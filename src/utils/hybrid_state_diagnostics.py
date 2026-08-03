from typing import Dict, List, Union

import torch
import torch.nn as nn


def decompose_feedback(
    feedback: torch.Tensor,
    gate: Union[torch.Tensor, float],
) -> Dict[str, torch.Tensor]:
    """Decompose gated particle feedback into global and centered components."""
    if feedback.ndim != 3:
        raise ValueError("feedback must have shape (B,N,C)")

    gate_tensor = torch.as_tensor(
        gate,
        device=feedback.device,
        dtype=torch.float32,
    )
    if gate_tensor.numel() != 1:
        raise ValueError("gate must be a scalar")
    if not torch.isfinite(gate_tensor).all():
        raise ValueError("gate must be finite")

    delta = torch.nan_to_num(
        feedback.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ) * gate_tensor
    if not torch.isfinite(delta).all():
        raise ValueError("applied feedback must be finite")

    global_component = delta.mean(dim=1)
    deform_component = delta - global_component[:, None]
    feedback_energy = delta.square().mean(dim=(1, 2))
    global_energy = global_component.square().mean(dim=1)
    deform_energy = deform_component.square().mean(dim=(1, 2))
    fraction = torch.where(
        feedback_energy > 0,
        global_energy / feedback_energy,
        torch.zeros_like(feedback_energy),
    )
    return {
        "feedback_rms": feedback_energy.sqrt().cpu(),
        "global_rms": global_energy.sqrt().cpu(),
        "deform_rms": deform_energy.sqrt().cpu(),
        "feedback_energy": feedback_energy.cpu(),
        "global_energy": global_energy.cpu(),
        "deform_energy": deform_energy.cpu(),
        "global_energy_fraction": fraction.cpu(),
    }


class HybridStateFeedbackRecorder:
    """Record gated HST feedback without changing the model forward path."""

    def __init__(self, exchange: nn.Module):
        self.exchange = exchange
        self._records: List[dict] = []
        self._pending_stage = None
        self._exchange_handle = None
        self._feedback_handle = None

    def __enter__(self):
        self.reset()
        self._exchange_handle = self.exchange.register_forward_pre_hook(
            self._capture_stage,
            with_kwargs=True,
        )
        self._feedback_handle = self.exchange.feedback_attention.register_forward_hook(
            self._capture_feedback,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._exchange_handle is not None:
            self._exchange_handle.remove()
            self._exchange_handle = None
        if self._feedback_handle is not None:
            self._feedback_handle.remove()
            self._feedback_handle = None
        return False

    def reset(self):
        self._records.clear()
        self._pending_stage = None

    def _capture_stage(self, module, args, kwargs):
        if self._pending_stage is not None:
            raise RuntimeError("previous exchange stage was not consumed")
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            raise ValueError("exchange forward must provide hidden_states")
        if hidden_states.shape[0] != 1:
            raise ValueError("HybridStateFeedbackRecorder requires batch size 1")
        if "stage_index" not in kwargs:
            raise ValueError("exchange forward must provide stage_index")
        self._pending_stage = int(kwargs["stage_index"])

    def _capture_feedback(self, module, args, output):
        if self._pending_stage is None:
            raise RuntimeError("feedback captured without exchange stage")
        if output.shape[0] != 1:
            self._pending_stage = None
            raise ValueError("HybridStateFeedbackRecorder requires batch size 1")

        stage = self._pending_stage
        stats = decompose_feedback(output, self.exchange.feedback_gates[stage])
        self._records.append(
            {
                "stage": stage,
                "gate": float(self.exchange.feedback_gates[stage].detach().cpu()),
                **{key: float(value[0]) for key, value in stats.items()},
            }
        )
        self._pending_stage = None

    def finalize(self, expected_rollout_steps: int) -> List[dict]:
        if not isinstance(expected_rollout_steps, int) or expected_rollout_steps < 0:
            raise ValueError("expected_rollout_steps must be a non-negative integer")
        if self._pending_stage is not None:
            raise RuntimeError("exchange stage was not consumed")

        expected_count = expected_rollout_steps * self.exchange.num_stages
        if len(self._records) != expected_count:
            raise ValueError(
                f"expected {expected_count} records, got {len(self._records)}"
            )

        expected_stages = list(range(self.exchange.num_stages)) * expected_rollout_steps
        actual_stages = [row["stage"] for row in self._records]
        if actual_stages != expected_stages:
            raise ValueError(
                f"stage order mismatch: expected {expected_stages}, got {actual_stages}"
            )

        for index, row in enumerate(self._records):
            rollout_step = index // self.exchange.num_stages
            row["rollout_step"] = rollout_step
            row["absolute_frame"] = self.exchange.history_frames + rollout_step
        return list(self._records)
