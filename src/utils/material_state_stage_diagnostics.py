import math
from contextlib import contextmanager

import torch


STAGE_KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
    ("stage3_off", (1, 1, 1, 0)),
)


@contextmanager
def masked_material_state_stages(adapter, mask):
    scales = adapter.stage_scales
    original = scales.detach().clone()
    mask_tensor = torch.as_tensor(mask, device=scales.device, dtype=scales.dtype)
    if scales.shape != (4,) or mask_tensor.shape != scales.shape:
        raise ValueError("material-state stage mask must contain four values")
    if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
        raise ValueError("material-state stage mask must be binary")
    if not torch.isfinite(original).all():
        raise ValueError("checkpoint stage scales must be finite")

    with torch.no_grad():
        scales.copy_(original * mask_tensor)
    try:
        yield scales.detach().clone()
    finally:
        with torch.no_grad():
            scales.copy_(original)
        if not torch.equal(scales.detach(), original):
            raise RuntimeError("material-state stage scales were not restored exactly")


class MaterialStateActivityCollector:
    def __init__(self, adapter):
        self.adapter = adapter
        self._rows = []
        self._capture_state = None
        self._handle = None

    def __enter__(self):
        self._rows.clear()
        self._capture_state = None
        self._handle = self.adapter.register_forward_hook(
            self._capture_activity,
            with_kwargs=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._capture_state = None
        return False

    @contextmanager
    def capture(self, model_name, mat_type, expected_calls_per_stage):
        if self._handle is None:
            raise ValueError("activity collector must be active")
        if self._capture_state is not None:
            raise ValueError("nested model activity contexts are not supported")
        if (
            isinstance(expected_calls_per_stage, bool)
            or not isinstance(expected_calls_per_stage, int)
            or expected_calls_per_stage <= 0
        ):
            raise ValueError("expected_calls_per_stage must be a positive integer")

        num_stages = self._num_stages()
        state = {
            "model": model_name,
            "mat_type": mat_type,
            "expected_calls_per_stage": expected_calls_per_stage,
            "call_count": [0] * num_stages,
            "delta_sq_sum": [0.0] * num_stages,
            "hidden_sq_sum": [0.0] * num_stages,
            "numel": [0] * num_stages,
        }
        self._capture_state = state
        try:
            yield self
        except BaseException:
            self._capture_state = None
            raise
        else:
            try:
                self._append_rows(state)
            finally:
                self._capture_state = None

    def rows(self):
        return [dict(row) for row in self._rows]

    def _num_stages(self):
        scales = self.adapter.stage_scales
        if scales.ndim != 1:
            raise ValueError("adapter stage_scales must be one-dimensional")
        return int(scales.numel())

    def _capture_activity(self, module, positional_args, kwargs, output):
        state = self._capture_state
        if state is None:
            return
        if not positional_args:
            raise ValueError("material-state adapter hook requires hidden states")
        if "stage_index" not in kwargs:
            raise ValueError("material-state adapter forward must provide stage_index")

        hidden = positional_args[0]
        stage_index = int(kwargs["stage_index"])
        if not isinstance(hidden, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError("material-state adapter hook requires tensor input and output")
        if hidden.shape != output.shape:
            raise ValueError("material-state adapter input and output shapes must match")
        if not 0 <= stage_index < len(state["call_count"]):
            raise ValueError("material-state adapter stage_index is outside the configured range")

        hidden = hidden.detach().float()
        output = output.detach().float()
        if not torch.isfinite(hidden).all() or not torch.isfinite(output).all():
            raise ValueError("material-state adapter hook received nonfinite input or output")
        delta = output - hidden
        hidden_cpu = hidden.to(device="cpu", dtype=torch.float64)
        delta_cpu = delta.to(device="cpu", dtype=torch.float64)
        state["call_count"][stage_index] += 1
        state["delta_sq_sum"][stage_index] += float(delta_cpu.square().sum())
        state["hidden_sq_sum"][stage_index] += float(hidden_cpu.square().sum())
        state["numel"][stage_index] += delta_cpu.numel()

    def _append_rows(self, state):
        rows = []
        expected = state["expected_calls_per_stage"]
        for stage_index, call_count in enumerate(state["call_count"]):
            if call_count != expected:
                raise ValueError(
                    f"stage {stage_index} call count {call_count} does not match "
                    f"expected {expected}"
                )

            numel = state["numel"][stage_index]
            delta_rms = math.sqrt(state["delta_sq_sum"][stage_index] / numel)
            hidden_rms = math.sqrt(state["hidden_sq_sum"][stage_index] / numel)
            if hidden_rms == 0.0:
                raise ValueError("hidden RMS must be nonzero")
            rows.append(
                {
                    "model": state["model"],
                    "mat_type": state["mat_type"],
                    "stage_index": stage_index,
                    "call_count": call_count,
                    "delta_rms": delta_rms,
                    "hidden_rms": hidden_rms,
                    "relative_rms": delta_rms / hidden_rms,
                }
            )
        self._rows.extend(rows)
