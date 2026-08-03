import csv
import math
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn


FEEDBACK_METRICS = (
    "feedback_rms",
    "global_rms",
    "deform_rms",
    "global_energy_fraction",
)
TRAJECTORY_METRICS = (
    "full_rollout_mse",
    "fde",
    "f24_centroid_error",
    "f24_shape_residual_mse",
)
_MATERIAL_GROUPS = {0: "elastic", 1: "plasticine", 2: "sand"}
_HORIZON_ORDER = ("short", "mid", "long")


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


def horizon_bucket(absolute_frame: int) -> str:
    """Return the fixed diagnostic horizon bucket for an absolute frame."""
    if isinstance(absolute_frame, bool) or not isinstance(absolute_frame, int):
        raise ValueError("absolute_frame must be an integer in [5, 24]")
    if 5 <= absolute_frame <= 10:
        return "short"
    if 11 <= absolute_frame <= 17:
        return "mid"
    if 18 <= absolute_frame <= 24:
        return "long"
    raise ValueError("absolute_frame must be in [5, 24]")


def _finite_float(row: dict, key: str) -> float:
    if key not in row:
        raise ValueError(f"missing field: {key}")
    value = row[key]
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"field {key} must be finite")
    return numeric


def _integer_field(row: dict, key: str) -> int:
    if key not in row:
        raise ValueError(f"missing field: {key}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"field {key} must be an integer")
    return int(value)


def _material_group(row: dict) -> str:
    value = row.get("mat_type", row.get("material"))
    if value is None:
        raise ValueError("missing field: mat_type")
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in _MATERIAL_GROUPS.values():
            return normalized
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError("mat_type must be 0, 1, or 2") from exc
    try:
        material_type = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mat_type must be 0, 1, or 2") from exc
    if isinstance(value, bool) or material_type not in _MATERIAL_GROUPS:
        raise ValueError("mat_type must be 0, 1, or 2")
    return _MATERIAL_GROUPS[material_type]


def _validate_diagnostic_rows(rows: list[dict], *, include_trajectory: bool) -> list[dict]:
    if not rows:
        raise ValueError("rows must not be empty")

    required = (
        "model",
        "absolute_frame",
        "stage",
        *FEEDBACK_METRICS,
    )
    if include_trajectory:
        required += TRAJECTORY_METRICS

    validated = []
    for row in rows:
        for key in required:
            if key not in row:
                raise ValueError(f"missing field: {key}")
        if row["model"] is None or str(row["model"]) == "":
            raise ValueError("model must not be empty")
        absolute_frame = _integer_field(row, "absolute_frame")
        stage = _integer_field(row, "stage")
        material = _material_group(row)
        values = {
            metric: _finite_float(row, metric)
            for metric in (*FEEDBACK_METRICS, *TRAJECTORY_METRICS)
            if metric in row
        }
        validated.append(
            {
                **row,
                "model": str(row["model"]),
                "absolute_frame": absolute_frame,
                "stage": stage,
                "horizon": horizon_bucket(absolute_frame),
                "material": material,
                **values,
            }
        )
    return validated


def _group_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups = [("overall", rows)]
    for material in _MATERIAL_GROUPS.values():
        material_rows = [row for row in rows if row["material"] == material]
        if material_rows:
            groups.append((material, material_rows))
    return groups


def _model_means(rows: list[dict], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    return {
        model: {
            metric: float(np.mean([row[metric] for row in model_rows]))
            for metric in metrics
        }
        for model, model_rows in grouped.items()
    }


def aggregate_feedback_rows(rows: list[dict]) -> list[dict]:
    """Aggregate feedback at equal model weight by stage and horizon."""
    validated = _validate_diagnostic_rows(rows, include_trajectory=False)
    output = []
    for group, group_rows in _group_rows(validated):
        for dimension, values in (
            ("stage", sorted({row["stage"] for row in group_rows})),
            ("horizon", [bucket for bucket in _HORIZON_ORDER if any(
                row["horizon"] == bucket for row in group_rows
            )]),
        ):
            for value in values:
                selected = [row for row in group_rows if row[dimension] == value]
                model_means = _model_means(selected, FEEDBACK_METRICS)
                result = {
                    "group": group,
                    "material": group,
                    "stage": value if dimension == "stage" else None,
                    "horizon": value if dimension == "horizon" else None,
                    "n_models": len(model_means),
                }
                result.update(
                    {
                        metric: float(np.mean([values[metric] for values in model_means.values()]))
                        for metric in FEEDBACK_METRICS
                    }
                )
                output.append(result)
    return output


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    if denominator == 0.0:
        return float("nan")
    return float(np.clip(np.sum(first_centered * second_centered) / denominator, -1.0, 1.0))


def feedback_correlations(rows: list[dict]) -> list[dict]:
    """Compute model-level Pearson and Spearman feedback correlations."""
    validated = _validate_diagnostic_rows(rows, include_trajectory=True)
    output = []
    for group, group_rows in _group_rows(validated):
        model_means = _model_means(
            group_rows,
            (*FEEDBACK_METRICS, *TRAJECTORY_METRICS),
        )
        for feedback_metric in FEEDBACK_METRICS:
            for trajectory_metric in TRAJECTORY_METRICS:
                feedback_values = np.asarray(
                    [values[feedback_metric] for values in model_means.values()],
                    dtype=np.float64,
                )
                trajectory_values = np.asarray(
                    [values[trajectory_metric] for values in model_means.values()],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "group": group,
                        "feedback_metric": feedback_metric,
                        "trajectory_metric": trajectory_metric,
                        "n_models": len(model_means),
                        "pearson": _correlation(feedback_values, trajectory_values),
                        "spearman": _correlation(
                            _average_rank(feedback_values),
                            _average_rank(trajectory_values),
                        ) if len(model_means) >= 2 else float("nan"),
                    }
                )
    return output


CSV_COLUMNS = (
    "model",
    "mat_type",
    "log10_e",
    "nu",
    "rollout_step",
    "absolute_frame",
    "stage",
    "horizon",
    "gate",
    *FEEDBACK_METRICS,
    *TRAJECTORY_METRICS,
)


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "N/A"
    return str(value)


def write_feedback_csv(path: Path, rows: list[dict]) -> None:
    """Write sorted raw feedback rows with a stable diagnostic schema."""
    validated = _validate_diagnostic_rows(rows, include_trajectory=True)
    for row in validated:
        for key in ("mat_type", "log10_e", "nu", "rollout_step", "gate"):
            if key not in row:
                raise ValueError(f"missing field: {key}")
            if key in ("log10_e", "nu", "gate"):
                _finite_float(row, key)
            else:
                _integer_field(row, key)
    sorted_rows = sorted(
        validated,
        key=lambda row: (row["model"], row["rollout_step"], row["stage"]),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in sorted_rows:
            output = {key: row.get(key) for key in CSV_COLUMNS}
            output["horizon"] = row["horizon"]
            writer.writerow({key: _format_value(value) for key, value in output.items()})


def write_feedback_report(path: Path, rows: list[dict], metadata: dict[str, str]) -> None:
    """Write the grouped feedback and model-level correlation report."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    for key in ("checkpoint", "config"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata must contain a non-empty string: {key}")

    validated = _validate_diagnostic_rows(rows, include_trajectory=True)
    summary = aggregate_feedback_rows(validated)
    correlations = feedback_correlations(validated)
    material_models: dict[str, set[str]] = {}
    for row in validated:
        material_models.setdefault(row["material"], set()).add(row["model"])
    material_counts = {
        material: len(models)
        for material, models in material_models.items()
    }
    material_count_text = ", ".join(
        f"{group}={count}" for group, count in material_counts.items()
    )
    lines = ["# HST Feedback Diagnostic", "", "## Metadata", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "- Material groups: overall, elastic, plasticine, sand.",
            "- Scope: 41-window start_idx=0 full-rollout diagnostic.",
            f"- Material model counts: {material_count_text or 'none'}.",
            "- correlation: model-level diagnostic association, not significance or causality.",
            "- Correlations are diagnostic associations only; no significance or causal claims are made.",
            "",
            "## Grouped Feedback",
            "",
            "| group | stage | horizon | n_models | feedback_rms | global_rms | deform_rms | global_energy_fraction |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            "| {group} | {stage} | {horizon} | {n_models} | {feedback_rms} | {global_rms} | {deform_rms} | {global_energy_fraction} |".format(
                group=row["group"],
                stage=_format_value(row["stage"]),
                horizon=_format_value(row["horizon"]),
                n_models=row["n_models"],
                feedback_rms=_format_value(row["feedback_rms"]),
                global_rms=_format_value(row["global_rms"]),
                deform_rms=_format_value(row["deform_rms"]),
                global_energy_fraction=_format_value(row["global_energy_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Correlation Diagnostics",
            "",
            "| group | feedback_metric | trajectory_metric | n_models | pearson | spearman |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in correlations:
        lines.append(
            "| {group} | {feedback_metric} | {trajectory_metric} | {n_models} | {pearson} | {spearman} |".format(
                group=row["group"],
                feedback_metric=row["feedback_metric"],
                trajectory_metric=row["trajectory_metric"],
                n_models=row["n_models"],
                pearson=_format_value(row["pearson"]),
                spearman=_format_value(row["spearman"]),
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
