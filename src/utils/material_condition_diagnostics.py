from dataclasses import dataclass

import numpy as np
import torch


def _to_numpy(value: np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_trajectory_pair(
    first: np.ndarray,
    second: np.ndarray,
    input_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = _to_numpy(first)
    second_array = _to_numpy(second)
    if (
        first_array.ndim != 3
        or second_array.ndim != 3
        or first_array.shape != second_array.shape
        or first_array.shape[-1] != 3
    ):
        raise ValueError(
            "trajectory inputs must share shape (T,N,3); "
            f"got {first_array.shape} and {second_array.shape}"
        )
    if not 0 < input_frames < first_array.shape[0]:
        raise ValueError("input_frames must satisfy 0 < input_frames < T")
    return first_array, second_array


def trajectory_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    input_frames: int = 5,
) -> dict[str, float]:
    """Compute per-trajectory rollout diagnostics for a ``(T,N,3)`` trajectory."""
    pred_array, gt_array = _validate_trajectory_pair(pred, gt, input_frames)
    squared_error = (pred_array - gt_array) ** 2
    prediction_mse = np.mean(squared_error[input_frames:], axis=(1, 2))
    long_start = max(input_frames, 18)
    long_stop = min(pred_array.shape[0], 25)
    long_mse = prediction_mse[long_start - input_frames : long_stop - input_frames]
    gm_mse = np.exp(np.mean(np.log(np.maximum(prediction_mse, 1e-30))))
    return {
        "full_rollout_mse": float(np.mean(squared_error)),
        "gm_mse": float(gm_mse),
        "long_seg_mse": float(np.mean(long_mse)) if long_mse.size else float("nan"),
        "fde": float(np.linalg.norm(pred_array[-1] - gt_array[-1], axis=-1).mean()),
    }


def condition_response_metrics(
    normal: np.ndarray,
    counterfactual: np.ndarray,
    input_frames: int = 5,
) -> dict[str, float]:
    """Measure counterfactual response only on frames after the shared condition."""
    normal_array, counterfactual_array = _validate_trajectory_pair(
        normal, counterfactual, input_frames
    )
    squared_difference = (counterfactual_array - normal_array) ** 2
    prediction_difference = squared_difference[input_frames:]
    return {
        "prediction_mse": float(np.mean(prediction_difference)),
        "final_prediction_mse": float(np.mean(prediction_difference[-1])),
    }


@dataclass(frozen=True)
class MaterialRecord:
    model: str
    mat_type: int
    log10_e: float
    nu: float


def build_parameter_donor_mapping(
    records: list[MaterialRecord], seed: int
) -> dict[str, MaterialRecord]:
    grouped: dict[int, list[MaterialRecord]] = {}
    for record in records:
        if record.mat_type not in (0, 1, 2):
            raise ValueError("mat_type expected one of 0, 1, 2")
        grouped.setdefault(record.mat_type, []).append(record)

    assignments: dict[str, MaterialRecord] = {}
    for mat_type, group in grouped.items():
        ordered = sorted(group, key=lambda record: record.model)
        if len(ordered) < 2:
            raise ValueError("each material group must contain at least two records")
        rng = np.random.default_rng(seed + mat_type)
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        for index, record in enumerate(shuffled):
            assignments[record.model] = shuffled[(index + 1) % len(shuffled)]
    return assignments


def build_parameter_derangement(
    records: list[MaterialRecord], seed: int
) -> dict[str, tuple[float, float]]:
    return {
        model: (donor.log10_e, donor.nu)
        for model, donor in build_parameter_donor_mapping(records, seed).items()
    }


def rotate_material_type(mat_type: int) -> int:
    if mat_type not in (0, 1, 2):
        raise ValueError("mat_type expected one of 0, 1, 2")
    return (mat_type + 1) % 3


_METRICS = ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde")
_MATERIAL_GROUPS = {0: "elastic", 1: "plasticine", 2: "sand"}
MATERIAL_INTERVENTIONS = (
    "shuffle_e",
    "shuffle_nu",
    "shuffle_params",
    "shuffle_class",
)


def paired_bootstrap(
    normal: np.ndarray,
    counterfactual: np.ndarray,
    samples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Summarize paired errors and a percentile bootstrap CI of their difference."""
    normal_array = np.asarray(normal, dtype=np.float64)
    counterfactual_array = np.asarray(counterfactual, dtype=np.float64)
    if normal_array.ndim != 1 or counterfactual_array.ndim != 1:
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    if normal_array.size == 0:
        raise ValueError("paired bootstrap inputs must not be empty")
    if normal_array.shape != counterfactual_array.shape:
        raise ValueError("paired bootstrap inputs must have equal lengths")
    if samples <= 0:
        raise ValueError("samples must be positive")

    normal_mean = float(normal_array.mean())
    if normal_mean <= 0:
        raise ValueError("normal mean must be positive")
    counterfactual_mean = float(counterfactual_array.mean())
    delta = counterfactual_array - normal_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    bootstrap_means = delta[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, (2.5, 97.5))
    mean_delta = float(delta.mean())
    return {
        "normal_mean": normal_mean,
        "counterfactual_mean": counterfactual_mean,
        "mean_delta": mean_delta,
        "relative_change_pct": mean_delta / normal_mean * 100.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def dependency_label(
    relative_change_pct: float,
    ci_low: float,
    ci_high: float,
    response_ratio_pct: float,
) -> str:
    """Classify condition dependence from effect size, confidence interval, and response."""
    if relative_change_pct >= 5.0 and ci_low > 0.0:
        return "used"
    if (
        abs(relative_change_pct) < 2.0
        and ci_low <= 0.0 <= ci_high
        and response_ratio_pct <= 2.0
    ):
        return "ignored"
    return "ambiguous"


def summarize_rows(
    rows: list[dict],
    intervention: str,
    samples: int = 10000,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Aggregate normal/counterfactual metrics overall and by material class."""
    if intervention not in MATERIAL_INTERVENTIONS:
        raise ValueError(
            "intervention must be one of: " + ", ".join(MATERIAL_INTERVENTIONS)
        )

    grouped_rows: dict[str, list[dict]] = {"overall": list(rows)}
    for mat_type, group_name in _MATERIAL_GROUPS.items():
        grouped_rows[group_name] = [row for row in rows if row.get("mat_type") == mat_type]

    summary: dict[str, dict[str, dict[str, float | str]]] = {}
    for group_name, group_rows in grouped_rows.items():
        if not group_rows:
            raise ValueError(f"{group_name} group must not be empty")
        normal_rollout = np.asarray(
            [row["normal_full_rollout_mse"] for row in group_rows], dtype=np.float64
        )
        prediction_mse = np.asarray(
            [row[f"{intervention}_prediction_mse"] for row in group_rows], dtype=np.float64
        )
        response_ratio_pct = float(prediction_mse.mean() / normal_rollout.mean() * 100.0)
        summary[group_name] = {}
        for metric in _METRICS:
            stats = paired_bootstrap(
                np.asarray([row[f"normal_{metric}"] for row in group_rows]),
                np.asarray([row[f"{intervention}_{metric}"] for row in group_rows]),
                samples=samples,
                seed=seed,
            )
            stats["response_ratio_pct"] = response_ratio_pct
            stats["label"] = dependency_label(
                stats["relative_change_pct"],
                stats["ci_low"],
                stats["ci_high"],
                response_ratio_pct,
            )
            summary[group_name][metric] = stats
    return summary
