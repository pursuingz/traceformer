from dataclasses import dataclass

import numpy as np


def _validate_trajectory_pair(
    first: np.ndarray,
    second: np.ndarray,
    input_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
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


def build_parameter_derangement(
    records: list[MaterialRecord], seed: int
) -> dict[str, tuple[float, float]]:
    grouped: dict[int, list[MaterialRecord]] = {}
    for record in records:
        if record.mat_type not in (0, 1, 2):
            raise ValueError("mat_type expected one of 0, 1, 2")
        grouped.setdefault(record.mat_type, []).append(record)

    assignments: dict[str, tuple[float, float]] = {}
    for mat_type, group in grouped.items():
        ordered = sorted(group, key=lambda record: record.model)
        if len(ordered) < 2:
            raise ValueError("each material group must contain at least two records")
        rng = np.random.default_rng(seed + mat_type)
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        for index, record in enumerate(shuffled):
            source = shuffled[(index + 1) % len(shuffled)]
            assignments[record.model] = (source.log10_e, source.nu)
    return assignments


def rotate_material_type(mat_type: int) -> int:
    if mat_type not in (0, 1, 2):
        raise ValueError("mat_type expected one of 0, 1, 2")
    return (mat_type + 1) % 3
