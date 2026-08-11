"""Position-only response metrics and validated B0.3 audit outputs."""

import csv
import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import rankdata, spearmanr


PRIMARY_RESPONSES = (
    "position_velocity_rms_trajectory",
    "position_acceleration_rms_trajectory",
    "centroid_displacement_f24",
    "centered_shape_mse_f24",
    "hull_volume_relative_change_f24",
    "hull_volume_relative_change_trajectory",
)
SECONDARY_RESPONSES = (
    "extent_change_x_f24",
    "extent_change_y_f24",
    "extent_change_z_f24",
    "future_contact_fraction",
)
RESPONSE_NAMES = (*PRIMARY_RESPONSES, *SECONDARY_RESPONSES)
MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
SUMMARY_GROUPS = ("overall", "elastic", "plasticine", "sand")
EXPECTED_MATERIAL_COUNTS = {0: 13, 1: 14, 2: 14}
OUTPUT_NAMES = {
    "models": "material_response_fidelity_b03_models.csv",
    "responses": "material_response_fidelity_b03_responses.csv",
    "fidelity": "material_response_fidelity_b03_fidelity.csv",
    "alignment": "material_response_fidelity_b03_alignment.csv",
    "metadata": "material_response_fidelity_b03_metadata.json",
    "report": "material_response_fidelity_b03.md",
}
MODEL_COLUMNS = (
    "model",
    "mat_type",
    "material",
    "log10_e",
    "nu",
    "checkpoint",
    "config",
    "seed",
    "sample_scope",
    "full_rollout_mse",
    "gm_mse",
    "long_seg_mse",
    "fde",
)
RESPONSE_COLUMNS = (
    "model",
    "mat_type",
    "material",
    "log10_e",
    "nu",
    "checkpoint",
    "config",
    "seed",
    "sample_scope",
    "response",
    "response_tier",
    "gt_value",
    "pred_value",
    "signed_error",
    "absolute_error",
)
FIDELITY_COLUMNS = (
    "group",
    "response",
    "response_tier",
    "n",
    "gt_mean",
    "gt_std",
    "pred_mean",
    "pred_std",
    "mae",
    "rmse",
    "bias",
    "spearman",
    "spearman_ci_low",
    "spearman_ci_high",
    "status",
)
ALIGNMENT_COLUMNS = (
    "material",
    "parameter",
    "response",
    "response_tier",
    "n",
    "gt_ordinary_rho",
    "pred_ordinary_rho",
    "gt_partial_rho",
    "pred_partial_rho",
    "gt_ordinary_ci_low",
    "gt_ordinary_ci_high",
    "pred_ordinary_ci_low",
    "pred_ordinary_ci_high",
    "gt_partial_ci_low",
    "gt_partial_ci_high",
    "pred_partial_ci_low",
    "pred_partial_ci_high",
    "ordinary_rho_gap",
    "partial_rho_gap",
    "magnitude_ratio",
    "alignment_label",
    "partial_alignment_label",
    "status",
)
_METADATA_REQUIRED_FIELDS = (
    "schema_version",
    "checkpoint",
    "config",
    "seed",
    "split",
    "model_counts",
    "response_schema",
    "bootstrap_samples",
)
_ROW_REQUIRED_FIELDS = (
    "model",
    "mat_type",
    "log10_e",
    "nu",
    "material",
    "response",
    "response_tier",
    "gt_value",
    "pred_value",
    "signed_error",
    "absolute_error",
)
_STAT_EPSILON = 1e-12


def _as_float64_trajectory(trajectory: Any) -> np.ndarray:
    value = trajectory
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be convertible to float64 NumPy") from exc

    if array.ndim != 3 or array.shape[0] != 25 or array.shape[2] != 3:
        raise ValueError("trajectory must have shape (25 frames, N, 3)")
    if array.shape[1] < 4:
        raise ValueError("trajectory must contain at least 4 particles")
    if not np.isfinite(array).all():
        raise ValueError("trajectory must contain only finite values")
    return array


def _convex_hull_volumes(trajectory: np.ndarray) -> np.ndarray:
    volumes = []
    for frame in trajectory:
        try:
            volume = float(ConvexHull(frame).volume)
        except QhullError as exc:
            raise ValueError(
                "trajectory frames must define a finite 3D convex hull"
            ) from exc
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("trajectory convex hull volumes must be positive and finite")
        volumes.append(volume)
    return np.asarray(volumes, dtype=np.float64)


def extract_position_responses(
    trajectory: Any,
    floor_height: float,
    contact_band_raw: float = 0.08,
) -> dict[str, float]:
    """Extract the frozen B0.3 position-response schema from one trajectory."""
    points = _as_float64_trajectory(trajectory)

    try:
        floor = float(floor_height)
        contact_band = float(contact_band_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("floor_height and contact_band_raw must be finite scalars") from exc
    if not np.isfinite(floor):
        raise ValueError("floor_height must be finite")
    if not np.isfinite(contact_band) or contact_band < 0.0:
        raise ValueError("contact_band_raw must be finite and non-negative")

    volumes = _convex_hull_volumes(points)
    centroid = points.mean(axis=1)
    centered = points - centroid[:, None, :]
    extent = np.ptp(points, axis=1)
    initial_volume = volumes[0]
    relative_volume_change = np.abs(volumes - initial_volume) / initial_volume
    contact = points[1:, :, 1] <= floor + contact_band

    result = {
        "position_velocity_rms_trajectory": float(
            np.sqrt(np.mean(np.diff(points, axis=0) ** 2))
        ),
        "position_acceleration_rms_trajectory": float(
            np.sqrt(np.mean(np.diff(points, n=2, axis=0) ** 2))
        ),
        "centroid_displacement_f24": float(
            np.linalg.norm(centroid[-1] - centroid[0])
        ),
        "centered_shape_mse_f24": float(
            np.mean((centered[-1] - centered[0]) ** 2)
        ),
        "hull_volume_relative_change_f24": float(relative_volume_change[-1]),
        "hull_volume_relative_change_trajectory": float(
            np.mean(relative_volume_change[1:])
        ),
        "extent_change_x_f24": float(extent[-1, 0] - extent[0, 0]),
        "extent_change_y_f24": float(extent[-1, 1] - extent[0, 1]),
        "extent_change_z_f24": float(extent[-1, 2] - extent[0, 2]),
        "future_contact_fraction": float(np.mean(contact)),
    }
    if set(result) != set(RESPONSE_NAMES) or not np.isfinite(
        np.asarray(tuple(result.values()), dtype=np.float64)
    ).all():
        raise ValueError("position responses must be finite and match the frozen schema")
    return result


def _finite_scalar(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{field} must be a finite number")
    return scalar


def _material_from_row(model_row: dict[str, Any]) -> tuple[int, str]:
    if not isinstance(model_row, dict):
        raise ValueError("model_row must be a dictionary")
    model = model_row.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model_row.model must be a non-empty string")
    mat_type = model_row.get("mat_type")
    if isinstance(mat_type, bool) or not isinstance(mat_type, (int, np.integer)):
        raise ValueError("model_row.mat_type must be 0, 1, or 2")
    mat_type = int(mat_type)
    if mat_type not in MATERIAL_NAMES:
        raise ValueError("model_row.mat_type must be 0, 1, or 2")
    _finite_scalar(model_row.get("log10_e"), "model_row.log10_e")
    _finite_scalar(model_row.get("nu"), "model_row.nu")
    return mat_type, MATERIAL_NAMES[mat_type]


def _response_tier(response: str) -> str:
    return "primary" if response in PRIMARY_RESPONSES else "secondary"


def build_response_rows(
    model_row: dict[str, Any],
    gt: Any,
    pred: Any,
    floor_height: float,
    contact_band_raw: float,
) -> list[dict[str, Any]]:
    """Pair the frozen GT and prediction response schemas for one model."""
    mat_type, material = _material_from_row(model_row)
    gt_responses = extract_position_responses(gt, floor_height, contact_band_raw)
    pred_responses = extract_position_responses(pred, floor_height, contact_band_raw)
    if set(gt_responses) != set(RESPONSE_NAMES) or set(pred_responses) != set(
        RESPONSE_NAMES
    ):
        raise ValueError("GT and prediction responses must match the frozen schema")

    provenance = dict(model_row)
    provenance["mat_type"] = mat_type
    provenance["material"] = material
    rows: list[dict[str, Any]] = []
    for response in RESPONSE_NAMES:
        gt_value = _finite_scalar(gt_responses[response], f"gt.{response}")
        pred_value = _finite_scalar(pred_responses[response], f"pred.{response}")
        signed_error = pred_value - gt_value
        rows.append(
            {
                **provenance,
                "response": response,
                "response_tier": _response_tier(response),
                "gt_value": gt_value,
                "pred_value": pred_value,
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
            }
        )
    return rows


def _is_constant(values: np.ndarray) -> bool:
    return values.size == 0 or float(np.ptp(values)) < _STAT_EPSILON


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if _is_constant(x) or _is_constant(y):
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if math.isfinite(rho) else None


def partial_spearman(x: Any, y: Any, control: Any) -> float | None:
    """Rank-transform values, OLS-residualize against ``control``, then correlate."""
    arrays = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in (x, y, control))
    if not all(array.ndim == 1 for array in arrays):
        return None
    x_values, y_values, control_values = arrays
    if (
        x_values.size < 3
        or x_values.shape != y_values.shape
        or x_values.shape != control_values.shape
        or not all(np.isfinite(array).all() for array in arrays)
        or _is_constant(x_values)
        or _is_constant(y_values)
        or _is_constant(control_values)
    ):
        return None

    ranked_x, ranked_y, ranked_control = (rankdata(array) for array in arrays)
    design = np.column_stack((np.ones_like(ranked_control), ranked_control))
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if _is_constant(residual_x) or _is_constant(residual_y):
        return 0.0
    rho = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return rho if math.isfinite(rho) else None


def classify_alignment(gt_rho: float | None, pred_rho: float | None) -> str:
    """Classify agreement with frozen precedence for overlapping thresholds."""
    if gt_rho is None or pred_rho is None:
        return "weak_or_unresolved"
    gt_value = _finite_scalar(gt_rho, "gt_rho")
    pred_value = _finite_scalar(pred_rho, "pred_rho")
    if abs(gt_value) >= 0.20 and np.sign(gt_value) != np.sign(pred_value):
        return "reversed"
    if (
        abs(gt_value) >= 0.20
        and np.sign(gt_value) == np.sign(pred_value)
        and abs(pred_value) < 0.50 * abs(gt_value)
    ):
        return "attenuated"
    if (
        abs(gt_value) >= 0.20
        and abs(pred_value) >= 0.20
        and np.sign(gt_value) == np.sign(pred_value)
    ):
        return "aligned"
    return "weak_or_unresolved"


def _validated_response_rows(
    response_rows: Sequence[dict[str, Any]], *, require_frozen_counts: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(response_rows, Sequence) or isinstance(response_rows, (str, bytes)):
        raise ValueError("response_rows must be a sequence of dictionaries")
    if not response_rows or any(not isinstance(row, dict) for row in response_rows):
        raise ValueError("response_rows must be a non-empty sequence of dictionaries")

    by_model: dict[str, list[dict[str, Any]]] = {}
    for source_row in response_rows:
        missing = [field for field in _ROW_REQUIRED_FIELDS if field not in source_row]
        if missing:
            raise ValueError(f"response row is missing fields: {missing}")
        row = dict(source_row)
        mat_type, material = _material_from_row(row)
        if row["material"] != material:
            raise ValueError("response row material must match mat_type")
        response = row["response"]
        if response not in RESPONSE_NAMES or row["response_tier"] != _response_tier(response):
            raise ValueError("response row has an invalid frozen response schema")
        for field in ("gt_value", "pred_value", "signed_error", "absolute_error"):
            row[field] = _finite_scalar(row[field], field)
        if not math.isclose(
            row["signed_error"], row["pred_value"] - row["gt_value"], abs_tol=1e-12
        ):
            raise ValueError("signed_error must equal pred_value minus gt_value")
        if not math.isclose(row["absolute_error"], abs(row["signed_error"]), abs_tol=1e-12):
            raise ValueError("absolute_error must equal abs(signed_error)")
        row["mat_type"] = mat_type
        by_model.setdefault(str(row["model"]), []).append(row)

    counts = {mat_type: 0 for mat_type in MATERIAL_NAMES}
    validated: list[dict[str, Any]] = []
    for model, model_rows in by_model.items():
        if len(model_rows) != len(RESPONSE_NAMES):
            raise ValueError(f"{model}: response schema must contain exactly one row per response")
        if {row["response"] for row in model_rows} != set(RESPONSE_NAMES):
            raise ValueError(f"{model}: response schema is incomplete or duplicated")
        metadata = tuple(
            model_rows[0][field] for field in ("mat_type", "log10_e", "nu", "material")
        )
        if any(
            tuple(row[field] for field in ("mat_type", "log10_e", "nu", "material"))
            != metadata
            for row in model_rows[1:]
        ):
            raise ValueError(f"{model}: metadata must agree across responses")
        counts[model_rows[0]["mat_type"]] += 1
        validated.extend(model_rows)

    if require_frozen_counts and counts != EXPECTED_MATERIAL_COUNTS:
        raise ValueError("material counts must be elastic=13, plasticine=14, sand=14")
    return validated


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    statistic: Any,
) -> tuple[float | None, float | None]:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        return None, None
    estimates = []
    for indices in rng.integers(0, x.size, size=(samples, x.size)):
        value = statistic(x[indices], y[indices])
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    if not estimates:
        return None, None
    low, high = np.percentile(np.asarray(estimates, dtype=np.float64), (2.5, 97.5))
    return float(low), float(high)


def _bootstrap_partial_ci(
    x: np.ndarray,
    y: np.ndarray,
    control: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    """Bootstrap a partial correlation by resampling full object triples together."""
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if (
        x.shape != y.shape
        or x.shape != control.shape
        or x.ndim != 1
        or x.size < 2
    ):
        return None, None
    estimates = []
    for indices in rng.integers(0, x.size, size=(samples, x.size)):
        value = partial_spearman(x[indices], y[indices], control[indices])
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    if not estimates:
        return None, None
    low, high = np.percentile(np.asarray(estimates, dtype=np.float64), (2.5, 97.5))
    return float(low), float(high)


def _group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "overall":
        return rows
    mat_type = next(
        mat_type for mat_type, material in MATERIAL_NAMES.items() if material == group
    )
    return [row for row in rows if row["mat_type"] == mat_type]


def build_fidelity_summary(
    response_rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Summarize factual GT-pred response fidelity with model-level pairing."""
    rows = _validated_response_rows(response_rows)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    summary: list[dict[str, Any]] = []
    for group in SUMMARY_GROUPS:
        grouped = _group_rows(rows, group)
        for response in RESPONSE_NAMES:
            selected = [row for row in grouped if row["response"] == response]
            gt_values = np.asarray([row["gt_value"] for row in selected], dtype=np.float64)
            pred_values = np.asarray([row["pred_value"] for row in selected], dtype=np.float64)
            errors = pred_values - gt_values
            constant = _is_constant(gt_values) or _is_constant(pred_values)
            rho = None if constant else _safe_spearman(gt_values, pred_values)
            ci_low, ci_high = (
                (None, None)
                if rho is None
                else _bootstrap_ci(
                    gt_values,
                    pred_values,
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=_safe_spearman,
                )
            )
            summary.append(
                {
                    "group": group,
                    "response": response,
                    "response_tier": _response_tier(response),
                    "n": len(selected),
                    "gt_mean": float(np.mean(gt_values)),
                    "gt_std": float(np.std(gt_values, ddof=0)),
                    "pred_mean": float(np.mean(pred_values)),
                    "pred_std": float(np.std(pred_values, ddof=0)),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "bias": float(np.mean(errors)),
                    "spearman": rho,
                    "spearman_ci_low": ci_low,
                    "spearman_ci_high": ci_high,
                    "status": "constant_response" if constant else "ok",
                }
            )
    return summary


def build_alignment_summary(
    response_rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Describe GT and predicted response alignment to factual material parameters."""
    rows = _validated_response_rows(response_rows)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    summary: list[dict[str, Any]] = []
    for material in MATERIAL_NAMES.values():
        material_rows = _group_rows(rows, material)
        for parameter, control_parameter in (("log10_e", "nu"), ("nu", "log10_e")):
            for response in RESPONSE_NAMES:
                selected = [row for row in material_rows if row["response"] == response]
                parameters = np.asarray([row[parameter] for row in selected], dtype=np.float64)
                control = np.asarray(
                    [row[control_parameter] for row in selected], dtype=np.float64
                )
                gt_values = np.asarray([row["gt_value"] for row in selected], dtype=np.float64)
                pred_values = np.asarray([row["pred_value"] for row in selected], dtype=np.float64)
                constant = _is_constant(gt_values) or _is_constant(pred_values)
                gt_ordinary = None if constant else _safe_spearman(parameters, gt_values)
                pred_ordinary = None if constant else _safe_spearman(parameters, pred_values)
                gt_partial = (
                    None
                    if constant
                    else partial_spearman(parameters, gt_values, control)
                )
                pred_partial = (
                    None
                    if constant
                    else partial_spearman(parameters, pred_values, control)
                )
                if constant:
                    ordinary_ci = (None, None, None, None)
                    partial_ci = (None, None, None, None)
                else:
                    gt_ordinary_ci = _bootstrap_ci(
                        parameters,
                        gt_values,
                        samples=bootstrap_samples,
                        rng=rng,
                        statistic=_safe_spearman,
                    )
                    pred_ordinary_ci = _bootstrap_ci(
                        parameters,
                        pred_values,
                        samples=bootstrap_samples,
                        rng=rng,
                        statistic=_safe_spearman,
                    )
                    gt_partial_ci = _bootstrap_partial_ci(
                        parameters,
                        gt_values,
                        control,
                        samples=bootstrap_samples,
                        rng=rng,
                    )
                    pred_partial_ci = _bootstrap_partial_ci(
                        parameters,
                        pred_values,
                        control,
                        samples=bootstrap_samples,
                        rng=rng,
                    )
                    ordinary_ci = (*gt_ordinary_ci, *pred_ordinary_ci)
                    partial_ci = (*gt_partial_ci, *pred_partial_ci)
                magnitude_ratio = (
                    None
                    if gt_ordinary is None or abs(gt_ordinary) < 0.05
                    else pred_ordinary / gt_ordinary
                    if pred_ordinary is not None
                    else None
                )
                summary.append(
                    {
                        "material": material,
                        "parameter": parameter,
                        "response": response,
                        "response_tier": _response_tier(response),
                        "n": len(selected),
                        "gt_ordinary_rho": gt_ordinary,
                        "pred_ordinary_rho": pred_ordinary,
                        "gt_partial_rho": gt_partial,
                        "pred_partial_rho": pred_partial,
                        "gt_ordinary_ci_low": ordinary_ci[0],
                        "gt_ordinary_ci_high": ordinary_ci[1],
                        "pred_ordinary_ci_low": ordinary_ci[2],
                        "pred_ordinary_ci_high": ordinary_ci[3],
                        "gt_partial_ci_low": partial_ci[0],
                        "gt_partial_ci_high": partial_ci[1],
                        "pred_partial_ci_low": partial_ci[2],
                        "pred_partial_ci_high": partial_ci[3],
                        "ordinary_rho_gap": (
                            None
                            if gt_ordinary is None or pred_ordinary is None
                            else pred_ordinary - gt_ordinary
                        ),
                        "partial_rho_gap": (
                            None
                            if gt_partial is None or pred_partial is None
                            else pred_partial - gt_partial
                        ),
                        "magnitude_ratio": magnitude_ratio,
                        "alignment_label": classify_alignment(
                            gt_ordinary, pred_ordinary
                        ),
                        "partial_alignment_label": classify_alignment(
                            gt_partial, pred_partial
                        ),
                        "status": "constant_response" if constant else "ok",
                    }
                )
    return summary


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite numeric values")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"metadata contains unsupported value: {type(value).__name__}")


def _validate_model_rows(model_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(model_rows, Sequence) or isinstance(model_rows, (str, bytes)):
        raise ValueError("model_rows must be a sequence of dictionaries")
    if len(model_rows) != sum(EXPECTED_MATERIAL_COUNTS.values()):
        raise ValueError("model_rows must contain exactly 41 models")

    seen_models: set[str] = set()
    counts = {material: 0 for material in MATERIAL_NAMES}
    validated: list[dict[str, Any]] = []
    for source_row in model_rows:
        if not isinstance(source_row, dict):
            raise ValueError("model_rows must be a sequence of dictionaries")
        missing = [field for field in MODEL_COLUMNS if field not in source_row]
        if missing:
            raise ValueError(f"model row is missing fields: {missing}")
        row = {field: source_row[field] for field in MODEL_COLUMNS}
        mat_type, material = _material_from_row(row)
        if row["material"] != material:
            raise ValueError("model row material must match mat_type")
        model = row["model"]
        if model in seen_models:
            raise ValueError(f"duplicate model row: {model}")
        seen_models.add(model)
        for field in ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde"):
            row[field] = _finite_scalar(row[field], field)
        row["mat_type"] = mat_type
        counts[mat_type] += 1
        validated.append(row)
    if counts != EXPECTED_MATERIAL_COUNTS:
        raise ValueError("model_rows material counts must be elastic=13, plasticine=14, sand=14")
    return sorted(validated, key=lambda row: str(row["model"]))


def _validate_summary_rows(
    rows: Sequence[dict[str, Any]],
    *,
    columns: tuple[str, ...],
    expected_count: int,
    row_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{row_name} must be a sequence of dictionaries")
    if len(rows) != expected_count:
        raise ValueError(f"{row_name} has an unexpected frozen schema size")
    output: list[dict[str, Any]] = []
    for source_row in rows:
        if not isinstance(source_row, dict):
            raise ValueError(f"{row_name} must contain dictionaries")
        missing = [field for field in columns if field not in source_row]
        if missing:
            raise ValueError(f"{row_name} is missing fields: {missing}")
        output.append({field: source_row[field] for field in columns})
    return output


def _validate_fidelity_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = _validate_summary_rows(
        rows,
        columns=FIDELITY_COLUMNS,
        expected_count=len(SUMMARY_GROUPS) * len(RESPONSE_NAMES),
        row_name="fidelity_rows",
    )
    identities = set()
    for row in validated:
        identity = (row["group"], row["response"])
        if row["group"] not in SUMMARY_GROUPS or row["response"] not in RESPONSE_NAMES:
            raise ValueError("fidelity_rows has an invalid frozen response schema")
        if identity in identities:
            raise ValueError("fidelity_rows has duplicate group-response rows")
        identities.add(identity)
        if row["response_tier"] != _response_tier(row["response"]):
            raise ValueError("fidelity_rows response tier must match response")
        expected_n = 41 if row["group"] == "overall" else EXPECTED_MATERIAL_COUNTS[
            next(kind for kind, name in MATERIAL_NAMES.items() if name == row["group"])
        ]
        if row["n"] != expected_n:
            raise ValueError("fidelity_rows n must match the frozen material counts")
        for field in ("gt_mean", "gt_std", "pred_mean", "pred_std", "mae", "rmse", "bias"):
            row[field] = _finite_scalar(row[field], field)
        for field in ("spearman", "spearman_ci_low", "spearman_ci_high"):
            if row[field] is not None:
                row[field] = _finite_scalar(row[field], field)
        if row["status"] not in ("ok", "constant_response"):
            raise ValueError("fidelity_rows status must be ok or constant_response")
        if row["status"] == "constant_response" and any(
            row[field] is not None
            for field in ("spearman", "spearman_ci_low", "spearman_ci_high")
        ):
            raise ValueError("constant fidelity response must leave correlation fields empty")
    if identities != {(group, response) for group in SUMMARY_GROUPS for response in RESPONSE_NAMES}:
        raise ValueError("fidelity_rows is incomplete or duplicated")
    return validated


def _validate_alignment_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = _validate_summary_rows(
        rows,
        columns=ALIGNMENT_COLUMNS,
        expected_count=len(MATERIAL_NAMES) * 2 * len(RESPONSE_NAMES),
        row_name="alignment_rows",
    )
    identities = set()
    nullable = tuple(field for field in ALIGNMENT_COLUMNS if field.endswith("rho") or field.endswith("ci_low") or field.endswith("ci_high") or field.endswith("rho_gap")) + ("magnitude_ratio",)
    for row in validated:
        identity = (row["material"], row["parameter"], row["response"])
        if row["material"] not in MATERIAL_NAMES.values() or row["parameter"] not in ("log10_e", "nu") or row["response"] not in RESPONSE_NAMES:
            raise ValueError("alignment_rows has an invalid frozen response schema")
        if identity in identities:
            raise ValueError("alignment_rows has duplicate material-parameter-response rows")
        identities.add(identity)
        if row["response_tier"] != _response_tier(row["response"]):
            raise ValueError("alignment_rows response tier must match response")
        expected_n = EXPECTED_MATERIAL_COUNTS[
            next(kind for kind, name in MATERIAL_NAMES.items() if name == row["material"])
        ]
        if row["n"] != expected_n:
            raise ValueError("alignment_rows n must match the frozen material counts")
        for field in nullable:
            if row[field] is not None:
                row[field] = _finite_scalar(row[field], field)
        for field in ("alignment_label", "partial_alignment_label"):
            if row[field] not in ("aligned", "attenuated", "reversed", "weak_or_unresolved"):
                raise ValueError("alignment_rows contains an invalid alignment label")
        if row["status"] not in ("ok", "constant_response"):
            raise ValueError("alignment_rows status must be ok or constant_response")
        if row["status"] == "constant_response" and any(
            row[field] is not None for field in nullable
        ):
            raise ValueError("constant alignment response must leave correlation fields empty")
    expected = {
        (material, parameter, response)
        for material in MATERIAL_NAMES.values()
        for parameter in ("log10_e", "nu")
        for response in RESPONSE_NAMES
    }
    if identities != expected:
        raise ValueError("alignment_rows is incomplete or duplicated")
    return validated


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    missing = [field for field in _METADATA_REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"metadata is missing fields: {missing}")
    result = _json_value(metadata)
    if result["model_counts"] != {"elastic": 13, "plasticine": 14, "sand": 14}:
        raise ValueError("metadata.model_counts must be elastic=13, plasticine=14, sand=14")
    if result["response_schema"] != list(RESPONSE_NAMES):
        raise ValueError("metadata.response_schema must match the frozen response schema")
    if not isinstance(result["seed"], int) or isinstance(result["seed"], bool) or result["seed"] < 0:
        raise ValueError("metadata.seed must be a non-negative integer")
    if not isinstance(result["bootstrap_samples"], int) or isinstance(result["bootstrap_samples"], bool) or result["bootstrap_samples"] < 1:
        raise ValueError("metadata.bootstrap_samples must be a positive integer")
    for field in ("schema_version", "checkpoint", "config", "split"):
        if not isinstance(result[field], str) or not result[field]:
            raise ValueError(f"metadata.{field} must be a non-empty string")
    return result


def _validate_output_payload(
    model_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    fidelity_rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validated_models = _validate_model_rows(model_rows)
    validated_responses = _validated_response_rows(response_rows)
    if {row["model"] for row in validated_models} != {
        row["model"] for row in validated_responses
    }:
        raise ValueError("model_rows and response_rows must describe the same 41 models")
    model_metadata = {
        row["model"]: tuple(
            row[field]
            for field in (
                "mat_type",
                "log10_e",
                "nu",
                "material",
                "checkpoint",
                "config",
                "seed",
                "sample_scope",
            )
        )
        for row in validated_models
    }
    for row in validated_responses:
        response_metadata = tuple(
            row[field]
            for field in (
                "mat_type",
                "log10_e",
                "nu",
                "material",
                "checkpoint",
                "config",
                "seed",
                "sample_scope",
            )
        )
        if response_metadata != model_metadata[row["model"]]:
            raise ValueError("model_rows and response_rows disagree on provenance")
    validated_metadata = _validate_metadata(metadata)
    for row in validated_models:
        if (
            row["checkpoint"],
            row["config"],
            row["seed"],
            row["sample_scope"],
        ) != (
            validated_metadata["checkpoint"],
            validated_metadata["config"],
            validated_metadata["seed"],
            validated_metadata["split"],
        ):
            raise ValueError("metadata and model_rows disagree on run provenance")
    return (
        validated_models,
        validated_responses,
        _validate_fidelity_rows(fidelity_rows),
        _validate_alignment_rows(alignment_rows),
        validated_metadata,
    )


def _write_csv(path: Path, columns: tuple[str, ...], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row[column] is None else row[column] for column in columns})


def _markdown_table(columns: tuple[str, ...], rows: Sequence[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join("" if row[column] is None else str(row[column]) for column in columns) + " |")
    return "\n".join((header, divider, *body))


def render_fidelity_markdown_report(
    fidelity_rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Render the Chinese, factual-only B0.3 decision report."""
    by_group = {group: [row for row in fidelity_rows if row["group"] == group] for group in SUMMARY_GROUPS}
    report = [
        "# B0.3 预测-GT 材料响应保真度审计",
        "",
        "## 审计边界",
        f"- checkpoint: `{metadata['checkpoint']}`",
        f"- config: `{metadata['config']}`",
        f"- seed: `{metadata['seed']}`；split: `{metadata['split']}`",
        "- 本报告仅比较预测和 GT 均可从 25 帧粒子位置轨迹计算的**位置可观测响应**。",
        "- 结果是 factual 条件下的保真度审计，不能证明 counterfactual 因果正确性，也不使用测试集选择 checkpoint 或训练超参数。",
        "- 凸包体积是几何可观测量，不等同于模拟器内部的 `det(F)` 或体积应变。",
        "",
        "## Factual 保真度（overall）",
        _markdown_table(("response", "n", "mae", "rmse", "spearman", "status"), [
            {key: row[key] for key in ("response", "n", "mae", "rmse", "spearman", "status")}
            for row in by_group["overall"]
        ]),
    ]
    for material in ("elastic", "plasticine", "sand"):
        report.extend((
            "",
            f"## {material} 的位置可观测响应",
            _markdown_table(("response", "n", "mae", "rmse", "spearman", "status"), [
                {key: row[key] for key in ("response", "n", "mae", "rmse", "spearman", "status")}
                for row in by_group[material]
            ]),
        ))
    report.extend((
        "",
        "## 材料参数对齐诊断",
        "下表为描述性 ordinary/partial Spearman 结果；每种材料只有 13 或 14 个对象，CI 只表达对象级重采样不确定性。",
        _markdown_table(
            ("material", "parameter", "response", "gt_partial_rho", "pred_partial_rho", "partial_rho_gap", "partial_alignment_label", "status"),
            [
                {key: row[key] for key in ("material", "parameter", "response", "gt_partial_rho", "pred_partial_rho", "partial_rho_gap", "partial_alignment_label", "status")}
                for row in alignment_rows
            ],
        ),
        "",
        "## 解释限制",
        "B0.3 只能说明冻结模型是否保留 factual 样本之间的位置响应排序和幅度；它不能证明更改 E/nu 后模型会产生正确的反事实轨迹。需要配对 counterfactual 数据或干预测试才可回答该问题。",
    ))
    return "\n".join(report) + "\n"


def _validate_output_targets(output_dir: Path, target_paths: dict[str, Path]) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"fidelity output path is not a directory: {output_dir}")
    for target in target_paths.values():
        if target.exists() and not target.is_file():
            raise IsADirectoryError(f"fidelity output target is not a regular file: {target}")


def preflight_fidelity_outputs(
    output_dir: str | Path, overwrite: bool
) -> dict[str, Path]:
    """Validate exact B0.3 targets without creating or replacing them."""
    directory = Path(output_dir)
    target_paths = {key: directory / name for key, name in OUTPUT_NAMES.items()}
    if not overwrite:
        existing = next((path for path in target_paths.values() if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"refusing to overwrite existing fidelity output: {existing}")
    _validate_output_targets(directory, target_paths)
    return target_paths


def _cleanup_transaction_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def _validate_staged_outputs(temporary_dir: Path) -> None:
    paths = [temporary_dir / name for name in OUTPUT_NAMES.values()]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("staged fidelity output set is incomplete")
    if {path.name for path in temporary_dir.iterdir()} != set(OUTPUT_NAMES.values()):
        raise RuntimeError("staged fidelity output set has an unexpected schema")


def _activate_output_files(temporary_dir: Path, target_paths: dict[str, Path]) -> None:
    backup_dir = Path(tempfile.mkdtemp(prefix=f".{temporary_dir.name}.backup.", dir=temporary_dir.parent))
    backups: dict[str, Path] = {}
    activated: list[str] = []
    try:
        for key, target_path in target_paths.items():
            if target_path.exists():
                backup_path = backup_dir / OUTPUT_NAMES[key]
                target_path.replace(backup_path)
                backups[key] = backup_path
        for key, target_path in target_paths.items():
            (temporary_dir / OUTPUT_NAMES[key]).replace(target_path)
            activated.append(key)
    except Exception as activation_error:
        rollback_errors = []
        for key in reversed(activated):
            try:
                target_paths[key].replace(temporary_dir / OUTPUT_NAMES[key])
            except OSError as error:
                rollback_errors.append(error)
        for key, backup_path in backups.items():
            try:
                backup_path.replace(target_paths[key])
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                f"fidelity output activation failed and rollback was incomplete; backups remain in {backup_dir}"
            ) from activation_error
        _cleanup_transaction_directory(backup_dir)
        raise
    else:
        _cleanup_transaction_directory(backup_dir)


def write_fidelity_outputs(
    output_dir: str | Path,
    model_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    fidelity_rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Path]:
    """Validate, stage, and transactionally activate the fixed six-file B0.3 output set."""
    directory = Path(output_dir)
    target_paths = preflight_fidelity_outputs(directory, overwrite=overwrite)
    (
        validated_models,
        validated_responses,
        validated_fidelity,
        validated_alignment,
        validated_metadata,
    ) = _validate_output_payload(
        model_rows, response_rows, fidelity_rows, alignment_rows, metadata
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        _write_csv(temporary_dir / OUTPUT_NAMES["models"], MODEL_COLUMNS, validated_models)
        _write_csv(temporary_dir / OUTPUT_NAMES["responses"], RESPONSE_COLUMNS, validated_responses)
        _write_csv(temporary_dir / OUTPUT_NAMES["fidelity"], FIDELITY_COLUMNS, validated_fidelity)
        _write_csv(temporary_dir / OUTPUT_NAMES["alignment"], ALIGNMENT_COLUMNS, validated_alignment)
        with (temporary_dir / OUTPUT_NAMES["metadata"]).open("w", encoding="utf-8") as handle:
            json.dump(validated_metadata, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (temporary_dir / OUTPUT_NAMES["report"]).write_text(
            render_fidelity_markdown_report(
                validated_fidelity, validated_alignment, validated_metadata
            ),
            encoding="utf-8",
        )
        _validate_staged_outputs(temporary_dir)
        directory.mkdir(parents=True, exist_ok=True)
        preflight_fidelity_outputs(directory, overwrite=overwrite)
        _activate_output_files(temporary_dir, target_paths)
    finally:
        _cleanup_transaction_directory(temporary_dir)
    return target_paths
