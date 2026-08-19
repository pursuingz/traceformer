"""Local position-derived deformation diagnostics for the B0.4 audit."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
LOCAL_RESPONSE_NAMES = (
    "legacy_strain_f24",
    "legacy_strain_trajectory",
    "volumetric_strain_f24",
    "volumetric_strain_trajectory",
    "green_strain_f24",
    "green_strain_trajectory",
    "deviatoric_strain_f24",
    "deviatoric_strain_trajectory",
    "edge_stretch_f24",
    "edge_stretch_trajectory",
)
OUTPUT_NAMES = {
    "calibration": "material_local_deformation_b04_calibration.csv",
    "models": "material_local_deformation_b04_models.csv",
    "frames": "material_local_deformation_b04_frames.csv",
    "responses": "material_local_deformation_b04_responses.csv",
    "fidelity": "material_local_deformation_b04_fidelity.csv",
    "metadata": "material_local_deformation_b04_metadata.json",
    "report": "material_local_deformation_b04.md",
}


@dataclass(frozen=True)
class RestNeighborhood:
    """Fixed rest-space neighborhood and precomputed least-squares factors."""

    rest_points: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    rest_edges: np.ndarray
    inverse_moment: np.ndarray
    valid: np.ndarray
    condition_number: np.ndarray
    k: int
    condition_threshold: float
    regularization_scale: float


@dataclass(frozen=True)
class LocalDeformationResult:
    """Per-frame, per-particle local deformation quantities."""

    f_hat: np.ndarray
    jacobian: np.ndarray
    legacy_strain: np.ndarray
    volumetric_strain: np.ndarray
    green_strain_norm: np.ndarray
    deviatoric_strain_norm: np.ndarray
    edge_stretch: np.ndarray
    valid: np.ndarray
    condition_number: np.ndarray


def _points_array(value: Any, *, field: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be convertible to float64") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{field} must have shape (N, 3)")
    if points.shape[0] < 5:
        raise ValueError(f"{field} must contain at least 5 particles")
    if not np.isfinite(points).all():
        raise ValueError(f"{field} must contain only finite values")
    return points


def build_rest_neighborhood(
    rest_points: Any,
    *,
    k: int = 16,
    condition_threshold: float = 1e6,
    regularization_scale: float = 1e-6,
) -> RestNeighborhood:
    """Build a fixed Gaussian-weighted kNN graph in the reference configuration."""
    rest = _points_array(rest_points, field="rest_points")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    k = int(k)
    if k < 4 or k >= rest.shape[0]:
        raise ValueError("k must satisfy 4 <= k < particle count")
    if not np.isfinite(condition_threshold) or condition_threshold <= 0.0:
        raise ValueError("condition_threshold must be finite and positive")
    if not np.isfinite(regularization_scale) or regularization_scale < 0.0:
        raise ValueError("regularization_scale must be finite and non-negative")

    _, queried_indices = cKDTree(rest).query(rest, k=k + 1)
    indices = np.empty((rest.shape[0], k), dtype=np.int64)
    for particle, candidates in enumerate(np.asarray(queried_indices)):
        without_self = candidates[candidates != particle]
        if without_self.size < k:
            raise ValueError(f"particle {particle} does not have {k} distinct neighbors")
        indices[particle] = without_self[:k]

    rest_edges = rest[indices] - rest[:, None, :]
    rest_lengths = np.linalg.norm(rest_edges, axis=-1)
    bandwidth = rest_lengths[:, -1]
    positive_bandwidth = np.isfinite(bandwidth) & (bandwidth > 0.0)
    safe_bandwidth = np.where(positive_bandwidth, bandwidth, 1.0)
    weights = np.exp(-np.square(rest_lengths / safe_bandwidth[:, None]))

    moment = np.einsum("nki,nkj,nk->nij", rest_edges, rest_edges, weights)
    condition_number = np.linalg.cond(moment)
    scale = np.trace(moment, axis1=-2, axis2=-1) / 3.0
    valid = (
        positive_bandwidth
        & np.isfinite(condition_number)
        & (condition_number <= float(condition_threshold))
        & np.isfinite(scale)
        & (scale > 0.0)
    )

    inverse_moment = np.full_like(moment, np.nan)
    if np.any(valid):
        identity = np.eye(3, dtype=np.float64)[None, :, :]
        regularized = moment[valid] + (
            float(regularization_scale) * scale[valid]
        )[:, None, None] * identity
        inverse_moment[valid] = np.linalg.inv(regularized)

    return RestNeighborhood(
        rest_points=rest,
        indices=indices,
        weights=weights,
        rest_edges=rest_edges,
        inverse_moment=inverse_moment,
        valid=valid,
        condition_number=condition_number,
        k=k,
        condition_threshold=float(condition_threshold),
        regularization_scale=float(regularization_scale),
    )


def _trajectory_array(value: Any, particle_count: int) -> np.ndarray:
    source = value
    if hasattr(source, "detach"):
        source = source.detach()
    if hasattr(source, "cpu"):
        source = source.cpu()
    if hasattr(source, "numpy"):
        source = source.numpy()
    try:
        trajectory = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be convertible to float64") from exc
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape (T, N, 3)")
    if trajectory.shape[0] < 2:
        raise ValueError("trajectory must contain at least two frames")
    if trajectory.shape[1] != particle_count:
        raise ValueError("trajectory particle count must match the rest neighborhood")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory must contain only finite values")
    return trajectory


def estimate_local_deformation(
    trajectory: Any,
    neighborhood: RestNeighborhood,
) -> LocalDeformationResult:
    """Estimate local affine deformation and strain from point trajectories."""
    points = _trajectory_array(trajectory, neighborhood.rest_points.shape[0])
    current_edges = points[:, neighborhood.indices, :] - points[:, :, None, :]
    spatial_moment = np.einsum(
        "tnka,nkb,nk->tnab",
        current_edges,
        neighborhood.rest_edges,
        neighborhood.weights,
    )

    shape = (points.shape[0], points.shape[1], 3, 3)
    f_hat = np.full(shape, np.nan, dtype=np.float64)
    if np.any(neighborhood.valid):
        f_hat[:, neighborhood.valid] = np.einsum(
            "tnab,nbc->tnac",
            spatial_moment[:, neighborhood.valid],
            neighborhood.inverse_moment[neighborhood.valid],
        )

    jacobian = np.full(points.shape[:2], np.nan, dtype=np.float64)
    jacobian[:, neighborhood.valid] = np.linalg.det(f_hat[:, neighborhood.valid])
    identity = np.eye(3, dtype=np.float64)
    legacy_strain = np.linalg.norm(f_hat - identity, axis=(-2, -1))
    volumetric_strain = np.abs(jacobian - 1.0)

    right_cauchy_green = np.einsum("tnba,tnbc->tnac", f_hat, f_hat)
    green = 0.5 * (right_cauchy_green - identity)
    green_strain_norm = np.linalg.norm(green, axis=(-2, -1))
    green_trace = np.trace(green, axis1=-2, axis2=-1)
    deviatoric = green - green_trace[:, :, None, None] * identity / 3.0
    deviatoric_strain_norm = np.linalg.norm(deviatoric, axis=(-2, -1))

    rest_lengths = np.linalg.norm(neighborhood.rest_edges, axis=-1)
    current_lengths = np.linalg.norm(current_edges, axis=-1)
    positive_edges = rest_lengths > np.finfo(np.float64).eps
    length_ratio = np.ones_like(current_lengths)
    np.divide(
        current_lengths,
        rest_lengths[None, :, :],
        out=length_ratio,
        where=positive_edges[None, :, :],
    )
    edge_weights = neighborhood.weights * positive_edges
    weighted_stretch = np.sum(
        edge_weights[None, :, :] * np.abs(length_ratio - 1.0), axis=-1
    ) / np.sum(edge_weights, axis=-1)[None, :]
    edge_stretch = np.where(neighborhood.valid[None, :], weighted_stretch, np.nan)

    return LocalDeformationResult(
        f_hat=f_hat,
        jacobian=jacobian,
        legacy_strain=legacy_strain,
        volumetric_strain=volumetric_strain,
        green_strain_norm=green_strain_norm,
        deviatoric_strain_norm=deviatoric_strain_norm,
        edge_stretch=edge_stretch,
        valid=neighborhood.valid.copy(),
        condition_number=neighborhood.condition_number.copy(),
    )


def summarize_local_deformation(result: LocalDeformationResult) -> dict[str, Any]:
    """Aggregate per-particle quantities into deterministic per-frame summaries."""
    frames = result.jacobian.shape[0]
    if np.any(result.valid):
        condition = result.condition_number[result.valid]
        frame_means = {
            "jacobian_mean": np.nanmean(result.jacobian, axis=1),
            "legacy_strain_mean": np.nanmean(result.legacy_strain, axis=1),
            "volumetric_strain_mean": np.nanmean(result.volumetric_strain, axis=1),
            "green_strain_norm_mean": np.nanmean(result.green_strain_norm, axis=1),
            "deviatoric_strain_norm_mean": np.nanmean(
                result.deviatoric_strain_norm, axis=1
            ),
            "edge_stretch_mean": np.nanmean(result.edge_stretch, axis=1),
        }
        condition_median = float(np.median(condition))
        condition_p95 = float(np.percentile(condition, 95.0))
    else:
        frame_means = {
            name: np.full(frames, np.nan, dtype=np.float64)
            for name in (
                "jacobian_mean",
                "legacy_strain_mean",
                "volumetric_strain_mean",
                "green_strain_norm_mean",
                "deviatoric_strain_norm_mean",
                "edge_stretch_mean",
            )
        }
        condition_median = float("nan")
        condition_p95 = float("nan")

    return {
        "valid_fraction": float(np.mean(result.valid)),
        "condition_number_median": condition_median,
        "condition_number_p95": condition_p95,
        **frame_means,
    }


def _true_f_array(value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        true_f = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("true_f must be convertible to float64") from exc
    if true_f.ndim == 3 and true_f.shape[-1] == 9:
        true_f = true_f.reshape(*true_f.shape[:2], 3, 3)
    expected = (*expected_shape, 3, 3)
    if true_f.shape != expected:
        raise ValueError(f"true_f must have shape {expected}; got {true_f.shape}")
    if not np.isfinite(true_f).all():
        raise ValueError("true_f must contain only finite values")
    return true_f


def compare_estimated_to_true_f(
    estimated: LocalDeformationResult,
    true_f: Any,
) -> dict[str, float]:
    """Compare a position-derived estimate with simulator deformation gradients."""
    truth = _true_f_array(true_f, estimated.jacobian.shape)
    valid = estimated.valid
    valid_fraction = float(np.mean(valid))
    if not np.any(valid):
        return {
            "valid_fraction": valid_fraction,
            "f_relative_error": float("nan"),
            "j_absolute_error": float("nan"),
            "estimated_legacy_strain_trajectory": float("nan"),
            "true_legacy_strain_trajectory": float("nan"),
            "estimated_volumetric_strain_trajectory": float("nan"),
            "true_volumetric_strain_trajectory": float("nan"),
        }

    future_estimated = estimated.f_hat[1:, valid]
    future_truth = truth[1:, valid]
    difference = np.linalg.norm(future_estimated - future_truth, axis=(-2, -1))
    denominator = np.maximum(
        np.linalg.norm(future_truth, axis=(-2, -1)), np.finfo(np.float64).eps
    )
    true_j = np.linalg.det(future_truth)
    estimated_j = estimated.jacobian[1:, valid]
    identity = np.eye(3, dtype=np.float64)
    true_legacy = np.linalg.norm(future_truth - identity, axis=(-2, -1))

    return {
        "valid_fraction": valid_fraction,
        "f_relative_error": float(np.mean(difference / denominator)),
        "j_absolute_error": float(np.mean(np.abs(estimated_j - true_j))),
        "estimated_legacy_strain_trajectory": float(
            np.mean(estimated.legacy_strain[1:, valid])
        ),
        "true_legacy_strain_trajectory": float(np.mean(true_legacy)),
        "estimated_volumetric_strain_trajectory": float(
            np.mean(estimated.volumetric_strain[1:, valid])
        ),
        "true_volumetric_strain_trajectory": float(np.mean(np.abs(true_j - 1.0))),
    }


def extract_local_responses(result: LocalDeformationResult) -> dict[str, float]:
    """Extract the frozen B0.4 model-level local response schema."""
    if not np.any(result.valid):
        raise ValueError("local deformation result has no valid particles")
    summaries = summarize_local_deformation(result)
    response = {
        "legacy_strain_f24": float(summaries["legacy_strain_mean"][-1]),
        "legacy_strain_trajectory": float(
            np.mean(summaries["legacy_strain_mean"][1:])
        ),
        "volumetric_strain_f24": float(
            summaries["volumetric_strain_mean"][-1]
        ),
        "volumetric_strain_trajectory": float(
            np.mean(summaries["volumetric_strain_mean"][1:])
        ),
        "green_strain_f24": float(summaries["green_strain_norm_mean"][-1]),
        "green_strain_trajectory": float(
            np.mean(summaries["green_strain_norm_mean"][1:])
        ),
        "deviatoric_strain_f24": float(
            summaries["deviatoric_strain_norm_mean"][-1]
        ),
        "deviatoric_strain_trajectory": float(
            np.mean(summaries["deviatoric_strain_norm_mean"][1:])
        ),
        "edge_stretch_f24": float(summaries["edge_stretch_mean"][-1]),
        "edge_stretch_trajectory": float(
            np.mean(summaries["edge_stretch_mean"][1:])
        ),
    }
    values = np.asarray(list(response.values()), dtype=np.float64)
    if set(response) != set(LOCAL_RESPONSE_NAMES) or not np.isfinite(values).all():
        raise ValueError("local responses must be finite and match the frozen schema")
    return response


def _model_provenance(model_row: dict[str, Any]) -> dict[str, Any]:
    required = (
        "model",
        "mat_type",
        "log10_e",
        "nu",
        "checkpoint",
        "config",
        "seed",
        "sample_scope",
    )
    missing = [field for field in required if field not in model_row]
    if missing:
        raise ValueError(f"model_row is missing fields: {missing}")
    mat_type = model_row["mat_type"]
    if isinstance(mat_type, bool) or int(mat_type) not in MATERIAL_NAMES:
        raise ValueError("model_row.mat_type must be 0, 1, or 2")
    material = MATERIAL_NAMES[int(mat_type)]
    provided_material = model_row.get("material", material)
    if provided_material != material:
        raise ValueError("model_row material and mat_type disagree")
    provenance = dict(model_row)
    provenance["mat_type"] = int(mat_type)
    provenance["material"] = material
    return provenance


def build_local_response_rows(
    model_row: dict[str, Any],
    *,
    gt: LocalDeformationResult,
    pred: LocalDeformationResult,
) -> list[dict[str, Any]]:
    """Pair GT and predicted local responses for B0.3-compatible statistics."""
    provenance = _model_provenance(model_row)
    gt_response = extract_local_responses(gt)
    pred_response = extract_local_responses(pred)
    rows = []
    for name in LOCAL_RESPONSE_NAMES:
        gt_value = gt_response[name]
        pred_value = pred_response[name]
        rows.append(
            {
                **provenance,
                "response": name,
                "response_tier": "primary",
                "gt_value": gt_value,
                "pred_value": pred_value,
                "signed_error": pred_value - gt_value,
                "absolute_error": abs(pred_value - gt_value),
            }
        )
    return rows


def build_local_test_rows(
    model_row: dict[str, Any],
    *,
    gt: LocalDeformationResult,
    pred: LocalDeformationResult,
    input_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build model, frame, and response rows from one aligned factual rollout."""
    provenance = _model_provenance(model_row)
    if gt.f_hat.shape != pred.f_hat.shape:
        raise ValueError("GT and prediction local deformation shapes must match")
    if not np.array_equal(gt.valid, pred.valid):
        raise ValueError("GT and prediction must share the same rest-neighborhood mask")
    if input_frames < 1 or input_frames >= gt.f_hat.shape[0]:
        raise ValueError("input_frames must leave at least one predicted frame")
    if input_frames != 5 or gt.f_hat.shape[0] != 25:
        raise ValueError("B0.4 requires the frozen 5-input/25-frame protocol")
    valid = gt.valid
    if not np.any(valid):
        raise ValueError("test model has no valid local neighborhoods")
    future = slice(input_frames, None)
    f_difference = pred.f_hat[future, valid] - gt.f_hat[future, valid]
    j_difference = pred.jacobian[future, valid] - gt.jacobian[future, valid]
    gt_summary = summarize_local_deformation(gt)
    pred_summary = summarize_local_deformation(pred)
    model_output = {
        **provenance,
        "valid_fraction": float(np.mean(valid)),
        "condition_number_median": gt_summary["condition_number_median"],
        "condition_number_p95": gt_summary["condition_number_p95"],
        "local_f_mse": float(np.mean(np.square(f_difference))),
        "local_j_mae": float(np.mean(np.abs(j_difference))),
    }
    for segment, segment_slice in (
        ("short", slice(5, 11)),
        ("mid", slice(11, 18)),
        ("long", slice(18, 25)),
    ):
        segment_f = pred.f_hat[segment_slice, valid] - gt.f_hat[segment_slice, valid]
        segment_j = (
            pred.jacobian[segment_slice, valid] - gt.jacobian[segment_slice, valid]
        )
        model_output[f"local_f_mse_{segment}"] = float(
            np.mean(np.square(segment_f))
        )
        model_output[f"local_j_mae_{segment}"] = float(
            np.mean(np.abs(segment_j))
        )

    metric_fields = {
        "jacobian": "jacobian_mean",
        "legacy_strain": "legacy_strain_mean",
        "volumetric_strain": "volumetric_strain_mean",
        "green_strain": "green_strain_norm_mean",
        "deviatoric_strain": "deviatoric_strain_norm_mean",
        "edge_stretch": "edge_stretch_mean",
    }
    frame_rows: list[dict[str, Any]] = []
    for frame in range(gt.f_hat.shape[0]):
        row = {
            "model": provenance["model"],
            "mat_type": provenance["mat_type"],
            "material": provenance["material"],
            "log10_e": provenance["log10_e"],
            "nu": provenance["nu"],
            "frame": frame,
            "phase": "condition" if frame < input_frames else "predicted",
        }
        for short_name, summary_name in metric_fields.items():
            gt_value = float(gt_summary[summary_name][frame])
            pred_value = float(pred_summary[summary_name][frame])
            row[f"gt_{short_name}"] = gt_value
            row[f"pred_{short_name}"] = pred_value
            row[f"error_{short_name}"] = pred_value - gt_value
        frame_rows.append(row)
    response_rows = build_local_response_rows(provenance, gt=gt, pred=pred)
    return model_output, frame_rows, response_rows


def _validate_local_response_rows(
    rows: list[dict[str, Any]], *, require_frozen_counts: bool = True
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("local response rows must not be empty")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        provenance = _model_provenance(source)
        response = source.get("response")
        if response not in LOCAL_RESPONSE_NAMES:
            raise ValueError("local response row has an invalid response")
        row = dict(source)
        row.update(provenance)
        for field in ("gt_value", "pred_value", "signed_error", "absolute_error"):
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
            row[field] = value
        if not np.isclose(
            row["signed_error"], row["pred_value"] - row["gt_value"], atol=1e-12
        ):
            raise ValueError("signed_error must equal pred_value minus gt_value")
        if not np.isclose(row["absolute_error"], abs(row["signed_error"]), atol=1e-12):
            raise ValueError("absolute_error must equal abs(signed_error)")
        by_model.setdefault(str(row["model"]), []).append(row)

    counts = {material: 0 for material in MATERIAL_NAMES.values()}
    validated: list[dict[str, Any]] = []
    for model, model_rows in by_model.items():
        if len(model_rows) != len(LOCAL_RESPONSE_NAMES) or {
            row["response"] for row in model_rows
        } != set(LOCAL_RESPONSE_NAMES):
            raise ValueError(f"{model}: local response schema is incomplete or duplicated")
        counts[str(model_rows[0]["material"])] += 1
        validated.extend(model_rows)
    if require_frozen_counts and counts != {"elastic": 13, "plasticine": 14, "sand": 14}:
        raise ValueError("material counts must be elastic=13, plasticine=14, sand=14")
    return validated


def _bootstrap_ci(
    arrays: tuple[np.ndarray, ...],
    *,
    samples: int,
    rng: np.random.Generator,
    statistic: Any,
) -> tuple[float | None, float | None]:
    size = arrays[0].size
    estimates = []
    for indices in rng.integers(0, size, size=(samples, size)):
        value = statistic(*(array[indices] for array in arrays))
        if value is not None and np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return None, None
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def build_local_fidelity_rows(
    response_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build prediction-GT fidelity and material-response alignment rows."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows = _validate_local_response_rows(response_rows)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for group in ("overall", "elastic", "plasticine", "sand"):
        group_rows = rows if group == "overall" else [
            row for row in rows if row["material"] == group
        ]
        for response in LOCAL_RESPONSE_NAMES:
            selected = [row for row in group_rows if row["response"] == response]
            gt_values = np.asarray([row["gt_value"] for row in selected], dtype=np.float64)
            pred_values = np.asarray([row["pred_value"] for row in selected], dtype=np.float64)
            errors = pred_values - gt_values
            rho = _spearman(gt_values.tolist(), pred_values.tolist())
            rho_ci = (
                (None, None)
                if not np.isfinite(rho)
                else _bootstrap_ci(
                    (gt_values, pred_values),
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=lambda x, y: _spearman(x.tolist(), y.tolist()),
                )
            )
            bias_ci = _bootstrap_ci(
                (errors,),
                samples=bootstrap_samples,
                rng=rng,
                statistic=lambda values: float(np.mean(values)),
            )
            output.append(
                {
                    "analysis": "prediction_gt_fidelity",
                    "group": group,
                    "material": "" if group == "overall" else group,
                    "parameter": "",
                    "response": response,
                    "n": len(selected),
                    "gt_mean": float(np.mean(gt_values)),
                    "pred_mean": float(np.mean(pred_values)),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "bias": float(np.mean(errors)),
                    "ci_low": bias_ci[0],
                    "ci_high": bias_ci[1],
                    "gt_ordinary_rho": None,
                    "pred_ordinary_rho": None,
                    "gt_partial_rho": None,
                    "pred_partial_rho": None,
                    "spearman": rho if np.isfinite(rho) else None,
                    "spearman_ci_low": rho_ci[0],
                    "spearman_ci_high": rho_ci[1],
                    "alignment": "",
                }
            )

    from utils.material_response_fidelity import classify_alignment, partial_spearman

    for material in MATERIAL_NAMES.values():
        material_rows = [row for row in rows if row["material"] == material]
        for parameter, control_name in (("log10_e", "nu"), ("nu", "log10_e")):
            for response in LOCAL_RESPONSE_NAMES:
                selected = [row for row in material_rows if row["response"] == response]
                parameter_values = np.asarray(
                    [float(row[parameter]) for row in selected], dtype=np.float64
                )
                control = np.asarray(
                    [float(row[control_name]) for row in selected], dtype=np.float64
                )
                gt_values = np.asarray([row["gt_value"] for row in selected], dtype=np.float64)
                pred_values = np.asarray([row["pred_value"] for row in selected], dtype=np.float64)
                gt_ordinary = _spearman(parameter_values.tolist(), gt_values.tolist())
                pred_ordinary = _spearman(parameter_values.tolist(), pred_values.tolist())
                gt_partial = partial_spearman(parameter_values, gt_values, control)
                pred_partial = partial_spearman(parameter_values, pred_values, control)
                gt_ordinary_ci = _bootstrap_ci(
                    (parameter_values, gt_values),
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=lambda x, y: _spearman(x.tolist(), y.tolist()),
                )
                pred_ordinary_ci = _bootstrap_ci(
                    (parameter_values, pred_values),
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=lambda x, y: _spearman(x.tolist(), y.tolist()),
                )
                gt_partial_ci = _bootstrap_ci(
                    (parameter_values, gt_values, control),
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=partial_spearman,
                )
                pred_partial_ci = _bootstrap_ci(
                    (parameter_values, pred_values, control),
                    samples=bootstrap_samples,
                    rng=rng,
                    statistic=partial_spearman,
                )
                output.append(
                    {
                        "analysis": "material_response_alignment",
                        "group": material,
                        "material": material,
                        "parameter": parameter,
                        "response": response,
                        "n": len(selected),
                        "gt_mean": float(np.mean(gt_values)),
                        "pred_mean": float(np.mean(pred_values)),
                        "mae": float(np.mean(np.abs(pred_values - gt_values))),
                        "rmse": float(np.sqrt(np.mean(np.square(pred_values - gt_values)))),
                        "bias": float(np.mean(pred_values - gt_values)),
                        "ci_low": None,
                        "ci_high": None,
                        "gt_ordinary_rho": (
                            gt_ordinary if np.isfinite(gt_ordinary) else None
                        ),
                        "pred_ordinary_rho": (
                            pred_ordinary if np.isfinite(pred_ordinary) else None
                        ),
                        "gt_partial_rho": gt_partial,
                        "pred_partial_rho": pred_partial,
                        "gt_ordinary_ci_low": gt_ordinary_ci[0],
                        "gt_ordinary_ci_high": gt_ordinary_ci[1],
                        "pred_ordinary_ci_low": pred_ordinary_ci[0],
                        "pred_ordinary_ci_high": pred_ordinary_ci[1],
                        "gt_partial_ci_low": gt_partial_ci[0],
                        "gt_partial_ci_high": gt_partial_ci[1],
                        "pred_partial_ci_low": pred_partial_ci[0],
                        "pred_partial_ci_high": pred_partial_ci[1],
                        "spearman": None,
                        "spearman_ci_low": None,
                        "spearman_ci_high": None,
                        "alignment": classify_alignment(gt_partial, pred_partial),
                    }
                )
    return output


def build_calibration_row(
    *,
    model: str,
    mat_type: int,
    log10_e: float,
    nu: float,
    k: int,
    estimated: LocalDeformationResult,
    true_f: Any,
) -> dict[str, Any]:
    """Build one model/k row for the train-side estimator calibration."""
    if mat_type not in MATERIAL_NAMES:
        raise ValueError("mat_type must be 0, 1, or 2")
    comparison = compare_estimated_to_true_f(estimated, true_f)
    condition = estimated.condition_number[estimated.valid]
    return {
        "model": str(model),
        "mat_type": int(mat_type),
        "material": MATERIAL_NAMES[int(mat_type)],
        "log10_e": float(log10_e),
        "nu": float(nu),
        "k": int(k),
        "condition_number_median": (
            float(np.median(condition)) if condition.size else float("nan")
        ),
        "condition_number_p95": (
            float(np.percentile(condition, 95.0)) if condition.size else float("nan")
        ),
        **comparison,
    }


def _spearman(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 3 or np.ptp(x_array) <= 1e-12 or np.ptp(y_array) <= 1e-12:
        return float("nan")
    return float(spearmanr(x_array, y_array).statistic)


def evaluate_calibration_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen B0.4 calibration and neighborhood-robustness gates."""
    if not rows:
        raise ValueError("calibration rows must not be empty")
    primary = [row for row in rows if int(row.get("k", -1)) == 16]
    if not primary:
        raise ValueError("calibration rows must include k=16")

    reasons: list[str] = []
    valid_by_material: dict[str, float] = {}
    for material in MATERIAL_NAMES.values():
        material_rows = [row for row in primary if row.get("material") == material]
        if not material_rows:
            reasons.append(f"{material}: missing k=16 calibration rows")
            valid_by_material[material] = float("nan")
            continue
        valid_fraction = float(
            np.mean([float(row["valid_fraction"]) for row in material_rows])
        )
        valid_by_material[material] = valid_fraction
        if valid_fraction < 0.95:
            reasons.append(f"{material}: valid_fraction={valid_fraction:.6f} < 0.95")

    legacy_rho = _spearman(
        [float(row["estimated_legacy_strain_trajectory"]) for row in primary],
        [float(row["true_legacy_strain_trajectory"]) for row in primary],
    )
    volume_rho = _spearman(
        [float(row["estimated_volumetric_strain_trajectory"]) for row in primary],
        [float(row["true_volumetric_strain_trajectory"]) for row in primary],
    )
    for name, rho in (("legacy", legacy_rho), ("volumetric", volume_rho)):
        if not np.isfinite(rho) or rho < 0.8:
            reasons.append(f"{name} k=16 Spearman={rho:.6g} < 0.8")

    primary_by_model = {str(row["model"]): row for row in primary}
    sensitivity: dict[str, float] = {}
    for k in (8, 32):
        candidate_by_model = {
            str(row["model"]): row for row in rows if int(row.get("k", -1)) == k
        }
        if set(candidate_by_model) != set(primary_by_model):
            reasons.append(f"k={k}: model set differs from k=16")
            continue
        for short_name, field in (
            ("legacy", "estimated_legacy_strain_trajectory"),
            ("volumetric", "estimated_volumetric_strain_trajectory"),
        ):
            ordered = sorted(primary_by_model)
            rho = _spearman(
                [float(primary_by_model[model][field]) for model in ordered],
                [float(candidate_by_model[model][field]) for model in ordered],
            )
            sensitivity[f"{short_name}_k16_vs_k{k}_spearman"] = rho
            if not np.isfinite(rho) or rho < 0.9:
                reasons.append(
                    f"{short_name} k=16 vs k={k} Spearman={rho:.6g} < 0.9"
                )

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "valid_fraction_by_material": valid_by_material,
        "legacy_spearman_k16": legacy_rho,
        "volumetric_spearman_k16": volume_rho,
        **sensitivity,
    }


def preflight_local_deformation_outputs(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Resolve the fixed output set and reject accidental replacement."""
    directory = Path(output_dir)
    targets = {name: directory / filename for name, filename in OUTPUT_NAMES.items()}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing B0.4 outputs: {formatted}")
    return targets


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _csv_cell(value: Any) -> Any:
    converted = _json_value(value)
    if converted is None:
        return ""
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, ensure_ascii=False, sort_keys=True)
    return converted


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = _csv_columns(rows)
    if not columns:
        raise ValueError(f"cannot write empty CSV schema: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})


def _markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = _json_value(row.get(column))
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_local_deformation_report(
    *,
    calibration_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Render a compact Chinese report with an explicit interpretation gate."""
    gate = metadata["calibration_gate"]
    passed = gate.get("status") == "passed"
    gate_heading = "校准通过" if passed else "校准失败"
    if passed:
        interpretation = (
            "位置估计器通过预注册门槛；可以比较 prediction/GT 的局部形变响应，"
            "但仍不能把 factual correlation 解释为 counterfactual 因果正确性。"
        )
    else:
        interpretation = (
            "位置估计器未通过预注册门槛。禁止将 test 上的 F_hat/J_hat 结果解释为"
            "本构响应证据；只能把邻边伸缩视为辅助几何诊断。"
        )
    reasons = gate.get("reasons", [])
    reason_lines = [f"- {reason}" for reason in reasons] or ["- 无"]
    material_rows = []
    for material in MATERIAL_NAMES.values():
        subset = [row for row in model_rows if row.get("material") == material]
        material_rows.append(
            {
                "material": material,
                "n": len(subset),
                "mean_valid_fraction": (
                    float(np.mean([float(row["valid_fraction"]) for row in subset]))
                    if subset
                    else float("nan")
                ),
                "mean_local_f_mse": (
                    float(np.mean([float(row["local_f_mse"]) for row in subset]))
                    if subset
                    else float("nan")
                ),
                "mean_local_j_mae": (
                    float(np.mean([float(row["local_j_mae"]) for row in subset]))
                    if subset
                    else float("nan")
                ),
            }
        )
    fidelity_preview = fidelity_rows[: min(12, len(fidelity_rows))]
    return "\n".join(
        [
            "# B0.4 局部形变响应保真度审计",
            "",
            f"## {gate_heading}",
            "",
            interpretation,
            "",
            "### 校准失败原因",
            "",
            *reason_lines,
            "",
            "## 冻结协议",
            "",
            f"- checkpoint: `{metadata['checkpoint']}`",
            f"- config: `{metadata['config']}`",
            f"- split: `{metadata['split']}`",
            f"- 主邻域 k: `{metadata['k_primary']}`",
            f"- 稳健性邻域: `{metadata['k_sensitivity']}`",
            f"- calibration rows: `{len(calibration_rows)}`",
            f"- test models: `{len(model_rows)}`（41-model test 协议）",
            "",
            "## 分材质 prediction-GT 局部误差",
            "",
            _markdown_table(
                material_rows,
                (
                    "material",
                    "n",
                    "mean_valid_fraction",
                    "mean_local_f_mse",
                    "mean_local_j_mae",
                ),
            ),
            "",
            "## Fidelity 摘要（前 12 行）",
            "",
            _markdown_table(
                fidelity_preview,
                tuple(_csv_columns(fidelity_preview)) if fidelity_preview else ("analysis",),
            ),
            "",
            "## 解释边界",
            "",
            "- 所有 test 结果来自事实条件，不证明反事实材料控制正确。",
            "- `F_hat` 来自固定 rest-kNN 的位置最小二乘估计，不是模拟器直接输出。",
            "- 必须分材质阅读，overall 均值不能替代 elastic/plasticine/sand 裁决。",
            "",
        ]
    )


def _validate_output_payload(
    *,
    calibration_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    for name, rows in (
        ("calibration_rows", calibration_rows),
        ("model_rows", model_rows),
        ("frame_rows", frame_rows),
        ("response_rows", response_rows),
        ("fidelity_rows", fidelity_rows),
    ):
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{name} must be a non-empty list")
        if not all(isinstance(row, dict) and row for row in rows):
            raise ValueError(f"{name} must contain non-empty dictionaries")
    required_metadata = (
        "schema_version",
        "checkpoint",
        "config",
        "seed",
        "split",
        "model_counts",
        "calibration_status",
        "calibration_gate",
        "k_primary",
        "k_sensitivity",
    )
    missing = [field for field in required_metadata if field not in metadata]
    if missing:
        raise ValueError(f"metadata is missing fields: {missing}")
    if metadata["calibration_gate"].get("status") not in ("passed", "failed"):
        raise ValueError("metadata.calibration_gate.status must be passed or failed")
    if metadata["calibration_status"] != metadata["calibration_gate"]["status"]:
        raise ValueError("metadata.calibration_status must match calibration_gate.status")


def write_local_deformation_outputs(
    *,
    output_dir: str | Path,
    calibration_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Path]:
    """Write and atomically activate the fixed seven-file B0.4 report."""
    _validate_output_payload(
        calibration_rows=calibration_rows,
        model_rows=model_rows,
        frame_rows=frame_rows,
        response_rows=response_rows,
        fidelity_rows=fidelity_rows,
        metadata=metadata,
    )
    targets = preflight_local_deformation_outputs(output_dir, overwrite=overwrite)
    directory = Path(output_dir)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging.", dir=directory.parent)
    )
    staged = {name: temporary_dir / filename for name, filename in OUTPUT_NAMES.items()}
    backup_dir: Path | None = None
    activated: list[Path] = []
    try:
        _write_csv(staged["calibration"], calibration_rows)
        _write_csv(staged["models"], model_rows)
        _write_csv(staged["frames"], frame_rows)
        _write_csv(staged["responses"], response_rows)
        _write_csv(staged["fidelity"], fidelity_rows)
        staged["metadata"].write_text(
            json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staged["report"].write_text(
            render_local_deformation_report(
                calibration_rows=calibration_rows,
                model_rows=model_rows,
                fidelity_rows=fidelity_rows,
                metadata=metadata,
            ),
            encoding="utf-8",
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in staged.values()):
            raise OSError("staged B0.4 output validation failed")

        directory.mkdir(parents=True, exist_ok=True)
        existing = {name: path for name, path in targets.items() if path.exists()}
        if existing:
            backup_dir = Path(
                tempfile.mkdtemp(prefix=f".{directory.name}.backup.", dir=directory.parent)
            )
            backed_up: list[str] = []
            try:
                for name, path in existing.items():
                    os.replace(path, backup_dir / OUTPUT_NAMES[name])
                    backed_up.append(name)
            except Exception:
                for name in reversed(backed_up):
                    backup = backup_dir / OUTPUT_NAMES[name]
                    if backup.exists():
                        os.replace(backup, targets[name])
                raise
        try:
            for name, source in staged.items():
                os.replace(source, targets[name])
                activated.append(targets[name])
        except Exception:
            for path in activated:
                if path.exists():
                    path.unlink()
            if backup_dir is not None:
                for name in existing:
                    backup = backup_dir / OUTPUT_NAMES[name]
                    if backup.exists():
                        os.replace(backup, targets[name])
            raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    return targets
