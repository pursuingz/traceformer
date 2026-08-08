"""H5 record extraction for the material identifiability audit."""

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
FRAME_INDICES = (5, 10, 15, 20, 24)

OUTPUT_NAMES = {
    "records": "material_identifiability_records.csv",
    "coverage": "material_identifiability_coverage.csv",
    "confounding": "material_identifiability_confounding.csv",
    "response": "material_identifiability_response.csv",
    "summary": "material_identifiability_summary.csv",
    "metadata": "material_identifiability_metadata.json",
    "report": "material_identifiability_b02.md",
}


@dataclass(frozen=True)
class AuditSettings:
    seed: int = 0
    folds: int = 5
    permutations: int = 500
    bootstrap_samples: int = 1000
    contact_band_raw: float = 0.08


class RecordValidationError(ValueError):
    pass


_FRAME_RESPONSE_PREFIXES = (
    "point_displacement_mse",
    "centroid_displacement",
    "centered_shape_mse",
    "extent_change_x",
    "extent_change_y",
    "extent_change_z",
    "future_contact_fraction",
)

STATIC_COLUMNS = (
    "model",
    "split",
    "material",
    "valid",
    "log10_e",
    "nu",
    "initial_centroid_x",
    "initial_centroid_y",
    "initial_centroid_z",
    "initial_extent_x",
    "initial_extent_y",
    "initial_extent_z",
    "initial_cov_eig_0",
    "initial_cov_eig_1",
    "initial_cov_eig_2",
    "radius_of_gyration",
    "initial_hull_volume",
    "total_particle_volume",
    "floor_gap",
    "gravity",
    "drag_magnitude",
    "drag_count",
    "drag_mask_ratio",
    "initial_contact_fraction",
)
NUISANCE_COLUMNS = (
    "log10_e",
    "nu",
    "initial_centroid_x",
    "initial_centroid_y",
    "initial_centroid_z",
    "initial_extent_x",
    "initial_extent_y",
    "initial_extent_z",
    "initial_cov_eig_0",
    "initial_cov_eig_1",
    "initial_cov_eig_2",
    "radius_of_gyration",
    "initial_hull_volume",
    "total_particle_volume",
    "floor_gap",
    "gravity",
    "drag_magnitude",
    "drag_count",
    "drag_mask_ratio",
    "initial_contact_fraction",
)
RESPONSE_COLUMNS = (
    "velocity_rms_trajectory",
    "velocity_rms_f24",
    "position_velocity_rms_trajectory",
    "position_acceleration_rms_trajectory",
    "f_strain_norm_f24",
    "f_strain_norm_trajectory",
    "volumetric_strain_f24",
    "volumetric_strain_trajectory",
    "c_norm_f24",
    "c_norm_trajectory",
    "contact_onset_frame",
    "future_contact_fraction",
    *(
        f"{prefix}_f{frame}"
        for prefix in _FRAME_RESPONSE_PREFIXES
        for frame in FRAME_INDICES
    ),
)
PRIMARY_RESPONSE_COLUMNS = (
    "centered_shape_mse_f24",
    "centroid_displacement_f24",
    "velocity_rms_trajectory",
    "f_strain_norm_f24",
    "volumetric_strain_f24",
)

_COVERAGE_DISTRIBUTION_COLUMNS = (
    "row_type",
    "split",
    "material",
    "parameter",
    "n",
    "unique_n",
    "min",
    "p05",
    "p25",
    "mean",
    "std",
    "p50",
    "p75",
    "p95",
    "max",
    "pearson_e_nu",
    "spearman_e_nu",
    "joint_grid_occupancy",
)
_COVERAGE_SUPPORT_COLUMNS = (
    "row_type",
    "split",
    "material",
    "parameter",
    "n_train",
    "n_test",
    "train_min",
    "train_max",
    "test_min",
    "test_max",
    "outside_train_fraction",
    "ks_statistic",
    "ks_pvalue",
    "wasserstein_distance",
    "joint_empty_bin_fraction",
    "support_status",
    "mahalanobis_feature_columns",
    "mahalanobis_train_p95",
    "mahalanobis_outside_fraction",
    "mahalanobis_nonfinite_test_fraction",
    *(
        f"smd_{column}"
        for column in NUISANCE_COLUMNS
        if column not in {"log10_e", "nu"}
    ),
)
_COVERAGE_COLUMNS = tuple(
    dict.fromkeys((*_COVERAGE_DISTRIBUTION_COLUMNS, *_COVERAGE_SUPPORT_COLUMNS))
)
_CONFOUNDING_COLUMNS = (
    "row_type",
    "material",
    "parameter",
    "feature",
    "feature_names",
    "fitted_features",
    "n",
    "pair_n",
    "pearson",
    "spearman",
    "cv_r2",
    "permutation_p",
    "confounded",
    "status",
    "seed",
    "folds",
    "invalid_record_count",
)
_RESPONSE_OUTPUT_COLUMNS = (
    "material",
    "parameter",
    "response",
    "response_tier",
    "n",
    "seed",
    "folds",
    "nuisance_features",
    "fitted_features",
    "r2_m0",
    "r2_me",
    "r2_mnu",
    "r2_mboth",
    "r2_augmented",
    "delta_r2",
    "partial_spearman",
    "permutation_p",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "q_value",
    "status",
    "invalid_record_count",
)
_SUMMARY_COLUMNS = (
    "material",
    "parameter",
    "status",
    "support_status",
    "reason_codes",
    "invalid_record_count",
)

PARAMETER_COLUMNS = ("log10_e", "nu")
_EXPECTED_SUMMARY_KEYS = tuple(
    (material, parameter)
    for material in MATERIAL_NAMES.values()
    for parameter in PARAMETER_COLUMNS
)
_STATIC_NUISANCE_COLUMNS = tuple(
    column for column in NUISANCE_COLUMNS if column not in PARAMETER_COLUMNS
)
_LOG10_E_RANGE = (4.0, 7.0)
_NU_RANGE = (0.05, 0.45)
_RIDGE_ALPHAS = 10.0 ** np.arange(-4, 4)
_STAT_EPSILON = 1e-12

_STATIC_REQUIRED_FIELDS = (
    "x",
    "vol",
    "E",
    "nu",
    "mat_type",
    "gravity",
    "floor_height",
    "drag_force",
    "drag_mask",
)
_TRAIN_DYNAMIC_FIELDS = ("v", "F", "C")


def _convex_hull_volume(points: np.ndarray) -> float:
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return float("nan")


def _require_fields(handle: h5py.File, fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in handle:
            raise RecordValidationError(f"missing mandatory field {field}")


def _scalar(handle: h5py.File, field: str) -> float:
    value = np.asarray(handle[field][()])
    if value.size != 1:
        raise RecordValidationError(f"{field} must be scalar")
    return float(value.reshape(()))


def _validate_metadata(handle: h5py.File) -> tuple[float, float, int]:
    e_value = _scalar(handle, "E")
    if not np.isfinite(e_value) or e_value <= 0:
        raise RecordValidationError("finite E is required and must be positive")

    nu = _scalar(handle, "nu")
    if not np.isfinite(nu):
        raise RecordValidationError("nu must be finite")

    mat_value = np.asarray(handle["mat_type"][()])
    if mat_value.size != 1:
        raise RecordValidationError("mat_type must be scalar")
    try:
        numeric_mat_type = float(mat_value.reshape(()))
    except (TypeError, ValueError):
        raise RecordValidationError("mat_type must be 0, 1, or 2") from None
    if not np.isfinite(numeric_mat_type) or not numeric_mat_type.is_integer():
        raise RecordValidationError("mat_type must be 0, 1, or 2")
    mat_type = int(numeric_mat_type)
    if mat_type not in MATERIAL_NAMES:
        raise RecordValidationError("mat_type must be 0, 1, or 2")
    return e_value, nu, mat_type


def _validate_initial_points(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RecordValidationError("x must have shape (particles, 3)")
    if not np.isfinite(points).all():
        raise RecordValidationError("x must be finite")


def _validate_x_dataset(dataset: h5py.Dataset) -> None:
    if dataset.ndim != 3 or dataset.shape[-1] != 3:
        raise RecordValidationError("x must have shape (frames, particles, 3)")
    if dataset.shape[0] == 0:
        raise RecordValidationError("x must contain at least one frame")
    if dataset.shape[1] == 0:
        raise RecordValidationError("x must contain at least one particle")


def _static_features(
    handle: h5py.File,
    initial_points: np.ndarray,
    *,
    e_value: float,
    nu: float,
    mat_type: int,
    split: str,
    settings: AuditSettings,
    model: str,
) -> dict[str, object]:
    particle_count = initial_points.shape[0]
    volume = np.asarray(handle["vol"][:], dtype=np.float64)
    if volume.ndim != 1:
        raise RecordValidationError("vol must have shape (particles,)")
    if volume.shape[0] != particle_count:
        raise RecordValidationError("vol particle dimension must match x")
    if not np.isfinite(volume).all():
        raise RecordValidationError("vol must be finite")

    gravity = _scalar(handle, "gravity")
    floor_height = _scalar(handle, "floor_height")
    if not np.isfinite(gravity) or not np.isfinite(floor_height):
        raise RecordValidationError("gravity and floor_height must be finite")

    drag_force = np.asarray(handle["drag_force"][:], dtype=np.float64)
    drag_mask = np.asarray(handle["drag_mask"][:], dtype=np.float64)
    if drag_force.ndim != 2 or drag_force.shape[1] != 3:
        raise RecordValidationError("drag_force must have shape (forces, 3)")
    if drag_mask.ndim != 2:
        raise RecordValidationError("drag_mask must have shape (forces, particles)")
    if drag_force.shape[0] != drag_mask.shape[0]:
        raise RecordValidationError("drag_force and drag_mask force count must match")
    if drag_mask.shape[1] != particle_count:
        raise RecordValidationError("drag_mask particle dimension must match x")
    if not np.isfinite(drag_force).all() or not np.isfinite(drag_mask).all():
        raise RecordValidationError("drag_force and drag_mask must be finite")

    centroid = initial_points.mean(axis=0)
    extents = initial_points.max(axis=0) - initial_points.min(axis=0)
    covariance = np.cov(initial_points, rowvar=False, bias=True)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    centered_points = initial_points - centroid
    contact = initial_points[:, 1] <= floor_height + settings.contact_band_raw

    return {
        "model": model,
        "split": split,
        "material": MATERIAL_NAMES[mat_type],
        "valid": True,
        "log10_e": float(np.log10(e_value)),
        "nu": nu,
        "initial_centroid_x": float(centroid[0]),
        "initial_centroid_y": float(centroid[1]),
        "initial_centroid_z": float(centroid[2]),
        "initial_extent_x": float(extents[0]),
        "initial_extent_y": float(extents[1]),
        "initial_extent_z": float(extents[2]),
        "initial_cov_eig_0": float(covariance_eigenvalues[0]),
        "initial_cov_eig_1": float(covariance_eigenvalues[1]),
        "initial_cov_eig_2": float(covariance_eigenvalues[2]),
        "radius_of_gyration": float(np.sqrt(np.mean(np.sum(centered_points ** 2, axis=1)))),
        "initial_hull_volume": _convex_hull_volume(initial_points),
        "total_particle_volume": float(volume.sum()),
        "floor_gap": float(initial_points[:, 1].min() - floor_height),
        "gravity": gravity,
        "drag_magnitude": float(np.linalg.norm(drag_force)),
        "drag_count": int(drag_force.shape[0]),
        "drag_mask_ratio": float(np.count_nonzero(drag_mask) / drag_mask.size)
        if drag_mask.size
        else 0.0,
        "initial_contact_fraction": float(contact.mean()),
    }


def _validate_train_dynamics(
    x: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
) -> None:
    if x.ndim != 3 or x.shape[-1] != 3:
        raise RecordValidationError("x must have shape (frames, particles, 3)")
    if x.shape[0] < 25:
        raise RecordValidationError("train records require at least 25 frames")
    if v.shape != x.shape:
        raise RecordValidationError("v particle dimension must match x")
    expected_matrix_shape = x.shape[:2] + (3, 3)
    if f.shape != expected_matrix_shape:
        raise RecordValidationError("F particle dimension must match x")
    if c.shape != expected_matrix_shape:
        raise RecordValidationError("C particle dimension must match x")
    if not all(np.isfinite(values).all() for values in (x, v, f, c)):
        raise RecordValidationError("train dynamics must be finite")


def _train_responses(
    x: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    *,
    floor_height: float,
    settings: AuditSettings,
) -> dict[str, object]:
    identity = np.eye(3, dtype=f.dtype)
    f_strain = np.linalg.norm(f - identity, axis=(-2, -1))
    j_error = np.abs(np.linalg.det(f) - 1.0)
    c_norm = np.linalg.norm(c, axis=(-2, -1))
    initial_centroid = x[0].mean(axis=0)
    initial_centered = x[0] - initial_centroid
    initial_extents = np.ptp(x[0], axis=0)
    contact = x[:, :, 1] <= floor_height + settings.contact_band_raw

    response: dict[str, object] = {
        "velocity_rms_trajectory": float(np.sqrt(np.mean(v ** 2))),
        "velocity_rms_f24": float(np.sqrt(np.mean(v[24] ** 2))),
        "position_velocity_rms_trajectory": float(
            np.sqrt(np.mean(np.diff(x, axis=0) ** 2))
        ),
        "position_acceleration_rms_trajectory": float(
            np.sqrt(np.mean(np.diff(x, n=2, axis=0) ** 2))
        ),
        "f_strain_norm_f24": float(f_strain[24].mean()),
        "f_strain_norm_trajectory": float(f_strain.mean()),
        "volumetric_strain_f24": float(j_error[24].mean()),
        "volumetric_strain_trajectory": float(j_error.mean()),
        "c_norm_f24": float(c_norm[24].mean()),
        "c_norm_trajectory": float(c_norm.mean()),
        "contact_onset_frame": _contact_onset_frame(contact),
        "future_contact_fraction": float(contact[1:25].mean()),
    }
    for frame in FRAME_INDICES:
        centered = x[frame] - x[frame].mean(axis=0)
        extent_change = np.ptp(x[frame], axis=0) - initial_extents
        response[f"point_displacement_mse_f{frame}"] = float(
            np.mean((x[frame] - x[0]) ** 2)
        )
        response[f"centered_shape_mse_f{frame}"] = float(
            np.mean((centered - initial_centered) ** 2)
        )
        response[f"centroid_displacement_f{frame}"] = float(
            np.linalg.norm(x[frame].mean(axis=0) - initial_centroid)
        )
        response[f"extent_change_x_f{frame}"] = float(extent_change[0])
        response[f"extent_change_y_f{frame}"] = float(extent_change[1])
        response[f"extent_change_z_f{frame}"] = float(extent_change[2])
        response[f"future_contact_fraction_f{frame}"] = float(contact[frame].mean())
    return response


def _contact_onset_frame(contact: np.ndarray) -> float:
    contact_frames = np.flatnonzero(contact.any(axis=1))
    return float(contact_frames[0]) if contact_frames.size else float("nan")


def _validate_bins(bins: int) -> None:
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")


def _audit_materials(
    train_records: list[dict[str, object]], test_records: list[dict[str, object]]
) -> tuple[str, ...]:
    observed = {str(record["material"]) for record in (*train_records, *test_records)}
    known = tuple(name for name in MATERIAL_NAMES.values() if name in observed)
    return (*known, *sorted(observed.difference(known)))


def _material_records(
    records: list[dict[str, object]], material: str
) -> list[dict[str, object]]:
    return [record for record in records if record["material"] == material]


def _column_values(records: list[dict[str, object]], column: str) -> np.ndarray:
    return np.asarray([float(record[column]) for record in records], dtype=np.float64)


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {
            "n": 0,
            "unique_n": 0,
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(values.size),
        "unique_n": int(np.unique(values).size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p50": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _correlations(log10_e: np.ndarray, nu: np.ndarray) -> tuple[float, float]:
    if log10_e.size < 2 or np.ptp(log10_e) == 0 or np.ptp(nu) == 0:
        return float("nan"), float("nan")
    return (
        float(np.corrcoef(log10_e, nu)[0, 1]),
        float(spearmanr(log10_e, nu)[0]),
    )


def _joint_bin_indices(
    log10_e: np.ndarray,
    nu: np.ndarray,
    *,
    bins: int,
) -> list[tuple[int, int] | None]:
    log10_e_edges = np.linspace(*_LOG10_E_RANGE, bins + 1)
    nu_edges = np.linspace(*_NU_RANGE, bins + 1)
    indices: list[tuple[int, int] | None] = []
    for e_value, nu_value in zip(log10_e, nu):
        if not (
            _LOG10_E_RANGE[0] <= e_value <= _LOG10_E_RANGE[1]
            and _NU_RANGE[0] <= nu_value <= _NU_RANGE[1]
        ):
            indices.append(None)
            continue
        e_bin = min(np.searchsorted(log10_e_edges, e_value, side="right") - 1, bins - 1)
        nu_bin = min(np.searchsorted(nu_edges, nu_value, side="right") - 1, bins - 1)
        indices.append((int(e_bin), int(nu_bin)))
    return indices


def _joint_grid_occupancy(log10_e: np.ndarray, nu: np.ndarray, *, bins: int) -> float:
    if not log10_e.size:
        return float("nan")
    occupied = {index for index in _joint_bin_indices(log10_e, nu, bins=bins) if index is not None}
    return float(len(occupied) / (bins * bins))


def build_coverage_rows(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    *,
    bins: int = 5,
) -> list[dict[str, object]]:
    """Summarize material-local parameter coverage for train and test objects."""
    _validate_bins(bins)
    rows: list[dict[str, object]] = []
    for split, records in (("train", train_records), ("test", test_records)):
        for material in _audit_materials(train_records, test_records):
            material_records = _material_records(records, material)
            log10_e = _column_values(material_records, "log10_e")
            nu = _column_values(material_records, "nu")
            pearson_e_nu, spearman_e_nu = _correlations(log10_e, nu)
            joint_grid_occupancy = _joint_grid_occupancy(log10_e, nu, bins=bins)
            for parameter in PARAMETER_COLUMNS:
                rows.append(
                    {
                        "split": split,
                        "material": material,
                        "parameter": parameter,
                        **_distribution_summary(_column_values(material_records, parameter)),
                        "pearson_e_nu": pearson_e_nu,
                        "spearman_e_nu": spearman_e_nu,
                        "joint_grid_occupancy": joint_grid_occupancy,
                    }
                )
    return rows


def _standardized_mean_differences(
    train_records: list[dict[str, object]], test_records: list[dict[str, object]]
) -> dict[str, float]:
    differences: dict[str, float] = {}
    for column in _STATIC_NUISANCE_COLUMNS:
        train_values = _column_values(train_records, column)
        test_values = _column_values(test_records, column)
        finite_train = train_values[np.isfinite(train_values)]
        finite_test = test_values[np.isfinite(test_values)]
        if finite_train.size < 2 or not finite_test.size:
            differences[f"smd_{column}"] = float("nan")
            continue
        train_std = float(finite_train.std(ddof=1))
        differences[f"smd_{column}"] = (
            float((finite_test.mean() - finite_train.mean()) / train_std)
            if train_std > 0
            else float("nan")
        )
    return differences


def _joint_empty_bin_fraction(
    train_log10_e: np.ndarray,
    train_nu: np.ndarray,
    test_log10_e: np.ndarray,
    test_nu: np.ndarray,
    *,
    bins: int,
) -> float:
    if not test_log10_e.size:
        return float("nan")
    occupied_train_bins = {
        index
        for index in _joint_bin_indices(train_log10_e, train_nu, bins=bins)
        if index is not None
    }
    test_bins = _joint_bin_indices(test_log10_e, test_nu, bins=bins)
    return float(sum(index not in occupied_train_bins for index in test_bins) / len(test_bins))


def _mahalanobis_diagnostics(
    train_records: list[dict[str, object]], test_records: list[dict[str, object]]
) -> dict[str, object]:
    empty_diagnostics: dict[str, object] = {
        "mahalanobis_feature_columns": (),
        "mahalanobis_train_p95": float("nan"),
        "mahalanobis_outside_fraction": float("nan"),
        "mahalanobis_nonfinite_test_fraction": float("nan"),
    }
    if not train_records:
        return empty_diagnostics

    feature_columns = (*PARAMETER_COLUMNS, *_STATIC_NUISANCE_COLUMNS)
    train_matrix = np.column_stack(
        [_column_values(train_records, column) for column in feature_columns]
    )
    finite_train_columns = np.isfinite(train_matrix).all(axis=0)
    train_matrix = train_matrix[:, finite_train_columns]
    selected_columns = np.asarray(feature_columns, dtype=object)[finite_train_columns]
    if not train_matrix.shape[1]:
        return empty_diagnostics

    train_mean = train_matrix.mean(axis=0)
    train_std = (
        train_matrix.std(axis=0, ddof=1)
        if len(train_matrix) > 1
        else np.zeros(train_matrix.shape[1])
    )
    varying_columns = train_std > 0
    train_matrix = train_matrix[:, varying_columns]
    train_mean = train_mean[varying_columns]
    train_std = train_std[varying_columns]
    selected_columns = selected_columns[varying_columns]
    if not train_matrix.shape[1]:
        return empty_diagnostics

    train_standardized = (train_matrix - train_mean) / train_std
    covariance = np.atleast_2d(np.cov(train_standardized, rowvar=False, ddof=1))
    covariance += 1e-6 * np.eye(covariance.shape[0])
    precision = np.linalg.inv(covariance)

    def distances(values: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", values, precision, values)

    train_distances = distances(train_standardized)
    threshold = np.percentile(train_distances, 95)
    diagnostics = {
        "mahalanobis_feature_columns": tuple(selected_columns.tolist()),
        "mahalanobis_train_p95": float(threshold),
        "mahalanobis_outside_fraction": float("nan"),
        "mahalanobis_nonfinite_test_fraction": float("nan"),
    }
    if not test_records:
        return diagnostics

    test_matrix = np.column_stack(
        [_column_values(test_records, column) for column in selected_columns]
    )
    finite_test_rows = np.isfinite(test_matrix).all(axis=1)
    diagnostics["mahalanobis_nonfinite_test_fraction"] = float(
        np.mean(~finite_test_rows)
    )
    test_outside = np.ones(len(test_matrix), dtype=bool)
    if np.any(finite_test_rows):
        test_standardized = (
            test_matrix[finite_test_rows] - train_mean
        ) / train_std
        test_outside[finite_test_rows] = distances(test_standardized) > threshold
    diagnostics["mahalanobis_outside_fraction"] = float(test_outside.mean())
    return diagnostics


def build_support_rows(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    *,
    bins: int = 5,
) -> list[dict[str, object]]:
    """Compare each material's test parameters and static nuisances to train."""
    _validate_bins(bins)
    rows: list[dict[str, object]] = []
    for material in _audit_materials(train_records, test_records):
        train_material_records = _material_records(train_records, material)
        test_material_records = _material_records(test_records, material)
        train_log10_e = _column_values(train_material_records, "log10_e")
        train_nu = _column_values(train_material_records, "nu")
        test_log10_e = _column_values(test_material_records, "log10_e")
        test_nu = _column_values(test_material_records, "nu")
        joint_empty_fraction = _joint_empty_bin_fraction(
            train_log10_e,
            train_nu,
            test_log10_e,
            test_nu,
            bins=bins,
        )
        mahalanobis_diagnostics = _mahalanobis_diagnostics(
            train_material_records,
            test_material_records,
        )
        outside_fractions: list[float] = []
        parameter_rows: list[dict[str, object]] = []
        for parameter in PARAMETER_COLUMNS:
            train_values = _column_values(train_material_records, parameter)
            test_values = _column_values(test_material_records, parameter)
            if train_values.size and test_values.size:
                outside_fraction = float(
                    np.mean((test_values < train_values.min()) | (test_values > train_values.max()))
                )
                ks_result = ks_2samp(train_values, test_values)
                ks_statistic = float(ks_result.statistic)
                ks_pvalue = float(ks_result.pvalue)
                wasserstein = float(wasserstein_distance(train_values, test_values))
            else:
                outside_fraction = float("nan")
                ks_statistic = float("nan")
                ks_pvalue = float("nan")
                wasserstein = float("nan")
            outside_fractions.append(outside_fraction)
            parameter_rows.append(
                {
                    "material": material,
                    "parameter": parameter,
                    "n_train": int(train_values.size),
                    "n_test": int(test_values.size),
                    "train_min": float(train_values.min()) if train_values.size else float("nan"),
                    "train_max": float(train_values.max()) if train_values.size else float("nan"),
                    "test_min": float(test_values.min()) if test_values.size else float("nan"),
                    "test_max": float(test_values.max()) if test_values.size else float("nan"),
                    "outside_train_fraction": outside_fraction,
                    "ks_statistic": ks_statistic,
                    "ks_pvalue": ks_pvalue,
                    "wasserstein_distance": wasserstein,
                }
            )
        out_of_support = (
            any(fraction > 0.05 for fraction in outside_fractions)
            or joint_empty_fraction > 0.20
            or mahalanobis_diagnostics["mahalanobis_outside_fraction"] > 0.20
            or mahalanobis_diagnostics["mahalanobis_nonfinite_test_fraction"] > 0
        )
        diagnostics = {
            "joint_empty_bin_fraction": joint_empty_fraction,
            "support_status": "out_of_support" if out_of_support else "in_support",
            **mahalanobis_diagnostics,
            **_standardized_mean_differences(
                train_material_records,
                test_material_records,
            ),
        }
        for parameter_row in parameter_rows:
            parameter_row.update(diagnostics)
            rows.append(parameter_row)
    return rows


def make_object_folds(
    model_names: list[str] | tuple[str, ...] | np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic object folds after sorting names before shuffling."""
    names = np.asarray(model_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("model_names must be one-dimensional")
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise ValueError("folds must be an integer greater than one")
    if folds > len(names):
        raise ValueError("folds cannot exceed the number of objects")
    normalized_names = np.asarray([str(name) for name in names], dtype=object)
    if len(set(normalized_names.tolist())) != len(normalized_names):
        raise ValueError("model_names must be unique")

    sorted_indices = np.argsort(normalized_names, kind="stable")
    shuffled_indices = np.random.default_rng(seed).permutation(sorted_indices)
    all_indices = np.arange(len(names), dtype=np.int64)
    object_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_indices in np.array_split(shuffled_indices, folds):
        test_indices = np.asarray(test_indices, dtype=np.int64)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        object_folds.append((train_indices, test_indices))
    return object_folds


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values in the input order."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0.0) | (values[finite] > 1.0)):
        raise ValueError("finite pvalues must lie in [0, 1]")
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    if not np.any(finite):
        return adjusted

    finite_indices = np.flatnonzero(finite)
    order = np.argsort(values[finite], kind="stable")
    sorted_values = values[finite][order]
    count = len(sorted_values)
    ranked = sorted_values * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(count, dtype=np.float64)
    restored[order] = np.clip(ranked, 0.0, 1.0)
    adjusted[finite_indices] = restored
    return adjusted


def _piecewise_basis(
    train_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a four-column hinge basis using train-only scaling and knots."""
    train = np.asarray(train_values, dtype=np.float64)
    evaluation = np.asarray(eval_values, dtype=np.float64)
    if train.ndim != 1 or evaluation.ndim != 1:
        raise ValueError("piecewise basis values must be one-dimensional")
    if not train.size or not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("piecewise basis values must be finite with non-empty train data")

    mean = float(train.mean())
    scale = float(train.std(ddof=0))
    if scale < _STAT_EPSILON:
        scale = 1.0
    standardized_train = (train - mean) / scale
    standardized_eval = (evaluation - mean) / scale
    knots = np.quantile(standardized_train, [0.25, 0.50, 0.75])

    def basis(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [values, *(np.maximum(values - knot, 0.0) for knot in knots)]
        )

    return basis(standardized_train), basis(standardized_eval)


def _prepare_nuisance_features(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Impute, expand, select, and scale nuisance columns from train only."""
    train = np.asarray(train_values, dtype=np.float64)
    evaluation = np.asarray(eval_values, dtype=np.float64)
    if train.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("nuisance values must be two-dimensional")
    if train.shape[1] != evaluation.shape[1]:
        raise ValueError("train and eval nuisance columns must match")
    if feature_names is None:
        names = tuple(f"nuisance_{index}" for index in range(train.shape[1]))
    else:
        names = tuple(feature_names)
        if len(names) != train.shape[1]:
            raise ValueError("feature_names must match nuisance columns")

    train_columns: list[np.ndarray] = []
    eval_columns: list[np.ndarray] = []
    selected_names: list[str] = []
    for index, name in enumerate(names):
        train_column = train[:, index]
        eval_column = evaluation[:, index]
        finite_train = np.isfinite(train_column)
        if not np.any(finite_train):
            continue
        median = float(np.median(train_column[finite_train]))
        imputed_train = np.where(finite_train, train_column, median)
        imputed_eval = np.where(np.isfinite(eval_column), eval_column, median)
        candidates = [(imputed_train, imputed_eval, name)]
        if np.any(~finite_train):
            candidates.append(
                (
                    (~finite_train).astype(np.float64),
                    (~np.isfinite(eval_column)).astype(np.float64),
                    f"{name}__missing",
                )
            )
        for candidate_train, candidate_eval, candidate_name in candidates:
            mean = float(candidate_train.mean())
            scale = float(candidate_train.std(ddof=0))
            if scale < _STAT_EPSILON:
                continue
            train_columns.append((candidate_train - mean) / scale)
            eval_columns.append((candidate_eval - mean) / scale)
            selected_names.append(candidate_name)

    if not train_columns:
        return (
            np.empty((len(train), 0), dtype=np.float64),
            np.empty((len(evaluation), 0), dtype=np.float64),
            (),
        )
    return (
        np.column_stack(train_columns),
        np.column_stack(eval_columns),
        tuple(selected_names),
    )


def _parameter_features(
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    parameters: dict[str, np.ndarray],
    augmented_parameter: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if augmented_parameter is None:
        return (
            np.empty((len(train_indices), 0), dtype=np.float64),
            np.empty((len(eval_indices), 0), dtype=np.float64),
        )
    if augmented_parameter in PARAMETER_COLUMNS:
        if augmented_parameter not in parameters:
            raise ValueError(f"missing parameter {augmented_parameter}")
        values = np.asarray(parameters[augmented_parameter], dtype=np.float64)
        return _piecewise_basis(values[train_indices], values[eval_indices])
    if augmented_parameter != "both":
        raise ValueError(f"unknown augmented_parameter {augmented_parameter}")
    if any(parameter not in parameters for parameter in PARAMETER_COLUMNS):
        raise ValueError("both parameter model requires log10_e and nu")

    e_values = np.asarray(parameters["log10_e"], dtype=np.float64)
    nu_values = np.asarray(parameters["nu"], dtype=np.float64)
    train_e, eval_e = _piecewise_basis(
        e_values[train_indices], e_values[eval_indices]
    )
    train_nu, eval_nu = _piecewise_basis(
        nu_values[train_indices], nu_values[eval_indices]
    )
    train_interaction = (train_e[:, 0] * train_nu[:, 0])[:, None]
    eval_interaction = (eval_e[:, 0] * eval_nu[:, 0])[:, None]
    return (
        np.column_stack((train_e, train_nu, train_interaction)),
        np.column_stack((eval_e, eval_nu, eval_interaction)),
    )


def _design_matrices(
    nuisance: np.ndarray,
    parameters: dict[str, np.ndarray],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    augmented_parameter: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    nuisance_train, nuisance_eval, _ = _prepare_nuisance_features(
        nuisance[train_indices], nuisance[eval_indices]
    )
    parameter_train, parameter_eval = _parameter_features(
        train_indices,
        eval_indices,
        parameters,
        augmented_parameter,
    )
    train = np.column_stack((nuisance_train, parameter_train))
    evaluation = np.column_stack((nuisance_eval, parameter_eval))
    if not train.shape[1]:
        return train, evaluation
    varying = train.std(axis=0, ddof=0) >= _STAT_EPSILON
    return train[:, varying], evaluation[:, varying]


def _ridge_prediction(
    train_features: np.ndarray,
    train_response: np.ndarray,
    eval_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    response = np.asarray(train_response, dtype=np.float64)
    response_mean = float(response.mean())
    if not train_features.shape[1]:
        return np.full(len(eval_features), response_mean, dtype=np.float64)

    feature_mean = train_features.mean(axis=0)
    centered_features = train_features - feature_mean
    centered_response = response - response_mean
    gram = centered_features.T @ centered_features
    regularized = gram + alpha * np.eye(gram.shape[0], dtype=np.float64)
    rhs = centered_features.T @ centered_response
    try:
        coefficients = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(regularized) @ rhs
    return response_mean + (eval_features - feature_mean) @ coefficients


def _inner_splits(
    outer_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer_set = set(outer_train.tolist())
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for _, candidate_validation in folds:
        validation = np.asarray(
            [index for index in candidate_validation if index in outer_set],
            dtype=np.int64,
        )
        if not validation.size:
            continue
        train = np.setdiff1d(outer_train, validation, assume_unique=True)
        if train.size:
            splits.append((train, validation))
    if len(splits) >= 2:
        return splits

    inner_count = min(max(2, len(folds) - 1), len(outer_train))
    splits = []
    for validation in np.array_split(np.asarray(outer_train), inner_count):
        train = np.setdiff1d(outer_train, validation, assume_unique=True)
        if train.size and validation.size:
            splits.append((train, np.asarray(validation, dtype=np.int64)))
    return splits


def _select_ridge_alpha(
    nuisance: np.ndarray,
    parameters: dict[str, np.ndarray],
    response: np.ndarray,
    outer_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    augmented_parameter: str | None,
) -> float:
    splits = _inner_splits(outer_train, folds)
    if not splits:
        return float(_RIDGE_ALPHAS[0])
    squared_errors = np.zeros(len(_RIDGE_ALPHAS), dtype=np.float64)
    validation_count = 0
    for inner_train, inner_validation in splits:
        train_features, validation_features = _design_matrices(
            nuisance,
            parameters,
            inner_train,
            inner_validation,
            augmented_parameter,
        )
        validation_response = response[inner_validation]
        validation_count += len(inner_validation)
        for alpha_index, alpha in enumerate(_RIDGE_ALPHAS):
            prediction = _ridge_prediction(
                train_features,
                response[inner_train],
                validation_features,
                float(alpha),
            )
            squared_errors[alpha_index] += float(
                np.sum((validation_response - prediction) ** 2)
            )
    if not validation_count:
        return float(_RIDGE_ALPHAS[0])
    return float(_RIDGE_ALPHAS[int(np.argmin(squared_errors / validation_count))])


def _nested_cv_predictions(
    nuisance: np.ndarray,
    parameters: dict[str, np.ndarray],
    response: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    augmented_parameter: str | None,
) -> np.ndarray:
    """Return pooled outer-fold predictions with inner-fold ridge tuning."""
    nuisance_values = np.asarray(nuisance, dtype=np.float64)
    response_values = np.asarray(response, dtype=np.float64)
    if nuisance_values.ndim != 2:
        raise ValueError("nuisance must be a two-dimensional array")
    if response_values.ndim != 1 or len(response_values) != len(nuisance_values):
        raise ValueError("response must align with nuisance rows")
    if not np.isfinite(response_values).all():
        raise ValueError("response must be finite")
    parameter_values = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in parameters.items()
    }
    if any(values.shape != response_values.shape for values in parameter_values.values()):
        raise ValueError("parameter arrays must align with response")

    predictions = np.full(len(response_values), np.nan, dtype=np.float64)
    held_out_counts = np.zeros(len(response_values), dtype=np.int64)
    for train_indices, test_indices in folds:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        if set(train_indices.tolist()).intersection(test_indices.tolist()):
            raise ValueError("train and held-out fold indices must be disjoint")
        alpha = _select_ridge_alpha(
            nuisance_values,
            parameter_values,
            response_values,
            train_indices,
            folds,
            augmented_parameter,
        )
        train_features, test_features = _design_matrices(
            nuisance_values,
            parameter_values,
            train_indices,
            test_indices,
            augmented_parameter,
        )
        predictions[test_indices] = _ridge_prediction(
            train_features,
            response_values[train_indices],
            test_features,
            alpha,
        )
        held_out_counts[test_indices] += 1
    if not np.all(held_out_counts == 1):
        raise ValueError("folds must hold out every object exactly once")
    return predictions


def _pooled_oof_r2(response: np.ndarray, prediction: np.ndarray) -> float:
    response_values = np.asarray(response, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    if response_values.shape != prediction_values.shape:
        raise ValueError("response and prediction shapes must match")
    total_sum_squares = float(
        np.sum((response_values - response_values.mean()) ** 2)
    )
    if total_sum_squares < _STAT_EPSILON:
        return float("nan")
    residual_sum_squares = float(np.sum((response_values - prediction_values) ** 2))
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _permutation_pvalue(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1:
        raise ValueError("null_values must be one-dimensional")
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + np.count_nonzero(null >= observed)) / (1 + len(null)))


def _bootstrap_delta_r2(
    base_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    response: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap paired pooled OOF object tuples without refitting models."""
    base = np.asarray(base_prediction, dtype=np.float64)
    augmented = np.asarray(augmented_prediction, dtype=np.float64)
    response_values = np.asarray(response, dtype=np.float64)
    if base.shape != augmented.shape or base.shape != response_values.shape:
        raise ValueError("bootstrap arrays must have matching shapes")
    if base.ndim != 1 or not len(base):
        raise ValueError("bootstrap arrays must be non-empty and one-dimensional")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("samples must be a positive integer")

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for indices in rng.integers(0, len(response_values), size=(samples, len(response_values))):
        sampled_response = response_values[indices]
        base_r2 = _pooled_oof_r2(sampled_response, base[indices])
        augmented_r2 = _pooled_oof_r2(sampled_response, augmented[indices])
        if np.isfinite(base_r2) and np.isfinite(augmented_r2):
            deltas.append(augmented_r2 - base_r2)
    if not deltas:
        return float("nan"), float("nan")
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return float(lower), float(upper)


def _derived_seed(seed: int, *parts: str) -> int:
    payload = "|".join((str(seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _validate_analysis_settings(settings: AuditSettings) -> None:
    if not isinstance(settings.folds, int) or isinstance(settings.folds, bool) or settings.folds < 2:
        raise ValueError("settings.folds must be an integer greater than one")
    if (
        not isinstance(settings.permutations, int)
        or isinstance(settings.permutations, bool)
        or settings.permutations < 1
    ):
        raise ValueError("settings.permutations must be a positive integer")
    if (
        not isinstance(settings.bootstrap_samples, int)
        or isinstance(settings.bootstrap_samples, bool)
        or settings.bootstrap_samples < 1
    ):
        raise ValueError("settings.bootstrap_samples must be a positive integer")


def _analysis_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        record
        for record in records
        if bool(record.get("valid", True)) and record.get("split", "train") == "train"
    ]
    return sorted(selected, key=lambda record: (str(record["material"]), str(record["model"])))


def _analysis_materials(records: list[dict[str, object]]) -> tuple[str, ...]:
    observed = {str(record["material"]) for record in records}
    known = tuple(name for name in MATERIAL_NAMES.values() if name in observed)
    return (*known, *sorted(observed.difference(known)))


def _nuisance_matrix(records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            [float(record.get(column, float("nan"))) for column in _STATIC_NUISANCE_COLUMNS]
            for record in records
        ],
        dtype=np.float64,
    )


def _fitted_nuisance_names(
    nuisance: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[str, ...]:
    selected: set[str] = set()
    empty_eval = np.empty((0, nuisance.shape[1]), dtype=np.float64)
    for train_indices, _ in folds:
        _, _, names = _prepare_nuisance_features(
            nuisance[train_indices],
            empty_eval,
            feature_names=_STATIC_NUISANCE_COLUMNS,
        )
        selected.update(names)
    ordered_names = tuple(
        name
        for column in _STATIC_NUISANCE_COLUMNS
        for name in (column, f"{column}__missing")
        if name in selected
    )
    return ordered_names


def _safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x_values = np.asarray(x[finite], dtype=np.float64)
    y_values = np.asarray(y[finite], dtype=np.float64)
    if (
        len(x_values) < 2
        or np.ptp(x_values) < _STAT_EPSILON
        or np.ptp(y_values) < _STAT_EPSILON
    ):
        return len(x_values), float("nan"), float("nan")
    return (
        len(x_values),
        float(np.corrcoef(x_values, y_values)[0, 1]),
        float(spearmanr(x_values, y_values)[0]),
    )


def analyze_confounding(
    records: list[dict[str, object]],
    settings: AuditSettings,
) -> list[dict[str, object]]:
    """Audit whether static nuisance features predict each material parameter."""
    _validate_analysis_settings(settings)
    selected_records = _analysis_records(records)
    rows: list[dict[str, object]] = []
    for material in _analysis_materials(selected_records):
        material_records = [
            record for record in selected_records if record["material"] == material
        ]
        for parameter in PARAMETER_COLUMNS:
            parameter_records = [
                record
                for record in material_records
                if np.isfinite(float(record[parameter]))
            ]
            parameter_values = _column_values(parameter_records, parameter)
            nuisance = _nuisance_matrix(parameter_records)
            for feature_index, feature in enumerate(_STATIC_NUISANCE_COLUMNS):
                pair_n, pearson, spearman = _safe_correlations(
                    parameter_values,
                    nuisance[:, feature_index] if len(nuisance) else np.asarray([]),
                )
                rows.append(
                    {
                        "row_type": "correlation",
                        "material": material,
                        "parameter": parameter,
                        "feature": feature,
                        "feature_names": (feature,),
                        "fitted_features": (),
                        "n": int(len(parameter_records)),
                        "pair_n": int(pair_n),
                        "pearson": pearson,
                        "spearman": spearman,
                        "cv_r2": float("nan"),
                        "permutation_p": float("nan"),
                        "confounded": False,
                        "status": "correlation",
                        "seed": settings.seed,
                        "folds": min(settings.folds, len(parameter_records)),
                    }
                )

            if len(parameter_records) < 2:
                rows.append(
                    {
                        "row_type": "summary",
                        "material": material,
                        "parameter": parameter,
                        "feature": "all_nuisance",
                        "feature_names": _STATIC_NUISANCE_COLUMNS,
                        "fitted_features": (),
                        "n": int(len(parameter_records)),
                        "pair_n": int(len(parameter_records)),
                        "pearson": float("nan"),
                        "spearman": float("nan"),
                        "cv_r2": float("nan"),
                        "permutation_p": 1.0,
                        "confounded": False,
                        "status": "insufficient_data",
                        "seed": settings.seed,
                        "folds": min(settings.folds, len(parameter_records)),
                    }
                )
                continue

            effective_folds = min(settings.folds, len(parameter_records))
            folds = make_object_folds(
                [str(record["model"]) for record in parameter_records],
                folds=effective_folds,
                seed=settings.seed,
            )
            fitted_features = _fitted_nuisance_names(nuisance, folds)
            if np.sum((parameter_values - parameter_values.mean()) ** 2) < _STAT_EPSILON:
                cv_r2 = float("nan")
                permutation_p = 1.0
                status = "constant_parameter"
            else:
                prediction = _nested_cv_predictions(
                    nuisance,
                    {},
                    parameter_values,
                    folds,
                    augmented_parameter=None,
                )
                cv_r2 = _pooled_oof_r2(parameter_values, prediction)
                rng = np.random.default_rng(
                    _derived_seed(settings.seed, "confounding", material, parameter)
                )
                null_values = np.empty(settings.permutations, dtype=np.float64)
                for permutation_index in range(settings.permutations):
                    permuted = rng.permutation(parameter_values)
                    null_prediction = _nested_cv_predictions(
                        nuisance,
                        {},
                        permuted,
                        folds,
                        augmented_parameter=None,
                    )
                    null_values[permutation_index] = _pooled_oof_r2(
                        permuted, null_prediction
                    )
                permutation_p = _permutation_pvalue(cv_r2, null_values)
                status = "ok"
            confounded = bool(cv_r2 > 0.05 and permutation_p < 0.05)
            rows.append(
                {
                    "row_type": "summary",
                    "material": material,
                    "parameter": parameter,
                    "feature": "all_nuisance",
                    "feature_names": _STATIC_NUISANCE_COLUMNS,
                    "fitted_features": fitted_features,
                    "n": int(len(parameter_records)),
                    "pair_n": int(len(parameter_records)),
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "cv_r2": cv_r2,
                    "permutation_p": permutation_p,
                    "confounded": confounded,
                    "status": status,
                    "seed": settings.seed,
                    "folds": effective_folds,
                }
            )
    return rows


def _empty_response_row(
    *,
    material: str,
    parameter: str,
    response: str,
    n: int,
    folds: int,
    settings: AuditSettings,
    status: str,
) -> dict[str, object]:
    constant = status == "constant_response"
    return {
        "material": material,
        "parameter": parameter,
        "response": response,
        "response_tier": "primary" if response in PRIMARY_RESPONSE_COLUMNS else "secondary",
        "n": n,
        "seed": settings.seed,
        "folds": folds,
        "nuisance_features": _STATIC_NUISANCE_COLUMNS,
        "fitted_features": (),
        "r2_m0": float("nan"),
        "r2_me": float("nan"),
        "r2_mnu": float("nan"),
        "r2_mboth": float("nan"),
        "r2_augmented": float("nan"),
        "delta_r2": 0.0 if constant else float("nan"),
        "partial_spearman": float("nan"),
        "permutation_p": 1.0,
        "bootstrap_ci_low": 0.0 if constant else float("nan"),
        "bootstrap_ci_high": 0.0 if constant else float("nan"),
        "q_value": 1.0,
        "status": status,
    }


def analyze_responses(
    records: list[dict[str, object]],
    settings: AuditSettings,
) -> list[dict[str, object]]:
    """Estimate material-local incremental parameter signal in GT responses."""
    _validate_analysis_settings(settings)
    selected_records = _analysis_records(records)
    rows: list[dict[str, object]] = []
    for material in _analysis_materials(selected_records):
        material_records = [
            record for record in selected_records if record["material"] == material
        ]
        available_responses = [
            response
            for response in RESPONSE_COLUMNS
            if any(response in record for record in material_records)
        ]
        for response_name in available_responses:
            response_records = [
                record
                for record in material_records
                if response_name in record
                and np.isfinite(float(record[response_name]))
                and all(
                    np.isfinite(float(record[parameter]))
                    for parameter in PARAMETER_COLUMNS
                )
            ]
            n = len(response_records)
            effective_folds = min(settings.folds, n)
            if n < 2:
                for parameter in PARAMETER_COLUMNS:
                    rows.append(
                        _empty_response_row(
                            material=material,
                            parameter=parameter,
                            response=response_name,
                            n=n,
                            folds=effective_folds,
                            settings=settings,
                            status="insufficient_data",
                        )
                    )
                continue

            response_values = _column_values(response_records, response_name)
            if np.sum((response_values - response_values.mean()) ** 2) < _STAT_EPSILON:
                for parameter in PARAMETER_COLUMNS:
                    rows.append(
                        _empty_response_row(
                            material=material,
                            parameter=parameter,
                            response=response_name,
                            n=n,
                            folds=effective_folds,
                            settings=settings,
                            status="constant_response",
                        )
                    )
                continue

            nuisance = _nuisance_matrix(response_records)
            parameters = {
                parameter: _column_values(response_records, parameter)
                for parameter in PARAMETER_COLUMNS
            }
            folds = make_object_folds(
                [str(record["model"]) for record in response_records],
                folds=effective_folds,
                seed=settings.seed,
            )
            fitted_features = _fitted_nuisance_names(nuisance, folds)
            predictions = {
                "m0": _nested_cv_predictions(
                    nuisance,
                    parameters,
                    response_values,
                    folds,
                    augmented_parameter=None,
                ),
                "me": _nested_cv_predictions(
                    nuisance,
                    parameters,
                    response_values,
                    folds,
                    augmented_parameter="log10_e",
                ),
                "mnu": _nested_cv_predictions(
                    nuisance,
                    parameters,
                    response_values,
                    folds,
                    augmented_parameter="nu",
                ),
                "mboth": _nested_cv_predictions(
                    nuisance,
                    parameters,
                    response_values,
                    folds,
                    augmented_parameter="both",
                ),
            }
            model_r2 = {
                name: _pooled_oof_r2(response_values, prediction)
                for name, prediction in predictions.items()
            }
            for parameter in PARAMETER_COLUMNS:
                augmented_model = "me" if parameter == "log10_e" else "mnu"
                augmented_prediction = predictions[augmented_model]
                delta_r2 = model_r2[augmented_model] - model_r2["m0"]
                parameter_prediction = _nested_cv_predictions(
                    nuisance,
                    {},
                    parameters[parameter],
                    folds,
                    augmented_parameter=None,
                )
                _, _, partial_spearman = _safe_correlations(
                    parameters[parameter] - parameter_prediction,
                    response_values - predictions["m0"],
                )
                rng = np.random.default_rng(
                    _derived_seed(
                        settings.seed,
                        "response",
                        material,
                        parameter,
                        response_name,
                        "permutation",
                    )
                )
                null_values = np.empty(settings.permutations, dtype=np.float64)
                for permutation_index in range(settings.permutations):
                    permuted_parameters = dict(parameters)
                    permuted_parameters[parameter] = rng.permutation(
                        parameters[parameter]
                    )
                    null_prediction = _nested_cv_predictions(
                        nuisance,
                        permuted_parameters,
                        response_values,
                        folds,
                        augmented_parameter=parameter,
                    )
                    null_values[permutation_index] = (
                        _pooled_oof_r2(response_values, null_prediction)
                        - model_r2["m0"]
                    )
                permutation_p = _permutation_pvalue(delta_r2, null_values)
                bootstrap_seed = _derived_seed(
                    settings.seed,
                    "response",
                    material,
                    parameter,
                    response_name,
                    "bootstrap",
                )
                bootstrap_ci_low, bootstrap_ci_high = _bootstrap_delta_r2(
                    predictions["m0"],
                    augmented_prediction,
                    response_values,
                    samples=settings.bootstrap_samples,
                    seed=bootstrap_seed,
                )
                rows.append(
                    {
                        "material": material,
                        "parameter": parameter,
                        "response": response_name,
                        "response_tier": "primary"
                        if response_name in PRIMARY_RESPONSE_COLUMNS
                        else "secondary",
                        "n": n,
                        "seed": settings.seed,
                        "folds": effective_folds,
                        "nuisance_features": _STATIC_NUISANCE_COLUMNS,
                        "fitted_features": fitted_features,
                        "r2_m0": model_r2["m0"],
                        "r2_me": model_r2["me"],
                        "r2_mnu": model_r2["mnu"],
                        "r2_mboth": model_r2["mboth"],
                        "r2_augmented": model_r2[augmented_model],
                        "delta_r2": delta_r2,
                        "partial_spearman": partial_spearman,
                        "permutation_p": permutation_p,
                        "bootstrap_ci_low": bootstrap_ci_low,
                        "bootstrap_ci_high": bootstrap_ci_high,
                        "q_value": float("nan"),
                        "status": "ok",
                    }
                )

    families = {
        (str(row["material"]), str(row["parameter"]))
        for row in rows
    }
    for material, parameter in sorted(families):
        family_indices = [
            index
            for index, row in enumerate(rows)
            if row["material"] == material and row["parameter"] == parameter
        ]
        q_values = benjamini_hochberg(
            np.asarray(
                [float(rows[index]["permutation_p"]) for index in family_indices],
                dtype=np.float64,
            )
        )
        for index, q_value in zip(family_indices, q_values):
            rows[index]["q_value"] = float(q_value)
    return rows


def read_h5_record(
    path: Path,
    *,
    split: str,
    settings: AuditSettings,
) -> dict[str, object]:
    """Read one audit row from an H5 trajectory record."""
    if split not in {"train", "test"}:
        raise RecordValidationError("split must be 'train' or 'test'")

    path = Path(path)
    with h5py.File(path, "r") as handle:
        _require_fields(handle, _STATIC_REQUIRED_FIELDS)
        e_value, nu, mat_type = _validate_metadata(handle)
        _validate_x_dataset(handle["x"])
        initial_points = np.asarray(handle["x"][0], dtype=np.float64)
        _validate_initial_points(initial_points)
        record = _static_features(
            handle,
            initial_points,
            e_value=e_value,
            nu=nu,
            mat_type=mat_type,
            split=split,
            settings=settings,
            model=path.name,
        )
        if split == "test":
            return record

        _require_fields(handle, _TRAIN_DYNAMIC_FIELDS)
        x = np.asarray(handle["x"][:], dtype=np.float64)
        v = np.asarray(handle["v"][:], dtype=np.float64)
        f = np.asarray(handle["F"][:], dtype=np.float64)
        c = np.asarray(handle["C"][:], dtype=np.float64)
        _validate_train_dynamics(x, v, f, c)
        record.update(
            _train_responses(
                x,
                v,
                f,
                c,
                floor_height=_scalar(handle, "floor_height"),
                settings=settings,
            )
        )
        return record


def _as_finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _validate_no_row_invalid_records(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if "invalid_record_count" in row:
            count = _as_finite_float(row.get("invalid_record_count"))
            if count is None or count != 0.0:
                raise ValueError(
                    "invalid file counts must come from metadata.invalid_records"
                )
        if row.get("invalid_records"):
            raise ValueError(
                "invalid file details must come from metadata.invalid_records"
            )


def _ordered_material_parameter_pairs(
    response_rows: list[dict[str, object]],
    confounding_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
) -> list[tuple[str, str]]:
    observed_pairs = {
        (str(row["material"]), str(row["parameter"]))
        for rows in (response_rows, confounding_rows, support_rows)
        for row in rows
        if "material" in row and "parameter" in row
    }
    unexpected = sorted(observed_pairs - set(_EXPECTED_SUMMARY_KEYS))
    if unexpected:
        raise ValueError(f"unexpected material-parameter rows: {unexpected}")
    return list(_EXPECTED_SUMMARY_KEYS)


def _qualifies_primary_response(row: dict[str, object]) -> bool:
    if row.get("response_tier") != "primary" or row.get("status", "ok") != "ok":
        return False
    delta_r2 = _as_finite_float(row.get("delta_r2"))
    permutation_p = _as_finite_float(row.get("permutation_p"))
    q_value = _as_finite_float(row.get("q_value"))
    ci_low = _as_finite_float(row.get("bootstrap_ci_low"))
    return bool(
        delta_r2 is not None
        and delta_r2 >= 0.05
        and permutation_p is not None
        and permutation_p < 0.05
        and q_value is not None
        and q_value < 0.05
        and ci_low is not None
        and ci_low > 0.0
    )


def _has_weak_response_evidence(row: dict[str, object]) -> bool:
    if row.get("status", "ok") != "ok":
        return False
    delta_r2 = _as_finite_float(row.get("delta_r2"))
    if delta_r2 is not None and delta_r2 >= 0.01:
        return True
    permutation_p = _as_finite_float(row.get("permutation_p"))
    q_value = _as_finite_float(row.get("q_value"))
    ci_low = _as_finite_float(row.get("bootstrap_ci_low"))
    return bool(
        (permutation_p is not None and permutation_p < 0.05)
        or (q_value is not None and q_value < 0.05)
        or (ci_low is not None and ci_low > 0.0)
    )


_PRIMARY_CLASSIFICATION_FIELDS = (
    "delta_r2",
    "partial_spearman",
    "permutation_p",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "q_value",
)


def _classification_completeness_reasons(
    response_rows: list[dict[str, object]],
    confounding_rows: list[dict[str, object]],
) -> list[str]:
    reasons: list[str] = []
    primary_rows = [
        row
        for row in response_rows
        if row.get("response_tier") == "primary"
        and row.get("response") in PRIMARY_RESPONSE_COLUMNS
    ]
    primary_counts = {
        response: sum(row.get("response") == response for row in primary_rows)
        for response in PRIMARY_RESPONSE_COLUMNS
    }
    if any(count == 0 for count in primary_counts.values()):
        reasons.append("missing_primary_responses")
    if any(count > 1 for count in primary_counts.values()):
        reasons.append("duplicate_primary_responses")
    if any(
        row.get("status") != "ok"
        or any(_as_finite_float(row.get(field)) is None for field in _PRIMARY_CLASSIFICATION_FIELDS)
        for row in primary_rows
    ):
        reasons.append("invalid_primary_statistics")

    if not confounding_rows:
        reasons.append("missing_confounding_summary")
    elif len(confounding_rows) > 1:
        reasons.append("duplicate_confounding_summary")
    else:
        confounding = confounding_rows[0]
        if (
            confounding.get("status") != "ok"
            or _as_finite_float(confounding.get("cv_r2")) is None
            or _as_finite_float(confounding.get("permutation_p")) is None
            or not isinstance(confounding.get("confounded"), (bool, np.bool_))
        ):
            reasons.append("invalid_confounding_statistics")
    return reasons


def classify_identifiability(
    response_rows: list[dict[str, object]],
    confounding_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Classify material-local parameter evidence without conflating support shift."""
    _validate_no_row_invalid_records(
        [*response_rows, *confounding_rows, *support_rows]
    )
    summary: list[dict[str, object]] = []
    for material, parameter in _ordered_material_parameter_pairs(
        response_rows, confounding_rows, support_rows
    ):
        response_subset = [
            row
            for row in response_rows
            if row.get("material") == material and row.get("parameter") == parameter
        ]
        confounding_subset = [
            row
            for row in confounding_rows
            if row.get("material") == material
            and row.get("parameter") == parameter
            and row.get("row_type") == "summary"
        ]
        support_subset = [
            row
            for row in support_rows
            if row.get("material") == material and row.get("parameter") == parameter
        ]
        completeness_reasons = _classification_completeness_reasons(
            response_subset,
            confounding_subset,
        )
        confounded = any(
            bool(row.get("confounded", False)) for row in confounding_subset
        )
        qualifying_primary = [
            row for row in response_subset if _qualifies_primary_response(row)
        ]
        weak_evidence = [
            row for row in response_subset if _has_weak_response_evidence(row)
        ]
        support_status = (
            str(support_subset[0].get("support_status", "unknown"))
            if support_subset
            else "unknown"
        )
        reason_codes: list[str] = []
        if completeness_reasons:
            status = "invalid"
            reason_codes.extend(completeness_reasons)
        elif confounded:
            status = "confounded"
            reason_codes.append("nuisance_predictable")
        elif qualifying_primary:
            status = "identifiable"
            reason_codes.extend(
                (
                    "primary_delta_r2",
                    "primary_permutation_significant",
                    "primary_fdr_significant",
                    "primary_bootstrap_positive",
                )
            )
        elif weak_evidence:
            status = "weak"
            if any(
                (_as_finite_float(row.get("delta_r2")) or float("-inf")) >= 0.01
                for row in weak_evidence
            ):
                reason_codes.append("response_delta_r2")
            if any(
                _as_finite_float(row.get("permutation_p")) is not None
                and _as_finite_float(row.get("permutation_p")) < 0.05
                or _as_finite_float(row.get("q_value")) is not None
                and _as_finite_float(row.get("q_value")) < 0.05
                or _as_finite_float(row.get("bootstrap_ci_low")) is not None
                and _as_finite_float(row.get("bootstrap_ci_low")) > 0.0
                for row in weak_evidence
            ):
                reason_codes.append("partial_significance")
        else:
            status = "not_detected"
            reason_codes.append("no_detectable_response")
        if support_status == "out_of_support":
            reason_codes.append("test_parameter_extrapolation")
        elif support_status == "in_support":
            reason_codes.append("test_parameter_in_support")
        else:
            reason_codes.append("test_support_unknown")
        summary.append(
            {
                "material": material,
                "parameter": parameter,
                "status": status,
                "support_status": support_status,
                "reason_codes": tuple(dict.fromkeys(reason_codes)),
                "invalid_record_count": 0,
            }
        )
    return summary


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _csv_value(value: object) -> str | int | float:
    value = _json_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _coverage_output_rows(
    coverage_rows: list[dict[str, object]], support_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        {**row, "row_type": "distribution"} for row in coverage_rows
    ] + [{**row, "row_type": "support"} for row in support_rows]


def _markdown_cell(value: object) -> str:
    value = _json_value(value)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|")


def _markdown_table(columns: tuple[str, ...], rows: list[dict[str, object]]) -> str:
    if not rows:
        return "无可用数据。"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _metadata_invalid_records(metadata: dict) -> list[object]:
    invalid_records = metadata.get("invalid_records", [])
    if not isinstance(invalid_records, list):
        raise TypeError("metadata.invalid_records must be a list")
    return invalid_records


def _summary_reason_codes(row: dict[str, object]) -> list[str]:
    reason_codes = row.get("reason_codes", ())
    if isinstance(reason_codes, str):
        return [reason_codes]
    if isinstance(reason_codes, (tuple, list)):
        return [str(code) for code in reason_codes]
    raise TypeError("summary reason_codes must be a string, tuple, or list")


def _validate_and_order_summary_rows(
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in summary_rows:
        key = (str(row.get("material", "")), str(row.get("parameter", "")))
        rows_by_key.setdefault(key, []).append(row)
    expected_keys = set(_EXPECTED_SUMMARY_KEYS)
    missing = [key for key in _EXPECTED_SUMMARY_KEYS if key not in rows_by_key]
    duplicate = [
        key for key, rows in rows_by_key.items() if key in expected_keys and len(rows) > 1
    ]
    unexpected = [key for key in rows_by_key if key not in expected_keys]
    problems: list[str] = []
    if missing:
        problems.append(f"missing={missing}")
    if duplicate:
        problems.append(f"duplicate={duplicate}")
    if unexpected:
        problems.append(f"unexpected={unexpected}")
    if problems:
        raise ValueError(
            "summary must contain six unique material-parameter decisions: "
            + "; ".join(problems)
        )
    return [dict(rows_by_key[key][0]) for key in _EXPECTED_SUMMARY_KEYS]


def _apply_metadata_invalid_records(
    summary_rows: list[dict[str, object]],
    metadata: dict,
) -> list[dict[str, object]]:
    summary_rows = _validate_and_order_summary_rows(summary_rows)
    invalid_record_count = len(_metadata_invalid_records(metadata))
    prepared_rows: list[dict[str, object]] = []
    for row in summary_rows:
        existing_count = _as_finite_float(row.get("invalid_record_count", 0))
        if (
            existing_count is None
            or existing_count < 0
            or not float(existing_count).is_integer()
            or int(existing_count) not in {0, invalid_record_count}
        ):
            raise ValueError(
                "summary invalid_record_count conflicts with "
                "metadata.invalid_records"
            )
        reason_codes = _summary_reason_codes(row)
        if invalid_record_count == 0 and "invalid_records" in reason_codes:
            raise ValueError(
                "summary invalid_records reason conflicts with "
                "metadata.invalid_records"
            )
        prepared = dict(row)
        prepared["invalid_record_count"] = invalid_record_count
        if invalid_record_count:
            prepared["status"] = "invalid"
            reason_codes.append("invalid_records")
        prepared["reason_codes"] = tuple(dict.fromkeys(reason_codes))
        prepared_rows.append(prepared)
    return prepared_rows


def _empty_test_response(value: object) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return isinstance(value, float) and not math.isfinite(value)


def _validate_test_records_have_no_responses(
    records: list[dict[str, object]],
) -> None:
    for record in records:
        if record.get("split") != "test":
            continue
        for response in RESPONSE_COLUMNS:
            if response in record and not _empty_test_response(record[response]):
                raise ValueError(
                    f"test record {record.get('model', '<unknown>')} contains "
                    f"response column {response}"
                )


def render_markdown_report(
    summary_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    confounding_rows: list[dict[str, object]],
    response_rows: list[dict[str, object]],
    metadata: dict,
) -> str:
    """Render a Chinese research-decision report from precomputed audit rows."""
    summary_rows = _apply_metadata_invalid_records(summary_rows, metadata)
    summary_table = _markdown_table(
        ("material", "parameter", "status", "support_status", "reason_codes"),
        summary_rows,
    )
    support_table = _markdown_table(
        (
            "material",
            "parameter",
            "outside_train_fraction",
            "joint_empty_bin_fraction",
            "mahalanobis_outside_fraction",
            "support_status",
        ),
        support_rows,
    )
    confounding_summary = [
        row for row in confounding_rows if row.get("row_type", "summary") == "summary"
    ]
    confounding_table = _markdown_table(
        ("material", "parameter", "cv_r2", "permutation_p", "confounded", "status"),
        confounding_summary,
    )
    primary_responses = [
        row for row in response_rows if row.get("response_tier") == "primary"
    ]
    response_table = _markdown_table(
        (
            "material",
            "parameter",
            "response",
            "delta_r2",
            "permutation_p",
            "q_value",
            "bootstrap_ci_low",
            "status",
        ),
        primary_responses,
    )
    invalid_count = len(_metadata_invalid_records(metadata))
    return "\n".join(
        (
            "# B0.2 材质参数可辨识性审计",
            "",
            "## 审计边界",
            "本审计只在各材质内部分析 `log10(E)` 与 `nu`，train 使用 GT 动力学响应，test 只用于参数和静态 nuisance 的 support 检查。",
            "该 observational audit 不能证明反事实物理正确；三种材质来自不同 UID 区间，不能构成配对反事实样本。",
            "报告不使用 test 动力学，也不以 test 轨迹响应选择模型。",
            "",
            "## 生成协议事实",
            "elastic 的 drag force 数量为 1；plasticine 和 sand 使用 `np.random.randint(0, 1)`，其结果恒为 0。",
            "因此材质类别携带外力和地板场景差异，跨材质比较只用于描述协议混杂，不作为连续参数效应证据。",
            "",
            "## 可辨识性裁决",
            summary_table,
            "",
            "`status` 只描述 train 内统计证据；`support_status` 独立描述 test 是否位于 train 支持范围内。`invalid` 表示存在未被静默忽略的无效记录，不能据此作出参数结论。",
            "",
            "## Train/Test Support",
            support_table,
            "",
            "## Nuisance 混杂",
            confounding_table,
            "",
            "## 主要 GT 响应",
            response_table,
            "",
            "## 数据完整性与下一步",
            f"invalid records: {invalid_count}。无效记录必须先定位并处理，再依据对应材质和参数的状态安排后续实验。",
            "若参数被判为 confounded，应优先修订数据生成协议；若 train 内有信号但 test 为 out_of_support，应先修订 split/coverage；若无可检测信号，应改变场景、载荷或参数采样，而不是把结果解释为模型条件注入失败。",
        )
    ) + "\n"


def _validate_output_targets(output_dir: Path, target_paths: dict[str, Path]) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"audit output path is not a directory: {output_dir}")
    for target_path in target_paths.values():
        if target_path.exists() and not target_path.is_file():
            raise IsADirectoryError(
                f"audit output target is not a regular file: {target_path}"
            )


def _activate_output_files(
    temporary_dir: Path,
    target_paths: dict[str, Path],
) -> None:
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{temporary_dir.name}.backup.",
            dir=temporary_dir.parent,
        )
    )
    backups: dict[str, Path] = {}
    activated: list[tuple[str, Path]] = []
    try:
        for key, target_path in target_paths.items():
            if target_path.exists():
                if not target_path.is_file():
                    raise IsADirectoryError(
                        f"audit output target is not a regular file: {target_path}"
                    )
                backup_path = backup_dir / OUTPUT_NAMES[key]
                target_path.replace(backup_path)
                backups[key] = backup_path

        for key, target_path in target_paths.items():
            (temporary_dir / OUTPUT_NAMES[key]).replace(target_path)
            activated.append((key, target_path))
    except Exception as activation_error:
        rollback_errors: list[OSError] = []
        for key, target_path in reversed(activated):
            try:
                target_path.replace(temporary_dir / OUTPUT_NAMES[key])
            except OSError as error:
                rollback_errors.append(error)
        for key, backup_path in backups.items():
            try:
                backup_path.replace(target_paths[key])
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                f"audit output activation failed and rollback was incomplete; "
                f"backups remain in {backup_dir}"
            ) from activation_error
        _cleanup_transaction_directory(backup_dir)
        raise
    else:
        _cleanup_transaction_directory(backup_dir)


def _cleanup_transaction_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def write_audit_outputs(
    output_dir: Path,
    *,
    records: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    confounding_rows: list[dict[str, object]],
    response_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    metadata: dict,
    overwrite: bool,
) -> dict[str, Path]:
    """Render all B0.2 artifacts before atomically activating any target file."""
    output_dir = Path(output_dir)
    target_paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    if not overwrite:
        existing = [path for path in target_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite existing audit output: {existing[0]}"
            )
    _validate_output_targets(output_dir, target_paths)
    _validate_test_records_have_no_responses(records)
    summary_rows = _apply_metadata_invalid_records(summary_rows, metadata)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_csv(
            temporary_dir / OUTPUT_NAMES["records"],
            (*STATIC_COLUMNS, *RESPONSE_COLUMNS),
            records,
        )
        _write_csv(
            temporary_dir / OUTPUT_NAMES["coverage"],
            _COVERAGE_COLUMNS,
            _coverage_output_rows(coverage_rows, support_rows),
        )
        _write_csv(
            temporary_dir / OUTPUT_NAMES["confounding"],
            _CONFOUNDING_COLUMNS,
            confounding_rows,
        )
        _write_csv(
            temporary_dir / OUTPUT_NAMES["response"],
            _RESPONSE_OUTPUT_COLUMNS,
            response_rows,
        )
        _write_csv(
            temporary_dir / OUTPUT_NAMES["summary"], _SUMMARY_COLUMNS, summary_rows
        )
        with (temporary_dir / OUTPUT_NAMES["metadata"]).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                _json_value(metadata),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        (temporary_dir / OUTPUT_NAMES["report"]).write_text(
            render_markdown_report(
                summary_rows,
                coverage_rows,
                support_rows,
                confounding_rows,
                response_rows,
                metadata,
            ),
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        _activate_output_files(temporary_dir, target_paths)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return target_paths
