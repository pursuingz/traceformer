"""H5 record extraction for the material identifiability audit."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
FRAME_INDICES = (5, 10, 15, 20, 24)


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

PARAMETER_COLUMNS = ("log10_e", "nu")
_STATIC_NUISANCE_COLUMNS = tuple(
    column for column in NUISANCE_COLUMNS if column not in PARAMETER_COLUMNS
)
_LOG10_E_RANGE = (4.0, 7.0)
_NU_RANGE = (0.05, 0.45)

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


def _mahalanobis_outside_fraction(
    train_records: list[dict[str, object]], test_records: list[dict[str, object]]
) -> float:
    if not train_records or not test_records:
        return float("nan")
    feature_columns = (*PARAMETER_COLUMNS, *_STATIC_NUISANCE_COLUMNS)
    train_matrix = np.column_stack(
        [_column_values(train_records, column) for column in feature_columns]
    )
    test_matrix = np.column_stack(
        [_column_values(test_records, column) for column in feature_columns]
    )
    finite_columns = np.isfinite(train_matrix).all(axis=0) & np.isfinite(test_matrix).all(axis=0)
    train_matrix = train_matrix[:, finite_columns]
    test_matrix = test_matrix[:, finite_columns]
    if not train_matrix.shape[1]:
        return float("nan")

    train_mean = train_matrix.mean(axis=0)
    train_std = train_matrix.std(axis=0, ddof=1) if len(train_matrix) > 1 else np.zeros(train_matrix.shape[1])
    varying_columns = train_std > 0
    train_matrix = train_matrix[:, varying_columns]
    test_matrix = test_matrix[:, varying_columns]
    train_mean = train_mean[varying_columns]
    train_std = train_std[varying_columns]
    if not train_matrix.shape[1]:
        return float("nan")

    train_standardized = (train_matrix - train_mean) / train_std
    test_standardized = (test_matrix - train_mean) / train_std
    covariance = np.atleast_2d(np.cov(train_standardized, rowvar=False, ddof=1))
    covariance += 1e-6 * np.eye(covariance.shape[0])
    precision = np.linalg.inv(covariance)

    def distances(values: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", values, precision, values)

    train_distances = distances(train_standardized)
    test_distances = distances(test_standardized)
    threshold = np.percentile(train_distances, 95)
    return float(np.mean(test_distances > threshold))


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
        mahalanobis_fraction = _mahalanobis_outside_fraction(
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
            or mahalanobis_fraction > 0.20
        )
        diagnostics = {
            "joint_empty_bin_fraction": joint_empty_fraction,
            "mahalanobis_outside_fraction": mahalanobis_fraction,
            "support_status": "out_of_support" if out_of_support else "in_support",
            **_standardized_mean_differences(
                train_material_records,
                test_material_records,
            ),
        }
        for parameter_row in parameter_rows:
            parameter_row.update(diagnostics)
            rows.append(parameter_row)
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
