"""H5 record extraction for the material identifiability audit."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import ConvexHull, QhullError


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
