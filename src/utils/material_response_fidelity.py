"""Position-only response metrics shared by B0.3 prediction and GT paths."""

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


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
