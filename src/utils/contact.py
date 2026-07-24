"""Contact-aware features, losses, and temporal sampling helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def _floor_for_points(floor_height: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    floor = torch.as_tensor(floor_height, device=points.device, dtype=points.dtype)
    if floor.numel() != points.shape[0]:
        raise ValueError(
            f"floor_height must contain one value per batch item; got {tuple(floor.shape)} "
            f"for batch size {points.shape[0]}"
        )
    return floor.reshape(points.shape[0], 1, 1, 1)


def build_contact_features(
    points: torch.Tensor,
    floor_height: torch.Tensor,
    start_velocity: Optional[torch.Tensor] = None,
    sigma: float = 0.04,
) -> torch.Tensor:
    """Build per-particle ``[signed gap, vertical velocity, proximity]`` features.

    ``points`` and ``floor_height`` must already be in the same normalized coordinate
    system. Proximity is one for penetrating particles and decays above the floor.
    """
    if points.ndim != 4 or points.shape[-1] < 3:
        raise ValueError(f"points must have shape (B,F,N,3+); got {tuple(points.shape)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")

    floor = _floor_for_points(floor_height, points)
    signed_gap = points[..., 1:2] - floor

    vertical_velocity = torch.zeros_like(signed_gap)
    if points.shape[1] > 1:
        vertical_velocity[:, 1:] = points[:, 1:, :, 1:2] - points[:, :-1, :, 1:2]

    if start_velocity is not None:
        velocity = start_velocity
        if velocity.ndim == 4:
            if velocity.shape[1] != 1:
                raise ValueError(
                    "4D start_velocity must have a singleton frame dimension; "
                    f"got {tuple(velocity.shape)}"
                )
            velocity = velocity[:, 0]
        if velocity.ndim != 3 or velocity.shape[:2] != points.shape[:1] + points.shape[2:3]:
            raise ValueError(
                f"start_velocity must have shape (B,N,3); got {tuple(velocity.shape)}"
            )
        vertical_velocity[:, 0] = velocity[..., 1:2].to(vertical_velocity)
    elif points.shape[1] > 1:
        vertical_velocity[:, 0] = vertical_velocity[:, 1]

    scaled_gap = torch.relu(signed_gap) / float(sigma)
    proximity = torch.exp(-(scaled_gap * scaled_gap))
    return torch.cat([signed_gap, vertical_velocity, proximity], dim=-1)


def apply_contact_feature_mask(
    features: torch.Tensor,
    feature_mask=(1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply a shared ``[gap, velocity, proximity]`` ablation mask."""
    if features.shape[-1] != 3:
        raise ValueError(
            f"contact features must have 3 channels; got {features.shape[-1]}"
        )
    mask = torch.as_tensor(
        feature_mask,
        device=features.device,
        dtype=features.dtype,
    )
    if mask.numel() != 3:
        raise ValueError(
            f"contact_feature_mask must contain 3 values; got {mask.numel()}"
        )
    return features * mask.reshape(1, 1, 1, 3)


def contact_channel_contributions(
    features: torch.Tensor,
    encoder_weight: torch.Tensor,
) -> torch.Tensor:
    """Estimate each input channel's mean hidden-vector magnitude."""
    if features.shape[-1] != 3:
        raise ValueError(
            f"contact features must have 3 channels; got {features.shape[-1]}"
        )
    if encoder_weight.ndim != 2 or encoder_weight.shape[1] != 3:
        raise ValueError(
            "contact encoder weight must have shape (hidden_dim, 3); "
            f"got {tuple(encoder_weight.shape)}"
        )
    mean_abs = features.abs().reshape(-1, 3).mean(dim=0)
    column_norms = torch.linalg.vector_norm(
        encoder_weight.to(features),
        dim=0,
    )
    return mean_abs * column_norms


def contact_weighted_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    last_source: torch.Tensor,
    floor_height: torch.Tensor,
    margin: float = 0.04,
    temperature: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contact-weighted position/velocity losses and soft contact fraction."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(
            f"pred and target must share shape (B,F,N,3); got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}"
        )
    if last_source.ndim != 4 or last_source.shape[0] != pred.shape[0] \
            or last_source.shape[2:] != pred.shape[2:]:
        raise ValueError(
            "last_source must have shape (B,Fsrc,N,3) with matching batch/points; "
            f"got {tuple(last_source.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    floor = _floor_for_points(floor_height, target)[..., 0]
    gt_gap = target[..., 1] - floor
    weights = torch.sigmoid((float(margin) - gt_gap) / float(temperature))
    denom = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)

    position_error = (pred - target).square().mean(dim=-1)
    position_loss = (weights * position_error).sum() / denom

    pred_with_prev = torch.cat([last_source[:, -1:], pred], dim=1)
    target_with_prev = torch.cat([last_source[:, -1:], target], dim=1)
    pred_velocity = pred_with_prev[:, 1:] - pred_with_prev[:, :-1]
    target_velocity = target_with_prev[:, 1:] - target_with_prev[:, :-1]
    velocity_error = (pred_velocity - target_velocity).square().mean(dim=-1)
    velocity_loss = (weights * velocity_error).sum() / denom

    return position_loss, velocity_loss, weights.mean()


def find_contact_window_starts(
    min_y: np.ndarray,
    floor_height: float,
    max_start: int,
    input_frames: int,
    output_frames: int,
    frame_interval: int = 1,
    margin: float = 0.04,
    frame_radius: int = 2,
) -> List[int]:
    """Find starts whose target frame(s) surround the first floor contact.

    H5 positions and floor heights are raw coordinates. The fixed dataset transform
    divides relative distances by two, so contact detection is performed in the
    normalized coordinates used by the model and losses. Restricting candidates to a
    small band around the first contact avoids spending the contact quota on the long
    settled tail after impact.
    """
    min_y = np.asarray(min_y)
    if min_y.ndim != 1:
        raise ValueError(f"min_y must be one-dimensional; got {min_y.shape}")
    if frame_interval <= 0:
        raise ValueError(f"frame_interval must be positive; got {frame_interval}")
    if frame_radius < 0:
        raise ValueError(f"frame_radius must be non-negative; got {frame_radius}")

    normalized_gap = (min_y - float(floor_height)) / 2.0
    contact_frames = np.flatnonzero(normalized_gap <= float(margin))
    if contact_frames.size == 0:
        return []
    first_contact = int(contact_frames[0])
    contact_lo = max(0, first_contact - int(frame_radius))
    contact_hi = min(min_y.shape[0] - 1, first_contact + int(frame_radius))

    starts = []
    output_offsets = np.arange(
        input_frames * frame_interval,
        (input_frames + output_frames) * frame_interval,
        frame_interval,
    )
    for start in range(max_start + 1):
        output_indices = start + output_offsets
        if output_indices[-1] >= min_y.shape[0]:
            continue
        if np.any((output_indices >= contact_lo) & (output_indices <= contact_hi)):
            starts.append(start)
    return starts
