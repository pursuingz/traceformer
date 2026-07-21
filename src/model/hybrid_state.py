from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention import Attention, FeedForward


def compute_explicit_frame_state(points: torch.Tensor) -> torch.Tensor:
    """Compute explicit whole-object state features for each frame."""
    if points.ndim != 4:
        raise ValueError("points must have shape (B, T, N, 3)")
    if points.shape[-1] != 3:
        raise ValueError("points must use XYZ coordinates in the last dimension")
    if points.shape[1] < 1:
        raise ValueError("points must contain at least one frame")
    if points.shape[2] < 1:
        raise ValueError("points must contain at least one particle")

    center = points.mean(dim=2)
    centered = points - center.unsqueeze(2)
    covariance = torch.einsum("btni,btnj->btij", centered, centered) / points.shape[2]
    upper_indices = torch.triu_indices(3, 3, device=points.device)
    covariance_upper = covariance[..., upper_indices[0], upper_indices[1]]

    relative_center = center - center[:, :1]
    center_velocity = torch.cat(
        (torch.zeros_like(center[:, :1]), center[:, 1:] - center[:, :-1]),
        dim=1,
    )
    covariance_delta = torch.cat(
        (
            torch.zeros_like(covariance_upper[:, :1]),
            covariance_upper[:, 1:] - covariance_upper[:, :-1],
        ),
        dim=1,
    )

    return torch.cat(
        (relative_center, center_velocity, covariance_upper, covariance_delta),
        dim=-1,
    )
