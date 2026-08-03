from contextlib import contextmanager
import math

import torch

from utils.eval_metrics import per_window_metrics


KNOCKOUT_METRICS = (
    "full_rollout_mse",
    "short_mse",
    "mid_mse",
    "long_mse",
    "gm_mse",
    "fde",
    "f24_centroid_error",
    "f24_shape_residual_mse",
    "penetration_rate",
    "penetration_depth",
)


KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
)


@contextmanager
def masked_feedback_gates(exchange, mask):
    gates = exchange.feedback_gates
    if gates.numel() != 4 or not torch.isfinite(gates.detach()).all():
        raise ValueError("feedback_gates must contain four finite values")
    mask_tensor = torch.as_tensor(mask, device=gates.device, dtype=gates.dtype)
    if mask_tensor.shape != gates.shape or not torch.all(
        (mask_tensor == 0) | (mask_tensor == 1)
    ):
        raise ValueError("gate mask must contain exactly four binary values")
    original = gates.detach().clone()
    with torch.no_grad():
        gates.copy_(original * mask_tensor)
    try:
        yield gates.detach().clone()
    finally:
        with torch.no_grad():
            gates.copy_(original)
        if not torch.equal(gates.detach(), original):
            raise RuntimeError("feedback gates were not restored exactly")


def reset_inference_seed(seed, device):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def trajectory_knockout_metrics(pred, gt, input_frames, floor_height):
    """Return strict trajectory, geometry, and floor-penetration metrics."""
    if not isinstance(pred, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise ValueError("pred and gt must be tensors")
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred and gt must share shape (25, N, 3)")
    if pred.shape[0] != 25 or pred.shape[1] < 2:
        raise ValueError("pred and gt must have shape (25, N, 3)")
    if input_frames != 5:
        raise ValueError("input_frames must be 5")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("pred and gt must contain only finite values")

    pred_f = pred.float()
    gt_f = gt.to(pred.device).float()
    floor = torch.as_tensor(floor_height, device=pred.device, dtype=pred_f.dtype)
    if floor.numel() != 1 or not torch.isfinite(floor).all():
        raise ValueError("floor_height must be one finite scalar")

    frame_mse = (pred_f[input_frames:] - gt_f[input_frames:]).square().mean((1, 2))
    base = per_window_metrics(
        pred_f, gt_f, input_frames, k=min(8, pred.shape[1] - 1)
    )
    centroid, _, _, shape = base["proc"][24]
    penetration = torch.clamp(
        floor.reshape(()) - pred_f[input_frames:, :, 1], min=0
    )
    result = {
        "full_rollout_mse": float(
            (pred_f[input_frames:] - gt_f[input_frames:]).square().mean()
        ),
        "short_mse": float(
            (pred_f[5:11] - gt_f[5:11]).square().mean()
        ),
        "mid_mse": float(
            (pred_f[11:18] - gt_f[11:18]).square().mean()
        ),
        "long_mse": float(
            (pred_f[18:25] - gt_f[18:25]).square().mean()
        ),
        "gm_mse": float(torch.exp(torch.log(frame_mse.clamp_min(1e-30)).mean())),
        "fde": float(base["fde"]),
        "f24_centroid_error": float(centroid),
        "f24_shape_residual_mse": float(shape),
        "penetration_rate": float((penetration > 0).float().mean()),
        "penetration_depth": float(penetration.mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("trajectory knockout metrics must be finite")
    return result
