from contextlib import contextmanager

import torch


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
