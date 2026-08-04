import math
from typing import Any

import torch

from model.material_state import FactorizedMaterialStateAdapter


def _linear_frobenius(module: torch.nn.Linear) -> float:
    squared = module.weight.detach().float().square().sum()
    if module.bias is not None:
        squared = squared + module.bias.detach().float().square().sum()
    return float(torch.sqrt(squared).cpu())


def summarize_material_state_adapter(model: Any) -> dict[str, Any]:
    """Return fail-closed structural and parameter diagnostics for B3a."""
    dit = getattr(model, "dit", None)
    adapter = getattr(dit, "material_state_exchange", None)
    if not isinstance(adapter, FactorizedMaterialStateAdapter):
        raise ValueError("model does not contain a material-state adapter")
    interval = getattr(dit, "material_state_interval", None)
    runtime_scale = getattr(dit, "material_state_runtime_scale", None)
    if adapter.rank != 64 or adapter.num_stages != 4 or interval != 2:
        raise ValueError(
            "material-state adapter structure must be rank=64, stages=4, interval=2"
        )

    parameters = list(adapter.parameters())
    if not parameters or not all(torch.isfinite(p.detach()).all() for p in parameters):
        raise ValueError("material-state adapter parameters must be finite")
    if not isinstance(runtime_scale, (int, float)) or not math.isfinite(
        float(runtime_scale)
    ):
        raise ValueError("material-state adapter runtime scale must be finite")

    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "rank": adapter.rank,
        "interval": interval,
        "num_stages": adapter.num_stages,
        "runtime_scale": float(runtime_scale),
        "stage_scales": [
            float(value) for value in adapter.stage_scales.detach().float().cpu()
        ],
        "state_projection_norm": _linear_frobenius(adapter.state_proj),
        "material_projection_norm": _linear_frobenius(adapter.material_proj),
        "output_projection_norm": _linear_frobenius(adapter.output_proj),
        "e_center": adapter.e_center,
        "e_scale": adapter.e_scale,
        "nu_center": adapter.nu_center,
        "nu_scale": adapter.nu_scale,
    }
