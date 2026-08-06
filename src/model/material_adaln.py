import torch
from torch import nn
from torch.nn import functional as F


class ContinuousMaterialConditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        e_center: float = 5.5,
        e_scale: float = 1.0,
        nu_center: float = 0.25,
        nu_scale: float = 0.15,
    ):
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        if e_scale <= 0 or nu_scale <= 0:
            raise ValueError("material normalization scales must be positive")
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.e_center = float(e_center)
        self.e_scale = float(e_scale)
        self.nu_center = float(nu_center)
        self.nu_scale = float(nu_scale)
        self.input_proj = nn.Linear(2, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def normalize_material_values(self, material_values: torch.Tensor) -> torch.Tensor:
        if material_values.ndim != 2 or material_values.shape[1] != 2:
            raise ValueError("material_values must have shape (B, 2)")
        if not torch.isfinite(material_values).all():
            raise ValueError("material_values must be finite")
        e = (material_values[:, :1] - self.e_center) / self.e_scale
        nu = (material_values[:, 1:2] - self.nu_center) / self.nu_scale
        return torch.cat((e, nu), dim=-1)

    def forward(self, material_values: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_material_values(material_values)
        hidden = F.silu(self.input_proj(normalized))
        return self.output_proj(hidden)
