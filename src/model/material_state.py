import torch
from torch import nn
from torch.nn import functional as F


class FactorizedMaterialStateAdapter(nn.Module):
    """Low-rank multiplicative interaction between particle state and material."""

    def __init__(
        self,
        particle_dim: int,
        rank: int,
        num_materials: int,
        num_stages: int,
        e_center: float = 5.5,
        e_scale: float = 1.0,
        nu_center: float = 0.25,
        nu_scale: float = 0.15,
    ):
        super().__init__()
        if particle_dim <= 0:
            raise ValueError("particle_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if num_materials <= 0:
            raise ValueError("num_materials must be positive")
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if e_scale <= 0:
            raise ValueError("e_scale must be positive")
        if nu_scale <= 0:
            raise ValueError("nu_scale must be positive")

        self.particle_dim = int(particle_dim)
        self.rank = int(rank)
        self.num_materials = int(num_materials)
        self.num_stages = int(num_stages)
        self.e_center = float(e_center)
        self.e_scale = float(e_scale)
        self.nu_center = float(nu_center)
        self.nu_scale = float(nu_scale)

        self.state_norm = nn.LayerNorm(self.particle_dim)
        self.state_proj = nn.Linear(self.particle_dim, self.rank)
        self.material_proj = nn.Linear(2 + self.num_materials, self.rank)
        self.output_proj = nn.Linear(self.rank, self.particle_dim)
        self.stage_scales = nn.Parameter(torch.ones(self.num_stages))
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        material_values: torch.Tensor,
        material_labels: torch.Tensor,
        stage_index: int,
    ) -> None:
        if hidden_states.ndim != 4 or hidden_states.shape[-1] != self.particle_dim:
            raise ValueError(
                "hidden_states must have shape (B, F, N, particle_dim)"
            )
        batch_size = hidden_states.shape[0]
        if material_values.shape != (batch_size, 2):
            raise ValueError("material_values must have shape (B, 2)")
        if material_labels.shape != (batch_size,):
            raise ValueError("material_labels must have shape (B,)")
        if not torch.isfinite(material_values).all():
            raise ValueError("material_values must be finite")
        labels = material_labels.long()
        if not torch.equal(labels.to(material_labels.dtype), material_labels):
            raise ValueError("material_labels must contain integer class indices")
        if ((labels < 0) | (labels >= self.num_materials)).any():
            raise ValueError("material_labels are outside the valid class range")
        if not isinstance(stage_index, int) or not 0 <= stage_index < self.num_stages:
            raise IndexError("stage_index is outside the configured stage range")

    def _material_input(
        self,
        material_values: torch.Tensor,
        material_labels: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        e = (material_values[:, :1] - self.e_center) / self.e_scale
        nu = (material_values[:, 1:2] - self.nu_center) / self.nu_scale
        one_hot = F.one_hot(
            material_labels.long(), num_classes=self.num_materials
        ).to(dtype=dtype, device=material_values.device)
        return torch.cat((e.to(dtype=dtype), nu.to(dtype=dtype), one_hot), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        material_values: torch.Tensor,
        material_labels: torch.Tensor,
        stage_index: int,
        runtime_scale: float = 1.0,
    ) -> torch.Tensor:
        self._validate_inputs(
            hidden_states,
            material_values,
            material_labels,
            stage_index,
        )
        if float(runtime_scale) == 0.0:
            return hidden_states.clone()

        state = F.silu(self.state_proj(self.state_norm(hidden_states)))
        material = F.silu(
            self.material_proj(
                self._material_input(material_values, material_labels, state.dtype)
            )
        )
        interaction = state * material[:, None, None, :]
        delta = self.output_proj(interaction)
        scale = self.stage_scales[stage_index].to(delta.dtype) * float(runtime_scale)
        return hidden_states + scale * delta
