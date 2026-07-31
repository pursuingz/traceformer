import torch
from torch import nn


CONTACT_FEATURE_ORDER = (
    "signed_gap",
    "displacement_x",
    "displacement_y",
    "displacement_z",
    "proximity",
)
BOUNDARY_FEATURE_INDICES = (0, 4)
NORMAL_FEATURE_INDICES = (2,)
TANGENTIAL_FEATURE_INDICES = (1, 3)


class FactorizedContactAdapter(nn.Module):
    def __init__(self, latent_dim: int):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        super().__init__()
        self.boundary_encoder = nn.Linear(2, latent_dim, bias=False)
        self.normal_encoder = nn.Linear(1, latent_dim, bias=False)
        self.tangential_encoder = nn.Linear(2, latent_dim, bias=False)
        self.shared_bias = nn.Parameter(torch.zeros(latent_dim))
        self.tangential_gate = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.boundary_encoder.weight)
        nn.init.zeros_(self.normal_encoder.weight)

    def forward(
        self,
        features: torch.Tensor,
        bias_scale: float = 1.0,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != 5:
            raise ValueError(
                "features must have shape (B, F, N, 5), "
                f"got {tuple(features.shape)}"
            )

        boundary = self.boundary_encoder(
            features[..., BOUNDARY_FEATURE_INDICES]
        )
        normal = self.normal_encoder(features[..., NORMAL_FEATURE_INDICES])
        tangential = self.tangential_encoder(
            features[..., TANGENTIAL_FEATURE_INDICES]
        )
        tangential_gate = torch.tanh(self.tangential_gate).to(
            dtype=tangential.dtype
        )
        shared_bias = self.shared_bias.to(dtype=boundary.dtype)
        return (
            boundary
            + normal
            + tangential_gate * tangential
            + bias_scale * shared_bias
        )
