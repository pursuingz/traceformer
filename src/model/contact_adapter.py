import torch
from torch import nn


class FactorizedContactAdapter(nn.Module):
    def __init__(self, latent_dim: int):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        super().__init__()
        self.boundary = nn.Linear(2, latent_dim, bias=False)
        self.normal = nn.Linear(1, latent_dim, bias=False)
        self.tangential = nn.Linear(2, latent_dim, bias=False)
        self.shared_bias = nn.Parameter(torch.zeros(latent_dim))
        self.tangential_gate = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.boundary.weight)
        nn.init.zeros_(self.normal.weight)

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

        boundary = self.boundary(features[..., [0, 4]])
        normal = self.normal(features[..., [2]])
        tangential = self.tangential(features[..., [1, 3]])
        return (
            boundary
            + normal
            + torch.tanh(self.tangential_gate) * tangential
            + bias_scale * self.shared_bias
        )
