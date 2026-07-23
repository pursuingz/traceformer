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


class HybridStateExchange(nn.Module):
    """Update shared physical state tokens and feed them to prediction particles."""

    def __init__(
        self,
        particle_dim: int,
        state_dim: int = 64,
        num_heads: int = 4,
        history_frames: int = 5,
        num_stages: int = 4,
    ):
        super().__init__()
        if particle_dim <= 0:
            raise ValueError("particle_dim must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if state_dim % num_heads != 0:
            raise ValueError("state_dim must be divisible by num_heads")
        if history_frames != 5:
            raise ValueError("history_frames must be exactly 5")
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")

        self.particle_dim = particle_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.history_frames = history_frames
        self.num_stages = num_stages

        self.pool_norm = nn.LayerNorm(particle_dim)
        self.pool_score = nn.Linear(particle_dim, 1)
        self.particle_projection = nn.Linear(particle_dim, state_dim)
        self.explicit_encoder = nn.Sequential(
            nn.Linear(18, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )
        self.material_encoder = nn.Sequential(
            nn.Linear(2, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        self.frame_embeddings = nn.Parameter(torch.empty(history_frames, state_dim))
        self.stage_embeddings = nn.Parameter(torch.empty(num_stages, state_dim))

        attention_head_dim = state_dim // num_heads
        self.state_attention_norm = nn.LayerNorm(state_dim)
        self.state_attention = Attention(
            query_dim=state_dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            qk_norm="layer_norm",
        )
        self.state_ff_norm = nn.LayerNorm(state_dim)
        self.state_ff = FeedForward(
            state_dim,
            inner_dim=state_dim * 4,
            activation_fn="gelu",
        )
        self.state_film = nn.Linear(state_dim, state_dim * 2)

        self.feedback_norm = nn.LayerNorm(particle_dim)
        self.feedback_film = nn.Linear(state_dim, particle_dim * 2)
        self.feedback_attention = Attention(
            query_dim=particle_dim,
            cross_attention_dim=state_dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            qk_norm="layer_norm",
        )
        self.feedback_gates = nn.Parameter(torch.zeros(num_stages))

        nn.init.normal_(self.frame_embeddings, std=0.02)
        nn.init.normal_(self.stage_embeddings, std=0.02)

    @staticmethod
    def _apply_film(
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = modulation.chunk(2, dim=-1)
        return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        explicit_frame_state: torch.Tensor,
        material_values: torch.Tensor,
        history_start: int,
        prediction_index: int,
        stage_index: int,
    ) -> None:
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape (B, F, N, C)")
        if not hidden_states.is_floating_point():
            raise ValueError("hidden_states must use a floating-point dtype")

        batch_size, frame_count, particle_count, particle_dim = hidden_states.shape
        if particle_dim != self.particle_dim:
            raise ValueError(
                f"hidden_states particle_dim must be {self.particle_dim}, got {particle_dim}"
            )
        if particle_count < 1:
            raise ValueError("hidden_states must contain at least one particle")
        if not isinstance(history_start, int) or history_start < 0:
            raise ValueError("history_start must be a non-negative integer")
        if not isinstance(prediction_index, int):
            raise ValueError("prediction_index must be an integer")

        expected_prediction = history_start + self.history_frames
        if prediction_index != expected_prediction:
            raise ValueError(
                "prediction_index must immediately follow the five physical history frames"
            )
        if prediction_index >= frame_count:
            raise ValueError("prediction_index is outside the hidden frame sequence")
        if frame_count != prediction_index + 1:
            raise ValueError(
                "hidden_states must contain exactly one prediction frame after history"
            )

        expected_explicit_shape = (batch_size, self.history_frames, 18)
        if tuple(explicit_frame_state.shape) != expected_explicit_shape:
            raise ValueError(
                "explicit_frame_state must have shape "
                f"{expected_explicit_shape}, got {tuple(explicit_frame_state.shape)}"
            )
        expected_material_shape = (batch_size, 2)
        if tuple(material_values.shape) != expected_material_shape:
            raise ValueError(
                "material_values must have shape "
                f"{expected_material_shape}, got {tuple(material_values.shape)}"
            )
        if state_tokens is not None:
            expected_state_shape = (
                batch_size,
                self.history_frames,
                self.state_dim,
            )
            if tuple(state_tokens.shape) != expected_state_shape:
                raise ValueError(
                    "state_tokens must have shape "
                    f"{expected_state_shape}, got {tuple(state_tokens.shape)}"
                )
        if not isinstance(stage_index, int) or not 0 <= stage_index < self.num_stages:
            raise ValueError(
                f"stage_index must be in [0, {self.num_stages - 1}]"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_tokens: Optional[torch.Tensor],
        explicit_frame_state: torch.Tensor,
        material_values: torch.Tensor,
        history_start: int,
        prediction_index: int,
        stage_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_forward_inputs(
            hidden_states,
            state_tokens,
            explicit_frame_state,
            material_values,
            history_start,
            prediction_index,
            stage_index,
        )

        compute_dtype = self.pool_norm.weight.dtype
        history = hidden_states[
            :, history_start : history_start + self.history_frames
        ].to(dtype=compute_dtype)
        pool_scores = self.pool_score(self.pool_norm(history))
        pool_weights = torch.softmax(pool_scores, dim=2)
        pooled_particles = (pool_weights * history).sum(dim=2)
        particle_state = self.particle_projection(pooled_particles)

        explicit_state = self.explicit_encoder(
            explicit_frame_state.to(dtype=compute_dtype)
        )
        material_context = self.material_encoder(material_values.to(dtype=compute_dtype))
        state_input = (
            particle_state
            + explicit_state
            + material_context.unsqueeze(1)
            + self.frame_embeddings.unsqueeze(0)
            + self.stage_embeddings[stage_index].view(1, 1, -1)
        )
        if state_tokens is None:
            state_tokens = torch.zeros_like(state_input)
        else:
            state_tokens = state_tokens.to(dtype=compute_dtype)

        state_tokens = state_tokens + state_input
        state_modulation = self.state_film(material_context)
        attention_input = self._apply_film(
            self.state_attention_norm(state_tokens),
            state_modulation,
        )
        state_tokens = state_tokens + self.state_attention(attention_input)
        ff_input = self._apply_film(
            self.state_ff_norm(state_tokens),
            state_modulation,
        )
        state_tokens = state_tokens + self.state_ff(ff_input)

        prediction = hidden_states[:, prediction_index]
        prediction_compute = prediction.to(dtype=compute_dtype)
        feedback_query = self._apply_film(
            self.feedback_norm(prediction_compute),
            self.feedback_film(material_context),
        )
        feedback = self.feedback_attention(
            feedback_query,
            encoder_hidden_states=state_tokens,
        )
        feedback = torch.nan_to_num(feedback, nan=0.0, posinf=0.0, neginf=0.0)
        updated_prediction = prediction + (
            self.feedback_gates[stage_index] * feedback
        ).to(dtype=prediction.dtype)
        updated_hidden = torch.cat(
            (
                hidden_states[:, :prediction_index],
                updated_prediction.unsqueeze(1),
                hidden_states[:, prediction_index + 1 :],
            ),
            dim=1,
        )
        return state_tokens, updated_hidden
