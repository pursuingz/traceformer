"""Inspect contact-feature distributions and learned encoder usage without inference."""

import argparse
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from dataset.traj_dataset import TrajDataset
from options import TestingConfig
from utils.contact import (
    build_contact_features,
    contact_channel_contributions,
    contact_feature_names,
)


MATERIAL_NAMES = {
    0: "elastic",
    1: "plasticine",
    2: "sand",
    3: "rigid",
}
def _state_tensor_by_suffix(state, suffix):
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one checkpoint tensor ending in {suffix!r}; "
            f"found {len(matches)}"
        )
    return matches[0].float()


def _load_factorized_contact_parameters(state):
    boundary_weight = _state_tensor_by_suffix(
        state,
        "contact_adapter.boundary_encoder.weight",
    )
    normal_weight = _state_tensor_by_suffix(
        state,
        "contact_adapter.normal_encoder.weight",
    )
    tangential_weight = _state_tensor_by_suffix(
        state,
        "contact_adapter.tangential_encoder.weight",
    )
    shared_bias = _state_tensor_by_suffix(
        state,
        "contact_adapter.shared_bias",
    )
    raw_gate = _state_tensor_by_suffix(
        state,
        "contact_adapter.tangential_gate",
    )

    if boundary_weight.ndim != 2 or boundary_weight.shape[1] != 2:
        raise ValueError(
            "factorized boundary weight must have shape (H, 2), "
            f"got {tuple(boundary_weight.shape)}"
        )
    hidden_dim = boundary_weight.shape[0]
    expected_shapes = {
        "normal": (hidden_dim, 1),
        "tangential": (hidden_dim, 2),
    }
    branch_weights = {
        "normal": normal_weight,
        "tangential": tangential_weight,
    }
    for name, weight in branch_weights.items():
        if tuple(weight.shape) != expected_shapes[name]:
            raise ValueError(
                f"factorized {name} weight must have shape "
                f"{expected_shapes[name]}, got {tuple(weight.shape)}"
            )
    if tuple(shared_bias.shape) != (hidden_dim,):
        raise ValueError(
            "factorized shared bias must have shape "
            f"({hidden_dim},), got {tuple(shared_bias.shape)}"
        )
    if raw_gate.ndim != 0:
        raise ValueError(
            "factorized tangential gate must be a scalar, "
            f"got shape {tuple(raw_gate.shape)}"
        )

    return (
        boundary_weight,
        normal_weight,
        tangential_weight,
        shared_bias,
        torch.tanh(raw_gate),
    )


def load_contact_projection(state, injection_mode, feature_dim):
    """Return the learned contact projection and any contact-only bias."""
    if injection_mode == "separate":
        weight = _state_tensor_by_suffix(state, "contact_encoder.weight")
        bias = _state_tensor_by_suffix(state, "contact_encoder.bias")
    elif injection_mode == "shared":
        full_weight = _state_tensor_by_suffix(
            state,
            "input_encoder.mlp.weight",
        )
        if full_weight.ndim != 2 or full_weight.shape[1] < feature_dim:
            raise ValueError(
                "shared input projection cannot contain the requested "
                f"{feature_dim} contact columns: shape={tuple(full_weight.shape)}"
            )
        weight = full_weight[:, -feature_dim:]
        bias = None
    elif injection_mode == "factorized":
        if feature_dim != 5:
            raise ValueError(
                "factorized contact projection requires feature_dim=5; "
                f"got {feature_dim}"
            )
        (
            boundary_weight,
            normal_weight,
            tangential_weight,
            bias,
            effective_gate,
        ) = _load_factorized_contact_parameters(state)
        weight = boundary_weight.new_zeros(boundary_weight.shape[0], 5)
        weight[:, [0, 4]] = boundary_weight
        weight[:, 2:3] = normal_weight
        weight[:, [1, 3]] = effective_gate * tangential_weight
    else:
        raise ValueError(
            "contact_injection_mode must be one of "
            "'separate', 'shared', or 'factorized'; "
            f"got {injection_mode!r}"
        )

    if weight.ndim != 2 or weight.shape[1] != feature_dim:
        raise ValueError(
            "contact projection width does not match the configured feature "
            f"dimension: shape={tuple(weight.shape)}, feature_dim={feature_dim}"
        )
    return weight, bias


def load_factorized_contact_stats(state):
    """Return effective gate and parameter norms for a factorized adapter."""
    (
        boundary_weight,
        normal_weight,
        tangential_weight,
        shared_bias,
        effective_gate,
    ) = _load_factorized_contact_parameters(state)
    return {
        "effective_gate": effective_gate.item(),
        "boundary_weight_norm": torch.linalg.vector_norm(
            boundary_weight
        ).item(),
        "normal_weight_norm": torch.linalg.vector_norm(normal_weight).item(),
        "tangential_weight_norm": torch.linalg.vector_norm(
            tangential_weight
        ).item(),
        "shared_bias_norm": torch.linalg.vector_norm(shared_bias).item(),
    }


def _mean_projected_token_norm(features, weight):
    features64 = features.double()
    weight64 = weight.double()
    gram = weight64.T @ weight64
    norm_sq = torch.einsum(
        "ti,ij,tj->t",
        features64,
        gram,
        features64,
    ).clamp_min(0)
    return torch.sqrt(norm_sq).mean()


def factorized_branch_hidden_norms(features, state):
    """Return mean per-token hidden norms for each factorized branch."""
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError(
            "features must have shape (tokens, 5), "
            f"got {tuple(features.shape)}"
        )
    (
        boundary_weight,
        normal_weight,
        tangential_weight,
        _,
        effective_gate,
    ) = _load_factorized_contact_parameters(state)
    boundary_norm = _mean_projected_token_norm(
        features[:, [0, 4]],
        boundary_weight,
    )
    normal_norm = _mean_projected_token_norm(
        features[:, 2:3],
        normal_weight,
    )
    tangential_norm = _mean_projected_token_norm(
        features[:, [1, 3]],
        tangential_weight,
    ) * effective_gate.double().abs()
    return {
        "boundary": boundary_norm.item(),
        "normal": normal_norm.item(),
        "tangential": tangential_norm.item(),
    }


def _quantile_text(values):
    levels = torch.tensor(
        [0.01, 0.05, 0.50, 0.95, 0.99],
        dtype=values.dtype,
    )
    quantiles = torch.quantile(values, levels)
    return " ".join(
        f"p{int(level * 100):02d}={value:.6g}"
        for level, value in zip(levels.tolist(), quantiles.tolist())
    )


def collect_grouped_features(loader, sigma, velocity_mode="vertical"):
    """Collect raw contact features by material from TrajDataset batches."""
    feature_dim = len(contact_feature_names(velocity_mode))
    grouped_features = defaultdict(list)
    for batch, _ in loader:
        features = build_contact_features(
            batch["points_src"].float(),
            batch["floor_height"].float(),
            start_velocity=batch.get("start_vel"),
            sigma=sigma,
            velocity_mode=velocity_mode,
        )
        material = batch.get("mat_type")
        if material is None:
            material = torch.zeros(features.shape[0], dtype=torch.long)
        for index, mat_type in enumerate(material.reshape(-1).tolist()):
            grouped_features[int(mat_type)].append(
                features[index].reshape(-1, feature_dim)
            )
    return grouped_features


def main(cfg):
    cfg.train_dataset.input_frames = cfg.get("input_frames", 5)
    cfg.train_dataset.output_frames = cfg.get("output_frames", 1)
    dataset = TrajDataset("test", cfg.train_dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
    )

    sigma = float(cfg.model_config.get("contact_feature_sigma", 0.04))
    velocity_mode = cfg.model_config.get("contact_velocity_mode", "vertical")
    feature_names = contact_feature_names(velocity_mode)
    injection_mode = cfg.model_config.get(
        "contact_injection_mode",
        "separate",
    )
    state = load_file(cfg.resume, device="cpu")
    weight, bias = load_contact_projection(
        state,
        injection_mode=injection_mode,
        feature_dim=len(feature_names),
    )
    factorized_stats = None
    if injection_mode == "factorized":
        factorized_stats = load_factorized_contact_stats(state)
    grouped_features = collect_grouped_features(
        loader,
        sigma,
        velocity_mode=velocity_mode,
    )
    print("===== contact feature diagnostics =====")
    print(f"checkpoint: {cfg.resume}")
    print(f"contact injection mode: {injection_mode}")
    print(f"contact velocity mode: {velocity_mode}")
    print(f"encoder weight column norms: {torch.linalg.vector_norm(weight, dim=0).tolist()}")
    if bias is None:
        print("encoder bias: shared point bias (not contact-specific)")
    else:
        print(f"encoder bias norm: {torch.linalg.vector_norm(bias).item():.6g}")
    if factorized_stats is not None:
        print(
            "factorized effective gate: "
            f"{factorized_stats['effective_gate']:.6g}"
        )
        print(
            "factorized parameter norms: "
            f"boundary={factorized_stats['boundary_weight_norm']:.6g}, "
            f"normal={factorized_stats['normal_weight_norm']:.6g}, "
            f"tangential="
            f"{factorized_stats['tangential_weight_norm']:.6g}, "
            f"shared_bias={factorized_stats['shared_bias_norm']:.6g}"
        )

    for mat_type in sorted(grouped_features):
        features = torch.cat(grouped_features[mat_type], dim=0)
        contributions = contact_channel_contributions(features, weight)
        gap = features[:, 0]
        proximity = features[:, -1]
        print(
            f"\n--- {MATERIAL_NAMES.get(mat_type, 'unknown')}"
            f" (mat_type={mat_type}, tokens={features.shape[0]}) ---"
        )
        for index, name in enumerate(feature_names):
            values = features[:, index]
            if name == "proximity":
                print(
                    f"{name}: mean={values.mean().item():.6g} "
                    f"nonzero={(values > 0).float().mean().item() * 100:.3f}%"
                )
            else:
                print(
                    f"{name}: mean={values.mean().item():.6g} "
                    f"std={values.std(unbiased=False).item():.6g} "
                    f"{_quantile_text(values)}"
                )
        print(
            f"contact-band fraction(gap <= sigma={sigma:g}): "
            f"{(gap <= sigma).float().mean().item() * 100:.3f}%"
        )
        print(
            "estimated hidden contribution: "
            + ", ".join(
                f"{name}={value:.6g}"
                for name, value in zip(
                    feature_names,
                    contributions.tolist(),
                )
            )
        )
        if injection_mode == "factorized":
            branch_norms = factorized_branch_hidden_norms(features, state)
            print(
                "factorized branch hidden norms: "
                f"boundary={branch_norms['boundary']:.6g}, "
                f"normal={branch_norms['normal']:.6g}, "
                f"tangential={branch_norms['tangential']:.6g}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    schema = OmegaConf.structured(TestingConfig)
    config = OmegaConf.merge(schema, OmegaConf.load(cli_args.config))
    main(config)
