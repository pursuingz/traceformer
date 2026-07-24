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
)


MATERIAL_NAMES = {
    0: "elastic",
    1: "plasticine",
    2: "sand",
    3: "rigid",
}
FEATURE_NAMES = ("signed_gap", "vertical_displacement", "proximity")


def _state_tensor_by_suffix(state, suffix):
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one checkpoint tensor ending in {suffix!r}; "
            f"found {len(matches)}"
        )
    return matches[0].float()


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


def collect_grouped_features(loader, sigma):
    """Collect raw contact features by material from TrajDataset batches."""
    grouped_features = defaultdict(list)
    for batch, _ in loader:
        features = build_contact_features(
            batch["points_src"].float(),
            batch["floor_height"].float(),
            start_velocity=batch.get("start_vel"),
            sigma=sigma,
        )
        material = batch.get("mat_type")
        if material is None:
            material = torch.zeros(features.shape[0], dtype=torch.long)
        for index, mat_type in enumerate(material.reshape(-1).tolist()):
            grouped_features[int(mat_type)].append(features[index].reshape(-1, 3))
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

    state = load_file(cfg.resume, device="cpu")
    weight = _state_tensor_by_suffix(state, "contact_encoder.weight")
    bias = _state_tensor_by_suffix(state, "contact_encoder.bias")

    sigma = float(cfg.model_config.get("contact_feature_sigma", 0.04))
    grouped_features = collect_grouped_features(loader, sigma)
    print("===== contact feature diagnostics =====")
    print(f"checkpoint: {cfg.resume}")
    print(f"encoder weight column norms: {torch.linalg.vector_norm(weight, dim=0).tolist()}")
    print(f"encoder bias norm: {torch.linalg.vector_norm(bias).item():.6g}")

    for mat_type in sorted(grouped_features):
        features = torch.cat(grouped_features[mat_type], dim=0)
        contributions = contact_channel_contributions(features, weight)
        gap = features[:, 0]
        velocity = features[:, 1]
        proximity = features[:, 2]
        print(
            f"\n--- {MATERIAL_NAMES.get(mat_type, 'unknown')}"
            f" (mat_type={mat_type}, tokens={features.shape[0]}) ---"
        )
        print(
            f"signed_gap: mean={gap.mean().item():.6g} "
            f"std={gap.std(unbiased=False).item():.6g} {_quantile_text(gap)}"
        )
        print(
            f"vertical_displacement: mean={velocity.mean().item():.6g} "
            f"std={velocity.std(unbiased=False).item():.6g} "
            f"{_quantile_text(velocity)}"
        )
        print(
            f"proximity: mean={proximity.mean().item():.6g} "
            f"nonzero={(proximity > 0).float().mean().item() * 100:.3f}%"
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
                    FEATURE_NAMES,
                    contributions.tolist(),
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    schema = OmegaConf.structured(TestingConfig)
    config = OmegaConf.merge(schema, OmegaConf.load(cli_args.config))
    main(config)
