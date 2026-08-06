import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.material_gate_dataset import (
    MATERIAL_NAMES,
    MaterialGateDataset,
    build_material_split,
)
from model.spacetime import MDM_ST
from utils.material_stage_gate_training import (
    calibrate_material_rows,
    freeze_for_gate_training,
    load_b3a_into_b3b,
    save_gate_artifacts,
)


def create_gate_model(config):
    if bool(config.get("use_diffusion", False)):
        raise ValueError("material-stage gate calibration requires deterministic inference")
    if int(config.get("output_frames", 1)) != 1:
        raise ValueError("material-stage gate calibration requires output_frames=1")
    if not bool(config.model_config.get("material_state_adapter", False)):
        raise ValueError("material_stage_gate requires material_state_adapter=true")
    if not bool(config.model_config.get("material_stage_gate", False)):
        raise ValueError("material_stage_gate must be enabled for gate calibration")

    input_frames = int(config.get("input_frames", 5))
    output_frames = int(config.get("output_frames", 1))
    config.train_dataset.input_frames = input_frames
    config.train_dataset.output_frames = output_frames
    config.model_config.cond_frames = input_frames
    return MDM_ST(
        int(config.pc_size),
        output_frames,
        n_feats=3,
        model_config=config.model_config,
    )


def _material_loaders(config, manifest):
    train_loaders = {}
    val_loaders = {}
    batch_size = int(config.get("gate_batch_size", 1))
    if batch_size != 1:
        raise ValueError("variable-horizon gate calibration requires gate_batch_size=1")
    workers = int(config.get("gate_dataloader_num_workers", 0))
    max_rollout_steps = int(config.gate_max_rollout_steps)
    seed = int(config.gate_split_seed)

    for material_id, material_name in MATERIAL_NAMES.items():
        material_split = manifest["materials"][material_name]
        train_dataset = MaterialGateDataset(
            config.train_dataset,
            material_split["train"],
            seed=seed + material_id,
            max_rollout_steps=max_rollout_steps,
        )
        val_dataset = MaterialGateDataset(
            config.train_dataset,
            material_split["val"],
            seed=seed + material_id,
            max_rollout_steps=max_rollout_steps,
            resample_random_start=False,
        )
        generator = torch.Generator().manual_seed(seed + material_id)
        train_loaders[material_id] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
        )
        val_loaders[material_id] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
    return train_loaders, val_loaders


def run(config):
    if float(config.get("gate_start0_probability", 0.5)) != 0.5:
        raise ValueError("B3b screening fixes gate_start0_probability at 0.5")
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "bf16"))
    if accelerator.num_processes != 1:
        raise RuntimeError("material-stage gate calibration supports exactly one process")
    set_seed(int(config.get("seed", 0)))

    manifest = build_material_split(
        config.train_dataset.dataset_path,
        config.train_dataset.dataset_list,
        train_fraction=float(config.gate_train_fraction),
        seed=int(config.gate_split_seed),
    )
    train_loaders, val_loaders = _material_loaders(config, manifest)
    model = create_gate_model(config)
    load_b3a_into_b3b(model, config.base_checkpoint)
    gate = freeze_for_gate_training(model)
    accelerator.print("base checkpoint loaded with only gate_logits missing")
    accelerator.print(f"total trainable parameters: {gate.numel()}")
    accelerator.print("dev-test access during calibration: disabled")
    model = accelerator.prepare(model)

    result = calibrate_material_rows(
        model,
        train_loaders,
        val_loaders,
        max_updates=int(config.gate_updates_per_material),
        validation_interval=int(config.gate_validation_interval),
        patience=int(config.gate_patience),
        learning_rate=float(config.gate_learning_rate),
        long_weight=float(config.gate_long_weight),
        reg_weight=float(config.gate_reg_weight),
        accelerator=accelerator,
        noise_seed=int(config.get("seed", 0)),
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = str(config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
        save_gate_artifacts(
            model,
            output_dir,
            split_manifest=manifest,
            training_history=result["history"],
            metadata={
                "base_checkpoint": str(config.base_checkpoint),
                "seed": int(config.get("seed", 0)),
                "gate_split_seed": int(config.gate_split_seed),
                "identity_scores": result["identity_scores"],
                "best_updates": result["best_updates"],
                "best_scores": result["best_scores"],
                "material_names": MATERIAL_NAMES,
            },
            accelerator=accelerator,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate B3b material-stage gates")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(OmegaConf.load(args.config))


if __name__ == "__main__":
    main()
