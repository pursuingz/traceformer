"""Run the frozen B3a material-state stage knockout diagnostic."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

try:  # Supports both ``python src/...`` and ``import src....``.
    from diagnose_hybrid_state_feedback import (
        _validate_records,
        load_frozen_test_manifest,
        select_start_zero_indices,
    )
    from diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from options import TestingConfig
    from utils.hybrid_state_gate_knockout import (
        reset_inference_seed,
        trajectory_knockout_metrics,
    )
    from utils.material_state_stage_diagnostics import (
        MaterialStateActivityCollector,
        STAGE_KNOCKOUT_CONDITIONS,
        build_stage_paired_rows,
        masked_material_state_stages,
        summarize_stage_paired_rows,
        validate_stage_raw_rows,
        write_stage_outputs,
    )
except ModuleNotFoundError:
    from src.diagnose_hybrid_state_feedback import (
        _validate_records,
        load_frozen_test_manifest,
        select_start_zero_indices,
    )
    from src.diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from src.options import TestingConfig
    from src.utils.hybrid_state_gate_knockout import (
        reset_inference_seed,
        trajectory_knockout_metrics,
    )
    from src.utils.material_state_stage_diagnostics import (
        MaterialStateActivityCollector,
        STAGE_KNOCKOUT_CONDITIONS,
        build_stage_paired_rows,
        masked_material_state_stages,
        summarize_stage_paired_rows,
        validate_stage_raw_rows,
        write_stage_outputs,
    )


B3A90_SAMPLE_SCOPE = "frozen 41-model start_idx=0 B3a90 stage knockout"
_DIRECT_CALL_CONFIG_PATH = "configs/eval_mm3_b3a_material_state_adapter_90k.yaml"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _rollout_autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _expected_model_names(manifest: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for names in manifest.values() for name in names]


def _assert_checkpoint_scales(adapter: Any, original_scales: torch.Tensor) -> None:
    if not torch.equal(adapter.stage_scales.detach(), original_scales):
        raise RuntimeError("material-state stage scales no longer match the checkpoint")


def run_stage_knockout_diagnostics(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
    config_path: Path | None = None,
) -> list[dict]:
    """Run all registered B3a stage masks over frozen start-0 test windows."""
    from safetensors.torch import load_file

    try:
        from dataset.traj_dataset import TrajDataset
        from model.spacetime import MDM_ST
        from pipeline_traj import TrajPipeline
    except ModuleNotFoundError:
        from src.dataset.traj_dataset import TrajDataset
        from src.model.spacetime import MDM_ST
        from src.pipeline_traj import TrajPipeline

    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    args.resume = str(checkpoint)
    _validate_b0_identity(args, profile="b3a90")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    dataset_root = Path(args.train_dataset.dataset_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
    manifest = load_frozen_test_manifest()
    expected_models = _expected_model_names(manifest)
    if len(expected_models) != 41 or len(set(expected_models)) != 41:
        raise ValueError("frozen manifest must contain exactly 41 unique models")

    args.train_dataset.input_frames = 5
    args.train_dataset.output_frames = 1
    args.model_config.cond_frames = 5
    dataset = TrajDataset("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    records_by_model = _validate_records(records, dataset.split_lst_save, manifest)
    start_zero_indices = select_start_zero_indices(dataset, expected_models)

    device = torch.device("cuda")
    model = MDM_ST(args.pc_size, 1, n_feats=3, model_config=args.model_config).to(device)
    _load_checkpoint_strict(model, checkpoint, load_file)
    model.eval().requires_grad_(False)
    pipeline = TrajPipeline(model=model, scheduler=None)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, start_zero_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    adapter = model.dit.material_state_exchange
    scales = adapter.stage_scales.detach()
    if scales.shape != (4,) or not torch.isfinite(scales).all():
        raise ValueError("stage_scales must contain four finite checkpoint values")
    original_scales = scales.clone()
    raw_rows = []
    evaluated_models = set()
    with MaterialStateActivityCollector(adapter) as collector:
        for batch, _ in dataloader:
            start_indices = batch.get("start_idx")
            if start_indices is None or not torch.all(torch.as_tensor(start_indices) == 0):
                raise ValueError("B3a stage knockout accepts only start_idx=0")
            names = [Path(name).name for name in batch.get("model", [])]
            if len(names) != 1:
                raise ValueError("B3a stage knockout requires eval_batch_size=1")
            model_name = names[0]
            if model_name in evaluated_models:
                raise ValueError(f"{model_name}: evaluated more than once")
            record = records_by_model.get(model_name)
            if record is None:
                raise ValueError(f"{model_name}: missing material metadata")
            _validate_normal_material_condition(batch, record)
            gt = _build_raw_reference(batch, args.train_dataset)

            for condition, mask in STAGE_KNOCKOUT_CONDITIONS:
                _assert_checkpoint_scales(adapter, original_scales)
                reset_inference_seed(args.seed, device)
                with masked_material_state_stages(adapter, mask):
                    with _rollout_autocast_context(device):
                        if condition == "normal":
                            with collector.capture(
                                model_name,
                                record.mat_type,
                                expected_calls_per_stage=20,
                            ):
                                pred = rollout_condition(
                                    pipeline,
                                    batch,
                                    args,
                                    record.log10_e,
                                    record.nu,
                                    record.mat_type,
                                )
                        else:
                            pred = rollout_condition(
                                pipeline,
                                batch,
                                args,
                                record.log10_e,
                                record.nu,
                                record.mat_type,
                            )
                _assert_checkpoint_scales(adapter, original_scales)
                metrics = trajectory_knockout_metrics(
                    pred[0],
                    gt[0],
                    input_frames=5,
                    floor_height=batch["floor_height"][0],
                )
                raw_rows.append(
                    {
                        "model": model_name,
                        "mat_type": record.mat_type,
                        "log10_e": record.log10_e,
                        "nu": record.nu,
                        "condition": condition,
                        **metrics,
                    }
                )
            evaluated_models.add(model_name)

    _assert_checkpoint_scales(adapter, original_scales)
    if evaluated_models != set(expected_models):
        missing = sorted(set(expected_models) - evaluated_models)
        unexpected = sorted(evaluated_models - set(expected_models))
        raise ValueError(
            "frozen model set mismatch after stage knockout: "
            f"missing={missing}; unexpected={unexpected}"
        )

    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(Path(config_path)) if config_path is not None else _DIRECT_CALL_CONFIG_PATH,
        "seed": int(args.seed),
        "sample_scope": B3A90_SAMPLE_SCOPE,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    validated_raw = validate_stage_raw_rows(raw_rows)
    output_raw = [{**row, **metadata} for row in validated_raw]
    paired_rows = build_stage_paired_rows(output_raw)
    summary_rows = summarize_stage_paired_rows(
        paired_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    activity_rows = [{**row, **metadata} for row in collector.rows()]
    paths = write_stage_outputs(
        output_dir,
        output_raw,
        paired_rows,
        activity_rows,
        summary_rows,
        metadata,
    )
    print(
        "B3a stage knockout complete: "
        f"models={len(evaluated_models)} raw_rows={len(output_raw)} "
        f"paired_rows={len(paired_rows)} activity_rows={len(activity_rows)}"
    )
    for name, path in paths.items():
        print(f"{name}: {Path(path).resolve()}")
    return output_raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen B3a material-state stage knockout diagnostic."
    )
    parser.add_argument("--config", required=True, help="Frozen B3a90 evaluation YAML.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="B3a90 checkpoint-90000 model.safetensors path.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for B3a outputs.")
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=10000)
    parser.add_argument("--bootstrap-seed", type=_non_negative_int, default=0)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    config_path = Path(cli.config)
    checkpoint = Path(cli.checkpoint)
    args = OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config_path))
    args.resume = str(checkpoint)
    run_stage_knockout_diagnostics(
        args,
        checkpoint=checkpoint,
        output_dir=Path(cli.output_dir),
        bootstrap_samples=cli.bootstrap_samples,
        bootstrap_seed=cli.bootstrap_seed,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
