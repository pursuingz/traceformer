"""Run the frozen B1b inference-only HST feedback-gate knockout diagnostic."""

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

try:  # Supports both ``python src/...`` and ``import src....``.
    from diagnose_hybrid_state_feedback import (
        B1B_SAMPLE_SCOPE,
        DIRECT_CALL_CONFIG_PATH,
        _rollout_autocast_context,
        _validate_records,
        load_frozen_test_manifest,
        select_start_zero_indices,
        validate_diagnostic_config,
    )
    from diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        load_material_records,
        rollout_condition,
    )
    from options import TestingConfig
    from utils.hybrid_state_gate_knockout import (
        KNOCKOUT_CONDITIONS,
        build_paired_rows,
        dynamic_gate_verdict,
        masked_feedback_gates,
        reset_inference_seed,
        summarize_paired_rows,
        trajectory_knockout_metrics,
        validate_raw_rows,
        write_knockout_report,
        write_paired_csv,
        write_raw_csv,
    )
except ModuleNotFoundError:
    from src.diagnose_hybrid_state_feedback import (
        B1B_SAMPLE_SCOPE,
        DIRECT_CALL_CONFIG_PATH,
        _rollout_autocast_context,
        _validate_records,
        load_frozen_test_manifest,
        select_start_zero_indices,
        validate_diagnostic_config,
    )
    from src.diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        load_material_records,
        rollout_condition,
    )
    from src.options import TestingConfig
    from src.utils.hybrid_state_gate_knockout import (
        KNOCKOUT_CONDITIONS,
        build_paired_rows,
        dynamic_gate_verdict,
        masked_feedback_gates,
        reset_inference_seed,
        summarize_paired_rows,
        trajectory_knockout_metrics,
        validate_raw_rows,
        write_knockout_report,
        write_paired_csv,
        write_raw_csv,
    )


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "hybrid_state_gate_knockout_b1b_90k_raw.csv",
        output_dir / "hybrid_state_gate_knockout_b1b_90k_paired.csv",
        output_dir / "hybrid_state_gate_knockout_b1b_90k.md",
    )


def _validate_bootstrap_protocol(samples: int, seed: int) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")


def _expected_model_names(manifest: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for names in manifest.values() for name in names]


def _assert_checkpoint_gates(exchange: Any, original_gates: torch.Tensor) -> None:
    gates = exchange.feedback_gates.detach()
    if not torch.equal(gates, original_gates):
        raise RuntimeError("feedback gates no longer match the loaded checkpoint")


def run_gate_knockout_diagnostics(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    config_path: Path | None = None,
) -> list[dict]:
    """Run the five registered gate conditions for every frozen start-0 model."""
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
    _validate_bootstrap_protocol(bootstrap_samples, bootstrap_seed)
    validate_diagnostic_config(args, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    manifest = load_frozen_test_manifest()
    expected_models = _expected_model_names(manifest)
    dataset_root = Path(args.train_dataset.dataset_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    args.train_dataset.input_frames = 5
    args.train_dataset.output_frames = 1
    args.model_config.cond_frames = 5
    dataset = TrajDataset("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    records_by_model = _validate_records(
        records,
        dataset.split_lst_save,
        manifest,
    )
    start_zero_indices = select_start_zero_indices(dataset, expected_models)

    device = torch.device("cuda")
    model = MDM_ST(
        args.pc_size,
        1,
        n_feats=3,
        model_config=args.model_config,
    ).to(device)
    _load_checkpoint_strict(model, checkpoint, load_file)
    model.eval().requires_grad_(False)
    pipeline = TrajPipeline(model=model, scheduler=None)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, start_zero_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    exchange = model.dit.hybrid_state_exchange
    gates = exchange.feedback_gates.detach()
    if gates.shape != (4,) or not torch.isfinite(gates).all():
        raise ValueError("feedback_gates must contain four finite checkpoint values")
    original_gates = gates.clone()
    raw_rows = []
    evaluated_models = set()
    for batch, _ in dataloader:
        start_indices = batch.get("start_idx")
        if start_indices is None or not torch.all(start_indices == 0):
            raise ValueError("B1b gate knockout accepts only start_idx=0")
        names = [Path(name).name for name in batch["model"]]
        if len(names) != 1:
            raise ValueError("B1b gate knockout requires eval_batch_size=1")
        model_name = names[0]
        if model_name in evaluated_models:
            raise ValueError(f"{model_name}: evaluated more than once")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")

        gt = _build_raw_reference(batch, args.train_dataset)
        for condition, mask in KNOCKOUT_CONDITIONS:
            _assert_checkpoint_gates(exchange, original_gates)
            reset_inference_seed(args.seed, device)
            with masked_feedback_gates(exchange, mask):
                with _rollout_autocast_context(device):
                    pred = rollout_condition(
                        pipeline,
                        batch,
                        args,
                        record.log10_e,
                        record.nu,
                        record.mat_type,
                    )
            _assert_checkpoint_gates(exchange, original_gates)
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

    _assert_checkpoint_gates(exchange, original_gates)
    if evaluated_models != set(expected_models):
        missing = sorted(set(expected_models) - evaluated_models)
        unexpected = sorted(evaluated_models - set(expected_models))
        raise ValueError(
            "frozen model set mismatch after gate knockout: "
            f"missing={missing}; unexpected={unexpected}"
        )

    validated_raw = validate_raw_rows(raw_rows)
    paired_rows = build_paired_rows(validated_raw)
    summary_rows = summarize_paired_rows(
        paired_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    verdict = dynamic_gate_verdict(summary_rows)
    metadata = {
        "checkpoint": str(checkpoint),
        "config": (
            str(Path(config_path))
            if config_path is not None
            else DIRECT_CALL_CONFIG_PATH
        ),
        "seed": int(args.seed),
        "sample_scope": B1B_SAMPLE_SCOPE,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    output_raw = [{**row, **metadata} for row in validated_raw]
    output_paired = [{**row, **metadata} for row in paired_rows]
    raw_path, paired_path, report_path = _output_paths(output_dir)
    write_raw_csv(raw_path, output_raw)
    write_paired_csv(paired_path, output_paired)
    write_knockout_report(
        report_path,
        output_raw,
        summary_rows,
        metadata,
        original_gates.detach().cpu().tolist(),
        verdict,
    )
    print(f"Raw CSV: {raw_path.resolve()}")
    print(f"Paired CSV: {paired_path.resolve()}")
    print(f"Markdown: {report_path.resolve()}")
    print(
        "B1b gate knockout complete: "
        f"models={len(evaluated_models)} raw_rows={len(output_raw)} "
        f"paired_rows={len(output_paired)}"
    )
    return output_raw


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen B1b HST feedback-gate knockout diagnostic."
    )
    parser.add_argument("--config", required=True, help="Frozen B1b evaluation YAML.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="B1b checkpoint-90000 model.safetensors path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for raw, paired, and Markdown outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=10000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=_non_negative_int,
        default=0,
    )
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    config_path = Path(cli_args.config)
    checkpoint = Path(cli_args.checkpoint)
    args = OmegaConf.merge(
        OmegaConf.structured(TestingConfig),
        OmegaConf.load(config_path),
    )
    args.resume = str(checkpoint)
    run_gate_knockout_diagnostics(
        args,
        checkpoint,
        Path(cli_args.output_dir),
        cli_args.bootstrap_samples,
        cli_args.bootstrap_seed,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
