"""Run the frozen B2 one-dimensional E/nu material-response sweep."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

try:  # Supports both ``python src/...`` and ``import src....``.
    from diagnose_material_condition import (
        EXPECTED_MODEL_COUNT,
        ROLLOUT_HORIZON,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from options import TestingConfig
    from utils.hybrid_state_gate_knockout import reset_inference_seed
    from utils.material_response_sweep import (
        SWEEP_CONDITIONS,
        build_group_summaries,
        build_model_summaries,
        build_sweep_conditions,
        response_metrics,
        trajectory_state_metrics,
        validate_raw_rows,
        write_sweep_outputs,
    )
except ModuleNotFoundError:
    from src.diagnose_material_condition import (
        EXPECTED_MODEL_COUNT,
        ROLLOUT_HORIZON,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from src.options import TestingConfig
    from src.utils.hybrid_state_gate_knockout import reset_inference_seed
    from src.utils.material_response_sweep import (
        SWEEP_CONDITIONS,
        build_group_summaries,
        build_model_summaries,
        build_sweep_conditions,
        response_metrics,
        trajectory_state_metrics,
        validate_raw_rows,
        write_sweep_outputs,
    )


B2_PROFILE = "contact_cond90"
B2_SAMPLE_SCOPE = "frozen 41-model start_idx=0 material-response sweep"


@dataclass(frozen=True)
class RuntimeComponents:
    dataset_cls: Any
    model_cls: Any
    pipeline_cls: Any
    checkpoint_loader: Callable[..., Any]
    dataloader_cls: Callable[..., Any]
    compile_model: Callable[[Any], Any]


def _load_runtime_components() -> RuntimeComponents:
    from safetensors.torch import load_file

    try:
        from dataset.traj_dataset import TrajDataset
        from model.spacetime import MDM_ST
        from pipeline_traj import TrajPipeline
    except ModuleNotFoundError:
        from src.dataset.traj_dataset import TrajDataset
        from src.model.spacetime import MDM_ST
        from src.pipeline_traj import TrajPipeline

    return RuntimeComponents(
        dataset_cls=TrajDataset,
        model_cls=MDM_ST,
        pipeline_cls=TrajPipeline,
        checkpoint_loader=load_file,
        dataloader_cls=torch.utils.data.DataLoader,
        compile_model=torch.compile,
    )


def _finite_batch_scalar(batch: dict[str, Any], field: str, model: str) -> float:
    if field not in batch:
        raise ValueError(f"{model}: batch is missing {field}")
    value = torch.as_tensor(batch[field]).detach().cpu().float()
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{model}: {field} must contain one finite scalar")
    return float(value.reshape(-1)[0])


def _prediction_horizon(
    output: Any,
    input_frames: int,
    model: str,
    condition: str,
) -> torch.Tensor:
    trajectory = torch.as_tensor(output).detach().cpu().float()
    expected_frames = input_frames + ROLLOUT_HORIZON
    if (
        trajectory.ndim != 4
        or trajectory.shape[0] != 1
        or trajectory.shape[1] != expected_frames
        or trajectory.shape[-1] != 3
    ):
        raise ValueError(
            f"{model}/{condition}: rollout must have shape "
            f"(1, {expected_frames}, N, 3); got {tuple(trajectory.shape)}"
        )
    if not bool(torch.isfinite(trajectory).all()):
        raise ValueError(f"{model}/{condition}: rollout contains non-finite values")
    return trajectory[0, input_frames:]


def _rollout_autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_sweep_conditions(
    pipeline: Any,
    batch: dict[str, Any],
    args: Any,
    record: Any,
    seed: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the seven paired B2 conditions for one frozen start-0 model."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_normal_material_condition(batch, record)

    input_frames = int(args.input_frames)
    points_src = torch.as_tensor(batch["points_src"]).detach().cpu().float()
    if (
        points_src.ndim != 4
        or points_src.shape[0] != 1
        or points_src.shape[1] != input_frames
        or points_src.shape[-1] != 3
    ):
        raise ValueError(
            f"{record.model}: points_src must have shape (1, {input_frames}, N, 3)"
        )
    reference_frame = points_src[0, -1]
    floor_height = _finite_batch_scalar(batch, "floor_height", record.model)
    conditions = build_sweep_conditions(record)
    device = torch.device(getattr(args, "device", "cuda"))

    predictions: dict[str, torch.Tensor] = {}
    with _rollout_autocast_context(device):
        for name in SWEEP_CONDITIONS:
            condition = conditions[name]
            reset_inference_seed(seed, device)
            output = rollout_condition(
                pipeline,
                batch,
                args,
                condition.log10_e,
                condition.nu,
                condition.mat_type,
            )
            predictions[name] = _prediction_horizon(
                output, input_frames, record.model, name
            )

    normal = predictions["normal"]
    rows: list[dict[str, Any]] = []
    for name in SWEEP_CONDITIONS:
        condition = conditions[name]
        prediction = predictions[name]
        row = {
            **metadata,
            "model": record.model,
            "mat_type": int(record.mat_type),
            "true_log10_e": float(record.log10_e),
            "true_nu": float(record.nu),
            "condition": name,
            "scanned_log10_e": float(condition.log10_e),
            "scanned_nu": float(condition.nu),
        }
        row.update(
            trajectory_state_metrics(prediction, reference_frame, floor_height)
        )
        row.update(response_metrics(normal, prediction))
        rows.append(row)
    return rows


def _build_inference_stack(
    args: Any,
    checkpoint: Path,
    runtime: RuntimeComponents,
) -> Any:
    device = "cuda"
    model = runtime.model_cls(
        args.pc_size,
        int(args.output_frames),
        n_feats=3,
        model_config=args.model_config,
    ).to(device)
    _load_checkpoint_strict(model, checkpoint, runtime.checkpoint_loader)
    model.eval().requires_grad_(False)
    model = runtime.compile_model(model)
    return runtime.pipeline_cls(model=model, scheduler=None)


def run_material_response_sweep(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
    config_path: Path,
    seed: int,
    runtime: RuntimeComponents | None = None,
) -> list[dict[str, Any]]:
    """Load the frozen arm once, run 287 paired rollouts, and write B2 outputs."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    config_path = Path(config_path)
    args.resume = str(checkpoint)
    _validate_b0_identity(args, profile=B2_PROFILE)

    dataset_root = Path(args.train_dataset.dataset_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    input_frames = int(args.input_frames)
    output_frames = int(args.output_frames)
    args.train_dataset.input_frames = input_frames
    args.train_dataset.output_frames = output_frames

    runtime = runtime or _load_runtime_components()
    dataset = runtime.dataset_cls("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    _validate_b0_identity(args, records=records, profile=B2_PROFILE)
    args.model_config.cond_frames = input_frames
    records_by_model = {record.model: record for record in records}
    if len(records_by_model) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODEL_COUNT} unique material records, "
            f"got {len(records_by_model)}"
        )

    # CLI seed controls both the explicit pipeline generator and global RNG reset.
    args.seed = seed
    pipeline = _build_inference_stack(args, checkpoint, runtime)
    dataloader = runtime.dataloader_cls(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": seed,
        "sample_scope": B2_SAMPLE_SCOPE,
    }

    raw_rows: list[dict[str, Any]] = []
    evaluated_models: set[str] = set()
    for batch, _ in dataloader:
        start_indices = batch.get("start_idx")
        if start_indices is None:
            raise ValueError("B2 batch is missing start_idx")
        start_indices = torch.as_tensor(start_indices)
        if not bool(torch.all(start_indices == 0)):
            if bool(torch.any(start_indices == 0)):
                raise ValueError("start-0 and nonzero windows must not share a batch")
            continue

        names = [Path(name).name for name in batch.get("model", [])]
        if len(names) != 1:
            raise ValueError("B2 requires eval_batch_size=1")
        model_name = names[0]
        if model_name in evaluated_models:
            raise ValueError(f"{model_name}: evaluated more than once")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")

        raw_rows.extend(
            evaluate_sweep_conditions(
                pipeline,
                batch,
                args,
                record,
                seed=seed,
                metadata=metadata,
            )
        )
        evaluated_models.add(model_name)
        print(
            f"[{len(evaluated_models):02d}/{EXPECTED_MODEL_COUNT}] "
            f"{model_name}: {len(SWEEP_CONDITIONS)} conditions"
        )

    expected_models = set(records_by_model)
    if evaluated_models != expected_models:
        missing = sorted(expected_models - evaluated_models)
        unexpected = sorted(evaluated_models - expected_models)
        raise ValueError(
            "B2 evaluated-model mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )

    condition_order = {name: index for index, name in enumerate(SWEEP_CONDITIONS)}
    raw_rows.sort(key=lambda row: (str(row["model"]), condition_order[row["condition"]]))
    validate_raw_rows(raw_rows)
    model_rows = build_model_summaries(raw_rows)
    summary_rows = build_group_summaries(model_rows)
    paths = write_sweep_outputs(
        output_dir,
        raw_rows,
        model_rows,
        summary_rows,
        metadata,
    )
    print(
        "B2 material-response sweep complete: "
        f"models={len(evaluated_models)} rollouts={len(raw_rows)}"
    )
    for name, path in paths.items():
        print(f"{name}: {Path(path).resolve()}")
    return raw_rows


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
        description="Run the frozen mm3_contact_cond@90k B2 E/nu response sweep."
    )
    parser.add_argument(
        "--config", required=True, help="Frozen mm3_contact_cond evaluation YAML."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Frozen mm3_contact_cond checkpoint-90000 model.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/material_response_sweep_b2",
        help="Directory for the four B2 output files.",
    )
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    config_path = Path(cli_args.config)
    checkpoint = Path(cli_args.checkpoint)
    args = OmegaConf.merge(
        OmegaConf.structured(TestingConfig),
        OmegaConf.load(config_path),
    )
    run_material_response_sweep(
        args,
        checkpoint=checkpoint,
        output_dir=Path(cli_args.output_dir),
        config_path=config_path,
        seed=cli_args.seed,
    )


if __name__ == "__main__":
    main()
