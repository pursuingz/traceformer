"""Run the frozen B0.3 factual prediction-GT material-response audit."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

try:  # Supports both ``python src/...`` and ``import src....``.
    from diagnose_material_condition import (
        EXPECTED_MATERIAL_COUNTS,
        EXPECTED_MODEL_COUNT,
        _build_raw_reference,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from options import TestingConfig
    from utils.hybrid_state_gate_knockout import reset_inference_seed
    from utils.material_condition_diagnostics import trajectory_metrics
    from utils.material_response_fidelity import (
        MATERIAL_NAMES,
        RESPONSE_NAMES,
        build_alignment_summary,
        build_fidelity_summary,
        build_response_rows,
        preflight_fidelity_outputs,
        write_fidelity_outputs,
    )
except ModuleNotFoundError:
    from src.diagnose_material_condition import (
        EXPECTED_MATERIAL_COUNTS,
        EXPECTED_MODEL_COUNT,
        _build_raw_reference,
        _load_checkpoint_strict,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from src.options import TestingConfig
    from src.utils.hybrid_state_gate_knockout import reset_inference_seed
    from src.utils.material_condition_diagnostics import trajectory_metrics
    from src.utils.material_response_fidelity import (
        MATERIAL_NAMES,
        RESPONSE_NAMES,
        build_alignment_summary,
        build_fidelity_summary,
        build_response_rows,
        preflight_fidelity_outputs,
        write_fidelity_outputs,
    )


B03_PROFILE = "contact_cond90"
B03_SAMPLE_SCOPE = "frozen 41-model test start_idx=0 factual rollout"
B03_SCHEMA_VERSION = "1.0"
_POSITION_NORMALIZATION_SCALE = 2.0
_CONDITION_FRAME_RTOL = 1e-6
_CONDITION_FRAME_ATOL = 1e-6


@dataclass(frozen=True)
class RuntimeComponents:
    """Runtime dependencies kept injectable for CPU fake-runtime tests."""

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


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _frozen_seed(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed != 0:
        raise argparse.ArgumentTypeError("B0.3 requires seed=0")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _validate_runtime_arguments(
    seed: int, bootstrap_samples: int, contact_band_raw: float
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise ValueError("B0.3 requires seed=0")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(contact_band_raw, bool):
        raise ValueError("contact_band_raw must be finite and non-negative")
    try:
        contact_band = float(contact_band_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("contact_band_raw must be finite and non-negative") from exc
    if not np.isfinite(contact_band) or contact_band < 0.0:
        raise ValueError("contact_band_raw must be finite and non-negative")


def _normalized_checkpoint_path(path: str | Path) -> str:
    """Resolve equivalent CLI/config paths without requiring the file to exist."""
    return os.path.normcase(
        os.path.normpath(str(Path(path).expanduser().resolve(strict=False)))
    )


def _validate_cli_checkpoint_matches_config(args: Any, checkpoint: Path) -> None:
    config_resume = Path(str(args.resume))
    if _normalized_checkpoint_path(config_resume) != _normalized_checkpoint_path(
        checkpoint
    ):
        raise ValueError(
            "CLI --checkpoint must match config resume: "
            f"cli={checkpoint}; config={config_resume}"
        )


def _rollout_autocast_context(device: str | torch.device):
    """Match eval.py's CUDA bf16 rollout context without forcing CUDA in tests."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _scalar_batch_value(batch: dict[str, Any], field: str, model: str) -> float:
    if field not in batch:
        raise ValueError(f"{model}: batch is missing {field}")
    value = torch.as_tensor(batch[field]).detach().cpu().float()
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{model}: {field} must contain one finite scalar")
    return float(value.reshape(-1)[0])


def _batch_model_name(batch: dict[str, Any]) -> str:
    names = [Path(name).name for name in batch.get("model", [])]
    if len(names) != 1:
        raise ValueError("B0.3 requires eval_batch_size=1")
    return names[0]


def _is_start_zero_batch(batch: dict[str, Any]) -> bool:
    if "start_idx" not in batch:
        raise ValueError("B0.3 batch is missing start_idx")
    indices = torch.as_tensor(batch["start_idx"])
    if indices.numel() != 1:
        if bool(torch.any(indices == 0)) and bool(torch.any(indices != 0)):
            raise ValueError("start-0 and nonzero windows must not share a batch")
        raise ValueError("B0.3 requires eval_batch_size=1")
    start_index = float(indices.reshape(-1)[0].item())
    if not np.isfinite(start_index) or not start_index.is_integer():
        raise ValueError("B0.3 start_idx must be one finite integer")
    return int(start_index) == 0


def _build_inference_stack(
    args: Any, checkpoint: Path, runtime: RuntimeComponents
) -> Any:
    device = getattr(args, "device", "cuda")
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


def _prediction_trajectory(output: Any, input_frames: int, model: str) -> torch.Tensor:
    trajectory = torch.as_tensor(output).detach().cpu().float()
    if (
        trajectory.ndim != 4
        or trajectory.shape[0] != 1
        or trajectory.shape[1] != 25
        or trajectory.shape[-1] != 3
    ):
        raise ValueError(
            f"{model}: factual rollout must have shape (1, 25, N, 3); "
            f"got {tuple(trajectory.shape)}"
        )
    if trajectory.shape[1] <= input_frames or not bool(torch.isfinite(trajectory).all()):
        raise ValueError(f"{model}: factual rollout must be finite and include predictions")
    return trajectory[0]


def _reference_trajectory(reference: Any, model: str) -> torch.Tensor:
    trajectory = torch.as_tensor(reference).detach().cpu().float()
    if trajectory.ndim != 4 or trajectory.shape[0] != 1 or trajectory.shape[1] != 25:
        raise ValueError(
            f"{model}: ground truth must have shape (1, 25, N, 3); "
            f"got {tuple(trajectory.shape)}"
        )
    if trajectory.shape[-1] != 3 or not bool(torch.isfinite(trajectory).all()):
        raise ValueError(f"{model}: ground truth must be finite (1, 25, N, 3)")
    return trajectory[0]


def _validate_condition_frame_alignment(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    input_frames: int,
    model: str,
) -> None:
    """Lock observed condition-frame particle correspondence before scoring futures."""
    if input_frames < 1 or input_frames > pred.shape[0] or input_frames > gt.shape[0]:
        raise ValueError(f"{model}: invalid input_frames for condition alignment")
    pred_condition = pred[:input_frames]
    gt_condition = gt[:input_frames]
    if not bool(torch.isfinite(pred_condition).all()) or not bool(
        torch.isfinite(gt_condition).all()
    ):
        raise ValueError(f"{model}: conditioning frames must be finite")
    if not torch.allclose(
        pred_condition,
        gt_condition,
        rtol=_CONDITION_FRAME_RTOL,
        atol=_CONDITION_FRAME_ATOL,
    ):
        max_error = float(torch.max(torch.abs(pred_condition - gt_condition)).item())
        raise ValueError(
            f"{model}: prediction/GT conditioning frames are not pointwise aligned "
            f"(rtol={_CONDITION_FRAME_RTOL}, atol={_CONDITION_FRAME_ATOL}, "
            f"max_abs_error={max_error:.6g})"
        )


def _model_row(
    *,
    record: Any,
    metadata: dict[str, Any],
    pred: torch.Tensor,
    gt: torch.Tensor,
    input_frames: int,
) -> dict[str, Any]:
    metrics = trajectory_metrics(
        pred,
        gt,
        input_frames,
        model=record.model,
        intervention="factual",
    )
    return {
        "model": record.model,
        "mat_type": int(record.mat_type),
        "material": MATERIAL_NAMES[int(record.mat_type)],
        "log10_e": float(record.log10_e),
        "nu": float(record.nu),
        **metadata,
        **metrics,
    }


def _validate_evaluated_models(evaluated: set[str], records_by_model: dict[str, Any]) -> None:
    expected = set(records_by_model)
    if evaluated != expected:
        missing = sorted(expected - evaluated)
        unexpected = sorted(evaluated - expected)
        raise ValueError(
            "B0.3 evaluated-model mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )


def run_material_response_fidelity(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
    config_path: Path,
    seed: int,
    bootstrap_samples: int,
    contact_band_raw: float,
    overwrite: bool,
    runtime: RuntimeComponents | None = None,
) -> dict[str, Path]:
    """Run exactly one normal-condition rollout for each frozen start-0 model."""
    _validate_runtime_arguments(seed, bootstrap_samples, contact_band_raw)
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    config_path = Path(config_path)

    # This occurs before model construction or HDF5 traversal, preventing costly work
    # when a complete six-file report already exists.
    preflight_fidelity_outputs(output_dir, overwrite=overwrite)

    # Validate the YAML identity before comparing it with the CLI path. Mutating
    # args.resume here would hide a config/checkpoint mismatch from the strict
    # contact_cond90 profile check.
    _validate_b0_identity(args, profile=B03_PROFILE)
    _validate_cli_checkpoint_matches_config(args, checkpoint)
    dataset_root = Path(args.train_dataset.dataset_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    input_frames = int(args.input_frames)
    output_frames = int(args.output_frames)
    if input_frames <= 0 or output_frames <= 0:
        raise ValueError("input_frames and output_frames must be positive")
    args.train_dataset.input_frames = input_frames
    args.train_dataset.output_frames = output_frames
    args.model_config.cond_frames = input_frames
    args.seed = seed
    # CLI uses raw MPM units. GT/pred/floor_height below use TrajDataset's
    # normalized coordinates: (raw - norm_fac) / 2.
    contact_band = float(contact_band_raw) / _POSITION_NORMALIZATION_SCALE

    runtime = runtime or _load_runtime_components()
    dataset = runtime.dataset_cls("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    _validate_b0_identity(args, records=records, profile=B03_PROFILE)
    records_by_model = {record.model: record for record in records}
    if len(records_by_model) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODEL_COUNT} unique material records, "
            f"got {len(records_by_model)}"
        )
    counts = Counter(record.mat_type for record in records_by_model.values())
    if dict(sorted(counts.items())) != EXPECTED_MATERIAL_COUNTS:
        raise ValueError("B0.3 material counts must be elastic=13, plasticine=14, sand=14")

    pipeline = _build_inference_stack(args, checkpoint, runtime)
    dataloader = runtime.dataloader_cls(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    provenance = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": seed,
        "sample_scope": B03_SAMPLE_SCOPE,
    }
    model_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    evaluated_models: set[str] = set()

    for batch, _ in dataloader:
        if not _is_start_zero_batch(batch):
            continue
        model_name = _batch_model_name(batch)
        if model_name in evaluated_models:
            raise ValueError(f"{model_name}: evaluated more than once")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")

        reset_inference_seed(seed, getattr(args, "device", "cuda"))
        _validate_normal_material_condition(batch, record)
        with _rollout_autocast_context(getattr(args, "device", "cuda")):
            output = rollout_condition(
                pipeline,
                batch,
                args,
                record.log10_e,
                record.nu,
                record.mat_type,
            )
        pred = _prediction_trajectory(output, input_frames, record.model)
        gt = _reference_trajectory(
            _build_raw_reference(batch, args.train_dataset), record.model
        )
        if pred.shape != gt.shape:
            raise ValueError(
                f"{record.model}: prediction/GT shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(gt.shape)}"
            )
        _validate_condition_frame_alignment(
            pred,
            gt,
            input_frames=input_frames,
            model=record.model,
        )
        floor_height = _scalar_batch_value(batch, "floor_height", record.model)
        row = _model_row(
            record=record,
            metadata=provenance,
            pred=pred,
            gt=gt,
            input_frames=input_frames,
        )
        model_rows.append(row)
        response_rows.extend(
            build_response_rows(
                row,
                gt=gt,
                pred=pred,
                floor_height=floor_height,
                contact_band_raw=contact_band,
            )
        )
        evaluated_models.add(model_name)
        print(f"[{len(evaluated_models):02d}/{EXPECTED_MODEL_COUNT}] {model_name}")

    _validate_evaluated_models(evaluated_models, records_by_model)
    fidelity_rows = build_fidelity_summary(
        response_rows, bootstrap_samples=bootstrap_samples, seed=seed
    )
    alignment_rows = build_alignment_summary(
        response_rows, bootstrap_samples=bootstrap_samples, seed=seed
    )
    metadata = {
        "schema_version": B03_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": seed,
        "split": B03_SAMPLE_SCOPE,
        "model_counts": {"elastic": 13, "plasticine": 14, "sand": 14},
        "response_schema": list(RESPONSE_NAMES),
        "bootstrap_samples": bootstrap_samples,
        "contact_band_raw": float(contact_band_raw),
        "contact_band_normalized": contact_band,
    }
    paths = write_fidelity_outputs(
        output_dir,
        model_rows=model_rows,
        response_rows=response_rows,
        fidelity_rows=fidelity_rows,
        alignment_rows=alignment_rows,
        metadata=metadata,
        overwrite=overwrite,
    )
    print(f"B0.3 factual response fidelity complete: models={len(evaluated_models)}")
    for name, path in paths.items():
        print(f"{name}: {Path(path).resolve()}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B0.3 factual prediction-GT material-response audit."
    )
    parser.add_argument("--config", required=True, help="Registered evaluation YAML.")
    parser.add_argument("--checkpoint", required=True, help="Strict matching checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="results/material_response_fidelity_b03",
        help="Directory for the fixed six-file B0.3 report.",
    )
    parser.add_argument("--seed", type=_frozen_seed, default=0)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=10000)
    parser.add_argument("--contact-band-raw", type=_non_negative_float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    config_path = Path(cli_args.config)
    args = OmegaConf.merge(
        OmegaConf.structured(TestingConfig), OmegaConf.load(config_path)
    )
    run_material_response_fidelity(
        args,
        checkpoint=Path(cli_args.checkpoint),
        output_dir=Path(cli_args.output_dir),
        config_path=config_path,
        seed=cli_args.seed,
        bootstrap_samples=cli_args.bootstrap_samples,
        contact_band_raw=cli_args.contact_band_raw,
        overwrite=cli_args.overwrite,
    )


if __name__ == "__main__":
    main()
