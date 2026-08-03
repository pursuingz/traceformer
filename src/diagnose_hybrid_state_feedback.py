"""Run eager B1b hybrid-state feedback diagnostics for the 90k checkpoint."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

try:  # Supports both ``python src/...`` and ``import src....``.
    from diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        load_material_records,
        rollout_condition,
    )
    from options import TestingConfig
    from utils.eval_metrics import per_window_metrics
    from utils.hybrid_state_diagnostics import (
        HybridStateFeedbackRecorder,
        write_feedback_csv,
        write_feedback_report,
    )
except ModuleNotFoundError:
    from src.diagnose_material_condition import (
        _build_raw_reference,
        _load_checkpoint_strict,
        load_material_records,
        rollout_condition,
    )
    from src.options import TestingConfig
    from src.utils.eval_metrics import per_window_metrics
    from src.utils.hybrid_state_diagnostics import (
        HybridStateFeedbackRecorder,
        write_feedback_csv,
        write_feedback_report,
    )


EXPECTED_MODEL_COUNT = 41
EXPECTED_MATERIAL_COUNTS = {0: 13, 1: 14, 2: 14}
ROLLOUT_STEPS = 20
STAGES_PER_STEP = 4
ROWS_PER_MODEL = ROLLOUT_STEPS * STAGES_PER_STEP
CHECKPOINT_SUFFIX = "checkpoint-90000/model.safetensors"


def _config_value(config: Any, field: str) -> Any:
    try:
        return getattr(config, field)
    except (AttributeError, KeyError):
        try:
            return config[field]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing diagnostic config field: {field}") from exc


def _checkpoint_matches(checkpoint: Path) -> bool:
    normalized = str(checkpoint).replace("\\", "/")
    return normalized.endswith(CHECKPOINT_SUFFIX)


def validate_diagnostic_config(args: Any, checkpoint: Path) -> None:
    """Reject configurations that cannot represent the fixed B1b diagnostic."""
    model_config = _config_value(args, "model_config")
    required = {
        "transformer_block": "SpatialTemporalTransformerBlockv11a",
        "contact_particle_cond": True,
    }
    for field, expected in required.items():
        actual = _config_value(model_config, field)
        if actual != expected:
            raise ValueError(
                f"B1b diagnostic requires {field}={expected!r}; got {actual!r}"
            )

    for field, expected in (("input_frames", 5), ("output_frames", 1), ("eval_batch_size", 1)):
        actual = _config_value(args, field)
        if actual != expected:
            raise ValueError(
                f"B1b diagnostic requires {field}={expected}; got {actual!r}"
            )
    if _config_value(args, "use_diffusion") is not False:
        raise ValueError("B1b diagnostic requires use_diffusion=False")
    if not _checkpoint_matches(checkpoint):
        raise ValueError(
            f"B1b diagnostic checkpoint must end with {CHECKPOINT_SUFFIX}: {checkpoint}"
        )


def trajectory_diagnostic_fields(
    pred: torch.Tensor,
    gt: torch.Tensor,
    input_frames: int,
) -> dict[str, float]:
    """Extract the trajectory fields repeated on every feedback row for one model."""
    if pred.shape != gt.shape:
        raise ValueError(f"trajectory shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
    if pred.ndim != 3 or pred.shape[0] <= 24:
        raise ValueError("trajectory diagnostic requires a (T,N,3) trajectory through frame 24")
    metrics = per_window_metrics(pred.float(), gt.float(), input_frames)
    try:
        centroid_error, _, _, shape_residual_mse = metrics["proc"][24]
    except KeyError as exc:
        raise ValueError("trajectory diagnostic requires Procrustes metrics at frame 24") from exc
    return {
        "full_rollout_mse": float(torch.mean((pred.float() - gt.float()).square()).item()),
        "fde": float(metrics["fde"]),
        "f24_centroid_error": float(centroid_error),
        "f24_shape_residual_mse": float(shape_residual_mse),
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "hybrid_state_feedback_b1b_90k.csv",
        output_dir / "hybrid_state_feedback_b1b_90k.md",
    )


def _validate_records(records: list[Any], dataset_models: list[str]) -> dict[str, Any]:
    record_names = [Path(record.model).name for record in records]
    dataset_names = [Path(name).name for name in dataset_models]
    if len(records) != EXPECTED_MODEL_COUNT or len(set(record_names)) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"expected {EXPECTED_MODEL_COUNT} unique material records")
    if len(dataset_names) != EXPECTED_MODEL_COUNT or len(set(dataset_names)) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"expected {EXPECTED_MODEL_COUNT} unique test models")
    if set(record_names) != set(dataset_names):
        raise ValueError("dataset and material metadata model sets differ")
    counts = dict(sorted(Counter(int(record.mat_type) for record in records).items()))
    if counts != EXPECTED_MATERIAL_COUNTS:
        raise ValueError(
            f"material counts mismatch: actual={counts}; expected={EXPECTED_MATERIAL_COUNTS}"
        )
    return {Path(record.model).name: record for record in records}


def _validate_completed_rows(rows: list[dict[str, Any]], evaluated_models: set[str]) -> None:
    if len(evaluated_models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODEL_COUNT} evaluated models, got {len(evaluated_models)}"
        )
    counts = Counter(int(row["mat_type"]) for row in rows)
    expected_rows = EXPECTED_MODEL_COUNT * ROWS_PER_MODEL
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} feedback rows, got {len(rows)}")
    expected_material_rows = {
        material: model_count * ROWS_PER_MODEL
        for material, model_count in EXPECTED_MATERIAL_COUNTS.items()
    }
    if dict(sorted(counts.items())) != expected_material_rows:
        raise ValueError(
            "feedback row material counts mismatch: "
            f"actual={dict(sorted(counts.items()))}; expected={expected_material_rows}"
        )


def run_feedback_diagnostics(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
) -> list[dict]:
    """Run exactly one eager start-0 rollout per B1b test model and write diagnostics."""
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
    validate_diagnostic_config(args, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    dataset_root = Path(_config_value(_config_value(args, "train_dataset"), "dataset_path"))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    args.train_dataset.input_frames = 5
    args.train_dataset.output_frames = 1
    args.model_config.cond_frames = 5
    dataset = TrajDataset("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    records_by_model = _validate_records(records, dataset.split_lst_save)

    model = MDM_ST(args.pc_size, 1, n_feats=3, model_config=args.model_config).to("cuda")
    _load_checkpoint_strict(model, checkpoint, load_file)
    model.eval().requires_grad_(False)
    pipeline = TrajPipeline(model=model, scheduler=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    rows: list[dict] = []
    evaluated_models: set[str] = set()
    for batch, _ in dataloader:
        start_indices = batch.get("start_idx")
        if start_indices is None or not torch.all(start_indices == 0):
            raise ValueError("B1b feedback diagnostics accepts only start_idx=0")
        names = [Path(name).name for name in batch["model"]]
        if len(names) != 1:
            raise ValueError("B1b feedback diagnostics requires eval_batch_size=1")
        model_name = names[0]
        if model_name in evaluated_models:
            raise ValueError(f"{model_name}: evaluated more than once")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")

        gt = _build_raw_reference(batch, args.train_dataset)
        with HybridStateFeedbackRecorder(model.dit.hybrid_state_exchange) as recorder:
            pred = rollout_condition(
                pipeline,
                batch,
                args,
                record.log10_e,
                record.nu,
                record.mat_type,
            )
        feedback_rows = recorder.finalize(expected_rollout_steps=ROLLOUT_STEPS)
        if len(feedback_rows) != ROWS_PER_MODEL:
            raise ValueError(
                f"{model_name}: expected {ROWS_PER_MODEL} feedback rows, got {len(feedback_rows)}"
            )
        trajectory = trajectory_diagnostic_fields(pred[0], gt[0], input_frames=5)
        for feedback_row in feedback_rows:
            feedback_row.update(
                {
                    "model": model_name,
                    "mat_type": record.mat_type,
                    "log10_e": record.log10_e,
                    "nu": record.nu,
                    **trajectory,
                }
            )
        rows.extend(feedback_rows)
        evaluated_models.add(model_name)

    _validate_completed_rows(rows, evaluated_models)
    csv_path, markdown_path = _output_paths(output_dir)
    config_path = getattr(args, "diagnostic_config_path", None)
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("diagnostic_config_path must record the CLI config path")
    write_feedback_csv(csv_path, rows)
    write_feedback_report(
        markdown_path,
        rows,
        {
            "checkpoint": str(checkpoint),
            "config": config_path,
        },
    )
    material_counts = Counter(int(record.mat_type) for record in records)
    print(f"CSV: {csv_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    print(
        "B1b feedback diagnostics complete: "
        f"models={len(evaluated_models)} rows={len(rows)} "
        f"material_counts=0:{material_counts[0]},1:{material_counts[1]},2:{material_counts[2]}"
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B1b eager HST feedback diagnostics.")
    parser.add_argument("--config", required=True, help="B1b evaluation YAML path.")
    parser.add_argument("--checkpoint", required=True, help="B1b checkpoint-90000 model.safetensors path.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic CSV and Markdown.")
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    config_path = Path(cli_args.config)
    checkpoint = Path(cli_args.checkpoint)
    args = OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config_path))
    args.resume = str(checkpoint)
    args.diagnostic_config_path = str(config_path)
    run_feedback_diagnostics(args, checkpoint, Path(cli_args.output_dir))


if __name__ == "__main__":
    main()
