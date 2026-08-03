"""Run eager B1b hybrid-state feedback diagnostics for the 90k checkpoint."""

import argparse
import json
import posixpath
from collections import Counter
from contextlib import nullcontext
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


ROLLOUT_STEPS = 20
STAGES_PER_STEP = 4
ROWS_PER_MODEL = ROLLOUT_STEPS * STAGES_PER_STEP
CHECKPOINT_SUFFIX = "checkpoint-90000/model.safetensors"
DIRECT_CALL_CONFIG_PATH = "<not-provided-direct-call>"
B1B_DATASET_PATH = "mm3_data/mm3_test"
B1B_SAMPLE_SCOPE = "frozen 41-model start_idx=0 full-horizon B1b test"
_SOURCE_DIR = Path(__file__).resolve().parent
_FROZEN_MANIFEST_PATH = _SOURCE_DIR / "configs" / "mm3_test_split.json"
_CANONICAL_EVAL_CONFIG_PATH = (
    _SOURCE_DIR / "configs" / "eval_mm3_v11a_contact_cond_8L_45k.yaml"
)
_MATERIAL_TYPES = {"elastic": 0, "plasticine": 1, "sand": 2}
_TOP_LEVEL_FIELDS = (
    "pc_size",
    "eval_batch_size",
    "dataloader_num_workers",
    "seed",
    "pred_offset",
    "model_type",
    "use_diffusion",
    "num_inference_steps",
    "output_frames",
)
_MODEL_CONFIG_FIELDS = (
    "n_layers",
    "latent_dim",
    "frame_cond",
    "point_embed",
    "mask_cond",
    "pred_offset",
    "num_neighbors",
    "floor_cond",
    "max_num_forces",
    "force_as_token",
    "force_as_latent",
    "gravity_emb",
    "coeff_cond",
    "num_mat",
    "class_token",
    "transformer_block",
    "hybrid_state_dim",
    "hybrid_state_heads",
    "hybrid_state_interval",
    "contact_particle_cond",
    "contact_feature_sigma",
)
_DATASET_FIELDS = (
    "category",
    "dataset_list",
    "has_gravity",
    "max_num_forces",
    "norm_fac",
    "stage",
    "mode",
    "pc_size",
    "repeat",
    "seed",
    "n_sample_pro_model",
    "n_frames_interval",
    "n_training_frames",
    "batch_size",
    "overfit",
    "input_frames",
    "output_frames",
)


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


def _normalized_dataset_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("B1b diagnostic dataset_path must be a non-empty string")
    normalized = posixpath.normpath(value.strip().replace("\\", "/"))
    return normalized[2:] if normalized.startswith("./") else normalized


def load_frozen_test_manifest() -> dict[str, tuple[str, ...]]:
    """Load the canonical B1b test-model manifest from the frozen split."""
    try:
        manifest_data = json.loads(_FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
        raw_manifest = manifest_data["frozen_test_models"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid frozen B1b manifest: {_FROZEN_MANIFEST_PATH}") from exc

    if not isinstance(raw_manifest, dict) or tuple(raw_manifest) != tuple(_MATERIAL_TYPES):
        raise ValueError("frozen B1b manifest must contain elastic, plasticine, and sand")
    manifest: dict[str, tuple[str, ...]] = {}
    for material in _MATERIAL_TYPES:
        names = raw_manifest[material]
        if not isinstance(names, list) or not names:
            raise ValueError(f"frozen B1b manifest has no {material} models")
        normalized_names = tuple(Path(name).name for name in names)
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError(f"frozen B1b manifest has invalid {material} model names")
        manifest[material] = normalized_names

    all_names = [name for names in manifest.values() for name in names]
    if len(all_names) != 41 or len(all_names) != len(set(all_names)):
        raise ValueError("frozen B1b manifest must contain exactly 41 unique model names")
    return manifest


def _frozen_model_names(manifest: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for material in _MATERIAL_TYPES for name in manifest[material]]


def _validate_field_matches_canonical(
    actual_config: Any,
    canonical_config: Any,
    fields: tuple[str, ...],
    section: str,
) -> None:
    for field in fields:
        actual = _config_value(actual_config, field)
        expected = _config_value(canonical_config, field)
        if actual != expected or (
            isinstance(expected, bool) and not isinstance(actual, bool)
        ):
            raise ValueError(
                f"B1b diagnostic requires {section}.{field}={expected!r}; got {actual!r}"
            )


def validate_diagnostic_config(args: Any, checkpoint: Path) -> None:
    """Reject configurations that cannot represent the fixed B1b diagnostic."""
    canonical = OmegaConf.load(_CANONICAL_EVAL_CONFIG_PATH)
    _validate_field_matches_canonical(args, canonical, _TOP_LEVEL_FIELDS, "config")
    _validate_field_matches_canonical(
        _config_value(args, "model_config"),
        _config_value(canonical, "model_config"),
        _MODEL_CONFIG_FIELDS,
        "model_config",
    )
    dataset_config = _config_value(args, "train_dataset")
    canonical_dataset = _config_value(canonical, "train_dataset")
    _validate_field_matches_canonical(
        dataset_config,
        canonical_dataset,
        _DATASET_FIELDS,
        "train_dataset",
    )
    if _normalized_dataset_path(_config_value(dataset_config, "dataset_path")) != B1B_DATASET_PATH:
        raise ValueError(
            f"B1b diagnostic requires train_dataset.dataset_path={B1B_DATASET_PATH!r}"
        )
    if _config_value(args, "input_frames") != 5:
        raise ValueError("B1b diagnostic requires input_frames=5")
    if not _checkpoint_matches(checkpoint):
        raise ValueError(
            f"B1b diagnostic checkpoint must end with {CHECKPOINT_SUFFIX}: {checkpoint}"
        )


def _move_ground_truth_to_prediction_device(pred: Any, gt: torch.Tensor) -> torch.Tensor:
    """Align ground truth with the rollout device before calculating metrics."""
    return gt.to(pred.device)


def _rollout_autocast_context(device: torch.device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    if not isinstance(input_frames, int) or not 0 < input_frames < pred.shape[0]:
        raise ValueError("trajectory diagnostic input_frames must select prediction frames")
    gt = _move_ground_truth_to_prediction_device(pred, gt)
    pred_float = pred.float()
    gt_float = gt.float()
    metrics = per_window_metrics(pred_float, gt_float, input_frames)
    try:
        centroid_error, _, _, shape_residual_mse = metrics["proc"][24]
    except KeyError as exc:
        raise ValueError("trajectory diagnostic requires Procrustes metrics at frame 24") from exc
    return {
        "full_rollout_mse": float(
            torch.mean((pred_float[input_frames:] - gt_float[input_frames:]).square()).item()
        ),
        "fde": float(metrics["fde"]),
        "f24_centroid_error": float(centroid_error),
        "f24_shape_residual_mse": float(shape_residual_mse),
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "hybrid_state_feedback_b1b_90k.csv",
        output_dir / "hybrid_state_feedback_b1b_90k.md",
    )


def select_start_zero_indices(
    dataset: Any,
    expected_models: list[str] | None = None,
) -> list[int]:
    """Select the one fixed start-0 diagnostic window for every test model."""
    try:
        entries = dataset.models
        dataset_models = [Path(name).name for name in dataset.split_lst_save]
    except AttributeError as exc:
        raise ValueError("diagnostic dataset must expose models and split_lst_save") from exc
    if expected_models is None:
        expected_models = dataset_models
    if len(expected_models) != len(set(expected_models)):
        raise ValueError("diagnostic dataset split contains duplicate model names")
    if set(dataset_models) != set(expected_models):
        raise ValueError("frozen model set mismatch between dataset and manifest")

    selected: dict[str, int] = {}
    for index, entry in enumerate(entries):
        try:
            model_name = Path(entry["model"]).name
            start_idx = int(entry["start_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid diagnostic dataset entry at index {index}") from exc
        if start_idx != 0:
            continue
        if model_name in selected:
            raise ValueError(f"{model_name}: duplicate start_idx=0 diagnostic window")
        selected[model_name] = index

    expected_set = set(expected_models)
    selected_set = set(selected)
    if selected_set != expected_set:
        missing = sorted(expected_set - selected_set)
        unexpected = sorted(selected_set - expected_set)
        raise ValueError(
            f"start_idx=0 selection mismatch: missing={missing}; unexpected={unexpected}"
        )
    return [selected[model_name] for model_name in expected_models]


def _validate_records(
    records: list[Any],
    dataset_models: list[str],
    manifest: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    record_names = [Path(record.model).name for record in records]
    dataset_names = [Path(name).name for name in dataset_models]
    expected_names = _frozen_model_names(manifest)
    if len(dataset_names) != len(set(dataset_names)) or set(dataset_names) != set(expected_names):
        raise ValueError("frozen model set mismatch between dataset and manifest")
    if len(record_names) != len(set(record_names)) or set(record_names) != set(expected_names):
        raise ValueError("frozen model set mismatch between material metadata and manifest")
    expected_types = {
        name: _MATERIAL_TYPES[material]
        for material, names in manifest.items()
        for name in names
    }
    for record in records:
        model_name = Path(record.model).name
        if int(record.mat_type) != expected_types[model_name]:
            raise ValueError(f"{model_name}: material metadata disagrees with frozen manifest")
    return {Path(record.model).name: record for record in records}


def _validate_completed_rows(
    rows: list[dict[str, Any]],
    evaluated_models: set[str],
    expected_models: list[str],
) -> None:
    if evaluated_models != set(expected_models):
        raise ValueError(
            "frozen model set mismatch after diagnostic evaluation"
        )
    counts = Counter(int(row["mat_type"]) for row in rows)
    expected_rows = len(expected_models) * ROWS_PER_MODEL
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} feedback rows, got {len(rows)}")
    if set(counts) != set(_MATERIAL_TYPES.values()):
        raise ValueError(
            f"feedback row material groups mismatch: {dict(sorted(counts.items()))}"
        )


def run_feedback_diagnostics(
    args: Any,
    checkpoint: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
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
    manifest = load_frozen_test_manifest()
    expected_models = _frozen_model_names(manifest)
    dataset_root = Path(_config_value(_config_value(args, "train_dataset"), "dataset_path"))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

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
            with _rollout_autocast_context(device):
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

    _validate_completed_rows(rows, evaluated_models, expected_models)
    csv_path, markdown_path = _output_paths(output_dir)
    report_config_path = (
        str(Path(config_path))
        if config_path is not None
        else DIRECT_CALL_CONFIG_PATH
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "config": report_config_path,
        "sample_scope": B1B_SAMPLE_SCOPE,
    }
    write_feedback_csv(csv_path, rows, metadata)
    write_feedback_report(
        markdown_path,
        rows,
        metadata,
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
    run_feedback_diagnostics(
        args,
        checkpoint,
        Path(cli_args.output_dir),
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
