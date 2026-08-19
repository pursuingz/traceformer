"""Run the two-stage B0.4 local deformation response fidelity audit."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import OmegaConf

try:  # Supports execution from src/ and import through src.*.
    from diagnose_material_condition import (
        EXPECTED_MATERIAL_COUNTS,
        EXPECTED_MODEL_COUNT,
        _build_raw_reference,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from diagnose_material_response_fidelity import (
        B03_PROFILE,
        RuntimeComponents,
        _batch_model_name,
        _build_inference_stack,
        _is_start_zero_batch,
        _load_runtime_components,
        _prediction_trajectory,
        _reference_trajectory,
        _rollout_autocast_context,
        _validate_cli_checkpoint_matches_config,
        _validate_condition_frame_alignment,
    )
    from options import TestingConfig
    from utils.hybrid_state_gate_knockout import reset_inference_seed
    from utils.local_deformation_fidelity import (
        MATERIAL_NAMES,
        build_calibration_row,
        build_local_fidelity_rows,
        build_local_test_rows,
        build_rest_neighborhood,
        estimate_local_deformation,
        evaluate_calibration_gate,
        preflight_local_deformation_outputs,
        write_local_deformation_outputs,
    )
except ModuleNotFoundError:
    from src.diagnose_material_condition import (
        EXPECTED_MATERIAL_COUNTS,
        EXPECTED_MODEL_COUNT,
        _build_raw_reference,
        _validate_b0_identity,
        _validate_normal_material_condition,
        load_material_records,
        rollout_condition,
    )
    from src.diagnose_material_response_fidelity import (
        B03_PROFILE,
        RuntimeComponents,
        _batch_model_name,
        _build_inference_stack,
        _is_start_zero_batch,
        _load_runtime_components,
        _prediction_trajectory,
        _reference_trajectory,
        _rollout_autocast_context,
        _validate_cli_checkpoint_matches_config,
        _validate_condition_frame_alignment,
    )
    from src.options import TestingConfig
    from src.utils.hybrid_state_gate_knockout import reset_inference_seed
    from src.utils.local_deformation_fidelity import (
        MATERIAL_NAMES,
        build_calibration_row,
        build_local_fidelity_rows,
        build_local_test_rows,
        build_rest_neighborhood,
        estimate_local_deformation,
        evaluate_calibration_gate,
        preflight_local_deformation_outputs,
        write_local_deformation_outputs,
    )


B04_SCHEMA_VERSION = "1.0"
B04_SAMPLE_SCOPE = "frozen 41-model test start_idx=0 factual rollout"
PRIMARY_K = 16
SENSITIVITY_K = (8, 32)
ALL_K = (8, 16, 32)
CONDITION_THRESHOLD = 1e6
REGULARIZATION_SCALE = 1e-6


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _frozen_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed != 0:
        raise argparse.ArgumentTypeError("B0.4 requires seed=0")
    return parsed


def _read_calibration_header(path: Path) -> int:
    required = ("x", "F", "E", "nu", "mat_type")
    with h5py.File(path, "r") as handle:
        missing = [field for field in required if field not in handle]
        if missing:
            raise ValueError(f"{path.name}: missing calibration fields {missing}")
        if handle["x"].ndim != 3 or handle["x"].shape[0] < 25 or handle["x"].shape[-1] != 3:
            raise ValueError(f"{path.name}: x must have shape (>=25, N, 3)")
        particle_count = int(handle["x"].shape[1])
        f_shape = tuple(handle["F"].shape)
        accepted = ((handle["x"].shape[0], particle_count, 9), (
            handle["x"].shape[0], particle_count, 3, 3
        ))
        if f_shape not in accepted:
            raise ValueError(f"{path.name}: F shape does not align with x")
        mat_value = np.asarray(handle["mat_type"][()])
        if mat_value.size != 1:
            raise ValueError(f"{path.name}: mat_type must be scalar")
        numeric = float(mat_value.reshape(()))
        if not np.isfinite(numeric) or not numeric.is_integer() or int(numeric) not in MATERIAL_NAMES:
            raise ValueError(f"{path.name}: mat_type must be 0, 1, or 2")
        return int(numeric)


def select_calibration_paths(
    train_dir: str | Path,
    *,
    per_material: int,
    seed: int,
) -> list[tuple[int, Path]]:
    """Select a deterministic stratified calibration subset without reading trajectories."""
    directory = Path(train_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"train directory does not exist: {directory}")
    if isinstance(per_material, bool) or not isinstance(per_material, int) or per_material < 1:
        raise ValueError("calibration_per_material must be a positive integer")
    if seed != 0:
        raise ValueError("B0.4 requires seed=0")
    grouped = {mat_type: [] for mat_type in MATERIAL_NAMES}
    paths = sorted(directory.glob("*.h5"), key=lambda path: path.name)
    if not paths:
        raise ValueError("train directory contains no *.h5 files")
    for path in paths:
        mat_type = _read_calibration_header(path)
        grouped[mat_type].append(path)

    rng = np.random.default_rng(seed)
    selected: list[tuple[int, Path]] = []
    for mat_type in MATERIAL_NAMES:
        candidates = grouped[mat_type]
        if not candidates:
            raise ValueError(f"calibration has no {MATERIAL_NAMES[mat_type]} models")
        count = min(per_material, len(candidates))
        indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
        selected.extend((mat_type, candidates[int(index)]) for index in indices)
    return selected


def _read_calibration_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["x"][:25], dtype=np.float64)
        true_f = np.asarray(handle["F"][:25], dtype=np.float64)
        if true_f.ndim == 3 and true_f.shape[-1] == 9:
            true_f = true_f.reshape(*true_f.shape[:2], 3, 3)
        e_value = float(np.asarray(handle["E"][()]).reshape(()))
        nu = float(np.asarray(handle["nu"][()]).reshape(()))
        mat_type = int(float(np.asarray(handle["mat_type"][()]).reshape(())))
    if x.ndim != 3 or x.shape[0] != 25 or x.shape[-1] != 3:
        raise ValueError(f"{path.name}: calibration x must have shape (25, N, 3)")
    if true_f.shape != (*x.shape[:2], 3, 3):
        raise ValueError(f"{path.name}: calibration F must align with x")
    if not np.isfinite(x).all() or not np.isfinite(true_f).all():
        raise ValueError(f"{path.name}: calibration x/F must be finite")
    if not np.isfinite(e_value) or e_value <= 0.0 or not np.isfinite(nu):
        raise ValueError(f"{path.name}: calibration E/nu must be finite and E positive")
    return x, true_f, float(np.log10(e_value)), nu, mat_type


def run_calibration(
    train_dir: str | Path,
    *,
    per_material: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_calibration_paths(train_dir, per_material=per_material, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, (expected_mat_type, path) in enumerate(selected, start=1):
        x, true_f, log10_e, nu, mat_type = _read_calibration_trajectory(path)
        if mat_type != expected_mat_type:
            raise ValueError(f"{path.name}: mat_type changed after calibration selection")
        for k in ALL_K:
            neighborhood = build_rest_neighborhood(
                x[0],
                k=k,
                condition_threshold=CONDITION_THRESHOLD,
                regularization_scale=REGULARIZATION_SCALE,
            )
            estimated = estimate_local_deformation(x, neighborhood)
            rows.append(
                build_calibration_row(
                    model=path.name,
                    mat_type=mat_type,
                    log10_e=log10_e,
                    nu=nu,
                    k=k,
                    estimated=estimated,
                    true_f=true_f,
                )
            )
        print(f"calibration [{index}/{len(selected)}] {path.name}")
    return rows, evaluate_calibration_gate(rows)


def _validate_test_models(evaluated: set[str], records_by_model: dict[str, Any]) -> None:
    expected = set(records_by_model)
    if evaluated != expected:
        raise ValueError(
            "B0.4 evaluated-model mismatch: "
            f"missing={sorted(expected - evaluated)}; unexpected={sorted(evaluated - expected)}"
        )


def run_test_fidelity(
    args: Any,
    *,
    checkpoint: Path,
    config_path: Path,
    seed: int,
    runtime: RuntimeComponents | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one k=16 factual local-deformation comparison per frozen test model."""
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
    args.seed = seed

    runtime = runtime or _load_runtime_components()
    dataset = runtime.dataset_cls("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    _validate_b0_identity(args, records=records, profile=B03_PROFILE)
    args.model_config.cond_frames = input_frames
    records_by_model = {record.model: record for record in records}
    if len(records_by_model) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"expected {EXPECTED_MODEL_COUNT} unique test records")
    counts = Counter(record.mat_type for record in records_by_model.values())
    if dict(sorted(counts.items())) != EXPECTED_MATERIAL_COUNTS:
        raise ValueError("B0.4 material counts must be elastic=13, plasticine=14, sand=14")

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
        "sample_scope": B04_SAMPLE_SCOPE,
    }
    model_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    evaluated: set[str] = set()

    for batch, _ in dataloader:
        if not _is_start_zero_batch(batch):
            continue
        model_name = _batch_model_name(batch)
        if model_name in evaluated:
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
        gt = _reference_trajectory(_build_raw_reference(batch, args.train_dataset), record.model)
        if pred.shape != gt.shape:
            raise ValueError(f"{record.model}: prediction/GT shape mismatch")
        _validate_condition_frame_alignment(
            pred, gt, input_frames=input_frames, model=record.model
        )
        gt_array = gt.numpy().astype(np.float64, copy=False)
        pred_array = pred.numpy().astype(np.float64, copy=False)
        neighborhood = build_rest_neighborhood(
            gt_array[0],
            k=PRIMARY_K,
            condition_threshold=CONDITION_THRESHOLD,
            regularization_scale=REGULARIZATION_SCALE,
        )
        gt_local = estimate_local_deformation(gt_array, neighborhood)
        pred_local = estimate_local_deformation(pred_array, neighborhood)
        base_row = {
            "model": record.model,
            "mat_type": int(record.mat_type),
            "material": MATERIAL_NAMES[int(record.mat_type)],
            "log10_e": float(record.log10_e),
            "nu": float(record.nu),
            **provenance,
        }
        model_row, model_frames, model_responses = build_local_test_rows(
            base_row,
            gt=gt_local,
            pred=pred_local,
            input_frames=input_frames,
        )
        model_rows.append(model_row)
        frame_rows.extend(model_frames)
        response_rows.extend(model_responses)
        evaluated.add(model_name)
        print(f"test [{len(evaluated)}/{EXPECTED_MODEL_COUNT}] {model_name}")
    _validate_test_models(evaluated, records_by_model)
    return model_rows, frame_rows, response_rows


def run_local_deformation_fidelity(
    args: Any,
    *,
    checkpoint: Path,
    config_path: Path,
    train_dir: Path,
    output_dir: Path,
    seed: int,
    bootstrap_samples: int,
    calibration_per_material: int,
    overwrite: bool,
    runtime: RuntimeComponents | None = None,
) -> dict[str, Path]:
    if seed != 0:
        raise ValueError("B0.4 requires seed=0")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if calibration_per_material < 1:
        raise ValueError("calibration_per_material must be positive")
    checkpoint = Path(checkpoint)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    preflight_local_deformation_outputs(output_dir, overwrite=overwrite)
    _validate_b0_identity(args, profile=B03_PROFILE)
    _validate_cli_checkpoint_matches_config(args, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    dataset_root = Path(args.train_dataset.dataset_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
    if not Path(train_dir).is_dir():
        raise FileNotFoundError(f"train directory does not exist: {train_dir}")

    calibration_rows, calibration_gate = run_calibration(
        train_dir,
        per_material=calibration_per_material,
        seed=seed,
    )
    print(f"calibration gate: {calibration_gate['status']}")
    model_rows, frame_rows, response_rows = run_test_fidelity(
        args,
        checkpoint=checkpoint,
        config_path=config_path,
        seed=seed,
        runtime=runtime,
    )
    print("statistics: prediction-GT fidelity and material response alignment")
    fidelity_rows = build_local_fidelity_rows(
        response_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    metadata = {
        "schema_version": B04_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": seed,
        "split": B04_SAMPLE_SCOPE,
        "model_counts": {"elastic": 13, "plasticine": 14, "sand": 14},
        "train_dir": str(Path(train_dir)),
        "calibration_per_material": calibration_per_material,
        "calibration_status": calibration_gate["status"],
        "calibration_gate": calibration_gate,
        "k_primary": PRIMARY_K,
        "k_sensitivity": list(SENSITIVITY_K),
        "condition_threshold": CONDITION_THRESHOLD,
        "regularization_scale": REGULARIZATION_SCALE,
        "bootstrap_samples": bootstrap_samples,
    }
    paths = write_local_deformation_outputs(
        output_dir=output_dir,
        calibration_rows=calibration_rows,
        model_rows=model_rows,
        frame_rows=frame_rows,
        response_rows=response_rows,
        fidelity_rows=fidelity_rows,
        metadata=metadata,
        overwrite=overwrite,
    )
    print(
        "B0.4 complete: "
        f"calibration={calibration_gate['status']}; test_models={len(model_rows)}"
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B0.4 local deformation response fidelity audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/material_local_deformation_b04"),
    )
    parser.add_argument("--seed", type=_frozen_seed, default=0)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=10000)
    parser.add_argument("--calibration-per-material", type=_positive_int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    config_path = Path(cli.config)
    args = OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config_path))
    run_local_deformation_fidelity(
        args,
        checkpoint=Path(cli.checkpoint),
        config_path=config_path,
        train_dir=cli.train_dir,
        output_dir=cli.output_dir,
        seed=cli.seed,
        bootstrap_samples=cli.bootstrap_samples,
        calibration_per_material=cli.calibration_per_material,
        overwrite=cli.overwrite,
    )


if __name__ == "__main__":
    main()
