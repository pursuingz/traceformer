"""Run B0 material-condition counterfactual diagnostics without importing eval.py."""

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:  # Supports both ``python src/...`` and ``import src....``.
    from utils.material_condition_diagnostics import (
        MaterialRecord,
        build_parameter_derangement,
        condition_response_metrics,
        rotate_material_type,
        summarize_rows,
        trajectory_metrics,
    )
except ModuleNotFoundError:
    from src.utils.material_condition_diagnostics import (
        MaterialRecord,
        build_parameter_derangement,
        condition_response_metrics,
        rotate_material_type,
        summarize_rows,
        trajectory_metrics,
    )


ROLLOUT_HORIZON = 20
EXPECTED_MODEL_COUNT = 41
EXPECTED_RESUME_SUFFIX = "outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors"
EXPECTED_DATASET_SUFFIX = "mm3_data/mm3_test"
EXPECTED_MATERIAL_COUNTS = {0: 13, 1: 14, 2: 14}
EXPECTED_MODEL_NAMES = frozenset(
    {
        "39478_000.h5",
        "45767_002.h5",
        "39999_000.h5",
        "43624_001.h5",
        "47781_002.h5",
        "42107_001.h5",
        "45583_002.h5",
        "39973_000.h5",
        "39511_000.h5",
        "43739_001.h5",
        "39726_000.h5",
        "39659_000.h5",
        "39959_000.h5",
        "40481_001.h5",
        "41341_001.h5",
        "48721_002.h5",
        "44431_001.h5",
        "47458_002.h5",
        "49860_002.h5",
        "47570_002.h5",
        "42798_001.h5",
        "39746_000.h5",
        "46809_002.h5",
        "39586_000.h5",
        "49196_002.h5",
        "39764_000.h5",
        "39653_000.h5",
        "47956_002.h5",
        "39714_000.h5",
        "45752_002.h5",
        "43014_001.h5",
        "43941_001.h5",
        "45790_002.h5",
        "43393_001.h5",
        "44732_001.h5",
        "46081_002.h5",
        "46213_002.h5",
        "42473_001.h5",
        "39560_000.h5",
        "44774_001.h5",
        "44503_001.h5",
    }
)
EXPECTED_TOP_LEVEL_CONFIG = {
    "pc_size": 2048,
    "eval_batch_size": 1,
    "seed": 0,
    "pred_offset": True,
    "model_type": "dit_st",
    "use_diffusion": False,
    "num_inference_steps": 1,
    "input_frames": 5,
    "output_frames": 1,
    "floor_projection": False,
}
EXPECTED_MODEL_CONFIG = {
    "n_layers": 8,
    "latent_dim": 256,
    "frame_cond": True,
    "point_embed": True,
    "mask_cond": True,
    "pred_offset": True,
    "num_neighbors": -1,
    "floor_cond": True,
    "max_num_forces": 1,
    "force_as_token": False,
    "force_as_latent": False,
    "gravity_emb": True,
    "coeff_cond": False,
    "num_mat": 4,
    "class_token": True,
    "transformer_block": "SpatialTemporalTransformerBlock",
    "contact_particle_cond": True,
    "contact_feature_sigma": 0.04,
}
EXPECTED_MODEL_DEFAULT_VALUES = {
    "contact_injection_mode": "separate",
    "contact_velocity_mode": "vertical",
    "contact_feature_mask": [1, 1, 1],
    "contact_bias_scale": 1.0,
}
EXPECTED_TRAIN_DATASET_CONFIG = {
    "category": "hf-objaverse-v1",
    "dataset_list": "DATASET_ITEM_LIST",
    "has_gravity": True,
    "max_num_forces": 1,
    "norm_fac": 5,
    "stage": "deform",
    "mode": "diff",
    "pc_size": 2048,
    "repeat": 1,
    "seed": 0,
    "n_sample_pro_model": 300,
    "n_frames_interval": 1,
    "n_training_frames": 24,
    "batch_size": 20,
    "overfit": False,
    "input_frames": 5,
    "output_frames": 1,
}
# Runtime/reporting fields are intentionally not identity-bearing:
# dataloader_num_workers, vis_dir, and per_model_csv. contact_eval_margin is also
# excluded because this standalone script neither reads it nor passes it to model forward.
# resume and train_dataset.dataset_path are validated separately by fixed suffixes.
ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    set(EXPECTED_TOP_LEVEL_CONFIG)
    | {
        "model_config",
        "train_dataset",
        "resume",
        "dataloader_num_workers",
        "vis_dir",
        "per_model_csv",
        "contact_eval_margin",
    }
)
ALLOWED_MODEL_CONFIG_FIELDS = frozenset(
    set(EXPECTED_MODEL_CONFIG) | set(EXPECTED_MODEL_DEFAULT_VALUES)
)
ALLOWED_TRAIN_DATASET_FIELDS = frozenset(
    set(EXPECTED_TRAIN_DATASET_CONFIG) | {"dataset_path"}
)
_MISSING = object()


def _scalar_from_h5(handle: h5py.File, field: str, model: str) -> float:
    if field not in handle:
        raise ValueError(f"{model}: missing HDF5 field '{field}'")
    value = np.asarray(handle[field])
    if value.size != 1:
        raise ValueError(f"{model}: HDF5 field '{field}' must be scalar")
    return float(value.reshape(-1)[0])


def load_material_records(
    dataset_root: str | Path, model_names: list[str]
) -> list[MaterialRecord]:
    """Read material metadata with the same ``log10(E)`` convention as TrajDataset."""
    root = Path(dataset_root)
    records: list[MaterialRecord] = []
    seen_models: set[str] = set()
    for model_name in model_names:
        model = Path(model_name).name
        if model in seen_models:
            raise ValueError(f"{model}: duplicate model record")
        seen_models.add(model)
        path = root / model
        if not path.is_file():
            raise ValueError(f"{model}: HDF5 file does not exist at {path}")
        try:
            with h5py.File(path, "r") as handle:
                e_value = _scalar_from_h5(handle, "E", model)
                nu_value = _scalar_from_h5(handle, "nu", model)
                mat_type_value = _scalar_from_h5(handle, "mat_type", model)
        except OSError as error:
            raise ValueError(f"{model}: cannot read HDF5 metadata") from error
        if not math.isfinite(e_value) or e_value <= 0:
            raise ValueError(f"{model}: E must be finite and positive")
        if not math.isfinite(nu_value):
            raise ValueError(f"{model}: nu must be finite")
        if not math.isfinite(mat_type_value) or not mat_type_value.is_integer():
            raise ValueError(f"{model}: mat_type must be an integer")
        records.append(
            MaterialRecord(
                model=model,
                mat_type=int(mat_type_value),
                log10_e=math.log10(e_value),
                nu=nu_value,
            )
        )
    return records


def _constant_like(value: torch.Tensor, replacement: float | int) -> torch.Tensor:
    return torch.full_like(value, replacement)


def _normalized_path(value: str | Path) -> str:
    return os.path.normpath(str(value)).replace("\\", "/")


def _nested_config_value(config: Any, field: str) -> Any:
    value = config
    for part in field.split("."):
        try:
            value = getattr(value, part)
        except (AttributeError, KeyError):
            try:
                value = value[part]
            except (KeyError, TypeError):
                return _MISSING
    return value


def _config_values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        try:
            actual_values = list(actual)
        except TypeError:
            return False
        return len(actual_values) == len(expected) and all(
            not isinstance(actual_value, bool)
            and isinstance(actual_value, (int, float))
            and math.isclose(
                float(actual_value), float(expected_value), rel_tol=0.0, abs_tol=1e-12
            )
            for actual_value, expected_value in zip(actual_values, expected)
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_config_value(
    args: Any, field: str, expected: Any, use_default_when_missing: bool
) -> None:
    actual = _nested_config_value(args, field)
    if actual is _MISSING and use_default_when_missing:
        actual = expected
    if actual is _MISSING or not _config_values_match(actual, expected):
        actual_repr = "<missing>" if actual is _MISSING else repr(actual)
        raise ValueError(
            f"B0 config mismatch for {field}: "
            f"actual={actual_repr}; expected={expected!r}"
        )


def _config_field_names(config: Any) -> set[str]:
    keys = getattr(config, "keys", None)
    if callable(keys):
        return {str(key) for key in keys()}
    try:
        return set(vars(config))
    except TypeError as exc:
        raise ValueError(f"B0 config section is not inspectable: {config!r}") from exc


def _validate_allowed_fields(config: Any, section: str, allowed: frozenset[str]) -> None:
    unexpected = sorted(_config_field_names(config) - allowed)
    if unexpected:
        raise ValueError(f"B0 config has unexpected {section} field(s): {unexpected}")


def _validate_b0_identity(args: Any, records: list[MaterialRecord] | None = None) -> None:
    _validate_allowed_fields(args, "top-level", ALLOWED_TOP_LEVEL_FIELDS)
    _validate_allowed_fields(
        args.model_config, "model_config", ALLOWED_MODEL_CONFIG_FIELDS
    )
    _validate_allowed_fields(
        args.train_dataset, "train_dataset", ALLOWED_TRAIN_DATASET_FIELDS
    )
    for field, expected in EXPECTED_TOP_LEVEL_CONFIG.items():
        _validate_config_value(args, field, expected, use_default_when_missing=False)
    for field, expected in EXPECTED_MODEL_CONFIG.items():
        _validate_config_value(
            args,
            f"model_config.{field}",
            expected,
            use_default_when_missing=False,
        )
    for field, expected in EXPECTED_MODEL_DEFAULT_VALUES.items():
        _validate_config_value(
            args,
            f"model_config.{field}",
            expected,
            use_default_when_missing=True,
        )
    for field, expected in EXPECTED_TRAIN_DATASET_CONFIG.items():
        _validate_config_value(
            args,
            f"train_dataset.{field}",
            expected,
            use_default_when_missing=False,
        )
    resume = _normalized_path(args.resume)
    if not resume.endswith(EXPECTED_RESUME_SUFFIX):
        raise ValueError(
            "B0 checkpoint mismatch: "
            f"actual={resume!r}; expected={EXPECTED_RESUME_SUFFIX!r}"
        )
    dataset_path = _normalized_path(args.train_dataset.dataset_path)
    if not dataset_path.endswith(EXPECTED_DATASET_SUFFIX):
        raise ValueError(
            "B0 dataset mismatch: "
            f"actual={dataset_path!r}; expected={EXPECTED_DATASET_SUFFIX!r}"
        )
    if records is not None:
        actual_counts = dict(sorted(Counter(record.mat_type for record in records).items()))
        if actual_counts != EXPECTED_MATERIAL_COUNTS:
            raise ValueError(
                "B0 material counts mismatch: "
                f"actual={actual_counts}; expected={EXPECTED_MATERIAL_COUNTS}"
            )
        if len(records) != EXPECTED_MODEL_COUNT:
            raise ValueError(
                f"expected {EXPECTED_MODEL_COUNT} material records, got {len(records)}"
            )
        name_counts = Counter(record.model for record in records)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"duplicate material metadata model(s): {duplicates}"
            )
        actual_names = set(name_counts)
        if actual_names != EXPECTED_MODEL_NAMES:
            missing = sorted(EXPECTED_MODEL_NAMES - actual_names)
            unexpected = sorted(actual_names - EXPECTED_MODEL_NAMES)
            raise ValueError(
                "B0 model set mismatch: "
                f"missing={missing}; unexpected={unexpected}"
            )


def _batch_material_scalar(
    batch: dict[str, Any], field: str, model: str
) -> float:
    if field not in batch:
        raise ValueError(f"{model}: batch is missing material field '{field}'")
    value = batch[field]
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{model}: batch material field '{field}' must be scalar")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"{model}: batch material field '{field}' must be finite")
    return scalar


def _validate_normal_material_condition(
    batch: dict[str, Any], record: MaterialRecord
) -> None:
    expected = {
        "E": record.log10_e,
        "nu": record.nu,
        "mat_type": record.mat_type,
    }
    for field, expected_value in expected.items():
        actual = _batch_material_scalar(batch, field, record.model)
        if field == "mat_type":
            matches = actual.is_integer() and int(actual) == expected_value
        else:
            matches = bool(
                np.isclose(actual, expected_value, rtol=1e-5, atol=1e-6)
            )
        if not matches:
            raise ValueError(
                f"{record.model}: batch/HDF5 {field} mismatch: "
                f"batch={actual!r}; HDF5={expected_value!r}"
            )


def _load_checkpoint_strict(model: Any, resume: str | Path, loader: Any) -> None:
    checkpoint = loader(str(resume), device="cpu")
    model.load_state_dict(checkpoint, strict=True)


@torch.no_grad()
def rollout_condition(
    pipeline: Any,
    batch: dict[str, Any],
    args: Any,
    e_value: float,
    nu_value: float,
    mat_type: int,
) -> torch.Tensor:
    """Generate a 20-frame autoregressive rollout under one material condition."""
    input_frames = int(args.input_frames)
    output_frames = int(args.output_frames)
    if input_frames <= 0 or output_frames <= 0:
        raise ValueError("input_frames and output_frames must be positive")

    device = getattr(args, "device", "cuda")
    current_input = batch["points_src"].to(device)
    if current_input.shape[1] != input_frames:
        raise ValueError(
            f"points_src has {current_input.shape[1]} frames; expected {input_frames}"
        )
    rollout_chunks = [current_input]
    previous_input = current_input
    steps = math.ceil(ROLLOUT_HORIZON / output_frames)
    e_condition = _constant_like(batch["E"], e_value)
    nu_condition = _constant_like(batch["nu"], nu_value)
    if "mat_type" in batch:
        y_condition = _constant_like(batch["mat_type"], mat_type)
    else:
        y_condition = torch.full(
            (current_input.shape[0],),
            mat_type,
            dtype=torch.long,
            device=current_input.device,
        )

    for step_idx in range(steps):
        if step_idx == 0:
            step_start_vel = batch.get("start_vel")
            if step_start_vel is not None:
                step_start_vel = step_start_vel.to(device)
        elif input_frames == 1:
            step_start_vel = current_input[:, -1] - previous_input[:, -1]
        elif output_frames >= 2:
            step_start_vel = current_input[:, 1] - previous_input[:, -1]
        else:
            step_start_vel = current_input[:, 1] - current_input[:, 0]

        pred_chunk = pipeline(
            current_input,
            batch["force"],
            e_condition,
            nu_condition,
            batch["mask"][..., :1],
            batch["drag_point"],
            batch["floor_height"],
            batch["gravity"],
            batch["base_drag_coeff"],
            start_vel=step_start_vel,
            points_rest=batch.get("points_rest"),
            y=y_condition,
            device=device,
            batch_size=args.eval_batch_size,
            generator=torch.Generator().manual_seed(args.seed),
            n_frames=output_frames,
            num_inference_steps=args.num_inference_steps,
        )
        if getattr(args, "floor_projection", False):
            floor_height = batch["floor_height"].to(device).view(-1, 1, 1)
            pred_chunk = pred_chunk.clone()
            pred_chunk[..., 1] = torch.maximum(pred_chunk[..., 1], floor_height)
        rollout_chunks.append(pred_chunk)
        previous_input = current_input
        current_input = torch.cat([current_input, pred_chunk], dim=1)[:, -input_frames:]

    return torch.cat(rollout_chunks, dim=1)[:, : input_frames + ROLLOUT_HORIZON]


def rollout_counterfactuals(
    pipeline: Any,
    batch: dict[str, Any],
    args: Any,
    record: MaterialRecord,
    shuffled_parameters: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all B0 conditions under eval.py's CUDA bfloat16 inference context."""
    _validate_normal_material_condition(batch, record)
    shuffled_e, shuffled_nu = shuffled_parameters
    with torch.autocast("cuda", dtype=torch.bfloat16):
        normal = rollout_condition(
            pipeline, batch, args, record.log10_e, record.nu, record.mat_type
        )
        shuffled_params = rollout_condition(
            pipeline, batch, args, shuffled_e, shuffled_nu, record.mat_type
        )
        shuffled_class = rollout_condition(
            pipeline,
            batch,
            args,
            record.log10_e,
            record.nu,
            rotate_material_type(record.mat_type),
        )
    return normal, shuffled_params, shuffled_class


def _build_raw_reference(batch: dict[str, Any], dataset_cfg: Any) -> torch.Tensor:
    interval = int(dataset_cfg.get("n_frames_interval", 1))
    sequences = []
    for index, model_name in enumerate(batch["model"]):
        path = Path(dataset_cfg.dataset_path) / Path(model_name).name
        with h5py.File(path, "r") as handle:
            raw_points = torch.from_numpy(np.asarray(handle["x"]))
        point_indices = batch["point_indices"][index].cpu().numpy()
        selected_frames = np.arange(25) * interval
        if selected_frames[-1] >= raw_points.shape[0]:
            raise ValueError(f"{model_name}: cannot provide 25 ground-truth frames")
        sequence = raw_points[selected_frames][:, point_indices].float()
        sequences.append((sequence - dataset_cfg.norm_fac) / 2)
    return torch.stack(sequences, dim=0)


def _metric_row(prefix: str, pred: torch.Tensor, gt: torch.Tensor, input_frames: int) -> dict[str, float]:
    return {
        f"{prefix}_{name}": value
        for name, value in trajectory_metrics(pred, gt, input_frames).items()
    }


def _response_row(prefix: str, normal: torch.Tensor, counterfactual: torch.Tensor, input_frames: int) -> dict[str, float]:
    return {
        f"{prefix}_{name}": value
        for name, value in condition_response_metrics(
            normal, counterfactual, input_frames
        ).items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty diagnostics CSV")
    ordered_rows = sorted(rows, key=lambda row: str(row["model"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_rows[0]))
        writer.writeheader()
        writer.writerows(ordered_rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any],],
    samples: int,
    seed: int,
) -> None:
    lines = ["# B0 Material-Condition Diagnostics", ""]
    model_names = sorted(str(row["model"]) for row in rows)
    if len(model_names) != len(set(model_names)):
        raise ValueError("diagnostics report contains duplicate model names")
    lines.extend(
        [
            "## Evaluated Models",
            "",
            f"Count: {len(model_names)}",
            "",
            *(f"- `{model_name}`" for model_name in model_names),
            "",
        ]
    )
    for intervention in ("shuffle_params", "shuffle_class"):
        summary = summarize_rows(rows, intervention, samples=samples, seed=seed)
        lines.extend([f"## {intervention}", ""])
        if intervention == "shuffle_class":
            lines.extend(
                [
                    "This intervention only measures class-condition dependence; "
                    "it does not represent physical accuracy.",
                    "",
                ]
            )
        lines.append(
            "| Group | Metric | Baseline | Counterfactual | Relative change | "
            "Paired delta 95% CI | Response ratio | Label |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for group, metrics in summary.items():
            for metric, stats in metrics.items():
                lines.append(
                    "| {group} | {metric} | {normal_mean:.6e} | {counterfactual_mean:.6e} | "
                    "{relative_change_pct:+.2f}% | [{ci_low:.6e}, {ci_high:.6e}] | "
                    "{response_ratio_pct:.2f}% | {label} |".format(
                        group=group,
                        metric=metric,
                        **stats,
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _output_paths(output_dir: Path, permutation_seed: int) -> tuple[Path, Path]:
    stem = f"material_condition_b0_seed{permutation_seed}"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.md"


def _print_model_progress(completed: int, total: int, model_name: str) -> None:
    print(f"[{completed}/{total}] {model_name} complete")


def _print_completion(
    output_dir: Path,
    permutation_seed: int,
    rows: list[dict[str, Any]],
    samples: int,
) -> None:
    csv_path, markdown_path = _output_paths(output_dir, permutation_seed)
    print(f"CSV: {csv_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    print("Overall summary:")
    for intervention in ("shuffle_params", "shuffle_class"):
        overall = summarize_rows(
            rows, intervention, samples=samples, seed=permutation_seed
        )["overall"]
        print(f"  {intervention}:")
        for metric, stats in overall.items():
            print(
                f"    {metric}: {stats['normal_mean']:.6e} -> "
                f"{stats['counterfactual_mean']:.6e} "
                f"({stats['relative_change_pct']:+.2f}%, {stats['label']})"
            )


def run_diagnostics(args: Any, output_dir: Path, permutation_seed: int, bootstrap_samples: int) -> list[dict[str, Any]]:
    """Load the model once, evaluate each test model's start-0 window once, and write rows."""
    from diffusers import DDIMScheduler
    from safetensors.torch import load_file

    try:
        from dataset.traj_dataset import TrajDataset
        from model.spacetime import MDM_ST
        from pipeline_traj import TrajPipeline
    except ModuleNotFoundError:
        from src.dataset.traj_dataset import TrajDataset
        from src.model.spacetime import MDM_ST
        from src.pipeline_traj import TrajPipeline

    _validate_b0_identity(args)
    input_frames = int(args.input_frames)
    output_frames = int(args.output_frames)
    args.train_dataset.input_frames = input_frames
    args.train_dataset.output_frames = output_frames
    resume = Path(args.resume)
    dataset_root = Path(args.train_dataset.dataset_path)
    if not resume.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resume}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    dataset = TrajDataset("test", args.train_dataset)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    _validate_b0_identity(args, records)
    args.model_config.cond_frames = input_frames
    if len(records) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"expected {EXPECTED_MODEL_COUNT} material records, got {len(records)}")
    records_by_model = {record.model: record for record in records}
    parameter_derangement = build_parameter_derangement(records, permutation_seed)

    device = "cuda"
    model = MDM_ST(args.pc_size, output_frames, n_feats=3, model_config=args.model_config).to(device)
    _load_checkpoint_strict(model, resume, load_file)
    model.eval().requires_grad_(False)
    model = torch.compile(model)
    scheduler = (
        DDIMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
        if args.use_diffusion
        else None
    )
    pipeline = TrajPipeline(model=model, scheduler=scheduler)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    rows: list[dict[str, Any]] = []
    evaluated_models: set[str] = set()
    for batch, _ in dataloader:
        start_indices = batch["start_idx"]
        if not torch.all(start_indices == 0):
            if torch.any(start_indices == 0):
                raise ValueError("start-0 and nonzero windows must not share a batch")
            continue
        names = [Path(name).name for name in batch["model"]]
        if len(names) != 1:
            raise ValueError("B0 diagnostics requires eval_batch_size=1 for per-model rows")
        model_name = names[0]
        if model_name in evaluated_models:
            raise ValueError(f"{model_name}: evaluated more than once")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")
        gt = _build_raw_reference(batch, args.train_dataset)
        shuffled_e, shuffled_nu = parameter_derangement[model_name]
        shuffled_class_type = rotate_material_type(record.mat_type)
        normal, shuffled_params, shuffled_class = rollout_counterfactuals(
            pipeline,
            batch,
            args,
            record,
            (shuffled_e, shuffled_nu),
        )
        row: dict[str, Any] = {
            "model": model_name,
            "mat_type": record.mat_type,
            "true_log10_e": record.log10_e,
            "true_nu": record.nu,
            "shuffled_log10_e": shuffled_e,
            "shuffled_nu": shuffled_nu,
            "shuffled_mat_type": shuffled_class_type,
        }
        row.update(_metric_row("normal", normal[0], gt[0], input_frames))
        row.update(_metric_row("shuffle_params", shuffled_params[0], gt[0], input_frames))
        row.update(_metric_row("shuffle_class", shuffled_class[0], gt[0], input_frames))
        row.update(_response_row("shuffle_params", normal[0], shuffled_params[0], input_frames))
        row.update(_response_row("shuffle_class", normal[0], shuffled_class[0], input_frames))
        rows.append(row)
        evaluated_models.add(model_name)
        _print_model_progress(len(evaluated_models), EXPECTED_MODEL_COUNT, model_name)

    if len(evaluated_models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODEL_COUNT} start-0 evaluations, got {len(evaluated_models)}"
        )
    if evaluated_models != set(records_by_model):
        missing = sorted(set(records_by_model) - evaluated_models)
        unexpected = sorted(evaluated_models - set(records_by_model))
        raise ValueError(f"evaluated models do not match metadata; missing={missing}, unexpected={unexpected}")

    rows.sort(key=lambda row: str(row["model"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, markdown_path = _output_paths(output_dir, permutation_seed)
    _write_csv(csv_path, rows)
    _write_markdown(
        markdown_path,
        rows,
        samples=bootstrap_samples,
        seed=permutation_seed,
    )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B0 material-condition counterfactual diagnostics.")
    parser.add_argument("--config", required=True, help="Evaluation YAML matching the checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="results/material_condition_b0",
        help="Directory for the per-model CSV and summary Markdown.",
    )
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    cli_args = _parse_args()
    if cli_args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    from omegaconf import OmegaConf

    try:
        from options import TestingConfig
    except ModuleNotFoundError:
        from src.options import TestingConfig

    args = OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(cli_args.config))
    rows = run_diagnostics(
        args,
        output_dir=Path(cli_args.output_dir),
        permutation_seed=cli_args.permutation_seed,
        bootstrap_samples=cli_args.bootstrap_samples,
    )
    _print_completion(
        Path(cli_args.output_dir),
        permutation_seed=cli_args.permutation_seed,
        rows=rows,
        samples=cli_args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
