"""Diagnose per-material gradient conflict in the frozen B3a adapter."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


_GRADIENT_STARTS = (0, 5, 10, 15)
_GRADIENT_MATERIAL_COUNTS = {"elastic": 13, "plasticine": 14, "sand": 14}
_GRADIENT_SAMPLE_COUNTS = {"elastic": 52, "plasticine": 56, "sand": 56}
_GRADIENT_GROUPS = (
    "all_adapter",
    "state_norm",
    "state_proj",
    "material_proj",
    "output_proj",
    "stage_scales",
)
_GRADIENT_PAIRS = (
    "elastic__plasticine",
    "elastic__sand",
    "plasticine__sand",
)
_GRADIENT_PROTOCOL = "164-window teacher-forced one-step coordinate MSE"
_GRADIENT_SAMPLE_SCOPE = "frozen 41-model x 4 fixed windows B3a90"


def _validated_manifest_models(manifest: dict[str, tuple[str, ...]]) -> list[str]:
    if not isinstance(manifest, dict) or tuple(manifest) != tuple(
        _GRADIENT_MATERIAL_COUNTS
    ):
        raise ValueError("gradient manifest must contain elastic, plasticine, and sand")
    for material, expected_count in _GRADIENT_MATERIAL_COUNTS.items():
        names = manifest[material]
        if not isinstance(names, (tuple, list)) or len(names) != expected_count:
            raise ValueError("gradient manifest must use material counts 13/14/14")
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError(f"gradient manifest contains invalid {material} model names")
    names = [name for material in manifest.values() for name in material]
    if len(names) != 41 or len(names) != len(set(names)):
        raise ValueError("gradient manifest must contain exactly 41 unique models")
    return names


def _sample_point_count(dataset: Any, index: int) -> int:
    sample = dataset[index]
    if isinstance(sample, (tuple, list)):
        if not sample:
            raise ValueError("gradient dataset item is empty")
        sample = sample[0]
    if not isinstance(sample, dict) or "points_src" not in sample:
        raise ValueError("gradient dataset item must contain points_src")
    points_src = sample["points_src"]
    if not isinstance(points_src, torch.Tensor) or points_src.ndim < 3:
        raise ValueError("gradient points_src must be a frame-point tensor")
    point_count = int(points_src.shape[-2])
    if point_count <= 0:
        raise ValueError("gradient point count must be positive")
    return point_count


def validate_fixed_gradient_windows(
    dataset: Any, manifest: dict[str, tuple[str, ...]]
) -> list[int]:
    """Validate and order the frozen 41-model by four-start gradient protocol."""
    expected_models = _validated_manifest_models(manifest)
    try:
        entries = dataset.models
        dataset_models = [Path(name).name for name in dataset.split_lst_save]
    except AttributeError as exc:
        raise ValueError("gradient dataset must expose models and split_lst_save") from exc
    if len(dataset_models) != len(set(dataset_models)) or set(dataset_models) != set(
        expected_models
    ):
        raise ValueError("frozen model set mismatch between dataset and manifest")

    selected = {}
    expected_model_set = set(expected_models)
    allowed_starts = set(_GRADIENT_STARTS)
    for index, entry in enumerate(entries):
        try:
            model = Path(entry["model"]).name
            start_idx = entry["start_idx"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid gradient dataset entry at index {index}") from exc
        if (
            isinstance(start_idx, bool)
            or not isinstance(start_idx, (int, torch.Tensor))
        ):
            raise ValueError(f"invalid start_idx at gradient dataset index {index}")
        if isinstance(start_idx, torch.Tensor):
            if start_idx.numel() != 1:
                raise ValueError(f"invalid start_idx at gradient dataset index {index}")
            start_idx = int(start_idx.item())
        else:
            start_idx = int(start_idx)
        if model not in expected_model_set:
            raise ValueError(f"unexpected gradient model: {model}")
        if start_idx not in allowed_starts:
            raise ValueError(f"{model}: gradient starts must be {_GRADIENT_STARTS}")
        key = (model, start_idx)
        if key in selected:
            raise ValueError(f"{model}: duplicate gradient start {start_idx}")
        selected[key] = index

    expected_keys = {
        (model, start_idx)
        for model in expected_models
        for start_idx in _GRADIENT_STARTS
    }
    if set(selected) != expected_keys:
        raise ValueError("gradient windows must contain exactly starts (0, 5, 10, 15)")

    ordered = [
        selected[(model, start_idx)]
        for model in expected_models
        for start_idx in _GRADIENT_STARTS
    ]
    point_counts = {
        model: _sample_point_count(dataset, selected[(model, _GRADIENT_STARTS[0])])
        for model in expected_models
    }
    if len(set(point_counts.values())) != 1:
        raise ValueError(f"gradient windows have mixed point count: {point_counts}")
    if len(ordered) != 164:
        raise ValueError("gradient protocol must contain exactly 164 windows")
    return ordered


def _tensor_to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def teacher_forced_one_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the deterministic one-frame training contract without updating weights."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("teacher-forced seed must be a non-negative integer")
    if not isinstance(batch, dict):
        raise ValueError("teacher-forced batch must be a dictionary")
    required = (
        "points_src",
        "points_tgt",
        "force",
        "E",
        "nu",
        "mask",
        "drag_point",
    )
    missing = [field for field in required if field not in batch]
    if missing:
        raise ValueError(f"teacher-forced batch is missing fields: {missing}")

    device = torch.device(device)
    points_src = _tensor_to_device(batch["points_src"], device)
    target = _tensor_to_device(batch["points_tgt"], device)
    if (
        not isinstance(points_src, torch.Tensor)
        or points_src.ndim != 4
        or points_src.shape[1] != 5
        or points_src.shape[-1] != 3
    ):
        raise ValueError("points_src must have shape (B, 5, N, 3)")
    if (
        not isinstance(target, torch.Tensor)
        or target.shape != (points_src.shape[0], 1, points_src.shape[2], 3)
    ):
        raise ValueError("points_tgt must have shape (B, 1, N, 3)")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_input = points_src[:, -1:].clone()
    model_input = model_input + torch.randn_like(model_input) * 0.02
    timesteps = torch.zeros(
        (points_src.shape[0],), device=device, dtype=torch.long
    )
    mask = _tensor_to_device(batch["mask"], device).to(dtype=model_input.dtype)
    pred = model(
        model_input,
        timesteps,
        points_src,
        _tensor_to_device(batch["force"], device),
        _tensor_to_device(batch["E"], device),
        _tensor_to_device(batch["nu"], device),
        mask[..., :1],
        _tensor_to_device(batch["drag_point"], device),
        floor_height=_tensor_to_device(batch.get("floor_height"), device),
        gravity_label=_tensor_to_device(batch.get("gravity"), device),
        coeff=_tensor_to_device(batch.get("base_drag_coeff"), device),
        y=_tensor_to_device(batch.get("mat_type"), device),
        null_emb=None,
        start_vel=_tensor_to_device(batch.get("start_vel"), device),
        points_rest=_tensor_to_device(batch.get("points_rest"), device),
    )
    if pred.shape != target.shape or not torch.isfinite(pred).all():
        raise ValueError("teacher-forced prediction must be finite and match points_tgt")
    return pred, target


def _finite_float(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{field} must be finite and non-negative")
    return value


def _validate_gradient_conflict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("gradient conflict payload must be a dictionary")
    validated = dict(payload)
    for field in ("checkpoint", "config", "sample_scope", "protocol"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"gradient conflict payload requires {field}")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("gradient conflict seed must be a non-negative integer")
    if payload.get("protocol") != _GRADIENT_PROTOCOL:
        raise ValueError(f"gradient protocol must be {_GRADIENT_PROTOCOL!r}")

    sample_counts = payload.get("sample_counts")
    if sample_counts != _GRADIENT_SAMPLE_COUNTS:
        raise ValueError("gradient sample counts must be 52/56/56")

    loss_means = payload.get("loss_means")
    expected_loss_groups = {"overall", *_GRADIENT_SAMPLE_COUNTS}
    if not isinstance(loss_means, dict) or set(loss_means) != expected_loss_groups:
        raise ValueError("gradient loss_means must cover overall and three materials")
    for group, value in loss_means.items():
        _finite_float(value, f"loss_means.{group}", nonnegative=True)

    groups = payload.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(_GRADIENT_GROUPS):
        raise ValueError("gradient groups are incomplete")
    for group in _GRADIENT_GROUPS:
        group_payload = groups[group]
        if not isinstance(group_payload, dict):
            raise ValueError(f"gradient group must be a dictionary: {group}")
        norms = group_payload.get("gradient_norms")
        if not isinstance(norms, dict) or set(norms) != set(_GRADIENT_SAMPLE_COUNTS):
            raise ValueError(f"gradient norms are incomplete: {group}")
        for material, value in norms.items():
            _finite_float(
                value, f"groups.{group}.gradient_norms.{material}", nonnegative=True
            )
        cosines = group_payload.get("pairwise_cosine")
        if not isinstance(cosines, dict) or set(cosines) != set(_GRADIENT_PAIRS):
            raise ValueError(f"pairwise cosine table is incomplete: {group}")
        for pair, value in cosines.items():
            if value is None:
                continue
            value = _finite_float(value, f"groups.{group}.pairwise_cosine.{pair}")
            if not -1.000001 <= value <= 1.000001:
                raise ValueError(f"pairwise cosine is outside [-1, 1]: {group}.{pair}")

    stage_gradients = payload.get("stage_scale_gradients")
    if not isinstance(stage_gradients, dict) or set(stage_gradients) != set(
        _GRADIENT_SAMPLE_COUNTS
    ):
        raise ValueError("stage-scale gradients must cover three materials")
    for material, values in stage_gradients.items():
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"{material} stage-scale gradients must contain four values")
        for index, value in enumerate(values):
            _finite_float(value, f"stage_scale_gradients.{material}.{index}")
    return validated


def _gradient_conflict_report(payload: dict[str, Any]) -> str:
    lines = [
        "# B3a 分材质梯度冲突诊断",
        "",
        "## 实验元数据",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- config: `{payload['config']}`",
        f"- seed: `{payload['seed']}`",
        f"- sample_scope: {payload['sample_scope']}",
        f"- protocol: `{payload['protocol']}`",
        "- 目标函数：164-window teacher-forced 单步坐标 MSE。",
        "",
        "## 样本与损失",
        "",
        "| group | samples | mean coordinate MSE |",
        "| --- | ---: | ---: |",
        f"| overall | 164 | {payload['loss_means']['overall']:.6e} |",
    ]
    for material in _GRADIENT_SAMPLE_COUNTS:
        lines.append(
            f"| {material} | {payload['sample_counts'][material]} | "
            f"{payload['loss_means'][material]:.6e} |"
        )
    lines.extend(
        [
            "",
            "## 梯度范数与两两 cosine",
            "",
            "| group | elastic norm | plasticine norm | sand norm | e-p | e-s | p-s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in _GRADIENT_GROUPS:
        values = payload["groups"][group]
        norms = values["gradient_norms"]
        cosines = values["pairwise_cosine"]

        def display(value):
            return "N/A" if value is None else f"{value:+.6f}"

        lines.append(
            f"| {group} | {norms['elastic']:.6e} | {norms['plasticine']:.6e} | "
            f"{norms['sand']:.6e} | {display(cosines['elastic__plasticine'])} | "
            f"{display(cosines['elastic__sand'])} | "
            f"{display(cosines['plasticine__sand'])} |"
        )
    lines.extend(
        [
            "",
            "## Stage-scale 带符号平均梯度",
            "",
            "| material | stage 0 | stage 1 | stage 2 | stage 3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for material, values in payload["stage_scale_gradients"].items():
        lines.append(
            f"| {material} | " + " | ".join(f"{value:+.6e}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "负 cosine 表示当前冻结 checkpoint、当前样本和单步目标下的局部下降方向冲突；",
            "它不能证明 material experts 会改善 rollout，也不能推翻 B3a 已失败的 accuracy gate。",
            "",
        ]
    )
    return "\n".join(lines)


def write_gradient_conflict_outputs(
    output_prefix: str | Path, payload: dict[str, Any]
) -> dict[str, Path]:
    """Validate the complete diagnostic before writing JSON and Chinese Markdown."""
    validated = _validate_gradient_conflict_payload(payload)
    json_text = json.dumps(validated, indent=2, ensure_ascii=False) + "\n"
    report_text = _gradient_conflict_report(validated)

    prefix = Path(output_prefix)
    if prefix.suffix:
        raise ValueError("gradient output must be a suffix-free path prefix")
    json_path = prefix.with_suffix(".json")
    report_path = prefix.with_suffix(".md")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return {"json": json_path, "report": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen B3a teacher-forced material-gradient diagnostic."
    )
    parser.add_argument("--config", required=True, help="Frozen B3a90 evaluation YAML.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="B3a90 checkpoint-90000 model.safetensors path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Suffix-free prefix for the B3a90 JSON and Markdown reports.",
    )
    return parser


def run_gradient_conflict_diagnostic(
    args: Any,
    checkpoint: str | Path,
    output_prefix: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Accumulate frozen-checkpoint adapter gradients over 164 fixed windows."""
    from safetensors.torch import load_file

    try:
        from diagnose_hybrid_state_feedback import (
            _validate_records,
            load_frozen_test_manifest,
        )
        from diagnose_material_condition import (
            _load_checkpoint_strict,
            _validate_b0_identity,
            _validate_normal_material_condition,
            load_material_records,
        )
        from dataset.traj_dataset import TrajDataset
        from model.spacetime import MDM_ST
        from utils.material_state_stage_diagnostics import (
            mean_named_gradients,
            snapshot_adapter_gradients,
            summarize_material_gradient_conflict,
        )
    except ModuleNotFoundError:
        from src.diagnose_hybrid_state_feedback import (
            _validate_records,
            load_frozen_test_manifest,
        )
        from src.diagnose_material_condition import (
            _load_checkpoint_strict,
            _validate_b0_identity,
            _validate_normal_material_condition,
            load_material_records,
        )
        from src.dataset.traj_dataset import TrajDataset
        from src.model.spacetime import MDM_ST
        from src.utils.material_state_stage_diagnostics import (
            mean_named_gradients,
            snapshot_adapter_gradients,
            summarize_material_gradient_conflict,
        )

    checkpoint = Path(checkpoint)
    args.resume = str(checkpoint)
    _validate_b0_identity(args, profile="b3a90")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    dataset_root = Path(args.train_dataset.dataset_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    args.train_dataset.input_frames = 5
    args.train_dataset.output_frames = 1
    args.model_config.cond_frames = 5
    manifest = load_frozen_test_manifest()
    dataset = TrajDataset("test", args.train_dataset)
    selected_indices = validate_fixed_gradient_windows(dataset, manifest)
    records = load_material_records(dataset_root, dataset.split_lst_save)
    records_by_model = _validate_records(records, dataset.split_lst_save, manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("B3a gradient conflict diagnostic requires CUDA")
    device = torch.device("cuda")
    model = MDM_ST(args.pc_size, 1, n_feats=3, model_config=args.model_config).to(device)
    _load_checkpoint_strict(model, checkpoint, load_file)
    model.eval().requires_grad_(False)
    adapter = model.dit.material_state_exchange
    adapter.requires_grad_(True)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    material_names = {0: "elastic", 1: "plasticine", 2: "sand"}
    gradient_sums = {material: None for material in material_names.values()}
    loss_sums = {material: 0.0 for material in material_names.values()}
    sample_counts = {material: 0 for material in material_names.values()}

    for selected_index, (batch, _) in zip(selected_indices, dataloader):
        names = [Path(name).name for name in batch.get("model", [])]
        if len(names) != 1 or int(batch["points_src"].shape[0]) != 1:
            raise ValueError("gradient diagnostic requires batch size 1")
        model_name = names[0]
        expected_entry = dataset.models[selected_index]
        expected_name = Path(expected_entry["model"]).name
        expected_start = int(expected_entry["start_idx"])
        actual_start = int(torch.as_tensor(batch["start_idx"]).reshape(-1)[0])
        if model_name != expected_name or actual_start != expected_start:
            raise ValueError("gradient DataLoader order does not match the fixed protocol")
        record = records_by_model.get(model_name)
        if record is None:
            raise ValueError(f"{model_name}: missing material metadata")
        _validate_normal_material_condition(batch, record)
        material = material_names.get(int(record.mat_type))
        if material is None:
            raise ValueError(f"{model_name}: unsupported material type {record.mat_type}")

        adapter.zero_grad(set_to_none=True)
        pred, target = teacher_forced_one_step(
            model,
            batch,
            device,
            seed=int(args.seed) + int(selected_index),
        )
        loss = F.mse_loss(pred.float(), target.float())
        if not torch.isfinite(loss):
            raise ValueError(f"{model_name}: teacher-forced loss is nonfinite")
        loss.backward()
        snapshot = snapshot_adapter_gradients(adapter)

        if gradient_sums[material] is None:
            gradient_sums[material] = {
                name: value.clone() for name, value in snapshot.items()
            }
        else:
            if tuple(gradient_sums[material]) != tuple(snapshot):
                raise ValueError("adapter gradient parameter order changed during diagnostic")
            for name, value in snapshot.items():
                if gradient_sums[material][name].shape != value.shape:
                    raise ValueError(f"adapter gradient shape changed: {name}")
                gradient_sums[material][name].add_(value)
        loss_sums[material] += float(loss.detach())
        sample_counts[material] += 1

    if sample_counts != _GRADIENT_SAMPLE_COUNTS:
        raise ValueError(
            f"gradient diagnostic sample counts mismatch: {sample_counts}"
        )
    adapter_ids = {id(parameter) for parameter in adapter.parameters()}
    if any(
        parameter.grad is not None
        for parameter in model.parameters()
        if id(parameter) not in adapter_ids
    ):
        raise RuntimeError("frozen backbone unexpectedly received gradients")

    material_gradients = {
        material: {
            "sample_count": sample_counts[material],
            "named_gradients": mean_named_gradients(
                gradient_sums[material], sample_counts[material]
            ),
        }
        for material in _GRADIENT_SAMPLE_COUNTS
    }
    summary = summarize_material_gradient_conflict(material_gradients)
    loss_means = {
        material: loss_sums[material] / sample_counts[material]
        for material in _GRADIENT_SAMPLE_COUNTS
    }
    loss_means["overall"] = sum(loss_sums.values()) / sum(sample_counts.values())
    payload = {
        "checkpoint": str(checkpoint),
        "config": str(config_path) if config_path is not None else "<direct-call>",
        "seed": int(args.seed),
        "sample_scope": _GRADIENT_SAMPLE_SCOPE,
        "protocol": _GRADIENT_PROTOCOL,
        "loss_means": loss_means,
        **summary,
    }
    paths = write_gradient_conflict_outputs(output_prefix, payload)
    print(
        "B3a gradient conflict complete: "
        f"windows={sum(sample_counts.values())} counts={sample_counts}"
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return payload


def main() -> None:
    from omegaconf import OmegaConf

    try:
        from options import TestingConfig
    except ModuleNotFoundError:
        from src.options import TestingConfig

    cli = build_parser().parse_args()
    config_path = Path(cli.config)
    checkpoint = Path(cli.checkpoint)
    args = OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config_path))
    args.resume = str(checkpoint)
    run_gradient_conflict_diagnostic(
        args,
        checkpoint=checkpoint,
        output_prefix=Path(cli.output),
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
