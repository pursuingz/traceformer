from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

try:
    from .eval_metrics import umeyama_decompose
except ImportError:  # direct script execution with PYTHONPATH=src
    from utils.eval_metrics import umeyama_decompose


SWEEP_CONDITIONS = (
    "normal",
    "e_low",
    "e_mid",
    "e_high",
    "nu_low",
    "nu_mid",
    "nu_high",
)
E_SWEEP_VALUES = (4.5, 5.5, 6.5)
NU_SWEEP_VALUES = (0.10, 0.25, 0.40)
SUPPORTED_MATERIALS = (0, 1, 2)
MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
EXPECTED_MATERIAL_COUNTS = {0: 13, 1: 14, 2: 14}
STATE_METRICS = (
    "motion_rms",
    "f24_centroid_displacement",
    "f24_shape_deformation",
    "f24_volume_relative_change",
    "penetration_rate",
    "penetration_depth",
)
RESPONSE_METRICS = (
    "prediction_response_mse",
    "final_response_mse",
    "f24_centroid_response",
    "f24_shape_response",
)
RAW_FIELDS = (
    "checkpoint",
    "config",
    "seed",
    "sample_scope",
    "model",
    "mat_type",
    "true_log10_e",
    "true_nu",
    "condition",
    "scanned_log10_e",
    "scanned_nu",
    *STATE_METRICS,
    *RESPONSE_METRICS,
)


@dataclass(frozen=True)
class SweepCondition:
    name: str
    log10_e: float
    nu: float
    mat_type: int


def _finite_scalar(value: Any, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def build_sweep_conditions(record: Any) -> dict[str, SweepCondition]:
    log10_e = _finite_scalar(record.log10_e, "log10_e")
    nu = _finite_scalar(record.nu, "nu")
    mat_type = int(record.mat_type)
    if mat_type not in SUPPORTED_MATERIALS:
        raise ValueError(f"unsupported mat_type {mat_type}")

    values = (
        SweepCondition("normal", log10_e, nu, mat_type),
        *(SweepCondition(f"e_{name}", value, nu, mat_type) for name, value in zip(
            ("low", "mid", "high"), E_SWEEP_VALUES
        )),
        *(SweepCondition(f"nu_{name}", log10_e, value, mat_type) for name, value in zip(
            ("low", "mid", "high"), NU_SWEEP_VALUES
        )),
    )
    return {condition.name: condition for condition in values}


def _trajectory_tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim != ndim or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., N, 3)")
    if tensor.shape[-2] < 4:
        raise ValueError(f"{name} must contain at least four particles")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _cloud_volume(cloud: torch.Tensor) -> float:
    try:
        volume = float(ConvexHull(cloud.numpy()).volume)
    except Exception as exc:
        raise ValueError("point cloud must define a finite 3D convex hull") from exc
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("point cloud volume must be positive and finite")
    return volume


def trajectory_state_metrics(
    prediction: Any,
    reference_frame: Any,
    floor_height: float,
) -> dict[str, float]:
    pred = _trajectory_tensor(prediction, "prediction", 3)
    reference = _trajectory_tensor(reference_frame, "reference_frame", 2)
    if pred.shape[1:] != reference.shape:
        raise ValueError("prediction and reference_frame particle shapes must match")
    floor = _finite_scalar(floor_height, "floor_height")

    displacement = pred - reference.unsqueeze(0)
    motion_rms = torch.sqrt(displacement.square().sum(dim=-1).mean()).item()
    centroid_displacement = (pred[-1].mean(0) - reference.mean(0)).norm().item()
    _, _, _, shape_deformation = umeyama_decompose(pred[-1], reference)
    reference_volume = _cloud_volume(reference)
    final_volume = _cloud_volume(pred[-1])
    volume_change = abs(final_volume - reference_volume) / reference_volume

    penetration = torch.clamp(floor - pred[..., 1], min=0.0)
    return {
        "motion_rms": float(motion_rms),
        "f24_centroid_displacement": float(centroid_displacement),
        "f24_shape_deformation": float(shape_deformation),
        "f24_volume_relative_change": float(volume_change),
        "penetration_rate": float((penetration > 0).float().mean()),
        "penetration_depth": float(penetration.mean()),
    }


def response_metrics(normal: Any, counterfactual: Any) -> dict[str, float]:
    normal_tensor = _trajectory_tensor(normal, "normal", 3)
    counterfactual_tensor = _trajectory_tensor(counterfactual, "counterfactual", 3)
    if normal_tensor.shape != counterfactual_tensor.shape:
        raise ValueError("normal and counterfactual trajectories must share shape")

    delta = counterfactual_tensor - normal_tensor
    _, _, _, shape_response = umeyama_decompose(
        counterfactual_tensor[-1], normal_tensor[-1]
    )
    return {
        "prediction_response_mse": float(delta.square().mean()),
        "final_response_mse": float(delta[-1].square().mean()),
        "f24_centroid_response": float(
            (counterfactual_tensor[-1].mean(0) - normal_tensor[-1].mean(0)).norm()
        ),
        "f24_shape_response": float(shape_response),
    }


def spearman_monotonicity(
    values: Sequence[float],
    responses: Sequence[float],
    expected_direction: Literal["increasing", "decreasing"],
) -> dict[str, float | bool]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.shape != (3,) or y.shape != (3,):
        raise ValueError("monotonicity requires exactly three values and responses")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("monotonicity inputs must be finite")
    if expected_direction not in ("increasing", "decreasing"):
        raise ValueError("expected_direction must be increasing or decreasing")
    if not np.all(np.diff(x) > 0):
        raise ValueError("scan values must be strictly increasing")

    differences = np.diff(y)
    if expected_direction == "decreasing":
        strict = bool(np.all(differences < 0))
        weak = bool(np.all(differences <= 0))
    else:
        strict = bool(np.all(differences > 0))
        weak = bool(np.all(differences >= 0))
    if np.all(y == y[0]):
        rho = 0.0
    else:
        rho = float(spearmanr(x, y).statistic)
        if not math.isfinite(rho):
            rho = 0.0
    return {"rho": rho, "strict_monotonic": strict, "weak_monotonic": weak}


def _row_float(row: dict[str, Any], field: str) -> float:
    if field not in row:
        raise ValueError(f"raw row is missing field {field!r}")
    return _finite_scalar(row[field], field)


def validate_raw_rows(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != 41 * len(SWEEP_CONDITIONS):
        raise ValueError("B2 raw output must contain exactly 287 rows")
    provenance_fields = ("checkpoint", "config", "seed", "sample_scope")
    expected_provenance = tuple(rows[0].get(field) for field in provenance_fields)
    if any(value in (None, "") for value in expected_provenance):
        raise ValueError("B2 provenance must be complete")

    seen: set[tuple[str, str]] = set()
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if tuple(row.get(field) for field in provenance_fields) != expected_provenance:
            raise ValueError("B2 provenance must be identical across rows")
        missing = [field for field in RAW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"raw row is missing fields: {missing}")
        model = str(row["model"])
        condition = str(row["condition"])
        if not model or condition not in SWEEP_CONDITIONS:
            raise ValueError("raw row has invalid model or condition")
        key = (model, condition)
        if key in seen:
            raise ValueError(f"duplicate model-condition row: {key}")
        seen.add(key)
        by_model.setdefault(model, []).append(row)
        mat_type_value = _row_float(row, "mat_type")
        if not mat_type_value.is_integer() or int(mat_type_value) not in SUPPORTED_MATERIALS:
            raise ValueError("mat_type must be one of 0, 1, 2")
        for field in (
            "true_log10_e",
            "true_nu",
            "scanned_log10_e",
            "scanned_nu",
            *STATE_METRICS,
            *RESPONSE_METRICS,
        ):
            _row_float(row, field)

    if len(by_model) != 41:
        raise ValueError("B2 output must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in SUPPORTED_MATERIALS}
    for model, model_rows in by_model.items():
        conditions = tuple(
            row["condition"]
            for row in sorted(model_rows, key=lambda row: SWEEP_CONDITIONS.index(row["condition"]))
        )
        if conditions != SWEEP_CONDITIONS:
            raise ValueError(f"{model}: incomplete B2 condition set")
        mat_types = {int(float(row["mat_type"])) for row in model_rows}
        true_values = {
            (_row_float(row, "true_log10_e"), _row_float(row, "true_nu"))
            for row in model_rows
        }
        if len(mat_types) != 1 or len(true_values) != 1:
            raise ValueError(f"{model}: material metadata changed across conditions")
        material_counts[next(iter(mat_types))] += 1
    if material_counts != EXPECTED_MATERIAL_COUNTS:
        raise ValueError(
            f"B2 material counts must be {EXPECTED_MATERIAL_COUNTS}; got {material_counts}"
        )


def _condition_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): row for row in rows}


def _classification(
    response_ratio: float,
    e_plausible: bool,
    nu_plausible: bool,
) -> str:
    if response_ratio <= 0.02:
        return "ignored"
    if response_ratio > 1.0:
        return "unstable_excessive"
    if e_plausible and nu_plausible:
        return "directionally_plausible"
    return "responsive_non_monotonic"


def build_model_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_raw_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    summaries = []
    for model in sorted(grouped):
        conditions = _condition_map(grouped[model])
        normal = conditions["normal"]
        e_rows = [conditions[name] for name in ("e_low", "e_mid", "e_high")]
        nu_rows = [conditions[name] for name in ("nu_low", "nu_mid", "nu_high")]
        e_shape = spearman_monotonicity(
            E_SWEEP_VALUES,
            [_row_float(row, "f24_shape_deformation") for row in e_rows],
            "decreasing",
        )
        e_motion = spearman_monotonicity(
            E_SWEEP_VALUES,
            [_row_float(row, "motion_rms") for row in e_rows],
            "decreasing",
        )
        nu_volume = spearman_monotonicity(
            NU_SWEEP_VALUES,
            [_row_float(row, "f24_volume_relative_change") for row in nu_rows],
            "decreasing",
        )
        e_response = float(
            np.mean([_row_float(row, "prediction_response_mse") for row in e_rows])
        )
        nu_response = float(
            np.mean([_row_float(row, "prediction_response_mse") for row in nu_rows])
        )
        normal_coordinate_motion = max(
            _row_float(normal, "motion_rms") ** 2 / 3.0,
            np.finfo(np.float64).eps,
        )
        response_ratio = max(e_response, nu_response) / normal_coordinate_motion
        summaries.append(
            {
                "checkpoint": normal["checkpoint"],
                "config": normal["config"],
                "seed": int(float(normal["seed"])),
                "sample_scope": normal["sample_scope"],
                "model": model,
                "mat_type": int(float(normal["mat_type"])),
                "true_log10_e": _row_float(normal, "true_log10_e"),
                "true_nu": _row_float(normal, "true_nu"),
                "mean_e_response_mse": e_response,
                "mean_nu_response_mse": nu_response,
                "max_response_ratio": response_ratio,
                "e_shape_rho": e_shape["rho"],
                "e_shape_strict_monotonic": e_shape["strict_monotonic"],
                "e_shape_weak_monotonic": e_shape["weak_monotonic"],
                "e_motion_rho": e_motion["rho"],
                "e_motion_strict_monotonic": e_motion["strict_monotonic"],
                "e_motion_weak_monotonic": e_motion["weak_monotonic"],
                "nu_volume_rho": nu_volume["rho"],
                "nu_volume_strict_monotonic": nu_volume["strict_monotonic"],
                "nu_volume_weak_monotonic": nu_volume["weak_monotonic"],
                "classification": _classification(
                    response_ratio,
                    bool(e_shape["weak_monotonic"]),
                    bool(nu_volume["weak_monotonic"]),
                ),
            }
        )
    return summaries


def build_group_summaries(model_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(model_rows) != 41:
        raise ValueError("model summary must contain exactly 41 rows")
    groups = [("overall", list(model_rows))]
    groups.extend(
        (name, [row for row in model_rows if int(row["mat_type"]) == mat_type])
        for mat_type, name in MATERIAL_NAMES.items()
    )
    output = []
    for name, rows in groups:
        if not rows:
            raise ValueError(f"{name} group must not be empty")
        output.append(
            {
                "group": name,
                "n": len(rows),
                "mean_e_response_mse": float(np.mean([row["mean_e_response_mse"] for row in rows])),
                "mean_nu_response_mse": float(np.mean([row["mean_nu_response_mse"] for row in rows])),
                "median_e_shape_rho": float(np.median([row["e_shape_rho"] for row in rows])),
                "median_nu_volume_rho": float(np.median([row["nu_volume_rho"] for row in rows])),
                "e_shape_weak_fraction": float(np.mean([row["e_shape_weak_monotonic"] for row in rows])),
                "nu_volume_weak_fraction": float(np.mean([row["nu_volume_weak_monotonic"] for row in rows])),
                "ignored_fraction": float(np.mean([row["classification"] == "ignored" for row in rows])),
                "responsive_non_monotonic_fraction": float(
                    np.mean([row["classification"] == "responsive_non_monotonic" for row in rows])
                ),
                "directionally_plausible_fraction": float(
                    np.mean([row["classification"] == "directionally_plausible" for row in rows])
                ),
                "unstable_excessive_fraction": float(
                    np.mean([row["classification"] == "unstable_excessive" for row in rows])
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"inconsistent CSV schema for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_outputs(
    output_dir: str | Path,
    raw_rows: Sequence[dict[str, Any]],
    model_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    validate_raw_rows(raw_rows)
    if len(model_rows) != 41 or len(summary_rows) != 4:
        raise ValueError("B2 summaries must contain 41 model rows and four groups")
    for field in ("checkpoint", "config", "seed", "sample_scope"):
        if metadata.get(field) in (None, ""):
            raise ValueError(f"metadata is missing {field}")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": directory / "material_response_sweep_b2_raw.csv",
        "model_summary": directory / "material_response_sweep_b2_model_summary.csv",
        "summary": directory / "material_response_sweep_b2_summary.csv",
        "report": directory / "material_response_sweep_b2.md",
    }
    _write_csv(paths["raw"], raw_rows)
    _write_csv(paths["model_summary"], model_rows)
    _write_csv(paths["summary"], summary_rows)

    lines = [
        "# B2 连续材料响应扫描",
        "",
        "## 实验元数据",
        "",
        f"- checkpoint: `{metadata['checkpoint']}`",
        f"- config: `{metadata['config']}`",
        f"- seed: `{metadata['seed']}`",
        f"- sample_scope: {metadata['sample_scope']}",
        "- 扫描: log10(E)=4.5/5.5/6.5；nu=0.10/0.25/0.40",
        "- 完整性: 287 condition rows；41 model rows；材质 13/14/14",
        "",
        "## 解释限制",
        "",
        "> 本实验没有 counterfactual GT，只能判断参数响应强度、单调性和方向；不能判断反事实轨迹的准确率。",
        "> `directionally_plausible` 不是准确性结论，只表示扫描趋势符合预注册的 constitutive sanity direction。",
        "",
        "## 分组结果",
        "",
        "| group | n | E response MSE | nu response MSE | median E-shape rho | median nu-volume rho | plausible | ignored | non-monotonic | excessive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {n} | {mean_e_response_mse:.6e} | {mean_nu_response_mse:.6e} | "
            "{median_e_shape_rho:+.3f} | {median_nu_volume_rho:+.3f} | "
            "{directionally_plausible_fraction:.1%} | {ignored_fraction:.1%} | "
            "{responsive_non_monotonic_fraction:.1%} | {unstable_excessive_fraction:.1%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 分类定义",
            "",
            "- `ignored`: 最大参数响应不超过 normal coordinate-motion energy 的 2%。",
            "- `responsive_non_monotonic`: 响应可见，但 E-shape 或 nu-volume 未满足宽松单调方向。",
            "- `directionally_plausible`: 两个主要 sanity 指标均满足宽松单调方向。",
            "- `unstable_excessive`: 参数响应超过 normal coordinate-motion energy 的 100%。",
            "",
        ]
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}
