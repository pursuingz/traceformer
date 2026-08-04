from contextlib import contextmanager
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from utils.eval_metrics import per_window_metrics


KNOCKOUT_METRICS = (
    "full_rollout_mse",
    "short_mse",
    "mid_mse",
    "long_mse",
    "gm_mse",
    "fde",
    "f24_centroid_error",
    "f24_shape_residual_mse",
    "penetration_rate",
    "penetration_depth",
)


KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
)


_MATERIAL_GROUPS = {0: "elastic", 1: "plasticine", 2: "sand"}
_RAW_METADATA_FIELDS = ("model", "mat_type", "log10_e", "nu")
_KNOCKOUT_CONDITION_NAMES = tuple(name for name, _ in KNOCKOUT_CONDITIONS)
_PROVENANCE_FIELDS = ("checkpoint", "config", "seed", "sample_scope")
_RAW_CSV_COLUMNS = (
    *_PROVENANCE_FIELDS,
    *_RAW_METADATA_FIELDS,
    "condition",
    *KNOCKOUT_METRICS,
)
_PAIRED_METRIC_COLUMNS = tuple(
    field
    for metric in KNOCKOUT_METRICS
    for field in (
        f"normal_{metric}",
        f"knockout_{metric}",
        f"delta_{metric}",
        f"relative_change_pct_{metric}",
    )
)
_PAIRED_CSV_COLUMNS = (
    *_PROVENANCE_FIELDS,
    *_RAW_METADATA_FIELDS,
    "condition",
    *_PAIRED_METRIC_COLUMNS,
)
_SUMMARY_GROUPS = (
    "overall",
    "elastic",
    "plasticine",
    "sand",
    "elastic_low_E",
    "elastic_high_E",
    "plasticine_low_E",
    "plasticine_high_E",
    "sand_low_E",
    "sand_high_E",
)


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def validate_raw_rows(rows):
    """Validate the pre-registered 41-model by five-condition raw table."""
    if not isinstance(rows, list):
        raise ValueError("raw rows must be a list")
    if len(rows) != 205:
        raise ValueError("raw rows must contain exactly 205 rows")

    by_model = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each raw row must be a dictionary")
        for field in (*_RAW_METADATA_FIELDS, "condition", *KNOCKOUT_METRICS):
            if field not in row:
                raise ValueError(f"missing raw field: {field}")
        model = row["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        mat_type = row["mat_type"]
        if isinstance(mat_type, bool) or not isinstance(mat_type, (int, np.integer)):
            raise ValueError("mat_type must be 0, 1, or 2")
        if mat_type not in _MATERIAL_GROUPS:
            raise ValueError("mat_type must be 0, 1, or 2")
        if row["condition"] not in _KNOCKOUT_CONDITION_NAMES:
            raise ValueError("condition is not pre-registered")
        _finite_number(row["log10_e"], "log10_e")
        _finite_number(row["nu"], "nu")
        for metric in KNOCKOUT_METRICS:
            value = _finite_number(row[metric], metric)
            if value < 0:
                raise ValueError(f"{metric} must be non-negative")
        by_model.setdefault(model, []).append(row)

    if len(by_model) != 41:
        raise ValueError("raw rows must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in _MATERIAL_GROUPS}
    for model, model_rows in by_model.items():
        if len(model_rows) != len(KNOCKOUT_CONDITIONS):
            raise ValueError(f"{model}: expected five conditions")
        conditions = tuple(row["condition"] for row in model_rows)
        if set(conditions) != set(_KNOCKOUT_CONDITION_NAMES):
            raise ValueError(f"{model}: condition set is incomplete or duplicated")
        metadata = tuple(model_rows[0][field] for field in _RAW_METADATA_FIELDS[1:])
        if any(
            tuple(row[field] for field in _RAW_METADATA_FIELDS[1:]) != metadata
            for row in model_rows[1:]
        ):
            raise ValueError(f"{model}: metadata must agree across conditions")
        material_counts[metadata[0]] += 1
    if material_counts != {0: 13, 1: 14, 2: 14}:
        raise ValueError("material counts must be elastic=13, plasticine=14, sand=14")
    return list(rows)


def build_paired_rows(rows):
    """Pair every knockout condition with the same model's normal rollout."""
    validated_rows = validate_raw_rows(rows)
    by_model = {}
    for row in validated_rows:
        by_model.setdefault(row["model"], {})[row["condition"]] = row

    paired_rows = []
    for model in sorted(by_model):
        by_condition = by_model[model]
        normal = by_condition["normal"]
        for condition in _KNOCKOUT_CONDITION_NAMES:
            if condition == "normal":
                continue
            knockout = by_condition[condition]
            paired = {
                field: normal[field] for field in _RAW_METADATA_FIELDS
            }
            paired["condition"] = condition
            for metric in KNOCKOUT_METRICS:
                normal_value = float(normal[metric])
                knockout_value = float(knockout[metric])
                delta = knockout_value - normal_value
                paired[f"normal_{metric}"] = normal_value
                paired[f"knockout_{metric}"] = knockout_value
                paired[f"delta_{metric}"] = delta
                paired[f"relative_change_pct_{metric}"] = (
                    delta / normal_value * 100.0
                    if normal_value > 0
                    else 0.0 if knockout_value == 0 else None
                )
            paired_rows.append(paired)
    return paired_rows


def paired_delta_summary(normal, knockout, samples, seed):
    """Summarize paired values and bootstrap only their model-level deltas."""
    normal = np.asarray(normal, dtype=float)
    knockout = np.asarray(knockout, dtype=float)
    if normal.ndim != 1 or knockout.ndim != 1 or normal.size == 0:
        raise ValueError("normal and knockout must be non-empty one-dimensional arrays")
    if normal.shape != knockout.shape:
        raise ValueError("normal and knockout must have the same shape")
    if not np.isfinite(normal).all() or not np.isfinite(knockout).all():
        raise ValueError("normal and knockout must contain only finite values")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    delta = knockout - normal
    indices = np.random.default_rng(seed).integers(
        0, delta.size, size=(samples, delta.size)
    )
    bootstrap_means = delta[indices].mean(axis=1)
    normal_mean = float(normal.mean())
    knockout_mean = float(knockout.mean())
    ci_low, ci_high = np.percentile(bootstrap_means, (2.5, 97.5))
    return {
        "n_models": int(delta.size),
        "normal_mean": normal_mean,
        "knockout_mean": knockout_mean,
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "relative_change_pct": (
            float(delta.mean() / normal_mean * 100.0)
            if normal_mean > 0
            else 0.0 if knockout_mean == 0 else None
        ),
        "improved_count": int(np.sum(delta < 0)),
        "degraded_count": int(np.sum(delta > 0)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _validate_paired_rows(rows):
    if not isinstance(rows, list) or len(rows) != 164:
        raise ValueError("paired rows must contain exactly 164 rows")
    expected_conditions = set(_KNOCKOUT_CONDITION_NAMES) - {"normal"}
    by_model = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each paired row must be a dictionary")
        for field in (*_RAW_METADATA_FIELDS, "condition"):
            if field not in row:
                raise ValueError(f"missing paired field: {field}")
        if row["condition"] not in expected_conditions:
            raise ValueError("paired condition is not pre-registered")
        mat_type = row["mat_type"]
        if isinstance(mat_type, bool) or mat_type not in _MATERIAL_GROUPS:
            raise ValueError("mat_type must be 0, 1, or 2")
        _finite_number(row["log10_e"], "log10_e")
        _finite_number(row["nu"], "nu")
        for metric in KNOCKOUT_METRICS:
            normal = _finite_number(row.get(f"normal_{metric}"), f"normal_{metric}")
            knockout = _finite_number(
                row.get(f"knockout_{metric}"), f"knockout_{metric}"
            )
            delta = _finite_number(row.get(f"delta_{metric}"), f"delta_{metric}")
            relative_field = f"relative_change_pct_{metric}"
            if relative_field not in row:
                raise ValueError(f"missing paired field: {relative_field}")
            relative = row[relative_field]
            if normal < 0 or knockout < 0:
                raise ValueError(f"{metric} values must be non-negative")
            if not math.isclose(delta, knockout - normal, abs_tol=1e-12):
                raise ValueError(f"delta_{metric} must equal knockout minus normal")
            expected_relative = (
                delta / normal * 100.0
                if normal > 0
                else 0.0 if knockout == 0 else None
            )
            if expected_relative is None:
                if relative is not None:
                    raise ValueError(f"{relative_field} must be None for zero baseline")
            else:
                relative = _finite_number(relative, relative_field)
                if not math.isclose(relative, expected_relative, abs_tol=1e-10):
                    raise ValueError(
                        f"{relative_field} must match knockout minus normal"
                    )
        by_model.setdefault(row["model"], []).append(row)

    if len(by_model) != 41:
        raise ValueError("paired rows must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in _MATERIAL_GROUPS}
    for model, model_rows in by_model.items():
        if len(model_rows) != len(expected_conditions):
            raise ValueError(f"{model}: expected four paired conditions")
        if {row["condition"] for row in model_rows} != expected_conditions:
            raise ValueError(f"{model}: paired condition set is incomplete or duplicated")
        metadata = tuple(model_rows[0][field] for field in _RAW_METADATA_FIELDS[1:])
        if any(
            tuple(row[field] for field in _RAW_METADATA_FIELDS[1:]) != metadata
            for row in model_rows[1:]
        ):
            raise ValueError(f"{model}: metadata must agree across paired conditions")
        material_counts[metadata[0]] += 1
    if material_counts != {0: 13, 1: 14, 2: 14}:
        raise ValueError("material counts must be elastic=13, plasticine=14, sand=14")
    return list(rows)


def summarize_paired_rows(rows, bootstrap_samples, bootstrap_seed):
    """Return equal-model-weighted overall, material, and within-material E summaries."""
    paired_rows = _validate_paired_rows(rows)
    group_rows = [("overall", paired_rows)]
    for mat_type, material in _MATERIAL_GROUPS.items():
        material_rows = [row for row in paired_rows if row["mat_type"] == mat_type]
        group_rows.append((material, material_rows))
        median_e = float(np.median([row["log10_e"] for row in material_rows]))
        group_rows.append(
            (
                f"{material}_low_E",
                [row for row in material_rows if row["log10_e"] <= median_e],
            )
        )
        group_rows.append(
            (
                f"{material}_high_E",
                [row for row in material_rows if row["log10_e"] > median_e],
            )
        )

    summary_rows = []
    for group, grouped_rows in group_rows:
        for condition in _KNOCKOUT_CONDITION_NAMES:
            if condition == "normal":
                continue
            condition_rows = [
                row for row in grouped_rows if row["condition"] == condition
            ]
            for metric in KNOCKOUT_METRICS:
                stats = paired_delta_summary(
                    np.asarray(
                        [row[f"normal_{metric}"] for row in condition_rows],
                        dtype=float,
                    ),
                    np.asarray(
                        [row[f"knockout_{metric}"] for row in condition_rows],
                        dtype=float,
                    ),
                    bootstrap_samples,
                    bootstrap_seed,
                )
                summary_rows.append(
                    {"group": group, "condition": condition, "metric": metric, **stats}
                )
    return summary_rows


def _summary_row_index(summary_rows):
    if not isinstance(summary_rows, list):
        raise ValueError("summary rows must be a list")
    index = {}
    required = {
        "group",
        "condition",
        "metric",
        "n_models",
        "normal_mean",
        "knockout_mean",
        "median_delta",
        "relative_change_pct",
        "improved_count",
        "degraded_count",
    }
    for row in summary_rows:
        if not isinstance(row, dict) or not required.issubset(row):
            continue
        key = (row["group"], row["condition"], row["metric"])
        if key in index:
            raise ValueError("summary rows must not contain duplicate group-condition-metric rows")
        index[key] = row
    return index


def _verdict_stats(row):
    if row is None:
        return None
    n_models = row["n_models"]
    improved_count = row["improved_count"]
    degraded_count = row["degraded_count"]
    if (
        isinstance(n_models, bool)
        or not isinstance(n_models, (int, np.integer))
        or n_models <= 0
        or isinstance(improved_count, bool)
        or not isinstance(improved_count, (int, np.integer))
        or not 0 <= improved_count <= n_models
        or isinstance(degraded_count, bool)
        or not isinstance(degraded_count, (int, np.integer))
        or not 0 <= degraded_count <= n_models
    ):
        return None
    try:
        normal_mean = _finite_number(row["normal_mean"], "normal_mean")
        knockout_mean = _finite_number(row["knockout_mean"], "knockout_mean")
        median_delta = _finite_number(row["median_delta"], "median_delta")
    except ValueError:
        return None
    relative_change_pct = row["relative_change_pct"]
    if relative_change_pct is not None:
        try:
            relative_change_pct = _finite_number(
                relative_change_pct, "relative_change_pct"
            )
        except ValueError:
            return None
    return {
        "n_models": int(n_models),
        "improved_count": int(improved_count),
        "degraded_count": int(degraded_count),
        "normal_mean": normal_mean,
        "knockout_mean": knockout_mean,
        "median_delta": median_delta,
        "relative_change_pct": relative_change_pct,
    }


def dynamic_gate_verdict(summary_rows):
    """Apply the pre-registered B1b dynamic-gate decision rule, fail-closed."""
    index = _summary_row_index(summary_rows)
    candidate_metrics = ("long_mse", "fde", "f24_centroid_error")
    safety_metrics = ("full_rollout_mse", "fde")
    penetration_metrics = ("penetration_rate", "penetration_depth")
    failure_reasons = []

    for stage in ("stage0_off", "stage2_off"):
        plasticine_metrics = []
        sand_opposite_metrics = []
        for metric in candidate_metrics:
            plasticine = _verdict_stats(index.get(("plasticine", stage, metric)))
            if (
                plasticine is not None
                and plasticine["n_models"] == 14
                and plasticine["relative_change_pct"] is not None
                and plasticine["relative_change_pct"] <= -5.0
                and plasticine["improved_count"] >= 8
                and plasticine["median_delta"] < 0.0
            ):
                plasticine_metrics.append(metric)
                sand = _verdict_stats(index.get(("sand", stage, metric)))
                if (
                    sand is not None
                    and sand["n_models"] == 14
                    and sand["relative_change_pct"] is not None
                    and sand["relative_change_pct"] >= 5.0
                    and sand["degraded_count"] >= 8
                    and sand["median_delta"] > 0.0
                ):
                    sand_opposite_metrics.append(metric)

        safe_trajectory = True
        for metric in safety_metrics:
            stats = _verdict_stats(index.get(("overall", stage, metric)))
            if (
                stats is None
                or stats["n_models"] != 41
                or stats["relative_change_pct"] is None
                or stats["relative_change_pct"] >= 10.0
            ):
                safe_trajectory = False

        safe_penetration = True
        for metric in penetration_metrics:
            stats = _verdict_stats(index.get(("overall", stage, metric)))
            if stats is None or stats["n_models"] != 41:
                safe_penetration = False
            elif stats["normal_mean"] == 0.0:
                if stats["knockout_mean"] != 0.0:
                    safe_penetration = False
            elif (
                stats["relative_change_pct"] is None
                or stats["relative_change_pct"] >= 25.0
            ):
                safe_penetration = False

        if (
            len(plasticine_metrics) >= 2
            and sand_opposite_metrics
            and safe_trajectory
            and safe_penetration
        ):
            return {
                "proceed_dynamic_gate": True,
                "qualifying_stage": stage,
                "plasticine_metrics": tuple(plasticine_metrics),
                "sand_opposite_metrics": tuple(sand_opposite_metrics),
                "reasons": (f"{stage} satisfies all pre-registered criteria",),
            }

        if len(plasticine_metrics) < 2:
            failure_reasons.append(f"{stage}: fewer than two plasticine metrics qualify")
        if not sand_opposite_metrics:
            failure_reasons.append(f"{stage}: no qualifying sand opposite response")
        if not safe_trajectory:
            failure_reasons.append(f"{stage}: overall trajectory safety threshold failed")
        if not safe_penetration:
            failure_reasons.append(f"{stage}: overall penetration safety threshold failed")

    return {
        "proceed_dynamic_gate": False,
        "qualifying_stage": None,
        "plasticine_metrics": (),
        "sand_opposite_metrics": (),
        "reasons": tuple(failure_reasons),
    }


def _validate_csv_provenance(rows):
    if not rows:
        raise ValueError("CSV rows must not be empty")
    expected = None
    for row in rows:
        values = {}
        for field in _PROVENANCE_FIELDS:
            if field not in row:
                raise ValueError(f"missing CSV provenance field: {field}")
            value = row[field]
            if field == "seed":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or value < 0
                ):
                    raise ValueError("CSV provenance seed must be a non-negative integer")
                value = int(value)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CSV provenance {field} must be a non-empty string"
                )
            values[field] = value
        if expected is None:
            expected = values
        elif values != expected:
            raise ValueError("CSV provenance must agree across all rows")
    return expected


def _csv_value(value):
    return "" if value is None else value


def _write_csv(path, rows, columns, sort_key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def write_raw_csv(path: Path, rows: list[dict]) -> None:
    """Write the fixed 205-row raw knockout table with repeated provenance."""
    validated = validate_raw_rows(rows)
    _validate_csv_provenance(validated)
    condition_order = {
        condition: index for index, condition in enumerate(_KNOCKOUT_CONDITION_NAMES)
    }
    _write_csv(
        path,
        validated,
        _RAW_CSV_COLUMNS,
        lambda row: (str(row["model"]), condition_order[row["condition"]]),
    )


def write_paired_csv(path: Path, rows: list[dict]) -> None:
    """Write the fixed 164-row paired knockout-minus-normal table."""
    validated = _validate_paired_rows(rows)
    _validate_csv_provenance(validated)
    condition_order = {
        condition: index for index, condition in enumerate(_KNOCKOUT_CONDITION_NAMES)
    }
    _write_csv(
        path,
        validated,
        _PAIRED_CSV_COLUMNS,
        lambda row: (str(row["model"]), condition_order[row["condition"]]),
    )


def _validate_report_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    validated = {}
    for field in ("checkpoint", "config", "sample_scope"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata must contain a non-empty string: {field}")
        validated[field] = value
    for field, positive in (
        ("seed", False),
        ("bootstrap_samples", True),
        ("bootstrap_seed", False),
    ):
        value = metadata.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < (1 if positive else 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"metadata {field} must be a {qualifier} integer")
        validated[field] = int(value)
    return validated


def _validate_report_summary(summary_rows):
    index = _summary_row_index(summary_rows)
    expected = {
        (group, condition, metric)
        for group in _SUMMARY_GROUPS
        for condition in _KNOCKOUT_CONDITION_NAMES
        if condition != "normal"
        for metric in KNOCKOUT_METRICS
    }
    if set(index) != expected:
        missing = sorted(expected - set(index))
        unexpected = sorted(set(index) - expected)
        raise ValueError(
            "summary rows must contain the complete registered table: "
            f"missing={missing}; unexpected={unexpected}"
        )
    required_stats = (
        "mean_delta",
        "ci_low",
        "ci_high",
    )
    for key, row in index.items():
        stats = _verdict_stats(row)
        if stats is None:
            raise ValueError(f"invalid summary row: {key}")
        for field in required_stats:
            _finite_number(row.get(field), field)
    return index


def _report_number(value, *, signed=False, percent=False):
    if value is None:
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    suffix = "%" if percent else ""
    if signed:
        return f"{number:+.6e}{suffix}" if not percent else f"{number:+.2f}{suffix}"
    return f"{number:.6e}{suffix}"


def _append_summary_table(lines, rows):
    lines.extend(
        [
            "| group | condition | metric | n | normal mean | knockout mean | mean delta | median delta | relative change | improved | degraded | paired delta 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {group} | {condition} | {metric} | {n_models} | {normal} | "
            "{knockout} | {mean_delta} | {median_delta} | {relative} | "
            "{improved}/{n_models} | {degraded}/{n_models} | [{ci_low}, {ci_high}] |".format(
                group=row["group"],
                condition=row["condition"],
                metric=row["metric"],
                n_models=row["n_models"],
                normal=_report_number(row["normal_mean"]),
                knockout=_report_number(row["knockout_mean"]),
                mean_delta=_report_number(row["mean_delta"], signed=True),
                median_delta=_report_number(row["median_delta"], signed=True),
                relative=_report_number(
                    row["relative_change_pct"], signed=True, percent=True
                ),
                improved=row["improved_count"],
                degraded=row["degraded_count"],
                ci_low=_report_number(row["ci_low"], signed=True),
                ci_high=_report_number(row["ci_high"], signed=True),
            )
        )
    lines.append("")


def write_knockout_report(
    path: Path,
    raw_rows: list[dict],
    summary_rows: list[dict],
    metadata: dict,
    original_gates: Sequence[float],
    verdict: dict,
) -> None:
    """Write the complete Chinese B1b protocol report and registered verdict."""
    provenance = _validate_report_metadata(metadata)
    validated_raw = validate_raw_rows(raw_rows)
    paired_count = len(build_paired_rows(validated_raw))
    summary_index = _validate_report_summary(summary_rows)
    gates = np.asarray(tuple(original_gates), dtype=float)
    if gates.shape != (4,) or not np.isfinite(gates).all():
        raise ValueError("original_gates must contain four finite values")
    if not isinstance(verdict, dict) or not isinstance(
        verdict.get("proceed_dynamic_gate"), bool
    ):
        raise ValueError("verdict must contain proceed_dynamic_gate")
    reasons = verdict.get("reasons")
    if not isinstance(reasons, (tuple, list)) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("verdict reasons must be a sequence of non-empty strings")

    decision = "proceed" if verdict["proceed_dynamic_gate"] else "close"
    lines = [
        "# B1b HST Gate Knockout 诊断",
        "",
        "## 实验元数据",
        "",
        f"- checkpoint: `{provenance['checkpoint']}`",
        f"- config: `{provenance['config']}`",
        f"- seed: `{provenance['seed']}`",
        f"- sample_scope: {provenance['sample_scope']}",
        f"- bootstrap_samples: `{provenance['bootstrap_samples']}`",
        f"- bootstrap_seed: `{provenance['bootstrap_seed']}`",
        "- 原始 feedback gates: "
        + " / ".join(f"{gate:+.6f}" for gate in gates),
        "",
        "## 完整性检查",
        "",
        f"- raw rows: {len(validated_raw)} / 205",
        f"- paired rows: {paired_count} / 164",
        "- 配对差值定义为 `knockout - normal`；所有指标越低越好，负值表示改善。",
        "",
    ]

    ordered_summary = [summary_index[key] for key in sorted(summary_index)]
    sections = (
        ("## Overall", lambda row: row["group"] == "overall"),
        (
            "## 材质分层",
            lambda row: row["group"] in {"elastic", "plasticine", "sand"},
        ),
        ("## E 分层", lambda row: row["group"].endswith(("low_E", "high_E"))),
    )
    for title, predicate in sections:
        lines.extend([title, ""])
        _append_summary_table(lines, [row for row in ordered_summary if predicate(row)])

    lines.extend(
        [
            "## 预注册判定",
            "",
            f"- 最终判定: **{decision}**",
            f"- proceed_dynamic_gate: `{verdict['proceed_dynamic_gate']}`",
            f"- qualifying_stage: `{verdict.get('qualifying_stage')}`",
            "- reasons:",
            *(f"  - {reason}" for reason in reasons),
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@contextmanager
def masked_feedback_gates(exchange, mask):
    gates = exchange.feedback_gates
    if gates.numel() != 4 or not torch.isfinite(gates.detach()).all():
        raise ValueError("feedback_gates must contain four finite values")
    mask_tensor = torch.as_tensor(mask, device=gates.device, dtype=gates.dtype)
    if mask_tensor.shape != gates.shape or not torch.all(
        (mask_tensor == 0) | (mask_tensor == 1)
    ):
        raise ValueError("gate mask must contain exactly four binary values")
    original = gates.detach().clone()
    with torch.no_grad():
        gates.copy_(original * mask_tensor)
    try:
        yield gates.detach().clone()
    finally:
        with torch.no_grad():
            gates.copy_(original)
        if not torch.equal(gates.detach(), original):
            raise RuntimeError("feedback gates were not restored exactly")


def reset_inference_seed(seed, device):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def trajectory_knockout_metrics(pred, gt, input_frames, floor_height):
    """Return strict trajectory, geometry, and floor-penetration metrics."""
    if not isinstance(pred, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise ValueError("pred and gt must be tensors")
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred and gt must share shape (25, N, 3)")
    if pred.shape[0] != 25 or pred.shape[1] < 2:
        raise ValueError("pred and gt must have shape (25, N, 3)")
    if input_frames != 5:
        raise ValueError("input_frames must be 5")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("pred and gt must contain only finite values")

    pred_f = pred.float()
    gt_f = gt.to(pred.device).float()
    floor = torch.as_tensor(floor_height, device=pred.device, dtype=pred_f.dtype)
    if floor.numel() != 1 or not torch.isfinite(floor).all():
        raise ValueError("floor_height must be one finite scalar")

    frame_mse = (pred_f[input_frames:] - gt_f[input_frames:]).square().mean((1, 2))
    base = per_window_metrics(
        pred_f, gt_f, input_frames, k=min(8, pred.shape[1] - 1)
    )
    centroid, _, _, shape = base["proc"][24]
    penetration = torch.clamp(
        floor.reshape(()) - pred_f[input_frames:, :, 1], min=0
    )
    result = {
        "full_rollout_mse": float(
            (pred_f[input_frames:] - gt_f[input_frames:]).square().mean()
        ),
        "short_mse": float(
            (pred_f[5:11] - gt_f[5:11]).square().mean()
        ),
        "mid_mse": float(
            (pred_f[11:18] - gt_f[11:18]).square().mean()
        ),
        "long_mse": float(
            (pred_f[18:25] - gt_f[18:25]).square().mean()
        ),
        "gm_mse": float(torch.exp(torch.log(frame_mse.clamp_min(1e-30)).mean())),
        "fde": float(base["fde"]),
        "f24_centroid_error": float(centroid),
        "f24_shape_residual_mse": float(shape),
        "penetration_rate": float((penetration > 0).float().mean()),
        "penetration_depth": float(penetration.mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("trajectory knockout metrics must be finite")
    return result
