import csv
import math
from contextlib import contextmanager
from pathlib import Path
import weakref

import numpy as np
import torch

from .hybrid_state_gate_knockout import paired_delta_summary


STAGE_KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
    ("stage3_off", (1, 1, 1, 0)),
)
MATERIAL_GROUPS = {0: "elastic", 1: "plasticine", 2: "sand"}
SUMMARY_GROUPS = ("overall", "elastic", "plasticine", "sand")
STAGE_METRICS = (
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
_STAGE_CONDITION_NAMES = tuple(name for name, _ in STAGE_KNOCKOUT_CONDITIONS)
_RAW_METADATA_FIELDS = ("model", "mat_type", "log10_e", "nu")
_PROVENANCE_FIELDS = ("checkpoint", "config", "seed", "sample_scope")
_ACTIVE_COLLECTORS = weakref.WeakKeyDictionary()
_GRADIENT_MATERIAL_COUNTS = {"elastic": 52, "plasticine": 56, "sand": 56}
_GRADIENT_PAIRS = (
    ("elastic", "plasticine"),
    ("elastic", "sand"),
    ("plasticine", "sand"),
)


def _adapter_leaf_group(parameter_name):
    if parameter_name == "stage_scales":
        return "stage_scales"
    group = parameter_name.split(".", 1)[0]
    if group not in {"state_norm", "state_proj", "material_proj", "output_proj"}:
        raise ValueError(f"unregistered adapter parameter: {parameter_name}")
    return group


def _parameter_groups_from_names(parameter_names):
    parameter_names = tuple(parameter_names)
    if not parameter_names or len(parameter_names) != len(set(parameter_names)):
        raise ValueError("adapter parameter names must be non-empty and unique")

    groups = {"all_adapter": parameter_names}
    for name in parameter_names:
        if not isinstance(name, str) or not name:
            raise ValueError("adapter parameter names must be non-empty strings")
        group = _adapter_leaf_group(name)
        groups.setdefault(group, []).append(name)
    required = {
        "all_adapter",
        "state_norm",
        "state_proj",
        "material_proj",
        "output_proj",
        "stage_scales",
    }
    if set(groups) != required:
        raise ValueError("adapter parameter groups are incomplete")
    return {
        group: names if isinstance(names, tuple) else tuple(names)
        for group, names in groups.items()
    }


def adapter_parameter_groups(adapter):
    """Return deterministic full and leaf parameter groups for the B3a adapter."""
    return _parameter_groups_from_names(name for name, _ in adapter.named_parameters())


def snapshot_adapter_gradients(adapter):
    """Copy every adapter gradient to detached CPU float64 tensors."""
    snapshot = {}
    for name, parameter in adapter.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"missing gradient for adapter parameter: {name}")
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient shape mismatch for adapter parameter: {name}")
        gradient = gradient.detach()
        if not torch.isfinite(gradient).all():
            raise ValueError(f"nonfinite gradient for adapter parameter: {name}")
        snapshot[name] = gradient.to(device="cpu", dtype=torch.float64).clone()
    _parameter_groups_from_names(snapshot)
    return snapshot


def mean_named_gradients(sums, sample_count):
    """Convert named sample-weighted gradient sums into detached mean gradients."""
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(sums, dict) or not sums:
        raise ValueError("gradient sums must be a non-empty dictionary")

    means = {}
    for name, value in sums.items():
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ValueError("gradient sums must map names to tensors")
        value = value.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError(f"gradient sum must be finite: {name}")
        means[name] = (value / int(sample_count)).clone()
    return means


def _gradient_group_vectors(named_gradients, groups):
    vectors = {}
    for group, names in groups.items():
        vectors[group] = torch.cat(
            [named_gradients[name].reshape(-1) for name in names], dim=0
        )
    return vectors


def _gradient_cosine(left, right):
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def summarize_material_gradient_conflict(material_gradients):
    """Summarize B3a per-material mean gradients by norm and pairwise cosine."""
    if not isinstance(material_gradients, dict) or set(material_gradients) != set(
        _GRADIENT_MATERIAL_COUNTS
    ):
        raise ValueError("material gradients must contain elastic, plasticine, and sand")

    sample_counts = {}
    validated = {}
    parameter_names = None
    parameter_shapes = None
    for material, expected_count in _GRADIENT_MATERIAL_COUNTS.items():
        payload = material_gradients[material]
        if not isinstance(payload, dict):
            raise ValueError(f"{material} gradient payload must be a dictionary")
        sample_count = payload.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, (int, np.integer))
            or sample_count != expected_count
        ):
            raise ValueError(
                f"{material} sample_count must equal the frozen count {expected_count}"
            )
        named = payload.get("named_gradients")
        if not isinstance(named, dict) or not named:
            raise ValueError(f"{material} named_gradients must be non-empty")
        names = tuple(named)
        if parameter_names is None:
            parameter_names = names
            parameter_shapes = {name: value.shape for name, value in named.items()}
        elif names != parameter_names:
            raise ValueError("gradient parameter names and order must agree by material")

        material_values = {}
        for name, value in named.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"gradient must be a tensor: {material}.{name}")
            value = value.detach().to(device="cpu", dtype=torch.float64)
            if value.shape != parameter_shapes[name]:
                raise ValueError(f"gradient shape must agree by material: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"gradient must be finite: {material}.{name}")
            material_values[name] = value.clone()
        validated[material] = material_values
        sample_counts[material] = int(sample_count)

    groups = _parameter_groups_from_names(parameter_names)
    vectors = {
        material: _gradient_group_vectors(named, groups)
        for material, named in validated.items()
    }
    group_summary = {}
    for group in groups:
        group_summary[group] = {
            "gradient_norms": {
                material: float(torch.linalg.vector_norm(vectors[material][group]))
                for material in _GRADIENT_MATERIAL_COUNTS
            },
            "pairwise_cosine": {
                f"{left}__{right}": _gradient_cosine(
                    vectors[left][group], vectors[right][group]
                )
                for left, right in _GRADIENT_PAIRS
            },
        }

    stage_scale_gradients = {}
    for material, named in validated.items():
        stage_values = named["stage_scales"].reshape(-1)
        if stage_values.numel() != 4:
            raise ValueError("stage_scales gradient must contain exactly four values")
        stage_scale_gradients[material] = [float(value) for value in stage_values]
    return {
        "sample_counts": sample_counts,
        "groups": group_summary,
        "stage_scale_gradients": stage_scale_gradients,
    }


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_optional_provenance(rows):
    present = {
        field for field in _PROVENANCE_FIELDS if any(field in row for row in rows)
    }
    if not present:
        return None
    if present != set(_PROVENANCE_FIELDS):
        raise ValueError("provenance fields must be present together")
    expected = None
    for row in rows:
        values = {}
        for field in _PROVENANCE_FIELDS:
            if field not in row:
                raise ValueError(f"missing provenance field: {field}")
            value = row[field]
            if field == "seed":
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError("provenance seed must be a non-negative integer")
                value = int(value)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"provenance {field} must be a non-empty string")
            values[field] = value
        if expected is None:
            expected = values
        elif values != expected:
            raise ValueError("provenance must agree across all rows")
    return expected


def validate_stage_raw_rows(rows):
    """Validate the fixed B3a 41-model by six-condition stage table."""
    if not isinstance(rows, list):
        raise ValueError("raw rows must be a list")
    if len(rows) != 246:
        raise ValueError("raw rows must contain exactly 246 rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each raw row must be a dictionary")
    _validate_optional_provenance(rows)

    by_model = {}
    for row in rows:
        for field in (*_RAW_METADATA_FIELDS, "condition", *STAGE_METRICS):
            if field not in row:
                raise ValueError(f"missing raw field: {field}")
        model = row["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        mat_type = row["mat_type"]
        if isinstance(mat_type, bool) or not isinstance(mat_type, (int, np.integer)):
            raise ValueError("mat_type must be 0, 1, or 2")
        if mat_type not in MATERIAL_GROUPS:
            raise ValueError("mat_type must be 0, 1, or 2")
        if row["condition"] not in _STAGE_CONDITION_NAMES:
            raise ValueError("condition is not pre-registered")
        _finite_number(row["log10_e"], "log10_e")
        _finite_number(row["nu"], "nu")
        for metric in STAGE_METRICS:
            if _finite_number(row[metric], metric) < 0:
                raise ValueError(f"{metric} must be non-negative")
        by_model.setdefault(model, []).append(row)

    if len(by_model) != 41:
        raise ValueError("raw rows must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in MATERIAL_GROUPS}
    for model, model_rows in by_model.items():
        if len(model_rows) != len(STAGE_KNOCKOUT_CONDITIONS):
            raise ValueError(f"{model}: expected six conditions")
        if {row["condition"] for row in model_rows} != set(_STAGE_CONDITION_NAMES):
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
    return [dict(row) for row in rows]


def build_stage_paired_rows(rows):
    """Pair each B3a stage knockout against the matching normal rollout."""
    validated_rows = validate_stage_raw_rows(rows)
    provenance = _validate_optional_provenance(validated_rows)
    by_model = {}
    for row in validated_rows:
        by_model.setdefault(row["model"], {})[row["condition"]] = row

    paired_rows = []
    for model in sorted(by_model):
        by_condition = by_model[model]
        normal = by_condition["normal"]
        for condition in _STAGE_CONDITION_NAMES:
            if condition == "normal":
                continue
            knockout = by_condition[condition]
            paired = {field: normal[field] for field in _RAW_METADATA_FIELDS}
            if provenance is not None:
                paired.update(provenance)
            paired["condition"] = condition
            for metric in STAGE_METRICS:
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


def _validate_stage_paired_rows(rows):
    if not isinstance(rows, list) or len(rows) != 205:
        raise ValueError("paired rows must contain exactly 205 rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each paired row must be a dictionary")
    _validate_optional_provenance(rows)
    expected_conditions = set(_STAGE_CONDITION_NAMES) - {"normal"}
    by_model = {}
    for row in rows:
        for field in (*_RAW_METADATA_FIELDS, "condition"):
            if field not in row:
                raise ValueError(f"missing paired field: {field}")
        if row["condition"] not in expected_conditions:
            raise ValueError("paired condition is not pre-registered")
        if row["mat_type"] not in MATERIAL_GROUPS:
            raise ValueError("mat_type must be 0, 1, or 2")
        _finite_number(row["log10_e"], "log10_e")
        _finite_number(row["nu"], "nu")
        for metric in STAGE_METRICS:
            normal = _finite_number(row.get(f"normal_{metric}"), f"normal_{metric}")
            knockout = _finite_number(row.get(f"knockout_{metric}"), f"knockout_{metric}")
            delta = _finite_number(row.get(f"delta_{metric}"), f"delta_{metric}")
            relative_field = f"relative_change_pct_{metric}"
            if relative_field not in row:
                raise ValueError(f"missing paired field: {relative_field}")
            if normal < 0 or knockout < 0:
                raise ValueError(f"{metric} values must be non-negative")
            if not math.isclose(delta, knockout - normal, abs_tol=1e-12):
                raise ValueError(f"delta_{metric} must equal knockout minus normal")
            expected_relative = (
                delta / normal * 100.0 if normal > 0 else 0.0 if knockout == 0 else None
            )
            if expected_relative is None:
                if row[relative_field] is not None:
                    raise ValueError(f"{relative_field} must be None for zero baseline")
            elif not math.isclose(
                _finite_number(row[relative_field], relative_field),
                expected_relative,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{relative_field} must match knockout minus normal")
        by_model.setdefault(row["model"], []).append(row)

    if len(by_model) != 41:
        raise ValueError("paired rows must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in MATERIAL_GROUPS}
    for model, model_rows in by_model.items():
        if len(model_rows) != len(expected_conditions):
            raise ValueError(f"{model}: expected five paired conditions")
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
    return [dict(row) for row in rows]


def summarize_stage_paired_rows(rows, bootstrap_samples, bootstrap_seed):
    """Summarize B3a paired deltas by overall and material group."""
    paired_rows = _validate_stage_paired_rows(rows)
    summary_rows = []
    for group in SUMMARY_GROUPS:
        group_rows = (
            paired_rows
            if group == "overall"
            else [
                row
                for row in paired_rows
                if row["mat_type"] == next(
                    mat_type
                    for mat_type, material in MATERIAL_GROUPS.items()
                    if material == group
                )
            ]
        )
        for condition in _STAGE_CONDITION_NAMES:
            if condition == "normal":
                continue
            condition_rows = [row for row in group_rows if row["condition"] == condition]
            for metric in STAGE_METRICS:
                stats = paired_delta_summary(
                    [row[f"normal_{metric}"] for row in condition_rows],
                    [row[f"knockout_{metric}"] for row in condition_rows],
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                summary_rows.append(
                    {"group": group, "condition": condition, "metric": metric, **stats}
                )
    return summary_rows


def _validate_stage_activity_rows(rows):
    if not isinstance(rows, list) or len(rows) != 164:
        raise ValueError("activity rows must contain exactly 164 rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each activity row must be a dictionary")
    _validate_optional_provenance(rows)
    by_model = {}
    for row in rows:
        for field in (
            "model",
            "mat_type",
            "stage_index",
            "call_count",
            "delta_rms",
            "hidden_rms",
            "relative_rms",
        ):
            if field not in row:
                raise ValueError(f"missing activity field: {field}")
        model = row["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("activity model must be a non-empty string")
        mat_type = row["mat_type"]
        if isinstance(mat_type, bool) or mat_type not in MATERIAL_GROUPS:
            raise ValueError("activity mat_type must be 0, 1, or 2")
        stage_index = row["stage_index"]
        if isinstance(stage_index, bool) or not isinstance(stage_index, (int, np.integer)) or stage_index not in range(4):
            raise ValueError("activity stage_index must be 0 through 3")
        call_count = row["call_count"]
        if isinstance(call_count, bool) or not isinstance(call_count, (int, np.integer)) or call_count <= 0:
            raise ValueError("activity call_count must be a positive integer")
        for field in ("delta_rms", "hidden_rms", "relative_rms"):
            if _finite_number(row[field], field) < 0:
                raise ValueError(f"{field} must be non-negative")
        by_model.setdefault(model, []).append(row)

    if len(by_model) != 41:
        raise ValueError("activity rows must contain exactly 41 unique models")
    material_counts = {mat_type: 0 for mat_type in MATERIAL_GROUPS}
    for model, model_rows in by_model.items():
        if len(model_rows) != 4 or {row["stage_index"] for row in model_rows} != set(range(4)):
            raise ValueError(f"{model}: activity stages are incomplete or duplicated")
        mat_type = model_rows[0]["mat_type"]
        if any(row["mat_type"] != mat_type for row in model_rows[1:]):
            raise ValueError(f"{model}: activity mat_type must agree across stages")
        material_counts[mat_type] += 1
    if material_counts != {0: 13, 1: 14, 2: 14}:
        raise ValueError("activity material counts must be elastic=13, plasticine=14, sand=14")
    return [dict(row) for row in rows]


def _validate_stage_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    validated = {}
    for field in ("checkpoint", "config", "sample_scope"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata must contain a non-empty string: {field}")
        validated[field] = value
    for field, minimum in (("seed", 0), ("bootstrap_samples", 1), ("bootstrap_seed", 0)):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            qualifier = "positive" if minimum else "non-negative"
            raise ValueError(f"metadata {field} must be a {qualifier} integer")
        validated[field] = int(value)
    return validated


def _validate_stage_summary_rows(rows):
    expected = {
        (group, condition, metric)
        for group in SUMMARY_GROUPS
        for condition in _STAGE_CONDITION_NAMES
        if condition != "normal"
        for metric in STAGE_METRICS
    }
    if not isinstance(rows, list):
        raise ValueError("summary rows must be a list")
    index = {}
    required = {
        "group",
        "condition",
        "metric",
        "n_models",
        "normal_mean",
        "knockout_mean",
        "mean_delta",
        "median_delta",
        "relative_change_pct",
        "improved_count",
        "degraded_count",
        "ci_low",
        "ci_high",
    }
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("summary rows contain incomplete statistics")
        key = (row["group"], row["condition"], row["metric"])
        if key in index:
            raise ValueError("summary rows must not contain duplicates")
        index[key] = row
        for field in (
            "normal_mean",
            "knockout_mean",
            "mean_delta",
            "median_delta",
            "ci_low",
            "ci_high",
        ):
            _finite_number(row[field], field)
        relative = row["relative_change_pct"]
        if relative is not None:
            _finite_number(relative, "relative_change_pct")
    if set(index) != expected:
        raise ValueError("summary rows must contain the complete registered table")
    return index


def _stage_csv_value(value):
    return "" if value is None else value


def _write_stage_csv(path, rows, columns, sort_key):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({field: _stage_csv_value(row.get(field)) for field in columns})


def _report_number(value, *, signed=False, percent=False):
    if value is None:
        return "N/A"
    value = float(value)
    suffix = "%" if percent else ""
    if signed:
        return f"{value:+.6e}{suffix}" if not percent else f"{value:+.2f}{suffix}"
    return f"{value:.6e}{suffix}"


def _write_stage_report(path, raw_rows, activity_rows, summary_index, metadata):
    lines = [
        "# B3a Material-State Stage Knockout 诊断",
        "",
        "## 实验元数据",
        "",
        f"- checkpoint: `{metadata['checkpoint']}`",
        f"- config: `{metadata['config']}`",
        f"- seed: `{metadata['seed']}`",
        f"- sample_scope: {metadata['sample_scope']}",
        f"- bootstrap_samples: `{metadata['bootstrap_samples']}`",
        f"- bootstrap_seed: `{metadata['bootstrap_seed']}`",
        "",
        "## 完整性检查",
        "",
        f"- raw rows: {len(raw_rows)} / 246",
        f"- paired rows: 205 / 205",
        f"- activity rows: {len(activity_rows)} / 164",
        "- 差值定义为 `knockout - normal`；所有指标越低越好，负值表示改善。",
        "- `all_off` 是同一冻结 checkpoint 内的因果 knockout，不是独立训练的 baseline。",
        "",
    ]
    for group in SUMMARY_GROUPS:
        lines.extend(
            [
                f"## {group}",
                "",
                "| condition | metric | n | mean delta | relative change | paired delta 95% CI |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in _STAGE_CONDITION_NAMES:
            if condition == "normal":
                continue
            for metric in STAGE_METRICS:
                row = summary_index[(group, condition, metric)]
                lines.append(
                    "| {condition} | {metric} | {n_models} | {mean_delta} | {relative} | [{ci_low}, {ci_high}] |".format(
                        condition=condition,
                        metric=metric,
                        n_models=row["n_models"],
                        mean_delta=_report_number(row["mean_delta"], signed=True),
                        relative=_report_number(
                            row["relative_change_pct"], signed=True, percent=True
                        ),
                        ci_low=_report_number(row["ci_low"], signed=True),
                        ci_high=_report_number(row["ci_high"], signed=True),
                    )
                )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_stage_outputs(output_dir, raw_rows, paired_rows, activity_rows, summary_rows, metadata):
    """Validate the complete B3a table, then write its four registered outputs."""
    validated_metadata = _validate_stage_metadata(metadata)
    validated_raw = validate_stage_raw_rows(raw_rows)
    if _validate_optional_provenance(validated_raw) != {
        field: validated_metadata[field] for field in _PROVENANCE_FIELDS
    }:
        raise ValueError("raw provenance must match metadata")
    validated_paired = _validate_stage_paired_rows(paired_rows)
    if _validate_optional_provenance(validated_paired) != {
        field: validated_metadata[field] for field in _PROVENANCE_FIELDS
    }:
        raise ValueError("paired provenance must match metadata")
    if validated_paired != build_stage_paired_rows(validated_raw):
        raise ValueError("paired rows must exactly match the validated raw table")
    validated_activity = _validate_stage_activity_rows(activity_rows)
    if _validate_optional_provenance(validated_activity) != {
        field: validated_metadata[field] for field in _PROVENANCE_FIELDS
    }:
        raise ValueError("activity provenance must match metadata")
    summary_index = _validate_stage_summary_rows(summary_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "material_state_stage_knockout_b3a90_raw.csv"
    paired_path = output_dir / "material_state_stage_knockout_b3a90_paired.csv"
    activity_path = output_dir / "material_state_stage_activity_b3a90.csv"
    report_path = output_dir / "material_state_stage_knockout_b3a90.md"
    condition_order = {condition: index for index, condition in enumerate(_STAGE_CONDITION_NAMES)}
    raw_columns = (*_PROVENANCE_FIELDS, *_RAW_METADATA_FIELDS, "condition", *STAGE_METRICS)
    paired_columns = (
        *_PROVENANCE_FIELDS,
        *_RAW_METADATA_FIELDS,
        "condition",
        *(
            field
            for metric in STAGE_METRICS
            for field in (
                f"normal_{metric}",
                f"knockout_{metric}",
                f"delta_{metric}",
                f"relative_change_pct_{metric}",
            )
        ),
    )
    activity_columns = (
        *_PROVENANCE_FIELDS,
        "model",
        "mat_type",
        "stage_index",
        "call_count",
        "delta_rms",
        "hidden_rms",
        "relative_rms",
    )
    _write_stage_csv(raw_path, validated_raw, raw_columns, lambda row: (row["model"], condition_order[row["condition"]]))
    _write_stage_csv(paired_path, validated_paired, paired_columns, lambda row: (row["model"], condition_order[row["condition"]]))
    _write_stage_csv(activity_path, validated_activity, activity_columns, lambda row: (row["model"], row["stage_index"]))
    _write_stage_report(report_path, validated_raw, validated_activity, summary_index, validated_metadata)
    return {
        "raw": raw_path,
        "paired": paired_path,
        "activity": activity_path,
        "report": report_path,
    }


@contextmanager
def masked_material_state_stages(adapter, mask):
    scales = adapter.stage_scales
    original = scales.detach().clone()
    mask_tensor = torch.as_tensor(mask, device=scales.device, dtype=scales.dtype)
    if scales.shape != (4,) or mask_tensor.shape != scales.shape:
        raise ValueError("material-state stage mask must contain four values")
    if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
        raise ValueError("material-state stage mask must be binary")
    if not torch.isfinite(original).all():
        raise ValueError("checkpoint stage scales must be finite")

    with torch.no_grad():
        scales.copy_(original * mask_tensor)
    try:
        yield scales.detach().clone()
    finally:
        with torch.no_grad():
            scales.copy_(original)
        if not torch.equal(scales.detach(), original):
            raise RuntimeError("material-state stage scales were not restored exactly")


class MaterialStateActivityCollector:
    def __init__(self, adapter):
        self.adapter = adapter
        self._rows = []
        self._capture_state = None
        self._handle = None

    def __enter__(self):
        if self._handle is not None or self.adapter in _ACTIVE_COLLECTORS:
            raise ValueError("nested collector contexts are not supported")
        self._rows.clear()
        self._capture_state = None
        _ACTIVE_COLLECTORS[self.adapter] = self
        try:
            self._handle = self.adapter.register_forward_hook(
                self._capture_activity,
                with_kwargs=True,
            )
        except BaseException:
            del _ACTIVE_COLLECTORS[self.adapter]
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
        finally:
            if _ACTIVE_COLLECTORS.get(self.adapter) is self:
                del _ACTIVE_COLLECTORS[self.adapter]
            self._capture_state = None
        return False

    @contextmanager
    def capture(self, model_name, mat_type, expected_calls_per_stage):
        if self._handle is None:
            raise ValueError("activity collector must be active")
        if self._capture_state is not None:
            raise ValueError("nested model activity contexts are not supported")
        if (
            isinstance(expected_calls_per_stage, bool)
            or not isinstance(expected_calls_per_stage, int)
            or expected_calls_per_stage <= 0
        ):
            raise ValueError("expected_calls_per_stage must be a positive integer")

        num_stages = self._num_stages()
        state = {
            "model": model_name,
            "mat_type": mat_type,
            "expected_calls_per_stage": expected_calls_per_stage,
            "call_count": [0] * num_stages,
            "delta_sq_sum": [0.0] * num_stages,
            "hidden_sq_sum": [0.0] * num_stages,
            "numel": [0] * num_stages,
        }
        self._capture_state = state
        try:
            yield self
        except BaseException:
            self._capture_state = None
            raise
        else:
            try:
                self._append_rows(state)
            finally:
                self._capture_state = None

    def rows(self):
        return [dict(row) for row in self._rows]

    def _num_stages(self):
        scales = self.adapter.stage_scales
        if scales.ndim != 1:
            raise ValueError("adapter stage_scales must be one-dimensional")
        return int(scales.numel())

    def _capture_activity(self, module, positional_args, kwargs, output):
        state = self._capture_state
        if state is None:
            return
        if not positional_args:
            raise ValueError("material-state adapter hook requires hidden states")
        if "stage_index" not in kwargs:
            raise ValueError("material-state adapter forward must provide stage_index")

        hidden = positional_args[0]
        stage_index = int(kwargs["stage_index"])
        if not isinstance(hidden, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError("material-state adapter hook requires tensor input and output")
        if hidden.shape != output.shape:
            raise ValueError("material-state adapter input and output shapes must match")
        if not 0 <= stage_index < len(state["call_count"]):
            raise ValueError("material-state adapter stage_index is outside the configured range")

        hidden = hidden.detach().float()
        output = output.detach().float()
        if not torch.isfinite(hidden).all() or not torch.isfinite(output).all():
            raise ValueError("material-state adapter hook received nonfinite input or output")
        delta = output - hidden
        hidden_cpu = hidden.to(device="cpu", dtype=torch.float64)
        delta_cpu = delta.to(device="cpu", dtype=torch.float64)
        state["call_count"][stage_index] += 1
        state["delta_sq_sum"][stage_index] += float(delta_cpu.square().sum())
        state["hidden_sq_sum"][stage_index] += float(hidden_cpu.square().sum())
        state["numel"][stage_index] += delta_cpu.numel()

    def _append_rows(self, state):
        rows = []
        expected = state["expected_calls_per_stage"]
        for stage_index, call_count in enumerate(state["call_count"]):
            if call_count != expected:
                raise ValueError(
                    f"stage {stage_index} call count {call_count} does not match "
                    f"expected {expected}"
                )

            numel = state["numel"][stage_index]
            delta_rms = math.sqrt(state["delta_sq_sum"][stage_index] / numel)
            hidden_rms = math.sqrt(state["hidden_sq_sum"][stage_index] / numel)
            if hidden_rms == 0.0:
                raise ValueError("hidden RMS must be nonzero")
            rows.append(
                {
                    "model": state["model"],
                    "mat_type": state["mat_type"],
                    "stage_index": stage_index,
                    "call_count": call_count,
                    "delta_rms": delta_rms,
                    "hidden_rms": hidden_rms,
                    "relative_rms": delta_rms / hidden_rms,
                }
            )
        self._rows.extend(rows)
