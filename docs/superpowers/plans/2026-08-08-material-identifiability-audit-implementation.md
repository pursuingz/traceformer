# B0.2 Material Identifiability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个不加载模型、不使用 GPU 的 MM3 数据可辨识性审计，判断每种材质内部的 `E/nu` 是否具有充分覆盖、是否被几何/场景混杂，以及是否能解释 GT 动力学响应。

**Architecture:** CLI 逐个流式读取 train/test H5，把每个 object 压缩为一条静态/动力学记录；纯函数统计模块负责 coverage、support shift、交叉验证、置换、bootstrap、FDR 和最终分类；输出层生成结构化 CSV/JSON 与中文 Markdown 报告。Train 用于全部动力学分析，test 只允许静态 coverage，不生成动力学响应。

**Tech Stack:** Python 3.10、`numpy`、`h5py`、`scipy`、标准库 `csv/json/dataclasses/pathlib/unittest`；不新增依赖，不导入模型、pipeline、checkpoint、Torch 或 CUDA。

## Global Constraints

- 设计来源：`docs/superpowers/specs/2026-08-08-material-identifiability-audit-design.md`。
- 每个 H5 是唯一统计样本；禁止把帧或粒子当作独立 object 样本。
- `mm3_train` 可读取完整动力学；`mm3_test` 只读取静态字段和 `x[0]`。
- 所有连续参数分析必须分别在 elastic、plasticine、sand 内完成。
- 只有仿真开始前确定的量可进入 nuisance；未来接触属于 response。
- 接触阈值使用原始坐标 `0.08`。
- 固定默认 `seed=0`、5 folds、500 permutations、1000 bootstrap samples。
- 不覆盖既有输出，除非 CLI 显式传入 `--overwrite`。
- 不修改数据生成代码，不运行 B4 sweep，不训练新模型。
- commit 作者只能是 Will，不添加 AI co-author/contributor；不执行 `git push`。

---

## File Structure

### Create

- `src/utils/material_identifiability.py`
  - 数据列定义、H5 特征提取、统计、分类、输出构造和 Markdown 渲染。
- `src/diagnose_material_identifiability.py`
  - CLI、目录扫描、流式执行、进度和产物写入。
- `src/tests/test_material_identifiability.py`
  - synthetic H5、统计回归、输出 schema 和 CLI smoke tests。

### Modify

- `实验记录_1.md`
  - 注册 B0.2 目的、数据隔离、统计协议、成功/停止条件、代码 commit 和服务器命令。

### Must Not Modify

- `src/data_generation/generate_mpm_data.py`
- `src/model/**`
- `src/train.py`
- `src/eval.py`
- 任何 B3/B4 config 或 checkpoint 路径。

---

### Task 1: Object-Level H5 Record Extraction

**Files:**
- Create: `src/utils/material_identifiability.py`
- Create: `src/tests/test_material_identifiability.py`

**Interfaces:**
- Produces:
  - `AuditSettings`
  - `RecordValidationError`
  - `read_h5_record(path: Path, *, split: str, settings: AuditSettings) -> dict[str, object]`
  - `STATIC_COLUMNS`, `NUISANCE_COLUMNS`, `RESPONSE_COLUMNS`, `PRIMARY_RESPONSE_COLUMNS`
- Consumes: H5 fields defined in the design spec.

- [ ] **Step 1: Write synthetic-H5 helpers and failing static-record tests**

Add a `unittest.TestCase` helper that writes a deterministic cube point cloud:

```python
def write_h5(self, path: Path, *, frames: int = 25, include_dynamics: bool = True):
    cube = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=np.float32)
    x = np.stack([cube + np.asarray([0, -0.01 * t, 0]) for t in range(frames)])
    with h5py.File(path, "w") as handle:
        handle["x"] = x
        handle["vol"] = np.ones(len(cube), dtype=np.float32) / len(cube)
        handle["E"] = np.asarray(1e6)
        handle["nu"] = np.asarray(0.3)
        handle["mat_type"] = np.asarray(1)
        handle["gravity"] = np.asarray(1)
        handle["floor_height"] = np.asarray(-0.1)
        handle["drag_force"] = np.zeros((0, 3), dtype=np.float32)
        handle["drag_mask"] = np.zeros((0, len(cube)), dtype=np.float32)
        if include_dynamics:
            handle["v"] = np.zeros_like(x)
            eye = np.eye(3, dtype=np.float32)
            handle["F"] = np.broadcast_to(eye, (frames, len(cube), 3, 3))
            handle["C"] = np.zeros((frames, len(cube), 3, 3), dtype=np.float32)
```

Add shared test lookup helpers so later tasks use exact row-selection semantics:

```python
def find_row(rows, **criteria):
    matches = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]

def serialise_folds(folds):
    return [(train.tolist(), test.tolist()) for train, test in folds]
```

Required assertions:

```python
test_path = self.root / "test_static.h5"
self.write_h5(test_path, frames=1, include_dynamics=False)
record = read_h5_record(test_path, split="test", settings=AuditSettings())
self.assertEqual(record["split"], "test")
self.assertNotIn("centered_shape_mse_f24", record)
self.assertNotIn("future_contact_fraction", NUISANCE_COLUMNS)

train_path = self.root / "train_dynamic.h5"
self.write_h5(train_path, frames=25, include_dynamics=True)
record = read_h5_record(train_path, split="train", settings=AuditSettings())
self.assertEqual(record["model"], "train_dynamic.h5")
self.assertAlmostEqual(record["log10_e"], 6.0)
self.assertIn("centered_shape_mse_f24", record)
self.assertIn("future_contact_fraction", RESPONSE_COLUMNS)
```

The first test must create a one-frame H5 without `v/F/C`; it must succeed for `split="test"`. This is the regression guard that test dynamics are never read.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `src/`:

```bash
python -m unittest \
  tests.test_material_identifiability.MaterialRecordTests -v
```

Expected: import failure because `utils.material_identifiability` does not exist.

- [ ] **Step 3: Implement record constants, validation, and static features**

Start the utility module with exact public definitions:

```python
from dataclasses import dataclass
from pathlib import Path

MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}
FRAME_INDICES = (5, 10, 15, 20, 24)

@dataclass(frozen=True)
class AuditSettings:
    seed: int = 0
    folds: int = 5
    permutations: int = 500
    bootstrap_samples: int = 1000
    contact_band_raw: float = 0.08

class RecordValidationError(ValueError):
    pass
```

`read_h5_record` must:

1. validate `split in {"train", "test"}`;
2. validate finite positive `E`, finite `nu`, and `mat_type in {0,1,2}`;
3. read only `x[0]` for test;
4. calculate centroid, extents, covariance eigenvalues, radius of gyration, convex-hull volume, total particle volume, floor gap, gravity, drag magnitude/count/mask ratio and initial contact fraction;
5. leave every response column absent for test records;
6. attach `model`, `split`, `material`, `valid=True`.

Use a strict hull helper:

```python
def _convex_hull_volume(points: np.ndarray) -> float:
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return float("nan")
```

Do not replace hull failure with covariance volume.

- [ ] **Step 4: Implement train-only GT responses**

For train records require `x/v/F/C` with at least 25 frames and aligned particle dimensions. Add:

```python
centered_shape_mse_f24
centroid_displacement_f24
velocity_rms_trajectory
f_strain_norm_f24
volumetric_strain_f24
```

and all secondary responses from the spec. Contact onset and future contact fractions belong only to `RESPONSE_COLUMNS`; assert they are absent from `NUISANCE_COLUMNS`.

The F metrics use:

```python
identity = np.eye(3, dtype=F.dtype)
f_strain = np.linalg.norm(F - identity, axis=(-2, -1))
j_error = np.abs(np.linalg.det(F) - 1.0)
```

- [ ] **Step 5: Add explicit invalid-record tests**

Required cases and assertions:

```python
with self.assertRaisesRegex(RecordValidationError, "missing.*F"):
    read_h5_record(path_without_f, split="train", settings=settings)
with self.assertRaisesRegex(RecordValidationError, "at least 25 frames"):
    read_h5_record(short_path, split="train", settings=settings)
with self.assertRaisesRegex(RecordValidationError, "finite.*E"):
    read_h5_record(nonfinite_e_path, split="train", settings=settings)
with self.assertRaisesRegex(RecordValidationError, "particle dimension"):
    read_h5_record(misaligned_path, split="train", settings=settings)

record = read_h5_record(coplanar_path, split="train", settings=settings)
self.assertTrue(np.isnan(record["initial_hull_volume"]))
self.assertTrue(record["valid"])
```

Missing mandatory physics fields must raise `RecordValidationError`. A degenerate hull is a recorded missing descriptor, not a reason to discard all other valid fields.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

```bash
python -m unittest \
  tests.test_material_identifiability.MaterialRecordTests -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/utils/material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Add material identifiability record extraction"
```

---

### Task 2: Parameter Coverage and Train/Test Support

**Files:**
- Modify: `src/utils/material_identifiability.py`
- Modify: `src/tests/test_material_identifiability.py`

**Interfaces:**
- Consumes: object records from Task 1.
- Produces:
  - `build_coverage_rows(train_records, test_records, *, bins=5) -> list[dict]`
  - `build_support_rows(train_records, test_records, *, bins=5) -> list[dict]`

- [ ] **Step 1: Write failing coverage tests**

Create deterministic records for all three materials. Required assertions:

```python
coverage = build_coverage_rows(train_records, test_records, bins=5)
elastic_e = find_row(coverage, split="train", material="elastic", parameter="log10_e")
self.assertEqual(elastic_e["unique_n"], len(set(elastic_train_e)))
self.assertAlmostEqual(elastic_e["p50"], float(np.median(elastic_train_e)))
self.assertGreaterEqual(elastic_e["joint_grid_occupancy"], 0.0)
self.assertLessEqual(elastic_e["joint_grid_occupancy"], 1.0)

support = build_support_rows(train_records, test_records, bins=5)
elastic_support = find_row(support, material="elastic", parameter="log10_e")
self.assertAlmostEqual(elastic_support["outside_train_fraction"], 2.0 / 3.0)
self.assertIn("ks_statistic", elastic_support)
self.assertIn("wasserstein_distance", elastic_support)
self.assertIn("mahalanobis_outside_fraction", elastic_support)
self.assertTrue(all(name not in elastic_support for name in RESPONSE_COLUMNS))
```

For the extrapolation test, train `log10_e=[4,5,6]` and test `log10_e=[3,5,7]` must report an out-of-range fraction of `2/3`.

- [ ] **Step 2: Run Task 2 tests and verify RED**

```bash
python -m unittest \
  tests.test_material_identifiability.CoverageTests -v
```

Expected: missing `build_coverage_rows`/`build_support_rows`.

- [ ] **Step 3: Implement distribution rows**

For each `split x material x parameter`, emit:

```text
n, unique_n, min, p05, p25, mean, std, p50, p75, p95, max,
pearson_e_nu, spearman_e_nu, joint_grid_occupancy
```

Use continuous `log10_e` and `nu`. Grid edges are fixed to generation ranges `[4,7]` and `[0.05,0.45]`, not fitted to test.

- [ ] **Step 4: Implement support rows**

For each material and parameter, report:

```text
train_min, train_max, test_min, test_max, outside_train_fraction,
ks_statistic, ks_pvalue, wasserstein_distance
```

Also report joint empty-bin fraction, standardized mean differences for static nuisance columns, and a material-local joint Mahalanobis diagnostic. Remove train-constant columns, standardize from train only, regularize covariance as `cov + 1e-6 * I`, and report the fraction of test distances above the train 95th percentile as `mahalanobis_outside_fraction`. `support_status="out_of_support"` if any parameter has more than 5% out-of-range test samples, more than 20% of test samples fall in train-empty joint bins, or more than 20% exceed the train Mahalanobis 95th-percentile threshold; otherwise `in_support`.

- [ ] **Step 5: Run Task 1-2 tests**

```bash
python -m unittest \
  tests.test_material_identifiability.MaterialRecordTests \
  tests.test_material_identifiability.CoverageTests -v
```

Expected: all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/utils/material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Add material parameter coverage audit"
```

---

### Task 3: Confounding and GT Response Statistics

**Files:**
- Modify: `src/utils/material_identifiability.py`
- Modify: `src/tests/test_material_identifiability.py`

**Interfaces:**
- Produces:
  - `make_object_folds(model_names, *, folds, seed) -> list[tuple[np.ndarray, np.ndarray]]`
  - `benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray`
  - `analyze_confounding(records, settings) -> list[dict]`
  - `analyze_responses(records, settings) -> list[dict]`
- Internal:
  - `_piecewise_basis(train_values: np.ndarray, eval_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `_nested_cv_predictions(nuisance: np.ndarray, parameters: dict[str, np.ndarray], response: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], *, augmented_parameter: str | None) -> np.ndarray`
  - `_permutation_pvalue(observed: float, null_values: np.ndarray) -> float`
  - `_bootstrap_delta_r2(base_prediction: np.ndarray, augmented_prediction: np.ndarray, response: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]`

- [ ] **Step 1: Write failing fold and basis tests**

Required assertions:

```python
folds_a = make_object_folds(model_names, folds=5, seed=0)
folds_b = make_object_folds(model_names, folds=5, seed=0)
self.assertEqual(serialise_folds(folds_a), serialise_folds(folds_b))
self.assertEqual(sorted(np.concatenate([test for _, test in folds_a]).tolist()), list(range(len(model_names))))
for train, test in folds_a:
    self.assertTrue(set(train).isdisjoint(set(test)))

train_basis, eval_basis = _piecewise_basis(
    np.asarray([0.0, 1.0, 2.0, 3.0]), np.asarray([100.0])
)
self.assertEqual(train_basis.shape[1], 4)
self.assertEqual(eval_basis.shape[1], 4)

q = benjamini_hochberg(np.asarray([0.01, 0.04, 0.03, 0.002]))
self.assertTrue(np.all((q >= 0.0) & (q <= 1.0)))
self.assertEqual(q.shape, (4,))
```

For constant nuisance handling, pass one varying and one constant column; assert the returned fitted feature list contains the varying name and excludes the constant name.

Use the known FDR example `p=[0.01, 0.04, 0.03, 0.002]`; assert returned q-values preserve original order and lie in `[0,1]`.

- [ ] **Step 2: Run statistical helper tests and verify RED**

```bash
python -m unittest \
  tests.test_material_identifiability.StatisticsHelperTests -v
```

- [ ] **Step 3: Implement deterministic folds and ridge primitives**

Requirements:

- sort model names before seeded permutation;
- no object appears in both train and held-out indices;
- standardization mean/std comes from train fold only;
- zero-variance columns are removed using train fold only;
- missing nuisance values are replaced by the train-fold median and accompanied by a train-fold-derived missingness indicator; a column missing for the entire train fold is removed;
- inner alpha grid is exactly `10.0 ** np.arange(-4, 4)`;
- held-out `R²` uses pooled out-of-fold predictions, not mean fold `R²`.

The piecewise basis is:

```python
z_train = standardize(train_values)
z_eval = apply_train_standardization(eval_values)
knots = np.quantile(z_train, [0.25, 0.50, 0.75])
basis = [z, relu(z-knot_1), relu(z-knot_2), relu(z-knot_3)]
```

- [ ] **Step 4: Write failing synthetic identifiability tests**

Create at least 180 records per synthetic material with fixed RNG:

```python
nuisance = rng.normal(size=(n, 4))
log10_e = rng.uniform(4, 7, size=n)
nu = rng.uniform(0.05, 0.45, size=n)
strong_response = 2.0 * log10_e + 0.2 * nuisance[:, 0] + rng.normal(0, 0.1, n)
null_response = nuisance[:, 0] + rng.normal(0, 1.0, n)
confounded_e = 5.5 + 0.8 * nuisance[:, 0] + rng.normal(0, 0.05, n)
```

Required assertions:

```python
strong_rows = analyze_responses(strong_records, test_settings)
strong = find_row(
    strong_rows, material="elastic", parameter="log10_e",
    response="centered_shape_mse_f24",
)
self.assertGreater(strong["delta_r2"], 0.05)
self.assertLessEqual(strong["permutation_p"], 0.05)
self.assertGreater(strong["bootstrap_ci_low"], 0.0)

null_rows = analyze_responses(null_records, test_settings)
null = find_row(
    null_rows, material="elastic", parameter="nu",
    response="centered_shape_mse_f24",
)
self.assertLess(null["delta_r2"], 0.05)

confounding = analyze_confounding(confounded_records, test_settings)
summary = find_row(
    confounding, row_type="summary", material="elastic", parameter="log10_e"
)
self.assertTrue(summary["confounded"])

self.assertEqual(
    analyze_responses(strong_records, test_settings),
    analyze_responses(strong_records, test_settings),
)
self.assertNotIn("future_contact_fraction", NUISANCE_COLUMNS)
```

Tests use reduced `permutations=20` and `bootstrap_samples=40`; construct a large signal so the tests are deterministic rather than relying on marginal p-values.

- [ ] **Step 5: Implement confounding analysis**

For each material and each parameter:

1. compute per-nuisance Pearson/Spearman rows;
2. predict the parameter from all nonconstant nuisance with object-level CV;
3. compare CV `R²` against material-local parameter permutations;
4. emit `confounded = (cv_r2 > 0.05 and permutation_p < 0.05)`.

All rows include `material`, `parameter`, `n`, `seed`, `folds`, and explicit feature names.

- [ ] **Step 6: Implement nested response models**

For each material and response, fit `M0/ME/Mnu/Mboth` on identical folds. Emit:

```text
material, parameter, response, response_tier, n,
r2_m0, r2_augmented, delta_r2,
partial_spearman, permutation_p,
bootstrap_ci_low, bootstrap_ci_high, q_value
```

Statistical details are fixed:

- `delta_r2 = pooled_oof_r2_augmented - pooled_oof_r2_m0`;
- if the held-out response total sum of squares is below `1e-12`, emit `status=constant_response` and do not classify it as evidence;
- partial Spearman uses two cross-fitted residual vectors: residualize the parameter from nuisance and the response from nuisance using train-fold-only fits, then compute Spearman on pooled held-out residuals;
- a parameter permutation is performed within material before rebuilding its basis; nuisance, response, folds and model names remain fixed;
- permutation p-value is `(1 + count(null_delta >= observed_delta)) / (1 + permutations)`, so it is never zero;
- bootstrap resamples object-level pooled out-of-fold tuples with replacement using deterministic derived seeds and recomputes `delta_r2`; it does not silently refit on individual frames or particles;
- bootstrap is stratified implicitly because every analysis call already contains exactly one material group.

Compute FDR independently for each `material x parameter` family across responses. Secondary responses receive full statistics but cannot independently trigger `identifiable`.

- [ ] **Step 7: Run Task 3 tests and the accumulated module**

```bash
python -m unittest \
  tests.test_material_identifiability.StatisticsHelperTests \
  tests.test_material_identifiability.IdentifiabilityStatisticsTests -v
```

Then:

```bash
python -m unittest tests.test_material_identifiability -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/utils/material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Add material identifiability statistics"
```

---

### Task 4: Classification and Structured Outputs

**Files:**
- Modify: `src/utils/material_identifiability.py`
- Modify: `src/tests/test_material_identifiability.py`

**Interfaces:**
- Produces:
  - `classify_identifiability(response_rows, confounding_rows, support_rows) -> list[dict]`
  - `write_audit_outputs(output_dir, *, records, coverage_rows, support_rows, confounding_rows, response_rows, summary_rows, metadata, overwrite) -> dict[str, Path]`
  - `render_markdown_report(summary_rows: list[dict], coverage_rows: list[dict], support_rows: list[dict], confounding_rows: list[dict], response_rows: list[dict], metadata: dict) -> str`

- [ ] **Step 1: Write failing classification boundary tests**

Required exact assertions:

```python
summary = classify_identifiability(identifiable_rows, clean_confounding, in_support)
self.assertEqual(find_row(summary, material="elastic", parameter="log10_e")["status"], "identifiable")

summary = classify_identifiability(secondary_only_rows, clean_confounding, in_support)
self.assertNotEqual(find_row(summary, material="elastic", parameter="log10_e")["status"], "identifiable")

summary = classify_identifiability(weak_rows, clean_confounding, in_support)
self.assertEqual(find_row(summary, material="elastic", parameter="log10_e")["status"], "weak")

summary = classify_identifiability(null_rows, clean_confounding, in_support)
self.assertEqual(find_row(summary, material="elastic", parameter="log10_e")["status"], "not_detected")

summary = classify_identifiability(identifiable_rows, confounded_rows, out_of_support)
row = find_row(summary, material="elastic", parameter="log10_e")
self.assertEqual(row["status"], "confounded")
self.assertEqual(row["support_status"], "out_of_support")
```

The `identifiable` fixture must satisfy:

```text
response_tier=primary, delta_r2=0.05, permutation_p=0.049,
q_value=0.049, bootstrap_ci_low>0, confounded=False
```

- [ ] **Step 2: Run classification tests and verify RED**

```bash
python -m unittest \
  tests.test_material_identifiability.ClassificationTests -v
```

- [ ] **Step 3: Implement classification precedence**

For each `material x parameter`:

```text
if confounded:
    status = confounded
elif any qualifying primary response:
    status = identifiable
elif any response has delta_R2 >= 0.01 or partial significance:
    status = weak
else:
    status = not_detected
```

Always copy `support_status` as a separate column. Include machine-readable `reason_codes`, e.g. `primary_delta_r2`, `nuisance_predictable`, `test_parameter_extrapolation`.

- [ ] **Step 4: Write failing output tests**

Required assertions:

```python
paths = write_audit_outputs(output_dir, overwrite=False, **payload)
self.assertEqual(set(paths), set(OUTPUT_NAMES))
self.assertTrue(all(path.exists() for path in paths.values()))

with self.assertRaises(FileExistsError):
    write_audit_outputs(output_dir, overwrite=False, **payload)

metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
self.assertEqual(len(metadata["invalid_records"]), 1)
self.assertEqual(metadata["invalid_records"][0]["path"], "bad.h5")
self.assertEqual(metadata["seed"], 0)

with paths["records"].open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
test_row = next(row for row in rows if row["split"] == "test")
self.assertEqual(test_row["centered_shape_mse_f24"], "")

report = paths["report"].read_text(encoding="utf-8")
self.assertIn("np.random.randint(0, 1)", report)
self.assertIn("elastic", report)
self.assertIn("plasticine", report)
self.assertIn("sand", report)
self.assertIn("不能证明反事实物理正确", report)
```

- [ ] **Step 5: Implement CSV/JSON/Markdown writers**

Target names are fixed:

```python
OUTPUT_NAMES = {
    "records": "material_identifiability_records.csv",
    "coverage": "material_identifiability_coverage.csv",
    "confounding": "material_identifiability_confounding.csv",
    "response": "material_identifiability_response.csv",
    "summary": "material_identifiability_summary.csv",
    "metadata": "material_identifiability_metadata.json",
    "report": "material_identifiability_b02.md",
}
```

Before writing, check all seven target paths. If any exists and `overwrite=False`, raise `FileExistsError` before writing any file. Write JSON with UTF-8 and `ensure_ascii=False`; write CSV with stable explicit column order.

`coverage.csv` contains both distribution and support rows, distinguished by `row_type=distribution/support`; `support_rows` must not create an eighth file. Render all seven files completely in a temporary sibling directory first, then move them into the final output directory only after every render succeeds. A render exception must leave the final target paths unchanged.

- [ ] **Step 6: Run output and complete utility tests**

```bash
python -m unittest \
  tests.test_material_identifiability.ClassificationTests \
  tests.test_material_identifiability.OutputTests -v
```

Then:

```bash
python -m unittest tests.test_material_identifiability -v
```

- [ ] **Step 7: Commit Task 4**

```bash
git add src/utils/material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Add material identifiability reporting"
```

---

### Task 5: CLI Orchestration and End-to-End Smoke Test

**Files:**
- Create: `src/diagnose_material_identifiability.py`
- Modify: `src/tests/test_material_identifiability.py`

**Interfaces:**
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `run_material_identifiability_audit(train_dir: Path, test_dir: Path, output_dir: Path, settings: AuditSettings, *, overwrite: bool = False) -> dict[str, Path]`

- [ ] **Step 1: Write failing parser and orchestration tests**

Required assertions:

```python
args = build_parser().parse_args([
    "--train-dir", str(train_dir), "--test-dir", str(test_dir),
])
self.assertEqual(args.seed, 0)
self.assertEqual(args.folds, 5)
self.assertEqual(args.permutations, 500)
self.assertEqual(args.bootstrap_samples, 1000)
self.assertEqual(args.contact_band_raw, 0.08)

paths = run_material_identifiability_audit(
    train_dir, test_dir, output_dir,
    AuditSettings(seed=0, folds=2, permutations=2, bootstrap_samples=4),
)
self.assertEqual(set(paths), set(OUTPUT_NAMES))
metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
self.assertEqual(metadata["train_valid_count"], 18)
self.assertEqual(metadata["test_valid_count"], 6)
self.assertEqual(len(metadata["invalid_records"]), 1)
self.assertTrue(metadata["invalid_records"][0]["path"].endswith("bad.h5"))
```

Create a separate fixture with no sand train records and assert `run_material_identifiability_audit` raises `ValueError` naming `sand` before statistical fitting.

The smoke fixture creates at least 6 train objects and 2 test objects per material, then uses `folds=2`, `permutations=2`, `bootstrap_samples=4`. It verifies plumbing and schema, not statistical power.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
python -m unittest \
  tests.test_material_identifiability.CliTests -v
```

- [ ] **Step 3: Implement CLI parser**

Exact arguments:

```python
parser.add_argument("--train-dir", type=Path, required=True)
parser.add_argument("--test-dir", type=Path, required=True)
parser.add_argument(
    "--output-dir", type=Path,
    default=Path("results/material_identifiability_b02"),
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--permutations", type=int, default=500)
parser.add_argument("--bootstrap-samples", type=int, default=1000)
parser.add_argument("--contact-band-raw", type=float, default=0.08)
parser.add_argument("--overwrite", action="store_true")
```

Reject negative seed, fewer than 2 folds, nonpositive permutation/bootstrap counts, and nonpositive contact band.

- [ ] **Step 4: Implement stream orchestration**

The runner must:

1. resolve and validate both directories;
2. sort `*.h5` paths;
3. process train with dynamics and test statically;
4. catch only `RecordValidationError`, append `{path, split, error}` to `invalid_records`, and continue;
5. abort before statistics if any material has fewer records than `folds`;
6. build coverage, support, confounding, response and summary rows;
7. write all seven outputs atomically after computation;
8. print progress as `train i/n`, `test i/n`, and each statistical stage.

Unexpected I/O/programming exceptions must propagate rather than being mislabeled as invalid data.

- [ ] **Step 5: Run CLI and full module tests**

```bash
python -m unittest \
  tests.test_material_identifiability.CliTests -v
```

```bash
python -m unittest tests.test_material_identifiability -v
```

- [ ] **Step 6: Run syntax validation**

```bash
python -m py_compile \
  diagnose_material_identifiability.py \
  utils/material_identifiability.py \
  tests/test_material_identifiability.py
```

- [ ] **Step 7: Commit Task 5**

```bash
git add src/diagnose_material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Add B0.2 material identifiability CLI"
```

---

### Task 6: Register B0.2 and Perform Final Verification

**Files:**
- Modify: `实验记录_1.md`
- Verify: all Task 1-5 files.

**Interfaces:**
- Consumes: final CLI names, output schema and implementation commit hashes.
- Produces: formal B0.2 pre-registration and reproducible server command.

- [ ] **Step 1: Add the B0.2 overview row and detailed section**

Register:

```text
名称：B0.2 Material Identifiability Audit
服务目标：B / C / D
状态：实现完成，等待服务器数据审计
数据：mm3_train 完整响应；mm3_test 仅静态 coverage
训练：无
checkpoint：无
```

The detailed entry must copy the frozen thresholds and explicitly state:

- future contact is a response, not nuisance;
- object is the statistical unit;
- test GT dynamics are forbidden;
- no B5 training is registered before the report is interpreted;
- `np.random.randint(0,1)` is recorded as a protocol fact but not fixed here.

- [ ] **Step 2: Add the exact server command**

From `src/`:

```bash
python diagnose_material_identifiability.py \
  --train-dir mm3_data/mm3_train \
  --test-dir mm3_data/mm3_test \
  --output-dir results/material_identifiability_b02 \
  --seed 0 \
  --folds 5 \
  --permutations 500 \
  --bootstrap-samples 1000 \
  --contact-band-raw 0.08
```

- [ ] **Step 3: Run the project-selected test intensity**

Always run the focused suite:

```bash
python -m unittest tests.test_material_identifiability -v
```

For medium or above:

```bash
python -m py_compile \
  diagnose_material_identifiability.py \
  utils/material_identifiability.py \
  tests/test_material_identifiability.py
```

For high or extreme, additionally run all material diagnostics tests:

```powershell
$modules = Get-ChildItem tests\test_*material*.py |
  ForEach-Object { 'tests.' + $_.BaseName }
python -m unittest $modules
```

- [ ] **Step 4: Run static repository checks**

From repository root:

```bash
git diff --check
git status --short
```

Confirm no checkpoint, CSV, result directory, `CLAUDE.md`, unrelated user file, or AI attribution is staged.

- [ ] **Step 5: Independent review**

Review against the design checklist:

```text
[ ] test never reads dynamics
[ ] object-level folds only
[ ] future contact excluded from nuisance
[ ] train-only statistics
[ ] no model/CUDA imports
[ ] fixed seeds/folds/permutations/bootstrap
[ ] primary vs secondary response enforced
[ ] support shift separate from confounding
[ ] output refusal is atomic
[ ] exact seven-file schema
```

Fix genuine findings and rerun the affected tests before committing.

- [ ] **Step 6: Commit registration and final review fixes**

```bash
git add 实验记录_1.md \
        src/diagnose_material_identifiability.py \
        src/utils/material_identifiability.py \
        src/tests/test_material_identifiability.py
git commit -m "Register B0.2 material identifiability audit"
```

If no code changed after the prior commit, stage and commit only `实验记录_1.md`.

Do not push. Do not delete the implementation worktree or branch without Will's explicit approval.
