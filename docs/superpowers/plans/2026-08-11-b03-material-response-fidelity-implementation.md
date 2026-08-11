# B0.3 预测-GT 材料响应保真度审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个不训练模型的 B0.3 诊断工具，在冻结 41-model test 上比较 `mm3_contact_cond@90k` 预测轨迹与 GT 的位置可观测物理响应，并定位 `E/nu` 响应是保留、衰减还是反向。

**Architecture:** `utils/material_response_fidelity.py` 只负责位置响应、统计、校验和六文件输出，不导入模型或 CUDA；`diagnose_material_response_fidelity.py` 复用现有 B0/B2 的 dataset、strict checkpoint、GT reference 和 factual rollout 路径。正式 `eval.py`、模型结构、训练配置和 checkpoint 均不修改。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、OmegaConf、safetensors、`unittest`、现有 `TrajDataset`/`TrajPipeline`。

## Global Constraints

- 测试强度固定为“高”：相关测试模块全量、问题回归、`py_compile`、`git diff --check` 和一次独立代码审查。
- 设计来源：`docs/superpowers/specs/2026-08-11-b03-material-response-fidelity-design.md`。
- 只使用预测与 GT 都能由 `(25,N,3)` 位置轨迹同口径计算的 response；不得把凸包体积称为 `det(F)`，不得伪造预测侧 `F/C`。
- 固定 checkpoint/config/split：`mm3_contact_cond@90k`、`configs/eval_mm3_contact_cond.yaml`、41-model test、13/14/14、`start_idx=0`、seed 0。
- 推理只跑 factual condition，每个 model 一次，共 41 次；不保存 `.npy` 轨迹。
- 新增报告和设计文档使用中文；代码标识符使用英文。
- 新输出默认拒绝覆盖，只有显式 `--overwrite` 才替换完整六文件输出。
- commit author 使用 Will，不添加任何 AI co-author/contributor。

---

## File Structure

- Create `src/utils/material_response_fidelity.py`：纯 CPU 响应公式、bootstrap、fidelity/alignment 汇总、schema 校验、事务式输出和中文报告。
- Create `src/diagnose_material_response_fidelity.py`：CLI、checkpoint/config identity、41-model factual rollout 和结果收集。
- Create `src/tests/test_material_response_fidelity.py`：上述两个模块的 TDD 单元与 fake-runtime 集成测试。
- Modify `实验记录_1.md`：实现完成后登记 B0.3 设计/实现 commit、命令和“等待服务器运行”状态；合并到 main 时与 main 上已有 B0.2 记录人工整合。
- Modify `研究目标与路线图.md`：只在 B0.3 实现完成后把“下一步”标记为已注册/待运行，不预写结果。

---

### Task 1: 冻结位置响应 schema 与公式

**Files:**
- Create: `src/utils/material_response_fidelity.py`
- Create: `src/tests/test_material_response_fidelity.py`

**Interfaces:**
- Produces `PRIMARY_RESPONSES: tuple[str, ...]`、`SECONDARY_RESPONSES: tuple[str, ...]`、`RESPONSE_NAMES`。
- Produces `extract_position_responses(trajectory: Any, floor_height: float, contact_band_raw: float = 0.08) -> dict[str, float]`。
- All later tasks consume this exact response dictionary.

- [ ] **Step 1: 写轨迹校验和公式的失败测试**

在 `src/tests/test_material_response_fidelity.py` 创建 `PositionResponseTests`，构造一个非共面的 8 点长方体轨迹，明确验证：

```python
class PositionResponseTests(unittest.TestCase):
    def test_extract_position_responses_matches_closed_form_motion_and_volume(self):
        trajectory = translating_scaling_box(frames=25, scale_step=0.01, dy=-0.02)
        result = extract_position_responses(
            trajectory,
            floor_height=-10.0,
            contact_band_raw=0.08,
        )
        expected_velocity = np.sqrt(np.mean(np.diff(trajectory, axis=0) ** 2))
        expected_acceleration = np.sqrt(
            np.mean(np.diff(trajectory, n=2, axis=0) ** 2)
        )
        self.assertAlmostEqual(result["position_velocity_rms_trajectory"], expected_velocity)
        self.assertAlmostEqual(result["position_acceleration_rms_trajectory"], expected_acceleration)
        self.assertAlmostEqual(result["centroid_displacement_f24"], 0.48)
        self.assertGreater(result["hull_volume_relative_change_f24"], 0.0)
        self.assertEqual(set(result), set(RESPONSE_NAMES))

    def test_extract_position_responses_rejects_wrong_frames_and_degenerate_hull(self):
        with self.assertRaisesRegex(ValueError, "25 frames"):
            extract_position_responses(np.zeros((24, 8, 3)), -2.0)
        with self.assertRaisesRegex(ValueError, "convex hull"):
            extract_position_responses(np.zeros((25, 8, 3)), -2.0)
```

同时覆盖 non-finite、少于 4 点、负 `contact_band_raw`、contact fraction 和 XYZ extent 符号。

- [ ] **Step 2: 运行测试并确认 RED**

Run from `src/`:

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity.PositionResponseTests -v
```

Expected: import 或符号不存在导致 FAIL；不是测试 fixture 自身报错。

- [ ] **Step 3: 实现最小响应提取器**

实现冻结 schema：

```python
PRIMARY_RESPONSES = (
    "position_velocity_rms_trajectory",
    "position_acceleration_rms_trajectory",
    "centroid_displacement_f24",
    "centered_shape_mse_f24",
    "hull_volume_relative_change_f24",
    "hull_volume_relative_change_trajectory",
)
SECONDARY_RESPONSES = (
    "extent_change_x_f24",
    "extent_change_y_f24",
    "extent_change_z_f24",
    "future_contact_fraction",
)
RESPONSE_NAMES = (*PRIMARY_RESPONSES, *SECONDARY_RESPONSES)
```

`extract_position_responses` 必须：转成 CPU `float64` NumPy；严格要求 `(25,N,3)`、`N>=4` 和 finite；使用 `scipy.spatial.ConvexHull` 计算每帧体积；以 frame 0 为响应锚；contact 使用帧 1--24；返回值全部 finite。

- [ ] **Step 4: 运行 PositionResponseTests 并确认 GREEN**

Run: 上一步同一命令。Expected: all PASS。

- [ ] **Step 5: Commit Task 1**

```bash
git add src/utils/material_response_fidelity.py src/tests/test_material_response_fidelity.py
git commit -m "Add B0.3 position response metrics"
```

---

### Task 2: Factual fidelity 与材料响应 alignment 统计

**Files:**
- Modify: `src/utils/material_response_fidelity.py`
- Modify: `src/tests/test_material_response_fidelity.py`

**Interfaces:**
- Produces `build_response_rows(model_row: dict[str, Any], gt: Any, pred: Any, floor_height: float, contact_band_raw: float) -> list[dict[str, Any]]`。
- Produces `build_fidelity_summary(response_rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]`。
- Produces `build_alignment_summary(response_rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]`。
- Produces `partial_spearman(x: Any, y: Any, control: Any) -> float | None` and `classify_alignment(gt_rho, pred_rho) -> str`。

- [ ] **Step 1: 写 long-form rows 和统计失败测试**

增加以下测试类：

```python
class ResponseStatisticsTests(unittest.TestCase):
    def test_build_response_rows_pairs_identical_schema(self):
        rows = build_response_rows(
            model_row=fake_model_row("elastic_00.h5", mat_type=0, log10_e=5.0, nu=0.2),
            gt=synthetic_trajectory(amplitude=2.0),
            pred=synthetic_trajectory(amplitude=1.0),
            floor_height=-2.0,
            contact_band_raw=0.08,
        )
        self.assertEqual(len(rows), len(RESPONSE_NAMES))
        self.assertTrue(all(row["absolute_error"] >= 0 for row in rows))
        self.assertTrue(all(row["signed_error"] == row["pred_value"] - row["gt_value"] for row in rows))

    def test_partial_spearman_removes_other_parameter_rank_effect(self):
        control = np.arange(20, dtype=float)
        x = control + np.tile([0.0, 1.0], 10)
        y = control.copy()
        rho = partial_spearman(x, y, control)
        self.assertLess(abs(rho), 0.2)

    def test_alignment_labels_are_frozen_at_boundaries(self):
        self.assertEqual(classify_alignment(-0.60, -0.40), "aligned")
        self.assertEqual(classify_alignment(-0.60, -0.20), "attenuated")
        self.assertEqual(classify_alignment(-0.60, +0.30), "reversed")
        self.assertEqual(classify_alignment(-0.19, -0.19), "weak_or_unresolved")
```

增加 deterministic bootstrap、组内常数 response、13/14/14 计数、seed 重现性、GT rho 小于 0.05 时 magnitude ratio 留空等边界测试。

- [ ] **Step 2: 运行 ResponseStatisticsTests 并确认 RED**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity.ResponseStatisticsTests -v
```

- [ ] **Step 3: 实现统计函数**

实现规则：

- `build_response_rows` 输出 provenance + `response/response_tier/gt_value/pred_value/signed_error/absolute_error`。
- Fidelity groups 固定为 overall/elastic/plasticine/sand；输出 n、GT/pred mean/std、MAE、RMSE、bias、Spearman 及 95% CI。
- Partial Spearman 对 `x/y/control` 分别 rank 后，从 `x` 和 `y` 中用带截距 OLS 去除 control，再相关残差。
- Alignment 对每种材质、`log10_e/nu`、每个 response 输出 GT/pred ordinary rho、partial rho、bootstrap CI、rho gap、可选 magnitude ratio 和冻结标签。
- 标签判定优先级固定为 `reversed > attenuated > aligned > weak_or_unresolved`，解决阈值边界重叠。
- 常数输入返回 `None`，相关和 CI CSV 字段留空，状态为 `constant_response`；不是 exception。
- Bootstrap 以 object/model 为单位成对采样，使用 `np.random.default_rng(seed)`，不得把 response 行当独立样本。

- [ ] **Step 4: 运行 ResponseStatisticsTests 与 PositionResponseTests**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity.PositionResponseTests \
  tests.test_material_response_fidelity.ResponseStatisticsTests -v
```

- [ ] **Step 5: Commit Task 2**

```bash
git add src/utils/material_response_fidelity.py src/tests/test_material_response_fidelity.py
git commit -m "Add B0.3 response fidelity statistics"
```

---

### Task 3: 固定六文件输出与完整性保护

**Files:**
- Modify: `src/utils/material_response_fidelity.py`
- Modify: `src/tests/test_material_response_fidelity.py`

**Interfaces:**
- Produces `OUTPUT_NAMES: dict[str, str]` with exactly six names from the design.
- Produces `preflight_fidelity_outputs(output_dir: str | Path, overwrite: bool) -> dict[str, Path]`。
- Produces `write_fidelity_outputs(output_dir, model_rows, response_rows, fidelity_rows, alignment_rows, metadata, overwrite) -> dict[str, Path]`。

- [ ] **Step 1: 写输出事务的失败测试**

增加 `OutputTests`，至少覆盖：

```python
class OutputTests(unittest.TestCase):
    def test_writer_emits_exact_six_outputs_and_chinese_report(self):
        paths = write_fidelity_outputs(
            self.tmpdir,
            model_rows=self.model_rows,
            response_rows=self.response_rows,
            fidelity_rows=self.fidelity_rows,
            alignment_rows=self.alignment_rows,
            metadata=self.metadata,
            overwrite=False,
        )
        self.assertEqual(set(paths), set(OUTPUT_NAMES))
        self.assertEqual(set(path.name for path in paths.values()), set(OUTPUT_NAMES.values()))
        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("位置可观测响应", report)
        self.assertIn("不能证明 counterfactual", report)

    def test_preflight_rejects_any_existing_target_without_overwrite(self):
        existing = self.tmpdir / OUTPUT_NAMES["models"]
        existing.write_text("old", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            preflight_fidelity_outputs(self.tmpdir, overwrite=False)
```

同时覆盖 metadata 缺字段、模型计数错误、材质计数错误、response schema 不完整、临时写入失败不产生 final 文件，以及 `overwrite=True` 替换完整旧输出。

- [ ] **Step 2: 运行 OutputTests 并确认 RED**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity.OutputTests -v
```

- [ ] **Step 3: 实现 schema 校验、临时目录写入和报告**

固定名称：

```python
OUTPUT_NAMES = {
    "models": "material_response_fidelity_b03_models.csv",
    "responses": "material_response_fidelity_b03_responses.csv",
    "fidelity": "material_response_fidelity_b03_fidelity.csv",
    "alignment": "material_response_fidelity_b03_alignment.csv",
    "metadata": "material_response_fidelity_b03_metadata.json",
    "report": "material_response_fidelity_b03.md",
}
```

先在 output parent 下创建临时目录并完成六文件写入与校验，再激活到 final path；失败时清理临时文件。`overwrite=True` 时沿用 B0.2 的备份-激活-回滚原则，不能先删除旧输出。Markdown 必须分别报告 elastic/plasticine/sand，不只输出 overall。

- [ ] **Step 4: 运行全部新模块测试**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity -v
```

- [ ] **Step 5: Commit Task 3**

```bash
git add src/utils/material_response_fidelity.py src/tests/test_material_response_fidelity.py
git commit -m "Add B0.3 validated report outputs"
```

---

### Task 4: 41-model factual rollout CLI

**Files:**
- Create: `src/diagnose_material_response_fidelity.py`
- Modify: `src/tests/test_material_response_fidelity.py`

**Interfaces:**
- Produces `RuntimeComponents` and `_load_runtime_components()` matching existing diagnostic dependency injection style.
- Produces `run_material_response_fidelity(args, checkpoint, output_dir, config_path, seed, bootstrap_samples, contact_band_raw, overwrite, runtime=None) -> dict[str, Path]`。
- Produces `build_parser()` and `main()`。

- [ ] **Step 1: 写 fake-runtime CLI 集成失败测试**

构造 fake dataset/dataloader/model/pipeline/checkpoint loader，并验证：

```python
class DiagnosticRunnerTests(unittest.TestCase):
    def test_runner_evaluates_each_start_zero_model_once(self):
        paths = run_material_response_fidelity(
            args=fake_contact_cond_args(),
            checkpoint=self.checkpoint,
            output_dir=self.output_dir,
            config_path=Path("configs/eval_mm3_contact_cond.yaml"),
            seed=0,
            bootstrap_samples=200,
            contact_band_raw=0.08,
            overwrite=False,
            runtime=fake_runtime_for_41_models(),
        )
        self.assertEqual(fake_runtime.rollout_calls, 41)
        self.assertEqual(len(read_csv(paths["models"])), 41)
        self.assertEqual(len(read_csv(paths["responses"])), 41 * len(RESPONSE_NAMES))
```

另外覆盖：重复 model、缺 model、start-0 与非零窗口混批、batch/H5 参数不一致、checkpoint/config identity 不匹配、checkpoint 或 dataset 不存在、seed/bootstrap/contact-band 非法、昂贵推理前 output preflight 失败、严格只调用 normal factual condition。

- [ ] **Step 2: 运行 DiagnosticRunnerTests 并确认 RED**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity.DiagnosticRunnerTests -v
```

- [ ] **Step 3: 实现 CLI 与收集路径**

实现流程必须按顺序：

1. 参数与六目标 preflight。
2. `OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config))`。
3. `_validate_b0_identity(args, profile="contact_cond90")`。
4. 校验 checkpoint/dataset，构造 test dataset 和 material records。
5. strict load checkpoint；`model.eval().requires_grad_(False)`；`torch.compile`；scheduler 必须为 `None`。
6. 遍历 dataloader，仅处理 `start_idx=0`；每个 model 前 reset seed。
7. `_validate_normal_material_condition` 后调用一次 `rollout_condition`；调用 `_build_raw_reference` 获取 25 帧 GT。
8. 使用 `trajectory_metrics` 写标准 accuracy model row，使用 Task 1/2 接口写 response rows。
9. 校验 41 个唯一 model 和 13/14/14；构造 summary；最后一次性写六文件。

CLI：

```text
--config (required)
--checkpoint (required)
--output-dir (default results/material_response_fidelity_b03)
--seed (default 0)
--bootstrap-samples (default 10000)
--contact-band-raw (default 0.08)
--overwrite (store_true)
```

- [ ] **Step 4: 运行新测试模块全量**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity -v
```

- [ ] **Step 5: Commit Task 4**

```bash
git add src/diagnose_material_response_fidelity.py src/tests/test_material_response_fidelity.py
git commit -m "Add B0.3 factual response fidelity runner"
```

---

### Task 5: 登记实验并执行高强度验证

**Files:**
- Modify: `实验记录_1.md`
- Modify: `研究目标与路线图.md`
- Verify: all B0/B0.2/B2/B0.3 diagnostic files

**Interfaces:**
- Produces a registered server command and “实现完成、等待服务器运行” state.
- Does not pre-register a model-success claim or invent result values.

- [ ] **Step 1: 更新正式实验记录**

在 `实验记录_1.md` 增加 B0.3 条目：目标、研究问题、无训练边界、shared observable、六输出、设计/实现 commit、高强度验证和服务器命令。更新 `研究目标与路线图.md` 当前状态，但不写结果结论。

- [ ] **Step 2: 运行相关测试模块全量**

从 `src/` 运行：

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_material_response_fidelity \
  tests.test_material_identifiability \
  tests.test_material_response_sweep \
  tests.test_material_condition_diagnostics -v
```

Expected: all PASS；现有 B0.2 baseline 为 76 tests。

- [ ] **Step 3: 运行语法与 diff 检查**

从 worktree root 运行：

```bash
D:/miniconda3/envs/physctrl/python.exe -m py_compile \
  src/utils/material_response_fidelity.py \
  src/diagnose_material_response_fidelity.py \
  src/tests/test_material_response_fidelity.py
git diff --check
```

- [ ] **Step 4: 独立代码审查**

审查必须逐项检查：GT/pred 是否同坐标同帧；bootstrap 是否以 object 为单位；13/14/14 是否冻结；常数 response 是否合法；是否错误使用 test 做模型选择；是否误称凸包为 `det(F)`；是否修改 `eval.py` 或训练路径；输出替换失败是否可恢复。

发现 Critical/Important 后先写回归测试，再修复并重跑相关模块；Minor 仅在不扩张协议时修复。

- [ ] **Step 5: Commit Task 5**

```bash
git add 实验记录_1.md 研究目标与路线图.md src
git commit -m "Register and verify B0.3 response fidelity audit"
```

- [ ] **Step 6: 给出服务器正式命令**

从服务器 `src/` 运行：

```bash
python diagnose_material_response_fidelity.py \
  --config configs/eval_mm3_contact_cond.yaml \
  --checkpoint outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors \
  --output-dir results/material_response_fidelity_b03 \
  --seed 0 \
  --bootstrap-samples 10000 \
  --contact-band-raw 0.08
```

只在旧输出确实需要替换时附加 `--overwrite`。

---

## Plan Self-Review

- Spec coverage：位置响应、两层统计、常数 response、41-model factual rollout、六输出、无覆盖默认、解释限制和高强度验证均有对应 task。
- Type consistency：Task 1 的 response dict 由 Task 2 构成长表；Task 2 的三组 rows 由 Task 3 写出；Task 4 只负责推理与调用；Task 5 不改变协议。
- Scope：不修改 `eval.py`、模型、训练配置或 checkpoint；不加入 counterfactual generation、F/C estimator 或新训练。
- No placeholders：所有接口、路径、CLI、输出名、测试命令和服务器命令均已冻结。
