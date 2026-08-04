# B2 Material Response Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为冻结的 `mm3_contact_cond@90k` 实现可复现的 `E/nu` 一维反事实扫描，输出逐模型响应、单调性和分材质报告。

**Architecture:** 新建独立 utility 和 CLI，复用 `diagnose_material_condition.py` 的冻结协议、rollout 与数据验证，不修改旧 B0/B0.1。utility 只处理有限数组、指标、聚合和 writer；CLI 负责一次加载、287 次配对 rollout 和 provenance。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy（已有依赖）、OmegaConf、unittest。

## Global Constraints

- 固定扫描条件：normal；`log10(E)=4.5/5.5/6.5`；`nu=0.10/0.25/0.40`。
- 41 个 frozen start0 models，共 287 rollouts，材质数 13/14/14。
- 扫描一个参数时另一个参数和 `mat_type` 保持该模型真实值。
- 每个 condition 重置同一 inference seed；模型和 checkpoint 只加载一次。
- 无 counterfactual GT；报告必须明确“响应与方向诊断，不是准确率”。
- 不修改模型、训练代码和旧 B0/B0.1 输出协议。
- 测试强度为中：相关单元测试、`py_compile`、`git diff --check`。

---

### Task 1: 扫描定义、轨迹状态指标与单调性

**Files:**
- Create: `src/utils/material_response_sweep.py`
- Create: `src/tests/test_material_response_sweep.py`

**Interfaces:**
- Produces: `SWEEP_CONDITIONS`、`build_sweep_conditions(record)`、`trajectory_state_metrics(prediction, reference_frame, floor_height)`、`response_metrics(normal, counterfactual)`、`spearman_monotonicity(values, responses, expected_direction)`。

- [ ] **Step 1: 写扫描条件和输入验证的失败测试**

断言七个 condition 的有序 tuple，检查 E 扫描保持真实 `nu/mat_type`、nu 扫描保持真实 `E/mat_type`，拒绝非有限材料参数。

- [ ] **Step 2: 运行测试确认 RED**

Run: `PYTHONPATH=src python -m unittest src.tests.test_material_response_sweep -v`

- [ ] **Step 3: 实现固定扫描条件和 trajectory/response metrics**

状态指标必须只使用预测 horizon，返回 motion RMS、f24 centroid displacement、f24 Procrustes shape deformation、f24 absolute volume change、penetration rate/depth。响应指标返回 prediction response MSE、final response MSE、centroid response、shape response。

- [ ] **Step 4: 实现三点 Spearman 和单调标记**

返回 `rho`、`strict_monotonic`、`weak_monotonic`；相等值允许 weak、不允许 strict；非有限值和长度不为 3 时失败。

- [ ] **Step 5: 运行 Task 1 测试**

Expected: Task 1 tests PASS。

### Task 2: 逐模型摘要、分组统计和 writer

**Files:**
- Modify: `src/utils/material_response_sweep.py`
- Modify: `src/tests/test_material_response_sweep.py`

**Interfaces:**
- Consumes: Task 1 condition rows。
- Produces: `validate_raw_rows(rows)`、`build_model_summaries(rows)`、`build_group_summaries(model_rows)`、`write_sweep_outputs(output_dir, raw_rows, model_rows, summary_rows, metadata)`。

- [ ] **Step 1: 写 287/41 完整性和分材质计数测试**

覆盖缺 condition、重复 model-condition、材质计数错误、provenance 不一致和非有限指标。

- [ ] **Step 2: 运行测试确认 RED**

Run: `PYTHONPATH=src python -m unittest src.tests.test_material_response_sweep -v`

- [ ] **Step 3: 实现 model-level sensitivity/monotonicity**

每个模型输出 E/nu 三点响应、Spearman、strict/weak 标记；E 的主要 sanity 变量为 shape deformation，nu 的主要 sanity 变量为 absolute volume change。

- [ ] **Step 4: 实现 overall/elastic/plasticine/sand 汇总**

报告均值、中位数、响应模型比例、strict/weak plausible 比例；不跨材质平均后替代分材质结果。

- [ ] **Step 5: 实现固定四文件输出**

CSV 固定列顺序并重复 config/checkpoint/seed/scope；中文 Markdown 必须包含扫描范围、样本数、分材质表、限制声明和 `ignored/responsive_non_monotonic/directionally_plausible/unstable_excessive` 分类解释。

- [ ] **Step 6: 运行 Task 2 测试**

Expected: Task 1–2 tests PASS。

### Task 3: 冻结协议 CLI 与 287 次 rollout 集成

**Files:**
- Create: `src/diagnose_material_response_sweep.py`
- Modify: `src/tests/test_material_response_sweep.py`

**Interfaces:**
- Consumes: B0 的 `DiagnosticProfile`、`load_material_records`、`_validate_b0_identity`、`_validate_normal_material_condition`、`_build_raw_reference`、`_rollout_condition` 和 frozen manifest。
- Produces: CLI `--config --checkpoint --output-dir --seed`，默认输出目录 `results/material_response_sweep_b2`。

- [ ] **Step 1: 写 CLI mock RED 测试**

断言 config/checkpoint 各加载一次、41 models、287 rollouts、每 condition 重置相同 seed、batch-record mismatch 在 rollout 前失败、旧 B0 writer 未调用。

- [ ] **Step 2: 实现 parser 和 fail-closed config/checkpoint guard**

只接受注册的 `mm3_contact_cond@90k` profile；路径和 model/dataset runtime defaults 复用现有 B0 校验。

- [ ] **Step 3: 实现主循环**

每个 start0 model 先验证 batch-record，再运行 normal 和六个扫描条件。所有条件保持真实 `mat_type`，生成 raw rows 后统一验证、聚合和写文件。

- [ ] **Step 4: 运行中等强度验证**

```bash
PYTHONPATH=src python -m unittest src.tests.test_material_response_sweep -v
python -m py_compile src/diagnose_material_response_sweep.py src/utils/material_response_sweep.py src/tests/test_material_response_sweep.py
git diff --check
```

Expected: 全部 exit 0。

### Task 4: 登记实现，不伪造实验结果

**Files:**
- Modify after implementation only: `实验记录_1.md`

- [ ] **Step 1: 在待执行实验表登记 B2**

记录目标 C、冻结 checkpoint/config、扫描网格、代码 commit 和服务器命令；状态写“代码完成/待服务器运行”。

- [ ] **Step 2: 不写结果结论**

真实 287-rollout 输出产生前，不填写 sensitivity、单调性或 proceed/close 结论。
