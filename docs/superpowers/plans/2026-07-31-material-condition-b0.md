# B0 Material-Condition Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个不训练新模型的独立诊断工具，在 `mm3_contact_cond@90k` 上通过配对反事实 rollout 判断模型对连续 `(E,nu)` 与离散 `mat_type` 的依赖。

**Architecture:** 将无 GPU 依赖的置换、轨迹指标、配对 bootstrap 与判据放入 `utils/material_condition_diagnostics.py`；独立 CLI `diagnose_material_condition.py` 负责读取现有 eval config、一次加载模型、执行 Normal/参数置换/类别置换三条 rollout，并输出逐模型 CSV 和汇总 Markdown。正式 `eval.py` 不修改。

**Tech Stack:** Python 3.10、PyTorch、NumPy、OmegaConf、h5py、safetensors、diffusers、`unittest`。

## Global Constraints

- 锚点固定为 `outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors` 与 `mm3_data/mm3_test`。
- 每个 model 只纳入一个 `start_idx=0` full-horizon 窗口。
- 三条路径使用相同 seed 和相同 rollout 逻辑；除 `E/nu` 或 `mat_type` 外不得改变其他输入。
- `(E,nu)` 只在同材质内做无固定点的一一置换；`mat_type` 使用 `0->1->2->0`。
- overall 与 elastic/plasticine/sand 必须分别汇总；不能只报告混合均值。
- 不修改 `src/eval.py`；不安装新依赖；不生成或提交 checkpoint/CSV 等训练产物。
- 供 Will 审阅的设计与实施说明使用中文；代码标识符使用英文。
- commit author 使用 Will 本人的 Git 配置，不添加任何 AI co-author/contributor。

---

### Task 1: 反事实置换与材料记录

**Files:**
- Create: `src/utils/material_condition_diagnostics.py`
- Create: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `MaterialRecord(model: str, mat_type: int, log10_e: float, nu: float)`。
- Produces: `build_parameter_derangement(records, seed) -> dict[str, tuple[float, float]]`。
- Produces: `rotate_material_type(mat_type: int) -> int`。

- [ ] **Step 1: 写无固定点同材质置换的失败测试**

```python
def test_derangement_is_within_material_one_to_one_and_reproducible(self):
    records = [
        MaterialRecord("e0.h5", 0, 4.0, 0.1),
        MaterialRecord("e1.h5", 0, 5.0, 0.2),
        MaterialRecord("e2.h5", 0, 6.0, 0.3),
        MaterialRecord("s0.h5", 2, 4.5, 0.15),
        MaterialRecord("s1.h5", 2, 6.5, 0.35),
    ]
    first = build_parameter_derangement(records, seed=7)
    second = build_parameter_derangement(records, seed=7)
    self.assertEqual(first, second)
    for group in ({"e0.h5", "e1.h5", "e2.h5"}, {"s0.h5", "s1.h5"}):
        source = {(r.log10_e, r.nu) for r in records if r.model in group}
        assigned = {first[name] for name in group}
        self.assertEqual(source, assigned)
        for record in (r for r in records if r.model in group):
            self.assertNotEqual(first[record.model], (record.log10_e, record.nu))
```

- [ ] **Step 2: 运行测试并确认因模块/函数缺失而失败**

Run: `python -m unittest src.tests.test_material_condition_diagnostics -v`

Expected: `ImportError` 或 `ModuleNotFoundError` 指向 `material_condition_diagnostics`。

- [ ] **Step 3: 实现最小置换逻辑**

按 `mat_type` 分组；每组先按 model 名排序，再用 `np.random.default_rng(seed + mat_type)` 打乱 model 顺序，最后将顺序中的第 `i+1` 个参数对分配给第 `i` 个 model。组大小小于 2 时抛出 `ValueError`。该循环移位天然保证无固定点和一一映射。

- [ ] **Step 4: 增加类别循环与非法类别失败测试**

```python
def test_rotates_supported_material_classes(self):
    self.assertEqual([rotate_material_type(i) for i in (0, 1, 2)], [1, 2, 0])

def test_rejects_unsupported_material_class(self):
    with self.assertRaisesRegex(ValueError, "expected one of"):
        rotate_material_type(3)
```

- [ ] **Step 5: 实现类别循环并运行测试**

Run: `python -m unittest src.tests.test_material_condition_diagnostics -v`

Expected: Task 1 全部 PASS。

- [ ] **Step 6: 提交 Task 1**

```bash
git add src/utils/material_condition_diagnostics.py src/tests/test_material_condition_diagnostics.py
git commit -m "Add material counterfactual permutation utilities"
```

---

### Task 2: 逐模型轨迹指标与条件响应

**Files:**
- Modify: `src/utils/material_condition_diagnostics.py`
- Modify: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `trajectory_metrics(pred, gt, input_frames=5) -> dict[str, float]`。
- Produces: `condition_response_metrics(normal, counterfactual, input_frames=5) -> dict[str, float]`。
- `pred`、`gt` 形状固定为 `(T,N,3)`，其中前 `input_frames` 是相同 GT 条件帧。

- [ ] **Step 1: 写 full-rollout、GM-MSE、long seg-MSE、FDE 的失败测试**

构造 `T=25,N=2` 的零 GT；预测帧 `f5..f24` 误差按已知常数递增。断言：

```python
metrics = trajectory_metrics(pred, gt, input_frames=5)
self.assertAlmostEqual(metrics["full_rollout_mse"], expected_full)
self.assertAlmostEqual(metrics["gm_mse"], expected_geometric_mean)
self.assertAlmostEqual(metrics["long_seg_mse"], expected_f18_to_f24_mean)
self.assertAlmostEqual(metrics["fde"], expected_final_point_l2)
```

`full_rollout_mse` 必须与现有 `eval.py` 一致，包含 5 个误差为零的条件帧；其余三个指标只使用预测帧。

- [ ] **Step 2: 运行单测确认新函数缺失导致失败**

Run: `python -m unittest src.tests.test_material_condition_diagnostics -v`

Expected: `ImportError` 指向 `trajectory_metrics`。

- [ ] **Step 3: 实现轨迹指标与输入校验**

校验 shape 相同、四维度要求为 `(T,N,3)`、`0 < input_frames < T`。逐预测帧计算 position MSE；GM 使用自然对数并以 `1e-30` 防止 `log(0)`；long segment 固定为绝对帧 `18..24` 中实际存在的帧；FDE 为末帧逐粒子 L2 均值。

- [ ] **Step 4: 写 prediction response 的失败测试并实现**

```python
response = condition_response_metrics(normal, counter, input_frames=5)
self.assertAlmostEqual(response["prediction_mse"], expected_prediction_mse)
self.assertAlmostEqual(response["final_prediction_mse"], expected_final_mse)
```

响应只比较 `f5..f24`，不把相同的条件帧稀释进结果。

- [ ] **Step 5: 运行 Task 1-2 单测**

Run: `python -m unittest src.tests.test_material_condition_diagnostics -v`

Expected: 全部 PASS。

- [ ] **Step 6: 提交 Task 2**

```bash
git add src/utils/material_condition_diagnostics.py src/tests/test_material_condition_diagnostics.py
git commit -m "Add material-condition trajectory metrics"
```

---

### Task 3: 配对 bootstrap、分组与自动判据

**Files:**
- Modify: `src/utils/material_condition_diagnostics.py`
- Modify: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `paired_bootstrap(normal, counterfactual, samples=10000, seed=0) -> dict[str, float]`。
- Produces: `dependency_label(relative_change_pct, ci_low, ci_high, response_ratio_pct) -> str`。
- Produces: `summarize_rows(rows, intervention, samples, seed) -> dict[str, dict[str, dict[str, float | str]]]`，分组键为 `overall/elastic/plasticine/sand`。

- [ ] **Step 1: 写配对而非非配对 bootstrap 的失败测试**

用固定数组构造所有 paired delta 都为 `+2` 的样本，断言均值差和 CI 都等于 2；再次调用 seed 相同结果必须逐值相等。空数组、长度不一致和 `samples<=0` 必须抛出 `ValueError`。

- [ ] **Step 2: 运行测试确认失败后实现 bootstrap**

每次 bootstrap 对 paired delta 的 model 索引有放回抽样，统计抽样均值的 2.5/97.5 percentile。相对变化定义为 `mean_delta / mean_normal * 100`；`mean_normal<=0` 时抛出错误。

- [ ] **Step 3: 写判据边界失败测试**

```python
self.assertEqual(dependency_label(5.0, 0.01, 0.2, 10.0), "used")
self.assertEqual(dependency_label(1.9, -0.1, 0.1, 1.9), "ignored")
self.assertEqual(dependency_label(4.0, 0.01, 0.2, 5.0), "ambiguous")
self.assertEqual(dependency_label(-6.0, -0.3, -0.1, 10.0), "ambiguous")
```

- [ ] **Step 4: 实现判据和 overall/per-material 汇总**

每个 intervention 分别汇总四项误差指标；`response_ratio_pct = prediction_mse / normal_full_rollout_mse * 100`。类别置换和参数置换使用同一统计实现，但报告标题必须明确类别置换只衡量依赖。

- [ ] **Step 5: 运行纯函数测试**

Run: `python -m unittest src.tests.test_material_condition_diagnostics -v`

Expected: Task 1-3 全部 PASS。

- [ ] **Step 6: 提交 Task 3**

```bash
git add src/utils/material_condition_diagnostics.py src/tests/test_material_condition_diagnostics.py
git commit -m "Add paired material-condition statistics"
```

---

### Task 4: 独立 GPU 诊断 CLI

**Files:**
- Create: `src/diagnose_material_condition.py`
- Modify: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `load_material_records(dataset_root, model_names) -> list[MaterialRecord]`。
- Produces: `rollout_condition(pipeline, batch, args, e_value, nu_value, mat_type) -> torch.Tensor`。
- Produces CLI：`python diagnose_material_condition.py --config ... --output-dir ... --permutation-seed 0 --bootstrap-samples 10000`。

- [ ] **Step 1: 写 HDF5 metadata 读取失败测试**

在临时目录创建两个最小 HDF5，字段为 `E`、`nu`、`mat_type`；断言读取结果中的 `log10_e` 与 dataset 语义一致。缺字段、同名重复记录和不存在文件分别抛出包含 model 名的错误。

- [ ] **Step 2: 运行失败测试后实现 metadata 读取**

使用 `h5py.File` 读取原始 `E` 并执行 `math.log10(E)`；要求 `E>0`。model 名统一为 basename，保证与 DataLoader 的 `batch['model']` 对齐。

- [ ] **Step 3: 写 rollout 条件隔离测试**

使用记录调用参数并返回确定性下一帧的轻量 fake pipeline，构造 5 帧输入和 `output_frames=1`。断言：

- 输出形状为 `(B,25,N,3)`；
- 前 5 帧与输入完全相同；
- 每次调用只收到指定 intervention 的 `E/nu/y`；
- 其余 force、floor、gravity、contact 输入在三条路径中相同；
- 单帧 rollout 的 `start_vel` 与现有 `eval.py` 公式一致。

- [ ] **Step 4: 实现与正式 eval 同语义的 rollout**

从 config 读取 `input_frames`、`output_frames`，固定预测 horizon 为 20 帧；每步重新创建 `torch.Generator().manual_seed(args.seed)`，滑动窗口使用：

```python
current_input = torch.cat([current_input, pred_chunk], dim=1)[:, -input_frames:]
```

只覆盖传给 pipeline 的 `E`、`nu` 或 `y`，绝不修改原始 batch。

- [ ] **Step 5: 实现 main 数据流与完整性检查**

1. 用 `TestingConfig` 合并 eval YAML。
2. 验证 `resume` 与 dataset 目录存在。
3. 构造 `TrajDataset('test', ...)` 和 metadata derangement。
4. 一次加载 `MDM_ST` checkpoint 并建立 `TrajPipeline`。
5. DataLoader 中跳过 `start_idx!=0`；每个 model 必须恰好评一次。
6. 对同一 batch 依次执行 Normal、shuffle params、shuffle class。
7. 从原 HDF5 与 `point_indices` 构造 25 帧 GT，计算平铺 row。
8. 断言最终 model 集与 metadata model 集完全一致。

- [ ] **Step 6: 实现 CSV/Markdown 输出**

CSV 每 model 一行，包含真实/置换条件、真实 material、三条路径的四项误差及两类 prediction response。Markdown 输出每个 intervention 下 overall 与三种材料的 baseline、counterfactual、relative change、paired delta 95% CI、response ratio 和 label。

默认输出：

```text
results/material_condition_b0/material_condition_b0_seed0.csv
results/material_condition_b0/material_condition_b0_seed0.md
```

- [ ] **Step 7: CLI help 测试**

Run: `python src/diagnose_material_condition.py --help`

Expected: exit 0，显示 `--config`、`--output-dir`、`--permutation-seed`、`--bootstrap-samples`；不访问 CUDA、checkpoint 或 dataset。

- [ ] **Step 8: 运行测试与语法检查**

```bash
python -m unittest src.tests.test_material_condition_diagnostics -v
python -m py_compile src/utils/material_condition_diagnostics.py src/diagnose_material_condition.py
```

Expected: 全部 PASS，py_compile exit 0。

- [ ] **Step 9: 提交 Task 4**

```bash
git add src/diagnose_material_condition.py src/utils/material_condition_diagnostics.py src/tests/test_material_condition_diagnostics.py
git commit -m "Add B0 material-condition diagnostic CLI"
```

---

### Task 5: 回归验证与服务器交付

**Files:**
- Verify only: `src/eval.py`
- Verify only: `src/configs/eval_mm3_contact_cond.yaml`

**Interfaces:**
- Consumes: Task 1-4 的 CLI。
- Produces: 可直接在服务器 `src/` 目录执行的诊断命令。

- [ ] **Step 1: 确认正式 eval 未被修改**

Run: `git diff -- src/eval.py src/configs/eval_mm3_contact_cond.yaml`

Expected: 无输出。

- [ ] **Step 2: 运行相关回归测试**

```bash
python -m unittest \
  src.tests.test_material_condition_diagnostics \
  src.tests.test_eval_contact_metrics \
  src.tests.test_contact -v
```

Expected: 0 failures、0 errors。

- [ ] **Step 3: 检查改动范围与仓库产物**

```bash
git diff --check
git status --short
```

只允许本功能源文件、测试与文档出现；不得 stage 用户现有未跟踪 PPT、图片、临时目录或结果文件。

- [ ] **Step 4: 给出服务器运行命令**

```bash
cd /root/code/traceformer/src
python diagnose_material_condition.py \
  --config configs/eval_mm3_contact_cond.yaml \
  --output-dir results/material_condition_b0 \
  --permutation-seed 0 \
  --bootstrap-samples 10000
```

明确说明本地未执行 GPU 诊断；真实结论只能在服务器结果生成后给出。
