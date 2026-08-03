# HST Feedback Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有 B1b 90k checkpoint 增加一个不改变模型 forward 的诊断工具，按材质、rollout horizon 和 exchange stage 分解实际 HST feedback，并与 full-rollout、FDE、f24 质心和 Procrustes 残余形状误差关联。

**Architecture:** 使用 `HybridStateExchange` 与其 `feedback_attention` 的 forward hooks 捕获 `gate × feedback`，在粒子轴分解均值与去均值分量。独立 CLI 复用 `diagnose_material_condition.rollout_condition()` 和 `utils.eval_metrics.per_window_metrics()`，只跑 41 个 `start_idx=0` 窗口；纯函数模块负责统计、相关性和报告，生产模型代码保持不变。

**Tech Stack:** Python 3.10、PyTorch、NumPy、OmegaConf、safetensors、现有 `TrajDataset`/`TrajPipeline`、`unittest`。

## Global Constraints

- 不修改 `src/model/hybrid_state.py`、`src/model/spacetime.py`、训练路径或 checkpoint 内容。
- 诊断必须使用 eager forward，不调用 `torch.compile`。
- 只分析 B1b checkpoint-90000、mm3 test split 中 41 个 `start_idx=0` 全程窗口。
- 每次 single-frame 模型 forward 必须捕获 stage `0,1,2,3` 各一次；异常立即报错。
- 不新增第三方依赖；不把相关性写成因果结论。
- 新 Python 文件完成后必须运行 `python -m py_compile` 和对应单元测试。
- 结果 CSV/Markdown 写入 `src/results/` 或命令指定目录，不纳入 commit。
- commit author 使用 Will，不添加任何 AI co-author/contributor；不执行 git push。

---

## File Structure

- Create `src/utils/hybrid_state_diagnostics.py`: feedback 分解、hook recorder、分组统计、相关性与报告纯函数。
- Create `src/tests/test_hybrid_state_diagnostics.py`: 上述纯函数与真实 `HybridStateExchange` hook 的 CPU 测试。
- Create `src/diagnose_hybrid_state_feedback.py`: 配置/checkpoint 校验、41-model rollout、CSV/Markdown 输出 CLI。
- Modify `实验记录.md`: 将 B1b 从“待训练”更新为 90k 混合负结果，并登记诊断工具状态与代码 commit。

### Task 1: Feedback 分解与只读 Hook Recorder

**Files:**
- Create: `src/utils/hybrid_state_diagnostics.py`
- Create: `src/tests/test_hybrid_state_diagnostics.py`

**Interfaces:**
- Produces: `decompose_feedback(feedback: torch.Tensor, gate: torch.Tensor | float) -> dict[str, torch.Tensor]`
- Produces: `HybridStateFeedbackRecorder(exchange: nn.Module)`，支持 context manager、`reset()`、`finalize(expected_rollout_steps: int) -> list[dict[str, float | int]]`
- Consumes: `exchange.feedback_attention`、`exchange.feedback_gates`、exchange pre-hook 的 `stage_index`。

- [ ] **Step 1: 写 feedback 分解的失败测试**

在 `src/tests/test_hybrid_state_diagnostics.py` 创建 `unittest.TestCase`，测试：

```python
def test_decompose_feedback_separates_particle_mean_and_centered_energy(self):
    feedback = torch.tensor([[[1.0, 3.0], [3.0, 1.0]]])
    stats = decompose_feedback(feedback, gate=torch.tensor(0.5))

    expected_delta = feedback * 0.5
    expected_global = expected_delta.mean(dim=1)
    expected_centered = expected_delta - expected_global[:, None]
    torch.testing.assert_close(stats["feedback_rms"], expected_delta.square().mean((1, 2)).sqrt())
    torch.testing.assert_close(stats["global_rms"], expected_global.square().mean(1).sqrt())
    torch.testing.assert_close(stats["deform_rms"], expected_centered.square().mean((1, 2)).sqrt())
    torch.testing.assert_close(
        stats["feedback_energy"],
        stats["global_energy"] + stats["deform_energy"],
    )
```

同时增加 shape 非 `(B,N,C)`、非有限 gate、零反馈能量占比应返回 0 的测试。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: FAIL，原因是 `src.utils.hybrid_state_diagnostics` 尚不存在。

- [ ] **Step 3: 最小实现 `decompose_feedback`**

在 `src/utils/hybrid_state_diagnostics.py` 实现：

```python
def decompose_feedback(feedback, gate):
    if feedback.ndim != 3:
        raise ValueError("feedback must have shape (B,N,C)")
    delta = torch.nan_to_num(feedback.detach().float()) * torch.as_tensor(
        gate, device=feedback.device, dtype=torch.float32
    )
    if not torch.isfinite(delta).all():
        raise ValueError("applied feedback must be finite")
    global_component = delta.mean(dim=1)
    deform_component = delta - global_component[:, None]
    feedback_energy = delta.square().mean(dim=(1, 2))
    global_energy = global_component.square().mean(dim=1)
    deform_energy = deform_component.square().mean(dim=(1, 2))
    fraction = torch.where(
        feedback_energy > 0,
        global_energy / feedback_energy,
        torch.zeros_like(feedback_energy),
    )
    return {
        "feedback_rms": feedback_energy.sqrt().cpu(),
        "global_rms": global_energy.sqrt().cpu(),
        "deform_rms": deform_energy.sqrt().cpu(),
        "feedback_energy": feedback_energy.cpu(),
        "global_energy": global_energy.cpu(),
        "deform_energy": deform_energy.cpu(),
        "global_energy_fraction": fraction.cpu(),
    }
```

显式检查 scalar gate；用 `torch.testing.assert_close` 验证能量恒等式的误差来自浮点容差而不是公式错误。

- [ ] **Step 4: 运行分解测试并确认 GREEN**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: 所有当前测试 PASS。

- [ ] **Step 5: 写 recorder 的失败测试**

使用真实小尺寸 `HybridStateExchange(particle_dim=8, state_dim=8, num_heads=2, num_stages=4)`，将四个 gate 设置为非零值。连续调用 stage 0..3，断言：

```python
with HybridStateFeedbackRecorder(exchange) as recorder:
    for stage in range(4):
        state, hidden = exchange(
            hidden_states=hidden,
            state_tokens=state,
            explicit_frame_state=explicit,
            material_values=material,
            history_start=1,
            prediction_index=6,
            stage_index=stage,
        )
records = recorder.finalize(expected_rollout_steps=1)
self.assertEqual([row["stage"] for row in records], [0, 1, 2, 3])
self.assertEqual([row["rollout_step"] for row in records], [0, 0, 0, 0])
```

另测 stage 缺失、重复、乱序、context 退出后再次 forward 不再记录。

- [ ] **Step 6: 运行 recorder 测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: FAIL，原因是 `HybridStateFeedbackRecorder` 尚不存在。

- [ ] **Step 7: 最小实现 recorder**

实现两个 hook：

```python
def _capture_stage(self, module, args, kwargs):
    if self._pending_stage is not None:
        raise RuntimeError("previous exchange stage was not consumed")
    self._pending_stage = int(kwargs["stage_index"])

def _capture_feedback(self, module, args, output):
    if self._pending_stage is None:
        raise RuntimeError("feedback captured without exchange stage")
    stage = self._pending_stage
    stats = decompose_feedback(output, self.exchange.feedback_gates[stage])
    for batch_index in range(output.shape[0]):
        self._records.append({
            "stage": stage,
            "gate": float(self.exchange.feedback_gates[stage].detach().cpu()),
            **{key: float(value[batch_index]) for key, value in stats.items()},
        })
    self._pending_stage = None
```

`finalize()` 验证总记录数等于 `expected_rollout_steps × num_stages`，验证每四条 stage 顺序，并填入 `rollout_step` 与 `absolute_frame=history_frames+rollout_step`。

- [ ] **Step 8: 运行 Task 1 全部测试并提交**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
python -m py_compile src/utils/hybrid_state_diagnostics.py src/tests/test_hybrid_state_diagnostics.py
```

Expected: PASS。

Commit:

```bash
git add src/utils/hybrid_state_diagnostics.py src/tests/test_hybrid_state_diagnostics.py
git commit -m "Add HST feedback recorder"
```

### Task 2: 分组统计、相关性和报告

**Files:**
- Modify: `src/utils/hybrid_state_diagnostics.py`
- Modify: `src/tests/test_hybrid_state_diagnostics.py`

**Interfaces:**
- Consumes: Task 1 recorder rows enriched with model/material/trajectory fields。
- Produces: `horizon_bucket(absolute_frame: int) -> str`
- Produces: `aggregate_feedback_rows(rows: list[dict]) -> list[dict]`
- Produces: `feedback_correlations(rows: list[dict]) -> list[dict]`
- Produces: `write_feedback_csv(path: Path, rows: list[dict]) -> None`
- Produces: `write_feedback_report(path: Path, rows: list[dict], metadata: dict[str, str]) -> None`

- [ ] **Step 1: 写 horizon 和聚合的失败测试**

构造两个 model、三种 horizon、四个 stage 的 synthetic rows，验证：

```python
self.assertEqual(horizon_bucket(5), "short")
self.assertEqual(horizon_bucket(10), "short")
self.assertEqual(horizon_bucket(11), "mid")
self.assertEqual(horizon_bucket(17), "mid")
self.assertEqual(horizon_bucket(18), "long")
self.assertEqual(horizon_bucket(24), "long")
```

`aggregate_feedback_rows()` 必须按 overall/material × stage/horizon 分组，输出真实 model 数而非重复 row 数。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: FAIL，缺少统计函数。

- [ ] **Step 3: 实现 horizon 与聚合**

固定边界：short=f5-10、mid=f11-17、long=f18-24。聚合均值字段限定为：

```python
FEEDBACK_METRICS = (
    "feedback_rms",
    "global_rms",
    "deform_rms",
    "global_energy_fraction",
)
```

输入为空、缺字段、非有限值必须 `ValueError`。

- [ ] **Step 4: 写相关性的失败测试**

相关性必须先按 model 聚合 feedback，防止同一个轨迹指标被 80 行重复造成伪样本扩增。构造严格线性 synthetic data，断言 Pearson/Spearman 为 `+1` 或 `-1`；常数数组返回 `nan` 并在报告显示 `N/A`。

- [ ] **Step 5: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: FAIL，缺少 `feedback_correlations`。

- [ ] **Step 6: 实现 model-level Pearson/Spearman**

对每个 overall/material group 和每个 feedback metric × trajectory metric 组合：

```python
TRAJECTORY_METRICS = (
    "full_rollout_mse",
    "fde",
    "f24_centroid_error",
    "f24_shape_residual_mse",
)
```

先对每个 model 的全部 step/stage feedback 取均值，再计算相关性。Spearman 使用 NumPy 平均秩实现，不引入 SciPy 新依赖。

- [ ] **Step 7: 写 CSV/Markdown 输出失败测试**

在 `tempfile.TemporaryDirectory()` 写文件，验证：

- CSV 有固定列头且按 model/step/stage 排序；
- Markdown 包含 checkpoint、config、41-window 口径说明；
- Markdown 包含 overall/elastic/plasticine/sand、stage、horizon 和 correlation 表；
- 非有限相关性显示 `N/A`，不输出字符串 `nan`。

- [ ] **Step 8: 实现 writer 并运行 Task 2 测试**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
python -m py_compile src/utils/hybrid_state_diagnostics.py src/tests/test_hybrid_state_diagnostics.py
```

Expected: PASS。

Commit:

```bash
git add src/utils/hybrid_state_diagnostics.py src/tests/test_hybrid_state_diagnostics.py
git commit -m "Summarize HST feedback diagnostics"
```

### Task 3: 41-Model 诊断 CLI

**Files:**
- Create: `src/diagnose_hybrid_state_feedback.py`
- Modify: `src/tests/test_hybrid_state_diagnostics.py`

**Interfaces:**
- Consumes: Task 1/2 的 recorder、writers；现有 `rollout_condition()`、`_build_raw_reference()`、`load_material_records()`、`per_window_metrics()`。
- Produces: `validate_diagnostic_config(args, checkpoint: Path) -> None`
- Produces: `trajectory_diagnostic_fields(pred: torch.Tensor, gt: torch.Tensor, input_frames: int) -> dict[str, float]`
- Produces: `run_feedback_diagnostics(args, checkpoint: Path, output_dir: Path) -> list[dict]`
- Produces CLI flags: `--config`、`--checkpoint`、`--output-dir`。

- [ ] **Step 1: 写配置校验和轨迹字段的失败测试**

配置校验必须拒绝：

- block 不是 `SpatialTemporalTransformerBlockv11a`；
- `contact_particle_cond != True`；
- `input_frames != 5` 或 `output_frames != 1`；
- diffusion 开启、batch size 不为 1；
- checkpoint 名称不含 `checkpoint-90000/model.safetensors`。

`trajectory_diagnostic_fields()` 使用 synthetic trajectory 验证 full MSE、FDE 和 `per_window_metrics(...)["proc"][24]` 的 centroid/residual 字段。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
```

Expected: FAIL，CLI 模块尚不存在。

- [ ] **Step 3: 实现 CLI 骨架和校验**

配置加载：

```python
args = OmegaConf.merge(
    OmegaConf.structured(TestingConfig),
    OmegaConf.load(cli_args.config),
)
checkpoint = Path(cli_args.checkpoint)
args.resume = str(checkpoint)
```

报告 metadata 必须保留原 config 路径和命令行 checkpoint 路径。

- [ ] **Step 4: 实现 eager rollout 与 hook 数据拼接**

复用现有逻辑：

```python
model = MDM_ST(args.pc_size, 1, n_feats=3, model_config=args.model_config).to("cuda")
_load_checkpoint_strict(model, checkpoint, load_file)
model.eval().requires_grad_(False)
pipeline = TrajPipeline(model=model, scheduler=None)

with HybridStateFeedbackRecorder(model.dit.hybrid_state_exchange) as recorder:
    pred = rollout_condition(
        pipeline, batch, args,
        record.log10_e, record.nu, record.mat_type,
    )
feedback_rows = recorder.finalize(expected_rollout_steps=20)
```

不得调用 `torch.compile`。只接受 start0、每个 model 一次；结束时严格验证 41 个唯一模型以及 13/14/14 材质计数。

- [ ] **Step 5: 将轨迹字段加入 80 条 model feedback rows**

对每个 model 只计算一次：

```python
trajectory = trajectory_diagnostic_fields(pred[0], gt[0], input_frames=5)
for feedback_row in feedback_rows:
    feedback_row.update({
        "model": model_name,
        "mat_type": record.mat_type,
        "log10_e": record.log10_e,
        "nu": record.nu,
        **trajectory,
    })
```

最终应有 `41 × 20 × 4 = 3280` 行。

- [ ] **Step 6: 写 CLI parser/output path 测试并实现**

输出固定为：

```text
<output-dir>/hybrid_state_feedback_b1b_90k.csv
<output-dir>/hybrid_state_feedback_b1b_90k.md
```

CLI 完成时打印绝对路径、模型数、row 数和各材质计数。

- [ ] **Step 7: 运行 Task 3 测试和静态验证**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_diagnostics -v
python -m py_compile \
  src/diagnose_hybrid_state_feedback.py \
  src/utils/hybrid_state_diagnostics.py \
  src/tests/test_hybrid_state_diagnostics.py
```

Expected: PASS；本机不要求 CUDA integration run。

Commit:

```bash
git add src/diagnose_hybrid_state_feedback.py src/tests/test_hybrid_state_diagnostics.py
git commit -m "Add B1b HST feedback diagnostic CLI"
```

### Task 4: B1b 台账与整体回归验证

**Files:**
- Modify: `实验记录.md`

**Interfaces:**
- Consumes: B1b 01/11@90k 已有结果和 Tasks 1-3 的真实 commit hash。
- Produces: B1b 完整判决、分材质 Procrustes 结论、诊断工具运行命令与“待服务器运行”状态。

- [ ] **Step 1: 更新 B1b 总览行**

将状态从“已实现·待训练”改为“混合负·未通过”，至少记录：

- full-rollout `3.501029e-3 -> 3.475232e-3`（-0.7%）；
- FDE `0.1008436 -> 0.1153226`（+14.4%）；
- plasticine long `5.397833e-4 -> 2.309489e-3`（+327.9%）；
- sand long `1.911320e-2 -> 1.731829e-2`（-9.4%）；
- penetration `3.340% -> 5.726%`（+71.4%）。

- [ ] **Step 2: 更新 B1b 详录**

明确写出：

- v11a 在无 contact 时 full-rollout 改善 17.4%，有 contact 时仅改善 0.7%；
- sand 形状 residual 改善，plasticine 质心和形状同时恶化；
- B1b 命中预注册停止条件，不调 v11a 超参；
- 下一步只做被动 feedback 诊断，诊断完成前不启动新训练；
- 诊断结果若无稳定关系，则关闭 v11a。

- [ ] **Step 3: 回填 Tasks 1-3 的真实 commit hash 与服务器命令**

运行命令写为：

```bash
python diagnose_hybrid_state_feedback.py \
  --config configs/eval_mm3_v11a_contact_cond_8L_45k.yaml \
  --checkpoint outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors \
  --output-dir results/hybrid_state_b1b_90k
```

- [ ] **Step 4: 全量相关测试与 diff 检查**

Run:

```bash
python -m unittest \
  src.tests.test_hybrid_state_diagnostics \
  src.tests.test_hybrid_state \
  src.tests.test_v11a_contact_config -v
python -m py_compile \
  src/diagnose_hybrid_state_feedback.py \
  src/utils/hybrid_state_diagnostics.py \
  src/tests/test_hybrid_state_diagnostics.py
git diff --check
```

Expected: 所有测试 PASS；无 whitespace error；`src/model/` 与训练 config 无 diff。

- [ ] **Step 5: 提交台账**

```bash
git add 实验记录.md
git commit -m "Record B1b result and diagnostic plan"
```

## Server Verification After Merge

在服务器拉取代码后，从 `src/` 目录运行：

```bash
python diagnose_hybrid_state_feedback.py \
  --config configs/eval_mm3_v11a_contact_cond_8L_45k.yaml \
  --checkpoint outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors \
  --output-dir results/hybrid_state_b1b_90k
```

验收输出：

- `3280` raw rows；
- `41` unique models；
- material counts `13/14/14`；
- 每个 model/rollout step 的 stage 顺序严格为 `0,1,2,3`；
- 报告没有非有限 feedback 数值；
- 根据报告选择 motion/deformation factorization、material-conditioned gate 或关闭 v11a，选择前另写预注册设计。
