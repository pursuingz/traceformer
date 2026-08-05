# B3a Stage 与梯度诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为冻结的 B3a@90k checkpoint 增加逐 stage knockout、真实 adapter 更新幅度和分材质梯度 cosine 诊断，且不修改训练、模型结构或正式 eval 路径。

**Architecture:** 纯统计与验证逻辑集中在一个新的 utility 模块；两个 CLI 分别负责 GPU rollout 和 teacher-forced backward。Stage knockout 只在上下文管理器内临时修改 `stage_scales` 并逐位恢复；梯度诊断冻结全模型，仅保留 adapter 参数梯度，按材质累积 164 个固定窗口的单步坐标 MSE 梯度。

**Tech Stack:** Python 3.10、PyTorch、OmegaConf、safetensors、NumPy、标准库 `csv/json/contextlib`、`unittest`。

## Global Constraints

- 固定 checkpoint：`outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors`。
- 固定 config：`configs/eval_mm3_b3a_material_state_adapter_90k.yaml`。
- 固定模型集合：41 个 test models，elastic/plasticine/sand = 13/14/14。
- Stage rollout 只使用 `start_idx=0`；梯度诊断使用每个 model 的 `start_idx={0,5,10,15}`。
- seed 固定为 0；同一 model 的 knockout conditions 必须重置到相同 RNG 状态。
- 不修改 `src/model/spacetime.py`、`src/model/material_state.py`、训练 config 或正式 eval config。
- 不创建 optimizer，不更新 checkpoint，不启用 full-rollout BPTT。
- 梯度目标只使用 teacher-forced 单步坐标 MSE。
- 本次测试强度为“中”：相关单元测试、`py_compile`、`git diff --check`。
- commit 只使用 Will 的 Git 身份，不加入任何 AI co-author；不执行 `git push`。

---

## File Structure

- Create `src/utils/material_state_stage_diagnostics.py`
  - B3a stage mask、activity collector、配对统计、梯度向量分组、cosine 和报告写入。
- Create `src/diagnose_material_state_stage_knockout.py`
  - 严格加载 B3a@90k，一次加载模型，执行六种 knockout conditions。
- Create `src/diagnose_material_state_gradient_conflict.py`
  - 严格加载 B3a@90k，遍历 164 个固定窗口并累计三材质 adapter 梯度。
- Create `src/tests/test_material_state_stage_diagnostics.py`
  - Stage mask、activity、配对统计和输出完整性测试。
- Create `src/tests/test_material_state_gradient_conflict.py`
  - 梯度分组、平均、cosine、零梯度和固定窗口协议测试。
- Modify `实验记录_1.md`
  - 登记诊断实现、命令、输出文件和实现 commit；结果仍标记为待服务器执行。

---

### Task 1: Stage Mask 与 Activity 纯工具

**Files:**
- Create: `src/utils/material_state_stage_diagnostics.py`
- Create: `src/tests/test_material_state_stage_diagnostics.py`

**Interfaces:**
- Produces: `STAGE_KNOCKOUT_CONDITIONS: tuple[tuple[str, tuple[int, ...]], ...]`
- Produces: `masked_material_state_stages(adapter, mask)` context manager
- Produces: `MaterialStateActivityCollector(adapter)` context manager with `capture(model_name, mat_type, expected_calls_per_stage)` and `rows()`
- Consumes: `FactorizedMaterialStateAdapter.stage_scales`

- [ ] **Step 1: Write failing tests for the registered conditions and exact restoration**

```python
def test_stage_conditions_cover_normal_all_and_four_single_knockouts(self):
    self.assertEqual(
        STAGE_KNOCKOUT_CONDITIONS,
        (
            ("normal", (1, 1, 1, 1)),
            ("all_off", (0, 0, 0, 0)),
            ("stage0_off", (0, 1, 1, 1)),
            ("stage1_off", (1, 0, 1, 1)),
            ("stage2_off", (1, 1, 0, 1)),
            ("stage3_off", (1, 1, 1, 0)),
        ),
    )

def test_mask_multiplies_checkpoint_scales_and_restores_exactly(self):
    adapter = make_adapter()
    with torch.no_grad():
        adapter.stage_scales.copy_(torch.tensor([1.1, 0.9, 0.7, 0.5]))
    original = adapter.stage_scales.detach().clone()
    with masked_material_state_stages(adapter, (1, 0, 1, 0)):
        self.assertTrue(torch.equal(
            adapter.stage_scales.detach(),
            torch.tensor([1.1, 0.0, 0.7, 0.0]),
        ))
    self.assertTrue(torch.equal(adapter.stage_scales.detach(), original))
```

- [ ] **Step 2: Run RED test**

```powershell
$env:PYTHONPATH='src;.'
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_material_state_stage_diagnostics
```

Expected: FAIL because `utils.material_state_stage_diagnostics` does not exist.

- [ ] **Step 3: Implement registered conditions and fail-closed mask context**

```python
STAGE_KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
    ("stage3_off", (1, 1, 1, 0)),
)

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
```

- [ ] **Step 4: Write failing activity collector tests**

Test a nonzero synthetic adapter for two calls to every stage. Assert:

```python
self.assertEqual(len(rows), 4)
self.assertEqual({row["stage_index"] for row in rows}, {0, 1, 2, 3})
self.assertTrue(all(row["call_count"] == 2 for row in rows))
self.assertTrue(all(row["delta_rms"] > 0 for row in rows))
self.assertTrue(all(row["hidden_rms"] > 0 for row in rows))
self.assertTrue(all(row["relative_rms"] > 0 for row in rows))
```

Also assert that `capture(...)` rejects nested model contexts, missing stages, nonfinite outputs, and any stage call count other than `expected_calls_per_stage` when the context exits.

- [ ] **Step 5: Run RED activity tests**

Run the Task 1 command again. Expected: FAIL because `MaterialStateActivityCollector` is missing.

- [ ] **Step 6: Implement the activity collector**

Use `adapter.register_forward_hook(hook, with_kwargs=True)`. The hook reads:

```python
hidden = positional_args[0]
stage_index = int(kwargs["stage_index"])
delta = output.detach().float() - hidden.detach().float()
```

For each stage accumulate `delta.square().sum()`, `hidden.square().sum()`, `numel`, and `call_count` on CPU float64 scalars. Exiting `capture(...)` validates the expected calls and appends one row per stage; `rows()` returns a defensive copy of all completed model-stage rows:

```python
{
    "model": model_name,
    "mat_type": mat_type,
    "stage_index": stage_index,
    "call_count": call_count,
    "delta_rms": math.sqrt(delta_sq_sum / numel),
    "hidden_rms": math.sqrt(hidden_sq_sum / numel),
    "relative_rms": delta_rms / hidden_rms,
}
```

- [ ] **Step 7: Run GREEN Task 1 tests**

Expected: all Task 1 tests PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/utils/material_state_stage_diagnostics.py src/tests/test_material_state_stage_diagnostics.py
git commit -m "Add B3a stage mask and activity diagnostics"
```

---

### Task 2: Stage Knockout 配对统计与 CLI

**Files:**
- Modify: `src/utils/material_state_stage_diagnostics.py`
- Modify: `src/tests/test_material_state_stage_diagnostics.py`
- Create: `src/diagnose_material_state_stage_knockout.py`

**Interfaces:**
- Consumes: `trajectory_knockout_metrics()` and `paired_delta_summary()` from `utils.hybrid_state_gate_knockout`
- Produces: `validate_stage_raw_rows(rows) -> list[dict]`
- Produces: `build_stage_paired_rows(rows) -> list[dict]`
- Produces: `summarize_stage_paired_rows(rows, bootstrap_samples, bootstrap_seed) -> list[dict]`
- Produces: `run_stage_knockout_diagnostics(args, checkpoint, output_dir, ...) -> list[dict]`

- [ ] **Step 1: Write failing tests for fixed row cardinality and pairing**

Create synthetic data for 41 models × 6 conditions × all ten trajectory metrics. Assert:

```python
self.assertEqual(len(validate_stage_raw_rows(raw_rows)), 246)
paired = build_stage_paired_rows(raw_rows)
self.assertEqual(len(paired), 205)
self.assertEqual(paired[0]["delta_long_mse"],
                 paired[0]["knockout_long_mse"] - paired[0]["normal_long_mse"])
```

Reject duplicate model-condition pairs, missing `stage3_off`, wrong material counts, nonfinite values, mixed provenance, and zero denominators represented as fabricated percentage values.

- [ ] **Step 2: Run RED pairing tests**

Expected: FAIL because pairing functions are missing.

- [ ] **Step 3: Implement B3a-specific validation, pairing and summaries**

Use these exact groups and metrics:

```python
MATERIAL_GROUPS = {0: "elastic", 1: "plasticine", 2: "sand"}
SUMMARY_GROUPS = ("overall", "elastic", "plasticine", "sand")
STAGE_METRICS = (
    "full_rollout_mse", "short_mse", "mid_mse", "long_mse",
    "gm_mse", "fde", "f24_centroid_error",
    "f24_shape_residual_mse", "penetration_rate", "penetration_depth",
)
```

For each non-normal condition and metric call the existing `paired_delta_summary()` with 10,000 paired bootstrap samples. Do not import the B1b `dynamic_gate_verdict()`.

- [ ] **Step 4: Write failing CSV/report tests**

Assert exact output names and cardinalities:

```text
material_state_stage_knockout_b3a90_raw.csv      246 rows
material_state_stage_knockout_b3a90_paired.csv   205 rows
material_state_stage_activity_b3a90.csv           164 rows
material_state_stage_knockout_b3a90.md
```

The report must state that deltas are `knockout - normal` and `all_off` is not an independently trained baseline.

- [ ] **Step 5: Run RED writer tests**

Expected: FAIL because the writers are missing.

- [ ] **Step 6: Implement CSV and Chinese Markdown writers**

Writers must sort by model and registered condition/stage order, include `checkpoint/config/seed/sample_scope`, and validate the complete table before writing any file.

- [ ] **Step 7: Write failing parser and profile tests for the CLI**

Test:

```python
parsed = build_parser().parse_args([
    "--config", "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
    "--checkpoint", "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors",
    "--output-dir", "results/b3a-stage",
])
self.assertEqual(parsed.bootstrap_samples, 10000)
self.assertEqual(parsed.bootstrap_seed, 0)
```

Also assert `_validate_b0_identity(args, profile="b3a90")` rejects a 45k checkpoint, wrong rank, wrong interval, disabled adapter, or wrong dataset.

- [ ] **Step 8: Run RED CLI tests**

Expected: FAIL because the CLI does not exist.

- [ ] **Step 9: Implement the stage knockout orchestration**

Follow the existing frozen HST diagnostic sequence:

1. Load `TestingConfig`, set `args.resume` to the CLI checkpoint, call `_validate_b0_identity(..., profile="b3a90")`.
2. Build `TrajDataset("test", ...)`, frozen manifest and material records.
3. Select exactly one `start_idx=0` window per model.
4. Load `MDM_ST` once, strict checkpoint load, `eval().requires_grad_(False)`, no `torch.compile`.
5. Enter one `MaterialStateActivityCollector` context for the complete diagnostic.
6. For each model and each registered condition, reset seed, apply `masked_material_state_stages`, call `rollout_condition`, and restore scales.
7. Wrap only the `normal` rollout in `collector.capture(model_name, mat_type, expected_calls_per_stage=20)`.
8. Compute trajectory metrics and write all four outputs only after complete validation.

- [ ] **Step 10: Run GREEN Task 1-2 tests**

```powershell
$env:PYTHONPATH='src;.'
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_material_state_stage_diagnostics src.tests.test_material_response_sweep
```

Expected: PASS.

- [ ] **Step 11: Commit Task 2**

```bash
git add src/diagnose_material_state_stage_knockout.py src/utils/material_state_stage_diagnostics.py src/tests/test_material_state_stage_diagnostics.py
git commit -m "Add frozen B3a stage knockout diagnostic"
```

---

### Task 3: 梯度向量分组与 Cosine 纯工具

**Files:**
- Modify: `src/utils/material_state_stage_diagnostics.py`
- Create: `src/tests/test_material_state_gradient_conflict.py`

**Interfaces:**
- Produces: `adapter_parameter_groups(adapter) -> dict[str, tuple[str, ...]]`
- Produces: `snapshot_adapter_gradients(adapter) -> dict[str, torch.Tensor]`
- Produces: `mean_named_gradients(sums, sample_count) -> dict[str, torch.Tensor]`
- Produces: `summarize_material_gradient_conflict(material_gradients) -> dict`

- [ ] **Step 1: Write failing parameter-group tests**

Require these exact groups:

```python
{
    "all_adapter",
    "state_norm",
    "state_proj",
    "material_proj",
    "output_proj",
    "stage_scales",
}
```

Assert every adapter parameter appears exactly once in a leaf group and all leaf groups concatenate to `all_adapter` in deterministic `named_parameters()` order.

- [ ] **Step 2: Run RED grouping tests**

Expected: FAIL because gradient helpers are missing.

- [ ] **Step 3: Implement deterministic gradient extraction and averaging**

`snapshot_adapter_gradients()` must reject missing or nonfinite gradients. It returns detached CPU float64 tensors keyed by full adapter parameter name. `mean_named_gradients()` divides accumulated sample-weighted sums by a positive integer sample count.

- [ ] **Step 4: Write failing cosine tests**

Use synthetic gradients with exact known directions:

```python
elastic = torch.tensor([1.0, 0.0])
plasticine = torch.tensor([2.0, 0.0])
sand = torch.tensor([-1.0, 0.0])
```

Assert cosine(elastic, plasticine) = +1, cosine(elastic, sand) = -1, symmetry holds, and zero-norm groups return `None` rather than 0.

- [ ] **Step 5: Run RED cosine tests**

Expected: FAIL because the summary function is missing.

- [ ] **Step 6: Implement gradient norms, pairwise cosine and stage-scale values**

The summary payload must contain:

```python
{
    "sample_counts": {"elastic": 52, "plasticine": 56, "sand": 56},
    "groups": {
        group_name: {
            "gradient_norms": {material: float},
            "pairwise_cosine": {
                "elastic__plasticine": float | None,
                "elastic__sand": float | None,
                "plasticine__sand": float | None,
            },
        }
    },
    "stage_scale_gradients": {material: [float, float, float, float]},
}
```

- [ ] **Step 7: Run GREEN Task 3 tests**

```powershell
$env:PYTHONPATH='src;.'
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_material_state_gradient_conflict
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/utils/material_state_stage_diagnostics.py src/tests/test_material_state_gradient_conflict.py
git commit -m "Add B3a material gradient conflict statistics"
```

---

### Task 4: Teacher-Forced 梯度 CLI

**Files:**
- Create: `src/diagnose_material_state_gradient_conflict.py`
- Modify: `src/tests/test_material_state_gradient_conflict.py`

**Interfaces:**
- Produces: `validate_fixed_gradient_windows(dataset, expected_models) -> list[int]`
- Produces: `teacher_forced_one_step(model, batch, device, seed) -> tuple[pred, target]`
- Produces: `run_gradient_conflict_diagnostic(args, checkpoint, output_prefix) -> dict`
- Consumes: Task 3 gradient summary functions

- [ ] **Step 1: Write failing fixed-window validation tests**

Create a synthetic dataset index containing exactly four rows per model at starts `(0, 5, 10, 15)`. Assert 164 selected indices and reject duplicate/missing starts, unknown models, wrong material counts and mixed point counts.

- [ ] **Step 2: Run RED window tests**

Expected: FAIL because fixed-window validation is missing.

- [ ] **Step 3: Implement fixed-window validation**

The validator must derive model names and `start_idx` from dataset records without relying on DataLoader order, return indices sorted by frozen model order then start index, and fail before loading the GPU model if the protocol is incomplete.

- [ ] **Step 4: Write failing teacher-forced forward test**

Use the existing tiny `MDM_ST` test configuration with two points. Verify:

- output shape equals `(B, 1, N, 3)`;
- target is exactly `batch["points_tgt"]`;
- the model input is the last source frame plus deterministic `0.02` Gaussian noise;
- identical seed yields identical prediction;
- backward creates finite gradients on every adapter parameter and no gradients on frozen main parameters.

Before this backward test, fill the tiny adapter's `output_proj.weight` with `1e-3`; otherwise the production zero initialization correctly gives zero first-step gradients to the upstream factorized projections and would not test the trained-checkpoint regime.

- [ ] **Step 5: Run RED forward test**

Expected: FAIL because teacher-forced helper is missing.

- [ ] **Step 6: Implement one-step forward and per-material accumulation**

Use the same call contract as `train.py`:

```python
pred = model(
    model_input,
    timesteps,
    points_src,
    batch["force"], batch["E"], batch["nu"],
    batch["mask"][..., :1], batch["drag_point"],
    batch["floor_height"], batch["gravity"], batch["base_drag_coeff"],
    y=batch.get("mat_type"), null_emb=None,
    start_vel=batch.get("start_vel"),
    points_rest=batch.get("points_rest"),
)
```

Move tensor fields to CUDA, use batch size 1, reset with `seed + selected_index`, and construct:

```python
model_input = points_src[:, -1:].clone()
model_input += torch.randn_like(model_input) * 0.02
loss = F.mse_loss(pred.float(), target.float())
```

Freeze all parameters, then re-enable only `model.dit.material_state_exchange.parameters()`. For each sample, clear adapter grads, backward once, multiply the mean-loss gradient by batch size, accumulate CPU float64 tensors by material, and finally divide by sample count.

- [ ] **Step 7: Write failing report and parser tests**

The CLI requires `--config`, `--checkpoint`, and `--output`. It writes exactly:

```text
material_state_gradient_conflict_b3a90.json
material_state_gradient_conflict_b3a90.md
```

The Markdown must state the 164-window teacher-forced coordinate-MSE protocol and explain that negative cosine indicates conflicting local descent directions, not proof that material experts will improve rollout.

- [ ] **Step 8: Run RED report tests**

Expected: FAIL because writers/parser are missing.

- [ ] **Step 9: Implement strict CLI, JSON and Chinese Markdown output**

Before any GPU work call `_validate_b0_identity(args, profile="b3a90")`. Include checkpoint, config, seed, sample scope, sample counts, loss means, group norms, pairwise cosine and signed stage-scale gradients. Validate all required fields before writing either file.

- [ ] **Step 10: Run GREEN Task 3-4 tests**

```powershell
$env:PYTHONPATH='src;.'
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_material_state_gradient_conflict src.tests.test_material_state_diagnostics
```

Expected: PASS.

- [ ] **Step 11: Commit Task 4**

```bash
git add src/diagnose_material_state_gradient_conflict.py src/tests/test_material_state_gradient_conflict.py
git commit -m "Add B3a material gradient conflict diagnostic"
```

---

### Task 5: 台账、完整验证与服务器命令

**Files:**
- Modify: `实验记录_1.md`

**Interfaces:**
- Consumes: completed implementation commits and exact CLI paths
- Produces: reproducible B3a diagnostic record with results still marked pending

- [ ] **Step 1: Update the B3a ledger entry**

Record:

- design commit `95950da`;
- all implementation commit hashes from Tasks 1-4 obtained with `git log --oneline`;
- frozen checkpoint/config/sample protocol;
- both server commands below;
- output filenames;
- status `实现完成，等待服务器执行`;
- no scientific verdict before output files are returned.

- [ ] **Step 2: Run the medium verification suite**

```powershell
$env:PYTHONPATH='src;.'
D:/miniconda3/envs/physctrl/python.exe -m unittest `
  src.tests.test_material_state_adapter `
  src.tests.test_material_state_diagnostics `
  src.tests.test_material_state_stage_diagnostics `
  src.tests.test_material_state_gradient_conflict `
  src.tests.test_material_response_sweep
D:/miniconda3/envs/physctrl/python.exe -m py_compile `
  src/diagnose_material_state_stage_knockout.py `
  src/diagnose_material_state_gradient_conflict.py `
  src/utils/material_state_stage_diagnostics.py `
  src/tests/test_material_state_stage_diagnostics.py `
  src/tests/test_material_state_gradient_conflict.py
git diff --check
```

Expected: all tests PASS, `py_compile` exit 0, and `git diff --check` has no output.

- [ ] **Step 3: Commit ledger backfill**

```bash
git add -f 实验记录_1.md
git commit -m "Record B3a stage and gradient diagnostic protocol"
```

- [ ] **Step 4: Provide server commands from `src/`**

```bash
python diagnose_material_state_stage_knockout.py \
  --config configs/eval_mm3_b3a_material_state_adapter_90k.yaml \
  --checkpoint outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors \
  --output-dir /root/results/traceformer/material_state_stage_knockout_b3a90 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0

python diagnose_material_state_gradient_conflict.py \
  --config configs/eval_mm3_b3a_material_state_adapter_90k.yaml \
  --checkpoint outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors \
  --output /root/results/traceformer/material_state_gradient_conflict_b3a90
```

- [ ] **Step 5: Stop before push**

Report local commit hashes and `git status --short --branch`. Do not execute `git push`; Will will push or explicitly authorize it separately.
