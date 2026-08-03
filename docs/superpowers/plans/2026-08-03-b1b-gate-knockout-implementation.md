# B1b HST Gate Knockout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 B1b checkpoint-90000 实现严格配对的 inference-only HST gate knockout 工具，输出逐模型原始指标、配对差值、分材质/分 `E` 统计和预注册停止判定。

**Architecture:** 模型与数据只加载一次；每个模型在相同输入和随机种子下依次运行 `normal/all_off/stage0_off/stage1_off/stage2_off`。纯函数模块负责 gate 上下文、轨迹指标、配对统计与报告，CLI 负责冻结协议校验、真实 rollout 和文件写入；训练路径及模型 `forward` 保持不变。

**Tech Stack:** Python 3.10、PyTorch、NumPy、OmegaConf、safetensors、现有 `TrajDataset`/`TrajPipeline`、`unittest`。

## Global Constraints

- checkpoint 固定为 `outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors`。
- config 固定为 `src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml`，`use_diffusion=false`、`floor_projection=false`。
- test split 固定为 `src/configs/mm3_test_split.json` 的 41 个模型，只取每模型 `start_idx=0`。
- 模型只加载一次；五个条件必须在同一进程、同一 batch、同一材料条件、同一 seed 下重跑。
- gate mask 乘在 checkpoint 的训练后 gate 上；任何正常或异常退出都必须精确恢复原 gate。
- 不修改 `src/model/hybrid_state.py`、`src/model/spacetime.py`、训练路径、checkpoint 或 eval config。
- 不新增第三方依赖；真实 41-model rollout 只在服务器 GPU 上执行。
- 输出写入命令指定的结果目录，不纳入 commit。
- commit author 使用 Will，不添加任何 AI co-author/contributor；不执行 git push。

---

## File Structure

- Create `src/utils/hybrid_state_gate_knockout.py`: 条件 mask、gate 恢复上下文、轨迹指标、配对行、分组统计、预注册判定和 CSV/Markdown writer。
- Create `src/tests/test_hybrid_state_gate_knockout.py`: 纯函数、异常恢复、统计口径、writer 与 CLI 集成测试。
- Create `src/diagnose_hybrid_state_gate_knockout.py`: 复用 B1b 冻结协议，加载模型一次并运行 41×5 次严格配对 rollout。

### Task 1: Gate Mask 与异常安全上下文

**Files:**
- Create: `src/utils/hybrid_state_gate_knockout.py`
- Create: `src/tests/test_hybrid_state_gate_knockout.py`

**Interfaces:**
- Produces: `KNOCKOUT_CONDITIONS: tuple[tuple[str, tuple[int, int, int, int]], ...]`
- Produces: `masked_feedback_gates(exchange: nn.Module, mask: Sequence[int]) -> ContextManager[torch.Tensor]`
- Produces: `reset_inference_seed(seed: int, device: torch.device) -> None`

- [ ] **Step 1: 写 mask 顺序和 gate 值的失败测试**

```python
class GateMaskTests(unittest.TestCase):
    def test_conditions_are_pre_registered_and_ordered(self):
        self.assertEqual(
            KNOCKOUT_CONDITIONS,
            (
                ("normal", (1, 1, 1, 1)),
                ("all_off", (0, 0, 0, 0)),
                ("stage0_off", (0, 1, 1, 1)),
                ("stage1_off", (1, 0, 1, 1)),
                ("stage2_off", (1, 1, 0, 1)),
            ),
        )

    def test_mask_multiplies_trained_values_and_restores_them(self):
        exchange = SimpleNamespace(
            feedback_gates=torch.nn.Parameter(
                torch.tensor([0.04, 0.015, -0.013, -0.002])
            )
        )
        original = exchange.feedback_gates.detach().clone()
        with masked_feedback_gates(exchange, (0, 1, 0, 1)) as applied:
            torch.testing.assert_close(
                exchange.feedback_gates,
                torch.tensor([0.0, 0.015, 0.0, -0.002]),
            )
            torch.testing.assert_close(applied, exchange.feedback_gates)
        self.assertTrue(torch.equal(exchange.feedback_gates.detach(), original))
```

另测 mask 长度不为 4、包含非 0/1 值、gate 非有限值、gate 数量不为 4 时抛出 `ValueError`。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.GateMaskTests -v
```

Expected: FAIL，原因是 `src.utils.hybrid_state_gate_knockout` 尚不存在。

- [ ] **Step 3: 实现条件和上下文管理器**

```python
from contextlib import contextmanager

KNOCKOUT_CONDITIONS = (
    ("normal", (1, 1, 1, 1)),
    ("all_off", (0, 0, 0, 0)),
    ("stage0_off", (0, 1, 1, 1)),
    ("stage1_off", (1, 0, 1, 1)),
    ("stage2_off", (1, 1, 0, 1)),
)

@contextmanager
def masked_feedback_gates(exchange, mask):
    gates = exchange.feedback_gates
    if gates.numel() != 4 or not torch.isfinite(gates.detach()).all():
        raise ValueError("feedback_gates must contain four finite values")
    mask_tensor = torch.as_tensor(mask, device=gates.device, dtype=gates.dtype)
    if mask_tensor.shape != gates.shape or not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
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
```

- [ ] **Step 4: 写异常恢复与 seed 重置的失败测试**

```python
def test_mask_restores_original_gates_when_rollout_raises(self):
    exchange = SimpleNamespace(
        feedback_gates=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    )
    with self.assertRaisesRegex(RuntimeError, "rollout failed"):
        with masked_feedback_gates(exchange, (0, 1, 1, 1)):
            raise RuntimeError("rollout failed")
    torch.testing.assert_close(
        exchange.feedback_gates,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

@mock.patch("src.utils.hybrid_state_gate_knockout.torch.cuda.manual_seed_all")
@mock.patch("src.utils.hybrid_state_gate_knockout.torch.manual_seed")
def test_reset_seed_resets_cpu_and_cuda_for_each_condition(self, cpu_seed, cuda_seed):
    reset_inference_seed(7, torch.device("cuda"))
    cpu_seed.assert_called_once_with(7)
    cuda_seed.assert_called_once_with(7)
```

- [ ] **Step 5: 实现 seed 重置并确认 GREEN**

```python
def reset_inference_seed(seed, device):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
```

Run:

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.GateMaskTests -v
python -m py_compile src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
```

Expected: PASS。

- [ ] **Step 6: 提交 Task 1**

```bash
git add src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git commit -m "Add safe HST gate knockout masks"
```

### Task 2: 轨迹与接触指标

**Files:**
- Modify: `src/utils/hybrid_state_gate_knockout.py`
- Modify: `src/tests/test_hybrid_state_gate_knockout.py`

**Interfaces:**
- Produces: `KNOCKOUT_METRICS: tuple[str, ...]`
- Produces: `trajectory_knockout_metrics(pred: torch.Tensor, gt: torch.Tensor, input_frames: int, floor_height: float | torch.Tensor) -> dict[str, float]`
- Consumes: `src.utils.eval_metrics.per_window_metrics()`。

- [ ] **Step 1: 写分段、GM、Procrustes 和穿透指标的失败测试**

构造 `(25, 8, 3)` 的 GT，令预测帧 5--10 偏移 1、11--17 偏移 2、18--24 偏移 3，并令两个预测点低于 `floor=0`：

```python
def test_trajectory_metrics_exclude_history_and_use_absolute_frame_segments(self):
    gt = torch.zeros(25, 8, 3)
    gt[:, :, 0] = torch.arange(8, dtype=torch.float32)
    pred = gt.clone()
    pred[5:11] += 1.0
    pred[11:18] += 2.0
    pred[18:25] += 3.0
    pred[24, :2, 1] = -0.5

    metrics = trajectory_knockout_metrics(pred, gt, input_frames=5, floor_height=0.0)

    self.assertAlmostEqual(metrics["short_mse"], torch.mean((pred[5:11] - gt[5:11]) ** 2).item())
    self.assertAlmostEqual(metrics["mid_mse"], torch.mean((pred[11:18] - gt[11:18]) ** 2).item())
    self.assertAlmostEqual(metrics["long_mse"], torch.mean((pred[18:25] - gt[18:25]) ** 2).item())
    self.assertAlmostEqual(metrics["full_rollout_mse"], torch.mean((pred[5:] - gt[5:]) ** 2).item())
    self.assertAlmostEqual(metrics["penetration_rate"], 2.0 / (20 * 8))
    self.assertAlmostEqual(metrics["penetration_depth"], 1.0 / (20 * 8))
```

另测 shape 不一致、`T != 25`、`input_frames != 5`、非有限输入和 floor 非标量时拒绝。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.TrajectoryMetricTests -v
```

Expected: FAIL，原因是 `trajectory_knockout_metrics` 尚不存在。

- [ ] **Step 3: 实现指标函数**

```python
KNOCKOUT_METRICS = (
    "full_rollout_mse", "short_mse", "mid_mse", "long_mse", "gm_mse",
    "fde", "f24_centroid_error", "f24_shape_residual_mse",
    "penetration_rate", "penetration_depth",
)

def trajectory_knockout_metrics(pred, gt, input_frames, floor_height):
    # 严格验证 pred/gt=(25,N,3)、input_frames=5 和 finite。
    pred_f, gt_f = pred.float(), gt.to(pred.device).float()
    frame_mse = (pred_f[input_frames:] - gt_f[input_frames:]).square().mean((1, 2))
    base = per_window_metrics(pred_f, gt_f, input_frames)
    centroid, _, _, shape = base["proc"][24]
    floor = torch.as_tensor(floor_height, device=pred.device, dtype=pred_f.dtype)
    if floor.numel() != 1 or not torch.isfinite(floor).all():
        raise ValueError("floor_height must be one finite scalar")
    penetration = torch.clamp(floor.reshape(()) - pred_f[input_frames:, :, 1], min=0)
    result = {
        "full_rollout_mse": float(frame_mse.mean()),
        "short_mse": float(frame_mse[0:6].mean()),
        "mid_mse": float(frame_mse[6:13].mean()),
        "long_mse": float(frame_mse[13:20].mean()),
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
```

- [ ] **Step 4: 运行 Task 2 测试并提交**

Run:

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.TrajectoryMetricTests -v
python -m unittest src.tests.test_eval_contact_metrics -v
```

Expected: PASS。

Commit:

```bash
git add src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git commit -m "Add gate knockout trajectory metrics"
```

### Task 3: 配对统计、E 分层与预注册判定

**Files:**
- Modify: `src/utils/hybrid_state_gate_knockout.py`
- Modify: `src/tests/test_hybrid_state_gate_knockout.py`

**Interfaces:**
- Produces: `validate_raw_rows(rows: list[dict]) -> list[dict]`
- Produces: `build_paired_rows(rows: list[dict]) -> list[dict]`
- Produces: `paired_delta_summary(normal: np.ndarray, knockout: np.ndarray, samples: int, seed: int) -> dict[str, float | int | None]`
- Produces: `summarize_paired_rows(rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]`
- Produces: `dynamic_gate_verdict(summary_rows: list[dict]) -> dict[str, object]`

- [ ] **Step 1: 写 205/164 行完整性和配对符号测试**

构造 41 个模型、五条件 synthetic rows。每个 knockout 指标设为 normal 的 0.9 倍，断言：

```python
paired = build_paired_rows(raw_rows)
self.assertEqual(len(paired), 41 * 4)
row = paired[0]
self.assertLess(row["delta_full_rollout_mse"], 0.0)
self.assertAlmostEqual(row["relative_change_pct_full_rollout_mse"], -10.0)
```

另测缺模型、重复条件、条件集合错误、材质元数据在条件间变化、非有限值和 normal 为非正数时拒绝。

- [ ] **Step 2: 运行配对测试并确认 RED**

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.PairedStatisticTests -v
```

Expected: FAIL，原因是配对函数尚不存在。

- [ ] **Step 3: 实现 raw 校验和宽格式 paired rows**

`validate_raw_rows()` 必须验证 205 行、41 个唯一模型、13/14/14 材质数，以及每个模型恰有五个条件。`build_paired_rows()` 对每个模型的四个 knockout 输出一行，并为每个指标写入：

```python
paired_row[f"normal_{metric}"] = normal_value
paired_row[f"knockout_{metric}"] = knockout_value
paired_row[f"delta_{metric}"] = knockout_value - normal_value
paired_row[f"relative_change_pct_{metric}"] = (
    (knockout_value - normal_value) / normal_value * 100.0
    if normal_value > 0
    else 0.0 if knockout_value == 0
    else None
)
```

`normal=0, knockout>0` 时相对变化数学上无定义，CSV 留空并由绝对差值表达退化；不得写成无穷大或伪造百分比。

- [ ] **Step 4: 写材质内 E 分层和 bootstrap 测试**

```python
summary = summarize_paired_rows(paired, bootstrap_samples=200, bootstrap_seed=11)
groups = {(row["group"], row["condition"], row["metric"]) for row in summary}
self.assertIn(("plasticine", "stage0_off", "fde"), groups)
self.assertIn(("plasticine_low_E", "stage0_off", "fde"), groups)
self.assertIn(("plasticine_high_E", "stage0_off", "fde"), groups)
```

断言每材质按本材质 `log10_e` 中位数分层；等于中位数归 `low_E`。统计字段固定为 `n_models/normal_mean/knockout_mean/mean_delta/median_delta/relative_change_pct/improved_count/ci_low/ci_high`。
另测 penetration normal/knockout 均为 0 时相对变化为 `0.0`，normal 为 0 而 knockout 大于 0 时相对变化为 `None`、绝对差值和 bootstrap CI 仍正常输出。

- [ ] **Step 5: 实现分组统计**

实现只对配对差值重采样的 `paired_delta_summary()`，不复用要求 `normal_mean>0` 的 B0 helper。这样 contact penetration 在某些子组 baseline 为 0 时仍可统计：

```python
delta = knockout - normal
indices = np.random.default_rng(seed).integers(0, delta.size, size=(samples, delta.size))
bootstrap_means = delta[indices].mean(axis=1)
stats["median_delta"] = float(np.median(delta))
stats["improved_count"] = int(np.sum(delta < 0))
stats["relative_change_pct"] = (
    float(delta.mean() / normal.mean() * 100.0)
    if normal.mean() > 0
    else 0.0 if knockout.mean() == 0
    else None
)
stats["ci_low"], stats["ci_high"] = np.percentile(bootstrap_means, (2.5, 97.5))
```

overall 直接模型等权；材质组按 `mat_type`；`low_E/high_E` 只在材质内部划分，不跨材质比较绝对 `E`。

- [ ] **Step 6: 写严格自动判定的失败测试**

构造两个情形：

1. `stage0_off` 在 plasticine 的 `long_mse/fde/centroid` 中两项 `<=-5%`、各至少 8/14 改善且 median<0；sand 对其中一项 `>=+5%`、至少 8/14 退化且 median>0；overall `full/FDE < +10%`、penetration rate/depth `< +25%`，应返回 `proceed_dynamic_gate=True`。
2. 只有均值改善、模型胜数不足或 sand 无相反响应，应返回 `False`。

- [ ] **Step 7: 实现预注册判定**

只检查 `stage0_off` 和 `stage2_off`。自动通过条件固定为：

- plasticine 的 `long_mse/fde/f24_centroid_error` 至少两项 `relative_change_pct <= -5`、`improved_count >= 8`、`median_delta < 0`；
- sand 在上述已通过项中至少一项 `relative_change_pct >= +5`、退化模型数 `>=8`、`median_delta > 0`；
- overall `full_rollout_mse` 与 `fde` 均 `< +10%`；
- overall `penetration_rate` 与 `penetration_depth` 均 `< +25%`；若 normal 为 0，则只有 knockout 仍为 0 才通过该项。

返回：

```python
{
    "proceed_dynamic_gate": bool,
    "qualifying_stage": str | None,
    "plasticine_metrics": tuple[str, ...],
    "sand_opposite_metrics": tuple[str, ...],
    "reasons": tuple[str, ...],
}
```

- [ ] **Step 8: 运行 Task 3 测试并提交**

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.PairedStatisticTests -v
python -m py_compile src/utils/hybrid_state_gate_knockout.py
```

Expected: PASS。

Commit:

```bash
git add src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git commit -m "Add paired HST knockout statistics"
```

### Task 4: CSV、Markdown 与 CLI 集成

**Files:**
- Modify: `src/utils/hybrid_state_gate_knockout.py`
- Create: `src/diagnose_hybrid_state_gate_knockout.py`
- Modify: `src/tests/test_hybrid_state_gate_knockout.py`

**Interfaces:**
- Produces: `write_raw_csv(path: Path, rows: list[dict]) -> None`
- Produces: `write_paired_csv(path: Path, rows: list[dict]) -> None`
- Produces: `write_knockout_report(path: Path, raw_rows: list[dict], summary_rows: list[dict], metadata: dict, original_gates: Sequence[float], verdict: dict) -> None`
- Produces: `run_gate_knockout_diagnostics(args, checkpoint: Path, output_dir: Path, bootstrap_samples: int, bootstrap_seed: int, config_path: Path | None = None) -> list[dict]`
- Reuses: `validate_diagnostic_config()`、`load_frozen_test_manifest()`、`select_start_zero_indices()`、`_validate_records()`、`_rollout_autocast_context()`、`_build_raw_reference()`、`rollout_condition()`。

- [ ] **Step 1: 写 writer 字段和输出路径失败测试**

固定输出名：

```python
def _output_paths(output_dir):
    return (
        output_dir / "hybrid_state_gate_knockout_b1b_90k_raw.csv",
        output_dir / "hybrid_state_gate_knockout_b1b_90k_paired.csv",
        output_dir / "hybrid_state_gate_knockout_b1b_90k.md",
    )
```

测试 CSV 每行包含 `checkpoint/config/seed/sample_scope`，Markdown 包含原 gate、205/164 行验收、overall/材质/`E` 分层表和最终 `proceed/close` 判定。

- [ ] **Step 2: 实现 writer 并确认 GREEN**

使用 `csv.DictWriter`，不在 CSV 前加入非表格注释；元数据作为重复字段写入 raw/paired rows。Markdown 的相对变化统一带符号，明确“负值=改善”。

Run:

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout.WriterTests -v
```

Expected: PASS。

- [ ] **Step 3: 写 CLI 单模型集成失败测试**

mock `TrajDataset/MDM_ST/TrajPipeline/load_file/rollout_condition`，构造冻结 manifest 的 41 个模型但使用轻量假轨迹。验证：

- `MDM_ST` 和 checkpoint loader 各调用一次；
- 每模型条件顺序严格为五个预注册条件；
- `reset_inference_seed(args.seed, cuda)` 调用 `41×5` 次；
- 每次 rollout 观察到的 gate 等于 `original × mask`；
- 执行完成后模型 gate 与 original bitwise 相等；
- 返回 205 raw rows，并写出三份文件；
- 不调用 `torch.compile`。

- [ ] **Step 4: 实现 CLI 主循环**

核心循环固定为：

```python
original_gates = model.dit.hybrid_state_exchange.feedback_gates.detach().clone()
for batch, _ in dataloader:
    gt = _build_raw_reference(batch, args.train_dataset)
    for condition, mask in KNOCKOUT_CONDITIONS:
        reset_inference_seed(args.seed, device)
        with masked_feedback_gates(model.dit.hybrid_state_exchange, mask):
            with _rollout_autocast_context(device):
                pred = rollout_condition(
                    pipeline, batch, args,
                    record.log10_e, record.nu, record.mat_type,
                )
        metrics = trajectory_knockout_metrics(
            pred[0], gt[0], input_frames=5,
            floor_height=batch["floor_height"][0],
        )
        raw_rows.append({
            "model": model_name,
            "mat_type": record.mat_type,
            "log10_e": record.log10_e,
            "nu": record.nu,
            "condition": condition,
            **metrics,
        })
```

完成后依次调用 `validate_raw_rows()`、`build_paired_rows()`、`summarize_paired_rows()`、`dynamic_gate_verdict()` 和三个 writer。

- [ ] **Step 5: 实现 CLI 参数和严格协议校验**

参数固定为：

```text
--config              required
--checkpoint          required
--output-dir          required
--bootstrap-samples   default 10000, >0
--bootstrap-seed      default 0, >=0
```

`main()` 用 `OmegaConf.merge(OmegaConf.structured(TestingConfig), OmegaConf.load(config))` 构造参数并把 `resume` 覆盖为 CLI checkpoint。调用现有 `validate_diagnostic_config()` 拒绝错误 checkpoint、数据路径、帧数、block、contact 开关和 floor projection。

- [ ] **Step 6: 运行 CLI 与回归测试并提交**

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout -v
python -m unittest src.tests.test_hybrid_state_diagnostics -v
python -m unittest src.tests.test_material_condition_diagnostics -v
python -m py_compile src/diagnose_hybrid_state_gate_knockout.py src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
```

Expected: 全部 PASS。

Commit:

```bash
git add src/diagnose_hybrid_state_gate_knockout.py src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git commit -m "Add B1b HST gate knockout diagnostic"
```

### Task 5: 最终审查与服务器命令

**Files:**
- Modify only if verification finds a defect: `src/diagnose_hybrid_state_gate_knockout.py`, `src/utils/hybrid_state_gate_knockout.py`, `src/tests/test_hybrid_state_gate_knockout.py`

**Interfaces:**
- Produces: 一个可在服务器执行、不会训练或修改 checkpoint 的诊断命令。

- [ ] **Step 1: 运行完整验证**

从项目根目录运行：

```bash
python -m unittest src.tests.test_hybrid_state_gate_knockout src.tests.test_hybrid_state_diagnostics src.tests.test_material_condition_diagnostics -v
python -m py_compile src/diagnose_hybrid_state_gate_knockout.py src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git diff --check
```

Expected: 测试全部 PASS、`py_compile` exit 0、`git diff --check` 无输出。

- [ ] **Step 2: 审查实验不变量**

逐项确认：

- 没有生产模型和训练文件 diff；
- `KNOCKOUT_CONDITIONS` 恰为五个条件，无 `stage3_off`；
- raw/paired 行数保护为 205/164；
- 配对差值定义为 `knockout-normal`；
- 同条件异常不会遗留 masked gate；
- 结果路径受 `.gitignore` 保护；
- commit 中不含 checkpoint、CSV、PPT、临时文件或 AI attribution。

- [ ] **Step 3: 如验证修复产生 diff，单独提交**

```bash
git add src/diagnose_hybrid_state_gate_knockout.py src/utils/hybrid_state_gate_knockout.py src/tests/test_hybrid_state_gate_knockout.py
git commit -m "Tighten HST gate knockout protocol guards"
```

没有修复 diff 时不创建空提交。

- [ ] **Step 4: 给出服务器执行命令，不代替用户 push**

从服务器项目根目录执行：

```bash
PYTHONPATH=src python src/diagnose_hybrid_state_gate_knockout.py \
  --config src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml \
  --checkpoint outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors \
  --output-dir src/results/hybrid_state_gate_knockout_b1b_90k \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0
```

预期生成 205-row raw CSV、164-row paired CSV 和一份中文 Markdown。真实结果返回后再更新 `实验记录.md`；在未得到真实 rollout 前不登记实验结论。
