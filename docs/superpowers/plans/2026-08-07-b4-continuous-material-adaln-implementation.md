# B4 Continuous Material AdaLN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 MM3 串行 8 层主干、contact、数据和 loss 的前提下，为连续 `E/nu` 增加 joint AdaLN conditioning 路径，并准备严格匹配的 45k screening 与 ON/OFF 评测配置。

**Architecture:** 新增一个独立 `ContinuousMaterialConditioner`，把归一化后的 `[log10(E), nu]` 经 `2→64→256` joint MLP 编成 material embedding，并加入 `timestep_emb + class_emb`。现有 8 层 `CogVideoXLayerNormZero` 读取该 embedding；原 `E/nu/class` token 路径全部保留，不添加 particle-space residual adapter。

**Tech Stack:** Python 3.10、PyTorch、OmegaConf、diffusers CogVideoX blocks、`unittest`、Accelerate、safetensors。

## Global Constraints

- 当前比较基线固定为 `config_mm3_contact_cond.yaml` / `mm3_contact_cond@45k`。
- 架构之外的 MM3 数据、随机窗口、8 层串行主干、contact、loss、optimizer、batch、LR、scheduler 均不得改变。
- 原 `E_cond_encoder`、`nu_cond_encoder`、class token 和 class embedding 路径必须保留。
- `material_adaln_cond` 与 `material_state_adapter` 必须互斥；不得把 B3 和 B4 混合。
- B4 关闭时不得实例化新增参数；旧 config、旧 checkpoint 和旧 forward 行为必须兼容。
- `Linear(64,256)` 的 weight 与 bias 必须 zero-init；B4 初始函数与 baseline 等价。
- B4 conditioner 参数量必须精确为 `16,832`。
- 训练保持 `max_train_steps: 90000`，screening 使用 `stop_after_steps: 45000`，不得缩短 scheduler horizon。
- 代码、config 和测试提交均使用 Will 的 git 身份，不添加任何 AI co-author/contributor，不 push。
- 供 Will 审阅的设计与实验文档使用中文。
- 除非步骤明确写“从 `src/` 运行”，所有 `git` 命令均从 worktree 根目录运行；所有 `PYTHONPATH=. python ...` 命令均从 `src/` 运行。

---

## File Structure

- Create `src/model/material_adaln.py`: 只负责连续材料值的校验、归一化和 joint MLP 编码。
- Modify `src/model/spacetime.py`: 读取 B4 配置、整理 `material_values`、把 material embedding 加入 block conditioning embedding。
- Create `src/tests/test_material_adaln.py`: conditioner 数学、zero-init、梯度、参数量和异常输入测试。
- Create `src/tests/test_material_adaln_integration.py`: `MDM_ST` 初始等价、runtime knockout、旧配置兼容和 B3/B4 互斥测试。
- Create `src/tests/test_b4_material_adaln_config.py`: 训练/评测唯一变量和镜像协议审计。
- Create `src/configs/config_mm3_b4_material_adaln.yaml`: B4 45k screening 训练配置。
- Create `src/configs/eval_mm3_b4_material_adaln_45k.yaml`: B4 ON 评测配置。
- Create `src/configs/eval_mm3_b4_material_adaln_45k_off.yaml`: 同 checkpoint B4 OFF 机制诊断。
- Modify `src/count_params.py`: B4 精确参数预算和主干不变审计。
- Modify `实验记录_1.md`: 回填 B3b `close`，预注册 B4 的假设、唯一变量、三道 gate 和代码 commit。

---

### Task 1: Continuous Material Conditioner

**Files:**
- Create: `src/model/material_adaln.py`
- Create: `src/tests/test_material_adaln.py`

**Interfaces:**
- Consumes: `material_values: torch.Tensor`，shape 必须为 `(B,2)`，两列依次是 `log10(E)` 和 `nu`。
- Produces: `ContinuousMaterialConditioner.forward(material_values) -> torch.Tensor`，shape `(B, output_dim)`。
- Public constructor:

```python
ContinuousMaterialConditioner(
    output_dim: int,
    hidden_dim: int = 64,
    e_center: float = 5.5,
    e_scale: float = 1.0,
    nu_center: float = 0.25,
    nu_scale: float = 0.15,
)
```

- Test-visible helper: `normalize_material_values(material_values) -> torch.Tensor`。

- [ ] **Step 1: Write failing validation and normalization tests**

Create `src/tests/test_material_adaln.py` with tests equivalent to:

```python
import unittest

import torch

from model.material_adaln import ContinuousMaterialConditioner


class ContinuousMaterialConditionerTest(unittest.TestCase):
    def test_normalizes_log_e_and_nu_jointly(self):
        module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
        values = torch.tensor([[4.5, 0.10], [6.5, 0.40]])
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        torch.testing.assert_close(
            module.normalize_material_values(values), expected
        )

    def test_rejects_invalid_constructor_scales_and_dims(self):
        for kwargs in (
            {"output_dim": 0},
            {"output_dim": 8, "hidden_dim": 0},
            {"output_dim": 8, "e_scale": 0.0},
            {"output_dim": 8, "nu_scale": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ContinuousMaterialConditioner(**kwargs)

    def test_rejects_wrong_shape_and_non_finite_values(self):
        module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            module(torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            module(torch.tensor([[5.5, float("nan")]]))
```

- [ ] **Step 2: Run the tests and verify RED**

Run from `src/`:

```bash
PYTHONPATH=. python -m unittest tests.test_material_adaln -v
```

Expected: FAIL because `model.material_adaln` does not exist.

- [ ] **Step 3: Implement input validation and normalization**

Create `src/model/material_adaln.py` with this structure:

```python
import torch
from torch import nn
from torch.nn import functional as F


class ContinuousMaterialConditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        e_center: float = 5.5,
        e_scale: float = 1.0,
        nu_center: float = 0.25,
        nu_scale: float = 0.15,
    ):
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        if e_scale <= 0 or nu_scale <= 0:
            raise ValueError("material normalization scales must be positive")
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.e_center = float(e_center)
        self.e_scale = float(e_scale)
        self.nu_center = float(nu_center)
        self.nu_scale = float(nu_scale)
        self.input_proj = nn.Linear(2, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def normalize_material_values(self, material_values: torch.Tensor) -> torch.Tensor:
        if material_values.ndim != 2 or material_values.shape[1] != 2:
            raise ValueError("material_values must have shape (B, 2)")
        if not torch.isfinite(material_values).all():
            raise ValueError("material_values must be finite")
        e = (material_values[:, :1] - self.e_center) / self.e_scale
        nu = (material_values[:, 1:2] - self.nu_center) / self.nu_scale
        return torch.cat((e, nu), dim=-1)

    def forward(self, material_values: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_material_values(material_values)
        hidden = F.silu(self.input_proj(normalized))
        return self.output_proj(hidden)
```

- [ ] **Step 4: Add zero-init, parameter-count, and gradient tests**

Append tests that verify:

```python
def test_zero_init_outputs_exact_zero_and_budget_is_exact(self):
    module = ContinuousMaterialConditioner(output_dim=256, hidden_dim=64)
    output = module(torch.tensor([[5.5, 0.25], [6.0, 0.40]]))
    self.assertTrue(torch.equal(output, torch.zeros_like(output)))
    self.assertEqual(sum(p.numel() for p in module.parameters()), 16832)

def test_both_continuous_inputs_receive_gradient_after_output_path_is_active(self):
    module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
    with torch.no_grad():
        module.output_proj.weight.fill_(0.1)
    values = torch.tensor([[5.0, 0.20]], requires_grad=True)
    module(values).sum().backward()
    self.assertGreater(values.grad[0, 0].abs().item(), 0.0)
    self.assertGreater(values.grad[0, 1].abs().item(), 0.0)
```

The gradient test intentionally activates `output_proj`; with strict zero-init, the first backward only updates the final projection and cannot yet propagate into `E/nu` or `input_proj`.

- [ ] **Step 5: Run Task 1 tests**

```bash
PYTHONPATH=. python -m unittest tests.test_material_adaln -v
python -m py_compile model/material_adaln.py tests/test_material_adaln.py
```

Expected: all Task 1 tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/model/material_adaln.py src/tests/test_material_adaln.py
git commit -m "Add continuous material conditioner"
```

---

### Task 2: Integrate Continuous Material AdaLN into MDM_ST

**Files:**
- Modify: `src/model/spacetime.py:2034-2288`
- Modify: `src/model/spacetime.py:2360-2450`
- Modify: `src/model/spacetime.py:2621-2860`
- Modify: `src/model/spacetime.py:2870-3135`
- Create: `src/tests/test_material_adaln_integration.py`

**Interfaces:**
- Consumes Task 1 `ContinuousMaterialConditioner`.
- Adds `material_values: Optional[torch.Tensor]` to the existing inner transformer forward path; shape `(B,2)`.
- Adds these `SpaitalTemporalTransformer.__init__` arguments with backward-compatible defaults:

```python
material_adaln_cond: bool = False
material_adaln_hidden_dim: int = 64
material_adaln_e_center: float = 5.5
material_adaln_e_scale: float = 1.0
material_adaln_nu_center: float = 0.25
material_adaln_nu_scale: float = 0.15
material_adaln_runtime_scale: float = 1.0
```

- [ ] **Step 1: Write failing integration tests**

Create `src/tests/test_material_adaln_integration.py` with a `small_config()` derived from existing B3 tests. Cover:

```python
def test_disabled_config_has_no_material_adaln_parameters():
    model = MDM_ST(2, 1, 3, small_config(material_adaln=False))
    self.assertFalse(any("material_conditioner" in key for key in model.state_dict()))

def test_zero_initialized_b4_matches_baseline_forward():
    torch.manual_seed(17)
    baseline = MDM_ST(2, 1, 3, small_config(material_adaln=False)).eval()
    torch.manual_seed(17)
    candidate = MDM_ST(2, 1, 3, small_config(material_adaln=True)).eval()
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    self.assertEqual(incompatible.unexpected_keys, [])
    self.assertTrue(all("material_conditioner" in key for key in incompatible.missing_keys))
    with torch.no_grad():
        expected = baseline(**small_inputs())
        actual = candidate(**small_inputs())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

def test_runtime_zero_disables_nonzero_conditioner():
    torch.manual_seed(23)
    baseline = MDM_ST(2, 1, 3, small_config(material_adaln=False)).eval()
    torch.manual_seed(23)
    model = MDM_ST(2, 1, 3, small_config(material_adaln=True)).eval()
    model.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        model.dit.material_conditioner.output_proj.weight.fill_(0.01)
    inputs = small_inputs()
    model.dit.material_adaln_runtime_scale = 1.0
    enabled = model(**inputs)
    model.dit.material_adaln_runtime_scale = 0.0
    disabled = model(**inputs)
    self.assertFalse(torch.equal(enabled, disabled))
    torch.testing.assert_close(
        disabled, baseline(**inputs), rtol=0.0, atol=1e-7
    )

def test_rejects_b3_and_b4_combination():
    config = small_config(material_adaln=True)
    config.material_state_adapter = True
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
        MDM_ST(2, 1, 3, config)
```

Also test that active B4 with missing `material_values` raises, changing `E` or `nu` after making `output_proj` nonzero changes output, and all state-dict entries shared by baseline/B4 receive identical initial values under the same manual seed. The last check guards against the new module shifting shared CPU RNG.

- [ ] **Step 2: Run integration tests and verify RED**

```bash
PYTHONPATH=. python -m unittest tests.test_material_adaln_integration -v
```

Expected: FAIL because B4 config and module integration do not exist.

- [ ] **Step 3: Add constructor fields and compatibility validation**

In `MDM_ST.__init__`:

```python
self.material_adaln_cond = bool(
    model_config.get("material_adaln_cond", False)
)
if self.material_adaln_cond and self.material_state_adapter:
    raise ValueError(
        "material_adaln_cond and material_state_adapter are mutually exclusive"
    )
```

Pass all B4 settings into `SpaitalTemporalTransformer`. In its constructor, instantiate only when enabled, inside `torch.random.fork_rng(devices=[])` so initialization of shared parameters is not shifted:

```python
self.material_adaln_enabled = bool(material_adaln_cond)
self.material_adaln_runtime_scale = float(material_adaln_runtime_scale)
if self.material_adaln_enabled:
    with torch.random.fork_rng(devices=[]):
        self.material_conditioner = ContinuousMaterialConditioner(
            output_dim=time_embed_dim,
            hidden_dim=material_adaln_hidden_dim,
            e_center=material_adaln_e_center,
            e_scale=material_adaln_e_scale,
            nu_center=material_adaln_nu_center,
            nu_scale=material_adaln_nu_scale,
        )
```

- [ ] **Step 4: Build and route material values**

In `MDM_ST.forward()`, before `E = E.unsqueeze(1)`, construct material values whenever B3 or B4 requires them:

```python
if self.material_state_adapter or self.material_adaln_cond:
    material_values = torch.cat(
        (E.reshape(bs, -1)[:, :1], nu.reshape(bs, -1)[:, :1]),
        dim=1,
    )
```

Do not require `mat_type` for B4. B4's joint MLP only consumes continuous values; class conditioning continues through existing paths.

Add `material_values` to `dit_kwargs` when B4 is active. Preserve B3's `material_classes` routing unchanged.

- [ ] **Step 5: Add material embedding to block conditioning**

In `SpaitalTemporalTransformer.forward()`, after the existing class embedding addition and before block execution:

```python
if self.material_adaln_enabled:
    if material_values is None:
        raise ValueError("continuous material AdaLN requires material_values")
    material_emb = self.material_conditioner(material_values)
    material_emb = material_emb.to(dtype=emb.dtype, device=emb.device)
    emb = emb + self.material_adaln_runtime_scale * material_emb
```

Do not modify `SpatialTemporalTransformerBlock.forward()` or create per-layer copies of the conditioner.

- [ ] **Step 6: Run new and compatibility tests**

```bash
PYTHONPATH=. python -m unittest \
  tests.test_material_adaln \
  tests.test_material_adaln_integration \
  tests.test_material_state_adapter \
  tests.test_material_stage_gate \
  tests.test_v11a_config \
  tests.test_v11a_contact_config -v

python -m py_compile \
  model/material_adaln.py \
  model/spacetime.py \
  tests/test_material_adaln.py \
  tests/test_material_adaln_integration.py
```

Expected: all selected tests PASS; old B3/v11a behavior remains valid.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/model/spacetime.py src/tests/test_material_adaln_integration.py
git commit -m "Inject continuous material conditions through AdaLN"
```

---

### Task 3: Register 45k Screening and ON/OFF Eval Configs

**Files:**
- Create: `src/configs/config_mm3_b4_material_adaln.yaml`
- Create: `src/configs/eval_mm3_b4_material_adaln_45k.yaml`
- Create: `src/configs/eval_mm3_b4_material_adaln_45k_off.yaml`
- Create: `src/tests/test_b4_material_adaln_config.py`

**Interfaces:**
- Training config is a literal baseline copy plus B4 fields, `output_dir`, and `stop_after_steps`.
- ON/OFF evals load the same `checkpoint-45000/model.safetensors`.
- OFF differs from ON only by `vis_dir` and `model_config.material_adaln_runtime_scale`.

- [ ] **Step 1: Write failing config audit tests**

Use the existing `flatten()` pattern and assert the exact train diff:

```python
expected_train_diff = {
    "output_dir",
    "stop_after_steps",
    "model_config.material_adaln_cond",
    "model_config.material_adaln_hidden_dim",
    "model_config.material_adaln_e_center",
    "model_config.material_adaln_e_scale",
    "model_config.material_adaln_nu_center",
    "model_config.material_adaln_nu_scale",
    "model_config.material_adaln_runtime_scale",
}
```

Assert:

```python
self.assertEqual(candidate.max_train_steps, 90000)
self.assertEqual(candidate.stop_after_steps, 45000)
self.assertEqual(candidate.model_config.n_layers, 8)
self.assertEqual(
    candidate.model_config.transformer_block,
    "SpatialTemporalTransformerBlock",
)
```

For ON eval, compare resolved `model_config` against training config and require test dataset plus `checkpoint-45000`. For OFF eval, require exact differences:

```python
{"vis_dir", "model_config.material_adaln_runtime_scale"}
```

- [ ] **Step 2: Run config tests and verify RED**

```bash
PYTHONPATH=. python -m unittest tests.test_b4_material_adaln_config -v
```

Expected: FAIL because the three B4 configs do not exist.

- [ ] **Step 3: Create training config**

Copy `config_mm3_contact_cond.yaml` without changing existing values. Set:

```yaml
output_dir: ./outputs/mm3_b4_material_adaln_8L
max_train_steps: 90000
stop_after_steps: 45000

model_config:
  material_adaln_cond: true
  material_adaln_hidden_dim: 64
  material_adaln_e_center: 5.5
  material_adaln_e_scale: 1.0
  material_adaln_nu_center: 0.25
  material_adaln_nu_scale: 0.15
  material_adaln_runtime_scale: 1.0
```

All baseline comments may be shortened, but values and key presence must remain identical except for the audited set.

- [ ] **Step 4: Create ON and OFF eval configs**

ON:

```yaml
resume: outputs/mm3_b4_material_adaln_8L/checkpoint-45000/model.safetensors
vis_dir: vis_results_mm3_b4_material_adaln_45k
```

OFF uses the same checkpoint:

```yaml
vis_dir: vis_results_mm3_b4_material_adaln_45k_off
model_config:
  material_adaln_runtime_scale: 0.0
```

Both use `mm3_data/mm3_test`, `input_frames: 5`, `output_frames: 1`, `use_diffusion: false`, and `num_inference_steps: 1`.

- [ ] **Step 5: Run config and regression tests**

```bash
PYTHONPATH=. python -m unittest \
  tests.test_b4_material_adaln_config \
  tests.test_b3_material_state_config \
  tests.test_contact_ablation_configs -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  src/configs/config_mm3_b4_material_adaln.yaml \
  src/configs/eval_mm3_b4_material_adaln_45k.yaml \
  src/configs/eval_mm3_b4_material_adaln_45k_off.yaml \
  src/tests/test_b4_material_adaln_config.py
git commit -m "Register B4 material AdaLN screening"
```

---

### Task 4: Parameter Audit, Experiment Registration, and Final Verification

**Files:**
- Modify: `src/count_params.py`
- Modify: `实验记录_1.md`
- Test: `src/tests/test_material_adaln.py`
- Test: `src/tests/test_material_adaln_integration.py`
- Test: `src/tests/test_b4_material_adaln_config.py`

**Interfaces:**
- Add `validate_material_adaln_parameter_budget(baseline, candidate) -> dict`.
- The report must prove exactly one `ContinuousMaterialConditioner`, 8 original serial blocks, and signed parameter delta `16,832`.
- `实验记录_1.md` becomes the formal B4 preregistration and cites the exact implementation commits produced by Tasks 1–3.

- [ ] **Step 1: Add a failing exact parameter-budget test**

Add a test that builds baseline and B4 MM3 models and calls:

```python
report = validate_material_adaln_parameter_budget(baseline, candidate)
self.assertEqual(report["signed_delta"], 16832)
self.assertEqual(report["conditioner_params"], 16832)
self.assertEqual(report["conditioner_count"], 1)
self.assertEqual(report["block_count"], 8)
self.assertEqual(
    report["block_types"],
    ["SpatialTemporalTransformerBlock"] * 8,
)
```

Run:

```bash
PYTHONPATH=. python -m unittest tests.test_material_adaln -v
```

Expected: FAIL because the validation function does not exist.

- [ ] **Step 2: Implement parameter-budget validation**

In `count_params.py`, import `ContinuousMaterialConditioner`, build a B4 MM3 model with `material_adaln_cond=True`, and verify:

```python
conditioners = [
    module for module in candidate.modules()
    if isinstance(module, ContinuousMaterialConditioner)
]
if len(conditioners) != 1:
    raise RuntimeError(...)
signed_delta = count(candidate) - count(baseline)
conditioner_params = count(conditioners[0])
if signed_delta != 16832 or conditioner_params != 16832:
    raise RuntimeError(...)
```

Use `find_blocks(candidate)` to assert 8 blocks and exact block type. Print baseline, candidate, delta and percentage.

- [ ] **Step 3: Run parameter audit**

From `src/` with the local CPU environment:

```bash
python count_params.py
```

Expected B4 section:

```text
baseline=... candidate=... delta=+16,832
conditioner=16,832 copies=1 blocks=8
```

- [ ] **Step 4: Update the formal experiment ledger**

In `实验记录_1.md`:

1. Mark B3b `close`, including near-identity gates and dev-test result.
2. Add B4 to the experiment overview.
3. Record the design commit `436b564` and the exact Task 1–3 commit hashes returned by git.
4. Record baseline, unique variable, 45k budget, Gate A/B/C, stop conditions and the three config paths.
5. Mark status as `implementation complete; awaiting server 45k screening`.

Do not add a result before training/eval exists, and do not claim counterfactual physical correctness from the B2 sweep.

- [ ] **Step 5: Run verification at the user-selected test intensity**

Minimum commands for **medium**:

```bash
PYTHONPATH=. python -m unittest \
  tests.test_material_adaln \
  tests.test_material_adaln_integration \
  tests.test_b4_material_adaln_config \
  tests.test_material_state_adapter \
  tests.test_material_stage_gate \
  tests.test_b3_material_state_config \
  tests.test_contact_ablation_configs \
  tests.test_v11a_config \
  tests.test_v11a_contact_config -v

python -m py_compile \
  model/material_adaln.py \
  model/spacetime.py \
  count_params.py \
  tests/test_material_adaln.py \
  tests/test_material_adaln_integration.py \
  tests/test_b4_material_adaln_config.py

python count_params.py
git diff --check
```

For **high**, add all `src/tests/test_*material*.py`, all config tests affected by `MDM_ST`, and an independent code review. For **extreme high**, add the complete local test suite and a server-side single-batch forward/backward smoke test under bf16/compile before starting the full run.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/count_params.py src/tests/test_material_adaln.py 实验记录_1.md
git commit -m "Audit and register B4 material AdaLN"
```

- [ ] **Step 7: Final branch audit**

```bash
git status --short --branch
git log --oneline --decorate -6
git diff main...HEAD --check
git diff main...HEAD --stat
```

Expected: worktree clean; branch contains the design plus four implementation commits; no checkpoint, CSV, output directory, `CLAUDE.md`, or AI attribution is present.
