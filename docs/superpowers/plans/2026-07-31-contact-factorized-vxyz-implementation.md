# Contact 因子化 v_xyz 适配器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个参数量与旧 `v_xyz` 几乎相同、主干初值严格一致的因子化 contact adapter，将 `[d,q]`、`v_y` 和 `[v_x,v_z]` 分开编码，并用零初始化标量门控渐进启用切向速度。

**Architecture:** 新建独立 `FactorizedContactAdapter`，按五维 contact feature 的固定顺序切分三个物理分支。`MDM_ST` 新增 `contact_injection_mode: factorized`，只允许配合 `contact_velocity_mode: xyz`；模型构造时精确复现旧 `Linear(5, latent_dim)` 的全局 RNG 消耗，实际分支在局部 RNG 中初始化，保证公共主干和 step-0 输出与旧 `v_xyz` 位级一致。

**Tech Stack:** Python 3.10、PyTorch、OmegaConf、unittest、safetensors、Accelerate YAML 配置。

---

## 文件结构

- Create: `src/model/contact_adapter.py`
  - 只负责五维 contact feature 的因子化投影、切向门控和参数初始化。
- Modify: `src/model/spacetime.py`
  - 校验新模式、构造 adapter、在条件帧 hidden 上执行残差注入。
- Modify: `src/tests/test_contact.py`
  - adapter 数学行为、梯度路径、RNG 和完整前向等价性测试。
- Create: `src/configs/config_mm3_contact_vxyz_factorized.yaml`
  - 45k 训练筛选臂，严格锚定旧 `config_mm3_contact_vxyz.yaml`。
- Create: `src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml`
  - 对应 checkpoint-45000 的镜像评测配置。
- Modify: `src/tests/test_contact_ablation_configs.py`
  - 配置唯一变量、train/eval 镜像和参数量测试。
- Modify: `src/contact_feature_diagnostics.py`
  - 从 factorized checkpoint 重建有效五列投影并输出 gate/分支范数。
- Modify: `src/tests/test_contact_feature_diagnostics.py`
  - factorized checkpoint 投影与诊断测试。

不修改 `src/utils/contact.py` 中五维特征的定义，不修改 Transformer block、loss、
数据采样和评测指标。

---

### Task 1: 用 TDD 新增独立 FactorizedContactAdapter

**Files:**
- Create: `src/model/contact_adapter.py`
- Modify: `src/tests/test_contact.py`

- [ ] **Step 1: 先写 adapter 的失败测试**

在 `src/tests/test_contact.py` 的 import 区加入：

```python
from model.contact_adapter import FactorizedContactAdapter
```

在 `ContactFeatureTests` 前新增：

```python
class FactorizedContactAdapterTests(unittest.TestCase):
    def test_factorized_adapter_splits_boundary_normal_and_tangential_features(self):
        adapter = FactorizedContactAdapter(latent_dim=1)
        with torch.no_grad():
            adapter.boundary_encoder.weight.copy_(torch.tensor([[10.0, 100.0]]))
            adapter.normal_encoder.weight.copy_(torch.tensor([[1000.0]]))
            adapter.tangential_encoder.weight.copy_(torch.tensor([[1.0, 10.0]]))
            adapter.shared_bias.fill_(7.0)
            adapter.tangential_gate.copy_(torch.atanh(torch.tensor(0.5)))

        # Fixed input order: [d, vx, vy, vz, q].
        features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])
        actual = adapter(features)

        # boundary=10*1+100*5=510, normal=1000*3=3000,
        # tangential=0.5*(1*2+10*4)=21, bias=7.
        torch.testing.assert_close(actual, torch.tensor([[[[3538.0]]]]))

    def test_factorized_adapter_has_vxyz_parameter_budget_plus_one_gate(self):
        adapter = FactorizedContactAdapter(latent_dim=256)

        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            1537,
        )
        self.assertEqual(tuple(adapter.boundary_encoder.weight.shape), (256, 2))
        self.assertEqual(tuple(adapter.normal_encoder.weight.shape), (256, 1))
        self.assertEqual(tuple(adapter.tangential_encoder.weight.shape), (256, 2))
        self.assertEqual(tuple(adapter.shared_bias.shape), (256,))
        self.assertEqual(tuple(adapter.tangential_gate.shape), ())

    def test_factorized_adapter_is_zero_output_but_gate_can_learn(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        with torch.no_grad():
            adapter.tangential_encoder.weight.fill_(1.0)
        features = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0, 5.0]]]],
            requires_grad=True,
        )

        output = adapter(features)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        output.sum().backward()

        self.assertGreater(
            torch.count_nonzero(adapter.boundary_encoder.weight.grad).item(),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(adapter.normal_encoder.weight.grad).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(adapter.tangential_encoder.weight.grad).item(),
            0,
        )
        self.assertNotEqual(adapter.tangential_gate.grad.item(), 0.0)

    def test_factorized_adapter_tangential_weight_learns_after_gate_opens(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        with torch.no_grad():
            adapter.tangential_gate.fill_(0.25)
        features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])

        adapter(features).sum().backward()

        self.assertGreater(
            torch.count_nonzero(adapter.tangential_encoder.weight.grad).item(),
            0,
        )

    def test_factorized_adapter_rejects_non_xyz_feature_width(self):
        adapter = FactorizedContactAdapter(latent_dim=4)

        with self.assertRaisesRegex(ValueError, "five features"):
            adapter(torch.zeros(1, 2, 3, 3))
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

从 `src/` 目录运行：

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact.FactorizedContactAdapterTests -v
```

Expected: `ERROR`，错误包含
`ModuleNotFoundError: No module named 'model.contact_adapter'`。

- [ ] **Step 3: 写最小 adapter 实现**

创建 `src/model/contact_adapter.py`：

```python
"""Factorized contact-state adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedContactAdapter(nn.Module):
    """Project boundary, normal-motion, and tangential-motion features separately."""

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")

        self.boundary_encoder = nn.Linear(2, latent_dim, bias=False)
        self.normal_encoder = nn.Linear(1, latent_dim, bias=False)
        self.tangential_encoder = nn.Linear(2, latent_dim, bias=False)
        self.shared_bias = nn.Parameter(torch.zeros(latent_dim))
        self.tangential_gate = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.boundary_encoder.weight)
        nn.init.zeros_(self.normal_encoder.weight)
        # Keep the default nonzero Kaiming-uniform tangential weight. The zero
        # scalar gate preserves the baseline forward while retaining gate gradient.

    def forward(
        self,
        features: torch.Tensor,
        bias_scale: float = 1.0,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != 5:
            raise ValueError(
                "factorized contact adapter requires five features "
                "[signed_gap, displacement_x, displacement_y, "
                f"displacement_z, proximity]; got {tuple(features.shape)}"
            )

        boundary = features[..., [0, 4]]
        normal = features[..., 2:3]
        tangential = features[..., [1, 3]]
        gate = torch.tanh(self.tangential_gate)
        return (
            self.boundary_encoder(boundary)
            + self.normal_encoder(normal)
            + gate * self.tangential_encoder(tangential)
            + self.shared_bias * float(bias_scale)
        )
```

- [ ] **Step 4: 运行 adapter 测试并确认通过**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact.FactorizedContactAdapterTests -v
```

Expected: 5 tests, `OK`。

- [ ] **Step 5: 提交 adapter 与单元测试**

```bash
git add src/model/contact_adapter.py src/tests/test_contact.py
git commit -m "Add factorized contact adapter"
```

---

### Task 2: 把 factorized 模式接入 MDM_ST，并锁定 RNG 公平性

**Files:**
- Modify: `src/model/spacetime.py:13-17`
- Modify: `src/model/spacetime.py:2550-2692`
- Modify: `src/model/spacetime.py:2922-2951`
- Modify: `src/tests/test_contact.py`

- [ ] **Step 1: 先写模型接线与 RNG 的失败测试**

在 `ContactFeatureTests` 中加入：

```python
    def test_factorized_mode_requires_xyz_velocity(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "factorized"

        with self.assertRaisesRegex(ValueError, "requires.*xyz"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_factorized_mode_builds_adapter_without_legacy_contact_encoder(self):
        cfg = _model_config(True)
        cfg.contact_velocity_mode = "xyz"
        cfg.contact_injection_mode = "factorized"

        model = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        self.assertTrue(hasattr(model, "contact_adapter"))
        self.assertFalse(hasattr(model, "contact_encoder"))

    def test_factorized_and_legacy_vxyz_share_exact_trunk_initialization(self):
        legacy_cfg = _model_config(True)
        legacy_cfg.contact_velocity_mode = "xyz"
        factorized_cfg = _model_config(True)
        factorized_cfg.contact_velocity_mode = "xyz"
        factorized_cfg.contact_injection_mode = "factorized"

        torch.manual_seed(1234)
        legacy = MDM_ST(8, 1, n_feats=3, model_config=legacy_cfg).eval()
        torch.manual_seed(1234)
        factorized = MDM_ST(8, 1, n_feats=3, model_config=factorized_cfg).eval()

        legacy_dit = legacy.dit.state_dict()
        factorized_dit = factorized.dit.state_dict()
        self.assertEqual(legacy_dit.keys(), factorized_dit.keys())
        for name in legacy_dit:
            torch.testing.assert_close(
                legacy_dit[name],
                factorized_dit[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_factorized_and_legacy_vxyz_match_step_zero_forward_exactly(self):
        legacy_cfg = _model_config(True)
        legacy_cfg.contact_velocity_mode = "xyz"
        factorized_cfg = _model_config(True)
        factorized_cfg.contact_velocity_mode = "xyz"
        factorized_cfg.contact_injection_mode = "factorized"

        torch.manual_seed(1234)
        legacy = MDM_ST(8, 1, n_feats=3, model_config=legacy_cfg).eval()
        torch.manual_seed(1234)
        factorized = MDM_ST(8, 1, n_feats=3, model_config=factorized_cfg).eval()
        batch = _small_model_batch()

        with torch.no_grad():
            legacy_output = legacy(**batch)
            factorized_output = factorized(**batch)

        torch.testing.assert_close(
            legacy_output,
            factorized_output,
            rtol=0,
            atol=0,
        )
```

- [ ] **Step 2: 运行新增测试并确认模式尚未支持**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact.ContactFeatureTests.test_factorized_mode_requires_xyz_velocity \
  tests.test_contact.ContactFeatureTests.test_factorized_mode_builds_adapter_without_legacy_contact_encoder \
  tests.test_contact.ContactFeatureTests.test_factorized_and_legacy_vxyz_share_exact_trunk_initialization \
  tests.test_contact.ContactFeatureTests.test_factorized_and_legacy_vxyz_match_step_zero_forward_exactly -v
```

Expected: FAIL/ERROR，原因是 `factorized` 不在允许的 injection mode 中。

- [ ] **Step 3: 在 MDM_ST 中注册并构造 factorized adapter**

在 `src/model/spacetime.py` import 区加入：

```python
from model.contact_adapter import FactorizedContactAdapter
```

将允许模式校验改为：

```python
        if self.contact_injection_mode not in (
            'separate',
            'shared',
            'factorized',
        ):
            raise ValueError(
                "contact_injection_mode must be one of "
                "('separate', 'shared', 'factorized'); "
                f"got {self.contact_injection_mode!r}"
            )
```

保留随后现有的：

```python
        self.contact_velocity_mode = model_config.get(
            'contact_velocity_mode',
            'vertical',
        )
        self.contact_feature_names = contact_feature_names(
            self.contact_velocity_mode
        )
```

并且必须在这两行之后加入：

```python
        if (
            self.contact_injection_mode == 'factorized'
            and not self.contact_particle_cond
        ):
            raise ValueError(
                "contact_injection_mode='factorized' requires "
                "contact_particle_cond=true"
            )
        if (
            self.contact_injection_mode == 'factorized'
            and self.contact_velocity_mode != 'xyz'
        ):
            raise ValueError(
                "contact_injection_mode='factorized' requires "
                "contact_velocity_mode='xyz'"
            )
```

在当前 `contact_encoder` 构造位置保留 `separate` 原逻辑，并新增：

```python
        elif (
            self.contact_particle_cond
            and self.contact_injection_mode == 'factorized'
        ):
            # Advance global RNG exactly as legacy Linear(5, latent_dim), then
            # construct the actual adapter without perturbing the trunk RNG.
            legacy_rng_anchor = nn.Linear(
                len(self.contact_feature_names),
                self.latent_dim,
            )
            del legacy_rng_anchor
            with torch.random.fork_rng(devices=[]):
                self.contact_adapter = FactorizedContactAdapter(
                    self.latent_dim
                )
```

- [ ] **Step 4: 在条件帧 hidden 注入 factorized 输出**

把当前仅处理 `separate` 的分支扩为：

```python
        if (
            self.contact_particle_cond
            and self.contact_injection_mode in ('separate', 'factorized')
        ):
            if floor_height is None:
                raise ValueError("contact_particle_cond requires floor_height")
            contact_features = build_contact_features(
                init_pc_cond,
                floor_height,
                start_velocity=start_vel,
                sigma=self.contact_feature_sigma,
                velocity_mode=self.contact_velocity_mode,
            )
            contact_features = apply_contact_feature_mask(
                contact_features,
                self.contact_feature_mask,
            )
            if self.contact_injection_mode == 'separate':
                contact_features = contact_features.to(
                    dtype=self.contact_encoder.weight.dtype
                )
                contact_bias = self.contact_encoder.bias
                if contact_bias is not None:
                    contact_bias = contact_bias * self.contact_bias_scale
                contact_hidden = F.linear(
                    contact_features,
                    self.contact_encoder.weight,
                    contact_bias,
                )
            else:
                contact_features = contact_features.to(
                    dtype=self.contact_adapter.shared_bias.dtype
                )
                contact_hidden = self.contact_adapter(
                    contact_features,
                    bias_scale=self.contact_bias_scale,
                )
            contact_hidden = contact_hidden.to(hidden_states.dtype)
            n_contact_frames = contact_hidden.shape[1]
            hidden_states = torch.cat([
                hidden_states[:, :n_contact_frames] + contact_hidden,
                hidden_states[:, n_contact_frames:],
            ], dim=1)
```

- [ ] **Step 5: 运行新增测试和旧 contact 回归测试**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest tests.test_contact -v
```

Expected: 全部 `OK`，包括旧 `separate/shared/v_xyz` 测试。

- [ ] **Step 6: 提交模型接线**

```bash
git add src/model/spacetime.py src/tests/test_contact.py
git commit -m "Integrate factorized contact conditioning"
```

---

### Task 3: 新增严格镜像的 45k 训练和评测配置

**Files:**
- Create: `src/configs/config_mm3_contact_vxyz_factorized.yaml`
- Create: `src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml`
- Modify: `src/tests/test_contact_ablation_configs.py`

- [ ] **Step 1: 先写配置唯一变量和参数量失败测试**

在 `ContactAblationConfigTests` 中加入：

```python
    def test_factorized_vxyz_changes_only_output_and_injection_mode(self):
        anchor = _load_structured(
            "config_mm3_contact_vxyz.yaml",
            TrainingConfig,
        )
        arm = _load_structured(
            "config_mm3_contact_vxyz_factorized.yaml",
            TrainingConfig,
        )
        ignored = {
            "output_dir",
            "model_config.contact_injection_mode",
        }

        self.assertEqual(arm.stop_after_steps, 45000)
        self.assertEqual(arm.model_config.contact_velocity_mode, "xyz")
        self.assertEqual(arm.model_config.contact_injection_mode, "factorized")
        self.assertEqual(
            _without_paths(arm, ignored),
            _without_paths(anchor, ignored),
        )

        anchor.model_config.cond_frames = anchor.get("input_frames", 5)
        arm.model_config.cond_frames = arm.get("input_frames", 5)
        torch.manual_seed(1234)
        anchor_model = MDM_ST(
            8, anchor.output_frames, 3, anchor.model_config
        )
        torch.manual_seed(1234)
        arm_model = MDM_ST(
            8, arm.output_frames, 3, arm.model_config
        )
        anchor_contact_params = sum(
            parameter.numel()
            for name, parameter in anchor_model.named_parameters()
            if name.startswith("contact_encoder.")
        )
        factorized_contact_params = sum(
            parameter.numel()
            for name, parameter in arm_model.named_parameters()
            if name.startswith("contact_adapter.")
        )
        self.assertEqual(anchor_contact_params, 1536)
        self.assertEqual(factorized_contact_params, 1537)

    def test_factorized_vxyz_eval_mirrors_training_and_45k_checkpoint(self):
        train = _load_structured(
            "config_mm3_contact_vxyz_factorized.yaml",
            TrainingConfig,
        )
        eval_cfg = _load_structured(
            "eval_mm3_contact_vxyz_factorized_45k.yaml",
            TestingConfig,
        )

        self.assertEqual(
            eval_cfg.resume,
            "outputs/mm3_contact_vxyz_factorized_8L/"
            "checkpoint-45000/model.safetensors",
        )
        self.assertEqual(
            eval_cfg.model_config.contact_injection_mode,
            "factorized",
        )
        self.assertEqual(
            eval_cfg.model_config.contact_velocity_mode,
            train.model_config.contact_velocity_mode,
        )
        self.assertEqual(eval_cfg.input_frames, train.input_frames)
        self.assertEqual(eval_cfg.output_frames, train.output_frames)
        train.train_dataset.input_frames = train.input_frames
        train.train_dataset.output_frames = train.output_frames
        eval_cfg.train_dataset.input_frames = eval_cfg.input_frames
        eval_cfg.train_dataset.output_frames = eval_cfg.output_frames
        self.assertEqual(
            OmegaConf.to_container(
                eval_cfg.model_config,
                resolve=True,
            ),
            OmegaConf.to_container(
                train.model_config,
                resolve=True,
            ),
        )
```

并在文件顶部加入：

```python
import torch
```

- [ ] **Step 2: 运行测试并确认因配置文件不存在而失败**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact_ablation_configs.ContactAblationConfigTests.test_factorized_vxyz_changes_only_output_and_injection_mode \
  tests.test_contact_ablation_configs.ContactAblationConfigTests.test_factorized_vxyz_eval_mirrors_training_and_45k_checkpoint -v
```

Expected: `FileNotFoundError`，指向两个新 YAML。

- [ ] **Step 3: 创建训练配置**

复制 `src/configs/config_mm3_contact_vxyz.yaml` 为
`src/configs/config_mm3_contact_vxyz_factorized.yaml`，只修改：

```yaml
# Factorized v_xyz screening arm.
# Anchor = config_mm3_contact_vxyz.yaml. Only output_dir and
# model_config.contact_injection_mode differ.
output_dir: ./outputs/mm3_contact_vxyz_factorized_8L
```

并在 `model_config.contact_velocity_mode: xyz` 后加入：

```yaml
  contact_injection_mode: factorized
```

不得改变 `max_train_steps: 90000`、`stop_after_steps: 45000` 或任何其他字段。

- [ ] **Step 4: 创建评测配置**

复制 `src/configs/eval_mm3_contact_vxyz_45k.yaml` 为
`src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml`，只修改：

```yaml
# Eval mirror for config_mm3_contact_vxyz_factorized.yaml at checkpoint 45000.
resume: 'outputs/mm3_contact_vxyz_factorized_8L/checkpoint-45000/model.safetensors'
vis_dir: 'vis_results_mm3_contact_vxyz_factorized_45k'
```

并在 `model_config.contact_velocity_mode: xyz` 后加入：

```yaml
  contact_injection_mode: factorized
```

- [ ] **Step 5: 运行配置测试**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact_ablation_configs -v
```

Expected: 全部 `OK`。

- [ ] **Step 6: 提交配置和测试**

```bash
git add \
  src/configs/config_mm3_contact_vxyz_factorized.yaml \
  src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml \
  src/tests/test_contact_ablation_configs.py
git commit -m "Add factorized vxyz screening configs"
```

---

### Task 4: 支持 factorized checkpoint 的特征贡献诊断

**Files:**
- Modify: `src/contact_feature_diagnostics.py`
- Modify: `src/tests/test_contact_feature_diagnostics.py`

- [ ] **Step 1: 先写有效投影和分支统计失败测试**

在 `src/tests/test_contact_feature_diagnostics.py` 的 import 中加入：

```python
    factorized_branch_hidden_norms,
    load_factorized_contact_stats,
```

在测试类中加入：

```python
    def test_loads_factorized_effective_projection_in_vxyz_order(self):
        state = {
            "model.contact_adapter.boundary_encoder.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            "model.contact_adapter.normal_encoder.weight": torch.tensor(
                [[5.0], [6.0]]
            ),
            "model.contact_adapter.tangential_encoder.weight": torch.tensor(
                [[7.0, 8.0], [9.0, 10.0]]
            ),
            "model.contact_adapter.shared_bias": torch.tensor([11.0, 12.0]),
            "model.contact_adapter.tangential_gate": torch.atanh(
                torch.tensor(0.5)
            ),
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="factorized",
            feature_dim=5,
        )

        expected = torch.tensor([
            [1.0, 3.5, 5.0, 4.0, 2.0],
            [3.0, 4.5, 6.0, 5.0, 4.0],
        ])
        torch.testing.assert_close(weight, expected)
        torch.testing.assert_close(bias, torch.tensor([11.0, 12.0]))

    def test_reports_factorized_gate_and_branch_norms(self):
        state = {
            "model.contact_adapter.boundary_encoder.weight": torch.ones(2, 2),
            "model.contact_adapter.normal_encoder.weight": torch.ones(2, 1) * 2,
            "model.contact_adapter.tangential_encoder.weight": torch.ones(2, 2) * 3,
            "model.contact_adapter.shared_bias": torch.ones(2) * 4,
            "model.contact_adapter.tangential_gate": torch.atanh(
                torch.tensor(0.25)
            ),
        }

        stats = load_factorized_contact_stats(state)

        self.assertAlmostEqual(stats["effective_gate"], 0.25, places=6)
        self.assertAlmostEqual(stats["boundary_weight_norm"], 2.0, places=6)
        self.assertAlmostEqual(
            stats["normal_weight_norm"],
            float(torch.linalg.vector_norm(torch.ones(2, 1) * 2)),
            places=6,
        )
        self.assertAlmostEqual(stats["tangential_weight_norm"], 6.0, places=6)
        self.assertAlmostEqual(
            stats["shared_bias_norm"],
            float(torch.linalg.vector_norm(torch.ones(2) * 4)),
            places=6,
        )

    def test_computes_factorized_branch_hidden_norms(self):
        state = {
            "model.contact_adapter.boundary_encoder.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "model.contact_adapter.normal_encoder.weight": torch.tensor(
                [[1.0], [1.0]]
            ),
            "model.contact_adapter.tangential_encoder.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "model.contact_adapter.shared_bias": torch.zeros(2),
            "model.contact_adapter.tangential_gate": torch.atanh(
                torch.tensor(0.5)
            ),
        }
        features = torch.tensor([
            [3.0, 4.0, 5.0, 6.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
        ])

        norms = factorized_branch_hidden_norms(features, state)

        expected_boundary = (
            torch.linalg.vector_norm(torch.tensor([3.0, 0.0]))
            + torch.linalg.vector_norm(torch.tensor([0.0, 4.0]))
        ) / 2
        expected_normal = (
            torch.linalg.vector_norm(torch.tensor([5.0, 5.0]))
            + 0.0
        ) / 2
        expected_tangential = (
            torch.linalg.vector_norm(torch.tensor([2.0, 3.0]))
            + 0.0
        ) / 2
        self.assertAlmostEqual(
            norms["boundary_hidden_norm"],
            expected_boundary.item(),
            places=6,
        )
        self.assertAlmostEqual(
            norms["normal_hidden_norm"],
            expected_normal.item(),
            places=6,
        )
        self.assertAlmostEqual(
            norms["tangential_hidden_norm"],
            expected_tangential.item(),
            places=6,
        )
```

- [ ] **Step 2: 运行诊断测试并确认 factorized 尚未支持**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact_feature_diagnostics.ContactFeatureDiagnosticTests.test_loads_factorized_effective_projection_in_vxyz_order \
  tests.test_contact_feature_diagnostics.ContactFeatureDiagnosticTests.test_reports_factorized_gate_and_branch_norms \
  tests.test_contact_feature_diagnostics.ContactFeatureDiagnosticTests.test_computes_factorized_branch_hidden_norms -v
```

Expected: import error 或 `contact_injection_mode` ValueError。

- [ ] **Step 3: 重建 factorized 的有效五列线性投影**

在 `src/contact_feature_diagnostics.py` import 区加入：

```python
import torch.nn.functional as F
```

在 `src/contact_feature_diagnostics.py` 新增：

```python
def _factorized_contact_tensors(state):
    return {
        "boundary": _state_tensor_by_suffix(
            state, "contact_adapter.boundary_encoder.weight"
        ),
        "normal": _state_tensor_by_suffix(
            state, "contact_adapter.normal_encoder.weight"
        ),
        "tangential": _state_tensor_by_suffix(
            state, "contact_adapter.tangential_encoder.weight"
        ),
        "bias": _state_tensor_by_suffix(
            state, "contact_adapter.shared_bias"
        ),
        "gate": _state_tensor_by_suffix(
            state, "contact_adapter.tangential_gate"
        ),
    }


def load_factorized_contact_stats(state):
    tensors = _factorized_contact_tensors(state)
    return {
        "effective_gate": torch.tanh(tensors["gate"]).item(),
        "boundary_weight_norm": torch.linalg.vector_norm(
            tensors["boundary"]
        ).item(),
        "normal_weight_norm": torch.linalg.vector_norm(
            tensors["normal"]
        ).item(),
        "tangential_weight_norm": torch.linalg.vector_norm(
            tensors["tangential"]
        ).item(),
        "shared_bias_norm": torch.linalg.vector_norm(
            tensors["bias"]
        ).item(),
    }


def factorized_branch_hidden_norms(features, state):
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError(
            "factorized branch diagnostics require features shaped (tokens, 5)"
        )
    tensors = _factorized_contact_tensors(state)
    boundary = F.linear(features[:, [0, 4]], tensors["boundary"])
    normal = F.linear(features[:, 2:3], tensors["normal"])
    tangential = (
        torch.tanh(tensors["gate"])
        * F.linear(features[:, [1, 3]], tensors["tangential"])
    )
    return {
        "boundary_hidden_norm": torch.linalg.vector_norm(
            boundary, dim=-1
        ).mean().item(),
        "normal_hidden_norm": torch.linalg.vector_norm(
            normal, dim=-1
        ).mean().item(),
        "tangential_hidden_norm": torch.linalg.vector_norm(
            tangential, dim=-1
        ).mean().item(),
    }
```

在 `load_contact_projection` 中加入：

```python
    elif injection_mode == "factorized":
        if feature_dim != 5:
            raise ValueError(
                "factorized contact projection requires feature_dim=5"
            )
        tensors = _factorized_contact_tensors(state)
        hidden_dim = tensors["boundary"].shape[0]
        weight = tensors["boundary"].new_zeros(hidden_dim, 5)
        weight[:, [0, 4]] = tensors["boundary"]
        weight[:, 2:3] = tensors["normal"]
        weight[:, [1, 3]] = (
            torch.tanh(tensors["gate"]) * tensors["tangential"]
        )
        bias = tensors["bias"]
```

同步把非法模式错误文本改成包含 `factorized`。

- [ ] **Step 4: 在命令行输出 gate 与分支范数**

在 `main` 加载 projection 后加入：

```python
    factorized_stats = (
        load_factorized_contact_stats(state)
        if injection_mode == "factorized"
        else None
    )
```

在 encoder bias 输出后加入：

```python
    if factorized_stats is not None:
        print(
            "factorized adapter: "
            + " ".join(
                f"{name}={value:.6g}"
                for name, value in factorized_stats.items()
            )
        )
```

在每个 material 的 `features` 拼接完成后加入：

```python
        if injection_mode == "factorized":
            branch_norms = factorized_branch_hidden_norms(features, state)
            print(
                "factorized hidden contribution: "
                + " ".join(
                    f"{name}={value:.6g}"
                    for name, value in branch_norms.items()
                )
            )
```

- [ ] **Step 5: 运行全部诊断测试**

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact_feature_diagnostics -v
```

Expected: 全部 `OK`。

- [ ] **Step 6: 提交诊断支持**

```bash
git add \
  src/contact_feature_diagnostics.py \
  src/tests/test_contact_feature_diagnostics.py
git commit -m "Diagnose factorized contact checkpoints"
```

---

### Task 5: 全量验证与实验命令核对

**Files:**
- Verify only; no production files should change.

- [ ] **Step 1: 运行语法检查**

从仓库根目录运行：

```bash
D:/miniconda3/envs/physctrl/python.exe -m py_compile \
  src/model/contact_adapter.py \
  src/model/spacetime.py \
  src/contact_feature_diagnostics.py \
  src/count_params.py
```

Expected: exit code 0，无输出。

- [ ] **Step 2: 运行三组相关测试**

从 `src/` 目录运行：

```bash
D:/miniconda3/envs/physctrl/python.exe -m unittest \
  tests.test_contact \
  tests.test_contact_ablation_configs \
  tests.test_contact_feature_diagnostics -v
```

Expected: 全部 `OK`，无失败或 error。

- [ ] **Step 3: 核参数脚本**

从 `src/` 目录运行：

```bash
D:/miniconda3/envs/physctrl/python.exe count_params.py
```

Expected:

- 命令成功；
- 原有架构参数报告不漂移；
- 单元测试已确认旧 `v_xyz` contact 参数为 1536，新 factorized 为 1537。

- [ ] **Step 4: 检查配置差异**

```bash
git diff --no-index \
  src/configs/config_mm3_contact_vxyz.yaml \
  src/configs/config_mm3_contact_vxyz_factorized.yaml
```

Expected: 只有文件头说明、`output_dir` 和
`contact_injection_mode: factorized`。`git diff --no-index` 在发现预期差异时
返回 exit code 1，这是正常行为。

```bash
git diff --no-index \
  src/configs/eval_mm3_contact_vxyz_45k.yaml \
  src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml
```

Expected: 只有文件头说明、`resume`、`vis_dir` 和
`contact_injection_mode: factorized`。exit code 1 同样表示“文件存在差异”，
不表示验证失败。

- [ ] **Step 5: 检查提交和工作区边界**

```bash
git status --short
git log -5 --oneline
```

Expected: 本功能涉及的 tracked 文件均已提交；用户已有的未跟踪 PPT、图片、结果和
临时目录没有被暂存或修改。不得执行 `git push`。

- [ ] **Step 6: 给出服务器训练命令**

从服务器仓库的 `src/` 目录运行：

```bash
accelerate launch \
  --config_file configs/acc/1gpu.yaml \
  train.py \
  --config configs/config_mm3_contact_vxyz_factorized.yaml
```

45000 步结束后运行：

```bash
python eval.py \
  --config configs/eval_mm3_contact_vxyz_factorized_45k.yaml

python contact_feature_diagnostics.py \
  --config configs/eval_mm3_contact_vxyz_factorized_45k.yaml
```

不得在本地启动 GPU 训练，不得自动 push。
