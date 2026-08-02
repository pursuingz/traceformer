# B1b v11a + contact_cond 组合实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个只把 v11a MC-HST 叠加到原版 additive `contact_cond` 上的 mm3 训练/评测臂，并用自动化测试证明数据、损失、采样、主干和训练预算没有发生未声明变化。

**Architecture:** 复用现有 `contact_encoder` 与 `HybridStateExchange`，不增加第三个模块。五帧 contact-conditioned particle hidden 继续经过 8 个原串行 block；共享 v11a exchange 在第 2/4/6/8 层后读取五帧历史并只反馈单个 prediction frame。

**Tech Stack:** Python 3.10、PyTorch、OmegaConf、Diffusers、`unittest`、Accelerate、YAML。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-08-02-b1b-v11a-contact-factorial-design.md`。
- 主比较：11（v11a + contact）vs 01（contact only）；00/10 只用于 2×2 解释。
- 训练锚点：`src/configs/config_mm3_contact_cond.yaml`。
- 只允许改变 `output_dir`、`stop_after_steps`、`transformer_block` 和三个固定 hybrid-state 字段。
- `max_train_steps` 必须保持 90000；`stop_after_steps` 固定 45000，只控制筛选停止点。
- contact 必须保持 `signed_gap + vertical_displacement + proximity`、`separate` additive 注入与 zero initialization。
- 8 层原串行 `SpatialTemporalTransformerBlock`、latent 256、数据、loss、batch、lr、scheduler、seed 全冻结。
- 默认不得修改 `src/model/spacetime.py`、`src/model/hybrid_state.py`、`src/train.py`、`src/eval.py` 或 dataset。
- 若组合测试失败，先保留失败证据，再只做接口级最小修复；不得改变算法定义。
- Python 验证使用本机 `D:/miniconda3/envs/physctrl/python.exe`。
- commit author 使用 Will 本人，不添加任何 AI co-author/contributor；不得 push。

---

## File Map

- Create `src/configs/config_mm3_v11a_contact_cond_8L.yaml`：B1b 45k screening 训练配置。
- Create `src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml`：严格镜像的 45k 评测配置。
- Create `src/tests/test_v11a_contact_config.py`：配置唯一变量、结构化合并、组合路由、梯度与参数预算测试。
- Modify `实验记录.md`：实现完成后回填实际代码 commit；不改变“待训练”的实验结论。
- Do not modify model/training/eval implementation unless a failing integration test proves an existing interface defect.

### Task 1: 用配置测试锁定 B1b 唯一变量

**Files:**
- Create: `src/tests/test_v11a_contact_config.py`
- Create: `src/configs/config_mm3_v11a_contact_cond_8L.yaml`
- Create: `src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml`

**Interfaces:**
- Consumes: `TrainingConfig`、`TestingConfig`、`config_mm3_contact_cond.yaml`、`eval_mm3_contact_cond_45k.yaml`。
- Produces: 可由 OmegaConf structured merge 加载的 B1b train/eval config；Task 2 直接从训练 config 构造组合模型。

- [ ] **Step 1: 先写缺少配置时必然失败的隔离测试**

创建 `src/tests/test_v11a_contact_config.py`：

```python
import copy
import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from options import TestingConfig, TrainingConfig


CONFIG_DIR = SRC_DIR / "configs"
TRAIN_BASE = CONFIG_DIR / "config_mm3_contact_cond.yaml"
TRAIN_ARM = CONFIG_DIR / "config_mm3_v11a_contact_cond_8L.yaml"
EVAL_BASE = CONFIG_DIR / "eval_mm3_contact_cond_45k.yaml"
EVAL_ARM = CONFIG_DIR / "eval_mm3_v11a_contact_cond_8L_45k.yaml"

HYBRID_CONFIG = {
    "hybrid_state_dim": 64,
    "hybrid_state_heads": 4,
    "hybrid_state_interval": 2,
}


def load_plain(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


class V11aContactConfigTests(unittest.TestCase):
    def assert_config_exists(self, path):
        self.assertTrue(path.is_file(), f"missing config: {path}")

    def assert_isolated_arm(self, baseline_path, arm_path, allowed_top_level):
        self.assert_config_exists(arm_path)
        baseline = copy.deepcopy(load_plain(baseline_path))
        arm = copy.deepcopy(load_plain(arm_path))

        for key, expected in allowed_top_level.items():
            self.assertEqual(baseline.pop(key, None), expected[0])
            self.assertEqual(arm.pop(key, None), expected[1])

        baseline_model = baseline["model_config"]
        arm_model = arm["model_config"]
        self.assertEqual(
            baseline_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlock",
        )
        self.assertEqual(
            arm_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlockv11a",
        )
        for key, value in HYBRID_CONFIG.items():
            self.assertNotIn(key, baseline_model)
            self.assertEqual(arm_model.pop(key), value)

        self.assertEqual(arm, baseline)

    def test_training_arm_changes_only_registered_fields(self):
        self.assert_isolated_arm(
            TRAIN_BASE,
            TRAIN_ARM,
            {
                "output_dir": (
                    "./outputs/mm3_contact_cond_8L",
                    "./outputs/mm3_v11a_contact_cond_8L",
                ),
                "stop_after_steps": (None, 45000),
            },
        )

    def test_eval_arm_changes_only_architecture_and_artifact_paths(self):
        self.assert_isolated_arm(
            EVAL_BASE,
            EVAL_ARM,
            {
                "resume": (
                    "outputs/mm3_contact_cond_8L/checkpoint-45000/model.safetensors",
                    "outputs/mm3_v11a_contact_cond_8L/checkpoint-45000/model.safetensors",
                ),
                "vis_dir": (
                    "vis_results_mm3_contact_cond_45k",
                    "vis_results_mm3_v11a_contact_cond_8L_45k",
                ),
            },
        )

    def test_configs_merge_and_keep_screening_contract(self):
        train = OmegaConf.merge(
            OmegaConf.structured(TrainingConfig), OmegaConf.load(TRAIN_ARM)
        )
        evaluation = OmegaConf.merge(
            OmegaConf.structured(TestingConfig), OmegaConf.load(EVAL_ARM)
        )

        self.assertEqual(train.max_train_steps, 90000)
        self.assertEqual(train.stop_after_steps, 45000)
        self.assertEqual(train.checkpointing_steps, 2500)
        self.assertEqual(train.seed, 0)
        self.assertEqual(train.model_config.n_layers, 8)
        self.assertEqual(train.model_config.latent_dim, 256)
        self.assertTrue(train.model_config.contact_particle_cond)
        self.assertEqual(train.model_config.contact_feature_sigma, 0.04)
        self.assertEqual(train.model_config.hybrid_state_interval, 2)
        self.assertFalse(evaluation.use_diffusion)
        self.assertEqual(evaluation.num_inference_steps, 1)
        self.assertEqual(evaluation.output_frames, 1)
        self.assertTrue(evaluation.model_config.contact_particle_cond)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因配置不存在而失败**

从仓库根目录运行：

```powershell
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_v11a_contact_config.V11aContactConfigTests -v
```

Expected: FAIL，断言信息为 `missing config: ...config_mm3_v11a_contact_cond_8L.yaml`；不得是 `FileNotFoundError`、import 或环境错误。

- [ ] **Step 3: 创建训练配置，只应用预注册差异**

以 `src/configs/config_mm3_contact_cond.yaml` 的完整内容创建 `src/configs/config_mm3_v11a_contact_cond_8L.yaml`，只做以下变更：

```yaml
# B1b screening: original additive contact_cond + v11a MC-HST.
# Frozen against config_mm3_contact_cond.yaml except output_dir,
# stop_after_steps, transformer_block and the three hybrid-state fields.
output_dir: ./outputs/mm3_v11a_contact_cond_8L
stop_after_steps: 45000

model_config:
  transformer_block: SpatialTemporalTransformerBlockv11a
  hybrid_state_dim: 64
  hybrid_state_heads: 4
  hybrid_state_interval: 2
```

其余键、注释对应的数值和顺序均保留；不得显式新增其他默认字段。

- [ ] **Step 4: 创建严格镜像的 45k eval 配置**

以 `src/configs/eval_mm3_contact_cond_45k.yaml` 的完整内容创建 `src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml`，只做以下变更：

```yaml
resume: 'outputs/mm3_v11a_contact_cond_8L/checkpoint-45000/model.safetensors'
vis_dir: 'vis_results_mm3_v11a_contact_cond_8L_45k'

model_config:
  transformer_block: SpatialTemporalTransformerBlockv11a
  hybrid_state_dim: 64
  hybrid_state_heads: 4
  hybrid_state_interval: 2
```

contact、数据和评测字段必须逐项保留。

- [ ] **Step 5: 运行隔离测试并确认通过**

```powershell
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_v11a_contact_config.V11aContactConfigTests -v
```

Expected: 3 tests，全部 PASS。

- [ ] **Step 6: 提交配置与隔离测试**

```powershell
git add src/configs/config_mm3_v11a_contact_cond_8L.yaml src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml src/tests/test_v11a_contact_config.py
git commit -m "Add B1b v11a contact experiment configs"
```

不得 stage 其他未跟踪文件。

### Task 2: 验证 contact 与 v11a 的真实组合路径

**Files:**
- Modify: `src/tests/test_v11a_contact_config.py`
- Read only: `src/model/spacetime.py`
- Read only: `src/model/hybrid_state.py`
- Read only: `src/count_params.py`

**Interfaces:**
- Consumes: Task 1 的训练 config 和现有 `MDM_ST`、`HybridStateExchange`、`validate_v11a_parameter_budget`。
- Produces: 一个 CPU 小规模组合测试，证明路由、zero-gate 等价、梯度可达和 `<1%` v11a 参数增量。

- [ ] **Step 1: 增加组合模型测试**

在 `src/tests/test_v11a_contact_config.py` 的 import 区加入：

```python
import torch

from count_params import build_mm3, validate_v11a_parameter_budget
from model.hybrid_state import HybridStateExchange
from model.spacetime import MDM_ST, SpatialTemporalTransformerBlock
```

在文件中追加：

```python
class V11aContactIntegrationTests(unittest.TestCase):
    @staticmethod
    def build_from_config(transformer_block):
        cfg = OmegaConf.load(TRAIN_ARM)
        cfg.model_config.cond_frames = cfg.get("input_frames", 5)
        cfg.model_config.transformer_block = transformer_block
        if transformer_block == "SpatialTemporalTransformerBlock":
            for key in HYBRID_CONFIG:
                cfg.model_config.pop(key, None)
        return MDM_ST(
            n_points=8,
            n_frame=1,
            n_feats=3,
            model_config=cfg.model_config,
        )

    @staticmethod
    def inputs(batch_size=1, point_count=8):
        return {
            "x": torch.randn(batch_size, 1, point_count, 3),
            "timesteps": torch.zeros(batch_size, dtype=torch.long),
            "init_pc": torch.randn(batch_size, 5, point_count, 3),
            "force": torch.randn(batch_size, 3),
            "E": torch.full((batch_size, 1), 6.0),
            "nu": torch.full((batch_size, 1), 0.35),
            "drag_mask": torch.zeros(batch_size, 1, point_count, 1),
            "drag_point": torch.zeros(batch_size, 4),
            "floor_height": torch.full((batch_size, 1), -2.0),
            "gravity_label": torch.ones(batch_size, dtype=torch.long),
            "y": torch.ones(batch_size, dtype=torch.long),
            "start_vel": torch.zeros(batch_size, point_count, 3),
        }

    def test_combined_model_keeps_original_serial_blocks_and_both_modules(self):
        model = self.build_from_config("SpatialTemporalTransformerBlockv11a")

        self.assertTrue(model.contact_particle_cond)
        self.assertEqual(model.contact_injection_mode, "separate")
        self.assertEqual(model.contact_encoder.in_features, 3)
        self.assertEqual(model.contact_encoder.out_features, 256)
        self.assertTrue(
            all(
                type(block) is SpatialTemporalTransformerBlock
                for block in model.dit.transformer_blocks
            )
        )
        exchanges = [
            module for module in model.modules()
            if isinstance(module, HybridStateExchange)
        ]
        self.assertEqual(exchanges, [model.dit.hybrid_state_exchange])
        self.assertEqual(model.dit.hybrid_state_interval, 2)

    def test_zero_gate_load_from_contact_anchor_preserves_output_bits(self):
        torch.manual_seed(101)
        contact = self.build_from_config("SpatialTemporalTransformerBlock").eval()
        torch.manual_seed(202)
        combined = self.build_from_config(
            "SpatialTemporalTransformerBlockv11a"
        ).eval()

        incompatible = combined.load_state_dict(contact.state_dict(), strict=False)
        expected_missing = {
            key for key in combined.state_dict()
            if key.startswith("dit.hybrid_state_exchange.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            torch.equal(
                combined.dit.hybrid_state_exchange.feedback_gates,
                torch.zeros(4),
            )
        )

        inputs = self.inputs()
        with torch.no_grad():
            contact_output = contact(**inputs)
            combined_output = combined(**inputs)
        torch.testing.assert_close(combined_output, contact_output, rtol=0, atol=0)

    def test_contact_encoder_and_exchange_gate_receive_gradients(self):
        torch.manual_seed(303)
        model = self.build_from_config(
            "SpatialTemporalTransformerBlockv11a"
        ).train()
        output = model(**self.inputs())
        output.square().mean().backward()

        contact_grad = model.contact_encoder.weight.grad
        gate_grad = model.dit.hybrid_state_exchange.feedback_gates.grad
        self.assertIsNotNone(contact_grad)
        self.assertIsNotNone(gate_grad)
        self.assertTrue(torch.isfinite(contact_grad).all())
        self.assertTrue(torch.isfinite(gate_grad).all())
        self.assertGreater(torch.count_nonzero(contact_grad).item(), 0)
        self.assertTrue(torch.all(gate_grad != 0))

    def test_combination_adds_only_the_existing_v11a_exchange_budget(self):
        contact = build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
        )
        combined = build_mm3(
            "SpatialTemporalTransformerBlockv11a",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
        )
        report = validate_v11a_parameter_budget(contact, combined)

        self.assertEqual(report["signed_delta"], 160_773)
        self.assertLess(report["delta_percent"], 1.0)
        self.assertEqual(report["block_count"], 8)
        self.assertEqual(report["exchange_count"], 1)
        self.assertEqual(report["exchange_calls"], 4)
```

- [ ] **Step 2: 运行组合测试，保留任何真实失败**

```powershell
D:/miniconda3/envs/physctrl/python.exe -m unittest src.tests.test_v11a_contact_config.V11aContactIntegrationTests -v
```

Expected: 4 tests PASS。若失败，先确认失败来自 contact/v11a 接口，而不是测试输入 shape、dtype 或环境；不得通过放宽断言掩盖失败。

- [ ] **Step 3: 仅在接口测试证明必要时做最小修复**

允许修复的范围只有：

- v11a kwargs 路由遗漏；
- contact-conditioned hidden 在 mask 插入前后的 frame 索引错误；
- structured config 缺失已批准字段；
- checkpoint load 的新增模块 key 集不正确。

不允许修改 contact feature 公式、contact encoder、state feature、exchange schedule、backbone block 或 loss。若 4 个测试直接通过，本步骤不产生代码改动。

- [ ] **Step 4: 运行组合与既有单机制回归测试**

```powershell
D:/miniconda3/envs/physctrl/python.exe -m unittest `
  src.tests.test_v11a_contact_config `
  src.tests.test_v11a_config `
  src.tests.test_hybrid_state `
  src.tests.test_contact_ablation_configs -v
```

Expected: 全部 PASS，无 skipped failure。

- [ ] **Step 5: 提交组合验证测试**

```powershell
git add src/tests/test_v11a_contact_config.py
git commit -m "Verify B1b contact and hybrid state composition"
```

若 Step 3 修改了模型接口，把对应文件一并显式加入 `git add`；commit message 改为准确描述修复，不得使用笼统的 `fix tests`。

### Task 3: 全量验收并回填实验台账

**Files:**
- Modify: `实验记录.md`
- Verify: all files created or modified in Tasks 1-2

**Interfaces:**
- Consumes: Tasks 1-2 的两个本地代码 commit。
- Produces: 可审计的 B1b 实现记录和服务器训练命令；实验状态仍为“待训练”。

- [ ] **Step 1: 获取两个实现 commit 的短 hash**

```powershell
git log -2 --format="%h %s"
```

Expected: 顶部两条分别对应 Task 2 的组合验证和 Task 1 的配置实现。记录实际 hash，不得猜测或写占位符。

- [ ] **Step 2: 在实验台账回填实际代码 commit**

使用 `apply_patch` 修改 `实验记录.md`：

1. 在 B1b 总览行的代码 commit 单元中，按时间顺序写入 Step 1 输出的两个真实短 hash；
2. 在 B1b 详录的“结论”之后增加“代码 commit”行，分别注明第一个 hash 对应 train/eval config 与隔离测试，第二个 hash 对应组合路由、梯度和参数验证；
3. 保留“待训练”状态，不增加任何结果数字。

提交前运行以下占位符扫描，必须无匹配：

```powershell
rg -n "<.*hash|实际短 hash" 实验记录.md
```

- [ ] **Step 3: 运行最终验证**

从仓库根目录运行：

```powershell
D:/miniconda3/envs/physctrl/python.exe -m unittest `
  src.tests.test_v11a_contact_config `
  src.tests.test_v11a_config `
  src.tests.test_hybrid_state `
  src.tests.test_contact_ablation_configs -v

D:/miniconda3/envs/physctrl/python.exe -m py_compile `
  src/tests/test_v11a_contact_config.py

Push-Location src
D:/miniconda3/envs/physctrl/python.exe count_params.py
Pop-Location

git diff --check
git status --short
```

Expected:

- unittest 全部 PASS；
- `py_compile` exit 0；
- `count_params.py` 报告 v11a 增量 160,773、增幅 `<1%`，8 个原串行 block、1 个 shared exchange、4 次调用；
- `git diff --check` 无错误；
- status 中只出现本任务的 `实验记录.md` 未提交修改和用户原有未跟踪文件。

- [ ] **Step 4: 提交台账更新**

```powershell
git add -- 实验记录.md
git commit -m "Record B1b implementation commits"
```

不修改 B1b 的结果结论；训练和 eval 尚未发生。

- [ ] **Step 5: 最终检查提交内容**

```powershell
git show --stat --oneline HEAD~3..HEAD
git status --short --branch
```

Expected: 三个本地 commit 只包含两份配置、一份测试和台账更新；所有现有 PPT、图片、结果目录及临时文件仍未跟踪且未纳入 commit。

## 服务器运行（实现验收后，不在本地自动执行）

从服务器仓库的 `src/` 目录训练到 screening step 45000：

```bash
accelerate launch --config_file configs/acc/1gpu.yaml train.py \
  --config configs/config_mm3_v11a_contact_cond_8L.yaml
```

评测 checkpoint-45000：

```bash
python eval.py --config configs/eval_mm3_v11a_contact_cond_8L_45k.yaml
```

必须先按设计文档 §8 对 45k 结果判门槛。通过后才创建 90k continuation config；本实施计划不预建该配置。
