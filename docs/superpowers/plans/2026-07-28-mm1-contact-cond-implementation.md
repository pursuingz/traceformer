# mm1 Contact Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加一个使用 `diff_E_2048_data` elastic 单材料数据、仅启用初版 separate contact conditioning 的 45k 筛选实验。

**Architecture:** 不修改模型代码。训练配置严格复制 diffE2048 single-frame baseline，仅增加初版三通道 contact conditioning、独立输出目录与 45k 提前停止；评测配置镜像训练模型并使用同一 diffE2048 test split。

**Tech Stack:** Python 3.10、PyTorch、OmegaConf、YAML、`unittest`

---

## 文件结构

- Create: `src/configs/config_mm1_contact_cond.yaml`
  - elastic 单材料训练配置。
- Create: `src/configs/eval_mm1_contact_cond_45k.yaml`
  - checkpoint-45000 评测配置。
- Create: `src/tests/test_mm1_contact_cond_configs.py`
  - 验证训练臂只有允许差异，并验证 train/eval 镜像和参数增量。

### Task 1: 添加训练配置与单变量测试

**Files:**
- Create: `src/tests/test_mm1_contact_cond_configs.py`
- Create: `src/configs/config_mm1_contact_cond.yaml`

- [ ] **Step 1: 写训练配置的失败测试**

在 `src/tests/test_mm1_contact_cond_configs.py` 写入：

```python
import copy
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from model.spacetime import MDM_ST
from options import TrainingConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_structured(name, schema):
    return OmegaConf.merge(
        OmegaConf.structured(schema),
        OmegaConf.load(CONFIG_DIR / name),
    )


def _without_paths(cfg, paths):
    data = copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))
    for path in paths:
        cursor = data
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor.pop(parts[-1], None)
    return data


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class Mm1ContactCondConfigTests(unittest.TestCase):
    def test_training_arm_changes_only_contact_and_screening_fields(self):
        baseline = _load_structured(
            "config_diffE2048_singleframe_geom_deform_d0001.yaml",
            TrainingConfig,
        )
        arm = _load_structured(
            "config_mm1_contact_cond.yaml",
            TrainingConfig,
        )
        ignored = {
            "output_dir",
            "stop_after_steps",
            "model_config.contact_particle_cond",
            "model_config.contact_injection_mode",
            "model_config.contact_velocity_mode",
            "model_config.contact_feature_sigma",
        }

        self.assertEqual(arm.max_train_steps, 90000)
        self.assertEqual(arm.stop_after_steps, 45000)
        self.assertEqual(
            arm.train_dataset.dataset_path,
            "diff_E_2048_data/2048_data/2048_train",
        )
        self.assertTrue(arm.model_config.contact_particle_cond)
        self.assertEqual(arm.model_config.contact_injection_mode, "separate")
        self.assertEqual(arm.model_config.contact_velocity_mode, "vertical")
        self.assertEqual(arm.model_config.contact_feature_sigma, 0.04)
        self.assertFalse(arm.model_config.get("class_token", False))
        self.assertFalse(arm.model_config.get("gravity_emb", False))
        self.assertFalse(arm.get("geom_elastic_only", False))
        self.assertEqual(
            _without_paths(arm, ignored),
            _without_paths(baseline, ignored),
        )

        baseline.model_config.cond_frames = baseline.get("input_frames", 5)
        arm.model_config.cond_frames = arm.get("input_frames", 5)
        baseline_model = MDM_ST(
            8,
            baseline.output_frames,
            n_feats=3,
            model_config=baseline.model_config,
        )
        arm_model = MDM_ST(
            8,
            arm.output_frames,
            n_feats=3,
            model_config=arm.model_config,
        )
        self.assertEqual(
            _parameter_count(arm_model) - _parameter_count(baseline_model),
            1024,
        )
        self.assertEqual(arm_model.contact_encoder.in_features, 3)
        self.assertEqual(arm_model.contact_encoder.out_features, 256)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因配置缺失而失败**

Run:

```bash
cd src
python -m unittest tests.test_mm1_contact_cond_configs
```

Expected: `ERROR`，包含
`config_mm1_contact_cond.yaml` 不存在。

- [ ] **Step 3: 创建训练配置**

以
`src/configs/config_diffE2048_singleframe_geom_deform_d0001.yaml`
为逐字段来源创建
`src/configs/config_mm1_contact_cond.yaml`，只执行以下变更：

```diff
-output_dir: ./outputs/diffE2048_singleframe_geom_deform_d0001_8L
+output_dir: ./outputs/mm1_contact_cond_8L

 max_train_steps: 90000
+stop_after_steps: 45000

 model_config:
   transformer_block: SpatialTemporalTransformerBlock
+  contact_particle_cond: true
+  contact_injection_mode: separate
+  contact_velocity_mode: vertical
+  contact_feature_sigma: 0.04
```

文件顶部使用中文注释明确：

```yaml
# mm1_contact_cond：diff_E_2048 elastic 单材料严格消融。
# 基线为 config_diffE2048_singleframe_geom_deform_d0001.yaml。
# 唯一方法变量是初版 separate contact conditioning；
# output_dir 与 stop_after_steps 仅用于实验隔离和 45k 筛选。
```

不要加入 `class_token`、`gravity_emb`、`num_mat` 或
`geom_elastic_only`。

- [ ] **Step 4: 运行训练配置测试**

Run:

```bash
cd src
python -m unittest tests.test_mm1_contact_cond_configs
```

Expected: `Ran 1 test`，`OK`。

- [ ] **Step 5: 提交 Task 1**

```bash
git add src/tests/test_mm1_contact_cond_configs.py \
        src/configs/config_mm1_contact_cond.yaml
git commit -m "Add mm1 contact conditioning training config"
```

提交作者必须是 Will，不添加 AI co-author/contributor。

### Task 2: 添加 45k 评测配置与镜像测试

**Files:**
- Modify: `src/tests/test_mm1_contact_cond_configs.py`
- Create: `src/configs/eval_mm1_contact_cond_45k.yaml`

- [ ] **Step 1: 写 eval 镜像的失败测试**

在 `Mm1ContactCondConfigTests` 中加入：

```python
    def test_eval_mirrors_training_and_uses_diffE2048_test_split(self):
        train_cfg = _load_structured(
            "config_mm1_contact_cond.yaml",
            TrainingConfig,
        )
        eval_cfg = OmegaConf.load(
            CONFIG_DIR / "eval_mm1_contact_cond_45k.yaml"
        )
        baseline_eval = OmegaConf.load(
            CONFIG_DIR
            / "eval_diffE2048_singleframe_geom_deform_d0001.yaml"
        )
        ignored = {
            "resume",
            "vis_dir",
            "model_config.contact_particle_cond",
            "model_config.contact_injection_mode",
            "model_config.contact_velocity_mode",
            "model_config.contact_feature_sigma",
        }

        self.assertEqual(
            eval_cfg.resume,
            "outputs/mm1_contact_cond_8L/"
            "checkpoint-45000/model.safetensors",
        )
        self.assertEqual(
            eval_cfg.train_dataset.dataset_path,
            "diff_E_2048_data/2048_data/2048_test",
        )
        self.assertEqual(
            OmegaConf.to_container(eval_cfg.model_config, resolve=True),
            OmegaConf.to_container(train_cfg.model_config, resolve=True),
        )
        self.assertEqual(eval_cfg.output_frames, train_cfg.output_frames)
        self.assertEqual(
            _without_paths(eval_cfg, ignored),
            _without_paths(baseline_eval, ignored),
        )
```

- [ ] **Step 2: 运行测试并确认因 eval 配置缺失而失败**

Run:

```bash
cd src
python -m unittest \
  tests.test_mm1_contact_cond_configs.Mm1ContactCondConfigTests.test_eval_mirrors_training_and_uses_diffE2048_test_split
```

Expected: `ERROR`，包含
`eval_mm1_contact_cond_45k.yaml` 不存在。

- [ ] **Step 3: 创建 eval 配置**

以
`src/configs/eval_diffE2048_singleframe_geom_deform_d0001.yaml`
为逐字段来源创建
`src/configs/eval_mm1_contact_cond_45k.yaml`，只执行以下变更：

```diff
-resume: 'outputs/diffE2048_singleframe_geom_deform_d0001_8L/checkpoint-90000/model.safetensors'
-vis_dir: 'vis_results_diffE2048_singleframe_geom_deform_d0001'
+resume: 'outputs/mm1_contact_cond_8L/checkpoint-45000/model.safetensors'
+vis_dir: 'vis_results_mm1_contact_cond_45k'

 model_config:
   transformer_block: SpatialTemporalTransformerBlock
+  contact_particle_cond: true
+  contact_injection_mode: separate
+  contact_velocity_mode: vertical
+  contact_feature_sigma: 0.04
```

保持：

```yaml
train_dataset:
  dataset_path: diff_E_2048_data/2048_data/2048_test
```

- [ ] **Step 4: 运行配置测试与完整测试**

Run:

```bash
cd src
python -m unittest tests.test_mm1_contact_cond_configs
python -m unittest discover -s tests
```

Expected:

- mm1 配置测试：`Ran 2 tests`，`OK`；
- 完整测试：0 failures，0 errors。

- [ ] **Step 5: 验证配置可解析和模型可构建**

Run:

```bash
cd src
python -m py_compile tests/test_mm1_contact_cond_configs.py
python -m unittest tests.test_mm1_contact_cond_configs
```

Expected: 两条命令 exit code 0。

- [ ] **Step 6: 检查提交范围**

Run:

```bash
git diff --check
git status --short
```

Expected: 仅出现本 Task 的 eval 配置和测试修改，无训练产物、
checkpoint、CSV 或 `CLAUDE.md`。

- [ ] **Step 7: 提交 Task 2**

```bash
git add src/tests/test_mm1_contact_cond_configs.py \
        src/configs/eval_mm1_contact_cond_45k.yaml
git commit -m "Add mm1 contact conditioning eval config"
```

提交作者必须是 Will，不添加 AI co-author/contributor；不得执行
`git push`。

## 服务器命令

训练：

```bash
cd /root/code/traceformer/src
accelerate launch --config_file configs/acc/1gpu.yaml \
  train.py --config configs/config_mm1_contact_cond.yaml
```

评测：

```bash
cd /root/code/traceformer/src
python eval.py --config configs/eval_mm1_contact_cond_45k.yaml
```

对照评测必须将 diffE2048 baseline 的 eval config 临时或独立指向
`checkpoint-45000`，不可拿 baseline 90k 与 mm1 contact 45k 下结论。
