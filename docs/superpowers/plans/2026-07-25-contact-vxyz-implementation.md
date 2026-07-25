# Contact v_xyz 条件实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变旧三通道模型和 checkpoint 的前提下，为新实验增加 `[gap,vx,vy,vz,proximity]` 五通道逐粒子条件。

**Architecture:** `contact_velocity_mode` 默认 `vertical`，显式设为 `xyz` 时才把 contact encoder 从 `Linear(3,256)` 扩展为 `Linear(5,256)`。两种模式共用特征构造、mask、诊断和注入路径，旧配置继续走原始行为。

**Tech Stack:** Python、PyTorch、OmegaConf、unittest、YAML、Accelerate。

---

### Task 1: 规范与设计文档

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-25-contact-vxyz-design.md`

- [ ] **Step 1: 记录中文文档规范**

在 `AGENTS.md` 配置约定中加入：供 Will 审阅的设计、计划和实验分析使用中文。

- [ ] **Step 2: 将已确认设计改为中文**

保留公式、代码标识符和命令英文，正文使用中文。

- [ ] **Step 3: 检查文档**

Run:

```bash
rg -n "TBD|TODO|PLACEHOLDER" AGENTS.md docs/superpowers/specs/2026-07-25-contact-vxyz-design.md
```

Expected: 无输出。

### Task 2: 先写五通道特征失败测试

**Files:**
- Modify: `src/tests/test_contact.py`
- Modify: `src/utils/contact.py`

- [ ] **Step 1: 写失败测试**

新增测试，要求：

```python
features = build_contact_features(
    points,
    floor,
    start_velocity=start_velocity,
    sigma=1.0,
    velocity_mode="xyz",
)
self.assertEqual(features.shape, (1, 3, 2, 5))
torch.testing.assert_close(features[..., 0], points[..., 1])
torch.testing.assert_close(features[0, 0, :, 1:4], start_velocity)
torch.testing.assert_close(
    features[0, 1:, :, 1:4],
    points[:, 1:, :, :3] - points[:, :-1, :, :3],
)
```

另加：

```python
with self.assertRaisesRegex(ValueError, "contact_velocity_mode"):
    build_contact_features(points, floor, velocity_mode="bad")
```

- [ ] **Step 2: 运行并确认 RED**

Run:

```bash
python -m unittest src.tests.test_contact.ContactFeatureTests.test_features_encode_full_xyz_displacement
```

Expected: FAIL，原因是 `build_contact_features` 尚不接受 `velocity_mode`。

- [ ] **Step 3: 实现最小特征构造**

在 `src/utils/contact.py` 增加：

```python
CONTACT_FEATURE_NAMES = {
    "vertical": ("signed_gap", "vertical_displacement", "proximity"),
    "xyz": (
        "signed_gap",
        "displacement_x",
        "displacement_y",
        "displacement_z",
        "proximity",
    ),
}

def contact_feature_names(velocity_mode: str):
    try:
        return CONTACT_FEATURE_NAMES[velocity_mode]
    except KeyError as exc:
        raise ValueError(
            f"contact_velocity_mode must be one of {tuple(CONTACT_FEATURE_NAMES)}; "
            f"got {velocity_mode!r}"
        ) from exc
```

`build_contact_features(..., velocity_mode="vertical")` 在 `xyz` 模式下构造完整三维帧差，最后返回：

```python
motion = full_displacement if velocity_mode == "xyz" else full_displacement[..., 1:2]
return torch.cat([signed_gap, motion, proximity], dim=-1)
```

- [ ] **Step 4: 运行并确认 GREEN**

Run:

```bash
python -m unittest src.tests.test_contact
```

Expected: 全部通过，旧三通道测试保持不变。

### Task 3: 动态 mask、诊断与模型构造

**Files:**
- Modify: `src/tests/test_contact.py`
- Modify: `src/utils/contact.py`
- Modify: `src/model/spacetime.py`
- Modify: `src/contact_feature_diagnostics.py`
- Modify: `src/eval.py`

- [ ] **Step 1: 写模型失败测试**

测试旧模式：

```python
self.assertEqual(vertical_model.contact_encoder.in_features, 3)
self.assertEqual(vertical_model.contact_feature_mask, (1.0, 1.0, 1.0))
```

测试新模式：

```python
cfg.contact_velocity_mode = "xyz"
xyz_model = MDM_ST(...)
self.assertEqual(xyz_model.contact_encoder.in_features, 5)
self.assertEqual(xyz_model.contact_feature_mask, (1.0,) * 5)
```

并验证五通道 mask 长度错误时抛出 `ValueError`。

- [ ] **Step 2: 运行并确认 RED**

Run:

```bash
python -m unittest src.tests.test_contact
```

Expected: FAIL，新模型仍固定创建 `Linear(3, latent_dim)`。

- [ ] **Step 3: 实现动态维度**

`MDM_ST.__init__`：

```python
self.contact_velocity_mode = model_config.get("contact_velocity_mode", "vertical")
self.contact_feature_names = contact_feature_names(self.contact_velocity_mode)
default_mask = [1.0] * len(self.contact_feature_names)
self.contact_feature_mask = tuple(
    float(v) for v in model_config.get("contact_feature_mask", default_mask)
)
self.contact_encoder = nn.Linear(len(self.contact_feature_names), self.latent_dim)
```

forward 调用：

```python
build_contact_features(
    init_pc_cond,
    floor_height,
    start_velocity=start_vel,
    sigma=self.contact_feature_sigma,
    velocity_mode=self.contact_velocity_mode,
)
```

`apply_contact_feature_mask` 和 `contact_channel_contributions` 不再硬编码 3，改为验证 mask/weight 的最后一维与 features 一致。

- [ ] **Step 4: 更新诊断和 eval CLI**

诊断脚本从配置读取 `contact_velocity_mode`，按 `contact_feature_names()` 输出列名；eval CLI 的 mask 改为 `nargs='+'`，实际长度由模型验证。

- [ ] **Step 5: 运行并确认 GREEN**

Run:

```bash
python -m unittest src.tests.test_contact src.tests.test_contact_feature_diagnostics
```

Expected: 全部通过。

### Task 4: 添加严格冻结的训练和评测配置

**Files:**
- Create: `src/configs/config_mm3_contact_vxyz.yaml`
- Create: `src/configs/eval_mm3_contact_vxyz_45k.yaml`
- Modify: `src/tests/test_contact_ablation_configs.py`

- [ ] **Step 1: 写失败配置测试**

加载新 YAML 并验证：

```python
self.assertEqual(train.model_config.contact_velocity_mode, "xyz")
self.assertEqual(train.stop_after_steps, 45000)
self.assertEqual(train.max_train_steps, 90000)
self.assertEqual(eval_cfg.model_config.contact_velocity_mode, "xyz")
```

删除允许变化的字段：

```text
output_dir
stop_after_steps
model_config.contact_velocity_mode
```

其余训练字段必须与 `config_mm3_contact_cond.yaml` 相同；eval 中模型和数据字段必须镜像训练配置。

- [ ] **Step 2: 运行并确认 RED**

Run:

```bash
python -m unittest src.tests.test_contact_ablation_configs
```

Expected: FAIL，新配置尚不存在。

- [ ] **Step 3: 创建配置**

训练配置：

```yaml
output_dir: ./outputs/mm3_contact_vxyz_8L
max_train_steps: 90000
stop_after_steps: 45000
model_config:
  contact_particle_cond: true
  contact_feature_sigma: 0.04
  contact_velocity_mode: xyz
```

评测配置：

```yaml
resume: outputs/mm3_contact_vxyz_8L/checkpoint-45000/model.safetensors
vis_dir: vis_results_mm3_contact_vxyz_45k
model_config:
  contact_velocity_mode: xyz
```

其余字段逐项复制匹配基线。

- [ ] **Step 4: 运行并确认 GREEN**

Run:

```bash
python -m unittest src.tests.test_contact_ablation_configs
```

Expected: 全部通过。

### Task 5: 完整验证

**Files:**
- Verify all modified files

- [ ] **Step 1: 语法检查**

```bash
python -m py_compile src/utils/contact.py src/model/spacetime.py src/contact_feature_diagnostics.py src/eval.py
```

- [ ] **Step 2: 全部相关测试**

```bash
python -m unittest \
  src.tests.test_contact \
  src.tests.test_contact_feature_diagnostics \
  src.tests.test_contact_ablation_configs
```

- [ ] **Step 3: 参数量检查**

```bash
python src/count_params.py
```

确认 v_xyz 仅比 vertical 模型多 512 个权重，旧模型参数量不变。

- [ ] **Step 4: 配置差异检查**

确认新训练臂相对完整 contact 基线只允许：

```text
output_dir
stop_after_steps
model_config.contact_velocity_mode
```

- [ ] **Step 5: 汇报服务器命令**

```bash
cd /root/code/traceformer/src
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/config_mm3_contact_vxyz.yaml
python eval.py --config configs/eval_mm3_contact_vxyz_45k.yaml
```
