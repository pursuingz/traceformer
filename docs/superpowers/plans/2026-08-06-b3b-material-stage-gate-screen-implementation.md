# B3b Material-Stage Gate Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 B3a@90k 主干与共享 adapter 的条件下，实现并训练 12 个 bounded material-stage gates，用 autoregressive rollout calibration 检验能否保住 sand 并修复 plasticine/elastic。

**Architecture:** 在 `FactorizedMaterialStateAdapter` 内可选注册 `gate_logits[3,4]`，用 `2*sigmoid(logit)` 调制已有 stage scale；默认关闭时 state dict 和 forward 完全不变。独立 calibration trainer 从 B3a checkpoint 初始化并冻结除 gate 外的所有参数，使用 `mm3_train` 的分层 train/val 划分和逐帧 detach rollout 优化，每种材质独立选取最佳 gate row，最后保存完整模型供现有 `eval.py` 使用。

**Tech Stack:** Python 3.10、PyTorch、Accelerate、OmegaConf、safetensors、h5py、NumPy、现有 `MDM_ST/TrajDataset/TrajPipeline`。

## Global Constraints

- `material_stage_gate` 默认 `false`；关闭时不得创建新参数或改变旧 checkpoint/forward。
- B3b 仅训练 12 个 gate 参数；B3a adapter、8 层串行主干、contact、材料 token、loss 和数据本体均不修改。
- `gate_logits=0` 必须得到 `g=1`，加载 B3a checkpoint 后函数等价。
- gate 范围固定为 `(0,2)`；rigid 类固定为 1。
- calibration 只读取 `mm3_train`；当前 41-model `mm3_test` 只在 gate 冻结后作为 dev-test 评一次。
- 每种材质最多 200 updates，Adam `lr=3e-3`，无 weight decay/scheduler，val interval 25，patience 3。
- loss 固定为全 rollout MSE + `0.5 *` 最后三分之一 MSE + `1e-3 * mean((g-1)^2)`；不做 dev-test 超参数 sweep。
- 供 Will 审阅的设计、计划与实验文档使用中文；commit 不添加任何 AI attribution。
- 不执行 `git push`；不删除 worktree；不修改 `.env`、CI/CD 或系统依赖。

## File Structure

- Modify `src/model/material_state.py`: bounded gate 参数、查表和 adapter scale 调制。
- Modify `src/model/spacetime.py`: config 透传、启用约束与 MDM_ST 集成。
- Modify `src/count_params.py`: B3b 参数量报告与 +12 断言。
- Modify `src/dataset/traj_dataset.py`: 新增按显式 model/start spec 读取的兼容入口，旧索引入口委托给它。
- Create `src/dataset/material_gate_dataset.py`: 分材质 split manifest、calibration window 与变长 future GT。
- Create `src/utils/material_stage_gate_training.py`: checkpoint 校验、冻结、loss、rollout、early stopping、gate row 合并和产物保存。
- Create `src/train_material_stage_gates.py`: Accelerate CLI 和三材质训练编排。
- Create `src/configs/config_mm3_b3b_material_stage_gate_screen.yaml`: 冻结 screening 协议。
- Create `src/configs/eval_mm3_b3b_material_stage_gate_screen.yaml`: dev-test 标准 eval。
- Create `src/tests/test_material_stage_gate.py`: 数学、集成、state dict、参数量测试。
- Create `src/tests/test_material_gate_dataset.py`: split、采样、future GT 测试。
- Create `src/tests/test_material_stage_gate_training.py`: trainer 纯函数、冻结、rollout 和保存测试。
- Modify `src/tests/test_b3_material_state_config.py`: B3b config/eval 镜像测试。
- Modify `实验记录_1.md`: 回填 spec、plan、实现 commit 与服务器命令。

---

### Task 1: Bounded Material-Stage Gate 数学模块

**Files:**
- Modify: `src/model/material_state.py`
- Create: `src/tests/test_material_stage_gate.py`

**Interfaces:**
- Produces: `FactorizedMaterialStateAdapter(particle_dim: int, rank: int, num_materials: int, num_stages: int, e_center: float = 5.5, e_scale: float = 1.0, nu_center: float = 0.25, nu_scale: float = 0.15, material_stage_gate: bool = False, gated_materials: int = 3, gate_max: float = 2.0)`。
- Produces: `material_stage_gates() -> Tensor[3,4]`，返回 bounded gate 值。
- Produces: `gate_for(material_labels, stage_index, dtype) -> Tensor[B]`，rigid/预留类返回 1。
- Preserves: gate 关闭时原 constructor、参数名和 forward 数学不变。

- [ ] **Step 1: 写 gate constructor 与范围的失败测试**

在 `src/tests/test_material_stage_gate.py` 创建 `MaterialStageGateMathTest`，覆盖：

```python
def make_adapter(enabled: bool):
    return FactorizedMaterialStateAdapter(
        particle_dim=16,
        rank=4,
        num_materials=4,
        num_stages=4,
        material_stage_gate=enabled,
        gated_materials=3,
        gate_max=2.0,
    )

def test_disabled_adapter_has_no_gate_parameter(self):
    adapter = make_adapter(False)
    self.assertNotIn("gate_logits", dict(adapter.named_parameters()))

def test_enabled_gate_is_identity_and_bounded(self):
    adapter = make_adapter(True)
    gates = adapter.material_stage_gates()
    self.assertEqual(tuple(gates.shape), (3, 4))
    self.assertTrue(torch.equal(gates, torch.ones_like(gates)))
    with torch.no_grad():
        adapter.gate_logits.fill_(100.0)
    self.assertTrue((adapter.material_stage_gates() <= 2.0).all())
```

- [ ] **Step 2: 运行测试并确认因新参数未实现而失败**

从 `src/` 运行：

```bash
python -m unittest tests.test_material_stage_gate.MaterialStageGateMathTest -v
```

Expected: constructor 报 `unexpected keyword argument 'material_stage_gate'`。

- [ ] **Step 3: 实现可选 gate 参数与输入校验**

在 `FactorizedMaterialStateAdapter.__init__` 中：

```python
self.material_stage_gate = bool(material_stage_gate)
self.gated_materials = int(gated_materials)
self.gate_max = float(gate_max)
if self.material_stage_gate:
    if not 0 < self.gated_materials < self.num_materials:
        raise ValueError("gated_materials must cover a strict subset of material classes")
    if self.gate_max <= 0:
        raise ValueError("gate_max must be positive")
    self.gate_logits = nn.Parameter(
        torch.zeros(self.gated_materials, self.num_stages)
    )
```

增加：

```python
def material_stage_gates(self) -> torch.Tensor:
    if not self.material_stage_gate:
        raise RuntimeError("material-stage gate is disabled")
    return self.gate_max * torch.sigmoid(self.gate_logits)

def gate_for(self, material_labels, stage_index, dtype):
    gate = torch.ones(material_labels.shape[0], device=material_labels.device, dtype=dtype)
    active = material_labels.long() < self.gated_materials
    if active.any():
        all_gates = self.material_stage_gates().to(dtype=dtype)
        gate[active] = all_gates[material_labels[active].long(), stage_index]
    return gate
```

- [ ] **Step 4: 写 per-material、rigid 和梯度隔离的失败测试**

测试把 `output_proj.weight` 设为非零后：elastic/plasticine/sand 的不同 logits 产生不同 scale；label 3 始终返回 1；只用 material 1 前向反向时仅 `gate_logits[1]` 有非零梯度。

- [ ] **Step 5: 在 adapter forward 中调制已有 scale**

保持 gate 关闭分支原表达式；开启时：

```python
scale = self.stage_scales[stage_index].to(delta.dtype) * float(runtime_scale)
if self.material_stage_gate:
    gate = self.gate_for(material_labels, stage_index, delta.dtype)
    scale = scale * gate[:, None, None]
return hidden_states + scale * delta
```

- [ ] **Step 6: 运行数学模块测试**

```bash
python -m unittest tests.test_material_stage_gate.MaterialStageGateMathTest -v
python -m unittest tests.test_material_state_adapter.FactorizedMaterialStateAdapterTest -v
```

Expected: 全部 PASS，旧 adapter identity 测试不变。

- [ ] **Step 7: 提交 Task 1**

```bash
git add src/model/material_state.py src/tests/test_material_stage_gate.py
git commit -m "Add bounded material-stage gate"
```

---

### Task 2: MDM_ST 集成、checkpoint 兼容与参数统计

**Files:**
- Modify: `src/model/spacetime.py`
- Modify: `src/count_params.py`
- Modify: `src/tests/test_material_stage_gate.py`
- Modify: `src/tests/test_count_params.py`

**Interfaces:**
- Consumes: Task 1 的 adapter constructor。
- Produces model config: `material_stage_gate: bool = False`、`material_stage_gate_max: float = 2.0`。
- Produces invariant: B3a state dict 加载到 B3b 时 missing keys 精确为 `dit.material_state_exchange.gate_logits` 对应实际完整前缀，且无 unexpected key。

- [ ] **Step 1: 写默认关闭与 identity 集成失败测试**

扩展 `small_config` 支持 `gate` 参数。新增测试：

```python
def test_gate_disabled_state_dict_matches_b3a(self):
    b3a = MDM_ST(2, 1, 3, self.small_config(gate=False))
    self.assertFalse(hasattr(b3a.dit.material_state_exchange, "gate_logits"))

def test_b3a_checkpoint_loads_into_b3b_with_only_gate_missing(self):
    b3a = MDM_ST(2, 1, 3, self.small_config(gate=False)).eval()
    b3b = MDM_ST(2, 1, 3, self.small_config(gate=True)).eval()
    incompatible = b3b.load_state_dict(b3a.state_dict(), strict=False)
    self.assertEqual(incompatible.unexpected_keys, [])
    self.assertEqual(
        incompatible.missing_keys,
        ["dit.material_state_exchange.gate_logits"],
    )
```

再将 B3a 的非零 adapter state 加载到 B3b，断言相同输入输出 `torch.equal` 或最大绝对差不超过 `1e-7`。

- [ ] **Step 2: 运行测试并确认 model config 尚未透传**

```bash
python -m unittest tests.test_material_stage_gate.MaterialStageGateIntegrationTest -v
```

Expected: gate 参数不存在或 missing key 断言失败。

- [ ] **Step 3: 透传 model config**

在 `SpaitalTemporalTransformer.__init__` 增加：

```python
material_stage_gate: bool = False,
material_stage_gate_max: float = 2.0,
```

约束 gate 只能在 `material_state_adapter=True` 时启用。构建 adapter 时传入 `material_stage_gate`、`gated_materials=3`、`gate_max`。在 `MDM_ST` 保存配置并向 `self.dit` 透传；默认值必须维持旧配置行为。

- [ ] **Step 4: 增加参数统计失败测试**

在 `count_params.py` 的 B3 report 增加 B3b model，断言：

```python
b3b_total - b3a_total == 12
count_trainable(b3b.dit.material_state_exchange) == adapter_params + 12
```

旧 baseline/B3a 数字必须不变。

- [ ] **Step 5: 实现 B3b 参数统计行**

输出至少包含：

```text
B3a total params
B3b total params
B3b gate params: 12
```

- [ ] **Step 6: 运行集成与参数测试**

```bash
python -m unittest tests.test_material_stage_gate -v
python -m unittest tests.test_material_state_adapter -v
python -m unittest tests.test_count_params -v
python count_params.py
```

Expected: B3b 比 B3a 精确增加 12 参数；旧模型测试全部 PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add src/model/spacetime.py src/count_params.py src/tests/test_material_stage_gate.py src/tests/test_count_params.py
git commit -m "Integrate material-stage gate into Traceformer"
```

---

### Task 3: Calibration Split 与变长 Rollout 数据

**Files:**
- Modify: `src/dataset/traj_dataset.py`
- Create: `src/dataset/material_gate_dataset.py`
- Create: `src/tests/test_material_gate_dataset.py`

**Interfaces:**
- Produces: `TrajDataset.get_deform_diff_from_spec(model_spec: dict) -> tuple[dict, dict]`。
- Produces: `build_material_split(dataset_path, dataset_list, train_fraction, seed) -> dict`。
- Produces: `MaterialGateDataset.load_window(model_name, start_idx) -> dict`，含 `future_gt[T,N,3]`。
- Produces manifest keys: `seed/train_fraction/materials/{elastic,plasticine,sand}/{train,val}`。

- [ ] **Step 1: 写显式 model spec 与旧索引等价的失败测试**

在临时 H5 fixture 中构造 25 帧、固定粒子和最小必需物理字段。用同一 `model_spec={"model": name, "start_idx": 0}` 分别调用旧索引和新方法，逐 key 断言 tensor 相等。

- [ ] **Step 2: 运行测试确认新入口不存在**

```bash
python -m unittest tests.test_material_gate_dataset.TrajDatasetExplicitSpecTest -v
```

Expected: `TrajDataset` 无 `get_deform_diff_from_spec`。

- [ ] **Step 3: 最小重构 TrajDataset 入口**

保持 `get_deform_diff` 对外语义：

```python
def get_deform_diff(self, index):
    return self.get_deform_diff_from_spec(self.models[index])

def get_deform_diff_from_spec(self, model):
    model_name = model["model"]
    start_idx = model["start_idx"]
```

其余原函数体移入新方法，不改变随机 marker `-1/-2`、归一化、点采样或物理字段。

- [ ] **Step 4: 写分层 split 的失败测试**

fixture 至少包含每材质 5 个 model，另含 rigid 1 个。断言：

- 三材质 train/val 均非空；
- train 与 val 互斥且并集覆盖三种材质全部 model；
- rigid 不进入 gate split；
- 相同 seed manifest 完全一致，不同 seed 至少一组顺序不同；
- 输入路径必须以 `mm3_train` 结尾，`mm3_test` 触发 `ValueError`。

- [ ] **Step 5: 实现 split 与 material 读取**

`mat_type` 缺失按现有约定视为 elastic 0；只接受 0/1/2。每类先排序，再用独立 `random.Random(seed + material_id)` shuffle；`val_count=max(1, round(n*(1-train_fraction)))`，并保证 train 至少 1 个。

- [ ] **Step 6: 写 future GT 与起点测试**

对 25 帧 fixture：

- start0 返回 5 帧输入和 20 帧 `future_gt`；
- start5 返回 5 帧输入和 15 帧 `future_gt`；
- `future_gt` 使用与 `points_src` 相同 `point_indices`；
- 归一化为 `(x-norm_fac)/2`；
- start0/random 两种 sample spec 数量相同，构成 50/50 采样池。

- [ ] **Step 7: 实现 MaterialGateDataset**

内部复用 `TrajDataset.get_deform_diff_from_spec` 生成模型输入和条件；然后只为相同 model/start/point_indices 读取剩余 `x` 作为 `future_gt`。禁止复制 force、mask、floor、E/nu 等解析代码。

- [ ] **Step 8: 运行数据测试和旧数据回归测试**

```bash
python -m unittest tests.test_material_gate_dataset -v
python -m unittest tests.test_contact -v
python -m unittest tests.test_v11a_config -v
```

- [ ] **Step 9: 提交 Task 3**

```bash
git add src/dataset/traj_dataset.py src/dataset/material_gate_dataset.py src/tests/test_material_gate_dataset.py
git commit -m "Add material-gate calibration dataset"
```

---

### Task 4: 冻结 Gate Rollout Trainer

**Files:**
- Create: `src/utils/material_stage_gate_training.py`
- Create: `src/train_material_stage_gates.py`
- Create: `src/tests/test_material_stage_gate_training.py`

**Interfaces:**
- Produces: `load_b3a_into_b3b(model, checkpoint_path) -> None`。
- Produces: `freeze_for_gate_training(model) -> nn.Parameter`。
- Produces: `gate_rollout_loss(model, sample, material_id, long_weight, reg_weight, accelerator) -> dict`。
- Produces: `MaterialBestRowTracker`，独立保存三种材质最佳 row。
- Produces CLI: `python train_material_stage_gates.py --config <yaml>`。

- [ ] **Step 1: 写 checkpoint 严格兼容失败测试**

用 fake loader/state dict 验证：只缺 `gate_logits` 时通过；额外 missing 或任何 unexpected key 时抛出包含 key 名的 `RuntimeError`。

- [ ] **Step 2: 写冻结参数失败测试**

构建小 B3b model，调用 `freeze_for_gate_training` 后断言：

```python
trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
self.assertEqual(trainable, [(expected_gate_name, 12)])
```

提供 `activate_material_row(model, material_id)` 后，backward 只允许当前行梯度非零。

- [ ] **Step 3: 实现加载与冻结 helper**

使用 `safetensors.torch.load_file(checkpoint_path, device="cpu")`，`strict=False` 后精确校验 missing/unexpected。先 `model.requires_grad_(False)`，再仅 `gate_logits.requires_grad_(True)`；optimizer 只接收这一张参数。

- [ ] **Step 4: 写 rollout 语义失败测试**

使用记录输入的 fake model，构造 5 帧输入和 3 帧 future GT，断言：

- 第一步使用 batch `start_vel`；
- 后续 single-frame step 使用滑动后窗口 `current[:,1]-current[:,0]`；
- 每步丢弃最老帧并 append `pred.detach()`；
- 模型收到的下一窗口不携带上一预测图；
- 三帧都贡献到同一 gate 参数的梯度。

- [ ] **Step 5: 实现逐帧 detach rollout**

复刻现有 `eval.py` 的 single-frame window 更新和 model 参数顺序。每步计算 position MSE，缓存标量 loss；为控制显存，每帧按最终预先可计算的权重归一后立即 backward。不得调用 `TrajPipeline`，因为 trainer 需要梯度。

- [ ] **Step 6: 写 loss 数学测试**

使用已知 `[1,2,3,4,5,6]` 六个 frame losses，最后三分之一为 `[5,6]`，断言：

```python
expected = mean([1, 2, 3, 4, 5, 6]) + 0.5 * mean([5, 6])
```

再加 `reg_weight=1e-3` 的 gate identity 正则；gate=1 时正则精确为 0。

- [ ] **Step 7: 写每材质 early stopping 与 row 合并测试**

模拟 validation 序列，断言每 25 updates 才评估、连续三次不改善停止、三种材质可来自不同 best update，合并后其他 row 不被覆盖。

- [ ] **Step 8: 实现 trainer orchestration**

流程固定为 elastic 0、plasticine 1、sand 2；每种材质重新从当前组合 gate 开始，只优化该 row，最多 200 updates。Adam 参数：`lr=3e-3`、`weight_decay=0`。validation score 以该材质 `full_mse + 0.5*long_mse` 最小为准，同时记录 identity gate 基准。

- [ ] **Step 9: 写保存与恢复测试**

临时目录中断言生成：`split_manifest.json/training_history.csv/best_gates.json/checkpoint-best/model.safetensors/checkpoint-best/gate_metadata.json`；重新构建 B3b 并加载完整模型后 gate 值精确一致。

- [ ] **Step 10: 实现 Accelerate CLI**

`train_material_stage_gates.py`：加载 OmegaConf、断言单进程、设置 seed、构建 model/dataset、运行三材质 trainer、保存产物。异常时输出当前 material/update，但不吞掉异常或保存伪 best checkpoint。

- [ ] **Step 11: 运行 trainer 测试**

```bash
python -m unittest tests.test_material_stage_gate_training -v
python -m py_compile utils/material_stage_gate_training.py train_material_stage_gates.py
```

Expected: 全部 PASS。

- [ ] **Step 12: 提交 Task 4**

```bash
git add src/utils/material_stage_gate_training.py src/train_material_stage_gates.py src/tests/test_material_stage_gate_training.py
git commit -m "Add frozen material-stage gate trainer"
```

---

### Task 5: 配置、评测镜像、台账与高强度验证

**Files:**
- Create: `src/configs/config_mm3_b3b_material_stage_gate_screen.yaml`
- Create: `src/configs/eval_mm3_b3b_material_stage_gate_screen.yaml`
- Modify: `src/tests/test_b3_material_state_config.py`
- Modify: `实验记录_1.md`

**Interfaces:**
- Produces training config with base checkpoint `outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors`。
- Produces eval config with checkpoint `outputs/mm3_b3b_material_stage_gate_screen/checkpoint-best/model.safetensors`。
- Preserves B3a model config，唯一模型差异为 `material_stage_gate=true` 与 `material_stage_gate_max=2.0`。

- [ ] **Step 1: 写配置镜像失败测试**

断言 training config 的 B3a model fields 完全一致，仅新增 gate fields；dataset 为 `mm3_data/mm3_train`。eval config model config 与 training model config 完全一致，dataset 为 `mm3_data/mm3_test`，`output_frames=1/use_diffusion=false/num_inference_steps=1`。

- [ ] **Step 2: 创建 frozen config**

顶层 gate trainer 字段固定：

```yaml
base_checkpoint: outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors
output_dir: outputs/mm3_b3b_material_stage_gate_screen
gate_train_fraction: 0.8
gate_split_seed: 0
gate_start0_probability: 0.5
gate_max_rollout_steps: 20
gate_updates_per_material: 200
gate_validation_interval: 25
gate_patience: 3
gate_learning_rate: 3.0e-3
gate_long_weight: 0.5
gate_reg_weight: 1.0e-3
```

复制 B3a model config 并增加：

```yaml
material_stage_gate: true
material_stage_gate_max: 2.0
```

- [ ] **Step 3: 创建 eval config**

镜像训练 model config；只把 dataset 切到 `mm3_data/mm3_test`，resume 指向 `checkpoint-best/model.safetensors`，vis_dir 使用独立 B3b 名称。

- [ ] **Step 4: 运行 config tests**

```bash
python -m unittest tests.test_b3_material_state_config -v
```

Expected: 全部 PASS，旧 B3a config tests 不变。

- [ ] **Step 5: 更新实验台账**

在 `实验记录_1.md` B3b 条目记录：设计 commit、plan commit、Task 1--5 commits、唯一变量、数据角色、训练命令、dev-test 只评一次的约束、成功标准和停止条件。状态设为“实现完成，等待服务器 calibration”，不预填结果。

- [ ] **Step 6: 执行所选测试强度的完整验证**

若 Will 选择推荐的“高”，从 `src/` 执行：

```bash
python -m unittest tests.test_material_stage_gate -v
python -m unittest tests.test_material_gate_dataset -v
python -m unittest tests.test_material_stage_gate_training -v
python -m unittest tests.test_material_state_adapter -v
python -m unittest tests.test_b3_material_state_config -v
python -m unittest tests.test_count_params -v
python -m unittest tests.test_contact -v
python -m unittest tests.test_v11a_config -v
python -m unittest tests.test_material_state_gradient_conflict -v
python -m unittest tests.test_material_state_stage_diagnostics -v
python -m py_compile model/material_state.py model/spacetime.py dataset/traj_dataset.py dataset/material_gate_dataset.py utils/material_stage_gate_training.py train_material_stage_gates.py count_params.py
python count_params.py
git diff --check
```

再执行一次独立代码审查，重点检查 checkpoint key、旧 config default-off、rollout start velocity、future GT 对齐、detach 边界、dev-test 隔离和仅 12 参数可训练。

- [ ] **Step 7: 执行 CPU smoke test**

用小模型、2 粒子、3-step future GT 跑一次完整 `load -> freeze -> rollout -> optimizer.step -> save -> reload`，断言只有 gate logits 改变，公共参数逐 tensor 相等。

- [ ] **Step 8: 提交 Task 5**

```bash
git add src/configs/config_mm3_b3b_material_stage_gate_screen.yaml src/configs/eval_mm3_b3b_material_stage_gate_screen.yaml src/tests/test_b3_material_state_config.py '实验记录_1.md'
git commit -m "Register B3b material-stage gate screening"
```

---

## Server Runbook After Merge

从服务器 `src/` 目录运行：

```bash
accelerate launch \
  --config_file configs/acc/1gpu.yaml \
  train_material_stage_gates.py \
  --config configs/config_mm3_b3b_material_stage_gate_screen.yaml
```

先检查日志必须包含：

```text
base checkpoint loaded with only gate_logits missing
total trainable parameters: 12
dev-test access during calibration: disabled
```

完成后确认：

```bash
ls outputs/mm3_b3b_material_stage_gate_screen/checkpoint-best/model.safetensors
cat outputs/mm3_b3b_material_stage_gate_screen/best_gates.json
```

然后只运行一次 dev-test：

```bash
python eval.py --config configs/eval_mm3_b3b_material_stage_gate_screen.yaml
```

将标准 eval、分材质结果、gate 值与训练时间返回后，再按预注册门槛裁决 `proceed/close`；不根据 dev-test 继续调 `lr/lambda/gate_max`。
