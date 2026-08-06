# B3b Material-Stage Gate Screening 设计

## 1. 目标与定位

B3b 是 B3a `FactorizedMaterialStateAdapter` 的一次有界机制筛选，不是完整模型训练臂。它服务一期目标 A/B/C/D，具体回答：在保持 B3a 主干和共享 adapter 方向不变的前提下，仅允许不同材质调整不同注入 stage 的修正强度，能否保留 sand 的长程收益，同时修复 plasticine 和 elastic 的退化。

B3b 由 B3a@90k 的冻结诊断直接驱动：

- stage0 对三种材质均承重，尤其支配 sand；
- stage1--3 对 sand 有益，但对 plasticine 多数有害；
- sand 的共享 adapter 梯度范数约为 plasticine 的 9.7 倍、elastic 的 52.3 倍；
- 全 adapter 梯度没有强烈的统一方向冲突，局部冲突主要出现在 `output_proj`；
- teacher-forced 单步 stage-scale 梯度与长程 knockout 结论不一致，因此 gate 不能只用单步目标训练。

B3b 不扩展到完整 per-material expert，不调整 adapter rank，不修改任意边界、contact、材料 token 或 Transformer block。若本筛选失败，关闭当前 B3 adapter 家族。

## 2. 数据角色与解释边界

当前 41-model `mm3_test` 已用于 B3a 裁决和 B3b 设计，后续改称 `dev-test`。它仍可用于阶段性方法筛选，但不能作为论文最终唯一的无偏测试集。

- `mm3_train/gate_train`：优化 gate。
- `mm3_train/gate_val`：选择每个材质的最佳 gate row 和提前停止。
- 当前 41-model `mm3_test`：方法冻结后只执行一次 B3b screening，报告为 `dev-test`。
- 新 `final-test`：最终候选方法确定后另行生成或保留，只用于论文最终统一评测。

`gate_train/gate_val` 从 `mm3_train` 按 model、按材质分层划分，比例约为 80%/20%，固定 seed 并保存实际 model 清单。B3a 主干曾见过 `mm3_train`，但在 B3b 中完全冻结；该划分只防止新增 gate 自身在相同轨迹上训练和选择。

## 3. 模型结构

### 3.1 B3a 原更新

B3a 在 stage `k` 的更新为：

\[
h' = h + a_k\Delta h_{\text{shared}},
\]

其中 `a_k` 是 B3a 已学习的共享 stage scale，所有材质共用。

### 3.2 B3b 更新

B3b 增加材质-stage gate：

\[
h' = h + a_k g_{m,k}\Delta h_{\text{shared}},
\]

其中 `m` 为 elastic、plasticine 或 sand，`k` 为 stage0--3。gate 参数化为：

\[
g_{m,k}=g_{\max}\,\sigma(\theta_{m,k}),\qquad g_{\max}=2.
\]

`theta` 初始化为 0，因此 `g=1`，B3b 在初始化和加载 B3a checkpoint 后与 B3a 函数一致。gate 范围为 `(0,2)`：小于 1 表示削弱该材质在该 stage 的 adapter 修正，大于 1 表示增强。禁止负 gate，避免在小 calibration set 上直接反转共享 adapter 方向。

只为三种实际训练材质创建 `3 x 4 = 12` 个有效参数。预留 rigid 类的 gate 固定为 1。

### 3.3 参数共享边界

- 8 层串行 `SpatialTemporalTransformerBlock` 不变；
- B3a 的 `state_norm/state_proj/material_proj/output_proj/stage_scales` 全部冻结；
- 原有 `E/nu/class` tokens、contact condition、start velocity 和输出头不变；
- 仅 `gate_logits` 可训练；
- gate 根据已知 `mat_type` 查表，不新增粒子级参数，也不随帧数或粒子数增长。

## 4. 向后兼容

新增 model config：

```yaml
material_stage_gate: false
material_stage_gate_max: 2.0
```

兼容性必须满足：

1. 未写或关闭 `material_stage_gate` 时，不创建 gate 参数，不改变旧 `state_dict` 和 forward 路径。
2. 所有旧 config 和 checkpoint 保持原行为。
3. 开启 B3b 时创建全零 `gate_logits`，不消耗随机数，不改变其他参数初始化。
4. 从 B3a checkpoint 初始化 B3b 时，只允许缺失 `gate_logits`；其他 missing/unexpected key 立即报错。
5. gate=1 时，对相同输入的 B3a/B3b 输出必须在测试容差内一致。
6. B3b 保存完整 `model.safetensors`，后续仍使用现有 `eval.py`，不改变旧评测路径。

## 5. Calibration 数据与起点采样

新增独立 gate calibration dataset，不修改 `TrajDataset` 的既有 train/val/test 语义。每个样本提供：

- 5 帧条件点云；
- 从下一帧直到轨迹末尾的 GT；
- 原有 `force/E/nu/mat_type/mask/drag_point/floor/gravity/base_drag_coeff/points_rest/start_vel`；
- 同一 rollout 内固定的 particle indices。

起点采样：

- 50% 使用 `start_idx=0`，覆盖完整 20-frame rollout；
- 50% 从其他合法起点随机选择，并 rollout 到轨迹末尾；
- 分层划分和随机起点均使用显式 seed；
- 不读取 `mm3_test`。

## 6. Rollout 优化

每个 rollout 保持现有 single-frame 推理语义：

```text
5 帧输入 -> 预测 1 帧 -> 滑动窗口 -> 继续预测
```

最长 rollout 20 个预测帧。为限制显存，每一步预测反馈前 detach，并逐步反向：

```python
pred_t = model(current_window)
(weighted_loss_t / normalizer).backward()
current_window = append(current_window, pred_t.detach())
```

该协议不是 full BPTT。它不能把第 20 帧误差穿过全部历史预测反传到第 1 帧，但同一组 gate 在每个 step 重复使用，晚期误差会在模型自身漂移后的状态上直接更新 gate。相比 teacher-forced 单步目标，它更贴近 B3a 暴露的 long-horizon 问题，同时避免 20-step 图导致的显存增长。

三种材质轮流训练，每次只启用当前材质对应的四个 gate logits。这样 sand 的大梯度不会直接缩放 elastic/plasticine gate 的更新。

## 7. Loss

对材质 `m`、当前 rollout 长度 `T`：

\[
L_m =
\frac{1}{T}\sum_{t=1}^{T}\operatorname{MSE}_t
+0.5\frac{1}{T_L}\sum_{t\in\text{last third}}\operatorname{MSE}_t
+\lambda_g\frac{1}{4}\sum_{k=0}^{3}(g_{m,k}-1)^2.
\]

- 第一项保护整体轨迹；
- 第二项明确提高最后三分之一 horizon 的权重；
- 第三项以 identity gate 为中心，抑制小 calibration set 上的极端补偿；
- `lambda_g` 在实现计划中冻结为单一值，不做 dev-test sweep；
- 不新增 `loss_F/contact/volume` 等训练项，保持 screening 变量单一。

## 8. 优化预算与模型选择

- 每种材质最多 200 个 gate updates，总计最多 600；
- Adam，学习率 `3e-3`，无 weight decay、无 scheduler；
- 每 25 updates 在对应材质的 `gate_val` 上执行完整 rollout；
- patience=3，连续三次无改善则停止该材质；
- 三种材质独立保存最佳 gate row，最终组合为一个 `3 x 4` gate；
- 最佳 row 由该材质的 full/long rollout 指标选择，不使用 dev-test；
- 训练前记录 identity gate 在相同 `gate_val` 上的基准。

预计总成本约为一万余次单帧 forward/backward，显著低于重新训练 90k-step 模型。该估算必须在服务器首次 25-update 区间后用实际 wall time 修正。

## 9. Screening 裁决

最终 gate 在 `gate_val` 选择完毕后冻结，然后只在当前 41-model dev-test 上执行一次标准 eval。

同时相对 `mm3_contact_cond@90k` 和 B3a@90k 判断：

1. plasticine long MSE 相对 B3a 改善至少 25%，并进入 baseline `+10%` 以内；
2. elastic long MSE 不超过 baseline `+10%`；
3. sand long MSE 相对 B3a 恶化不超过 5%，保留 B3a 的主要收益；
4. overall long MSE 和 FDE 不恶化；
5. 穿透率和穿透深度不比 B3a 更差；
6. gate 不得大面积饱和到接近 0 或 2。

通过后才允许从头训练公平 B3b，并在新的 final-test 上完成论文级验证。未通过则关闭当前 B3 adapter 家族，不继续调 rank、增加 expert 或追加 stage。

## 10. 实现文件与产物

计划新增或修改：

```text
src/model/material_state.py
src/model/spacetime.py
src/train_material_stage_gates.py
src/dataset/material_gate_dataset.py
src/configs/config_mm3_b3b_material_stage_gate_screen.yaml
src/configs/eval_mm3_b3b_material_stage_gate_screen.yaml
src/tests/test_material_stage_gate.py
src/tests/test_material_stage_gate_training.py
```

输出目录：

```text
outputs/mm3_b3b_material_stage_gate_screen/
  split_manifest.json
  training_history.csv
  best_gates.json
  checkpoint-best/
    model.safetensors
    gate_metadata.json
```

`best_gates.json` 记录 gate 值、每个材质的最佳 update、gate-val full/long MSE、identity 基准、seed、config 和源 checkpoint。

服务器入口：

```bash
accelerate launch \
  --config_file configs/acc/1gpu.yaml \
  train_material_stage_gates.py \
  --config configs/config_mm3_b3b_material_stage_gate_screen.yaml

python eval.py \
  --config configs/eval_mm3_b3b_material_stage_gate_screen.yaml
```

## 11. 不做的方案与后续分叉

本轮不实现：

- 完整 per-material expert；
- per-material `output_proj` 或 LoRA；
- GradNorm/PCGrad 等共享 adapter 优化器；
- 由 `E/nu` 连续生成 gate 的 hypernetwork；
- full-rollout BPTT；
- 新边界表示或新物理 loss。

若 B3b 成功，下一步优先从头训练公平 B3b；之后再根据 gate 是否随材质形成稳定规律，决定是否升级为 `E/nu/material` 连续 gate。若 B3b 失败，仅当结果明确指向 sand 梯度幅度支配时才考虑 material-balanced retraining；仅当更多重复证据确认 `output_proj` 局部方向冲突时才考虑小型 per-material LoRA。

## 12. 验证要求

实现前按 `AGENTS.md` 选择测试强度。由于本改动涉及模型参数、checkpoint 兼容、训练协议和数据划分，推荐使用“高”。至少验证：

- gate 数学范围、identity 初始化、材质/rigid 查表；
- 关闭 gate 时旧模型 state dict 和输出不变；
- B3a checkpoint 只缺 gate key；
- 冻结后只有 12 个参数可训练，每个材质阶段只更新 4 个；
- rollout detach、窗口滑动、start velocity 与 eval 语义一致；
- split 分层、互斥、可复现且不读取 mm3_test；
- 完整模型保存后可被现有 eval config 加载；
- Python 语法、相关测试、参数统计和 `git diff --check`。
