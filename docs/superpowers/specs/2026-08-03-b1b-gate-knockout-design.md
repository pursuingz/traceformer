# B1b HST Gate Knockout 设计规格

## 1. 目的

B1b（v11a + additive `contact_cond`）在 41-model held-out test 上表现为混合负：相对原版 `contact_cond@90k`，`full-rollout MSE` 仅改善 0.7%，但 `FDE`、plasticine 晚程误差和地面穿透明显恶化。已有 feedback diagnostics 进一步表明：

- 四个共享 gate 为 `+0.039516 / +0.015344 / -0.013035 / -0.001890`；
- stage 0/2 承担主要反馈幅值；
- feedback 的粒子均值能量占比为 98%--99.998%，近似全局广播；
- feedback 幅值与 `log10(E)` 高度相关，但相关性不能证明其导致 plasticine 退化。

本实验通过**同一 checkpoint 上的推理期 gate 因果干预**，判断固定 stage gate 是否在不同材质之间产生冲突，并据此决定是否值得训练 material-conditioned dynamic gate。

## 2. 实验问题

1. 整体关闭 HST feedback 后，B1b checkpoint 的轨迹误差和接触误差如何变化？
2. stage 0、1、2 分别对 elastic、plasticine、sand 的长程预测起什么作用？
3. 是否存在同一 stage 对 plasticine 有害、对 sand 有益的稳定相反响应？
4. 这种响应是否在材质内部随 `E` 分层而变化？

本实验不回答：

- HST 是否优于另一个独立训练的无 HST baseline；
- dynamic gate 应采用哪种网络结构；
- HST feedback 是否具有局部粒子关系表达能力；
- 任意边界或新 contact feature 的效果。

## 3. 冻结口径

- checkpoint：`outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors`
- config：`src/configs/eval_mm3_v11a_contact_cond_8L_45k.yaml`
- test split：`src/configs/mm3_test_split.json` 中冻结的 41 个模型
- 材质数量：elastic/plasticine/sand = 13/14/14
- 窗口：每个模型仅 `start_idx=0`
- 输入/输出：5 帧输入，单帧预测，逐帧 autoregressive rollout 到 frame 24
- 推理：eager、CUDA、bf16、`use_diffusion=false`、`floor_projection=false`
- 模型权重、输入点云、`E`、`nu`、`mat_type`、地板、外力、点采样索引均不变

## 4. 干预条件

模型只加载一次。设训练后的四个 gate 为：

\[
g=(g_0,g_1,g_2,g_3).
\]

每种条件使用二值 mask `m`，实际 gate 为逐元素乘积：

\[
g' = g \odot m.
\]

| 条件 | mask | 含义 |
|---|---|---|
| `normal` | `[1,1,1,1]` | 原 checkpoint，作为同次运行配对基准 |
| `all_off` | `[0,0,0,0]` | 关闭全部 HST feedback |
| `stage0_off` | `[0,1,1,1]` | 关闭第 2 层后的反馈 |
| `stage1_off` | `[1,0,1,1]` | 关闭第 4 层后的反馈 |
| `stage2_off` | `[1,1,0,1]` | 关闭第 6 层后的反馈 |

不单独运行 `stage3_off`：其 gate 只有 `-0.001890`，已有诊断显示幅值接近关闭；`all_off` 已覆盖移除 stage 3 的组合情形。

mask 必须乘在 checkpoint 中训练后的 gate 上，不能把保留的 gate 改成 1。

## 5. 执行架构

新增独立脚本 `src/diagnose_hybrid_state_gate_knockout.py`，不修改训练路径和模型 `forward`。

数据流：

1. 沿用现有 B1b diagnostics 的 config、checkpoint、dataset manifest 和 batch 完整性校验。
2. 构建一次模型和 pipeline，严格加载 checkpoint。
3. 对每个 `start_idx=0` batch，依次运行五个干预条件。
4. 每次 rollout 前重置同一个 seed，使五个条件使用相同随机协议。
5. 临时写入 masked gates；rollout 完成后立即恢复原 gate。
6. 即使 rollout 抛出异常，也通过上下文管理器的 `finally` 恢复 gate。
7. 对同一模型计算 `knockout - normal` 的配对差值。

脚本必须验证：

- gate 张量恰好包含 4 个有限标量；
- 进入干预前 gate 与 checkpoint 值一致；
- 每次条件运行后 gate 被精确恢复；
- 每个模型恰有五个条件结果；
- 总计 41 个唯一模型，无遗漏、重复或非有限指标。

## 6. 指标

每个模型、每个条件计算：

- `full_rollout_mse`：frame 5--24 的平均位置 MSE；
- `short_mse`：frame 5--10；
- `mid_mse`：frame 11--17；
- `long_mse`：frame 18--24；
- `gm_mse`：逐帧 MSE 的几何均值；
- `fde`：frame 24 的逐点 L2 均值；
- `f24_centroid_error`：frame 24 质心误差；
- `f24_shape_residual_mse`：frame 24 Procrustes 对齐后形状残余；
- `penetration_rate`：预测 frame 5--24 中低于地板的粒子-时间比例；
- `penetration_depth`：预测 frame 5--24 的平均穿透深度，归一化坐标单位。

配对差值统一定义为：

\[
\Delta = \text{knockout} - \text{normal}.
\]

以上指标均为越低越好，因此 `Delta < 0` 表示关闭该 gate 后改善。

## 7. 分组和统计

输出以下分组：

1. overall；
2. elastic、plasticine、sand；
3. 每种材质内部按该材质 `log10(E)` 中位数划分 `low_E` / `high_E`。

每个分组报告：

- normal 与 knockout 的模型等权均值；
- 配对绝对差值和相对变化；
- 配对差值中位数；
- 改善模型数 / 总模型数；
- 固定 seed 的 model-level paired bootstrap 95% CI。

`E` 分层只作为机制诊断，不作为独立显著性结论；样本量小，不以单一 p 值裁决。

## 8. 输出文件

默认输出目录由 CLI `--output-dir` 指定，生成：

1. `hybrid_state_gate_knockout_b1b_90k_raw.csv`
   - 41 models × 5 conditions = 205 行；
2. `hybrid_state_gate_knockout_b1b_90k_paired.csv`
   - 41 models × 4 knockout conditions = 164 行；
3. `hybrid_state_gate_knockout_b1b_90k.md`
   - 实验元数据、原 gate、完整性检查、overall/材质/`E` 分层统计和预注册裁决结果。

CSV 顶部或字段中必须记录 config、checkpoint、seed 和样本口径，防止结果文件脱离实验上下文。

## 9. 预注册判定门

优先检查 `stage0_off` 和 `stage2_off`，因为它们承担主要 feedback 幅值。

只有同时满足以下条件，才允许进入 material-conditioned dynamic gate 训练：

1. 同一个 stage 关闭后，plasticine 的 `long_mse`、`fde` 或 `f24_centroid_error` 中至少两个改善不低于 5%；
2. 对应指标至少 8/14 个 plasticine 模型同方向改善；
3. 同一 knockout 对 sand 的相应长程指标产生不低于 5% 的退化，或显示稳定、可解释的相反材质响应；
4. 结论不是由一个极端模型单独驱动，配对中位数方向与均值一致；
5. overall 与穿透指标不存在足以否定该机制的灾难性退化。

若没有稳定的材质相反响应，或所谓改善只由少量样本/单一指标驱动，则关闭 v11a/HST 路线，不再投入新的训练预算。

`all_off` 仅用于判断 HST feedback 在**同一个已训练 B1b checkpoint 内**的净作用。由于主干曾与 HST 联合训练，`all_off` 不能替代独立训练 baseline，也不能用于宣称 HST 架构本身的最终增益。

## 10. 测试和验收

先写测试，再实现脚本。至少覆盖：

- mask 构造和条件顺序；
- mask 使用训练后 gate，而不是二值替换；
- 正常结束和异常结束都恢复原 gate；
- 相同 batch 下每个条件重新使用同一 seed；
- 轨迹、分段、FDE、Procrustes、穿透指标的数值口径；
- paired row 数量、符号和相对变化；
- material/`E` 分层与 model-level 等权聚合；
- 41-model manifest、checkpoint、config 和 `start_idx=0` 保护；
- CLI 输出路径和元数据。

验证命令将在实施计划中固定，至少包括目标单元测试、完整诊断测试集和 `py_compile`。本地不具备 41-model 数据与 GPU 时，只运行单元测试；真实 rollout 在服务器执行。

## 11. 范围控制

本轮不做以下工作：

- 不训练新模型；
- 不修改 HST、contact adapter 或串行 Transformer 主干；
- 不实现 dynamic gate；
- 不新增 `stage3_off`；
- 不改变 test split、checkpoint、eval config 或指标定义；
- 不把 inference-only knockout 结果写成独立训练架构的公平对比。
