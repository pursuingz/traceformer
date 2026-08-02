# B1b：v11a 全局状态交换与 contact_cond 组合实验设计

**状态：** 已完成口头设计，等待书面复核后进入实施计划

**日期：** 2026-08-02

**实验名：** `mm3_v11a_contact_cond_8L`

## 1. 研究问题

当前两条已经独立验证的机制解决不同问题：

- `contact_cond` 把逐粒子的地板关系与竖直运动状态注入粒子 hidden，是当前 mm3 长程预测和接触表现的主力机制；
- v11a（MC-HST）每隔两层提取五帧全局状态，并把材料调制后的状态反馈给预测帧粒子，单独使用时能改善无 contact baseline 的多数整体指标，但无法阻止 plasticine 的晚程整体漂移和严重穿透。

B1b 不再提出第三套模块，而是回答一个严格的组合问题：

> 当粒子 hidden 已包含局部边界状态时，v11a 的全局状态交换能否进一步改善长程精度，同时保持 contact_cond 已取得的边界交互质量？

## 2. 阶段编号

- **B0：** 已完成。诊断 `mm3_contact_cond@90k` 是否依赖材料条件。
- **B0.1：** 已完成。分别置换 `E`、`nu`、参数对和 `mat_type`，分解材料条件依赖。
- **B1a：** 已完成。v11a MC-HST 单独对比无 contact 的 mm3 baseline。
- **B1b：** 本设计。v11a MC-HST 与原版 additive `contact_cond` 组合。
- **B2：** 暂缓。只有在确认材料参数可辨识后，才研究更强的材料-状态交互注入。

B1b 是 B0.1 之后的当前训练阶段，不把已完成的 B0/B0.1 重新当作未来任务。

## 3. 现有证据与假设

### 3.1 B0/B0.1 的约束

`contact_cond@90k` 的置换诊断表明：

- `mat_type` 是最强条件；
- `E` 在 plasticine 和 sand 上被明确使用，在 elastic 上基本被忽略；
- `nu` 整体上接近被忽略或结论不明确；
- 因而本实验不把“更强 nu 注入”混入 v11a/contact 的组合验证。

### 3.2 B1a 的结论

v11a@90k 相对同 step、无 contact baseline@90k：

| 指标（越低越好） | baseline@90k | v11a@90k | 相对变化 |
|---|---:|---:|---:|
| first-chunk MSE | 3.293603e-05 | 2.384503e-05 | -27.6% |
| full-rollout MSE | 1.433783e-01 | 1.184186e-01 | -17.4% |
| Chamfer | 6.195522e-01 | 4.744260e-01 | -23.4% |
| GM-MSE | 2.163055e-02 | 1.298758e-02 | -40.0% |
| FDE | 1.008294 | 0.902687 | -10.5% |
| MSE@f24 | 4.992508e-01 | 5.164555e-01 | +3.4% |
| 穿透率 | 22.004% | 18.796% | -14.6%（绝对值仍不可接受） |

分材料结果并不一致：elastic 和 sand 的长程指标改善；plasticine 的 short/mid 改善，但 long、FDE 和 f24 没有净胜。90k 相对 45k 已继续改善，因此这不是简单的欠训练，而是“全局状态有用，但缺少接触闭环”这一结构性现象。

### 3.3 B1b 假设

- **H1：互补。** `contact_cond` 先把局部接触状态写入五帧粒子 hidden，v11a 随后从这些 hidden 中池化全局状态，因此状态 token 可以自然携带接触后的整体运动信息。
- **H2：非冗余。** `contact_cond` 主要解决局部边界关系；v11a 主要解决跨帧整体状态与材料调制。二者组合应优于只使用 contact。
- **H0：无增量或干扰。** `contact_cond` 已足以支持原串行主干学习整体运动，v11a 只增加优化负担，或者把局部接触信息过度压缩为一个全局状态，从而使 11 臂不优于 01 臂。

## 4. 2×2 因子矩阵

| 代号 | v11a 全局状态交换 | additive contact_cond | 实验臂 |
|---|---:|---:|---|
| 00 | 关 | 关 | `mm3_singleframe_geom_deform_d0001` |
| 01 | 关 | 开 | `mm3_contact_cond` |
| 10 | 开 | 关 | `mm3_v11a_mc_hst_8L` |
| 11 | 开 | 开 | **`mm3_v11a_contact_cond_8L`（B1b）** |

主比较是 **11 vs 01**：在当前强基线 `contact_cond` 上，v11a 是否有净增量。

次比较是 **11 vs 10**：contact 是否修复 v11a 的 plasticine 晚程漂移与穿透。

00 只用于解释主效应和交互，不作为 B1b 的主要竞争基线。

对任意越低越好的误差指标 `M`，报告：

```text
v11a effect without contact = M10 - M00
v11a effect with contact    = M11 - M01
interaction                 = M11 - M10 - M01 + M00
```

`interaction < 0` 表示在该误差的加性尺度上存在协同；同时必须报告四个原始值和相对变化，不能只报告交互项。

## 5. 架构与信息路径

B1b 复用现有模块，不改变 8 个串行 `SpatialTemporalTransformerBlock`：

```text
五帧粒子位置
  ├─ Point/Fourier input encoder ───────────────┐
  └─ [signed_gap, vertical_displacement, proximity]
       └─ zero-init Linear(3 -> 256) ── 数值相加 ┘
                    ↓ contact-conditioned particle hidden
原串行 Blocks 1-2
                    ↓
共享 HybridStateExchange #1
  ├─ 从五帧粒子 hidden 做可学习池化
  ├─ 加入显式 COM/COM velocity/covariance/covariance delta
  ├─ 加入连续 [E, nu] 材料上下文
  └─ state -> prediction-particle cross-attention（zero gate）
                    ↓
Blocks 3-4 -> Exchange #2 -> Blocks 5-6 -> Exchange #3
                    ↓
Blocks 7-8 -> Exchange #4 -> 单帧位置输出
```

真实执行顺序是：

1. `contact_cond` 只对五个物理历史帧构造逐粒子特征；
2. 独立 `Linear(3, 256)` 编码后，与对应历史帧的粒子 hidden 做逐元素加法；
3. mask 伪帧随后插入，形成 `[mask, five history, one prediction]`；
4. v11a 每两层只池化 five history，不池化 mask 或 prediction；
5. v11a 只把共享状态反馈给 prediction frame，原串行块与输出头保持不变。

因此二者虽然没有新增显式“contact × state”乘法层，但并非彼此隔离：v11a 的 learned pooling 读取的正是已经加入 contact 信息的粒子 hidden，交互由现有状态交换自然学习。

## 6. 唯一变量与配置

训练锚点必须是：

```text
src/configs/config_mm3_contact_cond.yaml
```

B1b 训练配置只允许改变：

```yaml
output_dir: ./outputs/mm3_v11a_contact_cond_8L
stop_after_steps: 45000

model_config:
  transformer_block: SpatialTemporalTransformerBlockv11a
  hybrid_state_dim: 64
  hybrid_state_heads: 4
  hybrid_state_interval: 2
```

其余全部继承并冻结：

- mm3 数据与 41-model test split；
- 2048 粒子、5 帧输入、1 帧输出、随机窗口；
- `signed_gap + vertical_displacement + proximity`；
- `contact_injection_mode: separate` 和 zero-init additive encoder；
- 8 层原串行主干、latent 256；
- loss 与全部权重；
- batch、gradient accumulation、optimizer、learning rate、scheduler horizon、seed；
- `max_train_steps: 90000`。

`stop_after_steps: 45000` 只控制首次筛选何时停止，不改变 90000-step scheduler horizon。若通过 45k 门槛，再单独生成继续训练配置，使同一 output directory 从 checkpoint-45000 恢复到 90000；不得在门槛判定前预先消耗完整训练预算。

## 7. 评测协议

45k 必须使用与训练配置镜像的独立 eval 配置，并与已有 00/01/10 三臂的 checkpoint-45000 在完全相同条件下比较：

- deterministic single-frame rollout；
- seed 0（确定性臂只用于固定辅助随机过程）；
- 41 个 held-out model，164 个窗口，41 个 full-horizon 窗口；
- overall + elastic/plasticine/sand 分组；
- per-frame f5-f24、short/mid/long、GM-MSE、FDE；
- Procrustes 位姿/形状分解；
- vMSE/aMSE、体积、地面接触与穿透指标；
- 实测推理时间与参数量。

主判据不使用 first-chunk 单项，不用训练 loss，也不以 GIF 目测替代 held-out 指标。

## 8. 45k 预注册判定门槛

以 `mm3_contact_cond@45k` 为主基线：

| 指标 | 01 基线值 |
|---|---:|
| full-rollout MSE | 4.659667e-03 |
| GM-MSE | 2.060440e-03 |
| FDE | 1.390408e-01 |
| short / mid / long | 3.026180e-04 / 4.285225e-03 / 1.209705e-02 |
| 穿透率 | 1.222% |
| 穿透深度 | 1.111329e-04 |

**只有同时满足以下条件，才继续到 90k：**

1. `full-rollout MSE`、`GM-MSE`、`FDE` 中至少一项改善不低于 5%；
2. 上述另外两项均不得恶化超过 3%；
3. elastic、plasticine、sand 各自的 long-segment MSE 均不得恶化超过 5%；
4. 三种材料中至少两种的 long-segment MSE 改善不低于 5%；
5. 穿透率不高于 1.5%，穿透深度不高于 1.25e-04；
6. per-frame 曲线不得出现“前段改善、f18 后连续系统性反转”的模式。

出现以下任一情况立即停止该路线：

- full-rollout 或 FDE 恶化超过 5%；
- plasticine long-segment 恶化超过 5%；
- 接触指标越过上述上限；
- 只有 first-chunk 改善，长程指标无改善；
- 所有主指标改善均小于 5%，不足以覆盖训练随机性和额外成本。

门槛失败后不继续搜索 `state_dim`、heads、interval 或 gate 初始化，避免把一次预注册架构实验扩展成昂贵超参搜索。

## 9. 结果分支

### 9.1 B1b 通过

继续同一 run 到 90k，随后：

1. 与 01@90k 做主比较；
2. 计算完整 2×2 交互；
3. 重点检查 plasticine 的 centroid/FDE、f18-f24 和 penetration；
4. 若增益跨材料稳定，再考虑 B2 的材料-状态交互注入或任意边界扩展。

### 9.2 B1b 失败

关闭 v11a 组合路线，不调 v11a 超参。主线回到 `contact_cond`：

1. 保留 B1a 作为“全局状态单独有帮助但无法替代接触建模”的消融；
2. 优先把地板标量特征推广为任意边界点云/SDF/法向与切向状态；
3. B2 材料交互只在有可辨识数据或配对 counterfactual GT 后重启。

## 10. 实施范围与验证

预计不修改 `src/model/spacetime.py` 的算法逻辑。实施阶段只应新增：

- B1b 训练配置；
- B1b 45k eval 配置；
- 配置差异与组合 forward/backward 测试；
- 参数预算检查；
- `实验记录.md` 的待测条目。

最低验证要求：

1. B1b 相对 `config_mm3_contact_cond.yaml` 只存在预注册字段差异；
2. `contact_particle_cond=true`、`contact_injection_mode=separate` 与 v11a 同时生效；
3. 组合模型接受 `[mask, five history, one prediction]` 布局；
4. contact encoder 和 hybrid exchange 均收到非零梯度；
5. zero-init 条件下新增反馈不会在 step 0 静默破坏基线路径；
6. Python 编译、相关 unittest、参数统计和 `git diff --check` 全部通过。

若组合 smoke test 暴露真实接口不兼容，允许做最小路由修复，但必须先补失败测试；不得借机修改 contact 特征、v11a 状态定义或串行 backbone。
