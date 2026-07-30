# Contact 因子化 v_xyz 适配器设计

## 1. 目标

重新检验完整逐粒子运动状态 `v_xyz`，但避免已有 `v_xyz` 实验中
`v_x/v_z` 与边界、法向运动特征从训练开始就通过同一个线性映射共同更新。

新实验不是简单地把五维输入拆成两个 `Linear`。如果两个线性输出直接相加，
它仍与 `Linear(5 -> 256)` 仿射等价，不能形成新的方法。依据已有消融证据：

- `v_y` 对三种材料均有效，对 plasticine 尤其重要；
- `v_x/v_z` 在旧 `v_xyz` 中总体有害；
- `d/q` 描述边界状态，`v_y` 描述法向运动，`v_x/v_z` 描述切向运动，
  三者承担的物理角色不同。

因此采用三个分支，并只对风险最大的切向分支增加可学习零门控：

```text
boundary branch:   [signed_gap, proximity]
normal branch:     [v_y]
tangential branch: [v_x, v_z] * learned zero gate
```

## 2. 严格实验问题

核心比较为：

```text
旧 v_xyz:
    Linear([d, v_x, v_y, v_z, q]) + bias

新 factorized v_xyz:
    Linear_b([d, q])
  + Linear_n([v_y])
  + tanh(alpha_t) * Linear_t([v_x, v_z])
  + shared_bias
```

两臂使用完全相同的五维输入。数据、采样、损失、Transformer block、隐藏维度、
优化器、学习率计划、训练步数、seed 和评测 split 全部冻结。新实验相对
`config_mm3_contact_vxyz.yaml` 只允许改变：

- `output_dir`；
- `model_config.contact_injection_mode`。

`stop_after_steps` 两边均为 45000，不构成差异。

## 3. 数学定义

对条件帧中粒子 `i` 在帧 `t` 的特征，定义：

\[
c^b_{t,i}=[d_{t,i},q_{t,i}],\qquad
c^n_{t,i}=[v^y_{t,i}],\qquad
c^t_{t,i}=[v^x_{t,i},v^z_{t,i}]
\]

其中：

\[
d_{t,i}=y_{t,i}-h_{\mathrm{floor}},\qquad
q_{t,i}=\exp\left[-\left(\frac{\max(d_{t,i},0)}{\sigma}\right)^2\right],
\quad \sigma=0.04
\]

位移仍沿用现有定义，不除以 `dt`：

\[
v_{t,i}=x_{t,i}-x_{t-1,i}
\]

首个条件帧优先使用已有 `start_velocity`，否则复制第二帧对应的帧差。

因子化 contact hidden 为：

\[
h^c_{t,i}
=W_b c^b_{t,i}
+W_n c^n_{t,i}
+\tanh(\alpha_t)W_t c^t_{t,i}
+b_c
\]

随后仅对真实条件帧做逐元素残差相加：

\[
h_{t,i}\leftarrow h_{t,i}+h^c_{t,i}
\]

输出槽和 mask 伪帧不注入 contact hidden。

这里的 `+` 是 256 维向量逐元素相加，不是拼接。

## 4. 模块与参数

新增独立模块 `FactorizedContactAdapter`，接口接收五维 contact feature，输出
同形状的 `latent_dim` hidden：

```text
boundary_encoder:   Linear(2 -> 256, bias=False)
normal_encoder:     Linear(1 -> 256, bias=False)
tangential_encoder: Linear(2 -> 256, bias=False)
shared_bias:        Parameter(256)
tangential_gate:    Parameter(1)
```

参数量：

```text
boundary weight     2 * 256 =  512
normal weight       1 * 256 =  256
tangential weight   2 * 256 =  512
shared bias             256 =  256
tangential gate           1 =    1
total                         1537
```

旧 `Linear(5 -> 256)` 为 1536 参数，因此新旧只差一个标量门控参数。所有
Linear 仍在帧轴、粒子轴和 batch 轴共享权重。

## 5. 初始化与优化路径

初始化要求：

- `W_b=0`；
- `W_n=0`；
- `b_c=0`；
- `alpha_t=0`；
- `W_t` 使用 `nn.Linear(2, latent_dim, bias=False)` 的默认
  Kaiming-uniform 非零初始化。

因此 step 0：

\[
h^c_{t,i}=0
\]

新模型与旧 `v_xyz` 的前向输出必须位级一致。由于 `W_t` 非零，
`alpha_t` 在第一个反向传播中能获得梯度；若同时把 `W_t` 和
`alpha_t` 置零，切向分支会因两个梯度都为零而永久失活。

为了排除随机初始化混淆，新模块构造必须满足：

1. 先构造并立即丢弃一个临时 `nn.Linear(5, latent_dim)`，只用于推进全局
   RNG，精确复现旧 `v_xyz` contact encoder 的 RNG 消耗；
2. 三个实际分支和 `W_t` 的初始化在局部 `torch.random.fork_rng` 中完成，
   离开作用域后恢复到临时 Linear 构造后的 RNG 状态；
3. 相同 seed 下，新旧模型的公共 Transformer 参数逐张量位级一致。

这使实验真正比较参数化方式，而不是比较两套不同的主干初值。

## 6. 配置接口与兼容性

扩展：

```yaml
model_config:
  contact_particle_cond: true
  contact_velocity_mode: xyz
  contact_injection_mode: factorized
```

规则：

- `separate`：保留当前独立单 Linear 路径；
- `shared`：保留当前与 PointEmbed 拼接路径；
- `factorized`：启用本设计，仅允许 `contact_velocity_mode: xyz`。

默认值仍为 `separate`。任何旧配置、旧 checkpoint 和已有实验路径均不得改变。
非法组合必须在模型构造时抛出明确错误。

新增配置：

```text
src/configs/config_mm3_contact_vxyz_factorized.yaml
src/configs/eval_mm3_contact_vxyz_factorized_45k.yaml
```

训练配置严格复制 `config_mm3_contact_vxyz.yaml`，只修改：

```yaml
output_dir: ./outputs/mm3_contact_vxyz_factorized_8L
model_config:
  contact_injection_mode: factorized
```

评测配置镜像训练配置，指向：

```text
outputs/mm3_contact_vxyz_factorized_8L/checkpoint-45000/model.safetensors
```

## 7. 诊断与评测

45000 步筛选时首先与旧 `mm3_contact_vxyz@45k` 比较，再与
`mm3_contact_cond@45k` 比较。

主指标：

- full-rollout MSE；
- GM-MSE、FDE；
- 分材质 GM-MSE/FDE；
- vMSE、aMSE；
- contact-region MSE、normal-vMSE；
- contact precision/recall；
- 地面穿透率和穿透深度。

新增参数诊断：

- `tanh(alpha_t)` 的最终值；
- `W_b/W_n/W_t` 的 Frobenius norm；
- 三个分支输出 hidden 的平均范数；
- E/nu counterfactual sensitivity，确认切向捷径没有再次压低材料敏感性。

判据：

1. 若相对旧 `v_xyz`，full-rollout MSE 和 FDE 至少改善 10%，且不存在任一
   材料指标恶化超过 10%，说明因子化与门控有效；
2. 若仍明显差于 `contact_cond`，则它只能作为“修复了部分 v_xyz 退化”的
   消融结果，不能升级为主方法；
3. 若 `tanh(alpha_t)` 接近 0 且性能回到 `contact_cond`，说明数据不需要
   显式切向速度；
4. 若门控显著非零且改善 plasticine/sand，同时保持 E/nu 敏感性，才支持
   “法向/切向运动因子化”的方法主张。

## 8. 测试要求

测试先于实现，必须覆盖：

1. `FactorizedContactAdapter` 的特征切片顺序正确；
2. 参数形状、总参数量 1537 以及共享 bias 正确；
3. 初始化输出严格为零；
4. 初次 backward 时 `tangential_gate` 梯度非零；
5. gate 为零时 `W_t` 梯度为零，gate 非零后 `W_t` 能获得梯度；
6. 相同 seed 下 factorized 与旧 `v_xyz` 的公共主干参数位级一致；
7. 相同输入下，两模型 step-0 完整前向输出位级一致；
8. `factorized + vertical` 等非法组合明确报错；
9. 原 `separate/shared` 测试全部继续通过；
10. 训练与评测配置除声明字段外严格镜像。

验证命令：

```bash
python -m unittest src.tests.test_contact
python -m unittest src.tests.test_contact_ablation_configs
python -m py_compile src/utils/contact.py src/model/spacetime.py src/count_params.py
python src/count_params.py
```

## 9. 不在本次范围

- 不修改 8 层串行 Transformer 主干；
- 不增加 contact loss 或 floor projection；
- 不改变 `signed_gap/proximity` 定义；
- 不做逐层 contact 重注入；
- 不将 `v_xyz` 与 E/nu 做显式 FiLM 或门控；
- 不续训旧 checkpoint，新臂从头训练。

这些方向若需要测试，应作为后续独立实验，不能混入本次因子化消融。
