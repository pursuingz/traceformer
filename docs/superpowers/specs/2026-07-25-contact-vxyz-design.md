# Contact v_xyz 条件设计

## 1. 目标

检验相较于当前仅提供竖直位移的接触条件，显式提供完整逐粒子位移向量能否改善多材质动力学预测。

本实验只改变一个因素：

```text
基线:  [signed_gap, v_y, proximity]
v_xyz: [signed_gap, v_x, v_y, v_z, proximity]
```

数据、采样、损失、Transformer block、隐藏维度、优化器、学习率计划、训练 seed 和评测 split 全部保持不变。

## 2. 特征定义

对于形状为 `(B, F, N, 3)` 的条件点云 `X`，定义：

```text
v[:, t] = X[:, t] - X[:, t - 1]       当 t > 0
v[:, 0] = start_velocity              当 start_velocity 可用
v[:, 0] = v[:, 1]                     否则
```

位移不除以 `dt`，以保持现有 `v_y` 通道的数值尺度和语义。

接触特征顺序为：

```text
[signed_gap, v_x, v_y, v_z, proximity]
```

其中：

```text
signed_gap = y - floor_height
proximity  = exp(-(relu(signed_gap) / sigma)^2)
sigma      = 0.04
```

## 3. 编码器

推荐方案使用一个逐点共享编码器：

```python
contact_encoder = nn.Linear(5, latent_dim)
```

所有 batch 样本、条件帧和粒子复用同一组权重。相较于 `Linear(3, 256)`，只增加 512 个权重：

```text
(5 - 3) * 256 = 512
```

该层继续采用 zero-init，注入位置和 bias 行为与基线完全一致。

本实验刻意不为 gap、velocity 和 proximity 分别创建 Linear。如果这些分支只在线性变换后相加：

```text
W_gap gap + W_vel velocity + W_prox proximity
```

则在数学上等价于对拼接输入使用一个 Linear。只有当各分支具有独立的非线性、门控、归一化、分层注入或正则化时，多个 Linear 才具有更强表达能力；这些改动会混淆本实验。

## 4. 向后兼容

新增配置：

```yaml
model_config:
  contact_velocity_mode: vertical  # default
```

支持以下模式：

- `vertical`：保持现有三通道行为和旧 checkpoint 参数形状。
- `xyz`：启用新的五通道行为。

默认模式必须保证现有配置和 checkpoint 不受影响。

feature mask 随模式变化：

```text
vertical: [gap, v_y, proximity]
xyz:      [gap, v_x, v_y, v_z, proximity]
```

非法模式名和 mask 长度必须抛出明确错误。

## 5. 实验配置

新增：

```text
src/configs/config_mm3_contact_vxyz.yaml
src/configs/eval_mm3_contact_vxyz_45k.yaml
```

训练配置严格镜像 `config_mm3_contact_cond.yaml`，仅修改：

```yaml
output_dir: ./outputs/mm3_contact_vxyz_8L
max_train_steps: 90000
stop_after_steps: 45000
model_config:
  contact_velocity_mode: xyz
```

评测配置镜像对应训练配置中的模型和数据设置，指向 checkpoint 45000，并使用相同的 41 个 held-out model 和 seed 0。

由于 `Linear(3, 256)` 与 `Linear(5, 256)` 的 checkpoint 形状不同，新实验从头训练。

## 6. 评测

在相同步数下与 `mm3_contact_cond_8L/checkpoint-45000` 比较。

主指标：

- full-rollout MSE；
- GM-MSE 和 FDE；
- 分材质 MSE/FDE；
- contact-region MSE；
- contact normal-vMSE；
- contact precision 和 recall；
- 地面穿透率和穿透深度。

结果解释：

- elastic 改善：说明完整速度提供了通用运动学收益；
- plasticine/sand 改善：说明水平速度帮助切向接触、滑动或颗粒流动建模；
- 仅首帧改善而 rollout 不改善：说明没有长期收益；
- 没有改善：说明模型已经能以可接受代价从位置历史恢复 `v_x` 和 `v_z`。

## 7. 验证

测试必须覆盖：

- 旧三通道路径的精确数值和形状；
- 五通道路径的精确数值，包括 `start_velocity`；
- 非法 mode 和 mask 长度错误；
- 旧 checkpoint 兼容的模型构造；
- 新五通道模型构造和 forward 形状；
- 训练与评测配置镜像；
- 除声明的实验变量外，其余变量全部冻结。

运行：

```bash
python -m py_compile src/utils/contact.py src/model/spacetime.py
python -m unittest src.tests.test_contact
python -m unittest src.tests.test_contact_ablation_configs
python src/count_params.py
```
