# mm1_contact_cond 实验设计

## 1. 实验目的

在 `diff_E_2048_data` 的 elastic 单材料数据上，验证初版
`contact_cond` 是否能够独立改善物理轨迹预测。

该实验用于区分两种可能原因：

1. `contact_cond` 本身改善了 elastic 预测；
2. 之前 mm3 模型在 elastic 上的提升主要来自多材料联合训练。

## 2. 对照原则

训练基线为：

`src/configs/config_diffE2048_singleframe_geom_deform_d0001.yaml`

除 contact conditioning 和独立输出目录外，数据、采样、网络主干、
损失权重、优化器、batch size、学习率、随机种子与训练预算全部冻结。

不得引入 mm3 专属变量：

- 不启用 `class_token`；
- 不启用 `gravity_emb`；
- 不启用 `geom_elastic_only`；
- 不修改数据划分或 random-window sampling；
- 不从已有 checkpoint 续训。

## 3. 模型改动

使用初版 separate contact conditioning：

```yaml
model_config:
  contact_particle_cond: true
  contact_injection_mode: separate
  contact_velocity_mode: vertical
  contact_feature_sigma: 0.04
```

每个条件帧、每个粒子的 contact feature 为：

```text
[signed_gap, vertical_displacement, proximity]
```

三维特征经过共享的 `Linear(3 -> 256)`，再加到对应 condition-frame
hidden state。该实验不使用 shared encoder，也不使用 `v_xyz`。

## 4. 训练配置

新增：

`src/configs/config_mm1_contact_cond.yaml`

相对基线只允许以下差异：

```yaml
output_dir: ./outputs/mm1_contact_cond_8L
stop_after_steps: 45000

model_config:
  contact_particle_cond: true
  contact_injection_mode: separate
  contact_velocity_mode: vertical
  contact_feature_sigma: 0.04
```

训练预算：

- `max_train_steps: 90000`，保持完整训练配置；
- `stop_after_steps: 45000`，用于首轮筛选；
- 从随机初始化开始训练；
- 数据集保持
  `diff_E_2048_data/2048_data/2048_train`。

## 5. 评测配置

新增：

`src/configs/eval_mm1_contact_cond_45k.yaml`

要求：

- checkpoint：
  `outputs/mm1_contact_cond_8L/checkpoint-45000/model.safetensors`；
- 测试集：
  `diff_E_2048_data/2048_data/2048_test`；
- 模型与数据字段严格镜像训练配置；
- 与 diffE2048 baseline 的 `checkpoint-45000` 同步比较。

## 6. 指标与结论

主要指标：

- full-rollout MSE；
- FDE；
- long-horizon MSE；
- `loss_F`；
- 体积相对误差与体积漂移。

辅助诊断：

- 三个 contact 投影列的权重范数；
- elastic 数据上的 feature 分布与估计 hidden contribution。

结论判据：

- mm1 contact 优于 mm1 baseline：contact 表征本身有效；
- mm1 无提升、mm3 elastic 提升：多材料联合训练是主要原因；
- mm1 与 mm3 elastic 都提升：contact 和多材料训练都可能有贡献。

elastic 轨迹通常不接触地板，因此若该实验提升，优先检查
`vertical_displacement` 和 condition-frame bias，而不是将结果归因于
`proximity`。

## 7. 验证要求

- 配置测试必须断言实验臂与 baseline 只有允许字段不同；
- structured config 合并必须成功；
- CPU 构建 `MDM_ST` 必须成功；
- 参数增量应为初版 `Linear(3 -> 256)` 的 `1024` 个参数；
- Python 修改需通过 `py_compile`，配置需通过对应单元测试。
