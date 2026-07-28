# Contact Feature 与粒子特征共享编码器设计

## 1. 目标

在保持初版 `contact_cond` 三个输入不变的前提下，新增一个独立实验臂，将 contact feature 与粒子位置特征拼接后送入同一个线性编码器：

```text
Fourier(xyz) 96维 + raw xyz 3维 + contact feature 3维
                         |
                         v
                  Linear(102 -> 256)
                         |
                         v
                  particle hidden
```

实验回答的问题是：

> 初版 contact feature 是否需要独立 `Linear(3 -> 256)` 后再注入 hidden，还是与粒子位置共享输入编码器即可？

这是注入方式消融，不宣称增加模型表达能力。两个线性分支相加与拼接后使用一个线性层在数学上近似等价，可能产生的差异主要来自 bias、参数化方式和优化过程。

## 2. 实验约束

- 保持初版 contact feature：
  - `signed_gap`
  - `vertical_displacement`
  - `proximity`
- 不使用 `v_xyz`。
- 保持8层串行 `SpatialTemporalTransformerBlock` 主干不变。
- 数据、采样、损失、batch size、学习率、训练步数和随机种子与 `config_mm3_contact_cond.yaml` 一致。
- 新实验只改变 contact feature 的编码与注入方式及独立 `output_dir`。
- 原 `separate` 路径、已有配置和 checkpoint 必须继续可用。
- v_xyz 代码和配置保留用于实验复现，不删除历史。

## 3. 配置接口

新增模型配置：

```yaml
contact_injection_mode: separate  # separate | shared
```

- `separate`：默认值，保持当前实现：

  ```text
  PointEmbed(xyz) + ContactEncoder(contact)
  ```

- `shared`：新增路径：

  ```text
  SharedPointEmbed(Fourier(xyz), raw xyz, contact)
  ```

`shared` 第一版仅支持：

- `point_embed: true`
- `force_as_latent: false`
- `contact_velocity_mode: vertical`

不满足时应明确报错，避免静默进入错误路径。

## 4. 数据流

设5个条件帧和1个待预测槽构成模型输入：

```text
x = [condition frame 0 ... condition frame 4, output slot]
```

contact feature 仍只由5个真实条件帧计算：

```text
c_cond.shape = (B, 5, N, 3)
```

待预测槽没有真实 contact state，因此补零：

```text
c_all = [c_cond, zeros_for_output_slots]
```

随后对每个粒子执行：

```text
phi_xyz = concat(Fourier(xyz), xyz)     # 99维
joint   = concat(phi_xyz, c_all)        # 102维
hidden  = Linear(102 -> 256)(joint)
```

mask伪帧在粒子编码完成后才加入，沿用现有逻辑，不参与共享编码。

## 5. 初始化

共享编码器中：

- 前99列按原 `PointEmbed` 的方式初始化；
- 新增3列 contact 权重初始化为0；
- bias沿用粒子编码器的单一 bias。

因此训练开始时 contact feature 不改变前向输出，模型随后自行学习这3列权重。该路径不创建独立 `contact_encoder`，从而也不引入只作用于条件帧的 `contact_encoder.bias`。

参数变化：

```text
原 separate 额外参数：3 * 256 + 256 = 1024
新 shared 额外参数：  3 * 256       =  768
```

新旧 checkpoint 形状不同。`shared` 使用独立输出目录并从头训练；`separate` 配置仍可正常加载旧 checkpoint。

## 6. 实验配置

新增：

```text
src/configs/config_mm3_contact_concat.yaml
src/configs/eval_mm3_contact_concat_45k.yaml
```

训练配置以 `config_mm3_contact_cond.yaml` 为基线，仅允许改变：

- `output_dir`
- `stop_after_steps: 45000`
- `model_config.contact_injection_mode: shared`

45000步作为筛选点，与已有 `contact_cond@45k` 同步比较。若没有明确优势，不继续到90000步。

## 7. 验证

至少覆盖：

1. 默认 `separate` 模型结构、参数形状和前向路径不变。
2. `shared` 编码器输入维度为102，且不存在独立 `contact_encoder`。
3. 新增3列权重为0。
4. contact feature 只填入条件帧，输出槽补零。
5. shared路径前向输出形状正确。
6. 非法模式和不兼容配置明确报错。
7. 训练与评测配置除允许项外严格镜像基线。

验证命令：

```bash
python -m py_compile src/model/spacetime.py src/utils/contact.py
python -m unittest src.tests.test_contact src.tests.test_contact_ablation_configs
python src/count_params.py
```

## 8. 成功判据

主要比较 `contact_cond@45k` 与 `contact_concat@45k`：

- full-rollout MSE
- Chamfer
- FDE
- 分材料 rollout MSE
- penetration rate/depth
- E/nu counterfactual sensitivity（诊断，不作为唯一裁决）

只有当共享编码器在主要 rollout 指标不退化，并改善至少一个已确认问题时，才值得继续训练到90000步。
