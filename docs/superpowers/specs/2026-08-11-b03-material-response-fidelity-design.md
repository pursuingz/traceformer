# B0.3 预测-GT 材料响应保真度审计设计

## 1. 目标

B0.2 已证明 mm3 train 在每种材质内都包含未被静态 nuisance 严重混杂的 `E/nu -> GT dynamics` 关联，其中最强信号集中在体积与应变。B0/B0.1/B2 又表明冻结的 `mm3_contact_cond@90k` 对连续材料参数的利用不充分。

B0.3 不训练新模型，而是回答更具体的问题：

> 在冻结 test split 的 factual 条件下，`mm3_contact_cond@90k` 是否保留了 GT 轨迹中可以从粒子位置观测到的材料相关运动与形变响应；若没有，丢失发生在哪种材质、哪个参数、哪类响应上？

本审计服务一期目标 B/C/D，并用于决定下一种条件表示或监督机制。它不是新模型实验，也不能证明 counterfactual 因果正确。

## 2. 冻结实验边界

- Checkpoint：`outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors`。
- Config：`configs/eval_mm3_contact_cond.yaml`，必须通过现有 `contact_cond90` identity 校验。
- Split：冻结的 41-model test，elastic/plasticine/sand=`13/14/14`。
- 每个 model 只使用 `start_idx=0`，生成 20 帧预测，与 5 帧条件拼成 25 帧轨迹。
- 推理：`use_diffusion=false`、`num_inference_steps=1`、`seed=0`，复用 B0/B2 的 `rollout_condition`、strict checkpoint loader 和随机数重置方式。
- GT：允许读取这 41 个 test 对象的完整位置轨迹，因为 B0.3 是冻结 checkpoint 的正式预测评测；不得据此调训练超参数或选择 checkpoint。
- 不改变模型、数据、loss、checkpoint 或 rollout 逻辑。

## 3. 为什么不直接复用 B0.2 的全部响应

B0.2 的 `f_strain_norm_*` 和 `volumetric_strain_*` 分别由模拟器内部 `F` 及 `det(F)` 计算。当前模型只预测粒子位置，不预测 `F/C`，因此模型侧无法无歧义地计算同一响应。

B0.3 禁止通过以下方式制造虚假的同口径比较：

- 不把凸包体积直接命名为 `det(F)` 体积应变。
- 不从预测位置拟合未经验证的局部 deformation gradient。
- 不使用 GT `F/C` 代替预测内部状态。

本审计只使用预测和 GT 都能从 `(25,N,3)` 位置轨迹计算的 shared observable。`F/J` 响应保真度留给后续显式状态预测或经过验证的 deformation-gradient estimator。

## 4. Shared observable

所有响应均在同一归一化坐标中对 GT 与 prediction 使用同一函数计算；相对体积指标无量纲。

### 4.1 Primary responses

1. `position_velocity_rms_trajectory`：`sqrt(mean((x[t]-x[t-1])^2))`，使用全部 24 个帧差。
2. `position_acceleration_rms_trajectory`：`sqrt(mean((x[t+1]-2x[t]+x[t-1])^2))`。
3. `centroid_displacement_f24`：`||mean(x[24])-mean(x[0])||_2`。
4. `centered_shape_mse_f24`：末帧与初始帧去质心后的逐点 MSE。
5. `hull_volume_relative_change_f24`：`|V_hull(x[24])-V_hull(x[0])| / V_hull(x[0])`。
6. `hull_volume_relative_change_trajectory`：帧 1--24 的相对凸包体积变化绝对值均值。

### 4.2 Secondary responses

- `extent_change_x/y/z_f24`：末帧和初始帧的 axis-aligned extent 差。
- `future_contact_fraction`：帧 1--24 中 `y <= floor_height + contact_band` 的粒子比例。

凸包失败、轨迹非有限、帧数不是 25、粒子对应关系不一致或初始凸包体积非正时 fail-fast；不静默改用其他体积估计器。

## 5. 两层分析

### 5.1 Factual response fidelity

每个 test model、每个 response 保存：

- `gt_value`
- `pred_value`
- `signed_error = pred_value - gt_value`
- `absolute_error`

按 overall 和三种材质汇总：

- GT/pred 均值与标准差
- MAE、RMSE、mean signed bias
- `Spearman(pred, GT)`
- 对 mean signed bias 和 Spearman 进行 object-level paired bootstrap 95% CI

该层回答模型是否重现不同对象之间的真实响应排序和幅值。

### 5.2 Descriptive material-response alignment

对每个 `material × parameter × response`，分别计算 GT 和 prediction 的：

- 普通 Spearman `rho(parameter, response)`。
- 只控制另一个连续材料参数的 partial Spearman；不在 13/14 个对象上拟合 18 维静态 nuisance。
- object-level bootstrap 95% CI。
- `rho_gap = rho_pred - rho_gt`。
- 当 `|rho_gt| >= 0.05` 时报告 `magnitude_ratio = |rho_pred| / |rho_gt|`；GT 关联过小时留空，避免用接近零的分母制造无意义的大比例。

诊断标签只用于定位，不作为论文显著性结论：

- `aligned`：GT/pred 方向相同，且两者 `|partial rho| >= 0.20`。
- `attenuated`：方向相同，GT `|rho| >= 0.30`，但 prediction 幅度低于 GT 的 50%。
- `reversed`：GT/pred 方向相反，且两者 `|rho| >= 0.20`。
- `weak_or_unresolved`：其余情况。

由于每种材质只有 13/14 个 test 对象，该层只报告效应量和 bootstrap CI，不输出 permutation p-value，也不声称因果关系。B0.2 的 train-level conditional association仍是数据可辨识性的主证据。

若 GT 或 prediction 的某个 response 在组内为常数，相关系数和对应 CI 留空，并以 `constant_response` 状态保留该行；这属于合法的无响应证据，不视为审计损坏。

## 6. 软件结构

### 6.1 纯统计模块

新增 `src/utils/material_response_fidelity.py`：

- 定义冻结 response schema。
- 从单条 25 帧位置轨迹提取 shared observable。
- 构造 long-form model-response rows。
- 计算 factual fidelity、材料参数 alignment 和 bootstrap CI。
- 校验 41 个 model、13/14/14 材质计数、字段有限性和 schema。
- 原子写入固定输出及中文 Markdown 报告。

该模块不导入模型、CUDA、OmegaConf 或 dataset，单元测试可在 CPU 上完成。

### 6.2 推理入口

新增 `src/diagnose_material_response_fidelity.py`：

- 复用 `diagnose_material_condition.py` 的 dataset、metadata、strict checkpoint、GT reference 和 rollout 实现。
- 只执行 factual condition；每个 model 一次 rollout，共 41 次。
- 在收集完成后调用纯统计模块。
- CLI 参数：`--config`、`--checkpoint`、`--output-dir`、`--seed`、`--bootstrap-samples`、`--contact-band-raw`、`--overwrite`。
- 默认 `bootstrap_samples=10000`、`contact_band_raw=0.08`。

不修改 `eval.py`，防止正式指标路径发生无关变化。

## 7. 固定输出

输出目录恰好包含：

1. `material_response_fidelity_b03_models.csv`：41 个 model 的 provenance 和标准 rollout accuracy。
2. `material_response_fidelity_b03_responses.csv`：`41 × response` 的 GT/pred long-form 响应。
3. `material_response_fidelity_b03_fidelity.csv`：overall 与分材质 fidelity 汇总。
4. `material_response_fidelity_b03_alignment.csv`：分材质、分参数、分响应的 GT/pred 关联与标签。
5. `material_response_fidelity_b03_metadata.json`：checkpoint/config/seed/split/count/schema。
6. `material_response_fidelity_b03.md`：中文裁决报告。

任一目标存在时默认在昂贵推理前失败；只有显式 `--overwrite` 才允许替换完整输出集。禁止保存 `.npy` 预测轨迹或 checkpoint 副本。

## 8. 结果解释与下一步分流

- **GT shared observable 有材料信号、prediction 明显 attenuated/reversed：** 下一机制针对被丢失的状态量或监督目标，不再盲目扩大共享 condition encoder。
- **Factual fidelity 良好，但 B2 counterfactual sweep 仍弱：** 模型可能依靠对象状态或 `mat_type` 捷径；优先制作小规模 paired counterfactual test set。
- **GT shared observable 本身无稳定材料信号，而 B0.2 仅 `F/J` 强：** 仅从位置 MSE 学连续材料控制的监督信噪比不足；考虑显式形变状态、位置可辨识的局部应变 estimator，或针对材料响应的辅助目标。
- **不同材质出现相反缺失模式：** 下一机制必须 material-aware/factorized，不能再次用 B3/B4 式共享更新方向统一处理。

B0.3 不直接产生 B5 架构结论。完成后才根据缺失模式预注册下一项训练实验。

## 9. 高强度验证

- TDD：每个生产函数先写失败测试并确认失败原因。
- 新增 `src/tests/test_material_response_fidelity.py`，覆盖公式、partial Spearman、bootstrap、标签、schema、输出安全和 fake-runtime 端到端收集。
- 运行新增测试模块全量。
- 运行既有 `tests.test_material_identifiability`、`tests.test_material_response_sweep` 和与 B0 rollout 相关的测试模块，确认复用路径未回归。
- 对新增 Python 文件执行 `py_compile`，并执行 `git diff --check`。
- 完成一次独立代码审查；修复 Critical/Important 问题后重跑相关测试。
- 本机无 CUDA checkpoint，不把 CPU fake-runtime 测试宣称为服务器端真实推理验证；最终还需服务器运行 41-model B0.3。
