# B0.4 局部形变响应保真度审计设计

## 1. 目标

B0.3 已发现 sand 的事实轨迹存在稳定的 `nu -> centered_shape_mse` GT 响应，而模型预测未复现该响应；同时凸包体积响应明显被压缩。B0.4 不训练新模型，而是回答一个更靠近本构动力学的问题：

> 模型是否丢失了由粒子局部形变可观测的材料响应，还是 B0.3 的全局形状/凸包结果主要来自统计口径？

B0.4 必须先验证位置轨迹能否可靠估计局部形变，再把该估计器用于冻结的 41-model test。估计器未通过校准时，不解释 test 上的 `F/J` 结果。

## 2. 不做什么

- 不训练或微调模型。
- 不修改 `eval.py`、模型结构、训练配置或 checkpoint。
- 不使用 test split 调估计器超参数。
- 不把位置估计的 `F` 称为模拟器真实 `F`。
- 不根据凸包体积替代局部体积应变。
- 不声称 factual correlation 等价于 counterfactual causal correctness。

## 3. 两阶段协议

### 3.1 Train calibration

从 train split 按材质、seed 0 固定选择每种材质最多 20 个有效模型。H5 必须同时提供 `x` 和模拟器真实 `F`。对 GT 位置轨迹计算局部估计 `F_hat`，与同一粒子、同一帧的真实 `F` 比较。

校准只用于验证估计器，不参与模型选择。主邻域超参数在运行 test 前冻结。

### 3.2 Frozen 41-model test

复用 B0.3 的 `mm3_contact_cond@90k`、`configs/eval_mm3_contact_cond.yaml`、seed 0、`start_idx=0` factual rollout。对 prediction 和 GT 使用同一个位置估计器，分别计算局部应变响应和 prediction-GT 误差，再按 elastic/plasticine/sand 分材质分析 `E/nu` 响应。

## 4. 局部形变估计器

对粒子 `i`，在初始/rest 帧 `X` 上建立固定 kNN。整个轨迹保持邻接不变，避免预测误差改变图结构后混入额外变量。

令：

```text
p_ij = X_j - X_i
q_ij(t) = x_j(t) - x_i(t)
w_ij = exp(-(norm(p_ij) / h_i)^2)
h_i = 第 k 个邻居的距离
```

加权最小二乘解为：

```text
Dm_i = sum_j w_ij p_ij p_ij^T
Ds_i(t) = sum_j w_ij q_ij(t) p_ij^T
F_hat_i(t) = Ds_i(t) [Dm_i + epsilon_i I]^-1
epsilon_i = 1e-6 * trace(Dm_i) / 3
```

主设置固定为 `k=16`。`k=8` 和 `k=32` 仅作为稳健性检查，不用于选择最优 test 结果。

邻域在正则化前的 `Dm` 条件数超过 `1e6`、尺度非正或结果非有限时，该粒子标记为 invalid。报告 valid fraction 和 condition-number 分布。

## 5. 指标

### 5.1 粒子/帧级量

- `legacy_strain = ||F_hat - I||_F`，与 B0.2 的旧定义对齐。
- `J = det(F_hat)`。
- `volumetric_strain = |J - 1|`。
- `green_lagrange = 0.5 * (F_hat^T F_hat - I)`。
- `green_strain_norm = ||green_lagrange||_F`。
- `deviatoric_strain_norm = ||green_lagrange - tr(green_lagrange) I / 3||_F`。
- `edge_stretch = mean_j |norm(q_ij)/norm(p_ij) - 1|`，作为不依赖矩阵求逆的辅助 sanity check。

### 5.2 Calibration 指标

- `F_relative_error = ||F_hat - F_true||_F / max(||F_true||_F, eps)`。
- `J_absolute_error = |det(F_hat) - det(F_true)|`。
- 估计/真实 trajectory mean 的 Spearman correlation。
- 主指标 `legacy_strain`、`volumetric_strain` 的分材质与 overall valid fraction。

校准通过条件：

1. 每种材质 valid fraction 至少 95%；
2. trajectory-level `legacy_strain` 和 `volumetric_strain` 与真实值的 overall Spearman 均至少 0.8；
3. `k=8/16/32` 下 response 的主要方向一致。

若任一条件失败，报告 `calibration_status=failed`，test 轨迹可以完成位置计算，但报告必须禁止把 `F_hat/J_hat` 当作本构结论。

### 5.3 Test prediction-GT 指标

- 各局部响应的 GT/pred mean、bias、MAE、RMSE、Spearman。
- `local_F_mse`、`local_J_mae`。
- frame 5、10、15、20、24 以及 short/mid/long 分段误差。
- elastic/plasticine/sand 的 `log10(E)` 与 `nu` ordinary/partial Spearman response。
- B0.3 同口径的 `aligned/attenuated/reversed/weak_or_unresolved` 分类。

## 6. 输出

固定输出目录默认 `results/material_local_deformation_b04`，包含：

- `material_local_deformation_b04_calibration.csv`
- `material_local_deformation_b04_models.csv`
- `material_local_deformation_b04_frames.csv`
- `material_local_deformation_b04_responses.csv`
- `material_local_deformation_b04_fidelity.csv`
- `material_local_deformation_b04_metadata.json`
- `material_local_deformation_b04.md`

默认拒绝覆盖已有目标；`--overwrite` 才允许完整替换。报告使用中文，明确 calibration gate、分材质结果和解释边界。

## 7. 裁决路径

- GT 局部 `nu` 响应稳定而 prediction 衰减/反向：材料条件没有正确控制局部形变，下一步注册材料条件与形变状态耦合机制。
- GT/pred 局部响应一致但 long-horizon 误差增大：优先解决 rollout 稳定性，而不是继续增强材料注入。
- GT 局部响应不稳定：B0.3 的 centered-shape 结果不能单独支撑新训练机制，应转向 paired counterfactual 数据。
- 局部体积响应正确而凸包体积错误：优先检查全局位姿、边界和外点，不把问题归因于本构响应。

## 8. 验证强度

按用户本次授权采用项目默认“中”：

- 新函数全部先写失败测试，再实现；
- 运行 B0.4 相关测试模块；
- 对新增 Python 文件执行 `py_compile`；
- 执行 `git diff --check`；
- 本机没有真实 checkpoint/CUDA/H5 时，不声称完成真实端到端运行。
