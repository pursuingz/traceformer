# B2 连续材料响应扫描设计

## 目标

冻结 `mm3_contact_cond@90k`，分别扫描 `E` 与 `nu`，判断模型对连续材料参数的响应强度、单调性和物理方向。该实验没有 counterfactual GT，只诊断依赖和方向，不声称反事实预测准确。

## 冻结协议

- checkpoint：`outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors`
- config：对应 `mm3_contact_cond@90k` 的冻结 eval config
- 样本：B0/B0.1 的 41-model、`start_idx=0` full-horizon test，材质数 `13/14/14`
- 模型、checkpoint 和 dataloader 只加载一次
- 每个条件使用同一输入、同一模型真实材质类别、同一 seed
- 扫描一个参数时，另一个参数保持该模型真实值
- 不修改模型、训练代码或 checkpoint

## 扫描网格

每个模型运行七个条件，共 `41 × 7 = 287` 次 rollout：

- `normal`：真实 `log10(E)`、真实 `nu`
- `e_low/e_mid/e_high`：`log10(E) = 4.5/5.5/6.5`，`nu` 保持真实值
- `nu_low/nu_mid/nu_high`：`nu = 0.10/0.25/0.40`，`log10(E)` 保持真实值

## 指标

### 相对 normal 的条件响应

- prediction response MSE
- final-frame response MSE
- f24 centroid response
- f24 Procrustes shape response

### 每条反事实轨迹的运动状态

- rollout motion RMS（相对最后一帧条件输入）
- f24 centroid displacement
- f24 Procrustes shape deformation（相对最后一帧条件输入）
- f24 volume relative change
- floor penetration rate/depth

### 单调性

按模型分别计算：

- `log10(E)` 与 motion/shape deformation 的 Spearman 相关
- `nu` 与绝对 volume change 的 Spearman 相关
- 三点严格单调和宽松单调标记

单调方向仅作为 constitutive sanity check。预期高 `E` 通常减少形变，高 `nu` 通常减少体积变化；不同材质需分别报告，不以 overall 代替。

## 输出

- `material_response_sweep_b2_raw.csv`：287 条 condition-level 记录
- `material_response_sweep_b2_model_summary.csv`：41 条 model-level sensitivity/monotonicity
- `material_response_sweep_b2_summary.csv`：overall 和分材质统计
- `material_response_sweep_b2.md`：中文报告、限制和后续裁决

## 实现边界

采用独立脚本和 utility，复用 B0 的 config guard、manifest、loader、rollout 与材料一致性验证。不得修改旧 B0/B0.1 输出协议。

## 测试强度

用户选择“中”：相关单元测试、`py_compile` 和 `git diff --check`；不做多轮独立审查或真实 GPU rollout。
