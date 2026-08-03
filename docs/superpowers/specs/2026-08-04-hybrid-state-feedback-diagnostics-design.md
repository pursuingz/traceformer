# B1b HST Feedback 分解诊断设计

## 1. 目标

B1b（`v11a + contact_cond`）在 90k、seed0 上呈现明显的材质异质性：

- sand 的对齐后形状误差改善；
- plasticine 的质心、FDE 和形状误差显著恶化；
- 总体 full-rollout MSE 基本持平，但穿透与计算成本变差。

本诊断不训练新模型，也不修改模型计算路径。目标是利用现有 B1b checkpoint，判断 HST feedback 的粒子均值分量与去均值分量是否分别对应全局运动漂移和局部形变变化，并检查这种关系是否随材质变化。

本阶段只产生机制证据，不把相关性直接解释为因果性。

## 2. 研究问题

诊断需要回答以下问题：

1. plasticine 是否接收到幅度异常的粒子均值 feedback？
2. sand 的形状改善是否伴随更强或更有效的去均值 feedback？
3. f24 质心误差是否与粒子均值 feedback 能量相关？
4. Procrustes 残余形状误差是否与去均值 feedback 能量相关？
5. feedback 的差异主要来自材质、rollout horizon，还是 exchange stage？

## 3. 约束

- 不修改 `HybridStateExchange.forward`、`MDM_ST.forward` 或训练路径。
- 不新增模型参数，不改变 checkpoint 加载结果。
- 不使用 knockout 干预；第一阶段只做被动记录。
- 只分析 41 个 `start_idx=0` 全程窗口，使 feedback 与现有 full-rollout、FDE 和 Procrustes 结果严格对齐。
- 使用 B1b 的 `checkpoint-90000`、相同 mm3 test split 和 seed0。
- 不启用 `torch.compile`，避免 hook 被图捕获或重排；诊断结果必须来自 eager forward。

## 4. Feedback 定义与分解

第 `s` 个 exchange stage 的实际残差为：

\[
\Delta h_{bni}^{(s)} = g_s\,f_{bni}^{(s)},
\]

其中 `f` 是 `feedback_attention` 的有限值输出，`g_s` 是已训练的 `feedback_gates[s]`。

对粒子轴 `N` 分解：

\[
\Delta h_{\mathrm{global}}^{(s)}
=\frac{1}{N}\sum_{n=1}^{N}\Delta h_n^{(s)},
\]

\[
\Delta h_{\mathrm{deform},n}^{(s)}
=\Delta h_n^{(s)}-\Delta h_{\mathrm{global}}^{(s)}.
\]

记录以下标量：

- `feedback_rms`：全部粒子和通道的 RMS；
- `global_rms`：粒子均值向量的 RMS；
- `deform_rms`：去均值 feedback 的 RMS；
- `global_energy_fraction`：均值分量能量占总能量比例；
- `gate`：对应 stage 的门值。

由于去均值项在粒子轴均值为零，应满足：

\[
\operatorname{mean}(\Delta h^2)
=\operatorname{mean}(\Delta h_{\mathrm{global}}^2)
+\operatorname{mean}(\Delta h_{\mathrm{deform}}^2)
\]

（允许浮点误差）。该恒等式作为实现测试之一。

## 5. 捕获方式

采用只读 forward hook：

1. 在共享 `HybridStateExchange` 上注册 `forward_pre_hook(with_kwargs=True)`，读取当前 `stage_index`；
2. 在 `feedback_attention` 上注册 `forward_hook`，取得未乘 gate 的 attention 输出；
3. 对输出执行与模型一致的 `nan_to_num`，再乘当前 stage 的实际 gate；
4. 立即 detach 到 CPU 并聚合，不保留计算图；
5. 每个模型 rollout 完成后移除 hook。

记录器必须验证每次模型 forward 恰好捕获 4 个 stage，stage 顺序为 `0,1,2,3`。缺失、重复或乱序均视为诊断失败，不静默继续。

## 6. 轨迹指标

对每个全程窗口同时记录：

- `full_rollout_mse`；
- `fde`；
- `f24_centroid_error`；
- `f24_shape_residual_mse`。

Procrustes 计算复用 `utils/eval_metrics.py` 的既有口径，不复制一套不同实现。输入条件帧不计入上述预测误差。

## 7. 输出

### 7.1 原始 CSV

每行对应一个：

`model × material × rollout_step × exchange_stage`

字段至少包括：

- model、mat_type、log10(E)、nu；
- rollout_step、absolute_frame、stage；
- gate、feedback_rms、global_rms、deform_rms、global_energy_fraction；
- 该窗口的 full-rollout MSE、FDE、f24 centroid 和 shape residual。

### 7.2 Markdown 汇总

按 overall / elastic / plasticine / sand 输出：

- 各 stage 的 feedback 统计；
- short / mid / long horizon 的 feedback 统计；
- feedback 指标与轨迹指标的 Pearson 和 Spearman 相关系数；
- 样本数和 checkpoint/config 元数据。

相关系数只用于定位候选机制。`n=13/14` 的分材质相关性不以显著性结论表述。

## 8. 文件结构

- `src/utils/hybrid_state_diagnostics.py`
  - feedback 分解；
  - 记录器；
  - 分组统计与相关性；
  - CSV/Markdown 所需的纯函数。
- `src/diagnose_hybrid_state_feedback.py`
  - CLI、配置与 checkpoint 校验；
  - dataset/pipeline rollout；
  - 轨迹指标和输出文件。
- `src/tests/test_hybrid_state_diagnostics.py`
  - 分解、能量恒等式、gate、stage 顺序和汇总测试。

## 9. CLI

从服务器 `src/` 目录执行：

```bash
python diagnose_hybrid_state_feedback.py \
  --config configs/eval_mm3_v11a_contact_cond_8L_45k.yaml \
  --checkpoint outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors \
  --output-dir results/hybrid_state_b1b_90k
```

脚本必须在报告中写出实际 checkpoint，避免配置文件名中的 `45k` 与实际评测的 `90k` 混淆。

## 10. 成功标准与后续分支

诊断成功的工程标准：

- 41 个全程窗口全部完成；
- 每次 forward 捕获 4 个 stage；
- 所有反馈统计为有限值；
- 能量分解误差在数值容差内；
- CSV 与 Markdown 均包含 checkpoint 和样本口径。

研究分支：

- 若 plasticine 的 global feedback 与 centroid/FDE 同时异常，下一臂优先做 motion/deformation factorization；
- 若主要是不同材质的总 feedback 幅度不匹配，下一臂优先做 material-conditioned gate；
- 若没有稳定关系，不继续修改 v11a，返回 `contact_cond` 的其他总体仿真改进方向。

任何新训练都必须另写预注册设计，不能从本相关性诊断直接宣称因果机制。
