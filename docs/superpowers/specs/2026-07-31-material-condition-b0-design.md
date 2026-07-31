# B0 材料条件利用诊断设计

## 1. 目的

在训练新的材料条件注入结构前，先回答一个可证伪的问题：现有
`mm3_contact_cond` 模型是否真正利用了连续材料参数 `E`、`nu`，以及模型是否主要依赖
离散 `mat_type` token。

B0 只诊断现有 checkpoint，不更新模型权重，不评价新方法优劣。

## 2. 固定实验口径

- checkpoint：`outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors`
- 配置：`src/configs/eval_mm3_contact_cond.yaml`
- 数据：`mm3_data/mm3_test`，共 41 个 held-out model
- 材料分组：elastic 13、plasticine 14、sand 14
- 每个 model 只评估 `start_idx=0` 的完整 20 帧预测 horizon
- 推理 seed：0
- 正常和反事实 rollout 使用相同随机数序列
- 模型权重、输入点轨迹、force、gravity、floor、GT 及其余条件保持不变

## 3. 三条推理路径

### 3.1 Normal

使用每个样本真实的 `log10(E)`、`nu` 和 `mat_type`，作为配对基准。

### 3.2 Shuffle `(E, nu)`

在同一种材料内部，对完整 `(log10(E), nu)` 参数对做固定、可复现的一一置换：

- 不跨材料置换；
- 参数对保持在真实测试分布内；
- 同一个 model 的所有窗口使用同一个置换结果；
- 置换不得把参数对分配回原 model。

该路径只改变连续材料参数，用于判断模型是否依赖连续材料条件。

### 3.3 Shuffle `mat_type`

保持真实 `(log10(E), nu)` 不变，对 `mat_type` 做固定的跨材料循环置换：

- elastic -> plasticine
- plasticine -> sand
- sand -> elastic

该路径会产生不一致的反事实条件，因此只用于测量模型对离散类别 token 的依赖，
不能作为物理准确率实验。

## 4. 输出指标

每个 model 保存 Normal 和两条反事实路径的配对结果。汇总同时给出 overall、elastic、
plasticine 和 sand 四组结果。

主要指标：

- `GM-MSE`
- `FDE`
- long-horizon `seg-MSE`
- `MSE full-rollout`

响应指标：

- 反事实预测与 Normal 预测之间的预测帧 MSE；
- 反事实末帧与 Normal 末帧之间的 MSE。

响应指标只说明模型输出是否对条件敏感；相对 GT 的误差变化才说明这种依赖是否有助于
正确预测。

## 5. 配对统计与判据

对每个 model 和每项误差指标计算：

\[
\Delta_i = \mathrm{Error}^{\mathrm{counterfactual}}_i
          - \mathrm{Error}^{\mathrm{normal}}_i.
\]

在逐 model 配对差值上进行固定 seed 的 bootstrap，报告平均差值、相对变化和 95% CI。

- **明确利用条件**：主要长程指标恶化至少 5%，且配对 bootstrap 95% CI 不包含 0。
- **基本忽略条件**：主要指标变化低于 2%，CI 包含 0，同时预测响应很小。
- **不明确**：介于两者之间，或不同材料方向相反。后续再做 `E-only`、`nu-only` 和多组置换。

不能只根据三种材料混合后的总体均值下结论；至少需要报告各材料内部趋势。

## 6. 实现边界

新增独立脚本：

```text
src/diagnose_material_condition.py
```

正式 `src/eval.py` 的行为和配置语义保持不变。诊断脚本复用现有模型、pipeline、dataset
和 `utils.eval_metrics`，一次加载 checkpoint，针对每个完整窗口执行三条 rollout。

输出：

- 终端汇总；
- Markdown 汇总报告；
- 逐 model CSV。CSV 属于实验产物，不纳入 Git。

## 7. 错误处理

脚本在以下情况立即失败并给出明确错误：

- checkpoint 不存在；
- 测试集中某种材料少于两个 model，无法构造无固定点置换；
- 同一 model 出现不一致的 `E`、`nu` 或 `mat_type`；
- 不是 `start_idx=0` 的窗口被纳入完整 horizon；
- 数据缺少 `E`、`nu`、`mat_type` 或 model 名称。

## 8. 验证要求

- 纯函数单元测试覆盖：同材质无固定点置换、类别循环、分组聚合、bootstrap CI、判据边界。
- 测试必须先失败，再实现通过。
- `python -m py_compile src/diagnose_material_condition.py`
- CLI `--help` 冒烟测试不加载 CUDA/checkpoint。
- 不在本地伪造 GPU 结果；正式诊断由服务器 5090 执行。
