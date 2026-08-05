# B3a Stage Knockout 与分材质梯度冲突诊断设计

## 1. 目的

B3a Factorized Material-State Adapter 已证明能够增强部分材料参数响应，但
accuracy gate 与 material-response gate 均未通过。当前不训练新模型，而是对冻结的
`mm3_b3a_material_state_adapter@90k` checkpoint 做机制诊断，回答两个问题：

1. 四次 adapter 注入中，哪些 stage 对 elastic、plasticine、sand 的长程误差有益或有害？
2. 三种材质为了降低主坐标误差，对共享 adapter 参数提出的梯度方向是幅度不同，还是方向冲突？

诊断结果只用于决定 B3 adapter 家族的后续分支，不能替代独立训练 baseline 或证明反事实材料轨迹正确。

## 2. 冻结项

- checkpoint：`outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors`
- config：`configs/eval_mm3_b3a_material_state_adapter_90k.yaml`
- 样本集合：冻结的 41-model test，elastic / plasticine / sand = 13 / 14 / 14
- rollout knockout：每个 model 仅使用 `start_idx=0`，输入 5 帧，逐帧 rollout 到 25 帧
- seed：0；同一 model 的所有 knockout condition 重置到同一随机状态
- 模型结构、权重、数据、物理条件、点采样和推理协议全部冻结
- 梯度诊断：每个 model 使用 4 个固定窗口，共 164 个 teacher-forced 单步样本
- 梯度目标：只使用预测坐标与 GT 坐标之间的单步 MSE

## 3. 实现边界

采用独立诊断脚本，不向 `spacetime.py`、训练 config 或正式 eval config 增加 runtime mask。
stage knockout 通过上下文管理器临时修改已加载 checkpoint 的 `stage_scales`，退出时逐位恢复并验证完全一致。

新增：

- `src/diagnose_material_state_stage_knockout.py`
- `src/diagnose_material_state_gradient_conflict.py`
- `src/utils/material_state_stage_diagnostics.py`
- 对应测试文件

允许复用现有 `diagnose_hybrid_state_gate_knockout.py` 和
`utils/hybrid_state_gate_knockout.py` 中的冻结 manifest、配对 bootstrap、轨迹指标和 CSV 写入模式，
但 B3a 条件、字段和完整性检查必须独立注册，不能把 B1b/HST 的 verdict 直接套用到 B3a。

## 4. Stage Knockout

### 4.1 条件

| condition | stage mask |
|---|---|
| `normal` | `[1, 1, 1, 1]` |
| `all_off` | `[0, 0, 0, 0]` |
| `stage0_off` | `[0, 1, 1, 1]` |
| `stage1_off` | `[1, 0, 1, 1]` |
| `stage2_off` | `[1, 1, 0, 1]` |
| `stage3_off` | `[1, 1, 1, 0]` |

所有差值定义为 `knockout - normal`。误差指标中负值代表关闭该 stage 后改善，正值代表恶化。
`all_off` 只是同一 checkpoint 内的因果 knockout，不是公平的 no-adapter baseline。

### 4.2 指标

每个 model-condition 保存：

- `full_rollout_mse`
- `short_mse`、`mid_mse`、`long_mse`
- `gm_mse`、`fde`
- `f24_centroid_error`、`f24_shape_residual_mse`
- `penetration_rate`、`penetration_depth`

对 overall、elastic、plasticine、sand 做配对均值、均值差、相对变化、改善/退化模型数和
10,000 次 paired bootstrap 95% CI。

### 4.3 实际更新幅度

只在 `normal` condition 注册 adapter forward hook。每次调用计算：

```text
delta = adapter_output - adapter_input
delta_rms = sqrt(mean(delta^2))
hidden_rms = sqrt(mean(adapter_input^2))
relative_rms = delta_rms / max(hidden_rms, eps)
```

按 model、material、stage 聚合。该指标说明 stage 实际产生了多大 hidden 更新，但不等价于对最终误差的贡献；
贡献方向必须结合 knockout 结果判断。

## 5. 分材质梯度冲突

### 5.1 计算协议

- 主干与 adapter 权重均不更新，不创建 optimizer。
- 只令 adapter 参数 `requires_grad=True`，其余参数冻结。
- 对每种材质分别遍历其全部固定窗口，累积按样本平均的单步坐标 MSE 梯度。
- 使用与确定性单步推理一致的 model input 构造和固定随机种子。
- 每个材质得到一个针对完整 adapter 参数向量的平均梯度。

### 5.2 输出

报告以下参数组的梯度 L2 范数和两两 cosine：

- `all_adapter`
- `state_norm`
- `state_proj`
- `material_proj`
- `output_proj`
- `stage_scales`

另外输出四个 `stage_scales[i]` 的带符号平均梯度。若某个分组梯度范数为零，cosine 记为
`null/undefined`，不得伪造为 0。所有非零梯度必须有限。

## 6. 判读规则

1. **幅度冲突：**三材质主要梯度 cosine 为正，但梯度范数或 stage-scale 梯度幅度明显不同。
   后续候选为 material-conditioned stage/amplitude gate。
2. **方向冲突：**任意主要材质对在 `all_adapter` 或关键 projection 上出现稳定负 cosine，且对应
   knockout 呈现材质相反响应。后续才考虑 constitutive-specific low-rank experts。
3. **无明确冲突：**梯度方向一致，且 stage knockout 没有定位出稳定有害 stage。关闭 B3 adapter 家族，
   不再调 rank、interval 或学习率。
4. **证据不足：**cosine 接近零、bootstrap CI 跨零或结果主要由极少数 model 驱动。停止架构推断，
   保留为诊断结果，不据此启动完整训练。

无论材料响应是否增强，B3a 已失败的 accuracy gate 不会被本诊断推翻。

## 7. 输出文件

Stage knockout：

- `material_state_stage_knockout_b3a90_raw.csv`：41 × 6 = 246 行
- `material_state_stage_knockout_b3a90_paired.csv`：41 × 5 = 205 行
- `material_state_stage_activity_b3a90.csv`：41 × 4 = 164 行聚合记录
- `material_state_stage_knockout_b3a90.md`

梯度冲突：

- `material_state_gradient_conflict_b3a90.json`
- `material_state_gradient_conflict_b3a90.md`

所有文件必须写入 checkpoint、config、seed、样本范围和指标口径，且完整性检查失败时不输出部分结论。

## 8. 验证

本次测试强度为“中”：

- 新增统计、mask 恢复、cosine 和输出完整性的单元测试
- 测试必须先失败，再实现最小代码使其通过
- 对修改的 Python 文件执行 `python -m py_compile`
- 执行相关测试模块
- 执行 `git diff --check`

本机不执行完整 GPU rollout 或 backward；服务器运行前由严格 profile 验证 config 与 checkpoint 身份。
