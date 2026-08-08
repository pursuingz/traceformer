# B0.2 Material Identifiability Audit 设计

## 1. 背景与目标

当前多材质主线已经连续暴露出同一种风险：B3 的共享 material-state adapter 和 B4 的共享 Continuous Material AdaLN 都能显著改变预测，但收益主要集中在 sand，同时伤害 plasticine，且没有建立稳定的连续 `E/nu` 控制证据。继续增强条件注入前，必须先确认现有 MM3 数据是否允许模型从观测轨迹中辨识 `E/nu` 的独立作用。

B0.2 是无训练、无 checkpoint 的数据诊断。它只回答以下问题：

1. 每种材质内部是否有充分且连续的 `E/nu` 覆盖？
2. 参数是否与几何、外力、地板和接触状态产生意外相关？
3. 控制可观测 nuisance 后，GT 动力学是否仍包含可检出的 `E/nu` 信号？
4. train 与 test 的参数和元数据支持范围是否一致？

该诊断不证明反事实物理正确，也不使用 test 轨迹响应选择模型。其结果决定下一步应继续研究条件表示，还是先修订数据生成协议。

## 2. 已知生成协议事实

当前 `generate_mpm_data.py` 中：

- `log10(E) ~ Uniform(4, 7)`；
- `nu ~ Uniform(0.05, 0.45)`；
- 每个 H5 只保存一组 `E/nu` 和一条轨迹；
- elastic 使用 `force_num = 1`；
- plasticine 和 sand 使用 `np.random.randint(0, 1)`，其结果恒为 `0`；
- 因此 elastic 始终有 drag force、无重力，plasticine/sand 始终无 drag force、有重力；
- elastic 使用固定低地板，plasticine/sand 使用随机地板；
- 三种材质来自不同 UID 区间，不构成同几何、同初态、同外力下的配对反事实样本。

这些事实意味着 `mat_type` 同时携带材料、外力和接触场景信息。B0.2 的连续参数分析必须在每种材质内部进行；跨材质统计只用于报告生成协议混杂，不能作为 `E/nu` 作用证据。

## 3. 数据隔离

### 3.1 `mm3_train`

允许读取完整 H5 内容，并用于：

- 参数覆盖分析；
- nuisance 混杂分析；
- GT 动力学响应分析；
- 统计模型拟合、交叉验证、bootstrap 和参数置换。

### 3.2 `mm3_test`

只读取参数和静态元数据，用于检查：

- `E/nu` 是否落在 train 支持范围内；
- 几何和场景元数据是否发生明显分布偏移；
- 是否存在参数外插样本。

禁止使用 `mm3_test` 的后续轨迹、`F/C` 或任何 GT 动力学响应来决定下一结构。当前 41-model test 已多次参与方法诊断，本设计不进一步把它变成调参集。

## 4. 输入与样本单位

输入目录：

```text
mm3_data/mm3_train
mm3_data/mm3_test
```

每个 H5 文件是一个独立 object-level 样本。统计样本量等于 H5 文件数，不能把 25 帧或 2048 个粒子当成独立样本扩大显著性。

必需字段：

```text
x, v, F, C, vol, E, nu, mat_type, gravity, floor_height,
drag_force, drag_mask
```

字段缺失、形状不一致、NaN/Inf、粒子数或帧数异常必须明确报错或登记为 invalid record；不得静默填零后继续。

## 5. 每个 object 的特征提取

### 5.1 材料参数

- `log10_e = log10(E)`；
- `nu`；
- `mat_type`。

### 5.2 几何 nuisance

从原始坐标 `x[0]` 提取：

- 质心 `(cx, cy, cz)`；
- 三轴包围盒尺寸；
- 协方差矩阵三个排序特征值；
- radius of gyration；
- 初始凸包体积；凸包失败时记录缺失，不用其他数值静默替代；
- 粒子体积总和；
- 初始最低点与地板的 signed gap。

### 5.3 场景 nuisance

- `gravity`；
- `floor_height`；
- drag force 数量和总模长；
- drag mask 粒子比例；
- 初始接触粒子比例；
- 是否在预测 horizon 内发生接触。

接触阈值与模型 contact band 对齐。模型归一化空间中的 `sigma=0.04` 对应原始坐标 `0.08`，因此原始 H5 上使用：

```text
gap_raw <= 0.08
```

### 5.4 GT 动力学响应

在绝对帧 `f5/f10/f15/f20/f24` 以及整段轨迹上计算：

#### 位置与形状

- 相对 `x[0]` 的逐点位移 MSE；
- 质心位移；
- 去质心后的形变 MSE；
- 三轴 extent 变化。

#### 速度与加速度

- H5 原生 `v` 的 RMS 和末帧 RMS；
- 位置一阶差分的速度 RMS；
- 位置二阶差分的加速度 RMS。

原生 `v` 和有限差分同时保留，用于发现保存速度与帧间运动不一致的问题，不将二者混为同一指标。

#### 本构状态

- `mean(||F-I||_F)`：总形变强度；
- `mean(|det(F)-1|)`：体积应变；
- `mean(||C||_F)`：局部仿射速度梯度强度；
- 上述指标的末帧值和轨迹均值。

#### 接触

- 首次接触帧；
- 每个选定帧的接触粒子比例；
- horizon 内平均接触比例。

## 6. 四层审计

### 6.1 参数覆盖

按 split 和 material 报告：

- 样本数、唯一值数；
- min/max/mean/std 和分位数；
- `log10(E)` 与 `nu` 的 Pearson/Spearman 相关；
- 5x5 二维等宽网格覆盖率；
- test 超出 train min/max 的比例；
- train/test KS statistic 和 Wasserstein distance。

### 6.2 nuisance 混杂

在每种材质内部：

1. 计算 `log10(E)`、`nu` 与每个 nuisance 的 Spearman 相关；
2. 使用 object-level 5-fold CV，从全部 nuisance 预测 `log10(E)` 或 `nu`；
3. 与材质内参数置换后的 null distribution 比较。

若 nuisance 对参数的 held-out `R² > 0.05` 且置换 `p < 0.05`，该参数标记为 `confounded`。常量 nuisance 只登记为生成协议事实，不进入回归。

### 6.3 GT 动力学可辨识性

对每种材质和每个响应，比较四个 nested predictor set：

```text
M0    = nuisance
ME    = nuisance + basis(log10(E))
Mnu   = nuisance + basis(nu)
Mboth = nuisance + basis(log10(E)) + basis(nu) + E*nu
```

`basis(z)` 使用仅由当前 train fold 分位数构造的低维分段线性基：标准化 `z` 本身，以及位于 train fold `q25/q50/q75` 的三个 hinge 项 `relu(z-q)`。该设计允许有限非线性，同时不依赖额外机器学习库，也不会用 held-out fold 决定 knots。

约束：

- object-level 固定 5-fold，seed 0；
- 四个模型使用完全相同 folds；
- 连续输入只用 train fold 的统计量标准化；
- ridge 的 alpha 从固定网格 `[1e-4, 1e-3, ..., 1e3]` 中只在 train fold 内选择，不能查看 held-out fold；
- 主要量为相对 M0 的 held-out `delta_R2`；
- 同时报告 cross-fitted residual partial Spearman；
- 每种材质内部执行 500 次参数置换；
- 执行 1000 次 object bootstrap，报告 95% CI；
- 多响应检验使用 Benjamini-Hochberg FDR，报告 `q_value`。

预先冻结的**主要响应**为：

1. `centered_shape_mse_f24`；
2. `centroid_displacement_f24`；
3. `velocity_rms_trajectory`；
4. `f_strain_norm_f24 = mean(||F-I||_F)`；
5. `volumetric_strain_f24 = mean(|det(F)-1|)`。

其他帧、加速度、`C` 和接触指标属于次要解释性响应，不能单独把参数从 `weak/not_detected` 提升为 `identifiable`。

### 6.4 train/test 支持偏移

只使用 test 的参数和静态 nuisance：

- 参数 min/max 外插率；
- 5x5 网格中落入 train 空 bin 的比例；
- 各静态特征的标准化均值差；
- 参数与静态 nuisance 的联合 Mahalanobis distance。

该层不生成 test 动力学响应行。

## 7. 可辨识性分类

每种材质、每个参数分别分类：

### `identifiable`

- 至少一个主要 GT 响应满足 `delta_R2 >= 0.05`；
- 置换 `p < 0.05` 且 FDR `q < 0.05`；
- bootstrap 95% CI 下界大于 0；
- 不属于 `confounded`。

### `weak`

- `0.01 <= delta_R2 < 0.05`，或显著性/置信区间只满足部分条件；
- 信号只存在于少量次要响应时也归入该类，不能据此写强控制主张。

### `not_detected`

- 所有主要响应 `delta_R2 < 0.01`，且置换/FDR 不显著。

### `confounded`

- nuisance 能显著预测该参数。

Train/test 支持范围另设独立的 `support_status = in_support/out_of_support`，不把分布外问题误称为参数混杂。即使 train 内参数为 `identifiable`，只要 test 为 `out_of_support`，也禁止据此解释模型 test 响应。

方向合理性作为额外证据，不替代可辨识性：例如更高 `E` 在可比外力下通常应降低形变强度，更高 `nu` 通常应降低体积应变。对 plasticine/sand 的方向解释必须结合其本构模型，不能直接套用线弹性结论。

## 8. 输出文件

默认输出目录：

```text
results/material_identifiability_b02/
```

文件：

```text
material_identifiability_records.csv
material_identifiability_coverage.csv
material_identifiability_confounding.csv
material_identifiability_response.csv
material_identifiability_summary.csv
material_identifiability_metadata.json
material_identifiability_b02.md
```

### `records.csv`

每个 object 一行，保存 split、material、参数、nuisance、GT response 和 validity flags。test 行不得包含 GT response。

### `coverage.csv`

保存 train/test 的参数分布、二维覆盖和 support shift 指标。

### `confounding.csv`

保存参数-nuisance 单变量相关、nuisance-to-parameter CV 结果、置换统计和 verdict。

### `response.csv`

保存 material x parameter x response 的 M0/ME/Mnu/Mboth CV 指标、`delta_R2`、partial Spearman、bootstrap CI、`p/q`。

### `summary.csv`

每种材质、每个参数一行，给出 `identifiable/weak/not_detected/confounded` 及触发依据。

### `metadata.json`

保存命令参数、seed、fold、置换/bootstraps 数量、数据目录、文件数量、无效文件、字段版本和生成时间。

### `material_identifiability_b02.md`

面向研究决策的中文报告，必须包含：

- 生成协议事实；
- 三材质独立结果；
- E 与 nu 独立裁决；
- train/test support shift；
- 不可由本诊断推出的结论；
- 下一步决策建议。

## 9. 决策规则

| 审计结果 | 研究结论 | 后续动作 |
|---|---|---|
| GT 中 `E/nu` 清楚、混杂低，而冻结模型响应弱 | 条件表示/注入存在问题 | 允许预注册新的条件机制 |
| `E` 清楚、`nu` 弱 | 当前数据主要支持 `E` 控制 | 暂停强 `nu` 主张，补 nu-sensitive 数据 |
| 参数被 geometry/scenario 显著预测 | 数据存在混杂 | 先修数据生成协议，不训练新结构 |
| 覆盖充分但 GT 中无可检出信号 | 当前场景对参数不敏感 | 改变场景、载荷或参数采样 |
| train 有信号但 test `out_of_support` | 评测含外插问题 | 修 split/coverage 后再比较结构 |

B0.2 结果出来前，不注册新的完整训练臂。

## 10. 实现边界

建议组件：

```text
src/diagnose_material_identifiability.py
src/utils/material_identifiability.py
src/tests/test_material_identifiability.py
```

- CLI 负责参数解析、目录扫描、进度显示和文件输出；
- utils 模块负责纯特征提取、统计、分类和表格构造；
- 不导入模型、pipeline、checkpoint 或 CUDA；
- H5 按文件流式读取，不把全部粒子轨迹常驻内存；
- 所有随机过程由 seed 控制；
- 生成既有文件时默认拒绝覆盖，必须显式指定 overwrite；
- 任一物理字段异常必须进入 invalid-record 清单并影响最终状态，不能静默跳过。

## 11. 验证策略

实现前由 Will 选择测试强度。最低验证范围：

1. 人工构造 H5 的特征提取单元测试；
2. test split 不读取轨迹响应的回归测试；
3. object-level 样本计数测试，防止按帧/粒子伪增样本；
4. 固定 seed 的 fold/置换/bootstrap 可复现测试；
5. synthetic identifiable / confounded / not-detected 三类统计测试；
6. FDR 和分类边界测试；
7. 输出 schema、invalid record 和拒绝覆盖测试；
8. `py_compile` 与 `git diff --check`。

## 12. 明确不做

- 不运行 B4 checkpoint sweep；
- 不使用 test GT 响应选择结构；
- 不把 observational association 写成反事实因果正确性；
- 不重新生成完整 MM3 数据；
- 不训练 B5；
- 不在本审计中修复 `np.random.randint(0, 1)`；
- 不因某一个响应显著就忽略多重比较和其他材质。
