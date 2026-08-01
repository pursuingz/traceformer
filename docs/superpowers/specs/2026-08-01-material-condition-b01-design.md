# B0.1 材料条件分解与跨模型诊断设计

## 1. 目标

B0 已确认 `mm3_contact_cond@90k` 对材料条件的依赖具有明显类别差异：

- elastic 基本忽略连续 `E/nu`；
- plasticine 会使用连续参数，但样本间差异较大；
- sand 明确使用连续参数；
- 三类材料均强烈依赖 `mat_type`。

B0.1 在不训练新模型的前提下回答两个后续问题：

1. 现有模型主要使用 `E`、`nu` 中的哪一个，还是必须联合使用二者；
2. `v_xyz` factorized 注入是否降低了模型对连续材料参数的有效依赖。

B0.1 只诊断已有 checkpoint，不修改模型权重，也不把反事实条件下的误差直接解释为真实物理泛化能力。

## 2. 实验范围

### 2.1 冻结的模型 profile

只允许以下两个预注册 profile：

| Profile | Checkpoint | 关键 contact 配置 |
|---|---|---|
| `contact_cond90` | `outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors` | `vertical + separate` |
| `factorized90` | `outputs/mm3_contact_vxyz_factorized_8L/checkpoint-90000/model.safetensors` | `xyz + factorized` |

两个 profile 必须使用相同的：

- 41 个 held-out test model；
- 材料数量：elastic 13、plasticine 14、sand 14；
- `start_idx=0` 的完整 20 帧预测 horizon；
- deterministic single-frame rollout；
- seed、输入点云、外力、重力、地板和 GT；
- 8 层串行 `SpatialTemporalTransformerBlock` 主干；
- 2048 粒子和相同 test split。

新增显式评测配置：

```text
src/configs/eval_mm3_contact_vxyz_factorized_90k.yaml
```

该配置必须镜像对应训练配置，并明确指向 90000 checkpoint，避免继续使用文件名含 `45k`、运行时却覆盖为 `90k` 的混乱口径。

### 2.2 不在本阶段处理的内容

- 不训练或微调模型；
- 不修改正式 `eval.py`；
- 不改变 contact feature、material token 或 Transformer 主干；
- 不生成新的 MPM 配对数据；
- 不把 class rotation 结果当作物理准确率；
- 不开放任意 checkpoint/config 运行 B0.1。

## 3. 反事实路径

每个模型均保留 Normal 路径，并增加四种反事实路径。

### 3.1 Normal

使用真实：

```text
(log10(E), nu, mat_type)
```

Normal 输出必须与对应 profile 的常规 eval 在 full-rollout、long-segment 和 FDE 口径上对齐。

### 3.2 Shuffle E

只在同一 `mat_type` 内对 `log10(E)` 做无固定点置换：

```text
(E_i, nu_i, class_i) -> (E_j, nu_i, class_i)
```

其中 `j != i`，且 donor 与 receiver 属于同一种材料。该路径只测量模型对连续 `E` 输入的依赖。

### 3.3 Shuffle nu

只在同一 `mat_type` 内对 `nu` 做无固定点置换：

```text
(E_i, nu_i, class_i) -> (E_i, nu_j, class_i)
```

该路径只测量模型对连续 `nu` 输入的依赖。

### 3.4 Shuffle parameter pair

保留原 B0 的同材料参数对置换：

```text
(E_i, nu_i, class_i) -> (E_j, nu_j, class_i)
```

它用于判断联合扰动是否产生单参数扰动无法解释的交互效应。

### 3.5 Shuffle class

保留原 B0 的循环类别替换：

```text
elastic -> plasticine -> sand -> elastic
```

真实 `E/nu` 保持不变。该路径只测量 class-token 依赖，不代表物理准确率，因为历史轨迹、连续参数和类别会形成 OOD 不一致组合。

## 4. 实现方案

采用**单脚本、严格 profile registry** 方案扩展现有：

```text
src/diagnose_material_condition.py
```

CLI 新增：

```text
--profile {contact_cond90,factorized90}
```

`--profile` 不设默认值：

- 不传 `--profile` 时，严格执行 legacy B0，只运行原有 Normal、`shuffle_params`、`shuffle_class`，并保持原 CSV/Markdown 文件名和字段；
- 显式传入 profile 时，进入 B0.1，运行本设计的全部路径并使用 B0.1 输出文件名。

这样旧命令的运行时间、输出 schema 和身份锁均不发生静默变化。每个 B0.1 profile 独立声明并校验：

- checkpoint suffix；
- top-level config；
- `model_config`；
- dataset config；
- contact velocity mode、injection mode、feature mask 和 bias scale；
- test model 名单和材料计数。

不得将身份锁退化为“任意配置只要能加载就运行”。

置换逻辑放在 `src/utils/material_condition_diagnostics.py` 的纯函数中。`E-only`、`nu-only` 和 pair 置换必须复用同一个 permutation seed 和 donor 映射，使三个实验之间的 donor 关系可审计。

## 5. 输出与兼容性

输出目录由调用者按 profile 隔离：

```text
results/material_condition_b0/contact_cond90/
results/material_condition_b0/factorized90/
```

每个目录生成：

```text
material_condition_b01_<profile>_seed<seed>.csv
material_condition_b01_<profile>_seed<seed>.md
```

CSV 每个 model 一行，至少包含：

- 真实和置换后的 `log10(E)`、`nu`、`mat_type`；
- Normal、`shuffle_E`、`shuffle_nu`、`shuffle_params`、`shuffle_class` 的主指标；
- 各反事实预测相对 Normal 的 prediction MSE 和 final-frame prediction MSE。

Markdown 同时报告 overall、elastic、plasticine 和 sand，包含：

- baseline / counterfactual 均值；
- 相对变化；
- 逐模型配对 delta 的 bootstrap 95% CI；
- response ratio；
- `used / ignored / ambiguous` 标签。

现有 B0 的历史 CSV/Markdown 不修改；旧命令重新运行时仍生成相同 schema。新增字段和文件名明确标记为 B0.1，避免把旧报告静默覆盖。

## 6. 运行顺序与成本控制

第一轮只运行 `permutation-seed=0`：

1. `contact_cond90`；
2. `factorized90`。

只有出现以下任一情况，才补跑 seeds `1-4`：

- 相对变化落在 `2%-5%` 的判据灰区；
- 95% CI 包含 0；
- 不同材料方向相反；
- 结论由少数高误差样本主导；
- 两个 profile 的差值不足以支持稳定判断。

`shuffle_class` 在补充 seed 中不是重点；多 seed 的主要目的，是检验 `E/nu` donor 配对是否改变结论。

## 7. 判定规则

对每个 profile、材料组和误差指标，定义：

\[
\Delta_i = L_i^{\mathrm{counterfactual}} - L_i^{\mathrm{normal}}.
\]

沿用 B0 判据：

- **used**：主要长程指标恶化至少 5%，且配对 bootstrap 95% CI 不包含 0；
- **ignored**：主要指标变化低于 2%，CI 包含 0，且 prediction response 很小；
- **ambiguous**：其余情况。

跨 profile 的核心比较不是绝对 response 大小，而是：

1. Normal 精度是否可比；
2. `shuffle_E/nu` 后相对 GT 的误差变化；
3. 反事实 prediction response 相对各自 Normal error 的比例；
4. 分材料结论是否一致。

如果 factorized 的 Normal 误差更大但 response ratio 更小，支持“`v_xyz` 形成状态 shortcut、降低材料依赖”的假设；如果 response 更大但准确率仍差，则说明它使用材料但状态表示或闭环稳定性仍有问题。

## 8. 局限性

B0.1 测量的是模型依赖，不是材料因果规律是否正确。置换后的参数与前 5 帧历史可能不一致，即便参数仍处于真实测试分布，也属于条件反事实。

最终物理正确性仍需要：

```text
相同初始形状、速度、边界和外力
+ 只改变材料参数
-> 对应的配对 GT 未来轨迹
```

B0.1 只用于决定下一次昂贵训练应该修改材料注入、运动状态表示，还是优先补充配对数据。

## 9. 测试与验收

实现前先添加失败测试，至少覆盖：

1. `shuffle_E` 只改变 E，且同材料、无固定点；
2. `shuffle_nu` 只改变 nu，且同材料、无固定点；
3. 三种连续参数置换复用同一 donor mapping；
4. 不传 `--profile` 时，legacy B0 的运行路径、字段和文件名保持兼容；
5. `factorized90` 只接受严格匹配的 checkpoint 和模型配置；
6. profile/config/checkpoint 交叉使用时立即失败；
7. 两个新增 intervention 正确进入 CSV、Markdown、bootstrap 和终端汇总；
8. Normal rollout 不因新增 intervention 改变；
9. 41 个模型、材料计数和 `start_idx=0` 约束继续生效；
10. 新 90k eval config 与训练 config 除评测字段外严格镜像。

验证命令：

```bash
python -m unittest src.tests.test_material_condition_diagnostics -v
python -m unittest src.tests.test_contact_ablation_configs -v
python -m py_compile \
  src/diagnose_material_condition.py \
  src/utils/material_condition_diagnostics.py
python src/diagnose_material_condition.py --help
git diff --check
```

验收时不得生成、stage 或提交 checkpoint、CSV、训练输出、PPT、图片或其他现有未跟踪文件。
