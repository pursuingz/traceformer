# B4 Continuous Material AdaLN 设计

## 1. 背景与裁决依据

当前一期强基线为 `mm3_contact_cond@90k`。B0/B0.1 和 B2 已确认：模型明显依赖 `mat_type`，`E` 的有效响应主要集中于 sand，`nu` 整体很弱；五帧历史、材质类别、重力和接触状态可以替代连续材料参数，使 `E/nu` condition token 的 causal authority 不足。

B3a Joint Material-State Adapter 通过共享 particle-space 残差显著改善 sand，但伤害 elastic、plasticine 和地面穿透。B3b 冻结 B3a 后只训练 12 个 material-stage scalar gate，最终 gate 全部停留在 `0.9945–1.0013`，dev-test 与 B3a 几乎重合。该结果否定了“训练后只调整共享 adapter 幅度即可修复材料失衡”的假设，B3 adapter 家族已按预注册判定关闭。

B4 回到独立 `mm3_contact_cond` baseline，从连续材料条件的信息路径入手，不产生额外 particle-space 修正方向。主要服务目标为 C（`E/nu` 明确、连续地影响运动），A/B（长程精度与分材质均衡）作为不可违反的安全门槛。

## 2. 研究问题与机制假设

### 2.1 研究问题

在保留现有 `E/nu` condition token 和 `mat_type` 路径的前提下，把连续 `E/nu` 直接加入每层 AdaLN conditioning，能否提高连续材料参数对动力学算子的控制权，同时保持 elastic、plasticine、sand 的长程 rollout 精度？

### 2.2 机制假设

当前 `mat_type` 既作为 class token 进入联合 attention，又通过 `class_embedder` 加入 block conditioning embedding；`E/nu` 只作为两个普通 condition token。因此类别条件可以直接控制每层 `CogVideoXLayerNormZero` 的 shift、scale 和 residual gate，而连续参数必须等待粒子 token 通过 attention 间接读取。

B4 将联合连续材料 embedding 加入现有 conditioning embedding：

\[
e_{block}=e_{timestep}+e_{class}+\alpha m(E,\nu).
\]

`CogVideoXLayerNormZero` 的各层参数独立，因此同一个 `m(E,nu)` 可以在不同层形成不同调制。该机制调整原有 attention/FFN 如何处理粒子状态，不添加 B3 的共享粒子残差方向。

## 3. 模型设计

### 3.1 输入与归一化

`TrajDataset` 已输出：

\[
E_{log}=\log_{10}(E),\qquad \nu.
\]

使用固定归一化：

\[
\hat E=\frac{E_{log}-5.5}{1.0},\qquad
\hat\nu=\frac{\nu-0.25}{0.15}.
\]

中心和尺度与 B3 保持一致，覆盖当前数据生成区间。scale 必须为正，输入必须为有限值。

### 3.2 Joint continuous material encoder

新增 `ContinuousMaterialConditioner`：

\[
m(E,\nu)=W_2\,\operatorname{SiLU}(W_1[\hat E,\hat\nu]+b_1)+b_2,
\]

其中：

- `W1: Linear(2, 64)`；
- `W2: Linear(64, 256)`；
- `W2` 的 weight 和 bias 均 zero-init；
- `mat_type` 不再次拼入该 MLP，继续使用已有 class 路径。

参数量为：

\[
2\times64+64+64\times256+256=16,832.
\]

约占 16.09M baseline 的 0.10%。该模块按 batch 生成一个 256 维 embedding，不随粒子数和帧数增加参数。

### 3.3 注入位置

数据流为：

```text
[log10(E), nu]
       |
 fixed normalization
       |
 Linear(2,64) -> SiLU -> zero-init Linear(64,256)
       |
 continuous_material_emb
       |
timestep_emb + class_emb + runtime_scale * continuous_material_emb
       |
8 x existing serial SpatialTemporalTransformerBlock
       |
each block's CogVideoXLayerNormZero
       |
spatial attention -> FFN -> temporal attention
```

实现位置：

1. `MDM_ST.forward()` 从 batch 中整理 `(B,2)` 的 `[E_log, nu]`；
2. `SpaitalTemporalTransformer.forward()` 接收 `material_values`；
3. 在 time embedding 与 class embedding 相加后加入 continuous material embedding；
4. 不修改 `SpatialTemporalTransformerBlock.forward()` 的接口、运算顺序或参数。

### 3.4 保留的信息路径

B4 不替换现有条件：

- 保留 `E_cond_encoder(E)` token；
- 保留 `nu_cond_encoder(nu)` token；
- 保留 `mat_type` class token；
- 保留 `mat_type` 对 block conditioning embedding 的已有注入；
- 保留 gravity、floor、force、contact condition。

因此唯一变量是新增的连续材料 AdaLN 路径。旧 token 路径提供基线信息，B4 路径提高连续条件的逐层控制能力。

### 3.5 Runtime knockout

定义：

```yaml
material_adaln_runtime_scale: 1.0  # ON
material_adaln_runtime_scale: 0.0  # OFF
```

OFF 使用同一个 checkpoint，只关闭新增 embedding。它用于诊断训练后的模型是否依赖新路径，不是独立训练 baseline，也不能把 ON/OFF 差值解释为相对 baseline 的方法增益。

## 4. 配置与兼容性

### 4.1 新增配置字段

```yaml
material_adaln_cond: false
material_adaln_hidden_dim: 64
material_adaln_e_center: 5.5
material_adaln_e_scale: 1.0
material_adaln_nu_center: 0.25
material_adaln_nu_scale: 0.15
material_adaln_runtime_scale: 1.0
```

所有字段默认关闭或使用上述默认值。B4 关闭时不实例化新参数，旧模型参数量、state dict 和 forward 行为保持不变。

### 4.2 互斥约束

`material_adaln_cond` 与 `material_state_adapter` 不允许同时开启。B3 已关闭，同时开启会破坏唯一变量并使信息路径不可解释，应在构造阶段直接报错。

### 4.3 实验配置

新增：

```text
src/configs/config_mm3_b4_material_adaln.yaml
src/configs/eval_mm3_b4_material_adaln_45k.yaml
src/configs/eval_mm3_b4_material_adaln_45k_off.yaml
```

训练 config 相对 `config_mm3_contact_cond.yaml` 只允许改变：

- `output_dir`；
- `stop_after_steps: 45000`；
- B4 配置字段。

OFF eval 相对 ON 只允许改变 `material_adaln_runtime_scale` 和输出标签。train/eval 的 model config 与 dataset 必须严格镜像。

## 5. 训练协议

### 5.1 Screening 预算

从头训练 `seed=0`：

```yaml
max_train_steps: 90000
stop_after_steps: 45000
```

`max_train_steps` 保持 scheduler horizon，45k 只作为 screening 暂停点。对照为 matched-step `mm3_contact_cond@45k`。

冻结以下变量：

- MM3 数据、随机窗口和 point sampling；
- 8 层串行主干；
- contact condition；
- loss、optimizer、batch、gradient accumulation、LR 和 scheduler；
- 原有 `E/nu/class` token 路径。

### 5.2 Gate A：精度安全门槛

B4@45k 必须同时满足：

- overall full-rollout MSE、long MSE、FDE 相对 baseline@45k 恶化不超过 5%；
- elastic、plasticine、sand 各自 long MSE、FDE 恶化不超过 10%；
- first-chunk MSE 恶化不超过 10%；
- 地面穿透率增加不超过 0.25 个百分点；
- 不出现某一材质明显视觉失效。

任一项失败即停止 B4，不运行材料 sweep，也不事后调整 hidden dim、注入层数或学习率。

### 5.3 Gate B：连续材料控制能力

Gate A 通过后，对 baseline@45k 与 B4@45k 执行相同 B2 sweep。由于当前没有 counterfactual GT，该 gate 只判断 causal authority 和响应连续性，不宣称反事实预测准确。

要求：

- `ignored` 比例相对 matched baseline 至少下降 15 个百分点；
- 提升不能只来自 sand，elastic 和 plasticine 均须出现超过 2% effect-size 的模型；
- `E/nu` condition-distance 与 response-strength 的关系不能变弱；
- sand 已有 `E-shape` 单调证据不能明显下降；
- `responsive_non_monotonic + unstable_excessive` 不超过 15%；
- `nu` 至少在两个材质上呈现可辨认的连续响应。

### 5.4 Gate C：路径承重诊断

同 checkpoint 比较 ON/OFF：

- ON/OFF 预测必须存在稳定差异；
- OFF 后材料响应应下降；
- ON 相对 OFF 的变化不能只由少数 sand 长尾样本驱动。

该诊断用于证明模型使用新路径，不替代 Gate A/B。

### 5.5 继续到 90k

只有 Gate A、B、C 全部通过，才从 B4@45k checkpoint 继续到 90k。否则记录负结果并停止。继续训练只改变停止点，不改变 scheduler horizon 或实验变量。

## 6. 测试与验证

实现时至少覆盖：

1. conditioner 输入 shape、finite value 和 scale 检查；
2. 归一化与参数量的手工可验证结果；
3. zero-init 时 B4 与 baseline 前向严格等价；
4. `runtime_scale=0` 时新增路径严格关闭；
5. 非零权重时改变 `E/nu` 能改变输出；
6. `E` 与 `nu` 均能收到梯度；
7. B4 关闭时不产生新增参数，旧 checkpoint 可加载；
8. B3/B4 同时开启时明确报错；
9. train/eval config 镜像和唯一变量审计；
10. 实际 `MDM_ST` smoke test 输出保持 `(B,1,N,3)`；
11. `python -m py_compile`、相关测试模块、`python count_params.py` 和 `git diff --check`。

代码实现前按项目规范由 Will 选择低/中/高/极高测试强度。

## 7. 科学解释边界

- B4 若提高响应强度，只能说明连续条件更承重；没有 paired counterfactual GT 时，不能证明响应物理正确。
- ON/OFF knockout 只能证明新路径被使用，不能作为独立训练 baseline。
- 41-model split 已多次参与机制诊断，当前属于 dev-test。最终论文结论仍需新的冻结 test 或少量同几何、同激励的参数网格 GT。
- 若 B4 失败，不回到 B3 adapter 家族，也不在 dev-test 后调参制造新变体；应记录为“连续 AdaLN 调制在当前数据混杂条件下不足以解决材料控制”。
