# Traceformer / PhysCtrl-2 改进路线图(Phase 1–5)

A2 已完成:控制深度与参数后,串行全面优于并行(full-rollout +55%~+106%),劣势随 rollout 放大。
本文件规划 A2 之后的改进实验。

## 全程纪律(每个 Phase 都适用)

- **A2 冻结配置一行不动。** Phase 1–5 改的是损失/训练/输入/块结构 → 属独立实验轴,**不折回 A2 对比**。
- 每个 Phase 独立 config + 独立 output_dir,注释显式标注每处改动(CLAUDE.md §5)。
- eval config 严格镜像对应训练 config,只差 resume / vis_dir / split。
- 单卡 5090,从 `src/` 跑;每臂约 21h → 串行推进、每步设决策门,别同时铺开。
- 改 Python → `py_compile`;改架构/层数 → `count_params.py`。
- push、删文件等红线:准备好命令停下等确认,不自动执行。
- 主指标 = held-out 上的 **full-rollout MSE** + **per-step 曲线**。

---

## Phase 1 — Rollout-aware 训练 【整族已 CLOSED:证伪,论文不主张】

**动机:** 主指标被 rollout 漂移主导(per-step step1→step4 涨 ~130x)。训练单步、用干净 GT 当输入;eval 自回归时输入是模型自己的预测 → exposure bias → 漂移。Phase 1 想在训练时让模型直面 rollout 误差以抗漂移。

**结论(2026-06-08,整族关闭):** 四个变体(1a / 1b / 1b-v2 / 1c)全部训完 + eval,**无一净胜 run23**。同 step(55000)对照:

| 指标 | run23 | 1a(iid加噪) | 1b(K=2,w=1.0) | 1b-v2(w=0.5+warmup) | 1c(K=3,w=1.0) |
|---|---|---|---|---|---|
| first-chunk | 5.995e-4 | +27% | +51% | +46% | **+63%** |
| **full-rollout(主)** | 5.258e-3 | +19% | **+16%(族内最优)** | +43% | +44% |
| per-step 倍率 | 1.0 | — | 2.01→1.70→1.34→**1.05** | 1.92→1.86→1.57→1.34 | 3.00→2.42→1.83→**1.22** |

(1a 对照口径为 45000;其余为 55000。)

**为什么必然失败(论文 ablation 洞见,反向支撑确定性回归主线):** 数据生成仿真超参全固定 → p(P|c) 退化为确定 → 单步映射良态/光滑/确定。这种映射上 ① run23 用干净 GT 把真实单步算子学到最准、反复施加累积误差最小;② rollout-training 往输入塞漂移分布,直接钝化单步算子在干净输入上的精度(first-chunk 必退),而累积误差由单步精度主导,抗漂移补不回单步损失,K 越大污染越重越差(K=3 实证)。**一句话:对确定良态单步映射,最优 rollout 策略=把单步做到极致(run23),而非教它从自错恢复。**

### 各变体存档(代码默认全关,A2/run23 不受影响)

- **1a 输入加噪 / scheduled sampling** 【66b607b,负结果】:训练时给条件 `points_src` 注入 σ(前 20000 步 0→0.03 ramp),目标干净。根因 = iid 高斯噪声 ≠ 结构化逐步累积漂移,只起正则损了基础拟合。memory `physctrl2-phase1a-result`。
- **1b 多步展开 / DAgger** 【710e593,族内最优仍 +16% 净负】:训练时真把预测喂回当条件、对后续 chunk 加 MSE+vel(严格复刻 eval rollout:每 chunk 用上一预测当 `init_pc`、边界重算 `start_vel=pred[:,1]-prev_init[:,-1]`)。chunk0 原完整 loss 块不动,detach 不 BPTT。抗漂移机制真(per-step 单调收窄、末步 1.05),但 trade 从未净胜。
- **1b-v2 降权 + warmup** 【71fee2d,走反更差】:w=0.5、warmup 20000。证明 rollout 权重越大越好——降权后末步倍率退回 1.34、full-rollout 劣化到 +43%,且 warmup 也没救回 first-chunk。
- **1c K=3** 【8b76de3,该族最后一发,证伪】:加深展开到 3 chunk。每一步都比 K=2 退化(连新增监督的 step3:1.34→1.83、目标 step4:1.05→1.22 都更差)→ 加深无救,方向本身错。

实现细节(留档):`options.py` 加 `rollout_unroll_steps`/`rollout_bptt`/`rollout_loss_weight`/`rollout_warmup_steps`(默认 1/False/1.0/0=全关);`traj_dataset.py` 按 K 预留 `required_span`、新增 `points_tgt_roll`;`train.py` 原 loss 块后追加 rollout 段;eval **不设** rollout_* → 推理路径同 run23。configs:`config_v1_rollout_unroll{2,3}.yaml` + `_v2` 及对应 eval。详见 memory `physctrl2-phase1b-result`。

---

## Phase 2 — 速度/加速度预测(新输入方案,结构性抗漂移)

**动机:** 形变是二阶动力学;直接回归绝对位置每步无锚点。

- 输出头从「绝对位置」改为「速度(或加速度)」,rollout 积分出位置:`pos_t = pos_{t-1} + pred_vel` → 时间连续性内建,漂移更小。代码已算 `start_vel`,接得上。
- 变体:预测相对物理先验(匀速外推/重力弹道)的残差 → 目标更小更稳。
- 涉及:model 输出层、train.py 训练目标、eval.py rollout 积分;新 config。
- 论文价值:命中 CLAUDE.md「新输入方案」,物理驱动,故事强。

---

## Phase 3 — 打开几何正则损失(补论文诚信洞)

**动机:** run23 里 `lambda_laplacian/edge=0`、`collision/floor` 恒 0 → 论文主打的「几何正则」贡献≈0,是「声称了但没生效」的洞。

- 把 `lambda_laplacian/edge` 调到合理非零(保局部结构、防散架);查清 collision/floor 为何恒 0 并修。
- 训练前先单 batch 打印各 loss 分量,确认非零且量级不盖过 MSE。
- 涉及:config 损失权重;可能查 `utils/physics.py`。
- 价值:把虚假声称变真贡献;能压制后期 step 散架,改善 Chamfer/体积/晚期 MSE。

---

## Phase 4 — 改良并行(把负结果转成设计洞见)

**动机:** A2 诊断 = 并行块缺「块内跨轴(空间–时间)信息流」。验证一个定向修复能否补回差距。

- **v5(纯并行,不偷偷变串行):** 在 v4 基础上加 **门控融合**(g 由两支共同决定)+ **每支 LayerScale**。若仍输 → 钉死差距是结构性的;若追平 → 融合质量也是部分原因。
- **进一步(若 v5 不够):** 块内加**轻量跨轴 bridge**(temporal 支读 `h0 + α·spatial_attn`,α 可学;α=0 退回纯并行)。这是「软串行」,赢了也要诚实说明它向串行靠拢。
- 涉及:`spacetime.py` 加新 block + 4 处注册(选择器/路由/输出切片);新 train/eval config;`count_params.py` 核参数。
- ⚠️ 不建议直接上交叉注意力/联合注意力——那等于把并行换成串行/joint,赢了无意义。
- 对应给导师邮件里「仍尝试改良并行」的方向。

---

## Phase 5 — 串行块内雕花(最低 ROI,时间富裕才碰)

- spatial→temporal 顺序换成 temporal→spatial 或 S→T→S。
- temporal 后再加一个 FFN(每注意力配 FFN)= 增宽。
- LayerScale / 更好的 AdaLN 调制。
- ⚠️ 已验证深度>宽度,这些大概率不值一臂 21h,不优先。

---

## 当前状态与下一步(2026-06-08)

- **Phase 1(rollout-training)已 CLOSED 证伪。** 论文不主张这条线,转作支持确定性回归的负 ablation。
- **A2 已完成**(串行>并行,见 memory `physctrl2-a2-result`)。
- **论文现有三块支柱(够撑实验章):** ① 确定性回归 run23(主贡献)② A2 串行>并行 ablation ③ rollout-training 负 ablation(反向支撑①)。

**建议:优先收敛成稿,而非再开新实验线。** 现有「主张 + 双 ablation」结构已完整。是否再开 Phase 2–5 取决于成稿时暴露的缺口:

- **Phase 2(速度/加速度预测,新输入方案):** 唯一仍可能成正向贡献的方向(结构性抗漂移,与 rollout-training 的「训练技巧」路线正交)。若要补一个正向贡献,优先这个。但属开放式风险。
- **Phase 3(打开几何正则):** 补论文诚信洞(声称了但 lambda 恒 0),低风险、价值在「把虚假声称变真贡献」。
- **Phase 4(改良并行 v5):** 把 A2 负结果转设计洞见;对应导师邮件「仍尝试改良并行」。
- **Phase 5:** 最低 ROI,仅富余时。

每步用 held-out 量化指标下结论,不看训练 loss 曲线。

---

## aug6win 硬化计划(2026-06-16,搁置中)

**已确立:aug6win 是 Phase 1 第一个 n=14 站得住的正贡献。** 单步数据增强 = run23 固定窗口 {0,5,10,15} + 每 model 额外 2 个随机窗口(单步,不 rollout)。@45000 n=14 vs run23:full-rollout −10.5%、first-chunk −8.5%、start>0 每步 −9~13%、Chamfer/vMSE/aMSE 全赢;唯一退步体积自漂移 +21%。机制:additive 增强(canonical 不动 + 补多样性),与已证伪的 (10) replace 式形成干净对照。代码 commit `078b6d1`(train_extra_random_windows flag)+ `b6947bb`(n=14 eval 数据)。

**硬化成论文级证据,需补三步(按优先级):**
1. **@60000 收敛点复测**:run23 + aug6win 都训到 60000、同 n=14 eval,确认 @45000 优势在训练终点保住(不是中途的暂时领先)。
2. **N 扫描**:`train_extra_random_windows = 0 / 2 / 4 / 6`,画「随机窗口数 vs full-rollout」曲线。单调/有最优 → 从「一个幸运设置」升级为「有规律的方法」。N 已是 config 参数,扫描只改一个数,各 21h 一臂。
3. **(可选)seed 复跑**:n=14 多指标已稳,补 1 个 seed 让正贡献无懈可击(对照 force0 两 seed 暴露的方差问题)。

**论文定位:** 把 aug6win 写成正贡献(确定性回归之上的训练侧增益),(10) replace 式作配对负 ablation 解释「为何增强必须 additive」。这是除「去扩散」外的第二个候选正贡献。

**判据:** @60000 + N 扫描后 full-rollout 仍 ≤ run23 且 start>0 per-step 全步不输 → 确立;否则降级为「@45000 偶发、需更多验证」。
