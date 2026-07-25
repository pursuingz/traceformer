# AGENTS.md — Traceformer / PhysCtrl-2 项目约定

本文件是项目级规范,优先级高于默认行为,低于全局 `~/.Codex/AGENTS.md`(红线、沟通方式以全局为准)。
改规范时:先改本文件,再改实践。

## 1. 项目是什么

Will 的硕士课题。基于 PhysCtrl(NeurIPS 2025,arxiv 2509.20358)重构,代号 **Traceformer**。
任务:物理驱动的 3D 点云轨迹生成(给定初始点云 + 物理条件,预测后续形变轨迹)。
核心改动方向:**去扩散的确定性回归** + **autoregressive rollout** + 新输入方案。
论文初稿:`Transformer-Based_Physics_Simulation.docx`(有问题,见 `论文-代码对照表.md`)。

## 2. 必须知道的事实(2026-05 查实,改前先复核)

- **当前 SOTA = run23**(`23(sota)/`):确定性回归(`use_diffusion: false`)+ **原版串行块** `SpatialTemporalTransformerBlock`(不是 v3),8 层 / latent 256,**16.09M 参数**,训 60000 步。
- run23 里 `lambda_laplacian/edge=0`、`collision`/`floor` 实际恒为 0 → 论文主打的「几何正则损失」对结果贡献≈0。
- **v3 并行双流块**(`SpatialTemporalTransformerBlockv3`)是 physctrl_1 阶段产物,**从未做过对照实验**,是候选,不是已验证贡献。
- v3 参数 ≈ v1 的 1.59x(双 FFN + 重复 AdaLN 调制)。等参对齐点:**v3-5L(16.096M)≈ v1-8L(16.092M)**。
- 数据生成的仿真超参全固定 → p(P|c) 退化为确定 → 这是「确定性回归替代扩散」的原理依据。

## 3. 当前主线目标:A2 实验

对比 **v3 并行架构 vs v1 串行架构**,这是论文最终主张。
**铁律:唯一变量是架构。** 数据、损失权重(含为 0 的)、步数、batch、lr、val split 全部冻结一致,否则对比作废。

- baseline:run23 = v1 串行 8 层(`config_dit_base.yaml`)
- 实验臂:v3 并行 5 层(`config_v3_5L.yaml`)= base 仅改 3 处(`transformer_block` / `n_layers` / `output_dir`)
- 对比 checkpoint:两边取**同一 step**(当前定 `checkpoint-45000`)
- ⚠️ **不要用 `config_dit_large.yaml` 当 v3 臂**——它和 base 有 14 处差异(点数 2048、换数据集、num_mat=4、损失全不同),完全混淆,不可比。

## 4. 命令(单卡 5090,从 `src/` 目录跑)

```bash
# 训练(续训:同命令重跑,resume_from_checkpoint: latest 自动接最近 checkpoint)
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/config_v3_5L.yaml

# 评测(baseline / 实验臂各跑一次,只换 --config)
python eval.py --config configs/eval_23.yaml
python eval.py --config configs/eval_v3_5L.yaml
```

- 训练机:服务器上的 5090(Blackwell,需 PyTorch≥2.7/CUDA12.8;train.py 有 torch.compile)。
- 本机只用于不依赖 GPU 的活(如数 参数):`D:/miniconda3/envs/physctrl/python.exe`。

## 5. 配置约定

- 训练 config:`config_<arch>_<层数>.yaml`,以 `config_dit_base.yaml` 为基线,**显式标注每一处改动**(注释写明改了什么、为什么)。
- 评测 config:`eval_<run>.yaml`,**严格镜像对应训练 config 的 model_config 与 train_dataset**,只允许差:`resume` / `vis_dir` / split(由 eval.py 强制 'val')。镜像不一致 = 训练/评测分布错位 = 指标无意义。
- 供 Will 审阅的设计文档、实施计划和实验分析文档默认使用中文；代码标识符、公式和命令保持英文。
- val split 固定:`traj_dataset.py` 中 `random.seed(0)` 打乱后取末 8 个 model(held-out),可复现。
- 评测必须 `use_diffusion: false` + `num_inference_steps: 1`(与 run23 推理路径一致)。

## 6. 评测指标(eval.py)

对比看 **held-out val 上的量化指标**,不要用训练 loss 曲线下结论(那只反映训练集拟合)。
eval.py 现输出:`MSE first-chunk` / **`MSE full-rollout`(主指标)** / `Chamfer` / **`MSE per-step`(误差累积曲线,A2 最有说服力)** / `loss_F`。全部取均值。
范式对比扩展(`utils/eval_metrics.py`,仅全程窗口):全帧 per-frame MSE + **分段 seg-MSE(short f5-10 / mid f11-17 / long f18-24,段间勿再平均)** + GM-MSE + FDE;**Procrustes 位姿/形状分解**(质心偏移/旋转/尺度 s/残余形状 MSE——位置漂 vs 形状糊两轴分开看);固定边弥散度(⚠️ 与 lambda_edge 同族,仅诊断不裁决)。
⚠️ 跨臂口径:`full-rollout`/`first-chunk` 分母含条件 GT 帧数,**输入/输出帧数不同的臂之间不可直接比**;跨范式臂用 per-frame/seg/FDE。任何跨帧单标量都隐含 horizon 权重,下结论用逐帧曲线。

## 7. 项目红线(全局红线之外的补充)

- **不要 push 训练产物**:checkpoint(`*.safetensors`/`*.bin`/`*.pth`)、optimizer state(`*.bin` 单文件可达百 MB,超 GitHub 100MB 上限)、`outputs/`、`23(sota)/`、`*.csv` 已在 `.gitignore`。新增大文件先确认被忽略。
- **`dataset_path: site`** 必须指向服务器上 run23 用的同一批真实数据;路径不对 = 实验全错。
- commit 不署任何 AI 助手(不加 Codex/Claude co-author/contributor);author 用 Will 本人。
- 涉及 `git push` 等全局红线动作:准备好命令后**停下来等确认**,不自动执行。

## 8. 验证

- 改 Python 后跑:`python -m py_compile <file>`(本机 physctrl env 即可)。
- 改架构/层数后核参数:`python count_params.py`(本机 CPU,无需 GPU)。
- 改完主动验证,不要只改不验。

## 9. 实验台账

- 每完成一个实验(训练 + eval 出结果),**必须**按 `实验记录.md` §6 模板补一条记录(思路→实现→结果→分析→结论);结论变化时同步更新 `实验记录.md` 与对应 memory。负结果也要完整记录。
- **每个实验必须挂上对应代码 commit**:提交实验代码(实现 / eval config / 修正)后,立刻把 hash 回填 `实验记录.md`(§4 总览表 commit 列 + §5 详录「代码 commit」行)。一个实验跨多 commit 就全列出。
- `实验记录.md` 是唯一的实验主台账,量化结论以它为准;`phase1-5_roadmap.md` 是规划、memory 是索引,均不与它冲突。
