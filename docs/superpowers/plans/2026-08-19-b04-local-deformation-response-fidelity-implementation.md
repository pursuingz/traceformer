# B0.4 Local Deformation Response Fidelity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个无训练的两阶段诊断工具，先用 train H5 的真实 `F/J` 校准位置局部形变估计器，再审计冻结 41-model test 的 prediction-GT 局部材料响应保真度。

**Architecture:** `src/utils/local_deformation_fidelity.py` 提供纯 NumPy/SciPy 的固定 rest-kNN 估计器、聚合统计和输出；`src/diagnose_local_deformation_fidelity.py` 负责 train calibration、复用 B0.3 factual rollout 以及 CLI。模型、训练和正式评测代码不修改。

**Tech Stack:** Python 3.10、NumPy、SciPy、h5py、PyTorch、OmegaConf、现有 B0.3 诊断框架、`unittest`。

## Global Constraints

- 设计来源：`docs/superpowers/specs/2026-08-19-b04-local-deformation-response-fidelity-design.md`。
- 主设置 `k=16`；`k=8/32` 仅做稳健性检查；condition threshold `1e6`；regularization scale `1e-6`。
- calibration 每材质最多 20 个 train model，seed 0；test 固定 41 model、13/14/14、start_idx=0、seed 0。
- calibration 未通过时，报告必须显式禁止本构解释。
- 不修改模型、训练路径、`eval.py` 或 checkpoint。
- 测试强度为“中”；commit 不包含 AI attribution。

---

### Task 1: 局部形变估计器

**Files:**
- Create: `src/utils/local_deformation_fidelity.py`
- Create: `src/tests/test_local_deformation_fidelity.py`

**Interfaces:**
- `build_rest_neighborhood(rest_points, k, condition_threshold, regularization_scale) -> RestNeighborhood`
- `estimate_local_deformation(trajectory, neighborhood) -> LocalDeformationResult`
- `summarize_local_deformation(result) -> dict[str, np.ndarray | float]`

- [ ] 写刚体平移、均匀缩放、剪切和病态共面邻域失败测试。
- [ ] 运行测试并确认符号缺失导致 RED。
- [ ] 实现 fixed rest-kNN、加权最小二乘 `F_hat`、`J`、Green strain、deviatoric strain 和 edge stretch。
- [ ] 运行测试确认 GREEN。

### Task 2: Calibration 与统计

**Files:**
- Modify: `src/utils/local_deformation_fidelity.py`
- Modify: `src/tests/test_local_deformation_fidelity.py`

**Interfaces:**
- `compare_estimated_to_true_f(...)`
- `build_local_response_rows(...)`
- `build_local_fidelity_rows(...)`
- `evaluate_calibration_gate(...)`

- [ ] 写真实仿射轨迹、response pairing、Spearman 和 calibration gate 失败测试。
- [ ] 运行确认 RED。
- [ ] 实现 train calibration rows、test prediction-GT rows、分材质 fidelity 和冻结 gate。
- [ ] 运行确认 GREEN。

### Task 3: 固定七文件输出

**Files:**
- Modify: `src/utils/local_deformation_fidelity.py`
- Modify: `src/tests/test_local_deformation_fidelity.py`

**Interfaces:**
- `preflight_local_deformation_outputs(output_dir, overwrite)`
- `write_local_deformation_outputs(...)`

- [ ] 写 exact filename、schema、默认拒绝覆盖、中文报告和失败清理测试。
- [ ] 运行确认 RED。
- [ ] 实现事务式七文件输出与 calibration-status 解释限制。
- [ ] 运行确认 GREEN。

### Task 4: 两阶段 CLI

**Files:**
- Create: `src/diagnose_local_deformation_fidelity.py`
- Modify: `src/tests/test_local_deformation_fidelity.py`

**Interfaces:**
- `run_calibration(...)`
- `run_test_fidelity(...)`
- `run_local_deformation_fidelity(...)`
- `build_parser()` / `main()`

- [ ] 写临时 H5 calibration 与 fake B0.3 runtime 集成失败测试。
- [ ] 运行确认 RED。
- [ ] 实现分层 train 采样、真实 `F` 读取、41-model rollout 和结果汇总。
- [ ] 运行确认 GREEN。

### Task 5: 登记和中强度验证

**Files:**
- Modify: `实验记录_1.md`
- Modify: `研究目标与路线图.md`

- [ ] 登记 B0.3 结果与 B0.4 设计、实现状态、服务器命令和解释边界。
- [ ] 运行 `tests.test_local_deformation_fidelity` 与相关 B0.3 测试。
- [ ] 运行新增文件 `py_compile`。
- [ ] 运行 `git diff --check`。
- [ ] 提交设计、实现和台账，不 push。

## Plan Self-Review

- 两阶段协议、固定邻域、校准 gate、分材质响应、输出和解释限制均有实现任务。
- 生产函数先有失败测试，不修改现有模型和评测路径。
- 真实服务器端 checkpoint/H5 运行不属于本机完成声明；最终给出从 `src/` 执行的命令。
