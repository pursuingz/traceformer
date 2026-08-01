# B0.1 Material-Condition Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 legacy B0 命令、身份锁和输出 schema 不变的前提下，为 `contact_cond@90k` 与 `vxyz_factorized@90k` 增加 `E-only`、`nu-only`、联合参数和类别反事实诊断。

**Architecture:** 纯置换逻辑继续放在 `src/utils/material_condition_diagnostics.py`；`src/diagnose_material_condition.py` 增加严格 profile registry，并在显式传入 profile 时走 B0.1 分支。legacy 分支继续只生成 Normal、`shuffle_params`、`shuffle_class`，B0.1 分支复用同一 donor mapping 生成五条条件路径，并使用独立文件名输出。

**Tech Stack:** Python 3.10、PyTorch、NumPy、OmegaConf、safetensors、unittest、现有 `TrajPipeline`。

## Global Constraints

- 不训练或修改任何模型权重，不修改正式 `src/eval.py`。
- 不传 `--profile` 时，legacy B0 的运行路径、CSV/Markdown 字段和文件名必须保持兼容。
- B0.1 只允许 `contact_cond90` 与 `factorized90` 两个严格注册 profile，不接受任意 checkpoint/config。
- 两个 profile 固定使用相同 41 个 held-out model、13/14/14 材料计数、`start_idx=0` 和 20 帧 deterministic rollout。
- `shuffle_E`、`shuffle_nu`、`shuffle_params` 必须使用同一个同材料、无固定点 donor mapping。
- 第一轮服务器运行只使用 permutation seed 0；只有结果落入预注册灰区时才运行 seeds 1-4。
- 设计与实施文档使用中文；代码标识符和 CLI 使用英文 ASCII。
- 不 stage 或提交 checkpoint、CSV、结果目录、PPT、图片及工作区中已有未跟踪文件。
- commit 作者使用 Will 本人的 Git 配置，不添加 Codex/Claude 或其他 AI 署名。
- 不执行 `git push`。

---

## File Map

- Modify: `src/utils/material_condition_diagnostics.py`
  - 负责可复现 donor mapping、旧参数对置换兼容函数和汇总 intervention 白名单。
- Modify: `src/diagnose_material_condition.py`
  - 负责 profile identity、五路径 rollout、legacy/B0.1 编排、报告和 CLI。
- Create: `src/configs/eval_mm3_contact_vxyz_factorized_90k.yaml`
  - 明确绑定 factorized 90000 checkpoint，消除 `45k` 文件名与90k运行时覆盖的歧义。
- Modify: `src/tests/test_material_condition_diagnostics.py`
  - 覆盖 donor mapping、profile guard、legacy兼容、五路径编排和报告字段。
- Modify: `src/tests/test_contact_ablation_configs.py`
  - 覆盖 factorized 90k eval 与训练配置镜像关系。

---

### Task 1: 统一连续参数 donor mapping

**Files:**
- Modify: `src/utils/material_condition_diagnostics.py`
- Test: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `build_parameter_donor_mapping(records: list[MaterialRecord], seed: int) -> dict[str, MaterialRecord]`
- Preserves: `build_parameter_derangement(records, seed) -> dict[str, tuple[float, float]]`
- Extends: `summarize_rows(rows, intervention, samples, seed)` 支持 `shuffle_e` 与 `shuffle_nu`。

- [ ] **Step 1: 为 donor mapping 写失败测试**

在 `MaterialConditionDiagnosticsTests` 中新增：

```python
def test_component_interventions_share_one_same_material_derangement(self):
    from src.utils.material_condition_diagnostics import (
        MaterialRecord,
        build_parameter_derangement,
        build_parameter_donor_mapping,
    )

    records = [
        MaterialRecord(f"elastic_{i}.h5", 0, 4.0 + i, 0.10 + i * 0.01)
        for i in range(4)
    ] + [
        MaterialRecord(f"sand_{i}.h5", 2, 5.0 + i, 0.20 + i * 0.01)
        for i in range(4)
    ]
    donors = build_parameter_donor_mapping(records, seed=7)
    pairs = build_parameter_derangement(records, seed=7)
    by_name = {record.model: record for record in records}

    self.assertEqual(set(donors), set(by_name))
    for model, donor in donors.items():
        self.assertNotEqual(model, donor.model)
        self.assertEqual(by_name[model].mat_type, donor.mat_type)
        self.assertEqual(pairs[model], (donor.log10_e, donor.nu))
```

再新增 determinism 测试，断言相同 seed mapping 相同，不同 seed 至少一个 donor 不同。

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```bash
python -m unittest \
  src.tests.test_material_condition_diagnostics.MaterialConditionDiagnosticsTests.test_component_interventions_share_one_same_material_derangement -v
```

Expected: FAIL，原因是 `build_parameter_donor_mapping` 尚不存在。

- [ ] **Step 3: 实现 donor mapping，并让旧 API 委托给它**

在 `MaterialRecord` 后增加：

```python
def build_parameter_donor_mapping(
    records: list[MaterialRecord], seed: int
) -> dict[str, MaterialRecord]:
    grouped: dict[int, list[MaterialRecord]] = {}
    for record in records:
        if record.mat_type not in (0, 1, 2):
            raise ValueError("mat_type expected one of 0, 1, 2")
        grouped.setdefault(record.mat_type, []).append(record)

    assignments: dict[str, MaterialRecord] = {}
    for mat_type, group in grouped.items():
        ordered = sorted(group, key=lambda record: record.model)
        if len(ordered) < 2:
            raise ValueError("each material group must contain at least two records")
        rng = np.random.default_rng(seed + mat_type)
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        for index, record in enumerate(shuffled):
            assignments[record.model] = shuffled[(index + 1) % len(shuffled)]
    return assignments


def build_parameter_derangement(
    records: list[MaterialRecord], seed: int
) -> dict[str, tuple[float, float]]:
    return {
        model: (donor.log10_e, donor.nu)
        for model, donor in build_parameter_donor_mapping(records, seed).items()
    }
```

删除旧函数中重复的 grouped/shuffle 实现，不改变旧函数输出。

- [ ] **Step 4: 为新增 intervention 汇总白名单写失败测试**

扩展现有 summary fixture，使每行包含：

```python
row["shuffle_e_prediction_mse"] = 0.5
row["shuffle_nu_prediction_mse"] = 0.75
for metric in ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde"):
    row[f"shuffle_e_{metric}"] = row[f"normal_{metric}"] * 1.10
    row[f"shuffle_nu_{metric}"] = row[f"normal_{metric}"] * 1.20
```

分别调用：

```python
summarize_rows(rows, "shuffle_e", samples=100, seed=3)
summarize_rows(rows, "shuffle_nu", samples=100, seed=3)
```

Expected before implementation: `ValueError`。

- [ ] **Step 5: 扩展 intervention 常量并通过测试**

在 utility 中定义：

```python
MATERIAL_INTERVENTIONS = (
    "shuffle_e",
    "shuffle_nu",
    "shuffle_params",
    "shuffle_class",
)
```

将 `summarize_rows` 的硬编码检查改为该常量成员检查，错误信息列出合法值。

Run:

```bash
python -m unittest src.tests.test_material_condition_diagnostics -v
```

Expected: PASS，legacy 测试无回归。

- [ ] **Step 6: 提交 Task 1**

```bash
git add \
  src/utils/material_condition_diagnostics.py \
  src/tests/test_material_condition_diagnostics.py
git commit -m "Add component material derangements"
```

---

### Task 2: 严格 profile registry 与 factorized 90k config

**Files:**
- Modify: `src/diagnose_material_condition.py`
- Create: `src/configs/eval_mm3_contact_vxyz_factorized_90k.yaml`
- Modify: `src/tests/test_material_condition_diagnostics.py`
- Modify: `src/tests/test_contact_ablation_configs.py`

**Interfaces:**
- Produces: `DiagnosticProfile` dataclass 与 `B01_PROFILES` registry。
- Changes: `_validate_b0_identity(args, records=None, profile=None)`；`profile=None` 保持 legacy contact-cond guard。
- Consumes: Task 1 的 donor mapping，但本任务不运行新增 rollout。

- [ ] **Step 1: 创建 factorized 90k config 的失败测试**

在 `ContactAblationConfigTests` 中新增：

```python
def test_factorized_90k_eval_mirrors_factorized_training_config(self):
    train = _load_structured(
        "config_mm3_contact_vxyz_factorized.yaml", TrainingConfig
    )
    eval_cfg = _load_structured(
        "eval_mm3_contact_vxyz_factorized_90k.yaml", TestingConfig
    )

    self.assertEqual(
        eval_cfg.resume,
        "outputs/mm3_contact_vxyz_factorized_8L/"
        "checkpoint-90000/model.safetensors",
    )
    self.assertEqual(eval_cfg.vis_dir, "vis_results_mm3_contact_vxyz_factorized_90k")
    self.assertEqual(eval_cfg.model_config.contact_velocity_mode, "xyz")
    self.assertEqual(eval_cfg.model_config.contact_injection_mode, "factorized")

    train.train_dataset.input_frames = train.input_frames
    train.train_dataset.output_frames = train.output_frames
    eval_cfg.train_dataset.input_frames = eval_cfg.input_frames
    eval_cfg.train_dataset.output_frames = eval_cfg.output_frames
    OmegaConf.resolve(train)
    OmegaConf.resolve(eval_cfg)
    self.assertEqual(
        OmegaConf.to_container(eval_cfg.model_config, resolve=True),
        OmegaConf.to_container(train.model_config, resolve=True),
    )
    self.assertEqual(
        _without_paths(eval_cfg.train_dataset, {"dataset_path"}),
        _without_paths(train.train_dataset, {"dataset_path"}),
    )
```

- [ ] **Step 2: 运行 config 测试并确认失败**

Run:

```bash
python -m unittest \
  src.tests.test_contact_ablation_configs.ContactAblationConfigTests.test_factorized_90k_eval_mirrors_factorized_training_config -v
```

Expected: FAIL，配置文件不存在。

- [ ] **Step 3: 创建明确的90k eval config**

复制 `eval_mm3_contact_vxyz_factorized_45k.yaml` 的模型与数据字段，只修改：

```yaml
resume: 'outputs/mm3_contact_vxyz_factorized_8L/checkpoint-90000/model.safetensors'
vis_dir: 'vis_results_mm3_contact_vxyz_factorized_90k'
```

文件头注释写明它镜像 `config_mm3_contact_vxyz_factorized.yaml` 的90000 checkpoint。

- [ ] **Step 4: 为两个 profile 的身份锁写失败测试**

在 diagnostic tests 中构造：

```python
factorized_args = self._b0_args()
factorized_args.resume = (
    "outputs/mm3_contact_vxyz_factorized_8L/"
    "checkpoint-90000/model.safetensors"
)
factorized_args.model_config.contact_velocity_mode = "xyz"
factorized_args.model_config.contact_injection_mode = "factorized"
factorized_args.model_config.contact_feature_mask = [1, 1, 1, 1, 1]
```

测试：

```python
_validate_b0_identity(self._b0_args(), self._b0_records(), profile=None)
_validate_b0_identity(self._b0_args(), self._b0_records(), profile="contact_cond90")
_validate_b0_identity(factorized_args, self._b0_records(), profile="factorized90")
```

再断言以下组合失败：

```text
contact config + factorized90 profile
factorized config + contact_cond90 profile
未知 profile
factorized mask 长度或 injection mode 不匹配
```

- [ ] **Step 5: 实现 profile registry**

在 `diagnose_material_condition.py` 顶部定义：

```python
@dataclass(frozen=True)
class DiagnosticProfile:
    name: str
    resume_suffix: str
    model_defaults: dict[str, Any]


B01_PROFILES = {
    "contact_cond90": DiagnosticProfile(
        name="contact_cond90",
        resume_suffix=(
            "outputs/mm3_contact_cond_8L/"
            "checkpoint-90000/model.safetensors"
        ),
        model_defaults={
            "contact_injection_mode": "separate",
            "contact_velocity_mode": "vertical",
            "contact_feature_mask": [1, 1, 1],
            "contact_bias_scale": 1.0,
        },
    ),
    "factorized90": DiagnosticProfile(
        name="factorized90",
        resume_suffix=(
            "outputs/mm3_contact_vxyz_factorized_8L/"
            "checkpoint-90000/model.safetensors"
        ),
        model_defaults={
            "contact_injection_mode": "factorized",
            "contact_velocity_mode": "xyz",
            "contact_feature_mask": [1, 1, 1, 1, 1],
            "contact_bias_scale": 1.0,
        },
    ),
}
```

实现 `_resolve_profile(profile)`，`None` 返回 legacy contact 规格；非空未知值抛出 `ValueError`。将 checkpoint suffix 与 contact defaults 从全局单值改为 resolved profile，但保留其余 top-level/model/dataset/model-list 校验。

- [ ] **Step 6: 运行 profile 与 config 测试**

Run:

```bash
python -m unittest \
  src.tests.test_material_condition_diagnostics \
  src.tests.test_contact_ablation_configs -v
```

Expected: PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add \
  src/diagnose_material_condition.py \
  src/configs/eval_mm3_contact_vxyz_factorized_90k.yaml \
  src/tests/test_material_condition_diagnostics.py \
  src/tests/test_contact_ablation_configs.py
git commit -m "Register B0.1 diagnostic profiles"
```

---

### Task 3: B0.1 五路径 rollout、报告和 CLI

**Files:**
- Modify: `src/diagnose_material_condition.py`
- Modify: `src/tests/test_material_condition_diagnostics.py`

**Interfaces:**
- Produces: `rollout_b01_conditions(...) -> dict[str, torch.Tensor]`
- Changes: `run_diagnostics(..., profile: str | None = None)`、`_write_markdown(..., profile=None)`、`_output_paths(..., profile=None)`、`_print_completion(..., profile=None)`。
- Preserves: `rollout_counterfactuals` 和无 profile 的 legacy orchestration。
- Consumes: Task 1 的 `build_parameter_donor_mapping` 与 Task 2 的 registry。

- [ ] **Step 1: 为五路径 rollout 写失败测试**

新增测试，以 mock `rollout_condition` 返回由参数编码的不同 tensor，并调用：

```python
outputs = rollout_b01_conditions(
    pipeline,
    batch,
    args,
    record,
    donor,
)
```

断言键顺序和条件分别为：

```python
(
    "normal",
    "shuffle_e",
    "shuffle_nu",
    "shuffle_params",
    "shuffle_class",
)
```

并检查五次 `rollout_condition` 参数：

```text
normal         = true E, true nu, true class
shuffle_e      = donor E, true nu, true class
shuffle_nu     = true E, donor nu, true class
shuffle_params = donor E, donor nu, true class
shuffle_class  = true E, true nu, rotated class
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m unittest \
  src.tests.test_material_condition_diagnostics.MaterialConditionDiagnosticsTests.test_b01_rollout_uses_one_donor_for_all_component_interventions -v
```

Expected: FAIL，`rollout_b01_conditions` 不存在。

- [ ] **Step 3: 实现五路径 rollout**

保留原 `rollout_counterfactuals` 不变，新增：

```python
B01_INTERVENTIONS = (
    "shuffle_e",
    "shuffle_nu",
    "shuffle_params",
    "shuffle_class",
)


def rollout_b01_conditions(
    pipeline: Any,
    batch: dict[str, Any],
    args: Any,
    record: MaterialRecord,
    donor: MaterialRecord,
) -> dict[str, torch.Tensor]:
    _validate_normal_material_condition(batch, record)
    conditions = {
        "normal": (record.log10_e, record.nu, record.mat_type),
        "shuffle_e": (donor.log10_e, record.nu, record.mat_type),
        "shuffle_nu": (record.log10_e, donor.nu, record.mat_type),
        "shuffle_params": (donor.log10_e, donor.nu, record.mat_type),
        "shuffle_class": (
            record.log10_e,
            record.nu,
            rotate_material_type(record.mat_type),
        ),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return {
            name: rollout_condition(pipeline, batch, args, *condition)
            for name, condition in conditions.items()
        }
```

字典字面量顺序作为 CSV 和测试的稳定顺序。

- [ ] **Step 4: 为 legacy 与 B0.1 输出路径写失败测试**

断言：

```python
self.assertEqual(
    _output_paths(Path("out"), 3, profile=None),
    (
        Path("out/material_condition_b0_seed3.csv"),
        Path("out/material_condition_b0_seed3.md"),
    ),
)
self.assertEqual(
    _output_paths(Path("out"), 3, profile="factorized90"),
    (
        Path("out/material_condition_b01_factorized90_seed3.csv"),
        Path("out/material_condition_b01_factorized90_seed3.md"),
    ),
)
```

扩展 Markdown/CSV fixture，要求 B0.1 包含 `shuffle_e`、`shuffle_nu`，legacy 不包含这两列。

- [ ] **Step 5: 参数化报告函数而不改变 legacy schema**

增加：

```python
LEGACY_INTERVENTIONS = ("shuffle_params", "shuffle_class")


def _interventions_for_profile(profile: str | None) -> tuple[str, ...]:
    return LEGACY_INTERVENTIONS if profile is None else B01_INTERVENTIONS
```

修改 `_write_markdown`、`_print_completion` 和 `_output_paths` 接收 `profile`，只在 profile 非空时使用 B0.1 标题、文件名和新增 intervention。

- [ ] **Step 6: 为 B0.1 orchestrator 写失败测试**

复用现有 `test_run_diagnostics_orchestrates_start_zero_models_and_writes_reports` 的 fake dataset/model/pipeline，新增 profile 测试：

```python
rows = run_diagnostics(
    factorized_args,
    output_dir=output_dir,
    permutation_seed=3,
    bootstrap_samples=20,
    profile="factorized90",
)
```

mock `rollout_b01_conditions` 返回五条 shape 一致的轨迹。断言：

- 每个模型只调用一次 `rollout_b01_conditions`；
- CSV 包含 donor model、真实/置换参数和五路径指标；
- Markdown 包含四个 intervention 章节；
- 文件名包含 `material_condition_b01_factorized90_seed3`；
- legacy 现有测试的 call count、字段和文件名保持原样。

- [ ] **Step 7: 实现 run_diagnostics 的显式分支**

将签名改为：

```python
def run_diagnostics(
    args: Any,
    output_dir: Path,
    permutation_seed: int,
    bootstrap_samples: int,
    profile: str | None = None,
) -> list[dict[str, Any]]:
```

共同路径加载一次数据、模型和 checkpoint。之后：

- `profile is None`：继续调用旧 `rollout_counterfactuals`，构造原 schema；
- profile 非空：调用 `build_parameter_donor_mapping` 和 `rollout_b01_conditions`，为四种反事实逐一调用 `_metric_row` 和 `_response_row`；
- B0.1 metadata 增加 `donor_model`、`shuffled_log10_e`、`shuffled_nu`、`shuffled_mat_type`；
- 写报告与完成输出时传入 profile。

- [ ] **Step 8: 扩展 CLI，同时保持旧命令有效**

在 parser 中增加：

```python
parser.add_argument(
    "--profile",
    choices=tuple(B01_PROFILES),
    default=None,
    help="Explicit B0.1 profile; omit to preserve legacy B0 behavior.",
)
```

`main()` 将 `cli_args.profile` 传给 `run_diagnostics` 和 `_print_completion`。扩展 help 测试断言 `--profile`、`contact_cond90`、`factorized90` 出现在帮助文本。

- [ ] **Step 9: 运行完整 diagnostic 测试**

Run:

```bash
python -m unittest src.tests.test_material_condition_diagnostics -v
```

Expected: PASS；legacy 和 B0.1 测试全部通过。

- [ ] **Step 10: 提交 Task 3**

```bash
git add \
  src/diagnose_material_condition.py \
  src/tests/test_material_condition_diagnostics.py
git commit -m "Add B0.1 component condition rollouts"
```

---

### Task 4: 全量回归、配置审计与服务器命令

**Files:**
- Verify only: `src/diagnose_material_condition.py`
- Verify only: `src/utils/material_condition_diagnostics.py`
- Verify only: `src/configs/eval_mm3_contact_vxyz_factorized_90k.yaml`
- Verify only: `src/tests/test_material_condition_diagnostics.py`
- Verify only: `src/tests/test_contact_ablation_configs.py`

**Interfaces:**
- Consumes: Tasks 1-3 的全部实现。
- Produces: 可执行的 legacy、contact-cond B0.1 和 factorized B0.1 CLI，以及验证证据。

- [ ] **Step 1: 运行两个相关测试模块**

```bash
python -m unittest \
  src.tests.test_material_condition_diagnostics \
  src.tests.test_contact_ablation_configs -v
```

Expected: 0 failures、0 errors。

- [ ] **Step 2: 运行相关 contact/eval 回归测试**

```bash
python -m unittest \
  src.tests.test_contact \
  src.tests.test_eval_contact_metrics -v
```

Expected: 0 failures、0 errors。

- [ ] **Step 3: 运行语法与 CLI 冒烟测试**

```bash
python -m py_compile \
  src/diagnose_material_condition.py \
  src/utils/material_condition_diagnostics.py
python src/diagnose_material_condition.py --help
```

Expected: py_compile 无输出；help 同时列出 legacy 参数和两个 B0.1 profile。

- [ ] **Step 4: 审计改动和暂存范围**

```bash
git diff --check
git status --short
git diff --name-only HEAD~3..HEAD
```

只允许本计划 File Map 中的源码、测试、config 和已确认文档进入 feature commits。不得 stage 工作区现有未跟踪文件。

- [ ] **Step 5: 给出服务器 seed-0 运行命令**

从服务器 `src/` 目录运行：

```bash
cd /root/code/traceformer/src

python diagnose_material_condition.py \
  --config configs/eval_mm3_contact_cond.yaml \
  --profile contact_cond90 \
  --output-dir results/material_condition_b0/contact_cond90 \
  --permutation-seed 0 \
  --bootstrap-samples 10000

python diagnose_material_condition.py \
  --config configs/eval_mm3_contact_vxyz_factorized_90k.yaml \
  --profile factorized90 \
  --output-dir results/material_condition_b0/factorized90 \
  --permutation-seed 0 \
  --bootstrap-samples 10000
```

Expected artifacts:

```text
results/material_condition_b0/contact_cond90/material_condition_b01_contact_cond90_seed0.csv
results/material_condition_b0/contact_cond90/material_condition_b01_contact_cond90_seed0.md
results/material_condition_b0/factorized90/material_condition_b01_factorized90_seed0.csv
results/material_condition_b0/factorized90/material_condition_b01_factorized90_seed0.md
```

- [ ] **Step 6: 最终本地提交检查**

如果 Task 1-3 已各自提交，本任务不创建空提交。运行：

```bash
git log -4 --oneline
git status --short
```

报告 feature commits、验证结果和服务器命令；不执行 push。
