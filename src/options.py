from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class TrainingConfig:
    image_size: int
    # train_batch_size = 16
    # eval_batch_size = 16  # how many images to sample during evaluation
    # num_epochs = 50
    # gradient_accumulation_steps = 1
    # learning_rate = 1e-4
    # lr_warmup_steps = 500
    # save_image_epochs = 10
    # save_model_epochs = 30
    # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    # logging
    output_dir: str
    logging_dir: str
    vis_dir: str
    report_to: Optional[str]
    local_rank: int
    tracker_project_name: str

    # Training
    seed: Optional[int]
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    use_8bit_adam: bool
    allow_tf32: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    mixed_precision: Optional[str]
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_train_steps: int
    validation_sanity_check: bool
    resume_step: Optional[int]
    push_to_hub: bool
    set_grads_to_none: bool
    lambda_vel: float
    lambda_mask : float
    lambda_momentum: float
    lambda_deform: float
    lambda_laplacian: float
    lambda_collision: float
    lambda_edge: float
    lambda_floor: float      # Floor collision penalty weight
    laplacian_k: int
    collision_margin: float
    overfit: bool

    # Diffusion Specific
    condition_drop_rate: float
    # Dataset
    train_dataset: Dict

    # Model
    model_type: str
    pred_offset: bool
    model_config: Dict
    pc_size: int
    use_diffusion: bool = False
    # Rollout-aware training (scheduled sampling): perturb conditioning points_src with
    # noise to mimic autoregressive drift at rollout time; target stays clean. 0.0 = off.
    rollout_input_noise_std: float = 0.0
    rollout_noise_warmup_steps: int = 0
    # 1b multi-step rollout training: unroll K chunks during training, feeding each predicted
    # chunk back as conditioning (mirrors eval.py rollout). 1 = off (single chunk, == run23).
    rollout_unroll_steps: int = 1
    # If True keep the autograd graph across chunks (true BPTT). If False detach the fed-back
    # prediction (DAgger-style exposure: model trains on its own real errors, cheaper/stable).
    rollout_bptt: bool = False
    # 1b-v2: weight on the rollout-chunk loss (chunk 0 always keeps weight 1). Full weight (1.0)
    # cannibalises single-step/base-fit accuracy; down-weight to keep it sharp. Linearly ramped
    # from 0 over rollout_warmup_steps so base fit is established before rollout loss kicks in.
    rollout_loss_weight: float = 1.0
    rollout_warmup_steps: int = 0
    # Curriculum over (unroll horizon K, windows-per-model): list of [step_threshold, K, n_win],
    # step-ascending. e.g. [[0,1,8],[10000,2,4],[25000,3,2]] = K=1 sampling 8 windows/model, then
    # K=2 sampling 4, then K=3 sampling 2. The train loader is rebuilt at each stage boundary, so
    # each stage uses its own window length (=> its own start-frame range) and sampling density.
    # None = off (fall back to the fixed rollout_unroll_steps). Back-compatible with 2-element
    # entries [step, K] (then windows-per-model falls back to the stride-5 count).
    rollout_curriculum: Optional[List] = None
    # Draw each training window's start frame at random over the whole trajectory (data aug),
    # instead of the fixed stride-5 grid. Eval is unaffected (test split stays deterministic).
    rollout_random_window: bool = False
    # With random windows on, force one window per model per epoch to start at 0 (rest stay random).
    # Restores the start=0 training density that pure random sampling dilutes, while keeping the
    # mid-trajectory coverage of random windows. Only affects training; eval is unchanged.
    rollout_force_start0: bool = False
    # Single-step (no rollout) data aug: keep run23's fixed stride-5 windows {0,5,10,...} AND append
    # this many extra random-start windows per model per epoch. Default 0 = run23 unchanged.
    # Differs from rollout_random_window (which REPLACES the fixed grid); here the canonical windows
    # stay and randoms are additive. Only affects the train split; eval/val windows are unchanged.
    train_extra_random_windows: int = 0
    # Weights-only init from another run's checkpoint (path to a model.safetensors). Loads ONLY the
    # model weights, leaving a FRESH optimizer/scheduler and global_step=0, so you can fine-tune a
    # converged model (e.g. run23@60k) at a different (lower) lr. Distinct from resume_from_checkpoint,
    # which restores optimizer+scheduler+step from THIS run's own output_dir. Skipped if this run
    # already has its own checkpoint to resume (so an interrupted fine-tune resumes correctly).
    # None = off.
    init_from_checkpoint: Optional[str] = None
    # Prediction granularity (new axis): number of frames the model predicts per forward pass.
    # Default 5 = run23 (5-frame chunk, eval rolls back every 5 frames). Set to 1 for single-frame
    # autoregression (eval rolls one frame at a time). INPUT_FRAMES stays 5. With output_frames<5
    # the multi-frame losses (vel/deform) become structurally inactive (single frame has no intra-
    # output frame difference); training reduces to position MSE + floor. See plan / 实验记录.md.
    output_frames: int = 5
    # 时间感受野轴:输入历史帧数。默认 5=所有现有臂(无此键则 5)。1=单帧输入消融(见 plan / 实验记录.md)。
    input_frames: int = 5
    # Non-curriculum random-window sampling: number of random-start windows to emit per model per
    # epoch (only used when rollout_random_window=True without a curriculum). None -> stride-5 count
    # (~4). For output_frames=1, set ~20 so training covers every input-window start the eval rollout
    # visits (0,1,...,~19), matching the train/eval start distribution.
    windows_per_model: Optional[int] = None
    # Single-frame boundary loss_F: with output_frames=1 the multi-frame loss_F (DeformLoss.forward,
    # needs >=3 frames) is structurally inactive. When True, restore it via DeformLoss.forward_single_step
    # using the input_last->pred forward-difference velocity to advance GT F one MPM step vs GT F_next.
    # Default False -> all existing configs unchanged (does NOT alter lambda_deform's output=1 semantics).
    single_frame_deform: bool = False
    # mm3 多材质:弹性先验损失(laplacian/edge/deform)仅作用 mat_type==0(elastic)样本。
    # sand 颗粒流动/塑性屈服破坏邻边保持假设,DeformLoss 按弹性本构写,对非弹性样本强加=错误先验。
    # Default False -> 单材质臂字节不变(train.py 按 args.get 读取)。
    geom_elastic_only: bool = False

@dataclass
class TestingConfig:
    dataloader_num_workers: int 
    pc_size: int
    model_type: str
    pred_offset: bool
    model_config: Dict
    train_dataset: Dict
    resume: str
    vis_dir: str
    eval_batch_size: int
    seed: int
    num_inference_steps: int
    use_diffusion: bool = False
    # Prediction granularity (mirror of TrainingConfig.output_frames). Must match the value the
    # checkpoint was trained with so the model is reconstructed with the same frame count. Default 5.
    output_frames: int = 5
    # 时间感受野轴:mirror TrainingConfig.input_frames,须与 checkpoint 训练值一致。默认 5=现有臂。
    input_frames: int = 5
    # opt-in:dump 每模型明细 CSV(log10E/full-rollout/体积),供 E 分档。默认 False=不写、现有 eval 零影响。
    per_model_csv: bool = False
    # 推理时地板投影(诊断/缓解穿透用):rollout 每步预测出来后,把 y < floor_height 的点直接
    # 夹到 floor_height(不改训练、不改 loss,只在 eval 的 rollout 反馈路径生效)。用于验证/缓解
    # sf/sfG 单帧自回归的地板穿透痼疾(见 实验记录.md / physctrl2-mm3-experiment memory)。
    # 默认 False -> 现有所有 eval 结果零影响。
    floor_projection: bool = False