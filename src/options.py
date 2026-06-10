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