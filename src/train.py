import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
import sys
from pathlib import Path
from omegaconf import OmegaConf
from options import TrainingConfig

import numpy as np
import h5py
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDPMPipeline, DDIMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, UNet2DModel
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_cosine_schedule_with_warmup
from pipeline_traj import TrajPipeline
from accelerate.utils import DistributedDataParallelKwargs

from model.spacetime import MDM_ST
from dataset.traj_dataset import TrajDataset

from utils.visualization import save_pointcloud_video, save_pointcloud_json, save_threejs_html
from utils.physics import loss_momentum
from utils.physics import DeformLoss

logger = get_logger(__name__)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_file_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_file, log_path

INPUT_FRAMES = 5
OUTPUT_FRAMES = 5
ROLLOUT_STEPS = 4
ROLLOUT_HORIZON = 20   # predicted-frame horizon for rollout (ROLLOUT_STEPS = ceil(HORIZON/OUTPUT_FRAMES))


def current_stage(global_step, curriculum, default_K):
    """Curriculum over (unroll horizon K, windows-per-model). Entries are [step, K, n_win]
    (n_win optional), step-ascending. Returns (K, n_win) for the largest threshold <= global_step.
    None curriculum -> (default_K, None)."""
    if not curriculum:
        return default_K, None
    def _parse(e):
        return int(e[1]), (int(e[2]) if len(e) > 2 else None)
    K, nw = _parse(curriculum[0])
    for e in curriculum:
        if global_step >= int(e[0]):
            K, nw = _parse(e)
    return K, nw


def _seed_worker(worker_id):
    """Give each DataLoader worker a distinct numpy/random seed so random-window starts differ."""
    import numpy as _np, random as _r, torch as _t
    s = (_t.initial_seed() + worker_id) % (2 ** 32)
    _np.random.seed(s); _r.seed(s)


def build_raw_reference(batch, dataset_cfg, target_frames):
    raw_sequences = []
    for j, model_name in enumerate(batch['model']):
        h5_path = os.path.join(dataset_cfg.dataset_path, model_name)
        with h5py.File(h5_path, 'r') as model_metas:
            model_pcls = torch.from_numpy(np.array(model_metas['x']))

        point_indices = batch['point_indices'][j].cpu().numpy()
        end_idx = min(target_frames, model_pcls.shape[0])
        raw_seq = model_pcls[:end_idx][:, point_indices].float()
        raw_seq = (raw_seq - dataset_cfg.norm_fac) / 2
        if raw_seq.shape[0] < target_frames:
            pad_count = target_frames - raw_seq.shape[0]
            raw_seq = torch.cat([raw_seq, raw_seq[-1:].repeat(pad_count, 1, 1)], dim=0)
        raw_sequences.append(raw_seq)

    return torch.stack(raw_sequences, dim=0)


def build_knn_indices(points, k):
    if points.ndim == 4:
        points = points[:, 0]
    n_points = points.shape[1]
    if n_points <= 1:
        return None
    k_eff = min(max(int(k), 1), n_points - 1)
    dists = torch.cdist(points, points)
    knn_idx = torch.topk(dists, k=k_eff + 1, dim=-1, largest=False).indices[..., 1:]
    return knn_idx


def laplacian_deformation_loss(pred_points, ref_points, knn_idx):
    if knn_idx is None:
        return pred_points.new_tensor(0.0)
    batch_size, n_frames = pred_points.shape[:2]
    per_batch_losses = []
    for b in range(batch_size):
        pred_b = pred_points[b]           # [n_frames, N, 3]
        ref_b = ref_points[b]             # [n_frames, N, 3]
        idx_b = knn_idx[b]                # [N, k]
        loss_t = 0.0
        for t in range(n_frames):
            pred_nb = pred_b[t, idx_b, :]     # [N, k, 3]
            ref_nb = ref_b[t, idx_b, :]       # [N, k, 3]
            pred_lap = pred_b[t] - pred_nb.mean(dim=1)   # [N, 3]
            ref_lap = ref_b[t] - ref_nb.mean(dim=1)      # [N, 3]
            loss_t += F.mse_loss(pred_lap, ref_lap)
        per_batch_losses.append(loss_t / n_frames)
    return torch.stack(per_batch_losses).mean()


def edge_length_regularization(pred_points, ref_points, knn_idx):
    if knn_idx is None:
        return pred_points.new_tensor(0.0)
    batch_size, n_frames = pred_points.shape[:2]
    per_batch_losses = []
    for b in range(batch_size):
        pred_b = pred_points[b]           # [n_frames, N, 3]
        ref_b = ref_points[b]             # [n_frames, N, 3]
        idx_b = knn_idx[b]                # [N, k]
        loss_t = 0.0
        for t in range(n_frames):
            ref_edge_len = torch.norm(ref_b[t, idx_b, :] - ref_b[t, :, None, :], dim=-1)   # [N, k]
            pred_edge_len = torch.norm(pred_b[t, idx_b, :] - pred_b[t, :, None, :], dim=-1) # [N, k]
            loss_t += F.mse_loss(pred_edge_len, ref_edge_len)
        per_batch_losses.append(loss_t / n_frames)
    return torch.stack(per_batch_losses).mean()


def collision_loss(pred_points, floor_height=None, margin=0.01):
    batch_size, n_frame, n_points, _ = pred_points.shape
    pred_flat = pred_points.reshape(batch_size * n_frame, n_points, 3)
    dists = torch.cdist(pred_flat, pred_flat)
    eye = torch.eye(n_points, device=pred_points.device, dtype=torch.bool).unsqueeze(0)
    penetration = F.relu(margin - dists)
    penetration = penetration.masked_fill(eye, 0.0)
    self_collision = (penetration ** 2).sum() / (~eye).sum().clamp_min(1)
    '''
    floor_collision = pred_points.new_tensor(0.0)
    if floor_height is not None:
        floor_h = floor_height.reshape(batch_size, 1, 1)
        floor_pen = F.relu(floor_h - pred_points[..., 1])
        floor_collision = (floor_pen ** 2).mean()

    return self_collision + floor_collision
    '''
    return self_collision


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main(args):
    vis_dir = os.path.join(args.output_dir, args.vis_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    log_file = None
    if accelerator.is_main_process and args.output_dir is not None:
        log_file, log_path = setup_file_logging(args.output_dir)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = {}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

        src_snapshot_folder = os.path.join(cfg.output_dir, 'src')
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        for folder in ['model', 'dataset']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        shutil.copy(os.path.abspath(__file__), os.path.join(cfg.output_dir, 'src', 'train.py'))

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Prediction granularity (new axis): output_frames is config-driven (default 5 = run23). Reassign
    # the module constant so all downstream references (model build, model_input repeat, rollout,
    # validation) follow it. INPUT_FRAMES stays 5. output_frames=1 -> single-frame autoregression.
    # ROLLOUT_STEPS (used by the validation-viz rollout) is derived from a fixed ~20-frame horizon so
    # output=5 stays 4 (=20/5) and output=1 becomes 20; keeps the visualised horizon constant.
    global OUTPUT_FRAMES, ROLLOUT_STEPS, INPUT_FRAMES
    # 时间感受野轴:input_frames config 驱动(默认 5 = 所有现有臂,无此键 → 字节不变)。input=1 = 单帧输入消融。
    INPUT_FRAMES = args.get('input_frames', 5)
    OUTPUT_FRAMES = args.get('output_frames', 5)
    ROLLOUT_STEPS = -(-ROLLOUT_HORIZON // OUTPUT_FRAMES)
    args.train_dataset.input_frames = INPUT_FRAMES
    args.train_dataset.output_frames = OUTPUT_FRAMES
    # 1b: forward the top-level unroll setting to the dataset so it reserves K output chunks
    # per window and emits points_tgt_roll. Single source of truth = top-level config.
    # Curriculum: the train loader is rebuilt at each stage boundary (see the train loop), so each
    # stage uses its own K (-> window length / start-frame range) and windows-per-model sampling.
    # None -> fixed rollout_unroll_steps. Initial loader uses the stage at step 0; resume corrects
    # it via the rebuild check at the top of the loop.
    curriculum = OmegaConf.to_container(args.rollout_curriculum) if args.get('rollout_curriculum', None) else None
    args.train_dataset.rollout_random_window = args.get('rollout_random_window', False)
    args.train_dataset.rollout_force_start0 = args.get('rollout_force_start0', False)
    args.train_dataset.train_extra_random_windows = args.get('train_extra_random_windows', 0)
    if curriculum:
        K0, nw0 = current_stage(0, curriculum, args.rollout_unroll_steps)
        args.train_dataset.rollout_unroll_steps = K0
        args.train_dataset.windows_per_model = nw0
    else:
        args.train_dataset.rollout_unroll_steps = args.rollout_unroll_steps
        # Non-curriculum random-window sampling: forward windows-per-model so output_frames=1 can
        # sample ~20 random starts/model (cover every eval input-window start). None -> stride-5 count.
        args.train_dataset.windows_per_model = args.get('windows_per_model', None)
    args.model_config.cond_frames = INPUT_FRAMES
    model = MDM_ST(args.pc_size, OUTPUT_FRAMES, n_feats=3, model_config=args.model_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    output_str = f"Total trainable parameters: {total_params / 1e6:.2f}M\n"

    # 写入到 txt 文件
    with open("model_parameters.txt", "w") as f:
        f.write(output_str)
    # if args.gradient_checkpointing:
    #     model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params = model.parameters()
    # Optimizer creation
    optimizer = optimizer_class(
        [
            {"params": params, "lr": args.learning_rate},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # if args.model_type == 'dit_st_water':
    #     from dataset.water_dataset import TrajDataset
    # Dataset and DataLoaders creation:
    train_dataset = TrajDataset('train', args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True, worker_init_fn=_seed_worker)

    val_dataset = TrajDataset('val', args.train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    # noise = torch.randn(sample_image.shape)
    # timesteps = torch.LongTensor([50])
    # noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    # Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num of Trainable Parameters (M) = {sum(p.numel() for p in model.parameters()) / 1000000}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Log to = {args.output_dir}")
    global_step = 0
    first_epoch = 0
    loss_history = {}
    loss_rows = []
    loss_csv_path = os.path.join(args.output_dir, "loss_history.csv")
    loss_csv_header = [
        "step",
        "loss",
        "loss_xyz",
        "loss_mask",
        "loss_vel",
        "loss_p",
        "loss_F",
        "loss_floor",
        "loss_laplacian",
        "loss_collision",
        "loss_edge",
    ]

    # Potentially load in the weights and states from a previous save
    resumed = False
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resumed = True
    else:
        initial_global_step = 0

    # Weights-only init (fine-tune a converged model at a fresh/lower lr). Only when we did NOT
    # resume this run's own checkpoint, so an interrupted fine-tune resumes from its own latest
    # checkpoint instead of re-initialising from the source every restart. Optimizer/scheduler/
    # global_step stay fresh -> the run's own learning_rate governs the fine-tune.
    if not resumed and args.get('init_from_checkpoint', None):
        from safetensors.torch import load_file
        ckpt = load_file(args.init_from_checkpoint, device='cpu')
        missing, unexpected = accelerator.unwrap_model(model).load_state_dict(ckpt, strict=False)
        accelerator.print(
            f"Init weights from {args.init_from_checkpoint} "
            f"(loaded={len(ckpt)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if len(missing) > 50 or len(unexpected) > 50:
            accelerator.print(
                "  [WARN] large key mismatch -> check the fine-tune config's architecture "
                "matches the source checkpoint."
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    noise_scheduler = None
    if args.use_diffusion:
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    loss_deform = DeformLoss()

    # curriculum: rebuild the train loader whenever the stage (K, windows-per-model) changes.
    def _build_train_loader(K, n_win):
        args.train_dataset.rollout_unroll_steps = K
        args.train_dataset.windows_per_model = n_win
        ds = TrajDataset('train', args.train_dataset)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.train_batch_size, shuffle=True,
                                         num_workers=args.dataloader_num_workers, pin_memory=True,
                                         worker_init_fn=_seed_worker)
        return accelerator.prepare(dl)

    cur_stage = current_stage(0, curriculum, args.rollout_unroll_steps)  # matches the initial loader
    while global_step < args.max_train_steps:
        model.train()
        train_loss = 0.0
        if curriculum:
            stage_now = current_stage(global_step, curriculum, args.rollout_unroll_steps)
            if stage_now != cur_stage:
                cur_stage = stage_now
                train_dataloader = _build_train_loader(cur_stage[0], cur_stage[1])
                if accelerator.is_main_process:
                    logger.info(f"[curriculum] step {global_step}: rebuilt loader K={cur_stage[0]} windows/model={cur_stage[1]}")
        for step, (batch, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                latents = batch['points_tgt'] # (bsz, n_frames, n_points, 3)

                bsz = latents.shape[0]
                cond_points_src = batch['points_src']
                if args.use_diffusion:
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    model_input = noise_scheduler.add_noise(latents, noise, timesteps)
                else:
                    timesteps = torch.zeros((bsz,), device=latents.device, dtype=torch.long)
                    # Rollout-aware (scheduled sampling): perturb the conditioning frames so the
                    # model learns to correct its own drift at autoregressive rollout time. The
                    # target (latents) stays clean. std=0 -> disabled (default for all A2 configs).
                    sigma = args.rollout_input_noise_std
                    if sigma > 0:
                        if args.rollout_noise_warmup_steps > 0:
                            sigma = sigma * min(1.0, global_step / args.rollout_noise_warmup_steps)
                        cond_points_src = batch['points_src'] + torch.randn_like(batch['points_src']) * sigma
                    # Use the last source frame repeated as input, instead of zeros,
                    # so that PointEmbed produces differentiated per-point embeddings.
                    # Add small noise to break uniformity across the 5 output frames,
                    # otherwise all output-frame tokens would have identical embeddings.
                    last_src_frame = cond_points_src[:, -1:, :, :]  # (B, 1, N, 3)
                    model_input = last_src_frame.repeat(1, OUTPUT_FRAMES, 1, 1)  # (B, F, N, 3)
                    model_input = model_input + torch.randn_like(model_input) * 0.02
                    #print("Running training without diffusion.")

                if args.condition_drop_rate > 0:
                    # Randomly drop some of the latents
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    null_emb = (random_p > args.condition_drop_rate).float()[..., None, None]
                else:
                    null_emb = None

                # Predict the noise residual
                pred_sample = model(model_input, timesteps, cond_points_src, batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1], batch['drag_point'], batch['floor_height'], batch['gravity'], batch['base_drag_coeff'], y=None if 'mat_type' not in batch else batch['mat_type'], null_emb=null_emb, start_vel=batch.get('start_vel', None), points_rest=batch.get('points_rest', None))
                losses = {}

                loss = F.mse_loss(pred_sample.float(), latents.float())
                losses['xyz'] = loss.detach().item()

                if args.lambda_mask > 0:
                    loss_mask = F.mse_loss(pred_sample[batch['mask']], latents[batch['mask']])
                    loss += args.lambda_mask * loss_mask
                    losses['mask'] = loss_mask.detach().item()

                # Velocity loss needs >=2 output frames (intra-output frame difference). With
                # output_frames=1 the diff is an empty tensor -> mse_loss returns NaN and poisons the
                # loss; skip it (structurally undefined for single-frame output). output_frames>=5 unaffected.
                if args.lambda_vel > 0. and pred_sample.shape[1] >= 2:
                    target_vel = latents[:, 1:] - latents[:, :-1]
                    pred_vel = (pred_sample[:, 1:] - pred_sample[:, :-1])
                    loss_vel = F.mse_loss(target_vel.float(), pred_vel.float())
                    losses['loss_vel'] = loss_vel.detach().item()
                    loss = loss + args.lambda_vel * loss_vel

                if 'vol' in batch and args.lambda_momentum > 0.:
                    loss_p = loss_momentum(x=pred_sample, vol=batch['vol'], force=batch['weighted_force'],
                        drag_pt_num=batch['mask'][:, 0, :].sum(dim=1), norm_fac=args.train_dataset.norm_fac)
                    losses['loss_p'] = loss_p.detach().item()
                    loss = loss + args.lambda_momentum * loss_p
                
                # Deform loss is an MPM time-stepping residual: needs >=3 output frames
                # (frame_interval=2 -> end_t = frames-2 must be >=1). With output_frames=1 it would
                # build a tensor with negative dim and crash -> skip (structurally undefined here).
                if 'vol' in batch and args.lambda_deform > 0. and pred_sample.shape[1] >= 3:
                    pred_sample_mpm = pred_sample
                    vol_data = batch['vol']
                    F_data = batch['F']
                    C_data = batch['C']
                    if 'is_mpm' in batch:
                        is_mpm_mask = batch['is_mpm']
                        pred_sample_mpm = pred_sample[is_mpm_mask]
                        vol_data = vol_data[is_mpm_mask]
                        F_data = F_data[is_mpm_mask]
                        C_data = C_data[is_mpm_mask]
                    loss_F = loss_deform(x=pred_sample_mpm.clamp(min=-2.2, max=2.2), vol=vol_data, F=F_data,
                        C=C_data, frame_interval=2, norm_fac=args.train_dataset.norm_fac) if vol_data.shape[0] > 0 else torch.tensor(0.0, device=pred_sample.device)
                    losses['loss_deform'] = loss_F.detach().item()
                    loss = loss + args.lambda_deform * loss_F

                # single-frame boundary loss_F: output=1 下原多帧 loss_F(>=3 帧)失效;用 input 末帧→pred
                # 一步前向差分速度推进 GT F 一步、比 GT F_next。开关 single_frame_deform 显式控制(默认 off,
                # 不偷改 lambda_deform 在 output=1 的语义)。x_t=input 末帧(常数锚), 梯度只经 x_next=pred。
                elif args.get('single_frame_deform', False) and 'vol' in batch and args.lambda_deform > 0. \
                        and pred_sample.shape[1] == 1 and 'F_src_last' in batch:
                    loss_F = loss_deform.forward_single_step(
                        x_t=cond_points_src[:, -1].clamp(min=-2.2, max=2.2),
                        x_next=pred_sample[:, 0].clamp(min=-2.2, max=2.2),
                        vol=batch['vol'], F_t=batch['F_src_last'], F_next=batch['F'][:, 0],
                        C_t=batch['C_src_last'], frame_interval=2, norm_fac=args.train_dataset.norm_fac)
                    losses['loss_deform'] = loss_F.detach().item()
                    loss = loss + args.lambda_deform * loss_F

                if args.model_config.floor_cond and args.lambda_floor > 0:
                    floor_height = batch['floor_height'].reshape(bsz, 1, 1) # (B, 1, 1)
                    sample_min_height = torch.amin(latents[..., 1], dim=(1, 2)).reshape(bsz, 1, 1)
                    floor_height = torch.minimum(floor_height, sample_min_height)
                    loss_floor = (torch.relu(floor_height - pred_sample[..., 1]) ** 2).mean()
                    losses['loss_floor'] = loss_floor.detach().item()
                    loss += args.lambda_floor * loss_floor

                knn_idx = None
                if args.lambda_laplacian > 0.0 or args.lambda_edge > 0.0:
                    knn_idx = build_knn_indices(batch['points_src'], args.laplacian_k)

                if args.lambda_laplacian > 0.0:
                    loss_laplacian = laplacian_deformation_loss(pred_sample, batch['points_tgt'], knn_idx)
                    losses['loss_laplacian'] = loss_laplacian.detach().item()
                    loss = loss + args.lambda_laplacian * loss_laplacian

                if args.lambda_collision > 0.0:
                    floor_h = batch['floor_height'] if 'floor_height' in batch else None
                    loss_collision = collision_loss(pred_sample, floor_h, margin=args.collision_margin)
                    losses['loss_collision'] = loss_collision.detach().item()
                    loss = loss + args.lambda_collision * loss_collision

                if args.lambda_edge > 0.0:
                    loss_edge = edge_length_regularization(pred_sample, batch['points_tgt'], knn_idx)
                    losses['loss_edge'] = loss_edge.detach().item()
                    loss = loss + args.lambda_edge * loss_edge

                # ---- 1b: multi-step rollout training (DAgger / optional BPTT) ----
                # Continue from the chunk-0 prediction, feed predictions back as conditioning EXACTLY
                # like eval.py rollout (init_pc = last pred chunk; start_vel = pred[:,1]-prev[:,-1]),
                # and add MSE(+vel) loss on chunks 1..K-1. chunk 0 above keeps its full loss bundle.
                # rollout_unroll_steps==1 -> skipped entirely (== run23). bptt=False detaches the
                # fed-back chunk so the model trains on its own real errors without backprop-through-time.
                roll_w = args.rollout_loss_weight
                if args.rollout_warmup_steps > 0:
                    roll_w = roll_w * min(1.0, global_step / args.rollout_warmup_steps)
                # curriculum over the unroll horizon K (None -> fixed rollout_unroll_steps).
                # dataset reserved roll-GT for k_max chunks; here we use only the first cur_K-1.
                cur_K = cur_stage[0] if curriculum else args.rollout_unroll_steps
                if (not args.use_diffusion) and cur_K > 1 and 'points_tgt_roll' in batch and roll_w > 0:
                    roll_tgt = batch['points_tgt_roll'][:, :(cur_K - 1) * OUTPUT_FRAMES]   # (B, (cur_K-1)*F, N, 3)
                    timesteps0 = torch.zeros((bsz,), device=latents.device, dtype=torch.long)
                    prev_init = cond_points_src                              # conditioning of chunk 0
                    init_pc = pred_sample if args.rollout_bptt else pred_sample.detach()
                    step_start_vel = init_pc[:, 1, :, :] - prev_init[:, -1, :, :]
                    roll_loss = 0.0
                    for k in range(1, cur_K):
                        tgt_k = roll_tgt[:, (k - 1) * OUTPUT_FRAMES:k * OUTPUT_FRAMES]
                        # pure DAgger: feed back the model's own clean prediction, no extra input noise.
                        model_input_k = init_pc[:, -1:, :, :].repeat(1, OUTPUT_FRAMES, 1, 1)
                        pred_k = model(model_input_k, timesteps0, init_pc, batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1], batch['drag_point'], batch['floor_height'], batch['gravity'], batch['base_drag_coeff'], y=None if 'mat_type' not in batch else batch['mat_type'], null_emb=null_emb, start_vel=step_start_vel, points_rest=batch.get('points_rest', None))
                        loss_k = F.mse_loss(pred_k.float(), tgt_k.float())
                        if args.lambda_vel > 0.:
                            tv = tgt_k[:, 1:] - tgt_k[:, :-1]
                            pv = pred_k[:, 1:] - pred_k[:, :-1]
                            loss_k = loss_k + args.lambda_vel * F.mse_loss(pv.float(), tv.float())
                        roll_loss = roll_loss + loss_k
                        prev_init = init_pc
                        nxt = pred_k if args.rollout_bptt else pred_k.detach()
                        step_start_vel = nxt[:, 1, :, :] - prev_init[:, -1, :, :]
                        init_pc = nxt
                    roll_loss = roll_loss / (cur_K - 1)
                    losses['roll'] = roll_loss.detach().item()
                    loss = loss + roll_w * roll_loss
  

  
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if accelerator.is_main_process:
                    step_losses = {
                        "loss": loss.detach().item(),
                        "loss_xyz": losses.get("xyz", 0.0),
                        "loss_mask": losses.get("mask", 0.0),
                        "loss_vel": losses.get("loss_vel", 0.0),
                        "loss_p": losses.get("loss_p", 0.0),
                        "loss_F": losses.get("loss_deform", 0.0),
                        "loss_floor": losses.get("loss_floor", 0.0),
                        "loss_laplacian": losses.get("loss_laplacian", 0.0),
                        "loss_collision": losses.get("loss_collision", 0.0),
                        "loss_edge": losses.get("loss_edge", 0.0),
                    }
                    loss_rows.append([step_losses.get(h, 0.0) if h != "step" else global_step for h in loss_csv_header])

                    for key, value in step_losses.items():
                        if key not in loss_history:
                            loss_history[key] = {"steps": [], "values": []}
                        loss_history[key]["steps"].append(global_step)
                        loss_history[key]["values"].append(float(value))

                    if global_step % 500 == 0:
                        if not os.path.exists(loss_csv_path):
                            with open(loss_csv_path, "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(loss_csv_header)
                        if loss_rows:
                            with open(loss_csv_path, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerows(loss_rows)
                            loss_rows.clear()

                        plot_keys = [
                            "loss",
                            "loss_xyz",
                            "loss_mask",
                            "loss_vel",
                            "loss_p",
                            "loss_F",
                            "loss_floor",
                            "loss_laplacian",
                            "loss_collision",
                            "loss_edge",
                        ]
                        plot_window = 15000   # 曲线始终只显示最近 15000 step,避免长程压缩看不清近期
                        cutoff = global_step - plot_window
                        for key in plot_keys:
                            series = loss_history.get(key)
                            if series is None:
                                continue
                            if not series["steps"]:
                                continue
                            # 截取最近 plot_window 个 step(steps 单调递增)
                            win_steps = [s for s in series["steps"] if s >= cutoff]
                            win_values = series["values"][len(series["steps"]) - len(win_steps):]
                            if not win_steps:
                                continue
                            plt.figure(figsize=(8, 4))
                            plt.plot(win_steps, win_values, linewidth=1.5)
                            plt.xlabel("step")
                            plt.ylabel(key)
                            plt.title(f"{key} curve")
                            plt.tight_layout()
                            plot_path = os.path.join(
                                args.output_dir, f"{key}_curve_step_{global_step:06d}.png"
                            )
                            plt.savefig(plot_path, dpi=150)
                            plt.close()

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
                if global_step % cfg.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        model.eval()
                        eval_scheduler = DDIMScheduler.from_config(noise_scheduler.config) if args.use_diffusion else None
                        pipeline = TrajPipeline(model=accelerator.unwrap_model(model), scheduler=eval_scheduler)
                        logger.info(
                            f"Running validation... \n."
                        )
                        seen_models = set()
                        for i, (batch, _) in enumerate(val_dataloader):
                            with torch.autocast("cuda"):
                                gs = [1.0] if args.condition_drop_rate == 0 else [1.0, 2.0, 3.0]
                                for guidance_scale in gs:
                                    current_input = batch['points_src'].to(accelerator.device)
                                    rollout_chunks = [current_input]
                                    prev_chunk = current_input

                                    for step_idx in range(ROLLOUT_STEPS):
                                        if step_idx == 0:
                                            step_start_vel = batch.get('start_vel', None)
                                            if step_start_vel is not None:
                                                step_start_vel = step_start_vel.to(accelerator.device)
                                        elif INPUT_FRAMES == 1:
                                            # input=1 消融:窗口只剩单帧,无法窗口内差分。用跨 rollout step
                                            # 后向差分(current 帧 − 上一步窗口帧),与训练 causal_start_vel 一致。
                                            step_start_vel = current_input[:, -1, :, :] - prev_chunk[:, -1, :, :]
                                        elif OUTPUT_FRAMES >= 2:
                                            # original chunked-feedback formula (preserves existing arms exactly)
                                            step_start_vel = current_input[:, 1, :, :] - prev_chunk[:, -1, :, :]
                                        else:
                                            # single-frame autoregression: model adds start_vel to the FIRST
                                            # window token and was trained on velocity AT the first frame.
                                            # Feed velocity at current_input[:,0] (forward diff) -> matches
                                            # training scale/frame, not the tail velocity.
                                            step_start_vel = current_input[:, 1, :, :] - current_input[:, 0, :, :]

                                        pred_chunk = pipeline(
                                            current_input,
                                            batch['force'],
                                            batch['E'],
                                            batch['nu'],
                                            batch['mask'][..., :1],
                                            batch['drag_point'],
                                            batch['floor_height'],
                                            batch['gravity'],
                                            batch['base_drag_coeff'],
                                            start_vel=step_start_vel,
                                            points_rest=batch.get('points_rest', None),
                                            y=None if 'mat_type' not in batch else batch['mat_type'],
                                            device=accelerator.device,
                                            batch_size=current_input.shape[0],
                                            generator=torch.Generator().manual_seed(args.seed),
                                            n_frames=OUTPUT_FRAMES,
                                            guidance_scale=guidance_scale,
                                        )
                                        rollout_chunks.append(pred_chunk)

                                        prev_chunk = current_input
                                        # slide the input window: drop oldest, append new prediction,
                                        # keep last INPUT_FRAMES. output==input -> equals the old replace.
                                        current_input = torch.cat([current_input, pred_chunk], dim=1)[:, -INPUT_FRAMES:]

                                    output = torch.cat(rollout_chunks, dim=1).cpu().numpy()
                                    tgt = build_raw_reference(batch, args.train_dataset, output.shape[1]).cpu().numpy()
                                    save_dir = os.path.join(vis_dir, f'{global_step:06d}')
                                    os.makedirs(save_dir, exist_ok=True)
                                    for j in range(output.shape[0]):
                                        model_name = batch['model'][j]
                                        if model_name in seen_models:
                                            continue
                                        seen_models.add(model_name)
                                        save_pointcloud_video(output[j:j+1].squeeze(), tgt[j:j+1].squeeze(), os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j}_{guidance_scale}.gif'),
                                            drag_mask=batch['mask'][j:j+1, 0, :, 0].cpu().numpy().squeeze(), vis_flag=args.train_dataset.dataset_path)
                                        # pred_name = f'{i*batch["points_src"].shape[0]+j}_pred.json'
                                        # gt_name = f'{i*batch["points_src"].shape[0]+j}_gt.json'
                                        # save_pointcloud_json(output[j:j+1].squeeze(), os.path.join(save_dir, pred_name))
                                        # save_pointcloud_json(tgt[j:j+1].squeeze(), os.path.join(save_dir, gt_name))
                                        # save_threejs_html(pred_name, gt_name, os.path.join(save_dir, f'{j}.html'))
                                torch.cuda.empty_cache()
                        model.train()

            logs = losses
            logs.update({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            if curriculum and current_stage(global_step, curriculum, args.rollout_unroll_steps) != cur_stage:
                break   # stage changed mid-epoch -> exit inner loop so the loader is rebuilt

    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unet.to(torch.float32)
    if log_file is not None:
        log_file.flush()
        log_file.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
