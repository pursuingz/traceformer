from diffusers import DDPMScheduler, DDIMScheduler
from dataset.traj_dataset import TrajDataset
from model.mdm_dit import MDM_DiT
from model.spacetime import MDM_ST
import sys
from options import TrainingConfig, TestingConfig
from omegaconf import OmegaConf
from pipeline_traj import TrajPipeline
import torch
from safetensors.torch import load_file
import argparse
import os
import numpy as np
import h5py
from utils.physics import loss_momentum, DeformLoss
import torch.nn.functional as F
from tqdm import tqdm
from utils.visualization import save_pointcloud_video, save_pointcloud_json, save_threejs_html, generate_html_from_exts

INPUT_FRAMES = 5
OUTPUT_FRAMES = 5
ROLLOUT_STEPS = 4

def create_model(args):
    args.train_dataset.input_frames = INPUT_FRAMES
    args.train_dataset.output_frames = OUTPUT_FRAMES
    args.model_config.cond_frames = INPUT_FRAMES
    model = MDM_ST(args.pc_size, OUTPUT_FRAMES, n_feats=3, model_config=args.model_config)
    return model


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


def chamfer_distance(pred, gt):
    """双向 Chamfer(L2,置换不变)。pred/gt: (..., P, 3) -> 标量(对所有帧/样本取均值)。"""
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1]).float()
    gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1]).float()
    d = torch.cdist(pred, gt)                       # (BT, P, P) 欧氏距离
    cd = d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)
    return cd.mean()

loss_deform = DeformLoss().to('cuda')
def main(args):
    val_dataset = TrajDataset('test', args.train_dataset)   # 单独 eval 只评最后 4 个干净 held-out
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    device = 'cuda'
    model = create_model(args).to(device)
    ckpt = load_file(args.resume, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval().requires_grad_(False)
    model = torch.compile(model)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False) if args.use_diffusion else None
    pipeline = TrajPipeline(model=model, scheduler=noise_scheduler)

    total_loss_xyz = 0.0          # 首段 MSE(头 5 帧,保留向后兼容)
    total_mse_full = 0.0          # 全 rollout MSE(25 帧)
    total_chamfer = 0.0           # 全 rollout 平均 Chamfer
    total_mse_step = torch.zeros(ROLLOUT_STEPS)   # 每个 rollout step 的 MSE(误差累积曲线)
    total_loss_F = 0.0
    total_loss_F_gt = 0.0
    n_batches = 0
    seen_models = set()
    for i, (batch, _) in enumerate(tqdm(val_dataloader)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current_input = batch['points_src'].to(device)
            rollout_chunks = [current_input]
            prev_chunk = current_input
            for step_idx in range(ROLLOUT_STEPS):
                if step_idx == 0:
                    step_start_vel = batch.get('start_vel', None)
                    if step_start_vel is not None:
                        step_start_vel = step_start_vel.to(device)
                else:
                    step_start_vel = current_input[:, 1, :, :] - prev_chunk[:, -1, :, :]

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
                    y=None if args.model_config.get('num_mat', 0) == 0 else batch['mat_type'],
                    device=device,
                    batch_size=args.eval_batch_size,
                    generator=torch.Generator().manual_seed(args.seed),
                    n_frames=OUTPUT_FRAMES,
                    num_inference_steps=args.num_inference_steps,
                )
                rollout_chunks.append(pred_chunk)
                prev_chunk = current_input
                current_input = pred_chunk

            first_pred = rollout_chunks[1]
            output = torch.cat(rollout_chunks, dim=1)
            if 'vol' in batch:
                loss_F = loss_deform(x=first_pred.clamp(min=-2.2, max=2.2), vol=batch['vol'].to(device), F=batch['F'].to(device),
                        C=batch['C'].to(device), frame_interval=2, norm_fac=args.train_dataset.norm_fac)
                loss_F_gt = loss_deform(x=batch['points_tgt'].to(device), vol=batch['vol'].to(device), F=batch['F'].to(device),
                        C=batch['C'].to(device), frame_interval=2, norm_fac=args.train_dataset.norm_fac)
                total_loss_F += loss_F.item()
                total_loss_F_gt += loss_F_gt.item()
            total_loss_xyz += F.mse_loss(first_pred, batch['points_tgt'].to(device)).item()

            gt_vis = build_raw_reference(batch, args.train_dataset, output.shape[1])

            # ---- 全 rollout 量化指标(output 与 gt_vis 均为 25 帧,同归一化空间)----
            gt_vis_dev = gt_vis.to(device)
            out_f = output.float()
            gt_f = gt_vis_dev.float()
            total_mse_full += F.mse_loss(out_f, gt_f).item()
            total_chamfer += chamfer_distance(out_f, gt_f).item()
            for s in range(ROLLOUT_STEPS):
                lo = (s + 1) * OUTPUT_FRAMES       # 跳过 [0:5] 输入段,只评 4 个预测段
                hi = lo + OUTPUT_FRAMES
                total_mse_step[s] += F.mse_loss(out_f[:, lo:hi], gt_f[:, lo:hi]).item()
            n_batches += 1

            output = output.cpu().numpy()
            tgt = gt_vis.cpu().numpy()
            vis_dir = args.vis_dir
            save_dir = os.path.join(vis_dir, f'test_100_{args.num_inference_steps}steps_nips_debug')
            os.makedirs(save_dir, exist_ok=True)
            for j in range(output.shape[0]):
                model_name = batch['model'][j]
                if model_name in seen_models:
                    continue
                seen_models.add(model_name)
                save_pointcloud_video(output[j:j+1].squeeze(), tgt[j:j+1].squeeze(), os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j:03d}_{batch["E"][j].item():03f}_{batch["nu"][j].item():03f}.gif'), drag_mask=batch['mask'][j:j+1, 0, :, 0].cpu().numpy().squeeze(), vis_flag='objaverse')
                np.save(os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j}_{batch["E"][j].item():03f}_{batch["nu"][j].item():03f}.npy'), output[j:j+1].squeeze())
                np.save(os.path.join(save_dir, f'{batch["model"][j]}.npy'), output[j:j+1].squeeze())
            torch.cuda.empty_cache()
    generate_html_from_exts(save_dir, os.path.join(save_dir, f'visualize.html'), 'gif')

    n = max(n_batches, 1)
    per_step = (total_mse_step / n).tolist()
    print(f'===== eval metrics (mean over {n_batches} batches) =====')
    print(f'  MSE first-chunk : {total_loss_xyz / n:.6e}')
    print(f'  MSE full-rollout: {total_mse_full / n:.6e}')
    print(f'  Chamfer  (mean) : {total_chamfer / n:.6e}')
    print('  MSE per step    : ' + ', '.join(f'step{k+1}={v:.6e}' for k, v in enumerate(per_step)))
    print(f'  loss_F (pred)   : {total_loss_F / n:.6e}')
    print(f'  loss_F (gt ref) : {total_loss_F_gt / n:.6e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)