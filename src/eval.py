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

try:
    from scipy.spatial import ConvexHull
    _HAS_HULL = True
except Exception:
    _HAS_HULL = False


def cloud_volume(pc):
    """点云体积。优先凸包真实体积;无 scipy 时退化为协方差行列式 sqrt(det Σ)(~length^3)。
    退化/共面导致凸包失败时返回 None,该帧不计入。"""
    if _HAS_HULL:
        try:
            return float(ConvexHull(pc).volume)
        except Exception:
            return None
    c = pc - pc.mean(axis=0)
    cov = (c.T @ c) / pc.shape[0]
    return float(np.sqrt(max(float(np.linalg.det(cov)), 0.0)))


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
    valid_counts = []          # 每样本 GT 真实覆盖的帧数(< target_frames 即起点越界,用于 partial-rollout)
    full_valid = True
    interval = dataset_cfg.get('n_frames_interval', 1)
    for j, model_name in enumerate(batch['model']):
        h5_path = os.path.join(dataset_cfg.dataset_path, model_name)
        with h5py.File(h5_path, 'r') as model_metas:
            model_pcls = torch.from_numpy(np.array(model_metas['x']))

        point_indices = batch['point_indices'][j].cpu().numpy()
        start = int(batch['start_idx'][j]) if 'start_idx' in batch else 0
        # 与 rollout 输出逐帧对齐:从 start_idx 起、按 n_frames_interval 取帧
        sel = start + np.arange(target_frames) * interval
        valid_counts.append(int(np.sum(sel < model_pcls.shape[0])))   # GT 覆盖到第几帧
        if sel[-1] >= model_pcls.shape[0]:
            full_valid = False                       # GT 不够覆盖整段 rollout,该窗口不计入 rollout 指标
        sel = np.clip(sel, 0, model_pcls.shape[0] - 1)
        raw_seq = model_pcls[sel][:, point_indices].float()
        raw_seq = (raw_seq - dataset_cfg.norm_fac) / 2
        raw_sequences.append(raw_seq)

    return torch.stack(raw_sequences, dim=0), full_valid, valid_counts


def chamfer_distance(pred, gt):
    """双向 Chamfer(L2,置换不变)。pred/gt: (..., P, 3) -> 标量(对所有帧/样本取均值)。"""
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1]).float()
    gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1]).float()
    d = torch.cdist(pred, gt)                       # (BT, P, P) 欧氏距离
    cd = d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)
    return cd.mean()

loss_deform = DeformLoss().to('cuda')
def main(args):
    # Prediction granularity (new axis): config-driven output_frames (default 5 = run23). Must be set
    # before create_model so the model is built with the same frame count as the checkpoint. ROLLOUT_STEPS
    # is derived from a fixed ~20-frame horizon: output=5 -> 4 (unchanged), output=1 -> 20 (frame-by-frame).
    global OUTPUT_FRAMES, ROLLOUT_STEPS
    OUTPUT_FRAMES = args.get('output_frames', 5)
    ROLLOUT_HORIZON = 20
    ROLLOUT_STEPS = -(-ROLLOUT_HORIZON // OUTPUT_FRAMES)
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
    total_mse_step = torch.zeros(ROLLOUT_STEPS)   # 每个 rollout step 的 MSE(chunk 桶, 跨粒度不可比)
    ABS_FRAMES = [5, 10, 15, 20]                   # 绝对帧索引(跨预测粒度可比的累积曲线)
    total_mse_abs = torch.zeros(len(ABS_FRAMES))
    cnt_mse_abs = torch.zeros(len(ABS_FRAMES))
    total_loss_F = 0.0
    total_loss_F_gt = 0.0
    n_batches = 0
    n_full = 0                    # GT 帧数足够覆盖整段 rollout 的窗口数(rollout 指标的分母)
    # ---- 物理合理性指标 ----
    total_vel_mse = 0.0           # 速度误差(帧差, vs GT)
    total_acc_mse = 0.0           # 加速度误差(二阶差, vs GT)
    total_vol_err = 0.0           # 体积相对误差 |Vp-Vg|/Vg(vs GT)
    n_vol = 0                     # 成功算出体积的帧数(凸包可能退化失败)
    total_vol_drift = 0.0         # 预测体积自漂移 |Vp(t)-Vp(0)|/Vp(0)
    n_vol_drift = 0
    total_floor_rate = 0.0        # 地面穿透率(预测帧, 占全部点比例)
    total_floor_depth = 0.0       # 平均穿透深度(归一化单位)
    total_floor_rate_gt = 0.0     # GT 参考(应≈0)
    n_floor_gt = 0
    # ---- 非0起点 partial-rollout(检验 (01) 固定窗口优势是否只是 start=0 评测对齐 artifact)----
    # full-rollout 只测 start=0;(01) stride-5 网格恰好每 epoch 命中 start=0,(11) 随机起点几乎抽不到。
    # 这里对每个窗口按 GT 实际覆盖的 chunk 数累计 per-step,并单列「仅 start>0」,区分:
    #   (11) 在 start>0 上追平/反超 (01) → 优势是评测对齐 artifact;(01) 仍全面赢 → 真链质量更好。
    total_mse_step_all = torch.zeros(ROLLOUT_STEPS)   # 所有起点:GT 覆盖该 step 的 chunk 才计入
    cnt_step_all = torch.zeros(ROLLOUT_STEPS)
    total_mse_step_nz = torch.zeros(ROLLOUT_STEPS)    # 仅 start>0
    cnt_step_nz = torch.zeros(ROLLOUT_STEPS)
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
                elif OUTPUT_FRAMES >= 2:
                    # original chunked-feedback formula (preserves existing arms exactly)
                    step_start_vel = current_input[:, 1, :, :] - prev_chunk[:, -1, :, :]
                else:
                    # single-frame autoregression: boundary velocity at the tail of the sliding window
                    step_start_vel = current_input[:, -1, :, :] - current_input[:, -2, :, :]

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
                    y=None if args.model_config.get('num_mat', 0) == 0 else batch['mat_type'],
                    device=device,
                    batch_size=args.eval_batch_size,
                    generator=torch.Generator().manual_seed(args.seed),
                    n_frames=OUTPUT_FRAMES,
                    num_inference_steps=args.num_inference_steps,
                )
                rollout_chunks.append(pred_chunk)
                prev_chunk = current_input
                # slide input window: drop oldest, append prediction, keep last INPUT_FRAMES.
                # output==input -> last INPUT_FRAMES = pred_chunk (== old replace behavior).
                current_input = torch.cat([current_input, pred_chunk], dim=1)[:, -INPUT_FRAMES:]

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

            # ---- 地面穿透(只看预测帧, 不需要 GT; Y 轴朝上, gravity=[0,-1,0]) ----
            floor_h = batch['floor_height'].to(device).view(-1, 1, 1)   # (B,1,1) 归一化空间
            y_pred = output[:, INPUT_FRAMES:, :, 1]                     # (B, pred_frames, N)
            total_floor_rate += (y_pred < floor_h).float().mean().item()
            total_floor_depth += torch.clamp(floor_h - y_pred, min=0).mean().item()

            gt_vis, full_valid, valid_counts = build_raw_reference(batch, args.train_dataset, output.shape[1])

            n_batches += 1
            # ---- 全 rollout 指标:仅在 GT 帧数足够覆盖整段 rollout 的窗口上算 ----
            # start_idx>0 的窗口 rollout 会越过序列末尾,GT 不存在,计入只会污染指标。
            if full_valid:
                gt_vis_dev = gt_vis.to(device)
                out_f = output.float()
                gt_f = gt_vis_dev.float()
                total_mse_full += F.mse_loss(out_f, gt_f).item()
                total_chamfer += chamfer_distance(out_f, gt_f).item()
                for s in range(ROLLOUT_STEPS):
                    lo = (s + 1) * OUTPUT_FRAMES   # 跳过 [0:5] 输入段,只评 4 个预测段
                    hi = lo + OUTPUT_FRAMES
                    total_mse_step[s] += F.mse_loss(out_f[:, lo:hi], gt_f[:, lo:hi]).item()

                # ---- per-绝对帧 MSE(跨预测粒度可比:不依赖 chunk 大小,直接对 run23)----
                # output 索引 0..4 = 输入段,5.. = 预测;取绝对帧 {5,10,15,20} 的单帧 MSE 作累积曲线。
                for ai, af in enumerate(ABS_FRAMES):
                    if af < out_f.shape[1]:
                        total_mse_abs[ai] += F.mse_loss(out_f[:, af], gt_f[:, af]).item()
                        cnt_mse_abs[ai] += 1

                # ---- 速度 / 加速度误差(归一化空间, vs GT) ----
                v_pred = out_f[:, 1:] - out_f[:, :-1]
                v_gt = gt_f[:, 1:] - gt_f[:, :-1]
                total_vel_mse += F.mse_loss(v_pred, v_gt).item()
                total_acc_mse += F.mse_loss(v_pred[:, 1:] - v_pred[:, :-1],
                                            v_gt[:, 1:] - v_gt[:, :-1]).item()

                # ---- 体积保持(反归一化到真实尺度后算体积) ----
                norm_fac = args.train_dataset.norm_fac
                out_real = (out_f * 2 + norm_fac).cpu().numpy()
                gt_real = (gt_f * 2 + norm_fac).cpu().numpy()
                for b in range(out_real.shape[0]):
                    vp = [cloud_volume(out_real[b, t]) for t in range(out_real.shape[1])]
                    vg = [cloud_volume(gt_real[b, t]) for t in range(gt_real.shape[1])]
                    for t in range(len(vp)):
                        if vp[t] is not None and vg[t] is not None and vg[t] > 1e-9:
                            total_vol_err += abs(vp[t] - vg[t]) / vg[t]
                            n_vol += 1
                    if vp[0] is not None and vp[0] > 1e-9:
                        for t in range(1, len(vp)):
                            if vp[t] is not None:
                                total_vol_drift += abs(vp[t] - vp[0]) / vp[0]
                                n_vol_drift += 1

                # ---- GT 地面穿透参考(应≈0) ----
                y_gt = gt_f[:, INPUT_FRAMES:, :, 1]
                total_floor_rate_gt += (y_gt < floor_h).float().mean().item()
                n_floor_gt += 1
                n_full += 1

            # ---- 非0起点 partial-rollout per-step:对每个窗口按 GT 覆盖到的 chunk 数累计 ----
            # 对 start=0 窗口与上面 per-step 等价;对 start>0 窗口补上原被 full_valid 排除的早段 rollout。
            out_dev = output.float()
            gtv_dev = gt_vis.to(device).float()
            for b in range(out_dev.shape[0]):
                nvalid = valid_counts[b]
                start_b = int(batch['start_idx'][b]) if 'start_idx' in batch else 0
                for s in range(ROLLOUT_STEPS):
                    lo = (s + 1) * OUTPUT_FRAMES
                    hi = lo + OUTPUT_FRAMES
                    if hi <= nvalid:                          # 该 chunk 的 5 帧 GT 全部存在才计入
                        m = F.mse_loss(out_dev[b, lo:hi], gtv_dev[b, lo:hi]).item()
                        total_mse_step_all[s] += m
                        cnt_step_all[s] += 1
                        if start_b > 0:
                            total_mse_step_nz[s] += m
                            cnt_step_nz[s] += 1

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
    nf = max(n_full, 1)
    per_step = (total_mse_step / nf).tolist()
    print('===== eval metrics =====')
    print(f'  windows: {n_batches} total, {n_full} full-horizon (rollout 指标分母)')
    print(f'  MSE first-chunk : {total_loss_xyz / n:.6e}   (全 {n_batches} 窗口, 逐样本对齐)')
    print(f'  MSE full-rollout: {total_mse_full / nf:.6e}   (仅 {n_full} 全程窗口)')
    print(f'  Chamfer  (mean) : {total_chamfer / nf:.6e}')
    print('  MSE per step    : ' + ', '.join(f'step{k+1}={v:.6e}' for k, v in enumerate(per_step)))
    # ---- per-绝对帧 MSE(跨预测粒度可比, output=1 vs output=5 用这行对比, 不用上面的 chunk per-step)----
    per_abs = (total_mse_abs / torch.clamp(cnt_mse_abs, min=1)).tolist()
    print('  MSE per abs-frame: ' + ', '.join(f'f{ABS_FRAMES[k]}={v:.6e}(n={int(cnt_mse_abs[k])})' for k, v in enumerate(per_abs)))
    # ---- 非0起点 partial-rollout(评测对齐 artifact 判据)----
    safe_step = lambda tot, cnt: (tot / torch.clamp(cnt, min=1)).tolist()
    per_step_all = safe_step(total_mse_step_all, cnt_step_all)
    per_step_nz = safe_step(total_mse_step_nz, cnt_step_nz)
    print('  per-step(所有起点): ' + ', '.join(f'step{k+1}={v:.6e}(n={int(cnt_step_all[k])})' for k, v in enumerate(per_step_all)))
    print('  per-step(仅start>0): ' + ', '.join(f'step{k+1}={v:.6e}(n={int(cnt_step_nz[k])})' for k, v in enumerate(per_step_nz)))
    print(f'  loss_F (pred)   : {total_loss_F / n:.6e}')
    print(f'  loss_F (gt ref) : {total_loss_F_gt / n:.6e}')
    nv = max(n_vol, 1)
    nvd = max(n_vol_drift, 1)
    nfg = max(n_floor_gt, 1)
    print('  --- 物理合理性 ---')
    print(f'  速度误差 vMSE   : {total_vel_mse / nf:.6e}   (帧差, vs GT, {n_full} 全程窗口)')
    print(f'  加速度误差 aMSE : {total_acc_mse / nf:.6e}   (二阶差, vs GT)')
    print(f'  体积相对误差    : {total_vol_err / nv * 100:.3f}%   (|Vp-Vg|/Vg, {n_vol} 帧, {"凸包" if _HAS_HULL else "协方差近似"})')
    print(f'  体积自漂移      : {total_vol_drift / nvd * 100:.3f}%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)')
    print(f'  地面穿透率      : {total_floor_rate / n * 100:.3f}%   (预测帧占全部点比例, 全 {n_batches} 窗口)')
    print(f'  地面穿透深度    : {total_floor_depth / n:.6e}   (归一化单位)')
    print(f'  地面穿透率(GT)  : {total_floor_rate_gt / nfg * 100:.3f}%   (参考, 应≈0)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
