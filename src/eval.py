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
import io
import math
import os
import re
import time
import numpy as np
import h5py
from utils.physics import loss_momentum, DeformLoss
from utils.eval_metrics import per_window_metrics
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


class _Tee:
    """把 print 同时写到终端和缓冲——eval 指标块原样落盘,不改动任何现有 print 行。"""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()

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
def main(args, config_path=None):
    # Prediction granularity (new axis): config-driven output_frames (default 5 = run23). Must be set
    # before create_model so the model is built with the same frame count as the checkpoint. ROLLOUT_STEPS
    # is derived from a fixed ~20-frame horizon: output=5 -> 4 (unchanged), output=1 -> 20 (frame-by-frame).
    global OUTPUT_FRAMES, ROLLOUT_STEPS, INPUT_FRAMES
    # 时间感受野轴:input_frames config 驱动(默认 5 = 所有现有臂,无此键 → 字节不变)。input=1 = 单帧输入消融。
    INPUT_FRAMES = args.get('input_frames', 5)
    args.train_dataset.input_frames = INPUT_FRAMES   # 单一真源:top-level 同步给 dataset(须在下方建库前)
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
    # 逐帧体积自漂移的绝对帧(0-indexed;24=末帧,对应口径"第25帧")。不对帧平均,每帧单独出。
    VOL_DRIFT_FRAMES = [10, 15, 20, 24]
    vol_drift_abs_p = torch.zeros(len(VOL_DRIFT_FRAMES))   # 逐帧 pred 自漂移 |Vp(af)-Vp(0)|/Vp(0)
    vol_drift_abs_g = torch.zeros(len(VOL_DRIFT_FRAMES))   # 逐帧 GT  自漂移 |Vg(af)-Vg(0)|/Vg(0)
    cnt_vol_drift_abs = torch.zeros(len(VOL_DRIFT_FRAMES))
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
    total_vol_drift_gt = 0.0      # GT 体积自漂移 |Vg(t)-Vg(0)|/Vg(0)(基线)
    n_vol_drift_gt = 0            # 度量是凸包体积:弹性体(nu=0.4)大形变下 GT 本身凸包体积就漂,
                                  #   "自漂移≈0" 不成立 → 必须减此 GT 基线才能判预测是否"额外"漂
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
    per_model_rows = []   # opt-in(per_model_csv):每全程窗口一行(log10E/full-rollout/体积),供 E 分档分析
    gen_times = []        # 每窗口生成段 wall-clock(仅 rollout 循环,cuda synchronize 包裹;首窗含 compile warmup)
    # ---- 度量层扩展(范式对比;utils/eval_metrics.py;仅全程窗口,只算预测帧)----
    # 全帧 per-frame MSE(→分段 seg-MSE / 几何均值)、FDE、Procrustes 位姿-形状分解、固定边弥散度。
    PARADIGM_FRAMES = (5, 10, 15, 20, 24)
    pm_mse_frame_sum, pm_mse_frame_cnt = {}, {}
    pm_fde_sum, pm_fde_cnt = 0.0, 0
    pm_proc_sum = {af: [0.0] * 4 for af in PARADIGM_FRAMES}
    pm_proc_cnt = {af: 0 for af in PARADIGM_FRAMES}
    pm_edge_sum = {af: [0.0, 0.0] for af in PARADIGM_FRAMES}
    pm_edge_cnt = {af: 0 for af in PARADIGM_FRAMES}
    # ---- per-material 分桶(多材质数据集;单材质数据只有 1 个桶,打印时跳过)----
    # 与上面全局累加器并行累加,现有聚合输出不动;仅全程窗口、仅预测帧,口径与全局一致。
    MAT_NAMES = {0: 'elastic', 1: 'plasticine', 2: 'sand', 3: 'rigid'}
    mm_buckets = {}

    def _mm_bucket(m):
        if m not in mm_buckets:
            mm_buckets[m] = {
                'mse_frame_sum': {}, 'mse_frame_cnt': {}, 'fde_sum': 0.0, 'fde_cnt': 0,
                'proc_sum': {af: [0.0] * 4 for af in PARADIGM_FRAMES},
                'proc_cnt': {af: 0 for af in PARADIGM_FRAMES},
                'edge_sum': {af: [0.0, 0.0] for af in PARADIGM_FRAMES},
                'edge_cnt': {af: 0 for af in PARADIGM_FRAMES},
                'vol_p': torch.zeros(len(VOL_DRIFT_FRAMES)),
                'vol_g': torch.zeros(len(VOL_DRIFT_FRAMES)),
                'vol_cnt': torch.zeros(len(VOL_DRIFT_FRAMES)),
            }
        return mm_buckets[m]
    for i, (batch, _) in enumerate(tqdm(val_dataloader)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current_input = batch['points_src'].to(device)
            rollout_chunks = [current_input]
            prev_chunk = current_input
            torch.cuda.synchronize()
            _gen_t0 = time.perf_counter()
            for step_idx in range(ROLLOUT_STEPS):
                if step_idx == 0:
                    step_start_vel = batch.get('start_vel', None)
                    if step_start_vel is not None:
                        step_start_vel = step_start_vel.to(device)
                elif INPUT_FRAMES == 1:
                    # input=1 消融:窗口只剩单帧,无法窗口内差分。用跨 rollout step 后向差分
                    # (current 帧 − 上一步窗口帧),与训练 causal_start_vel 后向差分语义/scale 一致。
                    step_start_vel = current_input[:, -1, :, :] - prev_chunk[:, -1, :, :]
                elif OUTPUT_FRAMES >= 2:
                    # original chunked-feedback formula (preserves existing arms exactly)
                    step_start_vel = current_input[:, 1, :, :] - prev_chunk[:, -1, :, :]
                else:
                    # single-frame autoregression: model adds start_vel to the FIRST window token
                    # (hidden_states[:,0]) and was trained on the per-frame velocity AT the window's
                    # first frame. So feed velocity at current_input[:,0] (forward diff; frame -1 has
                    # slid off the window) -> matches training scale/frame. NOT the tail velocity.
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
            torch.cuda.synchronize()
            gen_times.append(time.perf_counter() - _gen_t0)

            first_pred = rollout_chunks[1]
            output = torch.cat(rollout_chunks, dim=1)
            # loss_F (DeformLoss) is a multi-frame MPM residual: needs >=3 frames. first_pred has
            # OUTPUT_FRAMES frames -> N/A for output_frames=1 (would crash). Skip and report 0.
            if 'vol' in batch and first_pred.shape[1] >= 3:
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

                # ---- 度量层扩展:全帧 per-frame / FDE / Procrustes 分解 / 固定边弥散度 ----
                # autocast 关闭:SVD / cdist 需全精度;逐窗口(eval_batch_size 实际=1)。
                with torch.autocast("cuda", enabled=False):
                    _pm_win = {}   # 本 batch 逐窗口 metrics,供下方 per-model CSV 长尾列引用
                    for b in range(out_f.shape[0]):
                        pm = per_window_metrics(out_f[b].float(), gt_f[b].float(),
                                                INPUT_FRAMES, PARADIGM_FRAMES)
                        _pm_win[b] = pm
                        # 材质桶(数据无 mat_type 时按 elastic=0 兜底,与 traj_dataset 一致)
                        _mt = int(batch['mat_type'][b].item()) if 'mat_type' in batch else 0
                        mb = _mm_bucket(_mt)
                        for af, v in pm['mse_frame'].items():
                            pm_mse_frame_sum[af] = pm_mse_frame_sum.get(af, 0.0) + v
                            pm_mse_frame_cnt[af] = pm_mse_frame_cnt.get(af, 0) + 1
                            mb['mse_frame_sum'][af] = mb['mse_frame_sum'].get(af, 0.0) + v
                            mb['mse_frame_cnt'][af] = mb['mse_frame_cnt'].get(af, 0) + 1
                        pm_fde_sum += pm['fde']
                        pm_fde_cnt += 1
                        mb['fde_sum'] += pm['fde']
                        mb['fde_cnt'] += 1
                        for af, tup in pm['proc'].items():
                            for j in range(4):
                                pm_proc_sum[af][j] += tup[j]
                                mb['proc_sum'][af][j] += tup[j]
                            pm_proc_cnt[af] += 1
                            mb['proc_cnt'][af] += 1
                        for af, (rp, rg) in pm['edge'].items():
                            pm_edge_sum[af][0] += rp
                            pm_edge_sum[af][1] += rg
                            pm_edge_cnt[af] += 1
                            mb['edge_sum'][af][0] += rp
                            mb['edge_sum'][af][1] += rg
                            mb['edge_cnt'][af] += 1

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
                    # GT 自漂移基线(同口径:vg[0] 为参考,逐帧 |Vg(t)-Vg(0)|/Vg(0))
                    if vg[0] is not None and vg[0] > 1e-9:
                        for t in range(1, len(vg)):
                            if vg[t] is not None:
                                total_vol_drift_gt += abs(vg[t] - vg[0]) / vg[0]
                                n_vol_drift_gt += 1
                    # 逐帧体积自漂移(不平均,核心新指标):VOL_DRIFT_FRAMES 各绝对帧单独累计 pred/GT 自漂移
                    if vp[0] is not None and vp[0] > 1e-9 and vg[0] is not None and vg[0] > 1e-9:
                        _mt_v = int(batch['mat_type'][b].item()) if 'mat_type' in batch else 0
                        mb_v = _mm_bucket(_mt_v)
                        for fi, af in enumerate(VOL_DRIFT_FRAMES):
                            if af < len(vp) and vp[af] is not None and vg[af] is not None:
                                vol_drift_abs_p[fi] += abs(vp[af] - vp[0]) / vp[0]
                                vol_drift_abs_g[fi] += abs(vg[af] - vg[0]) / vg[0]
                                cnt_vol_drift_abs[fi] += 1
                                mb_v['vol_p'][fi] += abs(vp[af] - vp[0]) / vp[0]
                                mb_v['vol_g'][fi] += abs(vg[af] - vg[0]) / vg[0]
                                mb_v['vol_cnt'][fi] += 1
                    # opt-in 按模型日志(B=1 → 一窗一模型):供 E 分档拆解。additive,不改任何聚合指标。
                    if args.get('per_model_csv', None):
                        vrel = [abs(vp[t]-vg[t])/vg[t] for t in range(len(vp)) if vp[t] and vg[t] and vg[t] > 1e-9]
                        dpr  = [abs(vp[t]-vp[0])/vp[0] for t in range(1, len(vp)) if vp[0] and vp[t] and vp[0] > 1e-9]
                        dgt  = [abs(vg[t]-vg[0])/vg[0] for t in range(1, len(vg)) if vg[0] and vg[t] and vg[0] > 1e-9]
                        row = {
                            'model': str(batch['model'][b]),
                            'mat_type': int(batch['mat_type'][b].item()) if 'mat_type' in batch else 0,
                            'log10E': float(batch['E'][b].reshape(-1)[0].item()),
                            'nu': float(batch['nu'][b].reshape(-1)[0].item()),
                            'mse_full': float(F.mse_loss(out_f[b], gt_f[b]).item()),
                            # 长尾定位列(范式对比):fde vs mse_f24 分离"末帧长尾 vs 均方",
                            # resid_f24 = Procrustes 残余(形状轴)→ 坏窗口是位姿飘还是形状糊
                            'fde': float(_pm_win[b]['fde']),
                            'mse_f24': float(_pm_win[b]['mse_frame'].get(24, float('nan'))),
                            'resid_f24': float(_pm_win[b]['proc'].get(24, (float('nan'),) * 4)[3]),
                            'vol_rel': float(np.mean(vrel)) if vrel else float('nan'),
                            'drift_pred': float(np.mean(dpr)) if dpr else float('nan'),
                            'drift_gt': float(np.mean(dgt)) if dgt else float('nan'),
                        }
                        p0ok = vp[0] is not None and vp[0] > 1e-9
                        g0ok = vg[0] is not None and vg[0] > 1e-9
                        for af in VOL_DRIFT_FRAMES:   # 逐帧 pred/GT 自漂移列(供 E 分档 x 帧)
                            row[f'dp_f{af}'] = float(abs(vp[af]-vp[0])/vp[0]) if (p0ok and af < len(vp) and vp[af] is not None) else float('nan')
                            row[f'dg_f{af}'] = float(abs(vg[af]-vg[0])/vg[0]) if (g0ok and af < len(vg) and vg[af] is not None) else float('nan')
                        per_model_rows.append(row)

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
    # opt-in:dump 按模型明细 CSV(*.csv 已 gitignore)。默认关 → 现有 eval 零影响。
    if args.get('per_model_csv', None) and per_model_rows:
        import csv
        # seed=0 保持旧文件名(兼容既有分析脚本);多 seed 评测时各 seed 单独落盘不互相覆盖
        csv_path = f'{args.vis_dir}_per_model.csv' if args.seed == 0 else f'{args.vis_dir}_per_model_seed{args.seed}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as _f:
            w = csv.DictWriter(_f, fieldnames=list(per_model_rows[0].keys()))
            w.writeheader(); w.writerows(per_model_rows)
        print(f'  [per-model] {len(per_model_rows)} 行 -> {csv_path}')
    per_step = (total_mse_step / nf).tolist()
    # ---- 指标块整体捕获落盘(_Tee):终端输出照旧,同时写入 results md;现有 print 行零改动 ----
    _res_buf = io.StringIO()
    _stdout0 = sys.stdout
    sys.stdout = _Tee(_stdout0, _res_buf)
    print('===== eval metrics =====')
    print(f'  windows: {n_batches} total, {n_full} full-horizon (rollout 指标分母)')
    print(f'  seed            : {args.seed}   (扩散采样 generator;确定性臂无影响)')
    # ---- 推理耗时(仅 rollout 生成段;首窗含 torch.compile warmup,单列剔除)----
    if gen_times:
        _ex = gen_times[1:] if len(gen_times) > 1 else gen_times
        _mean_ex = sum(_ex) / len(_ex)
        _pf = ROLLOUT_STEPS * OUTPUT_FRAMES   # 每窗生成的预测帧数
        print(f'  推理耗时(生成段): 总 {sum(gen_times):.1f}s / {len(gen_times)} 窗;'
              f' 均值 {sum(gen_times) / len(gen_times):.3f}s/窗,'
              f' 去首窗 {_mean_ex:.3f}s/窗(剔 compile warmup),'
              f' {_mean_ex / _pf * 1000:.1f}ms/预测帧({_pf} 帧/窗)')
    print(f'  MSE first-chunk : {total_loss_xyz / n:.6e}   (全 {n_batches} 窗口, 逐样本对齐)')
    print(f'  MSE full-rollout: {total_mse_full / nf:.6e}   (仅 {n_full} 全程窗口)')
    print(f'  Chamfer  (mean) : {total_chamfer / nf:.6e}')
    print('  MSE per step    : ' + ', '.join(f'step{k+1}={v:.6e}' for k, v in enumerate(per_step)))
    # ---- per-绝对帧 MSE(跨预测粒度可比, output=1 vs output=5 用这行对比, 不用上面的 chunk per-step)----
    per_abs = (total_mse_abs / torch.clamp(cnt_mse_abs, min=1)).tolist()
    print('  MSE per abs-frame: ' + ', '.join(f'f{ABS_FRAMES[k]}={v:.6e}(n={int(cnt_mse_abs[k])})' for k, v in enumerate(per_abs)))
    # ---- 度量层扩展输出(范式对比口径;仅全程窗口。设计动机见 utils/eval_metrics.py 头注)----
    if pm_mse_frame_cnt:
        _fs = sorted(pm_mse_frame_cnt.keys())
        _pf = {af: pm_mse_frame_sum[af] / pm_mse_frame_cnt[af] for af in _fs}
        print('  --- 全帧 per-frame + 分段/端点(单标量隐含 horizon 权重, 分段呈现) ---')
        print('  MSE per-frame(仅预测帧): ' + ', '.join(f'f{af}={_pf[af]:.6e}' for af in _fs)
              + f'   (n={pm_mse_frame_cnt[_fs[0]]})')
        _seg = lambda lo, hi: (lambda xs: sum(xs) / len(xs) if xs else float('nan'))(
            [_pf[af] for af in _fs if lo <= af <= hi])
        _gm_vals = [_pf[af] for af in _fs if 5 <= af <= 24 and _pf[af] > 0]
        _gm = 10 ** (sum(math.log10(v) for v in _gm_vals) / len(_gm_vals)) if _gm_vals else float('nan')
        print(f'  seg-MSE: short(f5-10)={_seg(5, 10):.6e}  mid(f11-17)={_seg(11, 17):.6e}  '
              f'long(f18-24)={_seg(18, 24):.6e}   (段内算术均值, 段间勿再平均)')
        print(f'  GM-MSE (f5-24)  : {_gm:.6e}   (几何均值, 各帧相对误差等权)')
        print(f'  FDE (末帧逐点L2): {pm_fde_sum / max(pm_fde_cnt, 1):.6e}   '
              f'MSE@f24={_pf.get(24, float("nan")):.6e}')
        print('  --- Procrustes 位姿/形状分解(pred→GT 相似对齐; MSE=位置敏感, 残余=纯形状) ---')
        for af in PARADIGM_FRAMES:
            if pm_proc_cnt[af]:
                _c = pm_proc_cnt[af]
                _t, _r, _s, _res = (x / _c for x in pm_proc_sum[af])
                print(f'    f{af:<3}: 质心偏移={_t:.4e}  旋转={_r:6.2f}°  尺度s={_s:.4f}'
                      f'(s<1=pred偏大)  残余形状MSE={_res:.6e}   (n={_c})')
        print('  --- 固定边弥散度(首帧kNN k=8 边长膨胀率, 相对首帧;⚠️与 lambda_edge 同族, 仅诊断) ---')
        for af in PARADIGM_FRAMES:
            if pm_edge_cnt[af]:
                _c = pm_edge_cnt[af]
                _rp, _rg = pm_edge_sum[af][0] / _c, pm_edge_sum[af][1] / _c
                print(f'    f{af:<3}: pred={(_rp - 1) * 100:+6.2f}%  gt={(_rg - 1) * 100:+6.2f}%  '
                      f'超额={(_rp - _rg) * 100:+6.2f}%   (n={_c})')
        # ---- per-material 分组(仅多材质数据集打印;口径与上方全局块逐项一致,仅全程窗口)----
        if len(mm_buckets) > 1:
            print('  --- per-material 分组(多材质;段间勿再平均,跨材质单标量禁止) ---')
            for _mt in sorted(mm_buckets):
                mb = mm_buckets[_mt]
                _mfs = sorted(mb['mse_frame_cnt'].keys())
                if not _mfs:
                    continue
                _mpf = {af: mb['mse_frame_sum'][af] / mb['mse_frame_cnt'][af] for af in _mfs}
                _mseg = lambda lo, hi: (lambda xs: sum(xs) / len(xs) if xs else float('nan'))(
                    [_mpf[af] for af in _mfs if lo <= af <= hi])
                _mgm_vals = [_mpf[af] for af in _mfs if 5 <= af <= 24 and _mpf[af] > 0]
                _mgm = 10 ** (sum(math.log10(v) for v in _mgm_vals) / len(_mgm_vals)) if _mgm_vals else float('nan')
                _name = MAT_NAMES.get(_mt, f'mat{_mt}')
                print(f'  === {_name}(mat_type={_mt}, n={mb["fde_cnt"]} 全程窗口)===')
                print('    MSE per-frame: ' + ', '.join(f'f{af}={_mpf[af]:.6e}' for af in _mfs))
                print(f'    seg-MSE: short(f5-10)={_mseg(5, 10):.6e}  mid(f11-17)={_mseg(11, 17):.6e}  '
                      f'long(f18-24)={_mseg(18, 24):.6e}')
                print(f'    GM-MSE(f5-24)={_mgm:.6e}   FDE={mb["fde_sum"] / max(mb["fde_cnt"], 1):.6e}   '
                      f'MSE@f24={_mpf.get(24, float("nan")):.6e}')
                for af in PARADIGM_FRAMES:
                    if mb['proc_cnt'][af]:
                        _c = mb['proc_cnt'][af]
                        _t2, _r2, _s2, _res2 = (x / _c for x in mb['proc_sum'][af])
                        print(f'    Procrustes f{af:<3}: 质心偏移={_t2:.4e}  旋转={_r2:6.2f}°  '
                              f'尺度s={_s2:.4f}  残余形状MSE={_res2:.6e}')
                for af in PARADIGM_FRAMES:
                    if mb['edge_cnt'][af]:
                        _c = mb['edge_cnt'][af]
                        _rp2, _rg2 = mb['edge_sum'][af][0] / _c, mb['edge_sum'][af][1] / _c
                        print(f'    固定边   f{af:<3}: pred={(_rp2 - 1) * 100:+6.2f}%  gt={(_rg2 - 1) * 100:+6.2f}%  '
                              f'超额={(_rp2 - _rg2) * 100:+6.2f}%')
                _mvc = mb['vol_cnt'].clamp(min=1)
                _mvp, _mvg = mb['vol_p'] / _mvc, mb['vol_g'] / _mvc
                for fi, af in enumerate(VOL_DRIFT_FRAMES):
                    if mb['vol_cnt'][fi] > 0:
                        print(f'    体积自漂移 f{af:<3}: pred={_mvp[fi] * 100:6.2f}%  gt={_mvg[fi] * 100:6.2f}%  '
                              f'超额={(_mvp[fi] - _mvg[fi]) * 100:+6.2f}%')
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
    nvd_gt = max(n_vol_drift_gt, 1)
    nfg = max(n_floor_gt, 1)
    print('  --- 物理合理性 ---')
    print(f'  速度误差 vMSE   : {total_vel_mse / nf:.6e}   (帧差, vs GT, {n_full} 全程窗口)')
    print(f'  加速度误差 aMSE : {total_acc_mse / nf:.6e}   (二阶差, vs GT)')
    print(f'  体积相对误差    : {total_vol_err / nv * 100:.3f}%   (|Vp-Vg|/Vg, {n_vol} 帧, {"凸包" if _HAS_HULL else "协方差近似"})')
    print(f'  体积自漂移      : {total_vol_drift / nvd * 100:.3f}%   (|Vp(t)-Vp(0)|/Vp(0), pred)')
    print(f'  体积自漂移(GT)  : {total_vol_drift_gt / nvd_gt * 100:.3f}%   (基线: GT 凸包自漂移, 弹性体大形变下本就>0)')
    print(f'  超额自漂移      : {(total_vol_drift / nvd - total_vol_drift_gt / nvd_gt) * 100:+.3f}%   (pred-GT; >0=模型额外引入的体积不稳定=病态信号, 这才是该看的)')
    # 逐帧体积自漂移(不平均):每帧单独出 pred / GT / 超额(=pred-GT)
    _cvd = cnt_vol_drift_abs.clamp(min=1)
    _dpf = vol_drift_abs_p / _cvd
    _dgf = vol_drift_abs_g / _cvd
    print(f'  --- 逐帧体积自漂移(不平均, 绝对帧; f24=末帧=第25帧) ---')
    for i, af in enumerate(VOL_DRIFT_FRAMES):
        print(f'    f{af:<3}: pred={_dpf[i]*100:6.2f}%  gt={_dgf[i]*100:6.2f}%  超额={ (_dpf[i]-_dgf[i])*100:+6.2f}%   (n={int(cnt_vol_drift_abs[i])})')
    print(f'  地面穿透率      : {total_floor_rate / n * 100:.3f}%   (预测帧占全部点比例, 全 {n_batches} 窗口)')
    print(f'  地面穿透深度    : {total_floor_depth / n:.6e}   (归一化单位)')
    print(f'  地面穿透率(GT)  : {total_floor_rate_gt / nfg * 100:.3f}%   (参考, 应≈0)')
    sys.stdout = _stdout0
    # ---- 结果落盘:eval 指标原样写 markdown(*.md 已 gitignore 不入库;文件名 config/step/seed 可追溯)----
    _res_dir = args.get('results_dir', 'eval_results')
    os.makedirs(_res_dir, exist_ok=True)
    _stem = os.path.splitext(os.path.basename(config_path))[0] if config_path else os.path.basename(str(args.vis_dir))
    _m = re.search(r'checkpoint-(\d+)', str(args.resume))
    _step_tag = _m.group(1) if _m else 'na'
    _res_path = os.path.join(_res_dir, f'{_stem}_step{_step_tag}_seed{args.seed}.md')
    with open(_res_path, 'w', encoding='utf-8') as _f:
        _f.write(f'# eval results: {_stem} @checkpoint-{_step_tag} seed={args.seed}\n\n')
        _f.write(f'- time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        _f.write(f'- config: {config_path}\n')
        _f.write(f'- resume: {args.resume}\n')
        _f.write(f'- use_diffusion: {args.use_diffusion}  num_inference_steps: {args.num_inference_steps}  '
                 f'input_frames: {INPUT_FRAMES}  output_frames: {OUTPUT_FRAMES}\n\n')
        _f.write('```\n' + _res_buf.getvalue() + '```\n')
    print(f'  [results] eval 指标已写入 {_res_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖 config 的 seed(多 seed DDIM 评测用;确定性臂无影响)')
    args = parser.parse_args()
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    if args.seed is not None:
        cfg.seed = args.seed
    main(cfg, config_path=args.config)
