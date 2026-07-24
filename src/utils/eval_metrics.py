# -*- coding: utf-8 -*-
"""度量层扩展(范式对比用,2026-07-13):
  1. Umeyama 位姿/形状误差分解 —— 把逐点 MSE 拆成正交的「位置/位姿漂移」与「对齐后纯形状误差」,
     解决「MSE 对位置敏感/体积对位置全盲」两指标打架的问题(实验记录.md physctrl-orig 节)。
  2. 固定边弥散度 —— 首帧 kNN 图固定拓扑,追踪边长膨胀率,直接量化点云「发糊」。
     ⚠️ Goodhart:该量与 sfG 臂训练正则 lambda_edge 同族(A 臂照着它训),只作诊断指标,
     不作跨臂裁决主指标;裁决靠渲染。
  3. 全帧 per-frame MSE / 分段 seg-MSE / 几何均值 / FDE —— 单标量隐含 horizon 权重
     (误差跨 3 个量级,算术均值被晚帧主导),改为分段 + 端点的最小标量集合。
纯张量函数,CPU/GPU 通用、无 cuda 硬依赖 → 可本机单测(scratchpad test_eval_metrics.py)。
被 eval.py 在 full-horizon(start=0)窗口上调用,只算预测帧(af >= input_frames)。
"""
import torch


def append_result_tag(base, result_tag):
    """Append a diagnostic tag while preserving legacy paths by default."""
    return str(base) if not result_tag else f"{base}_{result_tag}"


def contact_region_metrics(
    pred,
    gt,
    previous_frame,
    floor_height,
    margin=0.04,
):
    """Return contact/free-space metric sums and particle-time counts."""
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(
            "pred and gt must share shape (B,T,N,3); "
            f"got {tuple(pred.shape)} and {tuple(gt.shape)}"
        )
    if previous_frame.ndim == 4:
        if previous_frame.shape[1] != 1:
            raise ValueError("previous_frame may contain only one frame")
        previous_frame = previous_frame[:, 0]
    if (
        previous_frame.ndim != 3
        or previous_frame.shape[0] != pred.shape[0]
        or previous_frame.shape[1:] != pred.shape[2:]
    ):
        raise ValueError(
            "previous_frame must have shape (B,N,3); "
            f"got {tuple(previous_frame.shape)}"
        )

    floor = torch.as_tensor(
        floor_height,
        device=gt.device,
        dtype=gt.dtype,
    ).reshape(pred.shape[0], 1, 1)
    gt_contact = gt[..., 1] - floor <= float(margin)
    pred_contact = pred[..., 1] - floor <= float(margin)
    free = ~gt_contact

    point_mse = (pred - gt).square().mean(dim=-1)
    pred_y = torch.cat([previous_frame[:, None, :, 1], pred[..., 1]], dim=1)
    gt_y = torch.cat([previous_frame[:, None, :, 1], gt[..., 1]], dim=1)
    normal_velocity_mse = (
        (pred_y[:, 1:] - pred_y[:, :-1])
        - (gt_y[:, 1:] - gt_y[:, :-1])
    ).square()

    contact_float = gt_contact.to(point_mse.dtype)
    free_float = free.to(point_mse.dtype)
    return {
        "contact_mse_sum": (point_mse * contact_float).sum(),
        "contact_count": gt_contact.sum(),
        "free_mse_sum": (point_mse * free_float).sum(),
        "free_count": free.sum(),
        "normal_velocity_mse_sum": (
            normal_velocity_mse * contact_float
        ).sum(),
        "normal_velocity_count": gt_contact.sum(),
        "true_positive_count": (pred_contact & gt_contact).sum(),
        "pred_contact_count": pred_contact.sum(),
        "gt_contact_count": gt_contact.sum(),
    }


def umeyama_decompose(p, g):
    """已知逐点对应下的相似变换对齐(Umeyama 1991):min_{s,R,t} ||s R p + t - g||^2。
    p/g: (N,3),同 dtype/device。返回 (质心偏移L2, 旋转角deg, 尺度s, 对齐后残余MSE):
      - 质心偏移:||mean(p) - mean(g)||,整体错位幅度(归一化单位)
      - 旋转角:R 的转角,整体转错的幅度
      - 尺度 s:把 pred 缩放到 GT 的系数。s < 1 = pred 偏大(如「发胖」),s > 1 = pred 偏小
      - 残余 MSE:对齐(去平移/旋转/缩放)后的逐点 MSE = 纯形状误差,与 F.mse_loss 同口径
    """
    p = p.double()
    g = g.double()
    n = p.shape[0]
    mu_p, mu_g = p.mean(0), g.mean(0)
    pc, gc = p - mu_p, g - mu_g
    sigma = gc.T @ pc / n                                   # (3,3) 互协方差
    U, D, Vh = torch.linalg.svd(sigma)
    S = torch.eye(3, dtype=p.dtype, device=p.device)
    if torch.det(U @ Vh) < 0:                               # 防反射解
        S[2, 2] = -1.0
    R = U @ S @ Vh
    var_p = (pc ** 2).sum() / n
    s = (D * S.diagonal()).sum() / var_p
    resid = ((s * (pc @ R.T) - gc) ** 2).mean()
    trans = (mu_p - mu_g).norm()
    cos = ((R.trace() - 1) / 2).clamp(-1.0, 1.0)
    rot_deg = torch.rad2deg(torch.acos(cos))
    return trans.item(), rot_deg.item(), s.item(), resid.item()


def knn_edges(pc, k=8):
    """首帧 kNN 边索引(固定拓扑,不含自身)。pc: (N,3) → (N,k) long。"""
    d = torch.cdist(pc.unsqueeze(0), pc.unsqueeze(0)).squeeze(0)
    return d.topk(k + 1, largest=False).indices[:, 1:]


def edge_lengths(pc, idx):
    """固定边长。pc: (N,3), idx: (N,k) → (N,k)。"""
    return (pc.unsqueeze(1) - pc[idx]).norm(dim=-1)


def per_window_metrics(pred, gt, input_frames, anchor_frames=(5, 10, 15, 20, 24), k=8, eps=1e-8):
    """单个 full-horizon 窗口的范式对比度量。pred/gt: (T,N,3) float32,归一化空间,
    帧 0..input_frames-1 = GT 输入段(不计入)。返回 dict:
      mse_frame: {af: 单帧 MSE}(af >= input_frames 的全部预测帧)
      fde:       末帧逐点 L2 均值(Final Displacement Error,文献口径)
      proc:      {af: (质心偏移, 旋转deg, 尺度s, 残余形状MSE)}(仅 anchor 帧)
      edge:      {af: (pred 边长膨胀率, GT 边长膨胀率)}(相对首帧;超额 = pred - gt)
    """
    T = pred.shape[0]
    out = {'mse_frame': {}, 'proc': {}, 'edge': {}}
    for af in range(input_frames, T):
        out['mse_frame'][af] = torch.mean((pred[af] - gt[af]) ** 2).item()
    out['fde'] = (pred[T - 1] - gt[T - 1]).norm(dim=-1).mean().item()
    for af in anchor_frames:
        if input_frames <= af < T:
            out['proc'][af] = umeyama_decompose(pred[af], gt[af])
    idx = knn_edges(gt[0], k)                               # 拓扑建在首帧(两臂首帧同为 GT 输入)
    e0 = edge_lengths(gt[0], idx)
    valid = e0 > eps                                        # 防重合点除零
    for af in anchor_frames:
        if input_frames <= af < T:
            rp = (edge_lengths(pred[af], idx)[valid] / e0[valid]).mean().item()
            rg = (edge_lengths(gt[af], idx)[valid] / e0[valid]).mean().item()
            out['edge'][af] = (rp, rg)
    return out
