import torch
import numpy as np

def transform2origin(v, size=1):
    bmax = v.max(axis=0)
    bmin = v.min(axis=0)
    aabb = bmax - bmin
    center = (bmax + bmin) / 2
    scale = size / (aabb.max() * 0.5)
    new_v = (v - center) * scale
    return new_v, center, scale 

def shift2center_th(position_tensor, center=[5, 5, 5]):
    tensor = torch.tensor(center, dtype=torch.float32, device=position_tensor.device).contiguous()
    return position_tensor + tensor

def shift2center(position_tensor, center=[5, 5, 5]):
    tensor = np.array(center)
    return position_tensor + tensor