import sys
sys.path.append('./')
from omegaconf import OmegaConf
from model.spacetime import MDM_ST

base_mc = {
    'n_layers': 8, 'latent_dim': 256, 'frame_cond': True, 'point_embed': True,
    'mask_cond': True, 'pred_offset': True, 'num_neighbors': -1, 'floor_cond': True,
    'max_num_forces': 1, 'force_as_token': False, 'force_as_latent': False,
    'coeff_cond': False, 'class_token': False, 'cond_frames': 5,
}

import torch.nn as nn

def count(m):
    return sum(p.numel() for p in m.parameters())

def build(block, n_layers):
    mc = OmegaConf.create(dict(base_mc, transformer_block=block, n_layers=n_layers))
    return MDM_ST(512, 5, n_feats=3, model_config=mc)

def find_blocks(m):
    for name, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith('transformer_blocks'):
            return mod
    raise RuntimeError('transformer_blocks not found')

for name, block, L in [
    ('v1 serial      8L', 'SpatialTemporalTransformerBlock', 8),
    ('v3 parallel    8L', 'SpatialTemporalTransformerBlockv3', 8),
    ('v3 parallel    5L', 'SpatialTemporalTransformerBlockv3', 5),
    ('v4 lean-par    8L', 'SpatialTemporalTransformerBlockv4', 8),
    ('v4 lean-par    7L', 'SpatialTemporalTransformerBlockv4', 7),
    ('v6 local-global 8L', 'SpatialTemporalTransformerBlockv6', 8),
    ('v7 state-token 8L', 'SpatialTemporalTransformerBlockv7', 8),
]:
    m = build(block, L)
    tot = count(m)
    blocks = find_blocks(m)
    blk = count(blocks)
    print(f'{name}: total={tot/1e6:.3f}M  blocks={blk/1e6:.3f}M  per-block={blk/L/1e6:.4f}M')

# per-submodule breakdown of one v3 block
m3 = build('SpatialTemporalTransformerBlockv3', 8)
b = find_blocks(m3)[0]
print('--- one v3 block breakdown ---')
print(f'  spatial_block : {count(b.spatial_block)/1e6:.4f}M')
print(f'  temporal_block: {count(b.temporal_block)/1e6:.4f}M')
print(f'  fuse (h+enc)  : {(count(b.hidden_fuse)+count(b.encoder_fuse))/1e6:.4f}M')
m1 = build('SpatialTemporalTransformerBlock', 8)
print(f'  v1 one block  : {count(find_blocks(m1)[0])/1e6:.4f}M')
