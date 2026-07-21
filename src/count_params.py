import sys
sys.path.append('./')
from omegaconf import OmegaConf
from model.hybrid_state import HybridStateExchange
from model.spacetime import MDM_ST, SpatialTemporalTransformerBlock

base_mc = {
    'n_layers': 8, 'latent_dim': 256, 'frame_cond': True, 'point_embed': True,
    'mask_cond': True, 'pred_offset': True, 'num_neighbors': -1, 'floor_cond': True,
    'max_num_forces': 1, 'force_as_token': False, 'force_as_latent': False,
    'coeff_cond': False, 'class_token': False, 'cond_frames': 5,
}

import torch.nn as nn

def count(m):
    return sum(p.numel() for p in m.parameters())

def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def build(block, n_layers, **overrides):
    mc = OmegaConf.create(dict(
        base_mc,
        transformer_block=block,
        n_layers=n_layers,
        **overrides,
    ))
    return MDM_ST(512, 5, n_feats=3, model_config=mc)

def build_mm3(block):
    """Build the exact single-output-frame mm3 baseline or v11a model."""
    mc = OmegaConf.create({
        'n_layers': 8,
        'latent_dim': 256,
        'frame_cond': True,
        'cond_frames': 5,
        'point_embed': True,
        'mask_cond': True,
        'pred_offset': True,
        'num_neighbors': -1,
        'floor_cond': True,
        'max_num_forces': 1,
        'force_as_token': False,
        'force_as_latent': False,
        'gravity_emb': True,
        'coeff_cond': False,
        'num_mat': 4,
        'class_token': True,
        'class_dropout_prob': 0.0,
        'transformer_block': block,
        'hybrid_state_dim': 64,
        'hybrid_state_heads': 4,
        'hybrid_state_interval': 2,
    })
    return MDM_ST(n_points=2048, n_frame=1, n_feats=3, model_config=mc)

def estimate_exchange_compute(batch_size=1, n_points=2048, history_frames=5,
                              particle_dim=256, state_dim=64, exchange_calls=4):
    """Estimate v11a exchange forward compute; this is not measured runtime."""
    B, N, T, C, S = batch_size, n_points, history_frames, particle_dim, state_dim

    # Count only major linear, attention-matmul, and weighted-pooling operators.
    # Linear(a -> b) costs approximately tokens*a*b MACs. Attention adds QK^T
    # and AV (2*queries*keys*head-width). Pooling counts score projection and
    # weighted reduction. Norms, activations, softmax, elementwise ops, explicit
    # state statistics, the serial backbone, backward, and optimizer are omitted.
    per_call = {
        'particle score + weighted pool': 2 * B * T * N * C,
        'pooled particle projection': B * T * C * S,
        'explicit-state MLP': B * T * (18 * S + S * S),
        'material MLP': B * (2 * S + S * S),
        'state self-attention': B * (4 * T * S * S + 2 * T * T * S),
        'state feed-forward': B * 8 * T * S * S,
        'state + feedback FiLM': B * (2 * S * S + 2 * S * C),
        'particle-to-state cross-attention': B * (
            2 * N * C * S + 2 * T * S * S + 2 * N * T * S
        ),
    }
    per_call_macs = sum(per_call.values())
    total_macs = exchange_calls * per_call_macs
    return per_call, per_call_macs, total_macs

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
    ('v6b strain-gated 8L', 'SpatialTemporalTransformerBlockv6b', 8),
    ('v7 state-token 8L', 'SpatialTemporalTransformerBlockv7', 8),
    ('v8 physics-slice 8L', 'SpatialTemporalTransformerBlockv8', 8),
    ('v9 dual-graph 8L', 'SpatialTemporalTransformerBlockv9', 8),
    ('v10 particle-grid 8L', 'SpatialTemporalTransformerBlockv10', 8),
]:
    m = build(block, L)
    tot = count(m)
    blocks = find_blocks(m)
    blk = count(blocks)
    print(f'{name}: total={tot/1e6:.3f}M  blocks={blk/1e6:.3f}M  per-block={blk/L/1e6:.4f}M')

baseline_contact_ref = build('SpatialTemporalTransformerBlock', 8)
contact_model = build(
    'SpatialTemporalTransformerBlock',
    8,
    contact_particle_cond=True,
    contact_feature_sigma=0.04,
)
contact_delta = count(contact_model) - count(baseline_contact_ref)
print(
    'v1 serial + contact cond 8L: '
    f'total={count(contact_model)/1e6:.3f}M  delta={contact_delta} params'
)

# per-submodule breakdown of one v3 block
m3 = build('SpatialTemporalTransformerBlockv3', 8)
b = find_blocks(m3)[0]
print('--- one v3 block breakdown ---')
print(f'  spatial_block : {count(b.spatial_block)/1e6:.4f}M')
print(f'  temporal_block: {count(b.temporal_block)/1e6:.4f}M')
print(f'  fuse (h+enc)  : {(count(b.hidden_fuse)+count(b.encoder_fuse))/1e6:.4f}M')
m1 = build('SpatialTemporalTransformerBlock', 8)
print(f'  v1 one block  : {count(find_blocks(m1)[0])/1e6:.4f}M')

# v11a is intentionally separate from the generic table above: that table builds
# five output frames, while v11a is defined only for one output frame.
baseline_mm3 = build_mm3('SpatialTemporalTransformerBlock')
v11a_mm3 = build_mm3('SpatialTemporalTransformerBlockv11a')
baseline_trainable = count_trainable(baseline_mm3)
v11a_trainable = count_trainable(v11a_mm3)
absolute_delta = abs(v11a_trainable - baseline_trainable)
delta_percent = 100.0 * absolute_delta / baseline_trainable

v11a_blocks = find_blocks(v11a_mm3)
block_types = [type(block).__name__ for block in v11a_blocks]
exchange_modules = [
    module for module in v11a_mm3.modules()
    if isinstance(module, HybridStateExchange)
]
exchange_stages = [exchange.num_stages for exchange in exchange_modules]

assert len(v11a_blocks) == 8
assert all(type(block) is SpatialTemporalTransformerBlock for block in v11a_blocks)
assert len(exchange_modules) == 1
assert exchange_stages == [4]
assert delta_percent < 1.0, f'v11a parameter delta is {delta_percent:.6f}%'

print('--- v11a mm3 exact parameter budget ---')
print(f'baseline original serial trainable params: {baseline_trainable:,}')
print(f'v11a trainable params                   : {v11a_trainable:,}')
print(f'absolute delta                          : {absolute_delta:,}')
print(f'percentage delta                        : {delta_percent:.6f}%')
print(
    'v11a actual blocks                    : '
    f'type={sorted(set(block_types))} layers={len(v11a_blocks)}'
)
print(
    'v11a hybrid exchange                  : '
    f'copies={len(exchange_modules)} stages={exchange_stages[0]} calls=4'
)

compute_parts, per_call_macs, total_macs = estimate_exchange_compute()
print('--- v11a exchange approximate forward compute ---')
print('shape: B=1, N=2048, T=5, C=256, S=64; exchange calls=4')
for operator, macs in compute_parts.items():
    print(f'  {operator:<37}: {macs / 1e6:.6f} M MAC/call')
print(f'per exchange call                       : {per_call_macs / 1e9:.6f} G MAC')
print(f'four exchange calls                     : {total_macs / 1e9:.6f} G MAC')
print(f'four-call arithmetic (1 MAC ~= 2 FLOPs) : {2 * total_macs / 1e9:.6f} G FLOPs')
print(
    'scope: analytical exchange-only forward estimate; excludes the serial '
    'backbone, omitted operators, backward, optimizer, and hardware effects'
)
print('required follow-up: server-side 100-step iteration timing under the training config')
