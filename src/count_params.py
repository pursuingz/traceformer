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

def estimate_exchange_compute(exchange_calls, batch_size=1, n_points=2048,
                              history_frames=5, particle_dim=256, state_dim=64):
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

def validate_v11a_parameter_budget(baseline, v11a):
    baseline_total = count(baseline)
    baseline_trainable = count_trainable(baseline)
    if baseline_total != baseline_trainable:
        raise RuntimeError(
            'baseline total params must equal baseline trainable params: '
            f'{baseline_total} != {baseline_trainable}'
        )

    v11a_total = count(v11a)
    v11a_trainable = count_trainable(v11a)
    if v11a_total != v11a_trainable:
        raise RuntimeError(
            'v11a total params must equal v11a trainable params: '
            f'{v11a_total} != {v11a_trainable}'
        )

    signed_delta = v11a_total - baseline_total
    if signed_delta <= 0:
        raise RuntimeError(f'v11a signed parameter delta must be positive: {signed_delta}')

    blocks = find_blocks(v11a)
    if len(blocks) != 8:
        raise RuntimeError(f'v11a must contain 8 transformer blocks, got {len(blocks)}')
    if not all(type(block) is SpatialTemporalTransformerBlock for block in blocks):
        block_types = [type(block).__name__ for block in blocks]
        raise RuntimeError(
            'v11a blocks must all be original SpatialTemporalTransformerBlock: '
            f'{block_types}'
        )

    exchanges = [
        module for module in v11a.modules()
        if isinstance(module, HybridStateExchange)
    ]
    if len(exchanges) != 1:
        raise RuntimeError(
            f'v11a must contain exactly one HybridStateExchange, got {len(exchanges)}'
        )
    exchange = exchanges[0]
    if exchange.num_stages != 4:
        raise RuntimeError(
            f'v11a exchange must contain 4 stages, got {exchange.num_stages}'
        )

    exchange_total = count(exchange)
    exchange_trainable = count_trainable(exchange)
    if exchange_total != exchange_trainable:
        raise RuntimeError(
            'v11a exchange total params must equal trainable params: '
            f'{exchange_total} != {exchange_trainable}'
        )
    if signed_delta != exchange_total:
        raise RuntimeError(
            'v11a signed parameter delta must equal the unique exchange params: '
            f'{signed_delta} != {exchange_total}'
        )

    interval = getattr(v11a.dit, 'hybrid_state_interval', None)
    if not isinstance(interval, int) or interval <= 0:
        raise RuntimeError(f'v11a hybrid_state_interval must be positive: {interval}')
    if len(blocks) % interval != 0:
        raise RuntimeError(
            'v11a block count must be divisible by hybrid_state_interval: '
            f'{len(blocks)} % {interval}'
        )
    exchange_calls = len(blocks) // interval
    if exchange_calls != exchange.num_stages:
        raise RuntimeError(
            'v11a exchange calls derived from layer interval must equal stages: '
            f'{exchange_calls} != {exchange.num_stages}'
        )

    delta_percent = 100.0 * signed_delta / baseline_total
    if delta_percent >= 1.0:
        raise RuntimeError(f'v11a parameter delta is {delta_percent:.6f}%')

    return {
        'baseline_total': baseline_total,
        'baseline_trainable': baseline_trainable,
        'v11a_total': v11a_total,
        'v11a_trainable': v11a_trainable,
        'signed_delta': signed_delta,
        'delta_percent': delta_percent,
        'block_types': sorted({type(block).__name__ for block in blocks}),
        'block_count': len(blocks),
        'exchange_count': len(exchanges),
        'exchange_total': exchange_total,
        'exchange_trainable': exchange_trainable,
        'exchange_stages': exchange.num_stages,
        'exchange_interval': interval,
        'exchange_calls': exchange_calls,
    }


def main():
    for name, block, layers in [
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
        model = build(block, layers)
        total = count(model)
        blocks = find_blocks(model)
        block_total = count(blocks)
        print(
            f'{name}: total={total/1e6:.3f}M  blocks={block_total/1e6:.3f}M  '
            f'per-block={block_total/layers/1e6:.4f}M'
        )

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

    # Per-submodule breakdown of one v3 block.
    model_v3 = build('SpatialTemporalTransformerBlockv3', 8)
    block_v3 = find_blocks(model_v3)[0]
    print('--- one v3 block breakdown ---')
    print(f'  spatial_block : {count(block_v3.spatial_block)/1e6:.4f}M')
    print(f'  temporal_block: {count(block_v3.temporal_block)/1e6:.4f}M')
    print(
        '  fuse (h+enc)  : '
        f'{(count(block_v3.hidden_fuse)+count(block_v3.encoder_fuse))/1e6:.4f}M'
    )
    model_v1 = build('SpatialTemporalTransformerBlock', 8)
    print(f'  v1 one block  : {count(find_blocks(model_v1)[0])/1e6:.4f}M')

    # The generic table builds five output frames; v11a is single-output only.
    baseline_mm3 = build_mm3('SpatialTemporalTransformerBlock')
    v11a_mm3 = build_mm3('SpatialTemporalTransformerBlockv11a')
    report = validate_v11a_parameter_budget(baseline_mm3, v11a_mm3)

    print('--- v11a mm3 exact parameter budget ---')
    print(
        'baseline original serial trainable params: '
        f'{report["baseline_trainable"]:,}'
    )
    print(f'v11a trainable params                   : {report["v11a_trainable"]:,}')
    print(f'signed delta                            : {report["signed_delta"]:,}')
    print(f'percentage delta                        : {report["delta_percent"]:.6f}%')
    print(
        'v11a actual blocks                    : '
        f'type={report["block_types"]} layers={report["block_count"]}'
    )
    print(
        'v11a hybrid exchange                  : '
        f'copies={report["exchange_count"]} stages={report["exchange_stages"]} '
        f'interval={report["exchange_interval"]} calls={report["exchange_calls"]}'
    )

    shape = {
        'batch_size': 1,
        'n_points': 2048,
        'history_frames': 5,
        'particle_dim': 256,
        'state_dim': 64,
    }
    exchange_calls = report['exchange_calls']
    compute_parts, per_call_macs, total_macs = estimate_exchange_compute(
        **shape,
        exchange_calls=exchange_calls,
    )
    print('--- v11a exchange approximate forward compute ---')
    print(
        f'shape: B={shape["batch_size"]}, N={shape["n_points"]}, '
        f'T={shape["history_frames"]}, C={shape["particle_dim"]}, '
        f'S={shape["state_dim"]}; exchange calls={exchange_calls}'
    )
    for operator, macs in compute_parts.items():
        print(f'  {operator:<37}: {macs / 1e6:.6f} M MAC/call')
    print(f'per exchange call                       : {per_call_macs / 1e9:.6f} G MAC')
    calls_label = f'{exchange_calls} exchange calls'
    flops_label = f'{exchange_calls}-call arithmetic (1 MAC ~= 2 FLOPs)'
    print(f'{calls_label:<40}: {total_macs / 1e9:.6f} G MAC')
    print(f'{flops_label:<40}: {2 * total_macs / 1e9:.6f} G FLOPs')
    print(
        'scope: analytical exchange-only forward estimate; excludes the serial '
        'backbone, omitted operators, backward, optimizer, and hardware effects'
    )
    print(
        'required follow-up: server-side 100-step iteration timing under the '
        'training config'
    )


if __name__ == '__main__':
    main()
