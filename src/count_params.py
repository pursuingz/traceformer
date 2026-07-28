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

def build_mm3(block, **overrides):
    """Build a single-output-frame mm3 model with model-config overrides."""
    model_config = {
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
        'hybrid_state_dim': 64,
        'hybrid_state_heads': 4,
        'hybrid_state_interval': 2,
    }
    model_config.update(overrides)
    model_config['transformer_block'] = block
    mc = OmegaConf.create(model_config)
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


def validate_contact_parameter_budget(baseline, separate, shared):
    expected_contact_parameter_names = {
        'contact_encoder.weight',
        'contact_encoder.bias',
    }
    baseline_parameters = dict(baseline.named_parameters())
    separate_parameters = dict(separate.named_parameters())
    shared_parameters = dict(shared.named_parameters())

    separate_only_vs_baseline = (
        set(separate_parameters) - set(baseline_parameters)
    )
    baseline_only_vs_separate = (
        set(baseline_parameters) - set(separate_parameters)
    )
    if (
        separate_only_vs_baseline != expected_contact_parameter_names
        or baseline_only_vs_separate
    ):
        raise RuntimeError(
            'separate contact parameter names must differ from the no-contact '
            'baseline only by contact_encoder.weight and contact_encoder.bias; '
            f'separate-only={sorted(separate_only_vs_baseline)}, '
            f'baseline-only={sorted(baseline_only_vs_separate)}'
        )

    separate_only_vs_shared = (
        set(separate_parameters) - set(shared_parameters)
    )
    shared_only_vs_separate = (
        set(shared_parameters) - set(separate_parameters)
    )
    if (
        separate_only_vs_shared != expected_contact_parameter_names
        or shared_only_vs_separate
    ):
        raise RuntimeError(
            'separate/shared parameter names must differ only by the separate '
            'contact_encoder parameters; '
            f'separate-only={sorted(separate_only_vs_shared)}, '
            f'shared-only={sorted(shared_only_vs_separate)}'
        )

    contact_encoder = getattr(separate, 'contact_encoder', None)
    if not isinstance(contact_encoder, nn.Linear):
        raise RuntimeError(
            'separate contact model must contain contact_encoder as nn.Linear'
        )
    if getattr(shared, 'contact_encoder', None) is not None:
        raise RuntimeError(
            'shared contact model must not contain a contact_encoder module'
        )

    latent_dim = separate.latent_dim
    expected_contact_weight_shape = (latent_dim, 3)
    expected_contact_bias_shape = (latent_dim,)
    if tuple(contact_encoder.weight.shape) != expected_contact_weight_shape:
        raise RuntimeError(
            'separate contact_encoder.weight must have shape '
            f'{expected_contact_weight_shape}, got '
            f'{tuple(contact_encoder.weight.shape)}'
        )
    if tuple(contact_encoder.bias.shape) != expected_contact_bias_shape:
        raise RuntimeError(
            'separate contact_encoder.bias must have shape '
            f'{expected_contact_bias_shape}, got '
            f'{tuple(contact_encoder.bias.shape)}'
        )

    separate_input_encoder = getattr(separate, 'input_encoder', None)
    shared_input_encoder = getattr(shared, 'input_encoder', None)
    if getattr(separate_input_encoder, 'extra_feature_dim', None) != 0:
        raise RuntimeError(
            'separate input_encoder.extra_feature_dim must be 0'
        )
    if getattr(shared_input_encoder, 'extra_feature_dim', None) != 3:
        raise RuntimeError(
            'shared input_encoder.extra_feature_dim must be 3'
        )
    separate_input_mlp = getattr(separate_input_encoder, 'mlp', None)
    shared_input_mlp = getattr(shared_input_encoder, 'mlp', None)
    if not isinstance(separate_input_mlp, nn.Linear):
        raise RuntimeError(
            'separate input_encoder.mlp must be nn.Linear'
        )
    if not isinstance(shared_input_mlp, nn.Linear):
        raise RuntimeError(
            'shared input_encoder.mlp must be nn.Linear'
        )

    separate_input_shape = tuple(separate_input_mlp.weight.shape)
    shared_input_shape = tuple(shared_input_mlp.weight.shape)
    expected_shared_input_shape = (
        separate_input_shape[0],
        separate_input_shape[1] + 3,
    )
    if shared_input_shape != expected_shared_input_shape:
        raise RuntimeError(
            'shared input_encoder.mlp.weight must add exactly 3 input columns; '
            f'separate={separate_input_shape}, shared={shared_input_shape}'
        )

    common_shape_differences = {
        name: (
            tuple(separate_parameters[name].shape),
            tuple(shared_parameters[name].shape),
        )
        for name in set(separate_parameters) & set(shared_parameters)
        if separate_parameters[name].shape != shared_parameters[name].shape
    }
    expected_shape_difference_name = 'input_encoder.mlp.weight'
    if set(common_shape_differences) != {expected_shape_difference_name}:
        raise RuntimeError(
            'common separate/shared parameter shapes must match except for '
            'input_encoder.mlp.weight; '
            f'differences={common_shape_differences}'
        )

    baseline_total = count(baseline)
    separate_total = count(separate)
    shared_total = count(shared)
    separate_delta = separate_total - baseline_total
    shared_delta = shared_total - baseline_total
    shared_vs_separate_delta = shared_total - separate_total
    contact_weight_params = contact_encoder.weight.numel()
    condition_frame_bias_params = contact_encoder.bias.numel()
    shared_input_expansion_params = (
        shared_input_mlp.weight.numel()
        - separate_input_mlp.weight.numel()
    )

    structural_mismatches = []
    if separate_delta != contact_weight_params + condition_frame_bias_params:
        structural_mismatches.append(
            'separate delta must equal contact_encoder weight plus bias: '
            f'{separate_delta:+d} != '
            f'{contact_weight_params + condition_frame_bias_params:+d}'
        )
    if shared_delta != shared_input_expansion_params:
        structural_mismatches.append(
            'shared delta must equal the expanded input columns: '
            f'{shared_delta:+d} != {shared_input_expansion_params:+d}'
        )
    if shared_input_expansion_params != contact_weight_params:
        structural_mismatches.append(
            'shared expanded input columns must replace the separate contact '
            f'weight: {shared_input_expansion_params} != {contact_weight_params}'
        )
    if shared_vs_separate_delta != -condition_frame_bias_params:
        structural_mismatches.append(
            'shared/separate delta must equal the negative separate-only '
            'condition-frame bias size: '
            f'{shared_vs_separate_delta:+d} != '
            f'{-condition_frame_bias_params:+d}'
        )

    expected_deltas = {
        'separate relative to no-contact baseline': 1024,
        'shared relative to no-contact baseline': 768,
        'shared relative to separate contact': -256,
    }
    actual_deltas = {
        'separate relative to no-contact baseline': separate_delta,
        'shared relative to no-contact baseline': shared_delta,
        'shared relative to separate contact': shared_vs_separate_delta,
    }
    budget_mismatches = [
        f'{name}: expected {expected:+d}, got {actual_deltas[name]:+d}'
        for name, expected in expected_deltas.items()
        if actual_deltas[name] != expected
    ]
    mismatches = structural_mismatches + budget_mismatches
    if mismatches:
        raise RuntimeError(
            'contact parameter budget drifted; '
            + '; '.join(mismatches)
        )

    return {
        'baseline_total': baseline_total,
        'separate_total': separate_total,
        'shared_total': shared_total,
        'separate_delta': separate_delta,
        'shared_delta': shared_delta,
        'shared_vs_separate_delta': shared_vs_separate_delta,
        'contact_weight_params': contact_weight_params,
        'condition_frame_bias_params': condition_frame_bias_params,
        'shared_input_expansion_params': shared_input_expansion_params,
        'common_shape_differences': common_shape_differences,
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

    baseline_mm3 = build_mm3('SpatialTemporalTransformerBlock')
    separate_contact_mm3 = build_mm3(
        'SpatialTemporalTransformerBlock',
        contact_particle_cond=True,
        contact_feature_sigma=0.04,
        contact_injection_mode='separate',
    )
    shared_contact_mm3 = build_mm3(
        'SpatialTemporalTransformerBlock',
        contact_particle_cond=True,
        contact_feature_sigma=0.04,
        contact_injection_mode='shared',
    )
    contact_report = validate_contact_parameter_budget(
        baseline_mm3,
        separate_contact_mm3,
        shared_contact_mm3,
    )
    print('--- mm3 contact exact parameter budget ---')
    print(
        'mm3 no-contact v1 serial 8L: '
        f'total={contact_report["baseline_total"]:,}'
    )
    print(
        'mm3 separate contact v1 serial 8L: '
        f'total={contact_report["separate_total"]:,}  '
        f'delta-vs-no-contact={contact_report["separate_delta"]:+,} params'
    )
    print(
        'mm3 shared contact v1 serial 8L: '
        f'total={contact_report["shared_total"]:,}  '
        f'delta-vs-no-contact={contact_report["shared_delta"]:+,} params  '
        'delta-vs-separate='
        f'{contact_report["shared_vs_separate_delta"]:+,} params'
    )
    print(
        'separate/shared difference: '
        f'{contact_report["condition_frame_bias_params"]:,}-parameter '
        'separate-only condition-frame bias'
    )

    # Preserve the historical five-output-frame v_xyz comparison.
    generic_contact_model = build(
        'SpatialTemporalTransformerBlock',
        8,
        contact_particle_cond=True,
        contact_feature_sigma=0.04,
        contact_injection_mode='separate',
    )
    contact_xyz_model = build(
        'SpatialTemporalTransformerBlock',
        8,
        contact_particle_cond=True,
        contact_feature_sigma=0.04,
        contact_velocity_mode='xyz',
    )
    contact_xyz_delta = count(contact_xyz_model) - count(generic_contact_model)
    print(
        'v1 serial + contact v_xyz 8L: '
        f'total={count(contact_xyz_model)/1e6:.3f}M  '
        f'delta-vs-v_y={contact_xyz_delta} params'
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
