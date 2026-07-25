# Contact v_xyz Conditioning Design

## 1. Goal

Test whether explicitly providing the full per-particle displacement vector
improves multi-material dynamics prediction relative to the current vertical-only
contact conditioning.

The experiment changes one factor:

```text
baseline: [signed_gap, v_y, proximity]
v_xyz:    [signed_gap, v_x, v_y, v_z, proximity]
```

All data, sampling, losses, Transformer blocks, hidden width, optimizer,
learning-rate schedule, training seed, and evaluation split remain unchanged.

## 2. Feature Definition

For conditioning points `X` with shape `(B, F, N, 3)`, define:

```text
v[:, t] = X[:, t] - X[:, t - 1]       for t > 0
v[:, 0] = start_velocity              when available
v[:, 0] = v[:, 1]                     otherwise
```

The displacement is not divided by `dt`. This preserves the scale and meaning of
the existing `v_y` channel.

The contact feature order is:

```text
[signed_gap, v_x, v_y, v_z, proximity]
```

where:

```text
signed_gap = y - floor_height
proximity  = exp(-(relu(signed_gap) / sigma)^2)
sigma      = 0.04
```

## 3. Encoder

The recommended design uses one pointwise shared encoder:

```python
contact_encoder = nn.Linear(5, latent_dim)
```

The same weights are reused for every batch item, conditioning frame, and
particle. Relative to `Linear(3, 256)`, this adds only 512 weights:

```text
(5 - 3) * 256 = 512
```

The layer remains zero-initialized. Its injection point and bias behavior remain
identical to the baseline.

Using separate linear layers for gap, velocity, and proximity is deliberately
out of scope. If their outputs are only added,

```text
W_gap gap + W_vel velocity + W_prox proximity
```

is mathematically equivalent to one linear layer over concatenated inputs.
Separate layers become more expressive only when they have separate nonlinear
processing, gating, normalization, depth-wise injection, or regularization.
Those changes would confound this experiment.

## 4. Backward Compatibility

Add:

```yaml
model_config:
  contact_velocity_mode: vertical  # default
```

Supported modes:

- `vertical`: existing three-channel behavior and existing checkpoint shapes.
- `xyz`: new five-channel behavior.

The default must preserve existing configs and checkpoints byte-for-byte.

Feature masks are mode-specific:

```text
vertical: [gap, v_y, proximity]
xyz:      [gap, v_x, v_y, v_z, proximity]
```

Invalid mode names and mask lengths raise explicit errors.

## 5. Experiment Configuration

Create:

```text
src/configs/config_mm3_contact_vxyz.yaml
src/configs/eval_mm3_contact_vxyz_45k.yaml
```

The training config mirrors `config_mm3_contact_cond.yaml`, except:

```yaml
output_dir: ./outputs/mm3_contact_vxyz_8L
max_train_steps: 90000
stop_after_steps: 45000
model_config:
  contact_velocity_mode: xyz
```

The evaluation config mirrors the corresponding training model and dataset
settings, points to checkpoint 45000, and uses the same held-out 41-model test
split and seed 0.

Training starts from scratch because `Linear(3, 256)` and `Linear(5, 256)` have
different checkpoint shapes.

## 6. Evaluation

Compare against `mm3_contact_cond_8L/checkpoint-45000` at matched step.

Primary metrics:

- full-rollout MSE;
- GM-MSE and FDE;
- per-material MSE/FDE;
- contact-region MSE;
- contact normal-vMSE;
- contact precision and recall;
- floor penetration rate and depth.

Interpretation:

- elastic improvement indicates general kinematic benefit;
- plasticine/sand improvement indicates better tangential contact, sliding, or
  granular-flow modeling;
- first-frame improvement without rollout improvement indicates no long-horizon
  benefit;
- no improvement indicates that `v_x` and `v_z` are already recoverable from
  position history at acceptable cost.

## 7. Verification

Tests must cover:

- exact legacy three-channel values and shape;
- exact five-channel values, including `start_velocity`;
- invalid mode and mask-length errors;
- old checkpoint-compatible model construction;
- new five-channel model construction and forward shape;
- training/evaluation config parity;
- all variables except the declared experiment changes remain frozen.

Run:

```bash
python -m py_compile src/utils/contact.py src/model/spacetime.py
python -m unittest src.tests.test_contact
python -m unittest src.tests.test_contact_ablation_configs
python src/count_params.py
```
