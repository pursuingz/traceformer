# v11a MC-HST Design

**Status:** Approved for implementation planning
**Date:** 2026-07-21
**Working name:** Material-Conditioned Hybrid State Transformer (MC-HST)

## 1. Objective

Test whether an explicit frame-level representation of whole-object state improves:

1. one-step particle prediction;
2. autoregressive rollout stability; and
3. sensitivity and generalization to continuous material parameters `log10(E)` and `nu`.

The experiment must isolate the architecture. It must not change the dataset, temporal sampling, losses, optimizer, training steps, hidden width, number of backbone layers, or evaluation split.

The comparison baseline is:

```text
src/configs/config_mm3_singleframe_geom_deform_d0001.yaml
```

Its eight original `SpatialTemporalTransformerBlock` instances remain unchanged in v11a.

## 2. Scope

### In scope

- One shared `HybridStateExchange` module.
- Four state exchanges after backbone layers 2, 4, 6, and 8.
- One hybrid state token for each of the five observed physical frames.
- Learned particle pooling plus explicit whole-object statistics.
- Repeated continuous `E/nu` conditioning of the state and feedback paths.
- State-to-particle feedback into the single prediction frame.
- A matched train/eval configuration and parameter/FLOP reporting.

### Out of scope

- New auxiliary losses, including COM or covariance supervision.
- Changes to spatial attention, temporal attention, FFN order, or backbone depth.
- Replacing the original temporal attention.
- Convex-hull volume as an input or training target.
- Contact/floor changes.
- Rollout-aware training changes.
- Exact parameter matching by shrinking the backbone FFNs.

Those changes require separate experimental arms. In particular:

- `v11a-PM`: optional exact-parameter confirmation if v11a is positive;
- `v11b`: explicit global-state supervision;
- `v11c`: state-conditioned backbone temporal attention.

## 3. Motivation and Prior-Art Boundary

The current particle transformer has access to all particle features, but it has no explicit representation of whole-object motion over the five observed frames. v7 introduced per-frame latent state tokens inside every block, but it:

- created 16 latent tokens independently for each frame;
- did not construct a clear five-frame global motion sequence;
- had no explicit COM, covariance, or material-conditioned state representation; and
- duplicated the state module in every layer.

v11a instead represents each observed frame with one hybrid state token and evolves the five-token state sequence between groups of two unchanged serial blocks.

The following components have prior art and are not novelty claims by themselves:

- particle/set pooling;
- global or abstract tokens;
- token-to-particle cross-attention;
- FiLM/AdaLN material conditioning;
- COM and covariance statistics.

The research hypothesis concerns their controlled combination: a material-conditioned, physically interpretable frame-state sequence that repeatedly feeds whole-object motion context into particle prediction.

## 4. Backbone and Exchange Schedule

The eight-layer backbone is partitioned into four stages:

```text
Blocks 1-2 -> Exchange 1
Blocks 3-4 -> Exchange 2
Blocks 5-6 -> Exchange 3
Blocks 7-8 -> Exchange 4 -> Output head
```

All four exchange calls share the same pooling, state-attention, material-conditioning, and cross-attention weights. Each exchange point owns only:

- one learned 64-dimensional stage embedding;
- one scalar particle-feedback gate.

The shared module also owns five learned frame-position embeddings. These identify the chronological order of the observed state tokens; temporal attention must not rely only on displacement features to infer frame order.

Sharing is required to limit parameter growth and to make the four calls iterative refinements of one state operator rather than four unrelated modules.

## 5. Frame Partitioning

For the mm3 single-frame baseline, `mask_cond=true`. The hidden frame sequence is therefore:

```text
[mask pseudo-frame, five physical history frames, one prediction frame]
```

The implementation must retain both counts separately:

- `history_frames = 5`: physical frames used to build state;
- `cond_frames_total = 6`: physical frames plus the mask pseudo-frame used by the existing output slicing logic.

State pooling must select only the five physical history frames. It must never pool the mask pseudo-frame or prediction frame.

The state module is initially scoped to `output_frames=1`. Unsupported frame layouts must raise a descriptive error rather than silently selecting the wrong frames.

## 6. Explicit Frame State

For each observed point cloud `X_t` in normalized coordinates, compute:

```text
mu_t       = mean_i X_t,i
Sigma_t    = mean_i (X_t,i - mu_t)(X_t,i - mu_t)^T
dmu_t      = mu_t - mu_0
vmu_t      = mu_t - mu_(t-1)
dSigma_t   = Sigma_t - Sigma_(t-1)
```

For `t=0`, set `vmu_0` and `dSigma_0` to zero. The explicit state is:

```text
s_t = [dmu_t(3), vmu_t(3), upper_triangle(Sigma_t)(6),
       upper_triangle(dSigma_t)(6)]
```

This gives 18 values per observed frame. The statistics are computed only from `init_pc_cond`; no target position or predicted position is used.

The 18-dimensional state is encoded by a small MLP into the 64-dimensional state space.

## 7. Learned Frame Summary

After stage `k`, let `H_obs^k` be the particle hidden states of the five physical history frames, with shape `(B, 5, N, 256)`.

For each frame:

```text
score_t,i = Linear(LayerNorm(H_t,i))
a_t       = softmax(score_t, dim=particles)
p_t       = sum_i a_t,i H_t,i
```

`p_t` is projected from 256 to 64 dimensions. The scalar attention pooling is intentionally smaller than Transolver-style multi-slice tokenization because the hypothesis concerns one whole-object state per frame, not multiple spatial regions.

## 8. Material-Conditioned State Update

The dataset already provides:

```text
m = [log10(E), nu]
```

An MLP maps `m` to a 64-dimensional material context. The existing categorical `mat_type` class token remains unchanged and is not duplicated in the new material path.

At stage `k`, construct the state input:

```text
u_t^k = Wp(p_t^k) + Ws(s_t) + Wm(m)
        + frame_embedding_t + stage_embedding_k
```

The previous state tokens are retained between stages. A shared temporal self-attention and FFN update the five-token sequence:

```text
G^k = StateBlock(G^(k-1) + U^k; material=m)
```

`G^0` is a zero tensor. Stage 1 therefore constructs its state entirely from the first learned summaries, explicit statistics, material context, and frame/stage embeddings. Later stages refine the retained state with deeper particle representations.

The state block uses material-conditioned normalization so that `E/nu` affects state evolution at every exchange point, rather than appearing only as two initial condition tokens.

There is no causal mask over the five state tokens because all five frames are observed history at inference time.

## 9. State-to-Particle Feedback

Only the single prediction-frame particle features query the state sequence:

```text
Q = MaterialFiLM(LayerNorm(H_out^k), m)
M = CrossAttention(query=Q, key=G^k, value=G^k)
H_out^k <- H_out^k + beta_k * Wo(M)
```

The four scalar `beta_k` gates are initialized to zero. Therefore, before training updates the gates, the particle path is numerically equivalent to the unchanged baseline backbone.

Observed physical frames and the mask pseudo-frame are not modified by this feedback. Later backbone stages may still propagate the updated prediction representation through the original temporal attention.

## 10. Module Interface

The intended conceptual interface is:

```python
state_tokens, hidden_states = state_exchange(
    hidden_states=hidden_states,
    state_tokens=state_tokens,
    explicit_frame_state=explicit_frame_state,
    material_context=material_context,
    physical_history_slice=physical_history_slice,
    prediction_slice=prediction_slice,
    stage_index=stage_index,
)
```

Responsibilities are separated as follows:

- `MDM_ST`: compute leakage-safe explicit state and continuous material input;
- `SpaitalTemporalTransformer`: own stage scheduling and persistent state tokens;
- `HybridStateExchange`: pool, update state, and feed state back to prediction particles;
- existing serial block: unchanged.

The new arguments must be optional for all existing block variants. Existing configurations must preserve their current execution paths.

## 11. Parameter and Compute Control

Initial settings:

```text
backbone latent dimension: 256
state dimension:           64
state tokens:              5
state attention heads:     4
exchange module copies:    1 shared copy
exchange calls:            4
```

The implementation must report:

- total trainable parameters;
- absolute and percentage parameter increase over the mm3 baseline;
- approximate exchange FLOPs or measured iteration-time overhead.

The target is less than 1% additional parameters. v11a does not reduce the backbone FFN to force exact matching because that would change two mechanisms at once. If v11a is positive, `v11a-PM` will redistribute the parameter budget in a separate confirmation experiment.

## 12. Configuration

The planned configuration names are:

```text
src/configs/config_mm3_v11a_mc_hst_8L.yaml
src/configs/eval_mm3_v11a_mc_hst_8L.yaml
```

The training config must copy the baseline and change only:

- `output_dir`;
- the architecture selector;
- explicit v11a state-module settings.

The eval config must mirror all model and dataset settings and differ only in checkpoint and visualization paths.

## 13. Verification

Implementation is not complete until all of the following pass:

1. Python syntax compilation.
2. CPU parameter counting for baseline and v11a.
3. Forward shape test for `B=1`, five history frames, one output frame, and 2048 particles.
4. Backward test with and without gradient checkpointing.
5. Frame-selection test proving that the mask pseudo-frame and prediction frame are excluded from state pooling.
6. Leakage test proving that explicit state depends only on `init_pc_cond`.
7. Zero-gate equivalence test against the baseline with shared backbone weights.
8. Existing baseline config merge and forward path remain valid.

## 14. Experimental Protocol

The screening comparison uses identical:

- mm3 training and test data;
- random-window sampling;
- seed;
- losses and loss weights;
- optimizer and learning-rate schedule;
- batch and gradient accumulation;
- training steps;
- checkpoint step;
- evaluation code and test split.

Report:

- one-step MSE;
- full-rollout MSE;
- Chamfer distance;
- per-absolute-frame MSE;
- velocity and acceleration MSE;
- volume error and drift;
- floor penetration;
- metrics separately for elastic, plasticine, and sand;
- metrics stratified by `E/nu` ranges when the test data supports meaningful bins.

One seed is sufficient only for screening. A positive result must be repeated with at least three seeds before it supports a publication claim.

## 15. Decision Rules

- If v11a improves both one-step and full-rollout MSE without materially worsening physical metrics, proceed to multi-seed confirmation and `v11a-PM`.
- If v11a improves material-stratified or `E/nu` generalization but not aggregate MSE, inspect whether the gain is concentrated in underrepresented regimes before changing the architecture.
- If v11a only improves training loss, reject the mechanism as overfitting.
- If v11a is negative, do not immediately add global-state supervision. First determine whether state tokens carry predictive information using diagnostic probes.
- Only after v11a is established should `v11b` supervision or `v11c` state-conditioned temporal attention be designed and tested.

## 16. Main Risks

1. The original global spatial attention may already infer equivalent whole-object information, making the state path redundant.
2. COM and covariance may be too coarse for plasticine and sand dynamics.
3. Material class and `E/nu` may be statistically confounded in the current dataset, so apparent material gains may not be continuous-parameter generalization.
4. A zero feedback gate delays gradients into the exchange branch during the first update, although it provides an exact baseline initialization.
5. The test objects for different materials are not object-matched; cross-material comparisons must therefore be reported as stratified performance, not paired object-level causality.
