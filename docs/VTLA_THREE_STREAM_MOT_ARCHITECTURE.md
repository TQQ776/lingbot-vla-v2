# VTLA Three-Stream Cascaded-MoT Architecture

This document reports the implementation derived from
`lingbot_192token_to_trex_three_stream_mot_prompt.md` on the
`marker_token_192` baseline.

## Data Flow

```mermaid
flowchart TD
    RGB[Scene RGB] --> VIT[Shared Qwen3-VL ViT]
    TRGB[TacRGB] --> VIT
    LANG[Language and task queries] --> P[Prefix / Qwen3-VL stream, 36 layers]
    VIT --> P
    P --> KVP[Cache Prefix K/V]

    STATE[State] --> A[Action Expert stream, 36 layers]
    NOISE[X1] --> A
    KVP --> A
    A -->|Euler at 1.0, 0.9, ..., 0.5| XS[X0.4]
    XS --> REFRESH[Re-run Action at tau=0.4 and X0.4]
    STATE --> REFRESH
    KVP --> REFRESH
    REFRESH --> KVA[Cache Action K/V at boundary]

    MARKER[1 sensor x 4 frames x 48 markers] --> ME[Existing Marker encoder]
    ME --> MP[marker_to_tactile_proj]
    MP --> TS[Marker192 | tau1 | X_tau50]
    XS --> TS
    KVP --> T[Tactile Expert stream, 36 layers]
    KVA --> T
    TS --> T
    T --> LAST[Last 50 tactile hidden states]
    LAST --> HEAD[Linear 768 to 55]
    HEAD -->|Euler at 0.4, 0.3, 0.2, 0.1| X0[X0 action chunk]
```

The per-layer cross-stream visibility is:

| Query / K-V | Prefix | Action | Tactile |
|---|---:|---:|---:|
| Prefix | yes | no | no |
| Action | yes | yes | no |
| Tactile | yes | yes | yes |

The Prefix-Prefix and Action-Action submatrices are copied from the legacy
two-stream mask. Only the new cross-stream blocks are added.

## Tensor Contract

| Stream | Input shape | Hidden size |
|---|---|---:|
| Prefix | `[B, P, D_vlm]` | Qwen3-VL hidden size |
| Action | `[B, 51, 768]` (`State1 + Action50`) | 768 |
| Tactile | `[B, 243, 768]` | 768 |

The Tactile layout is fixed by configuration rather than absolute joint-sequence
indices:

```text
[0, 192)   existing Marker observation tokens
[192, 193) independent tactile flow-time token
[193, 243) projected X_tau action tokens
```

Existing Marker temporal/spatial sin/cos encoding and reference XY content are
retained. Transformer M-RoPE sequence positions continue in the order Prefix,
Action, Tactile and do not replace Marker content positions.

## Slow and Fast Schedules

Slow Action Euler evaluations:

```text
tau = 1.0, 0.9, 0.8, 0.7, 0.6, 0.5
X1 -> X0.9 -> X0.8 -> X0.7 -> X0.6 -> X0.5 -> X0.4
```

After the final update, Action is evaluated again using exactly
`state + tau=0.4 + X0.4`. Its per-layer K/V is appended to the immutable Prefix
cache. Therefore Fast never reads stale Action K/V from `X0.7`.

Fast Tactile Euler evaluations:

```text
tau = 0.4, 0.3, 0.2, 0.1
X0.4 -> X0.3 -> X0.2 -> X0.1 -> X0
```

Every Fast call clones the cache containers and starts from a clone of the same
`CascadedSlowPlan.x_split`. It does not feed the previous Fast result back as a
new boundary.

Marker encoding and contact routing happen once before the Fast Euler loop.
Gate ON reuses the encoded Marker tokens for all six Tactile steps. Gate OFF
skips the Tactile Expert entirely and runs only the Action fallback. A mixed
batch keeps static shapes and evaluates both lower-flow branches before the
per-sample selection; the real-robot batch-one path never pays for both.

## Training

Action cascade training samples `tau_A ~ U(0.4, 1)` and constructs
`X_tau = tau * noise + (1 - tau) * action` analytically. Tactile training samples
`tau_T ~ U(0, 0.4)` and uses the same Flow Matching target `noise - action`.

The Tactile branch builds Action K/V at the analytic `X0.4`. With probability
`0.25`, a no-gradient Slow Action rollout supplies the boundary and a tactile
loss is evaluated at `tau=0.4`. This improves boundary exposure but does not
claim to eliminate the complete rollout distribution mismatch.

The first-stage YAML freezes Qwen3-VL and Action Expert. Trainable modules are:

- existing Marker encoder parameters;
- `marker_to_tactile_proj`;
- independent 36-layer Tactile Expert;
- `tactile_time_embedder`;
- tactile Action input/output projections.

In this frozen-Action stage, the Action cascade loss remains a detached
monitoring metric but is not added to the optimization tensor. If Action is
unfrozen, configuration validation requires a positive full-range Action loss
weight so the gate-off `0.4 -> 0` fallback is not silently forgotten.

Resolved parameter counts for the configured Qwen3-VL hidden size 2560 were
computed on the `meta` device (shape allocation only):

| Trainable component | Parameters |
|---|---:|
| Existing tactile/Marker encoder | 1,396,928 |
| 36-layer Tactile Expert | 410,796,288 |
| Marker to Tactile projection | 1,966,848 |
| Tactile time embedder | 1,181,184 |
| Tactile Action input/output heads | 85,303 |
| Total first-stage tactile trainable | 415,426,551 |

Depth, future-depth, future-video, Action MoE metrics, and router losses remain
in the original return path. The optional full-range Action auxiliary loss is
implemented but disabled in the frozen-Action first-stage configuration; the
pretrained Action Expert retains its original full-range ability.

## Contact Gate and Legacy Mode

TacRGB stays in Prefix because `gate_tactile_rgb=false` and the gate target is
`marker_only`. When contact is off, inference runs Action Expert from `X0.4` to
`X0`; it does not execute an all-masked Tactile sequence. Mixed batches select
Tactile or Action fallback per sample.

When `tactile_refinement.enabled=false`, Marker remains in Prefix and the exact
legacy VLM+Action two-stream path is used.

Old 192-token checkpoints can be loaded with `strict=False`; original VLM,
Action, and Marker names are unchanged and new Tactile parameters are expected
missing keys. The project weight loader recognizes those missing namespaces and
copies shape-compatible loaded Action decoder/projection values into independent
Tactile parameters. A newly migrated checkpoint still requires Tactile training
before deployment.

## Online Slow/Fast Scheduler

The deployment policy now owns one Slow plan per active robot session. The
default `LINGBOT_V2_VTLA_SCHEDULER=auto` path makes a three-level decision before
every batch-one request:

| Decision | Computation |
|---|---|
| `reuse` | Reuse Prefix+Action K/V and run Fast tactile/fallback only |
| `refresh_action` | Reuse Prefix K/V, rerun Action `1 -> 0.4`, refresh boundary K/V, then Fast |
| `rebuild` | Re-encode RGB/TacRGB/language, rebuild Prefix and Action Slow plan, then Fast |

Validity is not a single 55-D norm. It independently checks raw 8-D robot
position drift, quaternion rotation drift, gripper drift, plan age, executed
action offset, instruction, and an optional explicit scene version. The Realman
client sends cumulative `executed_offset`; callers may additionally send:

Action refresh updates the Action-plan timestamp but preserves the original
Prefix timestamp. Therefore repeated Action-only refreshes cannot keep stale
RGB/TacRGB Prefix K/V alive indefinitely.

```text
metadata.vtla_mode = auto | slow | fast | slow_and_fast
metadata.scene_version = non-negative integer
metadata.executed_offset = non-negative integer
```

`fast` is strict: it fails instead of using a stale/missing Slow plan. `slow`
and `slow_and_fast` force a full rebuild followed by Fast refinement. Set
`LINGBOT_V2_VTLA_SCHEDULER=legacy` before server startup to retain the original
one-request full `sample_actions()` path.

Thresholds are configurable without changing model weights:

```text
LINGBOT_V2_VTLA_REUSE_MAX_AGE_S
LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_AGE_S
LINGBOT_V2_VTLA_REUSE_MAX_POSITION_DRIFT_M
LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_POSITION_DRIFT_M
LINGBOT_V2_VTLA_REUSE_MAX_ROTATION_DRIFT_RAD
LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_ROTATION_DRIFT_RAD
LINGBOT_V2_VTLA_REUSE_MAX_GRIPPER_DRIFT_M
LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_GRIPPER_DRIFT_M
LINGBOT_V2_VTLA_REUSE_MAX_EXECUTED_OFFSET
LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_EXECUTED_OFFSET
```

Every response reports the selected level/reason, drift metrics, plan version,
and wall-clock `slow_s`, `action_refresh_s`, and `fast_s` fields under
`metadata.vtla_scheduler`.

## Files

- `tactile_action_expert.py`: validated configuration, independent expert
  construction, Slow plan, cache cloning, copy initialization.
- `modeling_lingbot_vla_v2.py`: three-stream Joint Attention, 243-token builder,
  training, Slow/refresh/Fast inference, gate fallback, public online APIs.
- `utils.py`: exact P/A/T visibility matrix while preserving legacy submasks.
- `configuration_lingbot_vla.py`: serialized model configuration and first-stage
  shape validation.
- `optim/vtla.py` and `train_lingbotvla.py`: optimizer groups, freezing and
  runtime metadata.
- `test_tactile_three_stream_mot.py`: architecture and behavioral tests.
- `slow_fast_scheduler.py`: pure raw-state validity policy for cached plans.
- `lingbot_vla_v2_policy.py` and `http_server.py`: stateful online scheduler,
  response profiling, and backward-compatible request routing.
- `test_tactile_slow_fast_scheduler.py` and
  `test_tactile_online_scheduler_policy.py`: validity and policy routing tests.

## Verification and Profiling

Unit tests verify the 243-token shape, 36/36/36 layer contract, independent
parameters, mask matrix, Marker sensitivity, deterministic Fast behavior,
immutable boundary/cache semantics, gate fallback, schedule, checkpoint
compatibility, first-stage gradients, protocol metadata, and online plan
routing. The model/deployment and legacy VTLA regression groups report
`138 passed`; two existing PyTorch DCP deprecation warnings remain.

Slow and Fast latency must still be profiled on the target GPU with the actual
6B checkpoint. The response now exposes the required stage timings, but this
development environment does not contain a trained three-stream checkpoint, so
it cannot report representative latency or memory usage. The roughly 415M new
tactile-side parameters remain randomly/copy initialized until a `three_vtla`
checkpoint is trained; code and unit-test completion does not make the current
baseline checkpoint robot-ready.
