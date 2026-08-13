# VTLA slow/fast inference cache

This inference-only path is based on the `marker_token_192` contract:

- one TacThru sensor;
- four marker-history frames;
- 48 point-spatiotemporal tokens per frame (`4 x 48 = 192`);
- one current TacThru RGB frame;
- causal Qwen3-VL prefix attention;
- `marker_only` contact gating.

Training and the original `sample_actions()` remain unchanged.

## Data flow

```text
Slow refresh
  Scene RGB + TacThru RGB -> shared Qwen3-VL ViT
  Scene -> Language -> TacThru RGB -> 36-layer VLM KV_SLOW

Fast replan
  latest Marker[t-3:t] -> PointSpatiotemporalMarkerEncoder -> 192 tokens
  192 Marker -> Task Query | KV_SLOW -> disposable full-prefix KV
  State + noisy action + FM time -> Action Expert -> Flow Matching
```

State, noisy actions, and FM time still enter only through `embed_suffix()`.
They never enter the VLM prefix.

## Cache lifecycle

`POST /context/refresh` constructs a complete pending slow cache and atomically
replaces the active server context only after construction succeeds. A refresh
always rebuilds `Scene + Language + latest TacThru RGB`; RGB caches are never
appended over time.

`POST /predict_fast` contains no RGB. Every call encodes the latest Marker
history and task queries into a new working cache based on the immutable slow
KV. The working cache is discarded after the action chunk is generated. Marker
tokens therefore never accumulate in the permanent slow cache.

The active context is invalidated when:

- `session_id` changes;
- an episode reset is received;
- the legacy `/predict` endpoint resets the episode;
- a newer slow context completes and replaces it.

MarkerContactGate remains a sensor-side episode state machine. Fast replanning
does not reset it, and gate OFF masks Marker tokens without clearing TacThru RGB
from the slow context.

## Position IDs

The slow cache stores its input IDs, padding mask, visual grid, and mRoPE
positions. Fast continuation rebuilds metadata in the full logical order:

```text
Scene -> Language -> TacThru RGB -> Marker -> Task Query
```

Qwen3-VL computes full prefix position IDs from that metadata. Only the
Marker/Query suffix slice is used for continuation, so dynamic positions never
restart at zero. The causal mask is also built for the full prefix before its
dynamic query rows are selected.

## Realman rollout

The real-robot client enables this path automatically when `/health` reports a
compatible checkpoint. The default executable window is `[2,4)`, so two
waypoints are used before the next Marker/state replan. Disable the path with:

```bash
export LINGBOT_V2_SLOW_FAST_CACHE=0
```

By default RGB is refreshed only at episode start. Profiling should determine
the deployment refresh rate. An explicit experimental interval is available:

```bash
--slow-refresh-every N
```

## Profiling

Slow context metadata reports:

- `rgb_preprocess_ms`
- `scene_tacrgb_vit_ms`
- `slow_vlm_ms`
- `slow_cache_build_ms`

Fast response/log metadata reports:

- `fast_preprocess_ms`
- `marker_mlp_ms`
- `task_query_ms`
- `marker_query_continuation_ms`
- `fm_sampling_ms`
- `fast_replan_total_ms`
- `http_encode_ms`
- `network_ms`
- `server_total_ms`
- `robot_execution_ms`
- `slow_context_age_ms`

No target-GPU checkpoint profiling was run as part of this code change. These
fields must be measured on the deployment GPU before claiming a replan rate or
choosing an RGB refresh frequency.

## Current concurrency limitation

Context replacement is atomic, but GPU model execution is serialized by the
backend inference lock. A slow refresh cannot expose a partial cache, although
a fast request arriving during the build waits for that build to complete. True
overlapped slow-refresh and fast inference would require a separately validated
CUDA stream/concurrency design and is intentionally not enabled here.
