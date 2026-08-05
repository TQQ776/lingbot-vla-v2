# TacThru UMI V2 history8 training

## Contract

Marker training uses all 201 episodes at 30 Hz. Each stored row is one
normalized `[48,2]` displacement frame. The dataloader constructs the eight
same-episode frames ending at the current observation. Training and deployment
must use the same marker calibration, fixed marker ID order, image dimensions,
normalization formula and 30 Hz sampling rate.

Required marker configuration:

```yaml
marker_history_length: 8
marker_sample_hz: 30.0
marker_input_features: 2
marker_feature_mode: displacement_history
marker_temporal_embedding: true
marker_displacement_keys:
  - observation.tactile.marker_displacement_left
marker_valid_mask_keys:
  - observation.tactile.marker_valid_left
```

## Build statistics

From the repository root:

```bash
.uv/convert-venv/bin/python tools/compute_tacthru_marker_stats.py \
  /mnt/models/VTLA-RDT/tacthru/data/tasks/InsertEthernetCable/insert_ethernet_cable_ml_0721_201.zarr.zip \
  assets/norm_stats/insert_ethernet_cable_ml_0721_201_vtla_marker.json \
  --train-episodes 0:201
```

The output must have exactly two values in `marker_mean` and `marker_std`.
Padding replication is not counted. A four-channel file from the old explicit
velocity model is incompatible.

## Convert all 201 episodes

Check the source without writing output:

```bash
.uv/convert-venv/bin/python tools/convert_tacthru_zarr_to_lerobot_v2.py \
  /mnt/models/VTLA-RDT/tacthru/data/tasks/InsertEthernetCable/insert_ethernet_cable_ml_0721_201.zarr.zip \
  data/lerobot/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_history8_v2 \
  --include-tactile \
  --max-source-episodes 201 \
  --task 'Insert the Ethernet cable' \
  --check-only
```

Run the same command without `--check-only` in the LeRobot training environment
to create the dataset. The manifest must declare:

```text
marker_representation: normalized_displacement
formula: 2 * (current_xy - reference_xy) / [image_width, image_height]
marker_order: fixed
history_storage: per_frame_displacement
history_construction: dataset_same_episode_offsets
history_length: 8
history_padding: earliest_valid_frame_replication
marker_sample_hz: 30
```

If the complete converter-v5 `tactile_v1` dataset already exists, its videos,
state/action rows and normalized marker values are identical. Reuse those video
files without re-encoding them and migrate only the marker column/schema:

```bash
.venv/bin/python tools/convert_tacthru_zarr_to_lerobot_v2.py \
  data/zarr/insert_ethernet_cable_ml_0721_201.zarr.zip \
  data/lerobot/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_history8_v2 \
  --include-tactile \
  --max-source-episodes 201 \
  --task 'Insert the Ethernet cable' \
  --reuse-existing-dataset \
  data/lerobot/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_v1
```

This mode validates episode/frame counts, hard-links unchanged files, atomically
renames `marker_flow_left` to `marker_displacement_left` in metadata/parquet,
and then runs the same full output validation. The legacy source is not modified.

## Train

Marker only:

```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_marker.yaml
```

Tactile RGB plus marker history:

```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml
```

Both configs point to the `tactile_history8_v2` dataset and distinct history8
output directories, preventing an accidental resume from the incompatible
single-token checkpoint. No 20-episode validation split is reserved.

## Validation before deployment

Run the offline tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
  .uv/convert-venv/bin/python -m pytest \
  tests/test_tactile_vtla.py \
  tests/test_tacthru_umi_v2_converter.py \
  tests/test_tacthru_umi_v2_tactile_source.py \
  tests/test_tacthru_umi_v2_deploy_protocol.py \
  tests/test_tacthru_umi_v2_http_e2e.py -q
```

A newly trained history8 checkpoint should strict-load without any compatibility
flag. `LINGBOT_V2_ALLOW_LEGACY_MARKER_REINIT=1` is only an explicit structural
migration for an old one-token checkpoint; those reinitialized marker weights
still require history8 training before meaningful deployment.

No real-arm command is part of this training procedure.
