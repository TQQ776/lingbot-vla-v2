#!/usr/bin/env bash

# Shared validation for the tactile launcher and the wrist-compatible base
# launcher. Keep experiment-defining values out of the free-form training
# override tail so tactile_experiment.json remains authoritative.
tactile_validate_train_overrides() {
  local override override_key
  for override in "$@"; do
    override_key="${override%%=*}"
    case "$override_key" in
      --model.model_path|--model.tokenizer_path|--data.data_name|--data.train_path|\
      --data.robot_config_root|--data.norm_stats_file|--data.tactile_rgb_key|\
      --data.tactile_marker_key|--data.tactile_marker_valid_key|\
      --data.tactile_timestamp_key|--train.output_dir|--train.align_params|\
      --train.tactile_rgb_enabled|--train.tactile_marker_enabled|\
      --train.tactile_params|--train.tactile_train_stage|\
      --train.tactile_experiment_contract_sha256|\
      --train.tactile_dataset_manifest_sha256|\
      --train.tactile_rgb_backbone_sha256|\
      --eval.force_mask_tactile_rgb|--eval.force_mask_tactile_marker)
        echo "Protected tactile launcher option cannot be overridden after --: $override" >&2
        echo "Use --tactile-mode/--tactile-stage or the documented environment variables." >&2
        return 2
        ;;
    esac
  done
}
