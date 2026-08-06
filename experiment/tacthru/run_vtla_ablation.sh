#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${1:?Usage: run_vtla_ablation.sh EXPERIMENT_NAME [training wrapper options]}"
shift
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
RESOLVED="$ROOT_DIR/.deployment/ablations/${NAME}.yaml"

"$PYTHON" "$ROOT_DIR/tools/generate_vtla_ablation_config.py" "$NAME" "$RESOLVED"
DATASET="$($PYTHON - "$RESOLVED" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["data"]["train_path"])
PY
)"

CONFIG="$RESOLVED" bash "$ROOT_DIR/scripts/train_tacthru_umi_v2.sh" \
  "$ROOT_DIR/$DATASET" --skip-convert --skip-norm \
  --expected-source-episodes 201 \
  --output-dir "$ROOT_DIR/output/vtla_ablation_${NAME}" "$@"
