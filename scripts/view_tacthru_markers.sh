#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TACTHRU_REPO="${TACTHRU_REPO:-$(cd "$PROJECT_ROOT/../tacthru" && pwd)}"
REALMAN_PYTHON="${REALMAN_PYTHON:-$TACTHRU_REPO/.venv-realman/bin/python}"
TACTHRU_SENSOR_CONFIG="${TACTHRU_SENSOR_CONFIG:-$TACTHRU_REPO/cfg/sensor/ml.yaml}"

if [[ ! -x "$REALMAN_PYTHON" ]]; then
  echo "Expected TacThru Python environment at: $REALMAN_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TACTHRU_SENSOR_CONFIG" ]]; then
  echo "TacThru sensor config not found: $TACTHRU_SENSOR_CONFIG" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$TACTHRU_REPO:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
exec "$REALMAN_PYTHON" tools/view_tacthru_markers.py \
  --tacthru-repo "$TACTHRU_REPO" \
  --sensor-config "$TACTHRU_SENSOR_CONFIG" \
  "$@"
