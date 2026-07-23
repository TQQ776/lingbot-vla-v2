#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${TACTHRU_ZARR_PIPELINE_DIR:-$ROOT_DIR/tools/tacthru_zarr_pipeline}"
PYTHON="${TACTHRU_ZARR_PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "LingBot V2 project Python is missing: $PYTHON" >&2
  exit 1
fi

ENTRYPOINT="$PIPELINE_DIR/scripts/process/run.py"
if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "TacThru Zarr conversion entrypoint is missing: $ENTRYPOINT" >&2
  exit 1
fi

export PYTHONPATH="$PIPELINE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$ENTRYPOINT" "$@"
