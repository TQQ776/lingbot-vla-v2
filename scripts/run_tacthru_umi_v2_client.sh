#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TACTHRU_REPO="${TACTHRU_REPO:-$(cd "$PROJECT_ROOT/../tacthru" && pwd)}"
REALMAN_PYTHON="${REALMAN_PYTHON:-$TACTHRU_REPO/.venv-realman/bin/python}"

if [[ ! -x "$REALMAN_PYTHON" ]]; then
  echo "Expected TacThru Python 3.11 Realman environment at: $REALMAN_PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$TACTHRU_REPO:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
exec "$REALMAN_PYTHON" -m deploy.tacthru_umi_v2.realman_client "$@"
