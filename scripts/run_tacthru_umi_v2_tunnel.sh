#!/usr/bin/env bash
set -euo pipefail

# Keep the tunnel in the foreground so Ctrl-C is an immediate, reversible
# rollback.  The SSH host alias is resolved from ~/.ssh/config; do not put a
# LocalForward in the shared Host block because it can conflict with other
# local services.
SSH_TARGET="${LINGBOT_V2_SSH_TARGET:-172.16.41.254}"
LOCAL_PORT="${LINGBOT_V2_LOCAL_PORT:-18081}"
REMOTE_PORT="${LINGBOT_V2_REMOTE_PORT:-18081}"

if ! [[ "${LOCAL_PORT}" =~ ^[1-9][0-9]*$ && "${REMOTE_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LINGBOT_V2_LOCAL_PORT and LINGBOT_V2_REMOTE_PORT must be positive integers" >&2
  exit 2
fi

if ss -H -lnt "sport = :${LOCAL_PORT}" 2>/dev/null | grep -q .; then
  echo "Local TCP port ${LOCAL_PORT} is already listening; refusing to replace it." >&2
  echo "Inspect it with: ss -lntp '( sport = :${LOCAL_PORT} )'" >&2
  exit 2
fi

effective_config="$(ssh -G "${SSH_TARGET}")"
ssh_user="$(awk '$1 == "user" { print $2; exit }' <<<"${effective_config}")"
ssh_host="$(awk '$1 == "hostname" { print $2; exit }' <<<"${effective_config}")"
ssh_port="$(awk '$1 == "port" { print $2; exit }' <<<"${effective_config}")"

echo "[lingbot-v2-tunnel] forwarding 127.0.0.1:${LOCAL_PORT} -> 127.0.0.1:${REMOTE_PORT}" >&2
echo "[lingbot-v2-tunnel] SSH target ${ssh_user}@${ssh_host}:${ssh_port}" >&2
echo "[lingbot-v2-tunnel] Ctrl-C closes the tunnel; no project or server files are changed." >&2

exec ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${SSH_TARGET}"
