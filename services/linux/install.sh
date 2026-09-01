#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
MODE=broker
ENABLE=1

while (($#)); do
  case $1 in
    --direct) MODE=direct ;;
    --no-enable) ENABLE=0 ;;
    --help|-h)
      printf 'Usage: services/linux/install.sh [--direct] [--no-enable]\n'
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

SERVICE_USER=${SUDO_USER:-$USER}
SERVICE_GROUP=$(id -gn "$SERVICE_USER")
[[ -x "$REPO_ROOT/.venv/bin/qwen-omni-daemon" ]] || {
  printf 'Run ./scripts/bootstrap.sh before installing the service.\n' >&2
  exit 1
}
mkdir -p "$REPO_ROOT/runtime-data"
chmod 700 "$REPO_ROOT/runtime-data"

if [[ $MODE == broker ]]; then
  docker gpu discover >/dev/null
  EXEC_START="$REPO_ROOT/portal/start.sh --foreground"
else
  if docker gpu discover >/dev/null 2>&1; then
    printf 'Refusing --direct: ollama-unify is present; broker mode is mandatory on this host.\n' >&2
    exit 1
  fi
  EXEC_START="$REPO_ROOT/.venv/bin/qwen-omni-daemon serve --allow-direct-gpu"
fi

temporary=$(mktemp)
trap 'unlink "$temporary" 2>/dev/null || true' EXIT
sed \
  -e "s|@SERVICE_USER@|$SERVICE_USER|g" \
  -e "s|@SERVICE_GROUP@|$SERVICE_GROUP|g" \
  -e "s|@REPO_ROOT@|$REPO_ROOT|g" \
  -e "s|@EXEC_START@|$EXEC_START|g" \
  "$SCRIPT_DIR/qwen-omni-adapters.service.in" >"$temporary"
sudo install -m 0644 "$temporary" /etc/systemd/system/qwen-omni-adapters.service
sudo systemctl daemon-reload
if ((ENABLE)); then
  sudo systemctl enable --now qwen-omni-adapters.service
fi
printf 'Installed qwen-omni-adapters.service (%s mode).\n' "$MODE"
printf 'Status: sudo systemctl status qwen-omni-adapters.service\n'
printf 'Portal: %s/portal/start.sh --status\n' "$REPO_ROOT"
