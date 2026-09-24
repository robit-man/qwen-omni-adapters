#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
MODE=auto
ENABLE=1
HARNESS=0

while (($#)); do
  case $1 in
    --auto) MODE=auto ;;
    --broker) MODE=broker ;;
    --direct) MODE=direct ;;
    --no-enable) ENABLE=0 ;;
    --with-harness) HARNESS=1 ;;
    --help|-h)
      printf 'Usage: services/linux/install.sh [--auto|--broker|--direct] [--no-enable] [--with-harness]\n'
      printf '  --with-harness  also install the always-listening call harness as a user service\n'
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# NVIDIA Tegra modules have an integrated GPU and no ollama-unify broker, so
# the previous broker default could only ever fail there. Auto resolves to
# whichever mode the host can actually honour; --broker/--direct still pin it.
if [[ $MODE == auto ]]; then
  if command -v docker >/dev/null 2>&1 && docker gpu discover >/dev/null 2>&1; then
    MODE=broker
  else
    MODE=direct
  fi
  printf 'Auto-selected %s mode for this host.\n' "$MODE"
fi

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
if [[ $MODE == broker ]]; then
  printf 'Portal: %s/portal/start.sh --status\n' "$REPO_ROOT"
else
  printf 'Portal: %s/.venv/bin/qwen-omni-daemon status\n' "$REPO_ROOT"
fi

if ((HARNESS)); then
  # A user service, not a system one: the harness needs the desktop session it
  # speaks into -- its audio devices, its top bar, and the USB permissions the
  # logged-in user already has. Installing it system-wide would have none of
  # those and would listen on behalf of nobody.
  for dependency in parec paplay pactl; do
    command -v "$dependency" >/dev/null 2>&1 || {
      printf 'Missing desktop harness dependency: %s (rerun scripts/bootstrap.sh --with-harness).\n' \
        "$dependency" >&2
      exit 1
    }
  done
  # Probe in the user manager, which is also where the persistent harness runs.
  # An SSH installer normally has no DISPLAY and an X-forwarded DISPLAY would
  # disappear when SSH exits; neither describes the logged-in desktop session.
  systemd-run --user --wait --pipe --quiet --collect --service-type=exec \
    --unit="qwen-omni-indicator-probe-$$" \
    --working-directory="$REPO_ROOT" \
    "$REPO_ROOT/.venv/bin/python" -c \
    'from harness.indicator import probe_indicator; print(probe_indicator())' \
    >/dev/null || {
    printf 'The graphical user service cannot initialize a desktop AppIndicator; log into the desktop and rerun the deployment.\n' >&2
    exit 1
  }
  harness_unit="$HOME/.config/systemd/user/omni-call-harness.service"
  mkdir -p "$(dirname "$harness_unit")"
  harness_tmp=$(mktemp)
  sed \
    -e "s|@REPO_ROOT@|$REPO_ROOT|g" \
    -e "s|@HARNESS_EXEC_START@|$REPO_ROOT/.venv/bin/python -m harness|g" \
    "$SCRIPT_DIR/omni-call-harness.service.in" >"$harness_tmp"
  install -m 0644 "$harness_tmp" "$harness_unit"
  unlink "$harness_tmp" 2>/dev/null || true
  if [[ -z ${SSH_CONNECTION:-} ]]; then
    desktop_variables=()
    for variable in \
      DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_CURRENT_DESKTOP \
      DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR; do
      [[ -n ${!variable:-} ]] && desktop_variables+=("$variable")
    done
    if ((${#desktop_variables[@]})); then
      systemctl --user import-environment "${desktop_variables[@]}"
    fi
  fi
  systemctl --user daemon-reload
  if ((ENABLE)); then
    systemctl --user enable --now omni-call-harness.service
  fi
  printf 'Installed omni-call-harness.service (user).\n'
  printf 'Status: systemctl --user status omni-call-harness.service\n'
fi
