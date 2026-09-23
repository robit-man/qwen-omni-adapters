#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVICE_NAME=qwen-omni-adapters.service
PROFILE=""
ACTION=""
WITH_HARNESS=""
ASSUME_YES=0
DRY_RUN=0
ALLOW_UPDATE=1
ENV_BACKUP=""
ENV_EXISTED=0
CONFIG_INSTALLED=0
SERVICE_WAS_ACTIVE=0
SERVICE_EXISTED=0
SERVICE_START_EPOCH=0
DEPLOY_COMPLETE=0

restore_cursor() {
  if [[ -t 1 ]]; then
    printf '\033[?25h'
  fi
}

usage() {
  cat <<'EOF'
Usage: ./deploy.sh [profile] [options]

Run without arguments for the arrow-key guided Jetson installer.

Profiles:
  ornith15             Standard Ornith 1.5 9B audio bridge (about 8.15 GiB)
  qwen38               Qwen3.8 27B E03 Obliterated audio bridge (about 18.33 GiB)

Options:
  --profile NAME       Select a profile without opening the model menu
  --action ACTION      install, upgrade, deploy, or download
  --with-harness       Also install the always-listening user service
  --no-harness         Install only the core daemon/portal service
  --no-update          Do not fast-forward the checkout during an upgrade
  --yes                Accept the final confirmation (requires --profile)
  --dry-run            Print the resolved deployment plan without changing the host
  --list-models        Print the deployable bridge profiles
  --help               Show this help

The old full-Omni tags remain usable through explicit OMNI_MODEL settings, but
the guided installer deliberately offers only the two reduced audio bridges.
EOF
}

list_models() {
  cat <<'EOF'
ornith15|robit/ornith-1.5-omni-audio-bridge:q4km|8.15 GiB|Standard Ornith 1.5 9B; recommended on 32 GB Jetson
qwen38|robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km|18.33 GiB|Qwen3.8 27B E03 Obliterated; higher memory requirement
EOF
}

die() {
  printf 'deploy: %s\n' "$*" >&2
  exit 1
}

is_tegra() {
  [[ -r /proc/device-tree/compatible ]] \
    && tr '\0' '\n' </proc/device-tree/compatible 2>/dev/null \
      | grep -q '^nvidia,tegra[0-9]'
}

tegra_soc() {
  if ! is_tegra; then
    printf 'not detected'
    return
  fi
  tr '\0' '\n' </proc/device-tree/compatible 2>/dev/null \
    | grep -E '^nvidia,tegra[0-9]' | head -n 1 | cut -d, -f2
}

memory_gib() {
  local memory_kib
  memory_kib=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true)
  if [[ $memory_kib =~ ^[0-9]+$ ]]; then
    awk -v kib="$memory_kib" 'BEGIN {printf "%.1f", kib / 1024 / 1024}'
  else
    printf 'unknown'
  fi
}

runtime_installed() {
  [[ -x "$REPO_ROOT/.venv/bin/qwen-omni" ]] \
    && [[ -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server" ]] \
    && [[ -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-tts" ]]
}

service_installed() {
  command -v systemctl >/dev/null 2>&1 \
    && systemctl cat "$SERVICE_NAME" >/dev/null 2>&1
}

configured_model() {
  [[ -r "$REPO_ROOT/.env" ]] || return 0
  awk -F= '$1 == "OMNI_MODEL" {sub(/^[^=]*=/, ""); print; exit}' "$REPO_ROOT/.env"
}

deployment_state() {
  if service_installed; then
    printf 'managed service installed'
  elif runtime_installed; then
    printf 'runtime installed; service not installed'
  else
    printf 'fresh installation'
  fi
}

select_menu() {
  local result_name=$1
  local prompt=$2
  local default_index=$3
  shift 3
  local options=("$@")
  local index=$default_index
  local key rest line_count=${#options[@]} i

  [[ -t 0 && -t 1 ]] || die "guided menus require a terminal; pass --profile and --action"
  printf '\n%s\n' "$prompt"
  printf '\033[?25l'
  while true; do
    for ((i = 0; i < line_count; i++)); do
      printf '\r\033[2K'
      if ((i == index)); then
        printf '  \033[1;36m❯ %s\033[0m\n' "${options[$i]}"
      else
        printf '    %s\n' "${options[$i]}"
      fi
    done

    IFS= read -rsn1 key
    if [[ $key == $'\033' ]]; then
      rest=""
      IFS= read -rsn2 -t 0.2 rest || true
      key+=$rest
    fi
    case "$key" in
      $'\033[A'|k)
        ((index = (index - 1 + line_count) % line_count))
        ;;
      $'\033[B'|j)
        ((index = (index + 1) % line_count))
        ;;
      $'\033[H'|$'\033[1~')
        index=0
        ;;
      $'\033[F'|$'\033[4~')
        ((index = line_count - 1))
        ;;
      '')
        break
        ;;
      q|Q)
        printf '\033[?25h'
        exit 0
        ;;
      *)
        ;;
    esac
    printf '\033[%dA' "$line_count"
  done
  printf '\033[?25h'
  printf -v "$result_name" '%d' "$index"
}

resolve_profile() {
  case "$PROFILE" in
    ornith15|ornith15-audio-bridge|ornith15-bridge)
      PROFILE=ornith15
      OMNI_MODEL=robit/ornith-1.5-omni-audio-bridge:q4km
      PROFILE_TITLE='Standard Ornith 1.5 9B audio bridge'
      PROFILE_SIZE='8.15 GiB'
      ;;
    qwen38|qwen38-audio-bridge|qwen38-bridge)
      PROFILE=qwen38
      OMNI_MODEL=robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km
      PROFILE_TITLE='Qwen3.8 27B E03 Obliterated audio bridge'
      PROFILE_SIZE='18.33 GiB'
      ;;
    *)
      die "unknown profile '$PROFILE' (expected ornith15 or qwen38)"
      ;;
  esac

  # A trained-audio-bridge tag carries its sole language trunk as its standard
  # Ollama model layer. Naming another tag here would reintroduce the redundant
  # language runner that this release is designed to remove.
  OMNI_LANGUAGE_MODEL=$OMNI_MODEL
  export OMNI_PROFILE=$PROFILE OMNI_MODEL OMNI_LANGUAGE_MODEL
}

choose_action() {
  local selected
  if service_installed || runtime_installed; then
    select_menu selected 'Choose what to do (↑/↓, Enter):' 0 \
      'Upgrade checkout, runtime, and managed service (recommended)' \
      'Redeploy from this checkout without a Git update' \
      'Download and validate a model only' \
      'Exit'
    case $selected in
      0) ACTION=upgrade ;;
      1) ACTION=deploy ;;
      2) ACTION=download ;;
      3) exit 0 ;;
    esac
  else
    select_menu selected 'Choose what to do (↑/↓, Enter):' 0 \
      'Install runtime and managed service (recommended)' \
      'Download a model only' \
      'Exit'
    case $selected in
      0) ACTION=install ;;
      1) ACTION=download ;;
      2) exit 0 ;;
    esac
  fi
}

choose_profile() {
  local selected current
  current=$(configured_model)
  local default_index=0
  [[ $current == *qwen3.8-27b-e03-obliterated-omni-audio-bridge* ]] && default_index=1
  select_menu selected 'Select the logical Omni model (↑/↓, Enter):' "$default_index" \
    'Standard Ornith 1.5 9B bridge — 8.15 GiB (recommended for 32 GB)' \
    'Qwen3.8 27B E03 Obliterated bridge — 18.33 GiB (more capability, less headroom)'
  case $selected in
    0) PROFILE=ornith15 ;;
    1) PROFILE=qwen38 ;;
  esac
}

choose_harness() {
  local selected
  select_menu selected 'Select the service deployment (↑/↓, Enter):' 0 \
    'Core daemon and portal service' \
    'Core service plus always-listening desktop harness'
  ((selected == 1)) && WITH_HARNESS=1 || WITH_HARNESS=0
}

confirm_plan() {
  local selected
  printf '\nDeployment plan\n'
  printf '  Host:       %s (%s GiB unified/system memory)\n' "$(tegra_soc)" "$(memory_gib)"
  printf '  State:      %s\n' "$(deployment_state)"
  printf '  Action:     %s\n' "$ACTION"
  printf '  Model:      %s\n' "$PROFILE_TITLE"
  printf '  Ollama tag: %s\n' "$OMNI_MODEL"
  printf '  Weights:    %s (artifact total; runtime peak is higher)\n' "$PROFILE_SIZE"
  if [[ $ACTION != download ]]; then
    if ((WITH_HARNESS)); then
      printf '  Services:   core daemon + always-listening harness\n'
    else
      printf '  Services:   core daemon and portal\n'
    fi
  fi

  ((ASSUME_YES || DRY_RUN)) && return
  select_menu selected 'Proceed (↑/↓, Enter)?' 0 'Deploy this plan' 'Cancel'
  ((selected == 0)) || exit 0
}

check_deploy_prerequisites() {
  local required=(cmake curl ffmpeg git node ollama openssl sudo systemctl)
  local python_command=${PYTHON:-python3}
  required+=("$python_command")
  is_tegra && required+=(nvidia-smi)
  local missing=() dependency
  for dependency in "${required[@]}"; do
    command -v "$dependency" >/dev/null 2>&1 || missing+=("$dependency")
  done
  ((${#missing[@]} == 0)) \
    || die "missing deployment prerequisites: ${missing[*]}"
}

run() {
  if ((DRY_RUN)); then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

fast_forward_checkout() {
  ((ALLOW_UPDATE)) || return 0
  git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  git -C "$REPO_ROOT" remote get-url origin >/dev/null 2>&1 || return 0
  if [[ -n $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no) ]]; then
    die 'tracked checkout changes prevent an automatic upgrade; commit them or use --action deploy'
  fi

  printf '\nChecking origin/main for a runtime upgrade...\n'
  run git -C "$REPO_ROOT" fetch --quiet origin main
  if ((DRY_RUN)); then
    return
  fi
  local behind
  behind=$(git -C "$REPO_ROOT" rev-list --count HEAD..origin/main)
  if ((behind > 0)); then
    printf 'Fast-forwarding %s commit(s), then restarting the guided deployment.\n' "$behind"
    git -C "$REPO_ROOT" merge --ff-only origin/main
    local resume=(--profile "$PROFILE" --action deploy --yes)
    ((WITH_HARNESS)) && resume+=(--with-harness) || resume+=(--no-harness)
    exec "$REPO_ROOT/deploy.sh" "${resume[@]}"
  fi
  printf 'Checkout is already current.\n'
}

pull_and_validate_model() {
  command -v ollama >/dev/null 2>&1 || die 'ollama is required'
  printf '\nDownloading and verifying %s...\n' "$OMNI_MODEL"
  run ollama pull "$OMNI_MODEL"
  if [[ -x "$REPO_ROOT/.venv/bin/qwen-omni" ]]; then
    run "$REPO_ROOT/.venv/bin/qwen-omni" resolve "$OMNI_MODEL"
  elif ((DRY_RUN == 0)); then
    printf 'Runtime is not installed yet; sidecar validation will run after bootstrap.\n'
  fi
}

backup_environment() {
  ENV_BACKUP=$(mktemp)
  if [[ -f "$REPO_ROOT/.env" ]]; then
    cp -- "$REPO_ROOT/.env" "$ENV_BACKUP"
    ENV_EXISTED=1
  fi
}

install_environment() {
  local temporary
  temporary=$(mktemp "$REPO_ROOT/.env.deploy.XXXXXX")
  if [[ -f "$REPO_ROOT/.env" ]]; then
    awk '!/^OMNI_PROFILE=/ && !/^OMNI_MODEL=/ && !/^OMNI_LANGUAGE_MODEL=/' \
      "$REPO_ROOT/.env" >"$temporary"
  fi
  {
    printf 'OMNI_PROFILE=%s\n' "$PROFILE"
    printf 'OMNI_MODEL=%s\n' "$OMNI_MODEL"
    printf 'OMNI_LANGUAGE_MODEL=%s\n' "$OMNI_LANGUAGE_MODEL"
  } >>"$temporary"
  chmod 600 "$temporary"
  mv -- "$temporary" "$REPO_ROOT/.env"
  CONFIG_INSTALLED=1
}

restore_environment() {
  ((CONFIG_INSTALLED)) || return 0
  if ((ENV_EXISTED)); then
    cp -- "$ENV_BACKUP" "$REPO_ROOT/.env"
    chmod 600 "$REPO_ROOT/.env"
  else
    unlink "$REPO_ROOT/.env" 2>/dev/null || true
  fi
}

on_exit() {
  local status=$?
  restore_cursor
  if ((status != 0 && CONFIG_INSTALLED && DEPLOY_COMPLETE == 0)); then
    printf '\ndeploy: deployment failed; restoring the prior model configuration.\n' >&2
    restore_environment
    if ((SERVICE_WAS_ACTIVE)) && command -v systemctl >/dev/null 2>&1; then
      sudo systemctl restart "$SERVICE_NAME" >/dev/null 2>&1 || true
    elif command -v systemctl >/dev/null 2>&1; then
      sudo systemctl stop "$SERVICE_NAME" >/dev/null 2>&1 || true
      if ((SERVICE_EXISTED == 0)); then
        sudo systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 || true
      fi
    fi
  fi
  [[ -z $ENV_BACKUP ]] || unlink "$ENV_BACKUP" 2>/dev/null || true
}

trap on_exit EXIT
trap 'exit 130' INT TERM

wait_for_service() {
  local deadline=$((SECONDS + 1800)) state model updated_at
  printf 'Waiting for the selected model to pass startup smoke gates...\n'
  while ((SECONDS < deadline)); do
    if ! sudo systemctl is-active --quiet "$SERVICE_NAME"; then
      sudo systemctl status "$SERVICE_NAME" --no-pager -l >&2 || true
      die "$SERVICE_NAME stopped before readiness"
    fi
    if [[ -r "$REPO_ROOT/runtime-data/state/daemon-status.json" ]]; then
      read -r state model updated_at < <(
        "$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT/runtime-data/state/daemon-status.json" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    value = {}
print(value.get("state", ""), value.get("model", ""), int(value.get("updated_at", 0)))
PY
      )
      if [[ $state == ready && $model == "$OMNI_MODEL" ]] \
        && [[ $updated_at =~ ^[0-9]+$ ]] && ((updated_at >= SERVICE_START_EPOCH)); then
        return 0
      fi
    fi
    sleep 2
  done
  die 'service did not become ready within 30 minutes'
}

deploy_service() {
  local doctor=("$REPO_ROOT/.venv/bin/qwen-omni" doctor --deployment \
    --model "$OMNI_MODEL" --language-model "$OMNI_LANGUAGE_MODEL")
  command -v cloudflared >/dev/null 2>&1 || doctor+=(--no-tunnel)

  printf '\nRunning deployment doctor and regression gates...\n'
  run "${doctor[@]}"
  run "$REPO_ROOT/scripts/validate.sh"

  if ((DRY_RUN)); then
    printf '+ persist OMNI_PROFILE=%q OMNI_MODEL=%q OMNI_LANGUAGE_MODEL=%q in %q\n' \
      "$PROFILE" "$OMNI_MODEL" "$OMNI_LANGUAGE_MODEL" "$REPO_ROOT/.env"
    local dry_install=("$REPO_ROOT/services/linux/install.sh" --auto --no-enable)
    ((WITH_HARNESS)) && dry_install+=(--with-harness)
    run "${dry_install[@]}"
    run sudo systemctl enable "$SERVICE_NAME"
    run sudo systemctl restart "$SERVICE_NAME"
    if ((WITH_HARNESS)); then
      run systemctl --user enable --now omni-call-harness.service
    fi
    printf '+ wait up to 30 minutes for state=ready and model=%q\n' "$OMNI_MODEL"
    return 0
  fi

  backup_environment
  if service_installed; then
    SERVICE_EXISTED=1
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
      SERVICE_WAS_ACTIVE=1
    fi
  fi
  install_environment

  local install=("$REPO_ROOT/services/linux/install.sh" --auto --no-enable)
  ((WITH_HARNESS)) && install+=(--with-harness)
  "${install[@]}"

  sudo systemctl enable "$SERVICE_NAME"
  SERVICE_START_EPOCH=$(date +%s)
  if ((SERVICE_WAS_ACTIVE)); then
    sudo systemctl restart "$SERVICE_NAME"
  else
    sudo systemctl start "$SERVICE_NAME"
  fi
  wait_for_service

  if ((WITH_HARNESS)); then
    systemctl --user enable omni-call-harness.service
    if systemctl --user is-active --quiet omni-call-harness.service; then
      systemctl --user restart omni-call-harness.service
    else
      systemctl --user start omni-call-harness.service
    fi
  fi

  printf '\nDeployment ready.\n'
  sudo systemctl status "$SERVICE_NAME" --no-pager -l
  "$REPO_ROOT/.venv/bin/qwen-omni-daemon" status
  DEPLOY_COMPLETE=1
}

while (($#)); do
  case $1 in
    ornith15|ornith15-audio-bridge|ornith15-bridge|qwen38|qwen38-audio-bridge|qwen38-bridge)
      [[ -z $PROFILE ]] || die 'select only one profile'
      PROFILE=$1
      ;;
    --profile)
      (($# >= 2)) || die '--profile requires a value'
      PROFILE=$2
      shift
      ;;
    --action)
      (($# >= 2)) || die '--action requires a value'
      ACTION=$2
      shift
      ;;
    --with-harness) WITH_HARNESS=1 ;;
    --no-harness) WITH_HARNESS=0 ;;
    --no-update) ALLOW_UPDATE=0 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --list-models) list_models; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument '$1'" ;;
  esac
  shift
done

case "$ACTION" in
  ''|install|upgrade|deploy|download) ;;
  *) die "unknown action '$ACTION' (expected install, upgrade, deploy, or download)" ;;
esac

[[ -n $PROFILE && -z $ACTION ]] && ACTION=deploy
if [[ ! -t 0 || ! -t 1 ]]; then
  [[ -n $PROFILE && -n $ACTION ]] && ASSUME_YES=1
fi

printf 'Qwen Omni guided deployment\n'
printf '  Platform: %s\n' "$(if is_tegra; then printf 'NVIDIA Jetson/Tegra (%s)' "$(tegra_soc)"; else printf '%s/%s' "$(uname -s)" "$(uname -m)"; fi)"
printf '  Memory:   %s GiB\n' "$(memory_gib)"
printf '  State:    %s\n' "$(deployment_state)"
current=$(configured_model)
[[ -z $current ]] || printf '  Current:  %s\n' "$current"

if [[ -z $ACTION ]]; then
  [[ -t 0 && -t 1 ]] || die 'non-interactive use requires --profile and --action'
  choose_action
fi
if [[ -z $PROFILE ]]; then
  [[ -t 0 && -t 1 ]] || die 'non-interactive use requires --profile'
  choose_profile
fi
resolve_profile

if [[ $ACTION != download && -z $WITH_HARNESS ]]; then
  if [[ -t 0 && -t 1 && $ASSUME_YES == 0 ]]; then
    choose_harness
  else
    WITH_HARNESS=0
  fi
fi
confirm_plan

if [[ $PROFILE == qwen38 ]] && is_tegra; then
  memory_kib=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)
  if [[ $memory_kib =~ ^[0-9]+$ ]] && ((memory_kib < 29 * 1024 * 1024)); then
    die 'Qwen3.8 bridge requires a 32 GB-class Jetson or larger'
  fi
fi

if [[ $ACTION != download ]]; then
  check_deploy_prerequisites
fi

if [[ $ACTION == upgrade ]]; then
  fast_forward_checkout
fi

if [[ $ACTION == download ]]; then
  pull_and_validate_model
  exit 0
fi

if ((DRY_RUN)); then
  run "$REPO_ROOT/scripts/bootstrap.sh" --refresh-models
  deploy_service
  exit 0
fi

"$REPO_ROOT/scripts/bootstrap.sh" --refresh-models
deploy_service
