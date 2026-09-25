#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVICE_NAME=qwen-omni-adapters.service
SERVICE_UNIT=/etc/systemd/system/$SERVICE_NAME
HARNESS_NAME=omni-call-harness.service
HARNESS_UNIT=$HOME/.config/systemd/user/$HARNESS_NAME
PACKAGE_LOCK_POLICY_SOURCE=$REPO_ROOT/services/linux/49-qwen-omni-package-lock-query.pkla
PACKAGE_LOCK_POLICY_TARGET=/etc/polkit-1/localauthority/50-local.d/49-qwen-omni-package-lock-query.pkla
DEPLOYMENT_PORTS=8892,8901,8910,8920,8930,8940
PROFILE=""
ACTION=""
WITH_HARNESS=""
ASSUME_YES=0
DRY_RUN=0
ALLOW_UPDATE=1
ENV_BACKUP=""
ENV_EXISTED=0
CONFIG_INSTALLED=0
SERVICE_WAS_ENABLED=0
SERVICE_START_EPOCH=0
HARNESS_START_EPOCH=0
DEPLOY_COMPLETE=0
HANDOFF_STARTED=0
UNIT_BACKUP=""
UNIT_EXISTED=0
UNIT_INSTALLED=0
HARNESS_UNIT_BACKUP=""
HARNESS_UNIT_EXISTED=0
HARNESS_WAS_ENABLED=0
PRIOR_OMNI_MODEL=""
PRIOR_LANGUAGE_MODEL=""
declare -a STOPPED_SYSTEM_UNITS=()
declare -a STOPPED_USER_UNITS=()
declare -a LEGACY_USER_UNITS=()
declare -a STOPPED_MANUAL_PIDS=()

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
  --with-harness       Install the always-listening user service (default)
  --no-harness         Explicitly install only the core daemon/portal service
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

install_desktop_package_lock_policy() {
  ((WITH_HARNESS)) && is_tegra || return 0
  [[ -d ${PACKAGE_LOCK_POLICY_TARGET%/*} ]] || return 0
  run sudo install -m 0644 "$PACKAGE_LOCK_POLICY_SOURCE" "$PACKAGE_LOCK_POLICY_TARGET"
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

configured_value() {
  local key=$1
  [[ -r "$REPO_ROOT/.env" ]] || return 0
  awk -v key="$key" -F= '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$REPO_ROOT/.env"
}

configured_model() {
  configured_value OMNI_MODEL
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
    'Core service plus always-listening desktop indicator (recommended)' \
    'Core daemon and portal only (no desktop indicator)'
  ((selected == 0)) && WITH_HARNESS=1 || WITH_HARNESS=0
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

backup_service_unit() {
  UNIT_BACKUP=$(mktemp)
  if sudo test -f "$SERVICE_UNIT"; then
    # shellcheck disable=SC2024 # sudo grants the read; this shell owns the temp output.
    sudo cat "$SERVICE_UNIT" >"$UNIT_BACKUP"
    UNIT_EXISTED=1
  fi
  if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    SERVICE_WAS_ENABLED=1
  fi
  HARNESS_UNIT_BACKUP=$(mktemp)
  if [[ -f $HARNESS_UNIT ]]; then
    cp -- "$HARNESS_UNIT" "$HARNESS_UNIT_BACKUP"
    HARNESS_UNIT_EXISTED=1
  fi
  if systemctl --user is-enabled --quiet "$HARNESS_NAME" 2>/dev/null; then
    HARNESS_WAS_ENABLED=1
  fi
}

install_environment() {
  local temporary
  temporary=$(mktemp "$REPO_ROOT/.env.deploy.XXXXXX")
  if [[ -f "$REPO_ROOT/.env" ]]; then
    awk '!/^OMNI_PROFILE=/ && !/^OMNI_MODEL=/ && !/^OMNI_LANGUAGE_MODEL=/ \
      && !/^OMNI_ENABLE_COMPREHENSION=/ && !/^OMNI_ENABLE_POINTING=/ && !/^OMNI_STARTUP_SMOKE=/ \
      && !/^OMNI_COMPREHENSION_CONTEXT_TOKENS=/ \
      && !/^OMNI_VIRTUAL_CONTEXT_MODE=/ && !/^OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS=/ \
      && !/^OMNI_VIRTUAL_CONTEXT_TOKENIZE_URL=/ \
      && !/^OMNI_VIRTUAL_CONTEXT_RECURRENT_TOKENS=/ \
      && !/^OMNI_VIRTUAL_CONTEXT_RECURRENT_SOURCE_CHUNKS=/ \
      && !/^OMNI_TTS_PERSISTENT=/ && !/^OMNI_CALL_SPEECH_EVICT_UNIT=/' \
      "$REPO_ROOT/.env" >"$temporary"
  fi
  {
    printf 'OMNI_PROFILE=%s\n' "$PROFILE"
    printf 'OMNI_MODEL=%s\n' "$OMNI_MODEL"
    printf 'OMNI_LANGUAGE_MODEL=%s\n' "$OMNI_LANGUAGE_MODEL"
    printf 'OMNI_ENABLE_COMPREHENSION=1\n'
    printf 'OMNI_ENABLE_POINTING=1\n'
    # Treat resident attention as L0 RAM. History beyond this bound belongs in
    # the lossless virtual-context hierarchy rather than an oversized Tegra KV
    # allocation that competes with vision, TTS, and the desktop.
    printf 'OMNI_COMPREHENSION_CONTEXT_TOKENS=16384\n'
    # The 16K/256K live RULER gate is accepted: use the lossless hierarchy as
    # the production working-set allocator. Operators can still explicitly
    # select shadow/off in .env for diagnostic comparison.
    printf 'OMNI_VIRTUAL_CONTEXT_MODE=active\n'
    printf 'OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS=16384\n'
    printf 'OMNI_VIRTUAL_CONTEXT_TOKENIZE_URL=http://127.0.0.1:8901/tokenize\n'
    printf 'OMNI_VIRTUAL_CONTEXT_RECURRENT_TOKENS=512\n'
    printf 'OMNI_VIRTUAL_CONTEXT_RECURRENT_SOURCE_CHUNKS=200\n'
    printf 'OMNI_STARTUP_SMOKE=0\n'
    printf 'OMNI_TTS_PERSISTENT=1\n'
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

restore_service_unit() {
  ((UNIT_INSTALLED || HANDOFF_STARTED)) || return 0
  sudo systemctl stop "$SERVICE_NAME" >/dev/null 2>&1 || true
  if ((UNIT_EXISTED)); then
    sudo install -m 0644 "$UNIT_BACKUP" "$SERVICE_UNIT"
  else
    sudo rm -f -- "$SERVICE_UNIT"
  fi
  sudo systemctl daemon-reload
  if ((UNIT_EXISTED == 0 || SERVICE_WAS_ENABLED == 0)); then
    sudo systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 || true
  else
    sudo systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true
  fi
}

restore_harness_unit() {
  ((WITH_HARNESS)) || return 0
  systemctl --user stop "$HARNESS_NAME" >/dev/null 2>&1 || true
  if ((HARNESS_UNIT_EXISTED)); then
    mkdir -p "$(dirname "$HARNESS_UNIT")"
    install -m 0644 "$HARNESS_UNIT_BACKUP" "$HARNESS_UNIT"
  else
    unlink "$HARNESS_UNIT" 2>/dev/null || true
  fi
  systemctl --user daemon-reload
  if ((HARNESS_UNIT_EXISTED == 0 || HARNESS_WAS_ENABLED == 0)); then
    systemctl --user disable "$HARNESS_NAME" >/dev/null 2>&1 || true
  else
    systemctl --user enable "$HARNESS_NAME" >/dev/null 2>&1 || true
  fi
}

append_unique() {
  local array_name=$1 value=$2 existing
  local -n values="$array_name"
  for existing in "${values[@]}"; do
    [[ $existing == "$value" ]] && return 0
  done
  values+=("$value")
}

handoff_command() {
  "$REPO_ROOT/.venv/bin/python" -m qwen_omni_adapters.deployment_handoff "$@"
}

read_running_ollama_models() {
  local array_name=$1 output
  local -n result="$array_name"
  if ! output=$(ollama ps 2>&1); then
    printf '%s\n' "$output" >&2
    die 'could not inspect resident Ollama runners before deployment'
  fi
  # shellcheck disable=SC2034 # result is a nameref populated for the caller.
  mapfile -t result < <(printf '%s\n' "$output" | awk 'NR > 1 && NF {print $1}')
}

is_relevant_ollama_model() {
  local running=$1 candidate
  shift
  for candidate in "$@"; do
    [[ -n $candidate && $running == "$candidate" ]] && return 0
  done
  return 1
}

unload_relevant_ollama_models() {
  local candidates=(
    "$OMNI_MODEL"
    "$PRIOR_OMNI_MODEL"
    "$PRIOR_LANGUAGE_MODEL"
    robit/qwen3.8-27b-e03-obliterated-omni:q4km
    robit/qwen3.8-27b-obliterated-e03:27b
    robit/ornith-1.5-omni:q4km
    robit/ornith-1.5:9b
    robit/ornith-1.5-obliterated-omni:q4km
    robit/ornith-1.5-obliterated:9b
  )
  local running=() model stopped=0 deadline
  read_running_ollama_models running
  for model in "${running[@]}"; do
    if is_relevant_ollama_model "$model" "${candidates[@]}"; then
      printf 'Unloading prior Ollama runner: %s\n' "$model"
      ollama stop "$model"
      stopped=1
    fi
  done
  ((stopped)) || printf 'No relevant Ollama runner is resident.\n'

  deadline=$((SECONDS + 120))
  while ((SECONDS < deadline)); do
    local remaining=()
    read_running_ollama_models running
    for model in "${running[@]}"; do
      is_relevant_ollama_model "$model" "${candidates[@]}" && remaining+=("$model")
    done
    ((${#remaining[@]} == 0)) && return 0
    sleep 2
  done
  die "Ollama did not unload the prior model runner within 120 seconds"
}

wait_for_handoff_ports() {
  local deadline=$((SECONDS + 120))
  while ((SECONDS < deadline)); do
    if handoff_command ports-free --ports "$DEPLOYMENT_PORTS" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  handoff_command inventory --ports "$DEPLOYMENT_PORTS" >&2 || true
  die 'the old runtime did not release all Omni listener ports within 120 seconds'
}

prepare_runtime_handoff() {
  local target_file kind value unit pid unit_state invalid_target=""
  target_file=$(mktemp)
  if command -v docker >/dev/null 2>&1; then
    printf '\nInspecting the host CUDA lease policy before changing services...\n'
    if ! docker gpu discover; then
      is_tegra || die 'docker GPU discovery failed on a non-Tegra host; refusing an unscoped CUDA cutover'
      printf 'No discrete-GPU broker is active; Tegra direct lifecycle applies.\n'
    fi
  fi
  printf '\nInspecting live Omni ports and accelerator state before cutover...\n'
  handoff_command inventory --ports "$DEPLOYMENT_PORTS"
  if ! handoff_command targets --ports "$DEPLOYMENT_PORTS" >"$target_file"; then
    unlink "$target_file" 2>/dev/null || true
    die 'a deployment port is owned by an unknown process; leaving the live runtime untouched'
  fi

  while IFS=$'\t' read -r kind value; do
    [[ -n $kind && -n $value ]] || continue
    case $kind in
      system-unit) append_unique STOPPED_SYSTEM_UNITS "$value" ;;
      user-unit) append_unique STOPPED_USER_UNITS "$value" ;;
      process) append_unique STOPPED_MANUAL_PIDS "$value" ;;
      *) invalid_target=$kind ;;
    esac
  done <"$target_file"
  unlink "$target_file" 2>/dev/null || true
  [[ -z $invalid_target ]] || die "invalid handoff target: $invalid_target"

  # A managed runtime may still be starting and not own a port yet.  Include
  # the installed unit and legacy Omni units so they cannot race the new one.
  unit_state=$(systemctl show "$SERVICE_NAME" -p ActiveState --value 2>/dev/null || true)
  case $unit_state in
    active|activating|reloading|deactivating)
      append_unique STOPPED_SYSTEM_UNITS "$SERVICE_NAME"
      ;;
  esac
  while read -r unit _; do
    [[ -n $unit ]] || continue
    append_unique STOPPED_SYSTEM_UNITS "$unit"
  done < <(
    systemctl list-units --type=service --all --no-legend --plain 2>/dev/null \
      | awk 'tolower($1) ~ /omni/ && $3 ~ /^(active|activating|reloading|deactivating)$/ {print $1}'
  )
  if systemctl --user is-active --quiet "$HARNESS_NAME" 2>/dev/null; then
    append_unique STOPPED_USER_UNITS "$HARNESS_NAME"
  fi
  # Legacy per-user Egg units may be enabled but inactive while the current
  # system daemon owns every port. Discover them by their exact namespace so
  # they cannot return on the next graphical login and load a second CUDA
  # worker behind the successful deployment.
  while read -r unit _; do
    [[ $unit == egg-omni-*.service ]] || continue
    append_unique LEGACY_USER_UNITS "$unit"
    if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
      append_unique STOPPED_USER_UNITS "$unit"
    fi
  done < <(systemctl --user list-unit-files --type=service --no-legend --no-pager 2>/dev/null)

  HANDOFF_STARTED=1
  for unit in "${STOPPED_USER_UNITS[@]}"; do
    printf 'Stopping prior user service: %s\n' "$unit"
    systemctl --user stop "$unit"
  done
  for unit in "${STOPPED_SYSTEM_UNITS[@]}"; do
    printf 'Stopping prior system service: %s\n' "$unit"
    sudo systemctl stop "$unit"
  done
  for pid in "${STOPPED_MANUAL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      printf 'Stopping recognized unmanaged Omni listener PID: %s\n' "$pid"
      sudo kill -TERM "$pid"
    fi
  done

  wait_for_handoff_ports
  unload_relevant_ollama_models

  printf '\nPost-handoff port and accelerator snapshot...\n'
  handoff_command inventory --ports "$DEPLOYMENT_PORTS"
  if is_tegra; then
    printf 'Checking exact bundle bytes, unified-memory headroom, and sampled GPU utilization...\n'
    handoff_command admit \
      --model "$OMNI_MODEL" \
      --reserve-mib "${OMNI_DEPLOY_MEMORY_RESERVE_MIB:-6144}" \
      --max-utilization-percent "${OMNI_DEPLOY_MAX_GPU_UTILIZATION:-80}"
  else
    printf 'Jetson unified-memory admission is not applicable; the selected host lifecycle remains authoritative.\n'
  fi
}

retire_prior_services() {
  local unit
  for unit in "${STOPPED_SYSTEM_UNITS[@]}"; do
    [[ $unit == "$SERVICE_NAME" ]] && continue
    sudo systemctl disable "$unit" >/dev/null 2>&1 || true
  done
  for unit in "${STOPPED_USER_UNITS[@]}"; do
    if [[ $unit == "$HARNESS_NAME" && $WITH_HARNESS == 1 ]]; then
      continue
    fi
    systemctl --user disable "$unit" >/dev/null 2>&1 || true
  done
  for unit in "${LEGACY_USER_UNITS[@]}"; do
    systemctl --user disable "$unit" >/dev/null 2>&1 || true
  done
}

restart_prior_services() {
  local unit
  for unit in "${STOPPED_SYSTEM_UNITS[@]}"; do
    if ! sudo systemctl start "$unit" >/dev/null 2>&1; then
      printf 'deploy: failed to restart prior system service %s\n' "$unit" >&2
    elif ! sudo systemctl is-active --quiet "$unit"; then
      printf 'deploy: prior system service did not become active: %s\n' "$unit" >&2
    fi
  done
  for unit in "${STOPPED_USER_UNITS[@]}"; do
    if ! systemctl --user start "$unit" >/dev/null 2>&1; then
      printf 'deploy: failed to restart prior user service %s\n' "$unit" >&2
    elif ! systemctl --user is-active --quiet "$unit"; then
      printf 'deploy: prior user service did not become active: %s\n' "$unit" >&2
    fi
  done
  if ((${#STOPPED_MANUAL_PIDS[@]})); then
    printf 'deploy: unmanaged prior listener(s) cannot be reconstructed automatically: %s\n' \
      "${STOPPED_MANUAL_PIDS[*]}" >&2
  fi
}

on_exit() {
  local status=$?
  restore_cursor
  if ((status != 0 && HANDOFF_STARTED && DEPLOY_COMPLETE == 0)); then
    printf '\ndeploy: deployment failed; restoring the prior configuration, unit, and managed services.\n' >&2
    restore_environment
    restore_service_unit || true
    restore_harness_unit || true
    restart_prior_services
  fi
  [[ -z $ENV_BACKUP ]] || unlink "$ENV_BACKUP" 2>/dev/null || true
  [[ -z $UNIT_BACKUP ]] || unlink "$UNIT_BACKUP" 2>/dev/null || true
  [[ -z $HARNESS_UNIT_BACKUP ]] || unlink "$HARNESS_UNIT_BACKUP" 2>/dev/null || true
}

trap on_exit EXIT
trap 'exit 130' INT TERM

report_service_failure() {
  printf '\nService startup evidence\n' >&2
  sudo systemctl status "$SERVICE_NAME" --no-pager -l >&2 || true
  sudo journalctl -u "$SERVICE_NAME" -n 100 --no-pager >&2 || true
  if [[ -r "$REPO_ROOT/runtime-data/state/daemon-status.json" ]]; then
    "$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT/runtime-data/state/daemon-status.json" <<'PY' >&2 || true
import json
import sys

try:
    source = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as exc:
    print(json.dumps({"status_error": str(exc)}, indent=2))
else:
    allowed = {
        key: source[key]
        for key in (
            "state", "detail", "model", "updated_at", "accelerator",
            "comprehension", "startup_smoke", "decision_plane", "children",
            "co_resident_stack",
        )
        if key in source
    }
    print(json.dumps(allowed, indent=2, sort_keys=True))
PY
  fi
}

report_harness_failure() {
  printf '\nDesktop harness startup evidence\n' >&2
  systemctl --user status "$HARNESS_NAME" --no-pager -l >&2 || true
  journalctl --user -u "$HARNESS_NAME" -n 100 --no-pager >&2 || true
  if [[ -r "$REPO_ROOT/runtime-data/state/harness-status.json" ]]; then
    "$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT/runtime-data/state/harness-status.json" <<'PY' >&2 || true
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as exc:
    value = {"status_error": str(exc)}
print(json.dumps(value, indent=2, sort_keys=True))
PY
  fi
}

wait_for_harness() {
  local deadline=$((SECONDS + 120)) state backend updated_at pid active_state sub_state restarts
  printf 'Waiting for the desktop harness and real indicator backend...\n'
  while ((SECONDS < deadline)); do
    active_state=$(systemctl --user show "$HARNESS_NAME" -p ActiveState --value 2>/dev/null || true)
    sub_state=$(systemctl --user show "$HARNESS_NAME" -p SubState --value 2>/dev/null || true)
    restarts=$(systemctl --user show "$HARNESS_NAME" -p NRestarts --value 2>/dev/null || true)
    [[ $restarts =~ ^[0-9]+$ ]] || restarts=0
    if [[ -r "$REPO_ROOT/runtime-data/state/harness-status.json" ]]; then
      read -r state backend updated_at pid < <(
        "$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT/runtime-data/state/harness-status.json" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    value = {}
print(
    value.get("state", ""),
    value.get("indicator_backend", ""),
    int(value.get("updated_at", 0)),
    int(value.get("pid", 0)),
)
PY
      )
      if [[ $backend == ayatana-appindicator3 || $backend == appindicator3 ]] \
        && [[ $updated_at =~ ^[0-9]+$ ]] && ((updated_at >= HARNESS_START_EPOCH)) \
        && [[ $pid =~ ^[0-9]+$ ]] && ((pid > 0)) && kill -0 "$pid" 2>/dev/null \
        && [[ $state =~ ^(listening|hearing|thinking|speaking|muted)$ ]]; then
        return 0
      fi
    fi
    # Restart=always deliberately passes through failed/inactive and
    # auto-restart while the core daemon is still loading and rotating its
    # token. Only the deadline is terminal; an arbitrary restart count is not.
    sleep 2
  done
  report_harness_failure
  die "$HARNESS_NAME did not prove a visible indicator within 120 seconds (state=$active_state/$sub_state, restarts=$restarts)"
}

wait_for_service() {
  local started=$SECONDS deadline=$((SECONDS + 1800)) state model updated_at
  local active_state sub_state restarts
  printf 'Waiting for local services to report ready (generation smoke disabled)...\n'
  while ((SECONDS < deadline)); do
    active_state=$(sudo systemctl show "$SERVICE_NAME" -p ActiveState --value 2>/dev/null || true)
    sub_state=$(sudo systemctl show "$SERVICE_NAME" -p SubState --value 2>/dev/null || true)
    restarts=$(sudo systemctl show "$SERVICE_NAME" -p NRestarts --value 2>/dev/null || true)
    [[ $restarts =~ ^[0-9]+$ ]] || restarts=0
    case $active_state in
      active|activating|reloading) ;;
      failed|inactive|deactivating|*)
        # Restart=on-failure has a deliberate ten-second gap.  An immediate
        # is-active check used to misclassify both "activating" and that gap
        # as a terminal failure and hide the journal that explained it.
        if ((SECONDS - started >= 60)); then
          report_service_failure
          die "$SERVICE_NAME failed before readiness (state=$active_state/$sub_state, restarts=$restarts)"
        fi
        ;;
    esac
    if [[ -r "$REPO_ROOT/runtime-data/state/daemon-status.json" ]]; then
      read -r state model updated_at < <(
        "$REPO_ROOT/.venv/bin/python" - "$REPO_ROOT/runtime-data/state/daemon-status.json" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    value = {}
print(
    value.get("state", ""),
    value.get("model", ""),
    int(value.get("updated_at", 0)),
)
PY
      )
      if [[ $state == ready && $model == "$OMNI_MODEL" ]] \
        && [[ $updated_at =~ ^[0-9]+$ ]] && ((updated_at >= SERVICE_START_EPOCH)); then
        return 0
      fi
    fi
    if ((restarts >= 3)); then
      report_service_failure
      die "$SERVICE_NAME restarted repeatedly before readiness (state=$active_state/$sub_state, restarts=$restarts)"
    fi
    sleep 2
  done
  report_service_failure
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
    printf '+ inventory owners of ports %s and refuse unknown listeners\n' "$DEPLOYMENT_PORTS"
    printf '+ stop and record prior Omni system/user services; wait for all deployment ports to close\n'
    printf '+ unload only the selected, prior-configured, and known legacy Omni Ollama runners\n'
    if is_tegra; then
      printf '+ sample Tegra GPU utilization and require exact model bytes plus %s MiB free headroom\n' \
        "${OMNI_DEPLOY_MEMORY_RESERVE_MIB:-6144}"
    fi
    printf '+ persist the trained-bridge model, mandatory comprehension, smoke-free startup, and persistent TTS in %q\n' \
      "$REPO_ROOT/.env"
    printf '+ persist OMNI_PROFILE=%q OMNI_MODEL=%q OMNI_LANGUAGE_MODEL=%q in %q\n' \
      "$PROFILE" "$OMNI_MODEL" "$OMNI_LANGUAGE_MODEL" "$REPO_ROOT/.env"
    local dry_install=("$REPO_ROOT/services/linux/install.sh" --auto --no-enable)
    ((WITH_HARNESS)) && dry_install+=(--with-harness)
    run "${dry_install[@]}"
    run sudo systemctl enable "$SERVICE_NAME"
    run sudo systemctl start "$SERVICE_NAME"
    if ((WITH_HARNESS)); then
      run systemctl --user enable --now omni-call-harness.service
      printf '+ wait up to 120 seconds for a live GTK/AppIndicator harness status\n'
    fi
    printf '+ start the desktop indicator immediately after the core service (when selected)\n'
    printf '+ wait for local service health only; do not run generation smoke\n'
    return 0
  fi

  PRIOR_OMNI_MODEL=$(configured_value OMNI_MODEL)
  PRIOR_LANGUAGE_MODEL=$(configured_value OMNI_LANGUAGE_MODEL)
  backup_environment
  backup_service_unit
  prepare_runtime_handoff
  install_environment

  local install=("$REPO_ROOT/services/linux/install.sh" --auto --no-enable)
  ((WITH_HARNESS)) && install+=(--with-harness)
  "${install[@]}"
  UNIT_INSTALLED=1
  install_desktop_package_lock_policy

  sudo systemctl enable "$SERVICE_NAME"
  SERVICE_START_EPOCH=$(date +%s)
  sudo systemctl start "$SERVICE_NAME"

  if ((WITH_HARNESS)); then
    printf 'Starting the desktop indicator while the core service becomes ready...\n'
    systemctl --user enable omni-call-harness.service
    HARNESS_START_EPOCH=$(date +%s)
    if systemctl --user is-active --quiet omni-call-harness.service; then
      systemctl --user restart omni-call-harness.service
    else
      systemctl --user start omni-call-harness.service
    fi
  fi

  wait_for_service

  if ((WITH_HARNESS)); then
    wait_for_harness
  fi

  retire_prior_services

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
    # The safe/default deployment is visibly listening. Core-only operation
    # requires the explicit --no-harness opt-out; Enter-through and --yes
    # must never silently omit the desktop indicator.
    WITH_HARNESS=1
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

bootstrap=("$REPO_ROOT/scripts/bootstrap.sh" --refresh-models)
((WITH_HARNESS)) && bootstrap+=(--with-harness)

if ((DRY_RUN)); then
  run "${bootstrap[@]}"
  deploy_service
  exit 0
fi

"${bootstrap[@]}"
deploy_service
