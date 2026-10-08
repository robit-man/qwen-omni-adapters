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
WITH_CAMERA=1
WITH_AUTO_UPDATE=1
ECAM_REPO_URL=${OMNI_ECAM_REPO:-https://github.com/robit-man/jetson-ecam-gmsl.git}
ECAM_DIR=$REPO_ROOT/vendor/jetson-ecam-gmsl
AUTO_UPDATE_UNIT_DIR=$HOME/.config/systemd/user
AUTO_UPDATE_SUDOERS=/etc/sudoers.d/qwen-omni-auto-update
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
  --no-camera          Skip the e-con GMSL camera stack on JetPack 7 AGX Orin
  --no-auto-update     Disable the default timer that redeploys when main moves
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

NODE_MINIMUM_MAJOR=18
NODE_MINIMUM_MINOR=18
NODESOURCE_SETUP=https://deb.nodesource.com/setup_22.x
OLLAMA_INSTALLER=https://ollama.com/install.sh

node_is_current() {
  local version major minor
  version=$(node --version 2>/dev/null) || return 1
  version=${version#v}
  major=${version%%.*}
  minor=${version#*.}
  minor=${minor%%.*}
  [[ $major =~ ^[0-9]+$ && $minor =~ ^[0-9]+$ ]] || return 1
  ((major > NODE_MINIMUM_MAJOR || (major == NODE_MINIMUM_MAJOR && minor >= NODE_MINIMUM_MINOR)))
}

missing_deploy_prerequisites() {
  local python_command=${PYTHON:-python3}
  local required=(cmake curl ffmpeg git ollama openssl sudo systemctl "$python_command")
  local dependency
  for dependency in "${required[@]}"; do
    command -v "$dependency" >/dev/null 2>&1 || printf '%s\n' "$dependency"
  done
  node_is_current || printf 'node\n'
  # Host-camera capture: NVIDIA's JetPack 7 ffmpeg has no V4L2 input, so the
  # portal captures through v4l2-ctl there.
  command -v v4l2-ctl >/dev/null 2>&1 || printf 'v4l2-ctl\n'
  if command -v "$python_command" >/dev/null 2>&1 \
    && ! "$python_command" -c 'import ensurepip, venv' >/dev/null 2>&1; then
    printf 'python3-venv\n'
  fi
}

wait_for_ollama() {
  local attempt
  for attempt in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && return 0
    sleep 1
  done
  die 'Ollama was installed but its server did not answer within 30 seconds'
}

# A fresh Ubuntu/JetPack host lacks several runtime tools, and its archive
# Node.js (12.x on 22.04) is older than the portal requires. Install only what
# is missing: apt for system packages, NodeSource for Node.js, and Ollama's
# official installer, which selects the JetPack build on Tegra and registers
# the ollama systemd service.
install_deploy_prerequisites() {
  local missing=() packages=() dependency
  mapfile -t missing < <(missing_deploy_prerequisites)
  ((${#missing[@]} == 0)) && return 0

  printf 'Installing missing deployment prerequisites: %s\n' "${missing[*]}"
  for dependency in "${missing[@]}"; do
    case $dependency in
      sudo|systemctl) die "missing deployment prerequisites: ${missing[*]} (cannot auto-install $dependency)" ;;
    esac
  done
  [[ $(uname -s) == Linux ]] && command -v apt-get >/dev/null 2>&1 \
    || die "missing deployment prerequisites: ${missing[*]} (automatic install requires apt-get)"

  for dependency in "${missing[@]}"; do
    case $dependency in
      cmake) packages+=(cmake build-essential) ;;
      curl|ffmpeg|git|openssl) packages+=("$dependency") ;;
      v4l2-ctl) packages+=(v4l-utils) ;;
      python3|python3-venv|"${PYTHON:-python3}") packages+=(python3 python3-venv python3-pip) ;;
    esac
  done
  # NodeSource and Ollama setup both fetch over HTTPS.
  command -v curl >/dev/null 2>&1 || packages+=(curl ca-certificates)

  if ((${#packages[@]})); then
    run sudo apt-get -o DPkg::Lock::Timeout=600 update
    run sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y "${packages[@]}"
  fi
  if [[ " ${missing[*]} " == *' node '* ]]; then
    # Ubuntu's libnode-dev owns headers that the NodeSource nodejs package
    # also ships, so an old archive Node.js would block the upgrade.
    if dpkg-query -W -f='${Status}' libnode-dev 2>/dev/null | grep -q 'install ok installed'; then
      run sudo apt-get -o DPkg::Lock::Timeout=600 remove -y libnode-dev
    fi
    run bash -c "curl -fsSL $NODESOURCE_SETUP | sudo -E bash -"
    run sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y nodejs
    hash -r
  fi
  if [[ " ${missing[*]} " == *' ollama '* ]]; then
    run bash -c "curl -fsSL $OLLAMA_INSTALLER | sh"
    hash -r
    ((DRY_RUN)) || wait_for_ollama
  fi
}

OLLAMA_SERVICE_HOME=/usr/share/ollama

# The official Ollama installer keeps models under the `ollama` user's home,
# readable only by the `ollama` group, and adds the invoking user to that group.
# Group membership only applies to new logins, so a deployment that just
# installed Ollama (or a session that predates the membership) re-executes
# itself under the group with the same choices instead of asking for a logout.
ensure_ollama_store_access() {
  [[ -d $OLLAMA_SERVICE_HOME ]] || return 0
  [[ -r $OLLAMA_SERVICE_HOME && -x $OLLAMA_SERVICE_HOME ]] && return 0
  getent group ollama >/dev/null 2>&1 \
    || die "cannot read $OLLAMA_SERVICE_HOME and no ollama group exists"
  local user
  user=$(id -un)
  if ! id -nG "$user" | tr ' ' '\n' | grep -qx ollama; then
    printf 'Adding %s to the ollama group to read the Ollama model store.\n' "$user"
    run sudo usermod -aG ollama "$user"
  fi
  if id -nG | tr ' ' '\n' | grep -qx ollama; then
    die "cannot read $OLLAMA_SERVICE_HOME even with ollama group access"
  fi
  ((DRY_RUN)) && return 0
  ((${OMNI_DEPLOY_GROUP_REEXEC:-0})) \
    && die "cannot read $OLLAMA_SERVICE_HOME after joining the ollama group; log out and back in"
  printf 'Continuing the deployment with ollama group access.\n'
  local resume=(--profile "$PROFILE" --action "$ACTION" --yes)
  ((ALLOW_UPDATE)) || resume+=(--no-update)
  if [[ $ACTION != download ]]; then
    ((WITH_HARNESS)) && resume+=(--with-harness) || resume+=(--no-harness)
  fi
  exec sg ollama -c "$(printf '%q ' env OMNI_DEPLOY_GROUP_REEXEC=1 "$REPO_ROOT/deploy.sh" "${resume[@]}")"
}

# The daemon publishes the portal through a cloudflared quick tunnel; without
# the binary it serves loopback only and the tray has no public link to copy.
# Install Cloudflare's official package for this architecture when missing.
install_cloudflared() {
  command -v cloudflared >/dev/null 2>&1 && return 0
  [[ $(uname -s) == Linux ]] && command -v dpkg >/dev/null 2>&1 || {
    warn 'cloudflared is not installed; the portal will be local-only'
    return 0
  }
  local arch package
  arch=$(dpkg --print-architecture)
  package=$(mktemp --suffix=.deb)
  printf 'Installing cloudflared for the public portal link...\n'
  if run curl -fsSL -o "$package" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch.deb" \
    && run sudo dpkg -i "$package"; then
    hash -r
  else
    warn 'cloudflared install failed; the portal will be local-only'
  fi
  rm -f "$package"
}

# jtop (jetson-stats) is the standard Jetson monitor. It ships only on PyPI,
# installs system-wide with its own jtop.service, and is optional for the
# runtime, so a failure here warns instead of aborting the deployment.
install_jetson_monitor() {
  is_tegra || return 0
  command -v jtop >/dev/null 2>&1 && return 0
  printf 'Installing jtop (jetson-stats) for Jetson monitoring...\n'
  if ! python3 -m pip --version >/dev/null 2>&1; then
    run sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y python3-pip
  fi
  local pip_install=(sudo -H python3 -m pip install -U jetson-stats)
  # Ubuntu 24.04 marks the system Python externally managed (PEP 668).
  if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
  fi
  if ! run "${pip_install[@]}"; then
    printf 'deploy: warning: jtop install failed; continuing without it\n' >&2
    return 0
  fi
  run sudo systemctl enable --now jtop.service \
    || printf 'deploy: warning: jtop.service did not start\n' >&2
  if getent group jtop >/dev/null 2>&1; then
    run sudo usermod -aG jtop "$(id -un)"
  fi
}

check_deploy_prerequisites() {
  ((DRY_RUN)) && return 0
  local missing=()
  mapfile -t missing < <(missing_deploy_prerequisites)
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
  local temporary cache_type=f16 background_residency_mode=conversation context_tokens=16384
  # The Ornith bridge's q8 KV profile on the 32 GB AGX Orin exposes a 64K
  # ceiling. The launcher admits only a tier funded by measured live residency,
  # the GGUF-derived KV slope, and the shared operational reserve, so this is
  # not an unconditional 64K allocation. Keep f16 everywhere else until the
  # exact model/platform pair has equivalent live evidence.
  if [[ $PROFILE == ornith15 ]] && is_tegra; then
    cache_type=q8_0
    background_residency_mode=action
    context_tokens=65536
  fi
  temporary=$(mktemp "$REPO_ROOT/.env.deploy.XXXXXX")
  if [[ -f "$REPO_ROOT/.env" ]]; then
    awk '!/^OMNI_PROFILE=/ && !/^OMNI_MODEL=/ && !/^OMNI_LANGUAGE_MODEL=/ \
      && !/^OMNI_ENABLE_COMPREHENSION=/ && !/^OMNI_ENABLE_POINTING=/ && !/^OMNI_STARTUP_SMOKE=/ \
      && !/^OMNI_COMPREHENSION_CONTEXT_TOKENS=/ \
      && !/^OMNI_COMPREHENSION_CACHE_TYPE_K=/ && !/^OMNI_COMPREHENSION_CACHE_TYPE_V=/ \
      && !/^OMNI_BACKGROUND_RESIDENCY_MODE=/ \
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
    # Treat resident attention as L0 RAM. The ceiling is profile-qualified;
    # the live launcher still picks the largest tier that measured weights,
    # KV, and the shared operational reserve can safely fund.
    printf 'OMNI_COMPREHENSION_CONTEXT_TOKENS=%s\n' "$context_tokens"
    printf 'OMNI_COMPREHENSION_CACHE_TYPE_K=%s\n' "$cache_type"
    printf 'OMNI_COMPREHENSION_CACHE_TYPE_V=%s\n' "$cache_type"
    # Silent durable work keeps the fused language/audio/vision trunk resident.
    # Action mode sheds the independently reloadable TTS and point-head graphs
    # before a background slice, preserving headroom for prompt/KV work.
    printf 'OMNI_BACKGROUND_RESIDENCY_MODE=%s\n' "$background_residency_mode"
    # The 16K/256K live RULER gate is accepted: use the lossless hierarchy as
    # the production working-set allocator. Operators can still explicitly
    # select shadow/off in .env for diagnostic comparison.
    printf 'OMNI_VIRTUAL_CONTEXT_MODE=active\n'
    printf 'OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS=%s\n' "$context_tokens"
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
  if ! handoff_command targets --ports "$DEPLOYMENT_PORTS" >"$target_file" 2>/dev/null; then
    # A listener owned by root or another user cannot be read from this
    # account's /proc view. Identify it with root before deciding; only a
    # genuinely foreign owner stops the handoff.
    printf 'A deployment port owner is not readable as %s; inspecting it with sudo...\n' "$(id -un)"
    sudo "$REPO_ROOT/.venv/bin/python" -m qwen_omni_adapters.deployment_handoff \
      inventory --ports "$DEPLOYMENT_PORTS"
    if ! sudo "$REPO_ROOT/.venv/bin/python" -m qwen_omni_adapters.deployment_handoff \
      targets --ports "$DEPLOYMENT_PORTS" >"$target_file"; then
      unlink "$target_file" 2>/dev/null || true
      die 'a deployment port is owned by a non-Omni process (see the owner above); leaving the live runtime untouched'
    fi
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
    # The owner may belong to root (found through the sudo inventory).
    if sudo kill -0 "$pid" 2>/dev/null; then
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

warn() {
  printf 'deploy: warning: %s\n' "$*" >&2
}

l4t_major() {
  sed -n 's/^# R\([0-9]*\) (release).*/\1/p' /etc/nv_tegra_release 2>/dev/null | head -n 1
}

# JetPack 7 AGX Orin hosts get the e-con GMSL camera stack. Its installer
# detects the L4T release and kernel, rebuilds only what that kernel needs,
# and is a no-op when nothing changed. A camera failure never fails the
# runtime deployment.
install_camera_stack() {
  ((WITH_CAMERA)) || return 0
  is_tegra || return 0
  local major
  major=$(l4t_major)
  if [[ ! $major =~ ^[0-9]+$ ]] || ((major < 38)); then
    printf 'Camera stack: L4T R%s predates JetPack 7; e-con'"'"'s own JetPack 6 installer applies there.\n' \
      "${major:-?}"
    return 0
  fi
  if ! tr '\0' '\n' </proc/device-tree/compatible 2>/dev/null | grep -q '^nvidia,p3737-0000+p3701-'; then
    printf 'Camera stack: not an AGX Orin developer kit carrier; skipping.\n'
    return 0
  fi

  printf '\nInstalling the e-con GMSL camera stack for this kernel...\n'
  if [[ -d $ECAM_DIR/.git ]]; then
    run git -C "$ECAM_DIR" fetch --quiet origin main
    run git -C "$ECAM_DIR" reset --quiet --hard origin/main
  else
    run git clone --quiet --depth 1 "$ECAM_REPO_URL" "$ECAM_DIR"
  fi
  if ! run sudo "$ECAM_DIR/install.sh" --yes --if-needed; then
    warn 'camera stack was not installed (see above); the Omni runtime is unaffected'
  fi
}

# Unattended redeploys run deploy.sh, which needs sudo, so automatic updates
# (on by default; --no-auto-update opts out) install a passwordless sudo rule
# for the deploying account and say how to remove it.
ensure_unattended_sudo() {
  local user
  user=$(id -un)
  if sudo -n -l 2>/dev/null | grep -Eq 'NOPASSWD: *ALL'; then
    return 0
  fi
  local rule
  rule=$(mktemp)
  printf '# Installed by qwen-omni-adapters deploy.sh for unattended redeploys\n%s ALL=(ALL) NOPASSWD: ALL\n' \
    "$user" >"$rule"
  sudo visudo -cqf "$rule" || { rm -f "$rule"; warn 'generated sudoers rule failed validation'; return 1; }
  sudo install -m 0440 "$rule" "$AUTO_UPDATE_SUDOERS"
  rm -f "$rule"
  printf 'Automatic updates: granted %s passwordless sudo via %s (remove that file or redeploy with --no-auto-update to revoke).\n' \
    "$user" "$AUTO_UPDATE_SUDOERS"
}

# A user timer checks origin/main every five minutes and redeploys through
# scripts/auto_update.sh. Linger keeps it running without a desktop login.
install_auto_update() {
  if ((WITH_AUTO_UPDATE == 0)); then
    # Opting out also withdraws the unattended sudo rule this deployer added.
    ((DRY_RUN)) && return 0
    systemctl --user disable --now omni-auto-update.timer >/dev/null 2>&1 || true
    if [[ -f $AUTO_UPDATE_SUDOERS ]]; then
      sudo rm -f "$AUTO_UPDATE_SUDOERS"
    fi
    return 0
  fi
  [[ $(uname -s) == Linux ]] || return 0
  if ((DRY_RUN)); then
    printf '+ install omni-auto-update.timer (checks origin/main every 5 minutes)\n'
    return 0
  fi
  sudo -n -l 2>/dev/null | grep -Eq 'NOPASSWD: *ALL' || ensure_unattended_sudo || return 0
  mkdir -p "$AUTO_UPDATE_UNIT_DIR"
  sed "s|@REPO_ROOT@|$REPO_ROOT|g" "$REPO_ROOT/services/linux/omni-auto-update.service.in" \
    >"$AUTO_UPDATE_UNIT_DIR/omni-auto-update.service"
  install -m 0644 "$REPO_ROOT/services/linux/omni-auto-update.timer" \
    "$AUTO_UPDATE_UNIT_DIR/omni-auto-update.timer"
  sudo loginctl enable-linger "$(id -un)"
  systemctl --user daemon-reload
  systemctl --user enable --now omni-auto-update.timer
  printf 'Automatic updates: omni-auto-update.timer redeploys when origin/main changes.\n'
}

record_deployed_commit() {
  ((DRY_RUN)) && return 0
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$REPO_ROOT" rev-parse HEAD >"$REPO_ROOT/.deployed-commit"
  fi
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
    --no-camera) WITH_CAMERA=0 ;;
    --no-auto-update) WITH_AUTO_UPDATE=0 ;;
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

# Passwordless sudo is part of setup when automatic updates are enabled (the
# default): install it before anything else so a later failure cannot leave
# the host without it and the rest of setup needs no further password prompt.
if ((WITH_AUTO_UPDATE && DRY_RUN == 0)) && [[ $(uname -s) == Linux ]]; then
  ensure_unattended_sudo || warn 'passwordless sudo was not configured; automatic updates will not run'
fi

if [[ $ACTION == download ]]; then
  command -v ollama >/dev/null 2>&1 || install_deploy_prerequisites
else
  install_deploy_prerequisites
  check_deploy_prerequisites
  install_cloudflared
  install_jetson_monitor
fi
ensure_ollama_store_access

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
  install_camera_stack
  deploy_service
  install_auto_update
  exit 0
fi

"${bootstrap[@]}"
# The camera stack is independent of the Omni runtime handoff, so it runs
# first and a refused or failed handoff cannot skip it.
install_camera_stack || warn 'camera stack step failed; the Omni runtime is unaffected'
deploy_service
install_auto_update || warn 'automatic updates were not enabled'
record_deployed_commit
