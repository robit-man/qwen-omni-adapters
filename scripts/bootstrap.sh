#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
VENV=${OMNI_VENV:-$REPO_ROOT/.venv}
MODEL=${OMNI_MODEL:-robit/qwen3.8-27b-e03-obliterated-omni:q4km}
LANGUAGE_MODEL=${OMNI_LANGUAGE_MODEL:-robit/qwen3.8-27b-obliterated-e03:27b}
BUILD_LLAMA=1
PULL_MODELS=1
REFRESH_MODELS=0
PREPARE_COMPONENTS=0
INSTALL_LAYA=1
INSTALL_POINTING=1
INSTALL_HARNESS=0

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [options]

  --skip-llama       Do not clone, patch, or build llama.cpp
  --skip-models      Do not pull or resolve Ollama models
  --refresh-models   Pull selected tags even when already installed
  --prepare          Materialize the disposable component cache now
  --skip-laya        Do not install the isolated resident Laya runtime
  --skip-pointing    Do not install the isolated Tegra visual point head
  --with-harness     Install Linux desktop indicator/audio prerequisites
  --help             Show this help

Environment: OMNI_MODEL, OMNI_LANGUAGE_MODEL, OMNI_VENV, PYTHON,
OMNI_COMPONENT_CACHE, LLAMA_CPP_BUILD_JOBS.
EOF
}

while (($#)); do
  case $1 in
    --skip-llama) BUILD_LLAMA=0 ;;
    --skip-models) PULL_MODELS=0 ;;
    --refresh-models) REFRESH_MODELS=1 ;;
    --prepare) PREPARE_COMPONENTS=1 ;;
    --skip-laya) INSTALL_LAYA=0 ;;
    --skip-pointing) INSTALL_POINTING=0 ;;
    --with-harness) INSTALL_HARNESS=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v "$PYTHON" >/dev/null 2>&1 || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }

desktop_harness_dependencies_ready() {
  command -v parec >/dev/null 2>&1 \
    && command -v paplay >/dev/null 2>&1 \
    && command -v pactl >/dev/null 2>&1 \
    && "$PYTHON" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gtk", "3.0")
try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
except ValueError:
    gi.require_version("AppIndicator3", "0.1")
PY
}

if ((INSTALL_HARNESS)) && [[ $(uname -s) == Linux ]]; then
  if ! desktop_harness_dependencies_ready; then
    command -v apt-get >/dev/null 2>&1 || {
      printf 'Desktop harness dependencies are missing and apt-get is unavailable.\n' >&2
      exit 1
    }
    printf 'Installing desktop indicator and PulseAudio client dependencies...\n'
    sudo apt-get update
    indicator_package=gir1.2-ayatanaappindicator3-0.1
    if ! apt-cache show "$indicator_package" >/dev/null 2>&1; then
      indicator_package=gir1.2-appindicator3-0.1
    fi
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3-gi gir1.2-gtk-3.0 "$indicator_package" pulseaudio-utils
  fi
  desktop_harness_dependencies_ready || {
    printf 'Desktop harness dependencies remain unavailable after installation.\n' >&2
    exit 1
  }
fi

venv_options=()
if ((INSTALL_HARNESS)) && [[ $(uname -s) == Linux ]]; then
  # Ubuntu distributes PyGObject through apt. Expose that system binding to
  # the venv while keeping venv-installed packages first on sys.path.
  venv_options+=(--system-site-packages)
fi
"$PYTHON" -m venv "${venv_options[@]}" "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e "${REPO_ROOT}[dev]"

if ((INSTALL_LAYA)); then
  "$REPO_ROOT/scripts/bootstrap_laya.sh"
fi

if ((INSTALL_POINTING)); then
  "$REPO_ROOT/scripts/bootstrap_pointing.sh"
fi

if ((BUILD_LLAMA)); then
  build_commands=(cmake git)
  if [[ $(uname -s) == Linux ]]; then build_commands+=(nvidia-smi); fi
  for command in "${build_commands[@]}"; do
    command -v "$command" >/dev/null 2>&1 || {
      printf 'Missing build dependency: %s\n' "$command" >&2
      exit 1
    }
  done
  "$REPO_ROOT/scripts/build_llama_cpp.sh"
fi

if ((PULL_MODELS)); then
  command -v ollama >/dev/null 2>&1 || { printf 'Missing command: ollama\n' >&2; exit 1; }
  if ((REFRESH_MODELS)) || ! ollama show "$MODEL" >/dev/null 2>&1; then
    printf 'Pulling logical Omni tag %s\n' "$MODEL"
    ollama pull "$MODEL"
  fi
  if [[ $LANGUAGE_MODEL != "$MODEL" ]] \
    && { ((REFRESH_MODELS)) || ! ollama show "$LANGUAGE_MODEL" >/dev/null 2>&1; }; then
    printf 'Pulling language backend %s\n' "$LANGUAGE_MODEL"
    ollama pull "$LANGUAGE_MODEL"
  fi
  "$VENV/bin/python" -m qwen_omni_adapters resolve "$MODEL" >/dev/null
  printf 'Validated the Omni sidecar attached to %s\n' "$MODEL"
fi

if ((PREPARE_COMPONENTS)); then
  CACHE=${OMNI_COMPONENT_CACHE:-$REPO_ROOT/runtime-data/components}
  mkdir -p "$CACHE"
  "$VENV/bin/python" -m qwen_omni_adapters prepare \
    "$MODEL" --out "$CACHE" --overwrite >/dev/null
  printf 'Materialized disposable component views in %s\n' "$CACHE"
fi

printf 'Bootstrap complete. Validate with:\n  %s/bin/qwen-omni doctor --deployment\n' "$VENV"
