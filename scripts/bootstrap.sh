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
PREPARE_COMPONENTS=0

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [options]

  --skip-llama       Do not clone, patch, or build llama.cpp
  --skip-models      Do not pull or resolve Ollama models
  --prepare          Materialize the disposable component cache now
  --help             Show this help

Environment: OMNI_MODEL, OMNI_LANGUAGE_MODEL, OMNI_VENV, PYTHON,
OMNI_COMPONENT_CACHE, LLAMA_CPP_BUILD_JOBS.
EOF
}

while (($#)); do
  case $1 in
    --skip-llama) BUILD_LLAMA=0 ;;
    --skip-models) PULL_MODELS=0 ;;
    --prepare) PREPARE_COMPONENTS=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v "$PYTHON" >/dev/null 2>&1 || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e "$REPO_ROOT[dev]"

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
  if ! ollama show "$MODEL" >/dev/null 2>&1; then
    printf 'Pulling logical Omni tag %s\n' "$MODEL"
    ollama pull "$MODEL"
  fi
  if ! ollama show "$LANGUAGE_MODEL" >/dev/null 2>&1; then
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
