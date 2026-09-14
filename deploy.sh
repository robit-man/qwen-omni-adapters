#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

PROFILE=${1:-${OMNI_PROFILE:-qwen38}}
case "$PROFILE" in
  qwen38)
    : "${OMNI_MODEL:=robit/qwen3.8-27b-e03-obliterated-omni:q4km}"
    : "${OMNI_LANGUAGE_MODEL:=robit/qwen3.8-27b-obliterated-e03:27b}"
    ;;
  ornith15)
    : "${OMNI_MODEL:=robit/ornith-1.5-omni:q4km}"
    : "${OMNI_LANGUAGE_MODEL:=robit/ornith-1.5:9b}"
    ;;
  ornith15-obliterated)
    : "${OMNI_MODEL:=robit/ornith-1.5-obliterated-omni:q4km}"
    : "${OMNI_LANGUAGE_MODEL:=robit/ornith-1.5-obliterated:9b}"
    ;;
  *)
    printf 'usage: %s [qwen38|ornith15|ornith15-obliterated]\n' "$0" >&2
    exit 2
    ;;
esac
export OMNI_MODEL OMNI_LANGUAGE_MODEL

if [[ ! -x "$REPO_ROOT/.venv/bin/qwen-omni" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-tts" ]]; then
  "$REPO_ROOT/scripts/bootstrap.sh"
fi

exec "$REPO_ROOT/portal/start.sh" --daemon
