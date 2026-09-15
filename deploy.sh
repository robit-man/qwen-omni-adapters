#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# A Tegra module has one unified memory pool shared by the integrated GPU, the
# TTS worker, and the OS. The logical tag already carries the language model in
# its standard layers -- the combined tag and its core tag resolve to identical
# base/projector blobs -- but Ollama keys a loaded runner by tag name, so
# naming the core tag separately would load a second resident copy of the same
# weights. Run the language stage on the tag already being served there.
share_language_slot=0
if [[ -r /proc/device-tree/compatible ]] \
  && tr '\0' '\n' </proc/device-tree/compatible | grep -q '^nvidia,tegra[0-9]'; then
  share_language_slot=1
fi

PROFILE=${1:-${OMNI_PROFILE:-qwen38}}
case "$PROFILE" in
  qwen38)
    : "${OMNI_MODEL:=robit/qwen3.8-27b-e03-obliterated-omni:q4km}"
    default_language_model=robit/qwen3.8-27b-obliterated-e03:27b
    ;;
  ornith15)
    : "${OMNI_MODEL:=robit/ornith-1.5-omni:q4km}"
    default_language_model=robit/ornith-1.5:9b
    ;;
  ornith15-obliterated)
    : "${OMNI_MODEL:=robit/ornith-1.5-obliterated-omni:q4km}"
    default_language_model=robit/ornith-1.5-obliterated:9b
    ;;
  *)
    printf 'usage: %s [qwen38|ornith15|ornith15-obliterated]\n' "$0" >&2
    exit 2
    ;;
esac
(( share_language_slot )) && default_language_model=$OMNI_MODEL
: "${OMNI_LANGUAGE_MODEL:=$default_language_model}"
export OMNI_MODEL OMNI_LANGUAGE_MODEL

if [[ ! -x "$REPO_ROOT/.venv/bin/qwen-omni" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-tts" ]]; then
  "$REPO_ROOT/scripts/bootstrap.sh"
fi

# The ollama-unify GPU broker owns discrete-GPU allocation wherever it is
# installed, and the portal supervisor must not bypass its leases. Hosts
# without it -- every NVIDIA Tegra module, and any unmanaged workstation --
# run the portable direct supervisor instead, which proves residency with
# whatever per-process evidence the platform actually publishes.
if command -v docker >/dev/null 2>&1 && docker gpu discover >/dev/null 2>&1; then
  exec "$REPO_ROOT/portal/start.sh" --daemon
fi

direct=("$REPO_ROOT/.venv/bin/qwen-omni-daemon" serve --allow-direct-gpu)
if ! command -v cloudflared >/dev/null 2>&1; then
  direct+=(--no-tunnel)
  printf 'cloudflared is not installed; serving the portal on loopback only.\n' >&2
fi
printf 'No GPU broker on this host; starting the direct supervisor in the foreground.\n' >&2
exec "${direct[@]}"
