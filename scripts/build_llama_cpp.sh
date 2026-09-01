#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR=${1:-$REPO_ROOT/vendor/llama.cpp}
PINNED_COMMIT=458681e1d5d4a29a1463c4732e03226cf384b997
PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-pcm-stream.patch
BUILD_DIR=${LLAMA_CPP_BUILD_DIR:-$SOURCE_DIR/build}
if command -v nproc >/dev/null 2>&1; then
  default_jobs=$(nproc)
else
  default_jobs=$(sysctl -n hw.logicalcpu)
fi
BUILD_JOBS=${LLAMA_CPP_BUILD_JOBS:-$default_jobs}

cloned=0
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
  cloned=1
fi
if (( cloned )); then
  git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"
fi

current_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD)
if [[ "$current_commit" != "$PINNED_COMMIT" ]]; then
  printf 'llama.cpp must be at pinned commit %s; found %s\n' \
    "$PINNED_COMMIT" "$current_commit" >&2
  exit 1
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  printf 'Qwen3-TTS PCM stream patch is already applied\n'
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
  printf 'Applied Qwen3-TTS PCM stream patch\n'
else
  printf 'Qwen3-TTS PCM stream patch does not apply cleanly\n' >&2
  exit 1
fi

cmake_options=(-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release)
case $(uname -s) in
  Darwin) cmake_options+=(-DGGML_METAL=ON) ;;
  Linux) cmake_options+=(-DGGML_CUDA=ON) ;;
  *)
    printf 'Use scripts/build_llama_cpp.ps1 on Windows.\n' >&2
    exit 1
    ;;
esac
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" "${cmake_options[@]}"
cmake --build "$BUILD_DIR" --target llama-server llama-tts --parallel "$BUILD_JOBS"

"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-stream-frames'
printf 'Built patched llama-server and llama-tts in %s\n' "$BUILD_DIR/bin"
