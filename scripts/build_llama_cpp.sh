#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR=${1:-$REPO_ROOT/vendor/llama.cpp}
PINNED_COMMIT=458681e1d5d4a29a1463c4732e03226cf384b997
PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-pcm-stream.patch
PERSISTENT_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-persistent.patch
STREAM_STATE_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-stream-state.patch
JSON_PATTERN_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-json-pattern-safety.patch
JSON_STRING_BOUNDS_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-json-string-bounds.patch
CONDITIONING_CACHE_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-conditioning-cache.patch
TOKEN_ROWS_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-token-rows.patch
INDEXED_VIEWS_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-indexed-tts-views.patch
REFERENCE_ENCODER_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-reference-encoder.patch
REFERENCE_IGPU_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-reference-integrated-gpu.patch
REFERENCE_PROMPT_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-reference-prompt.patch
UTTERANCE_STATE_PATCH_FILE=$REPO_ROOT/patches/llama.cpp-qwen3tts-utterance-state.patch
BUILD_DIR=${LLAMA_CPP_BUILD_DIR:-$SOURCE_DIR/build}
if command -v nproc >/dev/null 2>&1; then
  default_jobs=$(nproc)
else
  default_jobs=$(sysctl -n hw.logicalcpu)
fi

# NVIDIA Tegra modules (Jetson) run an integrated GPU on unified memory: the
# SoC has exactly one known compute capability, and nvcc competes with that
# same RAM while compiling. Detect both so an arm64 build neither fans out to
# the discrete architecture list nor OOMs itself mid-compile.
tegra_soc() {
  local compatible
  [[ -r /proc/device-tree/compatible ]] || return 0
  compatible=$(tr '\0' '\n' </proc/device-tree/compatible)
  local entry
  while IFS= read -r entry; do
    case $entry in
      nvidia,tegra[0-9]*) printf '%s' "${entry#nvidia,}"; return 0 ;;
    esac
  done <<<"$compatible"
}

tegra_cuda_architecture() {
  case $(tegra_soc) in
    tegra210) printf '53' ;;  # TX1, Nano
    tegra186) printf '62' ;;  # TX2
    tegra194) printf '72' ;;  # Xavier AGX/NX
    tegra234) printf '87' ;;  # Orin AGX/NX/Nano
    tegra264) printf '101' ;; # Thor
  esac
}

TEGRA_SOC=$(tegra_soc)
if [[ -n "$TEGRA_SOC" ]] && command -v free >/dev/null 2>&1; then
  # nvcc peaks around 2 GiB per translation unit on the ggml CUDA kernels.
  # Budget against memory that is actually *available*, not installed: on a
  # module already serving models, the GPU has carved its allocations out of
  # this same pool, and sizing from total RAM would fan out into an OOM.
  memory_gib=$(free -g | awk '$1 == "Mem:" {print $7}')
  if ! [[ "$memory_gib" =~ ^[0-9]+$ ]]; then
    memory_gib=$(free -g | awk '$1 == "Mem:" {print $2}')
  fi
  if [[ "$memory_gib" =~ ^[0-9]+$ ]]; then
    memory_jobs=$(( memory_gib / 2 ))
    (( memory_jobs < 1 )) && memory_jobs=1
    (( memory_jobs < default_jobs )) && default_jobs=$memory_jobs
  fi
fi
BUILD_JOBS=${LLAMA_CPP_BUILD_JOBS:-$default_jobs}

# Check before cloning or patching: runtime libraries and Torch wheels can be
# installed without the native compiler. The helper repairs APT installations
# and verifies nvcc after installation; it never initializes a CUDA device.
CUDA_COMPILER=""
if [[ $(uname -s) == Linux ]]; then
  CUDA_COMPILER=$("${PYTHON:-python3}" "$REPO_ROOT/runtime/cuda_toolchain.py" --build-dir "$BUILD_DIR")
fi

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

apply_patch_file() {
  local patch_file=$1
  local label=$2
  if git -C "$SOURCE_DIR" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    printf '%s patch is already applied\n' "$label"
  elif git -C "$SOURCE_DIR" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$SOURCE_DIR" apply "$patch_file"
    printf 'Applied %s patch\n' "$label"
  else
    printf '%s patch does not apply cleanly\n' "$label" >&2
    exit 1
  fi
}

# These three patches overlap in common/common.h and tools/tts/tts.cpp, so once
# all of them are applied no single one reverse-applies on its own any more:
# its context lines carry the others' edits. Checking each patch individually
# made every rebuild fail with "does not apply cleanly" on a correctly patched
# tree, and `git apply` given several files applies them as one combined set
# rather than sequentially, so reversing them together does not help either.
#
# Detect the applied state from content each patch uniquely introduces.
patch_marker_present() {
  local file=$1
  local marker=$2
  [[ -f "$SOURCE_DIR/$file" ]] && grep -qF -- "$marker" "$SOURCE_DIR/$file"
}

patch_set_applied() {
  patch_marker_present common/arg.cpp '"--tts-stream-frames"' \
    && patch_marker_present common/arg.cpp '"--tts-persistent"' \
    && patch_marker_present tools/mtmd/clip.cpp \
      'state_out describes this exact output boundary'
}

if patch_set_applied; then
  printf 'Qwen3-TTS patch set is already applied\n'
else
  apply_patch_file "$PATCH_FILE" "Qwen3-TTS PCM stream"
  apply_patch_file "$PERSISTENT_PATCH_FILE" "Qwen3-TTS persistent worker"
  apply_patch_file "$STREAM_STATE_PATCH_FILE" "Qwen3-TTS streaming decoder state"
  patch_set_applied || {
    printf 'Qwen3-TTS patch set did not produce the expected sources\n' >&2
    exit 1
  }
fi

# Independent common-code fix; do not fold it into the overlapping TTS markers.
apply_patch_file "$JSON_PATTERN_PATCH_FILE" "JSON pattern string safety"
apply_patch_file "$JSON_STRING_BOUNDS_PATCH_FILE" "JSON string length bounds"
if patch_marker_present common/arg.cpp '"--tts-no-conditioning-cache"' \
  && patch_marker_present tools/mtmd/mtmd-helper-gen.cpp 'mtmd_helper_gen_audio_cache_init(' \
  && patch_marker_present tools/tts/tts.cpp 'conditioning_cache.get()'; then
  printf 'Qwen3-TTS immutable conditioning cache patch is already applied\n'
else
  apply_patch_file "$CONDITIONING_CACHE_PATCH_FILE" "Qwen3-TTS immutable conditioning cache"
fi
if patch_marker_present common/arg.cpp '"--tts-conditioning-cache-rows"' \
  && patch_marker_present src/llama-model.cpp 'uint32_t llama_model_get_tok_embd_row(' \
  && patch_marker_present tools/mtmd/mtmd-helper-gen.cpp 'conditioning->token_rows.find(t)'; then
  printf 'Qwen3-TTS bounded token rows patch is already applied\n'
else
  apply_patch_file "$TOKEN_ROWS_PATCH_FILE" "Qwen3-TTS bounded token rows"
fi
if patch_marker_present ggml/include/gguf.h 'gguf_select_tensor_view(' \
  && patch_marker_present src/llama-model-loader.cpp 'indexed_view = true;' \
  && patch_marker_present tools/mtmd/clip.cpp 'robit.ollama-audio-bridge.v2'; then
  printf 'Indexed TTS source views patch is already applied\n'
else
  apply_patch_file "$INDEXED_VIEWS_PATCH_FILE" "Indexed TTS source views"
fi
if patch_marker_present tools/tts/reference-encode.cpp 'params.reference_encoder = true;' \
  && patch_marker_present tools/mtmd/clip-model.h 'ref_semantic_book' \
  && patch_marker_present tools/mtmd/models/qwen3tts-refenc.cpp 'reference_downsample'; then
  printf 'Qwen3-TTS reference acoustic encoder patch is already applied\n'
else
  apply_patch_file "$REFERENCE_ENCODER_PATCH_FILE" "Qwen3-TTS reference acoustic encoder"
fi
apply_patch_file "$REFERENCE_IGPU_PATCH_FILE" "Qwen3-TTS integrated GPU residency"
if patch_marker_present tools/mtmd/mtmd-helper-gen.cpp 'reference_decoder_state' \
  && patch_marker_present tools/tts/tts.cpp 'robit.qwen3tts.reference-prompt.v1'; then
  printf 'Qwen3-TTS full reference prompt patch is already applied\n'
else
  apply_patch_file "$REFERENCE_PROMPT_PATCH_FILE" "Qwen3-TTS full reference prompt"
fi
if patch_marker_present common/arg.cpp '"--tts-utterance-controls"' \
  && patch_marker_present tools/tts/tts.cpp 'robit.qwen3tts.utterance-control.v1' \
  && patch_marker_present tools/mtmd/mtmd-helper-gen.cpp 'set_waveform_state'; then
  printf 'Qwen3-TTS utterance decoder state patch is already applied\n'
else
  apply_patch_file "$UTTERANCE_STATE_PATCH_FILE" "Qwen3-TTS utterance decoder state"
fi

cmake_options=(-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release)
case $(uname -s) in
  Darwin) cmake_options+=(-DGGML_METAL=ON) ;;
  Linux)
    cmake_options+=(-DGGML_CUDA=ON)
    if [[ -n "$TEGRA_SOC" ]]; then
      # ggml's CUDA allocator reserves a 32 GiB virtual address pool
      # (CUDA_POOL_VMM_MAX_SIZE = 1<<35) per device. A discrete card has the
      # address space for that; a Tegra module addresses one unified pool that
      # is itself smaller than the reservation, so cuMemAddressReserve fails
      # with "out of memory" and aborts the worker -- observed on an AGX Orin
      # as llama-tts exiting -6 whenever a new worker had to build a pool.
      # Plain cudaMalloc has no such reservation.
      cmake_options+=(-DGGML_CUDA_NO_VMM=ON)
      printf 'Disabling the CUDA VMM pool: its 32 GiB reservation cannot fit Tegra unified memory\n'
    fi
    cuda_architectures=${OMNI_CUDA_ARCHITECTURES:-$(tegra_cuda_architecture)}
    if [[ -n "$cuda_architectures" ]]; then
      # Pin the exact integrated-GPU capability. Left unset, ggml builds its
      # discrete-desktop architecture list, which both wastes hours of Jetson
      # compile time and can emit no kernel the on-board GPU can run.
      cmake_options+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_architectures")
      printf 'Building CUDA kernels for compute capability %s\n' "$cuda_architectures"
    fi
    ;;
  *)
    printf 'Use scripts/build_llama_cpp.ps1 on Windows.\n' >&2
    exit 1
    ;;
esac
build_options=()
if [[ $(uname -s) == Linux ]]; then
  compiler_configuration=$("${PYTHON:-python3}" "$REPO_ROOT/runtime/cuda_toolchain.py" \
    --build-dir "$BUILD_DIR" --configure "$SOURCE_DIR" --compiler "$CUDA_COMPILER" \
    -- "${cmake_options[@]}")
  if [[ "$compiler_configuration" == changed ]]; then
    # Make may retain objects after required compiler arguments change. Only
    # an actual toolchain change needs a clean build; ordinary redeploys do not.
    build_options+=(--clean-first)
  elif [[ "$compiler_configuration" != unchanged ]]; then
    printf 'Unexpected CUDA compiler configuration result\n' >&2
    exit 1
  fi
else
  cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" "${cmake_options[@]}"
fi
printf 'Checking patched llama.cpp targets incrementally in %s\n' "$BUILD_DIR"
cmake --build "$BUILD_DIR" "${build_options[@]}" --target llama-server llama-tts llama-tts-reference-encode --parallel "$BUILD_JOBS"
if [[ $(uname -s) == Linux ]]; then
  "${PYTHON:-python3}" "$REPO_ROOT/runtime/cuda_toolchain.py" --build-dir "$BUILD_DIR" --build-complete
fi

"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-stream-frames'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-persistent'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-no-conditioning-cache'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-conditioning-cache-rows'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-reference-prompt-file'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-utterance-controls'
[[ -x "$BUILD_DIR/bin/llama-tts-reference-encode" ]] || { printf 'Reference encoder was not built\n' >&2; exit 1; }
printf 'Built patched llama-server and llama-tts in %s\n' "$BUILD_DIR/bin"
