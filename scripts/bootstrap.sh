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
INSTALL_CAMERA_STREAM=1

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
  --no-camera-stream Keep persistent Jetson camera streaming disabled
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
    --no-camera-stream) INSTALL_CAMERA_STREAM=0 ;;
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
    && command -v xdotool >/dev/null 2>&1 \
    && command -v xwininfo >/dev/null 2>&1 \
    && command -v import >/dev/null 2>&1 \
    && "$1" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Atspi", "2.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
assert all(Gst.ElementFactory.find(name) is not None for name in ("pipewiresrc", "pngenc", "videoconvert"))
try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
except ValueError:
    gi.require_version("AppIndicator3", "0.1")
PY
}

if ((INSTALL_HARNESS)) && [[ $(uname -s) == Linux ]]; then
  if ! desktop_harness_dependencies_ready "$PYTHON" \
    || ! command -v cc >/dev/null 2>&1 \
    || ! command -v pkg-config >/dev/null 2>&1 \
    || ! pkg-config --exists libpulse; then
    command -v apt-get >/dev/null 2>&1 || {
      printf 'Desktop harness dependencies are missing and apt-get is unavailable.\n' >&2
      exit 1
    }
    printf 'Installing desktop indicator and PulseAudio client dependencies...\n'
    sudo apt-get -o DPkg::Lock::Timeout=600 update
    indicator_package=gir1.2-ayatanaappindicator3-0.1
    if ! apt-cache show "$indicator_package" >/dev/null 2>&1; then
      indicator_package=gir1.2-appindicator3-0.1
    fi
    sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y \
      python3-gi gir1.2-gtk-3.0 gir1.2-atspi-2.0 "$indicator_package" pulseaudio-utils \
      build-essential pkg-config libpulse-dev \
      xdotool x11-utils imagemagick gir1.2-gstreamer-1.0 gstreamer1.0-tools \
      gstreamer1.0-pipewire gstreamer1.0-plugins-base gstreamer1.0-plugins-good
  fi
  desktop_harness_dependencies_ready "$PYTHON" || {
    printf 'Desktop harness dependencies remain unavailable after installation.\n' >&2
    exit 1
  }
fi

"$PYTHON" -m venv "$VENV"
if ((INSTALL_HARNESS)) && [[ $(uname -s) == Linux ]]; then
  pulse_capture=$(mktemp "$VENV/bin/.omni-pulse-capture.XXXXXX")
  trap 'unlink "$pulse_capture" 2>/dev/null || true' EXIT
  read -r -a pulse_flags <<<"$(pkg-config --cflags --libs libpulse)"
  cc -Wall -Wextra -O2 "$REPO_ROOT/harness/pulse_capture.c" -o "$pulse_capture" "${pulse_flags[@]}"
  chmod 755 "$pulse_capture"
  mv -fT -- "$pulse_capture" "$VENV/bin/omni-pulse-capture"
  trap - EXIT
  # Use the PyGObject binding from the matching host Python without exposing
  # the host's Python distribution metadata to pip.
  gi_package=$("$PYTHON" -c 'import gi; from pathlib import Path; print(Path(gi.__file__).resolve().parent)')
  venv_site=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  if [[ ! -e $venv_site/gi ]]; then
    ln -sfnT -- "$gi_package" "$venv_site/gi"
  fi
  desktop_harness_dependencies_ready "$VENV/bin/python" || {
    printf 'Desktop harness dependencies are unavailable in the runtime venv.\n' >&2
    exit 1
  }
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e "${REPO_ROOT}[dev]"
if ! "$VENV/bin/python" -m pip install -e "${REPO_ROOT}[pdf]"; then
  printf 'Optional PDF layout parser could not be installed. Continuing bootstrap.\n' >&2
fi

if ((INSTALL_HARNESS && INSTALL_CAMERA_STREAM)); then
  OMNI_VENV="$VENV" "$SCRIPT_DIR/install-camera-stream.sh"
elif ((!INSTALL_CAMERA_STREAM)); then
  "$SCRIPT_DIR/install-camera-stream.sh" --disable
fi

# Media capture and conversion need an ffmpeg with V4L2 input and swscale.
# NVIDIA's JetPack 7 package is a hardware-decode build configured with
# --disable-everything --disable-swscale (no V4L2 input, no scale/crop, no
# x264), so the runtime would find no cameras and could not convert media.
# Fetch a self-contained static build into vendor/ffmpeg; the service units
# put vendor/ffmpeg/bin first on PATH. Hosts with a capable ffmpeg skip this.
ffmpeg_is_capable() {
  "$1" -hide_banner -devices 2>/dev/null | grep -Eq '^ *D[^ ]* +(video4linux2,)?v4l2([, ]|$)' \
    && "$1" -hide_banner -filters 2>/dev/null | grep -Eq '^ *[^ ]+ +scale +'
}
BUNDLED_FFMPEG=$REPO_ROOT/vendor/ffmpeg/bin/ffmpeg
if [[ $(uname -s) == Linux ]] && ! { command -v ffmpeg >/dev/null 2>&1 && ffmpeg_is_capable ffmpeg; } \
  && ! { [[ -x $BUNDLED_FFMPEG ]] && ffmpeg_is_capable "$BUNDLED_FFMPEG"; }; then
  case $(uname -m) in
    aarch64|arm64) ffmpeg_arch=linuxarm64 ;;
    x86_64) ffmpeg_arch=linux64 ;;
    *) ffmpeg_arch="" ;;
  esac
  if [[ -n $ffmpeg_arch ]]; then
    printf 'System ffmpeg lacks V4L2 capture or scaling; installing a static ffmpeg in vendor/ffmpeg\n'
    ffmpeg_tmp=$(mktemp -d)
    curl -fsSL --retry 3 -o "$ffmpeg_tmp/ffmpeg.tar.xz" \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-${ffmpeg_arch}-gpl-8.1.tar.xz"
    tar -xJf "$ffmpeg_tmp/ffmpeg.tar.xz" -C "$ffmpeg_tmp"
    rm -rf "$REPO_ROOT/vendor/ffmpeg"
    mkdir -p "$REPO_ROOT/vendor/ffmpeg"
    cp -a "$ffmpeg_tmp"/ffmpeg-n8.1-*/bin "$REPO_ROOT/vendor/ffmpeg/"
    rm -rf "$ffmpeg_tmp"
    ffmpeg_is_capable "$BUNDLED_FFMPEG" \
      || { printf 'The bundled ffmpeg lacks V4L2 capture or scaling\n' >&2; exit 1; }
  fi
fi

if ((INSTALL_LAYA)); then
  "$REPO_ROOT/scripts/bootstrap_laya.sh"
fi

if ((INSTALL_POINTING)); then
  if [[ ${OMNI_POINTING_MODEL:-} == moondream/moondream3.1-9B-A2B ]]; then
    "$REPO_ROOT/scripts/bootstrap_photon_pointing.sh"
  else
    "$REPO_ROOT/scripts/bootstrap_pointing.sh"
  fi
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
  OMNI_MODEL="$MODEL" "$REPO_ROOT/scripts/build_llama_cpp.sh"
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
