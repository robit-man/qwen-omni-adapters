#!/usr/bin/env bash
set -Eeuo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv=${OMNI_VENV:-$repo/.venv}
marker=$repo/runtime-data/state/camera-stream.enabled
if [[ ${1:-} == --disable ]]; then
  rm -f -- "$marker"
  exit 0
fi
[[ $(uname -m) == aarch64 && -r /proc/device-tree/model ]] || exit 0
grep -aq 'Jetson AGX Orin' /proc/device-tree/model || exit 0

ready() {
  command -v gst-launch-1.0 >/dev/null && command -v tesseract >/dev/null &&
    "$venv/bin/python" - <<'PY'
import aiortc
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
assert all(Gst.ElementFactory.find(name) is not None for name in (
    "v4l2src", "shmsink", "shmsrc", "videorate", "appsrc", "appsink",
    "nvvidconv", "nvv4l2h264enc", "h264parse",
))
PY
}
if ! ready >/dev/null 2>&1; then
  if ! command -v gst-inspect-1.0 >/dev/null || ! gst-inspect-1.0 shmsink >/dev/null 2>&1 ||
    ! command -v tesseract >/dev/null; then
    sudo apt-get -o DPkg::Lock::Timeout=600 update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y \
      gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
  fi
  "$venv/bin/python" -m pip install 'aiortc==1.15.0'
fi
ready
install -d -m 700 "$(dirname -- "$marker")"
(umask 077; touch "$marker")
printf 'Persistent Jetson camera streaming and passive QR/OCR enabled.\n'

# Keep OpenCV's system NumPy ABI isolated from the model runtime.
install_advanced_ocr() {
  local ocr="$repo/runtime-data/components/ocr-venv"
  if [[ -x $ocr/bin/python ]] && "$ocr/bin/python" -c 'import cv2, numpy, pytesseract, PIL'; then
    return 0
  fi
  if ! /usr/bin/python3 -c 'import cv2, numpy' >/dev/null 2>&1; then
    sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y python3-opencv || return
  fi
  /usr/bin/python3 -m venv --system-site-packages "$ocr" || return
  "$ocr/bin/python" -m pip install 'pytesseract==0.3.13' || return
  "$ocr/bin/python" -c 'import cv2, numpy, pytesseract, PIL'
}
install_advanced_ocr || printf 'Optional Omnius batch OCR could not be provisioned.\n' >&2
