"""Low-cost per-camera image sampling; QR data stays in memory until intake."""

from __future__ import annotations

import hashlib
import io
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from harness.camera import CameraSet
from harness.camera_ocr import RollingOCR

logger = logging.getLogger(__name__)
IDLE_PERIOD, ACTIVE_PERIOD = 1.5, 0.30
IDLE_WIDTH, ACTIVE_WIDTH = 640, 1280
ACTIVE_HOLD = 5.0
QR_REARM = 7.0


@dataclass
class CameraActivity:
    next_at: float = 0.0
    active_until: float = 0.0
    signature: np.ndarray | None = None

    def due(self, now: float) -> int | None:
        if now < self.next_at:
            return None
        active = now < self.active_until
        self.next_at = now + (ACTIVE_PERIOD if active else IDLE_PERIOD)
        return ACTIVE_WIDTH if active else IDLE_WIDTH

    def observe(self, image: Image.Image, now: float, *, detected: bool) -> None:
        thumbnail = image.resize((40, 24), Image.Resampling.BILINEAR)
        sampled = np.asarray(thumbnail, dtype=np.float32)
        # A uniform lighting change alone does not signal movement.
        signature = sampled - np.median(sampled)
        moving = self.signature is not None and np.mean(np.abs(signature - self.signature) > 20) > 0.10
        self.signature = signature
        if moving or detected:
            self.active_until = now + ACTIVE_HOLD
            self.next_at = min(self.next_at, now + ACTIVE_PERIOD)


@dataclass
class _Control:
    value: int
    low: int
    high: int
    step: int
    menu: dict[int, str] = field(default_factory=dict)


def _camera_controls(device: str, *arguments: str) -> str:
    if not shutil.which("v4l2-ctl"):
        return ""
    try:
        result = subprocess.run(["v4l2-ctl", "-d", device, *arguments],
                                capture_output=True, text=True, timeout=1.5, check=False)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _controls(text: str) -> dict[str, _Control]:
    entries = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"\s*(\w+)\s+0x[\da-f]+\s+\((\w+)\)\s*:\s*(.*)", line, re.I)
        if match:
            name, kind, fields = match.groups()
            values = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)\b", fields)}
            current = None
            if "value" not in values or "read-only" in fields or "disabled" in fields:
                continue
            low, high = values.get("min", 0), values.get("max", 1 if kind == "bool" else 0)
            current = entries[name] = _Control(values["value"], low, high, max(1, values.get("step", 1)))
        elif current is not None and (menu := re.match(r"\s*(\d+):\s*(.+)", line)):
            current.menu[int(menu[1])] = menu[2].casefold()
    return entries


class CameraTuner:
    """Use supported device autos plus slow, bounded brightness/contrast trims."""

    def __init__(self):
        self.original = {}
        self.current = {}
        self.quality = {}

    def _set(self, device: str, name: str, value: int) -> None:
        control = self.current[device][name]
        if control.value == value:
            return
        # v4l2-ctl emits no stdout on a successful set. Read back the control
        # before adopting its new value.
        _camera_controls(device, f"--set-ctrl={name}={value}")
        result = _camera_controls(device, f"--get-ctrl={name}")
        if match := re.search(r":\s*(-?\d+)", result):
            control.value = int(match[1])

    def observe(self, device: str, image: Image.Image, now: float) -> None:
        if device not in self.current:
            controls = _controls(_camera_controls(device, "--list-ctrls-menus"))
            self.current[device] = controls
            self.original[device] = {name: item.value for name, item in controls.items()}
            for name in ("focus_automatic_continuous", "focus_auto", "white_balance_automatic",
                         "white_balance_temperature_auto", "auto_white_balance", "autogain"):
                if name in controls:
                    self._set(device, name, 1)
            for name in ("auto_exposure", "exposure_auto"):
                if name in controls:
                    for value, label in controls[name].menu.items():
                        if label in {"auto", "auto mode", "aperture priority mode"}:
                            self._set(device, name, value)
                            break
        current = self.current[device]
        last, previous, samples = self.quality.get(device, (now - 10, "", 0))
        if now - last < 3 or not ({"brightness", "contrast"} & current.keys()):
            return
        low, median, high = np.percentile(np.asarray(image.resize((64, 48))), (5, 50, 95))
        target = ("brightness+" if median < 38 else "brightness-" if median > 222 else
                  "contrast+" if high - low < 58 and low > 8 and high < 245 else "")
        samples = samples + 1 if target == previous else 1
        self.quality[device] = (last, target, samples)
        if not target or samples < 3 or target[:-1] not in current:
            return
        name = target[:-1]
        control = current[name]
        original = self.original[device][name]
        span = control.high - control.low
        if span <= 0:
            return
        delta = max(control.step, round(span * 0.05 / control.step) * control.step)
        bound = max(delta, control.step * int(0.20 * span / control.step))
        candidate = control.value + (delta if target.endswith("+") else -delta)
        value = min(control.high, original + bound, max(control.low, original - bound, candidate))
        self._set(device, name, value)
        self.quality[device] = (now, "", 0)

    def restore(self) -> None:
        for device, controls in self.current.items():
            for name, original in self.original[device].items():
                if controls[name].value != original:
                    self._set(device, name, original)
        self.current.clear()
        self.original.clear()
        self.quality.clear()


class CameraScanner:
    def __init__(self, cameras: CameraSet, *, enabled: Callable[[], bool],
                 submit: Callable[..., None], auto_tune: bool = True, orient=None):
        self.cameras = cameras
        self.enabled = enabled
        self.submit = submit
        self.tuner = CameraTuner() if auto_tune else None
        self.orient = orient
        self.ocr = RollingOCR()
        self.ocr_candidates = queue.Queue(maxsize=1)
        self.last_ocr = {}
        self.ocr_seen = {}
        self.next_ocr = 0.
        self.activity = {}
        self.seen = {}
        self.pending = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=target, name=name, daemon=True) for target, name in (
            (self._scan, "omni-camera-scan"), (self._deliver, "omni-camera-intake"),
            (self._read_text, "omni-camera-ocr"))]

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=3)

    def _scan(self) -> None:
        try:
            import zxingcpp
        except ImportError:
            logger.warning("Camera scanning needs the zxing-cpp decoder")
            return
        try:
            while not self.stop.wait(0.10):
                if not self.enabled():
                    if self.tuner and self.tuner.current:
                        self.tuner.restore()
                    continue
                capture_at = time.monotonic()
                def width_for(device, capture_at=capture_at):
                    width = self.activity.setdefault(device, CameraActivity()).due(capture_at)
                    return 1920 if width and self.cameras.stream is not None else width

                for device, camera_id, frame in self.cameras.scan_frames(width_for):
                    if self.stop.is_set() or not self.enabled():
                        break
                    try:
                        with Image.open(io.BytesIO(frame)) as decoded:
                            gray = ImageOps.exif_transpose(decoded).convert("L")
                        codes = decode_qr(gray, zxingcpp)
                        now = time.monotonic()
                        detected = False
                        for code in codes:
                            detected = self._queue(code, camera_id, now) or detected
                        self.activity[device].observe(gray, now, detected=detected)
                        if self.tuner is not None:
                            self.tuner.observe(device, gray, now)
                        if self.orient is not None:
                            self.orient.observe(device, frame, now)
                        if self.ocr.binary and not codes and now >= self.next_ocr and now - self.last_ocr.get(device, -20) > 7:
                            try:
                                self.ocr_candidates.put_nowait((device, camera_id, frame))
                                self.next_ocr = now + 1.5
                                self.last_ocr[device] = now
                            except queue.Full:
                                pass
                    except (OSError, ValueError):
                        logger.debug("Camera scan could not decode an image")
        finally:
            if self.tuner is not None:
                self.tuner.restore()

    def _queue(self, code: str, camera_id: str, now: float) -> bool:
        identity = hashlib.sha256(code.encode()).digest()
        self.seen = {digest: at for digest, at in self.seen.items() if now - at < QR_REARM}
        previous = identity in self.seen
        if len(self.seen) < 128 or previous:
            self.seen[identity] = now
        else:
            return False
        if not previous:
            try:
                self.pending.put_nowait((code, camera_id, now))
                return True
            except queue.Full:
                self.seen.pop(identity, None)
        return False

    def _read_text(self) -> None:
        while not self.stop.is_set():
            try:
                device, camera, frame = self.ocr_candidates.get(timeout=.2)
            except queue.Empty:
                continue
            if not self.enabled():
                continue
            try:
                frame = self.cameras.correct_orientation(device, frame)
            except (ValueError, OSError):
                logger.debug("Could not correct OCR frame orientation")
            result = self.ocr.recognize(frame)
            if result and self.enabled() and not self.stop.is_set():
                now = time.monotonic()
                self.ocr_seen = {key: at for key, at in self.ocr_seen.items() if now - at < 30}
                identity = hashlib.sha256(" ".join(result.text.lower().split()).encode()).digest()
                if identity not in self.ocr_seen:
                    self.submit(result.text, camera, source="ocr", confidence=result.confidence)
                self.ocr_seen[identity] = now

    def _deliver(self) -> None:
        while not self.stop.is_set():
            try:
                code, camera, at = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.enabled() and time.monotonic() - at <= 45:
                self.submit(code, camera)


def decode_qr(gray: Image.Image, decoder) -> list[str]:
    def read(image):
        return decoder.read_barcodes(image, formats=decoder.BarcodeFormat.QRCode)

    result = read(gray) or read(ImageOps.autocontrast(gray, cutoff=1))
    codes = []
    for item in result[:8]:
        try:
            raw = item.bytes
            if 0 < len(raw) <= 4096:
                codes.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return codes
