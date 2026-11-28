"""Estimate full-circle camera roll off the voice path using RightWayUp Pico."""

from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def _difference(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def _roll(probabilities) -> tuple[float, float]:
    bins = np.asarray(probabilities, dtype=np.float32).reshape(360)
    peak = int(np.argmax(bins))
    offsets = np.arange(-10, 11)
    weights = bins[(peak + offsets) % 360]
    angle = (peak + float(np.dot(weights, offsets)) / max(float(weights.sum()), 1e-9)) % 360
    distance = np.abs((np.arange(360) - angle + 180) % 360 - 180)
    return _difference(angle, 0), float(bins[distance <= 10].sum())


def _image_input(frame: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(frame)) as source:
        upright = ImageOps.exif_transpose(source).convert("RGB")
    upright.thumbnail((70, 70), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (70, 70), (124, 116, 104))
    canvas.paste(upright, ((70 - upright.width) // 2, (70 - upright.height) // 2))
    values = np.asarray(canvas, dtype=np.float32) / 255
    values -= np.array((0.485, 0.456, 0.406), dtype=np.float32)
    values /= np.array((0.229, 0.224, 0.225), dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


class CameraOrientation:
    def __init__(self, model: Path):
        self.model = model
        self._session = None
        self._failed = False
        self._next = {}
        self._global_next = 0.0
        self._angles = {}
        self._history = {}
        self._lock = threading.Lock()

    def _predict(self, frame: bytes) -> tuple[float, float] | None:
        if self._failed:
            return None
        try:
            if self._session is None:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                self._session = ort.InferenceSession(str(self.model), options,
                                                     providers=["CPUExecutionProvider"])
            return _roll(self._session.run(["prob"], {"image": _image_input(frame)})[0])
        except Exception:  # optional estimator must never disable capture
            self._failed = True
            logger.warning("Camera orientation is unavailable; check the configured RightWayUp Pico ONNX model")
            return None

    def observe(self, device: str, frame: bytes, now: float) -> None:
        if now < self._global_next or now < self._next.get(device, 0):
            return
        self._global_next = now + 0.35
        self._next[device] = now + 4
        estimate = self._predict(frame)
        history = self._history.setdefault(device, deque(maxlen=3))
        if estimate is None or not np.isfinite(estimate).all() or estimate[1] < 0.833:
            history.clear()
            return
        angle, _ = estimate
        if any(abs(_difference(angle, earlier)) > 12 for earlier in history):
            history.clear()
        history.append(angle)
        if len(history) == history.maxlen:
            estimate = _difference(angle + float(np.mean([_difference(value, angle) for value in history])), 0)
            with self._lock:
                self._angles[device] = (estimate, now)

    def roll(self, device: str) -> float:
        with self._lock:
            angle, updated = self._angles.get(device, (0.0, 0.0))
        return angle if abs(angle) >= 2 and time.monotonic() - updated < 120 else 0.0

    def correct(self, device: str, frame: bytes) -> bytes:
        angle = self.roll(device)
        if not angle:
            return frame
        with Image.open(io.BytesIO(frame)) as source:
            corrected = ImageOps.exif_transpose(source).convert("RGB").rotate(
                angle, Image.Resampling.BICUBIC, expand=True)
        output = io.BytesIO()
        corrected.save(output, "JPEG", quality=90)
        return output.getvalue()
