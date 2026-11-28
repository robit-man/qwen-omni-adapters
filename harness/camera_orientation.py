"""Estimate full-circle camera roll off the voice path using RightWayUp Pico."""

from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

ESTIMATE_INTERVAL = 4
ESTIMATE_MAX_AGE = 16
CORRECTION_MAX_AGE = 120
STRICT_CONFIDENCE = .833


@dataclass(frozen=True)
class RollReading:
    angle: float
    confidence: float
    upright_confidence: float


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
        self._readings = {}
        self._history = {}
        self._lock = threading.Lock()

    def _predict(self, frame: bytes) -> RollReading | None:
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
            probabilities = self._session.run(["prob"], {"image": _image_input(frame)})[0].reshape(360)
            angle, confidence = _roll(probabilities)
            upright = float(probabilities[:11].sum() + probabilities[-10:].sum())
            return RollReading(angle, confidence, upright)
        except Exception:  # optional estimator must never disable capture
            self._failed = True
            logger.warning("Camera orientation is unavailable; check the configured RightWayUp Pico ONNX model")
            return None

    def due(self, device: str, now: float) -> bool:
        return not self._failed and now >= self._global_next and now >= self._next.get(device, 0)

    def observe(self, device: str, frame: bytes, now: float) -> None:
        if not self.due(device, now):
            return
        self._global_next = now + 0.35
        self._next[device] = now + ESTIMATE_INTERVAL
        estimate = self._predict(frame)
        if estimate is not None and not np.isfinite(
            [estimate.angle, estimate.confidence, estimate.upright_confidence]
        ).all():
            estimate = None

        with self._lock:
            self._readings[device] = (estimate, now)
            history = self._history.setdefault(device, deque(maxlen=3))
            if estimate is None or estimate.confidence < STRICT_CONFIDENCE:
                history.clear()
                return
            angle = estimate.angle
            if any(abs(_difference(angle, previous.angle)) > 12 for previous in history):
                history.clear()
            history.append(estimate)
            if len(history) == history.maxlen:
                agreed_angle = _difference(angle + sum(_difference(previous.angle, angle)
                                                       for previous in history) / len(history), 0)
                self._angles[device] = (agreed_angle, now)

    def status(self, device: str, *, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            reading, measured = self._readings.get(device, (None, 0.))
            angle, confirmed = self._angles.get(device, (None, 0.))
            available = not self._failed
        fresh_reading = reading is not None and now - measured < ESTIMATE_MAX_AGE
        has_correction = angle is not None and now - confirmed < CORRECTION_MAX_AGE
        confirmed_current = fresh_reading and has_correction and confirmed == measured
        state = "unavailable" if not available else "uncertain"
        if confirmed_current:
            state = "upright" if abs(angle) <= 10 else "tilted"
        return {
            "state": state,
            "confidence": round(min(1., max(0., reading.confidence)), 3) if fresh_reading else None,
            "upright_confidence": round(min(1., max(0., reading.upright_confidence)), 3) if fresh_reading else None,
            "correction_degrees": round(angle if abs(angle) >= 2 else 0., 1) if has_correction else None,
            "held": has_correction and not confirmed_current,
        }

    def roll(self, device: str) -> float:
        return self.status(device)["correction_degrees"] or 0.

    def correct(self, device: str, frame: bytes) -> bytes:
        return rotate_image(frame, self.roll(device))


def rotate_image(frame: bytes, angle: float) -> bytes:
    if not angle:
        return frame
    with Image.open(io.BytesIO(frame)) as source:
        corrected = ImageOps.exif_transpose(source).convert("RGB").rotate(
            angle, Image.Resampling.BICUBIC, expand=True)
    output = io.BytesIO()
    corrected.save(output, "JPEG", quality=90)
    return output.getvalue()
