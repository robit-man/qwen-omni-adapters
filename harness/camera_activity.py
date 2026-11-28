"""Measure a fixed camera analysis stream and choose independent send limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTHS = (320, 640, 960, 1280, 1920, 1920)
RATES = (1, 5, 10, 24, 30, 60)
RATES_KBPS = (30, 150, 400, 1000, 3000, 6000)
THRESHOLDS = (.05, .15, .30, .55, .80)
SIZE = (180, 320)


@dataclass(frozen=True)
class Activity:
    kind: str
    changed_fraction: float
    local_fraction: float
    detail: float
    translation: tuple[int, int] | None
    spatial: float
    temporal: float


@dataclass(frozen=True)
class SendLimits:
    width: int
    fps: int
    bitrate_kbps: int


def _residual(current, previous):
    diff = current - previous
    return np.abs(diff - np.median(diff))


def _translated(current, previous):
    # Use a coarse luma comparison only when the frame changed substantially.
    a, b = current[::4, ::4], previous[::4, ::4]
    a = a - np.mean(a)
    b = b - np.mean(b)
    if min(float(a.std()), float(b.std())) < 6:
        return None
    cross = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    cross /= np.maximum(np.abs(cross), 1e-6)
    correlation = np.fft.irfft2(cross, s=a.shape, axes=(-2, -1))
    if float(correlation.max()) < .18:
        return None
    winner = np.unravel_index(correlation.argmax(), correlation.shape)
    delta = tuple(int((index if index <= size // 2 else index - size) * 4)
                  for index, size in zip(winner, a.shape, strict=True))
    if delta == (0, 0) or max(map(abs, delta)) > 40:
        return None
    return delta


class ActivityDetector:
    def __init__(self):
        self.previous = None
        self.timestamp = None

    def update(self, image, timestamp: float, *, importance: float = 0) -> Activity:
        if image.shape != SIZE or image.dtype != np.uint8:
            raise ValueError("expected a 320x180 uint8 luma frame")
        frame = image.astype(np.float32)
        previous, earlier = self.previous, self.timestamp
        self.previous, self.timestamp = frame, timestamp
        detail = min(1.0, float((np.abs(np.diff(frame, axis=1)).mean()
                              + np.abs(np.diff(frame, axis=0)).mean()) / 70))
        importance = float(np.clip(importance, 0, 1))
        spatial = max(importance, min(.79, 1.15 * detail))
        if previous is None or not 0 < timestamp - earlier <= .4:
            return Activity("baseline", 0, 0, detail, None, spatial, importance)

        residual = _residual(frame, previous)
        changed = float(np.mean(residual > 18))
        local = changed
        motion = None
        if changed > .12 and (translation := _translated(frame, previous)):
            aligned = np.roll(previous, translation, axis=(0, 1))
            a = frame[40:-40, 40:-40]
            b = aligned[40:-40, 40:-40]
            composable = _residual(a, b)
            before = float(np.mean(_residual(a, previous[40:-40, 40:-40]) > 18))
            after = float(np.mean(composable > 18))
            if before > .12 and after < .58 * before:
                motion, local = translation, after
                residual = composable
        areas = [area for row in np.array_split(residual > 18, 6)
                      for area in np.array_split(row, 8, axis=1)]
        region = sorted(float(np.mean(area)) for area in areas)[-3:]
        strongest = sum(region) / len(region)
        speed = min(2.0, (1/30) / (timestamp - earlier))
        if motion is not None:
            kind = "camera_motion"
            temporal = min(.95, .30 + .012 * max(map(abs, motion)) * speed + .45 * local)
        elif changed >= .56:
            kind, temporal = "scene_transition", .93
        elif changed >= .015 or strongest >= .10:
            kind = "local_motion"
            temporal = min(.82, (.09 + .55 * local + .47 * strongest) * speed)
        else:
            kind, temporal = "quiet", 0.
        spatial = max(spatial, min(.78, .55 * strongest + .40 * detail))
        return Activity(kind, changed, local, detail, motion, spatial,
                        max(importance, temporal))


class StreamQuality:
    """Fast rise and delayed fall applied only to the transmission branch."""
    def __init__(self, hold_seconds: float = 4):
        self.hold_seconds = hold_seconds
        self.bands = [0, 0]
        self.until = [0.0, 0.0]

    def update(self, activity: Activity, now: float) -> SendLimits:
        for i, score in enumerate((activity.spatial, activity.temporal)):
            goal = sum(score >= threshold for threshold in THRESHOLDS)
            if goal >= self.bands[i]:
                self.bands[i] = goal
                self.until[i] = now + self.hold_seconds
            elif now >= self.until[i]:
                self.bands[i] = goal
        spatial, temporal = self.bands
        # Initial encoder budget follows both dimensions and rises monotonically.
        bitrate = max(30, round((RATES_KBPS[spatial] * RATES_KBPS[temporal]) ** .5))
        return SendLimits(WIDTHS[spatial], RATES[temporal], bitrate)


