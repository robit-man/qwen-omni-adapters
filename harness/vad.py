"""Speech detection for the local call harness.

A faithful port of ``portal/static/call_vad.js``. The browser's live-call mode
has been tuned against real rooms, and its behaviour is the part worth keeping:
calibrate against the room, track a noise floor, require a short confirmation
before believing speech started, hold through brief pauses, and keep a pre-roll
so the first syllable is not clipped.

Sharing the constants matters as much as sharing the shape. A local harness that
drifted from the browser would answer differently in the same room, and the
difference would be blamed on the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Event = Literal["calibrating", "idle", "candidate", "start", "active", "utterance", "rejected"]


@dataclass(frozen=True)
class VadConfig:
    """Thresholds from the browser harness, in the same units."""

    calibration_ms: float = 800.0
    calibration_escape_threshold: float = 0.030
    calibration_escape_multiplier: float = 3.25
    start_threshold: float = 0.014
    release_threshold: float = 0.008
    noise_multiplier: float = 2.75
    release_multiplier: float = 1.6
    start_confirm_ms: float = 200.0
    silence_ms: float = 760.0
    min_active_ms: float = 400.0
    pre_roll_frames: int = 10
    initial_noise_floor: float = 0.003


@dataclass
class Utterance:
    """Audio the speaker actually produced, with its pre-roll attached."""

    chunks: list[np.ndarray]
    active_duration_ms: float

    def samples(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks)


@dataclass
class VadResult:
    event: Event
    active: bool
    utterance: Utterance | None = None
    level: float = 0.0
    threshold: float = 0.0


@dataclass
class Vad:
    """Frame-by-frame speech detection with hysteresis."""

    config: VadConfig = field(default_factory=VadConfig)
    _ready_at: float = 0.0
    _noise_floor: float = 0.0
    _pre_roll: list[np.ndarray] = field(default_factory=list)
    _candidate_frames: list[np.ndarray] = field(default_factory=list)
    _candidate_started_at: float | None = None
    _frames: list[np.ndarray] = field(default_factory=list)
    _speaking: bool = False
    _active_ms: float = 0.0
    _last_active_at: float = 0.0

    def __post_init__(self) -> None:
        self._noise_floor = self.config.initial_noise_floor
        self._ready_at = self.config.calibration_ms

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def reset(self, now: float = 0.0, *, calibrate: bool = False) -> None:
        self._ready_at = now + (self.config.calibration_ms if calibrate else 0.0)
        self._pre_roll = []
        self._candidate_frames = []
        self._candidate_started_at = None
        self._frames = []
        self._speaking = False
        self._active_ms = 0.0
        self._last_active_at = 0.0

    def _update_noise_floor(self, level: float, weight: float = 0.035) -> None:
        self._noise_floor = max(
            0.0005, self._noise_floor * (1.0 - weight) + level * weight
        )

    def process(
        self,
        samples: np.ndarray,
        now_ms: float,
        frame_ms: float,
        *,
        native_speech: bool | None = None,
    ) -> VadResult:
        """Feed one frame and learn what it means for the turn in progress.

        ``native_speech`` is the ReSpeaker's post-AEC speech decision. It gates
        only onset: once a real utterance has begun, ordinary inter-word gaps
        must not cut it off. Frames still enter pre-roll while the gate is
        closed, so opening it does not cost the first syllable.
        """

        level = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if math.isnan(level):
            level = 0.0
        config = self.config

        if now_ms < self._ready_at:
            escape = max(
                config.calibration_escape_threshold,
                self._noise_floor * config.calibration_escape_multiplier,
            )
            if level >= escape:
                # Loud enough that waiting out the calibration window would
                # swallow the start of a real sentence.
                self._ready_at = now_ms
            else:
                self._update_noise_floor(level, 0.12)
                self._pre_roll = []
                return VadResult("calibrating", False, level=level, threshold=escape)

        start_threshold = max(
            config.start_threshold, self._noise_floor * config.noise_multiplier
        )
        release_threshold = max(
            config.release_threshold, self._noise_floor * config.release_multiplier
        )

        if not self._speaking:
            self._pre_roll.append(samples)
            if len(self._pre_roll) > config.pre_roll_frames:
                self._pre_roll.pop(0)
            if native_speech is False:
                # Far-end playback can be much louder than room tone. Do not
                # learn it as the new floor, and do not let it accumulate the
                # software VAD confirmation needed to interrupt itself.
                self._candidate_frames = []
                self._candidate_started_at = None
                return VadResult(
                    "idle", False, level=level, threshold=start_threshold
                )
            if level < start_threshold:
                self._update_noise_floor(level)
                self._candidate_frames = []
                self._candidate_started_at = None
                return VadResult("idle", False, level=level, threshold=start_threshold)

            if self._candidate_started_at is None:
                self._candidate_started_at = now_ms
                # The pre-roll becomes the head of the utterance, so the first
                # syllable survives the confirmation delay.
                self._candidate_frames = self._pre_roll
                self._pre_roll = []
            else:
                self._candidate_frames.append(samples)

            if now_ms - self._candidate_started_at < config.start_confirm_ms:
                return VadResult(
                    "candidate", False, level=level, threshold=start_threshold
                )

            self._speaking = True
            self._frames = self._candidate_frames
            self._candidate_frames = []
            self._active_ms = max(
                frame_ms, now_ms - self._candidate_started_at + frame_ms
            )
            self._last_active_at = now_ms
            return VadResult("start", True, level=level, threshold=start_threshold)

        self._frames.append(samples)
        if level >= release_threshold:
            self._last_active_at = now_ms
            self._active_ms += frame_ms
        if now_ms - self._last_active_at < config.silence_ms:
            return VadResult("active", True, level=level, threshold=release_threshold)

        utterance = Utterance(chunks=self._frames, active_duration_ms=self._active_ms)
        accepted = utterance.active_duration_ms >= config.min_active_ms
        self.reset(now_ms)
        return VadResult(
            "utterance" if accepted else "rejected",
            False,
            utterance=utterance if accepted else None,
            level=level,
            threshold=release_threshold,
        )
