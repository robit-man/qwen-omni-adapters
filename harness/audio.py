"""Microphone capture and speaker playback for the local call harness.

PulseAudio's own tools do the work. ``parecord`` and ``paplay`` are already on
any Ubuntu desktop, they respect whatever the user has selected as their default
devices, and they cost nothing to add as dependencies.

Playback streams: PCM blocks are written to ``paplay`` as they arrive from the
adapter rather than after the whole reply is generated, which is the difference
between a reply that starts speaking in a second and one that starts when the
model has finished thinking.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CAPTURE_RATE_HZ = 16_000
PLAYBACK_RATE_HZ = 24_000


def require_tools() -> None:
    missing = [name for name in ("parecord", "paplay") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "the call harness needs PulseAudio's command line tools: "
            + ", ".join(missing)
        )


@dataclass
class MicrophoneStream:
    """Raw mono frames from the default (or named) capture device."""

    device: str | None = None
    rate_hz: int = CAPTURE_RATE_HZ
    frame_ms: float = 20.0
    channels: int = 1
    channel: int = 0

    def __post_init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def frame_samples(self) -> int:
        return int(self.rate_hz * self.frame_ms / 1000.0)

    def __enter__(self) -> MicrophoneStream:
        command = [
            "parecord",
            "--raw",
            "--format=s16le",
            f"--rate={self.rate_hz}",
            f"--channels={self.channels}",
            "--latency-msec=20",
        ]
        if self.device:
            command.append(f"--device={self.device}")
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    def frames(self):
        """Yield fixed-size mono float32 frames until the stream is closed."""

        if self._process is None or self._process.stdout is None:
            raise RuntimeError("microphone stream is not open")
        frame_bytes = self.frame_samples * self.channels * 2
        stream = self._process.stdout
        while True:
            block = stream.read(frame_bytes)
            if not block or len(block) < frame_bytes:
                return
            interleaved = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768.0
            if self.channels > 1:
                interleaved = interleaved.reshape(-1, self.channels)[:, self.channel]
            yield interleaved


class SpeakerStream:
    """Play PCM as it arrives, and stop the moment it is no longer wanted.

    Stopping matters as much as playing: a reply the speaker has talked over is
    not worth finishing, and holding the device keeps the next turn waiting.
    """

    def __init__(self, rate_hz: int = PLAYBACK_RATE_HZ, device: str | None = None) -> None:
        self.rate_hz = rate_hz
        self.device = device
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._played_bytes = 0

    @property
    def playing(self) -> bool:
        with self._lock:
            process = self._process
        return process is not None and process.poll() is None

    @property
    def played_seconds(self) -> float:
        with self._lock:
            return self._played_bytes / (self.rate_hz * 2)

    def start(self) -> None:
        command = [
            "paplay",
            "--raw",
            "--format=s16le",
            f"--rate={self.rate_hz}",
            "--channels=1",
        ]
        if self.device:
            command.append(f"--device={self.device}")
        with self._lock:
            self._played_bytes = 0
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

    def write(self, pcm: bytes) -> bool:
        """Queue one block. Returns False once playback has been stopped."""

        with self._lock:
            process = self._process
            if process is None or process.stdin is None or process.poll() is not None:
                return False
            try:
                process.stdin.write(pcm)
                process.stdin.flush()
            except (BrokenPipeError, ValueError):
                return False
            self._played_bytes += len(pcm)
        return True

    def finish(self, timeout: float = 60.0) -> None:
        """Close the input and let whatever is queued finish playing."""

        with self._lock:
            process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=timeout)
        except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
            self.stop()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def stop(self) -> None:
        """Cut playback off now -- the speaker has started talking."""

        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def to_wav(samples: np.ndarray, rate_hz: int = CAPTURE_RATE_HZ) -> bytes:
    """Wrap float32 mono samples as a 16-bit WAV, which is what the adapter takes."""

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate_hz)
        target.writeframes(pcm)
    return buffer.getvalue()
