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
import json
import logging
import shutil
import signal
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CAPTURE_RATE_HZ = 16_000
PLAYBACK_RATE_HZ = 24_000


def require_tools() -> None:
    missing = [
        name
        for name in ("parecord", "paplay", "pactl")
        if shutil.which(name) is None
    ]
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
    """Play PCM as it arrives, with a conversational interruption envelope.

    A possible interruption ducks first. Sustained speech fades to silence and
    pauses without throwing away the rest of the generated sentence; rejected
    noise resumes and fades back up. A confirmed interruption fades out before
    playback is discarded, avoiding the click and chopped phoneme of SIGTERM.
    """

    def __init__(self, rate_hz: int = PLAYBACK_RATE_HZ, device: str | None = None) -> None:
        self.rate_hz = rate_hz
        self.device = device
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._played_bytes = 0
        self._gain = 1.0
        self._sink_input: str | None = None
        self._paused = False
        self._paused_pcm = bytearray()
        self._fade_serial = 0

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
            "--latency-msec=40",
            "--process-time-msec=20",
            "--stream-name=Omni conversational voice",
        ]
        if self.device:
            command.append(f"--device={self.device}")
        with self._lock:
            self._played_bytes = 0
            self._gain = 1.0
            self._sink_input = None
            self._paused = False
            self._paused_pcm.clear()
            self._fade_serial += 1
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

    @staticmethod
    def _find_sink_input(process: subprocess.Popen[bytes]) -> str | None:
        """Find this paplay stream so an in-flight buffer can be faded too."""

        try:
            completed = subprocess.run(
                ["pactl", "--format=json", "list", "sink-inputs"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if completed.returncode:
                return None
            items = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, dict):
                continue
            properties = item.get("properties")
            if not isinstance(properties, dict):
                continue
            if str(properties.get("application.process.id") or "") == str(process.pid):
                return str(item.get("index"))
        return None

    def _set_gain(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        with self._lock:
            process = self._process
            self._gain = value
            sink_input = self._sink_input
        if process is None or process.poll() is not None:
            return
        if sink_input is None:
            sink_input = self._find_sink_input(process)
            if sink_input is not None:
                with self._lock:
                    if self._process is process:
                        self._sink_input = sink_input
        if sink_input is None:
            return
        try:
            subprocess.run(
                [
                    "pactl",
                    "set-sink-input-volume",
                    sink_input,
                    str(round(value * 65536)),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.debug("could not change the live playback volume", exc_info=True)

    def _fade_to(
        self,
        target: float,
        duration: float,
        *,
        then: Callable[[], None] | None = None,
    ) -> None:
        """Fade asynchronously so microphone capture never misses a frame."""

        with self._lock:
            self._fade_serial += 1
            serial = self._fade_serial
            start = self._gain

        def fade() -> None:
            steps = max(1, round(max(0.0, duration) / 0.03))
            for step in range(1, steps + 1):
                with self._lock:
                    if serial != self._fade_serial:
                        return
                self._set_gain(start + (target - start) * step / steps)
                if duration > 0:
                    time.sleep(duration / steps)
            if then is not None:
                with self._lock:
                    if serial != self._fade_serial:
                        return
                then()

        threading.Thread(target=fade, name="omni-playback-fade", daemon=True).start()

    def duck(self, level: float = 0.28, duration: float = 0.18) -> None:
        """Lower a possible interruption without abandoning the sentence."""

        self._fade_to(level, duration)

    def pause(self, duration: float = 0.18) -> None:
        """Fade fully down and preserve later PCM while the person is speaking."""

        def hold() -> None:
            with self._lock:
                process = self._process
                if process is None or process.poll() is not None:
                    return
                self._paused = True
            try:
                process.send_signal(signal.SIGSTOP)
            except OSError:
                pass

        self._fade_to(0.0, duration, then=hold)

    def resume(self, duration: float = 0.24) -> None:
        """Continue a sentence after a false interruption, easing back up."""

        with self._lock:
            self._fade_serial += 1
            process = self._process
            paused = self._paused
        if process is None or process.poll() is not None:
            return
        if paused:
            try:
                process.send_signal(signal.SIGCONT)
            except OSError:
                return
            self._fade_to(1.0, duration)

            def flush() -> None:
                while True:
                    with self._lock:
                        if self._process is not process:
                            return
                        if not self._paused_pcm:
                            self._paused = False
                            break
                        block = bytes(self._paused_pcm)
                        self._paused_pcm.clear()
                    if not self.write(block, force=True):
                        return

            threading.Thread(
                target=flush, name="omni-playback-resume", daemon=True
            ).start()
            return
        self._fade_to(1.0, duration)

    def write(self, pcm: bytes, *, force: bool = False) -> bool:
        """Queue one block. Returns False once playback has been stopped."""

        with self._lock:
            process = self._process
            if process is None or process.stdin is None or process.poll() is not None:
                return False
            if self._paused and not force:
                self._paused_pcm.extend(pcm)
                limit = self.rate_hz * 2 * 60
                if len(self._paused_pcm) > limit:
                    del self._paused_pcm[: len(self._paused_pcm) - limit]
                return True
            sink_input = self._sink_input
            gain = self._gain
            if sink_input is None and gain < 0.999:
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                pcm = np.clip(samples * gain, -32768, 32767).astype("<i2").tobytes()
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
            # A late duck/pause callback must not SIGSTOP a process we are
            # already waiting on at the natural end of the sentence.
            self._fade_serial += 1
            process = self._process
            paused = self._paused
            pending = bytes(self._paused_pcm)
            self._paused_pcm.clear()
            self._paused = False
        if process is None:
            return
        self._set_gain(1.0)
        if paused:
            try:
                process.send_signal(signal.SIGCONT)
            except OSError:
                pass
            if pending:
                self.write(pending, force=True)
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

    def stop(self, fade_s: float = 0.0) -> None:
        """Release playback, optionally fading instead of chopping a phoneme."""

        if fade_s > 0:
            self._fade_to(0.0, fade_s, then=self.stop)
            return

        with self._lock:
            process, self._process = self._process, None
            self._fade_serial += 1
            was_paused = self._paused
            self._paused = False
            self._paused_pcm.clear()
        if process is None or process.poll() is not None:
            return
        if was_paused:
            try:
                process.send_signal(signal.SIGCONT)
            except OSError:
                pass
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
