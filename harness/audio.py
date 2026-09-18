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
import queue
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
                return_code = self._process.poll()
                detail = (
                    f"exit code {return_code}"
                    if return_code is not None
                    else "capture pipe closed"
                )
                raise RuntimeError(
                    f"microphone capture ended unexpectedly ({detail})"
                )
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
        self._fade_serial = 0
        self._pcm_queue: queue.Queue[tuple[bytes, float] | None] | None = None
        self._writer: threading.Thread | None = None
        self._accepting = False
        self._stopped = False
        self._audible_started_at: float | None = None
        self._first_arrival_at: float | None = None
        self._last_arrival_at: float | None = None
        self._chunk_count = 0
        self._pending_bytes = 0
        self._peak_pending_bytes = 0
        self._startup_wait_s = 0.0
        self._startup_buffer_bytes = 0
        self._max_arrival_gap_s = 0.0
        self._max_source_wait_s = 0.0
        self._max_write_block_s = 0.0
        self._producer_buffer_s = 0.0
        self._producer_clock_at: float | None = None
        self._max_predicted_starvation_s = 0.0

    @property
    def playing(self) -> bool:
        with self._lock:
            process = self._process
            accepting = self._accepting
        return accepting or (process is not None and process.poll() is None)

    @property
    def played_seconds(self) -> float:
        with self._lock:
            return self._played_bytes / (self.rate_hz * 2)

    @property
    def audible_started_at(self) -> float | None:
        with self._lock:
            return self._audible_started_at

    def timing(self) -> dict[str, float | int]:
        """Playback diagnostics in transport-independent units."""

        bytes_per_second = self.rate_hz * 2
        with self._lock:
            return {
                "chunks": self._chunk_count,
                "pcm_seconds": self._played_bytes / bytes_per_second,
                "startup_wait_ms": self._startup_wait_s * 1000.0,
                "startup_buffer_ms": self._startup_buffer_bytes
                / bytes_per_second
                * 1000.0,
                "max_arrival_gap_ms": self._max_arrival_gap_s * 1000.0,
                "max_source_wait_ms": self._max_source_wait_s * 1000.0,
                "max_write_block_ms": self._max_write_block_s * 1000.0,
                "max_predicted_starvation_ms": self._max_predicted_starvation_s
                * 1000.0,
                "peak_buffer_ms": self._peak_pending_bytes
                / bytes_per_second
                * 1000.0,
            }

    def _playback_command(self) -> list[str]:
        command = [
            "paplay",
            "--raw",
            "--format=s16le",
            f"--rate={self.rate_hz}",
            "--channels=1",
            "--stream-name=Omni conversational voice",
        ]
        if self.device:
            command.append(f"--device={self.device}")
        return command

    def start(self) -> None:
        with self._lock:
            self._played_bytes = 0
            self._gain = 1.0
            self._sink_input = None
            self._paused = False
            self._fade_serial += 1
            self._process = None
            self._pcm_queue = queue.Queue()
            self._accepting = True
            self._stopped = False
            self._audible_started_at = None
            self._first_arrival_at = None
            self._last_arrival_at = None
            self._chunk_count = 0
            self._pending_bytes = 0
            self._peak_pending_bytes = 0
            self._startup_wait_s = 0.0
            self._startup_buffer_bytes = 0
            self._max_arrival_gap_s = 0.0
            self._max_source_wait_s = 0.0
            self._max_write_block_s = 0.0
            self._producer_buffer_s = 0.0
            self._producer_clock_at = None
            self._max_predicted_starvation_s = 0.0
            self._writer = threading.Thread(
                target=self._playback_loop,
                name="omni-playback-writer",
                daemon=True,
            )
            writer = self._writer
        writer.start()

    def _write_process(self, process: subprocess.Popen[bytes], pcm: bytes) -> bool:
        if process.stdin is None:
            return False
        with self._lock:
            if self._stopped or self._process is not process:
                return False
            sink_input = self._sink_input
            gain = self._gain
        if sink_input is None and gain < 0.999:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            pcm = np.clip(samples * gain, -32768, 32767).astype("<i2").tobytes()
        started = time.monotonic()
        try:
            process.stdin.write(pcm)
            process.stdin.flush()
        except (BrokenPipeError, ValueError):
            return False
        blocked = time.monotonic() - started
        with self._lock:
            self._pending_bytes = max(0, self._pending_bytes - len(pcm))
            self._max_write_block_s = max(self._max_write_block_s, blocked)
        return True

    def _playback_loop(self) -> None:
        """Feed one Pulse stream while decoder intake runs independently."""

        with self._lock:
            pcm_queue = self._pcm_queue
        if pcm_queue is None:
            return
        process: subprocess.Popen[bytes] | None = None
        try:
            first = pcm_queue.get()
            if first is None:
                return
            startup = [first]
            # One packet cannot reveal producer cadence. Waiting for the next
            # packet (or natural EOF) creates a pre-roll whose size and delay
            # come from this decoder run rather than a fixed millisecond guess.
            second = pcm_queue.get()
            if second is not None:
                startup.append(second)
            with self._lock:
                if self._stopped:
                    return
                self._startup_wait_s = max(0.0, startup[-1][1] - startup[0][1])
                self._startup_buffer_bytes = sum(len(item[0]) for item in startup)
            process = subprocess.Popen(
                self._playback_command(),
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                if self._stopped:
                    process.terminate()
                    return
                self._process = process
                self._audible_started_at = time.monotonic()
                paused = self._paused
            if paused:
                process.send_signal(signal.SIGSTOP)
            if not self._write_process(
                process, b"".join(item[0] for item in startup)
            ):
                return
            finished = second is None
            while not finished:
                waited_at = time.monotonic()
                item = pcm_queue.get()
                waited = time.monotonic() - waited_at
                with self._lock:
                    self._max_source_wait_s = max(self._max_source_wait_s, waited)
                    stopped = self._stopped
                if stopped or item is None:
                    break
                if not self._write_process(process, item[0]):
                    break
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            process.wait()
        except OSError:
            logger.warning("speaker playback process failed", exc_info=True)
        finally:
            with self._lock:
                if process is not None and self._process is process:
                    self._process = None

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
            with self._lock:
                self._paused = False
            self._fade_to(1.0, duration)
            return
        self._fade_to(1.0, duration)

    def write(self, pcm: bytes, *, force: bool = False) -> bool:
        """Queue PCM without throttling decoder intake on speaker playback."""

        with self._lock:
            pcm_queue = self._pcm_queue
            if not self._accepting or self._stopped or pcm_queue is None:
                return False
            if not pcm:
                return True
            if len(pcm) % 2:
                raise ValueError("PCM block ended on a partial 16-bit sample")
            arrived = time.monotonic()
            if self._first_arrival_at is None:
                self._first_arrival_at = arrived
            if self._last_arrival_at is not None:
                self._max_arrival_gap_s = max(
                    self._max_arrival_gap_s, arrived - self._last_arrival_at
                )
            duration = len(pcm) / (self.rate_hz * 2)
            if self._chunk_count == 0:
                self._producer_buffer_s = duration
            elif self._chunk_count == 1:
                self._producer_buffer_s += duration
                self._producer_clock_at = arrived
            elif self._producer_clock_at is not None:
                elapsed = arrived - self._producer_clock_at
                starvation = max(0.0, elapsed - self._producer_buffer_s)
                self._max_predicted_starvation_s = max(
                    self._max_predicted_starvation_s, starvation
                )
                self._producer_buffer_s = max(
                    0.0, self._producer_buffer_s - elapsed
                ) + duration
                self._producer_clock_at = arrived
            self._last_arrival_at = arrived
            self._chunk_count += 1
            self._played_bytes += len(pcm)
            self._pending_bytes += len(pcm)
            self._peak_pending_bytes = max(
                self._peak_pending_bytes, self._pending_bytes
            )
        pcm_queue.put((pcm, arrived))
        return True

    def finish(self, timeout: float = 60.0) -> None:
        """Close the input and let whatever is queued finish playing."""

        with self._lock:
            # A late duck/pause callback must not SIGSTOP a process we are
            # already waiting on at the natural end of the sentence.
            self._fade_serial += 1
            process = self._process
            paused = self._paused
            self._paused = False
            pcm_queue = self._pcm_queue
            writer = self._writer
            self._accepting = False
        self._set_gain(1.0)
        if paused and process is not None:
            try:
                process.send_signal(signal.SIGCONT)
            except OSError:
                pass
        if pcm_queue is not None:
            pcm_queue.put(None)
        if writer is not None:
            writer.join(timeout=timeout + self.played_seconds)
        if writer is not None and writer.is_alive():
            self.stop()

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
            self._accepting = False
            self._stopped = True
            pcm_queue = self._pcm_queue
            writer = self._writer
        if pcm_queue is not None:
            pcm_queue.put(None)
        if process is None or process.poll() is not None:
            if writer is not None and writer is not threading.current_thread():
                writer.join(timeout=2)
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
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=2)


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
