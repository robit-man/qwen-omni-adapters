"""Persistent V4L2 capture isolated from the voice process and its consumers."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from harness.camera_activity import SIZE, ActivityDetector, SendLimits, StreamQuality

logger = logging.getLogger(__name__)
CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS = 1920, 1080, 60
ENCODER_WIDTHS = (320, 640, 960, 1280, 1920)


def _gst():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _caps(width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS):
    return f"video/x-raw,format=UYVY,width={width},height={height},framerate={fps}/1"


def _fit(width: int, height: int, maximum: int) -> tuple[int, int]:
    scale = min(1., maximum / width)
    return int(width * scale) // 2 * 2, int(height * scale) // 2 * 2


@dataclass(frozen=True)
class CameraSample:
    buffer: object
    width: int
    height: int
    captured: float

    def jpeg(self, width: int, *, grayscale: bool = False, roll: float = 0) -> bytes:
        raw = self.buffer.extract_dup(0, self.buffer.get_size())
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width // 2, 4)
        out_w, out_h = _fit(self.width, self.height, width)
        y_idx = np.linspace(0, self.height - 1, out_h).round().astype(np.intp)
        x_idx = np.linspace(0, self.width - 1, out_w).round().astype(np.intp)
        pairs = packed[y_idx[:, None], x_idx[None, :] // 2]
        luma = np.take_along_axis(pairs, (1 + 2 * (x_idx % 2))[None, :, None], axis=2)[..., 0]
        frame = Image.fromarray(luma if grayscale else np.stack((luma, pairs[..., 0], pairs[..., 2]), axis=2),
                                mode="L" if grayscale else "YCbCr")
        if roll:
            if not grayscale:
                frame = frame.convert("RGB")
            frame = frame.rotate(roll, Image.Resampling.BICUBIC, expand=True)
        output = io.BytesIO()
        frame.save(output, format="JPEG", quality=88)
        return output.getvalue()


class CameraFeed:
    """A stuck stream ioctl can block only its owned gst-launch child."""

    def __init__(self, device: str, socket_path: Path, *, test_source: bool = False):
        self.device, self.socket_path = device, socket_path
        self.test_source = test_source
        self.detector, self.controller = ActivityDetector(), StreamQuality()
        self._limits = SendLimits(640, 5, 150)
        self.gst = _gst()
        self.pipeline = None
        self.child = None
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.latest_sample = None
        self.image_lock = threading.Lock()
        self.analysed = 0
        self.analysis_started = None
        self.last_analysis = 0.
        self.thread = threading.Thread(target=self._run, name=f"camera-{Path(device).name}", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        path = str(self.socket_path)
        cap = _caps()
        source = ["videotestsrc", "is-live=true", "pattern=ball"] if self.test_source else [
            "v4l2src", f"device={self.device}", "io-mode=mmap"]
        try:
            binary = shutil.which("gst-launch-1.0")
            if not binary:
                return
            self.child = subprocess.Popen([binary, "-q", *source, "!", cap, "!", "shmsink",
                f"socket-path={path}", "shm-size=33177600", "wait-for-connection=true", "sync=false"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            until = time.monotonic() + 5
            while not self.socket_path.exists():
                if self.stop_event.wait(.025) or time.monotonic() > until or self.child.poll() is not None:
                    return
            convert = "nvvidconv" if self.gst.ElementFactory.find("nvvidconv") else "videoconvert ! videoscale"
            self.pipeline = self.gst.parse_launch(
                f'shmsrc socket-path="{path}" is-live=true do-timestamp=true ! {cap} ! tee name=source '
                'source. ! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
                'appsink name=full max-buffers=1 drop=true sync=false '
                'source. ! queue leaky=downstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
                f'videorate drop-only=true ! video/x-raw,framerate=30/1 ! {convert} ! '
                'video/x-raw,format=GRAY8,width=320,height=180 ! appsink name=analysis max-buffers=2 drop=true sync=false'
            )
            self.full = self.pipeline.get_by_name("full")
            sink = self.pipeline.get_by_name("analysis")
            self.pipeline.set_state(self.gst.State.PLAYING)
            while not self.stop_event.is_set() and self.child.poll() is None:
                sample = sink.emit("try-pull-sample", 150_000_000)
                if sample is None:
                    continue
                now = time.monotonic()
                buffer = sample.get_buffer()
                if buffer.get_size() != SIZE[0] * SIZE[1]:
                    continue
                frame = np.frombuffer(buffer.extract_dup(0, buffer.get_size()), dtype=np.uint8).reshape(SIZE)
                observation = self.detector.update(frame, now)
                self._limits = self.controller.update(observation, now)
                if self.analysis_started is None:
                    self.analysis_started = now
                self.analysed += 1
                self.last_analysis = now
                if not self.ready.is_set():
                    logger.info("Persistent capture ready for %s", Path(self.device).name)
                    self.ready.set()
        except Exception as error:  # noqa: BLE001 - isolate camera faults from the call
            logger.warning("Persistent camera %s failed: %s", Path(self.device).name, error)
        finally:
            if self.pipeline is not None:
                self.pipeline.set_state(self.gst.State.NULL)
            self._stop_child()

    def _stop_child(self):
        child = self.child
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=.4)
        except subprocess.TimeoutExpired:
            child.kill()
            # Never wait for a driver already wedged in uninterruptible sleep.

    @property
    def available(self):
        return self.ready.is_set() and time.monotonic() - self.last_analysis < 2

    @property
    def quality(self):
        return self._limits

    def latest(self) -> CameraSample | None:
        if not self.available:
            return None
        with self.image_lock:
            if sample := self.full.emit("try-pull-sample", 0):
                buffer = sample.get_buffer()
                if buffer.get_size() == 2 * CAPTURE_WIDTH * CAPTURE_HEIGHT:
                    self.latest_sample = CameraSample(buffer, CAPTURE_WIDTH, CAPTURE_HEIGHT, time.monotonic())
            current = self.latest_sample
            return current if current and time.monotonic() - current.captured < 1 else None

    def close(self):
        self.stop_event.set()
        self._stop_child()
        self.thread.join(timeout=1.5)


class CameraEncoder:
    """One H.264 branch. The consumer's timebase survives profile switches."""

    def __init__(self, feed: CameraFeed, profile: SendLimits):
        self.gst = feed.gst
        width, height = _fit(CAPTURE_WIDTH, CAPTURE_HEIGHT, profile.width)
        self.hardware = bool(self.gst.ElementFactory.find("nvv4l2h264enc"))
        bitrate = profile.bitrate_kbps
        if self.hardware:
            encode = (f'nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={width},height={height} ! '
                      f'nvv4l2h264enc bitrate={bitrate*1000} insert-sps-pps=true iframeinterval={profile.fps} '
                      f'idrinterval={profile.fps} control-rate=1')
        else:
            encode = (f'videoconvert ! videoscale ! video/x-raw,format=I420,width={width},height={height} ! '
                      f'x264enc bitrate={bitrate} tune=zerolatency speed-preset=ultrafast key-int-max={profile.fps}')
        self.pipeline = self.gst.parse_launch(
            f'appsrc name=input is-live=true format=time caps="{_caps(fps=profile.fps)}" ! {encode} ! '
            'h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au,profile=baseline ! '
            'appsink name=output sync=false max-buffers=3 drop=false'
        )
        self.input = self.pipeline.get_by_name("input")
        self.output = self.pipeline.get_by_name("output")
        self.step = self.gst.SECOND // profile.fps
        self.index = 0
        self.pipeline.set_state(self.gst.State.PLAYING)

    def encode(self, frame: CameraSample) -> bytes | None:
        buffer = frame.buffer.copy()
        buffer.dts = buffer.pts = self.index * self.step
        buffer.duration = self.step
        self.index += 1
        if self.input.emit("push-buffer", buffer) != self.gst.FlowReturn.OK:
            return None
        sample = self.output.emit("try-pull-sample", 220_000_000)
        if sample is None:
            return None
        packet = sample.get_buffer()
        return packet.extract_dup(0, packet.get_size())

    def close(self):
        self.pipeline.set_state(self.gst.State.NULL)
