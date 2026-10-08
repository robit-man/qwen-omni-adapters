"""Camera capture for the call harness: every camera at once, as one image.

A machine with several cameras sees several places at once, and a question
like "what am I holding" should not have to say which one to look at. Every
camera is snapped at the same moment, the frames are stitched left-to-right and
scaled down, and the model is handed a single image.

Stitching rather than attaching several images is deliberate: one image costs
one pass through the vision encoder instead of one per camera, and the spatial
arrangement survives, so the model can say "on the left" and mean it.

Short clips work the same way, for questions about what just happened rather
than what is there now.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Wide enough for detail, small enough that the encoder is not the turn's cost.
STITCH_WIDTH = 1024
CLIP_SECONDS = 3.0
CLIP_WIDTH = 512


@dataclass
class CameraSet:
    """The cameras this machine has, captured together."""

    devices: list[str] = field(default_factory=list)
    stitch_width: int = STITCH_WIDTH
    # Remembered so a later look can repeat the same search.
    _explicit: str | None = None
    _candidates: list[str] = field(default_factory=list)
    _capture_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_discovery_at: float = 0.0

    @classmethod
    def discover(cls, explicit: str | None = None) -> CameraSet:
        """List candidate nodes without opening a camera."""

        candidates = (
            [explicit]
            if explicit
            else [str(path) for path in sorted(Path("/dev").glob("video*"))]
        )
        # Listing character devices is intentionally not a capture. Probing a
        # V4L2 node runs ffmpeg and activates the camera privacy indicator, so
        # it belongs only to a structured request_camera_view action.
        logger.info(
            "camera candidates: %s",
            ", ".join(candidates) if candidates else "none",
        )
        return cls(devices=[], _explicit=explicit, _candidates=candidates)

    @property
    def available(self) -> bool:
        return bool(self.devices or self._candidates or self._explicit)

    def ensure_devices(self) -> bool:
        """Find cameras now if the last look found none.

        Discovery probes each device by taking a frame, so a camera held by
        something else -- the previous instance of this service during a
        restart, say -- looks like no camera at all. Doing it once at startup
        meant one unlucky moment disabled vision until the next restart.
        Vision is asked for rarely, so it costs nothing to look again.
        """

        now = time.monotonic()
        if self.devices and now - self._last_discovery_at < 10.0:
            return True
        candidates = self._candidates
        if not candidates:
            candidates = (
                [self._explicit]
                if self._explicit
                else [str(path) for path in sorted(Path("/dev").glob("video*"))]
            )
        # This is the first operation that opens a camera. It is reached only
        # after the model emitted a structured camera tool request.
        missing = [device for device in candidates if device not in self.devices]
        with ThreadPoolExecutor(max_workers=max(1, len(missing))) as pool:
            recovered = [
                device
                for device, available in zip(
                    missing, pool.map(_can_capture_with_retry, missing), strict=True
                )
                if available
            ]
        known = set(self.devices) | set(recovered)
        self.devices = [device for device in candidates if device in known]
        self._last_discovery_at = now
        return bool(self.devices)

    def describe(self, *, probe: bool = True) -> list[dict[str, str]]:
        """Return stable public descriptors without exposing host device paths.

        ``probe`` is deliberately explicit.  Listing candidates alone does not
        activate a camera, while the authenticated portal calls this with
        ``probe=True`` only after a person chooses the remote-camera source.
        """

        if probe and not self.ensure_devices():
            return []
        devices = self.devices if probe else (self.devices or self._candidates)
        return [
            {
                "id": _camera_identifier(device),
                "label": _camera_label(device, index),
            }
            for index, device in enumerate(devices)
        ]

    def snapshot_devices(
        self,
        camera_ids: list[str] | None = None,
        *,
        width: int = 960,
    ) -> list[dict[str, str]]:
        """Capture selected cameras concurrently as independent JPEG frames."""

        with self._capture_lock:
            if shutil.which("ffmpeg") is None or not self.ensure_devices():
                return []
            selected = [
                device
                for device in self.devices
                if camera_ids is None or _camera_identifier(device) in camera_ids
            ]
            if not selected:
                return []
            with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
                frames = list(pool.map(lambda device: _grab_frame(device, width), selected))
            return [
                {
                    "id": _camera_identifier(device),
                    "mime_type": "image/jpeg",
                    "encoding": "base64",
                    "data": base64.b64encode(frame).decode("ascii"),
                }
                for device, frame in zip(selected, frames, strict=True)
                if frame is not None
            ]

    def snapshot(self) -> dict[str, Any] | None:
        """One image of everything the machine can see, right now."""

        with self._capture_lock:
            if shutil.which("ffmpeg") is None or not self.ensure_devices():
                return None
            with ThreadPoolExecutor(max_workers=max(1, len(self.devices))) as pool:
                frames = [
                    frame
                    for frame in pool.map(_grab_frame, self.devices)
                    if frame is not None
                ]
            if not frames:
                return None
            stitched = (
                frames[0]
                if len(frames) == 1
                else _stitch(frames, self.stitch_width)
            )
            if stitched is None:
                return None
            return {
                "mime_type": "image/jpeg",
                "encoding": "base64",
                "data": base64.b64encode(stitched).decode("ascii"),
            }

    def clip(self, seconds: float = CLIP_SECONDS) -> dict[str, Any] | None:
        """A few seconds from every camera, stitched into one clip."""

        with self._capture_lock:
            if shutil.which("ffmpeg") is None or not self.ensure_devices():
                return None
            with tempfile.TemporaryDirectory(prefix="omni-clip-") as workspace:
                root = Path(workspace)
                with ThreadPoolExecutor(max_workers=max(1, len(self.devices))) as pool:
                    paths = [
                        path
                        for path in pool.map(
                            lambda item: _grab_clip(
                                item[1], root / f"{item[0]}.mp4", seconds
                            ),
                            enumerate(self.devices),
                        )
                        if path is not None
                    ]
                if not paths:
                    return None
                merged = root / "stitched.mp4"
                if not _stitch_clips(paths, merged):
                    return None
                try:
                    data = merged.read_bytes()
                except OSError:
                    return None
        return {
            "mime_type": "video/mp4",
            "encoding": "base64",
            "data": base64.b64encode(data).decode("ascii"),
        }


def _can_capture(device: str) -> bool:
    return _grab_frame(device, width=64) is not None


def _camera_identifier(device: str) -> str:
    """Stable opaque identifier for a V4L2 node."""

    return "camera-" + hashlib.sha256(device.encode("utf-8")).hexdigest()[:16]


def _camera_label(device: str, index: int) -> str:
    """Use the kernel's human label when available, with a neutral fallback."""

    name_path = Path("/sys/class/video4linux") / Path(device).name / "name"
    try:
        label = " ".join(name_path.read_text(encoding="utf-8").split())
    except OSError:
        label = ""
    return label[:80] or f"Camera {index + 1}"


def _can_capture_with_retry(device: str) -> bool:
    """Tolerate one transient V4L2 open/startup miss during discovery."""

    if _can_capture(device):
        return True
    time.sleep(0.15)
    return _can_capture(device)


# V4L2 fourcc -> ffmpeg rawvideo pixel format and bytes per pixel, for the
# v4l2-ctl capture path. YUV GMSL cameras deliver packed 4:2:2.
_RAW_PIXEL_FORMATS = {
    "UYVY": ("uyvy422", 2),
    "YUYV": ("yuyv422", 2),
    "YVYU": ("yvyu422", 2),
    "GREY": ("gray", 1),
}


@functools.lru_cache(maxsize=1)
def _ffmpeg_reads_v4l2() -> bool:
    """Whether this ffmpeg build includes the V4L2 capture input.

    NVIDIA's JetPack 7 ffmpeg package is built without libavdevice's V4L2
    input ("Unknown input format: 'v4l2'"), so capture goes through
    v4l2-ctl there instead.
    """

    try:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-devices"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return False
    return any(
        re.match(r"^\s*D\S*\s+(?:\S+,)?v4l2(?:,|\s)", line) for line in listing.splitlines()
    )


@dataclass(frozen=True)
class _RawFormat:
    pixel_format: str
    width: int
    height: int
    stride_pixels: int
    fps: float


def _raw_format(device: str) -> _RawFormat | None:
    """Read the node's active format so raw frames can be decoded exactly."""

    try:
        completed = subprocess.run(
            ["v4l2-ctl", "-d", device, "--get-fmt-video", "--get-parm"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = completed.stdout
    size = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", text)
    fourcc = re.search(r"Pixel Format\s*:\s*'(\w{4})'", text)
    stride = re.search(r"Bytes per Line\s*:\s*(\d+)", text)
    fps = re.search(r"Frames per second\s*:\s*([\d.]+)", text)
    if completed.returncode != 0 or not size or not fourcc:
        return None
    mapped = _RAW_PIXEL_FORMATS.get(fourcc.group(1))
    if mapped is None:
        logger.warning("camera %s uses unsupported pixel format %s", device, fourcc.group(1))
        return None
    pixel_format, bytes_per_pixel = mapped
    width, height = int(size.group(1)), int(size.group(2))
    # The VI pads rows to its stride alignment; decode the padded row, then crop.
    stride_pixels = max(width, int(stride.group(1)) // bytes_per_pixel if stride else width)
    rate = float(fps.group(1)) if fps else 30.0
    return _RawFormat(pixel_format, width, height, stride_pixels, rate if rate > 0 else 30.0)


def _raw_capture(
    device: str, frames: int, ffmpeg_output: list[str], timeout: float
) -> subprocess.CompletedProcess[bytes] | None:
    """Stream raw frames from v4l2-ctl into ffmpeg's built-in rawvideo input."""

    fmt = _raw_format(device)
    if fmt is None:
        return None
    capture = [
        "v4l2-ctl", "-d", device, "--stream-mmap",
        # The first frames after stream-on can be partial while GMSL locks.
        "--stream-skip=2", f"--stream-count={frames}", "--stream-to=-",
    ]
    decode = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", fmt.pixel_format,
        "-s", f"{fmt.stride_pixels}x{fmt.height}", "-framerate", f"{fmt.fps:g}",
        "-i", "-",
        *[
            argument.replace("{crop}", f"crop={fmt.width}:{fmt.height}:0:0")
            for argument in ffmpeg_output
        ],
    ]
    try:
        producer = subprocess.Popen(
            capture, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        return None
    try:
        completed = subprocess.run(
            decode, stdin=producer.stdout, capture_output=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        producer.kill()
        producer.wait()
        return None
    finally:
        if producer.stdout is not None:
            producer.stdout.close()
    try:
        producer.wait(timeout=5)
    except subprocess.TimeoutExpired:
        producer.kill()
        producer.wait()
    return completed


def _grab_frame(device: str, width: int = 640) -> bytes | None:
    if not _ffmpeg_reads_v4l2():
        if shutil.which("v4l2-ctl") is None:
            logger.warning("ffmpeg lacks V4L2 input and v4l2-ctl is missing; cannot capture %s", device)
            return None
        completed = _raw_capture(
            device, 1,
            ["-frames:v", "1", "-vf", f"{{crop}},scale={width}:-2",
             "-f", "image2", "-c:v", "mjpeg", "-"],
            timeout=8,
        )
        if completed is None or completed.returncode != 0 or not completed.stdout:
            return None
        return completed.stdout
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "v4l2", "-i", device,
                "-frames:v", "1", "-vf", f"scale={width}:-2",
                "-f", "image2", "-c:v", "mjpeg", "-",
            ],
            capture_output=True,
            timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


def _grab_clip(device: str, target: Path, seconds: float) -> Path | None:
    duration = max(0.5, seconds)
    encode = [
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        # The model reads this through a pipe, which cannot seek back
        # to a trailing index.
        "-movflags", "+faststart",
        str(target),
    ]
    if not _ffmpeg_reads_v4l2():
        fmt = _raw_format(device) if shutil.which("v4l2-ctl") else None
        if fmt is None:
            return None
        completed = _raw_capture(
            device, max(1, round(fmt.fps * duration)),
            ["-vf", f"{{crop}},scale={CLIP_WIDTH}:-2", *encode],
            timeout=duration + 25,
        )
        if completed is None:
            return None
        return target if completed.returncode == 0 and target.exists() else None
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "v4l2", "-i", device,
                "-t", f"{max(0.5, seconds):.1f}",
                "-vf", f"scale={CLIP_WIDTH}:-2",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                # The model reads this through a pipe, which cannot seek back
                # to a trailing index.
                "-movflags", "+faststart",
                str(target),
            ],
            capture_output=True,
            timeout=seconds + 25,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return target if completed.returncode == 0 and target.exists() else None


def _stitch(frames: list[bytes], width: int) -> bytes | None:
    """Lay every frame out in one stable left-to-right row."""

    tile_width = max(160, width // len(frames))
    with tempfile.TemporaryDirectory(prefix="omni-stitch-") as workspace:
        root = Path(workspace)
        inputs: list[str] = []
        for index, frame in enumerate(frames):
            path = root / f"{index}.jpg"
            path.write_bytes(frame)
            inputs += ["-i", str(path)]

        scale = "".join(
            f"[{index}:v]scale={tile_width}:-2,pad={tile_width}:ceil(ih/2)*2[v{index}];"
            for index in range(len(frames))
        )
        layout = "|".join(
            "+".join(["0"] + [f"w{i}" for i in range(index)]) + "_0"
            for index in range(len(frames))
        )
        chain = (
            scale
            + "".join(f"[v{index}]" for index in range(len(frames)))
            + f"xstack=inputs={len(frames)}:layout={layout}[out]"
        )
        output = root / "stitched.jpg"
        try:
            completed = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                 "-filter_complex", chain, "-map", "[out]",
                 "-frames:v", "1", "-q:v", "4", str(output)],
                capture_output=True,
                timeout=25,
            )
        except (subprocess.TimeoutExpired, OSError):
            return frames[0]
        if completed.returncode != 0 or not output.exists():
            logger.debug("stitch failed: %s", completed.stderr[:160])
            # One camera's view is better than none.
            return frames[0]
        try:
            return output.read_bytes()
        except OSError:
            return frames[0]


def _stitch_clips(paths: list[Path], target: Path) -> bool:
    if len(paths) == 1:
        try:
            target.write_bytes(paths[0].read_bytes())
            return True
        except OSError:
            return False
    inputs: list[str] = []
    for path in paths:
        inputs += ["-i", str(path)]
    layout = "|".join(
        "+".join(["0"] + [f"w{i}" for i in range(index)]) + "_0"
        for index in range(len(paths))
    )
    chain = (
        "".join(f"[{index}:v]scale={CLIP_WIDTH}:-2[v{index}];" for index in range(len(paths)))
        + "".join(f"[v{index}]" for index in range(len(paths)))
        + f"xstack=inputs={len(paths)}:layout={layout}:shortest=1[out]"
    )
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
             "-filter_complex", chain, "-map", "[out]",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(target)],
            capture_output=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0 and target.exists()
