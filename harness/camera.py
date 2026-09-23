"""Camera capture for the call harness: every camera at once, as one image.

A machine with several cameras sees several places at once, and a question
like "what am I holding" should not have to say which one to look at. Every
camera is snapped at the same moment, the frames are stitched into one grid and
scaled down, and the model is handed a single image.

Stitching rather than attaching several images is deliberate: one image costs
one pass through the vision encoder instead of one per camera, and the spatial
arrangement survives, so the model can say "on the left" and mean it.

Short clips work the same way, for questions about what just happened rather
than what is there now.
"""

from __future__ import annotations

import base64
import logging
import math
import shutil
import subprocess
import tempfile
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

        if self.devices:
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
        self.devices = [device for device in candidates if _can_capture(device)]
        return bool(self.devices)

    def snapshot(self) -> dict[str, Any] | None:
        """One image of everything the machine can see, right now."""

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
        stitched = frames[0] if len(frames) == 1 else _stitch(frames, self.stitch_width)
        if stitched is None:
            return None
        return {
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "data": base64.b64encode(stitched).decode("ascii"),
        }

    def clip(self, seconds: float = CLIP_SECONDS) -> dict[str, Any] | None:
        """A few seconds from every camera, stitched into one clip."""

        if shutil.which("ffmpeg") is None or not self.ensure_devices():
            return None
        with tempfile.TemporaryDirectory(prefix="omni-clip-") as workspace:
            root = Path(workspace)
            with ThreadPoolExecutor(max_workers=max(1, len(self.devices))) as pool:
                paths = [
                    path
                    for path in pool.map(
                        lambda item: _grab_clip(item[1], root / f"{item[0]}.mp4", seconds),
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


def _grab_frame(device: str, width: int = 640) -> bytes | None:
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


def _grid(count: int) -> tuple[int, int]:
    """Columns and rows for ``count`` tiles, kept as square as possible."""

    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return columns, rows


def _stitch(frames: list[bytes], width: int) -> bytes | None:
    """Lay frames out in a grid with ffmpeg's xstack."""

    columns, rows = _grid(len(frames))
    tile_width = max(160, width // columns)
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
        # xstack places each tile by naming the widths and heights that come
        # before it, so a grid is expressed as sums of earlier tiles.
        layout = "|".join(
            "+".join(["0"] + [f"w{i}" for i in range(index % columns)]) + "_"
            + "+".join(["0"] + [f"h{i}" for i in range(index // columns)])
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
    columns, rows = _grid(len(paths))
    inputs: list[str] = []
    for path in paths:
        inputs += ["-i", str(path)]
    layout = "|".join(
        "+".join(["0"] + [f"w{i}" for i in range(index % columns)]) + "_"
        + "+".join(["0"] + [f"h{i}" for i in range(index // columns)])
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
