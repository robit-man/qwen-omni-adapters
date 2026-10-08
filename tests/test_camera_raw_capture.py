from __future__ import annotations

import subprocess
from pathlib import Path

from harness import camera

FORMAT_OUTPUT = """Format Video Capture:
\tWidth/Height      : 1920/1080
\tPixel Format      : 'UYVY' (UYVY 4:2:2)
\tBytes per Line    : 3968
Streaming Parameters Video Capture:
\tFrames per second: 30.000 (30/1)
"""


def _fake_run(stdout: str, returncode: int = 0):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    return run


def test_ffmpeg_v4l2_input_detection_requires_the_demuxer_flag(monkeypatch) -> None:
    camera._ffmpeg_reads_v4l2.cache_clear()
    monkeypatch.setattr(
        camera.subprocess, "run",
        _fake_run(" DE video4linux2,v4l2 Video4Linux2 output device\n"),
    )
    assert camera._ffmpeg_reads_v4l2() is True

    camera._ffmpeg_reads_v4l2.cache_clear()
    # NVIDIA's JetPack 7 build lists no V4L2 input at all.
    monkeypatch.setattr(camera.subprocess, "run", _fake_run("  E alsa ALSA audio output\n"))
    assert camera._ffmpeg_reads_v4l2() is False
    camera._ffmpeg_reads_v4l2.cache_clear()


def test_raw_format_decodes_padded_vi_rows(monkeypatch) -> None:
    monkeypatch.setattr(camera.subprocess, "run", _fake_run(FORMAT_OUTPUT))

    fmt = camera._raw_format("/dev/video0")

    assert fmt == camera._RawFormat("uyvy422", 1920, 1080, 1984, 30.0)


def test_raw_format_rejects_unknown_pixel_formats(monkeypatch) -> None:
    monkeypatch.setattr(
        camera.subprocess, "run",
        _fake_run(FORMAT_OUTPUT.replace("'UYVY'", "'RG10'")),
    )
    assert camera._raw_format("/dev/video0") is None


def test_frame_grab_uses_v4l2_ctl_and_files_when_ffmpeg_cannot_read_v4l2(monkeypatch) -> None:
    calls: list[tuple[str, int, list[str]]] = []

    def raw_capture(device, frames, output_args, output, timeout):
        calls.append((device, frames, output_args))
        output.write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(camera, "_ffmpeg_reads_v4l2", lambda: False)
    monkeypatch.setattr(camera.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(camera, "_raw_capture", raw_capture)

    assert camera._grab_frame("/dev/video3", 960) == b"jpeg"
    device, frames, output_args = calls[0]
    assert (device, frames) == ("/dev/video3", 1)
    assert "{crop},scale=960:-2" in output_args
    # NVIDIA's JetPack 7 ffmpeg has no pipe: protocol, so nothing may use "-".
    assert "-" not in output_args


def test_raw_capture_feeds_ffmpeg_through_a_fifo_not_a_pipe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        camera, "_raw_format", lambda device: camera._RawFormat("uyvy422", 1920, 1080, 1984, 30.0)
    )
    seen: dict[str, list[str]] = {}

    class FakeProducer:
        returncode = 0

        def __init__(self, command, **kwargs):
            seen["capture"] = command

        def communicate(self, timeout=None):
            return b"", b""

        def kill(self):
            pass

    def fake_run(command, **kwargs):
        seen["decode"] = command
        Path(command[-1]).write_bytes(b"out")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(camera.subprocess, "Popen", FakeProducer)
    monkeypatch.setattr(camera.subprocess, "run", fake_run)
    output = tmp_path / "frame.jpg"

    assert camera._raw_capture("/dev/video0", 1, ["-f", "image2"], output, timeout=5)
    stream_to = next(arg for arg in seen["capture"] if arg.startswith("--stream-to="))
    fifo = stream_to.split("=", 1)[1]
    assert seen["decode"][seen["decode"].index("-i") + 1] == fifo
    assert "-" not in seen["decode"]
    assert seen["decode"][-1] == str(output)
