from __future__ import annotations

import subprocess

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


def test_frame_grab_uses_v4l2_ctl_when_ffmpeg_cannot_read_v4l2(monkeypatch) -> None:
    calls: list[tuple[str, int, list[str]]] = []

    def raw_capture(device, frames, output, timeout):
        calls.append((device, frames, output))
        return subprocess.CompletedProcess([], 0, stdout=b"jpeg", stderr=b"")

    monkeypatch.setattr(camera, "_ffmpeg_reads_v4l2", lambda: False)
    monkeypatch.setattr(camera.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(camera, "_raw_capture", raw_capture)

    assert camera._grab_frame("/dev/video3", 960) == b"jpeg"
    device, frames, output = calls[0]
    assert (device, frames) == ("/dev/video3", 1)
    assert "{crop},scale=960:-2" in output
