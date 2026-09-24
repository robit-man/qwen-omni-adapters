from __future__ import annotations

import urllib.request
from pathlib import Path

from harness import camera
from harness.camera_view import CameraLiveView


def test_camera_discovery_does_not_open_or_probe_the_device(monkeypatch) -> None:
    probed: list[str] = []
    monkeypatch.setattr(camera, "_can_capture", lambda device: probed.append(device) or True)

    cameras = camera.CameraSet.discover("/dev/video0")

    assert probed == []
    assert cameras.devices == []
    assert cameras.available is True


def test_camera_is_first_opened_only_when_a_structured_capture_runs(monkeypatch) -> None:
    events: list[tuple[str, str]] = []
    cameras = camera.CameraSet.discover("/dev/video0")
    monkeypatch.setattr(camera.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        camera,
        "_can_capture",
        lambda device: events.append(("probe", device)) or True,
    )
    monkeypatch.setattr(
        camera,
        "_grab_frame",
        lambda device: events.append(("frame", device)) or b"jpeg",
    )

    result = cameras.snapshot()

    assert result is not None
    assert events == [("probe", "/dev/video0"), ("frame", "/dev/video0")]


def test_camera_discovery_retries_a_transient_v4l2_miss(monkeypatch) -> None:
    attempts: dict[str, int] = {}
    cameras = camera.CameraSet(_candidates=["/dev/video0", "/dev/video1"])

    def capture(device: str) -> bool:
        attempts[device] = attempts.get(device, 0) + 1
        return device == "/dev/video0" or attempts[device] > 1

    monkeypatch.setattr(camera, "_can_capture", capture)

    assert cameras.ensure_devices() is True
    assert cameras.devices == ["/dev/video0", "/dev/video1"]
    assert attempts == {"/dev/video0": 1, "/dev/video1": 2}


def test_four_camera_still_is_one_left_to_right_row(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"stitched")
        return type("Completed", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(camera.subprocess, "run", run)

    assert camera._stitch([b"a", b"b", b"c", b"d"], 1024) == b"stitched"
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "layout=0_0|0+w0_0|0+w0+w1_0|0+w0+w1+w2_0" in filter_graph
    assert "_0+h" not in filter_graph


def test_loopback_live_view_serves_stitched_frames(monkeypatch) -> None:
    cameras = camera.CameraSet()
    monkeypatch.setattr(
        cameras,
        "snapshot",
        lambda: {
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "data": "anBlZw==",
        },
    )
    live = CameraLiveView(cameras)
    try:
        url = live.start()
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
            page = response.read().decode("utf-8")
        with urllib.request.urlopen(url + "frame.jpg", timeout=2) as response:  # noqa: S310
            frame = response.read()
        assert "Horizontally stitched live camera view" in page
        assert frame == b"jpeg"
    finally:
        live.close()
