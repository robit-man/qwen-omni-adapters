from __future__ import annotations

from harness import camera


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
