from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

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


def test_camera_descriptors_are_stable_and_do_not_expose_device_paths(monkeypatch) -> None:
    cameras = camera.CameraSet(devices=["/dev/video2", "/dev/video7"])
    cameras._last_discovery_at = camera.time.monotonic()
    monkeypatch.setattr(camera, "_camera_label", lambda _device, index: f"Lens {index + 1}")

    first = cameras.describe()
    second = cameras.describe()

    assert first == second
    assert [item["label"] for item in first] == ["Lens 1", "Lens 2"]
    assert all(item["id"].startswith("camera-") for item in first)
    assert "/dev/" not in repr(first)


def test_selected_camera_frames_remain_independent(monkeypatch) -> None:
    cameras = camera.CameraSet(devices=["/dev/video0", "/dev/video1"])
    cameras._last_discovery_at = camera.time.monotonic()
    monkeypatch.setattr(camera.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(camera, "_grab_frame", lambda device, _width=960: device.encode())
    second_id = cameras.describe()[1]["id"]

    frames = cameras.snapshot_devices([second_id])

    assert len(frames) == 1
    assert frames[0]["id"] == second_id
    assert frames[0]["mime_type"] == "image/jpeg"
    assert frames[0]["data"] == "L2Rldi92aWRlbzE="


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


@pytest.mark.parametrize("motion", [False, True], ids=["still", "clip"])
def test_native_camera_delivers_corrected_pixels_and_confidence(monkeypatch, tmp_path, motion) -> None:
    import base64
    import io

    from PIL import Image

    from harness.call import CallConfig, CallSession
    from harness.camera_orientation import CameraOrientation, RollReading, rotate_image
    from harness.camera_stream import CameraStreamHub
    from qwen_omni_adapters.contract import parse_adapter_request
    from runtime.adapter_server import Config, _language_messages, build_comprehension_payload

    device = "/dev/video0"
    image = Image.new("RGB", (80, 40), "red")
    image.paste("blue", (40, 0, 80, 40))
    source = io.BytesIO()
    image.save(source, "JPEG")
    reader = CameraOrientation(tmp_path / "roll.onnx")
    monkeypatch.setattr(reader, "_predict", lambda _: RollReading(90, .98, .02))
    now = camera.time.monotonic()
    for offset in (10, 5, 0):
        reader.observe(device, source.getvalue(), now - offset)

    class SampleStream:
        clip = CameraStreamHub.clip

        def available(self):
            return [device]

        def jpeg(self, _device, _width, *, roll=0):
            return rotate_image(source.getvalue(), roll)

    cameras = camera.CameraSet(stream=SampleStream(), orientation=reader)
    if motion:
        av = pytest.importorskip("av")
        capture = cameras.clip(.5)
    else:
        capture = cameras.snapshot()
    assert capture is not None

    call = CallSession(CallConfig(token="t", model="m"))
    payload = call._build_payload(b"", 1, capture, with_tools=False)
    payload["stream"] = False
    parsed = parse_adapter_request(payload)
    parts = build_comprehension_payload(parsed, Config("", "", "", "", 1))["messages"][-1]["content"]
    if motion:
        encoded = next(part["input_video"]["data"] for part in parts if part["type"] == "input_video")
        with av.open(io.BytesIO(base64.b64decode(encoded))) as container:
            corrected = next(container.decode(video=0)).to_image()
        assert "camera_orientation" not in payload["messages"][-1]["videos"][0]
    else:
        url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
        corrected = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert "camera_orientation" not in payload["messages"][-1]["images"][0]
    with corrected:
        assert corrected.size == (40, 80)
        assert corrected.getpixel((10, 10))[2] > 200
        assert corrected.getpixel((10, 70))[0] > 200
    messages = _language_messages(parsed, "<speech_transcript>Is the camera upright?</speech_transcript>")
    assert '"upright_confidence":0.02' in messages[0]["content"]
    assert '"correction_degrees":90.0' in messages[0]["content"]
