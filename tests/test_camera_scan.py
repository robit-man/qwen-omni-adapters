from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image

from harness.camera import CameraSet
from harness.camera_orientation import CameraOrientation, _roll
from harness.camera_scan import (
    ACTIVE_WIDTH,
    IDLE_WIDTH,
    CameraActivity,
    CameraScanner,
    CameraTuner,
    decode_qr,
)
from portal.qr_intake import classify_camera_code


@pytest.mark.parametrize("payload,kind", [
    ("https://example.org/archive", "link"),
    ("https://example.org/mcp", "mcp"),
    ("WIFI:T:WPA;S:Example;P:secret;;", "private"),
    ("A friendly text label", "text"),
    ('{"token":"opaque-value"}', "structured"),
    ('{"mcpServers":{"test":{"command":"curl","args":["test"]}}}', "unsupported_mcp"),
])
def test_qr_content_is_classified_without_executing_it(payload, kind):
    assert classify_camera_code(payload).kind == kind


def test_real_qr_decoder_reads_rotated_credential_and_deduplicates_across_cameras():
    zxing = pytest.importorskip("zxingcpp")
    payload = '{"mcpServers":{"demo":{"url":"https://example.org/mcp","headers":{"Authorization":"Bearer fixture-key"}}}}'
    barcode = zxing.create_barcode(payload, zxing.BarcodeFormat.QRCode)
    frame = Image.fromarray(barcode.to_image(scale=4)).convert("L").rotate(27, expand=True, fillcolor=255)
    assert decode_qr(frame, zxing) == [payload]
    code = classify_camera_code(payload)
    assert (code.endpoint, code.key) == ("https://example.org/mcp", "fixture-key")
    assert "fixture-key" not in repr(code)
    scanner = CameraScanner(CameraSet(), enabled=lambda: True, submit=lambda *_: None)
    assert scanner._queue(payload, "left", 1.0)
    assert not scanner._queue(payload, "right", 2.0)
    assert scanner._queue(payload, "right", 10.0)


def test_camera_activity_promotes_only_the_moving_view_and_returns_to_idle():
    left, right = CameraActivity(), CameraActivity()
    quiet = Image.new("L", (80, 48), color=90)
    changed = quiet.copy()
    changed.paste(230, (0, 0, 32, 40))
    for camera in (left, right):
        assert camera.due(1.0) == IDLE_WIDTH
        camera.observe(quiet, 1.0, detected=False)
    left.observe(changed, 1.4, detected=False)
    right.observe(Image.new("L", (80, 48), color=150), 1.4, detected=False)
    assert left.due(1.71) == ACTIVE_WIDTH
    assert right.due(1.71) is None
    assert right.due(2.5) == IDLE_WIDTH
    assert left.due(7.0) == IDLE_WIDTH


def test_auto_tuning_discovers_controls_trims_contrast_and_restores(monkeypatch):
    values = {"focus_auto": 0, "contrast": 100}

    def controls(device, argument):
        assert device == "/dev/video0"
        if argument == "--list-ctrls-menus":
            return "focus_auto 0x1 (bool) : default=1 value=0\ncontrast 0x2 (int) : min=0 max=200 step=5 default=100 value=100"
        action, name, *value = argument.split("=")
        if action == "--set-ctrl":
            values[name] = int(value[0])
        if action == "--get-ctrl":
            return f"{name}: {values[name]}"
        return ""

    monkeypatch.setattr("harness.camera_scan._camera_controls", controls)
    tuner = CameraTuner()
    flat = Image.new("L", (40, 24), color=120)
    for now in (1.0, 1.5, 2.0):
        tuner.observe("/dev/video0", flat, now)
    assert values == {"focus_auto": 1, "contrast": 110}
    tuner.restore()
    assert values == {"focus_auto": 0, "contrast": 100}


def test_orientation_waits_for_consensus_and_rotates_the_original_pixels(tmp_path, monkeypatch):
    reader = CameraOrientation(tmp_path / "roll.onnx")
    estimate = iter([(90, .95), (90, .25), (90, .98), (91, .96), (89, .99)])
    monkeypatch.setattr(reader, "_predict", lambda _: next(estimate))
    source = io.BytesIO()
    Image.new("RGB", (40, 20), color="green").save(source, "JPEG")
    frame = source.getvalue()
    now = time.monotonic() - 25
    for step in range(4):
        reader.observe("left", frame, now + step * 5)
        assert reader.correct("left", frame) == frame
    reader.observe("left", frame, now + 20)
    assert reader.roll("left") == pytest.approx(90)
    result = Image.open(io.BytesIO(reader.correct("left", frame)))
    assert result.size == (20, 40)
    assert reader.correct("right", frame) == frame
    bins = np.zeros(360, dtype=np.float32)
    bins[359], bins[0] = .5, .5
    assert _roll(bins) == pytest.approx((-.5, 1))


def test_rolling_ocr_recognizes_printed_text_without_opening_a_camera():
    import io
    import shutil

    import pytest
    from PIL import Image, ImageDraw, ImageFont

    from harness.camera_ocr import RollingOCR

    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is unavailable")
    image = Image.new("L", (950, 240), color=255)
    font = ImageFont.load_default(size=52)
    ImageDraw.Draw(image).text((35, 90), "CAMERA TEST 4817", fill=0, font=font)
    frame = io.BytesIO()
    image.save(frame, "PNG")
    reading = RollingOCR().recognize(frame.getvalue())
    assert reading is not None
    assert "CAMERA TEST 4817" in reading.text.upper()
    assert reading.confidence >= 50
