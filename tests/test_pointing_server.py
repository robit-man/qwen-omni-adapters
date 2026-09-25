"""The dedicated point worker accepts only bounded local image evidence."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from pointing_server import _decode_observation_request, _decode_request  # noqa: E402


def _request(target: str = "blue triangle") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 180), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return ('{"image":"' + encoded + '","target":"' + target + '"}').encode()


def test_pointing_request_decodes_bounded_pixels_and_referring_expression() -> None:
    image, target = _decode_request(_request("  blue   triangle "))

    assert image.size == (320, 180)
    assert image.mode == "RGB"
    assert target == "blue triangle"


def test_observation_request_accepts_bounded_pixels_without_a_prompt() -> None:
    image = _decode_observation_request(_request())

    assert image.size == (320, 180)
    assert image.mode == "RGB"


@pytest.mark.parametrize(
    "payload",
    [b"{}", b'{"image":"not base64","target":"shape"}', _request(241 * "x")],
)
def test_pointing_request_rejects_missing_or_unbounded_input(payload: bytes) -> None:
    with pytest.raises(ValueError):
        _decode_request(payload)
