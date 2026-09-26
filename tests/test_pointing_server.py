"""The dedicated point worker accepts only bounded local image evidence."""

from __future__ import annotations

import base64
import io
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from pointing_server import (  # noqa: E402
    PointingModel,
    _decode_observation_request,
    _decode_request,
)


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


def test_pointing_weights_shed_and_reload_before_the_next_grounded_action() -> None:
    events: list[str] = []

    class FakeCuda:
        @staticmethod
        def synchronize() -> None:
            events.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

    class FakeTorch:
        bfloat16 = "bf16"
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeLoadedModel:
        def eval(self):
            return self

        def point(self, _image: Image.Image, _target: str):
            return {"points": [{"x": 0.25, "y": 0.75}]}

    class FakeFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            events.append(f"load:{args[0]}:{kwargs['revision']}")
            return FakeLoadedModel()

    model = PointingModel.__new__(PointingModel)
    model.model_id = "test/point-head"
    model.revision = "revision-1"
    model._torch = FakeTorch()
    model._model_type = FakeFactory
    model._model = FakeLoadedModel()
    model._lock = threading.Lock()

    model.shed()
    assert model.resident is False
    assert events == ["synchronize", "empty_cache"]

    points = model.point(Image.new("RGB", (16, 16)), "target")
    assert points == [{"x": 0.25, "y": 0.75}]
    assert model.resident is True
    assert events[-1] == "load:test/point-head:revision-1"
