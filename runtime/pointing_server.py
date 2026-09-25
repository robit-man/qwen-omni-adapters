#!/usr/bin/env python3
"""Loopback-only structured visual pointing for rendered browser frames.

The conversational model identifies *what* should be clicked.  This worker has
one deliberately narrow job: map that referring expression to proportional
coordinates in the exact image supplied by the browser executor.  It never
navigates, clicks, or accepts remote image URLs.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import io
import json
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image, UnidentifiedImageError

MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAX_TARGET_CHARS = 240
MAX_RETURNED_POINTS = 32
MAX_OBSERVATION_CHARS = 6000

OBSERVATION_PROMPT = (
    "Read this browser screenshot as current visual evidence. Transcribe all visible "
    "status text and exact identifiers or completion markers, then briefly describe "
    "the current page state. Do not infer text or state that is not visible."
)


def _install_torch_24_gqa_compatibility() -> None:
    """Backport the small ``enable_gqa`` SDPA surface used by Moondream.

    JetPack 6.0's official NVIDIA wheel is Torch 2.4.  Current Moondream 2
    checkpoints use the later ``enable_gqa`` keyword.  Repeating K/V heads is
    the reference grouped-query-attention transformation and keeps inference
    on the official JetPack CUDA build instead of substituting a CPU path.
    """

    import torch
    from torch.nn import functional as functional

    try:
        parameters = inspect.signature(
            functional.scaled_dot_product_attention
        ).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "enable_gqa" in parameters:
        return
    original = functional.scaled_dot_product_attention

    def compatible_sdpa(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:
        if enable_gqa and query.size(-3) != key.size(-3):
            key_heads = int(key.size(-3))
            query_heads = int(query.size(-3))
            if key_heads <= 0 or query_heads % key_heads:
                raise RuntimeError("Moondream returned incompatible grouped-query heads")
            repeats = query_heads // key_heads
            key = key.repeat_interleave(repeats, dim=-3)
            value = value.repeat_interleave(repeats, dim=-3)
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    functional.scaled_dot_product_attention = compatible_sdpa
    # Keep an explicit reference for diagnostics and make the patch idempotent.
    functional.scaled_dot_product_attention._omni_gqa_compat = True  # type: ignore[attr-defined]
    functional.scaled_dot_product_attention._omni_original = original  # type: ignore[attr-defined]
    assert torch.cuda.is_available(), "the pointing worker requires JetPack CUDA"


class PointingModel:
    """Persistent Moondream point head with serialized CUDA inference."""

    def __init__(self, model_id: str, revision: str) -> None:
        _install_torch_24_gqa_compatibility()
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing a CPU pointing fallback")
        self.model_id = model_id
        self.revision = revision
        self._torch = torch
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        ).eval()
        self._lock = threading.Lock()

    def point(self, image: Image.Image, target: str) -> list[dict[str, float]]:
        with self._lock, self._torch.inference_mode():
            result = self._model.point(image, target)
        raw_points = result.get("points") if isinstance(result, dict) else None
        if not isinstance(raw_points, list):
            raise RuntimeError("Moondream returned no structured point list")
        points: list[dict[str, float]] = []
        for raw in raw_points[:MAX_RETURNED_POINTS]:
            if not isinstance(raw, dict):
                continue
            try:
                x, y = float(raw["x"]), float(raw["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                points.append({"x": x, "y": y})
        return points

    def observe(self, image: Image.Image) -> str:
        """Return a bounded semantic reading of the exact supplied frame."""

        with self._lock, self._torch.inference_mode():
            result = self._model.query(image, OBSERVATION_PROMPT)
        answer = result.get("answer") if isinstance(result, dict) else result
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Moondream returned no visual observation")
        return answer.strip()[:MAX_OBSERVATION_CHARS]


def _decode_payload(raw: bytes) -> tuple[dict[str, Any], Image.Image]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("request body must be one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be one JSON object")
    encoded = payload.get("image")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image must be a base64-encoded PNG or JPEG")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image is not valid base64") from exc
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("decoded image is empty or exceeds the byte limit")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image is not a valid PNG or JPEG") from exc
    if max(image.size) > MAX_IMAGE_EDGE or min(image.size) < 2:
        raise ValueError("image dimensions are outside the supported range")
    return payload, image


def _decode_request(raw: bytes) -> tuple[Image.Image, str]:
    payload, image = _decode_payload(raw)
    target = " ".join(str(payload.get("target") or "").split())
    if not target or len(target) > MAX_TARGET_CHARS:
        raise ValueError(f"target must contain 1-{MAX_TARGET_CHARS} characters")
    return image, target


def _decode_observation_request(raw: bytes) -> Image.Image:
    _payload, image = _decode_payload(raw)
    return image


class PointingHandler(BaseHTTPRequestHandler):
    server: PointingServer

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid logging referring expressions or request bodies.
        return

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._json(
            HTTPStatus.OK,
            {
                "ready": True,
                "model": self.server.model.model_id,
                "revision": self.server.model.revision,
                "device": "cuda",
                "capabilities": ["point", "observe"],
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/point", "/observe"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid request size"})
            return
        try:
            raw = self.rfile.read(length)
            if self.path == "/point":
                image, target = _decode_request(raw)
                result: dict[str, Any] = {
                    "points": self.server.model.point(image, target)
                }
            else:
                image = _decode_observation_request(raw)
                result = {"observation": self.server.model.observe(image)}
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - bound the server failure surface.
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"visual inference failed: {type(exc).__name__}"},
            )
            return
        self._json(
            HTTPStatus.OK,
            {
                **result,
                "model": self.server.model.model_id,
                "revision": self.server.model.revision,
            },
        )


class PointingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: PointingModel) -> None:
        self.model = model
        super().__init__(address, PointingHandler)


def main() -> int:
    host = os.environ.get("OMNI_POINTING_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("pointing server must bind to loopback")
    port = int(os.environ.get("OMNI_POINTING_PORT", "8940"))
    model = PointingModel(
        os.environ.get("OMNI_POINTING_MODEL", "vikhyatk/moondream2"),
        os.environ.get(
            "OMNI_POINTING_REVISION",
            "9a7d4024050840e001defacec2b00727e89149e6",
        ),
    )
    server = PointingServer((host, port), model)

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.serve_forever(poll_interval=0.25)
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
