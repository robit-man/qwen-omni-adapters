#!/usr/bin/env python3
"""Loopback-only structured visual pointing for current desktop/application frames.

The conversational model identifies *what* should be clicked.  This worker has
one deliberately narrow job: map that referring expression to proportional
coordinates in the exact image supplied by the browser executor.  It never
navigates, clicks, or accepts remote image URLs.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_omni_adapters.accelerator import is_tegra, process_is_gpu_resident

MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAX_TARGET_CHARS = 240
MAX_RETURNED_POINTS = 32
MAX_OBSERVATION_CHARS = 6000
PHOTON_MODEL = "moondream/moondream3.1-9B-A2B"
PHOTON_REVISION = "1da27e6c2d2f9118de14560df456881dc62d91f3"
PHOTON_WEIGHTS_SHA256 = "26dde16af961d3a486c7b86b9a34fc8487be7917f7e62324b751400ef4d391b4"
PHOTON_WEIGHTS_BYTES = 10505269576
PHOTON_TOKENIZER_REVISION = "35192e10a54e36eabe0a7cc57a2c1aab371cafc5"
PHOTON_TOKENIZER_SHA256 = "0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c"
PHOTON_TOKENIZER_BYTES = 3694206
PHOTON_CUDA12_SHA256 = "256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7"
PHOTON_CUDA12_BYTES = 741088
PHOTON_PACKAGES = {
    "moondream": "2.6.1", "kestrel": "0.9.1", "kestrel-native": "0.1.8",
    "kestrel-kernels": "0.7.4", "nvidia-cuda-runtime-cu12": "12.9.79", "torch": "2.12.0",
    **{f"kestrel-kernels-bundle-{letter}": "0.7.4" for letter in "abcdefgh"},
}

OBSERVATION_PROMPT = (
    "Read this screenshot as current visual evidence. Transcribe all visible "
    "status text and exact identifiers or completion markers, then briefly describe "
    "the current page state. Do not infer text or state that is not visible."
)


class _PinnedPhotonFile:
    """Hash an exact local inode once, retaining its descriptor for each load."""

    def __init__(self, path: Path, sha256: str, size: int, *, suffix: str = "") -> None:
        self.source = path
        self._alias_directory: tempfile.TemporaryDirectory[str] | None = None
        self._alias: Path | None = None
        self.stream = path.open("rb")
        try:
            self.identity = self._identity(os.fstat(self.stream.fileno()))
            if self.identity[2] != size:
                raise RuntimeError("Photon checkpoint/runtime file has the wrong byte length")
            digest = hashlib.sha256()
            for block in iter(lambda: self.stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != sha256:
                raise RuntimeError("Photon checkpoint/runtime file SHA-256 mismatch")
            self._descriptor_path = f"/proc/self/fd/{self.stream.fileno()}"
            self.path = self._descriptor_path
            if suffix:
                # Kestrel dispatches formats by filename suffix. A private
                # suffixed alias still opens this exact hash-checked inode.
                self._alias_directory = tempfile.TemporaryDirectory(prefix="omni-photon-pinned-")
                self._alias = Path(self._alias_directory.name) / ("checkpoint" + suffix)
                self._alias.symlink_to(self._descriptor_path)
                self.path = str(self._alias)
            self.assert_unchanged()
        except BaseException:
            self.stream.close()
            if self._alias_directory is not None:
                self._alias_directory.cleanup()
            raise

    @staticmethod
    def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    def assert_unchanged(self) -> None:
        if (self._identity(os.fstat(self.stream.fileno())) != self.identity
                or self._identity(self.source.stat()) != self.identity):
            raise RuntimeError("Pinned Photon checkpoint/runtime file changed; refusing inference")
        if self._alias is not None and (
            not self._alias.is_symlink() or os.readlink(self._alias) != self._descriptor_path
        ):
            raise RuntimeError("Pinned Photon checkpoint alias changed; refusing inference")


def _photon_artifacts() -> dict[str, _PinnedPhotonFile]:
    for name, expected in PHOTON_PACKAGES.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Photon requires pinned {name}=={expected}; run scripts/bootstrap_photon_pointing.sh")
    root = Path(__file__).resolve().parents[1] / "runtime-data" / "pointing"
    cuda12 = Path(importlib.metadata.distribution("nvidia-cuda-runtime-cu12").locate_file(
        "nvidia/cuda_runtime/lib/libcudart.so.12"))
    return {
        "weights": _PinnedPhotonFile(Path(os.environ.get("OMNI_PHOTON_WEIGHTS", str(root / "model.safetensors"))),
                                     PHOTON_WEIGHTS_SHA256, PHOTON_WEIGHTS_BYTES, suffix=".safetensors"),
        "tokenizer": _PinnedPhotonFile(Path(os.environ.get("OMNI_PHOTON_TOKENIZER", str(root / "tokenizer.json"))),
                                       PHOTON_TOKENIZER_SHA256, PHOTON_TOKENIZER_BYTES),
        "cuda12": _PinnedPhotonFile(cuda12, PHOTON_CUDA12_SHA256, PHOTON_CUDA12_BYTES),
    }


def _fatal_photon_cleanup(error: Exception | None = None, *, stage: str = "") -> None:
    # Native engine allocations cannot be cleared by torch.empty_cache alone.
    # Losing their owner requires process teardown before the broker may ready.
    if error is not None:
        diagnostic = {"stage": stage, "type": type(error).__name__}
        # Constructor failures cannot contain an inference request. Cleanup
        # exceptions can, so their text stays private and only the type is logged.
        if stage == "constructor":
            detail = str(error)
            for key in ("OLLAMA_UNIFY_GPU_LEASE", "OMNI_PORTAL_TOKEN", "MOONDREAM_API_KEY", "HF_TOKEN"):
                if os.environ.get(key):
                    detail = detail.replace(os.environ[key], "[credential]")
            detail = re.sub(r"(?:https?|wss?)://\S+", "[URL]", detail)
            detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [credential]", detail)
            detail = re.sub(r"(?:[A-Za-z]:[\\/]|/|\.\.?/)[^\s'\"<>;,]+", "[path]", detail)
            diagnostic["message"] = " ".join(detail.split())[:1000]
        print("Photon failure: " + json.dumps(diagnostic), file=sys.stderr, flush=True)
    print("Photon engine lifecycle became uncertain; terminating owned point worker", file=sys.stderr, flush=True)
    os._exit(1)


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
    assert torch.cuda.is_available(), "the pointing worker requires CUDA"


def _broker_present() -> bool:
    if is_tegra() or not shutil.which("docker"):
        return False
    result = subprocess.run(
        ["docker", "gpu", "discover"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
    )
    return result.returncode == 0


def _broker_transition(action: str, token: str) -> None:
    result = subprocess.run(
        ["docker", "gpu", action, token],
        check=False, capture_output=True, text=True, timeout=330,
    )
    if result.returncode != 0:
        # Broker output can contain the lease credential; never echo it.
        raise RuntimeError(f"GPU broker {action} failed (exit {result.returncode})")


class PointingModel:
    """On-demand Moondream point head with serialized CUDA inference."""

    def __init__(self, model_id: str, revision: str) -> None:
        self.model_id = model_id
        self.revision = revision
        self._photon = model_id == PHOTON_MODEL
        if self._photon and revision != PHOTON_REVISION:
            raise RuntimeError("Moondream3.1 requires its exact pinned checkpoint revision")
        self._photon_files: dict[str, _PinnedPhotonFile] | None = None
        self._cuda12_library: Any | None = None
        self._lease_token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "").strip()
        self._gpu_uuid = os.environ.get("OMNI_POINTING_GPU_UUID", "").strip()
        if self._lease_token:
            if not self._gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != self._gpu_uuid:
                raise RuntimeError("pointing worker must use exactly its leased GPU UUID")
        elif _broker_present():
            raise RuntimeError("broker-managed pointing requires a scoped GPU lease")
        # Keep the control plane free of Torch imports and CUDA initialization.
        # Even cuda.is_available() is deferred until the lease is prepared.
        self._torch: Any | None = None
        self._model_type: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._growth_pending = False

    @property
    def resident(self) -> bool:
        with self._lock:
            return self._model is not None

    def _load_locked(self) -> None:
        if self._lease_token and not self._growth_pending:
            self._growth_pending = True
            _broker_transition("prepare", self._lease_token)
        if self._model is not None:
            return
        if self._photon:
            self._load_photon_locked()
            return
        if self._model_type is None:
            import torch
            from transformers import AutoModelForCausalLM

            self._torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; refusing a CPU pointing fallback")
            _install_torch_24_gqa_compatibility()
            self._model_type = AutoModelForCausalLM
        self._model = self._model_type.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=self._torch.bfloat16,
            device_map={"": "cuda"},
        ).eval()

    def _load_photon_locked(self) -> None:
        if self._photon_files is None:
            self._photon_files = _photon_artifacts()
        for artifact in self._photon_files.values():
            artifact.assert_unchanged()
        if self._torch is None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; refusing a CPU pointing fallback")
            self._torch = torch
        if self._cuda12_library is None:
            # The packaged Kestrel loader opens libcudart.so.12. Use the exact
            # already-proven 12.9 inode instead of this host's older CUDA12.0.
            self._cuda12_library = ctypes.CDLL(self._photon_files["cuda12"].path, mode=ctypes.RTLD_GLOBAL)
        import moondream

        try:
            self._model = moondream.photon(
                "moondream3.1-9B-A2B", api_key="", device="cuda:0",
                model_path=self._photon_files["weights"].path,
                tokenizer_path=self._photon_files["tokenizer"].path,
                max_batch_size=1, kv_cache_pages=8192, enable_cuda_graphs=False,
                enable_prefix_cache=True, decode_path="native", service_name="qwen_omni_pointing",
            )
        except Exception as exc:
            # A failed native constructor may have no public close handle.
            _fatal_photon_cleanup(exc, stage="constructor")
            raise

    def _ready_locked(self) -> None:
        assert self._torch is not None
        self._torch.cuda.synchronize()
        deadline = time.monotonic() + 120
        while not process_is_gpu_resident(os.getpid(), self._gpu_uuid):
            if time.monotonic() >= deadline:
                raise RuntimeError("pointing worker did not prove CUDA residency on its GPU")
            time.sleep(0.1)
        if self._growth_pending:
            _broker_transition("ready", self._lease_token)
            self._growth_pending = False

    def warm(self) -> None:
        with self._lock:
            try:
                self._load_locked()
                self._ready_locked()
            except Exception as exc:
                traceback.clear_frames(exc.__traceback__)
                self._shed_locked()
                raise

    def _shed_locked(self) -> None:
        if self._photon:
            try:
                self._shed_impl_locked()
            except Exception as exc:
                # This includes CUDA synchronization and allocator failures;
                # none can establish that native allocations were freed.
                _fatal_photon_cleanup(exc, stage="cleanup")
                raise
        else:
            self._shed_impl_locked()

    def _shed_impl_locked(self) -> None:
        if (
            self._model is None and not self._growth_pending
            and (self._torch is None or not self._torch.cuda.is_initialized())
        ):
            return
        if self._torch is not None and self._torch.cuda.is_initialized():
            self._torch.cuda.synchronize()
        if self._photon and self._model is not None:
            client, self._model = self._model, None
            try:
                client.close()
            finally:
                del client
        else:
            self._model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_initialized():
            self._torch.cuda.empty_cache()
        if self._growth_pending:
            # Failed loads can leave allocator caches even without a model
            # object. Free those before restoring the stable shared lease.
            _broker_transition("ready", self._lease_token)
            self._growth_pending = False

    def shed(self) -> None:
        """Release optional pointing weights while retaining the HTTP control plane."""

        with self._lock:
            self._shed_locked()

    def point(self, image: Image.Image, target: str) -> list[dict[str, float]]:
        with self._lock:
            try:
                self._load_locked()
                assert self._model is not None and self._torch is not None
                with self._torch.inference_mode():
                    if self._photon:
                        result = self._model.point(image, target, settings={"max_tokens": 64, "temperature": 0.0})
                    else:
                        result = self._model.point(image, target)
                self._ready_locked()
            except Exception as exc:
                traceback.clear_frames(exc.__traceback__)
                raise
            finally:
                self._shed_locked()
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

        with self._lock:
            try:
                self._load_locked()
                assert self._model is not None and self._torch is not None
                with self._torch.inference_mode():
                    if self._photon:
                        result = self._model.query(image, OBSERVATION_PROMPT, settings={"max_tokens": 512, "temperature": 0.0})
                    else:
                        result = self._model.query(image, OBSERVATION_PROMPT)
                self._ready_locked()
            except Exception as exc:
                traceback.clear_frames(exc.__traceback__)
                raise
            finally:
                self._shed_locked()
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
                "resident": self.server.model.resident,
                "capabilities": ["point", "observe"],
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/point", "/observe", "/residency"}:
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
            if self.path == "/residency":
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be one JSON object")
                action = str(payload.get("action") or "").strip().lower()
                if action == "shed":
                    self.server.model.shed()
                elif action == "warm":
                    self.server.model.warm()
                else:
                    raise ValueError("residency action must be shed or warm")
                result: dict[str, Any] = {
                    "ok": True,
                    "action": action,
                    "resident": self.server.model.resident,
                }
            elif self.path == "/point":
                image, target = _decode_request(raw)
                result = {
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
    model_id = os.environ.get("OMNI_POINTING_MODEL", "vikhyatk/moondream2")
    model = PointingModel(
        model_id,
        os.environ.get(
            "OMNI_POINTING_REVISION",
            PHOTON_REVISION if model_id == PHOTON_MODEL else "9a7d4024050840e001defacec2b00727e89149e6",
        ),
    )
    server = PointingServer((host, port), model)

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        model.shed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
