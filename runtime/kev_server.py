#!/usr/bin/env python3
"""Pinned Kev-4B native finite Choice service on one broker-leased CUDA GPU.

Uses the eager FP32/unmerged-LoRA path demonstrated in Omnius's Kev service,
with upstream Kev source 2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f
(Apache-2.0). No adjacent checkout, hosted API, or generated confidence is used.
Start through ``docker gpu run``; that wrapper owns lease and process lifetime.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_omni_adapters.accelerator import process_is_gpu_resident

CODE_REVISION = "2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f"
WEIGHTS_REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
BASE_REPOSITORY = "Qwen/Qwen3.5-4B-Base"
MODEL = "jaredpalmer/kev-4b@" + WEIGHTS_REVISION
MAX_REQUEST_BYTES = 128 * 1024
MAX_ROW_TOKENS = 4096


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"model", "state", "questions"}:
        raise ValueError("request must contain only model, state and questions")
    if value["model"] != MODEL:
        raise ValueError("request must identify the exact pinned Kev model")
    if not isinstance(value["state"], str) or not 0 < len(value["state"]) <= 20_000:
        raise ValueError("state must be a bounded string")
    questions = value["questions"]
    if not isinstance(questions, dict) or set(questions) != {"action"}:
        raise ValueError("exactly one action question is supported")
    question = questions["action"]
    if (not isinstance(question, dict) or set(question) != {"type", "instructions", "criteria"}
            or question["type"] != "choice"):
        raise ValueError("action must be a finite Choice question")
    if not isinstance(question["instructions"], str) or not 0 < len(question["instructions"]) <= 2000:
        raise ValueError("instructions must be bounded")
    choices = question["criteria"]
    if not isinstance(choices, dict) or not 2 <= len(choices) <= 50:
        raise ValueError("two to fifty finite choices are required")
    for name, description in choices.items():
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", name) is None
                or not isinstance(description, str) or not 0 < len(description) <= 1000):
            raise ValueError("invalid action ID or description")
    return value


def validate_answer(answer: Any, criteria: dict[str, str]) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "choice" or answer.get("choice") not in criteria:
        raise RuntimeError("Kev returned an invalid finite choice")
    probabilities = answer.get("probabilities")
    if (not isinstance(probabilities, dict) or set(probabilities) != set(criteria)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1
                   for value in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > 0.02):
        raise RuntimeError("Kev returned an invalid native probability distribution")
    return {"type": "choice", "choice": answer["choice"], "probabilities": probabilities}


def broker_transition(action: str, token: str) -> None:
    result = subprocess.run(
        ["docker", "gpu", action, token], check=False,
        capture_output=True, text=True, timeout=330,
    )
    if result.returncode:
        raise RuntimeError(f"GPU broker {action} failed (exit {result.returncode})")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Path]:
    for field, expected in (("code_revision", CODE_REVISION), ("weights_revision", WEIGHTS_REVISION),
                            ("base_revision", BASE_REVISION)):
        if manifest.get(field) != expected:
            raise ValueError(f"Kev manifest {field} differs from the pinned runtime")
    package = Path(importlib.metadata.distribution("kev").locate_file("kev")).resolve()
    code_files = manifest["kev_code_files"]
    if (not {"checkpoint.py", "model.py", "api.py"}.issubset(code_files)
            or {str(path.relative_to(package)) for path in package.rglob("*.py")} != set(code_files)):
        raise ValueError("Kev source inventory differs from the pinned manifest")
    for name, digest in code_files.items():
        path = package / name
        if not path.resolve().is_relative_to(package) or file_digest(path) != digest:
            raise ValueError("installed Kev source differs from the pinned manifest")
    for name in ("torch", "transformers", "peft", "kev"):
        if importlib.metadata.version(name) != manifest["versions"][name]:
            raise ValueError(f"Kev dependency version changed: {name}")
    paths = {}
    for entry in manifest["checkpoints"]:
        path = Path(entry["path"])
        expected = {"adapter": WEIGHTS_REVISION, "base": BASE_REVISION}.get(entry["name"])
        if expected is None or path.name != expected or entry["revision"] != expected:
            raise ValueError("Kev checkpoint path differs from its immutable revision")
        for name, receipt in entry["files"].items():
            candidate = path / name
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or candidate.stat().st_size != receipt["size_bytes"] or file_digest(candidate) != receipt["sha256"]):
                raise ValueError("Kev checkpoint inventory changed")
        paths[entry["name"]] = path
    if set(paths) != {"adapter", "base"}:
        raise ValueError("Kev requires its pinned adapter and base checkpoints")
    return paths


class KevModel:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "").strip()
        self.gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not self.token or re.fullmatch(r"GPU-[a-fA-F0-9-]{36}", self.gpu_uuid) is None:
            raise RuntimeError("Kev must run with one exact broker-leased GPU UUID")
        paths = verify_manifest(manifest)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        # The caller's docker gpu run lease is pending before this first import.
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Kev requires its leased CUDA device; no CPU fallback")
        self.torch = torch
        checkpoint_api = importlib.import_module("kev.checkpoint")
        self.api = importlib.import_module("kev.api")
        checkpoint = checkpoint_api.Checkpoint.__new__(checkpoint_api.Checkpoint)
        checkpoint.path = checkpoint.requested = str(paths["adapter"])
        checkpoint.meta = checkpoint_api.Meta.from_dict(torch.load(
            str(paths["adapter"] / "head.pt"), map_location="cpu", weights_only=True,
        ))
        if (checkpoint.meta.base != BASE_REPOSITORY or checkpoint.meta.base_revision != BASE_REVISION
                or checkpoint.meta.weights != "lora" or checkpoint.meta.weights_dtype != "fp32"):
            raise ValueError("Kev head metadata differs from the pinned FP32 LoRA path")
        checkpoint.meta.base, checkpoint.meta.base_revision = str(paths["base"]), None
        options = checkpoint_api.LoadOptions(dtype=torch.float32, merge=False, attn="sdpa",
                                            backend="torch", cuda_graphs=False, fused=False)
        self.tokenizer, self.model = checkpoint.load("cuda:0", options)
        self.lock = threading.Lock()
        self.calls = 0
        self.assert_resident()

    def assert_resident(self) -> None:
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        if not tensors:
            raise RuntimeError("Kev has no CUDA-resident weights")
        for tensor in tensors:
            if str(tensor.device) != "cuda:0":
                raise RuntimeError("Kev weights left the leased CUDA device")
        self.torch.cuda.synchronize(0)
        if not process_is_gpu_resident(os.getpid(), self.gpu_uuid):
            raise RuntimeError("Kev did not prove own-PID residency on its leased GPU")

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        request = validate_request(request)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Kev is already processing a decision")
        try:
            record, metadata = self.api.to_record(self.api.SystemOneRequest(**request))
            encoded = self.model.encode(self.tokenizer, record, max_state=2048,
                                        max_branch=MAX_ROW_TOKENS, strict=True)
            if len(encoded["ids"]) > MAX_ROW_TOKENS or encoded.get("state_truncated", False):
                raise ValueError("Kev request exceeds the strict token bound")
            broker_transition("prepare", self.token)
            watchdog = threading.Timer(30, lambda: os._exit(124))
            watchdog.daemon = True
            watchdog.start()
            try:
                with self.torch.inference_mode():
                    probabilities = self.model.probs(encoded)
                self.assert_resident()
                answer = self.api.to_answers([value.tolist() for value in probabilities], metadata)["action"]
                answer = validate_answer(answer, request["questions"]["action"]["criteria"])
                broker_transition("ready", self.token)
            except Exception:
                # A failed CUDA interval must exit; the wrapper frees this PID
                # before releasing its scope. Do not resume an uncertain graph.
                os._exit(125)
            finally:
                watchdog.cancel()
            self.calls += 1
            return {"model": MODEL, "revision": WEIGHTS_REVISION, "answers": {"action": answer}}
        finally:
            self.lock.release()

    def close(self) -> None:
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()


class KevHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass  # Never log states, candidate text, or credentials.

    def reply(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            return self.reply(404, {"error": "unknown route"})
        self.reply(200, {"ready": True, "model": MODEL, "revision": WEIGHTS_REVISION,
                         "device": "cuda", "pid": os.getpid(), "calls": self.server.model.calls})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/systemone":
            return self.reply(404, {"error": "unknown route"})
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked requests are unsupported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request size exceeds the bound")
            self.connection.settimeout(10)
            request = validate_request(json.loads(self.rfile.read(length)))
            result = self.server.model.decide(request)
        except (ValueError, OSError):
            return self.reply(400, {"error": "invalid bounded Choice request"})
        except Exception as exc:
            return self.reply(503, {"error": f"decision unavailable: {type(exc).__name__}"})
        self.reply(200, result)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(os.environ.get(
        "OMNI_KEV_MANIFEST", str(repo_root / "runtime-data" / "kev-models.json"),
    )).read_text(encoding="utf-8"))
    model = KevModel(manifest)
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("OMNI_KEV_PORT", "8950"))), KevHandler)
    server.daemon_threads = True
    server.model = model

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
