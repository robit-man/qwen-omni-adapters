"""Resident loopback worker for the Laya System-1 decision plane.

This process is intentionally isolated from the portal and adapter. It owns the
large optional ML dependency, checkpoint lifecycle, warmup, and inference lock.
The rest of the runtime talks only to the typed DecisionPlane abstraction.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_omni_adapters.decision_plane import load_decision_config
from qwen_omni_adapters.memory import (
    MemoryGovernor,
    MemoryPressure,
    release_unused_process_memory,
)

MAX_REQUEST_BYTES = 512 * 1024


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    device: str | None
    preload: tuple[str, ...]
    allow_lazy_load: bool
    warmup: bool
    startup_reserve_gib: float
    max_batch_size: int
    max_wait_ms: int
    state_max_chars: int
    model_id: str
    checkpoint: str | None
    cpu_quantization: str

    @classmethod
    def from_environment(cls) -> ServerConfig:
        catalog = load_decision_config(
            os.environ.get("OMNI_DECISION_PLANE_CONFIG", "").strip() or None
        )
        backend = catalog.get("backend", {})
        batching = catalog.get("batching", {})
        configured_device = os.environ.get(
            "OMNI_LAYA_DEVICE", str(backend.get("device") or "auto")
        ).strip()
        preload_value = os.environ.get("OMNI_LAYA_PRELOAD", "").strip()
        preload = (
            tuple(item.strip() for item in preload_value.split(",") if item.strip())
            if preload_value
            else tuple(str(item) for item in backend.get("preload", ["english"]))
        )
        return cls(
            host=os.environ.get("OMNI_LAYA_HOST", "127.0.0.1"),
            port=int(os.environ.get("OMNI_LAYA_PORT", "8930")),
            device=None if configured_device == "auto" else configured_device,
            preload=preload,
            allow_lazy_load=_bool_value(
                os.environ.get("OMNI_LAYA_ALLOW_LAZY_LOAD"),
                bool(backend.get("allow_lazy_load", False)),
            ),
            warmup=_bool_value(
                os.environ.get("OMNI_LAYA_WARMUP"), bool(backend.get("warmup", True))
            ),
            startup_reserve_gib=float(
                os.environ.get(
                    "OMNI_LAYA_STARTUP_RESERVE_GIB",
                    str(backend.get("startup_reserve_gib", 1.25)),
                )
            ),
            max_batch_size=int(batching.get("max_batch_size", 32)),
            max_wait_ms=max(0, int(batching.get("max_wait_ms", 2))),
            state_max_chars=int(catalog.get("state_max_chars", 12000)),
            model_id=str(backend.get("model_id") or "convaiinnovations/laya"),
            checkpoint=(
                str(backend.get("checkpoint")).strip()
                if backend.get("checkpoint")
                else None
            ),
            cpu_quantization=str(
                os.environ.get(
                    "OMNI_LAYA_CPU_QUANTIZATION",
                    str(backend.get("cpu_quantization") or "none"),
                )
            ).strip().lower(),
        )


class LayaRuntime:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.governor = MemoryGovernor()
        self.lock = threading.Lock()
        self.ready = False
        self.started_at = time.time()
        self.warmup_ms = 0.0
        self.inferences = 0
        self.failures = 0
        self.busy_rejections = 0
        self.last_latency_ms = 0.0
        self.quantization = "none"
        self.router: Any = None
        try:
            self.package_version = importlib.metadata.version("laya")
        except importlib.metadata.PackageNotFoundError:
            self.package_version = "uninstalled"

    def load(self) -> None:
        self.governor.require(
            "Laya checkpoint preload", reserve_gib=self.config.startup_reserve_gib
        )
        from laya import Router

        self.router = Router(
            device=self.config.device,
            max_loaded=max(1, len(self.config.preload)),
            preload=False,
        )
        if self.config.preload:
            self.router.preload(list(self.config.preload))
        self._optimize_cpu_models()
        # Dynamic quantization replaces hundreds of large float tensors. glibc
        # may otherwise retain the dead arenas, making a safe warmup look as
        # though it would cross the shared memory floor.
        release_unused_process_memory()
        if self.config.warmup:
            start = time.perf_counter()
            self.predict(
                {"user_request": "Inspect the current state and choose the route."},
                {
                    "warmup_route": {
                        "type": "choice",
                        "instructions": "Choose the applicable bounded route.",
                        "criteria": {
                            "direct": "no external action",
                            "tool": "a tool action is needed",
                            "deliberate": "open-ended reasoning is needed",
                        },
                    }
                },
                warmup=True,
            )
            self.warmup_ms = (time.perf_counter() - start) * 1000
        self.ready = True

    def _optimize_cpu_models(self) -> None:
        """Reduce resident RAM without changing the application-facing API.

        Laya promotes CPU weights to float32.  That is especially expensive on
        an integrated-memory Jetson, where the deliberative model and desktop
        share the same pool.  The configured CPU mode keeps Laya resident
        without making the optimization an OOM risk.  CUDA and MPS retain
        their native path.
        """

        mode = self.config.cpu_quantization
        if mode in {"", "none", "off", "false"}:
            return
        if mode not in {"float16", "dynamic_int8"}:
            raise ValueError(f"unsupported Laya CPU quantization mode: {mode}")
        import torch

        cpu_agents = [
            agent
            for agent in self.router._agents.values()
            if getattr(getattr(agent, "device", None), "type", "") == "cpu"
        ]
        for agent in cpu_agents:
            if mode == "float16":
                agent.model.to(dtype=torch.float16).eval()
                # Laya deliberately casts the pooled decision features to
                # float32 before the small action head. Keep that tiny head in
                # float32 while the 421M-parameter encoder remains fp16.
                agent.model.act_head.to(dtype=torch.float32)
                agent.dtype = torch.float16
                continue
            torch.backends.quantized.engine = (
                "qnnpack"
                if "qnnpack" in torch.backends.quantized.supported_engines
                else torch.backends.quantized.engine
            )
            # PyTorch's TransformerEncoder fast-path introspects ``.weight``
            # as a Tensor. Dynamic quantized Linear exposes it as a callable,
            # so the generic fast-path must be disabled for this resident
            # process before the quantized module is evaluated.
            torch.backends.mha.set_fastpath_enabled(False)
            agent.model = torch.ao.quantization.quantize_dynamic(
                agent.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
                inplace=True,
            ).eval()
        if cpu_agents:
            self.quantization = mode

    def predict(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        warmup: bool = False,
    ) -> dict[str, Any]:
        if self.router is None:
            raise RuntimeError("Laya router is not loaded")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        if len(questions) > self.config.max_batch_size:
            raise ValueError(
                f"question batch has {len(questions)} entries; limit is {self.config.max_batch_size}"
            )
        state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
        if len(state_json) > self.config.state_max_chars:
            raise ValueError("state exceeds the configured compact-state limit")
        self.governor.require("Laya decision wave")
        route = self.router.route(state, questions, model=self.config.checkpoint)
        checkpoint = str(route.get("model") or "")
        if not self.config.allow_lazy_load and checkpoint not in self.router.loaded:
            raise RuntimeError(
                f"checkpoint {checkpoint!r} is not resident; lazy loading is disabled"
            )
        started = time.perf_counter()
        acquired = self.lock.acquire(timeout=self.config.max_wait_ms / 1000.0)
        if not acquired:
            self.busy_rejections += 1
            raise RuntimeError("decision plane is busy; deliberative fallback required")
        try:
            result = self.router.predict(state, questions, model=checkpoint)
        except Exception:
            self.failures += 1
            raise
        finally:
            self.lock.release()
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_latency_ms = latency_ms
        if not warmup:
            self.inferences += 1
        return {
            **result,
            "routing": dict(route),
            "latency_ms": round(latency_ms, 3),
            "model": self.config.model_id,
            "model_version": f"laya-{self.package_version}:{checkpoint}",
            "checkpoint": checkpoint,
        }

    def health(self) -> dict[str, Any]:
        loaded = list(self.router.loaded) if self.router is not None else []
        devices = []
        if self.router is not None:
            for name in loaded:
                agent = self.router._agents.get(name)  # Laya exposes no public device list.
                if agent is not None:
                    devices.append(str(agent.device))
        return {
            "status": "ok" if self.ready else "loading",
            "ready": self.ready,
            "model": self.config.model_id,
            "model_version": f"laya-{self.package_version}",
            "loaded": loaded,
            "models_loaded": loaded,
            "device": ",".join(sorted(set(devices))) or str(self.config.device or "auto"),
            "allow_lazy_load": self.config.allow_lazy_load,
            "checkpoint": self.config.checkpoint or "language_routed",
            "quantization": self.quantization,
            "warmup_ms": round(self.warmup_ms, 3),
            "inferences": self.inferences,
            "failures": self.failures,
            "busy_rejections": self.busy_rejections,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "uptime_s": round(time.time() - self.started_at, 3),
            "memory_available_gib": round(self.governor.available_gib(), 3),
        }


def create_handler(runtime: LayaRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OmniLaya/1"

        def log_message(self, format: str, *args: Any) -> None:
            # Never write request bodies or user state. The supervisor captures
            # this minimal method/path/status access log.
            sys.stderr.write("laya-server: " + (format % args) + "\n")

        def _send(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value, separators=(",", ":"), default=str).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A caller timeout cancels interest in the shadow result. It
                # must not create a traceback storm in the private service log.
                return

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, runtime.health())

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/predict":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                state = body.get("state")
                questions = body.get("questions")
                if not isinstance(questions, dict):
                    raise ValueError("questions must be an object")
                result = runtime.predict(state, questions)
            except MemoryPressure as exc:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "memory_pressure", "detail": str(exc)},
                )
                return
            except (ValueError, TypeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)[:500]})
                return
            except Exception as exc:  # noqa: BLE001 - isolate optional backend failures
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": type(exc).__name__, "detail": str(exc)[:500]},
                )
                return
            self._send(HTTPStatus.OK, result)

    return Handler


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def main() -> int:
    config = ServerConfig.from_environment()
    runtime = LayaRuntime(config)
    try:
        runtime.load()
    except Exception as exc:  # noqa: BLE001 - startup diagnostic goes to private log
        print(f"laya-server: startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((config.host, config.port), create_handler(runtime))
    server.daemon_threads = True
    server.block_on_close = False
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
