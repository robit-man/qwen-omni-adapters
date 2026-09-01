from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx

from qwen_omni_adapters.ollama_sidecar import (
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)


class DaemonError(RuntimeError):
    """Raised when the portable runtime cannot enter a safe ready state."""


def _repo_root() -> Path:
    configured = os.environ.get("OMNI_REPO_ROOT", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[2]
    )


def _load_env_file(root: Path) -> None:
    path = root / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _binary(root: Path, name: str) -> Path:
    candidates = [
        root / "vendor" / "llama.cpp" / "build" / "bin" / name,
        root / "vendor" / "llama.cpp" / "build" / "bin" / f"{name}.exe",
        root / "vendor" / "llama.cpp" / "build" / "bin" / "Release" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


@dataclass(frozen=True)
class DaemonConfig:
    repo_root: Path
    runtime_root: Path
    model: str
    language_model: str
    comprehension_port: int = 8901
    tts_port: int = 8892
    adapter_port: int = 8910
    portal_port: int = 8920
    context_tokens: int = 65_536
    tts_stream_frames: int = 4
    portal_token: str = ""
    cloudflare: bool = True
    keep_cache: bool = False
    allow_direct_gpu: bool = False

    @classmethod
    def from_environment(
        cls,
        *,
        cloudflare: bool | None = None,
        allow_direct_gpu: bool = False,
    ) -> DaemonConfig:
        root = _repo_root()
        _load_env_file(root)
        runtime = (
            Path(os.environ.get("OMNI_PORTAL_RUNTIME_ROOT", str(root / "runtime-data")))
            .expanduser()
            .resolve()
        )
        return cls(
            repo_root=root,
            runtime_root=runtime,
            model=os.environ.get(
                "OMNI_MODEL", "robit/qwen3.8-27b-e03-obliterated-omni:q4km"
            ).strip(),
            language_model=os.environ.get(
                "OMNI_LANGUAGE_MODEL", "robit/qwen3.8-27b-obliterated-e03:27b"
            ).strip(),
            context_tokens=int(os.environ.get("OMNI_COMPREHENSION_CONTEXT_TOKENS", "65536")),
            tts_stream_frames=int(os.environ.get("OMNI_TTS_STREAM_FRAMES", "4")),
            portal_token=os.environ.get("OMNI_PORTAL_TOKEN", "").strip(),
            cloudflare=(
                os.environ.get("OMNI_ENABLE_CLOUDFLARED", "1") != "0"
                if cloudflare is None
                else cloudflare
            ),
            keep_cache=os.environ.get("OMNI_KEEP_CACHE", "0") == "1",
            allow_direct_gpu=allow_direct_gpu,
        )


@dataclass
class Child:
    name: str
    process: subprocess.Popen[bytes]
    log: IO[bytes]


class OmniDaemon:
    """Portable foreground supervisor intended to be owned by an OS service manager."""

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.state_dir = config.runtime_root / "state"
        self.log_dir = config.runtime_root / "logs"
        self.cache_dir = config.runtime_root / "components"
        self.session_log_dir = config.runtime_root / "session-logs"
        self.pid_file = self.state_dir / "daemon.pid"
        self.status_file = self.state_dir / "daemon-status.json"
        self.stop_file = self.state_dir / "stop.request"
        self.token_file = self.state_dir / "access-token.txt"
        self.children: list[Child] = []
        self.stop_event = threading.Event()

    def _write_status(self, **fields: Any) -> None:
        value = {
            "schema": "robit.qwen-omni-daemon.status.v1",
            "pid": os.getpid(),
            "platform": platform.system(),
            "model": self.config.model,
            "updated_at": time.time(),
            **fields,
        }
        partial = self.status_file.with_suffix(".tmp")
        partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(partial, self.status_file)

    def _command(self, command: list[str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise DaemonError(f"{' '.join(command[:3])} failed: {detail}")
        return completed

    def _ensure_model(self, model: str) -> None:
        shown = subprocess.run(
            ["ollama", "show", model], check=False, capture_output=True, text=True
        )
        if shown.returncode != 0:
            self._write_status(state="pulling", detail=model)
            self._command(["ollama", "pull", model], timeout=6 * 3600)

    def _verify_shared_base(self) -> None:
        def sources(model: str) -> list[str]:
            result = self._command(["ollama", "show", model, "--modelfile"])
            return [
                line.split(maxsplit=1)[1]
                for line in result.stdout.splitlines()
                if line.strip().startswith("FROM ") and len(line.split(maxsplit=1)) == 2
            ]

        if not sources(self.config.model) or sources(self.config.model) != sources(
            self.config.language_model
        ):
            raise DaemonError(
                "OMNI_LANGUAGE_MODEL does not share the logical tag's standard base/projector blobs"
            )

    def _broker_present(self) -> bool:
        if platform.system() != "Linux" or not shutil.which("docker"):
            return False
        result = subprocess.run(
            ["docker", "gpu", "discover"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return result.returncode == 0

    def _preflight(self) -> None:
        if platform.system() == "Linux" and self._broker_present():
            raise DaemonError(
                "ollama-unify GPU broker detected: use the Linux systemd unit or portal/start.sh; "
                "the portable direct supervisor will not bypass broker leases"
            )
        if platform.system() == "Linux" and not self.config.allow_direct_gpu:
            raise DaemonError(
                "direct Linux GPU mode requires --allow-direct-gpu on a host without ollama-unify"
            )
        required = ["ffmpeg", "ollama"]
        if self.config.cloudflare:
            required.append("cloudflared")
        missing = [command for command in required if not shutil.which(command)]
        if missing:
            raise DaemonError(f"missing commands: {', '.join(missing)}")
        for binary_name in ("llama-server", "llama-tts"):
            binary = _binary(self.config.repo_root, binary_name)
            if not binary.is_file():
                raise DaemonError(
                    f"missing {binary_name}; run the platform bootstrap first: {binary}"
                )
        for port in (
            self.config.comprehension_port,
            self.config.tts_port,
            self.config.adapter_port,
            self.config.portal_port,
        ):
            if not _port_available("127.0.0.1", port):
                raise DaemonError(f"required loopback port is already in use: {port}")

    def prepare(self) -> None:
        os.umask(0o077)
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.config.runtime_root,
            self.state_dir,
            self.log_dir,
            self.cache_dir,
            self.session_log_dir,
        ):
            if directory.exists():
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass
        if self.pid_file.is_file():
            try:
                prior = int(self.pid_file.read_text().strip())
            except ValueError:
                prior = 0
            if prior and _pid_alive(prior):
                raise DaemonError(f"daemon is already running as pid {prior}")
        self.stop_file.unlink(missing_ok=True)
        self._write_status(state="preflight")
        self._preflight()
        self._ensure_model(self.config.model)
        self._ensure_model(self.config.language_model)
        self._verify_shared_base()
        resolved = resolve_ollama_sidecar(model=self.config.model)
        self._write_status(state="materializing", sidecar_digest=resolved["layer"]["digest"])
        required = [
            self.cache_dir / "comprehension-model.gguf",
            self.cache_dir / "comprehension-projector.gguf",
            self.cache_dir / "tts-model.gguf",
            self.cache_dir / "tts-projector.gguf",
        ]
        if not all(path.is_file() for path in required):
            prepare_ollama_sidecar(
                model=self.config.model,
                output_dir=self.cache_dir,
                overwrite=True,
            )

    def _spawn(self, name: str, command: list[str], env: dict[str, str] | None = None) -> Child:
        log = (self.log_dir / f"{name}.log").open("ab", buffering=0)
        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        process = subprocess.Popen(
            command,
            cwd=self.config.repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        child = Child(name=name, process=process, log=log)
        self.children.append(child)
        return child

    def _wait_http(self, child: Child, url: str, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.process.poll() is not None:
                raise DaemonError(f"{child.name} exited before readiness; inspect {self.log_dir}")
            try:
                response = httpx.get(url, timeout=5)
                if response.status_code < 400:
                    return
            except httpx.HTTPError:
                pass
            self.stop_event.wait(1)
        raise DaemonError(f"{child.name} readiness timed out: {url}")

    def _verify_direct_gpu(self, pid: int) -> None:
        if platform.system() == "Darwin":
            return  # bootstrap enforces a Metal-enabled llama.cpp build
        if not shutil.which("nvidia-smi"):
            raise DaemonError("nvidia-smi is required to prove direct CUDA residency")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                fields = [value.strip() for value in line.split(",")]
                if (
                    len(fields) == 2
                    and fields[0] == str(pid)
                    and fields[1].isdigit()
                    and int(fields[1]) > 0
                ):
                    return
            time.sleep(1)
        raise DaemonError(f"comprehension pid {pid} did not become CUDA-resident")

    def start_children(self) -> str:
        python = sys.executable
        common = os.environ.copy()
        common["PYTHONUNBUFFERED"] = "1"
        comprehension = self._spawn(
            "comprehension",
            [
                str(_binary(self.config.repo_root, "llama-server")),
                "-m",
                str(self.cache_dir / "comprehension-model.gguf"),
                "--mmproj",
                str(self.cache_dir / "comprehension-projector.gguf"),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.config.comprehension_port),
                "--jinja",
                "-ngl",
                "99",
                "-c",
                str(self.config.context_tokens),
            ],
            common,
        )
        self._wait_http(
            comprehension,
            f"http://127.0.0.1:{self.config.comprehension_port}/health",
            1200,
        )
        self._verify_direct_gpu(comprehension.process.pid)

        tts_env = {
            **common,
            "LLAMA_TTS_BIN": str(_binary(self.config.repo_root, "llama-tts")),
            "OMNI_TTS_MODEL_GGUF": str(self.cache_dir / "tts-model.gguf"),
            "OMNI_TTS_PROJECTOR_GGUF": str(self.cache_dir / "tts-projector.gguf"),
            "OMNI_COMPONENT_CACHE": str(self.cache_dir),
            "OMNI_TTS_GPU_LAYERS": "-1",
            "OMNI_TTS_REQUIRE_GPU": "0" if platform.system() == "Darwin" else "1",
            "OMNI_TTS_STREAM_FRAMES": str(self.config.tts_stream_frames),
            "OMNI_TTS_HOST": "127.0.0.1",
            "OMNI_TTS_PORT": str(self.config.tts_port),
        }
        tts = self._spawn(
            "tts", [python, str(self.config.repo_root / "runtime" / "tts_server.py")], tts_env
        )
        self._wait_http(tts, f"http://127.0.0.1:{self.config.tts_port}/healthz", 60)

        adapter_env = {
            **common,
            "OMNI_COMPREHENSION_URL": f"http://127.0.0.1:{self.config.comprehension_port}/v1/chat/completions",
            "OMNI_COMPREHENSION_MODEL": "local-qwen3-omni",
            "OMNI_COMPREHENSION_CONTEXT_TOKENS": str(self.config.context_tokens),
            "OMNI_LANGUAGE_URL": "http://127.0.0.1:11434",
            "OMNI_LANGUAGE_MODEL": self.config.language_model,
            "OMNI_TTS_URL": f"http://127.0.0.1:{self.config.tts_port}/synthesize",
            "OMNI_ADAPTER_HOST": "127.0.0.1",
            "OMNI_ADAPTER_PORT": str(self.config.adapter_port),
        }
        adapter = self._spawn(
            "adapter",
            [python, str(self.config.repo_root / "runtime" / "adapter_server.py")],
            adapter_env,
        )
        self._wait_http(adapter, f"http://127.0.0.1:{self.config.adapter_port}/healthz", 60)

        token = self.config.portal_token or secrets.token_urlsafe(32)
        if len(token) < 24:
            raise DaemonError("OMNI_PORTAL_TOKEN must contain at least 24 characters")
        self.token_file.write_text(token + "\n", encoding="utf-8")
        try:
            self.token_file.chmod(0o600)
        except OSError:
            pass
        portal_env = {
            **common,
            "OMNI_MODEL": self.config.model,
            "OMNI_PORTAL_TOKEN": token,
            "OMNI_ADAPTER_URL": f"http://127.0.0.1:{self.config.adapter_port}/api/chat",
            "OMNI_ADAPTER_HEALTH_URL": f"http://127.0.0.1:{self.config.adapter_port}/healthz",
            "OMNI_COMPREHENSION_HEALTH_URL": f"http://127.0.0.1:{self.config.comprehension_port}/health",
            "OMNI_TTS_HEALTH_URL": f"http://127.0.0.1:{self.config.tts_port}/healthz",
            "OMNI_PORTAL_SESSION_LOG_DIR": str(self.session_log_dir),
            "OMNI_PORTAL_HOST": "127.0.0.1",
            "OMNI_PORTAL_PORT": str(self.config.portal_port),
        }
        portal = self._spawn(
            "portal", [python, str(self.config.repo_root / "portal" / "app.py")], portal_env
        )
        self._wait_http(portal, f"http://127.0.0.1:{self.config.portal_port}/healthz", 60)

        self._command(
            [
                python,
                str(self.config.repo_root / "portal" / "smoke.py"),
                "--endpoint",
                f"http://127.0.0.1:{self.config.portal_port}",
                "--token-file",
                str(self.token_file),
                "--model",
                self.config.model,
                "--text",
                "--tts",
                "--stream",
            ],
            timeout=1200,
        )

        public = f"http://127.0.0.1:{self.config.portal_port}"
        if self.config.cloudflare:
            tunnel = self._spawn(
                "cloudflared",
                [
                    "cloudflared",
                    "tunnel",
                    "--no-autoupdate",
                    "--url",
                    public,
                    "--loglevel",
                    "info",
                ],
                common,
            )
            tunnel_log = self.log_dir / "cloudflared.log"
            deadline = time.monotonic() + 120
            pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
            while time.monotonic() < deadline:
                if tunnel.process.poll() is not None:
                    raise DaemonError("cloudflared exited before publishing a URL")
                text = tunnel_log.read_text(encoding="utf-8", errors="replace")
                matches = pattern.findall(text)
                if matches:
                    public = matches[-1]
                    break
                time.sleep(1)
            else:
                raise DaemonError("cloudflared did not publish a URL")
        return f"{public}/#access={token}"

    def request_stop(self) -> None:
        self.stop_event.set()

    def stop_children(self) -> None:
        for child in reversed(self.children):
            if child.process.poll() is None:
                if os.name == "nt":
                    child.process.terminate()
                else:
                    os.killpg(child.process.pid, signal.SIGTERM)
                try:
                    child.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        child.process.kill()
                    else:
                        os.killpg(child.process.pid, signal.SIGKILL)
                    child.process.wait(timeout=10)
            child.log.close()
        self.children.clear()

    def cleanup(self) -> None:
        self.stop_children()
        self.stop_file.unlink(missing_ok=True)
        self.pid_file.unlink(missing_ok=True)
        self.token_file.unlink(missing_ok=True)
        if not self.config.keep_cache:
            for name in (
                "comprehension-model.gguf",
                "comprehension-projector.gguf",
                "tts-model.gguf",
                "tts-projector.gguf",
            ):
                (self.cache_dir / name).unlink(missing_ok=True)
        self._write_status(state="stopped", children=[])

    def run(self, *, register_signals: bool = True) -> int:
        self.prepare()
        self.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        if register_signals:
            signal.signal(signal.SIGTERM, lambda *_: self.request_stop())
            signal.signal(signal.SIGINT, lambda *_: self.request_stop())
        try:
            access_url = self.start_children()
            self._write_status(
                state="ready",
                access_url=access_url,
                children=[
                    {"name": child.name, "pid": child.process.pid} for child in self.children
                ],
            )
            print(access_url, flush=True)
            while not self.stop_event.wait(1):
                if self.stop_file.exists():
                    self.request_stop()
                    continue
                for child in self.children:
                    if child.process.poll() is not None:
                        raise DaemonError(
                            f"{child.name} exited unexpectedly; inspect {self.log_dir}"
                        )
            return 0
        finally:
            self.cleanup()


def _state(config: DaemonConfig) -> dict[str, Any]:
    path = config.runtime_root / "state" / "daemon-status.json"
    if not path.is_file():
        return {"state": "not-installed", "runtime_root": str(config.runtime_root)}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"state": "invalid", "error": str(exc), "path": str(path)}
    pid = int(result.get("pid") or 0)
    result["process_alive"] = bool(pid and _pid_alive(pid))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-omni-daemon")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the portable supervisor in the foreground")
    serve.add_argument("--no-tunnel", action="store_true")
    serve.add_argument(
        "--allow-direct-gpu",
        action="store_true",
        help="Required for unmanaged Linux; rejected when ollama-unify is detected",
    )
    sub.add_parser("status", help="Print the daemon status record")
    sub.add_parser("stop", help="Request a graceful stop through the runtime control file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DaemonConfig.from_environment(
        cloudflare=False if getattr(args, "no_tunnel", False) else None,
        allow_direct_gpu=getattr(args, "allow_direct_gpu", False),
    )
    if args.command == "status":
        print(json.dumps(_state(config), indent=2, sort_keys=True))
        return 0
    if args.command == "stop":
        state = _state(config)
        if not state.get("process_alive"):
            print(json.dumps(state, indent=2, sort_keys=True))
            return 1
        stop = config.runtime_root / "state" / "stop.request"
        stop.write_text(f"requested {time.time()}\n", encoding="utf-8")
        print(f"stop requested for pid {state['pid']}")
        return 0
    try:
        return OmniDaemon(config).run()
    except (DaemonError, OSError, subprocess.SubprocessError) as exc:
        print(f"qwen-omni-daemon: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
