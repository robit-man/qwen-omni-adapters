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

from qwen_omni_adapters.accelerator import (
    accelerator_profile,
    compute_apps_supported,
    is_tegra,
    process_is_gpu_resident,
    residency_backend,
)
from qwen_omni_adapters.model_catalog import MODEL_BY_TAG
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


def _language_disable_thinking_default(logical_model: str) -> bool:
    managed = MODEL_BY_TAG.get(logical_model)
    return bool(managed and managed.language_disable_thinking)


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
        # The child servers use reusable listening sockets. Probe with the
        # same semantics: after a clean restart, old loopback connections can
        # remain in TIME_WAIT even though no process owns the port. A plain
        # bind calls that "in use" and creates a restart loop lasting until
        # the kernel expires those connections.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
    decision_port: int = 8930
    context_tokens: int = 65_536
    comprehension_parallel_slots: int = 1
    tts_stream_frames: int = 8
    portal_token: str = ""
    cloudflare: bool = True
    keep_cache: bool = False
    allow_direct_gpu: bool = False
    # Qwen3-Omni comprehension is by far the largest component (about 16.8 GiB
    # resident for the 30B-A3B Q4_K_M weights plus KV). A host that wants the
    # tag's language/vision through Ollama and Qwen3-TTS speech, but cannot
    # spare that, can run the adapter without it: audio, video, and image
    # comprehension then report unavailable and every other route still works.
    enable_comprehension: bool = True
    # The portable daemon normally proves the entire stack with a language
    # and TTS generation before declaring itself ready. A memory-brokered
    # deployment must not do that at service start: those generations bypass
    # the host's admission controller and can overlap weights that already
    # occupy the device. The host still probes each route on demand.
    startup_smoke: bool = True
    # When the language stage targets an OpenAI-compatible endpoint, the model
    # name belongs to that server, not to Ollama. Pulling or blob-verifying it
    # against Ollama is meaningless and hangs on a tag that cannot exist.
    language_api: str = "ollama"
    decision_plane_enabled: bool = True

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
        model = os.environ.get("OMNI_MODEL", "robit/qwen3.8-27b-e03-obliterated-omni:q4km").strip()
        tegra = is_tegra()
        default_context = "262144" if tegra and "-audio-bridge:" in model else "65536"
        return cls(
            repo_root=root,
            runtime_root=runtime,
            model=model,
            language_model=os.environ.get(
                "OMNI_LANGUAGE_MODEL",
                # The logical tag's standard layers are the language model:
                # the combined tag and its core tag resolve to byte-identical
                # base/projector blobs. Ollama still keys a loaded runner by
                # tag *name*, so naming the core tag separately loads a second
                # resident copy of the same weights. A discrete host with room
                # for both keeps the explicit core tag; a Tegra module shares
                # one GPU-visible pool with the OS and the TTS worker, so it
                # defaults to running the language stage on the tag it is
                # already serving.
                (
                    os.environ.get("OMNI_MODEL", "").strip()
                    or "robit/qwen3.8-27b-e03-obliterated-omni:q4km"
                )
                if tegra
                else "robit/qwen3.8-27b-obliterated-e03:27b",
            ).strip(),
            context_tokens=int(
                os.environ.get(
                    "OMNI_COMPREHENSION_CONTEXT_TOKENS",
                    # This is a ceiling, not an eager allocation. The Tegra
                    # launcher derives KV bytes from the selected GGUF and
                    # picks the largest tier that current MemAvailable can
                    # fund while retaining the shared runtime reserve.
                    default_context,
                )
            ),
            comprehension_parallel_slots=max(
                1, int(os.environ.get("OMNI_COMPREHENSION_PARALLEL", "1"))
            ),
            tts_stream_frames=int(os.environ.get("OMNI_TTS_STREAM_FRAMES", "8")),
            portal_token=os.environ.get("OMNI_PORTAL_TOKEN", "").strip(),
            cloudflare=(
                os.environ.get("OMNI_ENABLE_CLOUDFLARED", "1") != "0"
                if cloudflare is None
                else cloudflare
            ),
            keep_cache=os.environ.get("OMNI_KEEP_CACHE", "0") == "1",
            allow_direct_gpu=allow_direct_gpu,
            enable_comprehension=os.environ.get("OMNI_ENABLE_COMPREHENSION", "1").strip().lower()
            not in {"0", "false", "no"},
            startup_smoke=os.environ.get("OMNI_STARTUP_SMOKE", "1").strip().lower()
            not in {"0", "false", "no"},
            language_api=os.environ.get("OMNI_LANGUAGE_API", "ollama").strip().lower(),
            decision_plane_enabled=os.environ.get("OMNI_DECISION_PLANE_ENABLED", "1")
            .strip()
            .lower()
            not in {"0", "false", "no"},
        )


@dataclass
class Child:
    name: str
    process: subprocess.Popen[bytes]
    log: IO[bytes]
    resident_pid: int | None = None


def _connected_tunnel_url(log_path: Path, start_offset: int) -> str:
    """Return only a URL created and connected by the current tunnel process."""

    try:
        with log_path.open("rb") as source:
            source.seek(start_offset)
            text = source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    if "Registered tunnel connection" not in text:
        return ""
    matches = re.findall(r"https://[-a-z0-9]+\.trycloudflare\.com", text)
    return matches[-1] if matches else ""


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
        self.restart_file = self.state_dir / "restart.request"
        self.context_file = self.state_dir / "comprehension-context-tokens"
        self.comprehension_pid_file = self.state_dir / "comprehension-worker.pid"
        self.token_file = self.state_dir / "access-token.txt"
        self.children: list[Child] = []
        self.stop_event = threading.Event()
        self.sidecar_resolution: dict[str, Any] | None = None
        # DaemonConfig is intentionally immutable.  Tunnel availability is a
        # runtime condition, so keep the effective value on the supervisor
        # instead of trying to mutate the frozen configuration during
        # preflight.
        self.cloudflare_enabled = config.cloudflare

    def _write_status(self, **fields: Any) -> None:
        value = {
            "schema": "robit.qwen-omni-daemon.status.v1",
            "pid": os.getpid(),
            "platform": platform.system(),
            "accelerator": accelerator_profile(),
            "model": self.config.model,
            "updated_at": time.time(),
            **fields,
        }
        partial = self.status_file.with_suffix(".tmp")
        partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        try:
            partial.chmod(0o600)
        except OSError:
            pass
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
        missing = [command for command in required if not shutil.which(command)]
        if missing:
            raise DaemonError(f"missing commands: {', '.join(missing)}")
        if self.cloudflare_enabled and not shutil.which("cloudflared"):
            # Publishing is a convenience; the conversation this daemon exists
            # to serve happens over loopback. Losing the tunnel must not cost
            # the microphone.
            print(
                "qwen-omni-daemon: cloudflared is not installed; serving locally only",
                file=sys.stderr,
                flush=True,
            )
            self.cloudflare_enabled = False
        for binary_name in ("llama-server", "llama-tts"):
            binary = _binary(self.config.repo_root, binary_name)
            if not binary.is_file():
                raise DaemonError(
                    f"missing {binary_name}; run the platform bootstrap first: {binary}"
                )
        ports = [
            self.config.tts_port,
            self.config.adapter_port,
            self.config.portal_port,
        ]
        if self.config.decision_plane_enabled and self._laya_python().is_file():
            ports.append(self.config.decision_port)
        if self.config.enable_comprehension:
            # Only required when this supervisor spawns the worker. An
            # externally managed comprehension worker legitimately occupies
            # the port already, and refusing to start because of it means the
            # adapter can never run alongside one.
            ports.insert(0, self.config.comprehension_port)
        for port in ports:
            # A port a previous instance has just given up takes a moment to
            # come back. Failing on the first look is how a restart turned
            # into a permanent outage that needed a human.
            deadline = time.monotonic() + 20
            while not _port_available("127.0.0.1", port):
                if time.monotonic() >= deadline:
                    raise DaemonError(f"required loopback port is already in use: {port}")
                time.sleep(0.5)

    def _reclaim_from_prior_instance(self) -> None:
        """Take the ports back from an older daemon instead of refusing to start.

        Refusing was the wrong instinct for something run as a service. A
        supervisor restarting this daemon while the old one is still shutting
        down -- or an operator who once started it by hand -- left an instance
        holding the loopback ports, and every subsequent start died on "port
        already in use" until someone noticed and killed it manually. The
        daemon is the single owner of those ports, so the newest instance
        wins and says so.
        """

        if not self.pid_file.is_file():
            return
        try:
            prior = int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            self.pid_file.unlink(missing_ok=True)
            return
        if not prior or prior == os.getpid() or not _pid_alive(prior):
            self.pid_file.unlink(missing_ok=True)
            return

        print(
            f"qwen-omni-daemon: replacing previous instance (pid {prior})",
            file=sys.stderr,
            flush=True,
        )
        try:
            os.kill(prior, signal.SIGTERM)
        except OSError:
            self.pid_file.unlink(missing_ok=True)
            return
        # Its children hold the ports, and they are torn down as it exits.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and _pid_alive(prior):
            time.sleep(0.5)
        if _pid_alive(prior):
            try:
                os.kill(prior, signal.SIGKILL)
            except OSError:
                pass
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and _pid_alive(prior):
                time.sleep(0.5)
        self.pid_file.unlink(missing_ok=True)

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
        self._reclaim_from_prior_instance()
        self.stop_file.unlink(missing_ok=True)
        self.restart_file.unlink(missing_ok=True)
        self.comprehension_pid_file.unlink(missing_ok=True)
        self._write_status(state="preflight")
        self._preflight()
        self._ensure_model(self.config.model)
        resolved = resolve_ollama_sidecar(model=self.config.model)
        self.sidecar_resolution = resolved
        bridge_profile = resolved.get("profile") == "trained-audio-bridge"
        if bridge_profile:
            self._write_status(
                state="preflight",
                detail="trained audio bridge uses one local language trunk",
            )
        elif self.config.language_api == "ollama":
            self._ensure_model(self.config.language_model)
            self._verify_shared_base()
        else:
            # The language model lives on another server; Ollama has never
            # heard of it and there are no shared blobs to verify.
            self._write_status(
                state="preflight",
                detail=f"language stage on the {self.config.language_api} backend",
            )
        self._write_status(state="materializing", sidecar_digest=resolved["layer"]["digest"])
        required = [
            self.cache_dir / "tts-model.gguf",
            self.cache_dir / "tts-projector.gguf",
        ]
        if self.config.enable_comprehension and not bridge_profile:
            required[:0] = [
                self.cache_dir / "comprehension-model.gguf",
                self.cache_dir / "comprehension-projector.gguf",
            ]
        if not all(path.is_file() for path in required):
            prepare_ollama_sidecar(
                model=self.config.model,
                output_dir=self.cache_dir,
                overwrite=True,
            )

    def _resolved_sidecar(self) -> dict[str, Any]:
        if self.sidecar_resolution is None:
            self.sidecar_resolution = resolve_ollama_sidecar(model=self.config.model)
        return self.sidecar_resolution

    def _voice_reference(self) -> Path:
        """Resolve the configured default clone reference without exposing it."""

        override = os.environ.get("OMNI_TTS_WARM_SPEAKER_FILE", "").strip()
        if override:
            reference = Path(override).expanduser().resolve()
        else:
            configured = os.environ.get("OMNI_VOICE_PROFILE", "").strip()
            profile_path = (
                Path(configured).expanduser().resolve()
                if configured
                else self.config.repo_root / "portal" / "voice-profile.json"
            )
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise DaemonError(
                    f"could not load the voice-clone profile {profile_path}: {exc}"
                ) from exc
            speaker = str(profile.get("speaker_file") or "").strip()
            presets = profile.get("presets")
            if isinstance(presets, list):
                defaults = [
                    item
                    for item in presets
                    if isinstance(item, dict) and item.get("default") is True
                ]
                if len(defaults) == 1:
                    speaker = str(defaults[0].get("speaker_file") or "").strip()
            if not speaker:
                raise DaemonError("voice-clone readiness requires one default speaker reference")
            reference = Path(speaker).expanduser()
            if not reference.is_absolute():
                reference = profile_path.parent / reference
            reference = reference.resolve()
        if not reference.is_file():
            raise DaemonError(f"voice-clone speaker reference does not exist: {reference}")
        return reference

    def _comprehension_artifacts(self) -> tuple[Path, Path]:
        resolved = self._resolved_sidecar()
        if resolved.get("profile") == "trained-audio-bridge":
            layers = resolved.get("standard_layers") or {}
            try:
                model = Path(layers["language_model"]["path"])
                projector = Path(layers["projector"]["path"])
            except (KeyError, TypeError) as exc:
                raise DaemonError(
                    "trained audio bridge is missing its standard model/projector layers"
                ) from exc
            return model, projector
        return (
            self.cache_dir / "comprehension-model.gguf",
            self.cache_dir / "comprehension-projector.gguf",
        )

    def _language_route(self) -> tuple[str, str, str]:
        resolved = self._resolved_sidecar()
        if resolved.get("profile") == "trained-audio-bridge" and self.config.enable_comprehension:
            return (
                "openai",
                f"http://127.0.0.1:{self.config.comprehension_port}/v1/chat/completions",
                "local-audio-bridge",
            )
        return (
            self.config.language_api,
            os.environ.get("OMNI_LANGUAGE_URL", "http://127.0.0.1:11434"),
            self.config.language_model,
        )

    def _speculative_args(self) -> list[str]:
        configured = os.environ.get("OMNI_SPECULATIVE_TYPE")
        if configured is None:
            configured = (
                "ngram-simple"
                if self._resolved_sidecar().get("profile") == "trained-audio-bridge"
                else ""
            )
        selected = configured.strip()
        return ["--spec-type", selected] if selected else []

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

    def _laya_python(self) -> Path:
        configured = os.environ.get("OMNI_LAYA_PYTHON", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        name = "python.exe" if os.name == "nt" else "python"
        return (
            self.config.repo_root / ".laya-venv" / ("Scripts" if os.name == "nt" else "bin") / name
        )

    def _discard_child(self, child: Child) -> None:
        if child.process.poll() is None:
            if os.name == "nt":
                child.process.terminate()
            else:
                os.killpg(child.process.pid, signal.SIGTERM)
            try:
                child.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    child.process.kill()
                else:
                    os.killpg(child.process.pid, signal.SIGKILL)
                child.process.wait(timeout=5)
        if child in self.children:
            self.children.remove(child)
        child.log.close()

    def _start_decision_plane(self, common: dict[str, str]) -> bool:
        """Start the optional resident optimizer without making it a SPOF."""

        if not self.config.decision_plane_enabled:
            return False
        python = self._laya_python()
        if not python.is_file():
            print(
                "qwen-omni-daemon: Laya runtime is not installed; continuing with "
                "the deliberative fallback (run scripts/bootstrap_laya.sh)",
                file=sys.stderr,
                flush=True,
            )
            return False
        env = {
            **common,
            "OMNI_LAYA_PORT": str(self.config.decision_port),
            "OMNI_DECISION_TRACE_FILE": str(self.config.runtime_root / "decision-traces.jsonl"),
        }
        child = self._spawn(
            "laya",
            [
                python.as_posix(),
                str(self.config.repo_root / "runtime" / "laya_server.py"),
            ],
            env,
        )
        try:
            self._wait_http(
                child,
                f"http://127.0.0.1:{self.config.decision_port}/health",
                1800,
            )
            response = httpx.get(f"http://127.0.0.1:{self.config.decision_port}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            if not isinstance(health, dict) or health.get("ready") is not True:
                raise DaemonError("Laya decision plane did not report ready after warmup")
        except (DaemonError, httpx.HTTPError, ValueError) as exc:
            print(
                "qwen-omni-daemon: Laya unavailable; continuing with the "
                f"deliberative fallback: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._discard_child(child)
            return False
        return True

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

    def _wait_worker_pid(self, child: Child, path: Path, timeout: int = 120) -> int:
        """Read a live CUDA worker pid published by a supervised launcher."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.process.poll() is not None:
                raise DaemonError(f"{child.name} launcher exited before publishing its worker pid")
            try:
                pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = 0
            if pid > 0 and _pid_alive(pid):
                return pid
            time.sleep(0.2)
        raise DaemonError(f"{child.name} did not publish a live worker pid")

    def _active_context_tokens(self) -> int:
        if not is_tegra():
            return self.config.context_tokens
        try:
            selected = int(self.context_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return self.config.context_tokens
        return min(self.config.context_tokens, max(1024, selected))

    def _verify_direct_gpu(self, pid: int, component: str = "comprehension") -> None:
        if platform.system() == "Darwin":
            return  # bootstrap enforces a Metal-enabled llama.cpp build
        if is_tegra():
            # Tegra's integrated GPU has no compute-app accounting; residency is
            # proven by the worker's own handles on the nvgpu/nvmap nodes, which
            # only an initialized CUDA context opens.
            if not Path("/proc/self/fd").is_dir():
                raise DaemonError("procfs is required to prove Tegra CUDA residency")
        elif not (shutil.which("nvidia-smi") and compute_apps_supported()):
            raise DaemonError("nvidia-smi is required to prove direct CUDA residency")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process_is_gpu_resident(pid):
                return
            time.sleep(1)
        raise DaemonError(
            f"{component} pid {pid} did not become CUDA-resident (evidence: {residency_backend()})"
        )

    def _verify_co_resident_stack(self, comprehension: Child | None, tts: Child) -> dict[str, Any]:
        """Prove cloned TTS did not evict the comprehension worker."""

        if comprehension is not None:
            if comprehension.process.poll() is not None:
                raise DaemonError(
                    "comprehension exited during cloned TTS synthesis; the selected "
                    "bundle does not satisfy the co-residency gate"
                )
            self._verify_direct_gpu(
                comprehension.resident_pid or comprehension.process.pid,
                "comprehension",
            )
        if tts.process.poll() is not None:
            raise DaemonError("TTS wrapper exited during startup smoke")
        try:
            response = httpx.get(f"http://127.0.0.1:{self.config.tts_port}/healthz", timeout=10)
            response.raise_for_status()
            health = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DaemonError(f"could not verify resident cloned TTS: {exc}") from exc
        if not isinstance(health, dict):
            raise DaemonError("TTS health returned no object")
        tts_pid = health.get("persistent_pid")
        if (
            health.get("persistent_ready") is not True
            or health.get("speaker_reference_configured") is not True
            or health.get("speaker_reference_active") is not True
            or not isinstance(tts_pid, int)
            or tts_pid <= 0
            or not _pid_alive(tts_pid)
        ):
            raise DaemonError(
                "cloned TTS did not retain one live persistent speaker-profile worker"
            )
        if platform.system() != "Darwin":
            self._verify_direct_gpu(tts_pid, "tts")
        return {
            "comprehension_pid": (
                (comprehension.resident_pid or comprehension.process.pid)
                if comprehension is not None
                else None
            ),
            "tts_pid": tts_pid,
            "gpu_residency": residency_backend(),
            "speaker_reference_active": True,
        }

    def start_children(self) -> str:
        python = sys.executable
        common = os.environ.copy()
        bridge_profile = self._resolved_sidecar().get("profile") == "trained-audio-bridge"
        common["PYTHONUNBUFFERED"] = "1"
        decision_ready = self._start_decision_plane(common)
        common["OMNI_DECISION_PLANE_ENABLED"] = "1" if decision_ready else "0"
        common["OMNI_DECISION_PLANE_URL"] = f"http://127.0.0.1:{self.config.decision_port}"
        common["OMNI_DECISION_TRACE_FILE"] = str(self.config.runtime_root / "decision-traces.jsonl")
        comprehension_model, comprehension_projector = self._comprehension_artifacts()
        comprehension: Child | None = None
        if self.config.enable_comprehension:
            server_command = [
                str(_binary(self.config.repo_root, "llama-server")),
                "-m",
                str(comprehension_model),
                "--mmproj",
                str(comprehension_projector),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.config.comprehension_port),
                "--jinja",
                "-ngl",
                "99",
                "-c",
                "{context}" if is_tegra() else str(self.config.context_tokens),
                "--parallel",
                (
                    "{parallel}"
                    if is_tegra()
                    else str(self.config.comprehension_parallel_slots)
                ),
                *self._speculative_args(),
            ]
            command = server_command
            if is_tegra():
                command = [
                    python,
                    str(self.config.repo_root / "runtime" / "comprehension_launcher.py"),
                    "--state-file",
                    str(self.context_file),
                    "--calibration-file",
                    str(self.state_dir / "comprehension-memory.json"),
                    "--max-context",
                    str(self.config.context_tokens),
                    "--parallel-slots",
                    str(self.config.comprehension_parallel_slots),
                    "--health-url",
                    f"http://127.0.0.1:{self.config.comprehension_port}/health",
                    "--child-pid-file",
                    str(self.comprehension_pid_file),
                    "--",
                    *server_command,
                ]
            comprehension = self._spawn(
                "comprehension",
                command,
                common,
            )
            self._wait_http(
                comprehension,
                f"http://127.0.0.1:{self.config.comprehension_port}/health",
                1200,
            )
            if is_tegra():
                comprehension.resident_pid = self._wait_worker_pid(
                    comprehension, self.comprehension_pid_file
                )
            self._verify_direct_gpu(comprehension.resident_pid or comprehension.process.pid)
        else:
            self._write_status(state="starting", detail="comprehension disabled")

        tts_env = {
            **common,
            "LLAMA_TTS_BIN": str(_binary(self.config.repo_root, "llama-tts")),
            "OMNI_TTS_MODEL_GGUF": str(self.cache_dir / "tts-model.gguf"),
            "OMNI_TTS_PROJECTOR_GGUF": str(self.cache_dir / "tts-projector.gguf"),
            "OMNI_COMPONENT_CACHE": str(self.cache_dir),
            "OMNI_TTS_GPU_LAYERS": "-1",
            "OMNI_TTS_REQUIRE_GPU": "0" if platform.system() == "Darwin" else "1",
            "OMNI_TTS_PERSISTENT": "1",
            "OMNI_TTS_WARM_SPEAKER_FILE": str(self._voice_reference()),
            "OMNI_TTS_ACTIVE_PID_FILE": str(self.state_dir / "tts-worker.pid"),
            "OMNI_TTS_STREAM_FRAMES": str(self.config.tts_stream_frames),
            "OMNI_TTS_HOST": "127.0.0.1",
            "OMNI_TTS_PORT": str(self.config.tts_port),
        }
        tts = self._spawn(
            "tts",
            [python, str(self.config.repo_root / "runtime" / "tts_server.py")],
            tts_env,
        )
        self._wait_http(tts, f"http://127.0.0.1:{self.config.tts_port}/healthz", 60)

        language_api, language_url, language_model = self._language_route()
        # The direct llama.cpp route intentionally uses the runtime alias
        # ``local-audio-bridge``. Template policy belongs to the logical model
        # profile, not that transport alias.
        default_disable_language_thinking = _language_disable_thinking_default(
            self.config.model
        )
        adapter_env = {
            **common,
            # An empty URL is how the adapter reports comprehension as
            # unconfigured rather than pretending a dead port is a worker.
            # An explicitly set URL wins even when this supervisor does not
            # spawn the worker: that is the externally-managed case, where the
            # host starts and stops comprehension around demand itself.
            "OMNI_COMPREHENSION_URL": (
                f"http://127.0.0.1:{self.config.comprehension_port}/v1/chat/completions"
                if self.config.enable_comprehension
                else os.environ.get("OMNI_COMPREHENSION_URL", "")
            ),
            "OMNI_COMPREHENSION_MODEL": language_model,
            "OMNI_COMPREHENSION_CONTEXT_TOKENS": str(self.config.context_tokens),
            "OMNI_COMPREHENSION_CONTEXT_FILE": (str(self.context_file) if is_tegra() else ""),
            "OMNI_COMPREHENSION_DISABLE_THINKING": "1" if bridge_profile else "0",
            "OMNI_COMPREHENSION_REPEAT_PENALTY": "1.1" if bridge_profile else "1.0",
            "OMNI_LANGUAGE_API": language_api,
            "OMNI_LANGUAGE_URL": language_url,
            "OMNI_LANGUAGE_MODEL": language_model,
            "OMNI_LANGUAGE_DISABLE_THINKING": os.environ.get(
                "OMNI_LANGUAGE_DISABLE_THINKING",
                "1" if default_disable_language_thinking else "0",
            ),
            "OMNI_TTS_URL": f"http://127.0.0.1:{self.config.tts_port}/synthesize",
            "OMNI_TTS_STREAM_FRAMES": str(self.config.tts_stream_frames),
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
            "OMNI_COMPREHENSION_HEALTH_URL": (
                f"http://127.0.0.1:{self.config.comprehension_port}/health"
                if self.config.enable_comprehension
                else os.environ.get("OMNI_COMPREHENSION_HEALTH_URL", "")
            ),
            "OMNI_TTS_HEALTH_URL": f"http://127.0.0.1:{self.config.tts_port}/healthz",
            "OMNI_PORTAL_SESSION_LOG_DIR": str(self.session_log_dir),
            "OMNI_PORTAL_HOST": "127.0.0.1",
            "OMNI_PORTAL_PORT": str(self.config.portal_port),
        }
        portal = self._spawn(
            "portal",
            [python, str(self.config.repo_root / "portal" / "app.py")],
            portal_env,
        )
        self._wait_http(portal, f"http://127.0.0.1:{self.config.portal_port}/healthz", 60)

        if self.config.startup_smoke:
            smoke = [
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
            ]
            if self.config.enable_comprehension:
                # A text+TTS check can pass while the audio projector/ASR path
                # is broken. The desktop harness is useless in exactly that
                # state, so readiness includes a real tracked speech fixture
                # and its direct ASR→TTS route.
                smoke.extend(
                    [
                        "--audio",
                        str(self.config.repo_root / "portal" / "voices" / "default_voice.wav"),
                        "--voice-clone",
                    ]
                )
            self._command(smoke, timeout=1200)
            resident_stack = self._verify_co_resident_stack(comprehension, tts)
        else:
            resident_stack = None
        self._write_status(
            state="smoke-passed" if self.config.startup_smoke else "ready",
            comprehension=self.config.enable_comprehension,
            startup_smoke=self.config.startup_smoke,
            decision_plane=decision_ready,
            co_resident_stack=resident_stack,
        )

        public = f"http://127.0.0.1:{self.config.portal_port}"
        if self.cloudflare_enabled:
            tunnel_log = self.log_dir / "cloudflared.log"
            try:
                tunnel_log_offset = tunnel_log.stat().st_size
            except OSError:
                tunnel_log_offset = 0
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
            deadline = time.monotonic() + 120
            published = ""
            while time.monotonic() < deadline:
                if tunnel.process.poll() is not None:
                    break
                published = _connected_tunnel_url(tunnel_log, tunnel_log_offset)
                if published:
                    break
                time.sleep(1)
            if published:
                public = published
            else:
                # A machine with no internet still has a microphone, a camera
                # and speakers, and everything that matters here is loopback.
                # Refusing to start because a tunnel could not be published
                # would take the local conversation down over a remote
                # convenience.
                print(
                    "qwen-omni-daemon: no public tunnel (offline or cloudflared "
                    f"unavailable); serving locally at {public}",
                    file=sys.stderr,
                    flush=True,
                )
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
        self.restart_file.unlink(missing_ok=True)
        self.comprehension_pid_file.unlink(missing_ok=True)
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
        restart_requested = False
        try:
            access_url = self.start_children()
            self._write_status(
                state="ready",
                access_url=access_url,
                startup_smoke=self.config.startup_smoke,
                comprehension_context_tokens=self._active_context_tokens(),
                comprehension_context_ceiling=self.config.context_tokens,
                comprehension_parallel_slots=self.config.comprehension_parallel_slots,
                children=[
                    {
                        "name": child.name,
                        "pid": child.resident_pid or child.process.pid,
                        **(
                            {"supervisor_pid": child.process.pid}
                            if child.resident_pid is not None
                            else {}
                        ),
                    }
                    for child in self.children
                ],
            )
            if sys.stdout.isatty() or os.environ.get("OMNI_PRINT_ACCESS_URL") == "1":
                print(access_url, flush=True)
            else:
                public = access_url.split("/#access=", 1)[0]
                print(
                    f"qwen-omni-daemon: portal ready at {public}; protected access URL "
                    f"saved in {self.status_file}",
                    flush=True,
                )
            while not self.stop_event.wait(1):
                if self.restart_file.exists():
                    restart_requested = True
                    self._write_status(state="restarting")
                    self.request_stop()
                    continue
                if self.stop_file.exists():
                    self.request_stop()
                    continue
                for child in list(self.children):
                    if child.process.poll() is not None:
                        if child.name == "laya":
                            print(
                                "qwen-omni-daemon: Laya exited; continuing with "
                                "the deliberative fallback",
                                file=sys.stderr,
                                flush=True,
                            )
                            self.children.remove(child)
                            child.log.close()
                            continue
                        raise DaemonError(
                            f"{child.name} exited unexpectedly; inspect {self.log_dir}"
                        )
            return 75 if restart_requested else 0
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
