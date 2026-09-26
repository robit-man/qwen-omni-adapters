"""Exact-tag model lifecycle used by the desktop indicator."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from qwen_omni_adapters.model_catalog import MANAGED_MODELS, MODEL_BY_TAG

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PERCENT = re.compile(r"(?<!\d)(\d{1,3})\s*%")


class ModelActionError(RuntimeError):
    pass


def _table_tags(output: str) -> set[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return set()
    return {line.split()[0] for line in lines[1:] if line.split()}


def _replace_env(path: Path, values: dict[str, str], remove: set[str]) -> None:
    retained: list[str] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            key = raw.split("=", 1)[0].strip() if "=" in raw else ""
            if key in values or key in remove:
                continue
            retained.append(raw)
    retained.extend(f"{key}={value}" for key, value in values.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".env.models.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write("\n".join(retained).rstrip() + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class IndicatorModelManager:
    """Cache model state and serialize background Ollama operations."""

    def __init__(
        self,
        repo_root: Path,
        active_model: str,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.active_model = active_model
        self.ollama_url = ollama_url.rstrip("/")
        self._lock = threading.Lock()
        self._operations: dict[str, dict[str, Any]] = {}
        self._cached_inventory: tuple[float, set[str], set[str]] = (0.0, set(), set())
        self._inventory_refreshing = False
        self._schedule_inventory_refresh()

    @staticmethod
    def _require_tag(tag: str):
        model = MODEL_BY_TAG.get(tag)
        if model is None:
            raise ModelActionError("model is not in the managed allowlist")
        return model

    def _inventory(self) -> tuple[set[str], set[str]]:
        with self._lock:
            cached_at, installed, loaded = self._cached_inventory
        if time.monotonic() - cached_at >= 2.0:
            self._schedule_inventory_refresh()
        return set(installed), set(loaded)

    def _schedule_inventory_refresh(self) -> None:
        with self._lock:
            if self._inventory_refreshing:
                return
            self._inventory_refreshing = True

        def refresh() -> None:
            installed: set[str] = set()
            loaded: set[str] = set()
            try:
                listed = subprocess.run(
                    ["ollama", "list"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                running = subprocess.run(
                    ["ollama", "ps"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if listed.returncode == 0:
                    installed = _table_tags(listed.stdout)
                if running.returncode == 0:
                    loaded = _table_tags(running.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
            with self._lock:
                self._cached_inventory = (time.monotonic(), installed, loaded)
                self._inventory_refreshing = False

        threading.Thread(
            target=refresh,
            name="omni-model-inventory",
            daemon=True,
        ).start()

    def _invalidate(self) -> None:
        with self._lock:
            self._cached_inventory = (0.0, set(), set())
        self._schedule_inventory_refresh()

    def views(self) -> list[dict[str, Any]]:
        installed, ollama_loaded = self._inventory()
        with self._lock:
            operations = {key: dict(value) for key, value in self._operations.items()}
            active = self.active_model
            inventory_ready = self._cached_inventory[0] > 0
            busy = any(
                operation.get("state") == "running" for operation in self._operations.values()
            )
        service_loaded = ""
        service_context = 0
        service_context_ceiling = 0
        try:
            daemon = json.loads(
                (self.repo_root / "runtime-data/state/daemon-status.json").read_text(
                    encoding="utf-8"
                )
            )
            pid = int(daemon.get("pid") or 0)
            if daemon.get("state") == "ready" and pid > 0:
                os.kill(pid, 0)
                service_loaded = str(daemon.get("model") or "")
                service_context = int(daemon.get("comprehension_context_tokens") or 0)
                service_context_ceiling = int(daemon.get("comprehension_context_ceiling") or 0)
        except (OSError, TypeError, ValueError):
            service_loaded = ""
            service_context = 0
            service_context_ceiling = 0
        return [
            {
                **model.to_dict(),
                "installed": model.tag in installed,
                "loaded": model.tag in ollama_loaded or model.tag == service_loaded,
                "ollama_loaded": model.tag in ollama_loaded,
                "service_loaded": model.tag == service_loaded,
                "context_tokens": service_context if model.tag == service_loaded else 0,
                "context_ceiling": (service_context_ceiling if model.tag == service_loaded else 0),
                "active": model.tag == active,
                "inventory_ready": inventory_ready,
                "busy": busy,
                "operation": operations.get(model.tag),
            }
            for model in MANAGED_MODELS
        ]

    def _set_operation(self, tag: str, **fields: Any) -> None:
        with self._lock:
            self._operations[tag] = {"updated_at": time.time(), **fields}

    def _finish_operation(self, tag: str, *, ok: bool, detail: str) -> None:
        self._set_operation(tag, state="complete" if ok else "failed", detail=detail[:160])
        self._invalidate()

    def _background(self, tag: str, action: str, target) -> tuple[bool, str]:
        self._require_tag(tag)
        with self._lock:
            current = self._operations.get(tag) or {}
            if current.get("state") == "running":
                return False, f"{current.get('action', 'operation')} already running"
            self._operations[tag] = {
                "state": "running",
                "action": action,
                "progress": None,
                "detail": f"{action.title()} started",
                "updated_at": time.time(),
            }

        def run() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - retain bounded UI diagnostic
                self._finish_operation(tag, ok=False, detail=str(exc))

        threading.Thread(target=run, name=f"omni-model-{action}", daemon=True).start()
        return True, f"{action.title()} started"

    def download(self, tag: str) -> tuple[bool, str]:
        self._require_tag(tag)
        installed, _loaded = self._inventory()
        if tag in installed:
            return False, "Model is already downloaded"

        def pull() -> None:
            process = subprocess.Popen(
                ["ollama", "pull", tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            tail = ""
            if process.stdout is not None:
                for raw in process.stdout:
                    line = ANSI.sub("", raw).strip()
                    if line:
                        tail = line[-160:]
                    matches = PERCENT.findall(line)
                    progress = min(100, int(matches[-1])) if matches else None
                    self._set_operation(
                        tag,
                        state="running",
                        action="download",
                        progress=progress,
                        detail=tail or "Downloading",
                    )
            returncode = process.wait()
            if returncode != 0:
                raise ModelActionError(tail or f"ollama pull exited {returncode}")
            self._finish_operation(tag, ok=True, detail="Download complete")

        return self._background(tag, "download", pull)

    def load(self, tag: str) -> tuple[bool, str]:
        self._require_tag(tag)
        with self._lock:
            if tag == self.active_model:
                return False, "Active model is already loaded by the daemon"
        installed, _loaded = self._inventory()
        if tag not in installed:
            return False, "Download the model before loading it"

        def load_model() -> None:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": tag,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 0},
                },
                timeout=1800,
            )
            if response.status_code >= 400:
                raise ModelActionError(
                    f"Ollama load returned HTTP {response.status_code}: {response.text[-120:]}"
                )
            self._finish_operation(tag, ok=True, detail="Model loaded")

        return self._background(tag, "load", load_model)

    def unload(self, tag: str) -> tuple[bool, str]:
        self._require_tag(tag)
        installed, loaded = self._inventory()
        if tag not in installed:
            return False, "Model is not downloaded"
        if tag not in loaded:
            return False, "Model is not loaded in Ollama"

        def stop_model() -> None:
            completed = subprocess.run(
                ["ollama", "stop", tag],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise ModelActionError((completed.stderr or completed.stdout)[-160:])
            self._finish_operation(tag, ok=True, detail="Model unloaded")

        return self._background(tag, "unload", stop_model)

    def delete(self, tag: str) -> tuple[bool, str]:
        self._require_tag(tag)
        installed, _loaded = self._inventory()
        if tag not in installed:
            return False, "Model is not downloaded"
        with self._lock:
            if tag == self.active_model:
                return False, "Activate another model before deleting this one"

        def remove_model() -> None:
            subprocess.run(
                ["ollama", "stop", tag],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            completed = subprocess.run(
                ["ollama", "rm", tag],
                check=False,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if completed.returncode != 0:
                raise ModelActionError((completed.stderr or completed.stdout)[-160:])
            self._finish_operation(tag, ok=True, detail="Model deleted")

        return self._background(tag, "delete", remove_model)

    def activate(self, tag: str) -> tuple[bool, str]:
        model = self._require_tag(tag)
        installed, _loaded = self._inventory()
        if tag not in installed:
            return False, "Download the model before activating it"
        values = {
            "OMNI_PROFILE": model.key,
            "OMNI_MODEL": model.tag,
            "OMNI_LANGUAGE_MODEL": model.tag,
            "OMNI_LANGUAGE_API": "ollama",
            "OMNI_ENABLE_COMPREHENSION": "1",
            "OMNI_COMPREHENSION_CONTEXT_TOKENS": str(model.max_context_tokens),
            "OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS": str(model.max_context_tokens),
            "OMNI_STARTUP_SMOKE": "0",
            "OMNI_TTS_PERSISTENT": "1",
        }
        _replace_env(
            self.repo_root / ".env",
            values,
            {
                "OMNI_CALL_SPEECH_EVICT_UNIT",
                "OMNI_CALL_COMPREHENSION_HEALTH",
                "OMNI_COMPREHENSION_URL",
                "OMNI_LANGUAGE_URL",
                "OMNI_SPECULATIVE_TYPE",
            },
        )
        restart = self.repo_root / "runtime-data" / "state" / "restart.request"
        restart.parent.mkdir(parents=True, exist_ok=True)
        restart.write_text(
            json.dumps({"model": model.tag, "requested_at": time.time()}) + "\n",
            encoding="utf-8",
        )
        restart.chmod(0o600)
        with self._lock:
            self.active_model = tag
        return True, "Activating model; services will reconnect"

    def action(self, tag: str, action: str) -> tuple[bool, str]:
        actions = {
            "download": self.download,
            "activate": self.activate,
            "load": self.load,
            "unload": self.unload,
            "delete": self.delete,
        }
        selected = actions.get(action)
        if selected is None:
            return False, "Unsupported model action"
        with self._lock:
            if any(operation.get("state") == "running" for operation in self._operations.values()):
                return False, "Another model operation is already running"
        try:
            return selected(tag)
        except (ModelActionError, OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)[:160]
