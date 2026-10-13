"""Persistent pause of the managed voice runtime, without unloading other apps."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

UNIT = "qwen-omni-adapters.service"


class VoiceServiceControl:
    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root.resolve()
        self.marker = self.root / "runtime-data/state/voice-service.paused"

    def paused(self) -> bool:
        return self.marker.exists()

    def _systemctl(self, action: str) -> None:
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        result = subprocess.run(
            [*prefix, "systemctl", action, UNIT],
            capture_output=True, text=True, timeout=150,
        )
        if result.returncode:
            raise RuntimeError(f"Could not {action} the voice service")

    def _save_pause(self) -> None:
        with self.marker.open("w", encoding="utf-8") as target:
            target.write("Paused by operator; resume with qwen-omni service resume.\n")
            target.flush()
            os.fsync(target.fileno())
        self.marker.chmod(0o600)
        self._sync_directory()

    def _sync_directory(self) -> None:
        descriptor = os.open(self.marker.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _doctor(self) -> None:
        selected = {}
        env_file = self.root / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() in {"OMNI_MODEL", "OMNI_LANGUAGE_MODEL"}:
                    selected[key.strip()] = value.strip().strip("\"'")
        command = [str(self.root / ".venv/bin/qwen-omni"), "doctor", "--deployment"]
        for name, option in [("OMNI_MODEL", "--model"), ("OMNI_LANGUAGE_MODEL", "--language-model")]:
            value = selected.get(name) or os.environ.get(name)
            if value:
                command.extend([option, value])
        result = subprocess.run(command, cwd=self.root, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError("Deployment doctor failed; voice remains paused")

    def set_paused(self, paused: bool) -> tuple[bool, str]:
        if platform.system() != "Linux":
            return False, "Managed service pause is available on Linux"
        import fcntl

        self.marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (self.marker.parent / "voice-service-control.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if paused:
                    # Persist first: a concurrent auto-update cannot resurrect it.
                    self._save_pause()
                    self._systemctl("stop")
                    return True, "Voice paused"
                if not self.paused():
                    return True, "Voice is already resumed"
                self._doctor()
                self.marker.unlink()
                self._sync_directory()
                try:
                    self._systemctl("start")
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    self._save_pause()
                    raise
                return True, "Voice resuming"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            return False, str(error)
