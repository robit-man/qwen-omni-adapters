"""Small systemd residency boundary for constrained unified-memory hosts."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_UNIT = re.compile(r"[A-Za-z0-9_.@:-]+\.service")


@dataclass
class SpeechResidency:
    """Stop one model service for speech, then start it and await readiness."""

    unit: str
    health_url: str
    stop_timeout_s: float = 120.0
    ready_timeout_s: float = 900.0
    _restore: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not _UNIT.fullmatch(self.unit):
            raise ValueError(f"invalid systemd service name: {self.unit!r}")

    def _systemctl(self, action: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["systemctl", "--user", action, self.unit],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"systemctl {action} {self.unit} failed: {detail}")
        return completed

    def prepare_speech(self) -> None:
        """Synchronously free the weights that cannot coexist with TTS."""

        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", self.unit],
            timeout=10,
        )
        self._restore = active.returncode == 0
        if not self._restore:
            logger.info("speech headroom already available; %s is inactive", self.unit)
            return
        logger.info("stopping %s to make room for speech", self.unit)
        self._systemctl("stop", timeout=self.stop_timeout_s)

    def restore(self) -> None:
        """Restore the evicted worker and wait until it can hear the next turn."""

        if not self._restore:
            return
        self._restore = False
        logger.info("restoring %s after speech", self.unit)
        self._systemctl("start", timeout=self.stop_timeout_s)

        deadline = time.monotonic() + self.ready_timeout_s
        last = "not ready"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(self.health_url, timeout=3.0)
                if response.status_code == 200:
                    logger.info("%s is ready for the next turn", self.unit)
                    return
                last = f"HTTP {response.status_code}"
            except Exception as error:  # noqa: BLE001 - readiness retry loop
                last = f"{type(error).__name__}: {error}"
            time.sleep(1.0)
        raise TimeoutError(f"{self.unit} did not become ready ({last})")
