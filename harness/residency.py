"""Small systemd residency boundary for constrained unified-memory hosts."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
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
    # A foreground turn must never hold the conversation worker for fifteen
    # minutes. Systemd and the background restorer keep healing indefinitely;
    # this bound applies only to one caller waiting for usable weights.
    ready_timeout_s: float = 120.0
    _restore: bool = field(default=False, init=False)
    _restarted_at: float = field(default=0.0, init=False)
    _start_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _restore_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )

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
        # Comprehension is the desired steady state even if it was already
        # failed when speech began. The old code remembered False here, then
        # skipped restoration entirely after TTS and left the system dead.
        self._restore = True
        if active.returncode != 0:
            logger.info("speech headroom already available; %s is inactive", self.unit)
            return
        logger.info("stopping %s to make room for speech", self.unit)
        self._systemctl("stop", timeout=self.stop_timeout_s)

    def _is_active(self) -> bool:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", self.unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return active.returncode == 0

    def ready_now(self) -> bool:
        """Whether the evicted model is presently ready, without waiting."""

        try:
            return httpx.get(self.health_url, timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _ensure_started(self) -> bool:
        """Make one serialized start attempt and confirm it survived startup."""

        with self._start_lock:
            if self._is_active():
                return True
            try:
                self._systemctl("start", timeout=self.stop_timeout_s)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                logger.debug("could not yet restore %s: %s", self.unit, error)
                return False
            # The adaptive launcher can accept systemd's start and then refuse
            # once it measures memory. Give that short check time to finish so
            # an active result means the model is actually loading.
            time.sleep(0.3)
            return self._is_active()

    def _restore_until_started(self) -> None:
        """Retry transient memory refusals without blocking playback cleanup."""

        # CUDA/unified-memory accounting can lag slightly behind the TTS child
        # exiting. Starting in that instant is the failure this loop repairs.
        time.sleep(1.0)
        attempts = 0
        while True:
            if self._ensure_started():
                self._restarted_at = time.monotonic()
                return
            attempts += 1
            if attempts % 30 == 0:
                logger.warning(
                    "still restoring %s after %ds; recovery remains active",
                    self.unit,
                    attempts * 2,
                )
            time.sleep(2.0)

    def restore(self) -> None:
        """Start the evicted worker again, and do not wait for it.

        Waiting here was the whole cost. The worker takes about half a minute
        to read its weights back off disk, and doing that before the turn was
        allowed to finish meant nothing could be answered for that whole time
        -- while the speaker was usually still listening to the reply, or
        thinking about what to say next.

        Loading it in the background spends that half minute against time the
        conversation was going to take anyway. Nothing is lost by not waiting:
        the microphone is read throughout, and the next turn asks for
        readiness before it needs the weights.
        """

        if not self._restore:
            return
        self._restore = False
        logger.info("restoring %s in the background after speech", self.unit)
        if self._restore_thread is not None and self._restore_thread.is_alive():
            return
        self._restore_thread = threading.Thread(
            target=self._restore_until_started,
            name="omni-comprehension-restore",
            daemon=True,
        )
        self._restore_thread.start()

    def await_ready(self) -> None:
        """Block until the worker can hear again, called before it is needed.

        By the time a turn actually wants comprehension, the reload started
        when the last reply finished speaking has usually completed, and this
        returns immediately.
        """

        deadline = time.monotonic() + self.ready_timeout_s
        last = "not ready"
        waited_from = time.monotonic()
        next_start_attempt = waited_from
        while time.monotonic() < deadline:
            try:
                response = httpx.get(self.health_url, timeout=3.0)
                if response.status_code == 200:
                    waited = time.monotonic() - waited_from
                    if waited > 1.0:
                        logger.info(
                            "waited %.0fs for %s; the rest of its reload ran "
                            "while the reply was playing",
                            waited,
                            self.unit,
                        )
                    return
                last = f"HTTP {response.status_code}"
            except Exception as error:  # noqa: BLE001 - readiness retry loop
                last = f"{type(error).__name__}: {error}"
            now = time.monotonic()
            if now >= next_start_attempt and not self._is_active():
                self._ensure_started()
                next_start_attempt = now + 2.0
            time.sleep(0.5)
        raise TimeoutError(f"{self.unit} did not become ready ({last})")
