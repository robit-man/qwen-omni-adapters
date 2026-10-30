"""Bounded idle TTS weight retention; generation state stays worker-owned."""
from __future__ import annotations

import threading
import time

from qwen_omni_adapters.accelerator import gpu_facts, is_tegra


class IdleTTSResidency:
    """Release optional idle TTS weights without touching an active generation.

    The caller holds the request lock for begin/finish/shed. The monitor acquires
    it nonblocking, so cancellation still belongs to the streaming generator.
    Discrete CUDA admission remains the existing scoped broker's responsibility.
    """

    def __init__(self, worker, lock, *, seconds: float, minimum_available_mib: int):
        if not 0 <= seconds <= 300 or minimum_available_mib < 0:
            raise ValueError("TTS idle retention must be 0-300 seconds with a nonnegative memory floor")
        self.worker = worker
        self.lock = lock
        self.seconds = seconds
        self.minimum_available_mib = minimum_available_mib
        self.deadline: float | None = None
        self.last_transition = "empty"
        self.stopped = threading.Event()
        self.thread = None
        if seconds and worker is not None:
            self.thread = threading.Thread(target=self._monitor, name="tts-idle-residency", daemon=True)
            self.thread.start()

    def _headroom_allows_retention(self) -> bool:
        if not is_tegra():
            return True
        try:
            facts = gpu_facts()
        except (OSError, ValueError, TypeError):
            return False
        if not facts:
            return False
        total, used = facts[0].get("vram_total_mib"), facts[0].get("vram_used_mib")
        return (isinstance(total, (int, float)) and isinstance(used, (int, float))
                and total - used >= self.minimum_available_mib)

    def shed(self, reason="explicit"):
        self.deadline = None
        self.last_transition = "released:" + reason
        if self.worker is not None:
            self.worker.close()

    def _release_if_due(self):
        if self.deadline is not None:
            if time.monotonic() >= self.deadline:
                self.shed("expired")
            elif not self._headroom_allows_retention():
                self.shed("headroom")

    def begin(self):
        self._release_if_due()
        self.deadline = None
        self.last_transition = "active"

    def finish(self, successful: bool):
        if not successful:
            self.shed("incomplete")
        elif not self.seconds:
            self.shed("request")
        elif self.worker is None or not self.worker.ready:
            self.shed("not-ready")
        elif not self._headroom_allows_retention():
            self.shed("headroom")
        else:
            self.deadline = time.monotonic() + self.seconds
            self.last_transition = "retained"

    def _monitor(self):
        while not self.stopped.wait(.25):
            if not self.lock.acquire(blocking=False):
                continue
            try:
                self._release_if_due()
            finally:
                self.lock.release()

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
