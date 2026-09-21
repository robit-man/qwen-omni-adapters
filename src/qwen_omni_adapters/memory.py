"""Host-wide memory admission and emergency cancellation for runtime work.

The constrained Jetson profile shares one physical pool between CPU and GPU.
Checking only the process that is about to run, or special-casing one tool such
as Chromium, misses the actual failure mode: any inference or tool can consume
the last available pages and invoke the kernel OOM killer.  This module keeps a
single policy that callers apply at every task boundary and, where possible,
while the operation is running.

Policy is enabled automatically on Tegra and can be enabled or disabled
explicitly elsewhere with ``OMNI_MEMORY_GOVERNOR``.  Tests and non-runtime
library users may inject a sampler, so behavior never depends on the machine
running the test suite.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .accelerator import is_tegra

GIB_IN_KIB = 1024 * 1024


def release_unused_process_memory() -> bool:
    """Return unreachable native allocations to the host when supported.

    Large graph transforms can free Python and PyTorch objects while glibc
    retains their arenas in the process. On a unified-memory device that
    retained RSS is unavailable to every CPU and GPU component even though no
    live object can use it. Collection is portable; ``malloc_trim`` is a
    best-effort Linux/glibc optimization and never changes admission policy.
    """

    gc.collect()
    if platform.system() != "Linux":
        return False
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False


class MemoryPressure(RuntimeError):
    """A retryable scheduling condition, never task evidence."""

    def __init__(self, label: str, available_gib: float, required_gib: float):
        super().__init__(
            f"{label} deferred for memory headroom "
            f"({available_gib:.2f} GiB available, {required_gib:.2f} GiB required)"
        )
        self.label = label
        self.available_gib = available_gib
        self.required_gib = required_gib


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    """Return Linux MemAvailable in GiB, or infinity off Linux.

    A missing counter must not turn a portable deployment into a permanent
    resource-pressure failure. Explicitly enabled non-Linux deployments can
    inject a platform-native sampler.
    """

    if platform.system() != "Linux":
        return float("inf")
    try:
        lines = meminfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return float("inf")
    for line in lines:
        key, _, value = line.partition(":")
        if key != "MemAvailable":
            continue
        try:
            return float(value.strip().split()[0]) / GIB_IN_KIB
        except (IndexError, ValueError):
            break
    return float("inf")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MemoryPolicy:
    """One generic policy shared by inference and every task/tool type."""

    enabled: bool
    soft_floor_gib: float = 3.0
    hard_floor_gib: float = 2.0
    operation_reserve_gib: float = 1.0
    poll_interval_s: float = 0.1
    wait_initial_s: float = 2.0
    wait_max_s: float = 60.0

    @classmethod
    def from_environment(cls) -> MemoryPolicy:
        enabled = _env_bool("OMNI_MEMORY_GOVERNOR", is_tegra())
        soft = max(0.0, _env_float("OMNI_MEMORY_SOFT_FLOOR_GIB", 3.0))
        hard = max(0.0, _env_float("OMNI_MEMORY_HARD_FLOOR_GIB", 2.0))
        if hard > soft:
            hard = soft
        return cls(
            enabled=enabled,
            soft_floor_gib=soft,
            hard_floor_gib=hard,
            operation_reserve_gib=max(
                0.0, _env_float("OMNI_MEMORY_OPERATION_RESERVE_GIB", 1.0)
            ),
            poll_interval_s=max(
                0.02, _env_float("OMNI_MEMORY_POLL_INTERVAL_S", 0.1)
            ),
            wait_initial_s=max(
                0.1, _env_float("OMNI_MEMORY_WAIT_INITIAL_S", 2.0)
            ),
            wait_max_s=max(1.0, _env_float("OMNI_MEMORY_WAIT_MAX_S", 60.0)),
        )


class MemoryGovernor:
    """Admit work with reserve and cancel it before the kernel OOM boundary."""

    def __init__(
        self,
        policy: MemoryPolicy | None = None,
        *,
        sampler: Callable[[], float] = available_memory_gib,
    ) -> None:
        self.policy = policy or MemoryPolicy.from_environment()
        self._sampler = sampler

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    def available_gib(self) -> float:
        try:
            return max(0.0, float(self._sampler()))
        except (OSError, TypeError, ValueError):
            return float("inf")

    def required_gib(self, reserve_gib: float | None = None) -> float:
        reserve = (
            self.policy.operation_reserve_gib
            if reserve_gib is None
            else max(0.0, float(reserve_gib))
        )
        # The soft floor is already the normal operating cushion. Treat the
        # operation reserve as distance above the emergency cancellation
        # floor instead of adding both cushions together; adding them made a
        # healthy resident model permanently inadmissible even though it still
        # had the full hard-to-soft safety band available.
        return max(
            self.policy.soft_floor_gib,
            self.policy.hard_floor_gib + reserve,
        )

    def require(self, label: str, *, reserve_gib: float | None = None) -> None:
        """Admit work that may establish new memory residency.

        This is intentionally the soft-boundary check. Callers continuing an
        already-resident operation, or doing tightly bounded control work,
        should use :meth:`require_hard_floor` instead. Conflating those two
        boundaries can deadlock the operations that inspect or release
        resident state whenever available memory sits in the safety band.
        """

        if not self.enabled:
            return
        available = self.available_gib()
        required = self.required_gib(reserve_gib)
        if available < required:
            raise MemoryPressure(label, available, required)

    def require_hard_floor(self, label: str) -> None:
        """Admit bounded or continuing work above the emergency floor."""

        if not self.enabled:
            return
        available = self.available_gib()
        required = self.policy.hard_floor_gib
        if available < required:
            raise MemoryPressure(label, available, required)

    def require_capacity(self, label: str, additional_gib: float) -> None:
        """Admit known incremental residency while preserving the hard floor.

        The soft floor remains the conservative boundary for work whose peak
        growth is unknown. A measured, bounded executor can instead declare
        its peak incremental residency and use this check. Its emergency
        watcher remains responsible for cancelling if the estimate is wrong.
        """

        if not self.enabled:
            return
        available = self.available_gib()
        required = self.policy.hard_floor_gib + max(0.0, float(additional_gib))
        if available < required:
            raise MemoryPressure(label, available, required)

    def under_hard_pressure(self) -> bool:
        return self.enabled and self.available_gib() < self.policy.hard_floor_gib

    def watch(
        self,
        label: str,
        cancel: Callable[[], None],
        done: threading.Event,
    ) -> tuple[threading.Thread | None, threading.Event]:
        """Start an emergency watcher for an already admitted operation.

        The caller owns ``done`` and sets it in a finally block.  ``tripped``
        reports whether the watchdog invoked cancellation.  Cancellation is
        deliberately supplied by the operation: an HTTP stream closes its
        response, a shell kills its process group, and a browser closes its
        own session.  The governor itself never guesses process ownership.
        """

        tripped = threading.Event()
        if not self.enabled:
            return None, tripped

        def monitor() -> None:
            while not done.wait(self.policy.poll_interval_s):
                if not self.under_hard_pressure():
                    continue
                tripped.set()
                try:
                    cancel()
                except Exception:  # noqa: BLE001 - cancellation is best-effort
                    pass
                return

        thread = threading.Thread(
            target=monitor,
            name=f"omni-memory-watch-{label[:24]}",
            daemon=True,
        )
        thread.start()
        return thread, tripped
