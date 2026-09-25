"""Observable memory-hierarchy operations and per-answer traces."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MemoryOperation(str, Enum):
    PAGE_IN = "PAGE_IN"
    PAGE_OUT = "PAGE_OUT"
    PIN = "PIN"
    UNPIN = "UNPIN"
    EVICT = "EVICT"
    EXPAND = "EXPAND"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    RECONSTRUCT = "RECONSTRUCT"


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    timestamp: float
    operation: str
    item_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)


class TraceCollector:
    """Collect bounded, JSON-safe debug telemetry for one controller run."""

    def __init__(self, *, max_events: int = 2048) -> None:
        self.max_events = max(1, max_events)
        self._events: list[TraceEvent] = []

    def record(
        self,
        operation: MemoryOperation | str,
        item_id: str | None = None,
        **detail: Any,
    ) -> None:
        if len(self._events) >= self.max_events:
            return
        value = operation.value if isinstance(operation, MemoryOperation) else operation
        self._events.append(
            TraceEvent(
                sequence=len(self._events),
                timestamp=time.time(),
                operation=value,
                item_id=item_id,
                detail=detail,
            )
        )

    def export(self) -> tuple[dict[str, Any], ...]:
        return tuple(asdict(event) for event in self._events)
