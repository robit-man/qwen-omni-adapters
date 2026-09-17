"""Hold what has been said until the speaker is actually finished saying it.

A port of ``portal/static/call_queue.js``, for the same reason the VAD was
ported: the browser's live-call mode has been used in real rooms, and people do
not speak in one clean burst. They start, stop to think, and carry on -- and
each of those pauses looks exactly like the end of a turn to a voice detector.

So a finished utterance is not a turn yet. It waits a moment, and anything said
in that moment joins it. Segments are concatenated with a short silence between
them, which is what the pause sounded like, and the model is told how many
segments it received so it can read later words as corrections of earlier ones.

The buffer is bounded. Someone who talks for a minute without a real pause gets
the most recent window rather than an unbounded request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Browser defaults, kept identical so the same room behaves the same way.
SETTLE_MS = 220.0
GAP_MS = 120.0
MAX_SECONDS = 45.0


@dataclass
class Pending:
    """Speech waiting to become one turn."""

    chunks: list[np.ndarray] = field(default_factory=list)
    samples: int = 0
    active_ms: float = 0.0
    segments: int = 0
    truncated: bool = False

    def __bool__(self) -> bool:
        return self.samples > 0 and self.segments > 0

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks)


class CallQueue:
    """Accumulate consecutive utterances into a single turn."""

    def __init__(
        self,
        rate_hz: int,
        *,
        gap_ms: float = GAP_MS,
        max_seconds: float = MAX_SECONDS,
    ) -> None:
        self.rate_hz = rate_hz
        self.gap_ms = gap_ms
        self.max_seconds = max_seconds
        self._pending = Pending()

    def __bool__(self) -> bool:
        return bool(self._pending)

    @property
    def segments(self) -> int:
        return self._pending.segments

    @property
    def seconds(self) -> float:
        return self._pending.samples / self.rate_hz if self.rate_hz else 0.0

    def add(self, samples: np.ndarray, active_ms: float = 0.0) -> None:
        """Append one finished utterance to whatever is already waiting."""

        if samples is None or samples.size == 0:
            return
        if self._pending.segments and self.gap_ms > 0:
            # The pause is part of what was said: without it the two halves
            # run together into a word that was never spoken.
            gap = max(1, int(round(self.rate_hz * self.gap_ms / 1000.0)))
            self._pending.chunks.append(np.zeros(gap, dtype=np.float32))
            self._pending.samples += gap
        self._pending.chunks.append(np.asarray(samples, dtype=np.float32))
        self._pending.samples += int(samples.size)
        self._pending.segments += 1
        self._pending.active_ms += max(0.0, active_ms)
        self._trim()

    def prepend(self, samples: np.ndarray, active_ms: float = 0.0) -> None:
        """Put speech back at the front, ahead of what is already waiting.

        Used when a reply is interrupted: the question it was answering was
        never actually answered, so it belongs with whatever the speaker went
        on to say rather than being stranded in the history as a turn that
        got talked over.
        """

        if samples is None or samples.size == 0:
            return
        head = [np.asarray(samples, dtype=np.float32)]
        added = int(samples.size)
        if self._pending.segments and self.gap_ms > 0:
            gap = max(1, int(round(self.rate_hz * self.gap_ms / 1000.0)))
            head.append(np.zeros(gap, dtype=np.float32))
            added += gap
        self._pending.chunks = head + self._pending.chunks
        self._pending.samples += added
        self._pending.segments += 1
        self._pending.active_ms += max(0.0, active_ms)
        self._trim()

    def _trim(self) -> None:
        """Keep the most recent window, dropping the oldest audio first."""

        limit = max(1, int(self.rate_hz * self.max_seconds))
        overflow = self._pending.samples - limit
        while overflow > 0 and self._pending.chunks:
            first = self._pending.chunks[0]
            if first.size <= overflow:
                self._pending.chunks.pop(0)
                self._pending.samples -= int(first.size)
                overflow -= int(first.size)
            else:
                self._pending.chunks[0] = first[overflow:]
                self._pending.samples -= overflow
                overflow = 0
            self._pending.truncated = True
        self._pending.active_ms = min(
            self._pending.active_ms, self.max_seconds * 1000.0
        )

    def take(self) -> Pending:
        """Hand over everything waiting, and start again empty."""

        taken, self._pending = self._pending, Pending()
        return taken

    def clear(self) -> None:
        self._pending = Pending()
