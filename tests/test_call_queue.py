"""Speech that pauses and carries on is one turn, not two.

People stop mid-sentence to think, and every one of those pauses looks exactly
like the end of a turn to a voice detector. Answering each detected utterance
separately is what produced a reply to the first half while the second half was
still being spoken -- and then a second reply to that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.call_queue import GAP_MS, MAX_SECONDS, CallQueue  # noqa: E402

RATE = 16_000


def speech(seconds: float, level: float = 0.2) -> np.ndarray:
    return np.full(int(RATE * seconds), level, dtype=np.float32)


def test_carrying_on_makes_one_turn_not_two() -> None:
    queue = CallQueue(RATE)
    queue.add(speech(1.0), active_ms=1000)
    queue.add(speech(0.8), active_ms=800)

    pending = queue.take()

    assert pending.segments == 2
    assert pending.active_ms == 1800
    # Both halves are there, plus the pause between them.
    expected = int(RATE * 1.8) + int(round(RATE * GAP_MS / 1000))
    assert pending.audio().size == expected


def test_the_pause_is_kept_between_segments() -> None:
    """Without it the two halves run into a word nobody said."""

    queue = CallQueue(RATE)
    queue.add(speech(0.5))
    queue.add(speech(0.5))

    audio = queue.take().audio()
    gap = int(round(RATE * GAP_MS / 1000))
    middle = audio[int(RATE * 0.5) : int(RATE * 0.5) + gap]

    assert np.allclose(middle, 0.0)


def test_taking_leaves_the_queue_empty_for_what_comes_next() -> None:
    queue = CallQueue(RATE)
    queue.add(speech(0.5))

    assert queue
    queue.take()
    assert not queue
    assert queue.segments == 0


def test_an_interrupted_question_goes_back_in_front() -> None:
    """It was cut off, so it still stands, and belongs before the follow-on."""

    queue = CallQueue(RATE)
    queue.add(speech(0.4, level=0.3))          # what was said over the reply
    queue.prepend(speech(0.6, level=0.1), 600)  # the question that was cut off

    pending = queue.take()
    audio = pending.audio()

    assert pending.segments == 2
    # The re-queued question leads.
    assert float(audio[0]) == pytest.approx(0.1)
    assert float(audio[-1]) == pytest.approx(0.3)


def test_a_long_monologue_keeps_the_most_recent_window() -> None:
    """Someone who never pauses must not build an unbounded request."""

    queue = CallQueue(RATE, max_seconds=2.0)
    queue.add(speech(1.5, level=0.1))
    queue.add(speech(1.5, level=0.2))

    pending = queue.take()

    assert pending.truncated is True
    assert pending.audio().size <= int(RATE * 2.0)
    # The newest audio is what survived.
    assert float(pending.audio()[-1]) == pytest.approx(0.2)


def test_silence_is_never_queued() -> None:
    queue = CallQueue(RATE)
    queue.add(np.zeros(0, dtype=np.float32))

    assert not queue
    assert queue.take().segments == 0


def test_the_browser_defaults_are_shared() -> None:
    """The same room should behave the same way here and in the browser."""

    assert GAP_MS == 120.0
    assert MAX_SECONDS == 45.0
