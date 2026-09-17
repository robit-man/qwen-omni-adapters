"""Memory has to surface things because they are relevant, not because a rule fired.

The test that matters is the one no keyword list would pass: a question that
shares no words with what was remembered still finds it, and a question about
something else does not. Everything else here -- reinforcement, decay,
consolidation -- exists so the store behaves like something that grows rather
than a log that gets longer.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.memory import MemoryStore  # noqa: E402


class FakeEmbedder:
    """Deterministic vectors, so the tests measure the store and not a model.

    Each text becomes a point in a small topic space. Sentences about the same
    topic land near each other regardless of the words used, which is exactly
    the property real embeddings provide and the property being relied on.
    """

    TOPICS = {
        "dog": ("biscuit", "dog", "puppy", "bark", "collar", "pet"),
        "work": ("welder", "job", "work", "shift", "workshop", "trade"),
        "kitchen": ("kettle", "boil", "water", "kitchen", "tea", "hot"),
        "bike": ("bicycle", "bike", "wheel", "ride", "cycling"),
    }

    def __init__(self) -> None:
        self.calls = 0

    def close(self) -> None:  # noqa: D102
        pass

    def __call__(self, text: str) -> list[float] | None:
        self.calls += 1
        words = set(str(text).lower().replace("'", " ").split())
        vector = []
        for topic, terms in self.TOPICS.items():
            overlap = sum(1 for term in terms if any(term in word for word in words))
            vector.append(float(overlap))
        # A little mass everywhere, so unrelated things are distinguishable
        # without being orthogonal, as real embeddings are.
        vector.append(0.35)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else None


def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.sqlite3", embedder=FakeEmbedder())


# -- the property that matters --------------------------------------------


def test_a_question_finds_a_memory_it_shares_no_words_with(tmp_path: Path) -> None:
    """No rule could match "puppy" to "Biscuit"; meaning does."""

    memory = store(tmp_path)
    memory.remember("The user said their dog is called Biscuit.")
    memory.remember("The user mentioned they work as a welder.")
    memory.remember("The kettle in the kitchen was boiling.")

    found = memory.recall("what is my puppy called")

    assert found, "a related memory should surface"
    assert "Biscuit" in found[0].text
    memory.close()


def test_an_unrelated_question_surfaces_nothing(tmp_path: Path) -> None:
    """Interrupting with irrelevant history is worse than staying quiet."""

    memory = store(tmp_path)
    memory.remember("The user said their dog is called Biscuit.")

    assert memory.recall("what is the capital of France") == []
    memory.close()


# -- growing rather than accumulating -------------------------------------


def test_saying_the_same_thing_twice_leaves_one_stronger_memory(tmp_path: Path) -> None:
    memory = store(tmp_path)
    first = memory.remember("The user said their dog is called Biscuit.")
    again = memory.remember("The user said their dog is called Biscuit.")

    assert first == again
    assert memory.stats()["memories"] == 1
    assert memory.stats()["mean_strength"] > 0.5
    memory.close()


def test_being_recalled_makes_a_memory_stronger(tmp_path: Path) -> None:
    """What keeps turning out to be the answer should be easier to reach."""

    memory = store(tmp_path)
    memory.remember("The user said their dog is called Biscuit.")
    before = memory.stats()["mean_strength"]

    memory.recall("tell me about the dog")

    assert memory.stats()["mean_strength"] > before
    memory.close()


def test_the_strongest_of_several_relevant_memories_leads(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.remember("The dog barked at the postman.")
    wanted = memory.remember("The user said their dog is called Biscuit.")
    for _ in range(3):
        memory.recall("what is the dog called")

    found = memory.recall("tell me about the dog")

    assert found[0].id == wanted
    memory.close()


# -- forgetting, but only the right things --------------------------------


def test_something_never_needed_eventually_fades(tmp_path: Path) -> None:
    memory = store(tmp_path)
    memory.remember("A lorry went past outside.", kind="sound")

    # Long unused and never once relevant.
    memory._db.execute(
        "UPDATE memories SET last_used_at = ?, strength = 0.09, uses = 0",
        (time.time() - 400 * 86400,),
    )
    memory._db.commit()

    assert memory.decay() == 1
    assert memory.stats()["memories"] == 0
    memory.close()


def test_age_alone_never_forgets_something_that_was_needed(tmp_path: Path) -> None:
    """A thing said once and asked about a year later is the whole point."""

    memory = store(tmp_path)
    kept = memory.remember("The user said their dog is called Biscuit.")
    memory.recall("what is the dog called")
    memory._db.execute(
        "UPDATE memories SET created_at = ?, last_used_at = ?",
        (time.time() - 400 * 86400, time.time() - 400 * 86400),
    )
    memory._db.commit()

    memory.decay()

    assert memory.stats()["memories"] == 1
    assert memory.recall("what is my dog called")[0].id == kept
    memory.close()


# -- never load-bearing ---------------------------------------------------


def test_an_unreachable_embedder_disables_memory_rather_than_the_call(tmp_path: Path) -> None:
    """A conversation must survive its memory being unavailable."""

    class Dead:
        def __call__(self, text: str) -> None:
            return None

        def close(self) -> None:
            pass

    memory = MemoryStore(tmp_path / "memory.sqlite3", embedder=Dead())

    assert memory.remember("anything at all") is None
    assert memory.recall("anything at all") == []
    assert memory.stats()["memories"] == 0
    memory.close()


def test_memory_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path, embedder=FakeEmbedder())
    first.remember("The user said their dog is called Biscuit.")
    first.close()

    second = MemoryStore(path, embedder=FakeEmbedder())
    found = second.recall("what is the dog called")

    assert found and "Biscuit" in found[0].text
    second.close()
