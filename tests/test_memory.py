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
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.memory import Memory, MemoryStore, PassiveMemory  # noqa: E402


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
        for _topic, terms in self.TOPICS.items():
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


def test_passive_memory_embeds_after_the_caller_has_moved_on(tmp_path: Path) -> None:
    """A slow semantic write can never hold up the conversation thread."""

    started = threading.Event()
    release = threading.Event()
    stored: list[tuple[str, str]] = []

    class BlockingStore:
        def decay(self) -> int:
            return 0

        def stats(self) -> dict[str, int]:
            return {"memories": 0}

        def remember(self, text: str, *, kind: str) -> None:
            started.set()
            release.wait(2)
            stored.append((text, kind))

        def close(self) -> None:
            pass

    worker = PassiveMemory(
        tmp_path / "memory.sqlite3",
        store_factory=lambda _path: BlockingStore(),
    )
    worker.remember("The user said their dog is called Biscuit.", kind="exchange")
    assert started.wait(1)
    assert stored == []

    release.set()
    worker.close()
    assert stored == [("The user said their dog is called Biscuit.", "exchange")]


def test_passive_recall_is_taken_only_after_it_finishes(tmp_path: Path) -> None:
    """A slow encoder may enrich a later turn, but cannot hold this one up."""

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    expected = Memory(
        id=1,
        text="The user's dog is called Biscuit.",
        kind="exchange",
        created_at=time.time(),
        last_used_at=time.time(),
        uses=0,
        strength=0.5,
        similarity=0.8,
    )

    class BlockingStore:
        def decay(self) -> int:
            return 0

        def stats(self) -> dict[str, int]:
            return {"memories": 1}

        def recall(self, _text: str, *, limit: int) -> list[Memory]:
            assert limit == 4
            started.set()
            release.wait(2)
            finished.set()
            return [expected]

        def remember(self, _text: str, *, kind: str) -> None:
            pass

        def close(self) -> None:
            pass

    worker = PassiveMemory(
        tmp_path / "memory.sqlite3",
        store_factory=lambda _path: BlockingStore(),
    )
    worker.recall_later("what is my puppy's name", limit=4)
    assert started.wait(1)
    assert worker.take_recall() == []

    release.set()
    assert finished.wait(1)
    for _ in range(20):
        recalled = worker.take_recall()
        if recalled:
            break
        time.sleep(0.01)
    assert recalled == [expected]
    worker.close()


# -- a memory knows when it happened ---------------------------------------


def test_a_memory_says_when_it_happened_the_way_a_person_would(tmp_path: Path) -> None:
    """Both forms: "yesterday" places it, the date makes it checkable."""

    from harness.memory import Memory

    now = time.time()

    def at(days_ago: float) -> Memory:
        return Memory(
            id=1,
            text="the dog is called Biscuit",
            kind="exchange",
            created_at=now - days_ago * 86400,
            last_used_at=now,
            uses=0,
            strength=0.5,
        )

    assert at(0).when().startswith("today at ")
    assert at(1).when().startswith("yesterday at ")
    assert "days ago" in at(3).when()
    # Far enough back that a weekday name would be useless.
    assert "days ago" not in at(40).when()
    # Older than this year carries the year.
    assert str(datetime.fromtimestamp(now - 500 * 86400).year) in at(500).when()


def test_the_stamp_travels_with_the_text(tmp_path: Path) -> None:
    from harness.memory import Memory

    memory = Memory(
        id=1,
        text="the dog is called Biscuit",
        kind="exchange",
        created_at=time.time(),
        last_used_at=time.time(),
        uses=0,
        strength=0.5,
    )

    stamped = memory.stamped()
    assert stamped.startswith("[today at ")
    assert stamped.endswith("the dog is called Biscuit")


def test_the_timestamp_is_not_embedded_with_the_text(tmp_path: Path) -> None:
    """Dating every memory in its text would make them all look alike.

    The date becomes a feature shared by everything stored, competing with
    what each memory is actually about. It is kept beside the text and
    attached at the point of use instead.
    """

    embedder = FakeEmbedder()
    memory = MemoryStore(tmp_path / "memory.sqlite3", embedder=embedder)
    memory.remember("The user said their dog is called Biscuit.")

    row = memory._db.execute("SELECT text FROM memories").fetchone()
    assert row["text"] == "The user said their dog is called Biscuit."
    assert "[" not in row["text"]

    # And recall is unaffected by how long ago it was stored.
    found = memory.recall("what is the dog called")
    assert found and "Biscuit" in found[0].text
    memory.close()
