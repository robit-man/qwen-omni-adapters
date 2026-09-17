"""Rolling memory: what was said and seen, kept so it can come back.

The point is that recall happens because something is *relevant*, not because
a rule fired. Every turn is embedded and stored; every new turn is embedded and
compared against everything before it. What surfaces, surfaces on similarity
alone -- there is no list of words that mean "remember", and asking about a
conversation from last week works for the same reason asking about one from a
minute ago does.

Two properties make it behave like something that grows rather than a log:

*Strength.* A memory that keeps being recalled is reinforced and ranks higher
next time; one that is never recalled decays. Nothing is deleted for being old,
only for being consistently irrelevant, so a thing mentioned once six months ago
survives if it was the answer to something.

*Consolidation.* Near-duplicate memories are merged rather than accumulated, so
saying the same thing five times leaves one stronger memory instead of five
competing ones.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# How close two memories must be before they are treated as the same thing.
MERGE_SIMILARITY = 0.93
# Below this, a memory is not related enough to be worth interrupting with.
RECALL_FLOOR = 0.45
# A memory earns this much strength each time it turns out to be the answer.
REINFORCEMENT = 0.35
# And loses this share of its strength per day of never being needed.
DAILY_DECAY = 0.02
# Forgotten only once it has been both weak and unused for a long time.
FORGET_BELOW = 0.08


@dataclass
class Memory:
    """One thing worth remembering, and how well it has held up."""

    id: int
    text: str
    kind: str
    created_at: float
    last_used_at: float
    uses: int
    strength: float
    similarity: float = 0.0

    def age_days(self, now: float | None = None) -> float:
        return max(0.0, ((now or time.time()) - self.created_at) / 86400.0)

    def score(self, now: float | None = None) -> float:
        """Relevance, tempered by how well this memory has held up.

        Similarity decides what is even a candidate; strength decides which of
        several relevant memories is the one worth saying. Recency breaks ties
        so a fact restated today outranks the same fact from a year ago.
        """

        now = now or time.time()
        idle_days = max(0.0, (now - self.last_used_at) / 86400.0)
        freshness = 1.0 / (1.0 + idle_days / 30.0)
        return self.similarity * (0.65 + 0.25 * self.strength + 0.10 * freshness)


class Embedder:
    """Turns text into a vector, through whatever service is configured.

    Kept behind an interface because the choice is a deployment detail: the
    omni package itself is a chat and perception model, not an encoder, and
    using it for this ranked unrelated sentences within 0.002 of each other.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_s)
        self._warned = False

    def close(self) -> None:
        self._client.close()

    def __call__(self, text: str) -> list[float] | None:
        text = " ".join((text or "").split())
        if not text:
            return None
        try:
            response = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:4000]},
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
        except Exception as error:  # noqa: BLE001 - memory is never load-bearing
            if not self._warned:
                logger.warning(
                    "memory is disabled: the embedder is unreachable (%s)", error
                )
                self._warned = True
            return None
        self._warned = False
        return vector if isinstance(vector, list) and vector else None


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left = list(left)
    right = list(right)
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    magnitude = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return dot / magnitude if magnitude else 0.0


class MemoryStore:
    """Everything remembered, and the arithmetic that keeps it honest.

    SQLite with vectors as blobs: the working set here is a conversation's
    worth of memories, not a corpus, so a linear scan costs less than the
    dependency an index would add.
    """

    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._prepare()

    def close(self) -> None:
        self._db.close()
        self.embedder.close()

    def _prepare(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY,
                text         TEXT NOT NULL,
                kind         TEXT NOT NULL DEFAULT 'turn',
                vector       BLOB NOT NULL,
                created_at   REAL NOT NULL,
                last_used_at REAL NOT NULL,
                uses         INTEGER NOT NULL DEFAULT 0,
                strength     REAL NOT NULL DEFAULT 0.5
            );
            CREATE INDEX IF NOT EXISTS memories_recent
                ON memories (last_used_at DESC);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._db.commit()

    # -- writing ---------------------------------------------------------

    def remember(self, text: str, *, kind: str = "turn") -> int | None:
        """Store something, or strengthen it if it is already known."""

        text = " ".join((text or "").split())
        if len(text) < 8:
            return None
        vector = self.embedder(text)
        if vector is None:
            return None

        existing = self._nearest(vector, limit=1)
        if existing and existing[0].similarity >= MERGE_SIMILARITY:
            # The same thing said again. One stronger memory beats two rivals.
            self._reinforce(existing[0].id, text=max(text, existing[0].text, key=len))
            return existing[0].id

        now = time.time()
        cursor = self._db.execute(
            "INSERT INTO memories (text, kind, vector, created_at, last_used_at,"
            " uses, strength) VALUES (?, ?, ?, ?, ?, 0, 0.5)",
            (text, kind, json.dumps(vector).encode(), now, now),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def _reinforce(self, memory_id: int, *, text: str | None = None) -> None:
        now = time.time()
        if text is None:
            self._db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?,"
                " strength = MIN(1.0, strength + ?) WHERE id = ?",
                (now, REINFORCEMENT, memory_id),
            )
        else:
            self._db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?,"
                " strength = MIN(1.0, strength + ?), text = ? WHERE id = ?",
                (now, REINFORCEMENT, text, memory_id),
            )
        self._db.commit()

    # -- reading ---------------------------------------------------------

    def _rows(self) -> list[sqlite3.Row]:
        return list(self._db.execute("SELECT * FROM memories"))

    def _nearest(self, vector: list[float], limit: int) -> list[Memory]:
        found: list[Memory] = []
        for row in self._rows():
            try:
                stored = json.loads(bytes(row["vector"]).decode())
            except (ValueError, UnicodeDecodeError):
                continue
            memory = Memory(
                id=int(row["id"]),
                text=row["text"],
                kind=row["kind"],
                created_at=float(row["created_at"]),
                last_used_at=float(row["last_used_at"]),
                uses=int(row["uses"]),
                strength=float(row["strength"]),
                similarity=_cosine(vector, stored),
            )
            found.append(memory)
        found.sort(key=lambda item: item.similarity, reverse=True)
        return found[:limit]

    def recall(
        self, query: str, *, limit: int = 4, floor: float = RECALL_FLOOR
    ) -> list[Memory]:
        """What this turn is actually about, from everything remembered.

        Nothing here inspects the words for intent. A question about a dog
        surfaces the dog because the sentences are close in meaning, which is
        also why it works for a phrasing nobody anticipated.
        """

        vector = self.embedder(query)
        if vector is None:
            return []
        candidates = [
            memory
            for memory in self._nearest(vector, limit=max(limit * 4, 16))
            if memory.similarity >= floor
        ]
        now = time.time()
        candidates.sort(key=lambda item: item.score(now), reverse=True)
        chosen = candidates[:limit]
        for memory in chosen:
            self._reinforce(memory.id)
        return chosen

    # -- keeping it a memory rather than a log ---------------------------

    def decay(self) -> int:
        """Let unused memories fade, and drop the ones that faded to nothing.

        Age alone never forgets anything: something said once and asked about
        a year later is exactly what this is for. What fades is what has
        repeatedly failed to be relevant.
        """

        now = time.time()
        removed = 0
        for row in self._rows():
            idle_days = max(0.0, (now - float(row["last_used_at"])) / 86400.0)
            if idle_days < 1.0:
                continue
            strength = float(row["strength"]) * ((1.0 - DAILY_DECAY) ** idle_days)
            if strength < FORGET_BELOW and int(row["uses"]) == 0:
                self._db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                removed += 1
            else:
                self._db.execute(
                    "UPDATE memories SET strength = ? WHERE id = ?",
                    (strength, row["id"]),
                )
        self._db.commit()
        return removed

    def stats(self) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT COUNT(*) AS total, AVG(strength) AS strength,"
            " MIN(created_at) AS oldest FROM memories"
        ).fetchone()
        return {
            "memories": int(row["total"] or 0),
            "mean_strength": round(float(row["strength"] or 0.0), 3),
            "oldest_days": round(
                (time.time() - float(row["oldest"])) / 86400.0, 2
            )
            if row["oldest"]
            else 0.0,
        }


class PassiveMemory:
    """Run every embedding and database operation away from the call path.

    Hearing, answering, reasoning, tool use, and speech must never wait for an
    embedding server or SQLite. Recall is therefore speculative: the call loop
    asks for it as soon as the normal chat stream exposes a transcript and only
    consumes it later if it is already ready. A slow or failed memory operation
    is simply absent from that turn.
    """

    def __init__(
        self,
        path: Path,
        *,
        store_factory: Callable[[Path], MemoryStore] = MemoryStore,
        max_pending: int = 64,
    ) -> None:
        self.path = Path(path)
        self._store_factory = store_factory
        self._jobs: queue.Queue[tuple[str, str, str, int]] = queue.Queue(
            maxsize=max_pending
        )
        self._ready: dict[str, list[Memory]] = {}
        self._ready_lock = threading.Lock()
        self._closed = threading.Event()
        self._warned_full = False
        self._thread = threading.Thread(
            target=self._work,
            name="omni-passive-memory",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _key(text: str) -> str:
        return " ".join((text or "").split())

    def _submit(self, job: tuple[str, str, str, int]) -> None:
        if self._closed.is_set() or not job[1]:
            return
        try:
            self._jobs.put_nowait(job)
            self._warned_full = False
        except queue.Full:
            if not self._warned_full:
                logger.warning("memory queue is full; dropping background work")
                self._warned_full = True

    def remember(self, text: str, *, kind: str = "turn") -> None:
        """Queue a write and return immediately."""

        self._submit(("remember", self._key(text), kind, 0))

    def recall_later(self, query: str, *, limit: int = 4) -> None:
        """Queue semantic recall without making the conversation wait for it."""

        self._submit(("recall", self._key(query), "", max(1, limit)))

    def take_recall(self, query: str) -> list[Memory]:
        """Return a completed recall, or immediately return no context."""

        key = self._key(query)
        with self._ready_lock:
            return self._ready.pop(key, [])

    def close(self) -> None:
        """Ask the daemon worker to drain and close; never wait on a call exit."""

        self._closed.set()

    def _work(self) -> None:
        store: MemoryStore | None = None
        try:
            store = self._store_factory(self.path)
            faded = store.decay()
            logger.info(
                "memory ready in background: %s at %s%s",
                store.stats(),
                self.path,
                f"; {faded} forgotten" if faded else "",
            )
            while not (self._closed.is_set() and self._jobs.empty()):
                try:
                    action, text, kind, limit = self._jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if action == "remember":
                        store.remember(text, kind=kind)
                    elif action == "recall":
                        found = store.recall(text, limit=limit)
                        with self._ready_lock:
                            # A missed turn must not grow an unbounded cache.
                            if len(self._ready) >= 32:
                                self._ready.pop(next(iter(self._ready)))
                            self._ready[text] = found
                except Exception as error:  # noqa: BLE001 - strictly best effort
                    logger.warning("background memory operation failed: %s", error)
                finally:
                    self._jobs.task_done()
        except Exception as error:  # noqa: BLE001 - memory cannot break the call
            logger.warning("memory is disabled: background startup failed (%s)", error)
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:  # noqa: BLE001 - process teardown is best effort
                    logger.debug("could not close memory store", exc_info=True)
