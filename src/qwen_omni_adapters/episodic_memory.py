"""Source-preserving multimodal episodes and a typed, derived memory graph.

Sources commit independently of model work. An encoder may propose entities and
relations only with exact quotations from those sources. Graph scores express
retrieval relevance, never the truth of an observation or a generated claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

MODALITIES = frozenset({"text", "speech", "sound", "image", "video", "tool", "sensor", "environment", "other"})
AUTHORITIES = frozenset({"user_statement", "perceptual_observation", "tool_result", "assistant_generated", "unattributed", "client_report"})
RELATION_TYPES = frozenset({"semantic", "entity", "temporal", "causal"})
MAX_EVENT_CHARS = 24_000
MAX_ENCODING_EVENTS = 8


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _name(value: Any, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("invalid bounded graph label")
    return unicodedata.normalize("NFKC", value.strip())


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    if not all(math.isfinite(float(value)) for value in (*left, *right)):
        return 0.0
    norm = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / norm if norm else 0.0


class EpisodicMemory:
    """One authorization scope. Callers choose its private path, never the model."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        self.path.chmod(0o600)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, timeout=2, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS event_sources(
                id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
                episode TEXT NOT NULL, modality TEXT NOT NULL, authority TEXT NOT NULL,
                actor TEXT NOT NULL, text TEXT NOT NULL, occurred_at REAL NOT NULL,
                recorded_at REAL NOT NULL, content_sha256 TEXT NOT NULL,
                provenance_json TEXT NOT NULL, vector_json TEXT,
                encoded INTEGER NOT NULL DEFAULT 0, retry_after REAL NOT NULL DEFAULT 0,
                encoding_error TEXT
            );
            CREATE INDEX IF NOT EXISTS event_source_time ON event_sources(occurred_at,id);
            CREATE INDEX IF NOT EXISTS event_episode ON event_sources(episode,id);
            CREATE TABLE IF NOT EXISTS source_annotations(
                event_id INTEGER NOT NULL,name TEXT NOT NULL,value_json TEXT NOT NULL,
                producer TEXT NOT NULL,recorded_at REAL NOT NULL,
                PRIMARY KEY(event_id,name),FOREIGN KEY(event_id) REFERENCES event_sources(id)
            );
            CREATE TABLE IF NOT EXISTS source_annotation_history(
                id INTEGER PRIMARY KEY,event_id INTEGER NOT NULL,name TEXT NOT NULL,
                value_json TEXT NOT NULL,producer TEXT NOT NULL,recorded_at REAL NOT NULL,
                FOREIGN KEY(event_id) REFERENCES event_sources(id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(
                text, content='event_sources', content_rowid='id', tokenize='unicode61');
            CREATE TRIGGER IF NOT EXISTS event_fts_insert AFTER INSERT ON event_sources
                BEGIN INSERT INTO event_fts(rowid,text) VALUES(new.id,new.text); END;
            CREATE TABLE IF NOT EXISTS graph_entities(
                id TEXT PRIMARY KEY, canonical TEXT NOT NULL, aliases_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS graph_entity_fts USING fts5(entity_id UNINDEXED,canonical,aliases,tokenize='unicode61');
            CREATE TRIGGER IF NOT EXISTS graph_entity_insert AFTER INSERT ON graph_entities
                BEGIN INSERT INTO graph_entity_fts(rowid,entity_id,canonical,aliases) VALUES(new.rowid,new.id,new.canonical,new.aliases_json); END;
            CREATE TRIGGER IF NOT EXISTS graph_entity_update AFTER UPDATE ON graph_entities
                BEGIN DELETE FROM graph_entity_fts WHERE rowid=old.rowid;
                INSERT INTO graph_entity_fts(rowid,entity_id,canonical,aliases) VALUES(new.rowid,new.id,new.canonical,new.aliases_json); END;
            CREATE TABLE IF NOT EXISTS graph_facts(
                id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL,
                object TEXT NOT NULL, relation_type TEXT NOT NULL,
                authority TEXT NOT NULL, epistemic_status TEXT NOT NULL,
                valid_from REAL NOT NULL, valid_to REAL, support_json TEXT NOT NULL,
                supersedes_json TEXT NOT NULL, FOREIGN KEY(subject) REFERENCES graph_entities(id),
                FOREIGN KEY(object) REFERENCES graph_entities(id)
            );
            CREATE TABLE IF NOT EXISTS fact_sources(
                fact_id TEXT NOT NULL, event_id INTEGER NOT NULL,
                PRIMARY KEY(fact_id,event_id), FOREIGN KEY(fact_id) REFERENCES graph_facts(id),
                FOREIGN KEY(event_id) REFERENCES event_sources(id)
            );
            CREATE INDEX IF NOT EXISTS fact_source_events ON fact_sources(event_id,fact_id);
            CREATE INDEX IF NOT EXISTS graph_subject ON graph_facts(subject);
            CREATE INDEX IF NOT EXISTS graph_object ON graph_facts(object);
            CREATE TABLE IF NOT EXISTS episode_summaries(
                episode TEXT PRIMARY KEY, summary TEXT NOT NULL, support_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self.path.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def append(self, *, event_key: str, episode: str, modality: str, authority: str,
               actor: str, text: str, occurred_at: float | None = None,
               provenance: Mapping[str, Any] | None = None) -> int:
        if modality not in MODALITIES or authority not in AUTHORITIES:
            raise ValueError("unknown event modality or authority")
        key, episode = _name(event_key, 256), _name(episode, 256)
        actor = _name(actor, 128)
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_EVENT_CHARS:
            raise ValueError("event text must contain 1–24000 characters")
        stamp = time.time() if occurred_at is None else float(occurred_at)
        if not math.isfinite(stamp):
            raise ValueError("event time must be finite")
        digest = hashlib.sha256(text.encode()).hexdigest()
        source = dict(provenance or {})
        # A recalled source is historical, regardless of its original modality.
        source["current_visual_input"] = False
        source["historical"] = True
        source["current_environment_evidence"] = False
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            previous = self._db.execute("SELECT * FROM event_sources WHERE event_key=?", (key,)).fetchone()
            if previous:
                if (previous["content_sha256"], previous["modality"], previous["authority"], previous["episode"], previous["actor"]) != (digest, modality, authority, episode, actor):
                    raise ValueError("event key already identifies different evidence")
                return int(previous["id"])
            cursor = self._db.execute(
                "INSERT INTO event_sources(event_key,episode,modality,authority,actor,text,occurred_at,recorded_at,content_sha256,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (key, episode, modality, authority, actor, text, stamp, time.time(), digest, _json(source)),
            )
            event_id = int(cursor.lastrowid)
            eligible = source.get("conversation_eligible")
            if not isinstance(eligible, bool):
                eligible = authority == "user_statement" and modality == "text"
            self._annotate_locked(event_id, {"conversation_eligible": eligible}, "source_capture")
            return event_id

    def _annotate_locked(self, event_id: int, tags: Mapping[str, Any], producer: str) -> None:
        for name, value in tags.items():
            if name == "speech_addressee":
                if value not in {"SELF", "OTHER", "AMBIGUOUS", "UNKNOWN"}:
                    raise ValueError("invalid observed speech addressee")
            elif name == "conversation_eligible":
                if not isinstance(value, bool):
                    raise ValueError("conversation membership requires a boolean receipt")
            else:
                raise ValueError("unknown source annotation")
            encoded = _json(value)
            old = self._db.execute("SELECT value_json,producer FROM source_annotations WHERE event_id=? AND name=?", (event_id, name)).fetchone()
            if old and (old[0], old[1]) == (encoded, producer):
                continue
            stamp = time.time()
            self._db.execute("INSERT INTO source_annotation_history(event_id,name,value_json,producer,recorded_at) VALUES(?,?,?,?,?)", (event_id, name, encoded, producer, stamp))
            self._db.execute("INSERT INTO source_annotations VALUES(?,?,?,?,?) ON CONFLICT(event_id,name) DO UPDATE SET value_json=excluded.value_json,producer=excluded.producer,recorded_at=excluded.recorded_at", (event_id, name, encoded, producer, stamp))

    def annotate_episode(self, episode: str, tags: Mapping[str, Any], *, producer: str) -> None:
        """Append classifier receipts separately from immutable observed sources."""
        producer = _name(producer, 128)
        with self._lock, self._db:
            for row in self._db.execute("SELECT id,modality FROM event_sources WHERE episode=?", (episode,)).fetchall():
                applicable = {key: value for key, value in tags.items() if key != "speech_addressee" or row["modality"] == "speech"}
                self._annotate_locked(int(row["id"]), applicable, producer)

    def _annotations(self, event_id: int) -> dict[str, Any]:
        return {row[0]: json.loads(row[1]) for row in self._db.execute("SELECT name,value_json FROM source_annotations WHERE event_id=?", (event_id,))}

    def pending(self, limit: int = MAX_ENCODING_EVENTS, *, newest: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            direction = "DESC" if newest else "ASC"
            first = self._db.execute(f"SELECT episode FROM event_sources WHERE (encoded=0 OR vector_json IS NULL) AND retry_after<=? ORDER BY occurred_at {direction},id {direction} LIMIT 1", (time.time(),)).fetchone()
            if first is None:
                return []
            rows = self._db.execute(f"SELECT * FROM event_sources WHERE episode=? AND (encoded=0 OR vector_json IS NULL) AND retry_after<=? ORDER BY id {direction} LIMIT ?", (first["episode"], time.time(), max(1, min(MAX_ENCODING_EVENTS, limit)))).fetchall()
            return [self._event(row) for row in rows]

    def events(self, event_ids: Sequence[int]) -> list[dict[str, Any]]:
        identities = list(dict.fromkeys(int(value) for value in event_ids))[:16]
        with self._lock:
            return [self._event(row) for identity in identities
                    if (row := self._db.execute("SELECT * FROM event_sources WHERE id=?", (identity,)).fetchone()) is not None]

    def _event(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"event_id": int(row["id"]), "episode": row["episode"], "modality": row["modality"],
                "authority": row["authority"], "actor": row["actor"], "text": row["text"],
                "occurred_at": row["occurred_at"], "recorded_at": row["recorded_at"],
                "content_sha256": row["content_sha256"], "provenance": json.loads(row["provenance_json"]),
                "encoded": bool(row["encoded"]), "vector_available": row["vector_json"] is not None,
                "annotations": self._annotations(int(row["id"]))}

    def defer(self, event_ids: Sequence[int], reason: str, seconds: float = 30) -> None:
        code = reason if reason in {"model_unavailable", "invalid_encoding", "interrupted", "resource_pressure"} else "invalid_encoding"
        with self._lock, self._db:
            self._db.executemany("UPDATE event_sources SET retry_after=?,encoding_error=? WHERE id=? AND (encoded=0 OR vector_json IS NULL)",
                [(time.time() + max(0, seconds), code, int(event_id)) for event_id in event_ids])

    def _support(self, value: Any, allowed: set[int]) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not 1 <= len(value) <= 8:
            raise ValueError("derived memory requires bounded source quotations")
        support = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError("invalid source reference")
            event_id = int(item.get("event_id", -1))
            quote = item.get("quote")
            if event_id not in allowed or not isinstance(quote, str) or not quote.strip() or len(quote) > 2000:
                raise ValueError("source reference outside the encoding batch")
            row = self._db.execute("SELECT text,content_sha256,authority,actor,modality,occurred_at,provenance_json FROM event_sources WHERE id=?", (event_id,)).fetchone()
            if row is None:
                raise ValueError("unknown encoding source")
            start = row["text"].find(quote)
            if start < 0:
                raise ValueError("derived memory quote does not match its source")
            support.append({"event_id": event_id, "quote": quote, "char_start": start, "char_end": start + len(quote),
                            "content_sha256": row["content_sha256"], "authority": row["authority"], "actor": row["actor"],
                            "modality": row["modality"], "occurred_at": row["occurred_at"], "provenance": json.loads(row["provenance_json"]),
                            "annotations": self._annotations(event_id)})
        return support

    def _entity(self, name: str, aliases: Sequence[str] = ()) -> str:
        canonical = _name(name)
        identity = hashlib.sha256(canonical.casefold().encode()).hexdigest()[:24]
        previous = self._db.execute("SELECT aliases_json FROM graph_entities WHERE id=?", (identity,)).fetchone()
        names = set(json.loads(previous[0])) if previous else set()
        names.update(_name(alias) for alias in aliases[:12])
        names.add(canonical)
        self._db.execute("INSERT INTO graph_entities VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET aliases_json=excluded.aliases_json", (identity, canonical, _json(sorted(names))))
        return identity

    def contains_episode(self, episode: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM event_sources WHERE episode=? LIMIT 1", (episode,)).fetchone() is not None

    def context(self, event_ids: Sequence[int], limit: int = 6) -> list[dict[str, Any]]:
        """Recent same-scope context for referents, with original source IDs."""
        ids = {int(value) for value in event_ids}
        if not ids:
            return []
        with self._lock:
            rows = self._db.execute("SELECT * FROM event_sources WHERE id<=? ORDER BY id DESC LIMIT 24", (max(ids),)).fetchall()
            return [self._event(row) for row in reversed(rows) if int(row["id"]) not in ids][-max(1, min(8, limit)):]

    def apply_encoding(self, event_ids: Sequence[int], encoding: Mapping[str, Any], vectors: Mapping[int, Sequence[float]] | None = None,
                       *, context_event_ids: Sequence[int] = (), complete: bool = True) -> dict[str, int]:
        targets = {int(value) for value in event_ids}
        context = {int(value) for value in context_event_ids}
        allowed = targets | context
        facts = encoding.get("facts", [])
        entities = encoding.get("entities", [])
        if not targets or len(targets) > MAX_ENCODING_EVENTS or len(context) > 8 or not isinstance(facts, list) or len(facts) > 24 or not isinstance(entities, list) or len(entities) > 32:
            raise ValueError("encoding exceeds bounded graph limits")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            rows = [self._db.execute("SELECT episode FROM event_sources WHERE id=?", (value,)).fetchone() for value in targets]
            if any(row is None for row in rows) or len({row[0] for row in rows}) != 1:
                raise ValueError("encoding sources must belong to one recorded episode")
            for entity in entities:
                if not isinstance(entity, Mapping) or not isinstance(entity.get("aliases", []), list):
                    raise ValueError("invalid entity encoding")
                self._support(entity.get("support"), allowed)
                self._entity(entity.get("name"), entity.get("aliases", []))
            written = 0
            for fact in facts:
                if not isinstance(fact, Mapping) or fact.get("relation_type") not in RELATION_TYPES or fact.get("epistemic_status") not in {"explicit", "inferred", "uncertain"}:
                    raise ValueError("invalid typed fact")
                support = self._support(fact.get("support"), allowed)
                if not any(item["event_id"] in targets for item in support):
                    # Historical context is supplied for reference resolution.
                    # Its own encoding job owns facts that cite only that context.
                    continue
                authorities = {item["authority"] for item in support}
                if authorities == {"assistant_generated"}:
                    raise ValueError("generated narration cannot establish an external fact")
                authority = next(iter(authorities)) if len(authorities) == 1 else "mixed_sources"
                subject, object_ = self._entity(fact.get("subject")), self._entity(fact.get("object"))
                predicate = _name(fact.get("predicate"), 128)
                stamp = max(item["occurred_at"] for item in support)
                supersedes = fact.get("supersedes", [])
                if not isinstance(supersedes, list) or len(supersedes) > 8:
                    raise ValueError("invalid fact update")
                if supersedes and (authority not in {"user_statement", "perceptual_observation", "tool_result"} or fact["epistemic_status"] != "explicit"):
                    raise ValueError("uncertain or unattributed evidence cannot supersede a fact")
                for old_id in supersedes:
                    previous = self._db.execute("SELECT * FROM graph_facts WHERE id=?", (old_id,)).fetchone()
                    if previous is None or previous["subject"] != subject or previous["predicate"] != predicate or previous["authority"] != authority or previous["valid_from"] > stamp:
                        raise ValueError("update cannot supersede unrelated or differently authorized evidence")
                identity = hashlib.sha256(_json([subject, predicate, object_, fact["relation_type"], support]).encode()).hexdigest()[:24]
                self._db.execute("INSERT OR IGNORE INTO graph_facts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (identity, subject, predicate, object_, fact["relation_type"], authority, fact["epistemic_status"], stamp, None, _json(support), _json(supersedes)))
                self._db.executemany("INSERT OR IGNORE INTO fact_sources VALUES(?,?)", [(identity, item["event_id"]) for item in support])
                for old_id in supersedes:
                    self._db.execute("UPDATE graph_facts SET valid_to=? WHERE id=? AND valid_to IS NULL", (stamp, old_id))
                written += 1
            summary = encoding.get("summary")
            if summary:
                summary = _name(summary, 2000)
                support = self._support(encoding.get("summary_support"), allowed)
                self._db.execute("INSERT INTO episode_summaries VALUES(?,?,?,?) ON CONFLICT(episode) DO UPDATE SET summary=excluded.summary,support_json=excluded.support_json,updated_at=excluded.updated_at", (rows[0][0], summary, _json(support), time.time()))
            for event_id, vector in (vectors or {}).items():
                if int(event_id) not in targets or not vector or len(vector) > 8192 or not all(math.isfinite(float(value)) for value in vector):
                    raise ValueError("invalid source embedding")
                self._db.execute("UPDATE event_sources SET vector_json=? WHERE id=?", (_json([float(value) for value in vector]), event_id))
            if complete:
                self._db.executemany("UPDATE event_sources SET encoded=1,encoding_error=NULL,retry_after=0 WHERE id=?", [(value,) for value in targets])
        return {"events": len(targets), "facts": written}

    def facts(self, *, limit: int = 32, active_only: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT f.*,s.canonical AS subject_name,o.canonical AS object_name FROM graph_facts f JOIN graph_entities s ON s.id=f.subject JOIN graph_entities o ON o.id=f.object "
                + ("WHERE f.valid_to IS NULL " if active_only else "") + "ORDER BY f.valid_from DESC LIMIT ?", (max(1, min(128, limit)),)).fetchall()
            return [{"fact_id": row["id"], "subject": row["subject_name"], "predicate": row["predicate"], "object": row["object_name"],
                     "relation_type": row["relation_type"], "authority": row["authority"], "epistemic_status": row["epistemic_status"],
                     "valid_from": row["valid_from"], "valid_to": row["valid_to"], "support": json.loads(row["support_json"])} for row in rows]

    def search(self, query: str, *, vector: Sequence[float] | None = None, start: float | None = None,
               end: float | None = None, modalities: Sequence[str] = (), limit: int = 8,
               conversation_only: bool = False, exclude_episode: str | None = None) -> dict[str, Any]:
        if len(query) > 1000 or any(value not in MODALITIES for value in modalities):
            raise ValueError("invalid bounded memory query")
        clauses, arguments = ["occurred_at<?"], [time.time() + 1 if end is None else float(end)]
        if start is not None:
            clauses.append("occurred_at>=?")
            arguments.append(float(start))
        if exclude_episode is not None:
            clauses.append("episode!=?")
            arguments.append(exclude_episode)
        if modalities:
            clauses.append("modality IN (" + ",".join("?" for _ in modalities) + ")")
            arguments.extend(modalities)
        if conversation_only:
            clauses.append("EXISTS(SELECT 1 FROM source_annotations a WHERE a.event_id=event_sources.id AND a.name='conversation_eligible' AND a.value_json='true')")
        where = " AND ".join(clauses)
        bounded = max(1, min(20, int(limit)))
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM event_sources WHERE {where}", arguments).fetchone()[0]
            terms = re.findall(r"\w+", query, flags=re.UNICODE)[:40]
            lexical, entity_anchors = [], []
            if terms:
                expression = " OR ".join('"' + value.replace('"', '""') + '"' for value in terms)
                lexical = [int(row[0]) for row in self._db.execute("SELECT event_sources.id FROM event_fts JOIN event_sources ON event_sources.id=event_fts.rowid "
                    f"WHERE event_fts MATCH ? AND {where} ORDER BY bm25(event_fts) LIMIT 128", [expression, *arguments])]
                entity_anchors = [str(row[0]) for row in self._db.execute("SELECT entity_id FROM graph_entity_fts WHERE graph_entity_fts MATCH ? ORDER BY bm25(graph_entity_fts) LIMIT 32", (expression,))]
            dense = []
            if vector:
                @lru_cache(maxsize=512)
                def similarity(stored: str) -> float:
                    try:
                        value = _cosine(vector, json.loads(stored))
                        return value if math.isfinite(value) else 0.0
                    except (ValueError, TypeError, OverflowError):
                        return 0.0
                self._db.create_function("memory_similarity", 1, similarity, deterministic=True)
                dense = [(int(row[0]), float(row[1])) for row in self._db.execute(
                    f"SELECT id,memory_similarity(vector_json) AS score FROM event_sources WHERE vector_json IS NOT NULL AND {where} ORDER BY score DESC LIMIT 128", arguments)]
            seed: dict[int, float] = defaultdict(float)
            for ranking in (lexical, [identity for identity, score in dense if score >= 0.25]):
                for rank, identity in enumerate(ranking):
                    seed[identity] += 1 / (60 + rank)

            # Build a bounded neighborhood from actual seed evidence. Every
            # supporting event passes the time/modality filter before traversal.
            event_cache: dict[int, sqlite3.Row | None] = {}
            def allowed(identity: int) -> bool:
                if identity not in event_cache:
                    event_cache[identity] = self._db.execute(f"SELECT id,occurred_at FROM event_sources WHERE id=? AND {where}", [identity, *arguments]).fetchone()
                return event_cache[identity] is not None

            eligible: dict[str, sqlite3.Row] = {}
            frontier = set(entity_anchors)
            if seed:
                placeholders = ",".join("?" for _ in seed)
                initial = self._db.execute(f"SELECT DISTINCT f.* FROM graph_facts f JOIN fact_sources s ON s.fact_id=f.id WHERE s.event_id IN ({placeholders}) LIMIT 512", tuple(seed)).fetchall()
                for row in initial:
                    if all(allowed(int(item["event_id"])) for item in json.loads(row["support_json"])):
                        eligible[row["id"]] = row
                        frontier.update((row["subject"], row["object"]))
            visited: set[str] = set()
            for _ in range(2):
                current = sorted(frontier - visited)[:200]
                if not current or len(eligible) >= 512:
                    break
                visited.update(current)
                placeholders = ",".join("?" for _ in current)
                neighbors = self._db.execute(f"SELECT * FROM graph_facts WHERE subject IN ({placeholders}) OR object IN ({placeholders}) ORDER BY valid_from DESC LIMIT 512", (*current, *current)).fetchall()
                frontier = set()
                for row in neighbors:
                    if row["id"] in eligible:
                        continue
                    if all(allowed(int(item["event_id"])) for item in json.loads(row["support_json"])):
                        eligible[row["id"]] = row
                        frontier.update((row["subject"], row["object"]))
                        if len(eligible) >= 512:
                            break
            adjacency: dict[str, set[str]] = defaultdict(set)
            entity_seed: dict[str, float] = defaultdict(float)
            for rank, identity in enumerate(entity_anchors):
                entity_seed[identity] += 1 / (60 + rank)
            for row in eligible.values():
                adjacency[row["subject"]].add(row["object"])
                adjacency[row["object"]].add(row["subject"])
                weight = sum(seed.get(item["event_id"], 0) for item in json.loads(row["support_json"]))
                entity_seed[row["subject"]] += weight
                entity_seed[row["object"]] += weight
            mass = sum(entity_seed.values())
            restart = {key: value / mass for key, value in entity_seed.items()} if mass else {}
            scores = dict(restart)
            for _ in range(20):
                following: dict[str, float] = defaultdict(float, {key: 0.15 * value for key, value in restart.items()})
                for entity, score in scores.items():
                    neighbors = adjacency.get(entity, set())
                    for neighbor in neighbors:
                        following[neighbor] += 0.85 * score / len(neighbors)
                scores = dict(following)
            graph: dict[int, float] = defaultdict(float)
            for row in eligible.values():
                weight = scores.get(row["subject"], 0) + scores.get(row["object"], 0)
                for item in json.loads(row["support_json"]):
                    graph[item["event_id"]] += weight
            combined = dict(seed)
            for rank, identity in enumerate(sorted(graph, key=lambda identity: graph[identity], reverse=True)[:128]):
                if graph[identity] > 0:
                    combined[identity] = combined.get(identity, 0) + 0.7 / (60 + rank)
            if not query.strip():
                chosen = self._db.execute(f"SELECT * FROM event_sources WHERE {where} ORDER BY occurred_at DESC,id DESC LIMIT ?", [*arguments, bounded]).fetchall()
            else:
                identities = sorted([identity for identity in combined if allowed(identity)], key=lambda identity: (combined[identity], event_cache[identity]["occurred_at"]), reverse=True)[:bounded]
                chosen = [self._db.execute("SELECT * FROM event_sources WHERE id=?", (identity,)).fetchone() for identity in identities]
            ids = {int(row["id"]) for row in chosen}
            relevant_facts = []
            for row in eligible.values():
                support = json.loads(row["support_json"])
                if not any(item["event_id"] in ids for item in support):
                    continue
                subject = self._db.execute("SELECT canonical FROM graph_entities WHERE id=?", (row["subject"],)).fetchone()
                object_ = self._db.execute("SELECT canonical FROM graph_entities WHERE id=?", (row["object"],)).fetchone()
                relevant_facts.append({"fact_id": row["id"], "subject": subject[0], "predicate": row["predicate"], "object": object_[0],
                    "relation_type": row["relation_type"], "authority": row["authority"], "epistemic_status": row["epistemic_status"],
                    "valid_from": row["valid_from"], "valid_to": row["valid_to"], "support": support})
            dense_ids = {identity for identity, score in dense if score >= 0.25}
            return {"schema": "robit.omni.episodic-memory.v1", "trust": "untrusted_historical_evidence", "events_in_window": total,
                "retrieval_complete": True, "semantic_query_available": vector is not None, "graph_neighborhood_limited": len(eligible) >= 512,
                "results": [{**self._event(row), "text": row["text"][:2000], "text_truncated": len(row["text"]) > 2000,
                    "retrieval_score": round(combined.get(int(row["id"]), 0), 6),
                    "channels": [name for name, present in (("lexical", int(row["id"]) in lexical), ("semantic", int(row["id"]) in dense_ids), ("graph", graph.get(int(row["id"]), 0) > 0)) if present]} for row in chosen],
                "facts": relevant_facts[:24], "source_is_current_visual_evidence": False}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            aggregate = self._db.execute("SELECT COUNT(*) AS events,COUNT(DISTINCT episode) AS episodes,SUM(encoded=0 OR vector_json IS NULL) AS pending,SUM(encoded=0) AS encoding_pending,SUM(vector_json IS NULL) AS vector_pending,SUM(vector_json IS NOT NULL) AS vectors FROM event_sources").fetchone()
            return {key: int(aggregate[key] or 0) for key in ("events", "episodes", "pending", "encoding_pending", "vector_pending", "vectors")} | {
                "entities": self._db.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0],
                "facts": self._db.execute("SELECT COUNT(*) FROM graph_facts").fetchone()[0],
                "active_facts": self._db.execute("SELECT COUNT(*) FROM graph_facts WHERE valid_to IS NULL").fetchone()[0],
                "modalities": {row[0]: row[1] for row in self._db.execute("SELECT modality,COUNT(*) FROM event_sources GROUP BY modality")},
            }

    def view(self, *, include_content: bool = False) -> dict[str, Any]:
        """The device overview is aggregate-only; own-scope inspection is bounded."""
        with self._lock:
            result: dict[str, Any] = {"available": True, "content_available": include_content, **self.stats()}
            if include_content:
                result["recent"] = [{**self._event(row), "text": row["text"][:700], "text_truncated": len(row["text"]) > 700}
                    for row in self._db.execute("SELECT * FROM event_sources ORDER BY occurred_at DESC,id DESC LIMIT 12")]
                result["recent_facts"] = [{**fact, "support": [{**source, "quote": source["quote"][:300], "quote_truncated": len(source["quote"]) > 300}
                    for source in fact["support"][:8]]} for fact in self.facts(limit=16, active_only=False)]
            return result
