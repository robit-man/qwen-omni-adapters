"""Append-only evidence and versioned derived memory backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_omni_adapters.virtual_memory.chunking import StructureAwareChunker
from qwen_omni_adapters.virtual_memory.models import (
    MEMORY_POLICIES,
    EvidenceChunk,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
)

SCHEMA_VERSION = 3
_DEFAULT_TTL = object()
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_.-]*)(?:\s+[A-Z][A-Za-z0-9_.-]*){0,3}\b")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return dot / norm if norm else 0.0


class ImmutableEvidenceStore:
    """Lossless corpus, hybrid indexes, and provenance-bearing memories.

    Evidence tables have database triggers that reject UPDATE and DELETE.  A
    correction is a new document version or a superseding derived memory, not
    mutation of history.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        chunker: StructureAwareChunker | None = None,
        embedder: Callable[[str], Sequence[float] | None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunker = chunker or StructureAwareChunker()
        self.embedder = embedder
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._prepare()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ImmutableEvidenceStore:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _prepare(self) -> None:
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    message_id TEXT,
                    source TEXT NOT NULL,
                    captured_at REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (document_id, version)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    message_id TEXT,
                    source TEXT NOT NULL,
                    captured_at REAL NOT NULL,
                    ordinal INTEGER NOT NULL,
                    token_start INTEGER NOT NULL,
                    token_end INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    byte_start INTEGER NOT NULL,
                    byte_end INTEGER NOT NULL,
                    parent_kind TEXT,
                    parent_name TEXT,
                    previous_chunk_id TEXT,
                    next_chunk_id TEXT,
                    content_hash TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (document_id, version)
                        REFERENCES documents (document_id, version)
                );
                CREATE INDEX IF NOT EXISTS chunks_document
                    ON chunks (document_id, version, ordinal);
                CREATE INDEX IF NOT EXISTS chunks_recency
                    ON chunks (captured_at DESC);
                CREATE INDEX IF NOT EXISTS chunks_parent
                    ON chunks (parent_name, parent_kind);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    original_text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                );
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT NOT NULL,
                    symbol_folded TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (symbol, kind, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                );
                CREATE INDEX IF NOT EXISTS symbols_folded
                    ON symbols (symbol_folded);
                CREATE TABLE IF NOT EXISTS code_edges (
                    source_symbol TEXT NOT NULL,
                    source_folded TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    target_symbol TEXT NOT NULL,
                    target_folded TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (
                        source_symbol, predicate, target_symbol, chunk_id
                    ),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                );
                CREATE INDEX IF NOT EXISTS code_edges_source
                    ON code_edges (source_folded, predicate);
                CREATE INDEX IF NOT EXISTS code_edges_target
                    ON code_edges (target_folded, predicate);
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    canonical TEXT NOT NULL,
                    canonical_folded TEXT NOT NULL UNIQUE,
                    aliases_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunk_entities (
                    chunk_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, entity_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    edge_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    valid_from REAL NOT NULL,
                    valid_to REAL,
                    version INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (subject_id) REFERENCES entities (entity_id),
                    FOREIGN KEY (object_id) REFERENCES entities (entity_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                );
                CREATE INDEX IF NOT EXISTS relationships_subject
                    ON relationships (subject_id, predicate);
                CREATE INDEX IF NOT EXISTS relationships_object
                    ON relationships (object_id, predicate);
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    memory_class TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    valid_from REAL NOT NULL,
                    valid_to REAL,
                    ttl_seconds REAL,
                    importance REAL NOT NULL,
                    version INTEGER NOT NULL,
                    supersedes TEXT,
                    compression_generation INTEGER NOT NULL,
                    verified INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (supersedes) REFERENCES memories (memory_id)
                );
                CREATE INDEX IF NOT EXISTS memories_active
                    ON memories (memory_class, subject, valid_to, created_at DESC);
                CREATE TABLE IF NOT EXISTS memory_provenance (
                    memory_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    exact INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, ordinal),
                    FOREIGN KEY (memory_id) REFERENCES memories (memory_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                );
                CREATE TRIGGER IF NOT EXISTS documents_no_update
                BEFORE UPDATE ON documents BEGIN
                    SELECT RAISE(ABORT, 'immutable evidence');
                END;
                CREATE TRIGGER IF NOT EXISTS documents_no_delete
                BEFORE DELETE ON documents BEGIN
                    SELECT RAISE(ABORT, 'immutable evidence');
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_no_update
                BEFORE UPDATE ON chunks BEGIN
                    SELECT RAISE(ABORT, 'immutable evidence');
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_no_delete
                BEFORE DELETE ON chunks BEGIN
                    SELECT RAISE(ABORT, 'immutable evidence');
                END;
                """
            )
            provenance_columns = {
                str(row["name"])
                for row in self._db.execute("PRAGMA table_info(memory_provenance)")
            }
            if "ordinal" not in provenance_columns:
                self._db.execute(
                    "ALTER TABLE memory_provenance ADD COLUMN ordinal INTEGER NOT NULL DEFAULT 0"
                )
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._db.commit()

    def ingest(
        self,
        text: str,
        *,
        source: str,
        document_id: str | None = None,
        message_id: str | None = None,
        version: str | None = None,
        captured_at: float | None = None,
        media_type: str | None = None,
        kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        entities: Iterable[str] = (),
    ) -> list[EvidenceChunk]:
        if not isinstance(text, str) or not text:
            raise ValueError("evidence text must be non-empty")
        normalized_source = str(source or "").strip()
        if not normalized_source:
            raise ValueError("evidence source is required")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = str(document_id or hashlib.sha256(normalized_source.encode()).hexdigest()[:20])
        doc_version = str(version or digest[:16])
        timestamp = float(captured_at if captured_at is not None else time.time())
        document_metadata = dict(metadata or {})
        drafts = self.chunker.chunk(
            text,
            source=normalized_source,
            media_type=media_type,
            kind=kind,
        )
        if not drafts:
            raise ValueError("evidence chunker produced no chunks")
        chunk_ids = [
            hashlib.sha256(
                f"{doc_id}\0{doc_version}\0{draft.char_start}\0{draft.char_end}\0"
                f"{hashlib.sha256(draft.text.encode()).hexdigest()}".encode()
            ).hexdigest()[:24]
            for draft in drafts
        ]
        explicit_entities = {value.strip() for value in entities if value.strip()}
        with self._lock:
            existing = self._db.execute(
                "SELECT content_hash FROM documents WHERE document_id = ? AND version = ?",
                (doc_id, doc_version),
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] != digest:
                    raise ValueError("document version already exists with different content")
                return self.document_chunks(doc_id, doc_version)
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    """
                    INSERT INTO documents (
                        document_id, version, message_id, source, captured_at,
                        content_hash, original_text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        doc_version,
                        message_id,
                        normalized_source,
                        timestamp,
                        digest,
                        text,
                        _json(document_metadata),
                    ),
                )
                for ordinal, (draft, chunk_id) in enumerate(zip(drafts, chunk_ids, strict=True)):
                    chunk_hash = hashlib.sha256(draft.text.encode("utf-8")).hexdigest()
                    chunk_metadata = {**document_metadata, **draft.metadata}
                    self._db.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, document_id, version, message_id, source,
                            captured_at, ordinal, token_start, token_end,
                            char_start, char_end, byte_start, byte_end,
                            parent_kind, parent_name, previous_chunk_id,
                            next_chunk_id, content_hash, original_text, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            doc_id,
                            doc_version,
                            message_id,
                            normalized_source,
                            timestamp,
                            ordinal,
                            draft.token_start,
                            draft.token_end,
                            draft.char_start,
                            draft.char_end,
                            draft.byte_start,
                            draft.byte_end,
                            draft.parent_kind,
                            draft.parent_name,
                            chunk_ids[ordinal - 1] if ordinal else None,
                            chunk_ids[ordinal + 1] if ordinal + 1 < len(chunk_ids) else None,
                            chunk_hash,
                            draft.text,
                            _json(chunk_metadata),
                        ),
                    )
                    self._db.execute(
                        "INSERT INTO chunk_fts (chunk_id, original_text) VALUES (?, ?)",
                        (chunk_id, draft.text),
                    )
                    for symbol, symbol_kind in draft.symbols:
                        self._db.execute(
                            "INSERT OR IGNORE INTO symbols VALUES (?, ?, ?, ?)",
                            (symbol, symbol.casefold(), symbol_kind, chunk_id),
                        )
                    for source_symbol, predicate, target_symbol in draft.code_edges:
                        self._db.execute(
                            "INSERT OR IGNORE INTO code_edges VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                source_symbol,
                                source_symbol.casefold(),
                                predicate,
                                target_symbol,
                                target_symbol.casefold(),
                                chunk_id,
                            ),
                        )
                    inferred_entities = {
                        match.group(0).strip() for match in ENTITY_RE.finditer(draft.text)
                    }
                    for entity in sorted(explicit_entities | inferred_entities):
                        entity_id = self._ensure_entity_locked(entity)
                        self._db.execute(
                            "INSERT OR IGNORE INTO chunk_entities VALUES (?, ?)",
                            (chunk_id, entity_id),
                        )
                    if self.embedder is not None:
                        vector = self.embedder(draft.text)
                        if vector:
                            values = [float(value) for value in vector]
                            self._db.execute(
                                "INSERT INTO embeddings VALUES (?, ?, ?)",
                                (chunk_id, len(values), _json(values)),
                            )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.document_chunks(doc_id, doc_version)

    def _ensure_entity_locked(self, canonical: str, aliases: Iterable[str] = ()) -> str:
        folded = canonical.casefold()
        row = self._db.execute(
            "SELECT entity_id, aliases_json FROM entities WHERE canonical_folded = ?",
            (folded,),
        ).fetchone()
        if row is not None:
            # Entity rows are indexes, not evidence. Aliases may grow without
            # changing any original source record.
            known = set(json.loads(row["aliases_json"]))
            additions = {alias for alias in aliases if alias}
            if not additions.issubset(known):
                self._db.execute(
                    "UPDATE entities SET aliases_json = ? WHERE entity_id = ?",
                    (_json(sorted(known | additions)), row["entity_id"]),
                )
            return str(row["entity_id"])
        entity_id = hashlib.sha256(folded.encode()).hexdigest()[:20]
        self._db.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?)",
            (entity_id, canonical, folded, _json(sorted(set(aliases)))),
        )
        return entity_id

    def add_entity(self, canonical: str, *, aliases: Iterable[str] = ()) -> str:
        name = str(canonical or "").strip()
        if not name:
            raise ValueError("entity canonical name is required")
        with self._lock:
            entity_id = self._ensure_entity_locked(name, aliases)
            self._db.commit()
            return entity_id

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        chunk_id: str,
        valid_from: float | None = None,
        valid_to: float | None = None,
        version: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        relation = str(predicate or "").strip()
        if not relation:
            raise ValueError("relationship predicate is required")
        with self._lock:
            if self.get_chunk(chunk_id) is None:
                raise KeyError(f"unknown provenance chunk: {chunk_id}")
            subject_id = self._ensure_entity_locked(subject)
            object_id = self._ensure_entity_locked(object_)
            timestamp = float(valid_from if valid_from is not None else time.time())
            edge_id = hashlib.sha256(
                f"{subject_id}\0{relation}\0{object_id}\0{chunk_id}\0{version}".encode()
            ).hexdigest()[:24]
            self._db.execute(
                "INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge_id,
                    subject_id,
                    relation,
                    object_id,
                    chunk_id,
                    timestamp,
                    valid_to,
                    version,
                    _json(dict(metadata or {})),
                ),
            )
            self._db.commit()
            return edge_id

    def write_memory(
        self,
        memory_class: MemoryClass | str,
        subject: str,
        content: str,
        *,
        provenance: Iterable[ProvenancePointer],
        importance: float | None = None,
        ttl_seconds: float | None | object = _DEFAULT_TTL,
        supersedes: str | None = None,
        compression_generation: int = 0,
        verified: bool | None = None,
        valid_from: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        selected_class = MemoryClass(memory_class)
        policy = MEMORY_POLICIES[selected_class]
        selected_importance = (
            policy.default_importance if importance is None else float(importance)
        )
        selected_ttl = (
            policy.default_ttl_seconds if ttl_seconds is _DEFAULT_TTL else ttl_seconds
        )
        if selected_ttl is not None:
            selected_ttl = float(selected_ttl)
            if not math.isfinite(selected_ttl) or selected_ttl <= 0:
                raise ValueError("memory TTL must be a positive finite duration")
        normalized_subject = str(subject or "").strip()
        normalized_content = str(content or "").strip()
        pointers = tuple(provenance)
        if not normalized_subject or not normalized_content:
            raise ValueError("memory subject and content are required")
        if not pointers:
            raise ValueError("derived memory requires source provenance")
        timestamp = time.time()
        starts = float(valid_from if valid_from is not None else timestamp)
        with self._lock:
            for pointer in pointers:
                chunk = self.get_chunk(pointer.chunk_id)
                if chunk is None:
                    raise KeyError(f"unknown provenance chunk: {pointer.chunk_id}")
                if not 0 <= pointer.char_start <= pointer.char_end <= len(chunk.original_text):
                    raise ValueError("provenance offsets are outside the source chunk")
            previous = None
            if supersedes is not None:
                previous = self._db.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (supersedes,)
                ).fetchone()
                if previous is None:
                    raise KeyError(f"unknown superseded memory: {supersedes}")
                if previous["valid_to"] is not None:
                    raise ValueError("memory has already been superseded")
            current = self._db.execute(
                """
                SELECT MAX(version) AS version FROM memories
                WHERE memory_class = ? AND subject = ?
                """,
                (selected_class.value, normalized_subject),
            ).fetchone()
            memory_version = int(current["version"] or 0) + 1
            memory_id = uuid.uuid4().hex[:24]
            is_verified = bool(pointers) if verified is None else bool(verified)
            try:
                self._db.execute("BEGIN IMMEDIATE")
                if previous is not None:
                    self._db.execute(
                        "UPDATE memories SET valid_to = ? WHERE memory_id = ?",
                        (starts, supersedes),
                    )
                self._db.execute(
                    """
                    INSERT INTO memories VALUES (
                        ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        memory_id,
                        selected_class.value,
                        normalized_subject,
                        normalized_content,
                        timestamp,
                        starts,
                        selected_ttl,
                        max(0.0, min(1.0, selected_importance)),
                        memory_version,
                        supersedes,
                        max(0, int(compression_generation)),
                        int(is_verified),
                        _json(dict(metadata or {})),
                    ),
                )
                self._db.executemany(
                    """
                    INSERT INTO memory_provenance (
                        memory_id, ordinal, chunk_id, char_start, char_end, exact
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            memory_id,
                            ordinal,
                            pointer.chunk_id,
                            pointer.char_start,
                            pointer.char_end,
                            int(pointer.exact),
                        )
                        for ordinal, pointer in enumerate(pointers)
                    ],
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        record = self.get_memory(memory_id)
        if record is None:  # pragma: no cover - database invariant
            raise RuntimeError("memory write was not readable")
        return record

    def get_chunk(self, chunk_id: str) -> EvidenceChunk | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._chunk(row) if row is not None else None

    def document_chunks(self, document_id: str, version: str) -> list[EvidenceChunk]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM chunks WHERE document_id = ? AND version = ?
                ORDER BY ordinal
                """,
                (document_id, version),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def expand(self, chunk_id: str, *, neighbors: int = 0) -> list[EvidenceChunk]:
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ? AND version = ? AND ordinal BETWEEN ? AND ?
                ORDER BY ordinal
                """,
                (
                    chunk.document_id,
                    chunk.version,
                    max(0, chunk.ordinal - max(0, neighbors)),
                    chunk.ordinal + max(0, neighbors),
                ),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def exact_search(self, value: str, *, limit: int = 200) -> list[EvidenceChunk]:
        needle = str(value or "").strip()
        if not needle:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM chunks WHERE instr(lower(original_text), lower(?)) > 0
                ORDER BY captured_at DESC LIMIT ?
                """,
                (needle, max(1, limit)),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def lexical_search(self, query: str, *, limit: int = 200) -> list[tuple[EvidenceChunk, float]]:
        terms = re.findall(r"[\w.-]+", query, re.UNICODE)
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:32])
        try:
            with self._lock:
                rows = self._db.execute(
                    """
                    SELECT chunks.*, bm25(chunk_fts) AS rank
                    FROM chunk_fts JOIN chunks USING (chunk_id)
                    WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?
                    """,
                    (expression, max(1, limit)),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            (self._chunk(row), 1.0 / (1.0 + index))
            for index, row in enumerate(rows)
        ]

    def symbol_search(self, symbol: str, *, limit: int = 200) -> list[EvidenceChunk]:
        folded = str(symbol or "").strip().casefold()
        if not folded:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT DISTINCT chunks.* FROM symbols
                JOIN chunks USING (chunk_id)
                WHERE symbol_folded = ? OR symbol_folded LIKE ?
                ORDER BY captured_at DESC LIMIT ?
                """,
                (folded, f"{folded}%", max(1, limit)),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def recent(self, *, limit: int = 200) -> list[EvidenceChunk]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM chunks ORDER BY captured_at DESC, ordinal DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def code_search(
        self, symbol: str, *, max_hops: int = 2, limit: int = 200
    ) -> list[tuple[EvidenceChunk, int, tuple[str, ...]]]:
        """Find definitions, callers, callees, imports, and base classes."""

        folded = str(symbol or "").strip().casefold()
        if not folded:
            return []
        frontier = {folded}
        visited = set(frontier)
        hits: dict[str, tuple[int, set[str]]] = {}
        with self._lock:
            for distance in range(0, max(0, min(3, max_hops)) + 1):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                definitions = self._db.execute(
                    f"""
                    SELECT chunk_id FROM symbols
                    WHERE symbol_folded IN ({placeholders})
                    """,
                    tuple(frontier),
                ).fetchall()
                for row in definitions:
                    current = hits.setdefault(str(row["chunk_id"]), (distance, set()))
                    current[1].add("defines")
                edges = self._db.execute(
                    f"""
                    SELECT * FROM code_edges
                    WHERE source_folded IN ({placeholders})
                       OR target_folded IN ({placeholders})
                    """,
                    (*frontier, *frontier),
                ).fetchall()
                next_frontier: set[str] = set()
                for edge in edges:
                    chunk_id = str(edge["chunk_id"])
                    current = hits.setdefault(chunk_id, (distance, set()))
                    current[1].add(str(edge["predicate"]))
                    for candidate in (
                        str(edge["source_folded"]),
                        str(edge["target_folded"]),
                    ):
                        if candidate not in visited:
                            visited.add(candidate)
                            next_frontier.add(candidate)
                frontier = next_frontier
            if not hits:
                return []
            placeholders = ",".join("?" for _ in hits)
            rows = self._db.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(hits),
            ).fetchall()
        found = [
            (
                self._chunk(row),
                hits[str(row["chunk_id"])][0],
                tuple(sorted(hits[str(row["chunk_id"])][1])),
            )
            for row in rows
        ]
        found.sort(key=lambda item: (item[1], -item[0].captured_at))
        return found[: max(1, limit)]

    def metadata_search(
        self, filters: Mapping[str, Any], *, limit: int = 200
    ) -> list[EvidenceChunk]:
        if not filters:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM chunks ORDER BY captured_at DESC"
            ).fetchall()
        found = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if all(metadata.get(key) == value for key, value in filters.items()):
                found.append(self._chunk(row))
                if len(found) >= limit:
                    break
        return found

    def dense_search(
        self, query_vector: Sequence[float], *, limit: int = 200
    ) -> list[tuple[EvidenceChunk, float]]:
        if not query_vector:
            return []
        with self._lock:
            rows = self._db.execute(
                """
                SELECT chunks.*, embeddings.vector_json
                FROM embeddings JOIN chunks USING (chunk_id)
                WHERE embeddings.dimensions = ?
                """,
                (len(query_vector),),
            ).fetchall()
        ranked = [
            (self._chunk(row), _cosine(query_vector, json.loads(row["vector_json"])))
            for row in rows
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(1, limit)]

    def entity_search(self, query: str, *, limit: int = 200) -> list[EvidenceChunk]:
        folded = str(query or "").casefold()
        with self._lock:
            entities = self._db.execute("SELECT * FROM entities").fetchall()
            entity_ids = [
                row["entity_id"]
                for row in entities
                if row["canonical_folded"] in folded
                or any(str(alias).casefold() in folded for alias in json.loads(row["aliases_json"]))
            ]
            if not entity_ids:
                return []
            placeholders = ",".join("?" for _ in entity_ids)
            rows = self._db.execute(
                f"""
                SELECT DISTINCT chunks.* FROM chunk_entities
                JOIN chunks USING (chunk_id)
                WHERE entity_id IN ({placeholders})
                ORDER BY captured_at DESC LIMIT ?
                """,
                (*entity_ids, max(1, limit)),
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def graph_search(
        self, query: str, *, max_hops: int = 2, limit: int = 200
    ) -> list[tuple[EvidenceChunk, int]]:
        folded = str(query or "").casefold()
        with self._lock:
            entities = self._db.execute("SELECT * FROM entities").fetchall()
            frontier = {
                str(row["entity_id"])
                for row in entities
                if row["canonical_folded"] in folded
                or any(str(alias).casefold() in folded for alias in json.loads(row["aliases_json"]))
            }
            visited = set(frontier)
            hits: dict[str, int] = {}
            for distance in range(1, max(0, min(3, max_hops)) + 1):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                edges = self._db.execute(
                    f"""
                    SELECT * FROM relationships
                    WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
                    """,
                    (*frontier, *frontier),
                ).fetchall()
                next_frontier: set[str] = set()
                for edge in edges:
                    hits.setdefault(str(edge["chunk_id"]), distance)
                    for entity_id in (str(edge["subject_id"]), str(edge["object_id"])):
                        if entity_id not in visited:
                            visited.add(entity_id)
                            next_frontier.add(entity_id)
                frontier = next_frontier
            if not hits:
                return []
            placeholders = ",".join("?" for _ in hits)
            rows = self._db.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(hits),
            ).fetchall()
        ranked = [(self._chunk(row), hits[str(row["chunk_id"])]) for row in rows]
        ranked.sort(key=lambda item: (item[1], -item[0].captured_at))
        return ranked[: max(1, limit)]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            pointers = self._db.execute(
                """
                SELECT * FROM memory_provenance WHERE memory_id = ?
                ORDER BY ordinal
                """,
                (memory_id,),
            ).fetchall()
        return self._memory(row, pointers)

    def active_memories(
        self,
        *,
        classes: Iterable[MemoryClass | str] | None = None,
        subject: str | None = None,
        now: float | None = None,
    ) -> list[MemoryRecord]:
        timestamp = float(now if now is not None else time.time())
        clauses = ["valid_from <= ?", "(valid_to IS NULL OR valid_to > ?)"]
        values: list[Any] = [timestamp, timestamp]
        selected = [MemoryClass(value).value for value in classes] if classes else []
        if selected:
            clauses.append(f"memory_class IN ({','.join('?' for _ in selected)})")
            values.extend(selected)
        if subject is not None:
            clauses.append("subject = ?")
            values.append(subject)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                "ORDER BY importance DESC, created_at DESC",
                values,
            ).fetchall()
        found = []
        for row in rows:
            ttl = row["ttl_seconds"]
            if ttl is not None and float(row["created_at"]) + float(ttl) <= timestamp:
                continue
            record = self.get_memory(str(row["memory_id"]))
            if record is not None:
                found.append(record)
        return found

    def reconstruct(self, memory_id: str) -> list[tuple[ProvenancePointer, str]]:
        memory = self.get_memory(memory_id)
        if memory is None:
            return []
        recovered = []
        for pointer in memory.provenance:
            chunk = self.get_chunk(pointer.chunk_id)
            if chunk is not None:
                recovered.append(
                    (
                        pointer,
                        chunk.original_text[pointer.char_start : pointer.char_end],
                    )
                )
        return recovered

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                name: int(self._db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                for name in (
                    "documents",
                    "chunks",
                    "embeddings",
                    "symbols",
                    "code_edges",
                    "entities",
                    "relationships",
                    "memories",
                )
            }

    @staticmethod
    def _chunk(row: sqlite3.Row) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            message_id=str(row["message_id"]) if row["message_id"] is not None else None,
            source=str(row["source"]),
            version=str(row["version"]),
            captured_at=float(row["captured_at"]),
            ordinal=int(row["ordinal"]),
            token_start=int(row["token_start"]),
            token_end=int(row["token_end"]),
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            byte_start=int(row["byte_start"]),
            byte_end=int(row["byte_end"]),
            parent_kind=str(row["parent_kind"]) if row["parent_kind"] else None,
            parent_name=str(row["parent_name"]) if row["parent_name"] else None,
            previous_chunk_id=(
                str(row["previous_chunk_id"]) if row["previous_chunk_id"] else None
            ),
            next_chunk_id=str(row["next_chunk_id"]) if row["next_chunk_id"] else None,
            content_hash=str(row["content_hash"]),
            original_text=str(row["original_text"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _memory(row: sqlite3.Row, pointers: Sequence[sqlite3.Row]) -> MemoryRecord:
        return MemoryRecord(
            memory_id=str(row["memory_id"]),
            memory_class=MemoryClass(row["memory_class"]),
            subject=str(row["subject"]),
            content=str(row["content"]),
            created_at=float(row["created_at"]),
            valid_from=float(row["valid_from"]),
            valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
            ttl_seconds=(
                float(row["ttl_seconds"]) if row["ttl_seconds"] is not None else None
            ),
            importance=float(row["importance"]),
            version=int(row["version"]),
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            compression_generation=int(row["compression_generation"]),
            verified=bool(row["verified"]),
            provenance=tuple(
                ProvenancePointer(
                    chunk_id=str(pointer["chunk_id"]),
                    char_start=int(pointer["char_start"]),
                    char_end=int(pointer["char_end"]),
                    exact=bool(pointer["exact"]),
                )
                for pointer in pointers
            ),
            metadata=json.loads(row["metadata_json"]),
        )
