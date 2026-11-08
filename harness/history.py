"""Durable source events and rebuildable retrieval views for the device owner.

Saving an event never calls a model. Embeddings and extractive hour/day views
are derived state; neither consolidation nor cache decay deletes source events.
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
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def local_timezone() -> str:
    """Use an IANA zone so calendar boundaries retain DST transitions."""
    candidates = [os.environ.get("TZ", "").removeprefix(":")]
    try:
        candidates.append(Path("/etc/timezone").read_text().strip())
    except OSError:
        pass
    try:
        candidates.append(str(Path("/etc/localtime").resolve()).split("zoneinfo/")[1])
    except (OSError, IndexError):
        pass
    for name in candidates:
        try:
            ZoneInfo(name)
            return name
        except (ValueError, KeyError):
            continue
    return "UTC"


def time_window(
    period: str = "all", *, now: float | None = None, zone: str | None = None,
    start_at: str | None = None, end_at: str | None = None,
) -> dict[str, Any]:
    """Resolve typed temporal intent, never classify a natural-language query."""
    instant = time.time() if now is None else now
    name = zone or local_timezone()
    try:
        tz = ZoneInfo(name)
    except (ValueError, KeyError) as error:
        raise ValueError("timezone must be an installed IANA zone") from error
    current = datetime.fromtimestamp(instant, tz)
    midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = None
    if period == "all":
        end = current
    elif period == "today":
        start, end = midnight, current
    elif period == "yesterday":
        start, end = midnight - timedelta(days=1), midnight
    elif period == "last_week":
        end = midnight - timedelta(days=midnight.weekday())
        start = end - timedelta(days=7)
    elif period == "last_7_days":
        start, end = datetime.fromtimestamp(instant - 7 * 86400, tz), current
    elif period == "past_hour":
        start, end = datetime.fromtimestamp(instant - 3600, tz), current
    elif period == "hour_ago":
        start = datetime.fromtimestamp(instant - 5400, tz)
        end = datetime.fromtimestamp(instant - 1800, tz)
    elif period == "range":
        if not start_at or not end_at:
            raise ValueError("range requires start_at and end_at with explicit UTC offsets")
        start, end = (datetime.fromisoformat(s.replace("Z", "+00:00"))
                      for s in (start_at, end_at))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("range timestamps require explicit UTC offsets")
        if start.timestamp() >= end.timestamp():
            raise ValueError("start_at must precede end_at")
    else:
        raise ValueError("unknown history period")
    if period != "range" and (start_at is not None or end_at is not None):
        raise ValueError("start_at/end_at are only valid with period=range")
    return {"period": period, "timezone": name,
            "start": start.timestamp() if start else None,
            "end": end.timestamp() if end else None,
            "start_at": start.isoformat() if start else None,
            "end_at": end.isoformat() if end else None,
            "end_exclusive": True, "as_of": current.isoformat(),
            "interpretation": ("previous Monday through Sunday" if period == "last_week"
                               else "30–90 minutes before now" if period == "hour_ago"
                               else period)}


class ConversationHistory:
    """One device-owner archive. Caller enforces owner/session authorization."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, timeout=3, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS history_events (
                id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL, kind TEXT NOT NULL, occurred_at REAL NOT NULL,
                recorded_at REAL NOT NULL, source_json TEXT NOT NULL,
                content_sha256 TEXT NOT NULL, vector BLOB,
                retry_after REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS history_time ON history_events(occurred_at,id);
            CREATE INDEX IF NOT EXISTS history_pending ON history_events(retry_after)
                WHERE vector IS NULL;
            CREATE TABLE IF NOT EXISTS history_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS history_links (
                event_id INTEGER PRIMARY KEY,memory_id INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                text, content='history_events', content_rowid='id', tokenize='unicode61');
            CREATE TRIGGER IF NOT EXISTS history_fts_insert AFTER INSERT ON history_events
            BEGIN INSERT INTO history_fts(rowid,text) VALUES(new.id,new.text); END;
            CREATE TABLE IF NOT EXISTS history_buckets (
                bucket TEXT PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL,
                dirty INTEGER NOT NULL DEFAULT 1, source_ids TEXT,
                source_sha256 TEXT, excerpt TEXT, updated_at REAL
            );
        """)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if "speech_addressee" not in {r[1] for r in self._db.execute("PRAGMA table_info(history_events)")}:
                self._db.execute("ALTER TABLE history_events ADD COLUMN speech_addressee TEXT NOT NULL DEFAULT 'UNKNOWN'")
        os.chmod(self.path, 0o600)
        self._migrate_legacy()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _buckets(self, stamp: float) -> None:
        day = datetime.fromtimestamp(stamp, ZoneInfo(local_timezone())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        for key, start, end in [
            (f"hour:{int(stamp // 3600)}", stamp // 3600 * 3600, stamp // 3600 * 3600 + 3600),
            (f"day:{day.isoformat()}", day.timestamp(), (day + timedelta(days=1)).timestamp()),
        ]:
            self._db.execute("INSERT INTO history_buckets(bucket,start,end) VALUES(?,?,?) "
                             "ON CONFLICT(bucket) DO UPDATE SET dirty=1", (key, start, end))

    def _migrate_legacy(self) -> None:
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self._db.execute("SELECT 1 FROM history_meta WHERE key='legacy_import_complete'").fetchone():
                return
            if not self._db.execute("SELECT 1 FROM sqlite_master WHERE name='memories'").fetchone():
                self._db.execute("INSERT INTO history_meta VALUES('legacy_import_complete','1')")
                return
            # Legacy consolidation may have replaced text without recording the
            # replacement time. Preserve that limitation, never invent an event.
            for row in self._db.execute("SELECT * FROM memories WHERE NOT EXISTS "
                                        "(SELECT 1 FROM history_events WHERE event_key='legacy:'||memories.id)"):
                self._db.execute(
                    "INSERT INTO history_events(event_key,text,kind,occurred_at,recorded_at,"
                    "source_json,content_sha256,vector) VALUES(?,?,?,?,?,?,?,?)",
                    (f"legacy:{row['id']}", row["text"], row["kind"], row["created_at"], time.time(),
                     json.dumps({"type": "legacy_cache", "legacy_id": row["id"],
                                 "timestamp_uncertain": True}),
                     hashlib.sha256(row["text"].encode()).hexdigest(), row["vector"]))
                self._buckets(float(row["created_at"]))
            self._db.execute("INSERT INTO history_meta VALUES('legacy_import_complete','1')")

    def append(self, text: str, *, kind: str = "turn", occurred_at: float | None = None,
               source: dict[str, Any] | None = None, event_key: str | None = None) -> int:
        text = str(text).strip()
        if not text or len(text) > 100_000:
            raise ValueError("history text must contain 1–100000 characters")
        stamp = time.time() if occurred_at is None else float(occurred_at)
        if not math.isfinite(stamp):
            raise ValueError("history timestamp must be finite")
        key = event_key or uuid.uuid4().hex
        addressee = str((source or {}).get("speech_addressee") or "UNKNOWN").upper()
        if addressee not in {"SELF", "OTHER", "AMBIGUOUS", "UNKNOWN"}:
            raise ValueError("invalid speech addressee")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute("SELECT id,content_sha256 FROM history_events WHERE event_key=?",
                                        (key,)).fetchone()
            digest = hashlib.sha256(text.encode()).hexdigest()
            if existing:
                if existing["content_sha256"] != digest:
                    raise ValueError("event key already identifies different source content")
                return int(existing["id"])
            cursor = self._db.execute(
                "INSERT INTO history_events(event_key,text,kind,occurred_at,recorded_at,"
                "source_json,content_sha256,speech_addressee) VALUES(?,?,?,?,?,?,?,?)",
                (key, text, kind, stamp, time.time(), json.dumps(source or {"type": "voice_event"}), digest, addressee))
            self._buckets(stamp)
            return int(cursor.lastrowid)

    def vector(self, event_id: int, vector: list[float] | None) -> None:
        if vector is not None and (not vector or not all(math.isfinite(v) for v in vector)):
            raise ValueError("invalid history embedding")
        with self._lock, self._db:
            self._db.execute("UPDATE history_events SET vector=?,retry_after=? WHERE id=?",
                             (json.dumps(vector).encode() if vector else None,
                              0 if vector else time.time() + 60, event_id))

    def pending(self) -> tuple[int, str, str] | None:
        with self._lock:
            row = self._db.execute("SELECT id,text,kind FROM history_events WHERE vector IS NULL "
                                   "AND retry_after<=? ORDER BY id LIMIT 1", (time.time(),)).fetchone()
        return (int(row["id"]), row["text"], row["kind"]) if row else None

    def link(self, event_id: int, memory_id: int) -> None:
        """A derived semantic cluster link; never overwrite the source event."""
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO history_links VALUES(?,?)", (event_id, memory_id))

    def rebuild_views(self) -> None:
        """Recover derived lexical/temporal views from retained source records."""
        with self._lock, self._db:
            self._db.execute("INSERT INTO history_fts(history_fts) VALUES('rebuild')")
            for row in self._db.execute("SELECT occurred_at FROM history_events"):
                self._buckets(row[0])

    def consolidate(self, limit: int = 8) -> int:
        """Rebuild bounded extractive views; every line retains a source ID."""
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            buckets = self._db.execute("SELECT * FROM history_buckets WHERE dirty=1 LIMIT ?",
                                       (limit,)).fetchall()
            for bucket in buckets:
                rows = self._db.execute("SELECT id,content_sha256 FROM history_events "
                                        "WHERE occurred_at>=? AND occurred_at<? ORDER BY occurred_at,id",
                                        (bucket["start"], bucket["end"])).fetchall()
                ids = [row["id"] for row in rows]
                digest = hashlib.sha256(json.dumps(
                    [(r["id"], r["content_sha256"]) for r in rows]).encode()).hexdigest()
                excerpts = self._db.execute("SELECT id,substr(text,1,500) AS text FROM history_events "
                                            "WHERE occurred_at>=? AND occurred_at<? ORDER BY occurred_at,id LIMIT 12",
                                            (bucket["start"], bucket["end"])).fetchall()
                excerpt = "\n".join(f"[event:{r['id']}] {r['text']}" for r in excerpts)
                self._db.execute("UPDATE history_buckets SET dirty=0,source_ids=?,source_sha256=?,"
                                 "excerpt=?,updated_at=? WHERE bucket=?",
                                 (json.dumps(ids), digest, excerpt, time.time(), bucket["bucket"]))
            return len(buckets)

    def search(self, query: str = "", *, period: str = "all", limit: int = 8,
               now: float | None = None, zone: str | None = None,
               start_at: str | None = None, end_at: str | None = None,
               vector: list[float] | None = None, before_id: int | None = None,
               kind: str = "conversation", view: str = "search") -> dict[str, Any]:
        window = time_window(period, now=now, zone=zone, start_at=start_at, end_at=end_at)
        if view not in {"search", "overview"}:
            raise ValueError("view must be overview or search")
        query = query.strip()
        if view == "overview":
            query = ""
        limit = max(1, min(20, int(limit)))
        where, args = ["occurred_at<?"], [window["end"]]
        if window["start"] is not None:
            where.append("occurred_at>=?")
            args.append(window["start"])
        if before_id is not None:
            with self._lock:
                boundary = self._db.execute("SELECT occurred_at FROM history_events WHERE id=?",
                                            (int(before_id),)).fetchone()
            if boundary is None:
                raise ValueError("unknown history cursor")
            where.append("(occurred_at<? OR (occurred_at=? AND id<?))")
            args.extend([boundary[0], boundary[0], int(before_id)])
        temporal_clause = " AND ".join(where)
        if kind == "conversation":
            where.append("kind!='sound' AND (speech_addressee='SELF' OR "
                         "(speech_addressee='UNKNOWN' AND kind!='heard'))")
        elif kind == "sound":
            where.append("kind='sound'")
        elif kind != "all":
            raise ValueError("kind must be conversation, sound or all")
        clause = " AND ".join(where)
        with self._lock:
            unattributed = self._db.execute(
                f"SELECT count(*) FROM history_events WHERE {temporal_clause} "
                "AND kind!='sound' AND (speech_addressee IN ('AMBIGUOUS','OTHER') OR "
                "(kind='heard' AND speech_addressee='UNKNOWN'))", args).fetchone()[0]
            total = self._db.execute(f"SELECT count(*) FROM history_events WHERE {clause}", args).fetchone()[0]
            rows = list(self._db.execute(f"SELECT * FROM history_events WHERE {clause} "
                                        "ORDER BY occurred_at DESC,id DESC LIMIT 4096", args))
            lexical: dict[int, float] = {}
            terms = re.findall(r"\w+", query, flags=re.UNICODE)[:40]
            if terms:
                expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
                matched = list(self._db.execute(
                    "SELECT history_events.*,bm25(history_fts) AS rank FROM history_fts "
                    "JOIN history_events ON history_events.id=history_fts.rowid "
                    f"WHERE history_fts MATCH ? AND {clause} ORDER BY rank LIMIT 128", [expression, *args]))
                lexical = {r["id"]: 1 / (60 + i) for i, r in enumerate(matched)}
                extra = {r["id"]: r for r in matched}
                extra.update({r["id"]: r for r in rows})
                rows = list(extra.values())
        overview = None
        if view == "overview":
            # An extractive temporal hierarchy, not a generated factual summary.
            # Sample the whole day's sequence; a recent cluster cannot hide its
            # beginning. Every excerpt carries its exact source ID and hash.
            days: dict[str, list[sqlite3.Row]] = {}
            tz = ZoneInfo(window["timezone"])
            for row in sorted(rows, key=lambda r: (r["occurred_at"], r["id"])):
                key = datetime.fromtimestamp(row["occurred_at"], tz).date().isoformat()
                days.setdefault(key, []).append(row)
            sections = []
            for day, events in list(days.items())[-8:]:
                positions = sorted({0, len(events)//2, len(events)-1})
                sections.append({"date": day, "source_events": len(events),
                                 "excerpts": [{"event_id": events[i]["id"],
                                               "text": events[i]["text"][:220],
                                               "kind": events[i]["kind"],
                                               "speech_addressee": events[i]["speech_addressee"],
                                               "content_sha256": events[i]["content_sha256"],
                                               "timestamp_uncertain": bool(json.loads(
                                                   events[i]["source_json"]).get("timestamp_uncertain"))}
                                              for i in positions]})
            overview = {"schema": "robit.omni.period-overview.v1", "derived": True,
                        "method": "dated_source_excerpts", "days": sections,
                        "source_events_in_window": total,
                        "complete_period_index": total <= 4096 and len(days) <= 8,
                        "excerpts_are_samples_not_all_facts": True}
        ranked = []
        for row in rows:
            similarity = 0.0
            if query and vector is not None and row["vector"]:
                try:
                    stored = json.loads(bytes(row["vector"]).decode())
                    if len(stored) == len(vector) and all(math.isfinite(v) for v in stored + vector):
                        norm = math.sqrt(sum(v*v for v in vector) * sum(v*v for v in stored))
                        similarity = sum(a*b for a,b in zip(vector, stored, strict=True)) / norm if norm else 0
                except (ValueError, TypeError, UnicodeDecodeError):
                    pass
            if query and row["id"] not in lexical and similarity < 0.45:
                continue
            ranked.append((row, similarity))
        semantic = {r["id"]: 1 / (60+i) for i,(r,s) in enumerate(
            sorted(ranked, key=lambda v: v[1], reverse=True)) if s >= 0.45}
        ranked.sort(key=lambda item: (
            lexical.get(item[0]["id"], 0) + semantic.get(item[0]["id"], 0),
            item[0]["occurred_at"], item[0]["id"]), reverse=True)
        chosen = ranked[:limit]
        results = [{"event_id": r["id"], "text": r["text"][:2000], "kind": r["kind"],
                    "speech_addressee": r["speech_addressee"],
                    "occurred_at": datetime.fromtimestamp(r["occurred_at"], timezone.utc).isoformat(),
                    "local_time": datetime.fromtimestamp(r["occurred_at"], ZoneInfo(window["timezone"])).isoformat(),
                    "content_sha256": r["content_sha256"],
                    "source": {k: (v[:1000] if isinstance(v, str) else v)
                               for k,v in json.loads(r["source_json"]).items()
                               if k in {"type", "legacy_id", "timestamp_uncertain", "user", "assistant",
                                        "speech_addressee", "speaker_attribution", "transcript"}},
                    "semantic_similarity": round(s, 4), "text_truncated": len(r["text"]) > 2000}
                   for r,s in chosen]
        while len(json.dumps(results, ensure_ascii=False)) + len(json.dumps(overview, ensure_ascii=False)) > 12000 and len(results) > 1:
            results.pop()
        return {"schema": "robit.omni.conversation-history.v1", "trust": "untrusted_historical_conversation",
                "scope": "device_owner_voice_archive", "window": window, "query": query,
                "kind": kind, "view": view, "overview": overview,
                "retrieval_complete": total <= 4096,
                "events_in_window": total, "results": results,
                "unattributed_speech_in_window": unattributed,
                "semantic_query_available": vector is not None,
                "vector_scan_truncated": total > 4096,
                "next_before_id": (results[-1]["event_id"] if view == "search" and not query
                                   and len(ranked) > len(results) else None),
                "results_truncated": len(ranked) > len(results),
                "source_is_current_visual_evidence": False,
                "legacy_limit": "Retained events remain searchable. Imported timestamps may precede later merged text; only source events already overwritten/deleted before migration are unrecoverable."}
