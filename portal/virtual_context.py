"""Session bridge from the portal to the lossless virtual-context subsystem."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_omni_adapters.virtual_memory import (
    ContextBudget,
    HashingEmbedder,
    HybridRetriever,
    ImmutableEvidenceStore,
    RecursiveMemoryController,
    StructuredMemoryExtractor,
    VirtualContextEngine,
    WorkingContextPacker,
)
from qwen_omni_adapters.virtual_memory.engine import PreparedTurn
from qwen_omni_adapters.virtual_memory.packer import conservative_token_estimate

VALID_MODES = {"off", "shadow", "active"}


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and str(item.get("text") or "").strip()
        )
    return ""


def _message_records(
    messages: Sequence[Any],
) -> list[tuple[int, str, str, str, str]]:
    """Return stable append-only identities for textual conversation turns."""

    prefix = hashlib.sha256()
    records = []
    for ordinal, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "tool"}:
            continue
        content = _text_content(message.get("content"))
        if not content:
            continue
        prefix.update(role.encode())
        prefix.update(b"\0")
        prefix.update(content.encode())
        records.append(
            (
                ordinal,
                role,
                content,
                prefix.hexdigest()[:24],
                hashlib.sha256(content.encode()).hexdigest()[:16],
            )
        )
    return records


@dataclass
class _SessionEngine:
    store: ImmutableEvidenceStore
    engine: VirtualContextEngine
    extractor: StructuredMemoryExtractor


class SessionVirtualContext:
    """Lazily open one durable, session-isolated evidence corpus."""

    def __init__(
        self,
        root: Path,
        *,
        mode: str = "off",
        physical_context_tokens: int = 16_384,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in VALID_MODES:
            raise ValueError("virtual context mode must be off, shadow, or active")
        self.root = Path(root)
        self.mode = normalized_mode
        self.physical_context_tokens = max(4096, physical_context_tokens)
        self.token_counter = token_counter
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionEngine] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @staticmethod
    def _key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _path(self, session_id: str) -> Path:
        return self.root / f"{self._key(session_id)}.sqlite3"

    def _session(self, session_id: str) -> _SessionEngine:
        key = self._key(session_id)
        with self._lock:
            current = self._sessions.get(key)
            if current is not None:
                return current
            self.root.mkdir(parents=True, exist_ok=True)
            embedder = HashingEmbedder()
            store = ImmutableEvidenceStore(self._path(session_id), embedder=embedder)
            retriever = HybridRetriever(store, query_embedder=embedder)
            controller = RecursiveMemoryController(retriever)
            packer = WorkingContextPacker(
                budget=ContextBudget(max_tokens=self.physical_context_tokens),
                **({"token_counter": self.token_counter} if self.token_counter else {}),
            )
            current = _SessionEngine(
                store=store,
                engine=VirtualContextEngine(store, controller, packer),
                extractor=StructuredMemoryExtractor(store),
            )
            self._sessions[key] = current
            return current

    def observe_messages(self, session_id: str, messages: Sequence[Any]) -> int:
        if not self.enabled:
            return 0
        session = self._session(session_id)
        ingested = 0
        for ordinal, role, content, message_id, version in _message_records(messages):
            chunks = session.store.ingest(
                content,
                source=f"conversation:{role}",
                document_id=f"message-{message_id}",
                message_id=message_id,
                version=version,
                kind="conversation",
                metadata={"role": role, "ordinal": ordinal},
            )
            if role == "user":
                session.extractor.extract(chunks, authority="user")
            ingested += 1
        return ingested

    def observe_documents(
        self, session_id: str, documents: Sequence[Mapping[str, Any]]
    ) -> int:
        if not self.enabled:
            return 0
        session = self._session(session_id)
        ingested = 0
        for document in documents:
            text = str(document.get("text") or "")
            if not text:
                continue
            document_id = str(document.get("id") or "").strip()
            digest = str(document.get("digest") or "").strip()
            name = str(document.get("name") or document_id or "document")
            session.store.ingest(
                text,
                source=f"document:{name}",
                document_id=document_id or None,
                version=digest[:16] or None,
                media_type=str(document.get("mime_type") or "") or None,
                metadata={
                    "name": name,
                    "mime_type": str(document.get("mime_type") or ""),
                    "upload_digest": digest,
                },
            )
            ingested += 1
        return ingested

    def prepare(
        self,
        session_id: str,
        messages: Sequence[Any],
        *,
        system_contract: str,
        query_override: str | None = None,
        reserved_tokens: int = 0,
    ) -> PreparedTurn | None:
        if not self.enabled:
            return None
        query = str(query_override or "").strip()
        recent = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "")
            content = _text_content(message.get("content"))
            if not content or role == "system":
                continue
            recent.append(f"{role}: {content}")
            if role == "user" and query_override is None:
                query = content
        if not query:
            return None
        current_user = next(
            (
                (message_id, version)
                for _ordinal, role, _content, message_id, version in reversed(
                    _message_records(messages)
                )
                if role == "user"
            ),
            None,
        )
        excluded_chunk_ids = set(
            (
                chunk.chunk_id
                for chunk in self._session(session_id).store.document_chunks(
                    f"message-{current_user[0]}",
                    current_user[1],
                )
            )
            if current_user is not None
            else ()
        )
        if query_override is not None:
            excluded_chunk_ids.update(
                chunk.chunk_id
                for chunk in self._session(session_id).store.exact_search(
                    query,
                    limit=200,
                )
                if chunk.original_text.strip() == query
                and str(chunk.metadata.get("role") or "").lower() == "user"
            )
        return self._session(session_id).engine.prepare_turn(
            query,
            system_contract=system_contract,
            recent_context=recent if query_override is not None else recent[:-1],
            reserved_tokens=reserved_tokens,
            excluded_chunk_ids=tuple(excluded_chunk_ids),
        )

    def request_envelope_tokens(self, payload: Mapping[str, Any]) -> int:
        """Count the live tool/control envelope omitted from the working-set text."""

        envelope = {
            key: payload[key]
            for key in ("tools", "tool_choice", "omni", "think")
            if key in payload
        }
        serialized = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
        counter = self.token_counter or conservative_token_estimate
        # Reserve chat-template delimiters and the retained user control turn.
        return counter(serialized) + 96

    def repack_followup(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        query: str,
        system_contract: str,
    ) -> PreparedTurn | None:
        """Page tool-loop results into a fresh bounded working set."""

        if not self.enabled:
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None
        self.observe_messages(session_id, messages)
        prepared = self.prepare(
            session_id,
            messages,
            system_contract=system_contract,
            query_override=query,
            reserved_tokens=self.request_envelope_tokens(payload),
        )
        if prepared is not None:
            self.apply_active(payload, prepared)
        return prepared

    def apply_active(self, payload: dict[str, Any], prepared: PreparedTurn) -> None:
        if self.mode != "active":
            return
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return
        latest_user = next(
            (
                dict(message)
                for message in reversed(messages)
                if isinstance(message, Mapping) and message.get("role") == "user"
            ),
            None,
        )
        if latest_user is None:
            return
        # The complete current text is already the final <current_query> in
        # the bounded pack. Keep the user role (and any current media) without
        # paying for or semantically duplicating that query a second time.
        latest_user["content"] = "Act on <current_query> using the working set above."
        payload["messages"] = [
            {"role": "system", "content": prepared.context.text},
            latest_user,
        ]

    def stats(self, session_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": "off"}
        base = {
            "mode": self.mode,
            "physical_context_tokens": self.physical_context_tokens,
            "tokenizer": "exact_endpoint" if self.token_counter else "conservative_fallback",
        }
        key = self._key(session_id)
        with self._lock:
            current = self._sessions.get(key)
            exists = self._path(session_id).exists()
        if current is None and not exists:
            return {
                **base,
                **{
                    name: 0
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
                },
            }
        return {**base, **(current or self._session(session_id)).store.stats()}

    def clear(self, session_id: str) -> None:
        """Destroy an entire session corpus only on the explicit Trash path."""

        key = self._key(session_id)
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is not None:
                session.store.close()
            path = self._path(session_id)
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    @classmethod
    def from_environment(cls, *, default_root: Path) -> SessionVirtualContext:
        return cls(
            Path(os.environ.get("OMNI_VIRTUAL_CONTEXT_ROOT", str(default_root))).expanduser(),
            mode=os.environ.get("OMNI_VIRTUAL_CONTEXT_MODE", "off"),
            physical_context_tokens=int(
                os.environ.get("OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS", "16384")
            ),
        )
