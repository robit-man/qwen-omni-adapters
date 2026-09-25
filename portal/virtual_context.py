"""Session bridge from the portal to the lossless virtual-context subsystem."""

from __future__ import annotations

import copy
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
    QueryAwareRecurrentViewBuilder,
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


def _latest_tool_protocol_tail(
    messages: Sequence[Any],
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Keep the newest completed tool round in its native chat roles.

    Flattening a successful ``role=tool`` observation into generic user text
    makes small language trunks treat the original request as still pending
    and repeat the same call. Older tool rounds remain losslessly indexed; the
    newest round stays structured so the inference endpoint receives a valid
    assistant-tool-call -> tool-result protocol.
    """

    start: int | None = None
    for ordinal in range(len(messages) - 1, -1, -1):
        message = messages[ordinal]
        if (
            isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and isinstance(message.get("tool_calls"), list)
            and message.get("tool_calls")
        ):
            start = ordinal
            break
    if start is None:
        return [], ()
    retained: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for ordinal in range(start, len(messages)):
        message = messages[ordinal]
        if not isinstance(message, Mapping):
            break
        role = str(message.get("role") or "")
        if ordinal == start:
            if role != "assistant":  # pragma: no cover - guarded above
                break
        elif role != "tool":
            break
        retained.append(copy.deepcopy(dict(message)))
        ordinals.append(ordinal)
    return retained, tuple(ordinals)


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
        physical_context_state_file: Path | None = None,
        recurrent_memory_tokens: int = 512,
        recurrent_source_chunks: int = 200,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in VALID_MODES:
            raise ValueError("virtual context mode must be off, shadow, or active")
        self.root = Path(root)
        self.mode = normalized_mode
        self.physical_context_tokens = max(4096, physical_context_tokens)
        self.token_counter = token_counter
        self.recurrent_memory_tokens = max(0, int(recurrent_memory_tokens))
        self.recurrent_source_chunks = max(1, int(recurrent_source_chunks))
        if self.recurrent_memory_tokens and not 128 <= self.recurrent_memory_tokens <= 4096:
            raise ValueError("recurrent memory tokens must be zero or between 128 and 4096")
        if self.recurrent_source_chunks > 2000:
            raise ValueError("recurrent source chunks cannot exceed 2000")
        self.physical_context_state_file = (
            Path(physical_context_state_file).expanduser()
            if physical_context_state_file is not None
            else None
        )
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

    def _resident_context_tokens(self) -> tuple[int, str]:
        """Return the live KV window without ever exceeding the configured cap."""

        if self.physical_context_state_file is None:
            return self.physical_context_tokens, "configured"
        try:
            selected = int(
                self.physical_context_state_file.read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            # A configured-but-unavailable live state is not permission to
            # assume the largest window. Fail closed at the supported floor.
            return min(4096, self.physical_context_tokens), "resident_state_fallback"
        return (
            max(4096, min(self.physical_context_tokens, selected)),
            "resident_state",
        )

    def _refresh_budget(self, session: _SessionEngine) -> int:
        selected, _source = self._resident_context_tokens()
        current = session.engine.packer.budget
        if current.max_tokens != selected:
            session.engine.packer.budget = ContextBudget(
                max_tokens=selected,
                output_headroom=min(current.output_headroom, selected - 512),
                system_target=current.system_target,
                pinned_target=current.pinned_target,
                structured_target=current.structured_target,
                recent_target=current.recent_target,
                evidence_target=current.evidence_target,
            )
            session.engine.packer.budget.validate()
        return selected

    def _session(self, session_id: str) -> _SessionEngine:
        key = self._key(session_id)
        with self._lock:
            current = self._sessions.get(key)
            if current is not None:
                return current
            self.root.mkdir(parents=True, exist_ok=True)
            resident_tokens, _source = self._resident_context_tokens()
            embedder = HashingEmbedder()
            store = ImmutableEvidenceStore(self._path(session_id), embedder=embedder)
            retriever = HybridRetriever(store, query_embedder=embedder)
            controller = RecursiveMemoryController(retriever)
            packer = WorkingContextPacker(
                budget=ContextBudget(max_tokens=resident_tokens),
                **({"token_counter": self.token_counter} if self.token_counter else {}),
            )
            current = _SessionEngine(
                store=store,
                engine=VirtualContextEngine(
                    store,
                    controller,
                    packer,
                    recurrent_view_builder=(
                        QueryAwareRecurrentViewBuilder(
                            memory_tokens=self.recurrent_memory_tokens,
                            source_chunks=self.recurrent_source_chunks,
                            **(
                                {"token_counter": self.token_counter}
                                if self.token_counter
                                else {}
                            ),
                        )
                        if self.recurrent_memory_tokens
                        else None
                    ),
                ),
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
        retained_protocol_ordinals: Sequence[int] = (),
    ) -> PreparedTurn | None:
        if not self.enabled:
            return None
        query = str(query_override or "").strip()
        recent = []
        retained_ordinals = set(retained_protocol_ordinals)
        for ordinal, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "")
            content = _text_content(message.get("content"))
            if not content or role == "system":
                continue
            if ordinal in retained_ordinals:
                continue
            if (
                query_override is not None
                and role == "user"
                and "<current_query>" in content
            ):
                # Do not preserve the legacy symbolic handoff as dialogue.
                # The exact override is pinned separately as the real query.
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
        if retained_ordinals:
            for ordinal, _role, _content, message_id, version in _message_records(messages):
                if ordinal not in retained_ordinals:
                    continue
                excluded_chunk_ids.update(
                    chunk.chunk_id
                    for chunk in self._session(session_id).store.document_chunks(
                        f"message-{message_id}", version
                    )
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
        session = self._session(session_id)
        self._refresh_budget(session)
        return session.engine.prepare_turn(
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

    def _protocol_tail_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        if not messages:
            return 0
        counter = self.token_counter or conservative_token_estimate
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        # Reserve the model template's per-message role/tool delimiters in
        # addition to the serialized content.
        return counter(serialized) + 32 * len(messages)

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
        protocol_tail, protocol_ordinals = _latest_tool_protocol_tail(messages)
        prepared = self.prepare(
            session_id,
            messages,
            system_contract=system_contract,
            query_override=query,
            reserved_tokens=(
                self.request_envelope_tokens(payload)
                + self._protocol_tail_tokens(protocol_tail)
            ),
            retained_protocol_ordinals=protocol_ordinals,
        )
        if prepared is not None:
            self.apply_active(payload, prepared, protocol_tail=protocol_tail)
        return prepared

    def apply_active(
        self,
        payload: dict[str, Any],
        prepared: PreparedTurn,
        *,
        protocol_tail: Sequence[Mapping[str, Any]] = (),
    ) -> None:
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
        # Keep policy in the system role and present the bounded working set as
        # one user message, with exact evidence immediately before the real
        # query.  This matches the independently verified RULER path and avoids
        # language trunks treating evidence embedded in a system policy as
        # instructions rather than source material.
        bounded_system = "\n\n".join(
            item.text
            for item in prepared.context.items
            if item.category == "system_contract" and item.text
        )
        working_text = "\n\n".join(
            item.text
            for item in prepared.context.items
            if item.category != "system_contract" and item.text
        )
        original_content = latest_user.get("content")
        if isinstance(original_content, list):
            retained_media = [
                dict(item)
                for item in original_content
                if isinstance(item, Mapping) and item.get("type") != "text"
            ]
            latest_user["content"] = [
                {"type": "text", "text": working_text},
                *retained_media,
            ]
        else:
            latest_user["content"] = working_text
        payload["messages"] = [
            {"role": "system", "content": bounded_system},
            latest_user,
            *(copy.deepcopy(dict(message)) for message in protocol_tail),
        ]

    def stats(self, session_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": "off"}
        resident_tokens, context_source = self._resident_context_tokens()
        base = {
            "mode": self.mode,
            "physical_context_tokens": resident_tokens,
            "configured_physical_context_tokens": self.physical_context_tokens,
            "physical_context_source": context_source,
            "tokenizer": "exact_endpoint" if self.token_counter else "conservative_fallback",
            "recurrent_memory_tokens": self.recurrent_memory_tokens,
            "recurrent_source_chunks": self.recurrent_source_chunks,
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
            physical_context_state_file=(
                Path(os.environ["OMNI_COMPREHENSION_CONTEXT_FILE"])
                if os.environ.get("OMNI_COMPREHENSION_CONTEXT_FILE", "").strip()
                else None
            ),
            recurrent_memory_tokens=int(
                os.environ.get("OMNI_VIRTUAL_CONTEXT_RECURRENT_TOKENS", "512")
            ),
            recurrent_source_chunks=int(
                os.environ.get("OMNI_VIRTUAL_CONTEXT_RECURRENT_SOURCE_CHUNKS", "200")
            ),
        )
