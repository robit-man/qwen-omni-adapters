"""Explicit virtual-memory movement operations over the evidence store."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qwen_omni_adapters.virtual_memory.models import (
    EvidenceChunk,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
)
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import MemoryOperation, TraceCollector


@dataclass(frozen=True)
class ResidentPage:
    item_id: str
    level: str
    tokens: int
    pinned: bool
    metadata: dict[str, Any]


class MemoryHierarchy:
    """Track the ephemeral resident set while delegating truth to L4 storage."""

    def __init__(
        self,
        store: ImmutableEvidenceStore,
        *,
        trace: TraceCollector | None = None,
    ) -> None:
        self.store = store
        self.trace = trace or TraceCollector()
        self._resident: dict[str, ResidentPage] = {}

    def page_in(
        self,
        item_id: str,
        *,
        level: str,
        tokens: int,
        pinned: bool = False,
        **metadata: Any,
    ) -> ResidentPage:
        page = ResidentPage(
            item_id=item_id,
            level=level,
            tokens=max(0, int(tokens)),
            pinned=bool(pinned),
            metadata=dict(metadata),
        )
        self._resident[item_id] = page
        self.trace.record(
            MemoryOperation.PAGE_IN,
            item_id,
            level=level,
            tokens=page.tokens,
            pinned=page.pinned,
            **metadata,
        )
        if pinned:
            self.trace.record(MemoryOperation.PIN, item_id, level=level)
        return page

    def page_out(self, item_id: str, *, reason: str = "released") -> bool:
        page = self._resident.get(item_id)
        if page is None:
            return False
        if page.pinned:
            raise ValueError(f"cannot page out pinned item: {item_id}")
        self._resident.pop(item_id)
        self.trace.record(
            MemoryOperation.PAGE_OUT,
            item_id,
            level=page.level,
            tokens=page.tokens,
            reason=reason,
            recoverable=True,
        )
        return True

    def pin(self, item_id: str) -> ResidentPage:
        page = self._resident[item_id]
        pinned = ResidentPage(page.item_id, page.level, page.tokens, True, page.metadata)
        self._resident[item_id] = pinned
        self.trace.record(MemoryOperation.PIN, item_id, level=page.level)
        return pinned

    def unpin(self, item_id: str) -> ResidentPage:
        page = self._resident[item_id]
        unpinned = ResidentPage(page.item_id, page.level, page.tokens, False, page.metadata)
        self._resident[item_id] = unpinned
        self.trace.record(MemoryOperation.UNPIN, item_id, level=page.level)
        return unpinned

    def evict(self, item_id: str, *, reason: str) -> bool:
        page = self._resident.get(item_id)
        if page is None:
            return False
        if page.pinned:
            raise ValueError(f"cannot evict pinned item: {item_id}")
        self._resident.pop(item_id)
        self.trace.record(
            MemoryOperation.EVICT,
            item_id,
            level=page.level,
            tokens=page.tokens,
            reason=reason,
            recoverable=True,
        )
        return True

    def expand_chunk(self, chunk_id: str, *, neighbors: int = 0) -> list[EvidenceChunk]:
        chunks = self.store.expand(chunk_id, neighbors=neighbors)
        self.trace.record(
            MemoryOperation.EXPAND,
            chunk_id,
            kind="chunk",
            neighbors=neighbors,
            expanded_chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        return chunks

    def reconstruct(self, memory_id: str):
        recovered = self.store.reconstruct(memory_id)
        self.trace.record(
            MemoryOperation.RECONSTRUCT,
            memory_id,
            spans=len(recovered),
            exact=all(pointer.exact for pointer, _text in recovered),
        )
        return recovered

    def merge(
        self,
        memory_ids: Iterable[str],
        *,
        memory_class: MemoryClass | str,
        subject: str,
        content: str,
        importance: float = 0.5,
    ) -> MemoryRecord:
        sources: dict[tuple[str, int, int], ProvenancePointer] = {}
        selected_ids = []
        generation = 0
        for memory_id in memory_ids:
            memory = self.store.get_memory(memory_id)
            if memory is None:
                raise KeyError(f"unknown memory: {memory_id}")
            selected_ids.append(memory_id)
            generation = max(generation, memory.compression_generation)
            for pointer in memory.provenance:
                sources[(pointer.chunk_id, pointer.char_start, pointer.char_end)] = pointer
        merged = self.store.write_memory(
            memory_class,
            subject,
            content,
            provenance=sources.values(),
            importance=importance,
            compression_generation=generation + 1,
            verified=False,
            metadata={"merged_from": selected_ids},
        )
        self.trace.record(
            MemoryOperation.MERGE,
            merged.memory_id,
            merged_from=selected_ids,
            source_chunk_ids=[pointer.chunk_id for pointer in merged.provenance],
        )
        return merged

    def supersede(
        self,
        memory_id: str,
        *,
        content: str,
        provenance: Iterable[ProvenancePointer],
    ) -> MemoryRecord:
        previous = self.store.get_memory(memory_id)
        if previous is None:
            raise KeyError(f"unknown memory: {memory_id}")
        replacement = self.store.write_memory(
            previous.memory_class,
            previous.subject,
            content,
            provenance=provenance,
            supersedes=previous.memory_id,
            importance=previous.importance,
            compression_generation=previous.compression_generation,
            verified=previous.verified,
            metadata={"supersession_reason": "explicit hierarchy operation"},
        )
        self.trace.record(
            MemoryOperation.SUPERSEDE,
            replacement.memory_id,
            supersedes=previous.memory_id,
        )
        return replacement

    def snapshot(self) -> tuple[ResidentPage, ...]:
        return tuple(self._resident.values())
