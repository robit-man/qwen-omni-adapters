"""Model-agnostic façade for lossless ingestion and bounded context assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qwen_omni_adapters.virtual_memory.controller import (
    ControllerResult,
    RecursiveMemoryController,
)
from qwen_omni_adapters.virtual_memory.models import (
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
    RetrievalHit,
    WorkingContext,
)
from qwen_omni_adapters.virtual_memory.packer import WorkingContextPacker
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import MemoryOperation, TraceCollector


@dataclass(frozen=True)
class PreparedTurn:
    context: WorkingContext
    controller: ControllerResult
    answer_allowed: bool
    unresolved_reason: str | None


class VirtualContextEngine:
    """Coordinate external memory without coupling it to an inference backend."""

    def __init__(
        self,
        store: ImmutableEvidenceStore,
        controller: RecursiveMemoryController,
        packer: WorkingContextPacker,
    ) -> None:
        self.store = store
        self.controller = controller
        self.packer = packer

    def ingest(self, text: str, *, source: str, **metadata: Any):
        return self.store.ingest(text, source=source, **metadata)

    def derive_memory(
        self,
        memory_class: MemoryClass | str,
        subject: str,
        content: str,
        *,
        chunk_ids: Iterable[str],
        supersedes: str | None = None,
        importance: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        pointers = []
        for chunk_id in chunk_ids:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                raise KeyError(f"unknown source chunk: {chunk_id}")
            pointers.append(
                ProvenancePointer(
                    chunk_id=chunk_id,
                    char_start=0,
                    char_end=len(chunk.original_text),
                    exact=True,
                )
            )
        return self.store.write_memory(
            memory_class,
            subject,
            content,
            provenance=pointers,
            supersedes=supersedes,
            importance=importance,
            metadata=metadata,
        )

    def prepare_turn(
        self,
        query: str,
        *,
        system_contract: str,
        recent_context: Sequence[str] = (),
        recurrent_memory: str = "",
        active_subjects: Sequence[str] = (),
        reserved_tokens: int = 0,
    ) -> PreparedTurn:
        trace = TraceCollector()
        controller_result = self.controller.gather(query, trace=trace)
        memories = self._active_memories(active_subjects)
        conflicts = self._conflicts(memories)
        conflict_hits: list[RetrievalHit] = []
        for key, competing in conflicts.items():
            trace.record(
                "memory_conflict",
                subject=key[1],
                memory_class=key[0],
                memory_ids=[memory.memory_id for memory in competing],
            )
            for memory in competing:
                for pointer in memory.provenance:
                    chunk = self.store.get_chunk(pointer.chunk_id)
                    if chunk is not None:
                        conflict_hits.append(
                            RetrievalHit(
                                chunk=chunk,
                                score=1.1,
                                channels=("memory_conflict",),
                                channel_scores={"memory_conflict": 1.1},
                            )
                        )
        combined_evidence = {
            hit.chunk.chunk_id: hit
            for hit in (*controller_result.evidence, *conflict_hits)
        }
        context = self.packer.pack(
            query,
            system_contract=system_contract,
            evidence=sorted(
                combined_evidence.values(), key=lambda hit: hit.score, reverse=True
            ),
            retrieval_queries=controller_result.queries,
            memories=memories,
            recent_context=recent_context,
            recurrent_memory=recurrent_memory,
            reserved_tokens=reserved_tokens,
            trace=trace,
        )
        return PreparedTurn(
            context=context,
            controller=controller_result,
            answer_allowed=controller_result.sufficient and not conflicts,
            unresolved_reason=(
                "conflicting active memory requires provenance/chronology resolution"
                if conflicts
                else (
                    None
                    if controller_result.sufficient
                    else "retrieval budget exhausted before evidence sufficiency"
                )
            ),
        )

    def expand(self, memory_id: str, *, trace: TraceCollector | None = None):
        collector = trace or TraceCollector()
        recovered = self.store.reconstruct(memory_id)
        collector.record(
            MemoryOperation.EXPAND,
            memory_id,
            spans=len(recovered),
            exact=all(pointer.exact for pointer, _text in recovered),
        )
        return recovered

    def reconstruct(self, memory_id: str, *, trace: TraceCollector | None = None):
        collector = trace or TraceCollector()
        recovered = self.store.reconstruct(memory_id)
        collector.record(
            MemoryOperation.RECONSTRUCT,
            memory_id,
            spans=len(recovered),
            success=bool(recovered),
        )
        return recovered

    def _active_memories(self, subjects: Sequence[str]) -> list[MemoryRecord]:
        memories = self.store.active_memories()
        if not subjects:
            return memories
        selected = {subject.casefold() for subject in subjects}
        return [
            memory
            for memory in memories
            if memory.memory_class in {MemoryClass.CONSTRAINT, MemoryClass.CURRENT_PLAN}
            or memory.subject.casefold() in selected
        ]

    @staticmethod
    def _conflicts(
        memories: Sequence[MemoryRecord],
    ) -> dict[tuple[str, str], list[MemoryRecord]]:
        grouped: dict[tuple[str, str], list[MemoryRecord]] = {}
        for memory in memories:
            key = (memory.memory_class.value, memory.subject.casefold())
            grouped.setdefault(key, []).append(memory)
        return {
            key: values
            for key, values in grouped.items()
            if len({value.content for value in values}) > 1
        }
