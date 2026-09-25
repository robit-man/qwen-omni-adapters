"""Model-agnostic façade for lossless ingestion and bounded context assembly."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from qwen_omni_adapters.virtual_memory.compilation import QueryEvidenceCompiler
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


_MEMORY_SCOPE_STOP_WORDS = {
    "always",
    "and",
    "are",
    "constraint",
    "current",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "into",
    "must",
    "never",
    "not",
    "shall",
    "that",
    "the",
    "their",
    "then",
    "this",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _scope_terms(value: str) -> set[str]:
    terms = {
        raw.strip(".$:-").casefold()
        for raw in re.findall(r"[A-Za-z0-9_.$:-]{3,}", value)
    }
    return {
        term
        for term in terms
        if len(term) >= 3 and term not in _MEMORY_SCOPE_STOP_WORDS
    }


def select_relevant_memories(
    memories: Sequence[MemoryRecord],
    query: str,
    *,
    active_subjects: Sequence[str] = (),
) -> list[MemoryRecord]:
    """Deterministically scope derived memory without similarity-only recall.

    Relevant constraints and the current execution plan remain hard pins.
    Constraints are selected by explicit global scope, active subject, or exact
    lexical/entity overlap; pinning every constraint in the corpus leaks stale
    task policy into unrelated answers. Other memory classes must be explicitly
    active or share a stable subject identifier with the current request. Raw
    evidence is unaffected and remains recoverable.
    """

    selected_subjects = {subject.casefold() for subject in active_subjects}
    query_folded = query.casefold()
    query_terms = _scope_terms(query)
    selected = []
    for memory in memories:
        if memory.memory_class is MemoryClass.CURRENT_PLAN:
            selected.append(memory)
            continue
        subject = memory.subject.casefold()
        subject_terms = _scope_terms(memory.subject)
        explicitly_active = subject in selected_subjects
        subject_mentioned = subject in query_folded or bool(subject_terms & query_terms)
        if memory.memory_class is MemoryClass.CONSTRAINT:
            content_folded = memory.content.casefold()
            content_terms = _scope_terms(memory.content)
            active_entity_mentioned = any(
                active in content_folded for active in selected_subjects
            )
            globally_scoped = memory.metadata.get("scope") == "global"
            content_overlap = content_terms & query_terms
            if (
                globally_scoped
                or explicitly_active
                or subject_mentioned
                or active_entity_mentioned
                # Backward-compatible path for older opaque constraint
                # subjects: require two exact content anchors so an incidental
                # verb such as "replace" in a historical question does not
                # activate an unrelated policy.
                or len(content_overlap) >= 2
            ):
                selected.append(memory)
            continue
        if explicitly_active or subject_mentioned:
            selected.append(memory)
    return selected


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
        excluded_chunk_ids: Sequence[str] = (),
    ) -> PreparedTurn:
        trace = TraceCollector()
        controller_result = self.controller.gather(
            query,
            trace=trace,
            excluded_chunk_ids=excluded_chunk_ids,
        )
        compilation = QueryEvidenceCompiler(self.store).compile(
            query,
            controller_result.evidence,
            trace=trace,
        )
        memories = [
            *self._active_memories(query, active_subjects),
            *compilation.memories,
        ]
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
        controller_evidence = (
            () if compilation.consume_evidence else controller_result.evidence
        )
        combined_evidence = {
            hit.chunk.chunk_id: hit
            for hit in (*controller_evidence, *conflict_hits)
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
        resident_ids = set(context.evidence_chunk_ids)
        resident_evidence = [
            hit
            for chunk_id, hit in combined_evidence.items()
            if chunk_id in resident_ids
        ]
        final_score = self.controller.evidence_sufficiency_score(
            query,
            resident_evidence,
        )
        resident_controller_sufficient = (
            controller_result.sufficient
            and len(resident_evidence) >= self.controller.config.minimum_evidence
            and final_score >= self.controller.config.sufficiency_threshold
        )
        packed_memory_ids = {item.item_id for item in context.items}
        packed_relevant_constraint = any(
            memory.memory_class is MemoryClass.CONSTRAINT
            and memory.verified
            and memory.memory_id in packed_memory_ids
            for memory in memories
        )
        evidence_sufficient = compilation.complete or (
            controller_result.sufficient
            and (resident_controller_sufficient or packed_relevant_constraint)
        )
        trace.record(
            "final_evidence_sufficiency",
            sufficient=evidence_sufficient,
            score=final_score,
            resident_chunk_ids=sorted(resident_ids),
            dropped_chunk_ids=sorted(set(combined_evidence) - resident_ids),
            compilation_complete=compilation.complete,
            packed_relevant_constraint=packed_relevant_constraint,
        )
        context = replace(context, trace=trace.export())
        return PreparedTurn(
            context=context,
            controller=controller_result,
            answer_allowed=evidence_sufficient and not conflicts,
            unresolved_reason=(
                "conflicting active memory requires provenance/chronology resolution"
                if conflicts
                else (
                    None
                    if evidence_sufficient
                    else (
                        "final working set lost sufficient evidence during packing"
                        if controller_result.sufficient
                        else "retrieval budget exhausted before evidence sufficiency"
                    )
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

    def _active_memories(
        self, query: str, subjects: Sequence[str]
    ) -> list[MemoryRecord]:
        return select_relevant_memories(
            self.store.active_memories(), query, active_subjects=subjects
        )

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
