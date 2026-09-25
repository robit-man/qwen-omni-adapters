"""Deterministic query-time compilation of exact evidence relationships.

These operators are a training-free form of LazyMem/ReContext compression: they
run only after broad retrieval, retain exact source pointers, and replace a noisy
set of pages only when the requested relation is completely resolved.  They do
not inspect expected answers or mutate immutable evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qwen_omni_adapters.virtual_memory.models import (
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
    RetrievalHit,
)
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector

_VALUE_QUERY_RE = re.compile(
    r"\b(?:number|value|code|identifier|id)s?\b",
    re.IGNORECASE,
)
_RELATION_VALUE_RE = re.compile(
    r"(?:\bis\b|\bequals?\b|=|:)\s*:?[\s`'\"]*"
    r"(?P<value>[+-]?\d+(?:[._-]\d+)*)",
    re.IGNORECASE,
)
_ASSIGNMENT_QUERY_RE = re.compile(
    r"\bvariables?\b.*?\b(?:assigned|resolve[sd]?|equal)\w*\b.*?"
    r"\bvalue\b\s*[`'\"]?(?P<target>[A-Za-z0-9_.:+-]{1,100})",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNMENT_RE = re.compile(
    r"\bVAR\s+(?P<left>[A-Za-z_$][\w.$:-]{1,100})\s*=\s*"
    r"(?:VAR\s+)?(?P<right>[A-Za-z0-9_$][\w.$:+-]{0,100})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryCompilation:
    memories: tuple[MemoryRecord, ...]
    complete: bool
    consume_evidence: bool
    operators: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]


class QueryEvidenceCompiler:
    """Compile fully supported exact relations into bounded verified memory."""

    def __init__(self, store: ImmutableEvidenceStore) -> None:
        self.store = store

    def compile(
        self,
        query: str,
        evidence: Sequence[RetrievalHit],
        *,
        trace: TraceCollector | None = None,
    ) -> QueryCompilation:
        collector = trace or TraceCollector()
        memories: list[MemoryRecord] = []
        operators: list[str] = []

        assignment = self._compile_assignment_resolution(query, evidence, collector)
        if assignment is not None:
            memories.append(assignment)
            operators.append("assignment_resolution")
        else:
            lookup = self._compile_exact_value_lookup(query, evidence, collector)
            if lookup is not None:
                memories.append(lookup)
                operators.append("exact_value_lookup")

        complete = bool(memories)
        return QueryCompilation(
            memories=tuple(memories),
            complete=complete,
            consume_evidence=complete,
            operators=tuple(operators),
            trace=collector.export(),
        )

    def _compile_exact_value_lookup(
        self,
        query: str,
        evidence: Sequence[RetrievalHit],
        trace: TraceCollector,
    ) -> MemoryRecord | None:
        if not _VALUE_QUERY_RE.search(query):
            return None
        anchors = HybridRetriever(self.store).plan(query).exact_strings
        # A single exact value is already the ideal replay unit. Compilation
        # earns its complexity when several independent addresses must be
        # preserved and compared in one small working set.
        if len(anchors) < 2:
            return None
        found: dict[str, list[tuple[str, ProvenancePointer]]] = {
            anchor: [] for anchor in anchors
        }
        for hit in evidence:
            text = hit.chunk.original_text
            folded = text.casefold()
            for anchor in anchors:
                cursor = 0
                anchor_folded = anchor.casefold()
                while cursor < len(folded):
                    start = folded.find(anchor_folded, cursor)
                    if start < 0:
                        break
                    after = start + len(anchor)
                    relation = _RELATION_VALUE_RE.search(text, after, min(len(text), after + 200))
                    if relation is not None:
                        value = relation.group("value")
                        pointer = ProvenancePointer(
                            hit.chunk.chunk_id,
                            start,
                            relation.end("value"),
                            exact=True,
                        )
                        if value not in {item[0] for item in found[anchor]}:
                            found[anchor].append((value, pointer))
                    cursor = after
        missing = [anchor for anchor, values in found.items() if not values]
        if missing:
            trace.record(
                "COMPILE_RELATIONS",
                operator="exact_value_lookup",
                complete=False,
                requested_anchors=list(anchors),
                missing_anchors=missing,
            )
            return None
        content = "Verified exact-value compilation from immutable evidence:\n" + "\n".join(
            f"key={anchor} values=[{', '.join(value for value, _pointer in found[anchor])}]"
            for anchor in anchors
        )
        pointers = tuple(
            pointer
            for anchor in anchors
            for _value, pointer in found[anchor]
        )
        memory = self._write_once(
            subject=self._subject("exact_value_lookup", query),
            content=content,
            pointers=pointers,
            operator="exact_value_lookup",
        )
        trace.record(
            "COMPILE_RELATIONS",
            memory.memory_id,
            operator="exact_value_lookup",
            complete=True,
            requested_anchors=list(anchors),
            values={
                anchor: [value for value, _pointer in found[anchor]]
                for anchor in anchors
            },
            source_chunk_ids=list(dict.fromkeys(pointer.chunk_id for pointer in pointers)),
            exact_source_preserved=True,
        )
        return memory

    def _compile_assignment_resolution(
        self,
        query: str,
        evidence: Sequence[RetrievalHit],
        trace: TraceCollector,
    ) -> MemoryRecord | None:
        request = _ASSIGNMENT_QUERY_RE.search(query)
        if request is None:
            return None
        requested_target = request.group("target").strip("`'\".,;:!?")
        target = requested_target.casefold()
        assignments: dict[str, dict[str, ProvenancePointer]] = {}
        display: dict[str, str] = {}
        for hit in evidence:
            for match in _ASSIGNMENT_RE.finditer(hit.chunk.original_text):
                left = match.group("left")
                right = match.group("right")
                folded = left.casefold()
                display.setdefault(folded, left)
                assignments.setdefault(folded, {})[right.casefold()] = ProvenancePointer(
                    hit.chunk.chunk_id,
                    match.start(),
                    match.end(),
                    exact=True,
                )
        # Conflicting assignments need chronology/provenance resolution, not a
        # synthetic compromise.  Do not compile through an ambiguous node.
        unambiguous = {
            left: next(iter(rights.items()))
            for left, rights in assignments.items()
            if len(rights) == 1
        }

        def resolve(left: str) -> tuple[str, tuple[str, ...], tuple[ProvenancePointer, ...]] | None:
            path: list[str] = []
            pointers: list[ProvenancePointer] = []
            current = left
            seen: set[str] = set()
            while current in unambiguous:
                if current in seen:
                    return None
                seen.add(current)
                path.append(display.get(current, current))
                right, pointer = unambiguous[current]
                pointers.append(pointer)
                current = right
            path.append(current)
            return current, tuple(path), tuple(pointers)

        resolved: list[tuple[str, tuple[str, ...], tuple[ProvenancePointer, ...]]] = []
        for left in unambiguous:
            result = resolve(left)
            if result is not None and result[0] == target:
                resolved.append((display.get(left, left), result[1], result[2]))
        if not resolved:
            trace.record(
                "COMPILE_RELATIONS",
                operator="assignment_resolution",
                complete=False,
                target=target,
                assignments=len(assignments),
                ambiguous_nodes=sorted(
                    display.get(left, left)
                    for left, rights in assignments.items()
                    if len(rights) > 1
                ),
            )
            return None
        # Stable source order follows the first exact assignment offset rather
        # than lexical sorting, which preserves causal/chain readability.
        resolved.sort(
            key=lambda item: (
                item[2][0].chunk_id if item[2] else "",
                item[2][0].char_start if item[2] else 0,
            )
        )
        content = (
            f"Verified assignment-graph resolution for target={requested_target}:\n"
            + "\n".join(
                f"variable={variable} path={' -> '.join(path)}"
                for variable, path, _pointers in resolved
            )
        )
        pointers = tuple(
            dict.fromkeys(
                pointer
                for _variable, _path, path_pointers in resolved
                for pointer in path_pointers
            )
        )
        memory = self._write_once(
            subject=self._subject("assignment_resolution", query),
            content=content,
            pointers=pointers,
            operator="assignment_resolution",
        )
        trace.record(
            "COMPILE_RELATIONS",
            memory.memory_id,
            operator="assignment_resolution",
            complete=True,
            target=requested_target,
            variables=[variable for variable, _path, _pointers in resolved],
            source_chunk_ids=list(dict.fromkeys(pointer.chunk_id for pointer in pointers)),
            exact_source_preserved=True,
        )
        return memory

    @staticmethod
    def _subject(operator: str, query: str) -> str:
        digest = hashlib.sha256(query.casefold().encode()).hexdigest()[:16]
        return f"compiled:{operator}:{digest}"

    def _write_once(
        self,
        *,
        subject: str,
        content: str,
        pointers: Sequence[ProvenancePointer],
        operator: str,
    ) -> MemoryRecord:
        for memory in self.store.active_memories(
            classes=[MemoryClass.FACT],
            subject=subject,
        ):
            if memory.content == content:
                return memory
        return self.store.write_memory(
            MemoryClass.FACT,
            subject,
            content,
            provenance=pointers,
            importance=0.99,
            ttl_seconds=300,
            compression_generation=1,
            verified=True,
            metadata={
                "operation": "query_evidence_compilation",
                "operator": operator,
                "exact_source_preserved": True,
            },
        )
