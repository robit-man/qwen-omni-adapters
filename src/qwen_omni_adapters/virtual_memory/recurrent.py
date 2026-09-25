"""Bounded recurrent textual memory over externally recoverable chunks."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import (
    EvidenceChunk,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
)
from qwen_omni_adapters.virtual_memory.packer import conservative_token_estimate
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import MemoryOperation, TraceCollector


@dataclass(frozen=True)
class RecurrentConfig:
    memory_tokens: int = 1024
    chunk_tokens: int = 4096
    regenerate_every: int = 8

    def validate(self) -> None:
        if self.memory_tokens not in {512, 1024, 2048, 4096}:
            raise ValueError("memory_tokens must be one of 512, 1024, 2048, 4096")
        if self.chunk_tokens not in {2048, 4096, 8192}:
            raise ValueError("chunk_tokens must be one of 2048, 4096, 8192")
        if self.regenerate_every < 0:
            raise ValueError("regenerate_every cannot be negative")


@dataclass(frozen=True)
class RecurrentWriteRequest:
    query: str
    previous_memory: str
    chunk: EvidenceChunk
    memory_token_budget: int
    step: int


@dataclass(frozen=True)
class RecurrentResult:
    memory: MemoryRecord | None
    processed_chunk_ids: tuple[str, ...]
    trace: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RecurrentView:
    """Ephemeral query-specific L2 view with exact recovery pointers."""

    text: str
    provenance: tuple[ProvenancePointer, ...]
    processed_chunk_ids: tuple[str, ...]
    tokens: int
    trace: tuple[dict[str, object], ...]


_VIEW_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "have",
    "into",
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


def _view_terms(value: str) -> set[str]:
    return {
        term.casefold().strip(".$:-")
        for term in re.findall(r"[A-Za-z0-9_.$:-]{3,}", value)
        if term.casefold().strip(".$:-") not in _VIEW_STOP_WORDS
    }


class QueryAwareRecurrentViewBuilder:
    """Build a bounded MemAgent-shaped view without modifying source truth.

    This applies LazyMem's query-time principle to recurrent text: scan a
    bounded set of externally recoverable pages, repeatedly update a fixed-size
    state, and attach exact line provenance to everything that survives.  The
    view is deliberately unverified and never substitutes for retrieved L3
    evidence.
    """

    def __init__(
        self,
        *,
        memory_tokens: int = 512,
        source_chunks: int = 200,
        token_counter: Callable[[str], int] = conservative_token_estimate,
    ) -> None:
        if not 128 <= int(memory_tokens) <= 4096:
            raise ValueError("recurrent view tokens must be between 128 and 4096")
        if not 1 <= int(source_chunks) <= 2000:
            raise ValueError("recurrent source chunks must be between 1 and 2000")
        self.memory_tokens = int(memory_tokens)
        self.source_chunks = int(source_chunks)
        self.token_counter = token_counter

    def process(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        *,
        excluded_chunk_ids: Sequence[str] = (),
        trace: TraceCollector | None = None,
    ) -> RecurrentView:
        collector = trace or TraceCollector()
        query_terms = _view_terms(query)
        excluded = set(excluded_chunk_ids)
        candidates: list[
            tuple[float, int, str, ProvenancePointer]
        ] = []
        selected: list[tuple[float, int, str, ProvenancePointer]] = []
        processed: list[str] = []
        sequence = 0
        bounded = [chunk for chunk in chunks if chunk.chunk_id not in excluded][
            -self.source_chunks :
        ]
        for step, chunk in enumerate(bounded):
            processed.append(chunk.chunk_id)
            changed = False
            role = str(chunk.metadata.get("role") or "").casefold()
            role_penalty = 0.2 if role == "assistant" else 0.0
            for match in re.finditer(r"[^\r\n]+", chunk.original_text):
                line = match.group(0).strip()
                if not line:
                    continue
                terms = _view_terms(line)
                overlap = len(query_terms & terms)
                if not overlap:
                    continue
                durable = bool(
                    re.search(
                        r"\b(?:MUST|NEVER|ALWAYS|decided|chose|changed\s+from|"
                        r"configured\s+as|TODO|open\s+question|unresolved)\b|"
                        r"\b(?:fault|error|request|status)=",
                        line,
                        re.IGNORECASE,
                    )
                )
                exact_addresses = {
                    term
                    for term in query_terms
                    if re.search(r"[_.$:-]|\d", term)
                }
                exact_overlap = len(exact_addresses & terms)
                recency = (step + 1) / max(1, len(bounded))
                score = (
                    float(overlap)
                    + exact_overlap * 0.75
                    + (0.25 if durable else 0.0)
                    + recency * 0.1
                    - role_penalty
                )
                left_trim = len(match.group(0)) - len(match.group(0).lstrip())
                start = match.start() + left_trim
                pointer = ProvenancePointer(
                    chunk.chunk_id,
                    start,
                    start + len(line),
                    exact=True,
                )
                candidates.append((score, sequence, line, pointer))
                sequence += 1
                changed = True
            if not changed:
                continue
            selected = self._bounded_state(candidates)
            collector.record(
                MemoryOperation.MERGE,
                "query-recurrent-view",
                level="L4->L2",
                step=step,
                source_chunk_id=chunk.chunk_id,
                retained_lines=len(selected),
                memory_tokens=sum(self.token_counter(item[2]) for item in selected),
                authority="derived_unverified",
            )
        text = "\n".join(item[2] for item in selected)
        provenance = tuple(item[3] for item in selected)
        return RecurrentView(
            text=text,
            provenance=provenance,
            processed_chunk_ids=tuple(processed),
            tokens=self.token_counter(text),
            trace=collector.export(),
        )

    def _bounded_state(
        self,
        candidates: Sequence[tuple[float, int, str, ProvenancePointer]],
    ) -> list[tuple[float, int, str, ProvenancePointer]]:
        ranked = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
        retained: list[tuple[float, int, str, ProvenancePointer]] = []
        seen: set[str] = set()
        used = 0
        for item in ranked:
            normalized = " ".join(item[2].casefold().split())
            if normalized in seen:
                continue
            tokens = self.token_counter(item[2])
            if used + tokens > self.memory_tokens:
                continue
            retained.append(item)
            seen.add(normalized)
            used += tokens
        return sorted(retained, key=lambda item: item[1])


class RecurrentMemoryBuilder:
    """MemAgent-style recurrence whose raw inputs remain in the evidence store."""

    def __init__(
        self,
        store: ImmutableEvidenceStore,
        writer: Callable[[RecurrentWriteRequest], str],
        *,
        config: RecurrentConfig | None = None,
        token_counter: Callable[[str], int] = conservative_token_estimate,
    ) -> None:
        self.store = store
        self.writer = writer
        self.config = config or RecurrentConfig()
        self.config.validate()
        self.token_counter = token_counter

    def process(
        self,
        query: str,
        chunk_ids: Sequence[str],
        *,
        subject: str = "active recurrent memory",
        trace: TraceCollector | None = None,
    ) -> RecurrentResult:
        collector = trace or TraceCollector()
        current: MemoryRecord | None = None
        current_text = ""
        processed: list[str] = []
        provenance: dict[tuple[str, int, int], ProvenancePointer] = {}
        for step, chunk_id in enumerate(chunk_ids):
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                raise KeyError(f"unknown recurrent source chunk: {chunk_id}")
            if chunk.token_end - chunk.token_start > self.config.chunk_tokens:
                raise ValueError(
                    f"chunk {chunk_id} exceeds the configured recurrent chunk budget"
                )
            collector.record(
                MemoryOperation.PAGE_IN,
                chunk_id,
                level="L4->L2",
                step=step,
                source_tokens=chunk.token_end - chunk.token_start,
            )
            source_ids = [*processed, chunk_id]
            regenerate = bool(
                self.config.regenerate_every
                and (step + 1) % self.config.regenerate_every == 0
            )
            if regenerate:
                updated = self._regenerate_from_raw(query, source_ids)
                collector.record(
                    MemoryOperation.RECONSTRUCT,
                    current.memory_id if current is not None else None,
                    reason="compression_generation_refresh",
                    source_chunk_ids=source_ids,
                    prior_generation=(
                        current.compression_generation if current is not None else 0
                    ),
                )
            else:
                updated = str(
                    self.writer(
                        RecurrentWriteRequest(
                            query=query,
                            previous_memory=current_text,
                            chunk=chunk,
                            memory_token_budget=self.config.memory_tokens,
                            step=step,
                        )
                    )
                    or ""
                ).strip()
            tokens = self.token_counter(updated)
            if not updated:
                raise ValueError("recurrent writer returned empty memory")
            if tokens > self.config.memory_tokens:
                raise ValueError(
                    f"recurrent writer exceeded memory budget: {tokens} > "
                    f"{self.config.memory_tokens}"
                )
            pointer = ProvenancePointer(
                chunk_id=chunk.chunk_id,
                char_start=0,
                char_end=len(chunk.original_text),
                exact=True,
            )
            provenance[(pointer.chunk_id, pointer.char_start, pointer.char_end)] = pointer
            current = self.store.write_memory(
                MemoryClass.EPISODE,
                subject,
                updated,
                provenance=provenance.values(),
                supersedes=current.memory_id if current is not None else None,
                importance=0.65,
                compression_generation=(
                    1
                    if regenerate
                    else (current.compression_generation + 1 if current is not None else 1)
                ),
                verified=False,
                metadata={
                    "query": query,
                    "memory_token_budget": self.config.memory_tokens,
                    "chunk_token_budget": self.config.chunk_tokens,
                    "step": step,
                    "regenerated_from_raw": regenerate,
                },
            )
            current_text = updated
            processed.append(chunk_id)
            collector.record(
                MemoryOperation.MERGE,
                current.memory_id,
                step=step,
                memory_tokens=tokens,
                source_chunk_ids=list(processed),
                compression_generation=current.compression_generation,
            )
        return RecurrentResult(
            memory=current,
            processed_chunk_ids=tuple(processed),
            trace=collector.export(),
        )

    def _regenerate_from_raw(self, query: str, chunk_ids: Sequence[str]) -> str:
        """Rebuild from immutable chunks instead of another derived-memory generation."""

        memory = ""
        for step, chunk_id in enumerate(chunk_ids):
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:  # pragma: no cover - checked by the caller path
                raise KeyError(f"unknown recurrent source chunk: {chunk_id}")
            memory = str(
                self.writer(
                    RecurrentWriteRequest(
                        query=query,
                        previous_memory=memory,
                        chunk=chunk,
                        memory_token_budget=self.config.memory_tokens,
                        step=step,
                    )
                )
                or ""
            ).strip()
            if not memory:
                raise ValueError("recurrent writer returned empty memory during regeneration")
            tokens = self.token_counter(memory)
            if tokens > self.config.memory_tokens:
                raise ValueError(
                    f"recurrent writer exceeded memory budget during regeneration: {tokens} > "
                    f"{self.config.memory_tokens}"
                )
        return memory
