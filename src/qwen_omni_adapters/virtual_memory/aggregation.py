"""Deterministic corpus-wide aggregations backed by immutable provenance."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import (
    EvidenceChunk,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
)
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
}
_FREQUENCY_REQUEST = re.compile(
    r"\b(?:most\s+common|most\s+frequent(?:ly)?(?:\s+appeared)?)\s+words?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrequencyAggregationResult:
    memory: MemoryRecord
    terms: tuple[str, ...]
    counts: tuple[int, ...]
    trace: tuple[dict[str, object], ...]


def requested_frequency_count(query: str) -> int | None:
    """Return requested top-k for a corpus word-frequency question."""

    if not _FREQUENCY_REQUEST.search(query):
        return None
    before = query[: _FREQUENCY_REQUEST.search(query).start()]
    digit = re.search(r"\b(\d{1,3})\s*$", before)
    if digit:
        return max(1, min(100, int(digit.group(1))))
    word = re.search(
        rf"\b({'|'.join(_NUMBER_WORDS)})\s*$", before, re.IGNORECASE
    )
    if word:
        return _NUMBER_WORDS[word.group(1).casefold()]
    return 10


class FrequencyAggregationBuilder:
    """Compile a bounded top-k view without discarding its raw source."""

    def __init__(self, store: ImmutableEvidenceStore) -> None:
        self.store = store

    def build(
        self,
        query: str,
        source_text: str,
        chunks: Sequence[EvidenceChunk],
        *,
        trace: TraceCollector | None = None,
    ) -> FrequencyAggregationResult | None:
        limit = requested_frequency_count(query)
        if limit is None:
            return None
        numbered = re.findall(
            r"(?:^|\s)\d+\.\s+([A-Za-z][A-Za-z'-]*)",
            source_text,
        )
        # Numbered-list extraction avoids counting the benchmark/task prose.
        # Free-form coded streams have no list markers; their repeated codes
        # dominate the one-off instruction words by construction.
        values = numbered if len(numbered) >= limit * 2 else re.findall(
            r"\b[A-Za-z][A-Za-z'-]*\b", source_text
        )
        counter = Counter(value.casefold() for value in values)
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
        if len(ranked) < limit:
            return None
        pointers = [
            ProvenancePointer(
                chunk_id=chunk.chunk_id,
                char_start=0,
                char_end=len(chunk.original_text),
                exact=True,
            )
            for chunk in chunks
        ]
        content = "Deterministic corpus word-frequency aggregation:\n" + "\n".join(
            f"rank={rank} term={term} count={count}"
            for rank, (term, count) in enumerate(ranked, start=1)
        )
        memory = self.store.write_memory(
            MemoryClass.FACT,
            "corpus_word_frequency",
            content,
            provenance=pointers,
            importance=0.98,
            verified=True,
            compression_generation=1,
            metadata={
                "operation": "deterministic_word_frequency",
                "top_k": limit,
                "source_chunks": len(chunks),
                "numbered_list": bool(numbered),
            },
        )
        collector = trace or TraceCollector()
        collector.record(
            "AGGREGATE",
            memory.memory_id,
            aggregation="word_frequency",
            top_k=limit,
            source_chunk_ids=[chunk.chunk_id for chunk in chunks],
            terms=[term for term, _count in ranked],
            counts=[count for _term, count in ranked],
            exact_source_preserved=True,
        )
        return FrequencyAggregationResult(
            memory=memory,
            terms=tuple(term for term, _count in ranked),
            counts=tuple(count for _term, count in ranked),
            trace=collector.export(),
        )
