"""Bounded recurrent textual memory over externally recoverable chunks."""

from __future__ import annotations

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
