"""Typed records shared by virtual-context storage, retrieval, and packing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryClass(str, Enum):
    """Derived memory classes with independent lifecycle semantics."""

    FACT = "fact"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    EPISODE = "episode"
    OPEN_QUESTION = "open_question"
    CURRENT_PLAN = "current_plan"
    LATENT_MEMORY = "latent_memory"


@dataclass(frozen=True)
class MemoryPolicy:
    """Lifecycle defaults kept separate for each derived memory class."""

    default_importance: float
    default_ttl_seconds: float | None
    pinned_when_active: bool = False


MEMORY_POLICIES: dict[MemoryClass, MemoryPolicy] = {
    MemoryClass.FACT: MemoryPolicy(0.72, None),
    MemoryClass.CONSTRAINT: MemoryPolicy(1.0, None, pinned_when_active=True),
    MemoryClass.DECISION: MemoryPolicy(0.9, None),
    MemoryClass.ENTITY: MemoryPolicy(0.62, None),
    MemoryClass.RELATIONSHIP: MemoryPolicy(0.7, None),
    MemoryClass.EPISODE: MemoryPolicy(0.45, 90 * 24 * 60 * 60),
    MemoryClass.OPEN_QUESTION: MemoryPolicy(0.82, 30 * 24 * 60 * 60),
    MemoryClass.CURRENT_PLAN: MemoryPolicy(1.0, 7 * 24 * 60 * 60, pinned_when_active=True),
    MemoryClass.LATENT_MEMORY: MemoryPolicy(0.3, 7 * 24 * 60 * 60),
}


class ControllerAction(str, Enum):
    """Observable InfMem-style controller actions."""

    PRETHINK = "PRETHINK"
    RETRIEVE = "RETRIEVE"
    WRITE = "WRITE"
    ANSWER = "ANSWER"
    STOP = "STOP"


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    message_id: str | None
    source: str
    version: str
    captured_at: float
    ordinal: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    parent_kind: str | None
    parent_name: str | None
    previous_chunk_id: str | None
    next_chunk_id: str | None
    content_hash: str
    original_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvenancePointer:
    chunk_id: str
    char_start: int
    char_end: int
    exact: bool = True


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    memory_class: MemoryClass
    subject: str
    content: str
    created_at: float
    valid_from: float
    valid_to: float | None
    ttl_seconds: float | None
    importance: float
    version: int
    supersedes: str | None
    compression_generation: int
    verified: bool
    provenance: tuple[ProvenancePointer, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: EvidenceChunk
    score: float
    channels: tuple[str, ...]
    channel_scores: dict[str, float] = field(default_factory=dict)
    graph_distance: int | None = None


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    memory_level: str
    category: str
    text: str
    tokens: int
    pinned: bool
    provenance: tuple[ProvenancePointer, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class WorkingContext:
    text: str
    items: tuple[ContextItem, ...]
    token_usage: dict[str, int]
    total_tokens: int
    max_tokens: int
    evidence_chunk_ids: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]
