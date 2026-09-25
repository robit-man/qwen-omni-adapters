"""Conservative extraction of durable structured memories from authoritative text."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import (
    EvidenceChunk,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
)
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore

_CONSTRAINT_RE = re.compile(
    r"(?:\b(?:must(?:\s+not)?|shall(?:\s+not)?)\b|"
    r"^(?:never|always|do\s+not|don't)\b)",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"^(?P<subject>[A-Za-z_][\w.:-]{1,100})\s+"
    r"(?:changed\s+from\b.+\bto|is\s+now|was\s+(?:changed|set)\s+to|"
    r"is\s+configured\s+as)\s+",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"^(?P<subject>[A-Za-z_][\w.:-]{1,100})\s*(?:=|:)\s*(?P<value>\S.+)$"
)
_DECISION_RE = re.compile(
    r"\b(?:we|i)\s+(?:decided|chose|selected)\s+(?:to\s+use\s+|that\s+)?"
    r"(?P<subject>[A-Za-z_][\w.:-]{1,100})",
    re.IGNORECASE,
)
_OPEN_RE = re.compile(r"^(?:todo|open question|unresolved)\s*[:\-]", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedCandidate:
    memory_class: MemoryClass
    subject: str
    content: str
    char_start: int
    char_end: int
    explicit_update: bool = False


def _spans(text: str):
    """Yield line/sentence spans while retaining exact source offsets."""

    for line_match in re.finditer(r"[^\r\n]+", text):
        line = line_match.group(0)
        cursor = 0
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            start = line.find(sentence, cursor)
            if start < 0:
                continue
            cursor = start + len(sentence)
            stripped = sentence.strip()
            if not stripped:
                continue
            left = len(sentence) - len(sentence.lstrip())
            absolute = line_match.start() + start + left
            yield stripped, absolute, absolute + len(stripped)


class StructuredMemoryExtractor:
    """Promote only high-signal statements; raw evidence always remains primary."""

    def __init__(self, store: ImmutableEvidenceStore) -> None:
        self.store = store

    @staticmethod
    def _subject(value: str) -> str:
        return value.strip().strip("`'\"").rstrip(".:")

    def candidates(self, chunk: EvidenceChunk) -> list[ExtractedCandidate]:
        candidates = []
        for content, start, end in _spans(chunk.original_text):
            if _CONSTRAINT_RE.search(content) and not re.match(
                r"^I\s+must\b", content, re.IGNORECASE
            ):
                digest = hashlib.sha256(content.casefold().encode()).hexdigest()[:16]
                candidates.append(
                    ExtractedCandidate(
                        MemoryClass.CONSTRAINT,
                        f"constraint:{digest}",
                        content,
                        start,
                        end,
                    )
                )
                continue
            update = _UPDATE_RE.match(content)
            if update:
                candidates.append(
                    ExtractedCandidate(
                        MemoryClass.DECISION,
                        self._subject(update.group("subject")),
                        content,
                        start,
                        end,
                        explicit_update=True,
                    )
                )
                continue
            decision = _DECISION_RE.search(content)
            if decision:
                candidates.append(
                    ExtractedCandidate(
                        MemoryClass.DECISION,
                        self._subject(decision.group("subject")),
                        content,
                        start,
                        end,
                    )
                )
                continue
            assignment = _ASSIGNMENT_RE.match(content)
            if assignment:
                candidates.append(
                    ExtractedCandidate(
                        MemoryClass.FACT,
                        self._subject(assignment.group("subject")),
                        content,
                        start,
                        end,
                    )
                )
                continue
            if _OPEN_RE.match(content):
                digest = hashlib.sha256(content.casefold().encode()).hexdigest()[:16]
                candidates.append(
                    ExtractedCandidate(
                        MemoryClass.OPEN_QUESTION,
                        f"question:{digest}",
                        content,
                        start,
                        end,
                    )
                )
        return candidates

    def extract(
        self,
        chunks: list[EvidenceChunk],
        *,
        authority: str,
    ) -> list[MemoryRecord]:
        written = []
        for chunk in chunks:
            for candidate in self.candidates(chunk):
                active = self.store.active_memories(
                    classes=[candidate.memory_class], subject=candidate.subject
                )
                if any(memory.content == candidate.content for memory in active):
                    continue
                supersedes = (
                    active[0].memory_id
                    if candidate.explicit_update and active
                    else None
                )
                written.append(
                    self.store.write_memory(
                        candidate.memory_class,
                        candidate.subject,
                        candidate.content,
                        provenance=[
                            ProvenancePointer(
                                chunk.chunk_id,
                                candidate.char_start,
                                candidate.char_end,
                                exact=True,
                            )
                        ],
                        supersedes=supersedes,
                        metadata={
                            "extracted": True,
                            "authority": authority,
                            "source": chunk.source,
                        },
                    )
                )
        return written
