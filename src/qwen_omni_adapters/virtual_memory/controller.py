"""Recursive PRETHINK -> RETRIEVE -> WRITE -> ANSWER/STOP controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import ControllerAction, RetrievalHit
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector


@dataclass(frozen=True)
class ControllerConfig:
    max_rounds: int = 4
    minimum_evidence: int = 1
    sufficiency_threshold: float = 0.78


@dataclass(frozen=True)
class ControllerResult:
    evidence: tuple[RetrievalHit, ...]
    queries: tuple[str, ...]
    sufficient: bool
    stopped_early: bool
    trace: tuple[dict[str, object], ...]


class RecursiveMemoryController:
    """Training-free controller with optional model planning and sufficiency hooks."""

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        config: ControllerConfig | None = None,
        dependency_planner: Callable[
            [str, Sequence[RetrievalHit], Sequence[str]], Sequence[str]
        ]
        | None = None,
        sufficiency_judge: Callable[[str, Sequence[RetrievalHit]], float] | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or ControllerConfig()
        self.dependency_planner = dependency_planner
        self.sufficiency_judge = sufficiency_judge

    def gather(self, query: str, *, trace: TraceCollector | None = None) -> ControllerResult:
        collector = trace or TraceCollector()
        pending = [query]
        queries: list[str] = []
        evidence: dict[str, RetrievalHit] = {}
        sufficient = False
        stopped_early = False
        for round_index in range(max(1, self.config.max_rounds)):
            collector.record(
                ControllerAction.PRETHINK.value,
                round=round_index,
                known_chunk_ids=list(evidence),
                pending_queries=pending,
                missing_evidence=(not evidence),
            )
            current = next((item for item in pending if item not in queries), None)
            if current is None:
                current = query
            queries.append(current)
            collector.record(ControllerAction.RETRIEVE.value, query=current, round=round_index)
            found = self.retriever.retrieve(current, trace=collector)
            changed = False
            for hit in found:
                previous = evidence.get(hit.chunk.chunk_id)
                if previous is None or hit.score > previous.score:
                    evidence[hit.chunk.chunk_id] = hit
                    changed = True
            ranked = sorted(evidence.values(), key=lambda hit: hit.score, reverse=True)
            collector.record(
                ControllerAction.WRITE.value,
                round=round_index,
                changed=changed,
                retained_chunk_ids=[hit.chunk.chunk_id for hit in ranked],
                note=(
                    "Evidence view updated; immutable sources were not modified."
                    if changed
                    else "No stronger evidence found; working evidence was frozen."
                ),
            )
            score = self._sufficiency(query, ranked)
            sufficient = (
                len(ranked) >= self.config.minimum_evidence
                and score >= self.config.sufficiency_threshold
            )
            collector.record(
                "evidence_sufficiency",
                round=round_index,
                score=round(score, 6),
                threshold=self.config.sufficiency_threshold,
                sufficient=sufficient,
            )
            if sufficient:
                stopped_early = round_index + 1 < self.config.max_rounds
                collector.record(
                    ControllerAction.STOP.value,
                    reason="evidence_sufficient",
                    round=round_index,
                )
                break
            additions = (
                list(self.dependency_planner(query, ranked, tuple(queries)))
                if self.dependency_planner is not None
                else self._dependency_queries(query, ranked, queries)
            )
            pending.extend(
                candidate.strip()
                for candidate in additions
                if candidate.strip() and candidate.strip() not in pending
            )
            if not changed and not any(item not in queries for item in pending):
                collector.record(
                    ControllerAction.STOP.value,
                    reason="retrieval_stalled",
                    round=round_index,
                )
                break
        ranked = sorted(evidence.values(), key=lambda hit: hit.score, reverse=True)
        collector.record(
            ControllerAction.ANSWER.value,
            allowed=sufficient,
            evidence_chunk_ids=[hit.chunk.chunk_id for hit in ranked],
            unresolved=not sufficient,
        )
        return ControllerResult(
            evidence=tuple(ranked),
            queries=tuple(queries),
            sufficient=sufficient,
            stopped_early=stopped_early,
            trace=collector.export(),
        )

    def _sufficiency(self, query: str, evidence: Sequence[RetrievalHit]) -> float:
        if self.sufficiency_judge is not None:
            return max(0.0, min(1.0, float(self.sufficiency_judge(query, evidence))))
        if not evidence:
            return 0.0
        query_terms = {
            term.casefold()
            for term in query.replace("?", " ").split()
            if len(term.strip(".,:;()[]")) > 2
        }
        evidence_terms = {
            term.casefold().strip(".,:;()[]")
            for hit in evidence
            for term in hit.chunk.original_text.split()
        }
        coverage = len(query_terms & evidence_terms) / max(1, len(query_terms))
        channel_bonus = min(
            0.25,
            len({channel for hit in evidence for channel in hit.channels}) * 0.04,
        )
        return min(1.0, coverage * 0.7 + max(hit.score for hit in evidence) * 0.2 + channel_bonus)

    @staticmethod
    def _dependency_queries(
        query: str,
        evidence: Sequence[RetrievalHit],
        history: Sequence[str],
    ) -> list[str]:
        # Pull newly discovered exact identifiers into focused follow-up
        # queries.  This provides a deterministic multi-hop baseline; learned
        # planners can replace it without changing storage or provenance.
        known = " ".join(hit.chunk.original_text for hit in evidence)
        identifiers = []
        for token in known.replace("(", " ").replace(")", " ").split():
            cleaned = token.strip("`'\".,:;[]{}")
            if (
                3 <= len(cleaned) <= 80
                and ("_" in cleaned or "." in cleaned or cleaned[:1].isupper())
                and cleaned.casefold() not in query.casefold()
            ):
                identifiers.append(cleaned)
        result = []
        for identifier in dict.fromkeys(identifiers):
            candidate = f"{query} {identifier}"
            if candidate not in history:
                result.append(candidate)
            if len(result) >= 3:
                break
        return result
