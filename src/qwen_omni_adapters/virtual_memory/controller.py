"""Recursive PRETHINK -> RETRIEVE -> WRITE -> ANSWER/STOP controller."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import ControllerAction, RetrievalHit
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector


@dataclass(frozen=True)
class ControllerConfig:
    max_rounds: int = 6
    max_queries_per_round: int = 4
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
        terminated = False
        for round_index in range(max(1, self.config.max_rounds)):
            batch = [item for item in pending if item not in queries][
                : max(1, self.config.max_queries_per_round)
            ]
            if not batch:
                batch = [query] if not queries else []
            collector.record(
                ControllerAction.PRETHINK.value,
                round=round_index,
                known_chunk_ids=list(evidence),
                pending_queries=batch,
                missing_evidence=(not evidence),
            )
            changed = False
            for current in batch:
                queries.append(current)
                collector.record(
                    ControllerAction.RETRIEVE.value,
                    query=current,
                    round=round_index,
                )
                found = self.retriever.retrieve(current, trace=collector)
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
            additions = (
                list(self.dependency_planner(query, ranked, tuple(queries)))
                if self.dependency_planner is not None
                else self._dependency_queries(query, ranked, queries)
            )
            new_dependencies = [
                candidate.strip()
                for candidate in additions
                if candidate.strip()
                and candidate.strip() not in pending
                and candidate.strip() not in queries
            ]
            pending.extend(new_dependencies)
            unresolved_now = [item for item in pending if item not in queries]
            collector.record(
                "evidence_sufficiency",
                round=round_index,
                score=round(score, 6),
                threshold=self.config.sufficiency_threshold,
                sufficient=sufficient,
                unresolved_dependencies=unresolved_now,
            )
            # A coherent resident window is not sufficient while its own
            # evidence exposes uninspected dependencies.  Page those sources
            # in before permitting ANSWER/STOP.
            if sufficient and not unresolved_now:
                stopped_early = round_index + 1 < self.config.max_rounds
                collector.record(
                    ControllerAction.STOP.value,
                    reason="evidence_sufficient",
                    round=round_index,
                )
                terminated = True
                break
            if not changed and not any(item not in queries for item in pending):
                collector.record(
                    ControllerAction.STOP.value,
                    reason="retrieval_stalled",
                    round=round_index,
                )
                terminated = True
                break
        ranked = sorted(evidence.values(), key=lambda hit: hit.score, reverse=True)
        unresolved_dependencies = [item for item in pending if item not in queries]
        if unresolved_dependencies:
            sufficient = False
        if not terminated:
            collector.record(
                ControllerAction.STOP.value,
                reason="retrieval_budget_exhausted",
                round=max(0, self.config.max_rounds - 1),
                unresolved_dependencies=unresolved_dependencies,
            )
        collector.record(
            ControllerAction.ANSWER.value,
            allowed=sufficient,
            evidence_chunk_ids=[hit.chunk.chunk_id for hit in ranked],
            unresolved=not sufficient,
            unresolved_dependencies=unresolved_dependencies,
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
        # Pull identifiers from explicit relationships into focused follow-up
        # queries.  Merely starting with a capital letter is deliberately not
        # enough: prose such as "The" and "Here" previously consumed the
        # dependency budget and caused false evidence-sufficiency decisions.
        known = " ".join(hit.chunk.original_text for hit in evidence)
        relationships: list[tuple[str, str]] = []
        relation_patterns = (
            r"\bVAR\s+([A-Z][A-Z0-9_]{1,79})\s*=\s*(?:VAR\s+)?([A-Z0-9_][A-Z0-9_.:-]{1,79})",
            r"(?m)^\s*(?!VAR\b)([A-Za-z_$][\w.$:-]{1,79})\s*=\s*"
            r"(?!VAR\b)([A-Za-z_$][\w.$:-]{1,79})",
            r"\b([A-Za-z_$][\w.$:-]{1,79})\s+"
            r"(?:calls|imports|inherits|extends|uses|references|depends\s+on)\s+"
            r"([A-Za-z_$][\w.$:-]{1,79})",
        )
        for pattern in relation_patterns:
            for match in re.finditer(pattern, known, re.IGNORECASE):
                left, right = (
                    value.strip("`'\".,:;[]{}") for value in match.groups()
                )
                if left and right:
                    relationships.append((left, right))
        # Only traverse the relationship component anchored in the current
        # question or an already inspected dependency.  Broad lexical recall
        # can surface unrelated assignment graphs (RULER includes a few-shot
        # chain); following every identifier would falsely satisfy the task.
        inspected = " ".join(history).casefold()
        anchors = {
            token.casefold()
            for token in re.findall(r"[A-Za-z_$][\w.$:-]{1,79}|\d+", f"{query} {inspected}")
        }
        connected = set(anchors)
        changed = True
        while changed:
            changed = False
            for left, right in relationships:
                left_folded = left.casefold()
                right_folded = right.casefold()
                if left_folded in connected and right_folded not in connected:
                    connected.add(right_folded)
                    changed = True
                if right_folded in connected and left_folded not in connected:
                    connected.add(left_folded)
                    changed = True
        result = []
        seen_identifiers: set[str] = set()
        for identifier in (
            value for relationship in relationships for value in relationship
        ):
            cleaned = identifier.strip("`'\".,:;[]{}")
            if not 2 <= len(cleaned) <= 80:
                continue
            folded = cleaned.casefold()
            if folded in seen_identifiers:
                continue
            seen_identifiers.add(folded)
            if folded not in connected or folded in query.casefold() or folded in inspected:
                continue
            candidate = f'{query} "{cleaned}"'
            if candidate not in history:
                result.append(candidate)
            if len(result) >= 8:
                break
        return result
