"""High-recall hybrid retrieval with diversity and provenance intact."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qwen_omni_adapters.virtual_memory.models import EvidenceChunk, RetrievalHit
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector

WORD_RE = re.compile(r"[A-Za-z_][\w.:-]*|\d+(?:\.\d+)*")
QUOTED_RE = re.compile(r"['\"]([^'\"]{2,200})['\"]")
CODE_SYMBOL_RE = re.compile(
    r"\b(?:class|function|method|symbol|file|error|exception)\s+"
    r"[`'\"]?([A-Za-z_$][\w.$:-]*)"
)
TEMPORAL_RE = re.compile(r"\b(?:latest|recent|before|after|previous|current|when|timeline)\b", re.I)
ENTITY_LEADING_STOP_WORDS = {
    "are",
    "did",
    "do",
    "does",
    "in",
    "is",
    "question",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
}


@dataclass(frozen=True)
class QueryPlan:
    original: str
    subqueries: tuple[str, ...]
    exact_strings: tuple[str, ...]
    symbols: tuple[str, ...]
    entities: tuple[str, ...]
    metadata_filters: dict[str, Any]
    temporal: bool


def _terms(value: str) -> set[str]:
    return {term.casefold() for term in WORD_RE.findall(value)}


def _jaccard(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 0.0


class HybridRetriever:
    """Union broad indexes, rerank, and enforce source diversity."""

    def __init__(
        self,
        store: ImmutableEvidenceStore,
        *,
        query_embedder: Callable[[str], Sequence[float] | None] | None = None,
        reranker: Callable[[str, Sequence[EvidenceChunk]], Sequence[float]] | None = None,
        candidate_limit: int = 120,
        cheap_limit: int = 36,
        final_limit: int = 12,
        source_cap: int = 4,
        mmr_lambda: float = 0.76,
    ) -> None:
        self.store = store
        self.query_embedder = query_embedder
        self.reranker = reranker
        self.candidate_limit = max(50, min(200, candidate_limit))
        self.cheap_limit = max(20, min(40, cheap_limit))
        self.final_limit = max(1, min(15, final_limit))
        self.source_cap = max(1, source_cap)
        self.mmr_lambda = max(0.0, min(1.0, mmr_lambda))

    def plan(
        self,
        query: str,
        *,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> QueryPlan:
        normalized = " ".join(str(query or "").split())
        if not normalized:
            raise ValueError("retrieval query is required")
        exact_values = [match.group(1) for match in QUOTED_RE.finditer(query)]
        exact_values.extend(
            re.findall(
                r"\b[A-Za-z0-9]+(?:[_.$:-][A-Za-z0-9]+)+\b",
                query,
            )
        )
        exact = tuple(dict.fromkeys(exact_values))
        symbols = list(match.group(1) for match in CODE_SYMBOL_RE.finditer(query))
        symbols.extend(
            term.strip("`")
            for term in re.findall(r"`([A-Za-z_$][\w.$:-]{1,120})`", query)
        )
        # Clausal decomposition is deterministic and intentionally conservative;
        # a model planner may add dependency queries in the recursive controller.
        clauses = [
            clause.strip(" ,;:")
            for clause in re.split(r"\b(?:and then|then|versus|vs\.?|and|but)\b|[?;]", normalized)
            if len(clause.strip(" ,;:")) >= 3
        ]
        subqueries = tuple(dict.fromkeys([normalized, *clauses]))
        entity_candidates = []
        for match in re.finditer(
            r"\b[A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,3}\b",
            query,
        ):
            words = match.group(0).split()
            while words and words[0].casefold() in ENTITY_LEADING_STOP_WORDS:
                words.pop(0)
            if words:
                entity_candidates.append(" ".join(words))
        return QueryPlan(
            original=normalized,
            subqueries=subqueries,
            exact_strings=exact,
            symbols=tuple(dict.fromkeys(symbols)),
            entities=tuple(dict.fromkeys(entity_candidates)),
            metadata_filters=dict(metadata_filters or {}),
            temporal=bool(TEMPORAL_RE.search(query)),
        )

    def retrieve(
        self,
        query: str,
        *,
        plan: QueryPlan | None = None,
        trace: TraceCollector | None = None,
    ) -> list[RetrievalHit]:
        selected_plan = plan or self.plan(query)
        if trace is not None:
            trace.record(
                "query_decomposition",
                subqueries=list(selected_plan.subqueries),
                exact_strings=list(selected_plan.exact_strings),
                symbols=list(selected_plan.symbols),
                entities=list(selected_plan.entities),
                metadata_filters=selected_plan.metadata_filters,
            )
        candidates: dict[str, EvidenceChunk] = {}
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        graph_distances: dict[str, int] = {}

        def add(chunk: EvidenceChunk, channel: str, score: float) -> None:
            candidates[chunk.chunk_id] = chunk
            scores[chunk.chunk_id][channel] = max(
                score, scores[chunk.chunk_id].get(channel, 0.0)
            )

        for exact in selected_plan.exact_strings:
            for rank, chunk in enumerate(
                self.store.exact_search(exact, limit=self.candidate_limit)
            ):
                add(chunk, "exact", 1.0 / (1.0 + rank * 0.02))
        for symbol in selected_plan.symbols:
            for rank, chunk in enumerate(
                self.store.symbol_search(symbol, limit=self.candidate_limit)
            ):
                add(chunk, "symbol", 0.98 / (1.0 + rank * 0.03))
            for chunk, distance, predicates in self.store.code_search(
                symbol, max_hops=2, limit=self.candidate_limit
            ):
                add(chunk, "code_graph", 0.9 / (1.0 + distance * 0.4))
                if predicates:
                    scores[chunk.chunk_id]["code_graph"] += min(
                        0.08, len(predicates) * 0.02
                    )
        for subquery in selected_plan.subqueries:
            for chunk, score in self.store.lexical_search(
                subquery, limit=self.candidate_limit
            ):
                add(chunk, "bm25", score)
            if self.query_embedder is not None:
                vector = self.query_embedder(subquery)
                if vector:
                    for chunk, score in self.store.dense_search(
                        vector, limit=self.candidate_limit
                    ):
                        add(chunk, "dense", max(0.0, score) * 0.85)
            for chunk in self.store.entity_search(
                subquery, limit=self.candidate_limit
            ):
                add(chunk, "entity", 0.68)
            for chunk, distance in self.store.graph_search(
                subquery, max_hops=3, limit=self.candidate_limit
            ):
                add(chunk, "graph", 0.66 / distance)
                graph_distances[chunk.chunk_id] = min(
                    distance, graph_distances.get(chunk.chunk_id, distance)
                )
        for entity in selected_plan.entities:
            for chunk in self.store.entity_search(entity, limit=self.candidate_limit):
                add(chunk, "entity", 0.72)
        if selected_plan.metadata_filters:
            for chunk in self.store.metadata_search(
                selected_plan.metadata_filters, limit=self.candidate_limit
            ):
                add(chunk, "metadata", 0.74)
        if selected_plan.temporal:
            for rank, chunk in enumerate(self.store.recent(limit=32)):
                add(chunk, "recency", 0.35 / (1.0 + rank * 0.08))

        query_terms = _terms(selected_plan.original)
        prelim: list[RetrievalHit] = []
        for chunk_id, chunk in candidates.items():
            channel_scores = scores[chunk_id]
            lexical_overlap = len(query_terms & _terms(chunk.original_text)) / max(
                1, len(query_terms)
            )
            ordered = sorted(channel_scores.values(), reverse=True)
            combined = (ordered[0] if ordered else 0.0) + sum(ordered[1:]) * 0.18
            combined += lexical_overlap * 0.22
            if chunk.parent_name and chunk.parent_name.casefold() in selected_plan.original.casefold():
                combined += 0.18
            lines = [line.strip() for line in chunk.original_text.splitlines() if line.strip()]
            title = lines[1] if len(lines) > 1 and lines[0].lower().startswith("document ") else ""
            for entity in selected_plan.entities:
                if title and title.casefold() == entity.casefold():
                    combined += 0.5
                if "country" in selected_plan.original.casefold() and re.search(
                    rf"\b{re.escape(entity)}\b.{{0,100}}\b"
                    r"(?:region|city|town|village|province|located|situated)\b"
                    r".{0,60}\b(?:in|of)\b",
                    chunk.original_text,
                    re.IGNORECASE | re.DOTALL,
                ):
                    combined += 0.45
            prelim.append(
                RetrievalHit(
                    chunk=chunk,
                    score=combined,
                    channels=tuple(sorted(channel_scores)),
                    channel_scores=dict(channel_scores),
                    graph_distance=graph_distances.get(chunk_id),
                )
            )
        prelim.sort(key=lambda hit: (hit.score, hit.chunk.captured_at), reverse=True)
        cheap = prelim[: self.cheap_limit]
        if self.reranker is not None and cheap:
            reranked = list(self.reranker(selected_plan.original, [hit.chunk for hit in cheap]))
            if len(reranked) != len(cheap):
                raise ValueError("reranker returned a score count that does not match candidates")
            cheap = [
                RetrievalHit(
                    chunk=hit.chunk,
                    score=hit.score * 0.45 + float(score) * 0.55,
                    channels=hit.channels + ("reranker",),
                    channel_scores={**hit.channel_scores, "reranker": float(score)},
                    graph_distance=hit.graph_distance,
                )
                for hit, score in zip(cheap, reranked, strict=True)
            ]
            cheap.sort(key=lambda hit: hit.score, reverse=True)
        final = self._diverse(cheap)
        if trace is not None:
            trace.record(
                "retrieval_candidates",
                candidates=len(candidates),
                cheap_reranked=len(cheap),
                selected=[
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "score": round(hit.score, 6),
                        "channels": hit.channels,
                        "source": hit.chunk.source,
                    }
                    for hit in final
                ],
            )
        return final

    def _diverse(self, candidates: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        remaining = list(candidates)
        selected: list[RetrievalHit] = []
        source_counts: dict[str, int] = defaultdict(int)
        while remaining and len(selected) < self.final_limit:
            scored: list[tuple[float, RetrievalHit]] = []
            for hit in remaining:
                if source_counts[hit.chunk.source] >= self.source_cap:
                    continue
                redundancy = max(
                    (_jaccard(hit.chunk.original_text, item.chunk.original_text) for item in selected),
                    default=0.0,
                )
                mmr = self.mmr_lambda * hit.score - (1.0 - self.mmr_lambda) * redundancy
                scored.append((mmr, hit))
            if not scored:
                break
            _score, chosen = max(scored, key=lambda item: item[0])
            selected.append(chosen)
            source_counts[chosen.chunk.source] += 1
            remaining = [hit for hit in remaining if hit.chunk.chunk_id != chosen.chunk.chunk_id]
        return selected
