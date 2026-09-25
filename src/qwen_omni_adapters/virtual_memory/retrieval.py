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
    "answer",
    "are",
    "did",
    "do",
    "does",
    "find",
    "give",
    "identify",
    "in",
    "is",
    "list",
    "provide",
    "question",
    "recall",
    "reply",
    "respond",
    "return",
    "show",
    "tell",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
}
RETRIEVAL_CHANNELS = frozenset(
    {
        "bm25",
        "code_graph",
        "dense",
        "entity",
        "entity_exact",
        "exact",
        "graph",
        "metadata",
        "recency",
        "symbol",
    }
)


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


def _anchor_terms(plan: QueryPlan) -> tuple[str, ...]:
    """Return query-derived values that must survive broad reranking.

    Exact identifiers and named entities are address calculations, not merely
    soft relevance hints.  If a request names four independent keys, an MMR
    pass must not spend two slots on overlapping copies of one key while
    silently evicting another.
    """

    return tuple(
        dict.fromkeys(
            value.casefold()
            for value in (*plan.exact_strings, *plan.entities)
            if len(value.strip()) >= 2
        )
    )


def _covered_anchors(hit: RetrievalHit, anchors: Sequence[str]) -> set[str]:
    text = hit.chunk.original_text.casefold()
    return {anchor for anchor in anchors if anchor in text}


def _overlapping_source(left: EvidenceChunk, right: EvidenceChunk) -> bool:
    return (
        left.document_id == right.document_id
        and left.version == right.version
        and left.char_start < right.char_end
        and right.char_start < left.char_end
    )


class HybridRetriever:
    """Union broad indexes, rerank, and enforce source diversity."""

    def __init__(
        self,
        store: ImmutableEvidenceStore,
        *,
        query_embedder: Callable[[str], Sequence[float] | None] | None = None,
        reranker: Callable[[str, Sequence[EvidenceChunk]], Sequence[float]] | None = None,
        candidate_limit: int = 200,
        cheap_limit: int = 36,
        final_limit: int = 12,
        source_cap: int = 4,
        mmr_lambda: float = 0.76,
        enabled_channels: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.query_embedder = query_embedder
        self.reranker = reranker
        self.candidate_limit = max(50, min(200, candidate_limit))
        self.cheap_limit = max(20, min(40, cheap_limit))
        self.final_limit = max(1, min(15, final_limit))
        self.source_cap = max(1, source_cap)
        self.mmr_lambda = max(0.0, min(1.0, mmr_lambda))
        selected_channels = (
            RETRIEVAL_CHANNELS
            if enabled_channels is None
            else frozenset(str(channel).strip() for channel in enabled_channels)
        )
        unknown = selected_channels - RETRIEVAL_CHANNELS
        if unknown:
            raise ValueError(f"unknown retrieval channels: {', '.join(sorted(unknown))}")
        if not selected_channels:
            raise ValueError("at least one retrieval channel must be enabled")
        self.enabled_channels = selected_channels

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
            term.strip("`") for term in re.findall(r"`([A-Za-z_$][\w.$:-]{1,120})`", query)
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
                enabled_channels=sorted(self.enabled_channels),
            )
        candidates: dict[str, EvidenceChunk] = {}
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        graph_distances: dict[str, int] = {}

        def add(chunk: EvidenceChunk, channel: str, score: float) -> None:
            candidates[chunk.chunk_id] = chunk
            scores[chunk.chunk_id][channel] = max(score, scores[chunk.chunk_id].get(channel, 0.0))

        if "exact" in self.enabled_channels:
            for exact in selected_plan.exact_strings:
                for rank, chunk in enumerate(
                    self.store.exact_search(exact, limit=self.candidate_limit)
                ):
                    add(chunk, "exact", 1.0 / (1.0 + rank * 0.02))
        for symbol in selected_plan.symbols:
            if "symbol" in self.enabled_channels:
                for rank, chunk in enumerate(
                    self.store.symbol_search(symbol, limit=self.candidate_limit)
                ):
                    add(chunk, "symbol", 0.98 / (1.0 + rank * 0.03))
            if "code_graph" in self.enabled_channels:
                for chunk, distance, predicates in self.store.code_search(
                    symbol, max_hops=2, limit=self.candidate_limit
                ):
                    add(chunk, "code_graph", 0.9 / (1.0 + distance * 0.4))
                    if predicates:
                        scores[chunk.chunk_id]["code_graph"] += min(
                            0.08, len(predicates) * 0.02
                        )
        for subquery in selected_plan.subqueries:
            if "bm25" in self.enabled_channels:
                for chunk, score in self.store.lexical_search(
                    subquery, limit=self.candidate_limit
                ):
                    add(chunk, "bm25", score)
            if "dense" in self.enabled_channels and self.query_embedder is not None:
                vector = self.query_embedder(subquery)
                if vector:
                    for chunk, score in self.store.dense_search(vector, limit=self.candidate_limit):
                        add(chunk, "dense", max(0.0, score) * 0.85)
            if "entity" in self.enabled_channels:
                for chunk in self.store.entity_search(
                    subquery, limit=self.candidate_limit
                ):
                    add(chunk, "entity", 0.68)
            if "graph" in self.enabled_channels:
                for chunk, distance in self.store.graph_search(
                    subquery, max_hops=3, limit=self.candidate_limit
                ):
                    add(chunk, "graph", 0.66 / distance)
                    graph_distances[chunk.chunk_id] = min(
                        distance, graph_distances.get(chunk.chunk_id, distance)
                    )
        for entity in selected_plan.entities:
            # Entity indexes are broad by design (aliases and graph neighbors
            # are useful), but a literal mention is the strongest starting
            # page for a named subject.  Keep the two channels separate in
            # telemetry so callers can audit why the page was selected.
            if "entity_exact" in self.enabled_channels:
                for rank, chunk in enumerate(
                    self.store.exact_search(entity, limit=self.candidate_limit)
                ):
                    add(chunk, "entity_exact", 0.96 / (1.0 + rank * 0.02))
            if "entity" in self.enabled_channels:
                for chunk in self.store.entity_search(entity, limit=self.candidate_limit):
                    add(chunk, "entity", 0.72)
        if selected_plan.metadata_filters and "metadata" in self.enabled_channels:
            for chunk in self.store.metadata_search(
                selected_plan.metadata_filters, limit=self.candidate_limit
            ):
                add(chunk, "metadata", 0.74)
        if selected_plan.temporal and "recency" in self.enabled_channels:
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
            if (
                chunk.parent_name
                and chunk.parent_name.casefold() in selected_plan.original.casefold()
            ):
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
        anchors = _anchor_terms(selected_plan)
        # Preserve at least one high-scoring page for every explicitly named
        # address before the cheap reranker.  This is still bounded by the
        # normal 20--40 candidate stage and never consults an expected answer.
        anchor_pages = self._cover_anchors(prelim, anchors, self.cheap_limit)
        anchor_page_ids = {hit.chunk.chunk_id for hit in anchor_pages}
        cheap = [
            *anchor_pages,
            *(
                hit
                for hit in prelim
                if hit.chunk.chunk_id not in anchor_page_ids
            ),
        ][: self.cheap_limit]
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
        final = self._diverse(cheap, required_anchors=anchors)
        if trace is not None:
            covered = set().union(
                *(_covered_anchors(hit, anchors) for hit in final),
                set(),
            )
            trace.record(
                "retrieval_candidates",
                candidates=len(candidates),
                cheap_reranked=len(cheap),
                required_anchors=list(anchors),
                covered_anchors=sorted(covered),
                missing_anchors=sorted(set(anchors) - covered),
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

    @staticmethod
    def _cover_anchors(
        candidates: Sequence[RetrievalHit],
        anchors: Sequence[str],
        limit: int,
    ) -> list[RetrievalHit]:
        remaining = set(anchors)
        selected: list[RetrievalHit] = []
        selected_ids: set[str] = set()
        while remaining and len(selected) < limit:
            ranked = [
                (
                    len(_covered_anchors(hit, tuple(remaining))),
                    hit.score,
                    hit,
                )
                for hit in candidates
                if hit.chunk.chunk_id not in selected_ids
            ]
            useful = [item for item in ranked if item[0] > 0]
            if not useful:
                break
            _coverage, _score, chosen = max(
                useful,
                key=lambda item: (item[0], item[1]),
            )
            selected.append(chosen)
            selected_ids.add(chosen.chunk.chunk_id)
            remaining -= _covered_anchors(chosen, tuple(remaining))
        return selected

    def _diverse(
        self,
        candidates: Sequence[RetrievalHit],
        *,
        required_anchors: Sequence[str] = (),
    ) -> list[RetrievalHit]:
        remaining = list(candidates)
        selected = self._cover_anchors(
            remaining,
            required_anchors,
            self.final_limit,
        )
        selected_ids = {hit.chunk.chunk_id for hit in selected}
        remaining = [
            hit for hit in remaining if hit.chunk.chunk_id not in selected_ids
        ]
        source_counts: dict[str, int] = defaultdict(int)
        for hit in selected:
            source_counts[hit.chunk.source] += 1
        while remaining and len(selected) < self.final_limit:
            scored: list[tuple[float, RetrievalHit]] = []
            for hit in remaining:
                if source_counts[hit.chunk.source] >= self.source_cap:
                    continue
                candidate_anchors = _covered_anchors(hit, required_anchors)
                if any(
                    _overlapping_source(hit.chunk, item.chunk)
                    and candidate_anchors
                    <= _covered_anchors(item, required_anchors)
                    for item in selected
                ):
                    # Overlapping fallback chunks are alternative views of the
                    # same immutable bytes.  They add no new address coverage
                    # and must not displace an independent source location.
                    continue
                redundancy = max(
                    (
                        _jaccard(hit.chunk.original_text, item.chunk.original_text)
                        for item in selected
                    ),
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
