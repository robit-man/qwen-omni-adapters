"""Dynamic allocation of a bounded transformer working set."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qwen_omni_adapters.virtual_memory.models import (
    ContextItem,
    MemoryClass,
    MemoryRecord,
    ProvenancePointer,
    RetrievalHit,
    WorkingContext,
)
from qwen_omni_adapters.virtual_memory.telemetry import MemoryOperation, TraceCollector


def conservative_token_estimate(value: str) -> int:
    """Conservative fallback; production should inject the model tokenizer."""

    if not value:
        return 0
    return max(len(re.findall(r"\S+", value)), math.ceil(len(value.encode("utf-8")) / 3))


@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 16_384
    output_headroom: int = 2_384
    system_target: int = 1_280
    pinned_target: int = 1_800
    structured_target: int = 1_400
    recent_target: int = 2_200
    evidence_target: int = 8_000

    @property
    def input_ceiling(self) -> int:
        return self.max_tokens - self.output_headroom

    def validate(self) -> None:
        if self.max_tokens < 4096:
            raise ValueError("max_tokens must be at least 4096")
        if not 512 <= self.output_headroom < self.max_tokens:
            raise ValueError("output_headroom must be between 512 and max_tokens")
        if any(
            value < 0
            for value in (
                self.system_target,
                self.pinned_target,
                self.structured_target,
                self.recent_target,
                self.evidence_target,
            )
        ):
            raise ValueError("context category targets cannot be negative")


class ContextOverflow(RuntimeError):
    """Pinned input cannot fit without violating a hard context invariant."""


class WorkingContextPacker:
    """Allocate 16K by authority and recoverability instead of FIFO order."""

    def __init__(
        self,
        *,
        budget: ContextBudget | None = None,
        token_counter: Callable[[str], int] = conservative_token_estimate,
        kv_bytes_per_token: float | None = None,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.budget.validate()
        self.token_counter = token_counter
        self.kv_bytes_per_token = (
            max(0.0, float(kv_bytes_per_token))
            if kv_bytes_per_token is not None
            else None
        )

    def pack(
        self,
        query: str,
        *,
        system_contract: str,
        evidence: Sequence[RetrievalHit],
        retrieval_queries: Sequence[str] = (),
        memories: Sequence[MemoryRecord] = (),
        recent_context: Sequence[str] = (),
        recurrent_memory: str = "",
        reserved_tokens: int = 0,
        trace: TraceCollector | None = None,
    ) -> WorkingContext:
        collector = trace or TraceCollector()
        query_block = f"<current_query>\n{query.strip()}\n</current_query>"
        query_tokens = self.token_counter(query_block)
        system_tokens = self.token_counter(system_contract)
        external_reserve = max(0, int(reserved_tokens))
        ceiling = self.budget.input_ceiling - external_reserve
        if ceiling <= 0:
            raise ContextOverflow("request envelope leaves no working-context capacity")
        if system_tokens + query_tokens > ceiling:
            raise ContextOverflow("system contract and current query exceed the input ceiling")

        pinned_memories = [
            memory
            for memory in memories
            if memory.memory_class in {MemoryClass.CONSTRAINT, MemoryClass.CURRENT_PLAN}
        ]
        structured_memories = [memory for memory in memories if memory not in pinned_memories]
        pinned_items = [self._memory_item(memory, pinned=True) for memory in pinned_memories]
        pinned_tokens = sum(item.tokens for item in pinned_items)
        if system_tokens + query_tokens + pinned_tokens > ceiling:
            raise ContextOverflow(
                "active constraints/current plan do not fit; narrow task scope before generation"
            )
        for item in pinned_items:
            collector.record(
                MemoryOperation.PIN,
                item.item_id,
                category=item.category,
                tokens=item.tokens,
                provenance=[pointer.chunk_id for pointer in item.provenance],
            )

        fixed = system_tokens + query_tokens + pinned_tokens
        available = ceiling - fixed
        # Reserve a bounded structured-memory allocation before evidence can
        # consume the entire small physical window.  Verified facts and
        # deterministic aggregate views otherwise disappear at 8K even though
        # the 16K policy explicitly budgets 1--2K for them.  Exact evidence is
        # still rendered immediately adjacent to the final query and remains
        # higher authority than derived recurrent/recent state.
        structured_cap = min(self.budget.structured_target, available)
        structured_items = self._select_structured(
            structured_memories,
            structured_cap,
            collector,
        )
        available -= sum(item.tokens for item in structured_items)

        evidence_cap = min(self.budget.evidence_target, available)
        evidence_query = " ".join((query, *retrieval_queries))
        focus_terms = tuple(
            dict.fromkeys(
                term.casefold()
                for retrieval_query in retrieval_queries
                for term in re.findall(r'["\']([^"\']{2,200})["\']', retrieval_query)
            )
        )
        root_focus_terms = self._root_focus_terms(query)
        evidence_items = self._select_evidence(
            evidence_query,
            evidence,
            evidence_cap,
            collector,
            focus_terms=focus_terms,
            root_focus_terms=root_focus_terms,
        )
        available -= sum(item.tokens for item in evidence_items)

        recurrent_items: list[ContextItem] = []
        if recurrent_memory.strip() and available:
            block = (
                "<derived_recurrent_memory authority=\"derived\">\n"
                f"{recurrent_memory.strip()}\n</derived_recurrent_memory>"
            )
            tokens = self.token_counter(block)
            if tokens <= min(self.budget.structured_target, available):
                recurrent_items.append(
                    ContextItem(
                        item_id="recurrent-memory",
                        memory_level="L2",
                        category="recurrent_memory",
                        text=block,
                        tokens=tokens,
                        pinned=False,
                        score=0.3,
                    )
                )
                available -= tokens
                collector.record(
                    MemoryOperation.PAGE_IN,
                    "recurrent-memory",
                    level="L2",
                    tokens=tokens,
                    authority="derived",
                )

        recent_cap = min(self.budget.recent_target, available)
        recent_items = self._select_recent(recent_context, recent_cap, collector)

        system_item = ContextItem(
            item_id="system-contract",
            memory_level="L1",
            category="system_contract",
            text=system_contract,
            tokens=system_tokens,
            pinned=True,
            score=1.0,
        )
        collector.record(
            MemoryOperation.PIN,
            system_item.item_id,
            category=system_item.category,
            tokens=system_item.tokens,
        )
        query_item = ContextItem(
            item_id="current-query",
            memory_level="L0",
            category="current_query",
            text=query_block,
            tokens=query_tokens,
            pinned=True,
            score=1.0,
        )

        # Replayed exact evidence is deliberately adjacent to the final query.
        ordered = [
            system_item,
            *pinned_items,
            *structured_items,
            *recurrent_items,
            *recent_items,
            *evidence_items,
            query_item,
        ]
        total = sum(item.tokens for item in ordered)
        if total > ceiling:  # pragma: no cover - defensive invariant
            raise ContextOverflow(f"packer exceeded input ceiling: {total} > {ceiling}")
        usage: dict[str, int] = {}
        for item in ordered:
            usage[item.category] = usage.get(item.category, 0) + item.tokens
        usage["output_headroom"] = self.budget.output_headroom
        usage["request_envelope"] = external_reserve
        collector.record(
            "token_allocation",
            maximum=self.budget.max_tokens,
            input_ceiling=ceiling,
            input_used=total,
            output_headroom=self.budget.output_headroom,
            request_envelope=external_reserve,
            categories=usage,
            estimated_kv_bytes=(
                int(total * self.kv_bytes_per_token)
                if self.kv_bytes_per_token is not None
                else None
            ),
        )
        return WorkingContext(
            text="\n\n".join(item.text for item in ordered if item.text),
            items=tuple(ordered),
            token_usage=usage,
            total_tokens=total + self.budget.output_headroom + external_reserve,
            max_tokens=self.budget.max_tokens,
            evidence_chunk_ids=tuple(
                pointer.chunk_id
                for item in evidence_items
                for pointer in item.provenance
            ),
            trace=collector.export(),
        )

    def _memory_item(self, memory: MemoryRecord, *, pinned: bool) -> ContextItem:
        source_ids = tuple(dict.fromkeys(pointer.chunk_id for pointer in memory.provenance))
        displayed_sources = source_ids[:12]
        provenance = ",".join(displayed_sources)
        if len(source_ids) > len(displayed_sources):
            provenance += ",..."
        block = (
            f'<memory class="{memory.memory_class.value}" id="{memory.memory_id}" '
            f'verified="{str(memory.verified).lower()}" sources="{provenance}" '
            f'source_count="{len(source_ids)}">\n'
            f"{memory.content}\n</memory>"
        )
        generation_penalty = 1.0 / (1.0 + memory.compression_generation * 0.35)
        provenance_strength = 1.0 if memory.verified else 0.45
        return ContextItem(
            item_id=memory.memory_id,
            memory_level="L1" if pinned else "L2",
            category=memory.memory_class.value,
            text=block,
            tokens=self.token_counter(block),
            pinned=pinned,
            provenance=memory.provenance,
            score=memory.importance * generation_penalty * provenance_strength,
        )

    def _select_structured(
        self,
        memories: Sequence[MemoryRecord],
        cap: int,
        collector: TraceCollector,
    ) -> list[ContextItem]:
        candidates = sorted(
            (self._memory_item(memory, pinned=False) for memory in memories),
            key=lambda item: item.score,
            reverse=True,
        )
        selected = []
        used = 0
        for item in candidates:
            if used + item.tokens <= cap:
                selected.append(item)
                used += item.tokens
                collector.record(
                    MemoryOperation.PAGE_IN,
                    item.item_id,
                    level="L2",
                    category=item.category,
                    tokens=item.tokens,
                    score=item.score,
                )
            else:
                collector.record(
                    MemoryOperation.EVICT,
                    item.item_id,
                    reason="structured_memory_budget",
                    recoverable=True,
                    tokens=item.tokens,
                )
        return selected

    def _select_recent(
        self,
        recent_context: Sequence[str],
        cap: int,
        collector: TraceCollector,
    ) -> list[ContextItem]:
        selected_reversed = []
        used = 0
        for reverse_index, text in enumerate(reversed(recent_context)):
            block = f"<recent_turn>\n{text.strip()}\n</recent_turn>"
            tokens = self.token_counter(block)
            item_id = f"recent-{len(recent_context) - reverse_index - 1}"
            if used + tokens <= cap:
                selected_reversed.append(
                    ContextItem(
                        item_id=item_id,
                        memory_level="L1",
                        category="recent_context",
                        text=block,
                        tokens=tokens,
                        pinned=False,
                        score=1.0 / (1.0 + reverse_index),
                    )
                )
                used += tokens
            else:
                collector.record(
                    MemoryOperation.EVICT,
                    item_id,
                    reason="recent_context_budget",
                    recoverable=True,
                    tokens=tokens,
                )
        selected = list(reversed(selected_reversed))
        for item in selected:
            collector.record(
                MemoryOperation.PAGE_IN,
                item.item_id,
                level="L1",
                category=item.category,
                tokens=item.tokens,
            )
        return selected

    def _select_evidence(
        self,
        query: str,
        evidence: Sequence[RetrievalHit],
        cap: int,
        collector: TraceCollector,
        *,
        focus_terms: Sequence[str] = (),
        root_focus_terms: Sequence[str] = (),
    ) -> list[ContextItem]:
        selected = []
        selected_hits: list[RetrievalHit] = []
        used = 0
        candidates = list(evidence)
        if focus_terms or root_focus_terms:
            focused = [
                hit
                for hit in candidates
                if set(hit.channels)
                & {
                    "code_graph",
                    "graph",
                    "memory_conflict",
                    "oracle",
                    "oracle_reference_location",
                }
                or any(
                    self._contains_focus_term(
                        hit.chunk.original_text,
                        term,
                        identifier_components=self._code_like(hit.chunk),
                    )
                    for term in focus_terms
                )
                or any(
                    self._contains_focus_term(
                        hit.chunk.original_text,
                        term,
                        identifier_components=self._code_like(hit.chunk),
                    )
                    for term in root_focus_terms
                )
            ]
            if focused:
                focused_ids = {hit.chunk.chunk_id for hit in focused}
                for hit in candidates:
                    if hit.chunk.chunk_id not in focused_ids:
                        collector.record(
                            MemoryOperation.EVICT,
                            hit.chunk.chunk_id,
                            reason="dependency_focus",
                            recoverable=True,
                            focus_terms=list(focus_terms),
                        )
                candidates = focused
        ranked = sorted(
            candidates,
            key=lambda hit: self._evidence_priority(query, hit),
            reverse=True,
        )
        for hit in ranked:
            matched_terms = self._matched_query_terms(query, hit.chunk.original_text)
            duplicate = next(
                (
                    previous
                    for previous in selected_hits
                    if self._chunks_overlap(previous.chunk, hit.chunk)
                    and matched_terms
                    <= self._matched_query_terms(query, previous.chunk.original_text)
                ),
                None,
            )
            if duplicate is not None:
                collector.record(
                    MemoryOperation.EVICT,
                    hit.chunk.chunk_id,
                    reason="overlapping_evidence_duplicate",
                    recoverable=True,
                    overlapping_chunk_id=duplicate.chunk.chunk_id,
                )
                continue
            remaining = cap - used
            if remaining <= 0:
                collector.record(
                    MemoryOperation.EVICT,
                    hit.chunk.chunk_id,
                    reason="evidence_budget",
                    recoverable=True,
                )
                continue
            item = self._evidence_item(query, hit, remaining)
            if item is None:
                collector.record(
                    MemoryOperation.EVICT,
                    hit.chunk.chunk_id,
                    reason="evidence_span_did_not_fit",
                    recoverable=True,
                )
                continue
            selected.append(item)
            selected_hits.append(hit)
            used += item.tokens
            collector.record(
                MemoryOperation.PAGE_IN,
                item.item_id,
                level="L3",
                tokens=item.tokens,
                score=item.score,
                channels=hit.channels,
                exact_replay=True,
                eviction_priority=self._evidence_priority(query, hit),
            )
        return selected

    @staticmethod
    def _code_like(chunk: object) -> bool:
        return getattr(chunk, "parent_kind", None) in {
            "class",
            "function",
            "module_block",
            "module_gap",
            "module_preamble",
            "module_tail",
            "symbol_block",
        }

    @staticmethod
    def _contains_focus_term(
        text: str,
        term: str,
        *,
        identifier_components: bool = False,
    ) -> bool:
        """Match an address anchor without conflating identifier prefixes."""

        normalized = str(term or "").strip().casefold()
        if not normalized:
            return False
        identifier_class = "A-Za-z0-9" if identifier_components else "A-Za-z0-9_"
        left = rf"(?<![{identifier_class}])" if normalized[0].isalnum() or normalized[0] == "_" else ""
        right = rf"(?![{identifier_class}])" if normalized[-1].isalnum() or normalized[-1] == "_" else ""
        return bool(re.search(f"{left}{re.escape(normalized)}{right}", text.casefold()))

    @staticmethod
    def _matched_query_terms(query: str, text: str) -> set[str]:
        source = text.casefold()
        return {
            term
            for term in {
                value.casefold().strip(".$:-")
                for value in re.findall(r"[A-Za-z0-9_.$:-]{3,}", query)
            }
            if term and term in source
        }

    @staticmethod
    def _chunks_overlap(left: object, right: object) -> bool:
        return bool(
            getattr(left, "document_id", None) == getattr(right, "document_id", None)
            and getattr(left, "version", None) == getattr(right, "version", None)
            and getattr(left, "char_start", 0) < getattr(right, "char_end", 0)
            and getattr(right, "char_start", 0) < getattr(left, "char_end", 0)
        )

    @staticmethod
    def _root_focus_terms(query: str) -> tuple[str, ...]:
        """Distinctive root-query anchors that keep terminal dependency facts.

        Recursive retrieval queries quote intermediate nodes.  Restricting the
        evidence pack to those nodes removes disconnected retrieval noise, but
        a terminal fact can be phrased in terms of the requested property
        (for example ``bitrate``) rather than the last quoted node.  Preserve
        high-information root terms as a second, query-derived inclusion path.
        """

        stop = {
            "answer",
            "all",
            "are",
            "does",
            "from",
            "have",
            "including",
            "into",
            "only",
            "return",
            "special",
            "that",
            "the",
            "their",
            "then",
            "through",
            "using",
            "what",
            "when",
            "where",
            "which",
            "with",
            # Generic syntax keywords are not dependency identities.
            "var",
            "variables",
        }
        quoted = tuple(
            dict.fromkeys(
                term.casefold().strip()
                for term in re.findall(r"[`'\"]([^`'\"]{2,200})[`'\"]", query)
                if term.strip()
            )
        )
        addresses = tuple(
            dict.fromkeys(
                term.casefold().strip(".$:-")
                for term in re.findall(
                    r"\b[A-Za-z0-9]+(?:[_.$:-][A-Za-z0-9]+)+\b",
                    query,
                )
                if term.casefold().strip(".$:-") not in stop
            )
        )
        distinctive = tuple(dict.fromkeys((*quoted, *addresses)))
        if distinctive:
            # Once a request supplies an exact address, broad query prose such
            # as "exact value" or "which request" must not keep a prefix
            # collision resident.  Topology-authoritative graph/code hits are
            # retained separately by _select_evidence.
            return distinctive
        return tuple(
            dict.fromkeys(
                term.casefold().strip(".$:-")
                for term in re.findall(r"[A-Za-z0-9_.$:-]{4,}", query)
                if term.casefold().strip(".$:-") not in stop
            )
        )

    @staticmethod
    def _evidence_priority(query: str, hit: RetrievalHit) -> float:
        """Content-aware resident priority, independent of FIFO position."""

        authority = 0.0
        channels = set(hit.channels)
        if channels & {
            "exact",
            "symbol",
            "memory_conflict",
            "oracle",
            "oracle_reference_location",
        }:
            authority += 0.4
        if channels & {"code_graph", "graph"}:
            authority += 0.18
        if len(channels) > 1:
            authority += min(0.2, (len(channels) - 1) * 0.05)
        query_terms = {
            term.casefold() for term in re.findall(r"[A-Za-z0-9_.$:-]{3,}", query)
        }
        source_terms = {
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9_.$:-]{3,}", hit.chunk.original_text)
        }
        relevance = len(query_terms & source_terms) / max(1, len(query_terms))
        recovery_cost = min(0.15, 1200 / max(1200, len(hit.chunk.original_text)) * 0.15)
        dependency = 0.12 / hit.graph_distance if hit.graph_distance else 0.0
        return hit.score + authority + relevance * 0.25 + dependency + recovery_cost

    def _evidence_item(
        self, query: str, hit: RetrievalHit, remaining: int
    ) -> ContextItem | None:
        chunk = hit.chunk

        def format_block(text: str, start: int, end: int) -> str:
            return (
                f'<exact_evidence chunk_id="{chunk.chunk_id}" source="{chunk.source}" '
                f'document_id="{chunk.document_id}" version="{chunk.version}" '
                f'chunk_chars="{start}:{end}">\n{text}\n</exact_evidence>'
            )

        full = format_block(chunk.original_text, 0, len(chunk.original_text))
        full_tokens = self.token_counter(full)
        # Long chunks are storage pages, not ideal working-set units.  Replay
        # bounded exact spans even when the whole page technically fits so one
        # early page cannot crowd out later dependency edges.  Small source
        # records remain verbatim in full.
        span_budget = min(remaining, 768)
        prefer_spans = full_tokens > span_budget
        if full_tokens <= remaining and not prefer_spans:
            pointer = ProvenancePointer(
                chunk_id=chunk.chunk_id,
                char_start=0,
                char_end=len(chunk.original_text),
                exact=True,
            )
            return ContextItem(
                item_id=chunk.chunk_id,
                memory_level="L3",
                category="exact_evidence",
                text=full,
                tokens=full_tokens,
                pinned=False,
                provenance=(pointer,),
                score=hit.score,
            )
        spans = self._query_spans(query, chunk.original_text)
        selected_spans: list[tuple[int, int]] = []
        block = ""
        tokens = 0
        for start, end in spans:
            # Prefer distinct evidence locations. Paragraph, line, and direct
            # term windows around one match otherwise consume the entire pack
            # while representing the same source bytes.
            if any(not (end <= old_start or start >= old_end) for old_start, old_end in selected_spans):
                continue
            candidate_spans = [*selected_spans, (start, end)]
            body = "\n".join(
                f'<exact_span chunk_chars="{span_start}:{span_end}">\n'
                f"{chunk.original_text[span_start:span_end]}\n</exact_span>"
                for span_start, span_end in candidate_spans
            )
            candidate = (
                f'<exact_evidence chunk_id="{chunk.chunk_id}" source="{chunk.source}" '
                f'document_id="{chunk.document_id}" version="{chunk.version}" '
                f'span_count="{len(candidate_spans)}">\n{body}\n</exact_evidence>'
            )
            candidate_tokens = self.token_counter(candidate)
            if candidate_tokens <= span_budget:
                selected_spans = candidate_spans
                block = candidate
                tokens = candidate_tokens
            if len(selected_spans) >= 12:
                break
        if selected_spans:
            pointers = tuple(
                ProvenancePointer(
                    chunk_id=chunk.chunk_id,
                    char_start=start,
                    char_end=end,
                    exact=True,
                )
                for start, end in selected_spans
            )
            return ContextItem(
                item_id=f"{chunk.chunk_id}:spans:{len(selected_spans)}",
                memory_level="L3",
                category="exact_evidence",
                text=block,
                tokens=tokens,
                pinned=False,
                provenance=pointers,
                score=hit.score,
            )
        if full_tokens <= remaining:
            pointer = ProvenancePointer(
                chunk_id=chunk.chunk_id,
                char_start=0,
                char_end=len(chunk.original_text),
                exact=True,
            )
            return ContextItem(
                item_id=chunk.chunk_id,
                memory_level="L3",
                category="exact_evidence",
                text=full,
                tokens=full_tokens,
                pinned=False,
                provenance=(pointer,),
                score=hit.score,
            )
        return None

    @staticmethod
    def _query_spans(query: str, text: str) -> list[tuple[int, int]]:
        terms = {
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9_.$:-]{3,}", query)
        }
        candidates: set[tuple[int, int]] = set()
        folded = text.casefold()
        for term in sorted(terms, key=len, reverse=True):
            cursor = 0
            while cursor < len(folded):
                found = folded.find(term, cursor)
                if found < 0:
                    break
                candidates.add((max(0, found - 512), min(len(text), found + len(term) + 512)))
                cursor = found + max(1, len(term))
        start = 0
        for match in re.finditer(r"\n\s*\n", text):
            end = match.start()
            if end > start:
                candidates.add((start, end))
            start = match.end()
        if start < len(text):
            candidates.add((start, len(text)))
        # Logs, generated benchmarks, and compact source often use one event or
        # relationship per line without blank paragraph separators.  Preserve
        # the exact line and a bounded neighboring window as replay options.
        for match in re.finditer(r"[^\r\n]+", text):
            line_start, line_end = match.span()
            candidates.add((line_start, line_end))
            candidates.add((max(0, line_start - 512), min(len(text), line_end + 512)))

        ranked = []
        for span_start, span_end in candidates:
            span = text[span_start:span_end]
            span_terms = {
                term.casefold() for term in re.findall(r"[\w.$:-]+", span)
            }
            overlap = len(terms & span_terms)
            if overlap <= 0:
                continue
            length = max(1, span_end - span_start)
            density = overlap / max(1, len(span_terms))
            ranked.append((overlap, density, -length, span_start, span_end))
        ranked.sort(reverse=True)
        return [(span_start, span_end) for _overlap, _density, _length, span_start, span_end in ranked]
