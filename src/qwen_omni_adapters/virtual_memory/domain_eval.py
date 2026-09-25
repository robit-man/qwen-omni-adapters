"""Answer-level adversarial evaluation for lossless virtual context.

RULER measures general long-context behavior, but it does not exercise this
runtime's code topology, pinned constraints, supersession records, or explicit
entity graph.  This module turns the deterministic adversarial corpus into a
model-answer suite while preserving the same evaluation boundary:

* production retrieval sees only the current query;
* expected terms are used only after inference for scoring;
* the labelled oracle may locate source chunks with fixture annotations;
* every prompt remains bounded by the physical context budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_omni_adapters.virtual_memory.benchmark import (
    BenchmarkScenario,
    PreparedCorpus,
)
from qwen_omni_adapters.virtual_memory.compilation import QueryEvidenceCompiler
from qwen_omni_adapters.virtual_memory.controller import (
    ControllerConfig,
    RecursiveMemoryController,
)
from qwen_omni_adapters.virtual_memory.embedding import HashingEmbedder
from qwen_omni_adapters.virtual_memory.engine import select_relevant_memories
from qwen_omni_adapters.virtual_memory.extractor import StructuredMemoryExtractor
from qwen_omni_adapters.virtual_memory.models import RetrievalHit, WorkingContext
from qwen_omni_adapters.virtual_memory.packer import (
    ContextBudget,
    WorkingContextPacker,
    conservative_token_estimate,
)
from qwen_omni_adapters.virtual_memory.retrieval import (
    RETRIEVAL_CHANNELS,
    HybridRetriever,
)
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import TraceCollector

DOMAIN_EVAL_SCHEMA = "robit.virtual-context-domain-eval.v1"
VALID_DOMAIN_BASELINES = frozenset({"fifo", "hybrid", "oracle"})
DOMAIN_RETRIEVAL_PROFILES = {
    "bm25-only": frozenset({"bm25"}),
    "dense-only": frozenset({"dense"}),
    "hybrid": RETRIEVAL_CHANNELS,
    "hybrid-no-graph": RETRIEVAL_CHANNELS - {"code_graph", "graph"},
}

_DOMAIN_CONTRACT = (
    "Complete the current task using only the supplied source evidence. "
    "For factual questions, give the exact values and the shortest dependency "
    "chain needed to support them. Preserve source spelling and chronology. "
    "If an instruction conflicts with a governing constraint, do not perform "
    "it; state the exact governing constraint. Keep distinct conflicting or "
    "near-duplicate entities separate. Do not invent missing facts, mention "
    "the memory system, or produce a generic assistant preamble."
)


@dataclass(frozen=True)
class PreparedDomainScenario:
    prompt: str
    answer_allowed: bool
    source_tokens: int
    resident_tokens: int
    compression_ratio: float
    evidence_chunk_ids: tuple[str, ...]
    retrieval_queries: tuple[str, ...]
    sufficient: bool
    preparation_seconds: float
    trace: tuple[dict[str, Any], ...]
    retrieval_profile: str
    controller_rounds: int
    compilation_enabled: bool
    exact_provenance: bool


def score_domain_answer(
    scenario: BenchmarkScenario,
    prediction: str,
) -> dict[str, Any]:
    """Score source-exact task content without influencing preparation."""

    folded = str(prediction or "").casefold()
    required_matches = {
        term: term.casefold() in folded for term in scenario.required_terms
    }
    forbidden_matches = {
        term: term.casefold() in folded for term in scenario.forbidden_terms
    }
    required_found = sum(required_matches.values())
    required_total = len(required_matches)
    forbidden_found = sum(forbidden_matches.values())
    return {
        "required_found": required_found,
        "required_total": required_total,
        "required_recall": required_found / max(1, required_total),
        "forbidden_found": forbidden_found,
        "passed": required_found == required_total and forbidden_found == 0,
        "required_matches": required_matches,
        "forbidden_matches": forbidden_matches,
    }


def _dedupe_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    selected: dict[str, RetrievalHit] = {}
    for hit in hits:
        previous = selected.get(hit.chunk.chunk_id)
        if previous is None or hit.score > previous.score:
            selected[hit.chunk.chunk_id] = hit
    return list(selected.values())


def _oracle_hits(
    store: ImmutableEvidenceStore,
    terms: Sequence[str],
) -> list[RetrievalHit]:
    """Locate labelled source pages; the model still receives raw evidence."""

    return _dedupe_hits(
        RetrievalHit(
            chunk=chunk,
            score=1.0,
            channels=("oracle",),
            channel_scores={"oracle": 1.0},
        )
        for term in terms
        for chunk in store.exact_search(term, limit=200)
        if term.casefold() in chunk.original_text.casefold()
    )


def _tail_within_tokens(
    value: str,
    maximum: int,
    token_counter: Callable[[str], int],
) -> str:
    if maximum <= 0:
        return ""
    if token_counter(value) <= maximum:
        return value
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high) // 2
        if token_counter(value[middle:]) <= maximum:
            high = middle
        else:
            low = middle + 1
    return value[low:]


class DomainVirtualContextHarness:
    """Ingest one lossless corpus and prepare bounded domain task prompts."""

    def __init__(
        self,
        corpus: PreparedCorpus,
        *,
        database: Path,
        physical_context_tokens: int = 16_384,
        token_counter: Callable[[str], int] = conservative_token_estimate,
        retrieval_profile: str = "hybrid",
        controller_rounds: int = 6,
        compilation_enabled: bool = True,
    ) -> None:
        if retrieval_profile not in DOMAIN_RETRIEVAL_PROFILES:
            raise ValueError(f"unknown retrieval profile: {retrieval_profile}")
        if not 1 <= int(controller_rounds) <= 12:
            raise ValueError("controller_rounds must be between 1 and 12")
        self.corpus = corpus
        self.token_counter = token_counter
        self.retrieval_profile = retrieval_profile
        self.controller_rounds = int(controller_rounds)
        self.compilation_enabled = bool(compilation_enabled)
        self.budget = ContextBudget(max_tokens=physical_context_tokens)
        self.packer = WorkingContextPacker(
            budget=self.budget,
            token_counter=token_counter,
        )
        self.embedder = HashingEmbedder()
        self.store = ImmutableEvidenceStore(database, embedder=self.embedder)
        self._closed = False
        self._ingest()
        self.retriever = HybridRetriever(
            self.store,
            query_embedder=self.embedder,
            enabled_channels=DOMAIN_RETRIEVAL_PROFILES[retrieval_profile],
        )

    def _ingest(self) -> None:
        extractor = StructuredMemoryExtractor(self.store)
        for ordinal, document in enumerate(self.corpus.documents):
            chunks = self.store.ingest(
                document.text,
                source=document.source,
                document_id=f"domain-{self.corpus.source_tokens}-{ordinal}",
                captured_at=1_700_000_000.0 + ordinal,
                media_type=document.media_type,
                kind=document.kind,
                metadata=document.metadata,
                entities=document.entities,
            )
            extractor.extract(chunks, authority="domain_benchmark_source")
            for subject, predicate, object_ in document.relationships:
                self.store.add_relationship(
                    subject,
                    predicate,
                    object_,
                    chunk_id=chunks[0].chunk_id,
                    valid_from=1_700_000_000.0 + ordinal,
                )

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    def __enter__(self) -> DomainVirtualContextHarness:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def prepare(
        self,
        scenario: BenchmarkScenario,
        *,
        baseline: str = "hybrid",
    ) -> PreparedDomainScenario:
        if self._closed:
            raise RuntimeError("domain harness is closed")
        selected = str(baseline).lower()
        if selected not in VALID_DOMAIN_BASELINES:
            raise ValueError(f"unknown domain baseline: {baseline}")
        started = time.perf_counter()
        if selected == "fifo":
            prompt = self._fifo_prompt(scenario.query)
            resident = self.token_counter(prompt)
            return PreparedDomainScenario(
                prompt=prompt,
                answer_allowed=True,
                source_tokens=self.corpus.source_tokens,
                resident_tokens=resident,
                compression_ratio=self.corpus.source_tokens / max(1, resident),
                evidence_chunk_ids=(),
                retrieval_queries=(scenario.query,),
                sufficient=True,
                preparation_seconds=time.perf_counter() - started,
                trace=(),
                retrieval_profile=self.retrieval_profile,
                controller_rounds=self.controller_rounds,
                compilation_enabled=self.compilation_enabled,
                exact_provenance=False,
            )

        trace = TraceCollector()
        compilation = None
        if selected == "oracle":
            hits = _oracle_hits(self.store, scenario.oracle_terms)
            retrieval_queries = ("oracle_source_pages", *scenario.oracle_terms)
            sufficient = bool(hits)
            trace.record(
                "oracle_source_location",
                terms=list(scenario.oracle_terms),
                chunk_ids=[hit.chunk.chunk_id for hit in hits],
            )
        else:
            result = RecursiveMemoryController(
                self.retriever,
                config=ControllerConfig(max_rounds=self.controller_rounds),
            ).gather(scenario.query, trace=trace)
            hits = list(result.evidence)
            retrieval_queries = result.queries
            sufficient = result.sufficient
            if self.compilation_enabled:
                compilation = QueryEvidenceCompiler(self.store).compile(
                    scenario.query,
                    hits,
                    trace=trace,
                )
                sufficient = sufficient or compilation.complete
                if compilation.complete:
                    retrieval_queries = (
                        *retrieval_queries,
                        *(f"compiled:{item}" for item in compilation.operators),
                    )

        memories = select_relevant_memories(
            self.store.active_memories(),
            scenario.query,
            active_subjects=(
                (scenario.active_memory_subject,)
                if scenario.active_memory_subject is not None
                else ()
            ),
        )
        if compilation is not None:
            memories.extend(compilation.memories)
        evidence = (
            []
            if compilation is not None and compilation.consume_evidence
            else hits
        )
        context = self.packer.pack(
            scenario.query,
            system_contract=_DOMAIN_CONTRACT,
            evidence=evidence,
            retrieval_queries=retrieval_queries,
            memories=memories,
            trace=trace,
        )
        resident = context.total_tokens - self.budget.output_headroom
        exact_provenance = self._exact_provenance(context)
        return PreparedDomainScenario(
            prompt=context.text,
            answer_allowed=sufficient,
            source_tokens=self.corpus.source_tokens,
            resident_tokens=resident,
            compression_ratio=self.corpus.source_tokens / max(1, resident),
            evidence_chunk_ids=context.evidence_chunk_ids,
            retrieval_queries=tuple(retrieval_queries),
            sufficient=sufficient,
            preparation_seconds=time.perf_counter() - started,
            trace=context.trace,
            retrieval_profile=self.retrieval_profile,
            controller_rounds=self.controller_rounds,
            compilation_enabled=self.compilation_enabled,
            exact_provenance=exact_provenance,
        )

    def _fifo_prompt(self, query: str) -> str:
        def render(tail: str) -> str:
            return (
                f"{_DOMAIN_CONTRACT}\n\n<unverified_fifo_tail>\n{tail}"
                f"\n</unverified_fifo_tail>\n\n<current_query>\n{query}"
                "\n</current_query>"
            )

        fixed = self.token_counter(render(""))
        tail = _tail_within_tokens(
            self.corpus.source_text,
            self.budget.input_ceiling - fixed,
            self.token_counter,
        )
        prompt = render(tail)
        while self.token_counter(prompt) > self.budget.input_ceiling:
            excess = self.token_counter(prompt) - self.budget.input_ceiling
            tail = tail[min(len(tail), max(1, excess * 3)) :]
            prompt = render(tail)
        return prompt

    def _exact_provenance(self, context: WorkingContext) -> bool:
        pointers = [pointer for item in context.items for pointer in item.provenance]
        if not pointers:
            return False
        for pointer in pointers:
            chunk = self.store.get_chunk(pointer.chunk_id)
            if (
                chunk is None
                or not pointer.exact
                or pointer.char_start < 0
                or pointer.char_end < pointer.char_start
                or pointer.char_end > len(chunk.original_text)
            ):
                return False
        return True


def domain_prediction_record(
    scenario: BenchmarkScenario,
    prepared: PreparedDomainScenario,
    *,
    prediction: str,
    baseline: str,
    inference_seconds: float | None = None,
    inference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an auditable answer record after inference has completed."""

    return {
        "scenario": scenario.name,
        "family": scenario.family,
        "query": scenario.query,
        "prediction": prediction,
        "score": score_domain_answer(scenario, prediction),
        "expected": {
            "required_terms": list(scenario.required_terms),
            "forbidden_terms": list(scenario.forbidden_terms),
        },
        "virtual_context": {
            "schema": DOMAIN_EVAL_SCHEMA,
            "baseline": baseline,
            "inference_seconds": inference_seconds,
            "inference": dict(inference or {}),
            **{
                key: value
                for key, value in asdict(prepared).items()
                if key not in {"prompt", "trace"}
            },
            "trace": list(prepared.trace),
        },
    }


def run_domain_scenarios(
    scenarios: Iterable[BenchmarkScenario],
    *,
    harness: DomainVirtualContextHarness,
    baseline: str,
    responder: Callable[[str], str] | None,
    allow_insufficient: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for scenario in scenarios:
        prepared = harness.prepare(scenario, baseline=baseline)
        prediction = ""
        inference_seconds = None
        inference: Mapping[str, Any] | None = None
        if responder is not None and (prepared.answer_allowed or allow_insufficient):
            inference_started = time.perf_counter()
            prediction = str(responder(prepared.prompt)).strip()
            inference_seconds = time.perf_counter() - inference_started
            metadata = getattr(responder, "last_metadata", None)
            if isinstance(metadata, Mapping):
                inference = metadata
        records.append(
            domain_prediction_record(
                scenario,
                prepared,
                prediction=prediction,
                baseline=baseline,
                inference_seconds=inference_seconds,
                inference=inference,
            )
        )
    return records
