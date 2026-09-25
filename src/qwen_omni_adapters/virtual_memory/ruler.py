"""RULER v1 adapter for bounded-context retrieval and model evaluation.

The adapter consumes RULER-generated JSONL without vendoring or rewriting the
benchmark. It emits the original records plus ``pred`` so NVIDIA's official
scorer remains the scoring authority. Reference outputs are visible only to
the explicitly labelled oracle baseline; production hybrid retrieval never
ingests them.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_omni_adapters.virtual_memory.aggregation import FrequencyAggregationBuilder
from qwen_omni_adapters.virtual_memory.compilation import QueryEvidenceCompiler
from qwen_omni_adapters.virtual_memory.controller import (
    ControllerConfig,
    RecursiveMemoryController,
)
from qwen_omni_adapters.virtual_memory.embedding import HashingEmbedder
from qwen_omni_adapters.virtual_memory.models import RetrievalHit, WorkingContext
from qwen_omni_adapters.virtual_memory.packer import (
    ContextBudget,
    WorkingContextPacker,
    conservative_token_estimate,
)
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore

RULER_V1_REVISION = "e8bbff677ca2c239640dc90f93310dcf32408c93"
VALID_BASELINES = {"fifo", "hybrid", "oracle"}
_QUESTION_MARKERS = (
    "Question:",
    "What are all the special magic",
)


def ruler_string_match_score(
    task: str, prediction: str, references: Sequence[str]
) -> float:
    """Reproduce RULER v1's published all/part string-match metric."""

    normalized = str(prediction or "").casefold()
    expected = [str(reference).casefold() for reference in references if str(reference)]
    if not expected:
        return 0.0
    if str(task).startswith("qa_"):
        return 100.0 if any(reference in normalized for reference in expected) else 0.0
    return round(
        sum(reference in normalized for reference in expected) / len(expected) * 100,
        2,
    )


@dataclass(frozen=True)
class RulerSample:
    task: str
    sample_id: str
    source_text: str
    query: str
    answer_prefix: str
    references: tuple[str, ...]
    record: dict[str, Any]
    reported_tokens: int | None = None


@dataclass(frozen=True)
class PreparedRulerSample:
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


def _flatten_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for nested in value for item in _flatten_strings(nested))
    return ()


def split_ruler_prompt(value: str) -> tuple[str, str]:
    """Separate the immutable corpus from the final benchmark question."""

    prompt = str(value or "").strip()
    if not prompt:
        raise ValueError("RULER input must be non-empty")
    split_at = max(prompt.rfind(marker) for marker in _QUESTION_MARKERS)
    if split_at < 0:
        # Unknown custom tasks stay valid, but only the bounded tail can be
        # treated as the query. Never inspect reference answers to find it.
        newline = prompt.rfind("\n")
        split_at = newline if newline >= 0 else max(0, len(prompt) - 512)
    source = prompt[:split_at].rstrip()
    query = prompt[split_at:].strip()
    if not source or not query:
        raise ValueError("could not separate RULER source context and query")
    return source, query


def sample_from_record(record: Mapping[str, Any], *, task: str, ordinal: int) -> RulerSample:
    source, query = split_ruler_prompt(str(record.get("input") or ""))
    references = _flatten_strings(record.get("outputs", record.get("output", ())))
    others = record.get("others")
    other_id = others.get("id", "") if isinstance(others, Mapping) else ""
    sample_id = str(record.get("index") or other_id or ordinal)
    reported_tokens = None
    for field in ("length_w_model_temp", "length"):
        try:
            candidate = int(record.get(field, 0))
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            reported_tokens = candidate
            break
    return RulerSample(
        task=task,
        sample_id=sample_id,
        source_text=source,
        query=query,
        answer_prefix=str(record.get("answer_prefix") or ""),
        references=references,
        record=dict(record),
        reported_tokens=reported_tokens,
    )


def read_ruler_jsonl(path: Path, *, limit: int | None = None) -> list[RulerSample]:
    samples = []
    task = path.stem
    with path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if limit is not None and len(samples) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{ordinal + 1}: expected a JSON object")
            samples.append(sample_from_record(record, task=task, ordinal=ordinal))
    return samples


def _reference_terms(references: Sequence[str]) -> list[str]:
    terms = []
    trim = " \t\r\n.,:;[](){}'\""
    for reference in references:
        normalized = str(reference).strip()
        if normalized:
            terms.append(normalized)
        terms.extend(
            part.strip(trim)
            for part in re.split(r"[,;\n]", normalized)
            if len(part.strip(trim)) >= 2
        )
    return list(dict.fromkeys(terms))


def _locatable_reference_terms(references: Sequence[str]) -> list[str]:
    """Return answer strings specific enough to locate source evidence.

    Short class labels such as ``yes`` and ``no`` are conclusions, not useful
    source locators. Searching an entire corpus for them creates a misleading
    oracle pack from unrelated prose.
    """

    return [
        term
        for term in _reference_terms(references)
        if len(re.sub(r"[^A-Za-z0-9]", "", term)) >= 4
    ]


def _oracle_hits(
    store: ImmutableEvidenceStore, references: Sequence[str]
) -> list[RetrievalHit]:
    chunks = {}
    for term in _locatable_reference_terms(references):
        for chunk in store.exact_search(term, limit=200):
            chunks[chunk.chunk_id] = chunk
    return [
        RetrievalHit(
            chunk=chunk,
            score=1.0,
            channels=("oracle_reference_location",),
            channel_scores={"oracle_reference_location": 1.0},
        )
        for chunk in chunks.values()
    ]


def _merge_hits(*groups: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """Merge support and oracle hits without duplicating immutable chunks."""

    merged: dict[str, RetrievalHit] = {}
    for group in groups:
        for hit in group:
            previous = merged.get(hit.chunk.chunk_id)
            if previous is None:
                merged[hit.chunk.chunk_id] = hit
                continue
            distances = tuple(
                distance
                for distance in (previous.graph_distance, hit.graph_distance)
                if distance is not None
            )
            merged[hit.chunk.chunk_id] = RetrievalHit(
                chunk=previous.chunk,
                score=max(previous.score, hit.score),
                channels=tuple(dict.fromkeys((*previous.channels, *hit.channels))),
                channel_scores={**previous.channel_scores, **hit.channel_scores},
                graph_distance=min(distances) if distances else None,
            )
    return list(merged.values())


def _tail_within_tokens(
    value: str,
    maximum: int,
    token_counter: Callable[[str], int] = conservative_token_estimate,
) -> str:
    if maximum <= 0:
        return ""
    if token_counter(value) <= maximum:
        return value
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high) // 2
        candidate = value[middle:]
        if token_counter(candidate) <= maximum:
            high = middle
        else:
            low = middle + 1
    return value[low:]


class RulerVirtualContextHarness:
    """Build isolated working sets for official RULER samples."""

    def __init__(
        self,
        *,
        physical_context_tokens: int = 16_384,
        token_counter: Callable[[str], int] = conservative_token_estimate,
    ) -> None:
        self.budget = ContextBudget(max_tokens=physical_context_tokens)
        self.token_counter = token_counter
        self.packer = WorkingContextPacker(
            budget=self.budget,
            token_counter=token_counter,
        )

    def prepare(self, sample: RulerSample, *, baseline: str = "hybrid") -> PreparedRulerSample:
        selected = str(baseline).lower()
        if selected not in VALID_BASELINES:
            raise ValueError(f"unknown RULER baseline: {baseline}")
        started = time.perf_counter()
        question = sample.query + sample.answer_prefix
        contract = (
            "Answer the current query using only replayed source evidence. "
            "Do not infer missing facts. Return only the shortest source-exact "
            "answer span, preserving source spelling, punctuation, and ordering."
        )
        source_tokens = sample.reported_tokens or self.token_counter(sample.source_text)
        if selected == "fifo":
            def render(candidate: str) -> str:
                return (
                    f"{contract}\n\n<unverified_fifo_tail>\n{candidate}"
                    f"\n</unverified_fifo_tail>\n\n<current_query>\n{question}"
                    "\n</current_query>"
                )

            fixed = self.token_counter(render(""))
            tail = _tail_within_tokens(
                sample.source_text,
                self.budget.input_ceiling - fixed,
                self.token_counter,
            )
            prompt = render(tail)
            while self.token_counter(prompt) > self.budget.input_ceiling:
                excess = self.token_counter(prompt) - self.budget.input_ceiling
                tail = tail[min(len(tail), max(1, excess * 3)) :]
                prompt = render(tail)
            resident = self.token_counter(prompt)
            return PreparedRulerSample(
                prompt=prompt,
                answer_allowed=True,
                source_tokens=source_tokens,
                resident_tokens=resident,
                compression_ratio=source_tokens / max(1, resident),
                evidence_chunk_ids=(),
                retrieval_queries=(sample.query,),
                sufficient=True,
                preparation_seconds=time.perf_counter() - started,
                trace=(),
            )
        with tempfile.TemporaryDirectory(prefix="omni-ruler-") as temp_dir:
            embedder = HashingEmbedder()
            store = ImmutableEvidenceStore(
                Path(temp_dir) / "evidence.sqlite3", embedder=embedder
            )
            try:
                chunks = store.ingest(
                    sample.source_text,
                    source=f"ruler:{sample.task}:{sample.sample_id}",
                    document_id=f"ruler-{sample.task}-{sample.sample_id}",
                    kind="document",
                )
                retriever = HybridRetriever(store, query_embedder=embedder)
                aggregation = FrequencyAggregationBuilder(store).build(
                    sample.query,
                    sample.source_text,
                    chunks,
                )
                memories = [aggregation.memory] if aggregation is not None else []
                controller = RecursiveMemoryController(
                    retriever,
                    config=ControllerConfig(max_rounds=6),
                )
                if selected == "oracle":
                    # Literal answer location is not an evidence oracle for a
                    # derived relation (for example yes/no QA or a variable
                    # chain). Recursive query support supplies dependencies;
                    # high-information reference strings only add exact source
                    # locations. Answers are never written into the prompt.
                    support = controller.gather(sample.query)
                    # The deterministic aggregation is already a verified,
                    # provenance-bearing view over the whole corpus. Expanding
                    # every literal occurrence of its frequent output terms is
                    # redundant and can turn a bounded oracle into hundreds of
                    # equivalent chunks.
                    oracle_hits = (
                        []
                        if aggregation is not None
                        else _oracle_hits(store, sample.references)
                    )
                    hits = _merge_hits(support.evidence, oracle_hits)
                    sufficient = (
                        support.sufficient
                        or bool(oracle_hits)
                        or aggregation is not None
                    )
                    locators = _locatable_reference_terms(sample.references)
                    retrieval_queries = (
                        "oracle_assisted_recursive_retrieval",
                        *support.queries,
                        *(("oracle_reference_location", *locators) if oracle_hits else ()),
                    )
                    trace: tuple[dict[str, Any], ...] = (
                        *support.trace,
                        *(aggregation.trace if aggregation is not None else ()),
                    )
                else:
                    result = controller.gather(sample.query)
                    hits = list(result.evidence)
                    sufficient = result.sufficient or aggregation is not None
                    retrieval_queries = (
                        (*result.queries, "deterministic word-frequency aggregation")
                        if aggregation is not None
                        else result.queries
                    )
                    trace = (
                        (*result.trace, *aggregation.trace)
                        if aggregation is not None
                        else result.trace
                    )
                compilation = (
                    QueryEvidenceCompiler(store).compile(sample.query, hits)
                    if aggregation is None
                    else None
                )
                if compilation is not None and compilation.complete:
                    memories.extend(compilation.memories)
                    sufficient = True
                    retrieval_queries = (
                        *retrieval_queries,
                        *(f"compiled:{operator}" for operator in compilation.operators),
                    )
                    trace = (*trace, *compilation.trace)
                context: WorkingContext = self.packer.pack(
                    question,
                    system_contract=contract,
                    # A deterministic whole-corpus aggregation is the exact
                    # query-specific view.  Replaying a few arbitrary local
                    # source pages beside it is incomplete and can falsely
                    # overrule the verified global count.  All raw chunks stay
                    # immutable and EXPAND-able through the memory provenance.
                    evidence=(
                        []
                        if aggregation is not None
                        or (compilation is not None and compilation.consume_evidence)
                        else hits
                    ),
                    retrieval_queries=retrieval_queries,
                    memories=memories,
                )
            finally:
                store.close()
        resident = context.total_tokens - self.budget.output_headroom
        return PreparedRulerSample(
            prompt=context.text,
            answer_allowed=sufficient,
            source_tokens=source_tokens,
            resident_tokens=resident,
            compression_ratio=source_tokens / max(1, resident),
            evidence_chunk_ids=context.evidence_chunk_ids,
            retrieval_queries=tuple(retrieval_queries),
            sufficient=sufficient,
            preparation_seconds=time.perf_counter() - started,
            trace=trace,
        )


def prediction_record(
    sample: RulerSample,
    prepared: PreparedRulerSample,
    *,
    prediction: str,
    baseline: str,
    inference_seconds: float | None = None,
    inference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve official fields and attach namespaced diagnostic metadata."""

    record = dict(sample.record)
    record["pred"] = prediction
    record["virtual_context"] = {
        "schema": "robit.ruler-virtual-context.v1",
        "ruler_v1_revision": RULER_V1_REVISION,
        "baseline": baseline,
        "answer_prefix_reattached": bool(sample.answer_prefix),
        "inference_seconds": inference_seconds,
        "inference": dict(inference or {}),
        **{
            key: value
            for key, value in asdict(prepared).items()
            if key not in {"prompt", "trace"}
        },
        "trace": list(prepared.trace),
    }
    return record


def run_samples(
    samples: Iterable[RulerSample],
    *,
    harness: RulerVirtualContextHarness,
    baseline: str,
    responder: Callable[[str], str] | None,
    allow_insufficient: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        prepared = harness.prepare(sample, baseline=baseline)
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
            prediction_record(
                sample,
                prepared,
                prediction=prediction,
                baseline=baseline,
                inference_seconds=inference_seconds,
                inference=inference,
            )
        )
    return records
