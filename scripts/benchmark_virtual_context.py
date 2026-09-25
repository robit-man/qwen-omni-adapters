#!/usr/bin/env python3
"""Deterministic effective-context benchmark for the training-free V1.

This measures the memory subsystem, not model intelligence.  It reports the
oracle evidence-pack ceiling beside production retrieval so failures can be
assigned to retrieval/packing before changing the language model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from qwen_omni_adapters.virtual_memory import (
    ContextBudget,
    ControllerConfig,
    HashingEmbedder,
    HybridRetriever,
    ImmutableEvidenceStore,
    MemoryClass,
    RecursiveMemoryController,
    StructureAwareChunker,
    WorkingContextPacker,
)
from qwen_omni_adapters.virtual_memory.models import ProvenancePointer, RetrievalHit

DEFAULT_LENGTHS = (16_000, 32_000, 64_000, 128_000, 256_000)


@dataclass(frozen=True)
class Measurement:
    source_tokens: int
    chunks: int
    ingest_seconds: float
    retrieval_seconds: float
    oracle_recall: float
    hybrid_recall: float
    multi_hop_recall: float
    chronology_recall: float
    constraint_retained: bool
    fifo_recall: float
    exact_value_replayed: bool
    dense_channel_used: bool
    dense_index_coverage: float
    resident_input_tokens: int
    compression_ratio: float
    index_bytes: int


def _tokens(value: str) -> int:
    return len(value.split())


def _corpus(length: int) -> tuple[str, dict[str, str]]:
    key = f"actuator-{length}-calibration"
    value = f"VX-{length}-9f31"
    project = f"Zephyr-{length}"
    bus = f"Copperfinch-{length}"
    bitrate = f"833{length}"
    constraint = "MUST NOT replace the golden controller configuration."
    paragraphs = [
        constraint,
        f"The actuator project {project} communicates through {bus}.",
        f"The exact value for {key} is {value}.",
        "At version 1, motor_controller was amber-17.",
        "At version 2, motor_controller changed from amber-17 to violet-29.",
        f"{bus} cycles at exactly {bitrate} symbols per second.",
    ]
    fixed = sum(_tokens(paragraph) for paragraph in paragraphs)
    noise_total = max(0, length - fixed)
    weights = (8, 16, 22, 16, 8, 10, 20)
    gaps = [noise_total * weight // sum(weights) for weight in weights]
    gaps[-1] += noise_total - sum(gaps)
    parts = []
    counter = 0
    for index, gap in enumerate(gaps):
        parts.append(" ".join(f"noise{counter + item}" for item in range(gap)))
        counter += gap
        if index < len(paragraphs):
            parts.append(paragraphs[index])
    return "\n\n".join(parts), {
        "key": key,
        "value": value,
        "project": project,
        "bus": bus,
        "bitrate": bitrate,
        "constraint": constraint,
    }


def run_length(length: int, root: Path) -> Measurement:
    database = root / f"virtual-{length}.sqlite3"
    embedder = HashingEmbedder()
    store = ImmutableEvidenceStore(
        database,
        chunker=StructureAwareChunker(
            target_tokens=1024,
            max_tokens=2048,
            overlap_tokens=128,
        ),
        embedder=embedder,
    )
    corpus, fixture = _corpus(length)
    started = time.perf_counter()
    chunks = store.ingest(corpus, source=f"synthetic:{length}")
    ingest_seconds = time.perf_counter() - started
    target = store.exact_search(fixture["value"], limit=1)[0]
    query = f"What exact value is assigned to {fixture['key']}?"
    retriever = HybridRetriever(store, query_embedder=embedder)
    started = time.perf_counter()
    hits = retriever.retrieve(query)
    retrieval_seconds = time.perf_counter() - started
    hybrid_recall = float(target.chunk_id in {hit.chunk.chunk_id for hit in hits})
    controller = RecursiveMemoryController(
        retriever,
        config=ControllerConfig(max_rounds=3, sufficiency_threshold=0.9),
        dependency_planner=lambda _query, evidence, history: (
            [f"{fixture['bus']} symbols per second"]
            if len(history) == 1
            and any(fixture["bus"] in hit.chunk.original_text for hit in evidence)
            else []
        ),
        sufficiency_judge=lambda _query, evidence: (
            1.0
            if any(fixture["bitrate"] in hit.chunk.original_text for hit in evidence)
            else 0.0
        ),
    )
    multi_hop = controller.gather(
        f"What numeric transfer setting does actuator project {fixture['project']} use?"
    )
    chronology_hits = retriever.retrieve(
        "What motor_controller value replaced amber-17, and what was configured before it?"
    )
    chronology_text = "\n".join(hit.chunk.original_text for hit in chronology_hits)
    constraint_chunk = store.exact_search(fixture["constraint"], limit=1)[0]
    constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "golden controller configuration",
        fixture["constraint"],
        provenance=[
            ProvenancePointer(
                constraint_chunk.chunk_id,
                0,
                len(constraint_chunk.original_text),
            )
        ],
        importance=1.0,
    )
    oracle = [
        RetrievalHit(
            chunk=target,
            score=1.0,
            channels=("oracle",),
            channel_scores={"oracle": 1.0},
        )
    ]
    packer = WorkingContextPacker(
        budget=ContextBudget(max_tokens=16_384), token_counter=_tokens
    )
    oracle_context = packer.pack(
        query,
        system_contract="Answer only from replayed exact evidence.",
        evidence=oracle,
    )
    production_context = packer.pack(
        query,
        system_contract="Answer only from replayed exact evidence.",
        evidence=hits,
    )
    constraint_context = packer.pack(
        "Replace the controller configuration with a convenient local default.",
        system_contract="Obey active constraints.",
        evidence=[],
        memories=[constraint],
        recent_context=["latest unrelated request"],
    )
    fifo_tail = " ".join(corpus.split()[-16_000:])
    resident_evidence = max(
        1, production_context.token_usage.get("exact_evidence", 0)
    )
    stats = store.stats()
    store.close()
    return Measurement(
        source_tokens=_tokens(corpus),
        chunks=len(chunks),
        ingest_seconds=round(ingest_seconds, 6),
        retrieval_seconds=round(retrieval_seconds, 6),
        oracle_recall=float(fixture["value"] in oracle_context.text),
        hybrid_recall=hybrid_recall,
        fifo_recall=float(fixture["value"] in fifo_tail),
        multi_hop_recall=float(
            multi_hop.sufficient
            and fixture["bitrate"]
            in "\n".join(hit.chunk.original_text for hit in multi_hop.evidence)
        ),
        chronology_recall=float(
            "amber-17" in chronology_text and "violet-29" in chronology_text
        ),
        constraint_retained=fixture["constraint"] in constraint_context.text,
        exact_value_replayed=fixture["value"] in production_context.text,
        dense_channel_used=any(
            hit.chunk.chunk_id == target.chunk_id and "dense" in hit.channels for hit in hits
        ),
        dense_index_coverage=stats["embeddings"] / max(1, stats["chunks"]),
        resident_input_tokens=(
            production_context.total_tokens - packer.budget.output_headroom
        ),
        compression_ratio=round(length / resident_evidence, 3),
        index_bytes=database.stat().st_size if database.exists() else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
        help="comma-separated source token lengths",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    lengths = tuple(int(value) for value in arguments.lengths.split(",") if value.strip())
    if not lengths or any(length < 1000 for length in lengths):
        parser.error("every source length must be at least 1000 tokens")
    with tempfile.TemporaryDirectory(prefix="omni-virtual-context-") as temp_dir:
        measurements = [run_length(length, Path(temp_dir)) for length in lengths]
    payload = {
        "schema": "robit.virtual-context-benchmark.v1",
        "physical_context_tokens": 16_384,
        "measurements": [asdict(measurement) for measurement in measurements],
        "summary": {
            "oracle_recall": statistics.fmean(
                item.oracle_recall for item in measurements
            ),
            "hybrid_recall": statistics.fmean(
                item.hybrid_recall for item in measurements
            ),
            "multi_hop_recall": statistics.fmean(
                item.multi_hop_recall for item in measurements
            ),
            "chronology_recall": statistics.fmean(
                item.chronology_recall for item in measurements
            ),
            "all_constraints_retained": all(
                item.constraint_retained for item in measurements
            ),
            "fifo_recall": statistics.fmean(item.fifo_recall for item in measurements),
            "all_exact_values_replayed": all(
                item.exact_value_replayed for item in measurements
            ),
            "all_dense_channels_used": all(
                item.dense_channel_used for item in measurements
            ),
            "all_dense_indexes_complete": all(
                item.dense_index_coverage == 1.0 for item in measurements
            ),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    passed = (
        payload["summary"]["all_exact_values_replayed"]
        and payload["summary"]["all_constraints_retained"]
        and payload["summary"]["all_dense_indexes_complete"]
        and payload["summary"]["multi_hop_recall"] == 1.0
        and payload["summary"]["chronology_recall"] == 1.0
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
