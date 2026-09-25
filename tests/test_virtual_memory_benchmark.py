"""Adversarial effective-context benchmark invariants."""

from __future__ import annotations

from pathlib import Path

from qwen_omni_adapters.virtual_memory.benchmark import (
    BASELINES,
    VirtualContextBenchmarkMatrix,
    benchmark_payload,
    build_adversarial_corpus,
)
from qwen_omni_adapters.virtual_memory.engine import select_relevant_memories
from qwen_omni_adapters.virtual_memory.models import MemoryClass, ProvenancePointer
from qwen_omni_adapters.virtual_memory.packer import conservative_token_estimate
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore


def test_adversarial_corpus_has_exact_requested_size_and_no_answer_in_queries() -> None:
    corpus = build_adversarial_corpus(16_000)

    assert corpus.source_tokens == 16_000
    assert conservative_token_estimate(corpus.source_text) == 16_000
    single = next(item for item in corpus.scenarios if item.name == "single_needle")
    assert "Quartz-41927" not in single.query
    assert "Quartz-41927" in corpus.source_text
    assert {item.family for item in corpus.scenarios} >= {
        "sparse_retrieval",
        "temporal_contradiction",
        "multi_hop",
        "code_dependency",
        "constraint_persistence",
        "adversarial_conflict",
    }


def test_matrix_preserves_oracle_ceiling_and_training_free_production_gate(
    tmp_path: Path,
) -> None:
    result = VirtualContextBenchmarkMatrix().run_length(
        16_000, tmp_path / "matrix.sqlite3"
    )
    payload = benchmark_payload([result])

    assert payload["summary"]["gate_passed"] is True
    assert payload["summary"]["all_reconstruction_exact"] is True
    summaries = payload["summary"]["baseline_summary"]
    assert summaries["oracle"]["pass_rate"] == 1.0
    assert summaries["structured_replay"]["pass_rate"] == 1.0
    assert summaries["structured_replay"]["mean_provenance_accuracy"] == 1.0
    assert summaries["fifo"]["pass_rate"] < summaries["structured_replay"]["pass_rate"]
    assert set(summaries) == set(BASELINES)
    assert all(
        item["resident_input_tokens"] <= result["resident_input_ceiling"]
        for item in result["measurements"]
    )
    conflict = next(
        item
        for item in result["measurements"]
        if item["baseline"] == "structured_replay"
        and item["scenario"] == "near_duplicate_conflict"
    )
    assert conflict["distractor_terms_replayed"] == 0


def test_custom_physical_context_is_reported_not_hard_coded(tmp_path: Path) -> None:
    result = VirtualContextBenchmarkMatrix(physical_context_tokens=12_288).run_length(
        8_000, tmp_path / "matrix-small.sqlite3"
    )
    payload = benchmark_payload([result])

    assert payload["physical_context_tokens"] == 12_288
    assert result["resident_input_ceiling"] == 9_904


def test_derived_memories_and_constraints_are_deterministically_task_scoped(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "memory-scope.sqlite3")
    chunk = store.ingest("authoritative source", source="scope.txt")[0]
    pointer = ProvenancePointer(chunk.chunk_id, 0, len(chunk.original_text))
    relevant = store.write_memory(
        MemoryClass.DECISION,
        "dropbear_left_controller",
        "dropbear_left_controller is DBL-NEW-552",
        provenance=[pointer],
    )
    irrelevant = store.write_memory(
        MemoryClass.DECISION,
        "dropbear_right_controller",
        "dropbear_right_controller is DBR-742",
        provenance=[pointer],
    )
    constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "dropbear_left_controller",
        "MUST preserve the dropbear_left_controller configuration",
        provenance=[pointer],
    )
    irrelevant_constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "camera encoder profile",
        "MUST preserve the camera encoder profile",
        provenance=[pointer],
    )
    global_constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "verified answers",
        "MUST cite verified source evidence",
        provenance=[pointer],
        metadata={"scope": "global"},
    )

    selected = select_relevant_memories(
        store.active_memories(),
        "What is the current dropbear_left_controller?",
    )

    assert relevant in selected
    assert constraint in selected
    assert global_constraint in selected
    assert irrelevant not in selected
    assert irrelevant_constraint not in selected
    store.close()
