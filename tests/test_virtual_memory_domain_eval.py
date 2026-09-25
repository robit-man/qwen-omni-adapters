"""Answer-level topology and durable-state benchmark invariants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qwen_omni_adapters.virtual_memory.benchmark import build_adversarial_corpus
from qwen_omni_adapters.virtual_memory.domain_eval import (
    DomainVirtualContextHarness,
    run_domain_scenarios,
    score_domain_answer,
)


def test_domain_hybrid_replays_every_required_topology_span_at_4k(
    tmp_path: Path,
) -> None:
    corpus = build_adversarial_corpus(16_000)
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "domain.sqlite3",
        physical_context_tokens=4_096,
    ) as harness:
        prepared = {
            scenario.name: harness.prepare(scenario, baseline="hybrid")
            for scenario in corpus.scenarios
        }

    for scenario in corpus.scenarios:
        item = prepared[scenario.name]
        assert item.answer_allowed is True
        assert item.sufficient is True
        assert item.exact_provenance is True
        assert item.resident_tokens <= 4_096 - 2_384
        assert all(term.casefold() in item.prompt.casefold() for term in scenario.required_terms)
    graph = prepared["multi_hop_entity_graph"]
    assert any(
        "graph" in selected.get("channels", ())
        for event in graph.trace
        if event["operation"] == "retrieval_candidates"
        for selected in event["detail"]["selected"]
    )
    code = prepared["cross_file_symbol_trace"]
    assert any(
        "code_graph" in selected.get("channels", ())
        for event in code.trace
        if event["operation"] == "retrieval_candidates"
        for selected in event["detail"]["selected"]
    )
    numerical = prepared["numerical_aggregation_evidence"]
    assert "send_frame must surface" not in numerical.prompt
    assert "golden controller configuration" not in numerical.prompt


def test_seeded_domain_variant_replays_randomized_evidence_at_4k(
    tmp_path: Path,
) -> None:
    corpus = build_adversarial_corpus(16_000, seed=91_307)
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "seeded-domain.sqlite3",
        physical_context_tokens=4_096,
    ) as harness:
        prepared = {
            scenario.name: harness.prepare(scenario, baseline="hybrid")
            for scenario in corpus.scenarios
        }

    for scenario in corpus.scenarios:
        item = prepared[scenario.name]
        assert item.answer_allowed is True
        assert item.sufficient is True
        assert item.exact_provenance is True
        assert item.resident_tokens <= 4_096 - 2_384
        assert all(
            term.casefold() in item.prompt.casefold()
            for term in scenario.required_terms
        )


def test_domain_expected_terms_never_become_production_retrieval_input(
    tmp_path: Path,
) -> None:
    corpus = build_adversarial_corpus(8_000)
    original = corpus.scenarios[0]
    poisoned = replace(original, required_terms=("SECRET_EVALUATOR_ONLY_VALUE",))
    prompts = []
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "no-leak.sqlite3",
        physical_context_tokens=4_096,
    ) as harness:
        records = run_domain_scenarios(
            [poisoned],
            harness=harness,
            baseline="hybrid",
            responder=lambda prompt: prompts.append(prompt) or "answer",
            allow_insufficient=True,
        )

    assert "SECRET_EVALUATOR_ONLY_VALUE" not in prompts[0]
    assert records[0]["expected"]["required_terms"] == [
        "SECRET_EVALUATOR_ONLY_VALUE"
    ]


def test_domain_oracle_is_labelled_and_fifo_is_physically_bounded(
    tmp_path: Path,
) -> None:
    corpus = build_adversarial_corpus(16_000)
    scenario = next(
        item for item in corpus.scenarios if item.name == "cross_file_symbol_trace"
    )
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "baselines.sqlite3",
        physical_context_tokens=4_096,
    ) as harness:
        oracle = harness.prepare(scenario, baseline="oracle")
        fifo = harness.prepare(scenario, baseline="fifo")

    assert oracle.answer_allowed is True
    assert oracle.retrieval_queries[0] == "oracle_source_pages"
    assert any(event["operation"] == "oracle_source_location" for event in oracle.trace)
    assert fifo.resident_tokens <= 4_096 - 2_384
    assert fifo.exact_provenance is False


def test_entity_graph_pages_terminal_fact_in_one_controller_round(
    tmp_path: Path,
) -> None:
    corpus = build_adversarial_corpus(16_000)
    scenario = next(
        item for item in corpus.scenarios if item.name == "multi_hop_entity_graph"
    )
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "with-graph.sqlite3",
        physical_context_tokens=4_096,
        controller_rounds=1,
        retrieval_profile="hybrid",
    ) as harness:
        with_graph = harness.prepare(scenario, baseline="hybrid")
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "without-graph.sqlite3",
        physical_context_tokens=4_096,
        controller_rounds=1,
        retrieval_profile="hybrid-no-graph",
    ) as harness:
        without_graph = harness.prepare(scenario, baseline="hybrid")
    with DomainVirtualContextHarness(
        corpus,
        database=tmp_path / "without-graph-recursive.sqlite3",
        physical_context_tokens=4_096,
        controller_rounds=6,
        retrieval_profile="hybrid-no-graph",
    ) as harness:
        without_graph_recursive = harness.prepare(scenario, baseline="hybrid")

    assert "CAN42_BITRATE_BPS=1000000" in with_graph.prompt
    assert "CAN42_BITRATE_BPS=1000000" not in without_graph.prompt
    assert with_graph.answer_allowed is True
    assert without_graph.answer_allowed is False
    assert "CAN42_BITRATE_BPS=1000000" in without_graph_recursive.prompt
    assert without_graph_recursive.answer_allowed is True


def test_domain_score_requires_all_exact_terms_and_rejects_decoys() -> None:
    corpus = build_adversarial_corpus(8_000)
    scenario = next(
        item for item in corpus.scenarios if item.name == "near_duplicate_conflict"
    )

    correct = score_domain_answer(
        scenario,
        "DBL-NEW-552 superseded DBL-OLD-118.",
    )
    decoy = score_domain_answer(
        scenario,
        "DBL-NEW-552 superseded DBL-OLD-118, like DBL-DECOY-991.",
    )

    assert correct["passed"] is True
    assert decoy["required_recall"] == 1.0
    assert decoy["forbidden_found"] == 1
    assert decoy["passed"] is False


def test_domain_score_ignores_presentation_whitespace_around_assignments() -> None:
    corpus = build_adversarial_corpus(8_000)
    scenario = next(
        item
        for item in corpus.scenarios
        if item.name == "numerical_aggregation_evidence"
    )

    score = score_domain_answer(
        scenario,
        "**NUM_ATLAS_W** = 17, `NUM_BOREAL_W`=23, NUM_CYGNUS_W  =  31; total 71",
    )

    assert score["required_recall"] == 1.0
    assert score["passed"] is True

    table_score = score_domain_answer(
        scenario,
        "| NUM_ATLAS_W | 17 |\n| NUM_BOREAL_W | 23 |\n| NUM_CYGNUS_W | 31 |",
    )
    assert table_score["required_recall"] == 1.0
    assert table_score["passed"] is True

    qualified_score = score_domain_answer(
        replace(scenario, required_terms=("NUM_ATLAS_W=17",)),
        "DriveConfig.NUM_ATLAS_W -> power://atlas = 17",
    )
    assert qualified_score["passed"] is True
