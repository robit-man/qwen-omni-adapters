"""Official-RULER adapter invariants without requiring benchmark downloads."""

from __future__ import annotations

import json
from pathlib import Path

from qwen_omni_adapters.virtual_memory.ruler import (
    RulerVirtualContextHarness,
    read_ruler_jsonl,
    ruler_string_match_score,
    run_samples,
    sample_from_record,
    split_ruler_prompt,
)
from scripts.run_virtual_context_ruler import EndpointResponder


def _record() -> dict[str, object]:
    return {
        "index": 7,
        "input": (
            "Some special magic numbers are hidden within the following text.\n"
            "irrelevant alpha. The special magic number for silver-otter is 7319042. "
            "irrelevant omega.\n"
            "What are all the special magic numbers for silver-otter mentioned in "
            "the provided text?"
        ),
        "outputs": ["7319042"],
        "answer_prefix": " The special magic numbers for silver-otter are",
    }


def test_split_ruler_prompt_does_not_need_reference_answer() -> None:
    source, query = split_ruler_prompt(str(_record()["input"]))

    assert "7319042" in source
    assert "7319042" not in query
    assert query.startswith("What are all the special magic")


def test_ruler_jsonl_preserves_official_record_and_reattaches_prefix(tmp_path: Path) -> None:
    path = tmp_path / "niah_single_1.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    sample = read_ruler_jsonl(path)[0]
    prepared = RulerVirtualContextHarness().prepare(sample, baseline="hybrid")

    assert sample.task == "niah_single_1"
    assert sample.answer_prefix in prepared.prompt
    assert "7319042" in prepared.prompt
    assert prepared.resident_tokens < 14_000
    assert prepared.source_tokens > 0


def test_ruler_prefers_generator_reported_length_for_compression_measurement() -> None:
    record = _record()
    record["length"] = 256_000
    sample = sample_from_record(record, task="niah_single_1", ordinal=0)

    prepared = RulerVirtualContextHarness().prepare(sample, baseline="hybrid")

    assert sample.reported_tokens == 256_000
    assert prepared.source_tokens == 256_000
    assert prepared.compression_ratio == 256_000 / prepared.resident_tokens


def test_hybrid_never_exposes_reference_field_to_responder() -> None:
    record = _record()
    record["outputs"] = ["SECRET_REFERENCE_NOT_IN_SOURCE"]
    sample = sample_from_record(record, task="niah_single_1", ordinal=0)
    prompts = []

    results = run_samples(
        [sample],
        harness=RulerVirtualContextHarness(),
        baseline="hybrid",
        responder=lambda prompt: prompts.append(prompt) or "answer",
        allow_insufficient=True,
    )

    assert "SECRET_REFERENCE_NOT_IN_SOURCE" not in prompts[0]
    assert results[0]["pred"] == "answer"
    assert results[0]["outputs"] == ["SECRET_REFERENCE_NOT_IN_SOURCE"]


def test_oracle_uses_reference_only_to_locate_source_chunk() -> None:
    sample = sample_from_record(_record(), task="niah_single_1", ordinal=0)
    prepared = RulerVirtualContextHarness().prepare(sample, baseline="oracle")

    assert prepared.answer_allowed is True
    assert "7319042" in prepared.prompt
    assert prepared.evidence_chunk_ids
    assert prepared.retrieval_queries == (
        "oracle_reference_location",
        "7319042",
    )


def test_fifo_baseline_is_hard_bounded() -> None:
    record = _record()
    record["input"] = (
        str(record["input"]).split("What are all")[0]
        + (" distractor" * 30_000)
        + " What are all the special magic numbers for silver-otter mentioned?"
    )
    sample = sample_from_record(record, task="niah_single_1", ordinal=0)

    prepared = RulerVirtualContextHarness(physical_context_tokens=16_384).prepare(
        sample, baseline="fifo"
    )

    assert prepared.resident_tokens <= 14_000
    assert prepared.compression_ratio > 1.0


def test_frequency_task_uses_reference_free_provenance_bearing_aggregation() -> None:
    record = {
        "index": 8,
        "input": (
            "Read the following coded text and track frequency. "
            "alpha beta alpha gamma alpha beta delta.\n"
            "Question: What are the three most frequently appeared words in "
            "the above coded text?"
        ),
        # Deliberately wrong: hybrid preparation must not use this field.
        "outputs": ["SECRET_WRONG_REFERENCE"],
        "answer_prefix": " Answer:",
    }
    sample = sample_from_record(record, task="fwe", ordinal=0)

    prepared = RulerVirtualContextHarness().prepare(sample, baseline="hybrid")

    assert prepared.sufficient is True
    assert "SECRET_WRONG_REFERENCE" not in prepared.prompt
    assert "term=alpha count=3" in prepared.prompt
    assert "term=beta count=2" in prepared.prompt
    assert 'source_count="1"' in prepared.prompt
    assert any(event["operation"] == "AGGREGATE" for event in prepared.trace)


def test_ruler_string_match_scoring_matches_all_and_qa_part_semantics() -> None:
    assert ruler_string_match_score("vt", "A, C", ("A", "B", "C")) == 66.67
    assert ruler_string_match_score("qa_1", "The answer is France.", ("France",)) == 100.0
    assert ruler_string_match_score("qa_2", "no", ("yes",)) == 0.0


def test_endpoint_runner_explicitly_disables_hidden_thinking_by_default() -> None:
    openai = EndpointResponder(
        endpoint="http://127.0.0.1:8901/v1/chat/completions",
        endpoint_style="openai",
        model="local-audio-bridge",
        api_key=None,
        timeout=1,
        max_tokens=256,
        think=False,
    )
    ollama = EndpointResponder(
        endpoint="http://127.0.0.1:11434/api/chat",
        endpoint_style="ollama",
        model="test-model",
        api_key=None,
        timeout=1,
        max_tokens=256,
        think=False,
    )
    try:
        assert openai._payload("question")["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert ollama._payload("question")["think"] is False
    finally:
        openai.close()
        ollama.close()
