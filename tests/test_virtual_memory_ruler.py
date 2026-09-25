"""Official-RULER adapter invariants without requiring benchmark downloads."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from qwen_omni_adapters.virtual_memory.ruler import (
    RulerVirtualContextHarness,
    read_ruler_jsonl,
    ruler_string_match_score,
    run_samples,
    sample_from_record,
    split_ruler_prompt,
)
from scripts.run_virtual_context_ruler import (
    EndpointResponder,
    _default_tokenize_endpoint,
    _resident_physical_context,
)


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
    assert "do not arbitrarily discard one" in prepared.prompt
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
    assert results[0]["virtual_context"]["inference_seconds"] is not None
    assert results[0]["virtual_context"]["inference"] == {}


def test_oracle_adds_reference_location_to_recursive_support() -> None:
    sample = sample_from_record(_record(), task="niah_single_1", ordinal=0)
    prepared = RulerVirtualContextHarness().prepare(sample, baseline="oracle")

    assert prepared.answer_allowed is True
    assert "7319042" in prepared.prompt
    compilation = next(
        event for event in prepared.trace if event["operation"] == "COMPILE_RELATIONS"
    )
    assert compilation["detail"]["complete"] is True
    assert compilation["detail"]["exact_source_preserved"] is True
    assert compilation["detail"]["source_chunk_ids"]
    assert prepared.retrieval_queries[0] == "oracle_assisted_recursive_retrieval"
    assert sample.query in prepared.retrieval_queries
    assert "oracle_reference_location" in prepared.retrieval_queries
    assert "7319042" in prepared.retrieval_queries


def test_oracle_treats_short_derived_answer_as_conclusion_not_source_locator() -> None:
    record = {
        "index": 9,
        "input": (
            "Document 1:\nScott Derrickson is an American film director.\n\n"
            "Document 2:\nEd Wood was an American filmmaker.\n\n"
            "Question: Were Scott Derrickson and Ed Wood of the same nationality?"
        ),
        "outputs": ["yes"],
        "answer_prefix": " Answer:",
    }
    sample = sample_from_record(record, task="qa_2", ordinal=0)

    prepared = RulerVirtualContextHarness().prepare(sample, baseline="oracle")

    assert prepared.answer_allowed is True
    assert "Scott Derrickson is an American" in prepared.prompt
    assert "Ed Wood was an American" in prepared.prompt
    assert "oracle_reference_location" not in prepared.retrieval_queries
    assert "yes" not in prepared.prompt.casefold()


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


def test_fifo_uses_injected_model_tokenizer_for_its_physical_ceiling() -> None:
    record = _record()
    record["input"] = (
        str(record["input"]).split("What are all")[0]
        + (" punctuation:::heavy" * 4_000)
        + " What are all the special magic numbers for silver-otter mentioned?"
    )
    sample = sample_from_record(record, task="niah_single_1", ordinal=0)

    def exact_counter(text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 1) // 2)

    prepared = RulerVirtualContextHarness(
        physical_context_tokens=4_096,
        token_counter=exact_counter,
    ).prepare(sample, baseline="fifo")

    assert prepared.resident_tokens == exact_counter(prepared.prompt)
    assert prepared.resident_tokens <= 4_096 - 2_384


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
    oracle = RulerVirtualContextHarness().prepare(sample, baseline="oracle")

    assert prepared.sufficient is True
    assert "SECRET_WRONG_REFERENCE" not in prepared.prompt
    assert "term=alpha count=3" in prepared.prompt
    assert "term=beta count=2" in prepared.prompt
    assert 'source_count="1"' in prepared.prompt
    assert any(event["operation"] == "AGGREGATE" for event in prepared.trace)
    assert prepared.evidence_chunk_ids == ()
    assert oracle.sufficient is True
    assert "SECRET_WRONG_REFERENCE" not in oracle.prompt
    assert "oracle_reference_location" not in oracle.retrieval_queries
    assert any(event["operation"] == "AGGREGATE" for event in oracle.trace)


def test_multi_key_values_compile_without_reference_answers() -> None:
    record = {
        "index": 9,
        "input": (
            "Archive text. The number for alpha-key is: 1234567. More text. "
            "The number for beta-key is: 7654321.\n"
            "What are all the numbers for alpha-key and beta-key?"
        ),
        # Preparation must derive only from source evidence.
        "outputs": ["WRONG_REFERENCE"],
        "answer_prefix": " Answer:",
    }
    sample = sample_from_record(record, task="custom_values", ordinal=0)

    prepared = RulerVirtualContextHarness().prepare(sample, baseline="hybrid")

    assert prepared.sufficient is True
    assert "WRONG_REFERENCE" not in prepared.prompt
    assert "key=alpha-key values=[1234567]" in prepared.prompt
    assert "key=beta-key values=[7654321]" in prepared.prompt
    assert prepared.evidence_chunk_ids == ()
    assert any(
        event["operation"] == "COMPILE_RELATIONS"
        and event["detail"]["operator"] == "exact_value_lookup"
        for event in prepared.trace
    )


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
        assert openai._payload("question")["cache_prompt"] is False
        assert ollama._payload("question")["think"] is False
    finally:
        openai.close()
        ollama.close()


def test_endpoint_runner_derives_the_active_llama_tokenizer_route() -> None:
    assert (
        _default_tokenize_endpoint(
            "http://127.0.0.1:8901/v1/chat/completions", "openai"
        )
        == "http://127.0.0.1:8901/tokenize"
    )
    assert _default_tokenize_endpoint("http://127.0.0.1:11434/api/chat", "ollama") is None


def test_endpoint_runner_caps_configured_budget_to_live_worker_state(tmp_path: Path) -> None:
    state = tmp_path / "comprehension-context-tokens"
    state.write_text("8192\n", encoding="utf-8")

    assert _resident_physical_context(16_384, state) == 8192
    assert _resident_physical_context(4096, state) == 4096


def test_endpoint_runner_captures_usage_without_changing_the_prediction() -> None:
    responder = EndpointResponder(
        endpoint="http://llama/v1/chat/completions",
        endpoint_style="openai",
        model="model",
        api_key=None,
        timeout=1,
        max_tokens=32,
        think=False,
    )
    responder.client.close()
    responder.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 19, "completion_tokens": 2},
                },
            )
        )
    )
    try:
        assert responder("question") == "answer"
        assert responder.last_metadata == {
            "usage": {"prompt_tokens": 19, "completion_tokens": 2},
            "finish_reason": "stop",
        }
    finally:
        responder.close()
