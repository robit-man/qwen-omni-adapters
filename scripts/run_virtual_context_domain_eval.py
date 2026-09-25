#!/usr/bin/env python3
"""Run topology, constraint, chronology, and conflict model-answer gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
from pathlib import Path

from qwen_omni_adapters.virtual_memory.benchmark import build_adversarial_corpus
from qwen_omni_adapters.virtual_memory.domain_eval import (
    DOMAIN_CONTRACT_VERSION,
    DOMAIN_EVAL_SCHEMA,
    DOMAIN_RETRIEVAL_PROFILES,
    VALID_DOMAIN_BASELINES,
    DomainVirtualContextHarness,
    run_domain_scenarios,
)
from qwen_omni_adapters.virtual_memory.endpoint import (
    EndpointResponder,
    default_tokenize_endpoint,
    percentile,
    resident_physical_context,
)
from qwen_omni_adapters.virtual_memory.tokenization import LlamaCppTokenCounter


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model answers over a lossless adversarial corpus without "
            "placing the complete source in the transformer context."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-length", type=int, default=256_000)
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "build a deterministic held-out variant with randomized facts, "
            "symbols, decoys, and evidence placement"
        ),
    )
    parser.add_argument(
        "--baseline", choices=sorted(VALID_DOMAIN_BASELINES), default="hybrid"
    )
    parser.add_argument(
        "--retrieval-profile",
        choices=sorted(DOMAIN_RETRIEVAL_PROFILES),
        default="hybrid",
    )
    parser.add_argument("--controller-rounds", type=int, default=6)
    parser.add_argument("--disable-compilation", action="store_true")
    parser.add_argument("--physical-context", type=int, default=16_384)
    parser.add_argument("--physical-context-state-file", type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="run only a named scenario; repeat to select more than one",
    )
    parser.add_argument("--endpoint")
    parser.add_argument("--endpoint-style", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokenize-endpoint")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--allow-insufficient", action="store_true")
    arguments = parser.parse_args()

    if arguments.output.exists():
        parser.error(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.source_length < 4_000:
        parser.error("--source-length must be at least 4000")
    if arguments.physical_context < 4_096:
        parser.error("--physical-context must be at least 4096")
    if not 1 <= arguments.controller_rounds <= 12:
        parser.error("--controller-rounds must be between 1 and 12")
    if arguments.endpoint and not arguments.model:
        parser.error("--model is required with --endpoint")

    configured_physical_context = arguments.physical_context
    if arguments.physical_context_state_file is not None:
        try:
            arguments.physical_context = resident_physical_context(
                configured_physical_context,
                arguments.physical_context_state_file,
            )
        except ValueError as exc:
            parser.error(str(exc))

    corpus = build_adversarial_corpus(arguments.source_length, seed=arguments.seed)
    available = {scenario.name: scenario for scenario in corpus.scenarios}
    unknown = sorted(set(arguments.scenario) - set(available))
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")
    scenarios = tuple(
        available[name] for name in arguments.scenario
    ) if arguments.scenario else corpus.scenarios

    responder = None
    token_counter = None
    if arguments.endpoint:
        responder = EndpointResponder(
            endpoint=arguments.endpoint,
            endpoint_style=arguments.endpoint_style,
            model=arguments.model,
            api_key=os.environ.get(arguments.api_key_env),
            timeout=arguments.timeout,
            max_tokens=max(1, arguments.max_tokens),
            think=arguments.think,
        )
    tokenize_endpoint = arguments.tokenize_endpoint or default_tokenize_endpoint(
        arguments.endpoint,
        arguments.endpoint_style,
    )
    if tokenize_endpoint:
        token_counter = LlamaCppTokenCounter(
            tokenize_endpoint,
            timeout=min(30.0, max(1.0, arguments.timeout)),
        )

    records = []
    stats: dict[str, int] = {}
    index_bytes = 0
    try:
        with tempfile.TemporaryDirectory(prefix="omni-domain-eval-") as temp_dir:
            database = Path(temp_dir) / "evidence.sqlite3"
            with DomainVirtualContextHarness(
                corpus,
                database=database,
                physical_context_tokens=arguments.physical_context,
                retrieval_profile=arguments.retrieval_profile,
                controller_rounds=arguments.controller_rounds,
                compilation_enabled=not arguments.disable_compilation,
                **({"token_counter": token_counter} if token_counter is not None else {}),
            ) as harness:
                records = run_domain_scenarios(
                    scenarios,
                    harness=harness,
                    baseline=arguments.baseline,
                    responder=responder,
                    allow_insufficient=arguments.allow_insufficient,
                )
                stats = harness.store.stats()
                index_bytes = database.stat().st_size
    finally:
        if responder is not None:
            responder.close()
        if token_counter is not None:
            token_counter.close()

    inference_seconds = [
        float(record["virtual_context"]["inference_seconds"])
        for record in records
        if record["virtual_context"]["inference_seconds"] is not None
    ]
    prompt_tokens = 0
    completion_tokens = 0
    for record in records:
        usage = record["virtual_context"].get("inference", {}).get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
    executed = responder is not None
    answered = sum(bool(record["prediction"]) for record in records)
    passed = sum(bool(record["score"]["passed"]) for record in records)
    report = {
        "schema": DOMAIN_EVAL_SCHEMA,
        "answer_contract_version": DOMAIN_CONTRACT_VERSION,
        "source": {
            "tokens": corpus.source_tokens,
            "sha256": hashlib.sha256(corpus.source_text.encode("utf-8")).hexdigest(),
            "seed": corpus.seed,
            "index_bytes": index_bytes,
            "store_stats": stats,
        },
        "baseline": arguments.baseline,
        "retrieval_profile": arguments.retrieval_profile,
        "controller_rounds": arguments.controller_rounds,
        "compilation_enabled": not arguments.disable_compilation,
        "physical_context_tokens": arguments.physical_context,
        "configured_physical_context_tokens": configured_physical_context,
        "physical_context_source": (
            "resident_state"
            if arguments.physical_context_state_file is not None
            else "configured"
        ),
        "tokenizer": "exact_endpoint" if token_counter is not None else "conservative_fallback",
        "think": arguments.think,
        "model_executed": executed,
        "summary": {
            "scenarios": len(records),
            "answered": answered,
            "sufficient": sum(
                bool(record["virtual_context"]["sufficient"]) for record in records
            ),
            "exact_provenance": sum(
                bool(record["virtual_context"]["exact_provenance"])
                for record in records
            ),
            "passed": passed if executed else None,
            "pass_rate": passed / max(1, len(records)) if executed else None,
            "p50_inference_seconds": percentile(inference_seconds, 0.5),
            "p95_inference_seconds": percentile(inference_seconds, 0.95),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                3,
            ),
        },
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in report if key != "records"}, indent=2, sort_keys=True))
    return 0 if not executed or passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
