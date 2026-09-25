#!/usr/bin/env python3
"""Prepare or run official RULER JSONL through the virtual-context layer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx

from qwen_omni_adapters.virtual_memory.ruler import (
    RETRIEVAL_PROFILES,
    VALID_BASELINES,
    RulerVirtualContextHarness,
    read_ruler_jsonl,
    ruler_string_match_score,
    run_samples,
)
from qwen_omni_adapters.virtual_memory.tokenization import LlamaCppTokenCounter


class EndpointResponder:
    def __init__(
        self,
        *,
        endpoint: str,
        endpoint_style: str,
        model: str,
        api_key: str | None,
        timeout: float,
        max_tokens: int,
        think: bool,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        self.endpoint = endpoint
        self.endpoint_style = endpoint_style
        self.model = model
        self.max_tokens = max_tokens
        self.think = think
        self.last_metadata: dict[str, Any] = {}

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.endpoint_style == "openai":
            payload.update(
                {
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    # Benchmark baselines must not inherit slot KV state from
                    # the preceding condition. This also preserves the runtime
                    # invariant used for fresh multimodal evidence.
                    "cache_prompt": False,
                    # llama.cpp exposes the Qwen native switch through the
                    # chat-template arguments rather than Ollama's top-level
                    # ``think`` field. Keep answer tokens out of a hidden
                    # reasoning channel unless the benchmark explicitly asks
                    # to measure thinking mode.
                    "chat_template_kwargs": {"enable_thinking": self.think},
                }
            )
        else:
            payload["think"] = self.think
            payload["options"] = {"temperature": 0.0, "num_predict": self.max_tokens}
        return payload

    def __call__(self, prompt: str) -> str:
        payload = self._payload(prompt)
        response = self.client.post(self.endpoint, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise RuntimeError(
                f"inference endpoint returned HTTP {response.status_code}: {detail}"
            ) from exc
        body = response.json()
        if self.endpoint_style == "openai":
            choice = body["choices"][0]
            self.last_metadata = {
                "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
                "finish_reason": choice.get("finish_reason"),
            }
            return str(choice["message"]["content"])
        self.last_metadata = {
            "usage": {
                "prompt_tokens": body.get("prompt_eval_count"),
                "completion_tokens": body.get("eval_count"),
            },
            "finish_reason": body.get("done_reason"),
        }
        return str(body["message"]["content"])

    def close(self) -> None:
        self.client.close()


def _input_files(values: list[Path]) -> list[Path]:
    result = []
    for value in values:
        if value.is_dir():
            result.extend(sorted(value.glob("*.jsonl")))
        elif value.is_file():
            result.append(value)
        else:
            raise FileNotFoundError(value)
    return list(dict.fromkeys(path.resolve() for path in result))


def _default_tokenize_endpoint(endpoint: str | None, endpoint_style: str) -> str | None:
    if not endpoint or endpoint_style != "openai":
        return None
    normalized = endpoint.rstrip("/")
    suffix = "/v1/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)] + "/tokenize"
    return None


def _resident_physical_context(configured: int, state_file: Path) -> int:
    """Resolve a live worker window while retaining the CLI ceiling."""

    try:
        selected = int(state_file.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise ValueError(f"cannot read physical-context state file: {state_file}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid physical-context state file: {state_file}") from exc
    if selected < 4096:
        raise ValueError("resident physical context must be at least 4096 tokens")
    return min(configured, selected)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consume NVIDIA RULER JSONL and emit official-scorer-compatible predictions. "
            "With no --endpoint this performs a preparation-only retrieval run."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", choices=sorted(VALID_BASELINES), default="hybrid")
    parser.add_argument(
        "--retrieval-profile",
        choices=sorted(RETRIEVAL_PROFILES),
        default="hybrid",
        help="retrieval-channel ablation profile",
    )
    parser.add_argument(
        "--controller-rounds",
        type=int,
        default=6,
        help="bounded PRETHINK/RETRIEVE rounds; use 1 for no-recursion ablation",
    )
    parser.add_argument(
        "--disable-aggregation",
        action="store_true",
        help="disable deterministic whole-corpus frequency aggregation",
    )
    parser.add_argument(
        "--disable-compilation",
        action="store_true",
        help="disable deterministic exact-relation compilation",
    )
    parser.add_argument("--physical-context", type=int, default=16_384)
    parser.add_argument(
        "--physical-context-state-file",
        type=Path,
        help="read the live worker KV window and cap --physical-context to it",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint")
    parser.add_argument("--endpoint-style", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--tokenize-endpoint",
        help=(
            "exact tokenizer endpoint; defaults to /tokenize beside an OpenAI-style "
            "/v1/chat/completions endpoint"
        ),
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable the endpoint's native thinking mode (disabled by default)",
    )
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="diagnostic only: ask the model even when evidence sufficiency failed",
    )
    arguments = parser.parse_args()
    if arguments.physical_context < 4096:
        parser.error("--physical-context must be at least 4096")
    if not 1 <= arguments.controller_rounds <= 12:
        parser.error("--controller-rounds must be between 1 and 12")
    configured_physical_context = arguments.physical_context
    if arguments.physical_context_state_file is not None:
        try:
            arguments.physical_context = _resident_physical_context(
                configured_physical_context,
                arguments.physical_context_state_file,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if arguments.endpoint and not arguments.model:
        parser.error("--model is required with --endpoint")
    files = _input_files(arguments.inputs)
    if not files:
        parser.error("no JSONL files found")
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
    tokenize_endpoint = arguments.tokenize_endpoint or _default_tokenize_endpoint(
        arguments.endpoint,
        arguments.endpoint_style,
    )
    if tokenize_endpoint:
        token_counter = LlamaCppTokenCounter(
            tokenize_endpoint,
            timeout=min(30.0, max(1.0, arguments.timeout)),
        )
    harness = RulerVirtualContextHarness(
        physical_context_tokens=arguments.physical_context,
        retrieval_profile=arguments.retrieval_profile,
        controller_rounds=arguments.controller_rounds,
        aggregation_enabled=not arguments.disable_aggregation,
        compilation_enabled=not arguments.disable_compilation,
        **({"token_counter": token_counter} if token_counter is not None else {}),
    )
    summaries = []
    all_inference_seconds: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    try:
        for input_path in files:
            samples = read_ruler_jsonl(input_path, limit=arguments.limit)
            records = run_samples(
                samples,
                harness=harness,
                baseline=arguments.baseline,
                responder=responder,
                allow_insufficient=arguments.allow_insufficient,
            )
            output_path = arguments.output_dir / input_path.name
            if output_path.exists():
                parser.error(f"refusing to overwrite existing output: {output_path}")
            _write_jsonl(output_path, records)
            contexts = [record["virtual_context"] for record in records]
            inference_seconds = [
                float(item["inference_seconds"])
                for item in contexts
                if item.get("inference_seconds") is not None
            ]
            all_inference_seconds.extend(inference_seconds)
            for item in contexts:
                usage = item.get("inference", {}).get("usage", {})
                if isinstance(usage, dict):
                    total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    total_completion_tokens += int(usage.get("completion_tokens") or 0)
            scores = [
                ruler_string_match_score(sample.task, record["pred"], sample.references)
                for sample, record in zip(samples, records, strict=True)
            ]
            summaries.append(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "samples": len(records),
                    "answered": sum(bool(record["pred"]) for record in records),
                    "null_predictions": sum(not bool(record["pred"]) for record in records),
                    "score": (
                        sum(scores) / max(1, len(scores)) if responder is not None else None
                    ),
                    "p50_inference_seconds": _percentile(inference_seconds, 0.5),
                    "p95_inference_seconds": _percentile(inference_seconds, 0.95),
                    "sufficient": sum(bool(item["sufficient"]) for item in contexts),
                    "mean_compression_ratio": (
                        sum(float(item["compression_ratio"]) for item in contexts)
                        / max(1, len(contexts))
                    ),
                }
            )
    finally:
        if responder is not None:
            responder.close()
        if token_counter is not None:
            token_counter.close()
    report = {
        "schema": "robit.ruler-virtual-context-run.v1",
        "baseline": arguments.baseline,
        "retrieval_profile": arguments.retrieval_profile,
        "controller_rounds": arguments.controller_rounds,
        "aggregation_enabled": not arguments.disable_aggregation,
        "compilation_enabled": not arguments.disable_compilation,
        "physical_context_tokens": arguments.physical_context,
        "configured_physical_context_tokens": configured_physical_context,
        "physical_context_source": (
            "resident_state"
            if arguments.physical_context_state_file is not None
            else "configured"
        ),
        "think": arguments.think,
        "tokenizer": "exact_endpoint" if token_counter is not None else "conservative_fallback",
        "scoring": "Run NVIDIA RULER's official evaluate.py over output files.",
        "files": summaries,
        "mean_task_score": (
            sum(float(item["score"]) for item in summaries) / max(1, len(summaries))
            if responder is not None
            else None
        ),
        "p50_inference_seconds": _percentile(all_inference_seconds, 0.5),
        "p95_inference_seconds": _percentile(all_inference_seconds, 0.95),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "virtual-context-run.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
