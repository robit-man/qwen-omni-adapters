#!/usr/bin/env python3
"""Prepare or run official RULER JSONL through the virtual-context layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from qwen_omni_adapters.virtual_memory.ruler import (
    VALID_BASELINES,
    RulerVirtualContextHarness,
    read_ruler_jsonl,
    ruler_string_match_score,
    run_samples,
)


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
        response.raise_for_status()
        body = response.json()
        if self.endpoint_style == "openai":
            return str(body["choices"][0]["message"]["content"])
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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
    parser.add_argument("--physical-context", type=int, default=16_384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint")
    parser.add_argument("--endpoint-style", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=256)
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
    if arguments.endpoint and not arguments.model:
        parser.error("--model is required with --endpoint")
    files = _input_files(arguments.inputs)
    if not files:
        parser.error("no JSONL files found")
    responder = None
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
    harness = RulerVirtualContextHarness(
        physical_context_tokens=arguments.physical_context
    )
    summaries = []
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
    report = {
        "schema": "robit.ruler-virtual-context-run.v1",
        "baseline": arguments.baseline,
        "physical_context_tokens": arguments.physical_context,
        "think": arguments.think,
        "scoring": "Run NVIDIA RULER's official evaluate.py over output files.",
        "files": summaries,
        "mean_task_score": (
            sum(float(item["score"]) for item in summaries) / max(1, len(summaries))
            if responder is not None
            else None
        ),
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
