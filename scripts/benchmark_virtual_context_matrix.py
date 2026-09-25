#!/usr/bin/env python3
"""Run the adversarial effective-context benchmark matrix."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from qwen_omni_adapters.virtual_memory.benchmark import (
    DEFAULT_SOURCE_LENGTHS,
    VirtualContextBenchmarkMatrix,
    benchmark_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default=",".join(str(value) for value in DEFAULT_SOURCE_LENGTHS),
        help="comma-separated source token lengths",
    )
    parser.add_argument("--physical-context", type=int, default=16_384)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    lengths = tuple(
        int(value.strip()) for value in arguments.lengths.split(",") if value.strip()
    )
    if not lengths or any(value < 4_000 for value in lengths):
        parser.error("every source length must be at least 4000")
    harness = VirtualContextBenchmarkMatrix(
        physical_context_tokens=arguments.physical_context
    )
    with tempfile.TemporaryDirectory(prefix="omni-context-matrix-") as temp_dir:
        results = [
            harness.run_length(
                length,
                Path(temp_dir) / f"matrix-{length}.sqlite3",
            )
            for length in lengths
        ]
    payload = benchmark_payload(results)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
