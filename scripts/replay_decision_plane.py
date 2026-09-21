#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_omni_adapters.decision_plane import DecisionPlane  # noqa: E402
from qwen_omni_adapters.decision_replay import (  # noqa: E402
    load_traces,
    render_report,
    replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay traces through the Laya decision plane")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "tests" / "fixtures" / "decision_replay.jsonl"],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--max-fast-error", type=float, default=0.01)
    parser.add_argument("--min-calibration-samples", type=int, default=30)
    args = parser.parse_args()

    plane = DecisionPlane.from_environment()
    try:
        result = replay(
            load_traces(args.inputs),
            plane,
            max_fast_error=max(0.0, args.max_fast_error),
            min_calibration_samples=max(1, args.min_calibration_samples),
        )
    finally:
        plane.close()
    report = render_report(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report)
    if not args.output_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    if not args.output_report:
        print(report, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
