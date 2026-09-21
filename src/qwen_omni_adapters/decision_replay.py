"""Offline replay, calibration, and architecture comparison for decision waves."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_plane import DecisionPlane, DecisionState

REPLAY_SCHEMA = "robit.omni.decision-replay.v1"


@dataclass(frozen=True)
class ReplayTrace:
    trace_id: str
    wave: str
    state: Mapping[str, Any]
    labels: Mapping[str, Any]
    source: str = "fixture"
    original_llm_calls: int = 1
    bounded_llm_calls: int = 0
    original_wall_ms: float = 0.0
    bounded_llm_latency_ms: float = 0.0
    task_success: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ReplayTrace:
        trace_id = str(value.get("trace_id") or "").strip()
        wave = str(value.get("wave") or "").strip()
        state = value.get("state")
        labels = value.get("labels")
        if not trace_id or not wave or not isinstance(state, Mapping) or not isinstance(labels, Mapping):
            raise ValueError("replay trace requires trace_id, wave, state, and labels")
        return cls(
            trace_id=trace_id,
            wave=wave,
            state=dict(state),
            labels=dict(labels),
            source=str(value.get("source") or "fixture"),
            original_llm_calls=max(0, int(value.get("original_llm_calls", 1))),
            bounded_llm_calls=max(0, int(value.get("bounded_llm_calls", 0))),
            original_wall_ms=max(0.0, float(value.get("original_wall_ms", 0.0))),
            bounded_llm_latency_ms=max(
                0.0, float(value.get("bounded_llm_latency_ms", 0.0))
            ),
            task_success=bool(value.get("task_success", True)),
        )


def load_traces(paths: Sequence[Path | str]) -> list[ReplayTrace]:
    traces: list[ReplayTrace] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError("trace must be an object")
                    traces.append(ReplayTrace.from_mapping(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid replay trace {path}:{line_number}: {exc}") from exc
    return traces


def replay(
    traces: Sequence[ReplayTrace],
    plane: DecisionPlane,
    *,
    max_fast_error: float = 0.01,
    min_calibration_samples: int = 30,
) -> dict[str, Any]:
    started_calls = plane.metrics().get("backend_calls", 0)
    records: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_latencies: list[float] = []
    new_latencies: list[float] = []
    original_calls = 0
    new_calls = 0
    avoided = 0
    escalations = 0
    fast_count = 0
    incorrect_fast = 0
    original_successes = 0
    estimated_successes = 0
    decision_latencies: list[float] = []

    for trace in traces:
        result = plane.evaluate(state=DecisionState.from_value(trace.state), wave=trace.wave)
        original_calls += trace.original_llm_calls
        original_successes += int(trace.task_success)
        relevant = {
            name: item for name, item in result.results.items() if name in trace.labels
        }
        fast = bool(relevant) and all(item.fast_path_taken for item in relevant.values())
        correct = bool(relevant) and all(
            _same_value(item.value, trace.labels[name]) for name, item in relevant.items()
        )
        avoided_here = min(trace.bounded_llm_calls, trace.original_llm_calls) if fast else 0
        avoided += avoided_here
        new_calls += trace.original_llm_calls - avoided_here
        fast_count += int(fast)
        incorrect_fast += int(fast and not correct)
        escalations += int(any(item.escalated for item in relevant.values()))
        estimated_success = trace.task_success and not (fast and not correct)
        estimated_successes += int(estimated_success)
        original_latency = trace.original_wall_ms
        decision_latencies.append(result.latency_ms)
        # Production shadow waves are dispatched asynchronously and therefore
        # add no time to the authoritative path. Active replay includes their
        # measured cost in the end-to-end estimate.
        critical_decision_ms = 0.0 if plane.shadow_mode else result.latency_ms
        new_latency = max(
            0.0,
            original_latency
            - (trace.bounded_llm_latency_ms if avoided_here else 0.0)
            + critical_decision_ms,
        )
        original_latencies.append(original_latency)
        new_latencies.append(new_latency)
        decisions: dict[str, Any] = {}
        for name, item in relevant.items():
            expected = trace.labels[name]
            item_correct = _same_value(item.value, expected)
            sample = {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "expected": expected,
                "predicted": item.value,
                "confidence": item.confidence,
                "probabilities": dict(item.probabilities),
                "correct": item_correct,
                "fast_path": item.fast_path_taken,
                "escalated": item.escalated,
                "checkpoint": item.checkpoint,
                "model_version": item.model_version,
                "question_version": item.question_version,
                "latency_ms": item.latency_ms,
                "error": item.error,
            }
            samples[name].append(sample)
            decisions[name] = sample
        records.append(
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "wave": trace.wave,
                "original_llm_calls": trace.original_llm_calls,
                "new_llm_calls": trace.original_llm_calls - avoided_here,
                "calls_eliminated": avoided_here,
                "original_wall_ms": original_latency,
                "new_wall_ms": round(new_latency, 3),
                "decision_agreement": correct,
                "action_agreement": correct,
                "escalated": any(item.escalated for item in relevant.values()),
                "incorrect_fast_path": fast and not correct,
                "original_task_success": trace.task_success,
                "estimated_task_success": estimated_success,
                "decisions": decisions,
            }
        )

    calibration = {
        decision_id: _calibration(
            values,
            max_fast_error=max_fast_error,
            min_samples=min_calibration_samples,
        )
        for decision_id, values in sorted(samples.items())
    }
    count = len(traces)
    backend_calls = plane.metrics().get("backend_calls", 0) - started_calls
    return {
        "schema": REPLAY_SCHEMA,
        "mode": "shadow" if plane.shadow_mode else "active",
        "traces": count,
        "sources": dict(Counter(trace.source for trace in traces)),
        "summary": {
            "original_llm_calls": original_calls,
            "new_llm_calls": new_calls,
            "llm_calls_avoided": avoided,
            "llm_call_reduction": avoided / max(1, original_calls),
            "decision_plane_backend_calls": backend_calls,
            "fast_path_coverage": fast_count / max(1, count),
            "escalation_percentage": 100.0 * escalations / max(1, count),
            "incorrect_fast_paths": incorrect_fast,
            "original_task_success": original_successes / max(1, count),
            "estimated_task_success": estimated_successes / max(1, count),
            "original_latency_ms": _latency_summary(original_latencies),
            "new_latency_ms": _latency_summary(new_latencies),
            "estimated_latency_reduction": (
                1.0 - sum(new_latencies) / sum(original_latencies)
                if sum(original_latencies) > 0
                else 0.0
            ),
            "decision_plane_latency_ms": _latency_summary(decision_latencies),
        },
        "calibration": calibration,
        "records": records,
    }


def render_report(result: Mapping[str, Any]) -> str:
    summary = result.get("summary", {})
    original_latency = summary.get("original_latency_ms", {})
    new_latency = summary.get("new_latency_ms", {})
    lines = [
        "# Laya decision-plane replay report",
        "",
        f"Mode: **{result.get('mode', 'unknown')}**. Traces: **{result.get('traces', 0)}**.",
        "",
        "## Architecture comparison",
        "",
        "| Metric | Current | Laya-augmented |",
        "|---|---:|---:|",
        f"| LLM calls | {summary.get('original_llm_calls', 0)} | {summary.get('new_llm_calls', 0)} |",
        f"| p50 latency | {original_latency.get('p50', 0):.1f} ms | {new_latency.get('p50', 0):.1f} ms |",
        f"| p95 latency | {original_latency.get('p95', 0):.1f} ms | {new_latency.get('p95', 0):.1f} ms |",
        f"| p99 latency | {original_latency.get('p99', 0):.1f} ms | {new_latency.get('p99', 0):.1f} ms |",
        f"| Task success | {100 * summary.get('original_task_success', 0):.2f}% | {100 * summary.get('estimated_task_success', 0):.2f}% |",
        f"| Fast-path coverage | — | {100 * summary.get('fast_path_coverage', 0):.2f}% |",
        f"| Escalation | — | {summary.get('escalation_percentage', 0):.2f}% |",
        "",
        "## Per-decision calibration",
        "",
        "| Decision | available/total | Accuracy | Brier | ECE | Safe threshold | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    calibration = result.get("calibration", {})
    if isinstance(calibration, Mapping):
        for name, raw in calibration.items():
            value = raw if isinstance(raw, Mapping) else {}
            threshold = value.get("recommended_threshold")
            lines.append(
                f"| {name} | {value.get('successful_samples', 0)}/{value.get('samples', 0)} "
                f"| {value.get('accuracy', 0):.3f} "
                f"| {value.get('brier', 0):.3f} | {value.get('ece', 0):.3f} "
                f"| {threshold if threshold is not None else 'not calibrated'} "
                f"| {value.get('coverage_at_recommended_threshold', 0):.3f} |"
            )
    lines.extend(
        [
            "",
            "Thresholds are recommendations only when the configured minimum sample count and",
            "maximum false-fast-path error are both satisfied. Fixture-only results do not",
            "authorize production activation; real shadow traces must be added to the replay.",
            "",
        ]
    )
    return "\n".join(lines)


def _calibration(
    samples: Sequence[Mapping[str, Any]], *, max_fast_error: float, min_samples: int
) -> dict[str, Any]:
    total_count = len(samples)
    usable = [
        item
        for item in samples
        if not item.get("error")
        and item.get("predicted") is not None
        and isinstance(item.get("probabilities"), Mapping)
        and bool(item.get("probabilities"))
    ]
    count = len(usable)
    correct = sum(bool(item.get("correct")) for item in usable)
    labels = sorted(
        {
            str(item.get(key))
            for item in usable
            for key in ("expected", "predicted")
            if item.get(key) is not None
        }
    )
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    brier_values: list[float] = []
    for item in usable:
        expected = str(item.get("expected"))
        predicted = str(item.get("predicted"))
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1
        probabilities = item.get("probabilities")
        if isinstance(probabilities, Mapping) and probabilities:
            options = set(str(key) for key in probabilities) | {expected}
            brier_values.append(
                sum(
                    (float(probabilities.get(option, 0.0)) - (1.0 if option == expected else 0.0))
                    ** 2
                    for option in options
                )
                / max(1, len(options))
            )
    precision_recall: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        precision_recall[label] = {
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
        }
    thresholds: list[dict[str, Any]] = []
    recommended: float | None = None
    recommended_coverage = 0.0
    for step in range(50, 100):
        threshold = step / 100.0
        selected = [item for item in usable if float(item.get("confidence", 0.0)) >= threshold]
        errors = sum(not bool(item.get("correct")) for item in selected)
        error_rate = errors / max(1, len(selected))
        coverage = len(selected) / max(1, count)
        thresholds.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "error_rate": error_rate,
                "false_fast_paths": errors,
            }
        )
        if (
            count >= min_samples
            and selected
            and error_rate <= max_fast_error
            and coverage > recommended_coverage
        ):
            recommended = threshold
            recommended_coverage = coverage
    return {
        "samples": total_count,
        "successful_samples": count,
        "unavailable_samples": total_count - count,
        "accuracy": correct / max(1, count),
        "precision_recall": precision_recall,
        "confusion_matrix": confusion,
        "brier": sum(brier_values) / max(1, len(brier_values)),
        "ece": _ece(usable),
        "recommended_threshold": recommended,
        "coverage_at_recommended_threshold": recommended_coverage,
        "threshold_curve": thresholds,
        "calibration_sufficient": recommended is not None,
    }


def _ece(samples: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    total = max(1, len(samples))
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item in samples
            if lower <= float(item.get("confidence", 0.0)) <= upper
            and (index == bins - 1 or float(item.get("confidence", 0.0)) < upper)
        ]
        if not members:
            continue
        confidence = sum(float(item.get("confidence", 0.0)) for item in members) / len(members)
        accuracy = sum(bool(item.get("correct")) for item in members) / len(members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def _latency_summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "mean": sum(ordered) / max(1, len(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, math.ceil(len(values) * fraction) - 1)
    return values[max(0, index)]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return str(left) == str(right)
