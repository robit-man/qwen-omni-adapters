"""Typed System-1 decision plane with a replaceable resident backend.

Application code imports this module, never the Laya SDK. The SDK and weights
live in a separately supervised loopback worker so that a timeout, invalid
answer, dependency failure, or OOM degrades to the deliberative path instead
of taking down the portal or call harness.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

DECISION_CONFIG_SCHEMA = "robit.omni.decision-plane.v1"
DECISION_TRACE_SCHEMA = "robit.omni.decision-trace.v1"


class DecisionPlaneError(RuntimeError):
    """Invalid configuration or backend result."""


class DecisionCategory(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    LAYA_FAST = "LAYA_FAST"
    LAYA_GATED = "LAYA_GATED"
    LLM_REQUIRED = "LLM_REQUIRED"
    HUMAN_AUTHORITY_GATED = "HUMAN_AUTHORITY_GATED"


class DecisionKind(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True)
class DecisionDefinition:
    id: str
    category: DecisionCategory
    version: int
    kind: DecisionKind
    instructions: str
    criteria: Mapping[str, str] | tuple[str, ...]
    calibrated: bool = False
    fast_accept: float | None = None
    enabled: bool = True

    def question(self) -> dict[str, Any]:
        criteria: Any = (
            dict(self.criteria)
            if isinstance(self.criteria, Mapping)
            else list(self.criteria)
        )
        return {
            "type": self.kind.value,
            "instructions": self.instructions,
            "criteria": criteria,
        }


@dataclass(frozen=True)
class DecisionState:
    """Compact canonical state used by every decision wave."""

    user_request: str = ""
    current_goal: str = ""
    current_phase: str = ""
    last_action: Mapping[str, Any] | None = None
    last_observation: str = ""
    available_tool_families: tuple[str, ...] = ()
    pending_requirements: tuple[str, ...] = ()
    authorization_state: Mapping[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    candidate_context: str = ""
    selected_context: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: DecisionState | Mapping[str, Any]) -> DecisionState:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise DecisionPlaneError("decision state must be a mapping")
        known = {item.name for item in cls.__dataclass_fields__.values()}
        raw = {key: copy.deepcopy(item) for key, item in value.items() if key in known}
        for key in ("available_tool_families", "pending_requirements", "selected_context"):
            item = raw.get(key)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                raw[key] = tuple(str(part) for part in item)
        return cls(**raw)

    def normalized(self, max_chars: int) -> tuple[dict[str, Any], bool]:
        """Return bounded JSON state without silently dropping structural fields."""

        if max_chars < 1024:
            raise DecisionPlaneError("state_max_chars must be at least 1024")
        value = asdict(self)
        truncated = False
        per_text = max(256, max_chars // 4)
        for key in (
            "user_request",
            "current_goal",
            "last_observation",
            "candidate_context",
        ):
            text = str(value.get(key) or "")
            if len(text) > per_text:
                value[key] = text[: per_text - 32] + "\n[truncated by decision plane]"
                truncated = True
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if len(encoded) <= max_chars:
            return value, truncated
        # Preserve the decision-bearing scalar fields and reduce only the
        # reversible textual evidence. Never clip JSON bytes into invalid data.
        for key in ("selected_context", "candidate_context", "last_observation", "current_goal"):
            if key in value and value[key]:
                value[key] = "[omitted: compact state limit]"
                truncated = True
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if len(encoded) <= max_chars:
                    return value, truncated
        request = str(value.get("user_request") or "")
        budget = max(128, max_chars - (len(encoded) - len(request)) - 64)
        if len(request) > budget:
            value["user_request"] = request[:budget] + "\n[truncated]"
            truncated = True
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if len(encoded) > max_chars:
            raise DecisionPlaneError("canonical decision state exceeds configured limit")
        return value, truncated


@dataclass(frozen=True)
class DecisionResult:
    decision_id: str
    question_version: int
    value: Any
    confidence: float
    probabilities: Mapping[str, float]
    source: str
    checkpoint: str
    model_version: str
    latency_ms: float
    calibrated: bool
    threshold: float | None
    band: str
    fast_path_taken: bool
    escalated: bool
    input_hash: str
    shadow: bool
    cached: bool = False
    truncated_state: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionWaveResult:
    wave: str
    results: Mapping[str, DecisionResult]
    latency_ms: float
    backend_calls: int
    input_hash: str
    truncated_state: bool

    @property
    def fast_path_ready(self) -> bool:
        return bool(self.results) and all(
            result.fast_path_taken for result in self.results.values()
        )


@dataclass(frozen=True)
class AuthorizationResult:
    disposition: str
    authorized: bool
    reason: str


def authorize_action(
    prediction: DecisionResult,
    *,
    deterministic_validation_passed: bool,
    policy_permits: bool,
    human_confirmation_required: bool = False,
    human_confirmed: bool = False,
) -> AuthorizationResult:
    """Keep prediction, authorization, and execution as separate layers."""

    if not deterministic_validation_passed:
        return AuthorizationResult("deny", False, "deterministic validation failed")
    if not policy_permits:
        return AuthorizationResult("deny", False, "deterministic policy denied the action")
    if human_confirmation_required and not human_confirmed:
        return AuthorizationResult(
            "require_confirmation", False, "explicit human authority is required"
        )
    if prediction.error or prediction.escalated or not prediction.fast_path_taken:
        return AuthorizationResult(
            "escalate", False, "prediction did not clear its calibrated fast-path gate"
        )
    return AuthorizationResult("execute", True, "policy and calibrated prediction agree")


class DecisionBackend(Protocol):
    def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def health(self) -> Mapping[str, Any]: ...


class HttpLayaBackend:
    """Small synchronous client for the separately resident Laya worker."""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = max(0.05, timeout_ms / 1000.0)
        self._client = httpx.Client(timeout=self.timeout_s)

    def close(self) -> None:
        self._client.close()

    def health(self) -> Mapping[str, Any]:
        response = self._client.get(f"{self.endpoint}/health")
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, Mapping):
            raise DecisionPlaneError("decision backend returned invalid health data")
        return value

    def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        response = self._client.post(
            f"{self.endpoint}/predict",
            json={"state": dict(state), "questions": dict(questions)},
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, Mapping):
            raise DecisionPlaneError("decision backend returned a non-object")
        return value


@dataclass
class _CacheEntry:
    created_at: float
    result: DecisionWaveResult


class DecisionPlane:
    """Central typed decision plane and the sole Laya-facing application API."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: DecisionBackend | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.config = _validate_config(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.shadow_mode = bool(self.config.get("shadow_mode", True))
        self.policy_version = int(self.config.get("policy_version", 1))
        self.state_max_chars = int(self.config.get("state_max_chars", 12000))
        self.definitions = _definitions(self.config)
        self.waves = {
            str(name): tuple(str(item) for item in values)
            for name, values in self.config.get("waves", {}).items()
        }
        self.backend = backend
        self.trace_path = trace_path
        self._lock = threading.RLock()
        self._trace_lock = threading.Lock()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        cache = self.config.get("cache", {})
        self._cache_enabled = bool(cache.get("enabled", True))
        self._cache_max_entries = max(0, int(cache.get("max_entries", 512)))
        self._cache_ttl_s = max(0.0, float(cache.get("ttl_seconds", 300)))
        self._metrics: dict[str, Any] = {
            "waves": 0,
            "backend_calls": 0,
            "backend_failures": 0,
            "cache_hits": 0,
            "decisions": 0,
            "fast_paths": 0,
            "escalations": 0,
            "llm_calls_avoided": 0,
            "downstream_errors": 0,
            "latencies_ms": [],
            "by_decision": {},
        }

    @classmethod
    def from_environment(
        cls,
        *,
        backend: DecisionBackend | None = None,
        trace_path: Path | None = None,
    ) -> DecisionPlane:
        path = Path(
            os.environ.get("OMNI_DECISION_PLANE_CONFIG", "").strip()
            or Path(__file__).with_name("decision-plane.yaml")
        )
        config = load_decision_config(path)
        if os.environ.get("OMNI_DECISION_PLANE_ENABLED", "").strip():
            config["enabled"] = _env_bool("OMNI_DECISION_PLANE_ENABLED", True)
        if os.environ.get("OMNI_DECISION_PLANE_SHADOW", "").strip():
            config["shadow_mode"] = _env_bool("OMNI_DECISION_PLANE_SHADOW", True)
        endpoint = os.environ.get("OMNI_DECISION_PLANE_URL", "").strip()
        if endpoint:
            config["endpoint"] = endpoint
        timeout_ms = os.environ.get("OMNI_DECISION_PLANE_TIMEOUT_MS", "").strip()
        if timeout_ms:
            config["timeout_ms"] = max(50, int(timeout_ms))
        if trace_path is None:
            raw_trace = os.environ.get("OMNI_DECISION_TRACE_FILE", "").strip()
            trace_path = Path(raw_trace).expanduser() if raw_trace else None
        if backend is None and bool(config.get("enabled", True)):
            backend = HttpLayaBackend(
                str(config.get("endpoint") or "http://127.0.0.1:8930"),
                int(config.get("timeout_ms", 750)),
            )
        return cls(config, backend=backend, trace_path=trace_path)

    @property
    def ready(self) -> bool:
        return bool(self.health().get("ready"))

    @property
    def models_loaded(self) -> list[str]:
        value = self.health().get("models_loaded", [])
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def device(self) -> str:
        return str(self.health().get("device") or "unavailable")

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    def health(self) -> dict[str, Any]:
        base = {
            "enabled": self.enabled,
            "ready": False,
            "shadow_mode": self.shadow_mode,
            "models_loaded": [],
            "device": "unavailable",
        }
        if not self.enabled:
            base["status"] = "disabled"
            return base
        if self.backend is None:
            base["status"] = "unavailable"
            return base
        try:
            remote = dict(self.backend.health())
        except Exception as exc:  # noqa: BLE001 - an optimization must degrade safely
            base.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            return base
        loaded = remote.get("loaded", remote.get("models_loaded", []))
        base.update(remote)
        base["models_loaded"] = list(loaded) if isinstance(loaded, list) else []
        base["ready"] = bool(remote.get("ready", remote.get("status") == "ok"))
        return base

    def evaluate(
        self,
        *,
        state: DecisionState | Mapping[str, Any],
        decisions: Sequence[str | DecisionDefinition] | None = None,
        wave: str = "custom",
    ) -> DecisionWaveResult:
        started = time.perf_counter()
        canonical, truncated = DecisionState.from_value(state).normalized(
            self.state_max_chars
        )
        selected = self._select(decisions, wave)
        input_hash = _input_hash(canonical)
        eligible = [
            item
            for item in selected
            if item.enabled
            and item.category in {DecisionCategory.LAYA_FAST, DecisionCategory.LAYA_GATED}
        ]
        skipped = [item for item in selected if item not in eligible]
        results = {
            item.id: self._skipped_result(item, input_hash, truncated)
            for item in skipped
        }
        if not eligible or not self.enabled or self.backend is None:
            for item in eligible:
                results[item.id] = self._fallback_result(
                    item,
                    input_hash,
                    truncated,
                    "decision plane disabled or unavailable",
                )
            return self._finish_wave(
                wave, results, started, 0, input_hash, truncated, canonical
            )

        # The cache identity includes every requested definition, including
        # deterministic/policy bypasses.  Keying only the learned subset could
        # replay the wrong bypass results for two custom waves that happened to
        # ask the same Laya questions.
        key = self._cache_key(canonical, selected)
        cached = self._cache_get(key)
        if cached is not None:
            copied = {
                name: DecisionResult(**{**asdict(item), "cached": True})
                for name, item in cached.results.items()
            }
            with self._lock:
                self._metrics["cache_hits"] += 1
            # Cached decisions remain first-class observable decisions.  Run
            # them through the normal accounting/trace path while recording
            # zero backend calls and the current cache-hit latency.
            return self._finish_wave(
                wave,
                {**results, **copied},
                started,
                0,
                input_hash,
                truncated,
                canonical,
            )

        questions = {item.id: item.question() for item in eligible}
        backend_calls = 0
        try:
            backend_calls = 1
            backend_started = time.perf_counter()
            raw = self.backend.evaluate(canonical, questions)
            backend_latency = (time.perf_counter() - backend_started) * 1000
            answers = raw.get("answers")
            if not isinstance(answers, Mapping):
                raise DecisionPlaneError("decision backend response has no answers object")
            routing = raw.get("routing") if isinstance(raw.get("routing"), Mapping) else {}
            checkpoint = str(routing.get("model") or raw.get("checkpoint") or "unknown")
            model_version = str(
                raw.get("model_version")
                or raw.get("model")
                or self.config.get("backend", {}).get("package_version")
                or "unknown"
            )
            reported_latency = raw.get("latency_ms")
            latency_ms = (
                float(reported_latency)
                if isinstance(reported_latency, (int, float))
                else backend_latency
            )
            for item in eligible:
                answer = answers.get(item.id)
                if not isinstance(answer, Mapping):
                    raise DecisionPlaneError(f"missing answer for {item.id}")
                results[item.id] = self._validated_result(
                    item,
                    answer,
                    input_hash=input_hash,
                    checkpoint=checkpoint,
                    model_version=model_version,
                    latency_ms=latency_ms,
                    truncated=truncated,
                )
        except Exception as exc:  # noqa: BLE001 - Laya is an optimization
            with self._lock:
                self._metrics["backend_failures"] += 1
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            for item in eligible:
                results[item.id] = self._fallback_result(
                    item, input_hash, truncated, error
                )

        wave_result = self._finish_wave(
            wave, results, started, backend_calls, input_hash, truncated, canonical
        )
        if not any(item.error for item in results.values() if item.decision_id in questions):
            self._cache_put(key, wave_result)
        return wave_result

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            data = copy.deepcopy(self._metrics)
        latencies = sorted(float(value) for value in data.pop("latencies_ms"))
        for percentile, fraction in (("p50_ms", 0.50), ("p95_ms", 0.95), ("p99_ms", 0.99)):
            if not latencies:
                data[percentile] = 0.0
            else:
                index = min(len(latencies) - 1, int((len(latencies) - 1) * fraction))
                data[percentile] = round(latencies[index], 3)
        decisions = max(1, int(data.get("decisions", 0)))
        data["fast_path_coverage"] = data.get("fast_paths", 0) / decisions
        data["escalation_rate"] = data.get("escalations", 0) / decisions
        by_decision = data.get("by_decision", {})
        if isinstance(by_decision, dict):
            for bucket in by_decision.values():
                values = sorted(float(value) for value in bucket.pop("latencies_ms", []))
                bucket["p50_ms"] = _percentile(values, 0.50)
                bucket["p95_ms"] = _percentile(values, 0.95)
        return data

    def _select(
        self,
        decisions: Sequence[str | DecisionDefinition] | None,
        wave: str,
    ) -> list[DecisionDefinition]:
        values: Sequence[str | DecisionDefinition] = (
            decisions if decisions is not None else self.waves.get(wave, ())
        )
        selected: list[DecisionDefinition] = []
        for value in values:
            if isinstance(value, DecisionDefinition):
                selected.append(value)
                continue
            definition = self.definitions.get(str(value))
            if definition is None:
                raise DecisionPlaneError(f"unknown decision definition {value!r}")
            selected.append(definition)
        batch_max = int(self.config.get("batching", {}).get("max_batch_size", 32))
        if len(selected) > batch_max:
            raise DecisionPlaneError(
                f"decision wave contains {len(selected)} questions; batch limit is {batch_max}"
            )
        return selected

    def _validated_result(
        self,
        definition: DecisionDefinition,
        answer: Mapping[str, Any],
        *,
        input_hash: str,
        checkpoint: str,
        model_version: str,
        latency_ms: float,
        truncated: bool,
    ) -> DecisionResult:
        answer_type = str(answer.get("type") or definition.kind.value)
        if answer_type != definition.kind.value:
            raise DecisionPlaneError(
                f"answer type for {definition.id} is {answer_type!r}, expected {definition.kind.value!r}"
            )
        probabilities: dict[str, float]
        if definition.kind == DecisionKind.CHOICE:
            value = str(answer.get("choice") or "")
            options = set(definition.criteria) if isinstance(definition.criteria, Mapping) else set()
            if not value or value not in options:
                raise DecisionPlaneError(
                    f"answer for {definition.id} selected unknown choice {value!r}"
                )
            raw_probabilities = answer.get("probabilities")
            if not isinstance(raw_probabilities, Mapping):
                raise DecisionPlaneError(f"answer for {definition.id} has no probabilities")
            probabilities = {
                str(key): _probability(number)
                for key, number in raw_probabilities.items()
                if str(key) in options
            }
            if value not in probabilities:
                raise DecisionPlaneError(
                    f"answer for {definition.id} lacks selected probability"
                )
            confidence = _probability(answer.get("confidence", probabilities[value]))
        elif definition.kind == DecisionKind.NOUL:
            yes = _probability(answer.get("noul"))
            value = yes >= 0.5
            probabilities = {"yes": yes, "no": 1.0 - yes}
            confidence = max(yes, 1.0 - yes)
        else:
            score = answer.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise DecisionPlaneError(f"answer for {definition.id} has invalid score")
            value = float(score)
            raw_probabilities = answer.get("probabilities", {})
            probabilities = (
                {
                    str(key): _probability(number)
                    for key, number in raw_probabilities.items()
                }
                if isinstance(raw_probabilities, Mapping)
                else {}
            )
            confidence = _probability(answer.get("confidence", max(probabilities.values(), default=0.0)))

        threshold = definition.fast_accept
        fast = bool(
            not self.shadow_mode
            and definition.calibrated
            and threshold is not None
            and confidence >= threshold
        )
        if fast:
            band = "fast"
        elif threshold is not None and confidence >= max(0.0, threshold - 0.1):
            band = "contextual_check"
        else:
            band = "escalate"
        return DecisionResult(
            decision_id=definition.id,
            question_version=definition.version,
            value=value,
            confidence=confidence,
            probabilities=probabilities,
            source="laya",
            checkpoint=checkpoint,
            model_version=model_version,
            latency_ms=latency_ms,
            calibrated=definition.calibrated,
            threshold=threshold,
            band=band,
            fast_path_taken=fast,
            escalated=not fast,
            input_hash=input_hash,
            shadow=self.shadow_mode,
            truncated_state=truncated,
        )

    def _skipped_result(
        self, definition: DecisionDefinition, input_hash: str, truncated: bool
    ) -> DecisionResult:
        return DecisionResult(
            decision_id=definition.id,
            question_version=definition.version,
            value=None,
            confidence=1.0,
            probabilities={},
            source="deterministic" if definition.category == DecisionCategory.DETERMINISTIC else "policy",
            checkpoint="none",
            model_version="none",
            latency_ms=0.0,
            calibrated=True,
            threshold=None,
            band="bypass",
            fast_path_taken=False,
            escalated=definition.category == DecisionCategory.LLM_REQUIRED,
            input_hash=input_hash,
            shadow=self.shadow_mode,
            truncated_state=truncated,
        )

    def _fallback_result(
        self,
        definition: DecisionDefinition,
        input_hash: str,
        truncated: bool,
        error: str,
    ) -> DecisionResult:
        return DecisionResult(
            decision_id=definition.id,
            question_version=definition.version,
            value=None,
            confidence=0.0,
            probabilities={},
            source="fallback",
            checkpoint="unavailable",
            model_version="unavailable",
            latency_ms=0.0,
            calibrated=False,
            threshold=definition.fast_accept,
            band="escalate",
            fast_path_taken=False,
            escalated=True,
            input_hash=input_hash,
            shadow=self.shadow_mode,
            truncated_state=truncated,
            error=error,
        )

    def _finish_wave(
        self,
        wave: str,
        results: Mapping[str, DecisionResult],
        started: float,
        backend_calls: int,
        input_hash: str,
        truncated: bool,
        trace_state: Mapping[str, Any] | None = None,
    ) -> DecisionWaveResult:
        elapsed_ms = (time.perf_counter() - started) * 1000
        value = DecisionWaveResult(
            wave=wave,
            results=dict(results),
            latency_ms=elapsed_ms,
            backend_calls=backend_calls,
            input_hash=input_hash,
            truncated_state=truncated,
        )
        with self._lock:
            self._metrics["waves"] += 1
            self._metrics["backend_calls"] += backend_calls
            self._metrics["decisions"] += len(results)
            self._metrics["fast_paths"] += sum(
                1 for result in results.values() if result.fast_path_taken
            )
            self._metrics["escalations"] += sum(
                1 for result in results.values() if result.escalated
            )
            self._metrics["latencies_ms"].append(elapsed_ms)
            by_decision = self._metrics["by_decision"]
            for result in results.values():
                bucket = by_decision.setdefault(
                    result.decision_id,
                    {
                        "count": 0,
                        "fast_paths": 0,
                        "escalations": 0,
                        "errors": 0,
                        "latencies_ms": [],
                    },
                )
                bucket["count"] += 1
                bucket["fast_paths"] += int(result.fast_path_taken)
                bucket["escalations"] += int(result.escalated)
                bucket["errors"] += int(bool(result.error))
                bucket["latencies_ms"].append(result.latency_ms)
        self._trace(value, trace_state)
        return value

    def _trace(
        self,
        wave: DecisionWaveResult,
        state: Mapping[str, Any] | None,
    ) -> None:
        trace = self.config.get("trace", {})
        if not trace.get("enabled", True) or self.trace_path is None:
            return
        records = []
        at = time.time()
        for result in wave.results.values():
            definition = self.definitions.get(result.decision_id)
            record = {
                "schema": DECISION_TRACE_SCHEMA,
                "at": at,
                "wave": wave.wave,
                **result.to_dict(),
            }
            if definition is not None:
                record["decision_category"] = definition.category.value
                record["question"] = definition.question()
            if trace.get("include_state", False) and state is not None:
                # Explicit opt-in only. The default content-redacted trace is
                # sufficient for operations and can be joined to separately
                # labelled replay data through input_hash.
                record["state"] = dict(state)
            records.append(json.dumps(record, sort_keys=True, default=str))
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_lock, self.trace_path.open("a", encoding="utf-8") as target:
                target.write("\n".join(records) + "\n")
        except OSError:
            # Observability failure cannot become an availability failure.
            return

    def _cache_key(
        self, state: Mapping[str, Any], definitions: Sequence[DecisionDefinition]
    ) -> str:
        backend = self.config.get("backend", {})
        payload = {
            "state": state,
            "definitions": [
                {
                    "id": item.id,
                    "version": item.version,
                    "question": item.question(),
                }
                for item in definitions
            ],
            "model_id": backend.get("model_id"),
            "package_version": backend.get("package_version"),
            "preload": backend.get("preload"),
            "policy_version": self.policy_version,
            "shadow_mode": self.shadow_mode,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()

    def _cache_get(self, key: str) -> DecisionWaveResult | None:
        if not self._cache_enabled or self._cache_max_entries <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.created_at > self._cache_ttl_s:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(entry.result)

    def _cache_put(self, key: str, result: DecisionWaveResult) -> None:
        if not self._cache_enabled or self._cache_max_entries <= 0:
            return
        with self._lock:
            self._cache[key] = _CacheEntry(time.monotonic(), copy.deepcopy(result))
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)


def load_decision_config(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else Path(__file__).with_name("decision-plane.yaml")
    try:
        value = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DecisionPlaneError(f"cannot load decision-plane config {target}: {exc}") from exc
    return _validate_config(value)


def _validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != DECISION_CONFIG_SCHEMA:
        raise DecisionPlaneError(f"decision-plane config must use {DECISION_CONFIG_SCHEMA}")
    result = copy.deepcopy(dict(value))
    if not isinstance(result.get("decisions"), Mapping):
        raise DecisionPlaneError("decision-plane config requires decisions")
    if not isinstance(result.get("waves"), Mapping):
        raise DecisionPlaneError("decision-plane config requires waves")
    _definitions(result)
    known = set(result["decisions"])
    for wave, members in result["waves"].items():
        if not isinstance(members, list) or any(str(item) not in known for item in members):
            raise DecisionPlaneError(f"wave {wave!r} contains invalid decision ids")
    return result


def _definitions(config: Mapping[str, Any]) -> dict[str, DecisionDefinition]:
    definitions: dict[str, DecisionDefinition] = {}
    for decision_id, raw in config.get("decisions", {}).items():
        if not isinstance(raw, Mapping):
            raise DecisionPlaneError(f"decision {decision_id!r} must be an object")
        try:
            category = DecisionCategory(str(raw["category"]))
            kind = DecisionKind(str(raw["type"]))
            version = int(raw["version"])
            instructions = str(raw["instructions"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionPlaneError(f"invalid definition for {decision_id!r}: {exc}") from exc
        criteria = raw.get("criteria")
        if kind == DecisionKind.CHOICE:
            if not isinstance(criteria, Mapping) or len(criteria) < 2:
                raise DecisionPlaneError(
                    f"choice decision {decision_id!r} needs at least two criteria"
                )
            normalized_criteria: Mapping[str, str] | tuple[str, ...] = {
                str(key): str(item) for key, item in criteria.items()
            }
        elif kind == DecisionKind.SCORE:
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise DecisionPlaneError(
                    f"score decision {decision_id!r} needs an ordered criteria list"
                )
            normalized_criteria = tuple(str(item) for item in criteria)
        else:
            normalized_criteria = (
                {str(key): str(item) for key, item in criteria.items()}
                if isinstance(criteria, Mapping)
                else {}
            )
        threshold = raw.get("fast_accept")
        if threshold is not None:
            threshold = _probability(threshold)
        definition = DecisionDefinition(
            id=str(decision_id),
            category=category,
            version=version,
            kind=kind,
            instructions=instructions,
            criteria=normalized_criteria,
            calibrated=bool(raw.get("calibrated", False)),
            fast_accept=threshold,
            enabled=bool(raw.get("enabled", True)),
        )
        definitions[definition.id] = definition
    return definitions


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionPlaneError(f"invalid probability {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise DecisionPlaneError(f"probability is outside [0,1]: {number}")
    return number


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int((len(values) - 1) * fraction))
    return round(float(values[index]), 3)


def _input_hash(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
