from __future__ import annotations

import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from qwen_omni_adapters.decision_plane import (
    DECISION_TRACE_SCHEMA,
    DecisionCategory,
    DecisionDefinition,
    DecisionKind,
    DecisionPlane,
    DecisionState,
    authorize_action,
    load_decision_config,
)
from qwen_omni_adapters.decision_replay import load_traces, render_report, replay
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPolicy
from runtime.laya_server import LayaRuntime, ServerConfig


class FakeBackend:
    def __init__(
        self,
        answers: dict[str, str] | None = None,
        *,
        confidence: float = 0.97,
        delay_s: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.selected = answers or {}
        self.confidence = confidence
        self.delay_s = delay_s
        self.error = error
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "loaded": ["english"],
            "device": "cpu",
        }

    def evaluate(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        del state
        if self.error is not None:
            raise self.error
        with self.lock:
            self.calls += 1
            self.batch_sizes.append(len(questions))
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            answers: dict[str, Any] = {}
            for question_id, question in questions.items():
                if question["type"] == "choice":
                    options = list(question["criteria"])
                    selected = self.selected.get(question_id, options[0])
                    remainder = (1.0 - self.confidence) / max(1, len(options) - 1)
                    probabilities = {
                        option: self.confidence if option == selected else remainder
                        for option in options
                    }
                    answers[question_id] = {
                        "type": "choice",
                        "choice": selected,
                        "probabilities": probabilities,
                        "confidence": self.confidence,
                    }
                elif question["type"] == "noul":
                    answers[question_id] = {
                        "type": "noul",
                        "noul": self.confidence,
                    }
                else:
                    answers[question_id] = {
                        "type": "score",
                        "score": 1.0,
                        "probabilities": {"0": 0.1, "1": 0.9},
                        "confidence": self.confidence,
                    }
            return {
                "answers": answers,
                "routing": {"model": "english"},
                "latency_ms": 7.5,
                "model_version": "fake-laya-v1",
            }
        finally:
            with self.lock:
                self.active -= 1


def config(*, shadow: bool = True) -> dict[str, Any]:
    value = load_decision_config()
    value["shadow_mode"] = shadow
    value["trace"]["enabled"] = True
    return value


def state(text: str = "Open example.com in the browser") -> DecisionState:
    return DecisionState(
        user_request=text,
        current_goal=text,
        current_phase="input_routing",
        available_tool_families=("browser", "web", "shell"),
        authorization_state={"portal_tools_opted_in": True},
    )


def test_deterministic_and_authority_decisions_bypass_laya() -> None:
    backend = FakeBackend()
    plane = DecisionPlane(config(), backend=backend)
    deterministic = DecisionDefinition(
        id="exact_schema_check",
        category=DecisionCategory.DETERMINISTIC,
        version=1,
        kind=DecisionKind.CHOICE,
        instructions="Was exact validation successful?",
        criteria={"yes": "valid", "no": "invalid"},
    )
    authority = DecisionDefinition(
        id="human_authority",
        category=DecisionCategory.HUMAN_AUTHORITY_GATED,
        version=1,
        kind=DecisionKind.CHOICE,
        instructions="Is explicit authority present?",
        criteria={"yes": "present", "no": "absent"},
    )

    result = plane.evaluate(state=state(), decisions=[deterministic, authority])

    assert backend.calls == 0
    assert result.results["exact_schema_check"].source == "deterministic"
    assert result.results["human_authority"].source == "policy"


def test_eligible_questions_are_one_batch_and_choices_are_not_redundant() -> None:
    backend = FakeBackend(
        {
            "request_kind": "action",
            "work_shape": "one_bounded_action",
            "retrieval_need": "external_current",
            "tool_family": "browser",
            "risk_class": "routine",
        }
    )
    plane = DecisionPlane(config(), backend=backend)

    result = plane.evaluate(state=state(), wave="input_routing")

    assert backend.calls == 1
    assert backend.batch_sizes == [5]
    assert set(result.results) == {
        "request_kind",
        "work_shape",
        "retrieval_need",
        "tool_family",
        "risk_class",
    }
    assert plane.definitions["risk_class"].kind == DecisionKind.CHOICE
    assert set(plane.definitions["risk_class"].criteria) == {
        "routine",
        "consequential",
        "uncertain",
    }


@pytest.mark.parametrize(
    ("confidence", "expected_fast"),
    [(0.96, True), (0.70, False)],
)
def test_calibrated_confidence_gate(confidence: float, expected_fast: bool) -> None:
    value = config(shadow=False)
    value["decisions"]["tool_family"].update(calibrated=True, fast_accept=0.9)
    backend = FakeBackend({"tool_family": "browser"}, confidence=confidence)
    plane = DecisionPlane(value, backend=backend)

    result = plane.evaluate(state=state(), decisions=["tool_family"])

    assert result.results["tool_family"].fast_path_taken is expected_fast
    assert result.results["tool_family"].escalated is not expected_fast


def test_shadow_mode_never_changes_behavior_even_at_high_confidence() -> None:
    value = config(shadow=True)
    value["decisions"]["tool_family"].update(calibrated=True, fast_accept=0.5)
    plane = DecisionPlane(
        value,
        backend=FakeBackend({"tool_family": "browser"}, confidence=0.999),
    )

    result = plane.evaluate(state=state(), decisions=["tool_family"])

    decision = result.results["tool_family"]
    assert decision.shadow is True
    assert decision.fast_path_taken is False
    assert decision.escalated is True


def test_authorization_policy_overrides_model_prediction() -> None:
    value = config(shadow=False)
    value["decisions"]["action_consistency"].update(
        calibrated=True, fast_accept=0.9
    )
    plane = DecisionPlane(
        value,
        backend=FakeBackend({"action_consistency": "aligned"}, confidence=0.99),
    )
    prediction = plane.evaluate(
        state=DecisionState(current_goal="delete it", last_action={"tool": "shell"}),
        decisions=["action_consistency"],
    ).results["action_consistency"]

    denied = authorize_action(
        prediction,
        deterministic_validation_passed=True,
        policy_permits=False,
    )
    confirmation = authorize_action(
        prediction,
        deterministic_validation_passed=True,
        policy_permits=True,
        human_confirmation_required=True,
    )

    assert denied.disposition == "deny" and not denied.authorized
    assert confirmation.disposition == "require_confirmation"
    assert not confirmation.authorized


def test_backend_failure_and_timeout_escalate_cleanly() -> None:
    for error in (RuntimeError("checkpoint unavailable"), TimeoutError("too slow")):
        plane = DecisionPlane(config(shadow=False), backend=FakeBackend(error=error))
        result = plane.evaluate(state=state(), decisions=["tool_family"])
        decision = result.results["tool_family"]
        assert decision.source == "fallback"
        assert decision.escalated is True
        assert decision.value is None
        assert type(error).__name__ in decision.error


def test_cache_uses_immutable_state_question_model_and_policy_identity() -> None:
    value = config()
    backend = FakeBackend({"tool_family": "browser"})
    plane = DecisionPlane(value, backend=backend)

    first = plane.evaluate(state=state(), decisions=["tool_family"])
    second = plane.evaluate(state=state(), decisions=["tool_family"])
    changed_state = plane.evaluate(
        state=state("Run the unit tests"), decisions=["tool_family"]
    )
    plane.config["backend"]["package_version"] = "different-version"
    changed_model = plane.evaluate(state=state(), decisions=["tool_family"])

    assert first.results["tool_family"].cached is False
    assert second.results["tool_family"].cached is True
    assert changed_state.results["tool_family"].cached is False
    assert changed_model.results["tool_family"].cached is False
    assert backend.calls == 3


def test_large_state_is_safely_structurally_truncated() -> None:
    value = config()
    value["state_max_chars"] = 2048
    plane = DecisionPlane(value, backend=FakeBackend())

    result = plane.evaluate(
        state=DecisionState(
            user_request="open the file " + "x" * 8000,
            current_goal="play it " + "y" * 8000,
            last_observation="z" * 8000,
        ),
        decisions=["request_kind"],
    )

    assert result.truncated_state is True
    assert result.results["request_kind"].truncated_state is True


def test_trace_metadata_is_complete_and_content_redacted(tmp_path: Path) -> None:
    trace = tmp_path / "decisions.jsonl"
    plane = DecisionPlane(
        config(),
        backend=FakeBackend({"tool_family": "browser"}),
        trace_path=trace,
    )

    private_state = state("private spoken request")
    plane.evaluate(state=private_state, decisions=["tool_family"])
    plane.evaluate(state=private_state, decisions=["tool_family"])
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    record = records[0]

    assert record["schema"] == DECISION_TRACE_SCHEMA
    for field in (
        "decision_id",
        "question_version",
        "model_version",
        "checkpoint",
        "input_hash",
        "value",
        "confidence",
        "probabilities",
        "latency_ms",
        "threshold",
        "fast_path_taken",
        "escalated",
    ):
        assert field in record
    assert "private spoken request" not in trace.read_text()
    assert record["decision_category"] == "LAYA_GATED"
    assert record["question"]["type"] == "choice"
    assert records[1]["cached"] is True


def test_checkpoint_or_question_version_change_invalidates_cache() -> None:
    backend = FakeBackend({"tool_family": "browser"})
    plane = DecisionPlane(config(), backend=backend)
    plane.evaluate(state=state(), decisions=["tool_family"])
    original = plane.definitions["tool_family"]
    plane.definitions["tool_family"] = DecisionDefinition(
        **{**original.__dict__, "version": original.version + 1}
    )
    plane.evaluate(state=state(), decisions=["tool_family"])

    assert backend.calls == 2


def test_hierarchical_tool_families_are_small() -> None:
    value = config()
    families = value["tool_families"]
    all_tools = {tool for members in families.values() for tool in members}

    assert all(len(members) <= 4 for members in families.values())
    assert len(all_tools) > max(len(members) for members in families.values())
    assert "browser_interact" in families["browser"]
    assert "shell" not in families["browser"]


def test_concurrent_callers_share_one_plane_and_resident_backend() -> None:
    backend = FakeBackend({"tool_family": "browser"}, delay_s=0.03)
    value = config()
    value["cache"]["enabled"] = False
    plane = DecisionPlane(value, backend=backend)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda index: plane.evaluate(
                    state=state(f"Open site {index}"), decisions=["tool_family"]
                ),
                range(4),
            )
        )

    assert len(results) == 4
    assert backend.calls == 4
    assert all(result.backend_calls == 1 for result in results)
    assert plane.health()["models_loaded"] == ["english"]


def test_replay_compares_architectures_and_refuses_fixture_only_calibration() -> None:
    traces = load_traces(
        [Path(__file__).parent / "fixtures" / "decision_replay.jsonl"]
    )
    selected = {
        "request_kind": "action",
        "work_shape": "multi_step",
        "retrieval_need": "local",
        "tool_family": "shell",
        "risk_class": "routine",
        "action_outcome": "success",
        "next_disposition": "finish",
        "completion_state": "satisfied",
    }
    result = replay(
        traces,
        DecisionPlane(config(), backend=FakeBackend(selected)),
        min_calibration_samples=30,
    )
    report = render_report(result)

    assert result["traces"] == len(traces)
    assert result["summary"]["new_llm_calls"] == result["summary"]["original_llm_calls"]
    assert result["summary"]["llm_calls_avoided"] == 0
    assert all(
        not item["calibration_sufficient"]
        for item in result["calibration"].values()
    )
    assert "fixture-only" in report.lower()


def test_resident_server_preloads_and_warms_before_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        device = "cpu"

    class FakeRouter:
        def __init__(self, **_kwargs: Any) -> None:
            self.loaded: list[str] = []
            self._agents: dict[str, Any] = {}
            self.calls = 0

        def preload(self, names: list[str]) -> None:
            self.loaded.extend(names)
            self._agents.update({name: FakeAgent() for name in names})

        def route(
            self, _state: Any, _questions: Any, model: str | None = None
        ) -> dict[str, Any]:
            return {"model": model or "english", "reason": "test"}

        def predict(
            self, _state: Any, questions: dict[str, Any], model: str
        ) -> dict[str, Any]:
            assert model == "english"
            self.calls += 1
            return {
                "answers": {
                    name: {
                        "type": "choice",
                        "choice": next(iter(question["criteria"])),
                        "probabilities": {
                            option: 1.0 if index == 0 else 0.0
                            for index, option in enumerate(question["criteria"])
                        },
                        "confidence": 1.0,
                    }
                    for name, question in questions.items()
                }
            }

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=FakeRouter))
    runtime = LayaRuntime(
        ServerConfig(
            host="127.0.0.1",
            port=0,
            device="cpu",
            preload=("english",),
            allow_lazy_load=False,
            warmup=True,
            startup_reserve_gib=0.0,
            max_batch_size=32,
            max_wait_ms=2,
            state_max_chars=12000,
            model_id="convaiinnovations/laya",
            checkpoint=None,
            cpu_quantization="none",
        )
    )
    runtime.governor = MemoryGovernor(MemoryPolicy(enabled=False))
    runtime.package_version = "test"

    runtime.load()

    assert runtime.ready is True
    assert runtime.router.loaded == ["english"]
    assert runtime.router.calls == 1
    assert runtime.health()["models_loaded"] == ["english"]


def test_resident_server_rejects_concurrent_queueing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        device = types.SimpleNamespace(type="cpu")

    class SlowRouter:
        def __init__(self, **_kwargs: Any) -> None:
            self.loaded: list[str] = []
            self._agents: dict[str, Any] = {}

        def preload(self, names: list[str]) -> None:
            self.loaded.extend(names)
            self._agents.update({name: FakeAgent() for name in names})

        def route(
            self, _state: Any, _questions: Any, model: str | None = None
        ) -> dict[str, Any]:
            return {"model": model or "english", "reason": "test"}

        def predict(
            self, _state: Any, questions: dict[str, Any], model: str
        ) -> dict[str, Any]:
            del model
            time.sleep(0.05)
            return {
                "answers": {
                    name: {
                        "type": "choice",
                        "choice": next(iter(question["criteria"])),
                        "probabilities": {
                            option: 1.0 if index == 0 else 0.0
                            for index, option in enumerate(question["criteria"])
                        },
                        "confidence": 1.0,
                    }
                    for name, question in questions.items()
                }
            }

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=SlowRouter))
    runtime = LayaRuntime(
        ServerConfig(
            host="127.0.0.1",
            port=0,
            device="cpu",
            preload=("english",),
            allow_lazy_load=False,
            warmup=False,
            startup_reserve_gib=0.0,
            max_batch_size=32,
            max_wait_ms=1,
            state_max_chars=12000,
            model_id="convaiinnovations/laya",
            checkpoint="english",
            cpu_quantization="none",
        )
    )
    runtime.governor = MemoryGovernor(MemoryPolicy(enabled=False))
    runtime.load()
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"direct": "direct", "tool": "tool"},
        }
    }

    barrier = threading.Barrier(2)

    def invoke() -> str:
        barrier.wait()
        try:
            runtime.predict({"user_request": "open it"}, questions)
            return "ok"
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: invoke(), range(2)))

    assert outcomes.count("ok") == 1
    assert sum("busy" in outcome for outcome in outcomes) == 1
    assert runtime.health()["busy_rejections"] == 1


def test_configured_questions_are_versioned_and_no_prompts_live_in_call_sites() -> None:
    value = config()
    assert all(raw["version"] >= 1 for raw in value["decisions"].values())
    assert all(str(raw["instructions"]).strip() for raw in value["decisions"].values())
    assert value["backend"]["allow_lazy_load"] is False
    assert value["backend"]["checkpoint"] == "multilingual"


def test_config_validation_rejects_unknown_wave_member() -> None:
    value = config()
    value["waves"]["input_routing"].append("not_a_decision")
    with pytest.raises(Exception, match="invalid decision ids"):
        DecisionPlane(value, backend=FakeBackend())


def test_policy_version_change_invalidates_cache() -> None:
    value = config()
    backend = FakeBackend()
    plane = DecisionPlane(value, backend=backend)
    plane.evaluate(state=state(), decisions=["request_kind"])
    plane.policy_version += 1
    plane.evaluate(state=state(), decisions=["request_kind"])
    assert backend.calls == 2


def test_invalid_model_output_falls_back_instead_of_crashing() -> None:
    class InvalidBackend(FakeBackend):
        def evaluate(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
            del state, questions
            self.calls += 1
            return {
                "answers": {
                    "tool_family": {
                        "type": "choice",
                        "choice": "invented_flat_tool",
                        "probabilities": {"invented_flat_tool": 1.0},
                        "confidence": 1.0,
                    }
                }
            }

    plane = DecisionPlane(config(shadow=False), backend=InvalidBackend())
    result = plane.evaluate(state=state(), decisions=["tool_family"])
    assert result.results["tool_family"].source == "fallback"
    assert result.results["tool_family"].escalated is True
