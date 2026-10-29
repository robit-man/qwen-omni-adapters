"""Private replanning never becomes native input, progress, or an unbounded loop."""
import copy
import json
import threading

import httpx
import pytest
from test_native_working_state import envelope
from test_portal_native_task_routes import _config, _install_store, _post, _request
from test_ui_task import ALLOWED, GOAL, Adapter, final

from portal.ui_task import NativePhaseReplan, run_native_goal


def run_replans(*, repeated=False, cancel_after_proposal=False):
    adapter, packets, dispatched, stop = Adapter(), [], [], threading.Event()

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        if repeated or len(packets) == 1:
            if cancel_after_proposal:
                stop.set()
            raise NativePhaseReplan("The observed target needs a fresh proposal.")
        return envelope(packet, {"kind": "blocked", "reason": "Task criteria remain unverified."})

    def execute(name, arguments):
        dispatched.append((name, arguments))
        raise AssertionError("A replan must not reach an executor")

    events = list(run_native_goal(GOAL, observe=adapter.observe,
        compile_phase=compile_phase, execute=execute, cancel=stop.set,
        cancelled=stop.is_set, allowed_tools=ALLOWED, working_state=True,
        effect_contracts=True))
    return adapter, packets, dispatched, final(events)


def test_explicit_replan_is_no_input_and_next_phase_keeps_original_source_epoch():
    adapter, packets, dispatched, terminal = run_replans()
    assert not adapter.calls and not dispatched and len(packets) == 2
    first, second = (packet["working_state"] for packet in packets)
    assert first["source_epoch"] == second["source_epoch"]
    assert second["source_phase"] == 2 and second["recovery_mode"] is True
    assert second["consecutive_unverified_count"] == 1
    result = packets[1]["last_result"]
    assert result["error"] == NativePhaseReplan.code
    assert result["dispatch_status"] == "not_dispatched" and result["input_attempted"] is False
    assert result["new_effect_count"] == 0 and "unverified claim" in result["reason"]
    assert second["executed_facts"] == [] and second["phase_refs"][0][2] == 0
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_replanning_cannot_reset_existing_no_progress_bound():
    adapter, packets, dispatched, terminal = run_replans(repeated=True)
    assert len(packets) == 3 and not adapter.calls and not dispatched
    assert packets[-1]["working_state"]["consecutive_unverified_count"] == 2
    assert terminal["portal"]["native_task"]["status"] == "blocked"
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_cancellation_after_valid_replan_takes_precedence_without_executor_call():
    adapter, packets, dispatched, terminal = run_replans(cancel_after_proposal=True)
    assert len(packets) == 1 and not adapter.calls and not dispatched
    assert terminal["portal"]["native_task"]["status"] == "cancelled"


@pytest.mark.parametrize("reason", [None, "", " ", "x" * 501])
def test_replan_reason_is_bounded_and_has_no_raw_exception_document(reason):
    with pytest.raises(ValueError):
        NativePhaseReplan(reason)


def test_portal_converts_validated_atomic_replan_to_bounded_no_input_feedback(tmp_path, monkeypatch):
    from portal.app import create_app
    from portal.computer_use import SystemOneUiProvider
    monkeypatch.setenv("OMNI_NATIVE_COMPACT_PLANNER", "1")
    store, adapters = _install_store(monkeypatch)
    store.provider = SystemOneUiProvider("http://127.0.0.1:18959", mode="active")
    seen = []

    def upstream(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        name = payload["response_format"]["json_schema"]["name"]
        seen.append(name)
        if name == "native_ui_subgoal":
            response = {"current_view": "A captured owned editor.", "needed_evidence": [],
                "next_subgoal": "Read the observed source before authoring.",
                "next_operation": "Request fresh planning if this cannot proceed.",
                "immediate_action": {"family": "blocked"}}
        else:
            response = {"action": {"family": "replan", "reason": "The source needs expansion.",
                    "objective": context["planning_handoff"]["proposal"]["next_subgoal"]},
                "source_epoch": context["working_state"]["source_epoch"],
                "source_phase": context["working_state"]["source_phase"],
                "assessment": "No input or effect is claimed.",
                "intended_resource": "Captured source", "remaining_criteria": ["Read the source"]}
        return httpx.Response(200, json={"message": {"content": json.dumps(response)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    terminal, events = _post(app.test_client(), _request(True), True)
    assert seen == ["native_ui_subgoal", "native_ui_phase"] * 3
    assert not adapters[0].calls and not [event for event in events if event.get("type") == "tool"]
    task = terminal["portal"]["native_task"]
    assert task["status"] == "blocked" and task["task_verified"] is False
    assert task["working_state"]["latest_host_outcome"]["facts"]["input_attempted"] is False
