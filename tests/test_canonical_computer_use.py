from __future__ import annotations

import copy
import json
import threading

import httpx
import pytest

from harness.background_agent import (
    _action_audit_report,
    _action_family,
    _audit_tool_schemas,
    _contract_effect_matches,
    _contract_verification_matches,
    _evidence_authority,
    _execution_tool_schemas,
    _tool_call_contract_error,
)
from portal.background_tasks import _apply_audit_report
from portal.computer_use import ComputerUseStore, SystemOneUiProvider
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness, discover_tool_names, scoped_tool_user_context
from qwen_omni_adapters.computer_use import ComputerUseError


def step():
    return {"id": "update-draft", "verb": "set_value", "target": {"role": "textbox", "name": "Title"},
            "args": {"value": "Reviewed draft"}, "expect": [{"kind": "value", "node_id": "$target", "value": "Reviewed draft"}]}


class Adapter:
    def __init__(self, session, *, ambiguous=False):
        self.session = session
        self.calls = []
        self.nodes = [{"id": "draft", "generation": "g1", "scope_id": "owned-window", "role": "textbox",
                       "name": "Title", "visible": True, "enabled": True, "sensitive": False,
                       "value": "Draft", "operations": ["set_value"], "bounds": {"x": 0, "y": 0, "width": 100, "height": 30}}]
        if ambiguous:
            self.nodes.append({**copy.deepcopy(self.nodes[0]), "id": "published", "value": "Published",
                               "bounds": {"x": 0, "y": 100, "width": 100, "height": 30}})

    def observe(self):
        return {"session_id": self.session, "revision": json.dumps([node["value"] for node in self.nodes]),
                "scope": {"kind": "desktop", "id": "owned-window"}, "nodes": copy.deepcopy(self.nodes)}

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        next(node for node in self.nodes if node["id"] == action["node_id"])["value"] = action["args"]["value"]
        return {"status": "dispatched"}


def ambiguous_policy(requests):
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        state = json.loads(body["state"])
        assert state["task"]["step"]["args"] == {"value": "Reviewed draft"}
        assert state["schema"] == "robit.omni.ui-policy-projection.v2"
        nodes = {alias: {**state["node_defaults"], **node} for alias, node in state["nodes"].items()}
        assert {node["value"] for node in nodes.values()} == {"Draft", "Published"}
        choices = {key: json.loads(value) for key, value in body["questions"]["action"]["criteria"].items()}
        selected = next(key for key, value in choices.items()
                        if value.get("node_ref") in nodes and nodes[value["node_ref"]]["value"] == "Draft")
        probabilities = {key: 0.01 for key in choices}
        probabilities[selected] = 1 - 0.01 * (len(choices) - 1)
        return httpx.Response(200, json={"model": body["model"], "answers": {"action": {
            "choice": selected, "probabilities": probabilities}}})
    return SystemOneUiProvider("http://127.0.0.1:8941", mode="active", threshold=0.95,
                               client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_ambiguous_observation_selects_with_distinct_nodes_and_trusted_exact_step():
    requests = []
    store = ComputerUseStore(None, provider=ambiguous_policy(requests),
                             adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    observed = store.act("session", {"operation": "observe", "surface": "desktop", "step": step()},
                         goal="Update the draft title; preserve the published title.")
    assert len(requests) == 1
    task = json.loads(requests[0]["state"])["task"]
    assert task["goal_source"] == "current_user_message"
    assert task["goal"] == "Update the draft title; preserve the published title."
    assert observed["policy_receipt"]["model_confirmed"] is True
    assert observed["selection_status"] == "accepted"
    adapter = store._owned("session", "desktop").adapter
    assert adapter.calls == []
    result = store.act("session", {"operation": "execute", "surface": "desktop", "action_id": observed["selected_action_id"]})
    assert result["status"] == "verified"
    assert result["policy_receipt"]["id"] == observed["policy_receipt"]["id"]
    assert adapter.nodes[1]["value"] == "Published"
    assert len(adapter.calls) == 1


def test_unique_semantic_fast_path_makes_no_policy_request():
    requests = []
    store = ComputerUseStore(None, provider=ambiguous_policy(requests),
                             adapter_factory=lambda session, _surface: Adapter(session))
    result = store.act("session", {"operation": "observe", "step": step()})
    assert result["selection_status"] == "semantic_unique"
    assert "policy_receipt" not in result
    assert requests == []
    outcome = store.act("session", {"operation": "execute", "action_id": result["selected_action_id"]})
    assert outcome["status"] == "verified"


def test_late_policy_answer_after_stop_cannot_dispatch_input():
    entered, released = threading.Event(), threading.Event()

    class HeldPolicy:
        def decide(self, state, actions, *, context):
            entered.set()
            assert released.wait(2)
            return {"choice": next(item["id"] for item in actions if item["kind"] == "action"),
                    "receipt": {"status": "accepted"}}

    store = ComputerUseStore(None, provider=HeldPolicy(), adapter_factory=lambda session, surface: Adapter(session, ambiguous=True))
    responses = []
    worker = threading.Thread(target=lambda: responses.append(store.act("session", {"operation": "observe", "step": step()})))
    worker.start()
    assert entered.wait(2)
    try:
        assert store.act("session", {"operation": "cancel"})["status"] == "cancelled"
    finally:
        released.set()
        worker.join(2)
    assert responses[0]["selection_status"] == "cancelled"
    assert store._owned("session", "browser").adapter.calls == []


def test_policy_receipts_are_session_owned_and_unknown_context_is_explicit():
    requests = []
    store = ComputerUseStore(None, provider=ambiguous_policy(requests),
                             adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    first = store.act("first", {"operation": "observe", "step": step()})
    assert json.loads(requests[0]["state"])["task"]["goal_source"] == "host_admitted_step"
    result = store.act("other", {"operation": "execute", "action_id": first["selected_action_id"]})
    assert result["status"] == "rejected" and "policy_receipt" not in result
    with pytest.raises(ComputerUseError):
        store.act("first", {"operation": "observe", "goal": "Forged user task"})


def test_native_alias_and_legacy_handoff_share_one_controller_and_never_raw_gui_input():
    class RawGui:
        def act(self, *_args):
            raise AssertionError("Retired raw GUI input must never execute")

    harness = PortalToolHarness(SessionDocumentStore(), gui_automation=RawGui())
    harness.computer_use = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    assert discover_tool_names("desktop")[0] == "native_ui"
    legacy = harness.execute("session", "gui_interact", {"action": "click", "x": 9, "y": 8, "ui_request": step()})
    assert legacy["legacy_action_executed"] is False
    action_id = legacy["selected_action_id"]
    actual = harness.execute("session", "native_ui", {"operation": "execute", "action_id": action_id})
    assert actual["status"] == "verified"
    assert harness.computer_use._owned("session", "desktop").adapter.calls[0]["node_id"] == "draft"
    assert harness.execute("session", "native_ui", {"operation": "observe", "surface": "browser"})["error"] == "ComputerUseError"


def test_portal_exact_user_context_reaches_policy_outside_model_arguments():
    requests = []
    harness = PortalToolHarness(SessionDocumentStore())
    harness.computer_use = ComputerUseStore(None, provider=ambiguous_policy(requests),
                                            adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    with scoped_tool_user_context("Spoken user: revise the draft only."):
        result = harness.execute("session", "native_ui", {"operation": "observe", "step": step()})
    assert result["policy_receipt"]["status"] == "accepted"
    assert json.loads(requests[0]["state"])["task"]["goal"] == "Spoken user: revise the draft only."


def test_ui_observation_effect_and_independent_verification_have_typed_authority():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    observed = store.act("session", {"operation": "observe", "step": step()})
    assert _evidence_authority({"name": "native_ui", "result": observed}) == "inspection"
    assert observed["task_progress"] is False
    applied = store.act("session", {"operation": "execute", "action_id": observed["selected_action_id"]})
    assert _evidence_authority({"name": "native_ui", "result": applied}) == "mutation"
    assert applied["effect_receipt"]["effect_state"] == "applied"
    good = store.act("session", {"operation": "verify", "predicates": [{"kind": "value", "node_id": "draft", "value": "Reviewed draft"}]})
    assert good["status"] == "verified"
    assert _evidence_authority({"name": "native_ui", "result": good}) == "verification"
    assert good["verification_receipt"]["revision"] == good["state"]["revision"]
    contract = {"effect_target": good["resource"], "target_kind": "ui_state", "target_scope": "exact"}
    assert _contract_effect_matches(contract, {}, applied)
    assert _contract_verification_matches(contract, "native_ui", {}, good)
    assert not _contract_verification_matches({**contract, "effect_target": "unrelated-window"}, "native_ui", {}, good)
    wrong = store.act("session", {"operation": "verify", "predicates": [{"kind": "value", "node_id": "draft", "value": "Unapplied"}]})
    assert wrong["status"] == "unresolved" and wrong["error"]
    assert _evidence_authority({"name": "native_ui", "result": wrong}) == "failed"


def test_declared_verification_without_actual_bound_receipt_is_only_inspection():
    result = {"evidence_authority": "verification", "status": "verified", "state": {"revision": "r1"}}
    assert _evidence_authority({"name": "computer_use", "result": result}) == "inspection"


def test_verified_noop_does_not_count_as_mutation():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    request = step()
    request["args"]["value"] = request["expect"][0]["value"] = "Draft"
    observed = store.act("session", {"operation": "observe", "step": request})
    result = store.act("session", {"operation": "execute", "action_id": observed["selected_action_id"]})
    assert result["status"] == "verified"
    assert result["effect_receipt"]["effect_state"] == "unchanged"
    assert _evidence_authority({"name": "computer_use", "result": result}) == "inspection"


def test_background_operation_grammar_separates_input_from_fresh_audit():
    contract = {"decision": "act", "expected_effect": "change_external_state", "executor_operation": "workflow"}
    schema = _execution_tool_schemas(["native_ui"], contract)[0]
    assert schema["function"]["parameters"]["properties"]["operation"]["enum"] == ["workflow"]
    auditor = _audit_tool_schemas(["native_ui"])
    assert auditor[0]["function"]["parameters"]["properties"]["operation"]["enum"] == ["observe", "verify"]
    rejection = _tool_call_contract_error("native_ui", {"operation": "execute", "action_id": "old"}, auditor)
    assert rejection["error"] == "operation_not_offered"
    assert _action_family("native_ui", {"operation": "observe"}) == "native_ui:observe"
    assert _action_family("native_ui", {"operation": "execute"}) == "native_ui:execute"


def test_point_receipts_are_bound_to_current_adapter_observation_and_execution():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    adapter = store._owned("session", "browser").adapter
    adapter.point_receipt = {"model": "pinned-point-model", "revision": "weights-digest", "frame_sha256": "frame-digest"}
    observed = store.act("session", {"operation": "observe", "step": step()})
    assert observed["point_receipt"] == adapter.point_receipt
    result = store.act("session", {"operation": "execute", "action_id": observed["selected_action_id"]})
    assert result["point_receipt"] == adapter.point_receipt


def test_independent_verification_rejects_missing_predicates_and_target_placeholders():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    with pytest.raises(ComputerUseError):
        store.act("session", {"operation": "verify", "predicates": []})
    with pytest.raises(ComputerUseError):
        store.act("session", {"operation": "verify", "predicates": step()["expect"]})


def test_ui_mutations_advance_environment_without_fabricated_file_paths_and_replay_does_not():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    task = {"task_id": "audit-fixture", "objective": "Update the draft"}
    observed = store.act("session", {"operation": "observe", "step": step()})
    observed_report = _action_audit_report(task, call_id="read", name="native_ui", arguments={"operation": "observe"}, result=observed)
    _apply_audit_report(task, observed_report)
    assert task["task_state"]["environment"]["version"] == 0
    request = {"operation": "execute", "action_id": observed["selected_action_id"]}
    applied = store.act("session", request)
    report = _action_audit_report(task, call_id="input", name="native_ui", arguments=request, result=applied)
    assert report["ui_effect_applied"] and report["changed_paths"] == []
    _apply_audit_report(task, report)
    assert task["task_state"]["environment"]["version"] == 1
    duplicate = store.act("session", request)
    assert duplicate["status"] == "verified"
    assert duplicate["effect_receipt"]["effect_state"] == "replayed"
    assert _evidence_authority({"name": "native_ui", "result": duplicate}) == "inspection"


def test_explicit_task_epoch_invalidates_old_candidates_without_cancelling_browser():
    from test_portal_native_task_routes import Adapter, _phase

    adapters = []
    def factory(session, _surface):
        adapter = Adapter(session)
        adapters.append(adapter)
        return adapter
    store = ComputerUseStore(None, adapter_factory=factory)
    browser = store._owned("session", "browser")
    old_event = store.begin_task("session")
    step = _phase()["steps"][0]
    offer = store.act("session", {"surface": "desktop", "operation": "observe", "step": step})
    candidate = offer["selected_action_id"]
    before = store.act("session", {"surface": "desktop", "operation": "execute", "action_id": candidate})
    assert before["status"] == "verified"
    new_event = store.begin_task("session")
    assert old_event.is_set() and new_event is not old_event and not new_event.is_set()
    assert browser.cancel.is_set() is False
    rejected = store.act("session", {"surface": "desktop", "operation": "execute", "action_id": candidate})
    assert rejected["status"] == "rejected"
    assert adapters[-1].calls == []
    fresh = store.act("session", {"surface": "desktop", "operation": "observe", "step": step})
    current = store.act("session", {"surface": "desktop", "operation": "execute", "action_id": fresh["selected_action_id"]})
    assert current["status"] == "verified" and len(adapters[-1].calls) == 1
    assert current["action"]["id"] != before["action"]["id"]
    with store.task_scope("session", "desktop", old_event):
        assert store.planning_observation("session")["status"] == "cancelled"
    assert new_event.is_set() is False


def test_task_scope_cannot_rebind_to_another_session_or_surface():
    from test_portal_native_task_routes import Adapter

    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    event = store.begin_task("first")
    with store.task_scope("first", "desktop", event):
        assert store.act("second", {"surface": "desktop", "operation": "observe"})["status"] == "cancelled"
    assert event.is_set()
    assert ("second", "desktop") not in store._sessions
    newer = store.begin_task("first")
    with store.task_scope("first", "desktop", newer):
        assert store.act("first", {"surface": "browser", "operation": "observe"})["status"] == "cancelled"
    assert ("first", "browser") not in store._sessions
    assert newer.is_set()
