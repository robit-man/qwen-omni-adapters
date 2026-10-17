from __future__ import annotations

import copy
import json

import httpx
import pytest

from portal.computer_use import ComputerUseStore, SystemOneUiProvider
from qwen_omni_adapters.computer_use import ComputerUseError


class Adapter:
    def __init__(self, session_id):
        self.session_id = session_id
        self.value = ""
        self.calls = 0

    def observe(self):
        return {"session_id": self.session_id, "revision": self.value or "empty",
                "scope": {"id": "form", "kind": "browser"}, "text": "Title",
                "nodes": [{"id": "title", "generation": "document", "scope_id": "form",
                           "role": "textbox", "name": "Title", "visible": True,
                           "enabled": True, "sensitive": False,
                           "operations": ["set_value"], "value": self.value}]}

    def perform(self, action):
        self.calls += 1
        self.value = action["args"]["value"]
        return {"status": "dispatched"}


def _step():
    return {"id": "title", "verb": "set_value", "target": {"role": "textbox", "name": "Title"},
            "args": {"value": "Verified"}, "expect": [{"kind": "value", "node_id": "$target", "value": "Verified"}]}


def test_tool_store_isolates_candidates_and_trash_invalidates_authority():
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: Adapter(session))
    observed = store.act("first", {"operation": "observe", "step": _step()})
    action = next(item for item in observed["candidates"] if item["kind"] == "action")
    request = {"operation": "execute", "action_id": action["id"]}
    assert store.act("second", request)["status"] == "rejected"
    assert store.act("first", request)["status"] == "verified"
    assert store.act("first", request)["status"] == "verified"
    store.clear("first")
    assert store.act("first", request)["status"] == "rejected"


def test_unoffered_and_abstain_actions_report_no_dispatch_as_failure():
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: Adapter(session))
    unknown = store.act("session", {"operation": "execute", "action_id": "unoffered"})
    assert unknown["error"] == "ui_action_rejected"
    assert unknown["dispatch_status"] == "not_dispatched"
    step = _step()
    step["target"] = {"node_id": "unobserved"}
    observed = store.act("session", {"operation": "observe", "step": step})
    assert observed["selected_kind"] == "abstain"
    assert observed["dispatch_status"] == "not_dispatched"
    assert observed["executable_candidate_count"] == 0
    result = store.act("session", {"operation": "execute", "action_id": observed["selected_action_id"]})
    assert result["status"] == "abstained"
    assert result["error"] == "ui_action_abstained"
    assert result["dispatch_status"] == "not_dispatched"
    assert store._owned("session", "browser").adapter.calls == 0


def test_unique_offer_is_explicitly_pending_until_real_execution():
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: Adapter(session))
    observed = store.act("session", {"operation": "observe", "step": _step()})
    assert observed["selected_kind"] == "action"
    assert observed["dispatch_status"] == "not_dispatched"
    assert observed["executable_candidate_count"] == 1
    result = store.act("session", {"operation": "execute", "action_id": observed["selected_action_id"]})
    assert result["status"] == "verified"
    assert result["dispatch_status"] == "dispatched"
    assert "error" not in result


def test_tool_workflow_has_independent_terminal_receipt_and_no_reasoner_loop():
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: Adapter(session))
    result = store.act("session", {"operation": "workflow", "steps": [_step()],
                                  "final_predicates": [{"kind": "value", "node_id": "title", "value": "Verified"}]})
    assert result["status"] == "complete"
    assert result["terminal_receipt"]["schema"] == "robit.omni.ui-verification.v1"
    assert store._owned("session", "browser").adapter.calls == 1


def test_native_workflow_refresh_records_same_step_before_selecting_following_step():
    class ChangingAdapter(Adapter):
        def __init__(self, session_id):
            super().__init__(session_id)
            self.reads = 0

        def observe(self):
            self.reads += 1
            state = super().observe()
            if self.reads >= 2:
                state["revision"] += "-background-change"
            return state

    store = ComputerUseStore(None, adapter_factory=lambda session, surface: ChangingAdapter(session))
    owned = store._owned("session", "desktop")
    owned.controller.enable_native_task_protocol()
    second = {**_step(), "id": "second", "args": {"value": "Second"},
              "expect": [{"kind": "value", "node_id": "$target", "value": "Second"}]}
    result = store.act("session", {"surface": "desktop", "operation": "workflow", "steps": [_step(), second],
        "final_predicates": [{"kind": "value", "node_id": "title", "value": "Second"}]})
    assert result["status"] == "complete"
    assert owned.adapter.calls == 2
    assert [item["step_index"] for item in result["selection_receipts"]] == [0, 0, 1]
    assert "pre_input_refresh" in result["outcomes"][0]


def test_cancel_requires_fresh_observation_and_disallows_argument_overrides():
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: Adapter(session))
    observed = store.act("session", {"operation": "observe", "step": _step()})
    action = next(item for item in observed["candidates"] if item["kind"] == "action")
    assert store.act("session", {"operation": "cancel"})["status"] == "cancelled"
    assert store.act("session", {"operation": "execute", "action_id": action["id"]})["status"] == "cancelled"
    store.act("session", {"operation": "observe"})
    assert store.act("session", {"operation": "execute", "action_id": action["id"]})["status"] == "rejected"
    with pytest.raises(ComputerUseError):
        store.act("session", {"operation": "execute", "action_id": action["id"], "args": {"value": "Override"}})


@pytest.mark.parametrize("mode,invalid", [("shadow", False), ("active", False), ("active", True)])
def test_model_policy_is_bounded_and_shadow_does_not_authorize(mode, invalid):
    actions = [{"id": "one", "kind": "action", "description": "Set intended title"},
               {"id": "other", "kind": "action", "description": "Set distractor"},
               {"id": "observe", "kind": "observe", "description": "observe"},
               {"id": "abstain", "kind": "abstain", "description": "abstain"}]

    def respond(request):
        answer = {"choice": "forged" if invalid else "one",
                  "probabilities": {"one": 0.97, "other": 0.01, "observe": 0.01, "abstain": 0.01}}
        return httpx.Response(200, json={"answers": {"action": answer}})

    provider = SystemOneUiProvider("http://127.0.0.1:8930", mode=mode,
                                  client=httpx.Client(transport=httpx.MockTransport(respond)))
    state = Adapter("session").observe()
    chosen = provider.decide(state, copy.deepcopy(actions))["choice"]
    assert chosen == ("one" if mode == "active" and not invalid else "abstain")


def _calculator_policy_inputs():
    scope_id = "atspi:" + "a" * 32
    nodes = [{"id": f"atspi:{index:032x}", "generation": "1", "scope_id": scope_id,
              "role": "button", "name": f"Control {index}", "visible": True,
              "enabled": True, "sensitive": False, "focused": False,
              "operations": ["invoke", "focus", "click", "double_click", "drag_drop",
                             "right_click", "scroll"],
              "bounds": {"x": index * 10, "y": 20, "width": 64, "height": 44},
              "description": "Observed application control"} for index in range(27)]
    nodes[0].update(value="Observed value", checked=False, selected=["First"],
                    text_selection={"start": 0, "end": 5}, observation_hint="Keep this field")
    step = {"id": "find-controls", "verb": "focus", "target": {"role": "button"},
            "args": {}, "expect": [{"kind": "focused", "node_id": "$target"}]}
    context = {"goal": "Focus the requested application control.",
               "goal_source": "current_user_message", "step": step}
    state = {"session_id": "session", "scope": {"id": scope_id, "kind": "desktop",
             "provider": "atspi2", "window_id": "65011716"}, "revision": "b" * 64,
             "nodes": nodes, "text": "Current application controls", "truncated": False}
    actions = [{"id": kind, "kind": kind, "description": kind}
               for kind in ("observe", "abstain")]
    actions.extend({"id": f"action-{index}", "kind": "action", "node_id": node["id"],
                    "generation": node["generation"], "description": f"focus {node['name']}",
                    "verb": "focus", "args": {}, "expect": copy.deepcopy(step["expect"])}
                   for index, node in enumerate(nodes))
    return state, actions, context


def _capturing_policy(requests):
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        criteria = payload["questions"]["action"]["criteria"]
        return httpx.Response(200, json={"model": payload["model"], "answers": {"action": {
            "choice": "action-0", "probabilities": {
                action_id: 1.0 if action_id == "action-0" else 0.0 for action_id in criteria}}}})

    return SystemOneUiProvider("http://127.0.0.1:8950", mode="active",
                               client=httpx.Client(transport=httpx.MockTransport(respond)))


def _decode_policy_projection(payload, source_state, actions):
    packed = json.loads(payload["state"])
    assert packed["schema"] == "robit.omni.ui-policy-projection.v2"
    relevant = {action.get("node_id") for action in actions}
    bindings = {f"n{index}": node["id"] for index, node in enumerate(
        node for node in source_state["nodes"]
        if node["id"] in relevant and not node.get("sensitive"))}
    assert set(packed["nodes"]) == set(bindings)
    nodes = [{"id": bindings[alias], **packed["node_defaults"], **node}
             for alias, node in packed["nodes"].items()]
    contracts = {}
    for action_id, raw in payload["questions"]["action"]["criteria"].items():
        contract = json.loads(raw)
        if "node_ref" in contract:
            contract = {**packed["action_defaults"], **contract}
            contract["node_id"] = bindings[contract.pop("node_ref")]
        contracts[action_id] = contract
    return packed, nodes, contracts


def test_policy_projects_real_shaped_candidates_without_losing_semantics_or_authority():
    state, actions, context = _calculator_policy_inputs()
    original = copy.deepcopy((state, actions, context))
    old_compact = {"task": context, "scope": state["scope"], "revision": state["revision"],
                   "nodes": state["nodes"]}
    assert len(json.dumps(old_compact, ensure_ascii=False)) > 10000
    requests = []
    decision = _capturing_policy(requests).decide(state, actions, context=context)
    assert decision["choice"] == "action-0"
    assert decision["receipt"]["status"] == "accepted"
    assert len(requests) == 1
    payload = requests[0]
    assert len(payload["state"]) < 10000
    packed, nodes, contracts = _decode_policy_projection(payload, state, actions)
    assert packed["task"] == context
    assert packed["scope"] == {key: value for key, value in state["scope"].items()
                               if key not in {"id", "window_id"}}
    assert "revision" not in packed
    assert decision["receipt"]["state_revision"] == state["revision"]
    assert nodes == [
        {key: value for key, value in node.items()
         if key not in {"scope_id", "operations", "sensitive"}} for node in state["nodes"]]
    criteria = payload["questions"]["action"]["criteria"]
    assert list(criteria) == [action["id"] for action in actions]
    for action in actions:
        assert contracts[action["id"]] == {
            key: action[key] for key in
            ("kind", "description", "node_id", "generation", "verb", "args", "expect")
            if key in action}
    assert (state, actions, context) == original


def test_policy_projection_preserves_sensitive_and_relevance_filters():
    state, actions, context = _calculator_policy_inputs()
    state["nodes"][1].update(sensitive=True, name="Protected content", value="Protected value")
    state["nodes"].append({**copy.deepcopy(state["nodes"][0]), "id": "unrelated"})
    original = copy.deepcopy((state, actions, context))
    requests = []
    _capturing_policy(requests).decide(state, actions, context=context)
    _, nodes, _ = _decode_policy_projection(requests[0], state, actions)
    ids = [node["id"] for node in nodes]
    assert len(ids) == 26
    assert state["nodes"][1]["id"] not in ids
    assert "unrelated" not in ids
    assert "Protected content" not in requests[0]["state"]
    assert "Protected value" not in requests[0]["state"]
    assert (state, actions, context) == original


def test_policy_projection_still_rejects_oversized_semantic_state_before_http():
    state, actions, context = _calculator_policy_inputs()
    state["nodes"][0]["value"] = "x" * 10001
    original = copy.deepcopy((state, actions, context))
    requests = []
    decision = _capturing_policy(requests).decide(state, actions, context=context)
    assert decision["choice"] == "abstain"
    assert decision["receipt"]["status"] == "rejected"
    assert decision["receipt"]["reason"] == "ValueError"
    assert requests == []
    assert (state, actions, context) == original


def test_policy_defaults_factor_only_json_identical_fields_and_decode_exact_contracts():
    state, actions, context = _calculator_policy_inputs()
    for node in state["nodes"]:
        node["checked"] = False
    state["nodes"][1].update(generation="different-generation", checked=0,
                             name=state["nodes"][0]["name"])
    state["nodes"][0]["checked"] = False
    actions[3].update(generation="different-generation", verb="set_value",
                      args={"value": "Exact replacement"},
                      expect=[{"kind": "value", "node_id": "$target",
                               "value": "Exact replacement"}])
    original = copy.deepcopy((state, actions, context))
    requests = []
    _capturing_policy(requests).decide(state, actions, context=context)
    packed, nodes, contracts = _decode_policy_projection(requests[0], state, actions)
    assert set(packed["action_defaults"]) == {"kind"}
    assert "generation" not in packed["node_defaults"]
    assert "checked" not in packed["node_defaults"]
    assert nodes[0]["checked"] is False
    assert type(nodes[1]["checked"]) is int
    assert nodes[0]["name"] == nodes[1]["name"]
    assert nodes[0]["id"] != nodes[1]["id"]
    for action in actions:
        assert contracts[action["id"]] == {
            key: action[key] for key in
            ("kind", "description", "node_id", "generation", "verb", "args", "expect")
            if key in action}
    assert (state, actions, context) == original


@pytest.mark.parametrize("wrong_id", ["n0", "stale-action-id"])
def test_policy_aliases_and_stale_ids_cannot_replace_offered_action_ids(wrong_id):
    state, actions, context = _calculator_policy_inputs()

    def respond(request):
        payload = json.loads(request.content)
        criteria = payload["questions"]["action"]["criteria"]
        assert wrong_id not in criteria
        return httpx.Response(200, json={"model": payload["model"], "answers": {"action": {
            "choice": wrong_id, "probabilities": {key: 1.0 if key == "action-0" else 0.0
                                                  for key in criteria}}}})

    provider = SystemOneUiProvider("http://127.0.0.1:8950", mode="active",
                                  client=httpx.Client(transport=httpx.MockTransport(respond)))
    decision = provider.decide(state, actions, context=context)
    assert decision["choice"] == "abstain"
    assert decision["receipt"]["status"] == "rejected"
    assert decision["receipt"]["reason"] == "ValueError"


def test_policy_projection_does_not_change_executor_revision_or_identity_checks():
    class DuplicateAdapter(Adapter):
        def observe(self):
            state = super().observe()
            state["nodes"].append({**copy.deepcopy(state["nodes"][0]), "id": "other-title"})
            return state

    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        criteria = payload["questions"]["action"]["criteria"]
        chosen = next(key for key, raw in criteria.items() if "node_ref" in json.loads(raw))
        return httpx.Response(200, json={"model": payload["model"], "answers": {"action": {
            "choice": chosen, "probabilities": {key: 1.0 if key == chosen else 0.0
                                                for key in criteria}}}})

    provider = SystemOneUiProvider("http://127.0.0.1:8950", mode="active",
                                  client=httpx.Client(transport=httpx.MockTransport(respond)))
    store = ComputerUseStore(None, provider=provider,
                             adapter_factory=lambda session, surface: DuplicateAdapter(session))
    observed = store.act("session", {"operation": "observe", "step": _step()}, goal="Set the title")
    assert observed["selection_status"] == "accepted"
    assert len(requests) == 1
    adapter = store._owned("session", "browser").adapter
    assert store.act("session", {"operation": "execute", "action_id": "n0"})["status"] == "rejected"
    adapter.value = "External change"
    result = store.act("session", {"operation": "execute",
                                   "action_id": observed["selected_action_id"]})
    assert result["status"] == "rejected"
    assert adapter.calls == 0
    assert adapter.value == "External change"
