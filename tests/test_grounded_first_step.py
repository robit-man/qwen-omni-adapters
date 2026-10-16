"""Private first-position grounding preserves later causal actions and authority."""
from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from portal import ui_task
from qwen_omni_adapters.computer_use import parse_step


def node(identity="current", *, role="window", name="Current", operations=("key_chord",), **changes):
    return {"id": identity, "generation": "1", "scope_id": "scope", "role": role, "name": name,
            "visible": True, "enabled": True, "sensitive": False, "operations": list(operations), **changes}


def observation(*nodes):
    return {"state": {"session_id": "session", "revision": "rev", "scope": {"id": "scope", "kind": "desktop"},
                      "nodes": list(nodes)}}


def step(verb="focus", target=None, args=None):
    value = {"id": "first", "verb": verb, "target": target or {"node_id": "current"},
             "expect": [{"kind": "active_window" if verb == "switch_window" else "focused", "node_id": "$target"}]}
    if args is not None:
        value["args"] = args
    return value


def phase(*steps):
    return {"kind": "ui", "steps": list(steps), "final_predicates": [{"kind": "focused", "node_id": "current"}]}


def boundary(value):
    return {"kind": "ui_boundary", "step": {key: item for key, item in value.items() if key != "expect"}}


def test_real_planning_boundary_rejects_current_window_without_advertised_focus():
    packets, dispatches = [], []
    unsupported = phase(step())

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        return unsupported if len(packets) == 1 else {"kind": "complete", "message": "Claim only"}

    list(ui_task.run_native_goal("Original goal", observe=lambda: observation(node()), compile_phase=compile_phase,
         execute=lambda *args: dispatches.append(args) or {"status": "complete"}, cancel=lambda: None,
         allowed_tools={"native_ui"}))
    assert not Draft202012Validator(packets[0]["response_schema"]).is_valid(unsupported)
    assert dispatches == []


def grounded(current, allowed=("native_ui", "memory_write", "memory_search")):
    return ui_task._grounded_phase_schema(allowed, current)


def test_exact_id_and_semantic_selectors_keep_positive_pairs_without_cross_products():
    current = observation(node("button", role="button", name="Run", operations=("focus",)),
                          node("edit", role="textbox", name="Draft", operations=("type_text",), focused=True))
    validator = Draft202012Validator(grounded(current))
    for target in ({"node_id": "button"}, {"role": "button"}, {"name": "Run"},
                   {"role": "button", "name": "Run", "scope_id": "scope"}):
        assert validator.is_valid(phase(step(target=target)))
    for target in ({"node_id": "edit"}, {"node_id": "invented"}, {"role": "button", "name": "Draft"},
                   {"role": "textbox", "name": "Run"}, {"name": "Run", "scope_id": "elsewhere"},
                   {"node_id": "button", "name": "Invented"}, {"scope_id": "scope"}):
        assert not validator.is_valid(phase(step(target=target)))
    assert validator.is_valid(phase(step("type_text", {"node_id": "edit"}, {"text": "actual"})))
    assert not validator.is_valid(phase(step("type_text", {"node_id": "button"}, {"text": "actual"})))
    assert not validator.is_valid(phase(step("type_text", {"node_id": "edit"}, {"keys": ["Return"]})))


def test_exact_tuple_lengths_and_future_unseen_controls():
    schema = grounded(observation(node(operations=("focus",))))
    validator = Draft202012Validator(schema)
    for size in (1, 25):
        future = [step(target={"node_id": "future-dialog"}) for _ in range(size - 1)]
        assert validator.is_valid(phase(step(), *future))
    assert not validator.is_valid(phase())
    assert not validator.is_valid(phase(*[step() for _ in range(26)]))
    tuples = schema["oneOf"][0]["properties"]["steps"]["anyOf"]
    assert len(tuples) == 25 and all("items" not in item for item in tuples)


def test_inactive_window_supports_only_its_observed_switch_and_current_key_chord():
    validator = Draft202012Validator(grounded(observation(node(), node("inactive", name="Editor", operations=("switch_window",)))))
    assert validator.is_valid(phase(step("switch_window", {"node_id": "inactive"})))
    assert validator.is_valid(phase(step("key_chord", args={"keys": ["Control_L", "l"]})))
    assert not validator.is_valid(phase(step("switch_window")))
    assert not validator.is_valid(phase(step("focus", {"node_id": "inactive"})))


def test_hidden_disabled_sensitive_and_unprojected_ids_never_populate_first_finite_sets():
    current = observation(node("valid", operations=("focus",)),
                          node("hidden", visible=False, operations=("focus",)),
                          node("disabled", enabled=False, operations=("focus",)),
                          node("secret", sensitive=True, operations=("focus",)))
    validator = Draft202012Validator(grounded(current))
    assert validator.is_valid(phase(step(target={"node_id": "valid"})))
    for identity in ("hidden", "disabled", "secret", "outside-task-scope"):
        assert not validator.is_valid(phase(step(target={"node_id": identity})))


def test_only_existing_named_visual_discovery_keeps_no_id_and_no_fabricated_focus():
    validator = Draft202012Validator(grounded(observation()))
    for verb in ("invoke", "click", "double_click", "right_click"):
        assert validator.is_valid(boundary(step(verb, {"name": "Unmapped named control"})))
        assert not validator.is_valid(boundary(step(verb, {"node_id": "invented"})))
        assert not validator.is_valid(boundary(step(verb, {"role": "button"})))
    assert not validator.is_valid(phase(step("focus", {"name": "Unmapped named control"})))
    current = observation(node())
    current["state"]["truncated"] = True
    # Existing missing-name discovery has no global truncated guard. No new
    # promise of absence/protection coverage is manufactured by this schema.
    assert Draft202012Validator(grounded(current)).is_valid(boundary(step("click", {"name": "Unmapped"})))


def test_canonical_and_non_ui_contracts_stay_byte_equivalent_and_schema_does_not_mutate_observation():
    base = ui_task.phase_response_schema({"native_ui", "memory_write", "memory_search"})
    current = observation(node())
    before = copy.deepcopy(current)
    schema = grounded(current, iter(("native_ui", "memory_write", "memory_search")))
    assert current == before
    assert {key: schema["$defs"][key] for key in base["$defs"]} == base["$defs"]
    assert [item for item in schema["oneOf"][1:] if item["properties"]["kind"]["const"] != "ui_boundary"] == base["oneOf"][1:]
    for plan in ({"kind": "memory_write", "topic": "t", "key": "k", "value": "v"},
                 {"kind": "memory_search", "query": "q", "max_results": 1},
                 {"kind": "visual_inspection"}, {"kind": "complete", "message": "Claim"},
                 {"kind": "blocked", "reason": "Cannot proceed"}):
        assert Draft202012Validator(schema).is_valid(plan)


def test_duplicate_semantics_retain_existing_finite_policy_choice_and_two_internal_inputs():
    from test_ui_task import Fixture
    from test_ui_task import phase as real_phase
    fixture = Fixture([real_phase(), {"kind": "complete", "message": "Claim"}], ambiguous=True)
    class Policy:
        def decide(self, state, actions, *, context):
            fixture.policy_calls.append(copy.deepcopy((state, actions, context)))
            return {"choice": next(action["id"] for action in actions if action.get("node_id") == "draft"),
                    "receipt": {"status": "accepted", "model": "fast-policy"}}

    fixture.store.provider = Policy()
    results = list(fixture.run())
    assert results[-1]["response"]["portal"]["native_task"]["status"] == "planner_claim"
    assert len(fixture.adapter.calls) == 2 and len(fixture.packets) == 2
    assert len(fixture.policy_calls) == 1


def test_quotes_backslashes_and_unicode_targets_are_preserved():
    name = 'café "quoted" \\ path 雪'
    validator = Draft202012Validator(grounded(observation(node("λ-id", name=name, operations=("focus",)))))
    assert validator.is_valid(phase(step(target={"name": name})))
    assert validator.is_valid(phase(step(target={"node_id": "λ-id"})))
    assert not validator.is_valid(phase(step(target={"name": name + "x"})))


def test_schema_is_rederived_after_each_observation_and_invalid_first_plan_reaches_no_executor():
    observed = iter([observation(node(operations=())), observation(node(operations=("focus",)))])
    packets, executions = [], []
    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        return phase(step())
    list(ui_task.run_native_goal("unchanged exact goal", observe=lambda: next(observed), compile_phase=compile_phase,
         execute=lambda *args: executions.append(args) or {"status": "complete"}, cancel=lambda: None,
         allowed_tools={"native_ui"}, max_phases=2))
    assert len(executions) == 1
    assert not Draft202012Validator(packets[0]["response_schema"]).is_valid(phase(step()))
    assert Draft202012Validator(packets[1]["response_schema"]).is_valid(phase(step()))


def test_unknown_malformed_or_oversized_grounding_withholds_ui_but_preserves_non_ui_and_scope():
    huge = observation(*[node(str(i) + "x" * 500, role="r" + str(i), name="n" + str(i) + "x" * 500,
                              operations=("focus", "type_text", "set_value", "set_checked", "select", "scroll", "key_chord", "drag_drop", "select_text"))
                         for i in range(80)])
    malformed = observation(node(), node())
    for current in ({}, malformed, huge):
        schema = grounded(current)
        assert all(branch["properties"]["kind"]["const"] != "ui" for branch in schema["oneOf"])
        assert "grounded_first_step" not in schema["$defs"]
        assert Draft202012Validator(schema).is_valid({"kind": "visual_inspection"})
        assert Draft202012Validator(schema).is_valid({"kind": "memory_search", "query": "q", "max_results": 1})
    no_memory = grounded({}, ("native_ui",))
    assert not Draft202012Validator(no_memory).is_valid({"kind": "memory_search", "query": "q", "max_results": 1})


@pytest.mark.parametrize("verb,args", [("invoke", None), ("focus", None), ("click", None),
    ("double_click", None), ("right_click", None), ("switch_window", None),
    ("set_value", {"value": ""}), ("set_checked", {"checked": True}), ("select", {"option": ""}),
    ("scroll", {"direction": "down"}), ("key_chord", {"keys": ["Control_L", "a"]}),
    ("drag_drop", {"to_node_id": "future-drop"}), ("select_text", {"start": 0, "end": 1}),
    ("type_text", {"text": 'quote " \\ café'})])
def test_all_first_verb_argument_branches_roundtrip_through_canonical_host_parser(verb, args):
    validator = Draft202012Validator(grounded(observation(node(name="", operations=(verb,), focused=True))))
    for target in ({"node_id": "current"}, {"role": "window"}, {"role": "window", "name": ""}):
        value = step(verb, target, args)
        assert parse_step(value)["verb"] == verb
        assert Draft202012Validator(ui_task.phase_response_schema({"native_ui"})).is_valid(phase(value))
        if verb in {"invoke", "click", "double_click", "right_click"}:
            assert not validator.is_valid(phase(value))
            assert validator.is_valid(boundary(value))
        else:
            assert validator.is_valid(phase(value))
    assert not validator.is_valid(phase(step(verb, {"node_id": "invented"}, args)))


@pytest.mark.parametrize("stream", [False, True])
def test_sync_and_stream_send_derived_schema_and_host_rejects_decoder_bypass(tmp_path, monkeypatch, stream):
    import json

    import httpx
    from test_portal_native_task_routes import (
        DEFAULT_MODEL,
        GOAL,
        SESSION_COOKIE_NAME,
        _config,
        _install_store,
        _phase,
        _post,
        _request,
        create_app,
    )

    _, adapters = _install_store(monkeypatch)
    invalid = phase(step("focus", {"node_id": "other-window"}))
    Draft202012Validator(ui_task.phase_response_schema(["native_ui"])).validate(invalid)
    packets = []

    def upstream(request):
        payload = json.loads(request.content)
        packet = json.loads(payload["messages"][1]["content"])
        packets.append(packet)
        assert packet["goal"] == GOAL
        assert packet["response_schema"] == payload["response_format"]["json_schema"]["schema"]
        validator = Draft202012Validator(packet["response_schema"])
        assert not validator.is_valid(invalid)
        future = _phase()
        future["steps"].append(step("focus", {"node_id": "future-dialog"}))
        assert validator.is_valid(future)
        if len(packets) == 2:
            assert packet["last_result"]["error"] == "invalid_native_phase"
        return httpx.Response(200, json={"model": DEFAULT_MODEL, "message": {"role": "assistant",
            "content": json.dumps(invalid if len(packets) == 1 else {"kind": "complete", "message": "Claim"})}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "grounded-phase-owned-session-12345")
    result, _ = _post(client, _request(stream), stream)
    assert len(packets) == 2 and adapters[0].calls == []
    assert result["portal"]["native_task"]["status"] == "planner_claim"
