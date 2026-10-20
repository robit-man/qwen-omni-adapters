"""Observed caret insertion derives exact nonempty typing effects without guessing (#4)."""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
from test_native_recipient_freshness import recipient_fixture
from test_ui_effect_contract import observed, step

from portal import ui_decision
from portal.computer_use import ComputerUseStore
from qwen_omni_adapters.computer_use import ComputerUseError, RapidUiController
from qwen_omni_adapters.ui_effects import compile_immediate_transition

RANGE = "typing_insertion_range_not_observed"
LEGACY = "typing_complete_value_not_observed"


def at_caret(value, caret):
    state = observed(value)
    state["nodes"][0]["text_insertion"] = {"caret": caret}
    return state


@pytest.mark.parametrize("value,caret,inserted,expected", [
    ("Existing text", 9, "new ", "Existing new text"),
    ("Existing text", 13, "!", "Existing text!"),
    ("Existing text", 0, ">", ">Existing text"),
    ("café 中 👩🏽‍💻 é", 5, "Ελ", "café Ελ中 👩🏽‍💻 é"),  # Unicode codepoint offsets.
    ("first\nlast", 5, "\n", "first\n\nlast"),
    ("aaa", 1, "a", "aaaa"),
])
def test_observed_caret_derives_exact_nonempty_insertion(value, caret, inserted, expected):
    proposal = step("type_text", {"text": inserted})
    effect = compile_immediate_transition(proposal, at_caret(value, caret))
    assert effect.mode == "verified_step" and effect.reason == "host_operation_effect"
    assert effect.bind(proposal)["expect"] == [{"kind": "value", "node_id": "$target", "value": expected}]


def test_selection_replacement_remains_separate_from_caret_insertion():
    proposal = step("type_text", {"text": "N"})
    effect = compile_immediate_transition(proposal, observed("old", selection={"start": 0, "end": 2}))
    assert effect.bind(proposal)["expect"][0]["value"] == "Nd"


@pytest.mark.parametrize("insertion", [{"caret": True}, {"caret": -1}, {"caret": 99}, {"caret": "1"},
                                       {"caret": 1, "offset": 1}, "1", None, []])
def test_malformed_insertion_facts_never_infer_a_caret(insertion):
    state = observed("old")
    state["nodes"][0]["text_insertion"] = insertion
    effect = compile_immediate_transition(step("type_text", {"text": "x"}), state)
    assert effect.mode == "requires_replan" and effect.reason == RANGE


@pytest.mark.parametrize("fault", ["truncated", "unfocused", "sensitive", "selected"])
def test_incomplete_sensitive_or_selected_values_cannot_use_a_caret(fault):
    state = at_caret("old", 1)
    node = state["nodes"][0]
    if fault == "truncated":
        node["text_observation"].update(character_count=9, truncated=True)
    elif fault == "unfocused":
        node["focused"] = False
    elif fault == "sensitive":
        node["sensitive"] = True
    else:
        node["text_observation"]["selection_count"] = 1
        node["text_selection"] = {"start": 0, "end": 1}
    effect = compile_immediate_transition(step("type_text", {"text": "x"}), state)
    if fault == "selected":  # The observed selection, never the caret, decides replacement.
        assert effect.bind(step("type_text", {"text": "x"}))["expect"][0]["value"] == "xld"
    else:
        assert effect.mode == "requires_replan" and effect.reason == LEGACY


@pytest.mark.parametrize("fault", ["selection", "selection_count", "truncated", "beyond", "extra", "bool"])
def test_controller_snapshot_rejects_contradictory_insertion_facts(fault):
    state = at_caret("old", 1)
    node = state["nodes"][0]
    if fault == "selection":
        node["text_selection"] = {"start": 0, "end": 1}
        node["text_observation"]["selection_count"] = 1
    elif fault == "selection_count":
        node["text_observation"]["selection_count"] = 1
    elif fault == "truncated":
        node["text_observation"].update(character_count=9, truncated=True)
    elif fault == "beyond":
        node["text_insertion"]["caret"] = 4
    elif fault == "extra":
        node["text_insertion"]["end"] = 2
    else:
        node["text_insertion"]["caret"] = True
    with pytest.raises(ComputerUseError):
        RapidUiController(SimpleNamespace(observe=lambda: state, perform=lambda _a: pytest.fail("no input"))).observe()


def test_sensitive_controller_snapshot_drops_insertion():
    state = at_caret("secret", 2)
    state["nodes"][0]["sensitive"] = True
    controller = RapidUiController(SimpleNamespace(observe=lambda: state, perform=lambda _a: None))
    assert "text_insertion" not in controller.observe()["nodes"][0]


def test_caret_only_drift_changes_handoff_target_binding_despite_identical_prediction():
    first, second = at_caret("aaa", 1), at_caret("aaa", 2)
    proposal = step("type_text", {"text": "a"})
    assert (compile_immediate_transition(proposal, first).bind(proposal)["expect"]
            == compile_immediate_transition(proposal, second).bind(proposal)["expect"])
    assert ui_decision._target_binding(first["nodes"][0]) != ui_decision._target_binding(second["nodes"][0])
    legacy = observed("aaa")["nodes"][0]
    assert "text_insertion" not in legacy and ui_decision._target_binding(legacy) == ui_decision._target_binding(
        copy.deepcopy(legacy))


def title(state):
    return next(node for node in state["nodes"] if node["name"] == "Title")


def test_worker_publishes_revision_bound_insertion_only_for_unselected_complete_focus(monkeypatch):
    fixture, _commands, _geometry = recipient_fixture(monkeypatch, value="aaa", caret=1)
    first = fixture.adapter.observe()
    assert title(first)["text_insertion"] == {"caret": 1}
    fixture.title.caret = 2
    second = fixture.adapter.observe()
    assert title(second)["text_insertion"] == {"caret": 2} and second["revision"] != first["revision"]
    fixture.title.selection = (0, 1)
    assert "text_insertion" not in title(fixture.adapter.observe())
    fixture.title.selection = None
    fixture.title.state.values.discard("FOCUSED")
    assert "text_insertion" not in title(fixture.adapter.observe())
    fixture.title.state.values.add("FOCUSED")
    fixture.title.value, fixture.title.caret = "z" * 2100, 0
    assert "text_insertion" not in title(fixture.adapter.observe())
    del fixture.api.Text.get_caret_offset
    fixture.title.value = "aaa"
    assert "text_insertion" not in title(fixture.adapter.observe())


def inserting_fixture(monkeypatch, value, caret):
    fixture, commands, _geometry = recipient_fixture(monkeypatch, value=value, caret=caret)

    def run(argv, **kwargs):
        commands.append(argv)
        assert "shell" not in kwargs
        if argv[1] == "type":
            node = fixture.title
            node.value = node.value[:node.caret] + argv[-1] + node.value[node.caret:]
            node.caret += len(argv[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    return fixture, commands


@pytest.mark.parametrize("caret,inserted,expected", [(6, "brave ", "Hello brave world"), (11, "!", "Hello world!")])
def test_generated_path_compiles_dispatches_and_verifies_real_worker_insertion(monkeypatch, caret, inserted, expected):
    fixture, commands = inserting_fixture(monkeypatch, "Hello world", caret)
    store = ComputerUseStore(None, adapter_factory=lambda *_: fixture.adapter)
    cancel = store.begin_task("session", private_boundaries=True)
    node = title(fixture.adapter.observe())
    with store.task_scope("session", "desktop", cancel):
        result = store.immediate_transition("session", {"id": "insert", "verb": "type_text",
            "target": {"node_id": node["id"]}, "args": {"text": inserted}},
            goal="Edit the owned document", objective="Insert the requested words")
    assert result["immediate_effect_contract"]["mode"] == "verified_step"
    assert result["immediate_effect_contract"]["predicates"] == [
        {"kind": "value", "node_id": "$target", "value": expected}]
    assert result["status"] == "complete" and result["outcomes"][-1]["status"] == "verified", result
    assert fixture.title.value == expected and len(commands) == 1


def handoff(state, node, goal, objective):
    return {"schema": ui_decision.HANDOFF_CONTRACT,
        "authority": "planner_claim_not_effect_or_input_authority", "input_authority": False,
        "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(), "source_epoch": "a" * 32,
        "source_phase": 1, "source_observation": ui_decision._observation_binding(state),
        "target": ui_decision._target_binding(node), "operation_family": "text_entry",
        "objective": objective, "next_operation": "Type one character at the observed insertion point.",
        "needed_evidence": ["The edited value"], "expected_transition": {
            "schema": "robit.omni.native-expected-transition.v1", "mode": "observe_after_input",
            "operation_family": "text_entry", "authority": "host_operation_descriptor_not_observed_effect",
            "input_authority": False, "task_progress": False}}


@pytest.mark.parametrize("drift", [False, True])
def test_captured_handoff_rejects_caret_only_drift_with_identical_prediction(monkeypatch, drift):
    goal, objective = "Edit the owned document", "Insert one character"
    fixture, commands = inserting_fixture(monkeypatch, "aaa", 1)
    store = ComputerUseStore(None, adapter_factory=lambda *_: fixture.adapter)
    cancel = store.begin_task("session", private_boundaries=True)
    state = fixture.adapter.observe()
    claim = handoff(state, title(state), goal, objective)
    if drift:
        fixture.title.caret = 2  # Same predicted "aaaa"; a different recipient position.
    with store.task_scope("session", "desktop", cancel):
        result = store.immediate_transition("session", {"id": "insert", "verb": "type_text",
            "target": {"node_id": title(state)["id"]}, "args": {"text": "a"}},
            goal=goal, objective=objective, action_handoff=claim)
    if drift:
        assert result["error"] == "ui_action_handoff_stale" and result["input_attempted"] is False
        assert commands == [] and fixture.title.value == "aaa"
    else:
        assert result["status"] == "complete" and result["outcomes"][-1]["status"] == "verified", result
        assert fixture.title.value == "aaaa" and fixture.title.caret == 2 and len(commands) == 1


def test_legacy_no_handoff_path_still_acts_on_its_own_fresh_observation(monkeypatch):
    fixture, commands = inserting_fixture(monkeypatch, "Hello", 0)
    node = title(fixture.adapter.observe())
    fixture.title.caret = 5  # The compiler reads this fresh executor observation itself.
    store = ComputerUseStore(None, adapter_factory=lambda *_: fixture.adapter)
    cancel = store.begin_task("session", private_boundaries=True)
    with store.task_scope("session", "desktop", cancel):
        result = store.immediate_transition("session", {"id": "insert", "verb": "type_text",
            "target": {"node_id": node["id"]}, "args": {"text": "!"}},
            goal="Edit the owned document", objective="Append punctuation")
    assert result["immediate_effect_contract"]["predicates"][0]["value"] == "Hello!"
    assert fixture.title.value == "Hello!" and len(commands) == 1
