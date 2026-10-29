"""Independent effect checks using the captured Egg41 failure as input.

No recorded state is a current desktop observation; this exercises only the
compiler's interpretation, never real input or current task completion.
"""
import copy

from qwen_omni_adapters.computer_use import predicate_satisfied
from qwen_omni_adapters.ui_effects import compile_immediate_transition
from runtime.verify_orchestration_contracts import PANEL, read_json


def test_exact_archived_egg41_ctrl_l_does_not_use_prose_as_literal_effect():
    fixture = read_json(PANEL / "egg41-bad-workflow.json")
    proposal = fixture["workflow"]["steps"][0]
    assert proposal["expect"] == [{"kind": "text", "text": "Workbench file manager surface text"}]
    assert fixture["first_outcome"]["status"] == "unresolved"
    effect = compile_immediate_transition(proposal, fixture["first_outcome"]["before"])
    assert effect.mode == "observation_boundary"
    assert effect.as_dict()["predicates"] == []
    assert effect.as_dict()["task_progress"] is False


def test_focused_actual_egg41_location_field_cannot_satisfy_requested_write():
    fixture = read_json(PANEL / "egg41-bad-workflow.json")
    state = fixture["first_outcome"]["after"]
    node = next(n for n in state["nodes"] if n["role"] == "textbox" and n.get("focused"))
    exact = node["value"] + "/Input/corrections.json"
    proposal = {"id": "independent-write-control", "target": {"node_id": node["id"]},
                "verb": "set_value", "args": {"value": exact},
                "expect": [{"kind": "focused", "node_id": "$target"}]}
    effect = compile_immediate_transition(proposal, state)
    predicates = effect.as_dict()["predicates"]
    assert predicates == [{"kind": "value", "node_id": "$target", "value": exact}]
    predicate = {**predicates[0], "node_id": node["id"]}
    assert predicate_satisfied({"kind": "focused", "node_id": node["id"]}, state)
    assert not predicate_satisfied(predicate, state)
    changed = copy.deepcopy(state)
    target = next(n for n in changed["nodes"] if n["id"] == node["id"])
    target["value"] = exact
    target["text_observation"]["character_count"] = len(exact)
    target["text_observation"]["captured_range"]["end"] = len(exact)
    assert predicate_satisfied(predicate, changed)
    assert effect.as_dict()["task_progress"] is False


def test_focus_or_double_click_cannot_verify_source_acquisition():
    fixture = read_json(PANEL / "egg41-bad-workflow.json")
    focus, open_step = fixture["workflow"]["steps"][1:3]
    focus_effect = compile_immediate_transition(focus)
    open_effect = compile_immediate_transition(open_step)
    assert focus_effect.as_dict()["predicates"] == [{"kind": "focused", "node_id": "$target"}]
    assert open_effect.mode == "observation_boundary"
    assert open_effect.as_dict()["predicates"] == []
    assert focus_effect.as_dict()["task_progress"] is open_effect.as_dict()["task_progress"] is False
