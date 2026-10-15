from __future__ import annotations

import copy
import json

import pytest

from runtime.verify_native_manipulation import (
    FULL_FLAGS,
    GestureState,
    baseline_plan,
    validate_plan,
)


def test_wrong_continue_cannot_be_covered_by_later_successful_gesture_effects(tmp_path) -> None:
    state = GestureState(tmp_path, "MARKER", FULL_FLAGS)
    state.mark("wrong_continue")
    for name in FULL_FLAGS:
        state.mark(name, {"actual_event": True})
    receipt = json.loads((tmp_path / "gesture-receipt.json").read_text())
    assert receipt["wrong_choice"] is True
    assert receipt["complete"] is False
    assert receipt["events"][0]["event"] == "wrong_continue"


def test_native_gesture_completion_requires_every_independent_application_effect(tmp_path) -> None:
    state = GestureState(tmp_path, "MARKER", FULL_FLAGS)
    for name in FULL_FLAGS[:-1]:
        state.mark(name)
    assert json.loads((tmp_path / "gesture-receipt.json").read_text())["complete"] is False
    state.mark(FULL_FLAGS[-1])
    assert json.loads((tmp_path / "gesture-receipt.json").read_text())["complete"] is True


@pytest.mark.parametrize("change", ["raw_coordinates", "wrong_destination", "missing_effect", "wrong_terminal", "foreign_scope"])
def test_gesture_plan_cannot_expand_the_owned_task_authority(change) -> None:
    state = {"nodes": [{"id": "observed-drop", "name": "Drop target", "role": "panel"}]}
    reference = baseline_plan(state, "MARKER", "Owned fixture", True)
    proposed = copy.deepcopy(reference)
    if change == "raw_coordinates":
        proposed["steps"][0]["args"] = {"x": 10, "y": 20}
    elif change == "wrong_destination":
        proposed["steps"][6]["args"]["to_node_id"] = "unobserved-drop"
    elif change == "missing_effect":
        proposed["steps"][0]["expect"] = [{"kind": "text", "text": "Already visible"}]
    elif change == "foreign_scope":
        proposed["steps"][0]["target"]["scope_id"] = "foreign-window"
    else:
        proposed["final_predicates"] = [{"kind": "text", "text": "Already visible"}]
    with pytest.raises(ValueError):
        validate_plan(proposed, reference)
