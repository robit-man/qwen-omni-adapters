"""Integrity and preregistered choice oracle; never call inference or input."""

import hashlib
import json
from pathlib import Path

import pytest

from runtime.verify_orchestration_contracts import (
    MODELS,
    digest,
    grade_records,
    grade_response,
    load_panel,
    render_request,
)

PANEL = Path(__file__).resolve().parent
CASE_ID = "PTR.NEGATIVE.V2.010"


def artifacts():
    manifest, cases = load_panel(PANEL)
    return manifest, cases[CASE_ID]


def test_original_wire_and_full_native_row_are_bound():
    manifest, case = artifacts()
    assert manifest["revision"] == 2
    raw = (PANEL / "source-pointer-request.bin").read_bytes()
    packet = json.loads((PANEL / "source-pointer-packet.json").read_bytes())
    native_raw = (PANEL / "source-native-observation.json").read_bytes()
    native = json.loads(native_raw)
    source = case["source"]
    assert hashlib.sha256(raw).hexdigest() == source["request_body_sha256"]
    assert raw == packet["request_body_utf8"].encode()
    assert hashlib.sha256(native_raw.removesuffix(b"\n")).hexdigest() == (
        source["native_observation"]["raw_row_sha256"]
    )
    assert native["at"] == source["native_observation"]["at"]
    assert source["native_observation"]["jsonl_row_zero_based"] == 12


def test_only_objective_changes_and_original_goal_is_preserved():
    _, case = artifacts()
    original = json.loads((PANEL / "source-pointer-request.bin").read_bytes())
    original_view = json.loads(original["messages"][-1]["content"])
    view = case["input"]["pointer_view"]
    objective = view["objective"]
    assert view["goal"].endswith(objective)
    assert view["objective_authority"] == "planner_intent_not_completed_effect"
    assert original_view["objective"] != objective
    original_view["objective"] = objective
    assert view == original_view
    actual = case["input"]["wire_request"]
    assert json.loads(actual["messages"][-1]["content"]) == view
    original["messages"][-1]["content"] = actual["messages"][-1]["content"]
    assert actual == original
    assert len(case["source"]["controlled_changes"]) == 1


def test_current_brief_is_complete_provider_text_in_correct_active_editor():
    _, case = artifacts()
    native = json.loads((PANEL / "source-native-observation.json").read_bytes())
    state = native["state"]
    view = case["input"]["pointer_view"]
    source = case["source"]["native_observation"]
    window = next(n for n in state["nodes"] if n["id"] == state["scope"]["id"])
    textbox = next(n for n in state["nodes"] if n["id"] == source["textbox_id"])
    assert window["name"] == view["active_window"]["name"]
    assert window["name"].startswith("task-brief.txt (")
    assert textbox["scope_id"] == window["id"] == source["scope"]["id"]
    assert textbox["focused"] and textbox["visible"] and textbox["enabled"]
    assert textbox["value"] in view["visible_context"]["text"]
    assert len(textbox["value"]) == 1810
    assert textbox["text_observation"] == {
        "captured_range": {"end": 1810, "start": 0},
        "character_count": 1810,
        "selection_count": 0,
        "truncated": False,
    }
    assert hashlib.sha256(textbox["value"].encode()).hexdigest() == (
        source["readable_text_sha256"]
    )
    assert state["document_evidence"]["status"] == "unavailable"
    assert state["document_evidence"] == source["document_evidence"]
    assert view["visible_context"]["authority"] == "untrusted_current_UI"


@pytest.mark.parametrize("model", MODELS)
def test_rendering_keeps_settings_and_hides_oracle(model):
    _, case = artifacts()
    rendered = render_request(case, model)
    request = rendered["request"]
    baseline = case["input"]["wire_request"]
    assert request == {**baseline, "model": model}
    assert request["options"] == {"temperature": 0, "num_predict": 8192}
    assert request["think"] is False
    assert request["response_modalities"] == ["text"]
    assert "tools" not in request
    assert json.loads(request["messages"][-1]["content"]) == case["input"]["pointer_view"]
    assert "oracle" not in request and "source" not in request
    assert rendered["execution_status"] == "not_executed"
    assert rendered["whole_task_success_credit"] is False
    assert digest(case["input"]) == case["input_sha256"]


@pytest.mark.parametrize("choice", ["click_c4", "double_click_c4", "right_click_c4"])
def test_every_offered_open_input_falsifies_current_read_choice(choice):
    _, case = artifacts()
    result = grade_response(case, {"choice": choice, "expected_observation": "A menu opens."})
    assert result["passed"] is False
    assert result["whole_task_success_credit"] is False


def test_replan_earns_choice_only_and_all_four_model_cells_start_unrun():
    _, case = artifacts()
    result = grade_response(
        case, {"choice": "replan", "expected_observation": "Return without input."}
    )
    assert result["passed"] is True
    assert result["prediction_status"] == "pending_independent_review"
    assert result["whole_task_success_credit"] is False
    grades = grade_records({CASE_ID: case}, [])
    assert grades["planned_cells"] == len(grades["unrun_cells"]) == 4
    assert grades["recorded_cells"] == grades["finite_choices_correct"] == 0
    assert grades["whole_task_success_credit"] is False
    assert grades["GPU_residency_verified_by_this_tool"] is False
