"""Frozen source/oracle checks only; no inference, input or lifecycle calls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from portal import ui_decision

PANEL = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location("atomic_anchor_fixtures_v3", PANEL / "fixture_api.py")
fixtures = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(fixtures)


def test_exact_sources_requests_and_original_decisions_are_retained():
    spec = fixtures.load_frozen_panel()
    assert len(spec["sources"]) == 4
    for source_id, source in spec["sources"].items():
        material = fixtures.model_inputs(source_id)
        bind = json.loads((PANEL / source["bind_file"]).read_bytes())
        expand = json.loads((PANEL / source["expand_file"]).read_bytes())
        assert material["compact"] == bind["arguments"][0]
        assert material["binding"] == bind["arguments"][1]
        assert material["packet"] == expand["arguments"][1]
        assert material["compact"]["working_state"]["source_phase"] == source["phase"]
        assert expand["arguments"][0]["source_phase"] == source["phase"]
        assert "immediate_action" not in bind["arguments"][2]
        Draft202012Validator(material["compact"]["decision_schema"]).validate(expand["arguments"][0])
        for kind in ("proposal", "action"):
            request = material[kind + "_request"]
            raw = (PANEL / source["request_files"][kind]["file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == source["request_files"][kind]["request_body_sha256"]
            assert request["think"] is False
            assert request["options"] == {"temperature": 0, "num_predict": 8192}
            assert "tools" not in request and request["response_modalities"] == ["text"]
        assert "counterfactual_anchor" not in material and "oracle" not in material


def test_counterfactual_anchors_use_actual_source_controls_and_families():
    spec = fixtures.load_frozen_panel()
    for source_id, source in spec["sources"].items():
        controls = {c["ref"]: c for c in fixtures.model_inputs(source_id)["compact"]["controls"]}
        anchor = source["counterfactual_anchor"]
        assert source["observed_anchor_control"] == controls[anchor["target"]]
        assert anchor["family"] in controls[anchor["target"]]["families"]
    assert spec["sources"]["SRC.PHASE05"]["counterfactual_anchor"] == {
        "family": "switch_window", "target": "c1"
    }
    assert spec["sources"]["SRC.PHASE06"]["counterfactual_anchor"] == {
        "family": "switch_window", "target": "c0"
    }


def test_actual_mismatch_cases_and_positive_controls_are_not_relabelled():
    spec = fixtures.load_frozen_panel()
    expected = {"ANCHOR.HOST.010": True, "ANCHOR.HOST.020": True,
                "ANCHOR.HOST.030": False, "ANCHOR.HOST.040": False}
    for case_id, matching in expected.items():
        material = fixtures.host_inputs(case_id)
        anchor = material["proposal"]["immediate_action"]
        action = material["decision"]["action"]
        assert (all(action[k] == v for k, v in anchor.items())) is matching
        row = next(c for c in spec["host_cases"] if c["case_id"] == case_id)
        assert (row["expected"] == "accept_consistent_inert_phase") is matching


def test_distractor_and_operation_controls_are_source_legal_before_anchor():
    for case_id in ("ANCHOR.HOST.050", "ANCHOR.HOST.060"):
        material = fixtures.host_inputs(case_id)
        Draft202012Validator(material["compact"]["decision_schema"]).validate(material["decision"])
        action = material["decision"]["action"]
        anchor = material["proposal"]["immediate_action"]
        assert any(action[k] != v for k, v in anchor.items())


def test_missing_stale_and_explicit_replan_are_distinct_no_input_controls():
    assert "immediate_action" not in fixtures.host_inputs("ANCHOR.HOST.070")["proposal"]
    stale = fixtures.host_inputs("ANCHOR.HOST.090")
    assert stale["compact"]["working_state"]["source_phase"] == 1
    assert stale["decision"]["source_phase"] == 5
    replan = fixtures.host_inputs("ANCHOR.HOST.080")
    assert replan["decision"]["action"]["family"] == "replan"
    assert replan["decision"]["source_phase"] == replan["compact"]["working_state"]["source_phase"]


def test_separate_model_matrix_and_fact_control_keep_original_claim_limits():
    spec = fixtures.load_frozen_panel()
    cells = spec["model_chain_protocol"]["planned_cells"]
    assert len(cells) == 16
    assert all(c["status"] == "unrun" for c in cells)
    assert len({(c["source_id"], c["model"], c["condition"]) for c in cells}) == 16
    fact = spec["fact045_separate_control"]
    assert fact["status"] == "unrun_outcome_test"
    source = json.loads((PANEL / spec["sources"]["SRC.PHASE01"]["bind_file"]).read_bytes())
    assert source["result"][0]["planning_handoff"]["authority"] == (
        "planner_claim_not_effect_or_input_authority"
    )
    assert spec["whole_task_success_credit"] is False
    assert spec["original_panels_and_100_task_scope_unchanged"] is True


@pytest.mark.parametrize("case_id", [f"ANCHOR.HOST.{number:03d}" for number in range(10, 100, 10)])
def test_actual_candidate_boundary_against_preregistered_host_oracle(case_id):
    spec = fixtures.load_frozen_panel()
    case = next(c for c in spec["host_cases"] if c["case_id"] == case_id)
    material = fixtures.host_inputs(case_id)

    def compile_phase():
        _, binding = ui_decision.bind_subgoal(
            material["compact"], material["binding"], material["proposal"], atomic_anchor=True
        )
        return ui_decision.expand(material["decision"], material["packet"], binding)

    if case["expected"] == "reject_without_input":
        with pytest.raises(ui_decision.DecisionError):
            compile_phase()
    elif case["expected"] == "explicit_no_input_replan":
        with pytest.raises(ui_decision.DecisionReplan):
            compile_phase()
    else:
        result = compile_phase()
        assert result["phase"]["kind"] == "ui"
        assert result["phase"]["steps"][0]["verb"] == "switch_window"
        assert result.get("task_progress") is not True
        assert result.get("input_authority") is not True
        assert "effect_receipt" not in result
