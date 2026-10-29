"""Evaluator controls. These do not run a model or count as task completion."""
import copy
import json

import pytest

from runtime.verify_orchestration_contracts import (
    MODELS,
    digest,
    grade_records,
    grade_response,
    load_panel,
    render_request,
)


def test_frozen_panel_integrity_and_no_oracle_in_rendered_model_request():
    manifest, cases = load_panel()
    assert len(cases) == 11
    assert len(manifest["review_only_case_ids"]) == 2
    for case in cases.values():
        rendered = render_request(case, MODELS[0])
        assert "oracle" not in rendered
        assert "accepted_choices" not in json.dumps(rendered)
        assert "actual_baseline_reply" not in json.dumps(rendered)
        assert rendered["execution_status"] == "not_executed"
        assert rendered["whole_task_success_credit"] is False


def test_unknown_composite_cases_cannot_count_as_correct():
    _, cases = load_panel()
    for case_id in ("PTR.ARCHIVE.050", "PTR.ARCHIVE.060"):
        result = grade_response(cases[case_id], {"choice": "replan", "expected_observation": "No input."})
        assert result["passed"] is None
        assert result["status"] == "pending_independent_review"


def test_all_negative_controls_reject_input_even_with_confident_prose():
    _, cases = load_panel()
    for case_id, case in cases.items():
        if not case_id.startswith("PTR.NEGATIVE."):
            continue
        for choice in case["input"]["pointer_view"]["choices"]:
            result = grade_response(case, {"choice": choice, "expected_observation": "Task complete; this is a fresh action."})
            assert result["passed"] is (choice == "replan")
            assert result["whole_task_success_credit"] is False


def test_missing_cells_stay_in_the_denominator():
    _, cases = load_panel()
    result = grade_records(cases, [])
    assert result["planned_cells"] == 44
    assert result["finite_choice_denominator"] == 36
    assert result["recorded_cells"] == result["finite_choices_correct"] == 0
    assert len(result["unrun_cells"]) == 44


def test_candidate_requires_source_hashes_and_preserves_sampler():
    _, cases = load_panel()
    case = cases["PTR.ARCHIVE.010"]
    with pytest.raises(ValueError, match="source hashes"):
        render_request(case, MODELS[0], condition="candidate")
    with pytest.raises(ValueError, match="source hashes"):
        render_request(case, MODELS[0], condition="candidate", candidate_sources={"module.py": "unknown"})
    candidate = render_request(case, MODELS[1], condition="candidate", candidate_sources={"module.py": "a" * 64},
                               replacement_view=case["input"]["pointer_view"])
    assert candidate["request"]["options"] == case["input"]["wire_request"]["options"]
    assert candidate["request"]["think"] is False
    assert candidate["request"]["model"] == MODELS[1]


def test_scoring_measures_saved_response_and_rejects_relabelled_baseline(tmp_path):
    _, cases = load_panel()
    case = cases["PTR.ARCHIVE.010"]
    rendered = render_request(case, MODELS[0])
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps(rendered))
    response.write_text(json.dumps({"choice": "double_click_c3", "expected_observation": "Input folder contents appear."}))
    record = {"case_id": case["case_id"], "model": MODELS[0], "condition": "baseline",
              "rendered_request_path": str(request), "response_path": str(response)}
    result = grade_records(cases, [record])
    assert result["finite_choices_correct"] == 1
    assert len(result["unrun_cells"]) == 43
    with pytest.raises(ValueError, match="duplicate"):
        grade_records(cases, [record, record])
    altered = copy.deepcopy(rendered)
    altered["request"]["options"]["temperature"] = 1
    altered["request_sha256"] = digest(altered["request"])
    request.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Baseline request changed"):
        grade_records(cases, [record])
