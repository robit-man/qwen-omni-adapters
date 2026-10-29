"""Freeze scope and independent whole-task oracles; no live execution here."""
import hashlib
import json
from pathlib import Path

from runtime.verify_orchestration_contracts import PANEL, ROOT


def contract():
    return json.loads((PANEL / "whole-task-acceptance.json").read_text())


def test_primary_and_remote_denominators_preserve_all_original_cases():
    value = contract()
    cases = json.loads((ROOT / value["source_refs"]["tasks"]["path"]).read_text())["tasks"]
    expected_ids = {case["case_id"] for case in cases}
    matrix = value["breadth_matrix"]
    assert len(expected_ids) == 100
    assert matrix["local_primary_first_attempt_cells"] == 200
    assert matrix["remote_separate_extension_cells"] == 200
    assert len(matrix["cells"]) == matrix["total_tracked_configuration_cells"] == 400
    configurations = {item["configuration_id"] for item in value["configurations"]}
    assert configurations == {"local-ornith", "local-qwen", "egg31-ornith", "egg41-ornith"}
    assert {(cell["configuration_id"], cell["case_id"]) for cell in matrix["cells"]} == {
        (configuration, case_id) for configuration in configurations for case_id in expected_ids}
    assert all(cell["status"] == "unrun" and cell["attempt_refs"] == []
               and set(cell["criterion_verdicts"].values()) == {"unrun"} for cell in matrix["cells"])
    assert value["whole_requested_outcome_verified"] is False


def test_three_initial_families_keep_exact_goals_outputs_and_every_original_criterion():
    value = contract()
    original = {case["case_id"]: case for case in json.loads(
        (ROOT / value["source_refs"]["tasks"]["path"]).read_text())["tasks"]}
    selected = value["initial_task_contracts"]
    assert [case["case_id"] for case in selected] == ["LHN-047", "LHN-001", "LHN-053"]
    assert len({case["family_id"] for case in selected}) == 3
    for case in selected:
        source = original[case["case_id"]]
        assert case["task_signature_sha256"] == source["task_signature_sha256"]
        assert case["goal_text_sha256"] == hashlib.sha256(source["goal_text"].encode()).hexdigest()
        assert case["required_outputs"] == source["required_outputs"]
        assert case["goal_mutation_allowed"] is False
        assert {item["criterion_id"] for item in case["criteria"]} == {
            item["id"] for item in source["mandatory_criteria"]}
        for criterion in case["criteria"]:
            assert criterion["mandatory"] is True and criterion["verdict"] == "unrun"
            assert criterion["evidence_refs"] == []
            assert all(criterion[key] for key in ("oracle", "implementation", "evidence_types", "falsification"))
        assert not source["controlled_perturbation"]
        assert not source["fixture"]["additional_asset_requirements"]


def test_frozen_source_hashes_and_resource_uncertainties_are_explicit():
    value = contract()
    for name in ("tasks", "protocol", "work_order"):
        ref = value["source_refs"][name]
        assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert value["budgets"]["native_input_ceiling"] == 500
    assert value["budgets"]["native_input_ceiling_enforced_in_current_runner"] is False
    assert value["budgets"]["current_local_reservation_is_verified_now"] is False
    assert value["rollback_release_gates"]["rollback_proven"].startswith("unproven")
    assert value["must_not_be_given_to_solver"] is True


def test_missing_materializers_and_automatic_oracle_counts_are_not_hidden():
    value = contract()
    gaps = value["fixture_and_oracle_gaps"]
    assert len(gaps["media_cases"]) == 6
    assert len(gaps["controlled_event_cases"]) == 8
    assert gaps["distinct_missing_materializer_case_count"] == 14
    assert gaps["declared_automatic_criteria"] + gaps["pending_independent_criteria"] == 960
    assert sum(f["declared_automatic_oracles"] for f in value["breadth_matrix"]["families"]) == 87
    assert sum(f["pending_independent_oracles"] for f in value["breadth_matrix"]["families"]) == 873


def test_whole_contract_artifacts_have_independent_frozen_hashes():
    manifest = json.loads((PANEL / "whole-task-acceptance.manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert Path(name).name == name
        assert hashlib.sha256((PANEL / name).read_bytes()).hexdigest() == expected
