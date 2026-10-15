from __future__ import annotations

import copy
import json

import pytest

from runtime import verify_real_calculator
from runtime.verify_real_calculator import (
    POINT_MODEL,
    POINT_REVISION,
    criteria,
    display_nodes,
    method_policy,
    point_receipts,
    policy_receipts,
    selection_method_audit,
)


@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--think"], True)])
def test_calculator_cli_preserves_native_think_default_and_explicit_flag(monkeypatch, flags, expected) -> None:
    captured = []
    monkeypatch.setattr(verify_real_calculator, "run", lambda args: captured.append(args) or 17)
    monkeypatch.setattr("sys.argv", ["verify_real_calculator", *flags])
    assert verify_real_calculator.main() == 17
    assert captured[0].think is expected
    assert captured[0].scenario == "click-arithmetic"
    assert captured[0].require_policy is False


def visual_outcome():
    return {"status": "verified", "action": {"id": "actual-action", "node_id": "visual:one", "generation": "a"*64, "verb": "click"},
            "receipt": {"status": "dispatched", "point_receipt": {"model": POINT_MODEL, "revision": POINT_REVISION,
                "candidate_id": "visual:one", "frame_sha256": "a"*64}}}


def test_point_acceptance_requires_actual_verified_current_model_and_frame_binding() -> None:
    valid = visual_outcome()
    assert len(point_receipts([valid])) == 1
    for field, value in [("model", "vikhyatk/moondream2"), ("revision", "different"),
                         ("candidate_id", "another-action"), ("frame_sha256", "b"*64)]:
        bad = copy.deepcopy(valid)
        bad["receipt"]["point_receipt"][field] = value
        assert point_receipts([bad]) == []
    valid["status"] = "unresolved"
    assert point_receipts([valid]) == []


def test_calculator_button_labels_or_substrings_do_not_count_as_display_evidence() -> None:
    state = {"nodes": [{"role": "button", "name": "42", "visible": True},
                       {"role": "textbox", "value": "142", "visible": True}]}
    assert display_nodes(state, "42") == []
    state["nodes"].append({"role": "textbox", "value": "42", "visible": True})
    assert len(display_nodes(state, "42")) == 1


def test_existing_expected_display_does_not_prove_visual_calculation_effect(tmp_path) -> None:
    item = visual_outcome()
    state = {"scope": {"id": "owned"}, "nodes": [{"role": "textbox", "value": "15", "visible": True}]}
    item.update(before=state, after=state)
    checks = criteria([item], state, expected="15", owned_scope="owned", pointer_path=tmp_path / "missing", visual=True)
    assert checks["actual_moondream31_verified_grounding"]
    assert not checks["grounded_click_produced_expected_display"]
    assert not checks["actual_pointer_movement"]


def test_kev_receipt_must_be_active_ambiguous_confident_and_bound_to_executed_action() -> None:
    outcome = visual_outcome()
    receipt = {"id": "policy", "status": "accepted", "mode": "active", "model_confirmed": True,
               "model": "jaredpalmer/kev-4b@pinned", "probability_kind": "native", "selected_action_id": "actual-action",
               "candidate_ids": ["observe", "abstain", "actual-action", "other-action"],
               "probabilities": {"actual-action": .99}}
    def events(value):
        return [{"tools": [{"result": json.dumps({"policy_receipt": value})}]}]
    assert policy_receipts(events(receipt), [outcome]) == [receipt]
    for key, value in [("mode", "shadow"), ("selected_action_id", "unexecuted"),
                       ("probabilities", {"actual-action": .94}), ("candidate_ids", ["actual-action"])]:
        bad = {**receipt, key: value}
        assert policy_receipts(events(bad), [outcome]) == []


def policy_events():
    outcome = {"status": "verified", "receipt": {"status": "dispatched"}, "action": {
        "id": "selected", "node_id": "real-button", "verb": "click", "args": {},
        "expect": [{"kind": "value", "node_id": "display", "value": "42"}]}}
    step = {"verb": "click", "target": {"role": "button"}, "args": {}, "expect": outcome["action"]["expect"]}
    receipt = {"id": "policy", "status": "accepted", "mode": "active", "model_confirmed": True,
               "model": "jaredpalmer/kev-4b@pinned", "probability_kind": "native", "selected_action_id": "selected",
               "candidate_ids": ["observe", "abstain", "selected", "other"], "probabilities": {"selected": .99}}
    offers = {"selected_action_id": "selected", "candidates": [{"kind": "action"}, {"kind": "action"}],
              "policy_receipt": receipt}
    return outcome, [{"tools": [{"name": "native_ui", "status": "complete", "arguments": {"operation": "observe", "step": step},
                                  "result": offers}]},
                     {"tools": [{"name": "native_ui", "status": "complete", "arguments": {"operation": "execute", "action_id": "selected"},
                                  "result": outcome}]}]


def test_required_policy_method_carries_no_typed_task_values_and_default_is_unforced() -> None:
    ordinary = method_policy(require_policy=False, visual=False)
    required = method_policy(require_policy=True, visual=False)
    assert "target is exactly" not in ordinary and "target is exactly" in required
    assert not any(value in required for value in ["42", "17", "25", "15", "equals", "plus"])
    visual = method_policy(require_policy=False, visual=True)
    assert "spoken referring expression" in visual and "bottom right" not in visual


def test_required_policy_audits_real_ambiguity_returned_id_and_preserved_planner_effects() -> None:
    outcome, events = policy_events()
    assert all(selection_method_audit(events, [outcome], require_policy=True, visual=False).values())
    named = copy.deepcopy(events)
    named[0]["tools"][0]["arguments"]["step"]["target"]["name"] = "Unique button"
    assert not selection_method_audit(named, [outcome], require_policy=True, visual=False)["policy_role_only_real_button_ambiguity"]
    bypass = copy.deepcopy(events)
    bypass[1]["tools"][0]["arguments"]["action_id"] = "manually-chosen"
    assert not selection_method_audit(bypass, [outcome], require_policy=True, visual=False)["policy_returned_id_executed"]
    changed = copy.deepcopy(events)
    changed[0]["tools"][0]["arguments"]["step"]["expect"] = [{"kind": "text", "text": "Unrelated easy effect"}]
    assert not selection_method_audit(changed, [outcome], require_policy=True, visual=False)["policy_preserved_planner_intent"]
    typed = {"status": "verified", "receipt": {"status": "dispatched"}, "action": {"verb": "set_value"}}
    assert not selection_method_audit(events, [typed, outcome], require_policy=True, visual=False)["policy_no_keyboard_or_direct_value_substitution"]


def test_visual_audit_requires_actual_offered_referring_expression_not_semantic_substitution() -> None:
    outcome = visual_outcome()
    offer = {"selected_action_id": "actual-action", "state": {"nodes": [{"id": "native", "name": "Equals"}]}}
    step = {"target": {"role": "button", "name": "the control described by visual appearance"}}
    events = [{"tools": [{"name": "native_ui", "status": "complete", "arguments": {"operation": "observe", "step": step}, "result": offer}]},
              {"tools": [{"name": "native_ui", "status": "complete", "arguments": {"operation": "execute", "action_id": "actual-action"}, "result": outcome}]}]
    assert selection_method_audit(events, [outcome], require_policy=False, visual=True)["spoken_visual_reference_method_used"]
    step["target"]["name"] = "Equals"
    assert not selection_method_audit(events, [outcome], require_policy=False, visual=True)["spoken_visual_reference_method_used"]
