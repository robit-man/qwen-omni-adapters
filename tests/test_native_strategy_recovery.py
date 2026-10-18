"""Stable strategy recovery uses execution facts, never changed planner wording."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from portal.ui_recovery import NativeStrategyLedger, strategy_identity
from qwen_omni_adapters.computer_use import ComputerUseError
from qwen_omni_adapters.ui_effects import compile_immediate_transition


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state():
    return {"session_id": "owned-session", "revision": "r1", "truncated": False,
        "scope": {"kind": "native", "id": "window", "window_id": "17", "provider": "atspi2"}, "text": "Unrelated background",
        "nodes": [{"id": "open", "generation": "g1", "scope_id": "window", "role": "button", "name": "Open",
            "visible": True, "enabled": True, "sensitive": False, "focused": False,
            "operations": ["click", "double_click", "invoke", "focus"],
            "bounds": {"x": 30, "y": 40, "width": 50, "height": 20}},
            {"id": "draft", "generation": "g2", "scope_id": "window", "role": "textbox", "name": "Draft",
             "visible": True, "enabled": True, "sensitive": False, "focused": True,
             "operations": ["set_value", "type_text", "focus"], "value": "old"}]}


def action(verb="click", target="open", args=None):
    return {"id": "original-step", "verb": verb, "target": {"node_id": target}, "args": args or {}}


def outcome(*, status="unresolved", effect="unknown", attempted=True, verb="click", target="open", args=None):
    before = state()
    node = next(node for node in before["nodes"] if node["id"] == target)
    instruction = action(verb, target, args)
    offered = {key: value for key, value in instruction.items() if key != "target"}
    offered.update(kind="action", node_id=node["id"], generation=node["generation"],
                   session_id=before["session_id"], scope_id=before["scope"]["id"],
                   revision=before["revision"], idempotency_key="external-operation",
                   expect=[{"kind": "visible", "name": "Open", "visible": True}])
    after = copy.deepcopy(before)
    result = {"status": status, "action": offered, "before": before, "after": after,
        "receipt": {"status": "dispatched" if attempted else "rejected", "input_attempted": attempted},
        "effect_receipt": {"schema": "robit.omni.ui-effect-receipt.v1", "effect_state": effect,
            "action_id": offered["id"], "idempotency_key": offered["idempotency_key"],
            "before_revision": before["revision"], "after_revision": after["revision"],
            "before_semantic_revision": before["revision"], "after_semantic_revision": after["revision"],
            "predicates_sha256": digest(offered["expect"])}}
    return result


def test_changed_wording_ids_or_global_revision_do_not_reset_an_uncertain_input():
    ledger = NativeStrategyLedger()
    result = outcome()
    ledger.record(result, source_phase=1)
    current = state()
    current.update(revision="background-r100", text="Unrelated updated document text")
    proposal = {**action(), "id": "a fresh attempt", "objective": "Try a better new approach",
                "assessment": "The previous approach was entirely different"}
    # Claims are excluded from stable identity, never interpreted or authorized.
    assessment = ledger.assess(current, proposal)
    assert assessment["classification"] == "input_effect_uncertain" and assessment["blocked"]
    assert assessment["repeat_status"] == "uncertain_input_requires_reconciliation"
    assert assessment["input_authority"] is assessment["task_progress"] is False


@pytest.mark.parametrize("change", ["geometry", "focus", "value", "ancestry", "resource"])
def test_relevant_changes_allow_new_reasoning_but_never_replay_unknown_input(change):
    ledger = NativeStrategyLedger()
    ledger.record(outcome())
    current = state()
    if change == "geometry":
        current["nodes"][0]["bounds"]["x"] += 25
    elif change == "focus":
        current["nodes"][0]["focused"] = True
    elif change == "value":
        current["nodes"][0]["value"] = "different"
    elif change == "ancestry":
        current["nodes"][0]["structure"] = {"parent_id": "chooser", "ancestors": []}
    else:
        current["scope"]["url"] = "https://example.test/other"
    assessment = ledger.assess(current, action())
    assert assessment["relevant_state_changed"] is True
    assert assessment["blocked"] is True
    assert assessment["repeat_status"] == "uncertain_input_requires_reconciliation"


def test_proven_unchanged_verification_blocks_only_that_effective_strategy_in_same_state():
    ledger = NativeStrategyLedger()
    ledger.record(outcome(status="verified", effect="unchanged"))
    same = ledger.assess(state(), action())
    assert same["classification"] == "proven_no_effect" and same["blocked"]
    changed = state()
    changed["nodes"][0]["focused"] = True
    assessment = ledger.assess(changed, action())
    assert assessment["relevant_state_changed"] and not assessment["blocked"]
    assert assessment["repeat_status"] == "relevant_state_changed_replan"
    assert not ledger.assess(state(), action("double_click"))["blocked"]
    assert not ledger.assess(state(), action("focus", "draft"))["blocked"]


def test_a_dispatched_unchanged_snapshot_without_verification_remains_uncertain():
    ledger = NativeStrategyLedger()
    ledger.record(outcome(status="observed_unverified", effect="unchanged"))
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"


def test_capability_order_and_effect_wording_do_not_make_a_failed_strategy_new():
    ledger = NativeStrategyLedger()
    ledger.record(outcome(status="verified", effect="unchanged"))
    current = state()
    current["nodes"][0]["operations"].reverse()
    proposal = {**action(), "expect": [{"kind": "text", "text": "A fresh explanation"}]}
    assert ledger.assess(current, proposal)["blocked"]


def test_mismatched_effect_receipt_cannot_downgrade_uncertainty_to_proven_no_effect():
    result = outcome(status="verified", effect="unchanged")
    result["effect_receipt"]["action_id"] = "a different input"
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"


def test_canonical_workflow_abstention_is_a_policy_deferral_without_executed_strategy():
    ledger = NativeStrategyLedger()
    ledger.record({"outcomes": [{"status": "abstained", "action": {"kind": "abstain", "id": "abstain"}}]})
    assert ledger.view()["facts"][0]["classification"] == "policy_deferred_no_input"
    assert not ledger.assess(state(), action())["blocked"]


def test_contradictory_no_input_wrapper_cannot_erase_a_lost_acknowledged_dispatch():
    result = outcome()
    result.pop("receipt")
    result.pop("effect_receipt")
    result["dispatch_attempted"] = True
    ledger = NativeStrategyLedger()
    ledger.record({"dispatch_record": result,
                   "ui_boundary_receipt": {"input_attempted": False, "adapter_called": False}})
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"


def test_explicit_native_rejection_before_input_is_not_an_uncertain_external_effect():
    result = outcome(status="rejected", effect="absent", attempted=False)
    result["dispatch_attempted"] = True  # Adapter was called but refused input at its final guard.
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["classification"] == "rejected_no_input"
    assert not ledger.assess(state(), action())["blocked"]


def test_no_input_already_satisfied_result_is_distinct_from_failed_dispatch():
    result = outcome(status="verified", effect="absent", attempted=False)
    result["receipt"]["status"] = "already_satisfied"
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["classification"] == "already_satisfied_no_input"


def test_large_valid_alternative_context_is_bounded_with_explicit_omissions():
    current = state()
    current["nodes"] = []
    proposals = []
    for index in range(48):
        node = copy.deepcopy(state()["nodes"][0])
        node["id"], node["generation"], node["scope_id"] = str(index) + "x" * 480, "g" * 500, "s" * 500
        current["nodes"].append(node)
        proposals.append(action(target=node["id"]))
    ledger = NativeStrategyLedger()
    view = ledger.view(current, proposals)
    assert len(json.dumps(view, ensure_ascii=False).encode()) <= 12288
    assert len(view["allowed_alternatives"]) + view["omitted_alternative_count"] == 48
    assert view["omitted_alternative_count"] > 0


def observed_popup_transition():
    record = outcome(status="observed_unverified")
    record["dispatch_attempted"] = True
    before, after = record["before"], record["after"]
    after["revision"] = "r2"
    after["nodes"].append({**copy.deepcopy(before["nodes"][0]), "id": "popup", "role": "popup_menu",
        "structure": {"parent_id": "open", "ancestors": [{key: copy.deepcopy(before["nodes"][0][key])
            for key in ("id", "generation", "role", "name", "visible", "enabled")}]}})
    descriptor = {"kind": "popup_opened", "source_ref": {"session_id": before["session_id"],
        "scope_id": before["scope"]["id"], "revision": before["revision"]}}
    contract = compile_immediate_transition(action(), before, transition=descriptor)
    return {"status": "observed_unverified", "dispatch_record": record,
        "immediate_effect_contract": contract.as_dict(),
        "immediate_transition_receipt": {"schema": "robit.omni.native-immediate-transition.v1",
            "compiler": "host_operation_semantics", "input_authority": False, "task_progress": False,
            "goal_verified": False, "original_source_ref": None, "comparison": contract.compare(before, after)}}


def refresh_typed_comparison(result, kind=None):
    record = result["dispatch_record"]
    before, after, offered = record["before"], record["after"], record["action"]
    descriptor = copy.deepcopy(result["immediate_effect_contract"]["transition"])
    if kind is not None:
        descriptor["kind"] = kind
    proposal = {"id": offered["id"], "verb": offered["verb"], "target": {"node_id": offered["node_id"]},
                "args": offered["args"]}
    contract = compile_immediate_transition(proposal, before, transition=descriptor)
    result["immediate_effect_contract"] = contract.as_dict()
    result["immediate_transition_receipt"]["comparison"] = contract.compare(before, after)
    return result["immediate_transition_receipt"]["comparison"]


def test_expected_ui_transition_can_be_observed_without_task_credit_or_permanent_uncertain_lock():
    result = observed_popup_transition()
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assessment = ledger.assess(result["dispatch_record"]["after"], action())
    assert assessment["classification"] == "observed_transition" and not assessment["blocked"]
    assert assessment["input_authority"] is assessment["task_progress"] is False
    assert ledger.view()["facts"][0]["claim_limit"] == "observed_ui_history_not_external_effect_proof"


@pytest.mark.parametrize("fault", ["claimed_only", "contradictory_comparison", "source_revision", "source_session",
                                  "geometry_only", "no_native_evidence", "no_dispatch", "model_credit", "fake_rebind",
                                  "unrelated_popup"])
def test_missing_or_contradictory_transition_evidence_keeps_uncertain_input_locked(fault):
    result = observed_popup_transition()
    if fault == "claimed_only":
        result.pop("immediate_effect_contract")
    elif fault == "contradictory_comparison":
        result["immediate_transition_receipt"]["comparison"]["status"] = "absent"
    elif fault == "source_revision":
        result["immediate_effect_contract"]["transition"]["source_ref"]["revision"] = "older"
    elif fault == "source_session":
        result["immediate_effect_contract"]["transition"]["source_ref"]["session_id"] = "foreign"
    elif fault == "geometry_only":
        result["dispatch_record"]["after"]["nodes"].pop()
        result["dispatch_record"]["after"]["nodes"][0]["bounds"]["x"] += 4
    elif fault == "no_native_evidence":
        result["dispatch_record"]["after"]["scope"].pop("provider")
    elif fault == "no_dispatch":
        result["dispatch_record"].pop("dispatch_attempted")
    elif fault == "model_credit":
        result["immediate_transition_receipt"]["task_progress"] = True
    elif fault == "fake_rebind":
        result["immediate_transition_receipt"]["original_source_ref"] = {
            "session_id": "owned-session", "scope_id": "window", "revision": "unseen"}
    else:
        popup = result["dispatch_record"]["after"]["nodes"][-1]
        popup["structure"] = {"parent_id": "different-control", "ancestors": []}
        # Recompute a truthful generic comparison to reproduce the real gap:
        # unrelated same-scope UI changed, but this input's target did not.
        before, after = result["dispatch_record"]["before"], result["dispatch_record"]["after"]
        contract = compile_immediate_transition(action(), before,
            transition=result["immediate_effect_contract"]["transition"])
        result["immediate_transition_receipt"]["comparison"] = contract.compare(before, after)
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"
    assert ledger.assess(state(), action())["blocked"]


@pytest.mark.parametrize("fault", ["missing", "clipped", "wrong_generation", "contradictory_parent", "target_replaced",
                                  "contradictory_ancestor", "partial_ancestor", "ancestry_cycle"])
def test_truthful_generic_popup_observation_without_exact_target_relationship_remains_uncertain(fault):
    result = observed_popup_transition()
    after = result["dispatch_record"]["after"]
    popup = after["nodes"][-1]
    if fault == "missing":
        popup.pop("structure")
    elif fault == "clipped":
        popup["structure"] = {"parent_id": "other", "ancestors": [{"id": "other", "generation": "g3"}]}
    elif fault == "wrong_generation":
        popup["structure"]["ancestors"][0]["generation"] = "a different control generation"
    elif fault == "contradictory_parent":
        popup["structure"]["parent_id"] = "different-control"
    elif fault == "target_replaced":
        after["nodes"][0]["generation"] = "g2"
    elif fault == "contradictory_ancestor":
        popup["structure"]["ancestors"][0].update(role="textbox", name="A different hidden disabled control",
                                                     visible=False, enabled=False)
    elif fault == "partial_ancestor":
        popup["structure"]["ancestors"][0] = {"id": "open", "generation": "g1"}
    else:
        popup["structure"]["ancestors"].append(copy.deepcopy(popup["structure"]["ancestors"][0]))
    assert refresh_typed_comparison(result)["status"] == "observed"
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"
    assert ledger.assess(state(), action())["blocked"]


@pytest.mark.parametrize("related", [False, True])
def test_focus_transition_must_reach_actual_target_instead_of_an_unrelated_field(related):
    result = observed_popup_transition()
    after = result["dispatch_record"]["after"]
    after["nodes"].pop()
    after["nodes"][1]["focused"] = False
    if related:
        after["nodes"][0]["focused"] = True
    else:
        after["nodes"].append({**copy.deepcopy(after["nodes"][1]), "id": "unrelated-field", "focused": True})
    assert refresh_typed_comparison(result, "focus_changed_within_scope")["status"] == "observed"
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["blocked"] is not related


def test_unrelated_bound_document_transition_cannot_clear_the_clicked_control():
    result = observed_popup_transition()
    before, after = result["dispatch_record"]["before"], result["dispatch_record"]["after"]
    after["nodes"].pop()
    for current, url in ((before, "https://example.test/old"), (after, "https://example.test/new")):
        current["scope"]["url"] = url
        current["nodes"].append({**copy.deepcopy(current["nodes"][1]), "id": "document", "generation": "d1",
                                 "role": "document_web", "focused": False})
        current["document_evidence"] = {"status": "observed", "source": "atspi.Document.DocURL", "url": url,
            "node_id": "document", "generation": "d1", "scope_id": "window", "window_id": "17",
            "provider": "atspi2", "provider_identity_sha256": "d" * 64}
    assert refresh_typed_comparison(result, "document_changed")["status"] == "observed"
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(before, action())["blocked"]


def test_lost_acknowledgment_locks_operation_without_claiming_no_effect():
    result = outcome()
    result.pop("receipt")
    result.pop("effect_receipt")
    result.pop("after")
    result["dispatch_attempted"] = True
    ledger = NativeStrategyLedger()
    ledger.record(result)
    assert ledger.assess(state(), action())["blocked"]
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"


def test_policy_deferral_without_input_is_informational_and_keeps_alternatives_available():
    ledger = NativeStrategyLedger()
    deferred = {"status": "abstained", "intent_selection_receipt": {"schema": "robit.omni.ui-intent-selection.v1"},
        "ui_boundary_receipt": {"input_attempted": False, "adapter_called": False},
        "effect_receipt": {"effect_state": "absent", "environment_may_have_changed": False}}
    ledger.record(deferred, source_phase=1)
    ledger.record(deferred, source_phase=2)
    view = ledger.view(state(), [action(), action("double_click")])
    assert not ledger.assess(state(), action())["blocked"]
    assert view["facts"][-1]["classification"] == "policy_deferred_no_input"
    assert len(view["allowed_alternatives"]) == 2 and not view["input_authority"]


def test_only_caller_supplied_fresh_compatible_alternatives_are_described():
    ledger = NativeStrategyLedger()
    ledger.record(outcome(status="verified", effect="unchanged"))
    alternatives = [action(), action("double_click"), action("invoke"), action("click", "not-observed")]
    view = ledger.view(state(), alternatives)
    assert {item["verb"] for item in view["allowed_alternatives"]} == {"double_click", "invoke"}
    assert all(item["input_authority"] is False for item in view["allowed_alternatives"])
    assert any(item["kind"] == "different_strategy_or_evidence" for item in view["evidence_needed"])


@pytest.mark.parametrize("change", ["generation", "session", "scope", "arguments", "target"])
def test_actual_new_operation_identity_is_distinct(change):
    current, proposal = state(), action()
    original = strategy_identity(current, proposal)
    if change == "generation":
        current["nodes"][0]["generation"] = "g2"
    elif change == "session":
        current["session_id"] = "other"
    elif change == "scope":
        current["scope"]["id"] = current["nodes"][0]["scope_id"] = "other"
    elif change == "target":
        proposal = action("focus", "draft")
    else:
        proposal = action("set_value", "draft", {"value": "desired"})
    assert strategy_identity(current, proposal)["operation_id"] != original["operation_id"]


def test_uncertain_locks_survive_capacity_exhaustion_without_silent_eviction():
    ledger = NativeStrategyLedger(max_entries=1)
    ledger.record(outcome())
    ledger.record(outcome(verb="double_click"))
    assert ledger.assess(state(), action())["classification"] == "input_effect_uncertain"
    assert ledger.assess(state(), action())["blocked"]
    assert ledger.assess(state(), action("invoke"))["blocked"]
    assert ledger.view()["capacity_exhausted"] is True


def test_wrapper_outcomes_and_dispatch_records_preserve_actual_action_and_deduplicate_replay():
    ledger = NativeStrategyLedger()
    result = outcome()
    ledger.record({"outcomes": [result]})
    ledger.record({"dispatch_record": result, "ui_boundary_receipt": {"input_attempted": True}})
    view = ledger.view()
    assert len(view["facts"]) == 1
    assert view["facts"][0]["repeated_receipt_count"] == 1


def verification_result(current, predicates):
    return {"status": "verified", "state": current, "content_sha256": digest(current),
        "verification_receipt": {"schema": "robit.omni.ui-verification.v1", "id": "fresh-independent-read",
            "status": "verified", "revision": current["revision"], "scope_id": current["scope"]["id"],
            "session_id": digest(current["session_id"]), "predicates_sha256": digest(predicates)}}


def test_independent_exact_value_verification_can_reconcile_idempotent_write_uncertainty():
    ledger = NativeStrategyLedger()
    result = outcome(verb="set_value", target="draft", args={"value": "new"})
    ledger.record(result)
    current = state()
    current["nodes"][1]["value"] = "new"
    predicates = [{"kind": "value", "node_id": "draft", "value": "new"}]
    identity = strategy_identity(state(), action("set_value", "draft", {"value": "new"}))
    reconciled = ledger.reconcile(identity["operation_id"], verification_result(current, predicates), predicates=predicates)
    assert reconciled["status"] == "effect_independently_observed"
    assert not ledger.assess(current, action("set_value", "draft", {"value": "new"}))["blocked"]
    assert reconciled["input_authority"] is False


@pytest.mark.parametrize("fault", ["prose", "wrong_value", "foreign_session", "stale_revision", "wrong_hash", "pointer",
                                  "missing_revision", "changed_window"])
def test_model_claims_or_wrong_independent_evidence_never_clear_unknown_input(fault):
    ledger = NativeStrategyLedger()
    verb, target, args = ("click", "open", {}) if fault == "pointer" else ("set_value", "draft", {"value": "new"})
    ledger.record(outcome(verb=verb, target=target, args=args))
    current = state()
    current["nodes"][1]["value"] = "new"
    predicates = [{"kind": "value", "node_id": "draft", "value": "new"}]
    evidence = verification_result(current, predicates)
    if fault == "prose":
        evidence = {"status": "verified", "assessment": "It definitely worked"}
    elif fault == "wrong_value":
        current["nodes"][1]["value"] = "wrong"
        evidence = verification_result(current, predicates)
    elif fault == "foreign_session":
        current["session_id"] = "foreign"
        evidence = verification_result(current, predicates)
    elif fault == "stale_revision":
        evidence["verification_receipt"]["revision"] = "stale"
    elif fault == "wrong_hash":
        evidence["content_sha256"] = "a" * 64
    elif fault == "missing_revision":
        current.pop("revision")
        evidence = verification_result(current | {"revision": None}, predicates)
    elif fault == "changed_window":
        current["scope"]["window_id"] = "another-owned-window"
        evidence = verification_result(current, predicates)
    identity = strategy_identity(state(), action(verb, target, args))
    with pytest.raises(ComputerUseError):
        ledger.reconcile(identity["operation_id"], evidence, predicates=predicates)
    assert ledger.assess(state(), action(verb, target, args))["blocked"]
