"""Historical prompt projection preserves current facts and execution authority."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from portal.ui_task import _compact_result, _planner_ledger, run_native_goal


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()).hexdigest()


def row(phase=1, *, text="owned value"):
    return {"phase": phase, "kind": "ui_boundary", "ok": True,
            "remaining_criteria": ["Verify every original criterion"], "criteria_authority": "planner_claim",
            "result": {"status": "observed_unverified", "reason": "Guarded input; effect unknown",
                "ui_boundary_receipt": {"schema": "robit.omni.ui-boundary.v1", "verified": False,
                    "input_attempted": None, "cached": False, "observed": False,
                    "task_progress": False, "new_effect_count": 0},
                "effect_receipt": {"effect_state": "unknown", "environment_may_have_changed": True},
                "observed_changes": {"nodes": [{"value": text}], "task_progress": False,
                                     "absence_claims_supported": False}}}


def metadata(projected):
    return projected[-1]["history_projection"]


def test_historical_receipts_preserve_exact_unknowns_and_reference_omitted_facts():
    original = [row()]
    frozen = copy.deepcopy(original)
    projected = _planner_ledger(original)
    assert original == frozen
    view = projected[0]
    assert view["phase"] == 1 and view["kind"] == "ui_boundary" and view["ok"] is True
    assert view["criteria_authority"] == "planner_claim"
    assert view["remaining_criteria"] == original[0]["remaining_criteria"]
    assert view["result"] == {key: value for key, value in original[0]["result"].items() if key != "observed_changes"}
    assert view["result"]["ui_boundary_receipt"]["input_attempted"] is None
    assert view["result"]["ui_boundary_receipt"]["verified"] is False
    assert view["result"]["ui_boundary_receipt"]["new_effect_count"] == 0
    assert view["history_reference"] == {"row_sha256": digest(original[0]), "omitted_fields_sha256": {
        "result.observed_changes": digest(original[0]["result"]["observed_changes"])}}
    fact = metadata(projected)
    assert fact["source_ledger_sha256"] == digest(original)
    assert fact["temporal_scope"] == "historical" and fact["input_authority"] is False
    assert fact["current_ui_verified"] is False and fact["omitted_facts_are_not_absent"] is True
    assert fact["source_rows"] == fact["retained_rows"] == 1
    assert fact["omitted_phase_count"] == 0 and fact["omitted_field_count"] == 1
    assert fact["latest_criteria_status"] == "retained" and fact["latest_criteria_count"] == 1
    assert fact["latest_criteria_sha256"] == digest(original[0]["remaining_criteria"])
    view["result"]["ui_boundary_receipt"]["input_attempted"] = True
    assert original == frozen


@pytest.mark.parametrize("text", ["x" * 20000, "éï🙂" * 6000, '\\"\n雪' * 6000])
def test_forty_large_historical_phases_have_a_utf8_bound_and_recent_receipt_order(text):
    original = [row(index, text=text) for index in range(1, 41)]
    for item in original:
        item["remaining_criteria"] *= 25
    frozen = copy.deepcopy(original)
    projected = _planner_ledger(original)
    assert len(encoded(projected)) <= 8192
    fact = metadata(projected)
    retained = [item for item in projected if "phase" in item]
    assert retained and [item["phase"] for item in retained] == list(range(41 - len(retained), 41))
    assert fact["source_rows"] == 40 and fact["retained_rows"] == len(retained)
    assert fact["omitted_phase_count"] == 40 - len(retained)
    assert fact["omitted_reference_count"] + len(fact["omitted_phase_references"]) == fact["omitted_phase_count"]
    for reference in fact["omitted_phase_references"]:
        assert reference["row_sha256"] == digest(original[reference["phase"] - 1])
    assert retained[-1]["remaining_criteria"] == original[-1]["remaining_criteria"]
    assert all("remaining_criteria" not in item for item in retained[:-1])
    assert original == frozen


def test_oversized_receipt_is_omitted_whole_with_its_error_and_hash_not_truncated_or_relabelled():
    original = [row()]
    original[0]["ok"] = False
    original[0]["result"].update(error="owned_read_failed", reason="🙂" * 100000)
    projected = _planner_ledger(original)
    assert len(encoded(projected)) <= 8192
    assert "result" not in projected[0]
    fact = metadata(projected)
    assert fact["retained_rows"] == 0 and fact["omitted_phase_count"] == 1
    assert fact["omitted_phase_references"] == [{"phase": 1, "row_sha256": digest(original[0])}]
    assert fact["latest_criteria_status"] == "omitted" and fact["latest_criteria_count"] == 1
    assert fact["latest_criteria_sha256"] == digest(original[0]["remaining_criteria"])
    assert original[0]["result"]["error"] == "owned_read_failed" and original[0]["ok"] is False


def test_included_errors_and_non_boolean_flags_are_not_coerced_or_promoted():
    original = [row()]
    original[0]["ok"] = False
    original[0]["result"].update(status="rejected", error="native_pointer_target_occluded")
    original[0]["result"]["ui_boundary_receipt"].update(adapter_called=True, cached="false", input_attempted=None)
    projected = _planner_ledger(original)
    assert projected[0]["result"] == {key: value for key, value in original[0]["result"].items() if key != "observed_changes"}
    assert projected[0]["ok"] is False
    assert metadata(projected)["evidence_authority"] == "untrusted_receipt_summary"


def test_empty_history_preserves_initial_packet_shape():
    assert _planner_ledger([]) == []


def test_latest_pending_claims_are_retained_once_with_verifiable_unicode_hashes():
    original = [row(index) for index in range(1, 13)]
    # An approximately 800-byte latest pending set is useful planning evidence.
    latest = ["Verify original requirement é🙂 " + str(index) + "x" * 119
              + ("yyy" if index == 0 else "") for index in range(5)]
    assert len(encoded(latest)) == 798
    original[-1]["remaining_criteria"] = latest
    projected = _planner_ledger(original)
    assert projected[-1]["remaining_criteria"] == latest
    assert sum("remaining_criteria" in item for item in projected) == 1
    assert metadata(projected)["latest_criteria_sha256"] == digest(latest)
    assert "default separators" in metadata(projected)["hash_serialization"]
    compact = json.dumps(latest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    assert digest(latest) != hashlib.sha256(compact).hexdigest()


def test_missing_latest_claims_are_unknown_not_an_empty_remaining_set():
    original = [row()]
    original[0].pop("remaining_criteria")
    fact = metadata(_planner_ledger(original))
    assert fact["latest_criteria_status"] == "not_provided"
    assert fact["latest_criteria_count"] is None and fact["latest_criteria_sha256"] is None


def test_oversized_schema_valid_unicode_claims_are_explicitly_unavailable_not_none_remaining():
    original = [row()]
    original[0]["remaining_criteria"] = ["🙂" * 240] * 25
    projected = _planner_ledger(original)
    fact = metadata(projected)
    assert len(encoded(projected)) <= 8192 and fact["retained_rows"] == 0
    assert fact["latest_criteria_status"] == "omitted" and fact["latest_criteria_count"] == 25
    assert fact["latest_criteria_sha256"] == digest(original[0]["remaining_criteria"])
    assert fact["omitted_facts_are_not_absent"] is True


def observation():
    return {"schema": "robit.omni.ui-state-receipt.v1", "content_sha256": "current-hash", "state": {
        "session_id": "owned", "scope": {"id": "window", "kind": "desktop"}, "revision": "fresh",
        "nodes": [{"id": "button", "generation": "1", "scope_id": "window", "role": "button",
            "name": "Owned control", "visible": True, "enabled": True, "sensitive": False,
            "operations": ["click"]}]}}


def boundary_result(index, *, novel=True):
    return {"status": "observed_unverified", "reason": "Effect remains unknown",
        "ui_boundary_receipt": {"schema": "robit.omni.ui-boundary.v1", "observed": True,
            "verified": False, "input_attempted": True, "cached": False, "adapter_called": True,
            "task_progress": False, "new_effect_count": 0,
            "novelty_key": hashlib.sha256(str(index if novel else 1).encode()).hexdigest()},
        "effect_receipt": {"effect_state": "unknown"},
        "observed_changes": {"nodes": [{"id": "owned", "kind": "fields", "after": {
            "focused": False, "selected": [], "value": "readback🙂" * 600}}],
            "snapshot_only": True, "absence_claims_supported": False, "task_progress": False}}


def exercise(*, novel=True, mutate_packet=False):
    packets, results, cancelled = [], [], []
    current, goal = observation(), "Exact unchanged original goal é🙂"

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        assert packet["goal"] == goal and packet["current_state"] == current
        if results:
            assert packet["last_result"] == _compact_result(results[-1])
        assert len(encoded(packet["ledger"])) <= 8192
        if mutate_packet and packet["ledger"]:
            packet["ledger"][0]["ok"] = "forged"
            packet["last_result"]["observed_changes"].clear()
        return {"kind": "ui_boundary", "step": {"id": "intent", "verb": "click",
                "target": {"node_id": "button"}, "args": {}},
                "remaining_criteria": ["No whole-task verification yet"] * 25}

    def execute(name, arguments):
        assert name == "native_ui" and arguments["operation"] == "observation_boundary"
        result = boundary_result(len(results) + 1, novel=novel)
        results.append(copy.deepcopy(result))
        return result

    events = list(run_native_goal(goal, observe=lambda: copy.deepcopy(current), compile_phase=compile_phase,
        execute=execute, cancel=lambda: cancelled.append(True), allowed_tools={"native_ui"}, max_phases=40,
        max_seconds=1800, clock=lambda: 0))
    return packets, results, events[-1]["response"]["portal"], cancelled


@pytest.mark.parametrize("mutate_packet", [False, True])
def test_forty_phase_loop_preserves_current_readback_raw_ledger_traces_and_budget(mutate_packet):
    packets, results, final, cancelled = exercise(mutate_packet=mutate_packet)
    assert len(packets) == len(results) == 40
    assert final["native_task"]["status"] == "phase_budget_exceeded"
    assert final["native_task"]["phases"] == 40 and final["native_task"]["task_verified"] is False
    assert len(final["native_task"]["ledger"]) == 12
    assert all(item["ok"] is True and "observed_changes" in item["result"]
               and len(item["remaining_criteria"]) == 25 for item in final["native_task"]["ledger"])
    assert all("history_projection" not in item and "history_reference" not in item
               for item in final["native_task"]["ledger"])
    raw = [json.loads(tool["result"]) for tool in final["safe_tools_executed"]]
    assert raw == results and len(raw) == 40 and cancelled == [True]
    for packet in packets[1:]:
        assert packet["last_result"]["observed_changes"]["nodes"][0]["after"]["focused"] is False
        assert packet["last_result"]["ui_boundary_receipt"]["new_effect_count"] == 0
        assert packet["last_result"]["effect_receipt"]["effect_state"] == "unknown"


def test_history_projection_does_not_reset_or_extend_three_failure_counter():
    packets, results, final, cancelled = exercise(novel=False)
    assert len(packets) == len(results) == 4  # One novel boundary, then three repeated observations.
    assert final["native_task"]["status"] == "blocked" and cancelled == [True]
    assert final["native_task"]["task_verified"] is False
