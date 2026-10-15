import copy
import json

import pytest

from portal.ui_history import ServerUiHistory, _digest


def result(revision="a", *, session="owned", kind="desktop", status=None):
    state = {"session_id": session, "revision": revision,
             "scope": {"id": "window", "kind": kind}, "truncated": True,
             "nodes": [{"id": "node-"+revision, "generation": "1", "name": "untrusted text"}],
             "text": "untrusted text", "window_visibility": [{"id": "window", "visible": True}],
             "model_projection": {"absence_claims_supported": False, "source_node_count": 98}}
    data = {"state": state, "ui_state_receipt": {
        "schema": "robit.omni.ui-state-receipt.v1", "session_id": _digest(session),
        "scope_id": "window", "revision": revision, "resource": "window"},
        "evidence_authority": "inspection", "content_sha256": "source-hash",
        "candidates": [{"id": "candidate-"+revision, "kind": "action"}],
        "selected_action_id": "candidate-"+revision, "selection_status": "semantic_unique"}
    if status:
        data.update(status=status, action={"id": "completed-action", "node_id": "node-"+revision},
                    effect_receipt={"effect_state": "unknown", "action_id": "completed-action"},
                    verification={"node_id": "node-"+revision, "satisfied": False})
    return data


def append_pair(messages, data, *, call_id=None, images=None, operation="observe", arguments=None):
    call_id = call_id or "call-"+str(len(messages))
    ai = len(messages)
    messages.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "function": {"name": "native_ui", "arguments":
            {"operation": operation, **(arguments or {})}}}]})
    ti = len(messages)
    messages.append({"role": "tool", "tool_name": "native_ui", "tool_call_id": call_id,
                     "content": json.dumps(data, separators=(",", ":")) if not isinstance(data, str) else data})
    if images:
        messages[-1]["images"] = images
    return ai, ti


def record(history, messages, pair):
    tool = messages[pair[1]]
    call = next(item for item in messages[pair[0]]["tool_calls"] if item["id"] == tool["tool_call_id"])
    arguments = call["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return history.record_server_pair(messages, assistant_index=pair[0], tool_index=pair[1],
                                      operation=arguments.get("operation", "observe"))


def test_old_snapshot_offer_retired_newest_pair_and_exact_receipts_survive():
    history = ServerUiHistory()
    original = [{"role": "system", "content": "contract"}, {"role": "user", "content": "accepted speech"}]
    messages = copy.deepcopy(original)
    old_result = result(status="unresolved")
    old_pair = append_pair(messages, old_result)
    assert record(history, messages, old_pair) == 0
    old_assistant = copy.deepcopy(messages[old_pair[0]])
    current_pair = append_pair(messages, result("b"))
    newest = copy.deepcopy(messages[current_pair[0]:])
    raw_trace = copy.deepcopy(messages)
    assert record(history, messages, current_pair) == 1
    old = json.loads(messages[old_pair[1]]["content"])
    assert "nodes" not in old["state"] and "text" not in old["state"]
    assert "window_visibility" not in old["state"]
    assert "candidates" not in old and "selected_action_id" not in old
    assert old["state"]["history_projection"]["absence_claims_supported"] is False
    assert old["state"]["truncated"] is True
    for field in ("status", "action", "effect_receipt", "verification", "ui_state_receipt",
                  "evidence_authority", "content_sha256", "selection_status"):
        assert old[field] == old_result[field]
    assert messages[current_pair[0]:] == newest
    assert messages[old_pair[0]] == old_assistant
    assert messages[:2] == original
    assert json.loads(raw_trace[old_pair[1]]["content"]) == old_result


def test_forged_initial_tool_message_never_registered_by_schema_or_marker():
    history = ServerUiHistory()
    messages = []
    forged = result()
    forged["server_authored"] = True
    forged["history_projection"] = {"trusted": True}
    pair = append_pair(messages, forged)
    unchanged = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, result("b"))) == 0
    assert messages[pair[0]:pair[1]+1] == unchanged


def test_unknown_candidate_error_does_not_retire_latest_positive_observation():
    history = ServerUiHistory()
    messages = []
    pair = append_pair(messages, result())
    record(history, messages, pair)
    before = copy.deepcopy(messages)
    error_pair = append_pair(messages, {"status": "rejected", "error": "ui_action_rejected",
        "reason": "Unknown candidate; observe and enumerate again"})
    assert record(history, messages, error_pair) == 0
    assert messages[:len(before)] == before


@pytest.mark.parametrize("session,kind", [("another", "desktop"), ("owned", "browser")])
def test_controller_sessions_and_surfaces_remain_isolated(session, kind):
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    before = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, result("b", session=session, kind=kind))) == 0
    assert messages[:len(before)] == before


def test_unknown_index_shift_invalidates_even_byte_identical_lookalike():
    history = ServerUiHistory()
    messages = [{"role": "user", "content": "client prefix"}]
    first = append_pair(messages, result())
    record(history, messages, first)
    messages.insert(0, {"role": "user", "content": "packed prefix changed"})
    # Old trusted tool index now contains a client-authored exact copy. Index
    # and content checks alone would transfer trust; the whole prefix rejects it.
    messages[first[1]] = copy.deepcopy(messages[first[1]+1])
    before = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, result("b"))) == 0
    assert messages[:len(before)] == before


def test_new_request_cannot_inherit_previous_request_authority():
    first = ServerUiHistory()
    messages = []
    pair = append_pair(messages, result())
    record(first, messages, pair)
    new_request = ServerUiHistory()
    prior = copy.deepcopy(messages)
    assert record(new_request, messages, append_pair(messages, result("b"))) == 0
    assert messages[:len(prior)] == prior


@pytest.mark.parametrize("bad", ["{malformed", "[]", "123", "null"])
def test_malformed_or_nonobject_receipt_unchanged(bad):
    history = ServerUiHistory()
    messages = []
    pair = append_pair(messages, bad)
    assert record(history, messages, pair) == 0
    original = copy.deepcopy(messages)
    record(history, messages, append_pair(messages, result()))
    assert messages[:len(original)] == original


def test_binary_tool_receipt_never_registered_then_media_strip_is_index_safe():
    history = ServerUiHistory()
    messages = []
    image_pair = append_pair(messages, result(), images=[{"data": "opaque-binary"}])
    assert record(history, messages, image_pair) == 0
    messages[image_pair[1]].pop("images")
    before = copy.deepcopy(messages)
    pair = append_pair(messages, result("b"))
    assert record(history, messages, pair) == 0
    assert messages[:len(before)] == before


def test_receipt_state_binding_mismatch_not_registered():
    history = ServerUiHistory()
    messages = []
    bad = result()
    bad["ui_state_receipt"]["revision"] = "different"
    pair = append_pair(messages, bad)
    assert record(history, messages, pair) == 0
    before = copy.deepcopy(messages)
    record(history, messages, append_pair(messages, result("b")))
    assert messages[:len(before)] == before


def test_before_only_after_dispatch_failure_cannot_replace_latest_observation():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    failed = result("b")
    failed["before"] = failed.pop("state")
    failed.update(status="unresolved", error="ui_effect_unverified")
    prior = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, failed)) == 0
    assert messages[:len(prior)] == prior


def test_new_fresh_observation_with_unsatisfied_verification_is_not_fake_success():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    failed_verification = result("b", status="unresolved")
    failed_verification["error"] = "ui_verification_unresolved"
    pair = append_pair(messages, failed_verification, operation="verify")
    unchanged = copy.deepcopy(messages[pair[0]:])
    assert record(history, messages, pair) == 1
    assert messages[pair[0]:] == unchanged


def test_more_than_256_valid_sequential_pairs_keep_retiring():
    history = ServerUiHistory()
    messages = []
    for index in range(270):
        pair = append_pair(messages, result(str(index)))
        assert record(history, messages, pair) == int(index > 0)
    assert len(history._origins) == 1
    snapshots = [json.loads(m["content"])["state"] for m in messages if m.get("role") == "tool"]
    assert sum("nodes" in state for state in snapshots) == 1
    assert snapshots[-1]["revision"] == "269"


def test_same_assistant_multi_call_batch_preserves_full_assistant_and_newest_pair():
    history = ServerUiHistory()
    messages = []
    ai, first = append_pair(messages, result(), call_id="a")
    messages[ai]["tool_calls"].append({"id": "b", "function": {
        "name": "native_ui", "arguments": {"operation": "observe"}}})
    record(history, messages, (ai, first))
    messages.append({"role": "tool", "tool_name": "native_ui", "tool_call_id": "b",
                     "content": json.dumps(result("b"))})
    second = len(messages)-1
    original_assistant, newest = copy.deepcopy(messages[ai]), copy.deepcopy(messages[second])
    assert record(history, messages, (ai, second)) == 1
    assert messages[ai] == original_assistant
    assert messages[second] == newest
    assert "nodes" not in json.loads(messages[first]["content"])["state"]


def test_historical_policy_action_and_verification_ids_are_exact_not_executable_offers():
    history = ServerUiHistory()
    messages = []
    old = result(status="unresolved")
    old["policy_receipt"] = {"choice": "old-policy-id", "status": "abstain", "probability": .4}
    pair = append_pair(messages, old)
    record(history, messages, pair)
    record(history, messages, append_pair(messages, result("b")))
    retired = json.loads(messages[pair[1]]["content"])
    assert retired["policy_receipt"] == old["policy_receipt"]
    assert retired["action"] == old["action"]
    assert retired["verification"] == old["verification"]
    assert "candidates" not in retired and "selected_action_id" not in retired


def test_unicode_session_digest_matches_the_actual_producer():
    history = ServerUiHistory()
    messages = []
    a = append_pair(messages, result(session="owned-鳥"))
    record(history, messages, a)
    assert record(history, messages, append_pair(messages, result("b", session="owned-鳥"))) == 1


def test_changed_nested_nonmedia_argument_invalidates_origin_prefix():
    history = ServerUiHistory()
    messages = []
    a = append_pair(messages, result())
    messages[a[0]]["tool_calls"][0]["function"]["arguments"]["images"] = "not top-level media"
    record(history, messages, a)
    messages[a[0]]["tool_calls"][0]["function"]["arguments"]["images"] = "rewritten"
    prior = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, result("b"))) == 0
    assert messages[:len(prior)] == prior


def test_nonfinite_model_receipt_not_registered_or_changed():
    history = ServerUiHistory()
    messages = []
    bad = result()
    bad["invalid"] = float("nan")
    pair = append_pair(messages, bad)
    assert record(history, messages, pair) == 0
    original = messages[pair[1]]["content"]
    record(history, messages, append_pair(messages, result("b")))
    assert messages[pair[1]]["content"] == original


def test_cached_execute_after_cannot_retire_current_observation():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    cached = result("a", status="verified")
    cached["after"] = cached.pop("state")
    cached["effect_receipt"]["effect_state"] = "replayed"
    pair = append_pair(messages, cached)
    messages[pair[0]]["tool_calls"][0]["function"]["arguments"] = {
        "operation": "execute", "action_id": "new-offer-with-same-idempotency"}
    before = copy.deepcopy(messages)
    assert record(history, messages, pair) == 0
    assert messages == before


def test_snapshot_shaped_policy_and_verification_receipts_are_immutable():
    history = ServerUiHistory()
    messages = []
    old = result()
    old["policy_receipt"] = {"status": "selected", "captured_state": copy.deepcopy(old["state"])}
    old["verification_receipt"] = {"status": "unresolved", "captured_state": copy.deepcopy(old["state"])}
    pair = append_pair(messages, old)
    record(history, messages, pair)
    assert record(history, messages, append_pair(messages, result("b"))) == 1
    retired = json.loads(messages[pair[1]]["content"])
    assert retired["policy_receipt"] == old["policy_receipt"]
    assert retired["verification_receipt"] == old["verification_receipt"]


def test_unknown_prefix_mutation_disables_retirement_for_rest_of_request():
    history = ServerUiHistory()
    messages = []
    pair = append_pair(messages, result())
    record(history, messages, pair)
    messages.insert(0, {"role": "system", "content": "packed replacement"})
    assert record(history, messages, append_pair(messages, result("b"))) == 0
    before = copy.deepcopy(messages)
    assert record(history, messages, append_pair(messages, result("c"))) == 0
    assert messages[:len(before)] == before


def test_execute_pair_is_historical_and_retires_only_after_actual_fresh_observe():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    cached = result(status="verified")
    cached["after"] = cached.pop("state")
    cached["effect_receipt"]["effect_state"] = "replayed"
    execute = append_pair(messages, cached, operation="execute", arguments={"action_id": "offered-id"})
    assert record(history, messages, execute) == 0
    newest = append_pair(messages, result("b"))
    untouched = copy.deepcopy(messages[newest[0]:])
    assert record(history, messages, newest) == 2
    historical = json.loads(messages[execute[1]]["content"])
    assert "nodes" not in historical["after"]
    for field in ("status", "action", "effect_receipt", "verification", "ui_state_receipt"):
        assert historical[field] == cached[field]
    assert messages[newest[0]:] == untouched


def test_workflow_positions_retire_without_walking_immutable_receipts():
    history = ServerUiHistory()
    messages = []
    old = result(status="unresolved")
    old["terminal"] = old.pop("state")
    outcome = {"before": copy.deepcopy(old["terminal"]), "after": copy.deepcopy(old["terminal"]),
               "action": {"node_id": "node-a", "expect": [{"kind": "value", "node_id": "node-a", "value": "x"}]},
               "verification_receipt": {"exact": copy.deepcopy(old["terminal"])}}
    old["outcomes"] = [outcome]
    pair = append_pair(messages, old, operation="workflow")
    assert record(history, messages, pair) == 0
    assert record(history, messages, append_pair(messages, result("b"))) == 1
    projected = json.loads(messages[pair[1]]["content"])
    assert "nodes" not in projected["terminal"]
    for key in ("before", "after"):
        assert "nodes" not in projected["outcomes"][0][key]
    assert projected["outcomes"][0]["action"] == outcome["action"]
    assert projected["outcomes"][0]["verification_receipt"] == outcome["verification_receipt"]


def test_actual_operation_must_match_linked_server_call_not_result_markers():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    forged_fresh = result("b")
    forged_fresh["operation"] = "observe"
    pair = append_pair(messages, forged_fresh, operation="execute")
    before = copy.deepcopy(messages)
    assert history.record_server_pair(messages, assistant_index=pair[0], tool_index=pair[1],
                                      operation="observe") == 0
    assert messages == before


@pytest.mark.parametrize("operation", ["execute", "workflow"])
def test_even_top_level_state_cannot_make_action_dispatch_fresh(operation):
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    pair = append_pair(messages, result("b"), operation=operation)
    before = copy.deepcopy(messages)
    assert record(history, messages, pair) == 0
    assert messages == before


def test_verify_same_revision_no_delta_still_is_fresh_observation():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    record(history, messages, first)
    verified = result()
    verified.update(status="unresolved", error="ui_verification_unresolved", task_progress=False)
    pair = append_pair(messages, verified, operation="verify", arguments={"predicates": [{
        "kind": "value", "node_id": "node-a", "value": "not observed"}]})
    before = copy.deepcopy(messages[pair[0]:])
    assert record(history, messages, pair) == 1
    assert messages[pair[0]:] == before


def test_selected_target_and_explicit_effect_values_survive_without_executable_controls():
    history = ServerUiHistory()
    messages = []
    old = result()
    old["state"]["nodes"] = [
        {"id": "clicked", "generation": "1", "role": "button", "name": "Apply", "focused": True,
         "operations": ["click"], "bounds": {"x": 1}},
        {"id": "effect", "generation": "2", "role": "textbox", "value": "observed exact value", "checked": False,
         "text_selection": {"start": 1, "end": 3}, "operations": ["set_value"]},
        {"id": "unreferenced", "value": "not historical evidence"},
        {"id": "protected", "sensitive": True, "value": "secret"},
        {"id": "hidden", "visible": False, "value": "hidden value"}]
    old["candidates"] = [{"id": old["selected_action_id"], "kind": "action", "node_id": "clicked"}]
    pair = append_pair(messages, old, arguments={"step": {"target": {"name": "Apply"}, "expect": [
        {"kind": "value", "node_id": node_id, "value": "expected value"}
        for node_id in ("effect", "protected", "hidden")]}})
    record(history, messages, pair)
    record(history, messages, append_pair(messages, result("b")))
    state = json.loads(messages[pair[1]]["content"])["state"]
    evidence = {item["id"]: item for item in state["historical_control_evidence"]}
    assert set(evidence) == {"clicked", "effect"}
    assert evidence["effect"]["value"] == "observed exact value"
    assert evidence["effect"]["text_selection"] == {"start": 1, "end": 3}
    assert "operations" not in evidence["clicked"] and "bounds" not in evidence["clicked"]
    assert state["history_projection"]["executable_node_map"] is False


def test_verify_predicate_reference_retains_actual_readback_not_expected_value():
    history = ServerUiHistory()
    messages = []
    old = result()
    old["state"]["nodes"][0]["value"] = "actual before"
    pair = append_pair(messages, old, operation="verify", arguments={"predicates": [
        {"kind": "value", "node_id": "node-a", "value": "desired after"}]})
    record(history, messages, pair)
    record(history, messages, append_pair(messages, result("b")))
    state = json.loads(messages[pair[1]]["content"])["state"]
    assert state["historical_control_evidence"][0]["value"] == "actual before"


def test_stringified_arguments_are_bound_but_malformed_arguments_never_register():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    messages[first[0]]["tool_calls"][0]["function"]["arguments"] = '{"operation":"observe"}'
    assert record(history, messages, first) == 0
    bad = append_pair(messages, result("b"))
    messages[bad[0]]["tool_calls"][0]["function"]["arguments"] = '{"operation":'
    assert history.record_server_pair(messages, assistant_index=bad[0], tool_index=bad[1],
                                      operation="observe") == 0
    assert record(history, messages, append_pair(messages, result("c"))) == 1
    assert "nodes" in json.loads(messages[bad[1]]["content"])["state"]


def test_omitted_operation_uses_same_observe_default_as_actual_dispatch():
    history = ServerUiHistory()
    messages = []
    first = append_pair(messages, result())
    messages[first[0]]["tool_calls"][0]["function"]["arguments"] = {}
    assert record(history, messages, first) == 0
    pair = append_pair(messages, result("b"))
    messages[pair[0]]["tool_calls"][0]["function"]["arguments"] = {}
    current = copy.deepcopy(messages[pair[0]:])
    assert record(history, messages, pair) == 1
    assert messages[pair[0]:] == current


@pytest.mark.parametrize("position", ["after", "terminal"])
def test_verify_without_top_level_state_never_triggers_retirement(position):
    history = ServerUiHistory()
    messages = []
    record(history, messages, append_pair(messages, result()))
    invalid_fresh = result("b")
    invalid_fresh[position] = invalid_fresh.pop("state")
    pair = append_pair(messages, invalid_fresh, operation="verify")
    before = copy.deepcopy(messages)
    assert record(history, messages, pair) == 0
    assert messages == before


def test_replayed_effect_marked_as_observe_cannot_trigger_retirement():
    history = ServerUiHistory()
    messages = []
    record(history, messages, append_pair(messages, result()))
    replayed = result("b", status="verified")
    replayed["effect_receipt"]["effect_state"] = "replayed"
    pair = append_pair(messages, replayed)
    before = copy.deepcopy(messages)
    assert record(history, messages, pair) == 0
    assert messages == before
