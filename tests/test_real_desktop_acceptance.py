from __future__ import annotations

import json

import pytest

from runtime.verify_real_desktop import (
    EXPECTED,
    INITIAL,
    SCENARIOS,
    acceptance,
    actions_in_owned_window,
    actual_pointer_movement,
    context_terminal_evidence,
    dispatched_actions,
    physical_pointer_action,
    selection_terminal_evidence,
    word_terminal_evidence,
)

BUFFER = INITIAL.decode().removesuffix("\n")


def document(*, selected=False, value=BUFFER):
    node = {"id": "document", "generation": "1", "scope_id": "owned", "role": "textbox", "visible": True, "enabled": True,
            "focused": True, "sensitive": False,
            "operations": ["set_value", "select_text", "key_chord"], "value": value}
    if selected:
        node["text_selection"] = {"start": 0, "end": len(INITIAL.decode().removesuffix("\n"))}
    return node


def outcome(verb, before, after, *, args=None, truncated=False):
    return {"status": "verified", "receipt": {"status": "dispatched"},
            "action": {"id": verb + "-action", "verb": verb, "node_id": "document", "args": args or {}},
            "before": {"scope": {"id": "owned", "window_id": "17"}, "nodes": before},
            "after": {"scope": {"id": "owned", "window_id": "17"}, "nodes": after, "truncated": truncated}}


def test_selection_only_cannot_be_reported_as_full_desktop_acceptance() -> None:
    event, fresh = recovered_selection()
    events = [event]
    checks = acceptance(events, INITIAL, 18, scenario="selection-only", fresh=fresh,
                        owned_scope="owned", owned_window="17", document_id="document")
    assert all(checks.values())
    assert set(checks) == set(SCENARIOS["selection-only"]["criteria"])
    assert not all(acceptance(events, INITIAL, 18, scenario="full").values())


def recovered_selection():
    item = outcome("select_text", [document()], [document(selected=True)], args={"start": 0, "end": len(BUFFER)})
    item["status"] = "unresolved"
    item["action"].update(scope_id="owned", generation="1")
    return item, json.loads(json.dumps(item["after"]))


def test_selection_terminal_recovers_exact_dispatched_full_buffer_range_without_rewriting_action() -> None:
    item, fresh = recovered_selection()
    evidence = selection_terminal_evidence([item], fresh, owned_scope="owned", owned_window="17",
        document_id="document", initial_buffer=BUFFER, actual=INITIAL)
    assert evidence["status"] == "verified" and evidence["action_status_rewritten"] is False
    assert evidence["recovered_action_ids"] == ["select_text-action"]
    assert evidence["historical_action_statuses"] == [{"action_id": "select_text-action", "status": "unresolved"}]
    assert item["status"] == "unresolved"


@pytest.mark.parametrize("target", ["document", "window"])
def test_selection_terminal_accepts_actual_select_all_keyboard_only_in_focused_owned_document(target) -> None:
    item, fresh = recovered_selection()
    item["action"].update(verb="key_chord", args={"keys": ["Control_L", "a"]})
    if target == "window":
        window = {"id": "owned", "generation": "1", "scope_id": "owned", "role": "window", "visible": True,
                  "enabled": True, "sensitive": False, "operations": ["key_chord"]}
        item["before"]["nodes"].append(window)
        item["action"]["node_id"] = "owned"
    assert selection_terminal_evidence([item], fresh, owned_scope="owned", owned_window="17",
        document_id="document", initial_buffer=BUFFER, actual=INITIAL)["status"] == "verified"
    item["before"]["nodes"][0]["focused"] = False
    assert selection_terminal_evidence([item], fresh, owned_scope="owned", owned_window="17",
        document_id="document", initial_buffer=BUFFER, actual=INITIAL)["status"] == "unresolved"


@pytest.mark.parametrize("bad", ["abstain", "not-dispatched", "wrong-range-args", "wrong-range-after", "no-after", "no-fresh",
    "wrong-range-fresh", "changed-buffer", "changed-file", "wrong-scope", "wrong-window", "other-control", "wrong-generation",
    "before-changed", "sensitive", "disabled", "hidden", "keyboard-other-control", "keyboard-other-chord", "contradiction"])
def test_selection_recovery_rejects_missing_or_contradictory_fresh_target_evidence(bad) -> None:
    item, fresh = recovered_selection()
    items, actual = [item], INITIAL
    if bad == "abstain":
        item["action"]["kind"] = "abstain"
    elif bad == "not-dispatched":
        item["receipt"]["status"] = "rejected"
    elif bad == "wrong-range-args":
        item["action"]["args"]["end"] -= 1
    elif bad == "wrong-range-after":
        item["after"]["nodes"][0]["text_selection"]["end"] -= 1
    elif bad == "no-after":
        item["after"] = {}
    elif bad == "no-fresh":
        fresh = {}
    elif bad == "wrong-range-fresh":
        fresh["nodes"][0]["text_selection"]["end"] -= 1
    elif bad == "changed-buffer":
        fresh["nodes"][0]["value"] = "changed"
    elif bad == "changed-file":
        actual = b"changed\n"
    elif bad == "wrong-scope":
        fresh["scope"]["id"] = "other"
    elif bad == "wrong-window":
        fresh["scope"]["window_id"] = "18"
    elif bad == "other-control":
        item["action"]["node_id"] = "other-textbox"
    elif bad == "wrong-generation":
        fresh["nodes"][0]["generation"] = "2"
    elif bad == "before-changed":
        item["before"]["nodes"][0]["value"] = "changed"
    elif bad == "sensitive":
        fresh["nodes"][0]["sensitive"] = True
    elif bad == "disabled":
        fresh["nodes"][0]["enabled"] = False
    elif bad == "hidden":
        fresh["nodes"][0]["visible"] = False
    elif bad in {"keyboard-other-control", "keyboard-other-chord"}:
        item["action"].update(verb="key_chord", args={"keys": ["Control_L", "a"]})
        if bad == "keyboard-other-control":
            other = {**document(), "id": "other-control"}
            item["before"]["nodes"].append(other)
            item["action"]["node_id"] = "other-control"
        else:
            item["action"]["args"]["keys"][1] = "c"
    else:
        change = outcome("set_value", [document()], [document(value="changed")], args={"value": "changed"})
        change["action"].update(scope_id="owned", generation="1")
        items.append(change)
        restore = outcome("set_value", [document()], fresh["nodes"], args={"value": BUFFER})
        restore["action"].update(scope_id="owned", generation="1")
        items.append(restore)
    assert selection_terminal_evidence(items, fresh, owned_scope="owned", owned_window="17",
        document_id="document", initial_buffer=BUFFER, actual=actual)["status"] == "unresolved"
    assert not all(acceptance(items, actual, len(BUFFER), scenario="selection-only", fresh=fresh,
        owned_scope="owned", owned_window="17", document_id="document").values())


def test_saved_bytes_alone_cannot_replace_actual_save_click_receipt() -> None:
    events = [outcome("set_value", [document()], [document(value=EXPECTED.decode())], args={"value": EXPECTED.decode()})]
    checks = acceptance(events, EXPECTED, 18, scenario="replace-save")
    assert checks["replacement_verified"] and checks["exact_saved_file_bytes"]
    assert checks["actual_save_click_verified"] is False


def test_context_menu_absence_in_truncated_snapshot_does_not_prove_closure() -> None:
    events, fresh = recovered_context()
    events[-1]["after"].pop("window_visibility")
    checks = acceptance(events, INITIAL, 18, scenario="context-menu", fresh=fresh, owned_scope="owned", owned_window="17")
    assert checks["actual_context_menu_observed"] and checks["document_buffer_unchanged"]
    assert checks["escape_menu_closed_verified"] is False


def recovered_context():
    popup = {"id": "popup", "generation": "1", "scope_id": "popup", "role": "window", "visible": True}
    menu = {"id": "menu", "generation": "1", "scope_id": "popup", "role": "menu", "visible": True, "focused": True}
    entry = {"id": "paste", "scope_id": "popup", "role": "menuitem", "visible": True}
    click = outcome("right_click", [document()], [popup, menu, entry])
    click["status"] = "unresolved"
    click["receipt"]["backend"] = "x11_guarded_xtest"
    click["action"]["scope_id"] = "owned"
    click["after"]["scope"]["id"] = "popup"
    restored = document()
    restored.update(id="restored-document", generation="2")
    escape = outcome("key_chord", [popup, menu, entry], [restored], args={"keys": ["Escape"]}, truncated=True)
    escape["status"] = "unresolved"
    escape["action"].update(node_id="menu", scope_id="popup")
    escape["before"]["scope"]["id"] = "popup"
    escape["after"]["window_visibility"] = [{"id": "popup", "generation": "1", "owner_window_id": "17",
        "source": "retained_atspi_window", "active": False, "visible": False, "showing": False}]
    return [click, escape], json.loads(json.dumps(escape["after"]))


def test_context_terminal_uses_positive_same_popup_hidden_flags_after_and_fresh_without_rewriting(tmp_path) -> None:
    items, fresh = recovered_context()
    evidence = context_terminal_evidence(items, fresh, owned_scope="owned", owned_window="17", expected_buffer=BUFFER)
    assert evidence["status"] == "verified" and evidence["action_status_rewritten"] is False
    assert evidence["historical_action_statuses"] == [{"action_id": "right_click-action", "status": "unresolved"},
                                                     {"action_id": "key_chord-action", "status": "unresolved"}]
    assert evidence["recovered_action_ids"] == ["right_click-action"]
    assert all(acceptance(items, INITIAL, 18, scenario="context-menu", fresh=fresh, owned_scope="owned", owned_window="17").values())
    assert all(item["status"] == "unresolved" for item in items)
    assert fresh["truncated"] is True
    path = tmp_path / "pointer.jsonl"
    path.write_text('\n'.join(json.dumps({"phase": "model_task", "x": x, "y": 20}) for x in [10, 30]))
    assert not actual_pointer_movement(path, items)
    assert actual_pointer_movement(path, items, recovered_action_ids=tuple(evidence["recovered_action_ids"]))


def test_right_click_can_focus_document_after_save_but_escape_must_restore_document_focus() -> None:
    items, fresh = recovered_context()
    items[0]["before"]["nodes"][0]["focused"] = False
    assert context_terminal_evidence(items, fresh, owned_scope="owned", owned_window="17", expected_buffer=BUFFER)["status"] == "verified"
    items[-1]["after"]["nodes"][0]["focused"] = False
    assert context_terminal_evidence(items, fresh, owned_scope="owned", owned_window="17", expected_buffer=BUFFER)["status"] == "unresolved"


@pytest.mark.parametrize("bad", ["abstain", "not-dispatched", "not-xtest", "no-menu", "popup-wrong-window", "escape-rejected",
    "escape-wrong-scope", "escape-unfocused", "no-after-hidden", "no-fresh-hidden", "generation", "not-fully-hidden",
    "untrusted-source", "wrong-owner", "no-fresh", "final-wrong-scope", "final-content", "final-unfocused",
    "final-document", "final-generation", "intervening-mutation", "changed-file"])
def test_context_recovery_rejects_missing_stale_unowned_or_contradictory_positive_evidence(bad) -> None:
    items, fresh = recovered_context()
    actual = INITIAL
    click, escape = items
    if bad == "abstain":
        click["action"]["kind"] = "abstain"
    elif bad == "not-dispatched":
        click["receipt"]["status"] = "rejected"
    elif bad == "not-xtest":
        click["receipt"].pop("backend")
    elif bad == "no-menu":
        click["after"]["nodes"] = [node for node in click["after"]["nodes"] if node["role"] != "menuitem"]
    elif bad == "popup-wrong-window":
        click["after"]["scope"]["window_id"] = "18"
    elif bad == "escape-rejected":
        escape["receipt"]["status"] = "rejected"
    elif bad == "escape-wrong-scope":
        escape["action"]["scope_id"] = "owned"
    elif bad == "escape-unfocused":
        escape["before"]["nodes"][1]["focused"] = False
    elif bad == "no-after-hidden":
        escape["after"].pop("window_visibility")
    elif bad == "no-fresh-hidden":
        fresh.pop("window_visibility")
    elif bad == "generation":
        fresh["window_visibility"][0]["generation"] = "2"
    elif bad == "not-fully-hidden":
        fresh["window_visibility"][0]["showing"] = True
    elif bad == "untrusted-source":
        fresh["window_visibility"][0]["source"] = "model_claim"
    elif bad == "wrong-owner":
        fresh["window_visibility"][0]["owner_window_id"] = "18"
    elif bad == "no-fresh":
        fresh = {}
    elif bad == "final-wrong-scope":
        fresh["scope"]["id"] = "other"
    elif bad == "final-content":
        fresh["nodes"][0]["value"] = "changed"
    elif bad == "final-unfocused":
        fresh["nodes"][0]["focused"] = False
    elif bad == "final-document":
        fresh["nodes"][0]["id"] = "another-document"
    elif bad == "final-generation":
        fresh["nodes"][0]["generation"] = "3"
    elif bad == "intervening-mutation":
        wrong = json.loads(json.dumps(fresh["nodes"][0]))
        wrong["value"] = "changed"
        items.append(outcome("set_value", [], [wrong], args={"value": "changed"}))
        items.append(outcome("set_value", [], fresh["nodes"], args={"value": BUFFER}))
    else:
        actual = b"changed\n"
    assert not all(acceptance(items, actual, 18, scenario="context-menu", fresh=fresh, owned_scope="owned", owned_window="17").values())


def test_owned_window_audit_accepts_native_popup_scope_transition_but_rejects_other_xid() -> None:
    items, _fresh = recovered_context()
    assert actions_in_owned_window(items, "17")
    items[-1]["after"]["scope"]["window_id"] = "18"
    assert not actions_in_owned_window(items, "17")
    assert not actions_in_owned_window(items, "")


def test_spoken_claim_or_no_action_receipt_never_passes_any_scenario() -> None:
    for scenario in SCENARIOS:
        assert not all(acceptance([], EXPECTED, 18, scenario=scenario).values())


def test_pointer_movement_excludes_setup_and_requires_a_verified_pointer_action(tmp_path) -> None:
    path = tmp_path / "pointer.jsonl"
    samples = [{"phase": "speech_source", "x": 0, "y": 0}, {"phase": "model_task", "x": 10, "y": 20}]
    path.write_text("\n".join(json.dumps(item) for item in samples))
    click = [outcome("click", [], [])]
    assert not actual_pointer_movement(path, click)
    samples.append({"phase": "model_task", "x": 30, "y": 40})
    path.write_text("\n".join(json.dumps(item) for item in samples))
    assert actual_pointer_movement(path, click)
    assert not actual_pointer_movement(path, [outcome("select_text", [], [])])


def test_double_click_word_requires_actual_double_click_and_exact_alpha_selection() -> None:
    selected = document()
    selected["text_selection"] = {"start": 0, "end": 5}
    action = outcome("double_click", [document()], [selected])
    assert not all(acceptance([action], INITIAL, 18, scenario="double-click-word").values())
    action["action"].update(node_id="visual:word", generation="a"*64)
    action["receipt"].update(backend="x11_guarded_xtest", point_receipt={"candidate_id": "visual:word", "frame_sha256": "a"*64})
    oracle = {"fresh": action["after"], "owned_scope": "owned", "document_id": "document"}
    assert not all(acceptance([action], INITIAL, 18, scenario="double-click-word").values())
    assert all(acceptance([action], INITIAL, 18, scenario="double-click-word", **oracle).values())
    action["action"]["verb"] = "select_text"
    assert not all(acceptance([action], INITIAL, 18, scenario="double-click-word", **oracle).values())
    action["action"]["verb"] = "double_click"
    selected["text_selection"]["end"] = 6
    assert not all(acceptance([action], INITIAL, 18, scenario="double-click-word", **oracle).values())


def recovered_word():
    selected = document()
    selected["text_selection"] = {"start": 0, "end": 5}
    item = outcome("double_click", [document()], [selected])
    item["status"] = "unresolved"
    item["action"].update(node_id="visual:word", generation="a"*64)
    item["receipt"].update(backend="x11_guarded_xtest", point_receipt={"candidate_id": "visual:word", "frame_sha256": "a"*64})
    fresh = json.loads(json.dumps(item["after"]))
    return item, fresh


def test_fresh_word_terminal_oracle_recovers_goal_without_rewriting_unresolved_action(tmp_path) -> None:
    item, fresh = recovered_word()
    evidence = word_terminal_evidence([item], fresh, owned_scope="owned", document_id="document", initial_buffer=BUFFER)
    assert evidence["status"] == "verified" and evidence["action_status_rewritten"] is False
    assert evidence["historical_action_statuses"] == [{"action_id": "double_click-action", "status": "unresolved"}]
    assert all(acceptance([item], INITIAL, 18, scenario="double-click-word", fresh=fresh,
                          owned_scope="owned", document_id="document").values())
    assert item["status"] == "unresolved"
    path = tmp_path / "pointer.jsonl"
    path.write_text('\n'.join(json.dumps({"phase": "model_task", "x": x, "y": 20}) for x in [10, 30]))
    assert not actual_pointer_movement(path, [item])
    assert actual_pointer_movement(path, [item], recovered_action_ids=tuple(evidence["recovered_action_ids"]))


@pytest.mark.parametrize("bad", ["abstain", "not-dispatched", "no-fresh", "no-post-dispatch", "changed-selection", "changed-content",
                                 "wrong-scope", "wrong-document", "changed-file", "intervening-mutation"])
def test_word_terminal_recovery_rejects_missing_or_contradictory_evidence(bad) -> None:
    item, fresh = recovered_word()
    items, actual = [item], INITIAL
    if bad == "abstain":
        item["action"]["kind"] = "abstain"
    elif bad == "not-dispatched":
        item["receipt"]["status"] = "rejected"
    elif bad == "no-fresh":
        fresh = {}
    elif bad == "no-post-dispatch":
        item["after"] = {}
    elif bad == "changed-selection":
        fresh["nodes"][0]["text_selection"]["end"] = 4
    elif bad == "changed-content":
        fresh["nodes"][0]["value"] = "alpha corrupted"
    elif bad == "wrong-scope":
        fresh["scope"]["id"] = "other"
    elif bad == "wrong-document":
        fresh["nodes"][0]["id"] = "other-document"
    elif bad == "changed-file":
        actual = b"alpha corrupted\n"
    else:
        items.append(outcome("set_value", [document()], [document(value="changed")], args={"value": "changed"}))
        items.append(outcome("set_value", [document()], fresh["nodes"], args={"value": BUFFER}))
    assert not all(acceptance(items, actual, 18, scenario="double-click-word", fresh=fresh,
                              owned_scope="owned", document_id="document").values())


def test_visual_invoke_counts_as_pointer_only_with_matching_native_dispatch_receipt(tmp_path) -> None:
    path = tmp_path / "pointer.jsonl"
    path.write_text('\n'.join(json.dumps({"phase": "model_task", "x": x, "y": 20}) for x in [10, 30]))
    item = {"status": "verified", "action": {"kind": "action", "verb": "invoke", "node_id": "visual:save", "generation": "a"*64},
            "receipt": {"status": "dispatched", "backend": "x11_guarded_xtest",
                        "point_receipt": {"candidate_id": "visual:save", "frame_sha256": "a"*64}}}
    assert actual_pointer_movement(path, [item])
    item["receipt"].pop("backend")
    assert not physical_pointer_action(item)
    item["receipt"]["backend"] = "x11_guarded_xtest"
    item["receipt"]["point_receipt"]["candidate_id"] = "visual:unrelated"
    assert not physical_pointer_action(item)
    item["action"]["node_id"] = "atspi:save"
    item["receipt"]["point_receipt"]["candidate_id"] = "atspi:save"
    assert not physical_pointer_action(item)


def test_scope_audit_ignores_abstention_and_pre_dispatch_rejection() -> None:
    actual = {"action": {"kind": "action"}, "receipt": {"status": "dispatched"}, "before": {"scope": {"id": "owned"}}}
    abstention = {"action": {"kind": "abstain"}, "status": "abstained"}
    rejected = {"action": {"kind": "action"}, "receipt": {"status": "rejected"}}
    assert dispatched_actions([abstention, actual, rejected]) == [actual]


def test_spelled_speech_variant_keeps_full_independent_criteria_and_original_instruction() -> None:
    assert SCENARIOS["full-spelled"]["criteria"] == SCENARIOS["full"]["criteria"]
    assert "R E A D Y" in SCENARIOS["full-spelled"]["instruction"]
    assert "Do not add a blank line" in SCENARIOS["full-spelled"]["instruction"]
    assert "followed by exactly one newline" in SCENARIOS["full"]["instruction"]
    assert acceptance([], EXPECTED, 18, scenario="full-spelled")["exact_saved_file_bytes"]
    assert not acceptance([], b"copper finch redy\n\n", 18, scenario="full-spelled")["exact_saved_file_bytes"]
