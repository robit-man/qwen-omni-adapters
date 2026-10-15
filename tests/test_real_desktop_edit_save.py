from __future__ import annotations

import json

import pytest

from runtime import verify_real_desktop as runner

INITIAL = runner.INITIAL.decode().removesuffix("\n")
EXPECTED = runner.EXPECTED.decode().removesuffix("\n")


def document(value=INITIAL, selected=False):
    result = {"id": "document", "generation": "1", "scope_id": "owned", "role": "textbox", "visible": True,
              "enabled": True, "sensitive": False, "focused": True, "value": value,
              "operations": ["select_text", "type_text", "set_value"]}
    if selected:
        result["text_selection"] = {"start": 0, "end": len(INITIAL)}
    return result


def state(nodes):
    return {"scope": {"id": "owned", "window_id": "17"}, "nodes": nodes}


def action(verb, before, after, args, target="document"):
    return {"status": "unresolved", "receipt": {"status": "dispatched", "backend": "x11_guarded_xtest"},
            "action": {"id": verb + "-id", "kind": "action", "scope_id": "owned", "generation": "1",
                       "node_id": target, "verb": verb, "args": args},
            "before": state(before), "after": state(after)}


def chain():
    select = action("select_text", [document()], [document(selected=True)], {"start": 0, "end": len(INITIAL)})
    typing = action("type_text", [document(selected=True)], [document(EXPECTED)], {"text": EXPECTED})
    save = {"id": "save", "generation": "1", "scope_id": "owned", "role": "button", "name": "Save",
            "visible": True, "enabled": True, "sensitive": False, "operations": ["click", "invoke"]}
    click = action("click", [document(EXPECTED), save], [document(EXPECTED), save], {}, "save")
    return [select, typing, click], state([document(EXPECTED)])


def assess(items, fresh, actual=runner.EXPECTED):
    return runner.edit_save_terminal_evidence(items, fresh, owned_scope="owned", owned_window="17",
        document_id="document", initial_buffer=INITIAL, actual=actual)


def test_no_return_edit_save_requires_exact_buffer_disk_and_actual_method_receipts():
    assert "without pressing Return" in runner.SCENARIOS["edit-save-line"]["instruction"]
    assert len(EXPECTED) == 18 and runner.EXPECTED == b"copper finch ready\n"
    items, fresh = chain()
    result = assess(items, fresh)
    assert result["status"] == "verified" and result["action_status_rewritten"] is False
    assert result["recovered_action_ids"] == ["type_text-id", "click-id"]
    assert all(item["status"] == "unresolved" for item in items)
    checks = runner.acceptance(items, runner.EXPECTED, len(INITIAL), scenario="edit-save-line", fresh=fresh,
        owned_scope="owned", owned_window="17", document_id="document")
    assert all(checks.values())


@pytest.mark.parametrize("bad", ["no-selection", "selection-other-node", "selection-stale", "selection-wrong-range",
    "semantic-write", "wrong-typed-text", "typed-return", "typing-not-dispatched", "typing-no-backend", "typing-abstain",
    "typing-unfocused", "typing-stale", "typing-wrong-scope", "typing-wrong-window", "no-save", "semantic-save",
    "save-no-backend", "save-not-dispatched", "save-wrong-button", "save-as", "dont-save", "save-disabled", "save-sensitive", "save-stale",
    "fresh-missing", "fresh-wrong-scope", "fresh-wrong-window", "fresh-stale", "fresh-other-node", "fresh-hidden",
    "fresh-sensitive", "fresh-extra-newline", "disk-extra-newline", "disk-no-newline", "disk-other-bytes", "contradiction"])
def test_edit_save_requires_actual_typed_and_pointer_method_with_independent_owned_effects(bad):
    items, fresh = chain()
    select, typing, save = items
    actual = runner.EXPECTED
    if bad == "no-selection":
        items.remove(select)
    elif bad == "selection-other-node":
        select["action"]["node_id"] = "other"
    elif bad == "selection-stale":
        select["action"]["generation"] = "2"
    elif bad == "selection-wrong-range":
        select["after"]["nodes"][0]["text_selection"]["end"] -= 1
    elif bad == "semantic-write":
        typing["action"].update(verb="set_value", args={"value": EXPECTED})
    elif bad == "wrong-typed-text":
        typing["action"]["args"]["text"] = "copper finch redy"
    elif bad == "typed-return":
        typing["action"]["args"]["text"] += "\n"
    elif bad == "typing-not-dispatched":
        typing["receipt"]["status"] = "rejected"
    elif bad == "typing-no-backend":
        typing["receipt"].pop("backend")
    elif bad == "typing-abstain":
        typing["action"]["kind"] = "abstain"
    elif bad == "typing-unfocused":
        typing["before"]["nodes"][0]["focused"] = False
    elif bad == "typing-stale":
        typing["action"]["generation"] = "2"
    elif bad == "typing-wrong-scope":
        typing["before"]["scope"]["id"] = "other"
    elif bad == "typing-wrong-window":
        typing["after"]["scope"]["window_id"] = "18"
    elif bad == "no-save":
        items.remove(save)
    elif bad == "semantic-save":
        save["action"]["verb"] = "invoke"
    elif bad == "save-no-backend":
        save["receipt"].pop("backend")
    elif bad == "save-not-dispatched":
        save["receipt"]["status"] = "rejected"
    elif bad == "save-wrong-button":
        save["before"]["nodes"][1]["name"] = "Open"
    elif bad == "save-as":
        save["before"]["nodes"][1]["name"] = "Save As"
    elif bad == "dont-save":
        save["before"]["nodes"][1]["name"] = "Don't Save"
    elif bad == "save-disabled":
        save["before"]["nodes"][1]["enabled"] = False
    elif bad == "save-sensitive":
        save["before"]["nodes"][1]["sensitive"] = True
    elif bad == "save-stale":
        save["action"]["generation"] = "2"
    elif bad == "fresh-missing":
        fresh = {}
    elif bad == "fresh-wrong-scope":
        fresh["scope"]["id"] = "other"
    elif bad == "fresh-wrong-window":
        fresh["scope"]["window_id"] = "18"
    elif bad == "fresh-stale":
        fresh["nodes"][0]["generation"] = "2"
    elif bad == "fresh-other-node":
        fresh["nodes"][0]["id"] = "other"
    elif bad == "fresh-hidden":
        fresh["nodes"][0]["visible"] = False
    elif bad == "fresh-sensitive":
        fresh["nodes"][0]["sensitive"] = True
    elif bad == "fresh-extra-newline":
        fresh["nodes"][0]["value"] += "\n"
    elif bad == "disk-extra-newline":
        actual += b"\n"
    elif bad == "disk-no-newline":
        actual = actual.removesuffix(b"\n")
    elif bad == "disk-other-bytes":
        actual = b"copper finch redy\n"
    elif bad == "contradiction":
        items.append(action("set_value", [document(EXPECTED)], [document("changed")], {"value": "changed"}))
        items.append(action("set_value", [document("changed")], [document(EXPECTED)], {"value": EXPECTED}))
    assert assess(items, fresh, actual)["status"] == "unresolved"


def test_actual_pointer_movement_still_requires_independent_nonsetup_samples(tmp_path):
    items, fresh = chain()
    receipt = assess(items, fresh)
    path = tmp_path / "pointer-events.jsonl"
    rows = [{"phase": "setup", "x": 1, "y": 1}, {"phase": "model_task", "x": 2, "y": 2}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert not runner.actual_pointer_movement(path, items, recovered_action_ids=tuple(receipt["recovered_action_ids"]))
    rows.append({"phase": "model_task", "x": 3, "y": 3})
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert runner.actual_pointer_movement(path, items, recovered_action_ids=tuple(receipt["recovered_action_ids"]))
