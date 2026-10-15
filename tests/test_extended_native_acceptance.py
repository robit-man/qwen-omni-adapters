"""CPU oracle/ownership checks, never evidence of a live desktop/model pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from runtime import acceptance_native_support as support
from runtime import verify_extended_native as native


def doc(value=native.INITIAL, selection=None, **changes):
    result = {"id": "doc", "generation": "1", "role": "textbox", "name": "Document",
              "visible": True, "enabled": True, "sensitive": False, "operations": ["set_value", "type_text"], "value": value}
    if selection:
        result["text_selection"] = {"start": selection[0], "end": selection[1]}
    return {**result, **changes}


def state(*items, window="17", **changes):
    return {"scope": {"id": "scope", "window_id": window}, "nodes": list(items), **changes}


def record(ui, **changes):
    return {"state": ui, "window_id": ui["scope"]["window_id"], "phase": "model_task", **changes}


def item(verb, before, after, args=None, status="unresolved", dispatch="dispatched", name="Document"):
    return {"action": {"id": verb+"-id", "kind": "action", "verb": verb, "node_id": "doc", "args": args or {}},
            "before": before or state(doc(name=name)), "after": after, "status": status,
            "receipt": {"status": dispatch, "backend": "x11_guarded_xtest"}}


def context(tmp_path, *, value=native.INITIAL):
    (tmp_path/"source.txt").write_text(value+"\n")
    return {"workspace": tmp_path, "primary_window": "17", "document_bindings": {"17": doc(value)},
            "manifest_before": support.file_manifest(tmp_path), "initial_records": [record(state(doc(value)))]}


def evaluate(case, ctx, outcomes, fresh, records=(), actual=None):
    return native.evaluate(case, ctx, list(records), outcomes, record(fresh),
                           support.file_manifest(ctx["workspace"]) if actual is None else actual)


def test_exact_native_case_catalog_and_source_goals():
    assert tuple(f"CU-{n:03}" for n in range(11, 41)) == native.CASE_IDS
    assert len({native.case_spec(case)["spoken_instruction"] for case in native.CASE_IDS}) == 30
    with pytest.raises(ValueError):
        native.case_spec("CU-041")


@pytest.mark.parametrize("case", list(native.BLOCKED_METHODS))
def test_unsupported_physical_method_blocks_before_any_gui_or_model(tmp_path, monkeypatch, case):
    monkeypatch.setattr(support, "desktop_subprocess_environment", lambda: {})
    monkeypatch.setattr(native, "REPO", tmp_path)
    monkeypatch.setattr(native, "source_hashes", lambda: {})
    # Source plan still refers to the real checkout; only artifacts use tmp.
    called = []
    monkeypatch.setattr(support.OwnedDesktop, "open_application", lambda *_a, **_k: called.append(True))
    args = argparse.Namespace(case=case, model="exact", think=False, output=tmp_path/"runtime-data"/case,
                              endpoint="http://unused", token_file=tmp_path/"no-token", timeout=1)
    assert native.run(args) == 1
    report = json.loads((args.output/"summary.json").read_text())
    assert report["status"] == "blocked_capability" and report["passed"] is False
    assert report["owned_window_closed"] is True and report["cleanup"] == [] and called == []


def test_single_click_needs_exact_selection_bytes_and_no_editor_opened(tmp_path):
    ctx = context(tmp_path)
    ctx.update(editor_windows_before=["old"], editor_windows_after=["old"])
    ui = state({"id": "list", "visible": True, "sensitive": False, "selected": ["alpha.txt"]})
    click = item("click", state(doc(name="alpha.txt")), ui)
    assert all(evaluate("CU-011", ctx, [click], ui).values())
    assert not all(evaluate("CU-011", ctx, [], ui).values())
    intermediate = record(ui, properties={"editor_windows": ["old", "unwanted"]})
    assert not evaluate("CU-011", ctx, [click], ui, [intermediate])["no_editor_opened"]
    click["action"]["verb"] = "double_click"
    assert not evaluate("CU-011", ctx, [click], ui)["no_double_click_or_open"]


@pytest.mark.parametrize("range_", [(0, 11), (6, 12), (0, 18)])
def test_shift_extension_requires_exact_fresh_range(tmp_path, range_):
    ctx = context(tmp_path)
    event = item("key_chord", state(doc(selection=(0, 5))), state(doc(selection=range_)), {"keys": ["Shift_L", "Right"]})
    assert not evaluate("CU-013", ctx, [event], event["after"])["exact_phrase_selection"]


def test_shift_extension_retains_unresolved_receipt_and_checks_generation(tmp_path):
    ctx = context(tmp_path)
    event = item("key_chord", state(doc(selection=(0, 5))), state(doc(selection=(0, 12))), {"keys": ["Shift_L", "Right"]})
    assert all(evaluate("CU-013", ctx, [event], event["after"]).values())
    assert event["status"] == "unresolved"
    assert not evaluate("CU-013", ctx, [event], state(doc(selection=(0, 12), generation="2")))["exact_phrase_selection"]


def test_shift_extension_allows_multiple_arrows_with_bridge_local_identity(tmp_path):
    ctx = context(tmp_path)
    def model_state(end):
        return state(doc(selection=(0, end), id="model-doc", generation="4"),
                     scope={"id": "model-scope", "window_id": "17"})
    events = [item("key_chord", model_state(end), model_state(end+1), {"keys": ["Shift_R", "Right"]})
              for end in range(5, 12)]
    for event in events:
        event["action"]["node_id"] = "model-doc"
    assert all(evaluate("CU-013", ctx, events, state(doc(selection=(0, 12)))).values())


@pytest.mark.parametrize("defect", ["missing_alpha", "incidental_shift", "wrong_document", "wrong_generation",
                                     "wrong_window", "wrong_scope", "wrong_target", "wrong_action_generation", "wrong_backend", "reversed_order", "reset",
                                     "substituted_selection", "changed_buffer"])
def test_shift_extension_rejects_unbound_unordered_or_substituted_trajectory(tmp_path, defect):
    ctx = context(tmp_path)
    first = item("key_chord", state(doc(selection=(0, 5))), state(doc(selection=(0, 6))), {"keys": ["Shift_L", "Right"]})
    last = item("key_chord", first["after"], state(doc(selection=(0, 12))), {"keys": ["Shift_L", "Right"]})
    events = [first, last]
    if defect == "missing_alpha":
        first["before"] = state(doc(selection=(0, 4)))
    elif defect == "incidental_shift":
        events = [item("key_chord", last["after"], last["after"], {"keys": ["Shift_L", "Right"]})]
    elif defect in {"wrong_document", "wrong_generation"}:
        field = "id" if defect == "wrong_document" else "generation"
        last["after"]["nodes"][0][field] = "different"
    elif defect in {"wrong_window", "wrong_scope"}:
        field = "window_id" if defect == "wrong_window" else "id"
        last["after"]["scope"][field] = "different"
    elif defect == "wrong_target":
        last["action"]["node_id"] = "different"
    elif defect == "wrong_action_generation":
        last["action"]["generation"] = "different"
    elif defect == "wrong_backend":
        last["receipt"]["backend"] = "semantic"
    elif defect == "reversed_order":
        events.reverse()
    elif defect == "reset":
        reset = item("key_chord", first["after"], state(doc(selection=(0, 5))), {"keys": ["Home"]})
        last["before"] = reset["after"]
        events.insert(1, reset)
    elif defect == "substituted_selection":
        last["action"].update(verb="select_text", args={"start": 0, "end": 12})
    else:
        first["after"]["nodes"][0]["value"] = "changed then restored"
    checks = evaluate("CU-013", ctx, events, state(doc(selection=(0, 12))))
    assert all(value for criterion, value in checks.items() if criterion != "ordered_alpha_shift_extension")
    assert not checks["ordered_alpha_shift_extension"]


@pytest.mark.parametrize("selection", [None, "unobserved", []])
def test_shift_extension_requires_observed_after_selection(tmp_path, selection):
    ctx = context(tmp_path)
    event = item("key_chord", state(doc(selection=(0, 5))), state(doc(text_selection=selection)), {"keys": ["Shift_L", "Right"]})
    assert not evaluate("CU-013", ctx, [event], state(doc(selection=(0, 12))))["ordered_alpha_shift_extension"]


def test_shift_extension_rejects_independent_document_changes_or_selection_reset(tmp_path):
    ctx = context(tmp_path)
    event = item("key_chord", state(doc(selection=(0, 5))), state(doc(selection=(0, 12))), {"keys": ["Shift_L", "Right"]})
    for middle in [state(doc("changed then restored")), state(doc(generation="2")),
                   state(doc(id="different")), state(doc(), window="other")]:
        assert not evaluate("CU-013", ctx, [event], event["after"], [record(middle)])["ordered_alpha_shift_extension"]
    records = [record(state(doc(selection=(0, 5)))), record(state(doc(selection=(0, 0)))),
               record(state(doc(selection=(0, 12))))]
    assert not evaluate("CU-013", ctx, [event], event["after"], records)["ordered_alpha_shift_extension"]


def test_copy_between_documents_requires_source_and_destination_effects(tmp_path):
    ctx = context(tmp_path)
    ctx.update(destination_window="18", document_bindings={"17": doc(), "18": doc("", id="dest")})
    copy = item("key_chord", state(doc(selection=(6, 12))), state(doc(selection=(6, 12))), {"keys": ["Control_L", "c"]})
    dest = state(doc("copper", id="dest"), window="18")
    paste = item("key_chord", state(doc("", id="dest"), window="18"), dest, {"keys": ["Control_L", "v"]})
    assert all(evaluate("CU-016", ctx, [copy, paste], dest).values())
    copy["after"] = state(doc("alpha ready"))
    assert not evaluate("CU-016", ctx, [copy, paste], dest)["source_terminal_buffer_unchanged"]


def test_cut_and_undo_requires_ordered_real_intermediate_states(tmp_path):
    ctx = context(tmp_path)
    cut = item("key_chord", state(doc(selection=(6, 12))), state(doc("alpha  ready")), {"keys": ["Control_L", "x"]})
    undo = item("key_chord", cut["after"], state(doc()), {"keys": ["Control_L", "z"]})
    assert all(evaluate("CU-017", ctx, [cut, undo], undo["after"]).values())
    assert not evaluate("CU-017", ctx, [undo, cut], state(doc()))["cut_then_undo_observed"]
    assert not evaluate("CU-017", ctx, [undo], state(doc()))["cut_then_undo_observed"]


def test_redo_requires_edit_undo_redo_sequence_and_final_exact_buffer(tmp_path):
    ctx = context(tmp_path, value="alpha")
    typed = item("type_text", state(doc("alpha")), state(doc("alpha beta")), {"text": " beta"})
    undo = item("key_chord", typed["after"], state(doc("alpha")), {"keys": ["Control_L", "z"]})
    redo = item("key_chord", undo["after"], typed["after"], {"keys": ["Control_L", "Shift_L", "z"]})
    assert all(evaluate("CU-018", ctx, [typed, undo, redo], typed["after"]).values())
    assert not evaluate("CU-018", ctx, [typed, redo], typed["after"])["edit_undo_redo_ordered"]


@pytest.mark.parametrize("verb,args", [("set_value", {"value": "Copper, finch!\nReady?"}), ("key_chord", {"keys": ["Control_L", "v"]})])
def test_keyboard_typing_cannot_pass_via_semantic_setter_or_paste(tmp_path, verb, args):
    ctx = context(tmp_path, value="")
    ui = state(doc("Copper, finch!\nReady?"))
    event = item(verb, state(doc("")), ui, args)
    assert not all(evaluate("CU-019", ctx, [event], ui).values())


def test_typing_exact_punctuation_newline_and_selection_replacement(tmp_path):
    ctx = context(tmp_path)
    ui = state(doc("Copper, finch!\nReady?"))
    typed = item("type_text", state(doc("")), ui, {"text": "Copper, finch!\nReady?"})
    assert all(evaluate("CU-019", ctx, [typed], ui).values())
    assert not evaluate("CU-019", ctx, [typed], state(doc("Copper, finch! Ready?")))["exact_keyboard_buffer"]
    replacement = item("type_text", state(doc(selection=(6, 12))), state(doc("alpha silver ready")), {"text": "silver"})
    assert all(evaluate("CU-020", ctx, [replacement], replacement["after"]).values())
    replacement["before"] = state(doc(selection=(0, 18)))
    assert not evaluate("CU-020", ctx, [replacement], replacement["after"])["only_copper_selected_before_typing"]


def test_geometry_restore_needs_actual_maximized_state_and_bounded_tolerance(tmp_path):
    ctx = context(tmp_path)
    geometry = {"x": 20, "y": 30, "width": 600, "height": 400}
    ordinary = {"window_id": "17", "geometry": geometry, "wm_state": "normal"}
    maximized = {**ordinary, "wm_state": "_NET_WM_STATE_MAXIMIZED_VERT, _NET_WM_STATE_MAXIMIZED_HORZ"}
    ctx["initial_records"] = [record(state(doc()), windows=[ordinary])]
    fresh = record(state(doc()), windows=[ordinary])
    checks = native.evaluate("CU-024", ctx, [record(state(doc()), windows=[maximized])], [], fresh, ctx["manifest_before"])
    assert all(checks.values())
    fresh["windows"][0] = {**ordinary, "geometry": {**geometry, "width": 590}}
    assert not native.evaluate("CU-024", ctx, [], [], fresh, ctx["manifest_before"])["original_geometry_restored"]


def test_save_as_exact_disk_oracle_rejects_implicit_extra_newline(tmp_path):
    ctx = context(tmp_path, value="copper finch ready")
    dialog = state({"id": "dlg", "role": "dialog", "visible": True, "sensitive": False}, doc("copper note.txt"))
    current = state(doc("copper finch ready"), window_visibility=[{"id": "scope", "owner_window_id": "17", "active": False, "visible": False, "showing": False}])
    save = item("click", state(doc(name="Save")), current)
    (tmp_path/"copper note.txt").write_bytes(b"copper finch ready\n")
    assert all(evaluate("CU-025", ctx, [save], current, [record(dialog)]).values())
    (tmp_path/"copper note.txt").write_bytes(b"copper finch ready\n\n")
    checks = evaluate("CU-025", ctx, [save], current, [record(dialog)])
    assert checks["fresh_saved_buffer"] is True and checks["exact_saved_bytes"] is False
    assert checks["no_unexpected_files"] is False


def test_dialog_closure_requires_positive_known_hidden_metadata_even_if_truncated(tmp_path):
    ctx = context(tmp_path, value="")
    dialog = state({"id": "dlg", "role": "dialog", "visible": True, "sensitive": False}, doc("draft.txt"))
    current = state(doc(""), truncated=True)
    cancel = item("key_chord", dialog, current, {"keys": ["Escape"]})
    assert not evaluate("CU-026", ctx, [cancel], current, [record(dialog)])["positive_dialog_closed"]
    current["window_visibility"] = [{"id": "scope", "owner_window_id": "17", "active": False, "visible": False, "showing": False}]
    assert all(evaluate("CU-026", ctx, [cancel], current, [record(dialog)]).values())
    current["window_visibility"][0]["owner_window_id"] = "unrelated"
    assert not evaluate("CU-026", ctx, [cancel], current, [record(dialog)])["positive_dialog_closed"]


def test_folder_create_and_rename_need_real_native_dialog_evidence(tmp_path):
    ctx = context(tmp_path)
    ctx["manifest_before"] = []
    empty = state({"id": "folder", "name": "Copper Notes", "visible": True, "sensitive": False})
    actual = [{"path": "Copper Notes", "kind": "directory"}]
    assert not all(evaluate("CU-031", ctx, [], empty, actual=actual).values())
    dialog = state({"id": "dlg", "role": "dialog", "visible": True, "sensitive": False})
    assert all(evaluate("CU-031", ctx, [], empty, [record(dialog)], actual).values())


@pytest.mark.parametrize(("case", "source", "copy", "shortcut"), [
    ("CU-033", "alpha.txt", True, "c"), ("CU-034", "beta.txt", False, "x")])
def test_file_copy_cut_require_exact_hashes_and_real_shortcuts(tmp_path, case, source, copy, shortcut):
    (tmp_path/source).write_bytes(native.CONTENTS[source])
    (tmp_path/"Destination").mkdir()
    ctx = {"workspace": tmp_path, "primary_window": "17", "manifest_before": support.file_manifest(tmp_path)}
    original = tmp_path/source
    destination = tmp_path/"Destination"/source
    destination.write_bytes(original.read_bytes())
    if not copy:
        original.unlink()
    ui = state({"id": "destination", "name": "Destination", "visible": True, "sensitive": False},
               {"id": "file", "name": source, "visible": True, "sensitive": False})
    first = item("key_chord", ui, ui, {"keys": ["Control_L", shortcut]})
    paste = item("key_chord", ui, ui, {"keys": ["Control_L", "v"]})
    assert all(evaluate(case, ctx, [first, paste], ui).values())
    destination.write_bytes(b"wrong")
    assert not evaluate(case, ctx, [first, paste], ui)["exact_filesystem_readback"]


def test_drag_move_requires_actual_matching_destination_node(tmp_path):
    (tmp_path/"gamma.txt").write_bytes(native.CONTENTS["gamma.txt"])
    (tmp_path/"Destination").mkdir()
    ctx = {"workspace": tmp_path, "primary_window": "17", "manifest_before": support.file_manifest(tmp_path)}
    (tmp_path/"gamma.txt").rename(tmp_path/"Destination/gamma.txt")
    ui = state(doc(name="gamma.txt"), {"id": "destination", "name": "Destination", "visible": True, "sensitive": False})
    event = item("drag_drop", ui, ui, {"to_node_id": "destination"})
    assert all(evaluate("CU-035", ctx, [event], ui).values())
    event["action"]["args"]["to_node_id"] = "unrelated"
    assert not evaluate("CU-035", ctx, [event], ui)["actual_gamma_destination_drag"]


def test_native_typing_and_pointer_methods_cannot_be_replaced_by_prose_or_unowned_input(tmp_path):
    ui = state(doc())
    click = item("click", ui, ui)
    pointer = tmp_path/"pointer.jsonl"
    pointer.write_text('\n'.join(json.dumps({"x": x, "y": 1, "phase": "model_task"}) for x in (1, 2)))
    spec = {"required_evidence": ["pointer", "keyboard"]}
    checks = native.method_checks(spec, [click], {"17"}, pointer, [], {"terminal": True})
    assert checks["actual_pointer_movement"] and not checks["actual_keyboard_dispatch"]
    assert not native.method_checks(spec, [], {"17"}, pointer, [], {"terminal": True})["actual_ui_input_dispatched"]
    click["after"]["scope"]["window_id"] = "other"
    assert not native.method_checks(spec, [click], {"17"}, pointer, [], {"terminal": True})["all_inputs_bound_to_owned_app"]
    assert not native.method_checks(spec, [click], {"17"}, pointer, [], {"terminal": False})["actual_pointer_movement"]


def test_window_switch_requires_observed_owned_foreground_sequence(tmp_path):
    ctx = context(tmp_path)
    ctx["calculator_window"] = "18"
    observed = [record(state(window="18")), record(state(doc()))]
    assert all(evaluate("CU-021", ctx, [], state(doc()), observed).values())
    assert not evaluate("CU-021", ctx, [], state(doc()), list(reversed(observed)))["calculator_then_editor_foreground"]
    assert not evaluate("CU-021", ctx, [], state(doc()), [record(state(window="99")), observed[-1]])["calculator_then_editor_foreground"]


def test_preferences_require_actual_property_restore_and_ui_control(tmp_path):
    ctx = context(tmp_path)
    ctx["preferences_before"] = {"line_numbers": "false", "tab_width": "uint32 8"}
    control = {"id": "line", "name": "Display line numbers", "role": "checkbox", "visible": True, "sensitive": False, "checked": True}
    gutter = {"id": "gutter", "name": "1", "role": "label", "visible": True, "sensitive": False}
    current = record(state(doc()), properties={"preferences": {"line_numbers": "false"}})
    enabled = record(state(doc(), control, gutter), properties={"preferences": {"line_numbers": "true"}})
    checks = native.evaluate("CU-027", ctx, [enabled, current], [], current, ctx["manifest_before"])
    assert all(checks.values())
    current["properties"]["preferences"]["line_numbers"] = "true"
    assert not native.evaluate("CU-027", ctx, [enabled], [], current, ctx["manifest_before"])["effective_setting_exactly_restored"]
    current["properties"]["preferences"]["line_numbers"] = "false"
    no_gutter = record(state(doc(), control), properties=enabled["properties"])
    assert not native.evaluate("CU-027", ctx, [no_gutter, current], [], current, ctx["manifest_before"])["positive_editor_gutter_evidence"]


def test_tab_width_requires_observed_property_and_exact_restoration(tmp_path):
    ctx = context(tmp_path)
    ctx["preferences_before"] = {"tab_width": "uint32 8"}
    control = {"id": "tabs", "name": "Tab width", "role": "combobox", "visible": True, "sensitive": False, "value": "4"}
    middle = record(state(control), properties={"preferences": {"tab_width": "uint32 4"}})
    fresh = record(state(doc()), properties={"preferences": {"tab_width": "uint32 8"}})
    assert all(native.evaluate("CU-028", ctx, [middle, fresh], [], fresh, ctx["manifest_before"]).values())
    assert not native.evaluate("CU-028", ctx, [fresh], [], fresh, ctx["manifest_before"])["effective_setting_target_observed"]


def test_multiple_file_copy_requires_selected_exact_preimage_and_no_extra_files(tmp_path):
    for name in ("alpha.txt", "beta.txt"):
        (tmp_path/name).write_bytes(native.CONTENTS[name])
    (tmp_path/"Destination").mkdir()
    ctx = {"workspace": tmp_path, "manifest_before": support.file_manifest(tmp_path)}
    for name in ("alpha.txt", "beta.txt"):
        (tmp_path/"Destination"/name).write_bytes(native.CONTENTS[name])
    ui = state({"id": "dest", "name": "Destination", "visible": True, "sensitive": False},
               *[{"id": name, "name": name, "visible": True, "sensitive": False} for name in ("alpha.txt", "beta.txt")])
    selection = state({"id": "list", "visible": True, "sensitive": False, "selected": ["alpha.txt", "beta.txt"]})
    copy = item("key_chord", selection, ui, {"keys": ["Control_L", "c"]})
    paste = item("key_chord", ui, ui, {"keys": ["Control_L", "v"]})
    assert all(evaluate("CU-036", ctx, [copy, paste], ui).values())
    selection["nodes"][0]["selected"].append("unrelated.txt")
    assert not evaluate("CU-036", ctx, [copy, paste], ui)["exact_multiple_selection_before_copy"]
    (tmp_path/"Destination"/"extra.txt").touch()
    assert not evaluate("CU-036", ctx, [copy, paste], ui)["exact_filesystem_readback"]


def test_file_search_requires_real_query_selected_result_and_unchanged_owned_files(tmp_path):
    (tmp_path/"copper-report.txt").write_text("report")
    ctx = {"workspace": tmp_path, "manifest_before": support.file_manifest(tmp_path)}
    ui = state({"id": "query", "role": "textbox", "value": "copper report", "visible": True, "sensitive": False},
               {"id": "list", "selected": ["copper-report.txt"], "visible": True, "sensitive": False})
    assert all(evaluate("CU-037", ctx, [], ui, [record(ui)]).values())
    ui["nodes"][1]["selected"].append("other.txt")
    assert not evaluate("CU-037", ctx, [], ui, [record(ui)])["exact_owned_result_selected"]


def test_file_order_requires_distinct_observed_rows_and_both_orderings(tmp_path):
    for name, value in (("alpha.txt", b"aaa"), ("beta.txt", b"b"), ("gamma.txt", b"cc")):
        (tmp_path/name).write_bytes(value)
    ctx = {"workspace": tmp_path, "manifest_before": support.file_manifest(tmp_path)}
    def ordered(names):
        return state({"id": "table", "role": "table", "visible": True, "sensitive": False},
                     *[{"id": name, "name": name, "role": "table_cell", "visible": True, "sensitive": False,
                        "bounds": {"x": 10, "y": i*20, "width": 20, "height": 20}} for i, name in enumerate(names)])
    by_size = ordered(["beta.txt", "gamma.txt", "alpha.txt"])
    by_name = ordered(["alpha.txt", "beta.txt", "gamma.txt"])
    actions = [item("click", state(doc(name=name)), by_name) for name in ("Size", "Name")]
    assert all(evaluate("CU-038", ctx, actions, by_name, [record(by_size)]).values())
    assert not evaluate("CU-038", ctx, actions, by_name)["size_order_then_name_order"]
    by_size["nodes"][2]["bounds"]["y"] = by_size["nodes"][1]["bounds"]["y"]
    assert native.file_order(by_size, {"alpha.txt", "beta.txt", "gamma.txt"}) == []


def test_trash_oracle_requires_exact_owned_hash_restore_and_unchanged_other_metadata(tmp_path):
    (tmp_path/"restore-me.txt").write_text("owned")
    ctx = {"workspace": tmp_path, "manifest_before": support.file_manifest(tmp_path),
           "trash_before": {"owned": [], "unrelated_metadata_sha256": "unchanged"}}
    original = {"original_path": str((tmp_path/"restore-me.txt").resolve()), "sha256": ctx["manifest_before"][0]["sha256"]}
    middle = record(state(), properties={"trash": {"owned": [original], "unrelated_metadata_sha256": "unchanged"}})
    fresh = record(state(), properties={"trash": {"owned": [], "unrelated_metadata_sha256": "unchanged"}})
    restore = item("click", state(doc(name="Restore")), state())
    checks = native.evaluate("CU-039", ctx, [middle], [restore], fresh, ctx["manifest_before"])
    assert all(checks.values())
    middle["properties"]["trash"]["unrelated_metadata_sha256"] = "changed"
    assert not native.evaluate("CU-039", ctx, [middle], [restore], fresh, ctx["manifest_before"])["preexisting_trash_metadata_unchanged"]
    assert not native.evaluate("CU-039", ctx, [], [restore], fresh, ctx["manifest_before"])["exact_owned_item_entered_trash"]


def test_trash_readback_binds_encoded_exact_path_and_never_logs_other_names(tmp_path, monkeypatch):
    home = tmp_path/"home"
    trash = home/".local/share/Trash"
    (trash/"info").mkdir(parents=True)
    (trash/"files").mkdir()
    target = tmp_path/"owned folder"/"restore-me.txt"
    (trash/"info"/"item.trashinfo").write_text("[Trash Info]\nPath="+str(target).replace(" ", "%20")+"\n")
    (trash/"files"/"item").write_bytes(b"owned")
    (trash/"info"/"unrelated-secret.trashinfo").write_text("[Trash Info]\nPath=/unrelated/private\n")
    (trash/"files"/"unrelated-secret").write_bytes(b"unrelated")
    monkeypatch.setattr(native.Path, "home", lambda: home)
    receipt = native.trash_receipt(target)
    assert receipt["owned"] == [{"original_path": str(target), "sha256": hashlib.sha256(b"owned").hexdigest()}]
    assert "unrelated-secret" not in json.dumps(receipt) and "/unrelated/private" not in json.dumps(receipt)


def test_rename_collision_requires_real_conflict_and_both_original_hashes(tmp_path):
    for name in ("alpha.txt", "beta.txt"):
        (tmp_path/name).write_bytes(native.CONTENTS[name])
    ctx = {"workspace": tmp_path, "manifest_before": support.file_manifest(tmp_path)}
    conflict = state({"id": "dialog", "role": "dialog", "name": "beta.txt already exists", "visible": True, "sensitive": False})
    cancel = item("key_chord", conflict, state(), {"keys": ["Escape"]})
    assert all(evaluate("CU-040", ctx, [cancel], state(), [record(conflict)]).values())
    assert not evaluate("CU-040", ctx, [cancel], state())["native_name_conflict_observed"]
    (tmp_path/"beta.txt").write_bytes(native.CONTENTS["alpha.txt"])
    assert not evaluate("CU-040", ctx, [cancel], state(), [record(conflict)])["both_original_names_hashes_intact"]


def test_only_canonical_native_tool_method_allowed():
    assert native.tool_method_audit([{"tools": [{"name": "native_ui"}]}])
    assert not native.tool_method_audit([{"tools": [{"name": "shell"}]}])
    assert not native.tool_method_audit([{"tools": [{"name": "computer_use", "arguments": {"surface": "browser"}}]}])


def test_manifest_never_follows_symlink_or_ignores_unexpected_file(tmp_path):
    owned = tmp_path/"owned"
    owned.mkdir()
    (owned/"link").symlink_to(tmp_path/"unrelated")
    assert support.file_manifest(owned) == [{"path": "link", "kind": "symlink"}]
    for n in range(128):
        (owned/str(n)).touch()
    with pytest.raises(RuntimeError, match="128"):
        support.file_manifest(owned)


def test_register_and_close_never_target_reused_pid_or_window(monkeypatch):
    owned = support.OwnedDesktop({"DISPLAY": ":test"})
    monkeypatch.setattr(support, "process_start", lambda _pid: "start")
    monkeypatch.setattr(support, "command", lambda *_a, **_k: "20")
    owned.register(support.OwnedWindow("calc", "17", "20", "start"))
    calls = []
    monkeypatch.setattr(support.subprocess, "run", lambda argv, **_k: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "99", ""))
    assert owned.close()[0]["closed"] is False
    assert calls == [["xdotool", "getwindowpid", "17"]]
    with pytest.raises(RuntimeError, match="identity"):
        owned.register(support.OwnedWindow("calc", "18", "21", "start"))


def test_absent_app_has_true_empty_cleanup_and_failed_unknown_launch_is_not_true(monkeypatch):
    owned = support.OwnedDesktop({"DISPLAY": ":test"})
    monkeypatch.setattr(support.shutil, "which", lambda _name: None)
    with pytest.raises(support.CapabilityBlocked):
        owned.open_calculator()
    assert owned.close() == []
    owned._pending.append({"application": "shared", "standalone": False, "registered": False, "process": None})
    assert owned.close()[0]["closed"] is False


def application_discovery(monkeypatch, *, visible=("202",), pid="101", before=("17",), start_changed=False):
    calls, starts = [], []
    process = SimpleNamespace(pid=101, poll=lambda: None)
    monkeypatch.setattr(support.shutil, "which", lambda _name: "/usr/bin/gedit")
    monkeypatch.setattr(support.subprocess, "Popen", lambda *_a, **_k: process)
    def process_start(actual_pid):
        starts.append(actual_pid)
        return "reused" if start_changed and len(starts) > 1 else "start-"+actual_pid
    monkeypatch.setattr(support, "process_start", process_start)
    def command(argv, _environment, **_kwargs):
        calls.append(argv)
        if argv[1] == "search":
            return "\n".join(visible if "--onlyvisible" in argv else before)
        if argv[1] == "getwindowpid":
            return pid
        if argv[1] == "windowactivate":
            return ""
        pytest.fail("Unexpected setup command")
    monkeypatch.setattr(support, "command", command)
    return support.OwnedDesktop({"DISPLAY": ":test"}), calls


def test_owned_editor_discovery_ignores_unmapped_helper_and_retains_exact_pid_start(monkeypatch):
    # Measured CU087 shape: helper201 is absent from --onlyvisible, while202
    # is the single mapped window. They may share the same class and PID.
    owned, calls = application_discovery(monkeypatch, visible=("202",))
    window = owned.open_editor()
    assert (window.window_id, window.pid, window.process_start) == ("202", "101", "start-101")
    searches = [call for call in calls if call[1] == "search"]
    assert searches == [["xdotool", "search", "--class", "[Gg]edit"],
                        ["xdotool", "search", "--onlyvisible", "--class", "[Gg]edit"]]
    assert calls[-1] == ["xdotool", "windowactivate", "--sync", "202"]


@pytest.mark.parametrize("visible", [("202", "203"), ("17", "202", "203")])
def test_two_genuinely_visible_new_windows_still_reject_without_activation(monkeypatch, visible):
    owned, calls = application_discovery(monkeypatch, visible=visible)
    with pytest.raises(RuntimeError, match="Concurrent"):
        owned.open_editor()
    assert owned.windows == [] and all(call[1] != "windowactivate" for call in calls)


def test_preexisting_unmapped_window_that_becomes_visible_cannot_be_adopted(monkeypatch):
    owned, calls = application_discovery(monkeypatch, before=("17", "201"), visible=("201", "202"))
    assert owned.open_editor().window_id == "202"
    assert [call for call in calls if call[1] == "windowactivate"] == [["xdotool", "windowactivate", "--sync", "202"]]


@pytest.mark.parametrize(("pid", "changed"), [("99", False), ("101", True)])
def test_visible_window_wrong_pid_or_reused_process_start_cannot_be_adopted(monkeypatch, pid, changed):
    owned, calls = application_discovery(monkeypatch, pid=pid, start_changed=changed)
    with pytest.raises(RuntimeError, match="exact owned process identity"):
        owned.open_editor()
    assert owned.windows == [] and all(call[1] != "windowactivate" for call in calls)


def test_dbus_launcher_keeps_unique_new_window_pid_and_start_binding(monkeypatch):
    owned, _calls = application_discovery(monkeypatch, pid="99")
    window = owned.open_application("gedit", [], "shared-app", "class", standalone=False)
    assert (window.window_id, window.pid, window.process_start) == ("202", "99", "start-99")


def test_unmapped_only_startup_times_out_without_window_activation(monkeypatch):
    owned, calls = application_discovery(monkeypatch, visible=("17",))
    times = iter((0, 1, 13))
    monkeypatch.setattr(support.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(support.time, "sleep", lambda _seconds: None)
    with pytest.raises(support.CapabilityBlocked, match="did not expose"):
        owned.open_editor()
    assert owned.windows == [] and all(call[1] != "windowactivate" for call in calls)


def test_independent_observer_never_reads_unowned_foreground_or_calls_pointing(tmp_path, monkeypatch):
    owned = support.OwnedDesktop({"DISPLAY": ":test"})
    monkeypatch.setattr(support, "_x11_foreground", lambda _env: {"window_id": "other", "process_id": 99})
    monkeypatch.setattr(support, "NativeUiAdapter", lambda *_a, **_k: pytest.fail("Unowned GUI must not be read"))
    observer = support.NativeEvidenceObserver(tmp_path, owned)
    assert observer.fresh()["outside_owned_foreground"] is True
    observer.close()


def test_spoken_helper_keeps_actual_partial_asr_after_transport_failure(tmp_path, monkeypatch):
    raw = b"actual-wav"
    monkeypatch.setattr(support, "synthesize_instruction", lambda *_a: (raw, {"source": "generated_tts_audio"}))
    def failed(*_args, **kwargs):
        kwargs["evidence_path"].write_text('{"type":"observation","transcript":"redy"}\n')
        raise RuntimeError("transport stopped")
    monkeypatch.setattr(support, "audio_turn", failed)
    with pytest.raises(RuntimeError, match="transport"):
        support.run_spoken_turn(None, "unused", "exact", "ready", tmp_path, think=True)
    evidence = json.loads((tmp_path/"speech-evidence.json").read_text())
    assert evidence["asr_transcript"] == "redy" and evidence["authoritative_final_received"] is False
    assert evidence["think"] is True and evidence["input_wav_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("flags,think", [([], False), (["--think"], True)])
def test_extended_cli_preserves_native_boolean_default(monkeypatch, flags, think):
    captured = []
    monkeypatch.setattr(native, "run", lambda args: captured.append(args) or 7)
    monkeypatch.setattr("sys.argv", ["native", "--case", "CU-011", *flags])
    assert native.main() == 7 and captured[0].think is think


def test_ods_reads_real_formula_and_rejects_values_only(tmp_path):
    path = tmp_path/"totals.ods"
    content = '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"><table:table>'
    content += ''.join('<table:table-row><table:table-cell office:value="'+value+'"'+(' table:formula="of:=SUM([.A1:.A2])"' if value == "42" else '')+'/></table:table-row>' for value in ("12", "30", "42"))
    content += '</table:table></office:document-content>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
    assert native.ods_values(path) == {"a1": "12", "a2": "30", "a3": "42", "formula": "of:=SUM([.A1:.A2])"}
    path.write_bytes(b"not ODS")
    assert native.ods_values(path) == {}
