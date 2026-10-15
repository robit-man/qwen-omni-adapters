from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from runtime import verify_extended_browser as runner

QUOTE = "The interpreter acts as a simple calculator: you can type an expression at it and it will write the value."


def state(url=runner.INTRO, *, text=QUOTE, visible=True, document=True):
    return {"revision": "r1", "scope": {"id": "window", "kind": "desktop", "window_id": "123"},
        "acceptance_viewport": {"x": 0, "y": 0, "width": 1000, "height": 800}, "nodes": [
        {"id": "address", "role": "textbox", "name": "Address and search", "value": url, "visible": True, "operations": ["set_value"]},
        {"id": "document", "role": "document_web" if document else "textbox", "name": "An Informal Introduction to Python",
         "visible": visible, "operations": [] if document else ["set_value"]},
        {"id": "paragraph", "role": "paragraph", "value": text, "visible": visible and document,
         "operations": [], "bounds": {"x": 20, "y": 100, "width": 900, "height": 40}}]}


def call(name, arguments, result):
    return {"type": "tool", "phase": "complete", "tools": [{"name": name, "arguments": arguments,
                                                            "result": json.dumps(result), "status": "complete"}]}


def test_all_forty_cases_resolve_against_frozen_plan_and_empty_evidence_never_passes():
    assert len(runner.CASE_IDS) == 40 and len(set(runner.CASE_IDS)) == 40
    for case in runner.CASE_IDS:
        assert runner.case_spec(case)["id"] == case
        checks = runner.check_case(case, {})
        assert checks and not all(checks.values())
    with pytest.raises(ValueError):
        runner.case_spec("CU-001")


@pytest.mark.parametrize("url", ["https://docs.python.org.evil/3/tutorial/introduction.html",
                                  "http://docs.python.org/3/tutorial/introduction.html",
                                  "https://user:password@docs.python.org/3/tutorial/introduction.html",
                                  "https://docs.python.org:8443/3/tutorial/introduction.html"])
def test_source_urls_reject_spoofed_origins_and_credentials(url):
    assert runner.canonical_url(url) == ""


def test_actual_document_and_address_are_both_required():
    assert runner.committed_page(state(), runner.INTRO)
    assert not runner.committed_page(state(document=False), runner.INTRO)
    assert not runner.committed_page(state(visible=False), runner.INTRO)
    assert not runner.committed_page(state(runner.HOME), runner.INTRO)
    assert runner.canonical_url("docs.python.org/3.15/tutorial/introduction.html#numbers") == runner.INTRO


def test_noneditable_complete_source_sentence_is_required_for_quote():
    record = {"source_url": runner.INTRO, "quote": QUOTE}
    assert runner.quote_observed(record, [state()])
    for current in (state(document=False), state(visible=False), state(runner.PATHLIB), state(text="Invented calculator sentence.")):
        assert not runner.quote_observed(record, [current])
    assert not runner.quote_observed({**record, "quote": "acts as a simple calculator"}, [state()])


def test_find_or_address_fields_cannot_supply_quoted_evidence():
    current = state(text="Unrelated document.")
    current["nodes"].append({"id": "find", "role": "textbox", "name": "Find in page", "value": QUOTE,
                             "visible": True, "operations": ["set_value"]})
    assert not runner.quote_observed({"source_url": runner.INTRO, "quote": QUOTE}, [current])


def test_offscreen_document_aggregate_cannot_prove_visible_quote():
    current = state()
    current["nodes"][1]["value"] = QUOTE
    current["nodes"][2]["bounds"]["y"] = 900
    assert not runner.quote_observed({"source_url": runner.INTRO, "quote": QUOTE}, [current])
    current["nodes"][2].pop("bounds")
    assert not runner.quote_observed({"source_url": runner.INTRO, "quote": QUOTE}, [current])


def test_navigation_order_cannot_be_satisfied_by_final_page_alone():
    home = state(runner.HOME, text="Python Documentation")
    tutorial = state(runner.TUTORIAL, text="Python Tutorial")
    assert runner.ordered_pages([home, tutorial], [runner.HOME, runner.TUTORIAL])
    assert not runner.ordered_pages([tutorial, home], [runner.HOME, runner.TUTORIAL])
    assert not runner.ordered_pages([tutorial], [runner.HOME, runner.TUTORIAL])


def test_find_phrase_requires_exact_selected_substring_not_any_selection():
    current = state(text="Some unrelated text. The // operator performs floor division.")
    paragraph = current["nodes"][2]
    paragraph["text_selection"] = {"start": 0, "end": 4}
    current["nodes"].append({"id": "find", "role": "textbox", "name": "Find in page", "value": "floor division", "visible": True})
    assert runner.check_case("CU-046", {"states": [current], "fresh": current})["actual_find_and_document_match"] is False
    start = paragraph["value"].index("floor division")
    paragraph["text_selection"] = {"start": start, "end": start + len("floor division")}
    assert runner.check_case("CU-046", {"states": [current], "fresh": current})["actual_find_and_document_match"] is True


def test_tab_preservation_excludes_only_exact_owned_test_tab_not_git_named_baselines():
    before = state(runner.HOME, text="Python Documentation")
    before["nodes"] += [{"id": "baseline-git", "name": "Git private baseline", "role": "page_tab", "visible": True},
                        {"id": "python", "name": "Python", "role": "page_tab", "visible": True},
                        {"id": "test-git", "name": "Git", "role": "page_tab", "visible": True}]
    after = json.loads(json.dumps(before))
    after["nodes"] = [n for n in after["nodes"] if n.get("id") not in {"baseline-git", "test-git"}]
    evidence = {"before": before, "fresh": after, "owned_git_tab_id": "test-git"}
    assert runner.check_case("CU-045", evidence)["baseline_tabs_preserved"] is False
    after["nodes"].append(before["nodes"][-3])
    assert runner.check_case("CU-045", evidence)["baseline_tabs_preserved"] is True


def test_menu_hidden_identity_must_match_the_menu_actually_opened():
    before = state(runner.HOME, text="Python Documentation")
    opened = state(runner.HOME, text="Python Documentation")
    opened["nodes"] += [{"id": "actual-menu", "generation": "g1", "role": "menu", "visible": True},
                        {"id": "open-link", "role": "menuitem", "name": "Open Link in New Tab", "visible": True}]
    after = state(runner.HOME, text="Python Documentation")
    after["window_visibility"] = [{"id": "another-menu", "generation": "g1", "visible": False, "showing": False}]
    actions = [{"status": "verified", "action": {"id": "right", "verb": "right_click"}, "receipt": {"status": "dispatched"}},
               {"status": "verified", "action": {"id": "escape", "verb": "key_chord"}, "receipt": {"status": "dispatched"}}]
    evidence = {"events": [call("native_ui", {}, {"outcomes": actions})], "before": before, "states": [opened], "fresh": after}
    assert runner.check_case("CU-049", evidence)["link_menu_shown_then_positive_hidden"] is False
    after["nodes"].append({"id": "actual-menu", "generation": "g1", "visible": False, "role": "menu"})
    assert runner.check_case("CU-049", evidence)["link_menu_shown_then_positive_hidden"] is True
    after["nodes"][-1]["generation"] = "g2"
    assert runner.check_case("CU-049", evidence)["link_menu_shown_then_positive_hidden"] is False


@pytest.mark.parametrize("text,expected", [("3.1.2. Text¶", True), ("3.1.3 Lists", False), ("Text formatting", False), ("Text", True)])
def test_source_headings_accept_numbering_without_matching_other_sections(text, expected):
    assert runner.heading_is(text, "Text") is expected


def test_saved_note_requires_exact_sentence_and_actual_source_line(tmp_path):
    path = tmp_path / "note.txt"
    record = {"source_url": runner.INTRO, "quote": QUOTE}
    path.write_text(QUOTE + "\n" + runner.INTRO + "\n")
    assert runner.exact_note(path, [record])
    assert runner.file_source_records(path, [state()]) == [record]
    path.write_text("The calculator fact was saved.\n" + runner.INTRO)
    assert not runner.exact_note(path, [record])
    path.write_text(QUOTE + "\n" + runner.PATHLIB)
    assert not runner.exact_note(path, [record])
    path.unlink()
    path.symlink_to(tmp_path / "other.txt")
    assert not runner.exact_note(path, [record])


def make_bookmarks(path: Path):
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE moz_places(id INTEGER,url TEXT); CREATE TABLE moz_bookmarks(id INTEGER,guid TEXT,title TEXT,fk INTEGER,type INTEGER);")
        db.execute("INSERT INTO moz_places VALUES(1,?)", ("https://private.example/PRIVATE-HISTORY",))
        db.execute("INSERT INTO moz_bookmarks VALUES(1,'old-guid','PRIVATE-BOOKMARK',1,1)")


def test_bookmark_oracle_exports_only_aggregates_and_case_owned_entry(tmp_path):
    path = tmp_path / "places.sqlite"
    make_bookmarks(path)
    before = runner.bookmark_receipt(path)
    assert "PRIVATE" not in json.dumps(before)
    assert before["target"] == [] and set(before["rows"]) == {"1"}
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO moz_places VALUES(2,?)", (runner.INTRO,))
        db.execute("INSERT INTO moz_bookmarks VALUES(2,'new-guid','Copper Test Bookmark',2,1)")
    during = runner.bookmark_receipt(path)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM moz_bookmarks WHERE id=2")
    after = runner.bookmark_receipt(path)
    assert runner.bookmark_restored(before, after, [during])
    assert not runner.bookmark_restored(before, after, [])
    assert not runner.bookmark_restored(before, during, [during])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE moz_bookmarks SET title='changed' WHERE id=1")
    assert not runner.bookmark_restored(before, runner.bookmark_receipt(path), [during])


def test_existing_identical_bookmark_cannot_count_as_new_creation():
    existing = {"id": 1, "guid": "old", "title": "Copper Test Bookmark", "url": runner.INTRO}
    before = {"rows": {"1": "digest"}, "target": [existing]}
    assert not runner.bookmark_restored(before, before, [before])


def test_negative_memory_case_needs_real_empty_lookup_truthful_reply_and_preserved_other_session():
    marker = "random-marker"
    events = [call("memory_search", {"query": marker}, {"results": []})]
    evidence = {"events": events, "marker": marker, "private_canary": "PRIVATE-CANARY", "other_session_intact": True,
                "reply": json.dumps({"marker": marker, "memory_found": False})}
    assert all(runner.check_case("CU-066", evidence).values())
    for changed in ({"events": []}, {"other_session_intact": False}, {"reply": "I found it."},
                    {"reply": json.dumps({"marker": marker, "memory_found": True})}, {"private_canary": "random-marker"}):
        assert not all(runner.check_case("CU-066", {**evidence, **changed}).values())


def test_trash_requires_controller_invalidation_in_addition_to_empty_search():
    marker = "random-marker"
    evidence = {"events": [call("memory_search", {"query": marker}, {"results": []})], "marker": marker,
                "private_canary": "PRIVATE-CANARY", "other_session_intact": True,
                "reply": json.dumps({"marker": marker, "memory_found": False})}
    assert runner.check_case("CU-067", evidence)["trash_invalidated_controller_session"] is False
    evidence["trash_changed_controller_session"] = True
    assert all(runner.check_case("CU-067", evidence).values())


@pytest.mark.parametrize("name", ["shell", "browser_interact", "web_fetch", "web_search", "workspace_file", "computer_use"])
def test_any_forbidden_tool_remains_a_method_failure(name):
    assert runner.methods([call(name, {}, {})]) is False
    assert runner.methods([call("tool_search", {"family": "shell"}, {})]) is False


def test_spreadsheet_oracle_never_accepts_missing_or_success_named_files(tmp_path):
    assert not any(runner.spreadsheet_checks(tmp_path).values())
    (tmp_path / "expenses.ods").write_text("success total 50")
    (tmp_path / "expenses.pdf").write_text("success total 50")
    assert not any(runner.spreadsheet_checks(tmp_path).values())


def test_instruction_preconditions_do_not_contain_prescribed_runtime_action_ids():
    for number in (*range(41, 71), *range(91, 101)):
        instructions = runner.prerequisites(number, "actual audio marker")
        assert all("node_id" not in text and "action_id" not in text and "workflow" not in text for text in instructions)
