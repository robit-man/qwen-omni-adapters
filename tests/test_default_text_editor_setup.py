"""CPU controls for the explicit host text-handler supplement, never GUI work."""
from __future__ import annotations

import copy
import subprocess
from types import SimpleNamespace

import pytest

from runtime import long_horizon_desktop as desktop
from runtime import verify_long_horizon_native as runner
from runtime.acceptance_native_support import OwnedWindow
from runtime.long_horizon_cases import load_case


def test_supplement_is_explicit_and_baseline_never_queries_mime(monkeypatch, tmp_path):
    args = runner.parser().parse_args(["--output", str(tmp_path / "run")])
    assert args.include_default_text_editor is False
    monkeypatch.setattr(runner, "command", lambda *_: pytest.fail("No default query without opt-in"))
    condition = runner.host_editor_condition(load_case("LHN-059"), "gnome-text-editor", False, {})
    assert not condition["supplemental_blank_editor_planned"] and not condition["default_handler_verified"]
    assert condition["text_plain_default_desktop_id"] is None


@pytest.mark.parametrize(("editor", "handler"), [
    ("gnome-text-editor", "org.gnome.TextEditor.desktop"), ("gedit", "org.gnome.gedit.desktop")])
def test_exact_selected_native_handler_is_checked_read_only(monkeypatch, editor, handler):
    calls = []
    monkeypatch.setattr(runner, "command", lambda argv, env: calls.append((argv, env)) or handler)
    condition = runner.host_editor_condition(load_case("LHN-059"), editor, True, {"DISPLAY": ":fixture"})
    assert condition["default_handler_verified"] and condition["text_plain_default_desktop_id"] == handler
    assert condition["supplemental_blank_editor_planned"] and condition["planned_owned_window_count"] == 6
    assert calls == [(["xdg-mime", "query", "default", "text/plain"], {"DISPLAY": ":fixture"})]


@pytest.mark.parametrize("handler", ["", "unknown.desktop", "org.gnome.gedit.desktop",
    "ORG.GNOME.TextEditor.desktop", "org.gnome.TextEditor.desktop\nother.desktop", "secret-canary"])
def test_unknown_mismatched_or_ambiguous_default_fails_without_echo(monkeypatch, handler):
    monkeypatch.setattr(runner, "command", lambda *_: handler)
    with pytest.raises(desktop.CapabilityBlocked, match="^default_text_editor_handler_mismatch$"):
        runner.host_editor_condition(load_case("LHN-059"), "gnome-text-editor", True, {})


@pytest.mark.parametrize("error", [FileNotFoundError(), RuntimeError("secret-canary"),
                                  subprocess.TimeoutExpired("private-command", 3)])
def test_unavailable_default_check_fails_closed_and_redacted(monkeypatch, error):
    def unavailable(*_):
        raise error
    monkeypatch.setattr(runner, "command", unavailable)
    with pytest.raises(desktop.CapabilityBlocked, match="^default_text_editor_handler_unavailable$"):
        runner.host_editor_condition(load_case("LHN-059"), "gnome-text-editor", True, {})


def setup_fixture(monkeypatch, tmp_path, *, failed_observation=False):
    owned = desktop.TaskDesktop({}, editor="gnome-text-editor")
    calls, present, closed = [], {"9"}, []  # An older window shares the same PID.
    def open_blank(app, *_args, **_kwargs):
        calls.append(app)
        window = OwnedWindow(app, str(100 + len(calls)), "42", "123")
        owned.windows.append(window)
        present.add(window.window_id)
        owned.launches.append({"application": app, "window_class": "fixture", "registered": True,
                               "window_id": window.window_id, "pid": "42", "process_start": "123"})
        return window
    def fresh():
        window = owned.windows[-1]
        return {"window_id": window.window_id, "state": {"scope": {"window_id": window.window_id},
                "nodes": [] if failed_observation else [{"id": "fixture-control"}]}}
    def close(xid, *_):
        closed.append(xid)
        present.remove(xid)
        return True
    monkeypatch.setattr(owned, "open_blank", open_blank)
    monkeypatch.setattr(runner, "command", lambda *_: "42")
    monkeypatch.setattr(runner, "process_start", lambda *_: "123")
    monkeypatch.setattr(desktop, "command", lambda argv, _env: "42" if argv[1] == "getwindowpid" else "")
    monkeypatch.setattr(desktop, "process_start", lambda *_: "123")
    monkeypatch.setattr(desktop, "class_windows", lambda *_args, **_kwargs: present.copy())
    monkeypatch.setattr(desktop, "request_owned_window_close", close)
    return owned, SimpleNamespace(fresh=fresh), calls, present, closed


def test_one_supplement_has_fresh_scope_exact_cleanup_and_no_task_changes(monkeypatch, tmp_path):
    case = load_case("LHN-059")
    original = copy.deepcopy(case)
    owned, observer, calls, present, closed = setup_fixture(monkeypatch, tmp_path)
    receipts = runner.prepare_blank_apps(owned, observer, case, tmp_path / "Workbench", tmp_path,
                                        "firefox", supplement_editor=True)
    assert calls == ["gedit", *original["fixture"]["blank_apps"]]
    assert receipts[0]["setup_origin"] == "host_default_text_editor"
    assert receipts[0]["declared_application"] is None
    assert receipts[0]["effective_application"] == "gnome-text-editor"
    scope = runner.registered_native_scope(owned)
    assert len(scope["windows"]) == 6 and "9" not in {row["window_id"] for row in scope["windows"]}
    assert not owned.is_owned({"window_id": "9", "process_id": 42})
    goal = original["goal_text"]
    payload = runner.task_payload("exact/model", goal, native_scope=scope)
    assert {k: v for k, v in payload.items() if k != "portal_native_task_scope"} == runner.task_payload("exact/model", goal)
    assert "portal_native_task_scope" not in runner.task_payload("exact/model", "Recall", recall=True)
    assert case == original
    assert all(row["closed"] for row in owned.close()) and len(closed) == 6 and present == {"9"}


def test_declared_editor_is_not_duplicated(monkeypatch, tmp_path):
    case = load_case("LHN-021")
    monkeypatch.setattr(runner, "command", lambda *_: "org.gnome.TextEditor.desktop")
    assert not runner.host_editor_condition(case, "gnome-text-editor", True, {})["supplemental_blank_editor_planned"]
    owned, observer, calls, _, _ = setup_fixture(monkeypatch, tmp_path)
    receipts = runner.prepare_blank_apps(owned, observer, case, tmp_path, tmp_path, "firefox", supplement_editor=True)
    assert calls.count("gedit") == 1 and all(row["setup_origin"] == "declared_fixture" for row in receipts)


def test_supplement_requires_native_admission_before_other_apps(monkeypatch, tmp_path):
    owned, observer, calls, _, _ = setup_fixture(monkeypatch, tmp_path, failed_observation=True)
    with pytest.raises(desktop.CapabilityBlocked, match="owned_native_setup_observation_unavailable:gedit"):
        runner.prepare_blank_apps(owned, observer, load_case("LHN-059"), tmp_path, tmp_path,
                                  "firefox", supplement_editor=True)
    assert calls == ["gedit"]


def test_supplement_cannot_exceed_scope_limit_before_any_lookup(monkeypatch):
    case = load_case("LHN-059")
    case["fixture"]["blank_app_window_counts"]["browser"] = 12  # Existing 16 plus supplement.
    monkeypatch.setattr(runner, "command", lambda *_: pytest.fail("No query or launch above scope limit"))
    with pytest.raises(desktop.CapabilityBlocked, match="owned_native_setup_window_limit"):
        runner.host_editor_condition(case, "gnome-text-editor", True, {})


def test_supplement_accepts_exact_scope_limit(monkeypatch):
    case = load_case("LHN-059")
    case["fixture"]["blank_app_window_counts"]["browser"] = 11
    monkeypatch.setattr(runner, "command", lambda *_: "org.gnome.TextEditor.desktop")
    assert runner.host_editor_condition(case, "gnome-text-editor", True, {})["planned_owned_window_count"] == 16
