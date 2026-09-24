from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import indicator as indicator_module  # noqa: E402
from harness.indicator import (
    MAX_ACTION_PREVIEW_CHARS,
    MAX_MENU_WIDTH_CHARS,
    MAX_VISIBLE_TASKS,
    IndicatorUnavailable,
    build_indicator,
    model_views,
    task_views,
)


def _indicator_arguments() -> dict:
    return {
        "on_mute": lambda _value: None,
        "on_quit": lambda: None,
    }


def test_required_indicator_fails_instead_of_silently_running_headless(monkeypatch) -> None:
    monkeypatch.setattr(
        indicator_module,
        "_indicator_modules",
        lambda: (_ for _ in ()).throw(IndicatorUnavailable("missing GTK")),
    )

    with pytest.raises(IndicatorUnavailable, match="missing GTK"):
        build_indicator(**_indicator_arguments(), required=True)


def test_optional_indicator_can_still_run_headless(monkeypatch) -> None:
    monkeypatch.setattr(
        indicator_module,
        "_indicator_modules",
        lambda: (_ for _ in ()).throw(IndicatorUnavailable("missing GTK")),
    )

    result = build_indicator(**_indicator_arguments())

    assert result.backend == "headless"


def test_required_indicator_also_requires_a_desktop_status_notifier(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        indicator_module,
        "_indicator_modules",
        lambda: (sentinel, sentinel, sentinel, sentinel, "ayatana-appindicator3"),
    )
    monkeypatch.setattr(
        indicator_module,
        "_require_status_notifier",
        lambda _glib: (_ for _ in ()).throw(
            IndicatorUnavailable("no StatusNotifierWatcher")
        ),
    )

    with pytest.raises(IndicatorUnavailable, match="StatusNotifierWatcher"):
        build_indicator(**_indicator_arguments(), required=True)


def test_running_tasks_lead_and_expose_stage_steps_and_tools() -> None:
    views = task_views(
        [
            {
                "task_id": "done",
                "objective": "Older completed work",
                "status": "completed",
                "updated_at": 20,
                "progress": ["Finished."],
                "tools_used": ["shell"],
                "result": "Verified.",
            },
            {
                "task_id": "live",
                "objective": "Create the requested audio file",
                "status": "running",
                "updated_at": 10,
                "current_stage": "Running shell",
                "progress": ["Accepted.", "Inspected the destination."],
                "tools_used": ["tool_search", "shell"],
                "actions": [
                    {
                        "at": 1_700_000_000,
                        "tool": "shell",
                        "arguments": '{"command": "ls -l Desktop"}',
                        "outcome": '{"exit_code": 0}',
                        "ok": True,
                    }
                ],
            },
        ]
    )

    assert [view["task_id"] for view in views] == ["live", "done"]
    assert views[0]["label"].startswith("● Create the requested audio file")
    assert views[0]["current_stage"] == "Running shell"
    assert views[0]["steps"] == ["Accepted.", "Inspected the destination."]
    assert views[0]["tools"] == ["tool_search", "shell"]
    assert "shell" in views[0]["actions"][0]["label"]
    assert "ls -l Desktop" in views[0]["actions"][0]["label"]
    assert '"exit_code": 0' in views[0]["actions"][0]["label"]
    assert views[0]["actions"][0]["ok"] is True
    assert views[0]["actions"][0]["tooltip"].endswith('→ {"exit_code": 0}')
    assert views[1]["result"] == "Verified."


def test_old_task_steps_recover_explicit_tool_names_for_display() -> None:
    views = task_views(
        [
            {
                "task_id": "old",
                "objective": "Old work",
                "status": "completed",
                "progress": [
                    "Ran shell step (ffmpeg input output); exit=0.",
                    "Ran web_fetch and retained its result.",
                ],
            }
        ]
    )

    assert views[0]["tools"] == ["shell", "web_fetch"]


def test_task_menu_is_bounded_and_keeps_the_newest_live_work() -> None:
    tasks = [
        {
            "task_id": f"task-{index}",
            "objective": f"Task {index}",
            "status": "running" if index == 0 else "completed",
            "updated_at": index,
        }
        for index in range(MAX_VISIBLE_TASKS + 4)
    ]

    views = task_views(tasks)

    assert len(views) == MAX_VISIBLE_TASKS
    assert views[0]["task_id"] == "task-0"
    assert views[1]["task_id"] == f"task-{MAX_VISIBLE_TASKS + 3}"


def test_exact_tool_calls_are_nested_beneath_each_task() -> None:
    source = inspect.getsource(build_indicator)

    assert 'label=f"Tool calls — latest {len(view[\'actions\'])}"' in source
    assert "action_header.set_submenu(action_menu)" in source
    assert 'marker_text="✓" if action["ok"] else "!"' in source


def test_tool_call_rows_are_bounded_but_keep_full_tooltip_text() -> None:
    command = "python -c " + "print('wide') " * 80
    outcome = "result " * 80
    action = task_views(
        [
            {
                "task_id": "wide",
                "objective": "Show a wide action",
                "status": "running",
                "actions": [
                    {
                        "tool": "shell",
                        "arguments": command,
                        "outcome": outcome,
                        "ok": True,
                    }
                ],
            }
        ]
    )[0]["actions"][0]

    assert len(action["label"]) <= MAX_ACTION_PREVIEW_CHARS
    assert command in action["tooltip"]
    assert outcome.strip() in action["tooltip"]
    source = inspect.getsource(build_indicator)
    assert "label.set_line_wrap(True)" in source
    assert "label.set_max_width_chars(MAX_MENU_WIDTH_CHARS)" in source
    assert MAX_MENU_WIDTH_CHARS < 70


def test_indicator_exposes_the_complete_managed_model_lifecycle() -> None:
    source = inspect.getsource(build_indicator)

    assert 'Gtk.MenuItem(label="Models")' in source
    assert '(("Download", "download"' in source
    assert '"Activate",' in source
    assert '"activate",' in source
    assert '"Load into Ollama"' in source
    assert '"Unload from Ollama"' in source
    assert '"Delete local copy…"' in source
    assert model_views([]) == []


def test_indicator_exposes_live_camera_view_and_repository_updates() -> None:
    source = inspect.getsource(build_indicator)

    assert 'Gtk.MenuItem(label="Open live camera view")' in source
    assert 'Gtk.MenuItem(label="Checking for software updates…")' in source
    assert "Update {available} available — install and restart" in source
    assert "Software update failed — retry" in source
    assert "on_open_camera_view()" in source
    assert "on_update()" in source
