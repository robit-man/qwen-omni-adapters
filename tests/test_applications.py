from __future__ import annotations

import json
from pathlib import Path

from qwen_omni_adapters import cli
from qwen_omni_adapters.applications import APPLICATION_SCHEMA, discover_applications


def _launcher(
    root: Path,
    desktop_id: str,
    *,
    name: str,
    executable: str = "",
    categories: str = "",
    mime_types: str = "",
    hidden: bool = False,
    no_display: bool = False,
) -> Path:
    path = root / desktop_id
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        "[Desktop Entry]",
        "Type=Application",
        f"Name={name}",
        f"Exec={executable} %U" if executable else "DBusActivatable=true",
        f"Categories={categories}",
        f"MimeType={mime_types}",
        f"Hidden={'true' if hidden else 'false'}",
        f"NoDisplay={'true' if no_display else 'false'}",
    ]
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_discovers_launchable_media_app_and_default_handler(tmp_path: Path) -> None:
    applications = tmp_path / "applications"
    binaries = tmp_path / "bin"
    executable = _executable(binaries, "sample-player")
    _launcher(
        applications,
        "org.example.Sample.desktop",
        name="Sample Videos",
        executable=executable.name,
        categories="AudioVideo;Player;Video;",
        mime_types="video/mp4;video/webm;",
    )

    result = discover_applications(
        query="media player",
        mime_type="video/mp4",
        directories=[applications],
        environment={"PATH": str(binaries), "HOME": str(tmp_path)},
        default_handler=lambda value: (
            "org.example.Sample.desktop" if value == "video/mp4" else ""
        ),
    )

    assert result["schema"] == APPLICATION_SCHEMA
    assert result["matched"] == 1
    assert result["default_desktop_id"] == "org.example.Sample.desktop"
    application = result["applications"][0]
    assert application["name"] == "Sample Videos"
    assert application["executable_path"] == str(executable)
    assert application["launchable"] is True
    assert application["default_for_mime"] is True


def test_user_hidden_tombstone_suppresses_lower_precedence_launcher(
    tmp_path: Path,
) -> None:
    user = tmp_path / "user"
    system = tmp_path / "system"
    _launcher(user, "same.desktop", name="Hidden override", hidden=True)
    _launcher(system, "same.desktop", name="System application")

    result = discover_applications(
        directories=[user, system],
        environment={"PATH": "", "HOME": str(tmp_path)},
    )

    assert result["applications"] == []


def test_user_override_wins_and_no_display_is_opt_in(tmp_path: Path) -> None:
    user = tmp_path / "user"
    system = tmp_path / "system"
    _launcher(user, "same.desktop", name="User choice")
    _launcher(system, "same.desktop", name="System choice")
    _launcher(user, "internal.desktop", name="Internal helper", no_display=True)

    visible = discover_applications(
        directories=[user, system],
        environment={"PATH": "", "HOME": str(tmp_path)},
    )
    all_entries = discover_applications(
        directories=[user, system],
        environment={"PATH": "", "HOME": str(tmp_path)},
        include_no_display=True,
    )

    assert [item["name"] for item in visible["applications"]] == ["User choice"]
    assert {item["name"] for item in all_entries["applications"]} == {
        "Internal helper",
        "User choice",
    }


def test_cli_emits_machine_readable_dynamic_inventory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_home = tmp_path / "share"
    applications = data_home / "applications"
    _launcher(applications, "org.example.Editor.desktop", name="Text Workbench")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "empty-system"))

    assert cli.main(["applications", "--query", "workbench", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["matched"] == 1
    assert result["applications"][0]["desktop_id"] == "org.example.Editor.desktop"
