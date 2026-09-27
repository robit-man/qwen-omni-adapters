"""High-signal discovery of launchable Linux desktop applications.

Linux desktop applications are contracts expressed through Freedesktop
``.desktop`` files, not a reliable list of familiar executable names.  This
module indexes the XDG search path plus the standard Flatpak and Snap export
locations, applies user-over-system precedence, honors hidden tombstones, and
reports whether each advertised launcher can actually resolve its executable.
"""

from __future__ import annotations

import configparser
import locale
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

APPLICATION_SCHEMA = "robit.system-applications.v1"
_TOKEN_PATTERN = re.compile(r"[\w.+-]+", re.UNICODE)
_FIELD_CODE_PATTERN = re.compile(r"(?<!%)%[fFuUdDnNickvm]")
_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _unique_paths(values: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = str(value.expanduser())
        if key in seen:
            continue
        seen.add(key)
        found.append(Path(key))
    return found


def application_directories(
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Return desktop-entry roots in Freedesktop precedence order."""

    env = dict(os.environ if environment is None else environment)
    user_home = Path(home or env.get("HOME") or Path.home())
    data_home = Path(env.get("XDG_DATA_HOME") or user_home / ".local" / "share")
    raw_data_dirs = env.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    data_dirs = [Path(value) for value in raw_data_dirs.split(":") if value]
    return _unique_paths(
        [
            data_home / "applications",
            user_home / ".local/share/flatpak/exports/share/applications",
            *[directory / "applications" for directory in data_dirs],
            Path("/var/lib/flatpak/exports/share/applications"),
            Path("/var/lib/snapd/desktop/applications"),
        ]
    )


def _desktop_id(path: Path, root: Path) -> str:
    try:
        return "-".join(path.relative_to(root).parts)
    except ValueError:
        return path.name


def _boolean(section: configparser.SectionProxy, key: str) -> bool:
    try:
        return section.getboolean(key, fallback=False)
    except ValueError:
        return False


def _localized(section: configparser.SectionProxy, key: str) -> str:
    language = (locale.getlocale()[0] or "").replace("-", "_")
    candidates = []
    if language:
        candidates.extend((f"{key}[{language}]", f"{key}[{language.split('_', 1)[0]}]"))
    candidates.append(key)
    for candidate in candidates:
        value = str(section.get(candidate, fallback="") or "").strip()
        if value:
            return value
    return ""


def _split_list(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(";") if item.strip()))


def _exec_command(value: str) -> list[str]:
    cleaned = _FIELD_CODE_PATTERN.sub("", value).replace("%%", "%").strip()
    try:
        return shlex.split(cleaned)
    except ValueError:
        return []


def _launcher_executable(section: configparser.SectionProxy) -> str:
    try_exec = str(section.get("TryExec", fallback="") or "").strip()
    if try_exec:
        return try_exec
    words = _exec_command(str(section.get("Exec", fallback="") or ""))
    if not words:
        return ""
    if Path(words[0]).name != "env":
        return words[0]
    for word in words[1:]:
        if word.startswith("-") or _ASSIGNMENT_PATTERN.match(word):
            continue
        return word
    return ""


def _resolve_executable(value: str, path: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(value, path=path)


def _read_desktop_entry(
    path: Path,
    *,
    desktop_id: str,
    search_path: str | None,
) -> dict[str, Any] | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, UnicodeError, configparser.Error):
        return None
    if not parser.has_section("Desktop Entry"):
        return None
    section = parser["Desktop Entry"]
    if str(section.get("Type", fallback="Application")) != "Application":
        return None
    executable = _launcher_executable(section)
    executable_path = _resolve_executable(executable, search_path)
    dbus_activatable = _boolean(section, "DBusActivatable")
    return {
        "desktop_id": desktop_id,
        "name": _localized(section, "Name") or desktop_id.removesuffix(".desktop"),
        "generic_name": _localized(section, "GenericName"),
        "comment": _localized(section, "Comment"),
        "exec": str(section.get("Exec", fallback="") or "").strip(),
        "try_exec": str(section.get("TryExec", fallback="") or "").strip(),
        "executable": executable,
        "executable_path": executable_path,
        "launchable": bool(executable_path or dbus_activatable),
        "terminal": _boolean(section, "Terminal"),
        "no_display": _boolean(section, "NoDisplay"),
        "hidden": _boolean(section, "Hidden"),
        "dbus_activatable": dbus_activatable,
        "categories": _split_list(str(section.get("Categories", fallback="") or "")),
        "mime_types": _split_list(str(section.get("MimeType", fallback="") or "")),
        "source": str(path),
    }


def _tokens(value: str) -> list[str]:
    return [item.casefold() for item in _TOKEN_PATTERN.findall(value)]


def _query_score(application: Mapping[str, Any], query: str) -> int:
    query_text = query.casefold().strip()
    query_tokens = _tokens(query)
    if not query_tokens:
        return 1
    name = str(application.get("name") or "").casefold()
    desktop_id = str(application.get("desktop_id") or "").casefold()
    executable = str(application.get("executable") or "").casefold()
    primary = " ".join((name, desktop_id, executable))
    categories = [str(item) for item in application.get("categories") or []]
    mime_types = [str(item) for item in application.get("mime_types") or []]
    semantic_terms: list[str] = []
    # Freedesktop expresses media applications as Audio/Video categories and
    # MIME types.  Expose the ordinary umbrella word people use without
    # maintaining any product-name aliases.
    if any(
        category.casefold() in {"audio", "video", "audiovideo", "player"}
        for category in categories
    ) or any(value.casefold().startswith(("audio/", "video/")) for value in mime_types):
        semantic_terms.extend(("media", "multimedia"))
    descriptive = " ".join(
        (
            str(application.get("generic_name") or ""),
            str(application.get("comment") or ""),
            " ".join(categories),
            " ".join(mime_types),
            " ".join(semantic_terms),
        )
    ).casefold()
    haystack = f"{primary} {descriptive}"
    if any(token not in haystack for token in query_tokens):
        return 0
    score = 20 if query_text and query_text in name else 0
    for token in query_tokens:
        if token in name:
            score += 8
        elif token in primary:
            score += 5
        elif token in descriptive:
            score += 2
    return score


def _default_mime_handler(mime_type: str) -> str:
    if not mime_type or shutil.which("xdg-mime") is None:
        return ""
    try:
        completed = subprocess.run(
            ["xdg-mime", "query", "default", mime_type],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def discover_applications(
    *,
    query: str = "",
    mime_type: str = "",
    limit: int = 50,
    include_no_display: bool = False,
    environment: Mapping[str, str] | None = None,
    directories: Sequence[Path] | None = None,
    default_handler: Callable[[str], str] = _default_mime_handler,
) -> dict[str, Any]:
    """Return a bounded, precedence-correct Linux application inventory."""

    env = dict(os.environ if environment is None else environment)
    roots = list(directories or application_directories(environment=env))
    seen_ids: set[str] = set()
    applications: list[dict[str, Any]] = []
    parse_failures = 0
    desktop_files = 0
    for root in roots:
        if not root.is_dir():
            continue
        try:
            candidates = sorted(root.rglob("*.desktop"))
        except OSError:
            continue
        for path in candidates:
            desktop_files += 1
            desktop_id = _desktop_id(path, root)
            if desktop_id in seen_ids:
                continue
            seen_ids.add(desktop_id)
            entry = _read_desktop_entry(
                path,
                desktop_id=desktop_id,
                search_path=env.get("PATH"),
            )
            if entry is None:
                parse_failures += 1
                continue
            # A higher-precedence Hidden entry is a tombstone for any system
            # copy with the same desktop ID.  Keep it in ``seen_ids`` but never
            # advertise it as installed/launchable.
            if entry["hidden"]:
                continue
            if entry["no_display"] and not include_no_display:
                continue
            if mime_type and mime_type not in entry["mime_types"]:
                continue
            score = _query_score(entry, query)
            if score <= 0:
                continue
            entry["match_score"] = score
            applications.append(entry)

    applications.sort(
        key=lambda item: (
            -int(item["match_score"]),
            not bool(item["launchable"]),
            str(item["name"]).casefold(),
            str(item["desktop_id"]),
        )
    )
    bounded_limit = max(1, min(int(limit), 200))
    default_desktop_id = default_handler(mime_type) if mime_type else ""
    selected = applications[:bounded_limit]
    for item in selected:
        item["default_for_mime"] = bool(
            default_desktop_id and item["desktop_id"] == default_desktop_id
        )
        item["mime_type_count"] = len(item["mime_types"])
        item["mime_types"] = item["mime_types"][:24]
    return {
        "schema": APPLICATION_SCHEMA,
        "query": query,
        "mime_type": mime_type,
        "default_desktop_id": default_desktop_id,
        "scanned_directories": [str(root) for root in roots if root.is_dir()],
        "desktop_files_seen": desktop_files,
        "parse_failures": parse_failures,
        "matched": len(applications),
        "returned": len(selected),
        "truncated": len(applications) > len(selected),
        "applications": selected,
    }
