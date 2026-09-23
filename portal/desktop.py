"""Recover the signed-in user's desktop environment for supervised tools.

The portal is intentionally a system service, while Chromium and desktop
automation must join the graphical session owned by that same Unix user.
Systemd does not copy DISPLAY or the session-bus address into a system unit,
so inheriting ``os.environ`` alone makes desktop tools fail even while the
top-bar harness is visibly running.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path


def _unix_sockets(directory: Path, pattern: str) -> list[Path]:
    sockets: list[Path] = []
    try:
        candidates = sorted(directory.glob(pattern))
    except OSError:
        return sockets
    for candidate in candidates:
        try:
            if stat.S_ISSOCK(candidate.stat().st_mode):
                sockets.append(candidate)
        except OSError:
            continue
    return sockets


def desktop_subprocess_environment(
    base: Mapping[str, str] | None = None,
    *,
    runtime_dir: Path | None = None,
    x11_dir: Path = Path("/tmp/.X11-unix"),
    home: Path | None = None,
) -> dict[str, str]:
    """Return an environment connected to the current user's desktop.

    Existing explicit values always win. Missing values are derived only from
    standard sockets available to the service user, and an ambiguous display
    is left unset rather than guessing which desktop to control.
    """

    environment = dict(os.environ if base is None else base)
    if os.name != "posix" or not hasattr(os, "getuid"):
        return environment

    resolved_runtime = runtime_dir
    if resolved_runtime is None:
        configured_runtime = environment.get("XDG_RUNTIME_DIR", "").strip()
        resolved_runtime = (
            Path(configured_runtime)
            if configured_runtime
            else Path("/run/user") / str(os.getuid())
        )
    if resolved_runtime.is_dir():
        environment.setdefault("XDG_RUNTIME_DIR", str(resolved_runtime))
        bus = resolved_runtime / "bus"
        try:
            bus_available = stat.S_ISSOCK(bus.stat().st_mode)
        except OSError:
            bus_available = False
        if bus_available:
            environment.setdefault(
                "DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}"
            )

        if not environment.get("WAYLAND_DISPLAY"):
            wayland = _unix_sockets(resolved_runtime, "wayland-*")
            if len(wayland) == 1:
                environment["WAYLAND_DISPLAY"] = wayland[0].name

    if not environment.get("DISPLAY"):
        x11 = _unix_sockets(x11_dir, "X*")
        if len(x11) == 1 and x11[0].name[1:].isdigit():
            environment["DISPLAY"] = f":{x11[0].name[1:]}"

    if not environment.get("XAUTHORITY"):
        resolved_home = home or Path.home()
        candidates = []
        if resolved_runtime.is_dir():
            candidates.append(resolved_runtime / "gdm" / "Xauthority")
        candidates.append(resolved_home / ".Xauthority")
        for candidate in candidates:
            if candidate.is_file():
                environment["XAUTHORITY"] = str(candidate)
                break

    return environment
