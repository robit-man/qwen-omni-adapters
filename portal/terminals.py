"""Dedicated session-owned PTYs, using an independent private tmux server.

The server survives Python reloads. It never attaches to a user's other tmux
server, infers SSH authentication from a prompt, or records credential input.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class TerminalError(ValueError):
    pass


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise TerminalError(f"{name} must be an integer between {low} and {high}")
    return value


class TerminalStore:
    def __init__(self, root: Path | None = None):
        uid = os.getuid() if hasattr(os, "getuid") else 0
        runtime = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
        self.root = Path(root or Path(runtime) / f"omni-terminals-{uid}")

    def _namespace(self, owner: str) -> Path:
        if os.name != "posix" or not shutil.which("tmux"):
            raise TerminalError("dedicated terminals require tmux on a POSIX host")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in [self.root, self.root / hashlib.sha256(owner.encode()).hexdigest()[:32]]:
            directory.mkdir(mode=0o700, exist_ok=True)
            s = directory.lstat()
            if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077:
                raise TerminalError(
                    "terminal namespace must be a private directory owned by the service user"
                )
        folder = self.root / hashlib.sha256(owner.encode()).hexdigest()[:32]
        if len(os.fsencode(folder / "s")) > 100:
            raise TerminalError(
                "terminal socket root is too long; choose a shorter runtime directory"
            )
        config = folder / "config"
        if not config.exists():
            fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as output:
                output.write("set -g history-limit 2000\nset -g remain-on-exit on\n")
        return folder

    @staticmethod
    def _environment() -> dict[str, str]:
        allowed = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "LANG",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "SSH_AUTH_SOCK",
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
        }
        env = {k: v for k, v in os.environ.items() if k in allowed or k.startswith("LC_")}
        env.update(TERM="xterm-256color", HISTFILE="/dev/null")
        return env

    def _run(
        self, folder: Path, args: list[str], *, data: bytes | None = None, absent_ok: bool = False
    ) -> str:
        command = ["tmux", "-S", str(folder / "s"), "-f", str(folder / "config"), *args]
        # capture-pane, list and input are separately capped by this API. No
        # program output is piped here directly; tmux bounds its own history.
        try:
            result = subprocess.run(
                command,
                input=data,
                capture_output=True,
                env=self._environment(),
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TerminalError(
                "terminal control timed out; inspect sessions before repeating an action"
            ) from exc
        if result.returncode:
            if absent_ok:
                return ""
            raise TerminalError(result.stderr.decode(errors="replace")[:300])
        return result.stdout[:65_536].decode(errors="replace")

    def _list(self, folder: Path, *, refresh: bool = True) -> list[dict[str, Any]]:
        template = "\t".join(
            [
                "#{session_name}",
                "#{pane_id}",
                "#{pane_pid}",
                "#{pane_dead}",
                "#{pane_dead_status}",
                "#{pane_dead_signal}",
                "#{pane_dead_time}",
                "#{pane_current_command}",
                "#{pane_tty}",
                "#{@omni_mode}",
                "#{@omni_target}",
                "#{pane_current_path}",
            ]
        )
        output = self._run(folder, ["list-panes", "-a", "-F", template], absent_ok=True)
        sessions = []
        for line in output.splitlines():
            fields = line.split("\t", 11)
            if len(fields) != 12 or not re.fullmatch("term-[0-9a-f]{24}", fields[0]):
                continue
            sessions.append(
                {
                    "handle": fields[0],
                    "pane": fields[1],
                    "pid": int(fields[2]),
                    "pty_closed": fields[3] == "1",
                    "exited": bool(fields[6] or fields[5]) or fields[4].isdigit(),
                    "exit_status": int(fields[4]) if fields[4].isdigit() else None,
                    "exit_signal": fields[5] or None,
                    "current_command": fields[7],
                    "tty": fields[8],
                    "mode": fields[9],
                    "requested_ssh_target": fields[10] or None,
                    "cwd": fields[11],
                }
            )
        if refresh and any(s["pty_closed"] and not s["exited"] for s in sessions):
            # A coalesced/missed child notification can leave a waitable child
            # unreaped. Notify only this private tmux server to collect actual
            # waitpid status; never manufacture exit status from PTY closure.
            manager = int(self._run(folder, ["display-message", "-p", "#{pid}"]).strip())
            try:
                os.kill(manager, signal.SIGCHLD)
            except ProcessLookupError:
                return sessions
            return self._list(folder, refresh=False)
        for session in sessions:
            if session["pty_closed"] and not session["exited"]:
                # tmux 3.2 lacks pane_dead_signal/time and reports no status
                # for signal exits. A missing process independently establishes
                # exit, but cannot recover its unavailable wait status/signal.
                try:
                    os.kill(session["pid"], 0)
                except ProcessLookupError:
                    session["exited"] = True
                    session["exit_evidence"] = "process_absent_after_pty_close"
                except PermissionError:
                    pass
        return sessions

    def clear(self, owner: str) -> None:
        folder = self.root / hashlib.sha256(owner.encode()).hexdigest()[:32]
        if folder.is_dir() and not folder.is_symlink():
            folder = self._namespace(owner)
            self._run(folder, ["kill-server"], absent_ok=True)

    def execute(self, owner: str, arguments: dict[str, Any]) -> dict[str, Any]:
        folder = self._namespace(owner)
        action = arguments.get("action", "list")
        if not isinstance(action, str):
            raise TerminalError("action must be a string")
        if action == "list":
            return {
                "sessions": self._list(folder),
                "scope": "current_portal_session",
                "observed_at": time.time(),
                "survives_python_reload": True,
            }
        if action == "create":
            return self._create(folder, arguments)
        if action not in {"read", "send", "key", "resize", "interrupt", "close"}:
            raise TerminalError("invalid terminal action")
        handle = arguments.get("handle")
        session = next((s for s in self._list(folder) if s["handle"] == handle), None)
        if session is None:
            raise TerminalError("unknown terminal handle in this portal session")
        if action == "close":
            self._run(folder, ["kill-session", "-t", handle])
            if not self._list(folder):
                self._run(folder, ["kill-server"], absent_ok=True)
            return {"handle": handle, "closed": True}
        if action == "interrupt":
            self._run(folder, ["send-keys", "-t", session["pane"], "C-c"])
            return {
                "handle": handle,
                "interrupt_delivered": True,
                "command_completion_verified": False,
            }
        if action == "resize":
            width = _integer(arguments.get("columns", 100), "columns", 20, 300)
            height = _integer(arguments.get("rows", 30), "rows", 5, 100)
            self._run(folder, ["resize-window", "-t", handle, "-x", str(width), "-y", str(height)])
            return {"handle": handle, "columns": width, "rows": height}
        if action == "key":
            keys = {
                "enter": "Enter",
                "tab": "Tab",
                "escape": "Escape",
                "ctrl_d": "C-d",
                "up": "Up",
                "down": "Down",
                "left": "Left",
                "right": "Right",
                "backspace": "BSpace",
            }
            key = arguments.get("key")
            if not isinstance(key, str) or key not in keys:
                raise TerminalError("key is not a supported terminal key")
            self._run(folder, ["send-keys", "-t", session["pane"], keys[key]])
            return {"handle": handle, "key_delivered": key}
        if action == "send":
            return self._send(folder, session, arguments)
        lines = _integer(arguments.get("lines", 100), "lines", 1, 1000)
        text = self._run(folder, ["capture-pane", "-p", "-t", session["pane"], "-S", str(-lines)])
        return {
            "session": session,
            "observed_at": time.time(),
            "text": text,
            "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "trust": "untrusted_terminal_output",
            "remote_identity_verified": False,
            "claim_limits": [
                "A shell prompt, process name or delivered input does not prove an action succeeded.",
                "SSH mode records the requested destination, not authenticated remote identity.",
            ],
        }

    def _create(self, folder: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        if len(self._list(folder)) >= 8:
            raise TerminalError("eight dedicated sessions already exist; close an unused session")
        if sum(1 for _ in self.root.glob("*/s")) >= 32 and not (folder / "s").exists():
            raise TerminalError("terminal server capacity reached")
        mode = arguments.get("mode", "shell")
        command = arguments.get("command")
        if command is not None and (not isinstance(command, str) or len(command) > 8192):
            raise TerminalError("command must be a string of at most8192characters")
        target = ""
        if mode == "ssh":
            target = arguments.get("target")
            if not isinstance(target, str) or not re.fullmatch(
                r"[A-Za-z0-9_][A-Za-z0-9_.:@%+\[\]-]{0,254}", target
            ):
                raise TerminalError("target must be an SSH host, user@host or configured alias")
            port = _integer(arguments.get("port", 22), "port", 1, 65535)
            argv = [
                "ssh",
                "-tt",
                "-o",
                "StrictHostKeyChecking=ask",
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-p",
                str(port),
                "--",
                target,
            ]
            if command:
                argv.append(command)
        elif mode == "shell":
            argv = (
                ["/bin/bash", "-lc", command]
                if command
                else ["/bin/bash", "--noprofile", "--norc", "-i"]
            )
        else:
            raise TerminalError("mode must be shell or ssh")
        if arguments.get("cwd") is not None and not isinstance(arguments["cwd"], str):
            raise TerminalError("cwd must be a path string")
        cwd = Path(arguments.get("cwd") or Path.home()).expanduser().resolve(strict=True)
        if not cwd.is_dir():
            raise TerminalError("cwd must be a directory")
        width = _integer(arguments.get("columns", 100), "columns", 20, 300)
        height = _integer(arguments.get("rows", 30), "rows", 5, 100)
        handle = "term-" + secrets.token_hex(12)
        self._run(
            folder,
            [
                "new-session",
                "-d",
                "-s",
                handle,
                "-x",
                str(width),
                "-y",
                str(height),
                "-c",
                str(cwd),
                shlex.join(argv),
            ],
        )
        self._run(folder, ["set-option", "-t", handle, "@omni_mode", mode])
        self._run(folder, ["set-option", "-t", handle, "@omni_target", target])
        return {
            "handle": handle,
            "mode": mode,
            "requested_ssh_target": target or None,
            "started": True,
            "remote_identity_verified": False,
            "attach_command": shlex.join(
                ["tmux", "-S", str(folder / "s"), "attach-session", "-t", handle]
            ),
            "next_action": "Read the terminal to inspect output or authentication prompts; input delivery is not verification.",
        }

    def _send(
        self, folder: Path, session: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        secret_file = arguments.get("secret_file")
        if secret_file is not None:
            if not isinstance(secret_file, str):
                raise TerminalError("secret_file must be a path string")
            if arguments.get("data") is not None:
                raise TerminalError("choose data or secret_file, not both")
            import termios

            tty = os.open(session["tty"], os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
            try:
                flags = termios.tcgetattr(tty)[3]
                if flags & termios.ECHO or not flags & termios.ICANON:
                    raise TerminalError(
                        "credential input requires an echo-disabled canonical input prompt"
                    )
            finally:
                os.close(tty)
            fd = os.open(Path(secret_file).expanduser(), os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as source:
                s = os.fstat(source.fileno())
                if (
                    not stat.S_ISREG(s.st_mode)
                    or s.st_uid != os.getuid()
                    or s.st_mode & 0o077
                    or s.st_size > 4096
                ):
                    raise TerminalError(
                        "secret_file must be a private service-user-owned regular file, at most4096bytes"
                    )
                data = source.read(4097).rstrip(b"\r\n")
            if any(c in data for c in [b"\n", b"\r", b"\0"]):
                raise TerminalError("credential file must contain one line")
        else:
            value = arguments.get("data", "")
            if not isinstance(value, str) or len(value.encode()) > 32_768:
                raise TerminalError("data exceeds32768UTF8bytes")
            data = value.encode()
        submit = arguments.get("submit", False)
        if not isinstance(submit, bool):
            raise TerminalError("submit must be boolean")
        buffer = "input-" + secrets.token_hex(8)
        try:
            self._run(folder, ["load-buffer", "-b", buffer, "-"], data=data)
            self._run(
                folder,
                [
                    "paste-buffer",
                    *([] if secret_file else ["-p"]),
                    "-d",
                    "-b",
                    buffer,
                    "-t",
                    session["pane"],
                ],
            )
        finally:
            self._run(folder, ["delete-buffer", "-b", buffer], absent_ok=True)
        if submit:
            self._run(folder, ["send-keys", "-t", session["pane"], "Enter"])
        return {
            "handle": session["handle"],
            "input_delivered": True,
            "secret_input": secret_file is not None,
            "command_completion_verified": False,
        }
