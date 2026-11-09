"""Real private tmux sessions: persistence, isolation, literal input and secrets."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from portal.terminals import TerminalError, TerminalStore


@pytest.fixture
def terminals():
    if not shutil.which("tmux"):
        pytest.skip("tmux is not installed")
    # UNIX socket paths must fit108bytes even with long pytest test names.
    with tempfile.TemporaryDirectory(prefix="omni-term-test-") as root:
        store = TerminalStore(Path(root) / "t")
        try:
            yield store
        finally:
            for owner in ["alice", "bob"]:
                store.clear(owner)


def await_text(store, owner, handle, marker):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = store.execute(owner, {"action": "read", "handle": handle})
        if marker in result["text"]:
            return result
        time.sleep(0.03)
    pytest.fail("Expected terminal evidence did not arrive")


def test_real_dedicated_shell_survives_new_manager_and_reports_actual_exit(terminals):
    first = terminals.execute(
        "alice", {"action": "create", "command": "printf 'PERSISTENT-RECEIPT\\n'; exit 7"}
    )
    handle = first["handle"]
    resumed = TerminalStore(terminals.root)
    result = await_text(resumed, "alice", handle, "PERSISTENT-RECEIPT")
    deadline = time.monotonic() + 2
    while not result["session"]["exited"] and time.monotonic() < deadline:
        result = resumed.execute("alice", {"action": "read", "handle": handle})
    assert result["session"]["exited"] and result["session"]["exit_status"] == 7
    assert result["remote_identity_verified"] is False
    assert resumed.execute("bob", {"action": "list"})["sessions"] == []
    with pytest.raises(TerminalError):
        resumed.execute("bob", {"action": "close", "handle": handle})
    terminals.execute("alice", {"action": "close", "handle": handle})
    assert terminals.execute("alice", {"action": "list"})["sessions"] == []


def test_interactive_literal_input_resize_and_trash(terminals):
    handle = terminals.execute("alice", {"action": "create"})["handle"]
    receipt = terminals.execute(
        "alice",
        {
            "action": "send",
            "handle": handle,
            "data": "printf 'INTERACTIVE-RECEIPT\\n'",
            "submit": True,
        },
    )
    assert receipt["input_delivered"] and receipt["command_completion_verified"] is False
    result = await_text(terminals, "alice", handle, "INTERACTIVE-RECEIPT")
    assert result["trust"] == "untrusted_terminal_output"
    resized = terminals.execute(
        "alice", {"action": "resize", "handle": handle, "columns": 90, "rows": 22}
    )
    assert (resized["columns"], resized["rows"]) == (90, 22)
    terminals.clear("alice")
    with pytest.raises(TerminalError):
        terminals.execute("alice", {"action": "read", "handle": handle})


def test_interrupt_does_not_claim_command_success(terminals):
    handle = terminals.execute("alice", {"action": "create", "command": "sleep 100"})["handle"]
    assert (
        terminals.execute("alice", {"action": "interrupt", "handle": handle})[
            "command_completion_verified"
        ]
        is False
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = terminals.execute("alice", {"action": "read", "handle": handle})
        if result["session"]["exited"]:
            break
        time.sleep(0.03)
    assert result["session"]["exited"]


def test_ssh_target_and_port_are_not_shell_options_or_commands(terminals, monkeypatch, tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "ssh"
    fake.write_text(
        '#!/usr/bin/env python3\nimport json,sys\nprint("SSH-ARGV:"+json.dumps(sys.argv[1:]),flush=True)\n'
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + ":" + os.environ["PATH"])
    first = terminals.execute(
        "alice",
        {"action": "create", "mode": "ssh", "target": "someone@configured-alias", "port": 2222},
    )
    result = await_text(terminals, "alice", first["handle"], "SSH-ARGV:")
    assert "StrictHostKeyChecking=ask" in result["text"]
    assert "someone@configured-alias" in result["text"] and "2222" in result["text"]
    assert first["remote_identity_verified"] is False
    for target in [
        "-oProxyCommand=anything",
        "host; touch /tmp/forbidden",
        "host\ncommand",
        "$(program)",
    ]:
        with pytest.raises(TerminalError):
            terminals.execute("alice", {"action": "create", "mode": "ssh", "target": target})


def test_secret_file_requires_echo_disabled_and_is_not_in_terminal_history(terminals, tmp_path):
    secret = tmp_path / "credential"
    secret.write_text("sample-test-credential\n")
    secret.chmod(0o600)
    normal = terminals.execute("alice", {"action": "create"})["handle"]
    await_text(terminals, "alice", normal, "bash")
    with pytest.raises(TerminalError, match="echo-disabled"):
        terminals.execute(
            "alice",
            {"action": "send", "handle": normal, "secret_file": str(secret), "submit": True},
        )
    program = tmp_path / "prompt.py"
    program.write_text(
        'import sys,termios\na=termios.tcgetattr(0);a[3]&=~termios.ECHO;termios.tcsetattr(0,termios.TCSANOW,a)\nprint("INPUT-READY",flush=True)\ns=input();print("INPUT-LENGTH:"+str(len(s)),flush=True)\n'
    )
    handle = terminals.execute("alice", {"action": "create", "command": f"python3 {program}"})[
        "handle"
    ]
    await_text(terminals, "alice", handle, "INPUT-READY")
    receipt = terminals.execute(
        "alice", {"action": "send", "handle": handle, "secret_file": str(secret), "submit": True}
    )
    assert receipt["secret_input"] and "sample-test-credential" not in str(receipt)
    output = await_text(terminals, "alice", handle, "INPUT-LENGTH:22")
    assert "sample-test-credential" not in output["text"]
    assert terminals._run(terminals._namespace("alice"), ["list-buffers"]) == ""


def test_terminal_child_environment_does_not_inherit_service_credentials(monkeypatch):
    monkeypatch.setenv("OMNI_ACCESS_TOKEN", "must-not-leak")
    monkeypatch.setenv("SOME_OTHER_SERVICE_PASSWORD", "must-not-leak-either")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/user/123/example-agent")
    env = TerminalStore._environment()
    assert "OMNI_ACCESS_TOKEN" not in env and "SOME_OTHER_SERVICE_PASSWORD" not in env
    assert env["SSH_AUTH_SOCK"] == "/run/user/123/example-agent" and "TMUX" not in env


def test_world_readable_or_symlink_namespaces_are_refused(tmp_path):
    root = tmp_path / "untrusted"
    root.mkdir(mode=0o755)
    root.chmod(0o755)  # Keep this negative control independent of caller umask.
    with pytest.raises(TerminalError, match="private directory"):
        TerminalStore(root).execute("alice", {"action": "list"})
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(TerminalError, match="private directory"):
        TerminalStore(alias).execute("alice", {"action": "list"})
