from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from harness.update import RepositoryUpdateManager


def _git(path: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    deployed = tmp_path / "deployed"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(seed)], check=True, capture_output=True
    )
    _git(seed, "config", "user.email", "test@example.invalid")
    _git(seed, "config", "user.name", "Test User")
    (seed / "version.txt").write_text("one\n", encoding="utf-8")
    _git(seed, "add", "version.txt")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), str(deployed)],
        check=True,
        capture_output=True,
    )
    return seed, remote, deployed


def test_indicator_update_fast_forwards_preserves_untracked_and_requests_restart(
    tmp_path: Path,
) -> None:
    seed, remote, deployed = _checkout(tmp_path)
    installed = threading.Event()
    manager = RepositoryUpdateManager(
        deployed,
        install_command=["/bin/true"],
        on_installed=installed.set,
        autostart=False,
    )

    assert manager.check_now()["state"] == "current"
    (deployed / "local-notes.txt").write_text("preserve me\n", encoding="utf-8")
    (seed / "version.txt").write_text("two\n", encoding="utf-8")
    _git(seed, "add", "version.txt")
    _git(seed, "commit", "-m", "update")
    _git(seed, "push", str(remote), "main")

    available = manager.check_now()
    assert available["state"] == "available"
    assert available["actionable"] is True
    ok, detail = manager.install()
    assert ok is True
    assert "started" in detail.lower()
    assert installed.wait(5)
    deadline = time.monotonic() + 5
    while manager.view()["state"] != "installed" and time.monotonic() < deadline:
        time.sleep(0.02)

    assert manager.view()["state"] == "installed"
    assert _git(deployed, "rev-parse", "HEAD") == _git(seed, "rev-parse", "HEAD")
    assert (deployed / "local-notes.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert (deployed / "runtime-data/state/restart.request").is_file()
    manager.close()


def test_indicator_update_refuses_to_overwrite_tracked_local_changes(
    tmp_path: Path,
) -> None:
    seed, remote, deployed = _checkout(tmp_path)
    (seed / "version.txt").write_text("upstream\n", encoding="utf-8")
    _git(seed, "add", "version.txt")
    _git(seed, "commit", "-m", "upstream")
    _git(seed, "push", str(remote), "main")
    (deployed / "version.txt").write_text("local\n", encoding="utf-8")
    manager = RepositoryUpdateManager(deployed, autostart=False)

    state = manager.check_now()

    assert state["state"] == "blocked"
    assert state["actionable"] is False
    assert "tracked local changes" in state["detail"]
    manager.close()
