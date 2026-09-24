"""Fast-forward-only repository updates for the desktop indicator.

Polling is read-only apart from Git's remote-tracking ref. Installation happens
only after a person clicks the indicator action, preserves untracked files, and
refuses tracked edits, branch divergence, or a non-main checkout.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 15 * 60


class RepositoryUpdateError(RuntimeError):
    """A repository update could not be proven safe."""


class RepositoryUpdateManager:
    """Poll origin/main and serialize one user-requested installation."""

    def __init__(
        self,
        repo_root: Path,
        *,
        remote: str = "origin",
        branch: str = "main",
        poll_seconds: float | None = None,
        install_command: Sequence[str] | None = None,
        on_installed: Callable[[], None] | None = None,
        autostart: bool = True,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.remote = remote
        self.branch = branch
        try:
            configured_poll = float(
                os.environ.get("OMNI_UPDATE_INTERVAL_SECONDS", DEFAULT_POLL_SECONDS)
            )
        except ValueError:
            configured_poll = DEFAULT_POLL_SECONDS
        self.poll_seconds = max(60.0, poll_seconds or configured_poll)
        self.install_command = list(
            install_command
            or (
                str(self.repo_root / "scripts/bootstrap.sh"),
                "--skip-models",
                "--with-harness",
            )
        )
        self.on_installed = on_installed
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "state": "checking",
            "detail": "Checking origin/main",
            "current": "",
            "available": "",
            "actionable": False,
            "updated_at": time.time(),
        }
        self._operation_running = False
        self._pending_install = False
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        if autostart:
            self._poll_thread = threading.Thread(
                target=self._poll_loop,
                name="omni-repository-update-poll",
                daemon=True,
            )
            self._poll_thread.start()

    def _set_state(self, state: str, detail: str, **fields: Any) -> None:
        with self._lock:
            self._state = {
                **self._state,
                "state": state,
                "detail": " ".join(detail.split())[:240],
                "updated_at": time.time(),
                **fields,
            }

    def view(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(command),
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepositoryUpdateError(str(error)) from error
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RepositoryUpdateError(
                detail[-500:] or f"{' '.join(command)} exited {completed.returncode}"
            )
        return completed

    def _git(self, *arguments: str, timeout: float = 120.0) -> str:
        completed = self._run(
            ("git", "-C", str(self.repo_root), *arguments), timeout=timeout
        )
        return completed.stdout.strip()

    def _inspect(self, *, fetch: bool) -> tuple[str, str, bool, bool]:
        if not (self.repo_root / ".git").exists():
            raise RepositoryUpdateError("checkout has no Git metadata")
        current_branch = self._git("branch", "--show-current")
        if current_branch != self.branch:
            raise RepositoryUpdateError(
                f"automatic updates require branch {self.branch}; current branch is "
                f"{current_branch or 'detached'}"
            )
        self._git("remote", "get-url", self.remote)
        if fetch:
            self._git(
                "fetch",
                "--quiet",
                "--no-tags",
                self.remote,
                f"+refs/heads/{self.branch}:refs/remotes/{self.remote}/{self.branch}",
                timeout=180.0,
            )
        head = self._git("rev-parse", "HEAD")
        target = self._git("rev-parse", f"refs/remotes/{self.remote}/{self.branch}")
        dirty = bool(self._git("status", "--porcelain", "--untracked-files=no"))
        ancestor = self._run(
            (
                "git",
                "-C",
                str(self.repo_root),
                "merge-base",
                "--is-ancestor",
                head,
                target,
            ),
            timeout=30.0,
            check=False,
        )
        if ancestor.returncode not in {0, 1}:
            raise RepositoryUpdateError("could not compare the installed and remote revisions")
        return head, target, dirty, ancestor.returncode == 0

    def check_now(self) -> dict[str, Any]:
        with self._lock:
            if self._operation_running or self._pending_install:
                return dict(self._state)
        self._set_state("checking", f"Checking {self.remote}/{self.branch}", actionable=False)
        try:
            head, target, dirty, ancestor = self._inspect(fetch=True)
            current = head[:8]
            available = target[:8]
            if head == target:
                detail = "Repository is current"
                if dirty:
                    detail += "; tracked local changes are present"
                self._set_state(
                    "current",
                    detail,
                    current=current,
                    available="",
                    actionable=False,
                )
            elif not ancestor:
                self._set_state(
                    "blocked",
                    "Local and origin/main histories diverged; run the guided deploy manually",
                    current=current,
                    available=available,
                    actionable=False,
                )
            elif dirty:
                self._set_state(
                    "blocked",
                    "Update available, but tracked local changes prevent a safe fast-forward",
                    current=current,
                    available=available,
                    actionable=False,
                )
            else:
                self._set_state(
                    "available",
                    f"Update {available} is available",
                    current=current,
                    available=available,
                    actionable=True,
                )
        except RepositoryUpdateError as error:
            self._set_state(
                "failed",
                f"Update check failed: {error}",
                actionable=False,
            )
        return self.view()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self.check_now()
            if self._stop.wait(self.poll_seconds):
                return

    def install(self) -> tuple[bool, str]:
        with self._lock:
            if self._operation_running:
                return False, "An update is already installing"
            retrying = self._pending_install
            if not retrying and not (
                self._state.get("state") == "available"
                and self._state.get("actionable") is True
            ):
                return False, "No verified fast-forward update is ready"
            self._operation_running = True
            self._state = {
                **self._state,
                "state": "installing",
                "detail": "Installing the update",
                "actionable": False,
                "updated_at": time.time(),
            }

        def run() -> None:
            try:
                if not retrying:
                    head, target, dirty, ancestor = self._inspect(fetch=True)
                    if dirty:
                        raise RepositoryUpdateError(
                            "tracked files changed after the update check"
                        )
                    if not ancestor:
                        raise RepositoryUpdateError(
                            "origin/main is no longer a fast-forward"
                        )
                    if head != target:
                        self._git(
                            "merge",
                            "--ff-only",
                            f"refs/remotes/{self.remote}/{self.branch}",
                            timeout=180.0,
                        )
                    with self._lock:
                        self._pending_install = True
                completed = self._run(self.install_command, timeout=7200.0)
                tail = (completed.stdout or completed.stderr or "").strip().splitlines()
                detail = tail[-1] if tail else "Update installed"
                restart = self.repo_root / "runtime-data/state/restart.request"
                restart.parent.mkdir(parents=True, exist_ok=True)
                restart.write_text(
                    json.dumps(
                        {
                            "reason": "repository_update",
                            "requested_at": time.time(),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                restart.chmod(0o600)
                with self._lock:
                    self._pending_install = False
                self._set_state(
                    "installed",
                    f"{detail}; restarting services",
                    actionable=False,
                )
                logger.info("repository update installed; restarting managed services")
                if self.on_installed is not None:
                    try:
                        self.on_installed()
                    except Exception:  # noqa: BLE001 - update itself already succeeded
                        logger.warning(
                            "updated repository but could not reload the indicator",
                            exc_info=True,
                        )
            except (OSError, RepositoryUpdateError) as error:
                with self._lock:
                    pending = self._pending_install
                self._set_state(
                    "failed",
                    f"Update install failed: {error}",
                    actionable=pending,
                )
                logger.warning("repository update failed: %s", error)
            finally:
                with self._lock:
                    self._operation_running = False

        threading.Thread(target=run, name="omni-repository-update", daemon=True).start()
        return True, "Update installation started"

    def close(self) -> None:
        self._stop.set()
        thread = self._poll_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
