#!/usr/bin/env python3
"""Private drain/readiness hooks for the cooperative broker supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

from werkzeug.wsgi import ClosingIterator


class RequestDrainGate:
    """Atomically admit and count writes until their response streams close."""

    def __init__(self, application, marker: Path):
        self.application = application
        self.marker = marker
        self.lock = threading.Lock()
        self.active = 0

    def snapshot(self) -> int:
        with self.lock:
            return self.active

    def __call__(self, environ, start_response):
        if environ.get("REQUEST_METHOD") not in {"POST", "PUT", "PATCH"}:
            return self.application(environ, start_response)
        with self.lock:
            if self.marker.is_file():
                body = json.dumps({"error": "service_draining", "retryable": True,
                                   "message": "Service is briefly paused. Retry shortly."}).encode()
                start_response("503 Service Unavailable", [("Content-Type", "application/json"),
                               ("Content-Length", str(len(body))), ("Retry-After", "2")])
                return [body]
            self.active += 1
        try:
            response = self.application(environ, start_response)
        except BaseException:
            with self.lock:
                self.active -= 1
            raise

        closed = False

        def finish():
            nonlocal closed
            with self.lock:
                if not closed:
                    self.active -= 1
                    closed = True
        return ClosingIterator(response, finish)


def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    runtime = Path(os.environ.get("OMNI_PORTAL_RUNTIME_ROOT", root / "runtime-data"))
    state = Path(os.environ.get("OMNI_PORTAL_STATE_DIR", runtime / "state"))
    marker = Path(os.environ.get("OMNI_GPU_DRAIN_FILE", state / "gpu-relocation.draining"))
    return state, marker


def ownership(state: Path) -> str:
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not token or not gpu.startswith("GPU-") or "," in gpu:
        raise ValueError("one supervised GPU lease is required")
    if (state / "gpu-lease-token.txt").read_text().strip() != token:
        raise ValueError("portal has not adopted this lease")
    if (state / "comprehension-gpu.txt").read_text().strip() != gpu:
        raise ValueError("portal GPU differs from the supervised reservation")
    return hashlib.sha256(token.encode()).hexdigest()


def marker_owner(marker: Path, identity: str) -> bool:
    return not marker.exists() or marker.read_text().strip() == identity


def fetch(path: str, state: Path, *, authenticated: bool) -> dict:
    port = int(os.environ.get("OMNI_PORTAL_PORT", "8920"))
    headers = {}
    if authenticated:
        headers["Authorization"] = "Bearer " + (state / "access-token.txt").read_text().strip()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError("invalid portal status")
    return value


def control(action: str) -> bool:
    state, marker = paths()
    identity = ownership(state)
    if action in {"starting", "drain"}:
        # Startup is called only after start.sh verifies the new pending
        # broker lease. It replaces a previous instance's stale drain marker.
        if action != "starting" and not marker_owner(marker, identity):
            raise ValueError("another owner holds the drain marker")
        temporary = marker.with_name(marker.name + f".{os.getpid()}.tmp")
        temporary.write_text(identity + "\n")
        temporary.chmod(0o600)
        temporary.replace(marker)
        if action == "starting":
            return True
    if not marker_owner(marker, identity):
        raise ValueError("another owner holds the drain marker")
    if action == "drain":
        # The portal checks this before admitting new writes. Existing request
        # counters remain authoritative until their streamed work finishes.
        activity = fetch("/api/activity", state, authenticated=True)
        return all(type(activity.get(key)) is int and activity[key] == 0
                   for key in ("inflight", "active", "queued", "mutating_requests"))
    if action == "ready":
        expected_pid = os.environ.get("OLLAMA_UNIFY_SUPERVISED_CHILD_PID", "")
        if not expected_pid or (state / "supervisor.pid").read_text().strip() != expected_pid:
            return False
        if fetch("/healthz", state, authenticated=False).get("ok") is not True:
            return False
    elif action != "resume":
        raise ValueError("expected starting, drain, resume or ready")
    marker.unlink(missing_ok=True)
    return True


def main() -> int:
    try:
        return 0 if len(sys.argv) == 2 and control(sys.argv[1]) else 1
    except (OSError, ValueError) as error:
        # Credentials and protected access URLs never enter hook output.
        print(f"GPU coordination hook unavailable: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
