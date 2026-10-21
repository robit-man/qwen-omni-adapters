from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from portal import gpu_lease
from portal.app import PortalConfig, create_app


@pytest.fixture
def owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    state = tmp_path / "state"
    state.mkdir()
    marker = state / "draining"
    (state / "gpu-lease-token.txt").write_text("private-fixture-token")
    (state / "comprehension-gpu.txt").write_text("GPU-fixture")
    (state / "supervisor.pid").write_text("12345")
    monkeypatch.setenv("OMNI_PORTAL_STATE_DIR", str(state))
    monkeypatch.setenv("OMNI_GPU_DRAIN_FILE", str(marker))
    monkeypatch.setenv("OLLAMA_UNIFY_GPU_LEASE", "private-fixture-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    monkeypatch.setenv("OLLAMA_UNIFY_SUPERVISED_CHILD_PID", "12345")
    return state, marker


def test_busy_owner_stays_paused_until_requests_finish(owned, monkeypatch):
    _, marker = owned
    monkeypatch.setattr(gpu_lease, "fetch", lambda *_args, **_kwargs:
                        {"inflight": 1, "active": 1, "queued": 0, "mutating_requests": 1})
    assert not gpu_lease.control("drain")
    assert marker.is_file()
    assert "private-fixture-token" not in marker.read_text()
    monkeypatch.setattr(gpu_lease, "fetch", lambda *_args, **_kwargs:
                        {"inflight": 0, "active": 0, "queued": 0, "mutating_requests": 0})
    assert gpu_lease.control("drain")
    assert marker.is_file()
    assert gpu_lease.control("resume")
    assert not marker.exists()


@pytest.mark.parametrize("activity", [{}, {"inflight": False, "active": 0, "queued": 0},
                                      {"inflight": 0, "active": 0, "queued": 1}])
def test_unknown_or_queued_work_is_not_safe_to_stop(owned, monkeypatch, activity):
    monkeypatch.setattr(gpu_lease, "fetch", lambda *_args, **_kwargs: activity)
    assert not gpu_lease.control("drain")


def test_readiness_requires_new_supervisor_instance_before_reopening(owned, monkeypatch):
    state, marker = owned
    assert gpu_lease.control("starting")
    (state / "supervisor.pid").write_text("old-instance")
    monkeypatch.setattr(gpu_lease, "fetch", lambda *_args, **_kwargs: {"ok": True})
    assert not gpu_lease.control("ready")
    assert marker.exists()
    (state / "supervisor.pid").write_text("12345")
    assert gpu_lease.control("ready")
    assert not marker.exists()


def test_hook_cannot_act_for_another_lease_or_gpu(owned, monkeypatch):
    state, _ = owned
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(ValueError, match="differs"):
        gpu_lease.control("drain")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    (state / "gpu-lease-token.txt").write_text("another-owner")
    with pytest.raises(ValueError, match="adopted"):
        gpu_lease.control("drain")


def test_cancel_cannot_clear_foreign_marker_but_new_verified_start_reclaims_it(owned):
    _, marker = owned
    marker.write_text("previous-instance")
    with pytest.raises(ValueError, match="another owner"):
        gpu_lease.control("resume")
    assert marker.read_text() == "previous-instance"
    assert gpu_lease.control("starting")
    assert marker.read_text().strip() == hashlib.sha256(b"private-fixture-token").hexdigest()


def test_portal_pauses_new_writes_while_status_and_existing_work_remain_observable(owned):
    _, marker = owned
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    app = create_app(PortalConfig(
        adapter_url="http://fixture/chat", adapter_health_url="http://fixture/health",
        comprehension_health_url="http://fixture/health", tts_health_url="http://fixture/health",
        ollama_health_url="http://fixture/health", model="fixture-model", access_token="fixture-key"),
        httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    assert gpu_lease.control("starting")
    paused = client.post("/api/chat", json={}, headers={"Authorization": "Bearer fixture-key"})
    assert paused.status_code == 503
    assert paused.json["error"] == "service_draining"
    assert paused.headers["Retry-After"] == "2"
    assert not calls
    assert client.get("/healthz").status_code == 200
    activity = client.get("/api/activity", headers={"Authorization": "Bearer fixture-key"})
    assert activity.status_code == 200
    assert activity.json["inflight"] == 0
    marker.unlink()
    resumed = client.post("/api/chat", json=[], headers={"Authorization": "Bearer fixture-key"})
    assert resumed.status_code == 400  # Normal request validation is reachable again.


def test_drain_counts_admitted_response_until_stream_closes(tmp_path):
    marker = tmp_path / "draining"
    exited = []

    def application(_environ, start_response):
        start_response("200 OK", [])
        def stream():
            try:
                yield b"first"
                yield b"last"
            finally:
                exited.append(True)
        return stream()

    gate = gpu_lease.RequestDrainGate(application, marker)
    response = gate({"REQUEST_METHOD": "POST"}, lambda *_: None)
    assert gate.snapshot() == 1  # Count starts before response iteration.
    assert next(response) == b"first"
    marker.touch()
    status = []
    assert list(gate({"REQUEST_METHOD": "POST"}, lambda value, _: status.append(value)))
    assert status == ["503 Service Unavailable"]
    assert gate.snapshot() == 1
    response.close()
    assert exited == [True]
    assert gate.snapshot() == 0


def test_supervised_cleanup_retains_worker_past_grace_period(tmp_path):
    import os
    import subprocess
    import sys

    source = Path("portal/start.sh").read_text()
    function = "terminate_child() {" + source.split("terminate_child() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    worker = tmp_path / "worker.py"
    marker = tmp_path / "ready"
    worker.write_text("import signal, time, pathlib\n"
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                      f"pathlib.Path({str(marker)!r}).touch()\n"
                      "time.sleep(.3)\n")
    script = function + '''
log() { :; }
"$PYTHON" "$WORKER" &
pid=$!
while [[ ! -f "$MARKER" ]]; do command sleep .01; done
# Accelerate the grace deadline, without shortening the worker's natural exit.
sleep() { SECONDS=$((SECONDS + 46)); command sleep .01; }
terminate_child "$pid" fixture
'''
    result = subprocess.run(["bash", "-c", script], timeout=3, check=False,
                            env={**os.environ, "PYTHON": sys.executable, "WORKER": str(worker),
                                 "MARKER": str(marker), "OLLAMA_UNIFY_GPU_LEASE_SUPERVISED": "1"})
    assert result.returncode == 0  # A forced SIGKILL would produce exit 137.
