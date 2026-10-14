from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from qwen_omni_adapters import deployment_handoff as handoff


def _fake_listener(
    root: Path,
    *,
    pid: int = 321,
    port: int = 8901,
    inode: int = 4242,
    command: str = "/opt/omni/.venv/bin/qwen-omni-daemon serve",
    cgroup: str = "0::/system.slice/egg-omni-comprehension.service\n",
) -> None:
    (root / "net").mkdir(parents=True)
    (root / "net" / "tcp").write_text(
        "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt "
        "uid timeout inode\n"
        f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 1000 0 {inode} 1\n",
        encoding="utf-8",
    )
    (root / "net" / "tcp6").write_text("header\n", encoding="utf-8")
    process = root / str(pid)
    (process / "fd").mkdir(parents=True)
    os.symlink(f"socket:[{inode}]", process / "fd" / "8")
    (process / "cmdline").write_bytes(command.replace(" ", "\0").encode() + b"\0")
    (process / "cgroup").write_text(cgroup, encoding="utf-8")
    (process / "status").write_text("Name:\ttest\nPPid:\t1\n", encoding="utf-8")


def test_listener_inventory_resolves_the_managed_omni_unit(tmp_path: Path) -> None:
    _fake_listener(tmp_path)

    owners = handoff.listener_owners(
        [8901], proc_root=tmp_path, include_gpu_residency=False
    )

    assert owners == [
        {
            "pid": 321,
            "parent_pid": 1,
            "ports": [8901],
            "command": "/opt/omni/.venv/bin/qwen-omni-daemon serve",
            "service_scope": "system-unit",
            "service_unit": "egg-omni-comprehension.service",
            "recognized_omni": True,
            "gpu_resident": None,
        }
    ]
    assert handoff.handoff_targets(owners) == [
        ("system-unit", "egg-omni-comprehension.service")
    ]


def test_unknown_port_owner_fails_closed(tmp_path: Path) -> None:
    _fake_listener(tmp_path, command="/usr/bin/unrelated-server", cgroup="0::/system.slice/x.service\n")
    owners = handoff.listener_owners(
        [8901], proc_root=tmp_path, include_gpu_residency=False
    )

    assert not owners[0]["recognized_omni"]
    with pytest.raises(RuntimeError, match="unknown or unreadable owner"):
        handoff.handoff_targets(owners)


def test_unresolved_listener_inode_is_not_reported_as_free(tmp_path: Path) -> None:
    _fake_listener(tmp_path)
    (tmp_path / "321" / "fd" / "8").unlink()

    owners = handoff.listener_owners(
        [8901], proc_root=tmp_path, include_gpu_residency=False
    )

    assert owners[0]["pid"] is None
    assert owners[0]["unresolved_socket_inode"] == "4242"
    with pytest.raises(RuntimeError):
        handoff.handoff_targets(owners)


def test_artifact_bytes_deduplicates_layers(monkeypatch) -> None:
    monkeypatch.setattr(
        handoff,
        "resolve_ollama_sidecar",
        lambda model: {
            "layer": {"digest": "sha256:sidecar", "size": 30},
            "standard_layers": {
                "language_model": {"layer": {"digest": "sha256:model", "size": 100}},
                "projector": {"layer": {"digest": "sha256:model", "size": 100}},
            },
        },
    )

    size, layers = handoff.artifact_bytes("robit/test:q4km")

    assert size == 130
    assert {layer["digest"] for layer in layers} == {"sha256:sidecar", "sha256:model"}


def test_jetson_admission_requires_headroom_and_an_idle_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(handoff, "artifact_bytes", lambda _model: (6 * 1024**3, []))
    monkeypatch.setattr(
        handoff,
        "gpu_facts",
        lambda: [{"utilization_percent": 7.0, "memory_model": "unified"}],
    )
    monkeypatch.setattr(handoff, "accelerator_profile", lambda: {"tegra": True})
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {12 * 1024 * 1024} kB\n", encoding="utf-8")

    result = handoff.jetson_admission(
        "robit/test:q4km",
        reserve_mib=4096,
        samples=3,
        sample_interval=0,
        meminfo_path=meminfo,
    )

    assert result["admitted"] is True
    assert result["memory_ok"] is True
    assert result["utilization_ok"] is True
    assert result["utilization_samples_percent"] == [7.0, 7.0, 7.0]


def test_jetson_admission_rejects_sustained_gpu_activity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(handoff, "artifact_bytes", lambda _model: (1, []))
    monkeypatch.setattr(
        handoff,
        "gpu_facts",
        lambda: [{"utilization_percent": 96.0, "memory_model": "unified"}],
    )
    monkeypatch.setattr(handoff, "accelerator_profile", lambda: {"tegra": True})
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 99999999 kB\n", encoding="utf-8")

    result = handoff.jetson_admission(
        "robit/test:q4km", samples=3, sample_interval=0, meminfo_path=meminfo
    )

    assert result["admitted"] is False
    assert result["memory_ok"] is True
    assert result["utilization_ok"] is False


def test_jetson_admission_rejects_an_invalid_utilization_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        handoff.jetson_admission("robit/test:q4km", max_utilization_percent=101)


@pytest.fixture
def broker_runtime(tmp_path: Path, monkeypatch):
    state = tmp_path / "runtime-data" / "state"
    state.mkdir(parents=True)
    token = "private-test-credential-123456789"
    files = {
        "supervisor.pid": "321\n",
        "access-token.txt": token + "\n",
        "access-url.txt": f"https://portal.example/#access={token}\n",
    }
    for name, value in files.items():
        path = state / name
        path.write_text(value, encoding="utf-8")
        os.utime(path, (101, 101))
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"bash\0" + str(tmp_path / "portal" / "start.sh").encode() + b"\0--foreground\0"
    )
    (process / "status").write_text("State:\tS (sleeping)\n", encoding="utf-8")
    monkeypatch.setattr(handoff.os, "kill", lambda _pid, _signal: None)
    status = {
        "ok": True,
        "model": "robit/test:q4km",
        "stages": {name: {"ok": True} for name in ("adapter", "comprehension", "tts", "ollama")},
    }
    requests = []
    response_options = {"status_code": 200}

    def respond(request):
        requests.append(request)
        assert str(request.url) == "http://127.0.0.1:8920/api/status"
        assert request.headers["Authorization"] == f"Bearer {token}"
        return httpx.Response(json=status, **response_options)

    client_class = httpx.Client

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return client_class(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(handoff.httpx, "Client", client)
    return {
        "repo_root": tmp_path,
        "state": state,
        "proc_root": proc_root,
        "status": status,
        "requests": requests,
        "response_options": response_options,
        "token": token,
    }


def _broker_probe(runtime):
    return handoff.broker_readiness(
        runtime["repo_root"], "robit/test:q4km",
        started_at=100, main_pid=321, proc_root=runtime["proc_root"],
    )


def test_broker_readiness_proves_fresh_owned_authenticated_health(broker_runtime) -> None:
    result = _broker_probe(broker_runtime)

    assert result == {"state": "ready", "mode": "broker", "model": "robit/test:q4km", "pid": 321}
    assert len(broker_runtime["requests"]) == 1
    assert broker_runtime["token"] not in str(result)
    assert "portal.example" not in str(result)


@pytest.mark.parametrize("name", ["supervisor.pid", "access-token.txt", "access-url.txt"])
def test_broker_readiness_rejects_each_stale_artifact(broker_runtime, name) -> None:
    os.utime(broker_runtime["state"] / name, (99, 99))

    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_rejects_artifacts_from_an_earlier_supervisor(broker_runtime) -> None:
    os.utime(broker_runtime["state"] / "supervisor.pid", (101.5, 101.5))

    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


@pytest.mark.parametrize("name", ["supervisor.pid", "access-token.txt", "access-url.txt"])
def test_broker_readiness_rejects_each_missing_artifact(broker_runtime, name) -> None:
    (broker_runtime["state"] / name).unlink()

    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_rejects_dead_supervisor(broker_runtime, monkeypatch) -> None:
    def dead(_pid, _signal):
        raise ProcessLookupError

    monkeypatch.setattr(handoff.os, "kill", dead)
    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_rejects_zombie_supervisor(broker_runtime) -> None:
    (broker_runtime["proc_root"] / "321" / "status").write_text("State:\tZ (zombie)\n")

    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_rejects_foreign_supervisor(broker_runtime) -> None:
    (broker_runtime["proc_root"] / "321" / "cmdline").write_bytes(
        b"bash\0/another/checkout/portal/start.sh\0--foreground\0"
    )

    assert _broker_probe(broker_runtime) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_requires_the_service_main_pid(broker_runtime) -> None:
    assert handoff.broker_readiness(
        broker_runtime["repo_root"], "robit/test:q4km",
        started_at=100, main_pid=999, proc_root=broker_runtime["proc_root"],
    ) is None
    assert broker_runtime["requests"] == []


def test_broker_readiness_rejects_wrong_model(broker_runtime) -> None:
    broker_runtime["status"]["model"] = "robit/other:q4km"

    assert _broker_probe(broker_runtime) is None


def test_broker_readiness_rejects_unhealthy_aggregate(broker_runtime) -> None:
    broker_runtime["status"]["ok"] = False

    assert _broker_probe(broker_runtime) is None


@pytest.mark.parametrize("name", ["adapter", "comprehension", "tts", "ollama"])
def test_broker_readiness_requires_each_healthy_stage(broker_runtime, name) -> None:
    broker_runtime["status"]["stages"][name]["ok"] = False

    assert _broker_probe(broker_runtime) is None


@pytest.mark.parametrize("code", [401, 302, 500])
def test_broker_readiness_rejects_unsuccessful_http(broker_runtime, code) -> None:
    broker_runtime["response_options"]["status_code"] = code

    assert _broker_probe(broker_runtime) is None


def test_broker_readiness_rejects_exit_during_status_probe(broker_runtime, monkeypatch) -> None:
    calls = 0

    def exits(_pid, _signal):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProcessLookupError

    monkeypatch.setattr(handoff.os, "kill", exits)
    assert _broker_probe(broker_runtime) is None
    assert len(broker_runtime["requests"]) == 1


def test_broker_ready_cli_emits_only_redacted_proof(monkeypatch, capsys) -> None:
    proof = {"state": "ready", "mode": "broker", "model": "robit/test:q4km", "pid": 321}
    monkeypatch.setattr(handoff, "broker_readiness", lambda *args, **kwargs: proof)

    assert handoff.main([
        "broker-ready", "--repo-root", "/opt/omni", "--model", "robit/test:q4km",
        "--started-at", "100", "--main-pid", "321",
    ]) == 0
    assert capsys.readouterr().out.strip() == (
        '{"mode": "broker", "model": "robit/test:q4km", "pid": 321, "state": "ready"}'
    )
