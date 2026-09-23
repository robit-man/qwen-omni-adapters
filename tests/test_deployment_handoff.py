from __future__ import annotations

import os
from pathlib import Path

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
