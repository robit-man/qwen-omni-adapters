from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from qwen_omni_adapters import daemon


def _config(tmp_path: Path, **overrides) -> daemon.DaemonConfig:
    values = {
        "repo_root": tmp_path,
        "runtime_root": tmp_path / "runtime-data",
        "model": "robit/omni:q4km",
        "language_model": "robit/base:27b",
        "cloudflare": False,
        "allow_direct_gpu": True,
    }
    values.update(overrides)
    return daemon.DaemonConfig(**values)


def test_env_file_sets_defaults_without_overriding_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "OMNI_MODEL=from-file\nOMNI_LANGUAGE_MODEL='file-base'\n# ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNI_MODEL", "from-process")
    monkeypatch.delenv("OMNI_LANGUAGE_MODEL", raising=False)

    daemon._load_env_file(tmp_path)

    assert daemon.os.environ["OMNI_MODEL"] == "from-process"
    assert daemon.os.environ["OMNI_LANGUAGE_MODEL"] == "file-base"


def test_binary_finds_windows_release_layout(tmp_path: Path) -> None:
    binary = tmp_path / "vendor/llama.cpp/build/bin/Release/llama-server.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")

    assert daemon._binary(tmp_path, "llama-server") == binary


def test_linux_direct_supervisor_refuses_to_bypass_detected_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = daemon.OmniDaemon(_config(tmp_path))
    monkeypatch.setattr(daemon.platform, "system", lambda: "Linux")
    monkeypatch.setattr(supervisor, "_broker_present", lambda: True)

    with pytest.raises(daemon.DaemonError, match="will not bypass broker leases"):
        supervisor._preflight()


def test_ensure_model_pulls_only_when_ollama_show_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = daemon.OmniDaemon(_config(tmp_path))
    supervisor.state_dir.mkdir(parents=True)
    seen: list[list[str]] = []
    show_results = iter(
        [
            subprocess.CompletedProcess(["ollama", "show"], 1, "", "missing"),
            subprocess.CompletedProcess(["ollama", "show"], 0, "ok", ""),
        ]
    )
    monkeypatch.setattr(daemon.subprocess, "run", lambda *args, **kwargs: next(show_results))
    monkeypatch.setattr(
        supervisor,
        "_command",
        lambda command, timeout=3600: (
            seen.append(command) or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    supervisor._ensure_model("robit/omni:q4km")
    supervisor._ensure_model("robit/base:27b")

    assert seen == [["ollama", "pull", "robit/omni:q4km"]]
    state = json.loads(supervisor.status_file.read_text(encoding="utf-8"))
    assert state["state"] == "pulling"


def test_stop_command_uses_cross_platform_control_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    state_dir = config.runtime_root / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "daemon-status.json").write_text(
        json.dumps({"state": "ready", "pid": 1234}), encoding="utf-8"
    )
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: pid == 1234)
    monkeypatch.setattr(
        daemon.DaemonConfig,
        "from_environment",
        classmethod(lambda cls, **kwargs: config),
    )

    assert daemon.main(["stop"]) == 0
    assert (state_dir / "stop.request").is_file()
