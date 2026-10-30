from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_omni_adapters import cli, daemon, gpu_broker


@pytest.fixture
def probe_host(monkeypatch):
    monkeypatch.setattr(gpu_broker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_broker, "_installation_marker", lambda: "")
    monkeypatch.setattr(gpu_broker.shutil, "which", lambda _: "/test/docker")

    def command(*, code=1, stdout="", stderr="", error=None):
        def run(argv, **kwargs):
            assert argv == ["/test/docker", "gpu", "discover"]
            assert kwargs["timeout"] == 20 and kwargs["env"]["LC_ALL"] == "C"
            if error:
                raise error
            return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)
        monkeypatch.setattr(gpu_broker.subprocess, "run", run)
    return command


@pytest.mark.parametrize("diagnostic", [
    "docker: unknown command: docker gpu",
    "docker: unknown command: gpu",
    "docker: 'gpu' is not a docker command.",
])
def test_missing_plugin_selects_direct_without_installing_a_broker(probe_host, diagnostic):
    probe_host(stderr=diagnostic + "\n\nRun 'docker --help' for more information")
    result = gpu_broker.probe_gpu_broker()
    assert result.state == "absent" and not result.managed and result.mode == "direct"


@pytest.mark.parametrize("installed", [False, True])
def test_missing_docker_does_not_bypass_an_installed_broker(probe_host, monkeypatch, installed):
    monkeypatch.setattr(gpu_broker.shutil, "which", lambda _: None)
    monkeypatch.setattr(gpu_broker, "_installation_marker", lambda: "/broker" if installed else "")
    assert gpu_broker.probe_gpu_broker().state == ("unavailable" if installed else "absent")


def test_invalid_installed_plugin_remains_mandatory(probe_host, monkeypatch):
    probe_host(stderr="docker: unknown command: docker gpu")
    monkeypatch.setattr(gpu_broker, "_installation_marker", lambda: "/broken/docker-gpu")
    assert gpu_broker.probe_gpu_broker().state == "unavailable"


@pytest.mark.parametrize("result", [
    {"stderr": "permission denied connecting to GPU broker"},
    {"stderr": "broker unavailable: missing socket"},
    {"stderr": "unknown command: discover"},
    {"stderr": ""},
    {"code": 0, "stdout": "not JSON"},
    {"code": 0, "stdout": "[]"},
    {"code": 0, "stdout": '{"available":false}'},
    {"code": 0, "stdout": '{"available":"true"}'},
    {"code": 0, "stdout": '{}'},
    {"error": subprocess.TimeoutExpired("docker", 20)},
    {"error": OSError("cannot execute docker")},
])
def test_unverified_discovery_never_selects_direct(probe_host, result, capsys):
    probe_host(**result)
    probe = gpu_broker.probe_gpu_broker()
    assert probe.state == "unavailable" and probe.managed and probe.mode == "broker"
    assert gpu_broker.main(["--mode"]) == 1
    output = capsys.readouterr()
    assert output.out == "" and "refusing direct GPU mode" in output.err


def test_healthy_broker_preserves_raw_policy_for_handoff(probe_host, capsys):
    discovery = '{"available":true,"warnings":["coordinate"],"active_leases":[{"owner":"other"}]}'
    probe_host(code=0, stdout=discovery)
    assert gpu_broker.probe_gpu_broker().state == "available"
    assert gpu_broker.main([]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(discovery)
    assert gpu_broker.main(["--mode"]) == 0
    assert capsys.readouterr().out == "broker\n"


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_platform_native_supervisors_do_not_probe_docker(monkeypatch, system):
    monkeypatch.setattr(gpu_broker.platform, "system", lambda: system)
    monkeypatch.setattr(gpu_broker, "_installation_marker", lambda: pytest.fail("Linux marker probe"))
    assert gpu_broker.probe_gpu_broker().mode == "direct"


@pytest.mark.parametrize("kind", ["user-plugin", "extra-plugin", "unreadable"])
def test_installation_markers_include_custom_and_unreadable_plugins(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    plugin = tmp_path / ("extra" if kind == "extra-plugin" else "cli-plugins") / "docker-gpu"
    plugin.parent.mkdir()
    plugin.symlink_to(tmp_path / "missing-executable")
    if kind == "extra-plugin":
        (tmp_path / "config.json").write_text(json.dumps({"cliPluginsExtraDirs": [str(plugin.parent)]}))
    original = Path.lstat

    def lstat(path):
        if path == plugin:
            if kind == "unreadable":
                raise PermissionError(str(path))
            return original(path)
        raise FileNotFoundError(str(path))
    monkeypatch.setattr(Path, "lstat", lstat)
    assert gpu_broker._installation_marker() == str(plugin)


@pytest.mark.parametrize("failure", [PermissionError("unreadable"), OSError("I/O"), "invalid JSON"])
def test_unknown_docker_config_cannot_prove_plugin_absence(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(gpu_broker.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    monkeypatch.setattr(gpu_broker.shutil, "which", lambda _: "/test/docker")

    def read_config(path):
        assert path == tmp_path / "config.json"
        if isinstance(failure, OSError):
            raise failure
        return failure

    monkeypatch.setattr(Path, "read_text", read_config)
    monkeypatch.setattr(gpu_broker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr="docker: unknown command: docker gpu",
    ))
    assert gpu_broker._installation_marker() == str(tmp_path / "config.json")
    assert gpu_broker.probe_gpu_broker().state == "unavailable"


@pytest.mark.parametrize("system,tegra,state,expected", [
    ("Linux", False, "absent", "direct"),
    ("Linux", False, "available", "broker"),
    ("Linux", False, "unavailable", "broker"),
    ("Linux", True, "absent", "direct"),
    ("Darwin", False, "absent", "direct"),
    ("Windows", False, "absent", "direct"),
])
def test_doctor_selects_host_lifecycle_and_required_tools(
    tmp_path, monkeypatch, capsys, system, tegra, state, expected,
):
    for name in ("llama-server", "llama-tts"):
        binary = tmp_path / "vendor/llama.cpp/build/bin" / name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("fixture")
        binary.chmod(0o755)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src/qwen_omni_adapters/cli.py"))
    monkeypatch.setattr(cli.platform, "system", lambda: system)
    monkeypatch.setattr(cli, "is_tegra", lambda: tegra)
    monkeypatch.setattr(cli, "accelerator_profile", lambda: {"system": system, "tegra": tegra})
    monkeypatch.setattr(cli, "probe_gpu_broker", lambda: gpu_broker.BrokerProbe(state, "fixture"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: (
        None if state == "absent" and name in {"docker", "jq", "ss"} else f"/fixture/{name}"
    ))

    def show(command):
        assert command == ["ollama", "show", "local/omni:test"]  # Never load/pull weights.
        return True, ""
    monkeypatch.setattr(cli, "_run", show)
    code = cli._doctor(argparse.Namespace(
        deployment=True, no_tunnel=True, model="local/omni:test", language_model="local/omni:test",
    ))
    report = json.loads(capsys.readouterr().out)
    assert report["deployment_mode"] == expected
    assert report["ok"] == (state != "unavailable")
    assert code == (1 if state == "unavailable" else 0)
    assert {"docker", "jq", "ss"}.issubset(report["commands"]) == (expected == "broker")
    assert ("--allow-direct-gpu" in report["remediation"]["start"]) == (
        expected == "direct" and system == "Linux"
    )


@pytest.mark.parametrize("state", ["available", "unavailable"])
def test_direct_daemon_refuses_healthy_and_failed_brokers(tmp_path, monkeypatch, state):
    monkeypatch.setattr(daemon.platform, "system", lambda: "Linux")
    monkeypatch.setattr(daemon, "probe_gpu_broker", lambda: gpu_broker.BrokerProbe(state, "fixture"))
    supervisor = daemon.OmniDaemon(daemon.DaemonConfig(
        repo_root=tmp_path, runtime_root=tmp_path / "runtime-data", model="local/test",
        language_model="local/test", allow_direct_gpu=True, cloudflare=False,
    ))
    with pytest.raises(daemon.DaemonError, match="will not bypass broker leases"):
        supervisor._preflight()


def test_direct_daemon_still_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon.platform, "system", lambda: "Linux")
    monkeypatch.setattr(daemon, "probe_gpu_broker", lambda: gpu_broker.BrokerProbe("absent", "fixture"))
    supervisor = daemon.OmniDaemon(daemon.DaemonConfig(
        repo_root=tmp_path, runtime_root=tmp_path / "runtime-data", model="local/test",
        language_model="local/test", allow_direct_gpu=False, cloudflare=False,
    ))
    with pytest.raises(daemon.DaemonError, match="requires --allow-direct-gpu"):
        supervisor._preflight()
