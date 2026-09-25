from __future__ import annotations

import inspect
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def test_startup_smoke_can_be_disabled_for_memory_brokered_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNI_STARTUP_SMOKE", "0")
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.startup_smoke is False


def test_tegra_defaults_optional_decision_plane_off_for_shared_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("OMNI_DECISION_PLANE_ENABLED", raising=False)
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)
    monkeypatch.setattr(daemon, "is_tegra", lambda: True)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.decision_plane_enabled is False


def test_tegra_can_explicitly_enable_optional_decision_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNI_DECISION_PLANE_ENABLED", "1")
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)
    monkeypatch.setattr(daemon, "is_tegra", lambda: True)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.decision_plane_enabled is True


def test_tts_stream_window_has_measured_default_and_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("OMNI_TTS_STREAM_FRAMES", raising=False)
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)

    assert daemon.DaemonConfig.from_environment(cloudflare=False).tts_stream_frames == 8

    monkeypatch.setenv("OMNI_TTS_STREAM_FRAMES", "12")
    assert daemon.DaemonConfig.from_environment(cloudflare=False).tts_stream_frames == 12


def test_language_thinking_policy_uses_the_logical_model_not_the_runtime_alias() -> None:
    assert daemon._language_disable_thinking_default(
        "robit/ornith-1.5-omni-audio-bridge:q4km"
    ) is True
    assert daemon._language_disable_thinking_default(
        "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km"
    ) is False
    assert daemon._language_disable_thinking_default("local-audio-bridge") is False


def test_status_with_the_capability_url_is_owner_readable_only(tmp_path: Path) -> None:
    supervisor = daemon.OmniDaemon(_config(tmp_path))
    supervisor.state_dir.mkdir(parents=True)

    supervisor._write_status(
        state="ready",
        access_url="https://example.test/#access=secret-capability",
    )

    mode = stat.S_IMODE(supervisor.status_file.stat().st_mode)
    assert mode == 0o600


def test_binary_finds_windows_release_layout(tmp_path: Path) -> None:
    binary = tmp_path / "vendor/llama.cpp/build/bin/Release/llama-server.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")

    assert daemon._binary(tmp_path, "llama-server") == binary


def test_tunnel_discovery_ignores_urls_from_prior_processes(tmp_path: Path) -> None:
    log = tmp_path / "cloudflared.log"
    log.write_text(
        "https://stale-link.trycloudflare.com\nRegistered tunnel connection\n",
        encoding="utf-8",
    )
    offset = log.stat().st_size

    with log.open("a", encoding="utf-8") as target:
        target.write("https://current-link.trycloudflare.com\n")
    assert daemon._connected_tunnel_url(log, offset) == ""

    with log.open("a", encoding="utf-8") as target:
        target.write("Registered tunnel connection\n")
    assert (
        daemon._connected_tunnel_url(log, offset)
        == "https://current-link.trycloudflare.com"
    )


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


def test_restart_control_file_exits_for_systemd_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = daemon.OmniDaemon(_config(tmp_path))
    supervisor.state_dir.mkdir(parents=True)
    supervisor.restart_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(supervisor, "prepare", lambda: None)
    monkeypatch.setattr(supervisor, "start_children", lambda: "http://local/#access=x")
    monkeypatch.setattr(supervisor, "cleanup", lambda: None)
    monkeypatch.setattr(supervisor, "_write_status", lambda **_fields: None)

    assert supervisor.run(register_signals=False) == 75


def test_tegra_direct_daemon_uses_dynamic_context_launcher() -> None:
    source = inspect.getsource(daemon.OmniDaemon.start_children)

    assert '"comprehension_launcher.py"' in source
    assert '"--child-pid-file"' in source
    assert '"OMNI_COMPREHENSION_CONTEXT_FILE"' in source
    assert '"{context}" if is_tegra()' in source
    assert '"--parallel-slots"' in source
    assert '"{parallel}"' in source
    assert '"--image-min-tokens"' in source
    assert '"1024"' in source


def test_an_openai_language_backend_is_not_pulled_from_ollama(monkeypatch, tmp_path):
    """Its model name belongs to that server; Ollama has never heard of it.

    Pulling it hangs the supervisor on a tag that cannot exist, which is
    exactly what happened the first time the OpenAI backend was deployed.
    """

    from qwen_omni_adapters import daemon as daemon_module

    monkeypatch.setenv("OMNI_LANGUAGE_API", "openai")
    monkeypatch.setenv("OMNI_MODEL", "robit/ornith-1.5-omni:q4km")
    monkeypatch.setenv("OMNI_LANGUAGE_MODEL", "local-qwen3-omni")
    monkeypatch.setenv("OMNI_PORTAL_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(daemon_module, "_load_env_file", lambda _root: None)

    config = daemon_module.DaemonConfig.from_environment(cloudflare=False)
    assert config.language_api == "openai"

    supervisor = daemon_module.OmniDaemon(config)
    pulled: list[str] = []
    monkeypatch.setattr(supervisor, "_ensure_model", lambda model: pulled.append(model))

    def refuse() -> None:
        raise AssertionError("must not verify Ollama blobs for another backend")

    monkeypatch.setattr(supervisor, "_verify_shared_base", refuse)
    monkeypatch.setattr(supervisor, "_preflight", lambda: None)
    monkeypatch.setattr(
        daemon_module, "resolve_ollama_sidecar", lambda model: {"layer": {"digest": "x"}}
    )

    def refuse_materialize(**_kwargs) -> None:
        raise AssertionError("component views are already present")

    monkeypatch.setattr(daemon_module, "prepare_ollama_sidecar", refuse_materialize)
    supervisor.cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "comprehension-model.gguf", "comprehension-projector.gguf",
        "tts-model.gguf", "tts-projector.gguf",
    ):
        (supervisor.cache_dir / name).write_bytes(b"x")

    supervisor.prepare()

    # Only the logical tag is an Ollama concern.
    assert pulled == ["robit/ornith-1.5-omni:q4km"]


def test_an_ollama_language_backend_is_still_pulled_and_verified(monkeypatch, tmp_path):
    from qwen_omni_adapters import daemon as daemon_module

    monkeypatch.delenv("OMNI_LANGUAGE_API", raising=False)
    monkeypatch.setenv("OMNI_MODEL", "robit/ornith-1.5-omni:q4km")
    monkeypatch.setenv("OMNI_LANGUAGE_MODEL", "robit/ornith-1.5:9b")
    monkeypatch.setenv("OMNI_PORTAL_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(daemon_module, "_load_env_file", lambda _root: None)

    config = daemon_module.DaemonConfig.from_environment(cloudflare=False)
    supervisor = daemon_module.OmniDaemon(config)
    pulled: list[str] = []
    verified: list[bool] = []
    monkeypatch.setattr(supervisor, "_ensure_model", lambda model: pulled.append(model))
    monkeypatch.setattr(supervisor, "_verify_shared_base", lambda: verified.append(True))
    monkeypatch.setattr(supervisor, "_preflight", lambda: None)
    monkeypatch.setattr(
        daemon_module, "resolve_ollama_sidecar", lambda model: {"layer": {"digest": "x"}}
    )

    def refuse_materialize(**_kwargs) -> None:
        raise AssertionError("component views are already present")

    monkeypatch.setattr(daemon_module, "prepare_ollama_sidecar", refuse_materialize)
    supervisor.cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "comprehension-model.gguf", "comprehension-projector.gguf",
        "tts-model.gguf", "tts-projector.gguf",
    ):
        (supervisor.cache_dir / name).write_bytes(b"x")

    supervisor.prepare()

    assert pulled == ["robit/ornith-1.5-omni:q4km", "robit/ornith-1.5:9b"]
    assert verified == [True]


def test_trained_audio_bridge_uses_standard_layers_as_its_only_language_trunk(
    tmp_path,
) -> None:
    from qwen_omni_adapters import daemon as daemon_module

    config = _config(tmp_path)
    supervisor = daemon_module.OmniDaemon(config)
    model = tmp_path / "blobs" / "language.gguf"
    projector = tmp_path / "blobs" / "projector.gguf"
    supervisor.sidecar_resolution = {
        "profile": "trained-audio-bridge",
        "standard_layers": {
            "language_model": {"path": str(model)},
            "projector": {"path": str(projector)},
        },
    }

    assert supervisor._comprehension_artifacts() == (model, projector)
    assert supervisor._language_route() == (
        "openai",
        f"http://127.0.0.1:{config.comprehension_port}/v1/chat/completions",
        "local-audio-bridge",
    )
    assert supervisor._speculative_args() == ["--spec-type", "ngram-simple"]


def test_startup_smoke_includes_real_audio_asr_and_tts() -> None:
    source = inspect.getsource(daemon.OmniDaemon.start_children)

    assert '"--audio"' in source
    assert '"default_voice.wav"' in source
    assert '"--tts"' in source
    assert '"--voice-clone"' in source
    assert "_verify_co_resident_stack" in source


def test_tegra_prewarms_resident_tts_before_comprehension_admission() -> None:
    source = inspect.getsource(daemon.OmniDaemon.start_children)

    prewarm = source.index("if is_tegra():\n            tts = self._start_tts(common)")
    comprehension = source.index("comprehension_model, comprehension_projector")
    assert prewarm < comprehension
    assert "self._wait_resident_tts(tts)" in source[prewarm:comprehension]


def test_daemon_resolves_the_shipped_default_clone_reference(tmp_path: Path) -> None:
    profile_dir = tmp_path / "portal"
    voice_dir = profile_dir / "voices"
    voice_dir.mkdir(parents=True)
    female = voice_dir / "female.wav"
    male = voice_dir / "male.wav"
    female.write_bytes(b"female")
    male.write_bytes(b"male")
    (profile_dir / "voice-profile.json").write_text(
        json.dumps(
            {
                "speaker_file": "voices/male.wav",
                "presets": [
                    {
                        "id": "female",
                        "speaker_file": "voices/female.wav",
                        "default": True,
                    },
                    {
                        "id": "male",
                        "speaker_file": "voices/male.wav",
                        "default": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert daemon.OmniDaemon(_config(tmp_path))._voice_reference() == female.resolve()


def test_co_residency_gate_requires_live_cloned_tts_and_comprehension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = daemon.OmniDaemon(_config(tmp_path))
    comprehension = daemon.Child(
        "comprehension", SimpleNamespace(pid=101, poll=lambda: None), None
    )
    tts = daemon.Child("tts", SimpleNamespace(pid=202, poll=lambda: None), None)
    checked: list[tuple[int, str]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "persistent_ready": True,
                "persistent_pid": 303,
                "speaker_reference_configured": True,
                "speaker_reference_active": True,
            }

    monkeypatch.setattr(daemon.httpx, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: pid == 303)
    monkeypatch.setattr(
        supervisor,
        "_verify_direct_gpu",
        lambda pid, component="comprehension": checked.append((pid, component)),
    )
    monkeypatch.setattr(daemon.platform, "system", lambda: "Linux")

    evidence = supervisor._verify_co_resident_stack(comprehension, tts)

    assert checked == [(101, "comprehension"), (303, "tts")]
    assert evidence["comprehension_pid"] == 101
    assert evidence["tts_pid"] == 303
    assert evidence["speaker_reference_active"] is True


def test_an_externally_managed_comprehension_port_does_not_block_start(monkeypatch):
    """A worker this supervisor does not own may already hold that port."""

    from qwen_omni_adapters import daemon as daemon_module

    monkeypatch.setenv("OMNI_ENABLE_COMPREHENSION", "0")
    monkeypatch.setattr(daemon_module, "_load_env_file", lambda _root: None)
    config = daemon_module.DaemonConfig.from_environment(
        cloudflare=False, allow_direct_gpu=True
    )
    supervisor = daemon_module.OmniDaemon(config)

    checked: list[int] = []
    monkeypatch.setattr(
        daemon_module, "_port_available", lambda host, port: checked.append(port) or True
    )
    monkeypatch.setattr(daemon_module.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(supervisor, "_broker_present", lambda: False)
    monkeypatch.setattr(daemon_module, "_binary", lambda root, name: Path("/bin/sh"))
    supervisor._preflight()

    assert config.comprehension_port not in checked
    assert config.tts_port in checked


def test_an_owned_comprehension_port_is_still_required(monkeypatch):
    from qwen_omni_adapters import daemon as daemon_module

    monkeypatch.delenv("OMNI_ENABLE_COMPREHENSION", raising=False)
    monkeypatch.setattr(daemon_module, "_load_env_file", lambda _root: None)
    config = daemon_module.DaemonConfig.from_environment(
        cloudflare=False, allow_direct_gpu=True
    )
    supervisor = daemon_module.OmniDaemon(config)

    checked: list[int] = []
    monkeypatch.setattr(
        daemon_module, "_port_available", lambda host, port: checked.append(port) or True
    )
    monkeypatch.setattr(daemon_module.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(supervisor, "_broker_present", lambda: False)
    monkeypatch.setattr(daemon_module, "_binary", lambda root, name: Path("/bin/sh"))
    supervisor._preflight()

    assert config.comprehension_port in checked
