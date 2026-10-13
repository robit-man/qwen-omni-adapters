from pathlib import Path
from types import SimpleNamespace

from qwen_omni_adapters.service_control import VoiceServiceControl


def test_pause_is_persisted_before_stopping_only_owned_service(tmp_path, monkeypatch):
    control = VoiceServiceControl(tmp_path)
    calls = []

    def systemctl(action):
        assert control.paused()
        calls.append(action)

    monkeypatch.setattr(control, "_systemctl", systemctl)
    assert control.set_paused(True) == (True, "Voice paused")
    assert calls == ["stop"]
    assert VoiceServiceControl(tmp_path).paused()
    assert control.marker.stat().st_mode & 0o777 == 0o600


def test_failed_stop_keeps_pause_guard(tmp_path, monkeypatch):
    control = VoiceServiceControl(tmp_path)

    def fail(_action):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(control, "_systemctl", fail)
    assert control.set_paused(True) == (False, "stop failed")
    assert control.paused()


def test_resume_doctor_failure_preserves_pause_without_start(tmp_path, monkeypatch):
    control = VoiceServiceControl(tmp_path)
    monkeypatch.setattr(control, "_systemctl", lambda _action: None)
    control.set_paused(True)

    def fail():
        raise RuntimeError("doctor failed")

    monkeypatch.setattr(control, "_doctor", fail)
    calls = []
    monkeypatch.setattr(control, "_systemctl", calls.append)
    assert control.set_paused(False) == (False, "doctor failed")
    assert control.paused()
    assert calls == []


def test_resume_releases_guard_only_after_doctor(tmp_path, monkeypatch):
    control = VoiceServiceControl(tmp_path)
    monkeypatch.setattr(control, "_systemctl", lambda _action: None)
    control.set_paused(True)
    calls = []

    def doctor():
        assert control.paused()
        calls.append("doctor")

    def start(action):
        assert not control.paused()
        calls.append(action)

    monkeypatch.setattr(control, "_doctor", doctor)
    monkeypatch.setattr(control, "_systemctl", start)
    assert control.set_paused(False) == (True, "Voice resuming")
    assert calls == ["doctor", "start"]
    assert not control.paused()


def test_failed_start_restores_durable_pause(tmp_path, monkeypatch):
    control = VoiceServiceControl(tmp_path)
    monkeypatch.setattr(control, "_systemctl", lambda _action: None)
    control.set_paused(True)
    monkeypatch.setattr(control, "_doctor", lambda: None)

    def fail(_action):
        raise RuntimeError("start failed")

    monkeypatch.setattr(control, "_systemctl", fail)
    assert control.set_paused(False) == (False, "start failed")
    assert control.paused()


def test_doctor_checks_selected_tags_without_loading_models(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('OMNI_MODEL="local/voice:one"\nOMNI_LANGUAGE_MODEL=local/language:two\n')
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("qwen_omni_adapters.service_control.subprocess.run", run)
    VoiceServiceControl(tmp_path)._doctor()
    assert calls[0][0][1:] == ["doctor", "--deployment", "--model", "local/voice:one", "--language-model", "local/language:two"]
    assert calls[0][1]["capture_output"]


def test_installed_service_guard_preserves_pause_across_updates():
    assert "ConditionPathExists=!@REPO_ROOT@/runtime-data/state/voice-service.paused" in Path(
        "services/linux/qwen-omni-adapters.service.in"
    ).read_text()
