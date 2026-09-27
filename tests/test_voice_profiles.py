from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal.app import VoiceProfileReader, load_voice_profile
from qwen_omni_adapters.voice_profiles import IndicatorVoiceManager


def _write_wav(path: Path, *, seconds: float = 1.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * int(seconds * rate))


def _repo(tmp_path: Path) -> Path:
    voices = tmp_path / "portal" / "voices"
    _write_wav(voices / "female_voice.wav")
    _write_wav(voices / "default_voice.wav")
    (tmp_path / "portal" / "voice-profile.json").write_text(
        json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "name": "default",
                "language": "en",
                "speaker_file": "voices/female_voice.wav",
                "presets": [
                    {
                        "id": "female",
                        "label": "Female",
                        "speaker_file": "voices/female_voice.wav",
                        "default": True,
                    },
                    {
                        "id": "male",
                        "label": "Male",
                        "speaker_file": "voices/default_voice.wav",
                        "default": False,
                    },
                ],
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "seed": 42,
                "max_frames": 512,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_indicator_voice_manager_selects_shipped_presets_without_dirtying_source(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    source_before = (root / "portal/voice-profile.json").read_bytes()
    manager = IndicatorVoiceManager(root)

    assert manager.views() == [
        {"id": "female", "label": "Default female", "active": True, "custom": False},
        {"id": "male", "label": "Default male", "active": False, "custom": False},
    ]
    ok, detail = manager.select("male")

    assert ok is True
    assert detail == "Selected Default male for the next reply"
    assert [view["id"] for view in manager.views() if view["active"]] == ["male"]
    assert (root / "portal/voice-profile.json").read_bytes() == source_before
    managed = json.loads(manager.profile_path.read_text(encoding="utf-8"))
    assert Path(managed["speaker_file"]).name == "default_voice.wav"
    assert manager.profile_path.stat().st_mode & 0o777 == 0o600


def test_custom_clip_is_normalized_added_selected_and_deduplicated(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    manager = IndicatorVoiceManager(root)
    custom = tmp_path / "My Sample.wav"
    _write_wav(custom, seconds=1.25, rate=8000)

    first = manager.import_clip(custom)
    second = manager.import_clip(custom)

    assert first[0] is True
    assert second[0] is True
    views = manager.views()
    custom_views = [view for view in views if view["custom"]]
    assert len(custom_views) == 1
    assert custom_views[0]["label"] == "My Sample"
    assert custom_views[0]["active"] is True
    profile = json.loads(manager.profile_path.read_text(encoding="utf-8"))
    selected = next(preset for preset in profile["presets"] if preset["default"])
    selected_path = Path(selected["speaker_file"])
    assert selected_path.parent == root / "runtime-data" / "voices"
    with wave.open(str(selected_path), "rb") as normalized:
        assert normalized.getframerate() == 16000
        assert normalized.getnchannels() == 1
        assert normalized.getsampwidth() == 2


def test_voice_profile_reader_hot_reloads_an_atomic_selection(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    manager = IndicatorVoiceManager(root)
    reader = VoiceProfileReader(
        load_voice_profile(manager.profile_path),
        manager.profile_path,
    )

    assert reader.current()["name"] == "default"
    assert manager.select("male")[0] is True

    current = reader.current()
    assert current["name"] == "male"
    assert Path(current["speaker_file"]).name == "default_voice.wav"
