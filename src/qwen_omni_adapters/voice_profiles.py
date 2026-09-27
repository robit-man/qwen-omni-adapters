"""Persistent, runtime-local voice profiles for the desktop call harness.

The checked-in profile is a deployment default.  User-selected clone samples
belong in ``runtime-data`` so a voice choice neither dirties the checkout nor
disappears during an update.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

MANAGED_VOICE_PROFILE = Path("runtime-data/state/voice-profile.json")
MANAGED_VOICE_DIRECTORY = Path("runtime-data/voices")
MAX_SOURCE_BYTES = 50 * 1024 * 1024
MIN_REFERENCE_SECONDS = 0.5
MAX_REFERENCE_SECONDS = 30.0
SUPPORTED_SOURCE_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


class VoiceProfileError(RuntimeError):
    """A voice profile or imported clone reference is invalid."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _resolved_template(repo_root: Path) -> dict[str, Any]:
    source = repo_root / "portal" / "voice-profile.json"
    try:
        profile = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VoiceProfileError(f"could not read the shipped voice profile: {exc}") from exc
    if not isinstance(profile, dict):
        raise VoiceProfileError("the shipped voice profile is not a JSON object")

    def resolve(value: Any) -> str:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        if not path.is_file():
            raise VoiceProfileError(f"voice reference does not exist: {path}")
        return str(path)

    if profile.get("speaker_file"):
        profile["speaker_file"] = resolve(profile["speaker_file"])
    presets = profile.get("presets")
    if isinstance(presets, list):
        for preset in presets:
            if isinstance(preset, dict):
                preset["speaker_file"] = resolve(preset.get("speaker_file"))
    return profile


def ensure_managed_voice_profile(repo_root: Path) -> Path:
    """Create the mutable runtime profile once from the shipped defaults."""

    repo_root = repo_root.resolve()
    profile_path = repo_root / MANAGED_VOICE_PROFILE
    if not profile_path.is_file():
        _atomic_json(profile_path, _resolved_template(repo_root))
    return profile_path


def _read_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VoiceProfileError(f"could not read voice profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise VoiceProfileError("voice profile must be a JSON object")
    presets = profile.get("presets")
    if not isinstance(presets, list) or not presets:
        raise VoiceProfileError("voice profile must contain at least one preset")
    defaults = [item for item in presets if isinstance(item, dict) and item.get("default") is True]
    if len(defaults) != 1:
        raise VoiceProfileError("voice profile must contain exactly one selected preset")
    return profile


def _display_label(preset: dict[str, Any]) -> str:
    preset_id = str(preset.get("id") or "")
    if preset_id == "female":
        return "Default female"
    if preset_id == "male":
        return "Default male"
    return str(preset.get("label") or preset_id or "Custom voice")


class IndicatorVoiceManager:
    """Own the indicator's persistent preset list and selected clone voice."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.profile_path = ensure_managed_voice_profile(self.repo_root)
        self.voice_directory = self.repo_root / MANAGED_VOICE_DIRECTORY
        self._lock = threading.Lock()

    def views(self) -> list[dict[str, Any]]:
        with self._lock:
            profile = _read_profile(self.profile_path)
        views: list[dict[str, Any]] = []
        for preset in profile["presets"]:
            if not isinstance(preset, dict):
                continue
            preset_id = str(preset.get("id") or "")
            if not preset_id:
                continue
            views.append(
                {
                    "id": preset_id,
                    "label": _display_label(preset),
                    "active": preset.get("default") is True,
                    "custom": preset_id.startswith("custom-"),
                }
            )
        return views

    def select(self, preset_id: str) -> tuple[bool, str]:
        with self._lock:
            profile = _read_profile(self.profile_path)
            selected: dict[str, Any] | None = None
            for preset in profile["presets"]:
                if not isinstance(preset, dict):
                    continue
                active = str(preset.get("id") or "") == preset_id
                preset["default"] = active
                if active:
                    selected = preset
            if selected is None:
                return False, "Voice is no longer in the managed list"
            speaker = Path(str(selected.get("speaker_file") or "")).expanduser().resolve()
            if not speaker.is_file():
                return False, "Voice reference is missing"
            profile["speaker_file"] = str(speaker)
            profile["name"] = preset_id
            _atomic_json(self.profile_path, profile)
        return True, f"Selected {_display_label(selected)} for the next reply"

    def import_clip(self, source: Path) -> tuple[bool, str]:
        """Normalize an owned audio clip, add it to the list, and select it."""

        source = source.expanduser().resolve()
        if not source.is_file():
            return False, "Selected audio clip does not exist"
        if source.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            return False, "Choose a WAV, MP3, M4A, FLAC, OGG, or AAC audio clip"
        try:
            source_size = source.stat().st_size
        except OSError as exc:
            return False, f"Could not inspect audio clip: {exc}"
        if source_size <= 0 or source_size > MAX_SOURCE_BYTES:
            return False, "Voice clip must be between 1 byte and 50 MiB"

        self.voice_directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".voice-import.", suffix=".wav", dir=self.voice_directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-180:]
                return False, f"Could not decode voice clip: {detail or 'ffmpeg failed'}"
            try:
                with wave.open(str(temporary), "rb") as clip:
                    seconds = clip.getnframes() / max(1, clip.getframerate())
                    valid_format = (
                        clip.getnchannels() == 1
                        and clip.getsampwidth() == 2
                        and clip.getframerate() == 16000
                    )
            except (OSError, EOFError, wave.Error) as exc:
                return False, f"Decoded voice clip is invalid: {exc}"
            if not valid_format:
                return False, "Decoded voice clip is not 16 kHz mono PCM16"
            if not MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
                return False, "Voice clip must be between 0.5 and 30 seconds"

            payload = temporary.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            slug = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")[:36]
            destination = self.voice_directory / f"{slug or 'voice'}-{digest[:12]}.wav"
            if not destination.exists():
                shutil.move(str(temporary), destination)
                destination.chmod(0o600)

            preset_id = f"custom-{digest[:16]}"
            with self._lock:
                profile = _read_profile(self.profile_path)
                selected: dict[str, Any] | None = None
                for preset in profile["presets"]:
                    if isinstance(preset, dict) and str(preset.get("id") or "") == preset_id:
                        selected = preset
                        break
                if selected is None:
                    label = source.stem.strip()[:80] or "Custom voice"
                    selected = {
                        "id": preset_id,
                        "label": label,
                        "speaker_file": str(destination.resolve()),
                        "default": False,
                    }
                    profile["presets"].append(selected)
                for preset in profile["presets"]:
                    if isinstance(preset, dict):
                        preset["default"] = preset is selected
                profile["speaker_file"] = str(destination.resolve())
                profile["name"] = preset_id
                _atomic_json(self.profile_path, profile)
            return True, f"Added and selected {_display_label(selected)}"
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"Could not import voice clip: {exc}"
        finally:
            temporary.unlink(missing_ok=True)
