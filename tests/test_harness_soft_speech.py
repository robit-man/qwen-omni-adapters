"""Soft speech reaches comprehension whole and at a usable level."""

from __future__ import annotations

import numpy as np

from harness.audio import (
    SPEECH_MAX_GAIN_DB,
    SPEECH_PEAK_CEILING,
    SPEECH_TARGET_DBFS,
    normalize_speech,
)
from harness.vad import Vad, VadConfig

RATE = 16_000
FRAME = 320


def _db(value: float) -> float:
    return 20.0 * np.log10(value)


def _speech_level(audio: np.ndarray) -> float:
    count = audio.size // FRAME
    levels = np.sqrt(np.mean(np.square(audio[: count * FRAME].reshape(count, FRAME)), axis=1))
    return float(np.sqrt(np.mean(np.square(levels[levels >= np.median(levels)]))))


def _soft_utterance(rng: np.random.Generator, speech_dbfs: float = -32.0) -> np.ndarray:
    """Quiet room pre-roll, then a voice-like burst, then a short tail."""

    room = rng.normal(0.0, 10 ** (-46 / 20), RATE * 3).astype(np.float32)
    t = np.arange(int(RATE * 1.8)) / RATE
    voice = np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    voice *= 10 ** (speech_dbfs / 20) / np.sqrt(np.mean(np.square(voice)))
    room[int(RATE * 0.85): int(RATE * 0.85) + voice.size] += voice.astype(np.float32)
    return room


def test_soft_speech_in_a_quiet_room_is_raised_toward_the_target():
    audio = _soft_utterance(np.random.default_rng(1))
    raised = normalize_speech(audio)
    gain_db = _db(_speech_level(raised) / _speech_level(audio))
    assert 4.0 < gain_db <= SPEECH_MAX_GAIN_DB + 0.01
    assert _db(_speech_level(raised)) <= SPEECH_TARGET_DBFS + 0.01
    assert np.abs(raised).max() <= SPEECH_PEAK_CEILING


def test_steady_room_noise_without_a_clear_voice_is_not_amplified():
    noise = np.random.default_rng(2).normal(0.0, 10 ** (-40 / 20), RATE * 3).astype(np.float32)
    np.testing.assert_array_equal(normalize_speech(noise), noise)


def test_loud_speech_is_never_attenuated():
    audio = _soft_utterance(np.random.default_rng(3), speech_dbfs=-12.0)
    np.testing.assert_array_equal(normalize_speech(audio), audio)


def test_one_click_does_not_cap_the_gain_for_the_whole_sentence():
    audio = _soft_utterance(np.random.default_rng(4))
    clicked = audio.copy()
    clicked[RATE * 2] = 0.95
    plain = _db(_speech_level(normalize_speech(audio)) / _speech_level(audio))
    with_click = _db(_speech_level(normalize_speech(clicked)) / _speech_level(clicked))
    assert abs(plain - with_click) < 0.5


def test_silence_and_empty_input_pass_through():
    assert normalize_speech(np.zeros(0, dtype=np.float32)).size == 0
    silent = np.zeros(RATE, dtype=np.float32)
    np.testing.assert_array_equal(normalize_speech(silent), silent)


def _frames(levels_ms: list[tuple[float, int]], frame_ms: float = 20.0):
    for level, duration_ms in levels_ms:
        for _ in range(int(duration_ms / frame_ms)):
            yield np.full(int(RATE * frame_ms / 1000), level, dtype=np.float32)


def _run(vad: Vad, frames, frame_ms: float = 20.0):
    now = 0.0
    for frame in frames:
        result = vad.process(frame, now, frame_ms)
        now += frame_ms
        if result.utterance is not None:
            return result.utterance
    return None


def test_pre_roll_is_a_duration_not_a_frame_count():
    config = VadConfig()
    utterance = _run(Vad(config), _frames([(0.003, 2000), (0.05, 600), (0.003, 1000)]))
    assert utterance is not None
    quiet_head = sum(1 for chunk in utterance.chunks if float(chunk[0]) < 0.01)
    assert quiet_head * 20.0 >= config.pre_roll_ms - 20.0


def test_a_soft_onset_that_dips_before_confirmation_is_kept():
    # 120 ms above threshold, a 40 ms dip, then sustained speech: the first
    # syllable must still head the utterance.
    utterance = _run(
        Vad(),
        _frames([(0.003, 2000), (0.04, 120), (0.003, 40), (0.05, 600), (0.003, 1000)]),
    )
    assert utterance is not None
    levels = [round(float(chunk[0]), 3) for chunk in utterance.chunks]
    first_loud = levels.index(0.04)
    assert levels[first_loud: first_loud + 6] == [0.04] * 6
