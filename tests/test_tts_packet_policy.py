from __future__ import annotations

from dataclasses import replace

from runtime.tts_server import (
    Config,
    PersistentTTSWorker,
    _batch_specs,
    _synthesis_spec,
    _warm_spec,
)


def test_automatic_packets_follow_context_budget_and_preserve_explicit_overrides(tmp_path):
    base = Config(binary=tmp_path / "tts", model=tmp_path / "model", projector=tmp_path / "codec")
    assert _synthesis_spec(base, {"text": "Short reply."}).stream_frames == 8
    adaptive = replace(base, adaptive_context=True)
    assert _synthesis_spec(adaptive, {"text": "Short reply."}).stream_frames == 8
    assert _synthesis_spec(adaptive, {"text": "中" * 1800}).stream_frames == 8
    assert _synthesis_spec(adaptive, {"text": "Short reply.", "stream_frames": 12}).stream_frames == 12
    assert _synthesis_spec(replace(adaptive, stream_frames=8), {"text": "Short reply."}).stream_frames == 8


def test_mixed_length_batch_selects_one_cadence_without_a_mid_utterance_worker_reload(tmp_path):
    encoder = tmp_path / "encoder"
    encoder.write_bytes(b"encoder")
    config = Config(binary=tmp_path / "tts", model=tmp_path / "model", projector=tmp_path / "codec",
                    adaptive_context=True, reference_encoder=encoder)
    reference = {"speaker_file": "/voice.wav", "reference_text": "Reference words."}
    specs = _batch_specs(config, {**reference, "blocks": ["Short first phrase.", "中" * 1800]})
    assert {spec.context_size for spec in specs} == {8192}
    assert {spec.stream_frames for spec in specs} == {8}
    short = _batch_specs(config, {**reference, "blocks": ["First phrase.", "Next phrase."]})
    assert {spec.context_size for spec in short} == {4096}
    assert {spec.stream_frames for spec in short} == {4}
    explicit = _batch_specs(config, {"blocks": ["First phrase.", "中" * 1800], "stream_frames": 6})
    assert {spec.stream_frames for spec in explicit} == {6}


def test_full_reference_budget_cannot_select_an_unqualified_small_packet_profile(tmp_path):
    encoder = tmp_path / "encoder"
    encoder.write_bytes(b"encoder")
    config = Config(binary=tmp_path / "tts", model=encoder, projector=encoder,
                    reference_encoder=encoder, adaptive_context=True)
    short = _synthesis_spec(config, {"text": "Target.", "speaker_file": "/voice.wav", "reference_text": "Reference."})
    large = _synthesis_spec(config, {"text": "Target.", "speaker_file": "/voice.wav", "reference_text": "中" * 2000})
    assert short.stream_frames == 4
    assert large.stream_frames == 8


def test_automatic_warmup_keeps_the_matching_small_speech_profile(tmp_path):
    voice = tmp_path / "reference.wav"
    voice.write_bytes(b"reference")
    config = Config(binary=tmp_path / "tts", model=voice, projector=voice,
                    warm_speaker_file=str(voice), adaptive_context=True)
    warm = _warm_spec(config)
    speech = _synthesis_spec(config, {"text": "Spoken words.", "speaker_file": str(voice)})
    assert warm.stream_frames == speech.stream_frames == 8
    assert PersistentTTSWorker._profile(warm) == PersistentTTSWorker._profile(speech)
