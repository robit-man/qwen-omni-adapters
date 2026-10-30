from __future__ import annotations

import hashlib
import os
from pathlib import Path

import gguf
import numpy as np
import pytest

from qwen_omni_adapters import audio_prefill


def projector(path: Path, *, prefill: bool = False, prompt_hash: str | None = None) -> Path:
    writer = gguf.GGUFWriter(str(path), arch="clip")
    writer.add_string("robit.audio_bridge.schema", "robit.qwen3a-audio-bridge-projector.v1")
    writer.add_string("clip.audio.projector_type", "qwen3a")
    if prefill:
        writer.add_string(audio_prefill.PREFILL_KEY, audio_prefill.PREFILL_ABI)
        writer.add_string(audio_prefill.PROMPT_KEY, prompt_hash or audio_prefill.TRAINED_AUDIO_PROMPT_SHA256)
        writer.add_bool(audio_prefill.BOUNDARIES_KEY, False)
    writer.add_tensor("mm.a.mlp.2.weight", np.arange(64, dtype=np.float32).reshape(2, 32))
    writer.add_tensor("mm.a.mlp.2.bias", np.array([0.5, -1], dtype=np.float32))
    quantized = gguf.quants.quantize(np.arange(64, dtype=np.float32).reshape(2, 32),
                                    gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("v.frozen.weight", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_export_preserves_plain_and_quantized_weights_and_source(tmp_path):
    source = projector(tmp_path / "source.gguf")
    original = source.read_bytes()
    output = tmp_path / "output.gguf"
    report = audio_prefill.prepare_audio_prefill_projector(source, output)
    assert source.read_bytes() == original
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["all_tensors_byte_identical"] and report["tensor_count"] == 3
    before, after = gguf.GGUFReader(source), gguf.GGUFReader(output)
    assert [(t.name, t.tensor_type, t.data.tobytes()) for t in before.tensors] == [
        (t.name, t.tensor_type, t.data.tobytes()) for t in after.tensors
    ]


def test_export_refuses_an_existing_hardlink_without_changing_source(tmp_path):
    source = projector(tmp_path / "source.gguf")
    original = source.read_bytes()
    output = tmp_path / "existing.gguf"
    os.link(source, output)
    with pytest.raises(ValueError, match="fresh output"):
        audio_prefill.prepare_audio_prefill_projector(source, output)
    assert source.read_bytes() == output.read_bytes() == original


def test_new_prefill_cannot_start_with_a_stale_native_library(tmp_path, monkeypatch):
    source = projector(tmp_path / "new.gguf", prefill=True)
    monkeypatch.setattr(audio_prefill, "native_prefill_version", lambda _: 0)
    with pytest.raises(ValueError, match="requires native trained-prefill ABI 2"):
        audio_prefill.verify_audio_prefill_runtime(source, tmp_path / "llama-server")
    monkeypatch.setattr(audio_prefill, "native_prefill_version", lambda _: 2)
    assert audio_prefill.verify_audio_prefill_runtime(source, tmp_path / "llama-server")[
        "native_boundary_policy"
    ] == "none"


def test_unknown_training_prompt_fails_before_native_loading(tmp_path, monkeypatch):
    source = projector(tmp_path / "wrong.gguf", prefill=True, prompt_hash="0" * 64)
    def never_load(_):
        pytest.fail("an incompatible checkpoint reached native loading")
    monkeypatch.setattr(audio_prefill, "native_prefill_version", never_load)
    with pytest.raises(ValueError, match="mismatched trained audio prefill"):
        audio_prefill.verify_audio_prefill_runtime(source, tmp_path / "llama-server")


def test_legacy_projectors_keep_their_existing_boundary_policy(tmp_path, monkeypatch):
    source = projector(tmp_path / "legacy.gguf")
    monkeypatch.setattr(audio_prefill, "native_prefill_version", lambda _: pytest.fail("legacy ABI probe"))
    assert audio_prefill.verify_audio_prefill_runtime(source, tmp_path / "llama-server") == {
        "prefill_abi": None, "native_boundary_policy": "default"
    }
