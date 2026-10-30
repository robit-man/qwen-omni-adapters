from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from qwen_omni_adapters import component_views
from qwen_omni_adapters.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    manifest_path,
    prepare_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import ComponentViewIndex, SingleGGUFError


@pytest.fixture
def installed(tmp_path):
    models = tmp_path / "models"
    blobs = models / "blobs"
    blobs.mkdir(parents=True)
    source = blobs / "sha256-sidecar"
    manifest = {"schema": "robit.ollama-audio-bridge.v1", "components": [
        {"name": "tts_model"}, {"name": "tts_projector"},
    ]}
    writer = gguf.GGUFWriter(str(source), arch="clip")
    writer.add_string("robit.audio_bundle.schema", manifest["schema"])
    writer.add_string("robit.audio_bundle.manifest", json.dumps(manifest))
    for name, arch in (("tts_model", "qwen3tts"), ("tts_projector", "clip")):
        prefix = f"robit.audio_bundle.component.{name}.kv."
        writer.add_string(prefix + "general.architecture", arch)
        writer.add_string(prefix + "general.name", "unchanged cloning checkpoint")
        writer.add_array(prefix + "tokenizer.ggml.tokens", ["speaker", "audio"])
    packed = gguf.quantize(
        np.linspace(-2, 2, 64, dtype=np.float32).reshape(2, 32),
        gguf.GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor("s.t.m.tts.weight", packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("s.t.p.codec.weight", np.array([[2.0, 3.0]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    layers = []
    for name, media in (("language", "model"), ("projector", "projector")):
        (blobs / f"sha256-{name}").write_bytes(name.encode())
        layers.append({"mediaType": f"application/vnd.ollama.image.{media}",
                       "digest": f"sha256:{name}", "size": len(name)})
    layers.append({"mediaType": OMNI_LAYER_MEDIA_TYPE, "digest": "sha256:sidecar",
                   "size": source.stat().st_size})
    path = manifest_path("robit/test:bridge", models)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return {"model": "robit/test:bridge", "models_dir": models,
            "output_dir": tmp_path / "cache"}


def prepare(installed, **kwargs):
    return prepare_ollama_sidecar(**installed, reuse=True, **kwargs)


def test_cold_preparation_indexes_source_once_and_preserves_quantized_payload(installed, monkeypatch):
    original = gguf.GGUFReader
    opened = []

    def reader(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(gguf, "GGUFReader", reader)
    result = prepare(installed)
    # One existing resolver inspection plus one shared extraction inventory.
    assert opened.count(result["bundle"]) == 2
    assert result["component_index"]["reused_views"] == []
    assert result["component_index"]["materialized_views"] == ["tts_model", "tts_projector"]
    source = original(result["bundle"])
    for view, prefix in (("tts_model", "s.t.m."), ("tts_projector", "s.t.p.")):
        output = original(result["views"][view]["output"])
        selected = [tensor for tensor in source.tensors if tensor.name.startswith(prefix)]
        assert output.fields["general.name"].contents() == "unchanged cloning checkpoint"
        assert output.fields["tokenizer.ggml.tokens"].contents() == ["speaker", "audio"]
        assert len(output.tensors) == len(selected)
        for actual, expected in zip(output.tensors, selected, strict=True):
            assert actual.name == expected.name.removeprefix(prefix)
            assert actual.tensor_type == expected.tensor_type
            assert actual.shape.tolist() == expected.shape.tolist()
            assert actual.data.tobytes() == expected.data.tobytes()


def test_warm_preparation_writes_no_component_or_index(installed, monkeypatch):
    cold = prepare(installed)
    files = [Path(item["output"]) for item in cold["views"].values()]
    files.append(Path(cold["component_index"]["path"]))
    identities = {path: path.stat() for path in files}

    def forbidden(*args, **kwargs):
        pytest.fail("verified warm views must not be reconstructed")

    monkeypatch.setattr(ComponentViewIndex, "materialize", forbidden)
    warm = prepare(installed, overwrite=True)
    assert warm["component_index"]["materialized_bytes"] == 0
    assert warm["component_index"]["reused_bytes"] == sum(path.stat().st_size for path in files[:-1])
    assert all(report["reused"] for report in warm["views"].values())
    for path in files:
        assert (path.stat().st_ino, path.stat().st_mtime_ns) == (
            identities[path].st_ino, identities[path].st_mtime_ns,
        )
    # A valid cache is idempotent even without permission to replace stale files.
    assert prepare(installed)["component_index"]["materialized_bytes"] == 0


@pytest.mark.parametrize("fault", ["missing", "same_size_corruption"])
def test_only_the_missing_or_corrupt_view_is_rebuilt(installed, fault):
    cold = prepare(installed)
    bad = Path(cold["views"]["tts_model"]["output"])
    good = Path(cold["views"]["tts_projector"]["output"])
    expected = hashlib.sha256(bad.read_bytes()).hexdigest()
    identity = good.stat()
    if fault == "missing":
        bad.unlink()
    else:
        with bad.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(bytes([bad.read_bytes()[-1] ^ 1]))
    repaired = prepare(installed, overwrite=True)
    assert repaired["component_index"]["materialized_views"] == ["tts_model"]
    assert repaired["component_index"]["reused_views"] == ["tts_projector"]
    assert hashlib.sha256(bad.read_bytes()).hexdigest() == expected
    assert good.stat().st_ino == identity.st_ino
    assert good.stat().st_mtime_ns == identity.st_mtime_ns


@pytest.mark.parametrize("fault", ["truncated", "oversized", "version", "digest", "source_identity"])
def test_invalid_receipt_cannot_authorize_reuse(installed, fault):
    cold = prepare(installed)
    path = Path(cold["component_index"]["path"])
    data = json.loads(path.read_text())
    if fault == "truncated":
        path.write_text("{broken")
    elif fault == "oversized":
        path.write_bytes(b" " * (component_views.MAX_INDEX_BYTES + 1))
    else:
        field = {"version": "extractor_version", "digest": "bundle_digest",
                 "source_identity": "source_identity"}[fault]
        data["binding"][field] = "obsolete"
        path.write_text(json.dumps(data))
    repaired = prepare(installed, overwrite=True)
    assert repaired["component_index"]["reused_views"] == []
    assert repaired["component_index"]["materialized_views"] == ["tts_model", "tts_projector"]


def test_unverified_existing_output_requires_overwrite(installed):
    cold = prepare(installed)
    bad = Path(cold["views"]["tts_model"]["output"])
    bad.write_bytes(b"not the reference-conditioned TTS weights")
    before = bad.read_bytes()
    with pytest.raises(SingleGGUFError, match="output already exists"):
        prepare(installed)
    assert bad.read_bytes() == before
    with pytest.raises(SingleGGUFError, match="output already exists"):
        prepare_ollama_sidecar(**installed)


def test_changed_installed_blob_invalidates_receipt_even_at_the_same_size(installed):
    cold = prepare(installed)
    source = Path(cold["bundle"])
    with source.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(bytes([source.read_bytes()[-1] ^ 1]))
    rebuilt = prepare(installed, overwrite=True)
    assert rebuilt["component_index"]["reused_views"] == []


def test_output_symlink_is_rejected_without_overwriting_its_target(installed, tmp_path):
    cold = prepare(installed)
    output = Path(cold["views"]["tts_model"]["output"])
    output.unlink()
    other = tmp_path / "unrelated.gguf"
    other.write_bytes(b"keep")
    output.symlink_to(other)
    with pytest.raises(SingleGGUFError, match="symlink"):
        prepare(installed, overwrite=True)
    assert other.read_bytes() == b"keep"


def test_source_mutation_during_preparation_never_publishes_a_valid_index(installed, monkeypatch):
    original = ComponentViewIndex.materialize

    def changed(self, **kwargs):
        report = original(self, **kwargs)
        self.source.touch()
        return report

    monkeypatch.setattr(ComponentViewIndex, "materialize", changed)
    with pytest.raises(SingleGGUFError, match="bundle changed"):
        prepare(installed)
    assert not (installed["output_dir"] / component_views.INDEX_FILENAME).exists()


def test_failed_rebuild_does_not_publish_an_incomplete_receipt(installed, monkeypatch):
    original = ComponentViewIndex.materialize

    def fail_second(self, **kwargs):
        if kwargs["view"] == "tts_projector":
            raise OSError("disk full")
        return original(self, **kwargs)

    monkeypatch.setattr(ComponentViewIndex, "materialize", fail_second)
    with pytest.raises(OSError, match="disk full"):
        prepare(installed)
    assert not (installed["output_dir"] / component_views.INDEX_FILENAME).exists()


def test_subset_preparation_preserves_other_verified_entries(installed):
    cold = prepare(installed)
    subset = prepare(installed, views=("tts_model",))
    assert subset["component_index"]["reused_views"] == ["tts_model"]
    complete = prepare(installed)
    assert complete["component_index"]["reused_views"] == ["tts_model", "tts_projector"]
    assert Path(cold["component_index"]["path"]).is_file()
