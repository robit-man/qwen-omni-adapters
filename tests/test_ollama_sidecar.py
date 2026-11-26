from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from qwen_omni_adapters.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    OllamaSidecarError,
    attach_ollama_sidecar,
    manifest_path,
    prepare_ollama_sidecar,
    resolve_models_dir,
    resolve_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import pack_monolithic_gguf


def _make_lightweight_sidecar(path: Path, digests: dict[str, str] | None = None) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    manifest = {
        "schema": "robit.ollama-split-audio-bridge.v1" if digests else "robit.ollama-audio-bridge.v1",
        "profile": "trained-audio-bridge",
        "components": [
            {"name": "tts_model"},
            {"name": "tts_projector"},
        ],
    }
    if digests:
        manifest["standard_layers"] = {"digests": digests}
        manifest["components"].append({"name": "comprehension_projector"})
    writer = gguf.GGUFWriter(str(path), arch="clip")
    writer.add_key_value(
        "robit.audio_bundle.schema",
        manifest["schema"],
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "robit.audio_bundle.manifest",
        json.dumps(manifest),
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "robit.audio_bundle.component.tts_model.kv.general.architecture",
        "llama",
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "robit.audio_bundle.component.tts_projector.kv.general.architecture",
        "clip",
        gguf.GGUFValueType.STRING,
    )
    writer.add_tensor("s.t.m.tts.weight", np.asarray([[1.0]], dtype=np.float32))
    writer.add_tensor("s.t.p.projector.weight", np.asarray([[2.0]], dtype=np.float32))
    if digests:
        prefix = "robit.audio_bundle.component.comprehension_projector.kv."
        writer.add_key_value(prefix + "clip.has_audio_encoder", True, gguf.GGUFValueType.BOOL)
        writer.add_key_value(prefix + "clip.audio.projector_type", "qwen3a", gguf.GGUFValueType.STRING)
        writer.add_key_value(prefix + "clip.audio.projection_dim", 2, gguf.GGUFValueType.UINT32)
        writer.add_tensor("a.c.p.mm.a.mlp.2.weight", np.ones((2, 1280), dtype=np.float32))
        writer.add_tensor("a.c.p.mm.a.mlp.2.bias", np.zeros(2, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _make_gguf(path: Path, name: str, value: float) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_key_value("general.name", name, gguf.GGUFValueType.STRING)
    writer.add_key_value("llama.block_count", 1, gguf.GGUFValueType.UINT32)
    writer.add_tensor(f"{name}.weight", np.asarray([[value, value + 1]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("content_addressed", [False, True])
def test_attach_and_resolve_omni_sidecar_layer(tmp_path: Path, content_addressed: bool) -> None:
    base = tmp_path / "base.gguf"
    comprehension = tmp_path / "comprehension.gguf"
    tts = tmp_path / "tts.gguf"
    bundle = tmp_path / "bundle.gguf"
    _make_gguf(base, "base", 1)
    _make_gguf(comprehension, "comprehension", 2)
    _make_gguf(tts, "tts", 3)
    pack_monolithic_gguf(
        base_gguf=base,
        comprehension_gguf=comprehension,
        tts_gguf=tts,
        out_gguf=bundle,
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    path = manifest_path("robit/test:q4km", models_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:base",
                        "size": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original = path.read_bytes()
    if content_addressed:
        original_blob = models_dir / "blobs" / f"sha256-{hashlib.sha256(original).hexdigest()}"
        original_blob.parent.mkdir()
        path.replace(original_blob)
        path = models_dir / "manifests-v2" / "ollama.com" / "robit" / "test" / "q4km"
        path.parent.mkdir(parents=True)
        target = os.path.relpath(original_blob, start=path.parent)
        path.symlink_to(target)
        sibling = path.with_name("other")
        sibling.symlink_to(target)

    attached = attach_ollama_sidecar(
        model="robit/test:q4km",
        bundle_gguf=bundle,
        models_dir=models_dir,
    )
    resolved = resolve_ollama_sidecar(model="robit/test:q4km", models_dir=models_dir)

    assert attached["layer"]["mediaType"] == OMNI_LAYER_MEDIA_TYPE
    assert resolved["bundle"] == str(
        models_dir / "blobs" / attached["layer"]["digest"].replace(":", "-", 1)
    )
    assert resolved["inspection"]["view_tensor_counts"]["base"] == 1
    updated = json.loads(path.read_text(encoding="utf-8"))
    if content_addressed:
        assert path.is_symlink()
        assert path.resolve().name == f"sha256-{hashlib.sha256(path.read_bytes()).hexdigest()}"
        assert sibling.read_bytes() == original
    assert [layer["mediaType"] for layer in updated["layers"]] == [
        "application/vnd.ollama.image.model",
        OMNI_LAYER_MEDIA_TYPE,
    ]

    prepared = prepare_ollama_sidecar(
        model="robit/test:q4km",
        output_dir=tmp_path / "runtime-cache",
        views=("comprehension_model", "tts_model"),
        models_dir=models_dir,
    )
    assert prepared["disposable_cache"] is True
    assert set(prepared["views"]) == {"comprehension_model", "tts_model"}
    assert Path(prepared["views"]["tts_model"]["output"]).is_file()


def test_lightweight_bridge_resolves_standard_layers_and_materializes_only_tts(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    blobs = models_dir / "blobs"
    blobs.mkdir(parents=True)
    language = blobs / "sha256-language"
    projector = blobs / "sha256-projector"
    sidecar = blobs / "sha256-sidecar"
    language.write_bytes(b"language")
    projector.write_bytes(b"projector")
    _make_lightweight_sidecar(sidecar)
    path = manifest_path("robit/bridge:q4km", models_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:language",
                        "size": language.stat().st_size,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": "sha256:projector",
                        "size": projector.stat().st_size,
                    },
                    {
                        "mediaType": OMNI_LAYER_MEDIA_TYPE,
                        "digest": "sha256:sidecar",
                        "size": sidecar.stat().st_size,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_ollama_sidecar(
        model="robit/bridge:q4km",
        models_dir=models_dir,
    )
    prepared = prepare_ollama_sidecar(
        model="robit/bridge:q4km",
        output_dir=tmp_path / "runtime-cache",
        models_dir=models_dir,
    )

    assert resolved["profile"] == "trained-audio-bridge"
    assert resolved["standard_layers"]["language_model"]["path"] == str(language)
    assert resolved["standard_layers"]["projector"]["path"] == str(projector)
    assert set(prepared["views"]) == {"tts_model", "tts_projector"}
    assert not (tmp_path / "runtime-cache" / "comprehension-model.gguf").exists()


def test_split_bridge_preserves_native_vision_and_requires_bound_layers(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    layer_inputs = {
        "language_model": (tmp_path / "language.gguf", "model"),
        "projector": (tmp_path / "vision.gguf", "projector"),
    }
    language, _ = layer_inputs["language_model"]
    vision, _ = layer_inputs["projector"]
    _make_gguf(language, "language", 1)
    writer = gguf.GGUFWriter(str(vision), arch="clip")
    writer.add_key_value("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL)
    writer.add_key_value("clip.vision.projector_type", "inkling2v", gguf.GGUFValueType.STRING)
    writer.add_key_value("clip.vision.projection_dim", 2, gguf.GGUFValueType.UINT32)
    writer.add_tensor("v.weight", np.asarray([[3.0, 4.0]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    models_dir = tmp_path / "models"
    blobs = models_dir / "blobs"
    blobs.mkdir(parents=True)
    digests = {}
    layers = []
    for key, (file, media_type) in layer_inputs.items():
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        digests[key] = f"sha256:{digest}"
        (blobs / f"sha256-{digest}").write_bytes(data)
        layers.append({
            "mediaType": f"application/vnd.ollama.image.{media_type}",
            "digest": digests[key],
            "size": len(data),
        })
    bundle = tmp_path / "split.gguf"
    _make_lightweight_sidecar(bundle, digests)
    model = "robit/inkling-test:iq4-xs"
    manifest = manifest_path(model, models_dir)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": layers}), encoding="utf-8")
    attached = attach_ollama_sidecar(model=model, bundle_gguf=bundle, models_dir=models_dir)
    assert resolve_ollama_sidecar(model=model, models_dir=models_dir)["split_projector"]
    output = tmp_path / "runtime"
    prepared = prepare_ollama_sidecar(model=model, output_dir=output, models_dir=models_dir)
    assert set(prepared["views"]) == {"comprehension_projector", "tts_model", "tts_projector"}
    assert (output / "split-audio.sha256").read_text().strip() == attached["layer"]["digest"]
    merged = gguf.GGUFReader(str(output / "comprehension-projector.gguf"))
    assert merged.fields["clip.vision.projector_type"].contents() == "inkling2v"
    assert merged.fields["clip.audio.projector_type"].contents() == "qwen3a"
    assert {tensor.name for tensor in merged.tensors} == {
        "v.weight", "mm.a.mlp.2.weight", "mm.a.mlp.2.bias",
    }
    assert merged.tensors[0].data.tobytes() == gguf.GGUFReader(str(vision)).tensors[0].data.tobytes()

    altered = json.loads(manifest.read_text())
    altered["layers"][1]["digest"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(OllamaSidecarError, match="bound projector layer"):
        resolve_ollama_sidecar(model=model, models_dir=models_dir)


def test_a_populated_store_is_preferred_over_one_that_merely_exists(monkeypatch, tmp_path):
    """An empty ~/.ollama/models must not shadow the real service store.

    Any `ollama` client invocation creates that directory, so taking the first
    candidate that exists produced "manifest not found" for a tag that was
    plainly installed.
    """

    populated = tmp_path / "service"
    (populated / "manifests").mkdir(parents=True)
    empty = tmp_path / "home" / ".ollama" / "models"
    empty.mkdir(parents=True)

    monkeypatch.setenv("OLLAMA_MODELS", str(empty))

    assert resolve_models_dir(populated) == populated.resolve()


def test_an_env_override_still_wins_when_it_holds_manifests(monkeypatch, tmp_path):
    override = tmp_path / "override"
    (override / "manifests").mkdir(parents=True)
    other = tmp_path / "other"
    (other / "manifests").mkdir(parents=True)

    monkeypatch.setenv("OLLAMA_MODELS", str(override))

    assert resolve_models_dir() == override.resolve()


def test_the_linux_service_store_is_a_known_candidate():
    """The official installer runs ollama as its own user with this home."""

    source = Path(resolve_models_dir.__globals__["__file__"]).read_text()

    assert "/usr/share/ollama/.ollama/models" in source
