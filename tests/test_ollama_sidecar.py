from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_omni_adapters.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    attach_ollama_sidecar,
    manifest_path,
    prepare_ollama_sidecar,
    resolve_models_dir,
    resolve_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import pack_monolithic_gguf


def _make_lightweight_sidecar(path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    manifest = {
        "schema": "robit.ollama-audio-bridge.v1",
        "profile": "trained-audio-bridge",
        "components": [
            {"name": "tts_model"},
            {"name": "tts_projector"},
        ],
    }
    writer = gguf.GGUFWriter(str(path), arch="clip")
    writer.add_key_value(
        "robit.audio_bundle.schema",
        "robit.ollama-audio-bridge.v1",
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


def test_attach_and_resolve_omni_sidecar_layer(tmp_path: Path) -> None:
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
