from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import gguf
import numpy as np
import pytest

from qwen_omni_adapters.indexed_sidecar import pack_indexed_tts_sidecar
from qwen_omni_adapters.ollama_sidecar import (
    INDEXED_BINDING_ANNOTATION,
    INDEXED_BINDING_MEDIA_TYPE,
    OMNI_LAYER_MEDIA_TYPE,
    OllamaSidecarError,
    attach_ollama_sidecar,
    manifest_path,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import (
    SingleGGUFError,
    inspect_monolithic_gguf,
    materialize_component_view,
)
from runtime.export_tts_reference_encoder import export_reference_encoder


def checkpoint(path: Path, arch: str, *, metadata=None) -> Path:
    writer = gguf.GGUFWriter(str(path), arch=arch)
    writer.add_name("public checkpoint fixture")
    writer.add_key_value(arch + ".block_count", 1, gguf.GGUFValueType.UINT32)
    for key, value in (metadata or {}).items():
        writer.add_string(key, value)
    writer.add_tensor("plain.weight", np.arange(64, dtype=np.float32).reshape(2, 32))
    quantized = gguf.quants.quantize(np.arange(64, dtype=np.float32).reshape(2, 32),
                                    gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("quantized.weight", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def sidecar(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = checkpoint(tmp_path / "model.gguf", "llama")
    projector = checkpoint(tmp_path / "codec.gguf", "clip")
    bundle = tmp_path / "bundle.gguf"
    pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector, out_gguf=bundle)
    return bundle, model, projector


def standard_tag(root: Path, name: str) -> Path:
    layers = []
    for role in ("model", "projector"):
        payload = (name + role).encode()
        digest = hashlib.sha256(payload).hexdigest()
        blob = root / "blobs" / ("sha256-" + digest)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        layers.append({"mediaType": "application/vnd.ollama.image." + role,
                       "digest": "sha256:" + digest, "size": len(payload)})
    path = manifest_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return path


def test_paths_do_not_duplicate_sidecar_and_quantized_views_remain_byte_identical(tmp_path):
    bundle, model, projector = sidecar(tmp_path)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    other_model = relocated / "renamed-language.gguf"
    other_projector = relocated / "renamed-codec.gguf"
    shutil.copyfile(model, other_model)
    shutil.copyfile(projector, other_projector)
    other_bundle = relocated / "other-backbone-sidecar.gguf"
    pack_indexed_tts_sidecar(tts_gguf=other_model, tts_projector_gguf=other_projector,
                             out_gguf=other_bundle)
    assert bundle.read_bytes() == other_bundle.read_bytes()
    root = tmp_path / "store"
    names = ("robit/ornith-fixture:edge-v2", "robit/qwen-fixture:edge-v2")
    standard_tag(root, names[0])
    standard_tag(root, names[1])
    attached = [attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root) for name in names]
    assert attached[0]["layer"]["digest"] == attached[1]["layer"]["digest"]
    assert attached[0]["binding_layer"]["digest"] != attached[1]["binding_layer"]["digest"]
    assert attached[0]["layer"]["annotations"][INDEXED_BINDING_ANNOTATION] != \
        attached[1]["layer"]["annotations"][INDEXED_BINDING_ANNOTATION]
    for name in names:
        resolved = resolve_ollama_sidecar(model=name, models_dir=root)
        assert resolved["profile"] == "trained-audio-bridge"
    prepared = prepare_ollama_sidecar(model=names[0], output_dir=tmp_path / "views", models_dir=root,
                                      reuse=True)
    for view, original in (("tts_model", model), ("tts_projector", projector)):
        assert Path(prepared["views"][view]["output"]).read_bytes() == original.read_bytes()
    reused = prepare_ollama_sidecar(model=names[1], output_dir=tmp_path / "views", models_dir=root,
                                    reuse=True)
    assert reused["component_index"]["materialized_views"] == []
    assert set(reused["component_index"]["reused_views"]) == {"tts_model", "tts_projector"}


@pytest.mark.parametrize("changed", ["model", "projector", "binding", "sidecar"])
def test_bound_tag_rejects_changed_trunk_projector_or_sidecar(tmp_path, changed):
    bundle, _, _ = sidecar(tmp_path)
    root = tmp_path / "store"
    name = "robit/bound:edge-v2"
    path = standard_tag(root, name)
    other = standard_tag(root, "robit/other:edge-v2")
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    manifest = json.loads(path.read_text())
    layer = next(x for x in manifest["layers"] if x["mediaType"] == OMNI_LAYER_MEDIA_TYPE)
    if changed == "binding":
        del layer["annotations"][INDEXED_BINDING_ANNOTATION]
        manifest["layers"] = [item for item in manifest["layers"]
                              if item["mediaType"] != INDEXED_BINDING_MEDIA_TYPE]
    elif changed == "sidecar":
        layer["annotations"][INDEXED_BINDING_ANNOTATION] = layer["annotations"][INDEXED_BINDING_ANNOTATION].replace(
            layer["digest"], "sha256:" + "0" * 64)
    else:
        index = 0 if changed == "model" else 1
        manifest["layers"][index] = json.loads(other.read_text())["layers"][index]
    path.write_text(json.dumps(manifest))
    with pytest.raises(OllamaSidecarError, match="binding"):
        resolve_ollama_sidecar(model=name, models_dir=root)


@pytest.mark.parametrize("swapped", [None, "model", "projector"])
def test_stock_descriptor_roundtrip_keeps_exact_binding_without_annotations(tmp_path, swapped):
    bundle, _, _ = sidecar(tmp_path)
    root, name = tmp_path / "store", "robit/portable:edge-v2"
    path = standard_tag(root, name)
    other = standard_tag(root, "robit/other:edge-v2")
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    manifest = json.loads(path.read_text())
    if swapped:
        index = 0 if swapped == "model" else 1
        manifest["layers"][index] = json.loads(other.read_text())["layers"][index]
    # Actual stock push drops descriptor annotations, but retains unknown
    # content-addressed layers. Exercise that persisted package, not a mock
    # success return from the resolver.
    manifest["layers"] = [{k: v for k, v in item.items() if k != "annotations"}
                          for item in manifest["layers"]]
    path.write_text(json.dumps(manifest))
    if swapped:
        with pytest.raises(OllamaSidecarError, match="binding"):
            resolve_ollama_sidecar(model=name, models_dir=root)
    else:
        result = prepare_ollama_sidecar(model=name, models_dir=root,
                                        output_dir=tmp_path / "views")
        assert set(result["views"]) == {"tts_model", "tts_projector"}


def test_existing_annotation_only_v2_tags_remain_supported(tmp_path):
    bundle, _, _ = sidecar(tmp_path)
    root, name = tmp_path / "store", "robit/previous:edge-v2"
    path = standard_tag(root, name)
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    manifest = json.loads(path.read_text())
    manifest["layers"] = [item for item in manifest["layers"]
                          if item["mediaType"] != INDEXED_BINDING_MEDIA_TYPE]
    path.write_text(json.dumps(manifest))
    assert resolve_ollama_sidecar(model=name, models_dir=root)["profile"] == "trained-audio-bridge"


@pytest.mark.parametrize("damage", ["missing", "bytes", "size", "digest", "oversized",
                                     "duplicate", "invalid_json", "other_binding"])
def test_binding_blob_damage_cannot_fall_back_to_a_valid_annotation(tmp_path, damage):
    bundle, _, _ = sidecar(tmp_path)
    root, name = tmp_path / "store", "robit/checked:edge-v2"
    path = standard_tag(root, name)
    attached = attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    manifest = json.loads(path.read_text())
    layer = next(item for item in manifest["layers"]
                 if item["mediaType"] == INDEXED_BINDING_MEDIA_TYPE)
    blob = root / "blobs" / layer["digest"].replace(":", "-", 1)
    if damage == "missing":
        blob.unlink()
    elif damage == "bytes":
        before = blob.read_bytes()
        blob.write_bytes(before.replace(b"language_model", b"language_modex"))
        assert blob.stat().st_size == layer["size"]
    elif damage == "size":
        layer["size"] += 1
    elif damage == "digest":
        layer["digest"] = "sha256:../../outside"
    elif damage == "oversized":
        layer["size"] = 65_537
    elif damage == "duplicate":
        manifest["layers"].append(dict(layer))
    elif damage == "invalid_json":
        raw = b"not json"
        digest = hashlib.sha256(raw).hexdigest()
        (root / "blobs" / ("sha256-" + digest)).write_bytes(raw)
        layer.update(digest="sha256:" + digest, size=len(raw))
    elif damage == "other_binding":
        other = "robit/other:edge-v2"
        standard_tag(root, other)
        wrong = attach_ollama_sidecar(model=other, bundle_gguf=bundle, models_dir=root)
        layer.update(wrong["binding_layer"])
        assert layer["digest"] != attached["binding_layer"]["digest"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(OllamaSidecarError, match="binding"):
        resolve_ollama_sidecar(model=name, models_dir=root)


def test_reattachment_reuses_one_small_binding_and_one_unchanged_sidecar(tmp_path):
    bundle, _, _ = sidecar(tmp_path)
    before = bundle.read_bytes()
    root, name = tmp_path / "store", "robit/repeated:edge-v2"
    path = standard_tag(root, name)
    first = attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    second = attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    assert first["binding_layer"] == second["binding_layer"]
    assert second["binding_layer"]["size"] < 1024
    manifest = json.loads(path.read_text())
    assert sum(item["mediaType"] == INDEXED_BINDING_MEDIA_TYPE for item in manifest["layers"]) == 1
    assert sum(item["mediaType"] == OMNI_LAYER_MEDIA_TYPE for item in manifest["layers"]) == 1
    assert bundle.read_bytes() == before
    assert not list((root / "blobs").glob(".omni-binding-*.partial"))


def test_failed_repack_preserves_previous_output_and_cleans_only_own_partial(tmp_path, monkeypatch):
    bundle, model, projector = sidecar(tmp_path)
    before = bundle.read_bytes()
    unrelated = tmp_path / "unrelated.partial"
    unrelated.write_bytes(b"preserve")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(gguf.GGUFWriter, "write_tensors_to_file", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                                 out_gguf=bundle, overwrite=True)
    assert bundle.read_bytes() == before and unrelated.read_bytes() == b"preserve"
    assert list(tmp_path.glob("bundle.gguf.*.partial")) == []
    with pytest.raises(SingleGGUFError, match="overwrite an input"):
        pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                                 out_gguf=model, overwrite=True)


@pytest.mark.parametrize("direct", [False, True])
def test_reference_encoder_shares_one_tag_and_preserves_all_three_component_views(tmp_path, direct):
    if direct and os.name == "nt":
        pytest.skip("Windows uses materialized component files")
    _, model, projector = sidecar(tmp_path)
    reference = checkpoint(tmp_path / "reference.gguf", "clip", metadata={
        "clip.audio.projector_type": "qwen3tts_refenc",
        "robit.reference_encoder.abi": "robit.qwen3tts.reference-encoder.v1"})
    bundle = tmp_path / "expanded.gguf"
    packed = pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                                     tts_reference_encoder_gguf=reference, out_gguf=bundle)
    assert inspect_monolithic_gguf(bundle)["valid"]
    root, name = tmp_path / "store", "robit/reference:edge-v2"
    standard_tag(root, name)
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    prepared = prepare_ollama_sidecar(model=name, output_dir=tmp_path / "views",
                                     models_dir=root, prefer_direct=direct)
    assert set(prepared["views"]) == {"tts_model", "tts_projector", "tts_reference_encoder"}
    for view, source in (("tts_model", model), ("tts_projector", projector), ("tts_reference_encoder", reference)):
        path = Path(prepared["views"][view]["output"])
        if direct:
            assert path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == packed["sha256"]
            path = tmp_path / (view + "-standalone.gguf")
            materialize_component_view(bundle_gguf=bundle, view=view, out_gguf=path)
        assert path.read_bytes() == source.read_bytes()
    assert len([layer for layer in json.loads(manifest_path(name, root).read_text())["layers"]
                if layer["mediaType"] == OMNI_LAYER_MEDIA_TYPE]) == 1
    bundle.write_bytes(bundle.read_bytes().replace(b"robit.qwen3tts.reference-encoder.v1",
                                                  b"robit.qwen3tts.reference-encoder.v9"))
    inspection = inspect_monolithic_gguf(bundle)
    assert not inspection["valid"] and "invalid indexed reference encoder ABI" in inspection["errors"]


def test_unqualified_reference_architecture_and_unpinned_source_are_rejected(tmp_path):
    _, model, projector = sidecar(tmp_path)
    reference = checkpoint(tmp_path / "reference.gguf", "clip")
    out = tmp_path / "invalid.gguf"
    with pytest.raises(SingleGGUFError, match="qualified.*ABI"):
        pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                                 tts_reference_encoder_gguf=reference, out_gguf=out)
    source = tmp_path / "wrong.safetensors"
    source.write_bytes(b"wrong checkpoint must be rejected before Torch is loaded")
    with pytest.raises(ValueError, match="pinned"):
        export_reference_encoder(source, out)
    assert not out.exists()


@pytest.mark.parametrize("direct", [False, True])
def test_optional_ctc_weights_share_tag_without_entering_default_runtime_views(tmp_path, direct):
    if direct and os.name == "nt":
        pytest.skip("Windows uses materialized files")
    _, model, projector = sidecar(tmp_path)
    source = tmp_path / "ctc.gguf"
    writer = gguf.GGUFWriter(str(source), arch="robit_audio_ctc")
    writer.add_string("robit.audio_ctc.schema", "robit.omni.audio-ctc-head.v1")
    writer.add_string("robit.audio_ctc.encoder_sha256", "a" * 64)
    for i in range(8):
        writer.add_tensor(f"fixture_{i}.weight", np.arange(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    bundle = tmp_path / "with-head.gguf"
    pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                             audio_ctc_head_gguf=source, out_gguf=bundle)
    inspection = inspect_monolithic_gguf(bundle)
    assert inspection["valid"]
    assert inspection["manifest"]["runtime"]["disabled_components"] == ["audio_ctc_head"]
    root, name = tmp_path / "store", "robit/head-fixture:edge-v2"
    standard_tag(root, name)
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    prepared = prepare_ollama_sidecar(model=name, output_dir=tmp_path / "default", models_dir=root)
    assert set(prepared["views"]) == {"tts_model", "tts_projector"}
    requested = prepare_ollama_sidecar(model=name, output_dir=tmp_path / "explicit",
                                      models_dir=root, views=("audio_ctc_head",), prefer_direct=direct)
    view = Path(requested["views"]["audio_ctc_head"]["output"])
    if direct:
        assert view.is_symlink() and view.resolve() == Path(resolve_ollama_sidecar(model=name, models_dir=root)["bundle"])
        view = tmp_path / "materialized-head.gguf"
        materialize_component_view(bundle_gguf=bundle, view="audio_ctc_head", out_gguf=view)
    assert view.read_bytes() == source.read_bytes()
    bundle.write_bytes(bundle.read_bytes().replace(b"robit.omni.audio-ctc-head.v1", b"robit.omni.audio-ctc-head.v9"))
    inspection = inspect_monolithic_gguf(bundle)
    assert not inspection["valid"] and "invalid indexed audio CTC head ABI" in inspection["errors"]


def test_optional_head_rejects_generic_wrong_architecture_before_packing(tmp_path):
    _, model, projector = sidecar(tmp_path)
    wrong = checkpoint(tmp_path / "wrong-head.gguf", "llama")
    output = tmp_path / "invalid-head-bundle.gguf"
    with pytest.raises(SingleGGUFError, match="head.*ABI"):
        pack_indexed_tts_sidecar(tts_gguf=model, tts_projector_gguf=projector,
                                 audio_ctc_head_gguf=wrong, out_gguf=output)
    assert not output.exists()


def test_index_does_not_accept_tensors_outside_its_declared_views(tmp_path):
    bundle, _, _ = sidecar(tmp_path)
    raw = bundle.read_bytes()
    # A same-length tensor-name mutation preserves the binary file structure.
    bundle.write_bytes(raw.replace(b"s.t.m.plain.weight", b"alien.plain.weight"))
    inspection = inspect_monolithic_gguf(bundle)
    assert not inspection["valid"]
    assert any("outside" in error for error in inspection["errors"])


@pytest.mark.skipif(os.name == "nt", reason="Windows retains materialized views")
def test_direct_views_alias_one_bound_blob_without_copying_payloads_and_cleanup_preserves_it(tmp_path):
    bundle, _, _ = sidecar(tmp_path)
    root = tmp_path / "store"
    name = "robit/direct:edge-v2"
    standard_tag(root, name)
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    resolved = resolve_ollama_sidecar(model=name, models_dir=root)
    before = Path(resolved["bundle"]).read_bytes()
    directory = tmp_path / "aliases"
    prepared = prepare_ollama_sidecar(model=name, output_dir=directory, models_dir=root, prefer_direct=True)
    assert prepared["component_index"]["materialized_bytes"] == 0
    assert set(prepared["component_index"]["direct_views"]) == {"tts_model", "tts_projector"}
    for report in prepared["views"].values():
        alias = Path(report["output"])
        assert alias.is_symlink() and alias.resolve() == Path(resolved["bundle"])
        assert report["read_mode"] == "indexed-source-alias"
    repeated = prepare_ollama_sidecar(model=name, output_dir=directory, models_dir=root, prefer_direct=True)
    assert repeated["component_index"]["materialized_bytes"] == 0
    # The ordinary materializer continues to reject aliases; it must never
    # silently treat the multigraph container as an extracted standalone GGUF.
    with pytest.raises(SingleGGUFError, match="symlink"):
        prepare_ollama_sidecar(model=name, output_dir=directory, models_dir=root, reuse=True)
    for path in directory.glob("*.gguf"):
        path.unlink()
    assert Path(resolved["bundle"]).read_bytes() == before


@pytest.mark.skipif(os.name == "nt", reason="Windows retains materialized views")
def test_direct_rebinding_replaces_only_aliases_and_never_writes_through_targets(tmp_path):
    bundle, _, _ = sidecar(tmp_path)
    root = tmp_path / "store"
    name = "robit/direct:edge-v2"
    standard_tag(root, name)
    attach_ollama_sidecar(model=name, bundle_gguf=bundle, models_dir=root)
    directory = tmp_path / "aliases"
    directory.mkdir()
    unrelated = tmp_path / "preserved-reference.gguf"
    unrelated.write_bytes(b"do not follow this link")
    (directory / "tts-model.gguf").symlink_to(unrelated)
    (directory / "tts-projector.gguf").write_bytes(b"older derived view")
    with pytest.raises(SingleGGUFError, match="exists"):
        prepare_ollama_sidecar(model=name, output_dir=directory, models_dir=root, prefer_direct=True)
    assert (directory / "tts-model.gguf").readlink() == unrelated
    prepare_ollama_sidecar(model=name, output_dir=directory, models_dir=root, prefer_direct=True, overwrite=True)
    assert unrelated.read_bytes() == b"do not follow this link"
    assert (directory / "tts-model.gguf").resolve() == (directory / "tts-projector.gguf").resolve()
