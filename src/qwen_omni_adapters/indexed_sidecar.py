"""Backbone-independent, deterministic packaging of the bundled TTS graphs."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_omni_adapters.single_gguf import (
    BUNDLE_NAMESPACE,
    INDEXED_AUDIO_BRIDGE_SCHEMA,
    TTS_COMPONENT,
    TTS_PROJECTOR_COMPONENT,
    SingleGGUFError,
    _add_reader_tensor,
    _architecture,
    _copy_metadata,
    _require_gguf,
    _sha256,
    inspect_monolithic_gguf,
)


def pack_indexed_tts_sidecar(*, tts_gguf: Path, tts_projector_gguf: Path,
                             out_gguf: Path, overwrite: bool = False) -> dict[str, Any]:
    """Identical checkpoints produce one blob regardless of paths or backbone.

    Exact language/projector bindings belong to each tag's layer descriptor;
    putting them into this payload would duplicate all TTS weights per backbone.
    This still uses independent executable views, not a stock Ollama graph.
    """
    gguf, np = _require_gguf()
    inputs = [path.expanduser().resolve() for path in (tts_gguf, tts_projector_gguf)]
    for path in inputs:
        if not path.is_file():
            raise SingleGGUFError(f"GGUF input does not exist: {path}")
    identities = [(path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns,
                   path.stat().st_ctime_ns) for path in inputs]
    out = out_gguf.expanduser().resolve()
    if out in inputs:
        raise SingleGGUFError("sidecar output must not overwrite an input checkpoint")
    if out.exists() and not overwrite:
        raise SingleGGUFError(f"output already exists: {out}")
    readers = [(component, path, gguf.GGUFReader(str(path))) for component, path in
               zip((TTS_COMPONENT, TTS_PROJECTOR_COMPONENT), inputs, strict=True)]
    components = []
    index = {}
    for component, path, reader in readers:
        if not reader.tensors or not _architecture(reader):
            raise SingleGGUFError(f"component has no architecture or tensors: {path}")
        digest = _sha256(path)
        components.append(replace(component, source="sha256:" + digest).to_dict())
        index[component.name] = {
            "tensor_prefix": component.tensor_prefix, "metadata_prefix": component.metadata_prefix,
            "architecture": _architecture(reader), "tensor_count": len(reader.tensors),
            "gguf_sha256": digest, "gguf_size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": INDEXED_AUDIO_BRIDGE_SCHEMA, "profile": "trained-audio-bridge",
        "physical_bundle_artifacts": 1, "components": components,
        "component_index": {"schema": "robit.gguf-component-index.v2", "components": index},
        "standard_layers": {"binding": "io.robit.omni.binding.v2 in the Ollama sidecar layer descriptor"},
        "runtime": {"language_trunk_copies": 1, "omni_thinker_required": False,
                    "materialized_views": ["tts_model", "tts_projector"],
                    "combined_projector_is_standard_ollama_layer": True},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=out.name + ".", suffix=".partial", dir=out.parent)
    os.close(fd)
    partial = Path(filename)
    writer = gguf.GGUFWriter(str(partial), arch="clip")
    try:
        writer.add_name("Robit Omni shared TTS sidecar v2")
        for component, _, reader in readers:
            _, skipped = _copy_metadata(reader=reader, writer=writer, prefix=component.metadata_prefix)
            unexpected = set(skipped) - {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
            if unexpected:
                raise SingleGGUFError(f"cannot preserve component metadata: {sorted(unexpected)}")
        writer.add_key_value(BUNDLE_NAMESPACE + ".schema", INDEXED_AUDIO_BRIDGE_SCHEMA,
                             gguf.GGUFValueType.STRING)
        writer.add_key_value(BUNDLE_NAMESPACE + ".manifest",
                             json.dumps(manifest, separators=(",", ":"), sort_keys=True),
                             gguf.GGUFValueType.STRING)
        for component, _, reader in readers:
            for tensor in reader.tensors:
                _add_reader_tensor(writer, tensor, component.tensor_prefix + tensor.name, np)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        current = [(path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns,
                    path.stat().st_ctime_ns) for path in inputs]
        if current != identities:
            raise SingleGGUFError("checkpoint changed while packing; require immutable inputs")
        inspection = inspect_monolithic_gguf(partial)
        if not inspection["valid"]:
            raise SingleGGUFError(f"indexed sidecar inspection failed: {inspection['errors']}")
        os.replace(partial, out)
    finally:
        writer.close()
        partial.unlink(missing_ok=True)
    return {"schema": INDEXED_AUDIO_BRIDGE_SCHEMA, "output": str(out),
            "size_bytes": out.stat().st_size, "sha256": _sha256(out),
            "component_index": manifest["component_index"]}
