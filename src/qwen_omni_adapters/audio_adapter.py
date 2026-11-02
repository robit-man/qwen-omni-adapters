"""An exact-pair-bound FFN LoRA; audio perception is its only execution scope."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from qwen_omni_adapters.single_gguf import _reader_metadata, _require_gguf

AUDIO_ADAPTER_MEDIA_TYPE = "application/vnd.robit.ollama.omni.audio-adapter.v1+gguf"
AUDIO_ADAPTER_ABI = "robit.omni.conditional-audio-lora.v1"
AUDIO_ADAPTER_FULL_FFN_ABI = "robit.omni.conditional-audio-lora.v2"
AUDIO_ADAPTER_MAX_BYTES = 256 * 1024**2
_SCOPE_ABI = {"last8-ffn": AUDIO_ADAPTER_ABI, "full-ffn": AUDIO_ADAPTER_FULL_FFN_ABI}
_SCOPE_MAX_BYTES = {"last8-ffn": 64 * 1024**2, "full-ffn": AUDIO_ADAPTER_MAX_BYTES}
_PREFIX = "robit.audio_adapter."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            and all(c in "0123456789abcdef" for c in value[7:]))


def _validate(reader: Any, model: Any, *, scope: str = "last8-ffn") -> None:
    if not isinstance(scope, str) or scope not in _SCOPE_ABI:
        raise ValueError("unsupported audio adapter scope")
    gguf, np = _require_gguf()
    metadata, base = _reader_metadata(reader), _reader_metadata(model)
    arch = base.get("general.architecture")
    if (arch != "qwen35" or metadata.get("general.architecture") != arch
            or metadata.get("general.type") != "adapter"
            or metadata.get("adapter.type") != "lora"
            or metadata.get("adapter.lora.alpha") != 8.0):
        raise ValueError("audio adapter requires the matching qwen35 rank8/alpha8 FFN LoRA")
    blocks = base.get(f"{arch}.block_count")
    width = base.get(f"{arch}.embedding_length")
    hidden = base.get(f"{arch}.feed_forward_length")
    if any(type(v) is not int or v <= 0 for v in (blocks, width, hidden)) or blocks < 8:
        raise ValueError("audio adapter base dimensions are unavailable")
    layers = range(blocks - 8, blocks) if scope == "last8-ffn" else range(blocks)
    expected_count = 6 * len(layers)
    names = [tensor.name for tensor in reader.tensors]
    description = "last8" if scope == "last8-ffn" else "full"
    # Reject impossible counts before allocating a base-sized name inventory.
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError(f"audio adapter must contain exactly the matching {description} FFN pairs")
    expected = {}
    for layer in layers:
        for module, inputs, outputs in (("ffn_down", hidden, width),
                                         ("ffn_gate", width, hidden),
                                         ("ffn_up", width, hidden)):
            stem = f"blk.{layer}.{module}.weight."
            expected[stem + "lora_a"] = (inputs, 8)
            expected[stem + "lora_b"] = (8, outputs)
    if set(names) != set(expected):
        raise ValueError(f"audio adapter must contain exactly the matching {description} FFN pairs")
    for tensor in reader.tensors:
        if (tuple(map(int, tensor.shape)) != expected[tensor.name]
                or tensor.tensor_type != gguf.GGMLQuantizationType.F32
                or not np.isfinite(tensor.data).all()):
            raise ValueError(f"audio adapter tensor shape, precision or values invalid: {tensor.name}")


def inspect_audio_adapter(path: Path, *, base_model: Path, base_digest: str,
                          projector_digest: str) -> dict[str, Any]:
    """Reject a transplanted, damaged, unsupported or unbound required component."""
    gguf, _ = _require_gguf()
    size = path.stat().st_size
    if not 0 < size <= AUDIO_ADAPTER_MAX_BYTES:
        raise ValueError("audio adapter size exceeds the component limit")
    reader = gguf.GGUFReader(str(path), mode="r")
    metadata = _reader_metadata(reader)
    abi = metadata.get(_PREFIX + "abi")
    if abi == AUDIO_ADAPTER_ABI:
        scope = "last8-ffn"
        if metadata.get(_PREFIX + "scope", scope) != scope:
            raise ValueError("audio adapter ABI and scope do not match")
    elif abi == AUDIO_ADAPTER_FULL_FFN_ABI:
        scope = "full-ffn"
        if metadata.get(_PREFIX + "scope") != scope:
            raise ValueError("audio adapter ABI requires explicit full-ffn scope")
    else:
        raise ValueError("unsupported audio adapter ABI")
    if size > _SCOPE_MAX_BYTES[scope]:
        raise ValueError("audio adapter size exceeds its ABI limit")
    if (not _digest(base_digest) or not _digest(projector_digest)
            or metadata.get(_PREFIX + "base_digest") != base_digest
            or metadata.get(_PREFIX + "projector_digest") != projector_digest):
        raise ValueError("audio adapter ABI or exact language/projector binding does not match")
    _validate(reader, gguf.GGUFReader(str(base_model), mode="r"), scope=scope)
    return {"abi": abi, "digest": "sha256:" + sha256(path),
            "size": size, "base_digest": base_digest,
            "projector_digest": projector_digest, "execution_scope": "audio-perception",
            "tensors": len(reader.tensors)}


def pack_audio_adapter(*, source: Path, base_model: Path, projector: Path,
                       output: Path, scope: str = "last8-ffn") -> dict[str, Any]:
    """Bind an unchanged export to its pair, without copying either large model."""
    if not isinstance(scope, str) or scope not in _SCOPE_ABI:
        raise ValueError("unsupported audio adapter scope")
    from qwen_omni_adapters.audio_prefill import (
        BOUNDARIES_KEY,
        PREFILL_ABI,
        PREFILL_KEY,
        PROMPT_KEY,
        TRAINED_AUDIO_PROMPT_SHA256,
    )

    gguf, np = _require_gguf()
    source, base_model, projector, output = (
        p.expanduser().resolve() for p in (source, base_model, projector, output)
    )
    if output.exists():
        raise ValueError("audio adapter output already exists")
    if not 0 < source.stat().st_size <= _SCOPE_MAX_BYTES[scope]:
        raise ValueError("audio adapter source size exceeds its scope limit")
    reader = gguf.GGUFReader(str(source), mode="r")
    _validate(reader, gguf.GGUFReader(str(base_model), mode="r"), scope=scope)
    projection = _reader_metadata(gguf.GGUFReader(str(projector), mode="r"))
    if (projection.get(PREFILL_KEY) != PREFILL_ABI
            or projection.get(PROMPT_KEY) != TRAINED_AUDIO_PROMPT_SHA256
            or projection.get(BOUNDARIES_KEY) is not False):
        raise ValueError("audio adapter requires the trained-prefill.v2 projector contract")
    base_digest, projector_digest = "sha256:" + sha256(base_model), "sha256:" + sha256(projector)
    metadata = _reader_metadata(reader)
    if any(key.startswith(_PREFIX) for key in metadata):
        raise ValueError("source is already bound; use the original verified LoRA export")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps failed builds from deleting a previous partial.
    partial = output.with_name(output.name + f".{os.getpid()}.partial")
    with partial.open("xb"):
        pass
    try:
        writer = gguf.GGUFWriter(str(partial), arch=metadata["general.architecture"])
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 sub_type=field.types[1] if len(field.types) > 1 else None)
        for key, value in (("abi", _SCOPE_ABI[scope]), ("base_digest", base_digest),
                           ("projector_digest", projector_digest),
                           ("source_sha256", sha256(source))):
            writer.add_string(_PREFIX + key, value)
        if scope == "full-ffn":
            writer.add_string(_PREFIX + "scope", scope)
        for tensor in reader.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        check = gguf.GGUFReader(str(partial), mode="r")
        if len(check.tensors) != len(reader.tensors) or any(
            a.name != b.name or a.tensor_type != b.tensor_type
            or not np.array_equal(a.data, b.data)
            for a, b in zip(reader.tensors, check.tensors, strict=True)
        ):
            raise ValueError("audio adapter binding changed tensor content")
        report = inspect_audio_adapter(partial, base_model=base_model,
                                      base_digest=base_digest, projector_digest=projector_digest)
        del check
        # Link publication never overwrites an output another writer installed.
        os.link(partial, output)
        return {**report, "path": str(output), "all_tensor_values_unchanged": True,
                "source_sha256": sha256(source), "release_qualified": False}
    finally:
        partial.unlink(missing_ok=True)
