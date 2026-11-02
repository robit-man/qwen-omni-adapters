"""An exact-pair-bound FFN LoRA; audio perception is its only execution scope."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from qwen_omni_adapters.single_gguf import _reader_metadata, _require_gguf

AUDIO_ADAPTER_MEDIA_TYPE = "application/vnd.robit.ollama.omni.audio-adapter.v1+gguf"
AUDIO_ADAPTER_ABI = "robit.omni.conditional-audio-lora.v1"
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


def _validate(reader: Any, model: Any) -> None:
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
    expected = {}
    for layer in range(blocks - 8, blocks):
        for module, inputs, outputs in (("ffn_down", hidden, width),
                                         ("ffn_gate", width, hidden),
                                         ("ffn_up", width, hidden)):
            stem = f"blk.{layer}.{module}.weight."
            expected[stem + "lora_a"] = (inputs, 8)
            expected[stem + "lora_b"] = (8, outputs)
    names = [tensor.name for tensor in reader.tensors]
    if len(names) != 48 or len(set(names)) != 48 or set(names) != set(expected):
        raise ValueError("audio adapter must contain exactly the matching last8 FFN pairs")
    for tensor in reader.tensors:
        if (tuple(map(int, tensor.shape)) != expected[tensor.name]
                or tensor.tensor_type != gguf.GGMLQuantizationType.F32
                or not np.isfinite(tensor.data).all()):
            raise ValueError(f"audio adapter tensor shape, precision or values invalid: {tensor.name}")


def inspect_audio_adapter(path: Path, *, base_model: Path, base_digest: str,
                          projector_digest: str) -> dict[str, Any]:
    """Reject a transplanted, damaged, unsupported or unbound required component."""
    gguf, _ = _require_gguf()
    reader = gguf.GGUFReader(str(path), mode="r")
    metadata = _reader_metadata(reader)
    if (not _digest(base_digest) or not _digest(projector_digest)
            or metadata.get(_PREFIX + "abi") != AUDIO_ADAPTER_ABI
            or metadata.get(_PREFIX + "base_digest") != base_digest
            or metadata.get(_PREFIX + "projector_digest") != projector_digest):
        raise ValueError("audio adapter ABI or exact language/projector binding does not match")
    _validate(reader, gguf.GGUFReader(str(base_model), mode="r"))
    return {"abi": AUDIO_ADAPTER_ABI, "digest": "sha256:" + sha256(path),
            "size": path.stat().st_size, "base_digest": base_digest,
            "projector_digest": projector_digest, "execution_scope": "audio-perception",
            "tensors": 48}


def pack_audio_adapter(*, source: Path, base_model: Path, projector: Path,
                       output: Path) -> dict[str, Any]:
    """Bind an unchanged export to its pair, without copying either large model."""
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
    reader = gguf.GGUFReader(str(source), mode="r")
    _validate(reader, gguf.GGUFReader(str(base_model), mode="r"))
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
        for key, value in (("abi", AUDIO_ADAPTER_ABI), ("base_digest", base_digest),
                           ("projector_digest", projector_digest),
                           ("source_sha256", sha256(source))):
            writer.add_string(_PREFIX + key, value)
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
