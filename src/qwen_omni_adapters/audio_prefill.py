"""Versioned framing for the frozen-encoder audio projection checkpoints."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from qwen_omni_adapters.context import context_text

TRAINED_AUDIO_SYSTEM_PROMPT = context_text("prompts", "transcribe_system")
TRAINED_AUDIO_SUFFIX_PROMPT = context_text("directives", "media_extract_audio")
TRAINED_AUDIO_EXTRACTION_PROMPT = "\n" + TRAINED_AUDIO_SUFFIX_PROMPT
TRAINED_AUDIO_TEMPLATE_SUFFIX = (
    TRAINED_AUDIO_EXTRACTION_PROMPT
    + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
TRAINED_AUDIO_PROMPT_SHA256 = hashlib.sha256(
    (TRAINED_AUDIO_SYSTEM_PROMPT + "\0" + TRAINED_AUDIO_TEMPLATE_SUFFIX).encode()
).hexdigest()
PREFILL_ABI = "robit.qwen3a.trained-prefill.v2"
PREFILL_KEY = "robit.audio_bridge.prefill_abi"
PROMPT_KEY = "robit.audio_bridge.prompt_sha256"
BOUNDARIES_KEY = "clip.audio.add_boundary_tokens"


def native_prefill_version(binary: Path) -> int:
    """Query the matching library without creating a model or CUDA context."""
    directory = binary.resolve().parent
    for name in ("libmtmd.so", "libmtmd.dylib", "mtmd.dll", "libmtmd.dll"):
        library = directory / name
        if not library.is_file():
            continue
        try:
            handle = ctypes.CDLL(str(library))
            function = handle.mtmd_audio_prefill_abi_version
            function.argtypes = []
            function.restype = ctypes.c_uint32
            return int(function())
        except (OSError, AttributeError):
            continue
    return 0


def verify_audio_prefill_runtime(projector: Path, binary: Path) -> dict[str, Any]:
    from qwen_omni_adapters.single_gguf import _reader_metadata, _require_gguf

    gguf, _ = _require_gguf()
    metadata = _reader_metadata(gguf.GGUFReader(str(projector), mode="r"))
    abi = metadata.get(PREFILL_KEY)
    if abi is None and metadata.get(BOUNDARIES_KEY, True) is True:
        return {"prefill_abi": None, "native_boundary_policy": "default"}
    if (abi != PREFILL_ABI or metadata.get(BOUNDARIES_KEY) is not False
            or metadata.get(PROMPT_KEY) != TRAINED_AUDIO_PROMPT_SHA256
            or metadata.get("clip.audio.projector_type") != "qwen3a"):
        raise ValueError("unsupported or mismatched trained audio prefill contract")
    version = native_prefill_version(binary)
    if version < 2:
        raise ValueError(
            "this audio projector requires native trained-prefill ABI 2; "
            "rebuild with scripts/build_llama_cpp.sh (Windows: build_llama_cpp.ps1)"
        )
    return {"prefill_abi": abi, "prompt_sha256": TRAINED_AUDIO_PROMPT_SHA256,
            "native_prefill_version": version, "native_boundary_policy": "none"}


def prepare_audio_prefill_projector(source: Path, output: Path) -> dict[str, Any]:
    """Add explicit framing metadata, preserving every neural tensor byte."""
    from qwen_omni_adapters.single_gguf import (
        _add_reader_tensor,
        _reader_metadata,
        _require_gguf,
        _sha256,
    )

    gguf, np = _require_gguf()
    source, output = source.resolve(strict=True), output.resolve()
    if source == output or output.exists():
        raise ValueError("use a fresh output; never overwrite a source or installed blob")
    original_stat = source.stat()
    reader = gguf.GGUFReader(str(source), mode="r")
    metadata = _reader_metadata(reader)
    names = {tensor.name for tensor in reader.tensors}
    if (metadata.get("general.architecture") != "clip"
            or metadata.get("robit.audio_bridge.schema") != "robit.qwen3a-audio-bridge-projector.v1"
            or metadata.get("clip.audio.projector_type") != "qwen3a"
            or not {"mm.a.mlp.2.weight", "mm.a.mlp.2.bias"} <= names):
        raise ValueError("source must be a trained Qwen3A final-projection GGUF")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    partial = Path(filename)
    writer = gguf.GGUFWriter(str(partial), arch="clip")
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture",
            PREFILL_KEY, PROMPT_KEY, BOUNDARIES_KEY}
    try:
        for key, field in reader.fields.items():
            if key not in skip:
                value = field.contents()
                if not isinstance(value, list) or value:
                    writer.add_key_value(key, value, field.types[0])
        writer.add_key_value(PREFILL_KEY, PREFILL_ABI, gguf.GGUFValueType.STRING)
        writer.add_key_value(PROMPT_KEY, TRAINED_AUDIO_PROMPT_SHA256, gguf.GGUFValueType.STRING)
        writer.add_key_value(BOUNDARIES_KEY, False, gguf.GGUFValueType.BOOL)
        for tensor in reader.tensors:
            _add_reader_tensor(writer, tensor, tensor.name, np)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        check = gguf.GGUFReader(str(partial), mode="r")
        if len(check.tensors) != len(reader.tensors):
            raise ValueError("audio prefill export changed tensor count")
        for before, after in zip(reader.tensors, check.tensors, strict=True):
            if (before.name != after.name or before.tensor_type != after.tensor_type
                    or not np.array_equal(before.shape, after.shape)
                    or not np.array_equal(before.data.view(np.uint8), after.data.view(np.uint8))):
                raise ValueError(f"audio prefill export changed tensor: {before.name}")
        current = source.stat()
        def identity(st: os.stat_result) -> tuple[int, ...]:
            return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

        if identity(original_stat) != identity(current):
            raise ValueError("source changed during audio prefill export")
        if output.exists():
            raise ValueError("output appeared during audio prefill export")
        check.data._mmap.close()
        # Atomic no-clobber publication, even if another exporter wins the name.
        os.link(partial, output)
    finally:
        writer.close()
        partial.unlink(missing_ok=True)
    return {"source_sha256": _sha256(source), "sha256": _sha256(output),
            "size_bytes": output.stat().st_size, "tensor_count": len(reader.tensors),
            "all_tensors_byte_identical": True, "prefill_abi": PREFILL_ABI,
            "prompt_sha256": TRAINED_AUDIO_PROMPT_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--projector", type=Path, required=True)
    verify.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    report = (prepare_audio_prefill_projector(args.source, args.output) if args.action == "prepare"
              else verify_audio_prefill_runtime(args.projector, args.binary))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
