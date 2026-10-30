"""Export the frozen Qwen3-TTS reference acoustic encoder for indexed packaging.

Build tool only. Requires local official safetensors, Torch and GGUF tooling;
runtime inference neither imports Torch nor downloads this checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

SOURCE_REPOSITORY = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SOURCE_REVISION = "7dd38ad4e9bad454aae9cd937d0cd577604fe229"
SOURCE_SHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
CONFIG_SHA256 = "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167"
REFERENCE_ENCODER_ABI = "robit.qwen3tts.reference-encoder.v1"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def export_reference_encoder(source: Path, output: Path, *, overwrite: bool = False) -> dict:
    source, output = source.expanduser().resolve(strict=True), output.expanduser().resolve()
    before = source.stat()
    if source == output or (output.exists() and not overwrite):
        raise ValueError("output exists or would replace the input checkpoint")
    if digest(source) != SOURCE_SHA256:
        raise ValueError("source is not the pinned Qwen3-TTS tokenizer checkpoint; no weights exported")
    import torch
    from gguf import GGUFWriter
    from safetensors import safe_open

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    partial = Path(name)
    writer = GGUFWriter(str(partial), "clip")
    count = 0
    try:
        writer.add_name("Qwen3-TTS-12Hz frozen reference encoder")
        writer.add_string("general.license", "apache-2.0")
        writer.add_clip_has_audio_encoder(True)
        writer.add_clip_has_vision_encoder(False)
        writer.add_clip_has_gen_audio_encoder(False)
        writer.add_string("clip.audio.projector_type", "qwen3tts_refenc")
        for key, value in {"embedding_length": 512, "feed_forward_length": 2048,
                           "block_count": 8, "projection_dim": 16, "attention.head_count": 8,
                           "num_mel_bins": 1}.items():
            writer.add_uint32("clip.audio." + key, value)
        writer.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)
        for key, value in {"abi": REFERENCE_ENCODER_ABI, "source.repository": SOURCE_REPOSITORY,
                           "source.revision": SOURCE_REVISION, "source.sha256": SOURCE_SHA256,
                           "input": "24kHz-mono-float32-bounded-0.5-30s"}.items():
            writer.add_string("robit.reference_encoder." + key, value)
        with safe_open(source, framework="pt", device="cpu") as checkpoint:
            def put(name, source_tensor):
                nonlocal count
                tensor = checkpoint.get_tensor(source_tensor) if isinstance(source_tensor, str) else source_tensor
                writer.add_tensor(name, tensor.float().contiguous().numpy())
                count += 1
            for suffix in ("weight", "bias"):
                put("a.seanet.conv_in." + suffix, "encoder.encoder.layers.0.conv." + suffix)
                put("a.seanet.conv_out." + suffix, "encoder.encoder.layers.14.conv." + suffix)
                for stage in range(4):
                    put(f"a.seanet.blk.{stage}.res_conv1.{suffix}",
                        f"encoder.encoder.layers.{1 + 3 * stage}.block.1.conv.{suffix}")
                    put(f"a.seanet.blk.{stage}.res_conv2.{suffix}",
                        f"encoder.encoder.layers.{1 + 3 * stage}.block.3.conv.{suffix}")
                    put(f"a.seanet.blk.{stage}.scale_conv.{suffix}",
                        f"encoder.encoder.layers.{3 + 3 * stage}.conv.{suffix}")
            mapping = {
                "input_layernorm.weight": "ln1.weight", "input_layernorm.bias": "ln1.bias",
                "post_attention_layernorm.weight": "ln2.weight", "post_attention_layernorm.bias": "ln2.bias",
                "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_out.weight",
                "mlp.fc1.weight": "ffn_up.weight", "mlp.fc2.weight": "ffn_down.weight",
                "self_attn_layer_scale.scale": "ls1.weight", "mlp_layer_scale.scale": "ls2.weight",
            }
            for layer in range(8):
                for key, target in mapping.items():
                    put(f"a.blk.{layer}.{target}", f"encoder.encoder_transformer.layers.{layer}.{key}")
            put("a.downsample.conv.weight", "encoder.downsample.conv.weight")
            for family, layers in (("semantic", 1), ("acoustic", 15)):
                prefix = f"encoder.quantizer.{family}_residual_vector_quantizer"
                put(f"a.ref.{family}.in.weight", checkpoint.get_tensor(prefix + ".input_proj.weight").squeeze(-1))
                books = []
                for layer in range(layers):
                    base = f"{prefix}.layers.{layer}.codebook."
                    books.append(checkpoint.get_tensor(base + "embed_sum") /
                                 checkpoint.get_tensor(base + "cluster_usage").clamp_min(1e-5).unsqueeze(-1))
                put(f"a.ref.{family}.codebook.weight", torch.stack(books) if layers > 1 else books[0])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        after = source.stat()
        if any(getattr(before, key) != getattr(after, key) for key in
               ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("source checkpoint changed during export")
        os.replace(partial, output)
    finally:
        writer.close()
        partial.unlink(missing_ok=True)
    return {"abi": REFERENCE_ENCODER_ABI, "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA256,
            "output": str(output), "sha256": digest(output), "size_bytes": output.stat().st_size,
            "tensor_count": count, "required_by_current_synthesis_path": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-safetensors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_reference_encoder(args.source_safetensors, args.out,
                                             overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
