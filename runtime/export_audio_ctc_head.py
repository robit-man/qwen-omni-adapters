"""Export the qualified English CTC pilot to a bounded native GGUF component.

This build-only tool requires safetensors and Torch. It verifies the joint
head/decoder qualification before exporting; it does not enable a runtime route
or qualify streaming, multilingual use, or either complete Omni release.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("head", "head-config", "tokenizer", "decoder-report", "lexicon", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.decoder_report.read_text())
    config = json.loads(args.head_config.read_text())
    if not (
        report.get("validation_gate")
        and report.get("test_gate")
        and report.get("lexicon_training_only")
        and report.get("external_language_model") is False
    ):
        parser.error("a qualified head and training-only decoder are required")
    if report["head_sha256"] != digest(args.head) or report["lexicon_sha256"] != digest(
        args.lexicon
    ):
        parser.error("head or decoder identities differ from the qualified artifacts")
    if config["tokenizer_sha256"] != digest(args.tokenizer):
        parser.error("tokenizer differs from the trained head")
    if (
        config["feature_width"],
        config["hidden_width"],
        config["vocabulary_size"],
        config["kernel_size"],
        config["norm_epsilon"],
        config["activation"],
        config["languages"],
    ) != (1280, 512, 2048, 3, 1e-5, "gelu_erf", ["en"]):
        parser.error("unsupported native head layout")
    if args.out.exists():
        parser.error("output already exists")
    import numpy as np
    from gguf import GGUFWriter
    from safetensors.torch import load_file

    weights = load_file(str(args.head))
    expected = {
        "norm.weight": (1280,),
        "norm.bias": (1280,),
        "input.weight": (512, 1280),
        "input.bias": (512,),
        "temporal.weight": (512, 512, 3),
        "temporal.bias": (512,),
        "output.weight": (2048, 512),
        "output.bias": (2048,),
    }
    if set(weights) != set(expected) or any(
        tuple(weights[k].shape) != shape for k, shape in expected.items()
    ):
        parser.error("unexpected head tensor inventory")
    writer = GGUFWriter(str(args.out), "robit_audio_ctc")
    writer.add_string("general.name", "Qwen Omni shared-encoder English CTC pilot")
    writer.add_string("robit.audio_ctc.schema", "robit.omni.audio-ctc-head.v1")
    writer.add_string("robit.audio_ctc.config_json", json.dumps(config, sort_keys=True))
    writer.add_string("robit.audio_ctc.head_sha256", digest(args.head))
    writer.add_string("robit.audio_ctc.encoder_sha256", config["encoder_tensor_sha256"])
    writer.add_string("robit.audio_ctc.tokenizer_json", args.tokenizer.read_text())
    writer.add_string("robit.audio_ctc.lexicon_json", args.lexicon.read_text())
    writer.add_string("robit.audio_ctc.qualification_sha256", digest(args.decoder_report))
    writer.add_string(
        "robit.audio_ctc.scope",
        "English full-clip pilot; no runtime activation or streaming qualification",
    )
    for name in expected:
        value = weights[name].float().contiguous().numpy()

        if not np.isfinite(value).all():
            parser.error("nonfinite head weights")
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": digest(args.out),
                "bytes": args.out.stat().st_size,
                "parameters": sum(v.numel() for v in weights.values()),
                "runtime_enabled": False,
            }
        )
    )


if __name__ == "__main__":
    main()
