"""Build frozen CPU reference fixtures, or compare a native GPU encoder with them.

Fixture building is an offline numerical oracle, not a runtime CPU fallback.
Comparison requires scoped CUDA admission or guarded Tegra residency. It never
starts or reloads a reasoning trunk and does not qualify full voice cloning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import select
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import accelerator_profile, is_tegra, process_is_gpu_resident
from runtime.export_tts_reference_encoder import CONFIG_SHA256, SOURCE_SHA256, digest


def build_fixtures(args):
    if digest(args.source_weights) != SOURCE_SHA256 or digest(args.source_config) != CONFIG_SHA256:
        raise ValueError("fixtures require the pinned Qwen3-TTS tokenizer weights and configuration")
    if args.directory.exists():
        raise ValueError("fixture output must be a new directory")
    import soxr
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import MimiConfig, MimiModel

    torch.set_num_threads(8)
    config = MimiConfig(**json.loads(args.source_config.read_text())["encoder_config"])
    config._attn_implementation = "eager"
    with torch.device("cpu"):
        model = MimiModel(config)
    model.upsample = model.decoder_transformer = model.decoder = None
    with safe_open(args.source_weights, framework="pt", device="cpu") as checkpoint:
        state = {key.removeprefix("encoder."): checkpoint.get_tensor(key)
                 for key in checkpoint.keys() if key.startswith("encoder.")}  # noqa: SIM118 - safetensors handle
    model.load_state_dict(state, strict=True)
    model.eval()
    args.directory.mkdir(parents=True, mode=0o700)
    manifest = {"schema": "robit.omni.reference-encoder-fixtures.v1", "source_sha256": SOURCE_SHA256,
                "config_sha256": CONFIG_SHA256, "transformers": transformers.__version__,
                "normalization": "24kHz-soxr-HQ-PCM16-roundtrip-v1", "fixtures": []}
    for index, reference in enumerate(args.reference):
        with wave.open(str(reference), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
            if (handle.getnchannels(), handle.getsampwidth()) != (1, 2) or not rate / 2 <= frames <= rate * 30:
                raise ValueError("reference requires 0.5-30 seconds of mono PCM16 WAV")
            audio = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(np.float32) / 32768
        if rate != 24000:
            audio = soxr.resample(audio, rate, 24000, quality="HQ")
        audio = (np.round(np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2")
                 .astype("<f4") / 32768)
        input_file, expected_file = f"reference-{index}.f32", f"reference-{index}.npy"
        audio.tofile(args.directory / input_file)
        with torch.inference_mode():
            result = model.encode(torch.from_numpy(audio).reshape(1, 1, -1), num_quantizers=16, return_dict=True)
        codes = result.audio_codes[0].transpose(0, 1).contiguous().numpy()
        np.save(args.directory / expected_file, codes)
        manifest["fixtures"].append({"input": input_file, "expected": expected_file,
                                     "input_sha256": digest(args.directory / input_file),
                                     "expected_sha256": digest(args.directory / expected_file),
                                     "reference_sha256": digest(reference), "samples": len(audio),
                                     "codes_shape": list(codes.shape)})
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"fixtures": len(manifest["fixtures"]), "manifest": str(args.directory / "manifest.json")}


def compare(args):
    from runtime.benchmark_tts_conditioning import memory_guard, wait_for_broker_ready

    if platform.system() != "Linux":
        raise ValueError("this qualification currently supports Linux CUDA and Tegra")
    tegra = is_tegra()
    uuid, token = os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if tegra:
        if uuid or token or args.minimum_available_mib <= 0:
            raise ValueError("Tegra requires positive memory floor and no discrete GPU scope")
    elif not uuid.startswith("GPU-") or "," in uuid or not token or args.ready_file is None:
        raise ValueError("use one exact scoped GPU UUID and fresh --ready-file")
    if args.ready_file is not None:
        if args.ready_file.exists():
            raise ValueError("readiness file already exists")
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    extras = [path.resolve(strict=True) for path in args.extra_library_dir]
    manifest = json.loads(args.fixtures.read_text())
    fixtures = manifest.get("fixtures", [])
    if (manifest.get("schema") != "robit.omni.reference-encoder-fixtures.v1"
            or manifest.get("source_sha256") != SOURCE_SHA256 or not 1 <= len(fixtures) <= 32):
        raise ValueError("unsupported or empty reference fixtures")
    report = {"schema": "robit.omni.reference-encoder-qualification.v1", "complete": False,
              "native_encoder_qualified": False, "full_voice_cloning_qualified": False,
              "accelerator": accelerator_profile(), "model_sha256": digest(model),
              "binary_sha256": digest(binary), "fixtures_sha256": digest(args.fixtures), "samples": []}
    worker, admitted = SimpleNamespace(process=None), tegra
    try:
        with (tempfile.TemporaryDirectory(prefix="reference-encoder-") as temporary,
              memory_guard(worker, args.minimum_available_mib) as headroom):
            report["system_headroom"] = headroom
            for index, fixture in enumerate(fixtures):
                data, expected = (args.fixtures.parent / fixture[key] for key in ("input", "expected"))
                if digest(data) != fixture["input_sha256"] or digest(expected) != fixture["expected_sha256"]:
                    raise ValueError("reference fixture changed")
                samples = fixture["samples"]
                if not 12000 <= samples <= 720000 or data.stat().st_size != samples * 4:
                    raise ValueError("reference fixture exceeds duration bounds")
                oracle = np.load(expected, allow_pickle=False)
                if oracle.shape != (math.ceil(samples / 1920), 16) or oracle.dtype.kind not in "iu":
                    raise ValueError("invalid oracle code shape")
                output = Path(temporary) / f"{index}.i32"
                with tempfile.TemporaryFile(mode="w+b") as log:
                    process = subprocess.Popen([str(binary), str(model), str(data.resolve()), str(output)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                        env={**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(map(str, [binary.parent, *extras]))})
                    worker.process = process
                    try:
                        if not select.select([process.stdout], [], [], 60)[0] or process.stdout.readline() != b"R\n":
                            size = os.fstat(log.fileno()).st_size
                            detail = os.pread(log.fileno(), min(size, 1500), max(0, size - 1500))
                            raise RuntimeError("reference encoder did not become ready: " + detail.decode(errors="replace"))
                        if not process_is_gpu_resident(process.pid, uuid):
                            raise RuntimeError("reference encoder has no proven GPU residency")
                        if not admitted:
                            args.ready_file.touch()
                            wait_for_broker_ready(token)
                            admitted = True
                        started = time.monotonic()
                        process.communicate(b"GO\n", timeout=120)
                        if process.returncode:
                            log.seek(0)
                            raise RuntimeError("reference encoder failed: " + log.read().decode(errors="replace")[-1500:])
                        actual = np.fromfile(output, dtype="<i4").reshape(-1, 16)
                        equal = oracle.shape == actual.shape and np.array_equal(oracle, actual)
                        report["samples"].append({"reference_sha256": fixture["reference_sha256"],
                            "frames": actual.shape[0], "codes": actual.size, "codes_identical": equal,
                            "encode_and_exit_s": time.monotonic() - started})
                        if not equal:
                            raise RuntimeError("native reference codes diverged from the frozen oracle")
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        worker.process = None
        report["complete"] = report["native_encoder_qualified"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("fixtures")
    build.add_argument("--source-weights", type=Path, required=True)
    build.add_argument("--source-config", type=Path, required=True)
    build.add_argument("--reference", type=Path, action="append", required=True)
    build.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser("compare")
    for name in ("binary", "model", "fixtures", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    run.add_argument("--ready-file", type=Path)
    run.add_argument("--minimum-available-mib", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_fixtures(args) if args.command == "fixtures" else compare(args), indent=2))


if __name__ == "__main__":
    main()
