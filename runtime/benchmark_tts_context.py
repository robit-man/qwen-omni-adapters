"""Qualify bounded TTS context against 32K using the exact bundled checkpoint.

Use a scoped broker run on discrete CUDA, or an explicit memory floor on Tegra.
This runs private native workers through the production command/profile path.
It does not start an inference service or reload the primary reasoning trunk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import accelerator_profile, is_tegra
from runtime.benchmark_tts_conditioning import (
    PROMPTS,
    ensure_with_matching_libraries,
    memory_guard,
    wait_for_broker_ready,
)
from runtime.tts_server import Config, PersistentTTSWorker, _context_size, _synthesis_spec
from runtime.verify_pcm_stream import analyze_pcm

LANGUAGE_CASES = (
    ("en", PROMPTS[0]), ("en", PROMPTS[1]), ("en", PROMPTS[0]),
    ("zh", "蓝色的自行车停在花园门旁边。"),
    ("ja", "青い自転車が庭の門のそばにあります。"),
    ("ko", "파란 자전거가 정원 문 옆에 있습니다."),
    ("de", "Das blaue Fahrrad steht neben dem Gartentor."),
    ("fr", "Le vélo bleu est garé près de la porte du jardin."),
    ("ru", "Синий велосипед стоит у садовых ворот."),
    ("pt", "A bicicleta azul está ao lado do portão do jardim."),
    ("es", "La bicicleta azul está junto a la puerta del jardín."),
    ("it", "La bicicletta blu è accanto al cancello del giardino."),
)


def discrete_resident_mib(pid: int, uuid: str) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=15)
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if fields[0] == str(pid):
            assert fields[1] == uuid, "worker escaped its reserved UUID"
            return int(fields[2])
    raise RuntimeError("resident worker missing from discrete memory accounting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "projector", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--minimum-available-mib", type=int, default=0)
    args = parser.parse_args()
    tegra = is_tegra()
    uuid, token = os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if tegra:
        if uuid or token or args.minimum_available_mib <= 0:
            parser.error("Tegra requires positive --minimum-available-mib and no discrete scope")
    elif not uuid.startswith("GPU-") or "," in uuid or not token or args.ready_file is None:
        parser.error("use one scoped GPU UUID, a broker token and fresh --ready-file")
    if args.ready_file is not None:
        if args.ready_file.exists():
            parser.error("readiness file already exists; choose a fresh directory")
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    extras = [path.resolve(strict=True) for path in args.extra_library_dir]
    report = {"schema": "robit.omni.tts-context-qualification.v1", "complete": False,
              "accelerator": accelerator_profile(), "samples": [],
              "memory_measurement": "per-worker resident samples after allocation/generation; Tegra system headroom at 100ms",
              "scope": "short speech in all ten supported languages and bounded-generation long-input capacity; not perceptual scoring or long-utterance fidelity"}
    expected, admitted = {}, tegra
    try:
        for adaptive in (False, True):
            for reference in args.reference:
                config = Config(binary=args.binary.resolve(strict=True), model=args.model.resolve(strict=True),
                                projector=args.projector.resolve(strict=True), adaptive_context=adaptive,
                                require_gpu=True, gpu_uuid=uuid, timeout_s=180, max_frames=96)
                worker = PersistentTTSWorker(config)
                sample = {"adaptive": adaptive, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                          "runs": [], "cpu_peak_rss_kib": 0, "gpu_resident_samples_mib": []}
                report["samples"].append(sample)
                cases = [*LANGUAGE_CASES,
                         ("en", ("This sentence verifies the complete input capacity. " * 90)[:4096]),
                         ("zh", ("这句话用于验证完整输入的上下文容量。" * 240)[:4096])]
                try:
                    with memory_guard(worker, args.minimum_available_mib) as headroom:
                        sample["system_headroom"] = headroom
                        for index, (language, text) in enumerate(cases):
                            spec = _synthesis_spec(config, {"text": text, "language": language,
                                                          "speaker_file": str(reference.resolve(strict=True)),
                                                          "max_frames": 16 if index >= len(LANGUAGE_CASES) else 96,
                                                          "stream_frames": 2, "seed": 42})
                            libraries = ensure_with_matching_libraries(worker, spec, extras, 32768)
                            sample["native_libraries"] = libraries
                            if not admitted:
                                args.ready_file.touch()
                                wait_for_broker_ready(token)
                                admitted = True
                            started, first, chunks = time.monotonic(), None, []
                            for chunk in worker.stream(spec):
                                if first is None:
                                    first = time.monotonic() - started
                                chunks.append(chunk)
                            elapsed = time.monotonic() - started
                            pcm = b"".join(chunks)
                            continuity = analyze_pcm(pcm, stream_frames=2)
                            assert pcm
                            key = (str(reference), index)
                            if not adaptive:
                                expected[key] = pcm
                            else:
                                assert pcm == expected[key], f"PCM changed for language={language} case={index}"
                            if not tegra:
                                sample["gpu_resident_samples_mib"].append(discrete_resident_mib(worker.pid, uuid))
                            status = Path(f"/proc/{worker.pid}/status").read_text().splitlines()
                            peak = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
                            sample["cpu_peak_rss_kib"] = max(sample["cpu_peak_rss_kib"], peak)
                            sample["runs"].append({"language": language, "case": index, "utf8_bytes": len(text.encode()),
                                                   "context_size": _context_size(config, spec) or 32768,
                                                   "generation_frames_limit": spec.frames, "first_pcm_s": first,
                                                   "total_s": elapsed, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                                                   "pcm_bytes": len(pcm), "continuity": continuity})
                finally:
                    worker.close()
        report["all_baseline_continuity_checks_passed"] = all(
            run["continuity"]["ok"] for sample in report["samples"] if not sample["adaptive"]
            for run in sample["runs"])
        report["release_qualified"] = False
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "pcm_comparisons": len(expected)}, indent=2))


if __name__ == "__main__":
    main()
