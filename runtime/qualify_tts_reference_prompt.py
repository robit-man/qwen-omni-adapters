"""Qualify a native full-reference prompt without changing a deployed service.

The JSON contains reference-only codes from the bundled frozen encoder, plus
the corresponding transcript. This bounded Linux/Tegra qualification exchanges
the real persistent protocol. It does not add a public speech field or claim
perceptual cloning quality from deterministic PCM checks.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import accelerator_profile, is_tegra, process_is_gpu_resident
from runtime.benchmark_tts_conditioning import (
    PROMPTS,
    counter_log,
    memory_guard,
    wait_for_broker_ready,
)
from runtime.tts_server import Config, PersistentTTSWorker, _command, _pcm_wav, _synthesis_spec
from runtime.verify_pcm_stream import analyze_pcm


def qualify(args: argparse.Namespace) -> dict:
    tegra = is_tegra()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if os.name != "posix" or (tegra and (uuid or token or args.minimum_available_mib <= 0)):
        raise ValueError("Linux qualification; Tegra requires a positive memory floor and no discrete scope")
    if not tegra and (not uuid.startswith("GPU-") or "," in uuid or not token):
        raise ValueError("run through docker gpu run with one exact UUID and a VRAM budget")
    if args.ready_file.exists():
        raise ValueError("use a fresh readiness path")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    libraries = [args.binary.resolve().parent, *(p.resolve(strict=True) for p in args.extra_library_dir)]
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(map(str, libraries))}
    report = {"schema": "robit.omni.tts-reference-prompt-qualification.v1",
              "accelerator": accelerator_profile(), "gpu_uuid": uuid,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "reference_prompt_sha256": hashlib.sha256(args.reference_prompt.read_bytes()).hexdigest(),
              "context_size": args.context_size, "cases": [],
              "public_voice_contract_qualified": False, "perceptual_cloning_quality_qualified": False}
    admitted = False
    for enabled in (False, True):
        config = Config(binary=args.binary.resolve(), model=args.model.resolve(),
                        projector=args.projector.resolve(), timeout_s=120, require_gpu=True,
                        gpu_uuid=uuid, conditioning_cache=enabled)
        spec = _synthesis_spec(config, {"text": PROMPTS[0], "speaker_file": str(args.reference.resolve()),
                                        "seed": 42, "max_frames": 96, "stream_frames": 2})
        worker = PersistentTTSWorker(config)
        runs, outputs = [], []
        try:
            with memory_guard(worker, args.minimum_available_mib) as headroom:
                if headroom["minimum_available_mib"] < args.minimum_available_mib:
                    raise RuntimeError("insufficient headroom before loading")
                command = _command(config, spec, Path(os.devnull), stream=True, persistent=True)
                command.extend(["--ctx-size", str(args.context_size), "--tts-reference-prompt-file",
                                str(args.reference_prompt.resolve())])
                worker.stderr_log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")  # noqa: SIM115
                loaded = time.monotonic()
                worker.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=worker.stderr_log, start_new_session=True, env=env)
                process = worker.process
                try:
                    ready = worker._read_frame(process, time.monotonic() + 120)
                    if ready != ("R", b"") or not process_is_gpu_resident(process.pid, uuid):
                        raise RuntimeError("native readiness or exact GPU residency failed")
                except Exception:
                    print(counter_log(worker)[-1500:], file=sys.stderr)
                    raise
                mapped = {}
                for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
                    fields = line.split(maxsplit=5)
                    if len(fields) != 6 or not fields[5].startswith("/"):
                        continue
                    path = Path(fields[5])
                    if path.name.startswith(("libllama.so", "libllama-common.so", "libmtmd.so")):
                        if path.parent.resolve() != libraries[0]:
                            raise RuntimeError("native comparison loaded mismatched model libraries")
                        mapped[path.name] = str(path)
                    elif path.name.startswith("libggml"):
                        if path.parent.resolve() not in libraries:
                            raise RuntimeError("native comparison loaded unrecorded GGML libraries")
                        mapped[path.name] = str(path)
                if not any(name.startswith("libmtmd.so") for name in mapped):
                    raise RuntimeError("native generation library mapping was not proven")
                load_s = time.monotonic() - loaded
                if not admitted:
                    args.ready_file.touch()
                    if not tegra:
                        wait_for_broker_ready(token)
                    admitted = True
                for index, text in enumerate(PROMPTS):
                    started, first = time.monotonic(), None
                    process.stdin.write(base64.b64encode(text.encode()) + b"\n")
                    process.stdin.flush()
                    pcm = bytearray()
                    while True:
                        kind, data = worker._read_frame(process, started + 120)
                        if kind == "A" and data:
                            if first is None:
                                first = time.monotonic() - started
                            pcm.extend(data)
                        elif kind == "D" and not data:
                            break
                        else:
                            raise RuntimeError(f"unexpected native frame: {kind}")
                    elapsed = time.monotonic() - started
                    raw = bytes(pcm)
                    if not raw or first is None or not process_is_gpu_resident(process.pid, uuid):
                        raise RuntimeError("missing PCM or GPU residency")
                    continuity = analyze_pcm(raw, stream_frames=2)
                    if not continuity["ok"]:
                        raise RuntimeError("raw PCM continuity failed")
                    outputs.append(raw)
                    name = f"reference-{int(enabled)}-{index}"
                    (args.work_dir / f"{name}.wav").write_bytes(_pcm_wav(raw))
                    time.sleep(.02)
                    log = counter_log(worker)
                    counters = re.findall(r"embeddings=(\d+) primes=(\d+) hits=(\d+) cached_bytes=(\d+)", log)
                    runs.append({"index": index, "first_pcm_s": first, "total_s": elapsed,
                                 "pcm_bytes": len(raw), "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                                 "continuity": continuity,
                                 "reference_counters": list(map(int, counters[-1])) if counters else None})
                if outputs[0] != outputs[2]:
                    raise RuntimeError("A/B/A carried target generation state")
                (args.work_dir / f"native-{int(enabled)}.log").write_text(counter_log(worker))
                report["cases"].append({"cache_enabled": enabled, "worker_pid": process.pid,
                                        "gpu_residency_verified": True, "load_s": load_s,
                                        "native_libraries": mapped,
                                        "memory_headroom": headroom, "a_b_a_identical": True, "runs": runs})
        finally:
            worker.close()
    before, after = report["cases"]
    if [r["pcm_sha256"] for r in before["runs"]] != [r["pcm_sha256"] for r in after["runs"]]:
        raise RuntimeError("reference cache changed PCM")
    counters = after["runs"][-1]["reference_counters"]
    if not counters or counters[:3] != [1, 1, 2] or not 0 < counters[3] <= 16 * 1024 * 1024:
        raise RuntimeError("reference conditioning did not reuse one bounded immutable entry")
    warm_before = statistics.median(r["first_pcm_s"] for r in before["runs"][1:])
    warm_after = statistics.median(r["first_pcm_s"] for r in after["runs"][1:])
    report.update(all_pcm_exact=True, reference_conditioning_reused=True,
                  warm_first_pcm_reduction_percent=(1 - warm_after / warm_before) * 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "projector", "reference", "reference-prompt", "work-dir", "ready-file", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--minimum-available-mib", type=int, default=0)
    parser.add_argument("--context-size", type=int, choices=(4096, 8192, 16384, 32768), default=4096)
    args = parser.parse_args()
    if args.minimum_available_mib < 0:
        parser.error("memory floor must be nonnegative")
    result = qualify(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_pcm_exact", "reference_conditioning_reused",
                                          "warm_first_pcm_reduction_percent")}), flush=True)


if __name__ == "__main__":
    main()
