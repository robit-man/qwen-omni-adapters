"""Bounded A/B/A comparison using the same bundled TTS checkpoint and references.

Run through docker gpu run with a scoped budget. The readiness file is written
only after actual residency and the native ready frame, and every worker is
closed before returning. No public service is started or restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.tts_server import Config, PersistentTTSWorker, _pcm_wav, _synthesis_spec
from runtime.verify_pcm_stream import analyze_pcm

PROMPTS = (
    "The blue bicycle is parked beside the garden gate.",
    "Seven small birds are sitting on the wooden fence.",
    "The blue bicycle is parked beside the garden gate.",
)


def wait_for_broker_ready(token: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        result = subprocess.run(["docker", "gpu", "status"], capture_output=True,
                                text=True, check=True, timeout=10)
        leases = json.loads(result.stdout).get("leases", [])
        if any(lease.get("token") == token and lease.get("state") == "active" for lease in leases):
            return
        time.sleep(.1)
    raise RuntimeError("foreground lease did not become broker-ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if not uuid.startswith("GPU-") or "," in uuid or not token:
        parser.error("use one exact UUID through the scoped GPU run wrapper")
    if len(args.reference) > 2:
        parser.error("at most two references per bounded comparison")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    args.ready_file.unlink(missing_ok=True)
    config = Config(binary=args.binary.resolve(), model=args.model.resolve(),
                    projector=args.projector.resolve(), gpu_uuid=uuid, require_gpu=True,
                    timeout_s=120, residency_timeout_s=90, max_frames=96, stream_frames=2)
    report = {"schema": "robit.omni.tts-conditioning-benchmark.v1", "gpu_uuid": uuid,
              "scope": "same native graph, cache disabled/enabled, seeded PCM equality",
              "samples": [], "prompts": list(PROMPTS)}
    admitted = False
    for reference in args.reference:
        reference = reference.resolve()
        for enabled in (False, True):
            worker = PersistentTTSWorker(replace(config, conditioning_cache=enabled))
            pcm_results = []
            samples = []
            try:
                first = _synthesis_spec(config, {"text": PROMPTS[0], "speaker_file": str(reference),
                                                "seed": 42, "stream_frames": 2})
                load_started = time.monotonic()
                worker.ensure(first)
                load_seconds = time.monotonic() - load_started
                if not admitted:
                    args.ready_file.touch()
                    wait_for_broker_ready(token)
                    admitted = True
                pid = worker.pid
                for index, text in enumerate(PROMPTS):
                    spec = replace(first, text=text)
                    started = time.monotonic()
                    first_pcm = None
                    pcm = bytearray()
                    for chunk in worker.stream(spec):
                        if first_pcm is None:
                            first_pcm = time.monotonic() - started
                        pcm.extend(chunk)
                    elapsed = time.monotonic() - started
                    assert worker.pid == pid and pcm and len(pcm) % 2 == 0
                    raw = bytes(pcm)
                    pcm_results.append(raw)
                    filename = f"{reference.stem}-{int(enabled)}-{index}"
                    (work / (filename + ".pcm")).write_bytes(raw)
                    (work / (filename + ".wav")).write_bytes(_pcm_wav(raw))
                    time.sleep(.02)  # allow redacted native counter logs to flush after D
                    counters = re.findall(r"token_tables=(\d+) speaker_encodes=(\d+) speaker_hits=(\d+)",
                                          worker._diagnostic())
                    hwm = next((line.split()[1] for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                                if line.startswith("VmHWM:")), None)
                    sample = {"index": index, "first_pcm_s": first_pcm, "total_s": elapsed,
                              "pcm_bytes": len(raw), "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                              "audio_s": len(raw) / 48000, "worker_pid": pid,
                              "cpu_peak_rss_kib": int(hwm) if hwm else None,
                              "conditioning_counters": list(map(int, counters[-1])) if counters else None,
                              "pcm_continuity": analyze_pcm(raw, stream_frames=2)}
                    samples.append(sample)
                assert pcm_results[0] == pcm_results[2], "A/B/A retained decoded request state"
                report["samples"].append({"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                    "reference_name": reference.name, "cache_enabled": enabled, "load_s": load_seconds,
                    "a_b_a_identical": True, "runs": samples})
                print(json.dumps({"reference": reference.name, "cache": enabled,
                                  "first_pcm_s": [x["first_pcm_s"] for x in samples]}), flush=True)
            finally:
                worker.close()
    for offset in range(0, len(report["samples"]), 2):
        baseline, cached = report["samples"][offset:offset + 2]
        assert [x["pcm_sha256"] for x in baseline["runs"]] == [x["pcm_sha256"] for x in cached["runs"]], \
            "conditioning cache changed synthesized PCM"
        assert cached["runs"][-1]["conditioning_counters"] == [1, 1, 2]
        before = statistics.median(x["first_pcm_s"] for x in baseline["runs"][1:])
        after = statistics.median(x["first_pcm_s"] for x in cached["runs"][1:])
        cached["warm_first_pcm_reduction_percent"] = (1 - after / before) * 100
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
