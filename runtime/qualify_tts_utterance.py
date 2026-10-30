"""Compare full-reference decoder continuation and strict utterance isolation.

Run in a scoped exact-GPU foreground lease, or with a positive Tegra memory
floor. Baseline and candidate use matching private libraries and the same pinned
reference prompt. No deployed service is started or restarted. Clean existing
joins do not establish a sound-quality improvement; this gate proves decoder
state retention, ordinary-request equality and end/abort/error isolation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_omni_adapters.accelerator import is_tegra, process_is_gpu_resident
from runtime import tts_server as tts
from runtime.benchmark_tts_conditioning import (
    PROMPTS,
    ensure_with_matching_libraries,
    memory_guard,
    wait_for_broker_ready,
)
from runtime.verify_pcm_stream import analyze_pcm


def main():
    p = argparse.ArgumentParser()
    for k in (
        "binary",
        "baseline-binary",
        "model",
        "projector",
        "reference",
        "reference-prompt",
        "work",
        "ready",
        "out",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--library", type=Path, action="append", default=[])
    p.add_argument("--minimum-available-mib", "--floor", dest="floor", type=int, default=0)
    a = p.parse_args()
    assert not a.ready.exists()
    a.work.mkdir(parents=True, exist_ok=False)
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    assert (is_tegra() and not uuid and not token and a.floor > 0) or (
        not is_tegra() and uuid.startswith("GPU-") and "," not in uuid and token
    )
    report = {
        "schema": "robit.omni.tts-utterance-prototype.v1",
        "cases": [],
        "perceptual_cloning_quality_qualified": False,
        "passed": False,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "baseline_binary_sha256": hashlib.sha256(a.baseline_binary.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "reference_prompt_sha256": hashlib.sha256(a.reference_prompt.read_bytes()).hexdigest(),
    }
    original = tts._command
    admitted = False
    baseline = None
    for control in (False, True):

        def command(*args, use_controls=control, **kwargs):
            cmd = original(*args, **kwargs) + [
                "--tts-reference-prompt-file",
                str(a.reference_prompt),
            ]
            return cmd + (["--tts-utterance-controls"] if use_controls else [])

        tts._command = command
        cfg = tts.Config(
            binary=(a.binary if control else a.baseline_binary).resolve(),
            model=a.model.resolve(),
            projector=a.projector.resolve(),
            require_gpu=True,
            gpu_uuid=uuid,
            timeout_s=120,
            max_frames=96,
            stream_frames=2,
            adaptive_context=True,
        )
        spec = tts._synthesis_spec(
            cfg, {"text": PROMPTS[0], "speaker_file": str(a.reference.resolve()), "seed": 42}
        )
        worker = tts.PersistentTTSWorker(cfg)
        rows = []
        chunks = {}

        def speak(name, text, worker=worker, spec=spec, rows=rows, chunks=chunks):
            started = time.monotonic()
            raw = b"".join(worker.stream(replace(spec, text=text)))
            elapsed = time.monotonic() - started
            assert process_is_gpu_resident(worker.pid, uuid)
            quality = analyze_pcm(raw, stream_frames=2)
            (a.work / (name + ".wav")).write_bytes(tts._pcm_wav(raw))
            rows.append(
                {
                    "name": name,
                    "text": text,
                    "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                    "pcm_bytes": len(raw),
                    "total_s": elapsed,
                    "continuity": quality,
                }
            )
            chunks[name] = raw
            print(json.dumps(rows[-1]), flush=True)
            return raw

        def ctl(kind, identifier="1" * 32, expected="C", worker=worker):
            body = json.dumps(
                {"schema": "robit.qwen3tts.utterance-control.v1", "kind": kind, "id": identifier},
                separators=(",", ":"),
            ).encode()
            worker.process.stdin.write(body + b"\n")
            worker.process.stdin.flush()
            response = worker._read_frame(worker.process, time.monotonic() + 120)
            assert response[0] == expected, response
            if expected == "C":
                assert response[1] == body

        try:
            with memory_guard(worker, a.floor) as memory:
                libs = ensure_with_matching_libraries(
                    worker, spec, [x.resolve() for x in a.library], 4096
                )
                if not admitted:
                    a.ready.touch()
                    if not is_tegra():
                        wait_for_broker_ready(token)
                    admitted = True
                outside = [speak(f"{int(control)}-outside-{i}", t) for i, t in enumerate(PROMPTS)]
                assert outside[0] == outside[2]
                if not control:
                    baseline = outside
                else:
                    assert outside == baseline, "ordinary request PCM changed"
                    for sequence in range(2):
                        ctl("begin")
                        first = speak(f"joined-{sequence}-0", PROMPTS[0])
                        second = speak(f"joined-{sequence}-1", PROMPTS[1])
                        assert first == outside[0]
                        assert second != outside[1], "continuation did not affect waveform state"
                        ctl("end")
                        assert speak(f"after-end-{sequence}", PROMPTS[0]) == outside[0]
                    assert chunks["joined-0-1"] == chunks["joined-1-1"]
                    ctl("begin")
                    speak("abort-target", PROMPTS[1])
                    ctl("abort")
                    assert speak("after-abort", PROMPTS[0]) == outside[0]
                    ctl("begin")
                    speak("bad-id-target", PROMPTS[1])
                    ctl("end", "2" * 32, "E")
                    assert speak("after-invalid", PROMPTS[0]) == outside[0]
                    log = os.pread(
                        worker.stderr_log.fileno(), os.fstat(worker.stderr_log.fileno()).st_size, 0
                    ).decode()
                    states = re.findall(
                        r"block=(\d+) prior_bytes=(\d+) bytes=(\d+) position=(\d+) frames=(\d+)",
                        log,
                    )
                    report["state_transfers"] = [list(map(int, x)) for x in states]
                    assert len(states) == 6, states
                    for first, second in ((states[0], states[1]), (states[2], states[3])):
                        x = list(map(int, first))
                        y = list(map(int, second))
                        assert x[1] == 0 and y[1] == x[2] == y[2] and y[3] == x[3] + y[4]

                    def join_stats(x, y):
                        left = struct.unpack("<" + str(len(x) // 2) + "h", x)
                        right = struct.unpack("<" + str(len(y) // 2) + "h", y)
                        local = left[-480:] + right[:480]
                        rms = math.sqrt(statistics.fmean(v * v for v in local))
                        return {
                            "jump": abs(right[0] - left[-1]),
                            "local_rms": rms,
                            "jump_over_rms": abs(right[0] - left[-1]) / max(rms, 1e-12),
                        }

                    report["joins"] = {
                        "reset": join_stats(outside[0], outside[1]),
                        "continuous": join_stats(chunks["joined-0-0"], chunks["joined-0-1"]),
                    }
                report["cases"].append(
                    {"control": control, "runs": rows, "memory": memory, "libraries": libs}
                )
        finally:
            worker.close()
            a.out.write_text(json.dumps(report, indent=2) + "\n")
    assert all(r["continuity"]["ok"] for c in report["cases"] for r in c["runs"])
    report["passed"] = True
    a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
