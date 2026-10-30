"""Bounded native full-reference packet-cadence comparison.

Precomputed reference prompts must come from the corresponding bundled encoder.
Run with an exact scoped CUDA reservation, or a positive Tegra memory floor.
A short/unscorable sample is a failed quality gate, not a waived assertion.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_omni_adapters.accelerator import accelerator_profile, is_tegra
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
    for name in (
        "binary",
        "model",
        "projector",
        "reference",
        "reference-prompt",
        "work",
        "ready",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--library", type=Path, action="append", default=[])
    p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    p.add_argument("--minimum-available-mib", type=int, default=0)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    assert not a.ready.exists()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    assert (is_tegra() and not uuid and not token and a.minimum_available_mib > 0) or (
        not is_tegra() and uuid.startswith("GPU-") and "," not in uuid and token
    )
    command = tts._command

    def full(*args, **kwargs):
        result = command(*args, **kwargs)
        result.extend(["--tts-reference-prompt-file", str(a.reference_prompt)])
        return result

    tts._command = full
    report = {
        "schema": "robit.omni.native-cadence-pilot.v1",
        "accelerator": accelerator_profile(),
        "cases": [],
        "scope": "Full-reference native chunk timing and raw continuity; no product change.",
    }
    admitted = False
    for frames in a.windows:
        config = tts.Config(
            binary=a.binary.resolve(),
            model=a.model.resolve(),
            projector=a.projector.resolve(),
            gpu_uuid=uuid,
            require_gpu=True,
            timeout_s=120,
            max_frames=96,
            stream_frames=frames,
            adaptive_context=True,
        )
        worker = tts.PersistentTTSWorker(config)
        rows = []
        try:
            with memory_guard(worker, a.minimum_available_mib) as memory:
                spec = tts._synthesis_spec(
                    config,
                    {
                        "text": PROMPTS[0],
                        "speaker_file": str(a.reference.resolve()),
                        "seed": 42,
                        "stream_frames": frames,
                    },
                )
                libraries = ensure_with_matching_libraries(
                    worker, spec, [x.resolve() for x in a.library], 4096
                )
                if not admitted:
                    a.ready.touch()
                    if not is_tegra():
                        wait_for_broker_ready(token)
                    admitted = True
                for i, text in enumerate(
                    (PROMPTS[0], PROMPTS[1], PROMPTS[0], PROMPTS[1], PROMPTS[0])
                ):
                    start = time.monotonic()
                    raw = bytearray()
                    arrivals = []
                    for chunk in worker.stream(replace(spec, text=text)):
                        arrivals.append(
                            {
                                "time_s": time.monotonic() - start,
                                "audio_s": len(chunk) / 48000,
                                "bytes": len(chunk),
                            }
                        )
                        raw.extend(chunk)
                    elapsed = time.monotonic() - start
                    (a.work / f"frames-{frames}-{i}.wav").write_bytes(tts._pcm_wav(bytes(raw)))
                    try:
                        continuity = analyze_pcm(bytes(raw), stream_frames=frames)
                    except ValueError as exc:
                        continuity = {"ok": False, "unscored_reason": str(exc)}
                    rows.append(
                        {
                            "index": i,
                            "text_chars": len(text),
                            "text": text,
                            "first_pcm_s": arrivals[0]["time_s"],
                            "total_s": elapsed,
                            "audio_s": len(raw) / 48000,
                            "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                            "arrivals": arrivals,
                            "continuity": continuity,
                        }
                    )
                assert rows[0]["pcm_sha256"] == rows[2]["pcm_sha256"] == rows[4]["pcm_sha256"]
                report["cases"].append(
                    {"frames": frames, "runs": rows, "memory": memory, "libraries": libraries}
                )
                print(
                    json.dumps(
                        {
                            "frames": frames,
                            "first_pcm_s": [round(r["first_pcm_s"], 4) for r in rows],
                            "total_s": [round(r["total_s"], 4) for r in rows],
                            "continuity": [r["continuity"]["ok"] for r in rows],
                        }
                    ),
                    flush=True,
                )
        finally:
            worker.close()
    report["raw_continuity_qualified"] = all(
        r["continuity"]["ok"] for c in report["cases"] for r in c["runs"]
    )
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["raw_continuity_qualified"]:
        raise SystemExit(
            "raw continuity or minimum scoring length failed; all observations retained in the report"
        )


if __name__ == "__main__":
    main()
