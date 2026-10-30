"""Qualify native CUDA head execution against frozen encoder-feature fixtures.

The manifest binds the head, input features and reference logits by SHA-256.
This gate loads only the small head; encoder sharing, streaming and a public
runtime route require separate integration. Tegra retains 3 GiB admission and
2 GiB active headroom. Discrete CUDA requires an owned 4096 MiB scoped lease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_omni_adapters.accelerator import gpu_facts, is_tegra, process_is_gpu_resident
from qwen_omni_adapters.audio_ctc import CTCLexiconDecoder
from runtime.train_audio_ctc_head import require_scope, wait_for_broker_ready, word_error_rate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "head", "manifest", "work", "ready", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    tegra = is_tegra()
    if tegra:
        if uuid or token:
            parser.error("Tegra uses integrated GPU residency, without a discrete lease")
    else:
        require_scope(uuid, token)
    if args.ready.exists() or args.out.exists():
        parser.error("fresh readiness and output paths are required")
    data = json.loads(args.manifest.read_text())
    assert digest(args.head) == data["head_sha256"]
    cases = data["cases"]
    assert 1 <= len(cases) <= 4096 and len({x["id"] for x in cases}) == len(cases)
    assert {x["split"] for x in cases} == {"validation", "test"}
    decoder = CTCLexiconDecoder(data["lexicon"], data["pieces"])
    base = args.manifest.resolve().parent
    for case in cases:
        for field, key in (
            ("feature_file", "feature_sha256"),
            ("expected_file", "expected_sha256"),
        ):
            assert digest(base / case[field]) == case[key]
    args.work.mkdir(exist_ok=False)
    (args.work / "logits").mkdir()
    inputs = args.work / "inputs.txt"
    inputs.write_text("".join(str((base / x["feature_file"]).resolve()) + "\n" for x in cases))
    process = None
    memory = []
    stopped = threading.Event()
    invalid_memory = threading.Event()

    def available():
        facts = gpu_facts()
        assert facts
        return facts[0]["vram_total_mib"] - facts[0]["vram_used_mib"]

    def observe():
        while not stopped.wait(0.1):
            try:
                value = available()
            except (OSError, ValueError, TypeError, KeyError, AssertionError):
                invalid_memory.set()
                value = -1
            memory.append(value)
            if value < 2048 and process is not None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                return

    if tegra:
        assert available() >= 3072, "insufficient integrated-memory headroom"
    observer = threading.Thread(target=observe, daemon=True) if tegra else None
    observations = []
    started = time.monotonic()
    try:
        if observer:
            observer.start()
        with (args.work / "native.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    str(args.binary.resolve()),
                    str(args.head.resolve()),
                    str(inputs.resolve()),
                    str((args.work / "logits").resolve()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
            assert select.select([process.stdout], [], [], 120)[0]
            assert process.stdout.read(2) == b"R\n"
            assert process_is_gpu_resident(process.pid, uuid)
            producer_pid = process.pid
            args.ready.touch()
            if not tegra:
                wait_for_broker_ready(token)
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            for index, case in enumerate(cases):
                assert select.select([process.stdout], [], [], 120)[0]
                fields = process.stdout.readline().decode().split()
                assert len(fields) == 2 and int(fields[0]) == case["frames"]
                logits = np.fromfile(args.work / "logits" / f"{index}.f32", dtype="<f4").reshape(
                    -1, 2048
                )
                expected = np.fromfile(base / case["expected_file"], dtype="<f4").reshape(-1, 2048)
                assert logits.shape == expected.shape == (case["frames"], 2048)
                assert np.isfinite(logits).all() and np.isfinite(expected).all()
                difference = float(np.max(abs(logits - expected)))
                assert difference <= 0.005, (case["id"], difference)
                ids = decoder.decode(logits - np.logaddexp.reduce(logits, axis=1, keepdims=True))
                if ids is None:
                    path = logits.argmax(-1).tolist()
                    ids = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i - 1])]
                text = "".join(data["pieces"][i] for i in ids).strip()
                assert text == case["hypothesis"], (case["id"], text, case["hypothesis"])
                observations.append(
                    {
                        "id": case["id"],
                        "split": case["split"],
                        "empty_reference": not case["reference"],
                        "has_transcript": bool(text),
                        "wer": word_error_rate(case["reference"], text),
                        "native_ms": float(fields[1]),
                        "maximum_absolute_logit_error": difference,
                    }
                )
                if index % 50 == 0:
                    print(json.dumps({"compared": index + 1, "total": len(cases)}), flush=True)
            assert process.wait(timeout=20) == 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        stopped.set()
        if observer:
            observer.join(timeout=5)
    metrics = {}
    for split in ("validation", "test"):
        speech = [x for x in observations if x["split"] == split and not x["empty_reference"]]
        noise = [x for x in observations if x["split"] == split and x["empty_reference"]]
        assert speech and noise
        wer = statistics.mean(x["wer"] for x in speech)
        false_rate = statistics.mean(x["has_transcript"] for x in noise)
        assert wer <= 0.15 and false_rate <= 0.05
        metrics[split] = {
            "mean_speech_wer": wer,
            "empty_false_transcript_rate": false_rate,
            "speech_rows": len(speech),
            "noise_rows": len(noise),
        }
    assert not tegra or memory and min(memory) >= 2048 and not invalid_memory.is_set()
    report = {
        "schema": "robit.omni.native-ctc-fixtures.v1",
        "head_sha256": digest(args.head),
        "manifest_sha256": digest(args.manifest),
        "binary_sha256": digest(args.binary),
        "producer_pid": producer_pid,
        "gpu_uuid": uuid,
        "tegra": tegra,
        "native_cuda_only": True,
        "all_decoder_text_exact": True,
        "maximum_absolute_logit_error": max(
            x["maximum_absolute_logit_error"] for x in observations
        ),
        "native_median_ms": statistics.median(x["native_ms"] for x in observations),
        "minimum_available_mib": min(memory) if memory else None,
        "metrics": metrics,
        "fixture_rows": len(cases),
        "elapsed_s": time.monotonic() - started,
        "language_weights_loaded": False,
        "encoder_executed_in_this_probe": False,
        "runtime_route_enabled": False,
        "streaming_qualified": False,
        "whole_release_qualified": False,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
