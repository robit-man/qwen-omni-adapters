"""Compare request-scoped and bounded idle TTS through real HTTP route handlers.

Linux CUDA only: use a scoped broker run on discrete GPUs, or a positive
memory floor on Tegra. Supply a fresh readiness filename to the broker wrapper.
No inference service or primary reasoning trunk is started or restarted.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import statistics
import sys
import time
import wave
from dataclasses import replace
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
from runtime.tts_server import Config, _warm_spec, create_app
from runtime.verify_pcm_stream import analyze_pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "projector", "reference", "alternate-reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--minimum-available-mib", type=int, default=6144)
    args = parser.parse_args()
    tegra = is_tegra()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if tegra:
        if uuid or token or args.minimum_available_mib <= 0:
            parser.error("Tegra requires a positive memory floor and no discrete GPU scope")
    elif not uuid.startswith("GPU-") or "," in uuid or not token or args.ready_file is None:
        parser.error("use one exact scoped GPU UUID and a fresh --ready-file")
    if args.ready_file is not None and args.ready_file.exists():
        parser.error("readiness file already exists; use a fresh run directory")
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    if not 0 < args.idle_seconds <= 30 or args.context_size <= 0:
        parser.error("bounded qualification needs 0 < idle seconds <= 30 and a positive context")
    extras = [path.resolve(strict=True) for path in args.extra_library_dir]
    config = Config(binary=args.binary.resolve(strict=True), model=args.model.resolve(strict=True),
                    projector=args.projector.resolve(strict=True), require_gpu=True, gpu_uuid=uuid,
                    warm_speaker_file=str(args.reference.resolve(strict=True)), max_frames=96,
                    stream_frames=2, timeout_s=120, idle_min_available_mib=args.minimum_available_mib)
    body = {"speaker_file": config.warm_speaker_file, "seed": 42, "stream_frames": 2, "max_frames": 96}
    previous = {key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LLAMA_ARG_CTX_SIZE")}
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, [config.binary.parent, *extras]))
    os.environ["LLAMA_ARG_CTX_SIZE"] = str(args.context_size)
    report = {"schema": "robit.omni.idle-tts-benchmark.v1", "accelerator": accelerator_profile(),
              "context_size": args.context_size, "scope": "real route handlers with in-process client and native GPU workers",
              "samples": [], "complete": False}
    expected = []
    alternate_pcm = None
    admitted = tegra

    def streamed(client, payload):
        started = time.monotonic()
        response = client.post("/synthesize/stream", json=payload, buffered=False)
        assert response.status_code == 200
        chunks, first, pid, peak = [], None, None, 0
        try:
            for chunk in response.response:
                if first is None:
                    first = time.monotonic() - started
                    health = client.get("/healthz").get_json()
                    assert health["gpu_resident"]
                    pid = health["persistent_pid"]
                status = Path(f"/proc/{pid}/status").read_text().splitlines()
                peak = max(peak, next(int(line.split()[1]) for line in status if line.startswith("VmHWM:")))
                chunks.append(chunk)
        finally:
            response.close()
        raw = b"".join(chunks)
        assert raw and analyze_pcm(raw, stream_frames=2)["ok"]
        return raw, {"first_pcm_s": first, "total_s": time.monotonic() - started,
                     "pid": pid, "cpu_peak_rss_kib": peak,
                     "pcm_sha256": hashlib.sha256(raw).hexdigest()}

    try:
        for seconds in (0, args.idle_seconds):
            app = create_app(replace(config, idle_keepalive_s=seconds))
            client, idle = app.test_client(), app.extensions["tts_idle_residency"]
            sample = {"idle_keepalive_s": seconds, "runs": []}
            report["samples"].append(sample)
            try:
                with memory_guard(idle.worker, 2048 if tegra else 0) as memory:
                    sample["memory_headroom"] = memory
                    # Prime both layouts identically, then unload before timing.
                    assert client.post("/residency", json={"action": "warm"}).status_code == 200
                    assert client.get("/healthz").get_json()["gpu_resident"]
                    sample["native_libraries"] = ensure_with_matching_libraries(
                        idle.worker, _warm_spec(config), extras, args.context_size)
                    if not admitted:
                        args.ready_file.touch()
                        wait_for_broker_ready(token)
                        admitted = True
                    client.post("/residency", json={"action": "shed"})
                    for index, text in enumerate(PROMPTS):
                        pcm, run = streamed(client, {**body, "text": text})
                        run["text_index"] = index
                        if seconds == 0:
                            expected.append(pcm)
                        else:
                            assert pcm == expected[index], "retention changed PCM"
                        assert idle.worker.pid == (run["pid"] if seconds else None)
                        sample["runs"].append(run)
                    assert expected[0] == expected[2]
                    assert len({run["pid"] for run in sample["runs"]}) == (1 if seconds else 3)
                    alternate = {**body, "speaker_file": str(args.alternate_reference.resolve(strict=True)), "text": PROMPTS[0]}
                    if seconds == 0:
                        alternate_pcm, _ = streamed(client, alternate)
                    else:
                        # Verify expiry while the core measured profile still
                        # has headroom; later reference controls may cause a
                        # legitimate memory-pressure eviction.
                        process = idle.worker.process
                        assert process is not None
                        deadline = time.monotonic() + seconds + 2
                        while process.poll() is None and time.monotonic() < deadline:
                            time.sleep(.1)
                        assert process.poll() is not None and idle.worker.pid is None
                        assert idle.last_transition in {"released:expired", "released:headroom"}
                        report["post_sequence_release"] = idle.last_transition
                        pcm, _ = streamed(client, {**body, "text": PROMPTS[0]})
                        assert pcm == expected[0]
                        process = idle.worker.process
                        assert process is not None, "core reference no longer has retention headroom"
                        response = client.post("/synthesize", json=alternate)
                        assert response.status_code == 200
                        with wave.open(io.BytesIO(response.data), "rb") as wav:
                            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
                            assert wav.readframes(wav.getnframes()) == alternate_pcm
                        assert process.poll() is not None and idle.worker.pid != process.pid
                        report["profile_change_reloads_only_tts"] = True
                        inline = {key: value for key, value in body.items() if key != "speaker_file"}
                        inline.update(blocks=[PROMPTS[0]], speaker_audio=base64.b64encode(args.reference.read_bytes()).decode())
                        response = client.post("/synthesize/stream/batch", json=inline, buffered=False)
                        assert b"".join(response.response) == expected[0]
                        response.close()
                        assert idle.worker.pid is None
                        report["inline_reference_ephemeral"] = True
                        response = client.post("/synthesize/stream", json={**body, "text": PROMPTS[0]}, buffered=False)
                        next(iter(response.response))
                        process = idle.worker.process
                        response.close()
                        assert process.poll() is not None and idle.worker.pid is None
                        report["cancellation_closes_worker"] = True
                        pcm, _ = streamed(client, {**body, "text": PROMPTS[1]})
                        assert pcm == expected[1]
                        process = idle.worker.process
                        if process is None:
                            assert idle.last_transition == "released:headroom"
                            report["post_controls_retention_declined_for_headroom"] = True
                        else:
                            deadline = time.monotonic() + seconds + 2
                            while process.poll() is None and time.monotonic() < deadline:
                                time.sleep(.1)
                            assert process.poll() is not None and idle.worker.pid is None
                            assert idle.last_transition in {"released:expired", "released:headroom"}
                            report["post_controls_release"] = idle.last_transition
            finally:
                client.post("/residency", json={"action": "shed"})
                idle.stop()
        before, after = report["samples"]
        for field in ("first_pcm_s", "total_s"):
            base = statistics.median(run[field] for run in before["runs"])
            new = statistics.median(run[field] for run in after["runs"])
            report[field + "_median_reduction_percent"] = 100 * (1 - new / base)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
