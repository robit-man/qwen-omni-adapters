"""Compare cold saved-reference preparation through actual adapter and TTS HTTP paths.

This bounded Tegra qualification uses the already resident Ornith primary, one
private loopback TTS server, unchanged checkpoints and a fixed committed reply.
It never restarts a deployed service. The private precursor produced the B3c
receipt; this maintained reproducer follows the same measured path. Physical
speaker playback, broad prompt performance and whole-model quality are separate
gates. Use fresh work paths and retain the runtime memory admission floors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx
from werkzeug.serving import make_server

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import is_tegra, process_is_gpu_resident
from qwen_omni_adapters.contract import parse_adapter_request
from runtime import adapter_server as adapter
from runtime import tts_server as tts
from runtime.verify_pcm_stream import analyze_pcm


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "sidecar", "reference", "transcript", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 5:
        parser.error("--pairs must be 1..5")
    assert is_tegra(), "This qualification uses the existing integrated-GPU primary on Orin only"
    args.work.mkdir(exist_ok=False)
    endpoint = "http://127.0.0.1:8901/v1/chat/completions"
    with httpx.Client(timeout=120) as client:
        models = client.get("http://127.0.0.1:8901/v1/models").json()
        model = models["data"][0]["id"]
        assert model.endswith("852922174ee4f76621df26105333f1dfe2171cdfb60ebe5a4b013836681a8a77")
    target = "The blue bicycle is parked beside the garden gate."
    payload = {
        "model": "robit/ornith-1.5-omni:edge-v2-reference-candidate-20261030",
        "stream": False,
        "think": False,
        "cache_prompt": False,
        "messages": [
            {
                "role": "user",
                "content": "Repeat this sentence exactly, without commentary: " + target,
            }
        ],
        "options": {"temperature": 0, "seed": 42, "num_predict": 80},
        "response_modalities": ["text", "audio"],
        "speech": {
            "speaker_file": str(args.reference),
            "reference_text": args.transcript.read_text().strip(),
            "seed": 42,
            "max_frames": 96,
        },
    }
    report = {
        "schema": "robit.omni.reference-overlap-pipeline.v1",
        "model": model,
        "tts_sha256": tts._reference_digest(args.sidecar),
        "observations": [],
        "physical_playback_measured": False,
        "whole_model_qualified": False,
    }
    original_encode = tts._encode_reference
    try:
        for pair in range(args.pairs):
            for enabled in [False, True] if pair % 2 == 0 else [True, False]:
                label = f"pair-{pair}-" + ("prepared" if enabled else "baseline")
                directory = args.work / label
                directory.mkdir()
                origin = [time.monotonic()]
                spans = []

                def encode(*a, clock=origin, rows=spans, **kw):
                    start = time.monotonic() - clock[0]
                    codes = original_encode(*a, **kw)
                    rows.append({"start_s": start, "end_s": time.monotonic() - clock[0]})
                    return codes

                tts._encode_reference = encode
                cfg = tts.Config(
                    binary=args.binary,
                    model=args.sidecar,
                    projector=args.sidecar,
                    reference_encoder=args.sidecar,
                    adaptive_context=True,
                    prepare_reference=enabled,
                    require_gpu=True,
                    max_frames=96,
                    active_pid_file=directory / "worker.pid",
                    timeout_s=120,
                )
                app = tts.create_app(cfg)
                health = app.test_client()
                preparations = []

                @app.after_request
                def record(response, clock=origin, rows=preparations, application=app):
                    if tts.request.path == "/prepare-reference":
                        rows.append(
                            {
                                "end_s": time.monotonic() - clock[0],
                                "result": response.get_json(),
                                "tts_loaded": application.extensions["tts_idle_residency"].worker.ready,
                            }
                        )
                    return response

                server = make_server("127.0.0.1", 0, app, threaded=True)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                acfg = adapter.Config(
                    comprehension_url="",
                    comprehension_model="",
                    language_url=endpoint,
                    language_api="openai",
                    language_model=model,
                    timeout_s=120,
                    tts_url=f"http://127.0.0.1:{server.server_port}/synthesize",
                    prepare_reference=enabled,
                    language_max_output_tokens=80,
                )
                stages, pcm = [], bytearray()
                memory, stopped = [], threading.Event()

                def observe(stop=stopped, rows=memory):
                    while not stop.wait(0.1):
                        available = tts._reference_headroom()
                        if available is not None:
                            rows.append(available)

                monitor = threading.Thread(target=observe, daemon=True)
                monitor.start()
                try:
                    origin[0] = time.monotonic()
                    with httpx.Client(timeout=120, trust_env=False) as client:
                        for raw in adapter.execute_stream(
                            parse_adapter_request(payload), acfg, client
                        ):
                            event = json.loads(raw)
                            kind = event["type"]
                            if kind in {"stage", "audio_start", "audio_end"}:
                                stages.append(
                                    {
                                        "type": kind,
                                        "stage": event.get("stage"),
                                        "time_s": time.monotonic() - origin[0],
                                    }
                                )
                            if kind == "audio_delta":
                                pcm.extend(base64.b64decode(event["audio"]["data"]))
                            if kind == "audio_start":
                                pid = int(cfg.active_pid_file.read_text())
                                assert process_is_gpu_resident(pid, "")
                            if kind == "final":
                                final = event["response"]
                    total = time.monotonic() - origin[0]
                    assert final["message"]["content"] == target, final["message"]["content"]
                    assert not final["message"].get("tool_calls") and not final["message"].get(
                        "thinking"
                    )
                    assert len(spans) == 1 and pcm
                    continuity = analyze_pcm(bytes(pcm), stream_frames=4)
                    assert continuity["ok"], continuity
                    assert memory and min(memory) >= 2048
                    if enabled:
                        assert preparations == [
                            {
                                "end_s": preparations[0]["end_s"],
                                "result": {"prepared": True},
                                "tts_loaded": False,
                            }
                        ]
                    (directory / "output.wav").write_bytes(tts._pcm_wav(pcm))
                    row = {
                        "label": label,
                        "prepared": enabled,
                        "pair": pair,
                        "stages": stages,
                        "reference_encoder_spans": spans,
                        "preparations": preparations,
                        "first_audio_s": next(
                            x["time_s"] for x in stages if x["type"] == "audio_start"
                        ),
                        "total_s": total,
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "pcm_bytes": len(pcm),
                        "continuity": continuity,
                        "minimum_available_mib": min(memory),
                        "reference_encodes": health.get("/healthz").json["reference_encodes"],
                    }
                    assert row["reference_encodes"] == 1
                    report["observations"].append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "label",
                                    "first_audio_s",
                                    "total_s",
                                    "minimum_available_mib",
                                )
                            }
                        ),
                        flush=True,
                    )
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(5)
                    idle = app.extensions["tts_idle_residency"]
                    idle.stop()
                    idle.shed()
                    stopped.set()
                    monitor.join(5)
                (args.work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert len({x["pcm_sha256"] for x in report["observations"]}) == 1
        for key in ("first_audio_s", "total_s"):
            baseline = statistics.median(
                x[key] for x in report["observations"] if not x["prepared"]
            )
            prepared = statistics.median(x[key] for x in report["observations"] if x["prepared"])
            report[key] = {
                "baseline_median": baseline,
                "prepared_median": prepared,
                "reduction_percent": 100 * (1 - prepared / baseline),
            }
        report["all_pcm_exact"] = True
        (args.work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "observations"}), flush=True)
    finally:
        tts._encode_reference = original_encode


if __name__ == "__main__":
    main()
