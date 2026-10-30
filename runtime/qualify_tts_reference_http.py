"""Exercise bundled full-reference conditioning through the real synthesis routes.

Use a scoped foreground GPU lease on discrete CUDA. The first reference encoder
is the initial reserved load; subsequent stages use prepare/ready. On Tegra no
lease is used and the runtime's integrated-memory admission remains authoritative.
No public service or port is started, and no deployed service is restarted.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_omni_adapters.accelerator import is_tegra, process_is_gpu_resident
from runtime import tts_server as tts
from runtime.benchmark_tts_conditioning import PROMPTS
from runtime.verify_pcm_stream import analyze_pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "binary",
        "model",
        "projector",
        "encoder",
        "reference",
        "transcript-file",
        "work-dir",
        "ready-file",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--automatic-packets", action="store_true")
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if not is_tegra() and (not uuid.startswith("GPU-") or "," in uuid or not token):
        parser.error("reserve one exact GPU UUID and positive VRAM budget through docker gpu run")
    if is_tegra() and (uuid or token):
        parser.error("Tegra uses its integrated GPU without a discrete lease")
    if not is_tegra():
        status = subprocess.run(
            ["docker", "gpu", "status"], capture_output=True, text=True, check=True, timeout=10
        )
        own = next(
            (item for item in json.loads(status.stdout)["leases"] if item.get("token") == token),
            None,
        )
        if not own or own.get("requested_mib", 0) < 12000 or own.get("gpu_uuids") != [uuid]:
            parser.error(
                "qualification requires its own exact-GPU reservation of at least 12000 MiB"
            )
    work = args.work_dir
    work.mkdir(exist_ok=False)
    (work / "temporary").mkdir()
    tempfile.tempdir = str(work / "temporary")
    ready = args.ready_file
    assert not ready.exists()
    original = tts._broker_transition
    transitions = []

    def supervised(action, token, timeout_s=330):
        if action == "ready":
            subprocess.run(
                ["docker", "gpu", "ready", token, "--pid", str(os.getpid())],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
            ready.touch()
        elif ready.exists():
            original(action, token, timeout_s)
        else:
            assert (
                action == "prepare"
            )  # The foreground wrapper already reserved the initial 12000 MiB.
        transitions.append(action)

    tts._broker_transition = supervised

    config = tts.Config(
        binary=args.binary.resolve(),
        model=args.model.resolve(),
        projector=args.projector.resolve(),
        reference_encoder=args.encoder.resolve(),
        require_gpu=True,
        gpu_uuid=uuid,
        lease_token=token,
        active_pid_file=work / "worker.pid",
        adaptive_context=True,
        idle_keepalive_s=30,
        max_frames=96,
        stream_frames=None if args.automatic_packets else 2,
        timeout_s=120,
    )
    app = tts.create_app(config)
    client = app.test_client()
    body = {
        "text": PROMPTS[0],
        "speaker_file": str(args.reference.resolve()),
        "reference_text": args.transcript_file.read_text(encoding="utf-8").strip(),
        "seed": 42,
        "stream_frames": 2,
        "max_frames": 96,
    }
    if args.automatic_packets:
        body.pop("stream_frames")
    report = {
        "schema": "robit.omni.tts-reference-http.v1",
        "observations": [],
        "transitions": transitions,
        "public_voice_contract_qualified": False,
        "perceptual_cloning_quality_qualified": False,
        "checkpoint_sha256": tts._reference_digest(args.encoder.resolve()),
        "reference_sha256": tts._reference_digest(args.reference),
        "transcript_sha256": tts._reference_digest(args.transcript_file),
        "automatic_packets": args.automatic_packets,
    }
    outputs = []

    def stream(payload, label, route="/synthesize/stream"):
        started = time.monotonic()
        response = client.post(route, json=payload, buffered=False)
        assert response.status_code == 200, (response.status_code, response.get_data())
        result = bytearray()
        first = None
        native_libraries = {}
        pcm_pid = None
        try:
            for chunk in response.response:
                if first is None:
                    first = time.monotonic() - started
                    pcm_pid = int(config.active_pid_file.read_text())
                    assert process_is_gpu_resident(pcm_pid, uuid), (
                        "first PCM was not produced by a resident GPU worker"
                    )
                    for line in Path(f"/proc/{pcm_pid}/maps").read_text().splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and Path(fields[5]).name.startswith(
                            ("libllama", "libmtmd", "libggml")
                        ):
                            path = Path(fields[5])
                            if path.name.startswith(("libllama", "libmtmd")):
                                assert path.parent.resolve() == args.binary.resolve().parent
                            native_libraries[path.name] = str(path)
                result.extend(chunk)
        finally:
            response.close()
        pcm = bytes(result)
        assert pcm and len(pcm) % 2 == 0
        (work / (label + ".wav")).write_bytes(tts._pcm_wav(pcm))
        selected_frames = int(response.headers["X-Audio-Stream-Frames"])
        if args.automatic_packets:
            assert selected_frames == 4, "small-context automatic packet policy was not exercised"
        continuity = analyze_pcm(pcm, stream_frames=selected_frames)
        assert continuity["ok"], json.dumps({"label": label, "continuity": continuity})
        health = client.get("/healthz").json
        report["observations"].append(
            {
                "label": label,
                "first_pcm_s": first,
                "total_s": time.monotonic() - started,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "pcm_bytes": len(pcm),
                "continuity": continuity,
                "stream_frames": selected_frames,
                "producer_pid": pcm_pid,
                "producer_gpu_resident": True,
                "native_libraries": native_libraries,
                "reference_encodes": health["reference_encodes"],
                "reference_cache_hits": health["reference_cache_hits"],
                "worker_pid": health["persistent_pid"],
                "gpu_resident": health["gpu_resident"],
                "idle_residency_state": health["idle_residency_state"],
            }
        )
        assert not list((work / "temporary").iterdir()), list((work / "temporary").iterdir())
        return pcm

    try:
        for i, text in enumerate(PROMPTS):
            outputs.append(stream({**body, "text": text}, f"saved-{i}"))
        assert outputs[0] == outputs[2]
        health = client.get("/healthz").json
        assert health["reference_encodes"] == 1
        assert health["gpu_resident"] or health["idle_residency_state"] == "released:headroom"
        if all(x["worker_pid"] for x in report["observations"]):
            assert len({x["worker_pid"] for x in report["observations"]}) == 1
        assert client.post("/residency", json={"action": "shed"}).status_code == 200
        assert stream(body, "saved-reloaded") == outputs[0]
        assert client.get("/healthz").json["reference_encodes"] == 1
        inline = dict(body)
        inline.pop("speaker_file")
        inline["speaker_audio"] = {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(args.reference.read_bytes()).decode(),
        }
        assert stream(inline, "inline") == outputs[0]
        complete = client.post("/synthesize", json=inline)
        assert complete.status_code == 200, complete.get_data()
        tts._validate_wav(complete.data)
        with wave.open(io.BytesIO(complete.data), "rb") as wav:
            assert wav.readframes(wav.getnframes()) == outputs[0]
        response = client.post("/synthesize/stream", json=inline, buffered=False)
        assert response.status_code == 200
        assert next(iter(response.response))
        response.close()
        assert not list((work / "temporary").iterdir())
        batch = {**inline, "blocks": list(PROMPTS[:2])}
        batch.pop("text")
        assert stream(batch, "inline-batch", "/synthesize/stream/batch") == outputs[0] + outputs[1]
        assert stream(body, "saved-after-inline") == outputs[0]
        report.update(
            all_saved_a_b_a_exact=True,
            inline_matches_saved=True,
            complete_wav_matches_stream=True,
            cancel_cleans_request=True,
            inline_batch_exact=True,
            temporary_reference_files_remaining=0,
        )
    finally:
        app.extensions["tts_idle_residency"].stop()
        client.post("/residency", json={"action": "shed"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["observations", "transitions"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
