"""Extract actual pre-projection Qwen3A features without loading language weights.

Build runtime/audio_feature_probe.cpp against the pinned native libraries. This
Linux/discrete-CUDA training helper requires a scoped reservation of at least
4096 MiB. The child loads vocabulary metadata only; only the existing audio
encoder executes. Cached features are bound to their source audio, projector,
producer and dataset hashes. Reuse requires the exact producer and projector.
"""

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_omni_adapters.accelerator import process_is_gpu_resident
from runtime.train_audio_ctc_head import require_scope


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--projector", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reuse", type=Path)
    a = p.parse_args()
    assert not a.ready.exists()
    a.work.mkdir(exist_ok=False)
    (a.work / "features").mkdir()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    require_scope(uuid, token)
    rows = [json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    metadata = {}

    if a.reuse:
        old = json.loads((a.reuse / "feature-manifest.json").read_text())
        assert (
            old["native_features_binary_sha256"]
            == hashlib.sha256(a.binary.read_bytes()).hexdigest()
        )
        assert old["projector_sha256"] == file_sha256(a.projector)
        for path, item in old["features"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == item["audio_sha256"]
            assert (
                hashlib.sha256(Path(item["feature_path"]).read_bytes()).hexdigest()
                == item["feature_sha256"]
            )
            metadata[path] = item
    paths = sorted({r["audio"] for r in rows} - set(metadata))
    (a.work / "inputs.txt").write_text("".join(path + "\n" for path in paths))
    projector_sha256 = file_sha256(a.projector)
    started = time.monotonic()
    with (a.work / "native.log").open("wb") as log:
        process = subprocess.Popen(
            [
                str(a.binary.resolve()),
                str(a.model.resolve()),
                str(a.projector.resolve()),
                str(a.work / "inputs.txt"),
                str(a.work / "features"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        try:
            assert select.select([process.stdout], [], [], 120)[0]
            assert process.stdout.read(2) == b"R\n" and process_is_gpu_resident(process.pid, uuid)
            a.ready.touch()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                state = json.loads(subprocess.check_output(["docker", "gpu", "status"], text=True))
                if any(
                    row.get("token") == token and row.get("state") == "active"
                    for row in state["leases"]
                ):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("native feature admission failed")
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            for index, path in enumerate(paths):
                assert select.select([process.stdout], [], [], 120)[0], index
                line = process.stdout.readline().decode()
                fields = line.strip().split()
                assert len(fields) == 2, (index, line)
                frames, tokens = map(int, fields)
                assert frames == tokens and frames > 0
                feature = a.work / "features" / (str(index) + ".f32")
                assert feature.stat().st_size == frames * 1280 * 4
                metadata[path] = {
                    "frames": frames,
                    "audio_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "feature_path": str(feature),
                    "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                }
                if index % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "processed": index + 1,
                                "total": len(paths),
                                "elapsed_s": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
            assert process.wait(timeout=20) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report = {
        "schema": "robit.omni.native-audio-feature-cache.v1",
        "dtype": "float32",
        "width": 1280,
        "language_weights_loaded": False,
        "language_forward_passes": 0,
        "native_features_binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "projector_sha256": projector_sha256,
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "features": metadata,
        "elapsed_s": time.monotonic() - started,
    }
    (a.work / "feature-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "features": len(metadata), "elapsed_s": report["elapsed_s"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
