"""Compare projected speaker conditioning using the existing bundled encoder.

Build tts_speaker_probe.cpp against the pinned native libraries and run this
qualifier through a scoped CUDA reservation. Input JSON rows contain id/path;
generated rows also declare pair, reference_id and frames (4 or 8). All paired
WAVs use one encoder/backend. This is not an independent calibrated speaker
verification metric or a substitute for perceptual release qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import process_is_gpu_resident
from runtime.benchmark_tts_conditioning import wait_for_broker_ready
from runtime.tts_server import _reference_digest, _stop_process_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "projector", "inputs", "work-dir", "ready-file", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    if not uuid.startswith("GPU-") or "," in uuid or not token:
        parser.error("use one exact GPU UUID through the scoped CUDA wrapper")
    if args.ready_file.exists():
        parser.error("use a fresh readiness path")
    rows = json.loads(args.inputs.read_text())
    if not isinstance(rows, list) or not 2 <= len(rows) <= 32:
        parser.error("input JSON must contain 2..32 WAV descriptors")
    if len({r["id"] for r in rows}) != len(rows):
        parser.error("input ids must be unique")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    paths = args.work_dir / "inputs.txt"
    resolved = [Path(r["path"]).resolve(strict=True) for r in rows]
    if any(
        "\n" in str(p) or not p.is_file() or p.stat().st_size > 50 * 1024 * 1024 for p in resolved
    ):
        parser.error("inputs must be bounded regular files")
    paths.write_text("".join(str(path) + "\n" for path in resolved))
    features_path = args.work_dir / "speaker-features.f32"
    with (args.work_dir / "native.log").open("wb") as log:
        process = subprocess.Popen(
            [
                str(args.binary),
                str(args.model),
                str(args.projector),
                str(paths),
                str(features_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while not select.select([process.stdout], [], [], 1)[0]:
                if time.monotonic() >= deadline or process.poll() is not None:
                    raise RuntimeError("speaker probe failed readiness")
            if process.stdout.read(2) != b"R\n" or not process_is_gpu_resident(process.pid, uuid):
                raise RuntimeError("speaker probe did not prove exact GPU residency")
            args.ready_file.touch()
            wait_for_broker_ready(token)
            process.communicate(input=b"GO\n", timeout=120)
            if process.returncode:
                raise RuntimeError(f"speaker probe failed: {process.returncode}")
        finally:
            _stop_process_group(process)
    values = np.fromfile(features_path, dtype="<f4").reshape(len(rows), 2048)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not (norms > 0).all():
        raise RuntimeError("speaker encoder returned invalid features")
    values /= norms
    features = {r["id"]: vector for r, vector in zip(rows, values, strict=True)}
    pairs = {}
    for row in rows:
        if row.get("pair"):
            pairs.setdefault(row["pair"], {})[row["frames"]] = row
    report = {
        "schema": "robit.omni.projected-speaker-conditioning-comparison.v1",
        "scope": "Same-bundle projected speaker encoder on one matched backend; not calibrated speaker verification.",
        "model_sha256": _reference_digest(args.model),
        "projector_sha256": _reference_digest(args.projector),
        "input_wav_sha256": {
            r["id"]: hashlib.sha256(p.read_bytes()).hexdigest()
            for r, p in zip(rows, resolved, strict=True)
        },
        "pairs": [],
        "tolerance_absolute_cosine": 0.001,
        "perceptual_cloning_quality_qualified": False,
    }
    for name, pair in pairs.items():
        if set(pair) != {4, 8} or pair[4]["reference_id"] != pair[8]["reference_id"]:
            raise ValueError("each pair needs four/eight-frame WAVs and one reference")
        ref = features[pair[4]["reference_id"]]
        four, eight = features[pair[4]["id"]], features[pair[8]["id"]]
        delta = float(four @ ref - eight @ ref)
        report["pairs"].append(
            {
                "id": name,
                "four_vs_eight_cosine": float(four @ eight),
                "four_vs_reference_cosine": float(four @ ref),
                "eight_vs_reference_cosine": float(eight @ ref),
                "delta_reference_cosine": delta,
                "no_material_regression": delta >= -0.001,
            }
        )
    if not pairs:
        raise ValueError("no paired observations declared")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pairs"]), flush=True)
    if not all(pair["no_material_regression"] for pair in report["pairs"]):
        raise SystemExit("projected speaker-conditioning comparison failed")


if __name__ == "__main__":
    main()
