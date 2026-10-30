"""Qualify shared sidecar bytes on separately cloned candidate Ollama tags.

This attaches the supplied bundle to the target tags. It never modifies source
tags, downloads weights, loads a model, or starts/stops an inference service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.ollama_sidecar import (
    attach_ollama_sidecar,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import INDEXED_AUDIO_BRIDGE_SCHEMA


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", nargs=2, required=True, metavar=("SOURCE", "TARGET"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    args = parser.parse_args()
    if len(args.pair) != 2 or len({name for pair in args.pair for name in pair}) != 4:
        parser.error("provide two distinct source/target pairs")
    sources = {}
    # Finish both identity checks before the first mutation.
    for source, target in args.pair:
        original = resolve_ollama_sidecar(model=source, models_dir=args.models_dir)
        candidate = resolve_ollama_sidecar(model=target, models_dir=args.models_dir)
        assert original["profile"] == candidate["profile"] == "trained-audio-bridge"
        assert original["standard_layers"] == candidate["standard_layers"], "target changed primary weights"
        sources[source] = original
    report = {"schema": "robit.omni.shared-sidecar-qualification.v2", "pairs": []}
    installed = set()
    old_sidecars = {}
    old_weights = {}
    new_weights = {}
    args.work_dir.mkdir(parents=True, exist_ok=False)
    for source, target in args.pair:
        original = sources[source]
        attach_ollama_sidecar(model=target, bundle_gguf=args.bundle, models_dir=args.models_dir)
        candidate = resolve_ollama_sidecar(model=target, models_dir=args.models_dir)
        assert candidate["inspection"]["manifest"]["schema"] == INDEXED_AUDIO_BRIDGE_SCHEMA
        assert original["standard_layers"] == candidate["standard_layers"]
        indexed = candidate["inspection"]["manifest"]["component_index"]["components"]
        prepared = prepare_ollama_sidecar(model=target, output_dir=args.work_dir / "views",
                                          models_dir=args.models_dir, reuse=True)
        components = {}
        for view, item in indexed.items():
            actual = digest(Path(prepared["views"][view]["output"]))
            assert actual == item["gguf_sha256"]
            assert original["inspection"]["manifest"]["component_files"][view]["sha256"] == actual
            components[view] = actual
        installed.add(candidate["bundle"])
        old_sidecars[original["layer"]["digest"]] = original["layer"]["size"]
        for resolved, weights in ((original, old_weights), (candidate, new_weights)):
            for layer in [resolved["layer"], *(entry["layer"] for entry in resolved["standard_layers"].values())]:
                weights[layer["digest"]] = layer["size"]
        report["pairs"].append({"source": source, "target": target,
            "source_sidecar": original["layer"]["digest"], "candidate_sidecar": candidate["layer"]["digest"],
            "primary_weights_unchanged": True,
            "standard_layers": {key: entry["layer"]["digest"] for key, entry in candidate["standard_layers"].items()},
            "extracted_components_sha256": components,
            "materialized_views": prepared["component_index"]["materialized_views"],
            "reused_views": prepared["component_index"]["reused_views"]})
    assert len(installed) == 1, "targets did not resolve the same physical sidecar"
    shared = Path(next(iter(installed)))
    report.update({"source_sidecar_bytes": sum(old_sidecars.values()),
        "candidate_sidecar_bytes": shared.stat().st_size,
        "sidecar_bytes_avoided_in_new_pair_install": sum(old_sidecars.values()) - shared.stat().st_size,
        "old_unique_weight_layer_bytes": sum(old_weights.values()),
        "new_unique_weight_layer_bytes": sum(new_weights.values()),
        "shared_installed_blob_sha256": digest(shared),
        "limits": ["Existing legacy blobs and tags retained; no current disk-space reclamation claimed",
                   "CPU packaging and exact checkpoint identity only; no new neural quality or GPU memory claim",
                   "Executable views still materialized; direct filtered loader remains pending",
                   "Candidate tags are not qualified Edge v2 releases"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"pairs", "limits"}}, indent=2))


if __name__ == "__main__":
    main()
