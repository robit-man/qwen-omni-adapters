"""Compare installed sidecar preparation with a baseline checkout, without CUDA.

Uses the same Python environment for both implementations, separate fresh child
processes, and an explicitly new work directory. Never downloads weights, starts
inference, changes an Ollama manifest, or removes an existing runtime cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CHILD = """
import contextlib, importlib.metadata, io, json, resource, sys, time
from pathlib import Path
from qwen_omni_adapters.ollama_sidecar import prepare_ollama_sidecar
options = json.loads(sys.argv[1])
options['output_dir'] = Path(options['output_dir'])
if options.get('models_dir'):
    options['models_dir'] = Path(options['models_dir'])
start = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    result = prepare_ollama_sidecar(**options)
elapsed = time.perf_counter() - start
print(json.dumps({'elapsed_s': elapsed,
    'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'implementation': prepare_ollama_sidecar.__code__.co_filename,
    'gguf_version': importlib.metadata.version('gguf'), 'result': result}))
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    if len(args.model) > 2:
        parser.error("at most two installed profiles per bounded benchmark")
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline_repo.expanduser().resolve()
    if not (baseline / "src/qwen_omni_adapters/ollama_sidecar.py").is_file():
        parser.error("baseline repo must contain the original sidecar implementation")
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=False)

    def run(repo, model, cache, *, reuse=False):
        options = {"model": model, "output_dir": str(cache), "overwrite": True}
        if args.models_dir:
            options["models_dir"] = str(args.models_dir.resolve())
        if reuse:
            options["reuse"] = True
        env = {**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps(options)], cwd=repo, env=env,
            capture_output=True, text=True, timeout=180, check=False,
        )
        if process.returncode:
            raise RuntimeError(f"preparation failed: {process.stderr[-4000:]}")
        return json.loads(process.stdout)

    report = {
        "schema": "robit.omni.component-view-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU artifact preparation only; no inference or GPU-residency measurement",
        "python": platform.python_version(), "platform": platform.system(),
        "architecture": platform.machine(), "trials_per_mode": args.trials,
        "baseline_revision": subprocess.check_output(
            ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "baseline_source_sha256": sha256(baseline / "src/qwen_omni_adapters/ollama_sidecar.py"),
        "upgraded_source_sha256": {str(path.relative_to(root)): sha256(path) for path in (
            root / "src/qwen_omni_adapters/ollama_sidecar.py",
            root / "src/qwen_omni_adapters/single_gguf.py",
            root / "src/qwen_omni_adapters/component_views.py",
        )},
        "work_dir": str(work), "profiles": [],
    }
    for number, model in enumerate(args.model):
        cache = work / f"profile-{number}"
        base_runs = [run(baseline, model, cache) for _ in range(args.trials)]
        original = base_runs[-1]["result"]
        expected = {view: sha256(Path(item["output"])) for view, item in original["views"].items()}
        size = sum(item["output_size_bytes"] for item in original["views"].values())
        prime = run(root, model, cache, reuse=True)
        assert prime["result"]["component_index"]["materialized_bytes"] == size
        files = [Path(item["output"]) for item in prime["result"]["views"].values()]
        identities = [(path.stat().st_ino, path.stat().st_mtime_ns) for path in files]
        warm_runs = [run(root, model, cache, reuse=True) for _ in range(args.trials)]
        assert identities == [(path.stat().st_ino, path.stat().st_mtime_ns) for path in files]
        assert all(item["result"]["component_index"]["materialized_bytes"] == 0 for item in warm_runs)
        assert expected == {view: sha256(Path(item["output"]))
                            for view, item in warm_runs[-1]["result"]["views"].items()}
        baseline_median = statistics.median(item["elapsed_s"] for item in base_runs)
        warm_median = statistics.median(item["elapsed_s"] for item in warm_runs)
        summary = {
            "model": model, "bundle_digest": original["bundle_digest"],
            "component_bytes": size, "baseline_materialized_bytes_per_run": size,
            "warm_materialized_bytes_per_run": 0, "exact_component_sha256": expected,
            "component_bytes_identical": True, "warm_component_identities_unchanged": True,
            "baseline_median_s": baseline_median, "warm_median_s": warm_median,
            "median_preparation_reduction_percent": (1 - warm_median / baseline_median) * 100,
            "baseline_runs": base_runs, "indexed_prime": prime, "warm_runs": warm_runs,
        }
        report["profiles"].append(summary)
        print(json.dumps({key: value for key, value in summary.items()
                          if key not in {"baseline_runs", "indexed_prime", "warm_runs"}}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
