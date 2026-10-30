"""Bounded A/B/A comparison using the same bundled TTS checkpoint and references.

On discrete CUDA, run through docker gpu run with a scoped budget. On Tegra,
require an explicit available-memory floor; residency is proven by the runtime's
integrated-device checks. The readiness file is written
only after actual residency and the native ready frame, and every worker is
closed before returning. No public service is started or restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.accelerator import accelerator_profile, is_tegra, process_is_gpu_resident
from runtime.tts_server import Config, PersistentTTSWorker, _pcm_wav, _synthesis_spec
from runtime.verify_pcm_stream import analyze_pcm

PROMPTS = (
    "The blue bicycle is parked beside the garden gate.",
    "Seven small birds are sitting on the wooden fence.",
    "The blue bicycle is parked beside the garden gate.",
)


def ensure_with_matching_libraries(worker: PersistentTTSWorker, spec,
                                   extra_directories: list[Path], context_size: int) -> dict[str, str]:
    """A copied ELF must use its matching libraries rather than build RUNPATH."""
    directory = worker.config.binary.parent.resolve()
    previous = os.environ.get("LD_LIBRARY_PATH")
    previous_context = os.environ.get("LLAMA_ARG_CTX_SIZE")
    try:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, [directory, *extra_directories]))
        if context_size:
            os.environ["LLAMA_ARG_CTX_SIZE"] = str(context_size)
        worker.ensure(spec)
    finally:
        if previous is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous
        if previous_context is None:
            os.environ.pop("LLAMA_ARG_CTX_SIZE", None)
        else:
            os.environ["LLAMA_ARG_CTX_SIZE"] = previous_context
    paths = {}
    for line in Path(f"/proc/{worker.pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if path.name.startswith(("libllama.so", "libllama-common.so", "libmtmd.so")):
            assert path.parent.resolve() == directory, f"wrong native library for comparison: {path}"
            paths[path.name] = str(path)
        elif path.name.startswith("libggml"):
            assert path.parent.resolve() in {directory, *extra_directories}, \
                f"unrecorded GGML library for comparison: {path}"
            paths[path.name] = str(path)
    assert any(name.startswith("libllama.so") for name in paths)
    assert any(name.startswith("libmtmd.so") for name in paths)
    return paths


def available_memory_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable is required for this benchmark")


@contextmanager
def memory_guard(worker: PersistentTTSWorker, floor_mib: int):
    """Observe system headroom and terminate only this benchmark's child if low."""
    state = {"minimum_available_mib": available_memory_mib(), "floor_mib": floor_mib,
             "floor_crossed": False, "sampling_interval_s": .1}
    stopped = threading.Event()

    def observe():
        while not stopped.is_set():
            available = available_memory_mib()
            state["minimum_available_mib"] = min(state["minimum_available_mib"], available)
            if floor_mib and available < floor_mib:
                state["floor_crossed"] = True
                process = worker.process
                if process is not None and process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            stopped.wait(.1)

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    try:
        yield state
        assert not state["floor_crossed"], "benchmark crossed the available-memory floor"
    finally:
        stopped.set()
        thread.join(timeout=1)
        print(json.dumps({"memory_guard": state}), flush=True)


def counter_log(worker: PersistentTTSWorker) -> str:
    # pread does not move the descriptor shared with the native stderr writer.
    handle = worker.stderr_log
    size = os.fstat(handle.fileno()).st_size
    return os.pread(handle.fileno(), min(size, 8192), max(0, size - 8192)).decode(errors="replace")


def wait_for_broker_ready(token: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        result = subprocess.run(["docker", "gpu", "status"], capture_output=True,
                                text=True, check=True, timeout=10)
        leases = json.loads(result.stdout).get("leases", [])
        if any(lease.get("token") == token and lease.get("state") == "active" for lease in leases):
            return
        time.sleep(.1)
    raise RuntimeError("foreground lease did not become broker-ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--baseline-keep-cache", action="store_true")
    parser.add_argument("--expected-full-table-loads", type=int, choices=(0, 1), default=0)
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--minimum-available-mib", type=int, default=0)
    parser.add_argument("--context-size", type=int, default=0,
                        help="matched native context; zero preserves the native/environment default")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    tegra = is_tegra()
    if tegra and (args.minimum_available_mib <= 0 or uuid or token):
        parser.error("Tegra requires a positive memory floor and no discrete GPU scope")
    if not tegra and (not uuid.startswith("GPU-") or "," in uuid or not token):
        parser.error("use one exact UUID through the scoped GPU run wrapper")
    if args.minimum_available_mib < 0:
        parser.error("memory floor must be nonnegative")
    if args.context_size < 0:
        parser.error("context size must be nonnegative")
    extra_directories = [path.resolve(strict=True) for path in args.extra_library_dir]
    if len(args.reference) > 2:
        parser.error("at most two references per bounded comparison")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    args.ready_file.unlink(missing_ok=True)
    config = Config(binary=args.binary.resolve(), model=args.model.resolve(),
                    projector=args.projector.resolve(), gpu_uuid=uuid, require_gpu=True,
                    timeout_s=120, residency_timeout_s=90, max_frames=96, stream_frames=2)
    report = {"schema": "robit.omni.tts-conditioning-benchmark.v1", "gpu_uuid": uuid,
              "accelerator": accelerator_profile(),
              "context_size": args.context_size or os.environ.get("LLAMA_ARG_CTX_SIZE", "native-default"),
              "scope": "same checkpoint, baseline/candidate seeded PCM equality",
              "baseline_binary": str((args.baseline_binary or args.binary).resolve()),
              "candidate_binary": str(args.binary.resolve()),
              "baseline_conditioning_cache_enabled": args.baseline_keep_cache,
              "samples": [], "prompts": list(PROMPTS)}
    admitted = False
    for reference in args.reference:
        reference = reference.resolve()
        for enabled in (False, True):
            active_binary = args.binary if enabled else (args.baseline_binary or args.binary)
            worker = PersistentTTSWorker(replace(config, binary=active_binary.resolve(),
                conditioning_cache=enabled or args.baseline_keep_cache))
            pcm_results = []
            samples = []
            guard = None
            try:
                guard = memory_guard(worker, args.minimum_available_mib)
                memory = guard.__enter__()
                assert memory["minimum_available_mib"] >= args.minimum_available_mib
                first = _synthesis_spec(config, {"text": PROMPTS[0], "speaker_file": str(reference),
                                                "seed": 42, "stream_frames": 2})
                load_started = time.monotonic()
                libraries = ensure_with_matching_libraries(worker, first, extra_directories, args.context_size)
                load_seconds = time.monotonic() - load_started
                if not admitted:
                    args.ready_file.touch()
                    if not tegra:
                        wait_for_broker_ready(token)
                    admitted = True
                pid = worker.pid
                assert process_is_gpu_resident(pid, uuid), "worker lost GPU residency"
                for index, text in enumerate(PROMPTS):
                    spec = replace(first, text=text)
                    started = time.monotonic()
                    first_pcm = None
                    pcm = bytearray()
                    for chunk in worker.stream(spec):
                        if first_pcm is None:
                            first_pcm = time.monotonic() - started
                        pcm.extend(chunk)
                    elapsed = time.monotonic() - started
                    assert worker.pid == pid and pcm and len(pcm) % 2 == 0
                    raw = bytes(pcm)
                    pcm_results.append(raw)
                    filename = f"{reference.stem}-{int(enabled)}-{index}"
                    (work / (filename + ".pcm")).write_bytes(raw)
                    (work / (filename + ".wav")).write_bytes(_pcm_wav(raw))
                    time.sleep(.02)  # allow redacted native counter logs to flush after D
                    counters = re.findall(r"token_tables=(\d+) speaker_encodes=(\d+) speaker_hits=(\d+)",
                                          counter_log(worker))
                    rows = re.findall(r"reads=(\d+) hits=(\d+) cached_bytes=(\d+)", counter_log(worker))
                    hwm = next((line.split()[1] for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                                if line.startswith("VmHWM:")), None)
                    sample = {"index": index, "first_pcm_s": first_pcm, "total_s": elapsed,
                              "pcm_bytes": len(raw), "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                              "audio_s": len(raw) / 48000, "worker_pid": pid,
                              "cpu_peak_rss_kib": int(hwm) if hwm else None,
                              "conditioning_counters": list(map(int, counters[-1])) if counters else None,
                              "row_counters": list(map(int, rows[-1])) if rows else None,
                              "pcm_continuity": analyze_pcm(raw, stream_frames=2)}
                    assert sample["pcm_continuity"]["ok"], "raw PCM continuity gate failed"
                    samples.append(sample)
                assert pcm_results[0] == pcm_results[2], "A/B/A retained decoded request state"
                report["samples"].append({"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                    "reference_name": reference.name, "cache_enabled": enabled, "load_s": load_seconds,
                    "native_libraries": libraries,
                    "gpu_residency_verified": True, "memory_headroom": memory,
                    "a_b_a_identical": True, "cold_request_first_pcm_s": load_seconds + samples[0]["first_pcm_s"],
                    "runs": samples})
                print(json.dumps({"reference": reference.name, "cache": enabled,
                                  "first_pcm_s": [x["first_pcm_s"] for x in samples]}), flush=True)
            finally:
                worker.close()
                if guard is not None:
                    guard.__exit__(*sys.exc_info())
    for offset in range(0, len(report["samples"]), 2):
        baseline, cached = report["samples"][offset:offset + 2]
        assert [x["pcm_sha256"] for x in baseline["runs"]] == [x["pcm_sha256"] for x in cached["runs"]], \
            "conditioning cache changed synthesized PCM"
        assert cached["runs"][-1]["conditioning_counters"] == [args.expected_full_table_loads, 1, 2]
        if args.expected_full_table_loads == 0:
            assert cached["runs"][-1]["row_counters"][2] <= 8 * 1024 * 1024
        before = statistics.median(x["first_pcm_s"] for x in baseline["runs"][1:])
        after = statistics.median(x["first_pcm_s"] for x in cached["runs"][1:])
        cached["warm_first_pcm_reduction_percent"] = (1 - after / before) * 100
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
