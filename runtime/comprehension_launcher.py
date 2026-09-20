"""Start llama.cpp with the largest context that safely fits right now.

The Jetson shares RAM between the CPU and GPU. A context size that is safe on
an otherwise idle machine can take the desktop down when another worker is
resident, so the systemd unit cannot sensibly bake in one ``-c`` value. This
launcher samples ``MemAvailable`` immediately before each model load, combines
it with the observed resident footprint and the KV-cache slope derived from
the installed GGUF, and publishes the largest standard window that fits while
retaining one model-derived context-tier increment as live headroom. No
machine-specific footprint or fixed reserve is baked into the launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qwen_omni_adapters.memory import MemoryPolicy

GIB_IN_KIB = 1024 * 1024
GIB_IN_BYTES = 1024**3


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    """Return Linux ``MemAvailable`` in GiB, or zero when it is unavailable."""

    try:
        lines = meminfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0
    for line in lines:
        key, _, value = line.partition(":")
        if key == "MemAvailable":
            try:
                return float(value.strip().split()[0]) / GIB_IN_KIB
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def sampled_available_memory_gib() -> float:
    """Use the low point of a short live sample, including current jitter."""

    readings: list[float] = []
    for _ in range(8):
        readings.append(available_memory_gib())
        time.sleep(0.05)
    return min(readings, default=0.0)


def estimated_resident_gib(
    context_tokens: int,
    *,
    base_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int = 1,
) -> float:
    """Footprint from a live-sampled base plus the model-derived KV slope."""

    return base_gib + context_tokens * kv_gib_per_token * max(1, parallel_slots)


def candidate_windows(minimum: int, maximum: int) -> list[int]:
    """Return power-of-two windows plus an explicit non-power-of-two cap."""

    if minimum < 1024 or maximum < minimum:
        raise ValueError("context bounds must satisfy 1024 <= minimum <= maximum")
    windows: list[int] = []
    value = minimum
    while value <= maximum:
        windows.append(value)
        value *= 2
    if windows[-1] != maximum:
        windows.append(maximum)
    return sorted(set(windows))


def choose_context_tokens(
    available_gib: float,
    *,
    minimum: int = 4096,
    maximum: int = 65_536,
    base_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int = 1,
    runtime_reserve_gib: float = 0.0,
) -> int | None:
    """Choose the largest window that retains generic runtime headroom."""

    chosen: int | None = None
    windows = candidate_windows(minimum, maximum)
    for context_tokens in windows:
        needed = estimated_resident_gib(
            context_tokens,
            base_gib=base_gib,
            kv_gib_per_token=kv_gib_per_token,
            parallel_slots=parallel_slots,
        )
        headroom = context_headroom_gib(
            context_tokens,
            windows=windows,
            kv_gib_per_token=kv_gib_per_token,
            parallel_slots=parallel_slots,
        )
        if needed + max(headroom, runtime_reserve_gib) <= available_gib:
            chosen = context_tokens
    return chosen


def context_headroom_gib(
    context_tokens: int,
    *,
    windows: list[int],
    kv_gib_per_token: float,
    parallel_slots: int = 1,
) -> float:
    """Reserve one adjacent context tier's KV growth, derived from this model.

    ``MemAvailable`` is not stationary: the desktop, audio server, cameras,
    allocator scratch, and CUDA bookkeeping all move after admission. Spending
    the final byte merely because the weights mathematically fit is what made
    a large window start successfully and later starve the host. The adjacent
    tier is already derived from the live GGUF rather than a device-specific
    magic GiB number.
    """

    ordered = sorted(set(windows))
    try:
        index = ordered.index(context_tokens)
    except ValueError as error:
        raise ValueError("context_tokens must be present in windows") from error
    if len(ordered) == 1:
        delta = context_tokens
    elif index + 1 < len(ordered):
        delta = ordered[index + 1] - context_tokens
    else:
        delta = context_tokens - ordered[index - 1]
    return max(1, delta) * kv_gib_per_token * max(1, parallel_slots)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _write_selected_context(path: Path, context_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{context_tokens}\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _command_value(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"llama-server command is missing {flag}") from error


def _component_fingerprint(command: list[str]) -> tuple[list[dict[str, Any]], float]:
    components: list[dict[str, Any]] = []
    total_bytes = 0
    for flag in ("-m", "--mmproj"):
        path = Path(_command_value(command, flag)).resolve()
        stat = path.stat()
        components.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
        total_bytes += stat.st_size
    return components, total_bytes / GIB_IN_BYTES


def _gguf_scalar(reader: Any, name: str) -> int:
    field = reader.fields[name]
    value = field.parts[field.data[-1]].tolist()
    if isinstance(value, list):
        value = value[-1]
    return int(value)


def _cache_bytes(command: list[str], flag: str) -> int:
    """Bytes per cache element from llama.cpp's explicit or default type."""

    try:
        name = _command_value(command, flag).lower()
    except ValueError:
        name = "f16"
    if name == "f32":
        return 4
    # Quantized cache formats occupy no more than fp16; using two bytes keeps
    # admission conservative without embedding a device-specific estimate.
    return 2


def _kv_gib_per_token(model: Path, command: list[str]) -> float:
    """Derive KV growth directly from this GGUF's architecture metadata."""

    from gguf import GGUFReader

    reader = GGUFReader(model, mode="r")
    architecture_field = reader.fields["general.architecture"]
    architecture = bytes(
        architecture_field.parts[architecture_field.data[-1]].tolist()
    ).decode("utf-8")
    prefix = f"{architecture}."
    layers = _gguf_scalar(reader, prefix + "block_count")
    heads = _gguf_scalar(reader, prefix + "attention.head_count_kv")
    key = _gguf_scalar(reader, prefix + "attention.key_length")
    value = _gguf_scalar(reader, prefix + "attention.value_length")
    per_token = layers * heads * (
        key * _cache_bytes(command, "--cache-type-k")
        + value * _cache_bytes(command, "--cache-type-v")
    )
    return per_token / GIB_IN_BYTES


def _load_calibration(
    path: Path, fingerprint: list[dict[str, Any]], command: list[str]
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if payload.get("components") != fingerprint:
        payload = {}
    # Migrate calibration written by the old fixed-reserve admission logic.
    # Capacity is now recalculated from each live sample instead.
    payload.pop("reserve_gib", None)
    if "kv_gib_per_token" not in payload:
        payload["kv_gib_per_token"] = _kv_gib_per_token(
            Path(_command_value(command, "-m")), command
        )
    payload["components"] = fingerprint
    return payload


def _last_context(path: Path, minimum: int, maximum: int) -> int:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return minimum
    return min(maximum, max(minimum, value))


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _live_calibrated_base(calibration: Mapping[str, Any]) -> float | None:
    """Prefer repeated live residency samples over the on-disk byte floor.

    GGUF files are memory mapped, so their total file size is a conservative
    first-load bound rather than the resident footprint. Once successful live
    loads exist, using that byte floor can deadlock restoration on a few MiB
    rounding edge even though the same model repeatedly leaves the required
    runtime reserve intact.
    """

    samples = calibration.get("base_samples")
    if isinstance(samples, list):
        valid = [
            float(value)
            for value in samples[-16:]
            if isinstance(value, (int, float)) and float(value) > 0
        ]
        if valid:
            return _median(valid)
    base = calibration.get("base_gib")
    return float(base) if isinstance(base, (int, float)) and base > 0 else None


def _record_live_sample(
    path: Path,
    calibration: dict[str, Any],
    *,
    before_gib: float,
    after_gib: float,
    context_tokens: int,
    parallel_slots: int,
    floor_gib: float,
) -> None:
    kv = float(calibration["kv_gib_per_token"])
    measured = max(0.0, before_gib - after_gib)
    sampled_base = max(0.0, measured - context_tokens * kv * parallel_slots)
    samples = calibration.get("base_samples")
    if not isinstance(samples, list) or not samples:
        samples = [float(calibration.get("base_gib") or sampled_base)]
    samples.append(sampled_base)
    del samples[:-16]
    calibration["base_gib"] = max(floor_gib, _median(samples))
    calibration["base_samples"] = samples
    calibration["last_sample"] = {
        "available_before_gib": before_gib,
        "available_after_gib": after_gib,
        "context_tokens": context_tokens,
        "parallel_slots": parallel_slots,
        "measured_resident_gib": measured,
        "sampled_base_gib": sampled_base,
        "sampled_at": time.time(),
    }
    calibration["samples"] = int(calibration.get("samples") or 0) + 1
    _atomic_json(path, calibration)


def _record_failed_context(
    path: Path,
    calibration: dict[str, Any],
    *,
    context_tokens: int,
    minimum: int,
    maximum: int,
    available_gib: float,
) -> None:
    """Step down after an abnormal child exit instead of crash-looping."""

    lower = [
        window
        for window in candidate_windows(minimum, maximum)
        if window < context_tokens
    ]
    if lower:
        calibration["context_cap"] = lower[-1]
    if context_tokens == minimum:
        calibration["probe_backoff_until"] = time.time() + float(
            os.environ.get("OMNI_COMPREHENSION_PROBE_COOLDOWN", "120")
        )
    calibration["last_failure"] = {
        "available_before_gib": available_gib,
        "context_tokens": context_tokens,
        "failed_at": time.time(),
    }
    _atomic_json(path, calibration)


def _effective_context_maximum(
    calibration: dict[str, Any],
    *,
    configured_maximum: int,
    available_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int,
) -> int:
    """Honor crash backoff until live capacity can fund retrying that tier."""

    cap = calibration.get("context_cap")
    failure = calibration.get("last_failure")
    if not isinstance(cap, int) or not isinstance(failure, Mapping):
        return configured_maximum
    failed_context = failure.get("context_tokens")
    failed_available = failure.get("available_before_gib")
    if not isinstance(failed_context, int) or not isinstance(
        failed_available, (int, float)
    ):
        return min(configured_maximum, cap)
    retry_cost = max(0, failed_context - cap) * kv_gib_per_token * max(
        1, parallel_slots
    )
    if available_gib >= float(failed_available) + retry_cost:
        calibration.pop("context_cap", None)
        calibration.pop("last_failure", None)
        return configured_maximum
    return min(configured_maximum, cap)


def _component_window_fits(
    *,
    component_gib: float,
    context_tokens: int,
    minimum: int,
    maximum: int,
    kv_gib_per_token: float,
    parallel_slots: int,
    available_gib: float,
    runtime_reserve_gib: float = 0.0,
) -> bool:
    """Admission against component bytes alone, retaining one tier of KV."""

    windows = candidate_windows(minimum, maximum)
    needed = component_gib + context_tokens * kv_gib_per_token * max(
        1, parallel_slots
    )
    headroom = context_headroom_gib(
        context_tokens,
        windows=windows,
        kv_gib_per_token=kv_gib_per_token,
        parallel_slots=parallel_slots,
    )
    return needed + max(headroom, runtime_reserve_gib) <= available_gib


def _probe_backed_off(calibration: Mapping[str, Any]) -> bool:
    until = calibration.get("probe_backoff_until")
    return isinstance(until, (int, float)) and time.time() < until


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(
            os.environ.get(
                "OMNI_COMPREHENSION_CONTEXT_FILE",
                f"/run/user/{os.getuid()}/egg-omni-comprehension-context",
            )
        ),
    )
    parser.add_argument(
        "--min-context",
        type=_positive_int,
        default=int(os.environ.get("OMNI_COMPREHENSION_MIN_CONTEXT_TOKENS", "4096")),
    )
    parser.add_argument(
        "--max-context",
        type=_positive_int,
        default=int(os.environ.get("OMNI_COMPREHENSION_CONTEXT_TOKENS", "65536")),
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=Path(
            os.environ.get(
                "OMNI_COMPREHENSION_CALIBRATION_FILE",
                "runtime-data/state/comprehension-memory.json",
            )
        ),
    )
    parser.add_argument(
        "--parallel-slots",
        type=_positive_int,
        default=int(os.environ.get("OMNI_COMPREHENSION_PARALLEL", "1")),
    )
    parser.add_argument(
        "--health-url",
        default=os.environ.get(
            "OMNI_COMPREHENSION_HEALTH", "http://127.0.0.1:8901/health"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a llama-server command is required after --")

    available = sampled_available_memory_gib()
    memory_policy = MemoryPolicy.from_environment()
    runtime_reserve = (
        memory_policy.soft_floor_gib + memory_policy.operation_reserve_gib
        if memory_policy.enabled
        else 0.0
    )
    fingerprint, component_gib = _component_fingerprint(command)
    calibration = _load_calibration(args.calibration_file, fingerprint, command)
    base = _live_calibrated_base(calibration)
    kv = float(calibration["kv_gib_per_token"])
    effective_maximum = _effective_context_maximum(
        calibration,
        configured_maximum=args.max_context,
        available_gib=available,
        kv_gib_per_token=kv,
        parallel_slots=args.parallel_slots,
    )
    if isinstance(base, (int, float)):
        selected = choose_context_tokens(
            available,
            minimum=args.min_context,
            maximum=effective_maximum,
            base_gib=float(base),
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
            runtime_reserve_gib=runtime_reserve,
        )
        if selected is None and not _probe_backed_off(calibration):
            # A stale calibrated baseline can block every window even though
            # the installed components fit. Re-probe at the minimum window so
            # residency can be remeasured from a live load instead of refusing.
            selected = (
                args.min_context
                if _component_window_fits(
                    component_gib=component_gib,
                    context_tokens=args.min_context,
                    minimum=args.min_context,
                    maximum=args.max_context,
                    kv_gib_per_token=kv,
                    parallel_slots=args.parallel_slots,
                    available_gib=available,
                    runtime_reserve_gib=runtime_reserve,
                )
                else None
            )
    else:
        # The first successful load is the calibration probe. Reuse the last
        # installed components for a conservative minimum-window probe. The
        # probe also retains one minimum-tier KV increment; component bytes
        # fitting by themselves says nothing about cache or runtime headroom.
        probe = args.min_context
        selected = (
            probe
            if _component_window_fits(
                component_gib=component_gib,
                context_tokens=probe,
                minimum=args.min_context,
                maximum=args.max_context,
                kv_gib_per_token=kv,
                parallel_slots=args.parallel_slots,
                available_gib=available,
                runtime_reserve_gib=runtime_reserve,
            )
            else None
        )
    if selected is None:
        print(
            "refusing comprehension load: "
            f"{available:.2f} GiB available; live calibration leaves no safe "
            f"window between {args.min_context} and {args.max_context} tokens",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(float(os.environ.get("OMNI_COMPREHENSION_RETRY_SECONDS", "15")))
        return 75

    _write_selected_context(args.state_file, selected)
    if isinstance(base, (int, float)):
        estimated = estimated_resident_gib(
            selected,
            base_gib=float(base),
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
        )
        headroom = max(0.0, available - estimated)
        required_headroom = max(
            runtime_reserve,
            context_headroom_gib(
            selected,
            windows=candidate_windows(args.min_context, args.max_context),
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
            ),
        )
        detail = (
            f"~{estimated:.2f} GiB from live calibration, "
            f"{headroom:.2f} GiB live headroom "
            f"({required_headroom:.2f} GiB runtime minimum)"
        )
    else:
        detail = f"first live calibration probe; {component_gib:.2f} GiB components"
    print(
        f"selected {selected}-token comprehension context: "
        f"{available:.2f} GiB available, {detail}, {args.parallel_slots} slot(s)",
        flush=True,
    )
    rendered = [
        part.replace("{context}", str(selected)).replace(
            "{parallel}", str(args.parallel_slots)
        )
        for part in command
    ]
    process = subprocess.Popen(rendered, env=os.environ.copy())
    sampled = False
    pressure_downshift = False
    while process.poll() is None:
        if not sampled and _healthy(args.health_url):
            after = available_memory_gib()
            _record_live_sample(
                args.calibration_file,
                calibration,
                before_gib=available,
                after_gib=after,
                context_tokens=selected,
                parallel_slots=args.parallel_slots,
                floor_gib=component_gib,
            )
            print(
                f"calibrated {selected}-token comprehension residency from live "
                f"memory: {available:.2f} -> {after:.2f} GiB available",
                flush=True,
            )
            sampled = True
            required_headroom = max(
                runtime_reserve,
                context_headroom_gib(
                    selected,
                    windows=candidate_windows(args.min_context, args.max_context),
                    kv_gib_per_token=kv,
                    parallel_slots=args.parallel_slots,
                ),
            )
            if after < required_headroom:
                print(
                    "controlled comprehension downshift: "
                    f"only {after:.2f} GiB remained after load, below the "
                    f"{required_headroom:.2f} GiB runtime reserve",
                    file=sys.stderr,
                    flush=True,
                )
                _record_failed_context(
                    args.calibration_file,
                    calibration,
                    context_tokens=selected,
                    minimum=args.min_context,
                    maximum=args.max_context,
                    available_gib=available,
                )
                pressure_downshift = True
                process.terminate()
        time.sleep(0.5)
    returncode = int(process.returncode or 0)
    if returncode != 0 and not pressure_downshift:
        _record_failed_context(
            args.calibration_file,
            calibration,
            context_tokens=selected,
            minimum=args.min_context,
            maximum=args.max_context,
            available_gib=available,
        )
    return 75 if pressure_downshift else returncode


if __name__ == "__main__":
    raise SystemExit(main())
