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
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from qwen_omni_adapters.accelerator import is_tegra
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPolicy

GIB_IN_KIB = 1024 * 1024
GIB_IN_BYTES = 1024**3

# Exact storage slopes for the KV element formats exposed by the pinned
# llama.cpp build. Quantized formats are block encoded, so treating every
# value as two bytes silently defeats cache-quantization admission: the
# launcher would reserve fp16-sized KV even when the worker uses q8/q4.
# Values include each block's scale/min metadata.
_CACHE_BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "iq4_nl": 18.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q5_1": 24.0 / 32.0,
}


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


def choose_context_tokens_with_recovery(
    available_gib: float,
    *,
    minimum: int,
    maximum: int,
    base_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int,
    startup_reserve_gib: float,
    recovery_reserve_gib: float,
) -> tuple[int | None, bool]:
    """Select normally, or recover at the shared governor's safe floor.

    Normal startup keeps the soft floor *plus* an inference-operation cushion.
    Immediately after TTS, allocator/page-cache lag can make that stricter sum
    miss the smallest safe window by a few hundred MiB. Only when no normal
    window fits, retry with the generic governor threshold, which still keeps
    the full soft/hard safety band and prevents a permanent reload deadlock.
    """

    common = {
        "minimum": minimum,
        "maximum": maximum,
        "base_gib": base_gib,
        "kv_gib_per_token": kv_gib_per_token,
        "parallel_slots": parallel_slots,
    }
    selected = choose_context_tokens(
        available_gib,
        runtime_reserve_gib=startup_reserve_gib,
        **common,
    )
    if selected is not None:
        return selected, False
    selected = choose_context_tokens(
        available_gib,
        runtime_reserve_gib=min(startup_reserve_gib, recovery_reserve_gib),
        **common,
    )
    return selected, selected is not None


def choose_calibrated_context(
    available_gib: float,
    *,
    minimum: int,
    maximum: int,
    live_base_gib: float,
    component_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int,
    startup_reserve_gib: float,
    recovery_reserve_gib: float,
    safe_context_tokens: int,
) -> tuple[int | None, bool, bool]:
    """Use measured residency and the conservative component-byte ceiling.

    Memory-mapped component bytes are deliberately conservative: page cache
    means they can overstate steady live residency. They remain useful for an
    unproven larger window, while successful live samples are authoritative.
    Select the largest unproven tier that the complete component bytes, KV
    slope, and runtime reserve can fund. A healthy compact model should not be
    stranded at 4K/8K for several service restarts merely because its first
    calibration began conservatively.
    """

    proven_maximum = min(maximum, max(minimum, safe_context_tokens))
    selected, recovery = choose_context_tokens_with_recovery(
        available_gib,
        minimum=minimum,
        maximum=proven_maximum,
        base_gib=live_base_gib,
        kv_gib_per_token=kv_gib_per_token,
        parallel_slots=parallel_slots,
        startup_reserve_gib=startup_reserve_gib,
        recovery_reserve_gib=recovery_reserve_gib,
    )
    probe = choose_context_tokens(
        available_gib,
        minimum=minimum,
        maximum=maximum,
        base_gib=component_gib,
        kv_gib_per_token=kv_gib_per_token,
        parallel_slots=parallel_slots,
        runtime_reserve_gib=startup_reserve_gib,
    )
    if probe is not None and probe > proven_maximum:
        return probe, False, True
    return selected, recovery, False


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


def _cache_type(command: list[str], flag: str) -> str:
    try:
        return _command_value(command, flag).lower()
    except ValueError:
        return "f16"


def _cache_contract(command: list[str]) -> dict[str, str]:
    return {
        "key": _cache_type(command, "--cache-type-k"),
        "value": _cache_type(command, "--cache-type-v"),
    }


def _cache_bytes(command: list[str], flag: str) -> float:
    """Bytes per cache element from llama.cpp's explicit or default type."""

    name = _cache_type(command, flag)
    try:
        return _CACHE_BYTES_PER_ELEMENT[name]
    except KeyError as error:
        raise ValueError(f"unsupported llama.cpp KV cache type: {name}") from error


def _kv_gib_per_token(model: Path, command: list[str]) -> float:
    """Derive KV growth directly from this GGUF's architecture metadata."""

    from gguf import GGUFReader

    reader = GGUFReader(model, mode="r")
    architecture_field = reader.fields["general.architecture"]
    architecture = bytes(architecture_field.parts[architecture_field.data[-1]].tolist()).decode(
        "utf-8"
    )
    prefix = f"{architecture}."
    layers = _gguf_scalar(reader, prefix + "block_count")
    heads = _gguf_scalar(reader, prefix + "attention.head_count_kv")
    key = _gguf_scalar(reader, prefix + "attention.key_length")
    value = _gguf_scalar(reader, prefix + "attention.value_length")
    per_token = (
        layers
        * heads
        * (
            key * _cache_bytes(command, "--cache-type-k")
            + value * _cache_bytes(command, "--cache-type-v")
        )
    )
    return per_token / GIB_IN_BYTES


def _load_calibration(
    path: Path, fingerprint: list[dict[str, Any]], command: list[str]
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    cache_contract = _cache_contract(command)
    if (
        payload.get("components") != fingerprint
        or payload.get("cache_types") != cache_contract
    ):
        payload = {}
    # Migrate calibration written by the old fixed-reserve admission logic.
    # Capacity is now recalculated from each live sample instead.
    payload.pop("reserve_gib", None)
    payload["kv_gib_per_token"] = _kv_gib_per_token(
        Path(_command_value(command, "-m")), command
    )
    payload["components"] = fingerprint
    payload["cache_types"] = cache_contract
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
    """Return robust measured residency instead of mapped component bytes."""

    persisted = calibration.get("base_gib")
    floor = (
        float(persisted) if isinstance(persisted, (int, float)) and float(persisted) > 0 else None
    )
    samples = calibration.get("base_samples")
    if isinstance(samples, list):
        valid = [
            float(value)
            for value in samples[-16:]
            if isinstance(value, (int, float)) and float(value) > 0
        ]
        if valid:
            return _median(valid)
    return floor


def _safe_context_tokens(
    calibration: Mapping[str, Any],
    *,
    minimum: int,
    maximum: int,
    runtime_reserve_gib: float,
    kv_gib_per_token: float,
    parallel_slots: int,
) -> int:
    """Read a proven tier, migrating a sufficiently healthy last sample."""

    safe = calibration.get("safe_context_tokens")
    if isinstance(safe, int) and safe >= minimum:
        return min(maximum, safe)
    sample = calibration.get("last_sample")
    if isinstance(sample, Mapping):
        context = sample.get("context_tokens")
        after = sample.get("available_after_gib")
        if isinstance(context, int) and isinstance(after, (int, float)):
            context = min(maximum, max(minimum, context))
            required = max(
                runtime_reserve_gib,
                context_headroom_gib(
                    context,
                    windows=candidate_windows(minimum, maximum),
                    kv_gib_per_token=kv_gib_per_token,
                    parallel_slots=parallel_slots,
                ),
            )
            if float(after) >= required:
                return context
    return minimum


def _record_live_sample(
    path: Path,
    calibration: dict[str, Any],
    *,
    before_gib: float,
    after_gib: float,
    context_tokens: int,
    parallel_slots: int,
    required_headroom_gib: float = 0.0,
) -> None:
    kv = float(calibration["kv_gib_per_token"])
    measured = max(0.0, before_gib - after_gib)
    sampled_base = max(0.0, measured - context_tokens * kv * parallel_slots)
    samples = calibration.get("base_samples")
    if not isinstance(samples, list) or not samples:
        samples = [float(calibration.get("base_gib") or sampled_base)]
    samples.append(sampled_base)
    del samples[:-16]
    calibration["base_gib"] = _median(samples)
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
    if after_gib >= required_headroom_gib:
        prior = calibration.get("safe_context_tokens")
        calibration["safe_context_tokens"] = max(
            context_tokens,
            prior if isinstance(prior, int) else 0,
        )
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

    lower = [window for window in candidate_windows(minimum, maximum) if window < context_tokens]
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
    if not isinstance(failed_context, int) or not isinstance(failed_available, (int, float)):
        return min(configured_maximum, cap)
    retry_cost = max(0, failed_context - cap) * kv_gib_per_token * max(1, parallel_slots)
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
    needed = component_gib + context_tokens * kv_gib_per_token * max(1, parallel_slots)
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


def _pressure_started_at(
    available_gib: float,
    required_gib: float,
    started_at: float | None,
    *,
    now: float,
) -> float | None:
    """Track continuous pressure while ignoring short inference transients."""

    if available_gib >= required_gib:
        return None
    return now if started_at is None else started_at


def _next_context_tier(selected: int, minimum: int, maximum: int) -> int | None:
    """Return the next configured tier above the live allocation."""

    return next(
        (
            window
            for window in candidate_windows(minimum, maximum)
            if window > selected
        ),
        None,
    )


def _expansion_required_gib(
    selected: int,
    target: int,
    *,
    minimum: int,
    maximum: int,
    kv_gib_per_token: float,
    parallel_slots: int,
    runtime_reserve_gib: float,
) -> float:
    """Charge incremental KV plus the target tier's complete live reserve."""

    growth = max(0, target - selected) * kv_gib_per_token * max(1, parallel_slots)
    target_headroom = context_headroom_gib(
        target,
        windows=candidate_windows(minimum, maximum),
        kv_gib_per_token=kv_gib_per_token,
        parallel_slots=parallel_slots,
    )
    return growth + max(runtime_reserve_gib, target_headroom)


def _expansion_backed_off(
    calibration: Mapping[str, Any], *, now: float, cooldown_s: float
) -> bool:
    """Prevent an idle/pressure resize oscillation after a failed larger tier."""

    failure = calibration.get("last_failure")
    if not isinstance(failure, Mapping):
        return False
    failed_at = failure.get("failed_at")
    return isinstance(failed_at, (int, float)) and now < float(failed_at) + cooldown_s


def _server_idle(health_url: str) -> bool:
    """Fail closed unless every llama.cpp slot reports that it is idle."""

    parsed = urlsplit(health_url)
    slots_url = urlunsplit((parsed.scheme, parsed.netloc, "/slots", "", ""))
    try:
        with urllib.request.urlopen(slots_url, timeout=1.0) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False
    return bool(payload) and isinstance(payload, list) and all(
        isinstance(slot, Mapping) and slot.get("is_processing") is False
        for slot in payload
    )


def _runtime_resize_ready(
    available_gib: float,
    *,
    hard_floor_gib: float,
    server_idle: bool,
) -> bool:
    """Resize between requests unless unified memory reached the emergency floor."""

    return server_idle or available_gib < hard_floor_gib


def _runtime_resize_enabled(*, tegra: bool, configured: str | None) -> bool:
    """Keep Tegra CUDA character-device ownership stable for one boot session."""

    normalized = str(configured or "").strip().lower()
    if not normalized:
        return not tegra
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "OMNI_COMPREHENSION_RUNTIME_RESIZE must be a boolean value"
    )


def _launcher_restart_command(argv: list[str]) -> list[str]:
    """Render an in-place launcher restart without involving the parent daemon."""

    return [sys.executable, str(Path(__file__).resolve()), *argv]


def _reexec_launcher(argv: list[str]) -> None:
    """Re-evaluate live capacity while retaining the supervised launcher PID."""

    command = _launcher_restart_command(argv)
    os.execvpe(command[0], command, os.environ.copy())
    raise RuntimeError("comprehension launcher re-exec unexpectedly returned")


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
        default=os.environ.get("OMNI_COMPREHENSION_HEALTH", "http://127.0.0.1:8901/health"),
    )
    parser.add_argument(
        "--child-pid-file",
        type=Path,
        help="publish the actual CUDA worker pid for supervisor residency checks",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a llama-server command is required after --")

    available = sampled_available_memory_gib()
    memory_policy = MemoryPolicy.from_environment()
    governor = MemoryGovernor(memory_policy)
    runtime_resize = _runtime_resize_enabled(
        tegra=is_tegra(),
        configured=os.environ.get("OMNI_COMPREHENSION_RUNTIME_RESIZE"),
    )
    runtime_reserve = governor.required_gib() if memory_policy.enabled else 0.0
    recovery_reserve = runtime_reserve
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
        safe_context = _safe_context_tokens(
            calibration,
            minimum=args.min_context,
            maximum=effective_maximum,
            runtime_reserve_gib=runtime_reserve,
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
        )
        selected, recovery_window, probing_window = choose_calibrated_context(
            available,
            minimum=args.min_context,
            maximum=effective_maximum,
            live_base_gib=float(base),
            component_gib=component_gib,
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
            startup_reserve_gib=runtime_reserve,
            recovery_reserve_gib=recovery_reserve,
            safe_context_tokens=safe_context,
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
        recovery_window = False
        probing_window = True
        # Component bytes overstate mapped steady residency but provide a safe
        # first-load base. Use all capacity that remains after model-derived KV
        # growth and the shared runtime reserve instead of hard-coding 4K.
        selected = choose_context_tokens(
            available,
            minimum=args.min_context,
            maximum=effective_maximum,
            base_gib=component_gib,
            kv_gib_per_token=kv,
            parallel_slots=args.parallel_slots,
            runtime_reserve_gib=runtime_reserve,
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

    admitted_reserve = recovery_reserve if recovery_window else runtime_reserve
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
            admitted_reserve,
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
            f"({required_headroom:.2f} GiB runtime minimum"
            f"{'; recovery window' if recovery_window else ''}"
            f"{'; probing next tier' if probing_window else ''})"
        )
    else:
        detail = f"first live calibration probe; {component_gib:.2f} GiB components"
    print(
        f"selected {selected}-token comprehension context: "
        f"{available:.2f} GiB available, {detail}, {args.parallel_slots} slot(s)",
        flush=True,
    )
    if not runtime_resize:
        print(
            "runtime comprehension process resizing disabled for this host; "
            "the startup-selected KV allocation remains fixed while prompt budgets "
            "continue to adapt within it",
            flush=True,
        )
    rendered = [
        part.replace("{context}", str(selected)).replace("{parallel}", str(args.parallel_slots))
        for part in command
    ]
    process = subprocess.Popen(rendered, env=os.environ.copy())
    if args.child_pid_file is not None:
        _write_selected_context(args.child_pid_file, process.pid)
    sampled = False
    pressure_downshift = False
    pressure_started_at: float | None = None
    runtime_pressure_deferred = False
    expansion_restart = False
    expansion_started_at: float | None = None
    runtime_required_headroom = 0.0
    pressure_grace_s = max(
        1.0,
        float(os.environ.get("OMNI_COMPREHENSION_PRESSURE_GRACE_SECONDS", "8")),
    )
    expansion_grace_s = max(
        10.0,
        float(os.environ.get("OMNI_COMPREHENSION_EXPANSION_GRACE_SECONDS", "60")),
    )
    expansion_cooldown_s = max(
        expansion_grace_s,
        float(os.environ.get("OMNI_COMPREHENSION_EXPANSION_COOLDOWN_SECONDS", "900")),
    )
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        if process.poll() is None:
            process.terminate()

    if os.name != "nt":
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    try:
        while process.poll() is None:
            if not sampled and _healthy(args.health_url):
                after = available_memory_gib()
                required_headroom = max(
                    admitted_reserve,
                    context_headroom_gib(
                        selected,
                        windows=candidate_windows(args.min_context, args.max_context),
                        kv_gib_per_token=kv,
                        parallel_slots=args.parallel_slots,
                    ),
                )
                runtime_required_headroom = required_headroom
                _record_live_sample(
                    args.calibration_file,
                    calibration,
                    before_gib=available,
                    after_gib=after,
                    context_tokens=selected,
                    parallel_slots=args.parallel_slots,
                    required_headroom_gib=required_headroom,
                )
                print(
                    f"calibrated {selected}-token comprehension residency from live "
                    f"memory: {available:.2f} -> {after:.2f} GiB available",
                    flush=True,
                )
                sampled = True
                if after < required_headroom:
                    _record_failed_context(
                        args.calibration_file,
                        calibration,
                        context_tokens=selected,
                        minimum=args.min_context,
                        maximum=args.max_context,
                        available_gib=available,
                    )
                    if runtime_resize:
                        print(
                            "controlled comprehension downshift: "
                            f"only {after:.2f} GiB remained after load, below the "
                            f"{required_headroom:.2f} GiB runtime reserve",
                            file=sys.stderr,
                            flush=True,
                        )
                        pressure_downshift = True
                        process.terminate()
                    else:
                        print(
                            "deferred comprehension downshift until the next supervised "
                            f"start: {after:.2f} GiB remained below the "
                            f"{required_headroom:.2f} GiB reserve",
                            file=sys.stderr,
                            flush=True,
                        )
            elif (
                sampled
                and runtime_required_headroom > 0
                and not runtime_pressure_deferred
            ):
                current_available = available_memory_gib()
                now = time.monotonic()
                pressure_started_at = _pressure_started_at(
                    current_available,
                    runtime_required_headroom,
                    pressure_started_at,
                    now=now,
                )
                if (
                    pressure_started_at is not None
                    and now - pressure_started_at >= pressure_grace_s
                    and (
                        not runtime_resize
                        or _runtime_resize_ready(
                            current_available,
                            hard_floor_gib=memory_policy.hard_floor_gib,
                            server_idle=_server_idle(args.health_url),
                        )
                    )
                ):
                    emergency = current_available < memory_policy.hard_floor_gib
                    _record_failed_context(
                        args.calibration_file,
                        calibration,
                        context_tokens=selected,
                        minimum=args.min_context,
                        maximum=args.max_context,
                        # The next-tier retry threshold is anchored to the
                        # pre-load capacity. Recording the transient low point
                        # would immediately reselect the unsafe tier after the
                        # old process released its pages.
                        available_gib=available,
                    )
                    if runtime_resize:
                        print(
                            "controlled comprehension downshift after sustained "
                            f"{'emergency' if emergency else 'idle'} runtime "
                            f"pressure: {current_available:.2f} GiB available remained "
                            f"below the {runtime_required_headroom:.2f} GiB reserve for "
                            f"{pressure_grace_s:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        pressure_downshift = True
                        process.terminate()
                    else:
                        print(
                            "recorded a lower comprehension tier for the next "
                            "supervised start without terminating the Tegra worker: "
                            f"{current_available:.2f} GiB available remained below the "
                            f"{runtime_required_headroom:.2f} GiB reserve for "
                            f"{pressure_grace_s:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        runtime_pressure_deferred = True
                        pressure_started_at = None
                elif pressure_started_at is None and runtime_resize:
                    target = _next_context_tier(
                        selected, args.min_context, args.max_context
                    )
                    expansion_required = (
                        _expansion_required_gib(
                            selected,
                            target,
                            minimum=args.min_context,
                            maximum=args.max_context,
                            kv_gib_per_token=kv,
                            parallel_slots=args.parallel_slots,
                            runtime_reserve_gib=admitted_reserve,
                        )
                        if target is not None
                        else float("inf")
                    )
                    expansion_eligible = (
                        target is not None
                        and current_available >= expansion_required
                        and not _expansion_backed_off(
                            calibration,
                            now=time.time(),
                            cooldown_s=expansion_cooldown_s,
                        )
                        and _server_idle(args.health_url)
                    )
                    if expansion_eligible:
                        if expansion_started_at is None:
                            expansion_started_at = now
                        elif now - expansion_started_at >= expansion_grace_s:
                            print(
                                "controlled comprehension expansion after sustained "
                                f"idle surplus: {current_available:.2f} GiB available "
                                f"funds {selected}->{target} tokens with "
                                f"{expansion_required:.2f} GiB required",
                                flush=True,
                            )
                            calibration.pop("context_cap", None)
                            calibration.pop("last_failure", None)
                            calibration["last_expansion"] = {
                                "from_context_tokens": selected,
                                "to_context_tokens": target,
                                "available_gib": current_available,
                                "expanded_at": time.time(),
                            }
                            _atomic_json(args.calibration_file, calibration)
                            expansion_restart = True
                            process.terminate()
                    else:
                        expansion_started_at = None
                else:
                    expansion_started_at = None
            time.sleep(0.5)
    finally:
        if args.child_pid_file is not None:
            args.child_pid_file.unlink(missing_ok=True)
    returncode = int(process.returncode or 0)
    if (
        returncode != 0
        and not pressure_downshift
        and not expansion_restart
        and not stopping
    ):
        _record_failed_context(
            args.calibration_file,
            calibration,
            context_tokens=selected,
            minimum=args.min_context,
            maximum=args.max_context,
            available_gib=available,
        )
    if stopping:
        return 0
    if pressure_downshift or expansion_restart:
        # A planned context resize is internal to the comprehension component.
        # Replacing this launcher process preserves the PID supervised by the
        # daemon, so the portal, TTS, point head, and indicator remain live while
        # the smaller/larger llama worker is selected from a fresh memory sample.
        if os.name != "nt":
            _reexec_launcher(raw_argv)
        return 75
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
