"""Start llama.cpp with the largest context that safely fits right now.

The Jetson shares RAM between the CPU and GPU. A context size that is safe on
an otherwise idle machine can take the desktop down when another worker is
resident, so the systemd unit cannot sensibly bake in one ``-c`` value. This
launcher measures ``MemAvailable`` immediately before the model load, retains
an explicit reserve, publishes the selected window for the adapter, and then
replaces itself with llama-server.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

GIB_IN_KIB = 1024 * 1024
CONTEXT_QUANTUM = 4096


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


def estimated_resident_gib(
    context_tokens: int,
    *,
    base_gib: float = 16.3,
    kv_gib_per_4k: float = 0.4,
) -> float:
    """Conservative measured footprint for this Qwen3-Omni worker.

    llama-server allocates the configured window for each of its four default
    slots. On the target AGX Orin the whole process measured about 16.7 GiB at
    4K and 22.1 GiB at 64K, making 0.4 GiB per additional 4K a conservative
    aggregate slope. Both calibration values remain configurable.
    """

    quanta = max(1.0, context_tokens / CONTEXT_QUANTUM)
    return base_gib + quanta * kv_gib_per_4k


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
    reserve_gib: float = 3.0,
    base_gib: float = 16.3,
    kv_gib_per_4k: float = 0.4,
) -> int | None:
    """Choose the largest window whose model estimate leaves the reserve."""

    chosen: int | None = None
    for context_tokens in candidate_windows(minimum, maximum):
        needed = estimated_resident_gib(
            context_tokens,
            base_gib=base_gib,
            kv_gib_per_4k=kv_gib_per_4k,
        )
        if needed + reserve_gib <= available_gib:
            chosen = context_tokens
    return chosen


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _write_selected_context(path: Path, context_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{context_tokens}\n", encoding="utf-8")
    temporary.replace(path)


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
        "--reserve-gib",
        type=_nonnegative_float,
        default=float(os.environ.get("OMNI_COMPREHENSION_MEMORY_RESERVE_GIB", "3.0")),
    )
    parser.add_argument(
        "--base-gib",
        type=_nonnegative_float,
        default=float(os.environ.get("OMNI_COMPREHENSION_BASE_GIB", "16.3")),
    )
    parser.add_argument(
        "--kv-gib-per-4k",
        type=_nonnegative_float,
        default=float(os.environ.get("OMNI_COMPREHENSION_KV_GIB_PER_4K", "0.4")),
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

    available = available_memory_gib()
    selected = choose_context_tokens(
        available,
        minimum=args.min_context,
        maximum=args.max_context,
        reserve_gib=args.reserve_gib,
        base_gib=args.base_gib,
        kv_gib_per_4k=args.kv_gib_per_4k,
    )
    if selected is None:
        minimum_cost = estimated_resident_gib(
            args.min_context,
            base_gib=args.base_gib,
            kv_gib_per_4k=args.kv_gib_per_4k,
        )
        print(
            "refusing comprehension load: "
            f"{available:.2f} GiB available, but the {args.min_context}-token "
            f"worker needs about {minimum_cost:.2f} GiB plus the "
            f"{args.reserve_gib:.2f} GiB reserve",
            file=sys.stderr,
            flush=True,
        )
        return 75

    _write_selected_context(args.state_file, selected)
    estimated = estimated_resident_gib(
        selected,
        base_gib=args.base_gib,
        kv_gib_per_4k=args.kv_gib_per_4k,
    )
    print(
        f"selected {selected}-token comprehension context: "
        f"{available:.2f} GiB available, ~{estimated:.2f} GiB model, "
        f"{args.reserve_gib:.2f} GiB reserve",
        flush=True,
    )
    rendered = [part.replace("{context}", str(selected)) for part in command]
    os.execvpe(rendered[0], rendered, os.environ.copy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
