"""Safe live-runtime inventory and Jetson admission checks for deployment.

The guided deployer uses this module before it changes a service.  Listener
ownership comes from procfs so Jetson does not need ``ss`` or ``lsof``.  GPU
load and unified-memory facts come from :mod:`qwen_omni_adapters.accelerator`;
the Tegra path deliberately never treats ``nvidia-smi`` as an authority.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qwen_omni_adapters.accelerator import (
    accelerator_profile,
    gpu_facts,
    is_tegra,
    process_is_gpu_resident,
)
from qwen_omni_adapters.ollama_sidecar import resolve_ollama_sidecar

DEFAULT_PORTS = (8892, 8901, 8910, 8920, 8930)
_LISTEN_STATE = "0A"
_SOCKET_PATTERN = re.compile(r"^socket:\[(\d+)]$")
_SERVICE_PATTERN = re.compile(r"(?:^|/)([^/]+\.service)(?:/|$)")
_OMNI_COMMAND_MARKERS = (
    "qwen-omni-daemon",
    "qwen_omni_adapters",
    "qwen-omni-adapters",
    "runtime/adapter_server.py",
    "runtime/tts_server.py",
    "runtime/comprehension_launcher.py",
    "portal/app.py",
)


def _read_text(path: Path, limit: int = 32_768) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _parse_ports(value: str) -> tuple[int, ...]:
    ports = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not ports or any(port < 1 or port > 65_535 for port in ports):
        raise argparse.ArgumentTypeError("ports must be comma-separated values from 1 to 65535")
    return ports


def _listener_inodes(proc_root: Path, ports: Iterable[int]) -> dict[str, set[int]]:
    selected = set(ports)
    listeners: dict[str, set[int]] = {}
    for name in ("tcp", "tcp6"):
        text = _read_text(proc_root / "net" / name, limit=8 * 1024 * 1024)
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != _LISTEN_STATE:
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if port in selected:
                listeners.setdefault(fields[9], set()).add(port)
    return listeners


def _service_identity(cgroup: str) -> tuple[str, str]:
    units = _SERVICE_PATTERN.findall(cgroup)
    unit = next((item for item in reversed(units) if "omni" in item.lower()), "")
    if not unit:
        return "", ""
    scope = "user-unit" if "/user.slice/" in cgroup else "system-unit"
    return scope, unit


def _recognized_owner(command: str, unit: str) -> bool:
    lowered = command.lower()
    return "omni" in unit.lower() or any(marker in lowered for marker in _OMNI_COMMAND_MARKERS)


def listener_owners(
    ports: Iterable[int] = DEFAULT_PORTS,
    *,
    proc_root: Path = Path("/proc"),
    include_gpu_residency: bool = True,
) -> list[dict[str, Any]]:
    """Return processes owning listening sockets on the deployment ports.

    A listening inode that cannot be mapped through ``/proc/<pid>/fd`` is
    returned as an unresolved owner.  Deployment treats that as unsafe rather
    than assuming the port is free.
    """

    listeners = _listener_inodes(proc_root, ports)
    if not listeners:
        return []
    owners: list[dict[str, Any]] = []
    matched: set[str] = set()
    try:
        processes = sorted(
            (path for path in proc_root.iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except OSError:
        processes = []
    tegra = include_gpu_residency and proc_root == Path("/proc") and is_tegra()
    for process in processes:
        found: set[str] = set()
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            descriptors = []
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = _SOCKET_PATTERN.match(target)
            if match and match.group(1) in listeners:
                found.add(match.group(1))
        if not found:
            continue
        matched.update(found)
        pid = int(process.name)
        command = _read_text(process / "cmdline").replace("\0", " ").strip()
        cgroup = _read_text(process / "cgroup")
        scope, unit = _service_identity(cgroup)
        try:
            parent_pid = int(_read_text(process / "status").split("PPid:\t", 1)[1].splitlines()[0])
        except (IndexError, ValueError):
            parent_pid = 0
        owners.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "ports": sorted({port for inode in found for port in listeners[inode]}),
                "command": command[:1000],
                "service_scope": scope,
                "service_unit": unit,
                "recognized_omni": _recognized_owner(command, unit),
                "gpu_resident": process_is_gpu_resident(pid) if tegra else None,
            }
        )
    for inode in sorted(set(listeners) - matched):
        owners.append(
            {
                "pid": None,
                "parent_pid": None,
                "ports": sorted(listeners[inode]),
                "command": "",
                "service_scope": "",
                "service_unit": "",
                "recognized_omni": False,
                "gpu_resident": None,
                "unresolved_socket_inode": inode,
            }
        )
    return owners


def handoff_targets(owners: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Collapse recognized listener owners to exact units or process IDs."""

    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for owner in owners:
        if not owner.get("recognized_omni"):
            raise RuntimeError(
                "deployment port has an unknown or unreadable owner: "
                f"ports={owner.get('ports')} pid={owner.get('pid')} "
                f"command={owner.get('command')!r}"
            )
        scope = str(owner.get("service_scope") or "")
        unit = str(owner.get("service_unit") or "")
        target = (
            (scope, unit) if scope and unit else ("process", str(int(owner["pid"])))
        )
        if target not in seen:
            seen.add(target)
            result.append(target)
    return result


def _mem_available_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int | None:
    for line in _read_text(meminfo_path).splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            return int(fields[1]) * 1024
    return None


def artifact_bytes(model: str) -> tuple[int, list[dict[str, Any]]]:
    """Return unique installed layer bytes for one logical bridge tag."""

    resolved = resolve_ollama_sidecar(model=model)
    layers = [resolved["layer"]]
    layers.extend(
        entry["layer"] for entry in (resolved.get("standard_layers") or {}).values()
    )
    unique: dict[str, dict[str, Any]] = {}
    for layer in layers:
        digest = str(layer.get("digest") or "")
        size = layer.get("size")
        if not digest or not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"model layer has no valid digest/size: {layer!r}")
        unique[digest] = {"digest": digest, "size_bytes": size}
    return sum(layer["size_bytes"] for layer in unique.values()), list(unique.values())


def jetson_admission(
    model: str,
    *,
    reserve_mib: int = 6144,
    max_utilization_percent: float = 80.0,
    samples: int = 5,
    sample_interval: float = 0.25,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> dict[str, Any]:
    """Measure post-handoff memory and sustained Tegra GPU utilization."""

    if reserve_mib < 0:
        raise ValueError("reserve_mib must be non-negative")
    if not 0 <= max_utilization_percent <= 100:
        raise ValueError("max_utilization_percent must be between 0 and 100")
    if samples < 1 or samples > 60:
        raise ValueError("samples must be between 1 and 60")
    if sample_interval < 0 or sample_interval > 10:
        raise ValueError("sample_interval must be between 0 and 10 seconds")
    weight_bytes, layers = artifact_bytes(model)
    available_bytes = _mem_available_bytes(meminfo_path)
    utilization: list[float] = []
    facts: list[dict[str, Any]] | None = None
    for index in range(samples):
        facts = gpu_facts()
        if facts:
            value = facts[0].get("utilization_percent")
            if isinstance(value, int | float):
                utilization.append(float(value))
        if index + 1 < samples and sample_interval > 0:
            time.sleep(sample_interval)
    median_utilization = statistics.median(utilization) if utilization else None
    required_bytes = weight_bytes + reserve_mib * 1024 * 1024
    memory_ok = available_bytes is not None and available_bytes >= required_bytes
    utilization_ok = (
        median_utilization is not None and median_utilization <= max_utilization_percent
    )
    return {
        "accelerator": accelerator_profile(),
        "gpu": facts,
        "model": model,
        "layers": layers,
        "artifact_bytes": weight_bytes,
        "artifact_gib": round(weight_bytes / 1024**3, 2),
        "memory_available_bytes": available_bytes,
        "memory_available_gib": (
            round(available_bytes / 1024**3, 2) if available_bytes is not None else None
        ),
        "reserve_mib": reserve_mib,
        "required_available_bytes": required_bytes,
        "required_available_gib": round(required_bytes / 1024**3, 2),
        "utilization_samples_percent": utilization,
        "median_utilization_percent": median_utilization,
        "max_allowed_utilization_percent": max_utilization_percent,
        "memory_ok": memory_ok,
        "utilization_ok": utilization_ok,
        "admitted": memory_ok and utilization_ok,
    }


def inventory(ports: Iterable[int]) -> dict[str, Any]:
    owners = listener_owners(ports)
    return {
        "accelerator": accelerator_profile(),
        "gpu": gpu_facts(),
        "ports": list(ports),
        "owners": owners,
        "ports_free": not owners,
        "all_owners_recognized": all(owner["recognized_omni"] for owner in owners),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_omni_adapters.deployment_handoff")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "targets", "ports-free"):
        item = sub.add_parser(command)
        item.add_argument("--ports", type=_parse_ports, default=DEFAULT_PORTS)
    admit = sub.add_parser("admit")
    admit.add_argument("--model", required=True)
    admit.add_argument("--reserve-mib", type=int, default=6144)
    admit.add_argument("--max-utilization-percent", type=float, default=80.0)
    admit.add_argument("--samples", type=int, default=5)
    admit.add_argument("--sample-interval", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "admit":
        if not is_tegra():
            print("Jetson admission is valid only on NVIDIA Tegra", file=sys.stderr)
            return 2
        try:
            result = jetson_admission(
                args.model,
                reserve_mib=args.reserve_mib,
                max_utilization_percent=args.max_utilization_percent,
                samples=args.samples,
                sample_interval=args.sample_interval,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"deployment admission failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["admitted"] else 1

    result = inventory(args.ports)
    if args.command == "inventory":
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "ports-free":
        if not result["ports_free"]:
            print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        return 0
    try:
        targets = handoff_targets(result["owners"])
    except RuntimeError as exc:
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3
    for kind, value in targets:
        print(f"{kind}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
