"""Accelerator identification and process-residency proofs across CUDA hosts.

Discrete NVIDIA hosts prove that a specific worker actually holds device
memory with ``nvidia-smi --query-compute-apps``. NVIDIA Tegra modules (Jetson
AGX/Orin/Xavier/Nano) run an integrated GPU behind the ``nvgpu``/``nvhost``
kernel drivers: ``nvidia-smi`` exists but reports ``Not Supported`` for memory
usage and never enumerates compute apps, so the discrete proof can only ever
time out there.

This module keeps one residency question -- "is this pid actually on the
GPU?" -- and answers it with whatever evidence the host can genuinely supply:

* discrete NVIDIA: per-process compute-app accounting, unchanged.
* Tegra: the worker's own open handles on the integrated GPU character
  devices (``/dev/nvgpu/igpu*`` on JetPack 6, ``/dev/nvhost-*gpu*`` on
  JetPack 4/5) plus the ``nvmap`` GPU allocator. A process that has mapped
  those nodes has an initialized CUDA context on the integrated GPU; a CPU
  fallback build never opens them.

Nothing here weakens the discrete path: a host that can answer with
compute-app accounting still must.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "accelerator_profile",
    "cuda_architectures",
    "gpu_facts",
    "is_tegra",
    "process_is_gpu_resident",
    "residency_backend",
    "tegra_soc",
]

# Integrated-GPU control nodes. A CUDA context on Tegra always holds at least
# one of these open for the lifetime of the context.
_TEGRA_GPU_DEVICE_PATTERNS = (
    re.compile(r"^/dev/nvgpu/(?:i|d)gpu\d+/"),  # JetPack 6 (L4T R36+)
    re.compile(r"^/dev/nvhost-(?:ctrl-)?gpu$"),  # JetPack 4/5 (L4T R32/R35)
    re.compile(r"^/dev/nvhost-(?:as|dbg|prof|ctxsw|sched)[-a-z_]*-gpu$"),
)
_TEGRA_ALLOCATOR_DEVICES = ("/dev/nvmap",)

# SoC compatible string -> CUDA compute capability of its integrated GPU.
_TEGRA_SOC_ARCHITECTURES = {
    "tegra210": "53",  # TX1, Nano
    "tegra186": "62",  # TX2
    "tegra194": "72",  # Xavier AGX/NX
    "tegra234": "87",  # Orin AGX/NX/Nano
    "tegra264": "101",  # Thor
}

_SYSFS_GPU_LOAD_GLOBS = (
    "/sys/devices/platform/bus@0/*.gpu/load",
    "/sys/devices/platform/*.gpu/load",
    "/sys/devices/gpu.0/load",
    "/sys/devices/platform/gpu.0/load",
)
_SYSFS_GPU_FREQ_GLOBS = (
    "/sys/devices/platform/bus@0/*.gpu/devfreq/*/cur_freq",
    "/sys/devices/platform/*.gpu/devfreq/*/cur_freq",
    "/sys/class/devfreq/17000000.gpa/cur_freq",
)
_SYSFS_GPU_TEMPERATURE_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"


def _read_text(path: Path | str, limit: int = 4096) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit).strip()
    except OSError:
        return ""


@lru_cache(maxsize=1)
def tegra_soc() -> str | None:
    """Return the Tegra SoC identifier (e.g. ``tegra234``), or None off-Tegra."""

    if platform.system() != "Linux":
        return None
    compatible = ""
    try:
        compatible = Path("/proc/device-tree/compatible").read_bytes().decode(
            "ascii", errors="replace"
        )
    except OSError:
        compatible = ""
    for token in compatible.split("\x00"):
        _, _, name = token.partition("nvidia,")
        if name.startswith("tegra") and name in _TEGRA_SOC_ARCHITECTURES:
            return name
    for token in compatible.split("\x00"):
        _, _, name = token.partition("nvidia,")
        if name.startswith("tegra"):
            return name
    if Path("/etc/nv_tegra_release").is_file():
        return "tegra"
    return None


@lru_cache(maxsize=1)
def is_tegra() -> bool:
    """Return True on an NVIDIA Tegra module with an integrated GPU."""

    if os.environ.get("OMNI_FORCE_TEGRA", "").strip() in {"1", "true", "yes"}:
        return True
    if tegra_soc() is not None:
        return True
    return platform.machine() in {"aarch64", "arm64"} and Path("/dev/nvmap").exists()


@lru_cache(maxsize=1)
def l4t_release() -> str:
    """Return the L4T release string (``R36.3.0``) when the host publishes one."""

    text = _read_text("/etc/nv_tegra_release", 512)
    match = re.search(r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", text)
    if not match:
        return ""
    return f"R{match.group(1)}.{match.group(2)}"


def cuda_architectures() -> str | None:
    """Return the ``CMAKE_CUDA_ARCHITECTURES`` value this host must build for.

    Tegra ships exactly one integrated GPU, so the answer is a single, exact
    compute capability. Discrete hosts keep llama.cpp's own detection, which
    already handles multi-GPU and newer silicon; None means "do not override".
    """

    override = os.environ.get("OMNI_CUDA_ARCHITECTURES", "").strip()
    if override:
        return override
    soc = tegra_soc()
    if soc and soc in _TEGRA_SOC_ARCHITECTURES:
        return _TEGRA_SOC_ARCHITECTURES[soc]
    if not is_tegra():
        return None
    capability = _nvidia_smi_compute_capability()
    return capability or None


def _nvidia_smi_compute_capability() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    completed = _run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
        timeout=5,
    )
    for line in completed.splitlines():
        value = line.strip()
        if re.fullmatch(r"\d+\.\d+", value):
            return value.replace(".", "")
    return ""


def _run(command: list[str], timeout: float = 10) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


@lru_cache(maxsize=1)
def compute_apps_supported() -> bool:
    """Return True when ``nvidia-smi`` can enumerate per-process GPU memory.

    Tegra's driver has no compute-app accounting at all, so the query returns
    success with an empty body forever. Treating that as "not yet resident"
    is what made every Jetson start-up fail its residency gate.
    """

    if is_tegra():
        return False
    if not shutil.which("nvidia-smi"):
        return False
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        return False
    return "not supported" not in completed.stdout.lower()


def residency_backend() -> str:
    """Name the evidence this host uses to prove GPU residency."""

    if platform.system() == "Darwin":
        return "metal-build"
    if is_tegra():
        return "tegra-device-handles"
    if compute_apps_supported():
        return "nvidia-smi-compute-apps"
    return "unavailable"


def _process_device_handles(pid: int) -> set[str]:
    """Return the /dev nodes a pid currently has open or mapped."""

    handles: set[str] = set()
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("/dev/"):
            handles.add(target)
    try:
        with open(f"/proc/{pid}/maps", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                path = line.rstrip("\n").rsplit(" ", 1)[-1]
                if path.startswith("/dev/"):
                    handles.add(path)
    except OSError:
        pass
    return handles


def _tegra_process_is_resident(pid: int) -> bool:
    handles = _process_device_handles(pid)
    if not handles:
        return False
    on_gpu = any(
        pattern.match(handle) for handle in handles for pattern in _TEGRA_GPU_DEVICE_PATTERNS
    )
    if not on_gpu:
        return False
    # The GPU control node alone proves a device handle; nvmap proves the
    # process also owns GPU-visible memory through the Tegra allocator.
    return any(device in handles for device in _TEGRA_ALLOCATOR_DEVICES)


def _discrete_process_is_resident(pid: int, gpu_uuid: str = "") -> bool:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            process_id = int(fields[0])
            used_mib = int(fields[2])
        except ValueError:
            continue
        if process_id == pid and (not gpu_uuid or fields[1] == gpu_uuid) and used_mib > 0:
            return True
    return False


def process_is_gpu_resident(pid: int, gpu_uuid: str = "") -> bool:
    """Return True when ``pid`` demonstrably holds the host's GPU.

    ``gpu_uuid`` pins the answer to one reserved discrete device. Tegra has a
    single integrated GPU that cannot be partitioned, so a uuid there is
    satisfied by the module's own GPU.
    """

    if is_tegra():
        return _tegra_process_is_resident(pid)
    return _discrete_process_is_resident(pid, gpu_uuid)


def _sysfs_first(globs: tuple[str, ...]) -> Path | None:
    root = Path("/")
    for pattern in globs:
        for candidate in sorted(root.glob(pattern.lstrip("/"))):
            if candidate.is_file():
                return candidate
    return None


def _tegra_memory_mib() -> tuple[float | None, float | None]:
    total_kib = available_kib = None
    for line in _read_text("/proc/meminfo", 4096).splitlines():
        key, _, value = line.partition(":")
        digits = value.strip().split(" ")[0]
        if not digits.isdigit():
            continue
        if key == "MemTotal":
            total_kib = int(digits)
        elif key == "MemAvailable":
            available_kib = int(digits)
    if total_kib is None:
        return None, None
    total_mib = total_kib / 1024
    if available_kib is None:
        return total_mib, None
    return total_mib, total_mib - available_kib / 1024


def _tegra_temperature_c() -> float | None:
    root = Path("/")
    for zone in sorted(root.glob(_SYSFS_GPU_TEMPERATURE_GLOB.lstrip("/"))):
        name = _read_text(zone / "type", 64).lower()
        if "gpu" not in name:
            continue
        raw = _read_text(zone / "temp", 32)
        if raw.lstrip("-").isdigit():
            return int(raw) / 1000
    return None


def tegra_gpu_facts() -> list[dict[str, Any]]:
    """Describe the integrated GPU from sysfs, mirroring the discrete fields.

    ``nvidia-smi --query-gpu`` answers ``[N/A]`` for every interesting column
    on Tegra, so the host would otherwise report no GPU at all.
    """

    if not is_tegra():
        return []
    utilization = None
    load_path = _sysfs_first(_SYSFS_GPU_LOAD_GLOBS)
    if load_path is not None:
        raw = _read_text(load_path, 32)
        if raw.isdigit():
            # Tegra publishes per-mille load.
            utilization = min(100.0, int(raw) / 10)
    frequency_mhz = None
    freq_path = _sysfs_first(_SYSFS_GPU_FREQ_GLOBS)
    if freq_path is not None:
        raw = _read_text(freq_path, 32)
        if raw.isdigit():
            frequency_mhz = int(raw) / 1_000_000
    total_mib, used_mib = _tegra_memory_mib()
    soc = tegra_soc() or "tegra"
    return [
        {
            "index": 0,
            "name": f"NVIDIA Tegra integrated GPU ({soc})"[:160],
            "vram_total_mib": round(total_mib, 1) if total_mib is not None else None,
            "vram_used_mib": round(used_mib, 1) if used_mib is not None else None,
            "utilization_percent": utilization,
            "temperature_c": _tegra_temperature_c(),
            "power_w": None,
            "power_limit_w": None,
            "memory_model": "unified",
            "frequency_mhz": round(frequency_mhz, 1) if frequency_mhz is not None else None,
        }
    ]


def gpu_facts() -> list[dict[str, Any]] | None:
    """Return Tegra GPU facts, or None when the discrete path should be used."""

    return tegra_gpu_facts() if is_tegra() else None


def accelerator_profile() -> dict[str, Any]:
    """Summarize the accelerator for diagnostics and the doctor command."""

    profile: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "tegra": is_tegra(),
        "residency_backend": residency_backend(),
        "gpu_memory_model": "unified" if is_tegra() else "discrete",
    }
    if is_tegra():
        profile["tegra_soc"] = tegra_soc()
        profile["l4t_release"] = l4t_release()
    architectures = cuda_architectures()
    if architectures:
        profile["cuda_architectures"] = architectures
    return profile
