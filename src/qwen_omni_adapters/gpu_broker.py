"""Distinguish an unmanaged CUDA host from an unavailable lease broker."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BrokerProbe:
    state: Literal["absent", "available", "unavailable"]
    detail: str

    @property
    def managed(self) -> bool:
        # A failed discovery is never permission to bypass an installed broker.
        return self.state != "absent"

    @property
    def mode(self) -> str:
        return "broker" if self.managed else "direct"


def _installation_marker() -> str:
    config = Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
    paths = [
        Path("/usr/local/share/ollama-unify/gpu-negotiator.json"),
        Path("/run/ollama-unify/gpu-negotiator.sock"),
        Path("/usr/local/bin/ollama-unify-gpu-lease"),
        config / "cli-plugins/docker-gpu",
        *(Path(base) / "docker-gpu" for base in (
            "/usr/local/lib/docker/cli-plugins", "/usr/local/libexec/docker/cli-plugins",
            "/usr/lib/docker/cli-plugins", "/usr/libexec/docker/cli-plugins",
        )),
    ]
    try:
        settings = json.loads((config / "config.json").read_text())
    except FileNotFoundError:
        settings = {}
    except (OSError, ValueError):
        # Custom plugin directories may be configured here. An unreadable or
        # invalid config cannot prove that no broker plugin is installed.
        return str(config / "config.json")
    extra_dirs = settings.get("cliPluginsExtraDirs", []) if isinstance(settings, dict) else []
    if isinstance(extra_dirs, list):
        paths.extend(Path(base) / "docker-gpu" for base in extra_dirs if isinstance(base, str))
    for path in paths:
        try:
            path.lstat()  # Broken symlinks also indicate an installed, broken plugin.
        except FileNotFoundError:
            continue
        except OSError:
            return str(path)  # An unreadable installation cannot establish absence.
        return str(path)
    return ""


def probe_gpu_broker() -> BrokerProbe:
    if platform.system() != "Linux":
        return BrokerProbe("absent", "Platform-native direct GPU lifecycle.")
    marker = _installation_marker()
    docker = shutil.which("docker")
    if not docker:
        if marker:
            return BrokerProbe("unavailable", f"Docker is missing; broker installation found: {marker}")
        return BrokerProbe("absent", "Docker and the ollama-unify GPU broker are not installed.")
    try:
        result = subprocess.run(
            [docker, "gpu", "discover"], check=False, capture_output=True, text=True,
            timeout=20, env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return BrokerProbe("unavailable", str(exc))
    detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode == 0:
        try:
            discovery = json.loads(result.stdout)
        except (ValueError, TypeError):
            discovery = None
        if isinstance(discovery, dict) and discovery.get("available") is True:
            return BrokerProbe("available", detail)
        return BrokerProbe("unavailable", detail or "GPU broker returned no valid discovery.")
    missing_command = detail.splitlines()[:1] in (
        ["docker: unknown command: docker gpu"],
        ["docker: unknown command: gpu"],
        ["docker: 'gpu' is not a docker command."],
    )
    if missing_command and not marker:
        return BrokerProbe("absent", detail)
    return BrokerProbe("unavailable", detail or f"GPU discovery failed (exit {result.returncode}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="store_true", help="Print the safe service mode only")
    args = parser.parse_args(argv)
    probe = probe_gpu_broker()
    if probe.state == "unavailable":
        print(f"GPU broker unavailable; refusing direct GPU mode: {probe.detail}", file=sys.stderr)
        return 1
    if args.mode:
        print(probe.mode)
    elif probe.managed:
        print(probe.detail)
    else:
        print("No ollama-unify GPU broker installed; using the direct GPU lifecycle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
