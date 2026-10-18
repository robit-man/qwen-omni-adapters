"""Find a native CUDA compiler, repairing a missing APT compiler installation.

CUDA libraries (including PyTorch's wheels) do not establish that nvcc exists.
This build-time helper never imports CUDA, loads weights, or changes drivers.
Only the selected compiler path is written to stdout, for the shell builder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def toolkit_roots(env: Mapping[str, str]) -> list[Path]:
    roots = [Path(env[key]) for key in ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH")
             if env.get(key)]
    roots.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))
    roots.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
    return list(dict.fromkeys(roots))


def usable_compiler(command: list[str], env: Mapping[str, str]) -> str | None:
    path = shutil.which(command[0], path=env.get("PATH", os.defpath))
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, *command[1:], "--version"], env=dict(env),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return os.path.abspath(path) if result.returncode == 0 else None


def find_compiler(env: Mapping[str, str], cache: Path | None = None) -> str | None:
    if explicit := env.get("CUDACXX"):
        # CMake allows a compiler command with required arguments. Preserve
        # that original command for CMake; never eval a shell expression.
        try:
            command = [explicit] if Path(explicit).is_file() else shlex.split(explicit)
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDACXX: {exc}") from exc
        compiler = usable_compiler(command, env) if command else None
        if not compiler:
            raise RuntimeError("CUDACXX does not name an executable CUDA compiler; correct or unset it.")
        return compiler
    if cache and cache.is_file():
        cached = {}
        for line in cache.read_text().splitlines():
            name, separator, value = line.partition("=")
            name = name.partition(":")[0]
            if separator and name in {"CMAKE_CUDA_COMPILER", "CMAKE_CUDA_COMPILER_ARG1"}:
                cached[name] = value
        if path := cached.get("CMAKE_CUDA_COMPILER"):
            try:
                command = [path, *shlex.split(cached.get("CMAKE_CUDA_COMPILER_ARG1", ""))]
            except ValueError as exc:
                raise RuntimeError(f"Invalid cached CUDA compiler arguments: {exc}") from exc
            if compiler := usable_compiler(command, env):
                return compiler
    roots = toolkit_roots(env)
    candidates = [str(root / "bin/nvcc") for root in roots]
    candidates.append("nvcc")
    for candidate in candidates:
        if compiler := usable_compiler([candidate], env):
            return compiler
    return None


def toolkit_version(roots: list[Path]) -> str | None:
    for root in roots:
        metadata = root / "version.json"
        if metadata.is_file():
            try:
                data = json.loads(metadata.read_text())
                version = data.get("cuda", {}).get("version", "")
                if match := re.match(r"^(\d+)\.(\d+)(?:\.|$)", str(version)):
                    return f"{match[1]}-{match[2]}"
            except (OSError, ValueError, AttributeError):
                pass
        version_file = root / "version.txt"
        if version_file.is_file() and (match := re.search(r"CUDA Version\s+(\d+)\.(\d+)", version_file.read_text())):
            return f"{match[1]}-{match[2]}"
        for header in (root / "include/cuda.h", *root.glob("targets/*/include/cuda.h")):
            if header.is_file() and (match := re.search(r"^\s*#define\s+CUDA_VERSION\s+(\d+)", header.read_text(), re.M)):
                version = int(match[1])
                return f"{version // 1000}-{version % 1000 // 10}"
        if root.exists() and (match := re.fullmatch(r"cuda-(\d+)\.(\d+)", root.resolve().name)):
            return f"{match[1]}-{match[2]}"
    return None


def ensure_compiler(env: Mapping[str, str], cache: Path | None = None) -> str:
    if compiler := find_compiler(env, cache):
        return compiler
    if not all(shutil.which(tool, path=env.get("PATH", os.defpath)) for tool in ("apt-get", "apt-cache")):
        raise RuntimeError("Missing nvcc. Install the native CUDA compiler and set CUDACXX to its path; automatic installation requires APT.")
    prefix = []
    if os.geteuid() != 0:
        if not shutil.which("sudo", path=env.get("PATH", os.defpath)):
            raise RuntimeError("Missing nvcc and sudo; install the CUDA compiler as administrator.")
        prefix = ["sudo", "env", "DEBIAN_FRONTEND=noninteractive", "LC_ALL=C"]
    install_env = {**env, "DEBIAN_FRONTEND": "noninteractive", "LC_ALL": "C"}

    def apt(*args: str) -> None:
        subprocess.run(
            [*prefix, "apt-get", "-o", "DPkg::Lock::Timeout=600", *args],
            env=install_env, stdout=sys.stderr, check=True,
        )

    print("CUDA headers/libraries do not provide nvcc; installing the missing compiler...", file=sys.stderr)
    apt("update")
    version = toolkit_version(toolkit_roots(env))
    packages = ([f"cuda-nvcc-{version}", f"cuda-compiler-{version}"] if version
                else ["nvidia-cuda-toolkit"])
    package = None
    for candidate in packages:
        policy = subprocess.run(
            ["apt-cache", "policy", candidate], env=install_env,
            capture_output=True, text=True, check=True,
        ).stdout
        if re.search(r"^\s*Candidate:\s+(?!\(none\))\S+", policy, re.M):
            package = candidate
            break
    if not package:
        raise RuntimeError(
            f"APT has no available CUDA compiler package ({', '.join(packages)}). "
            "Enable the matching NVIDIA CUDA repository or install nvcc and set CUDACXX."
        )
    print(f"Installing {package} (native compiler and its dependencies).", file=sys.stderr)
    apt("install", "-y", "--no-install-recommends", "--reinstall", package)
    if compiler := find_compiler(env, cache):
        return compiler
    raise RuntimeError(f"Installed {package}, but no executable nvcc was found; CUDA build cannot continue.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    try:
        compiler = ensure_compiler(
            os.environ, args.build_dir / "CMakeCache.txt" if args.build_dir else None,
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"CUDA compiler setup failed: {exc}", file=sys.stderr)
        return 1
    print(compiler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
