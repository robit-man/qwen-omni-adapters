"""Find a native CUDA compiler, repairing a missing APT compiler installation.

CUDA libraries (including PyTorch's wheels) do not establish that nvcc exists.
This build-time helper never imports CUDA, loads weights, or changes drivers.
By default stdout contains only the selected compiler path. Configure mode
instead reports changed/unchanged for the shell builder's incremental build.
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
import tempfile
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


def _compiler_info(build: Path, language: str, env: Mapping[str, str]) -> tuple[Path, str, str]:
    # Let the running CMake interpret its own generated file, including escaped
    # paths/arguments. HEX avoids treating those values as a line protocol.
    script = '''set(info "$ENV{OMNI_CMAKE_BUILD}/CMakeFiles/${CMAKE_VERSION}/CMake$LANGCompiler.cmake")
if(EXISTS "${info}")
  include("${info}")
endif()
string(HEX "${info}" info_hex)
string(HEX "${CMAKE_$LANG_COMPILER}" compiler_hex)
string(HEX "${CMAKE_$LANG_COMPILER_ARG1}" arguments_hex)
message("OMNI_INFO=${info_hex}")
message("OMNI_COMPILER=${compiler_hex}")
message("OMNI_ARGUMENTS=${arguments_hex}")
'''.replace("$LANG", language)
    with tempfile.TemporaryDirectory(prefix="omni-cmake-compiler-") as directory:
        probe = Path(directory) / "compiler.cmake"
        probe.write_text(script)
        result = subprocess.run(
            ["cmake", "-P", str(probe)], env={**env, "OMNI_CMAKE_BUILD": str(build)},
            capture_output=True, text=True, check=True,
        )
    values = re.findall(r"^OMNI_(INFO|COMPILER|ARGUMENTS)=([0-9a-f]*)$", result.stderr, re.M)
    if len(values) != 3 or {key for key, _ in values} != {"INFO", "COMPILER", "ARGUMENTS"}:
        raise RuntimeError("CMake did not report its effective compiler configuration.")
    decoded = {key: bytes.fromhex(value).decode() for key, value in values}
    return Path(decoded["INFO"]), decoded["COMPILER"], decoded["ARGUMENTS"]


def _configure_compiler(
    source: Path, build: Path, compiler: str, options: list[str], env: Mapping[str, str],
    *, language: str = "CUDA", compiler_environment: str = "CUDACXX",
) -> bool:
    """Configure normally, resetting only a changed/incomplete compiler record."""
    source, build = source.resolve(), build.resolve()
    compiler = os.path.abspath(compiler)
    key = f"CMAKE_{language}_COMPILER"
    cached = {}
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            name, separator, value = line.partition("=")
            name = name.partition(":")[0]
            if separator and name in {key, key + "_ARG1"}:
                cached[name] = value
    try:
        if explicit := env.get(compiler_environment):
            command = [explicit] if Path(explicit).is_file() else shlex.split(explicit)
            selected = shutil.which(command[0], path=env.get("PATH", os.defpath)) if command else None
            if not selected or os.path.abspath(selected) != compiler:
                raise RuntimeError(f"{compiler_environment} does not match the selected compiler.")
            arguments = command[1:]
        else:
            arguments = (shlex.split(cached.get(key + "_ARG1", ""))
                         if cached.get(key) and os.path.abspath(cached[key]) == compiler else [])
        info, effective, effective_arguments = _compiler_info(build, language, env)
        previous = ((os.path.abspath(effective), shlex.split(effective_arguments))
                    if effective else None)
        cached_compiler = cached.get(key, "")
        if previous is None and cached_compiler and not cached_compiler.endswith("-NOTFOUND"):
            previous = (os.path.abspath(cached_compiler), shlex.split(cached.get(key + "_ARG1", "")))
    except ValueError as exc:
        raise RuntimeError(f"Invalid {compiler_environment} compiler arguments: {exc}") from exc
    desired = (compiler, arguments)
    marker = build / ".omni-cuda-compiler-change"
    changed = previous is not None and previous != desired
    if changed and not marker.exists():
        # A failed configure/build must not consume the required object clean.
        marker.write_text("pending\n")
    reset = not effective or (os.path.abspath(effective), shlex.split(effective_arguments)) != desired
    configure_env = {**env, compiler_environment: shlex.join([compiler, *arguments])}
    if reset:
        info.unlink(missing_ok=True)
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), *options,
         *(["-U", key, "-U", key + "_ARG1"] if reset else [])],
        env=configure_env, stdout=sys.stderr, stderr=sys.stderr, check=True,
    )
    _, configured, configured_arguments = _compiler_info(build, language, env)
    if not configured or (os.path.abspath(configured), shlex.split(configured_arguments)) != desired:
        raise RuntimeError("CMake did not retain the requested compiler and required arguments.")
    return marker.exists()


def configure_cuda(
    source: Path, build: Path, compiler: str, options: list[str], env: Mapping[str, str],
) -> bool:
    return _configure_compiler(source, build, compiler, options, env)


def complete_build(build: Path) -> None:
    """Acknowledge a successful native build, never a configure alone."""
    (build / ".omni-cuda-compiler-change").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--configure", type=Path, metavar="SOURCE")
    parser.add_argument("--compiler")
    parser.add_argument("--build-complete", action="store_true")
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.build_complete:
            if not args.build_dir or args.configure or args.compiler or args.options:
                parser.error("--build-complete requires only --build-dir")
            complete_build(args.build_dir)
            return 0
        if args.configure:
            if not args.build_dir or not args.compiler:
                parser.error("--configure requires --build-dir and --compiler")
            options = args.options[1:] if args.options[:1] == ["--"] else args.options
            changed = configure_cuda(args.configure, args.build_dir, args.compiler, options, os.environ)
            print("changed" if changed else "unchanged")
            return 0
        if args.compiler or args.options:
            parser.error("--compiler and CMake options require --configure")
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
