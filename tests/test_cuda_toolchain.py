"""Build setup controls: real subprocesses, isolated compiler/package shims.

These test installer behavior only; they do not install host packages or make
GPU/model/desktop acceptance claims.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import pytest

from runtime import cuda_toolchain as cuda


@pytest.fixture
def host(tmp_path, monkeypatch):
    commands, root = tmp_path / "commands", tmp_path / "cuda"
    commands.mkdir()
    root.mkdir()
    calls = tmp_path / "calls.jsonl"
    env = {"PATH": str(commands), "CALLS": str(calls),
           "INSTALL_NVCC": str(root / "bin/nvcc"), "PACKAGE": "cuda-nvcc-13-2"}
    monkeypatch.setattr(cuda, "toolkit_roots", lambda env: [root])
    monkeypatch.setattr(cuda.os, "geteuid", lambda: 0)

    def executable(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return str(path)

    def python_command(name, body):
        path = commands / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    python_command("apt-get", '''import json, os, sys
from pathlib import Path
with open(os.environ["CALLS"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("APT_FAIL") in sys.argv:
    sys.exit(42)
if "install" in sys.argv and not os.environ.get("NO_COMPILER"):
    path = Path(os.environ["INSTALL_NVCC"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\\nprintf 'nvcc fixture\\\\n'\\n")
    path.chmod(0o755)
''')
    python_command("apt-cache", '''import os, sys
print("  Candidate: " + ("13.2.86-1" if sys.argv[-1] == os.environ["PACKAGE"] else "(none)"))
''')
    return commands, root, calls, env, executable


def records(calls):
    return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []


def test_compiler_outside_path_is_reused_without_apt(host):
    _, root, calls, env, executable = host
    path = executable(root / "bin/nvcc", "exit 0")
    assert cuda.ensure_compiler(env) == path
    assert not calls.exists()


def test_path_compiler_is_reused_without_apt(host):
    commands, _, calls, env, executable = host
    path = executable(commands / "nvcc", "exit 0")
    assert cuda.ensure_compiler(env) == path
    assert not calls.exists()


def test_explicit_compiler_and_required_arguments_are_respected(host):
    commands, root, calls, env, executable = host
    executable(root / "bin/nvcc", "exit 0")
    path = executable(commands / "custom nvcc", '[ "$1" = --required ] && [ "$2" = --version ]')
    env["CUDACXX"] = shlex.quote(path) + " --required"
    assert cuda.ensure_compiler(env) == path
    assert not calls.exists()


@pytest.mark.parametrize("override", ["missing-nvcc", "'unterminated", "   "])
def test_invalid_explicit_override_does_not_choose_or_install_another_compiler(host, override):
    _, root, calls, env, executable = host
    executable(root / "bin/nvcc", "exit 0")
    env["CUDACXX"] = override
    with pytest.raises(RuntimeError, match="CUDACXX"):
        cuda.ensure_compiler(env)
    assert not calls.exists()


def test_valid_cached_compiler_is_preserved_with_unrelated_options(host, tmp_path):
    commands, root, calls, env, executable = host
    executable(root / "bin/nvcc", "exit 0")
    path = executable(commands / "cached-nvcc", "exit 0")
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(f"CMAKE_CUDA_COMPILER:FILEPATH={path}\nUSER_SETTING:STRING=preserve\n")
    before = cache.read_bytes()
    assert cuda.ensure_compiler(env, cache) == path
    assert cache.read_bytes() == before
    assert not calls.exists()


def test_cached_compiler_required_arguments_are_preserved(host, tmp_path):
    commands, root, calls, env, executable = host
    executable(root / "bin/nvcc", "exit 0")
    path = executable(commands / "cached-nvcc", '[ "$1" = --required ] && [ "$2" = --version ]')
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(f"CMAKE_CUDA_COMPILER:FILEPATH={path}\nCMAKE_CUDA_COMPILER_ARG1:STRING=--required\nUSER_SETTING:STRING=preserve\n")
    before = cache.read_bytes()
    assert cuda.ensure_compiler(env, cache) == path
    assert cache.read_bytes() == before
    assert not calls.exists()


@pytest.mark.parametrize("old", ["CMAKE_CUDA_COMPILER-NOTFOUND", "/removed/cuda/bin/nvcc", ""])
def test_invalid_cache_does_not_hide_installed_compiler_or_delete_cache(host, tmp_path, old):
    _, root, calls, env, executable = host
    path = executable(root / "bin/nvcc", "exit 0")
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(f"CMAKE_CUDA_COMPILER:FILEPATH={old}\nUSER_SETTING:STRING=preserve\n")
    before = cache.read_bytes()
    assert cuda.ensure_compiler(env, cache) == path
    assert cache.read_bytes() == before
    assert not calls.exists()


@pytest.mark.parametrize("metadata", ["json", "text", "header", "target-header"])
def test_runtime_only_install_gets_matching_compiler_then_verifies_it(host, metadata):
    _, root, calls, env, _ = host
    if metadata == "json":
        (root / "version.json").write_text('{"cuda":{"version":"13.2.86"}}')
    elif metadata == "text":
        (root / "version.txt").write_text("CUDA Version 13.2.86\n")
    else:
        header = root / ("include/cuda.h" if metadata == "header" else "targets/x86_64-linux/include/cuda.h")
        header.parent.mkdir(parents=True)
        header.write_text("#define CUDA_VERSION 13020\n")
    assert cuda.ensure_compiler(env) == str(root / "bin/nvcc")
    assert records(calls) == [
        ["-o", "DPkg::Lock::Timeout=600", "update"],
        ["-o", "DPkg::Lock::Timeout=600", "install", "-y", "--no-install-recommends", "--reinstall", "cuda-nvcc-13-2"],
    ]
    cuda.ensure_compiler(env)
    assert len(records(calls)) == 2  # Idempotent rerun; no repeated package work.


def test_unavailable_matching_package_stops_without_unversioned_upgrade(host):
    _, root, calls, env, _ = host
    (root / "version.json").write_text('{"cuda":{"version":"13.2.86"}}')
    env["PACKAGE"] = "cuda-nvcc-13-3"
    with pytest.raises(RuntimeError, match="no available CUDA compiler package"):
        cuda.ensure_compiler(env)
    assert all("install" not in row for row in records(calls))
    assert not (root / "bin/nvcc").exists()


@pytest.mark.parametrize("operation", ["update", "install"])
def test_apt_failure_propagates_without_success_or_retry_loop(host, operation):
    _, root, calls, env, _ = host
    (root / "version.txt").write_text("CUDA Version 13.2.86\n")
    env["APT_FAIL"] = operation
    with pytest.raises(subprocess.CalledProcessError) as exc:
        cuda.ensure_compiler(env)
    assert exc.value.returncode == 42
    assert len(records(calls)) == (1 if operation == "update" else 2)
    assert not (root / "bin/nvcc").exists()


def test_successful_package_command_without_nvcc_is_not_success(host):
    _, root, calls, env, _ = host
    (root / "version.txt").write_text("CUDA Version 13.2.86\n")
    env["NO_COMPILER"] = "1"
    with pytest.raises(RuntimeError, match="no executable nvcc"):
        cuda.ensure_compiler(env)
    assert len(records(calls)) == 2


def test_missing_apt_reports_actionable_error_without_running_packages(host):
    commands, _, calls, env, _ = host
    (commands / "apt-get").unlink()
    with pytest.raises(RuntimeError, match="automatic installation requires APT"):
        cuda.ensure_compiler(env)
    assert not calls.exists()


def test_missing_sudo_stops_before_apt(host, monkeypatch):
    _, _, calls, env, _ = host
    monkeypatch.setattr(cuda.os, "geteuid", lambda: 1000)
    with pytest.raises(RuntimeError, match="sudo"):
        cuda.ensure_compiler(env)
    assert not calls.exists()


def test_unknown_toolkit_uses_distro_development_package(host):
    _, root, calls, env, _ = host
    env["PACKAGE"] = "nvidia-cuda-toolkit"
    assert cuda.ensure_compiler(env) == str(root / "bin/nvcc")
    assert records(calls)[1][-1] == "nvidia-cuda-toolkit"


def test_broken_compiler_is_not_treated_as_installed(host):
    _, root, _, env, executable = host
    executable(root / "bin/nvcc", "exit 42")
    (root / "version.txt").write_text("CUDA Version 13.2.86\n")
    assert cuda.ensure_compiler(env) == str(root / "bin/nvcc")
    assert subprocess.run([str(root / "bin/nvcc"), "--version"], env=env).returncode == 0


def test_configured_cuda_roots_are_searched_first(monkeypatch, tmp_path):
    roots = cuda.toolkit_roots({"CUDA_HOME": str(tmp_path / "home"), "CUDAToolkit_ROOT": str(tmp_path / "selected")})
    assert roots[:2] == [tmp_path / "selected", tmp_path / "home"]
    assert os.environ.get("CUDA_HOME") != str(tmp_path / "home")
