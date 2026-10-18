"""Actual CMake builds exercise incremental reuse without loading a CUDA device.

The compiler-configuration helper uses the same path for C and CUDA. These
portable C builds test object reuse, invalidation and failure recovery; they
do not establish CUDA inference or application deployment acceptance.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from runtime import cuda_toolchain as cuda


@pytest.fixture(params=["Unix Makefiles", "Ninja"])
def native_build(request, tmp_path):
    if not shutil.which("cmake") or not (compiler := shutil.which("cc")):
        pytest.skip("Native incremental-build control requires CMake and a C compiler")
    if request.param == "Ninja" and not shutil.which("ninja"):
        pytest.skip("Ninja is not installed")
    source, build = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    calls = tmp_path / "compiler-calls.jsonl"
    wrappers = []
    for name in ("compiler one", "compiler two"):
        wrapper = tmp_path / name
        wrapper.write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            f"with open({str(calls)!r}, 'a') as out:\n"
            "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            f"os.execv({compiler!r}, [{compiler!r}, *sys.argv[1:]])\n"
        )
        wrapper.chmod(0o755)
        wrappers.append(wrapper)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(reuse_control C)\n"
        'if(DEFINED ENV{OMNI_TEST_CONFIGURE_FAILURE})\nmessage(FATAL_ERROR "injected configure failure")\nendif()\n'
        'set(USER_SETTING "preserve" CACHE STRING "Unrelated user setting")\n'
        'set(VALUE "0" CACHE STRING "Target compile option")\n'
        'add_executable(probe main.c)\ntarget_compile_definitions(probe PRIVATE OPTION_VALUE=${VALUE})\n'
    )
    code = source / "main.c"
    code.write_text(
        "#include <stdio.h>\n#ifndef REQUIRED\n#error Missing required compiler argument\n#endif\n"
        'int main(void) { printf("%d %d\\n", REQUIRED, OPTION_VALUE); return 0; }\n'
    )
    return source, build, wrappers, calls, request.param


def configure(fixture, index=0, required=1, *, value=0, fail=False):
    source, build, wrappers, _, generator = fixture
    env = {**os.environ, "CC": shlex.join([str(wrappers[index]), f"-DREQUIRED={required}"])}
    if fail:
        env["OMNI_TEST_CONFIGURE_FAILURE"] = "1"
    return cuda._configure_compiler(
        source, build, str(wrappers[index]), ["-G", generator, f"-DVALUE={value}"], env,
        language="C", compiler_environment="CC",
    )


def build_target(fixture, changed):
    _, build, _, _, _ = fixture
    subprocess.run(
        ["cmake", "--build", str(build), *(["--clean-first"] if changed else []),
         "--target", "probe", "--parallel", "1"],
        capture_output=True, text=True, check=True,
    )
    cuda.complete_build(build)


def artifacts(fixture):
    _, build, _, _, _ = fixture
    paths = [build / "probe", *build.rglob("main.c.o")]
    assert len(paths) == 2
    return {str(path.relative_to(build)): (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}


def invocations(fixture):
    return [json.loads(line) for line in fixture[3].read_text().splitlines()]


def output(fixture):
    return subprocess.check_output([str(fixture[1] / "probe")], text=True).strip()


def test_unchanged_deploy_reuses_objects_and_real_changes_rebuild(native_build):
    fixture = native_build
    assert configure(fixture) is False
    build_target(fixture, False)
    assert output(fixture) == "1 0"
    original, calls = artifacts(fixture), invocations(fixture)

    # An application-only pull must not touch native objects or linked output.
    (fixture[0].parent / "portal.py").write_text("updated_application = True\n")
    assert configure(fixture) is False
    build_target(fixture, False)
    assert artifacts(fixture) == original
    assert invocations(fixture) == calls

    code = fixture[0] / "main.c"
    code.write_text(code.read_text().replace("return 0;", "return 0; /* changed native source */"))
    assert configure(fixture) is False
    build_target(fixture, False)
    assert artifacts(fixture) != original
    assert any("-c" in args for args in invocations(fixture)[len(calls):])

    assert configure(fixture, value=7) is False
    build_target(fixture, False)
    assert output(fixture) == "1 7"
    assert configure(fixture, required=2, value=7) is True
    build_target(fixture, True)
    assert output(fixture) == "2 7"
    assert configure(fixture, index=1, required=3, value=7) is True
    build_target(fixture, True)
    assert output(fixture) == "3 7"
    assert "USER_SETTING:STRING=preserve" in (fixture[1] / "CMakeCache.txt").read_text()
    final, calls = artifacts(fixture), invocations(fixture)
    assert configure(fixture, index=1, required=3, value=7) is False
    build_target(fixture, False)
    assert artifacts(fixture) == final
    assert invocations(fixture) == calls


def test_failed_toolchain_reconfiguration_keeps_clean_build_requirement(native_build):
    fixture = native_build
    assert configure(fixture) is False
    build_target(fixture, False)
    with pytest.raises(subprocess.CalledProcessError):
        configure(fixture, required=2, fail=True)
    assert configure(fixture, required=2) is True
    # A failed build must also leave the change pending for the next deploy.
    code = fixture[0] / "main.c"
    original = code.read_text()
    code.write_text("#error injected build failure\n")
    with pytest.raises(subprocess.CalledProcessError):
        build_target(fixture, True)
    code.write_text(original)
    assert configure(fixture, required=2) is True
    build_target(fixture, True)
    assert output(fixture) == "2 0"
    assert configure(fixture, required=2) is False
