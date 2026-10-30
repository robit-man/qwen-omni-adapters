"""Pinned native grammar integration and unchanged host dispatch authority."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from portal.ui_task import _grounded_phase_schema, phase_response_schema, run_native_goal
from qwen_omni_adapters.computer_use import parse_step

ROOT = Path(__file__).resolve().parents[1]
PIN = "458681e1d5d4a29a1463c4732e03226cf384b997"
PATCH = "llama.cpp-json-pattern-safety.patch"
BOUNDS_PATCH = "llama.cpp-json-string-bounds.patch"
CONDITIONING_PATCH = "llama.cpp-qwen3tts-conditioning-cache.patch"
TOKEN_ROWS_PATCH = "llama.cpp-qwen3tts-token-rows.patch"
INDEXED_VIEWS_PATCH = "llama.cpp-indexed-tts-views.patch"
REFERENCE_ENCODER_PATCH = "llama.cpp-qwen3tts-reference-encoder.patch"
INDEPENDENT_PATCHES = (PATCH, BOUNDS_PATCH, CONDITIONING_PATCH, TOKEN_ROWS_PATCH, INDEXED_VIEWS_PATCH, REFERENCE_ENCODER_PATCH)


def _patch_preimage(text: str, source: Path) -> None:
    """Minimal apply fixture from old hunk lines; actual pinned proof is separate."""
    current, position = None, 0
    files: dict[str, list[str]] = {}
    for line in text.splitlines(keepends=True):
        if line.startswith("--- a/"):
            current = line[6:].strip()
            files[current] = []
        elif line.startswith("--- /dev/null"):
            current = None
        elif line.startswith("@@"):
            position = int(re.match(r"@@ -(\d+)", line)[1]) - 1
        elif current and line[:1] in {" ", "-"}:
            lines = files[current]
            while len(lines) <= position:
                lines.append("\n")
            lines[position] = line[1:]
            position += 1
    for name, lines in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))


def _build_fixture(tmp_path: Path, *, pin: str = PIN, patch_name: str = PATCH) -> tuple[Path, Path, dict[str, str]]:
    repo, source, commands = tmp_path / "repo", tmp_path / "source", tmp_path / "commands"
    (repo / "scripts").mkdir(parents=True)
    (repo / "runtime").mkdir()
    (repo / "patches").mkdir()
    source.mkdir()
    commands.mkdir()
    shutil.copyfile(ROOT / "scripts/build_llama_cpp.sh", repo / "scripts/build_llama_cpp.sh")
    shutil.copyfile(ROOT / "runtime/cuda_toolchain.py", repo / "runtime/cuda_toolchain.py")
    for name in INDEPENDENT_PATCHES:
        shutil.copyfile(ROOT / "patches" / name, repo / "patches" / name)
    _patch_preimage((ROOT / "patches" / patch_name).read_text(), source)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    for name, text in {"common/arg.cpp": '"--tts-stream-frames"\n"--tts-persistent"\n',
                       "tools/mtmd/clip.cpp": "state_out describes this exact output boundary\n"}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((path.read_text() if path.exists() else "") + text)
    git = commands / "git"
    other_patches = "|".join(f'"{repo / "patches" / name}"'
                             for name in INDEPENDENT_PATCHES if name != patch_name)
    # Only the selected independent patch is exercised by this control-flow
    # fixture; the other is an already-applied prerequisite. Full real pinned
    # source composition is covered by the native proof, not this mock.
    git.write_text(f'#!/bin/sh\nif [ "$3" = rev-parse ]; then printf "%s\\n" "{pin}"; exit 0; fi\n'
                  f'if [ "$3" = apply ] && [ "$4" = --reverse ]; then case "$6" in '
                  f'{other_patches}) exit 0;; esac; fi\nexec {shutil.which("git")} "$@"\n')
    cmake = commands / "cmake"
    # This patch-control fixture models configuration, not compilation. Let
    # real CMake read generated compiler metadata, as the builder now does.
    cmake.write_text(
        f"#!{sys.executable}\nimport os, shlex, subprocess, sys\nfrom pathlib import Path\n"
        f"real_cmake = {shutil.which('cmake')!r}\n"
        "if sys.argv[1] == '-P':\n    os.execv(real_cmake, [real_cmake, *sys.argv[1:]])\n"
        "with open(os.environ['BUILD_CALLS'], 'a') as out:\n    out.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == '-S':\n"
        "    version = subprocess.check_output([real_cmake, '--version'], text=True).split()[2]\n"
        "    info = Path(sys.argv[4]) / 'CMakeFiles' / version / 'CMakeCUDACompiler.cmake'\n"
        "    info.parent.mkdir(parents=True, exist_ok=True)\n"
        "    compiler = shlex.split(os.environ['CUDACXX'])[0]\n"
        "    info.write_text('set(CMAKE_CUDA_COMPILER [==[' + compiler + ']==])\\n')\n"
    )
    git.chmod(0o755)
    cmake.chmod(0o755)
    nvcc = commands / "nvcc"
    nvcc.write_text('#!/bin/sh\nprintf "%s\\n" "CUDA compiler fixture"\n')
    nvcc.chmod(0o755)
    target = tmp_path / "build/bin/llama-tts"
    target.parent.mkdir(parents=True)
    target.write_text('#!/bin/sh\nprintf "%s\\n" "--tts-stream-frames --tts-persistent --tts-no-conditioning-cache --tts-conditioning-cache-rows"\n')
    target.chmod(0o755)
    reference_encoder = target.parent / "llama-tts-reference-encode"
    reference_encoder.write_text("#!/bin/sh\nexit 0\n")
    reference_encoder.chmod(0o755)
    env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
           "BUILD_CALLS": str(tmp_path / "cmake-calls"), "LLAMA_CPP_BUILD_DIR": str(target.parent.parent),
           "CUDACXX": str(nvcc),
           "LLAMA_CPP_BUILD_JOBS": "1"}
    return repo, source, env


@pytest.mark.parametrize("patch_name,label", [(PATCH, "JSON pattern string safety"),
                                              (BOUNDS_PATCH, "JSON string length bounds"),
                                              (CONDITIONING_PATCH, "Qwen3-TTS immutable conditioning cache"),
                                              (TOKEN_ROWS_PATCH, "Qwen3-TTS bounded token rows"),
                                              (INDEXED_VIEWS_PATCH, "Indexed TTS source views"),
                                              (REFERENCE_ENCODER_PATCH, "Qwen3-TTS reference acoustic encoder")])
def test_linux_builder_applies_each_independent_patch_once_and_keeps_pin(tmp_path, patch_name, label):
    repo, source, env = _build_fixture(tmp_path, patch_name=patch_name)
    command = ["bash", str(repo / "scripts/build_llama_cpp.sh"), str(source)]
    first = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*.cpp")}
    second = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert f"Applied {label} patch" in first.stdout
    assert f"{label} patch is already applied" in second.stdout
    assert after == {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*.cpp")}
    calls = Path(env["BUILD_CALLS"]).read_text().splitlines()
    assert len(calls) == 4
    assert all("-DGGML_CUDA=ON" in calls[index] for index in (0, 2))
    assert all("--target llama-server llama-tts" in calls[index] for index in (1, 3))


def test_wrong_vendor_pin_stops_before_patch_or_build(tmp_path):
    repo, source, env = _build_fixture(tmp_path, pin="0" * 40)
    before = (source / "common/json-schema-to-grammar.cpp").read_bytes()
    result = subprocess.run(["bash", str(repo / "scripts/build_llama_cpp.sh"), str(source)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 1 and f"pinned commit {PIN}" in result.stderr
    assert (source / "common/json-schema-to-grammar.cpp").read_bytes() == before
    assert not Path(env["BUILD_CALLS"]).exists()


def _text_observation(focused=True):
    return {"state": {"nodes": [{"id": "button", "generation": "1", "scope_id": "scope", "role": "button",
                               "name": "Open", "visible": True, "enabled": True, "sensitive": False,
                               "operations": ["focus"], "focused": False},
                              {"id": "document", "generation": "1", "scope_id": "scope", "role": "textbox",
                               "name": "Document", "visible": True, "enabled": True, "sensitive": False,
                               "operations": ["type_text"], **({"focused": focused} if focused is not None else {})}]}}


def _text_phase(value):
    return {"kind": "ui", "steps": [
        {"id": "earlier", "verb": "focus", "target": {"node_id": "button"},
         "expect": [{"kind": "focused", "node_id": "$target"}]},
        {"id": "write", "verb": "type_text", "target": {"node_id": "document"},
         "args": {"text": value}, "expect": [{"kind": "focused", "node_id": "$target"}]}],
        "final_predicates": [{"kind": "focused", "node_id": "document"}]}


@pytest.mark.parametrize("value", ["", "\x00", "prefix\x00suffix"])
def test_full_canonical_host_validation_blocks_fallback_empty_or_nul_before_any_dispatch(value):
    allowed = {"native_ui"}
    schema = phase_response_schema(allowed)
    assert schema["$defs"]["step_args"]["oneOf"][7]["properties"]["text"]["pattern"] == r"^[^\u0000]+$"
    plan, dispatched = _text_phase(value), []
    # Prove the first step is supported: only the later malformed text blocks
    # this otherwise admissible phase, before either action can dispatch.
    assert Draft202012Validator(_grounded_phase_schema(allowed, _text_observation())).is_valid(_text_phase("valid"))
    events = list(run_native_goal("Owned goal", observe=_text_observation, compile_phase=lambda _: copy.deepcopy(plan),
                                 execute=lambda *args: dispatched.append(args), cancel=lambda: None,
                                 allowed_tools=allowed))
    assert not Draft202012Validator(schema).is_valid(plan)
    assert dispatched == []  # Including the otherwise valid first action.
    assert events[-1]["response"]["portal"]["native_task"]["status"] == "blocked"


@pytest.mark.parametrize("value", ['quote " and backslash \\', "line\ncontrol\tallowed", "café"])
def test_valid_decoded_text_preserves_host_schema_and_arguments(value):
    plan, dispatched = _text_phase(value), []
    Draft202012Validator(phase_response_schema({"native_ui"})).validate(plan)
    public_invoke = copy.deepcopy(plan)
    public_invoke["steps"][0]["verb"] = "invoke"
    assert Draft202012Validator(phase_response_schema({"native_ui"})).is_valid(public_invoke)
    assert not Draft202012Validator(_grounded_phase_schema({"native_ui"}, _text_observation())).is_valid(public_invoke)
    plans = iter([plan, {"kind": "complete", "message": "Claim"}])
    events = list(run_native_goal("Owned goal", observe=_text_observation, compile_phase=lambda _: copy.deepcopy(next(plans)),
                                 execute=lambda name, args: dispatched.append((name, copy.deepcopy(args))) or {"status": "complete"},
                                 cancel=lambda: None, allowed_tools={"native_ui"}))
    assert len(dispatched) == 1
    assert dispatched[0][1]["steps"] == plan["steps"]
    assert events[-1]["response"]["portal"]["native_task"]["status"] == "planner_claim"


@pytest.mark.parametrize("focused", [None, False])
def test_first_typing_requires_positive_focus_while_public_and_future_typing_remain_valid(focused):
    value = 'quote " and backslash \\ café'
    plan = _text_phase(value)
    first_typing = {**plan, "steps": [plan["steps"][1]]}
    assert parse_step(first_typing["steps"][0])["args"]["text"] == value
    assert Draft202012Validator(phase_response_schema({"native_ui"})).is_valid(first_typing)
    assert Draft202012Validator(_grounded_phase_schema({"native_ui"}, _text_observation(True))).is_valid(first_typing)
    assert Draft202012Validator(_grounded_phase_schema({"native_ui"}, _text_observation(focused))).is_valid(plan)
    dispatched = []
    events = list(run_native_goal("Owned goal", observe=lambda: _text_observation(focused),
                                 compile_phase=lambda _: copy.deepcopy(first_typing),
                                 execute=lambda *args: dispatched.append(args), cancel=lambda: None,
                                 allowed_tools={"native_ui"}))
    assert dispatched == []
    assert events[-1]["response"]["portal"]["native_task"]["status"] == "blocked"


def test_host_rejects_decoded_short_surrogate_pair_even_when_native_units_satisfy_minimum():
    # Companion native regression records the existing two-wire-unit admission.
    # The decoded schema authority still rejects this one-codepoint value.
    value = json.loads(r'"\ud83d\ude00"')
    assert len(value) == 1
    assert not Draft202012Validator({"type": "string", "minLength": 2, "maxLength": 2}).is_valid(value)
