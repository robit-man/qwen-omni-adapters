"""Pinned native grammar integration and unchanged host dispatch authority."""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from portal.ui_task import phase_response_schema, run_native_goal

ROOT = Path(__file__).resolve().parents[1]
PIN = "458681e1d5d4a29a1463c4732e03226cf384b997"
PATCH = "llama.cpp-json-pattern-safety.patch"


def _patch_preimage(text: str, source: Path) -> None:
    """Minimal apply fixture from old hunk lines; actual pinned proof is separate."""
    current, position = None, 0
    files: dict[str, list[str]] = {}
    for line in text.splitlines(keepends=True):
        if line.startswith("--- a/"):
            current = line[6:].strip()
            files[current] = []
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


def _build_fixture(tmp_path: Path, *, pin: str = PIN) -> tuple[Path, Path, dict[str, str]]:
    repo, source, commands = tmp_path / "repo", tmp_path / "source", tmp_path / "commands"
    (repo / "scripts").mkdir(parents=True)
    (repo / "patches").mkdir()
    source.mkdir()
    commands.mkdir()
    shutil.copyfile(ROOT / "scripts/build_llama_cpp.sh", repo / "scripts/build_llama_cpp.sh")
    shutil.copyfile(ROOT / "patches" / PATCH, repo / "patches" / PATCH)
    _patch_preimage((ROOT / "patches" / PATCH).read_text(), source)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    for name, text in {"common/arg.cpp": '"--tts-stream-frames"\n"--tts-persistent"\n',
                       "tools/mtmd/clip.cpp": "state_out describes this exact output boundary\n"}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git = commands / "git"
    git.write_text(f'#!/bin/sh\nif [ "$3" = rev-parse ]; then printf "%s\\n" "{pin}"; else exec {shutil.which("git")} "$@"; fi\n')
    cmake = commands / "cmake"
    cmake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$BUILD_CALLS"\n')
    git.chmod(0o755)
    cmake.chmod(0o755)
    target = tmp_path / "build/bin/llama-tts"
    target.parent.mkdir(parents=True)
    target.write_text('#!/bin/sh\nprintf "%s\\n" "--tts-stream-frames --tts-persistent"\n')
    target.chmod(0o755)
    env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
           "BUILD_CALLS": str(tmp_path / "cmake-calls"), "LLAMA_CPP_BUILD_DIR": str(target.parent.parent),
           "LLAMA_CPP_BUILD_JOBS": "1"}
    return repo, source, env


def test_linux_builder_applies_independent_pattern_patch_once_and_keeps_pin(tmp_path):
    repo, source, env = _build_fixture(tmp_path)
    command = ["bash", str(repo / "scripts/build_llama_cpp.sh"), str(source)]
    first = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*.cpp")}
    second = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert "Applied JSON pattern string safety patch" in first.stdout
    assert "JSON pattern string safety patch is already applied" in second.stdout
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


def _text_phase(value):
    return {"kind": "ui", "steps": [
        {"id": "earlier", "verb": "invoke", "target": {"node_id": "button"},
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
    events = list(run_native_goal("Owned goal", observe=lambda: {}, compile_phase=lambda _: copy.deepcopy(plan),
                                 execute=lambda *args: dispatched.append(args), cancel=lambda: None,
                                 allowed_tools=allowed))
    assert not Draft202012Validator(schema).is_valid(plan)
    assert dispatched == []  # Including the otherwise valid first action.
    assert events[-1]["response"]["portal"]["native_task"]["status"] == "blocked"


@pytest.mark.parametrize("value", ['quote " and backslash \\', "line\ncontrol\tallowed", "café"])
def test_valid_decoded_text_preserves_host_schema_and_arguments(value):
    plan, dispatched = _text_phase(value), []
    Draft202012Validator(phase_response_schema({"native_ui"})).validate(plan)
    plans = iter([plan, {"kind": "complete", "message": "Claim"}])
    events = list(run_native_goal("Owned goal", observe=lambda: {}, compile_phase=lambda _: copy.deepcopy(next(plans)),
                                 execute=lambda name, args: dispatched.append((name, copy.deepcopy(args))) or {"status": "complete"},
                                 cancel=lambda: None, allowed_tools={"native_ui"}))
    assert len(dispatched) == 1
    assert dispatched[0][1]["steps"] == plan["steps"]
    assert events[-1]["response"]["portal"]["native_task"]["status"] == "planner_claim"
