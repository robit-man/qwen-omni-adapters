#!/usr/bin/env python3
"""Owned GTK/gedit task gate through the authenticated computer_use tool.

--plan makes exactly one real model planning call. Without it, this is a
separately labeled deterministic executor baseline. The host creates, focuses
and closes only its own windows; all task input uses finite desktop actions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime-data"
FIELDS = {"Request title": "Native semantic dispatch", "Owner": "Roko Fixture",
          "Reference": "NATIVE-4827", "Count": "3", "Due date": "2026-10-14",
          "Contact": "fixture@example.test", "Notes": "Preserve exact bytes.",
          "Output name": "native-report.json"}
CHECKS = {"Ready": True, "Include audit": True, "Urgent": False}
SELECTIONS = {"Priority": "High", "Channel": "Local"}


def expected_data() -> dict[str, Any]:
    return {"fields": dict(FIELDS), "checks": dict(CHECKS), "selections": dict(SELECTIONS)}


def artifact_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


class NativeFixtureState:
    """The application owns its state and bytes; the agent cannot assert success."""

    def __init__(self, root: Path, marker: str) -> None:
        self.root, self.marker, self.stage = root, marker, "editing"

    def advance(self, operation: str, data: dict[str, Any]) -> str:
        required = {"Validate": "editing", "Preview": "validated", "Save": "previewed"}
        if self.stage != required.get(operation) or data != expected_data():
            return "FIXTURE-REJECTED"
        self.stage = {"Validate": "validated", "Preview": "previewed", "Save": "saved"}[operation]
        if operation == "Save":
            raw = artifact_bytes(data)
            (self.root / "native-report.json").write_bytes(raw)
            (self.root / "fixture-receipt.json").write_text(json.dumps({
                "stage": self.stage, "marker": self.marker,
                "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            }) + "\n")
        return f"{self.marker}:{self.stage}"


def _gtk_fixture(root: Path) -> None:
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk

    config = json.loads((root / "config.json").read_text())
    state = NativeFixtureState(root, config["marker"])
    window = Gtk.Window(title=config["title"])
    window.set_default_size(760, 620)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    outer.set_border_width(15)
    window.add(outer)
    grid = Gtk.Grid(column_spacing=12, row_spacing=8)
    outer.pack_start(grid, False, False, 0)
    entries = {}
    for index, name in enumerate(FIELDS):
        row, col = divmod(index, 2)
        entry = Gtk.Entry()
        entry.get_accessible().set_name(name)
        entries[name] = entry
        grid.attach(Gtk.Label(label=name), col * 2, row, 1, 1)
        grid.attach(entry, col * 2 + 1, row, 1, 1)
    checks = {}
    for name, wanted in CHECKS.items():
        control = Gtk.CheckButton(label=name)
        control.set_active(not wanted)
        checks[name] = control
        outer.pack_start(control, False, False, 0)
    choices, choice_box = {}, Gtk.Box(spacing=25)
    outer.pack_start(choice_box, False, False, 0)
    for name, options in {"Priority": ["Low", "Normal", "High"], "Channel": ["Remote", "Local", "Archive"]}.items():
        column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        column.pack_start(Gtk.Label(label=name), False, False, 0)
        selection = Gtk.ListBox()
        selection.get_accessible().set_name(name)
        selection.set_selection_mode(Gtk.SelectionMode.SINGLE)
        for option in options:
            row = Gtk.ListBoxRow()
            row.get_accessible().set_name(option)
            row.add(Gtk.Label(label=option))
            selection.add(row)
        choices[name] = selection
        column.pack_start(selection, True, True, 0)
        choice_box.pack_start(column, True, True, 0)
    status = Gtk.Label(label="Editing owned fixture")
    status.get_accessible().set_description(
        "Successful Validate changes this status to " + config["marker"] + ":validated; "
        "successful Preview changes it to " + config["marker"] + ":previewed; "
        "successful Save changes it to " + config["marker"] + ":saved."
    )
    buttons, button_box = {}, Gtk.Box(spacing=10)
    outer.pack_start(button_box, False, False, 0)
    outer.pack_start(status, False, False, 0)

    def collect() -> dict[str, Any]:
        return {"fields": {name: widget.get_text() for name, widget in entries.items()},
                "checks": {name: widget.get_active() for name, widget in checks.items()},
                "selections": {name: (widget.get_selected_row().get_accessible().get_name()
                    if widget.get_selected_row() else "") for name, widget in choices.items()}}

    def clicked(button: Any, operation: str) -> None:
        button.set_sensitive(False)
        status.set_text(f"{operation} pending")

        def finish() -> bool:
            status.set_text(state.advance(operation, collect()))
            if state.stage == "validated":
                buttons["Preview"].set_sensitive(True)
            elif state.stage == "previewed":
                buttons["Save"].set_sensitive(True)
            return False

        GLib.timeout_add(250, finish)

    for name in ("Validate", "Preview", "Save"):
        button = Gtk.Button(label=name)
        button.set_sensitive(name == "Validate")
        button.connect("clicked", clicked, name)
        buttons[name] = button
        button_box.pack_start(button, True, True, 0)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()
    Gtk.main()


def baseline_plan(marker: str) -> dict[str, Any]:
    steps = []
    for name, value in FIELDS.items():
        steps.append({"id": f"field-{len(steps)}", "verb": "set_value", "target": {"role": "textbox", "name": name},
                      "args": {"value": value}, "expect": [{"kind": "value", "node_id": "$target", "value": value}]})
    for name, checked in CHECKS.items():
        steps.append({"id": f"check-{len(steps)}", "verb": "set_checked", "target": {"role": "checkbox", "name": name},
                      "args": {"checked": checked}, "expect": [{"kind": "checked", "node_id": "$target", "checked": checked}]})
    for name, option in SELECTIONS.items():
        steps.append({"id": f"select-{len(steps)}", "verb": "select", "target": {"role": "listbox", "name": name},
                      "args": {"option": option}, "expect": [{"kind": "selected", "node_id": "$target", "values": [option]}]})
    for name, stage in (("Validate", "validated"), ("Preview", "previewed"), ("Save", "saved")):
        steps.append({"id": f"commit-{len(steps)}", "verb": "invoke", "target": {"role": "button", "name": name},
                      "args": {}, "expect": [{"kind": "text", "text": f"{marker}:{stage}"}]})
    return {"steps": steps, "final_predicates": [{"kind": "text", "text": f"{marker}:saved"}]}


def validate_plan(value: Any, marker: str, editor: dict[str, Any] | None = None) -> dict[str, Any]:
    from qwen_omni_adapters.computer_use import parse_step

    if not isinstance(value, dict) or set(value) != ({"steps", "final_predicates"} | ({"editor_steps"} if editor else set())):
        raise ValueError("Planner must return only the bounded workflow object")
    baseline = baseline_plan(marker)
    expected = {step["target"]["name"]: step for step in baseline["steps"]}
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(expected):
        raise ValueError("The native task requires all 16 distinct semantic operations")
    seen, ids = set(), set()
    for raw in raw_steps:
        step = parse_step(raw)
        name = step["target"].get("name")
        reference = expected.get(name)
        if (reference is None or name in seen or step["id"] in ids or step["target"] != reference["target"]
                or step["verb"] != reference["verb"] or step["args"] != reference["args"]
                or reference["expect"][0] not in step["expect"]):
            raise ValueError("A proposed action exceeds the owned fixture task or omits its real effect")
        seen.add(name)
        ids.add(step["id"])
    if value.get("final_predicates") != baseline["final_predicates"]:
        raise ValueError("Independent terminal marker must be the final criterion")
    if editor is not None:
        if not isinstance(value.get("editor_steps"), list) or len(value["editor_steps"]) != 2:
            raise ValueError("The editor task requires an edit and a save")
        for raw, reference in zip(value["editor_steps"], editor["steps"], strict=True):
            step = parse_step(raw)
            if any(step[key] != reference[key] for key in ("verb", "target", "args", "expect")):
                raise ValueError("Editor actions exceed the owned note and exact Save contract")
    return copy.deepcopy(value)


def model_plan(client: Any, endpoint: str, model: str, marker: str, editor: dict[str, Any] | None) -> dict[str, Any]:
    task = {"fields": FIELDS, "checks": CHECKS, "selections": SELECTIONS,
            "commit_sequence": [{"button": name, "expected_text": f"{marker}:{stage}"}
                for name, stage in (("Validate", "validated"), ("Preview", "previewed"), ("Save", "saved"))],
            "terminal_text": f"{marker}:saved", "editor": editor}
    prompt = ("Plan the owned desktop task as JSON only: {steps:[...],final_predicates:[...]"
              + (",editor_steps:[...]" if editor else "") + "}. Create all 16 GTK steps from the task data. "
              "Use textbox/set_value args {value}, checkbox/set_checked args {checked}, listbox/select args {option}, "
              "button/invoke args {}. Each step has id,verb,target:{role,name},args,expect. "
              "Field effects are {kind:value,node_id:$target,value:...}; checkbox effects use kind:checked and checked; "
              "selection effects use kind:selected, node_id:$target, values:[option]. Button effects use kind:text,text. "
              "All JSON keys and strings need double quotes. final_predicates is [{kind:text,text:terminal_text}]. "
              "Validate, then Preview, then Save; these are delayed state transitions. "
              "If editor is present, include its exact two typed steps with your own IDs as editor_steps. "
              "Task data: " + json.dumps(task, ensure_ascii=False))
    response = client.post(endpoint + "/api/chat", json={"model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}], "portal_auto_tools": False,
        "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
        "response_modalities": ["text"], "speech_mode": "never", "options": {"num_predict": 8192}})
    response.raise_for_status()
    body = response.json()
    if body.get("model") != model or body.get("message", {}).get("tool_calls"):
        raise ValueError("The selected model must return a plan without executing tools")
    content = body.get("message", {}).get("content", "").strip()
    if content.startswith("```json\n") and content.endswith("```"):
        content = content[8:-3].strip()
    return validate_plan(json.loads(content), marker, editor)


def tool(client: Any, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = client.post(endpoint + "/api/tools/computer_use/call", json={"arguments": {"surface": "desktop", **arguments}})
    response.raise_for_status()
    result = response.json()["result"]
    if "error" in result and "outcomes" not in result:
        raise RuntimeError("Desktop tool failed: " + str(result.get("code", result["error"])))
    return result


def verify_artifact(root: Path, marker: str) -> dict[str, Any]:
    expected = artifact_bytes(expected_data())
    raw = (root / "native-report.json").read_bytes()
    receipt = json.loads((root / "fixture-receipt.json").read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if raw != expected or receipt != {"stage": "saved", "marker": marker, "sha256": digest, "size_bytes": len(raw)}:
        raise RuntimeError("Independent native artifact bytes/receipt verification failed")
    return {"path": str(root / "native-report.json"), "sha256": digest, "size_bytes": len(raw), "verified": True}


def _fixed_command(arguments: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(arguments, env=environment, capture_output=True, text=True, timeout=3, check=False)  # noqa: S603
    if result.returncode:
        raise RuntimeError("Owned fixture window setup failed")
    return result.stdout.strip()


def focus_process(process: subprocess.Popen, environment: dict[str, str]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            windows = _fixed_command(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)], environment).splitlines()
            if len(windows) == 1 and windows[0].isdecimal():
                _fixed_command(["xdotool", "windowactivate", "--sync", windows[0]], environment)
                return
        except RuntimeError:
            pass
        time.sleep(0.1)
    raise RuntimeError("Owned process did not expose exactly one visible window")


def observe_owned(client: Any, endpoint: str, title: str) -> dict[str, Any]:
    for _ in range(20):
        result = tool(client, endpoint, {"operation": "observe"})
        state = result["state"]
        if any(node["role"] == "window" and title in node["name"] for node in state["nodes"]):
            return state
        time.sleep(0.1)
    raise RuntimeError("Desktop observation does not belong to the owned fixture")


def editor_contract(state: dict[str, Any], text: str) -> dict[str, Any]:
    eligible = [node for node in state["nodes"] if node["visible"] and node["enabled"] and not node["sensitive"]]
    editors = [node for node in eligible if "set_value" in node["operations"] and node["role"] == "textbox"]
    windows = [node for node in eligible if node["role"] == "window" and "key_chord" in node["operations"]]
    if len(editors) != 1 or len(windows) != 1:
        raise RuntimeError("Owned gedit lacks a unique editable semantic document/window")
    return {"steps": [
        {"id": "editor-write", "verb": "set_value", "target": {"role": "textbox", "name": editors[0]["name"]},
         "args": {"value": text}, "expect": [{"kind": "value", "node_id": "$target", "value": text}]},
        {"id": "editor-save", "verb": "key_chord", "target": {"role": "window"},
         "args": {"keys": ["ctrl", "s"]}, "expect": [{"kind": "visible", "role": "window", "name": windows[0]["name"], "visible": True}]},
    ], "final_predicates": [{"kind": "text", "text": text.strip()}]}


def _scope_steps(steps: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    steps = copy.deepcopy(steps)
    for step in steps:
        step["target"]["scope_id"] = state["scope"]["id"]
    return steps


def speech_instruction(marker: str, editor_text: str | None = None) -> str:
    parts = ["Complete the owned native dispatch form using semantic desktop controls."]
    parts.extend(f"Set the {name} field to {value}." for name, value in FIELDS.items())
    parts.extend(f"Set the {name} checkbox to {'checked' if wanted else 'unchecked'}." for name, wanted in CHECKS.items())
    parts.extend(f"Select {option} in {name}." for name, option in SELECTIONS.items())
    parts.append(f"Click Validate, then Preview, then Save. Wait for each operation's observed effect. The final marker must be {marker}:saved.")
    if editor_text is not None:
        parts.append("Then replace the owned gedit document with exactly two lines. The first line is "
                     + editor_text.splitlines()[0] + ". The second line is " + editor_text.splitlines()[1]
                     + ". End both lines with a newline. Save with Control S and verify the original unmodified window title returns.")
    return " ".join(parts)


def main() -> int:
    import httpx

    from portal.desktop import desktop_subprocess_environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8920")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    plan_mode = parser.add_mutually_exclusive_group()
    plan_mode.add_argument("--plan", action="store_true")
    plan_mode.add_argument("--speech", action="store_true", help="Actual deployed TTS WAV → ASR → model plan; distinct from a physical microphone")
    parser.add_argument("--gedit", action="store_true")
    args = parser.parse_args()
    marker = "NATIVE-COMPLEX-" + secrets.token_hex(5).upper()
    root = RUNTIME / ("native-computer-use-" + secrets.token_hex(6))
    root.mkdir(parents=True)
    title = "Owned native dispatch " + marker
    (root / "config.json").write_text(json.dumps({"marker": marker, "title": title}) + "\n")
    environment = desktop_subprocess_environment()
    processes, started = [], time.monotonic()
    report: dict[str, Any] = {"schema": "robit.omni.native-computer-use-live.v1", "model": args.model,
        "mode": "generated_tts_audio_plan" if args.speech else "real_model_plan" if args.plan else "deterministic_baseline", "status": "failed", "plan_calls": 0}
    with httpx.Client(headers={"Authorization": "Bearer " + args.token_file.read_text().strip()},
                      cookies={"omni_portal_session": secrets.token_urlsafe(24)}, timeout=600) as client:
        try:
            fixture = subprocess.Popen(["/usr/bin/python3", str(Path(__file__).resolve()), "--gtk-fixture", str(root)],
                env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # noqa: S603
            processes.append(fixture)
            editor, editor_state, editor_spec = None, None, None
            editor_path, editor_text = root / "editor-note.txt", marker + "\nVerified through native gedit Save.\n"
            if args.gedit:
                editor_path.write_text("Owned initial note.\n")
                editor = subprocess.Popen(["gedit", "--standalone", "--new-window", str(editor_path)],
                    env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # noqa: S603
                processes.append(editor)
                focus_process(editor, environment)
                editor_state = observe_owned(client, args.endpoint, editor_path.name)
                editor_spec = editor_contract(editor_state, editor_text)
            focus_process(fixture, environment)
            initial = observe_owned(client, args.endpoint, title)
            if args.speech:
                from runtime.computer_use_speech import spoken_plan

                report["plan_calls"] = 1
                compiled, speech = spoken_plan(client, args.endpoint, args.model,
                    speech_instruction(marker, editor_text if editor_spec else None),
                    {"ui": initial, "editor_initial_ui": editor_state,
                     "terminal_criteria": baseline_plan(marker)["final_predicates"]}, root,
                    extra_keys=",editor_steps:[...]" if editor_spec else "")
                plan = validate_plan(compiled, marker, editor_spec)
                report["speech_evidence"] = {"source": "generated_tts_audio", "physical_microphone": False,
                    "typed_goal_substituted": speech["speech"]["typed_goal_substituted"],
                    "wav_sha256": speech["tts"]["wav_sha256"],
                    "asr_transcript_sha256": hashlib.sha256(speech["speech"]["asr_transcript"].encode()).hexdigest()}
            elif args.plan:
                report["plan_calls"] = 1
                plan = model_plan(client, args.endpoint, args.model, marker, editor_spec)
            else:
                plan = baseline_plan(marker)
                if editor_spec:
                    plan["editor_steps"] = editor_spec["steps"]
            (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            workflow = tool(client, args.endpoint, {"operation": "workflow", "steps": _scope_steps(plan["steps"], initial),
                "final_predicates": plan["final_predicates"]})
            report["gtk_steps"] = [outcome["status"] for outcome in workflow.get("outcomes", [])]
            if workflow.get("status") != "complete":
                raise RuntimeError("GTK workflow: " + str(workflow.get("reason", workflow.get("status"))))
            report["gtk_artifact"] = verify_artifact(root, marker)
            report["gtk_terminal_receipt"] = workflow["terminal_receipt"]
            if editor is not None:
                guard_step = baseline_plan(marker)["steps"][0]
                offered = tool(client, args.endpoint, {"operation": "observe", "step": guard_step})
                candidate = next(item for item in offered["candidates"] if item["kind"] == "action")
                focus_process(editor, environment)
                rejected = tool(client, args.endpoint, {"operation": "execute", "action_id": candidate["id"]})
                if rejected.get("status") != "rejected":
                    raise RuntimeError("Focus-switch stale candidate was not rejected")
                report["focus_switch_guard"] = "rejected_without_dispatch"
                assert editor_state is not None and editor_spec is not None
                current = observe_owned(client, args.endpoint, editor_path.name)
                edited = tool(client, args.endpoint, {"operation": "workflow", "steps": _scope_steps(plan["editor_steps"], current),
                    "final_predicates": editor_spec["final_predicates"]})
                report["gedit_steps"] = [outcome["status"] for outcome in edited.get("outcomes", [])]
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and editor_path.read_bytes() != editor_text.encode():
                    time.sleep(0.1)
                raw = editor_path.read_bytes()
                if edited.get("status") != "complete" or raw != editor_text.encode():
                    raise RuntimeError("Independent gedit Save exact-byte readback failed")
                report["gedit_artifact"] = {"path": str(editor_path), "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(), "verified": True}
            report["status"] = "complete"
        except Exception as error:
            report["error"] = str(error)[:1000]
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=2)
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "artifact_directory": str(root)}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--gtk-fixture":
        _gtk_fixture(Path(sys.argv[2]))
    else:
        sys.path.insert(0, str(REPO))
        raise SystemExit(main())
