#!/usr/bin/env python3
"""Owned native gestures, real policy ambiguity and visual-point acceptance.

--executor-probe exercises only native input/effects with no model service.
--plan uses a real Omni plan; --speech uses real TTS→ASR→Omni input. Full runs
require actual Kev selection between two Continue controls and Moondream
grounding of an inaccessible canvas target. Raw model coordinates are forbidden.
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
PROBE_FLAGS = ["single", "double", "context", "selection", "scroll", "drop"]
FULL_FLAGS = [*PROBE_FLAGS, "visual", "local_save"]


class GestureState:
    def __init__(self, root: Path, marker: str, required: list[str]) -> None:
        self.root, self.marker, self.required = root, marker, required
        self.flags: dict[str, bool] = {}
        self.events: list[dict[str, Any]] = []
        self.wrong_choice = False
        self.publish()

    def publish(self) -> None:
        value = {"marker": self.marker, "flags": self.flags, "events": self.events,
                 "wrong_choice": self.wrong_choice,
                 "complete": not self.wrong_choice and all(self.flags.get(key) for key in self.required)}
        (self.root / "gesture-receipt.json").write_text(json.dumps(value, sort_keys=True) + "\n")

    def mark(self, name: str, details: dict[str, Any] | None = None) -> None:
        self.events.append({"event": name, "details": details or {}})
        if name == "wrong_continue":
            self.wrong_choice = True
        else:
            self.flags[name] = True
        self.publish()


def gtk_fixture(root: Path) -> None:
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    gi.require_version("Atk", "1.0")
    from gi.repository import Atk, Gdk, Gtk

    config = json.loads((root / "config.json").read_text())
    state = GestureState(root, config["marker"], config["required"])
    primary, secondary = Gtk.Window(title=config["title"]), Gtk.Window(title="Owned gesture secondary")
    primary.set_default_size(880, 720)
    secondary.add(Gtk.Label(label="Owned companion window; no user documents"))
    secondary.show_all()
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=9)
    outer.set_border_width(12)
    primary.add(outer)
    results = {name: Gtk.Label(label=f"{name} pending") for name in FULL_FLAGS}
    terminal = Gtk.Label(label="Native gesture task pending")
    for name, result in results.items():
        result.get_accessible().set_description(
            "After the application observes the actual " + name + " effect, this text becomes "
            + config["marker"] + ":" + name + "."
        )
    terminal.get_accessible().set_description(
        "After all required application effects and no wrong Continue choice, this text becomes "
        + config["marker"] + ":complete."
    )

    def mark(name: str, details: dict[str, Any] | None = None) -> None:
        state.mark(name, details)
        if name in results:
            results[name].set_text(config["marker"] + ":" + name)
        if not state.wrong_choice and all(state.flags.get(key) for key in state.required):
            terminal.set_text(config["marker"] + ":complete")
        elif state.wrong_choice:
            terminal.set_text("WRONG-CONTINUE-CHOSEN")

    areas = Gtk.Box(spacing=12)
    outer.pack_start(areas, False, False, 0)

    def area(name: str, label: str) -> Any:
        widget = Gtk.EventBox()
        widget.set_size_request(160, 55)
        widget.get_accessible().set_name(name)
        widget.get_accessible().set_role(Atk.Role.PANEL)
        widget.add(Gtk.Label(label=label))
        widget.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK)
        return widget

    single, double, context = area("Single target", "Click once"), area("Double target", "Double click"), area("Context target", "Right click")
    for widget in (single, double, context):
        areas.pack_start(widget, True, True, 0)
    single.connect("button-release-event", lambda _widget, event: mark("single", {"button": int(event.button)}) if event.button == 1 else None)
    presses = []

    def double_press(_widget: Any, event: Any) -> bool:
        if event.button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
            presses.append(time.monotonic())
            if len(presses) >= 2 and presses[-1] - presses[-2] < .8:
                mark("double", {"press_count": len(presses)})
        return False

    double.connect("button-press-event", double_press)
    popover = Gtk.Popover.new(context)
    accept = Gtk.Button(label="Accept context action")
    popover.add(accept)
    accept.connect("clicked", lambda _button: (mark("context"), popover.popdown()))

    def context_press(_widget: Any, event: Any) -> bool:
        if event.button == 3:
            popover.show_all()
            popover.popup()
            return True
        return False

    context.connect("button-press-event", context_press)
    sentence = Gtk.Entry()
    sentence.set_text("Native text selection proves real offsets.")
    sentence.get_accessible().set_name("Selectable sentence")
    outer.pack_start(sentence, False, False, 0)

    def selection_changed(*_args: Any) -> None:
        bounds = sentence.get_selection_bounds()
        if len(bounds) == 2 and tuple(bounds) == (0, 6):
            mark("selection", {"start": 0, "end": 6})

    sentence.connect("notify::selection-bound", selection_changed)
    sentence.connect("notify::cursor-position", selection_changed)
    scroll = Gtk.ScrolledWindow()
    scroll.set_size_request(-1, 85)
    scroll.get_accessible().set_name("Scroll area")
    rows = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    for index in range(15):
        rows.pack_start(Gtk.Label(label=f"Owned scroll row {index+1}"), False, False, 0)
    scroll.add(rows)
    outer.pack_start(scroll, False, False, 0)
    adjustment = scroll.get_vadjustment()
    adjustment.connect("value-changed", lambda value: mark("scroll", {"offset": round(value.get_value(), 2)}) if value.get_value() > 0 else None)
    drag_row = Gtk.Box(spacing=90)
    source, destination = area("Drag source", "Drag token"), area("Drop target", "Drop token here")
    targets = [Gtk.TargetEntry.new("application/x-owned-native-gesture", Gtk.TargetFlags.SAME_APP, 0)]
    source.drag_source_set(Gdk.ModifierType.BUTTON1_MASK, targets, Gdk.DragAction.COPY)
    destination.drag_dest_set(Gtk.DestDefaults.ALL, targets, Gdk.DragAction.COPY)
    source.connect("drag-data-get", lambda _widget, _ctx, selection, _info, _stamp: selection.set_text(config["marker"], -1))

    def dropped(_widget: Any, ctx: Any, _x: int, _y: int, selection: Any, _info: int, stamp: int) -> None:
        valid = selection.get_text() == config["marker"]
        if valid:
            mark("drop", {"nonce_sha256": hashlib.sha256(config["marker"].encode()).hexdigest()})
        ctx.finish(valid, False, stamp)

    destination.connect("drag-data-received", dropped)
    drag_row.pack_start(source, True, True, 0)
    drag_row.pack_start(destination, True, True, 0)
    outer.pack_start(drag_row, False, False, 0)
    canvas = Gtk.DrawingArea()
    canvas.set_size_request(640, 100)
    canvas.get_accessible().set_name("Visual canvas")
    canvas.add_events(Gdk.EventMask.BUTTON_RELEASE_MASK)

    def draw(_widget: Any, cairo: Any) -> bool:
        cairo.set_source_rgb(.1, .12, .16)
        cairo.paint()
        cairo.set_source_rgb(.1, .4, .9)
        cairo.rectangle(150, 25, 50, 50)
        cairo.fill()
        cairo.set_source_rgb(.82, .6, .13)
        cairo.arc(480, 50, 28, 0, 6.283185)
        cairo.fill()
        return False

    canvas.connect("draw", draw)
    canvas.connect("button-release-event", lambda _widget, event: mark("visual", {"x": round(event.x), "y": round(event.y)})
                   if event.button == 1 and (event.x-480)**2 + (event.y-50)**2 < 28**2 else None)
    outer.pack_start(canvas, False, False, 0)
    continuation = Gtk.Box(spacing=180)
    for label, description, flag in (("Discard draft", "Discard local draft without saving", "wrong_continue"),
                                     ("Save locally", "Save local draft and confirm LOCAL-SAVE-CONFIRMED", "local_save")):
        group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        group.pack_start(Gtk.Label(label=label), False, False, 0)
        button = Gtk.Button(label="Continue")
        button.get_accessible().set_description(description)
        button.connect("clicked", lambda _button, name=flag: mark(name))
        group.pack_start(button, False, False, 0)
        continuation.pack_start(group, True, True, 0)
    outer.pack_start(continuation, False, False, 0)
    result_grid = Gtk.Grid(column_spacing=20)
    for index, label in enumerate(results.values()):
        result_grid.attach(label, index % 4, index // 4, 1, 1)
    outer.pack_start(result_grid, False, False, 0)
    outer.pack_start(terminal, False, False, 0)
    primary.connect("destroy", Gtk.main_quit)
    primary.show_all()
    primary.present()
    Gtk.main()


def baseline_plan(state: dict[str, Any], marker: str, title: str, full: bool) -> dict[str, Any]:
    destination = next(node for node in state["nodes"] if node["name"] == "Drop target" and node["role"] == "panel")
    steps = []

    def add(verb: str, name: str, role: str, args: dict[str, Any], effect: dict[str, Any]) -> None:
        steps.append({"id": f"gesture-{len(steps)}", "verb": verb, "target": {"role": role, "name": name}, "args": args, "expect": [effect]})

    for verb, name, flag in (("click", "Single target", "single"), ("double_click", "Double target", "double")):
        add(verb, name, "panel", {}, {"kind": "text", "text": marker+":"+flag})
    add("right_click", "Context target", "panel", {}, {"kind": "visible", "role": "button", "name": "Accept context action", "visible": True})
    add("invoke", "Accept context action", "button", {}, {"kind": "text", "text": marker+":context"})
    add("select_text", "Selectable sentence", "textbox", {"start": 0, "end": 6}, {"kind": "text_selection", "node_id": "$target", "start": 0, "end": 6})
    add("scroll", "Scroll area", "scroll_pane", {"direction": "down"}, {"kind": "text", "text": marker+":scroll"})
    add("drag_drop", "Drag source", "panel", {"to_node_id": destination["id"]}, {"kind": "text", "text": marker+":drop"})
    if full:
        add("click", "amber circle", "button", {}, {"kind": "text", "text": marker+":visual"})
        add("invoke", "Continue", "button", {}, {"kind": "text", "text": marker+":local_save"})
    for name in ("Owned gesture secondary", title):
        add("switch_window", name, "window", {}, {"kind": "active_window", "node_id": "$target"})
    return {"steps": steps, "final_predicates": [{"kind": "text", "text": marker+":complete"}]}


def validate_plan(value: Any, reference: dict[str, Any]) -> dict[str, Any]:
    from qwen_omni_adapters.computer_use import parse_step

    if not isinstance(value, dict) or set(value) != {"steps", "final_predicates"} or len(value.get("steps", [])) != len(reference["steps"]):
        raise ValueError("Expected the complete bounded native gesture workflow")
    for proposed, required in zip(value["steps"], reference["steps"], strict=True):
        parsed = parse_step(proposed)
        if any(parsed[key] != required[key] for key in ("verb", "target", "args", "expect")):
            raise ValueError("Native gesture plan exceeds exact owned task/effect authority")
    if value["final_predicates"] != reference["final_predicates"]:
        raise ValueError("A fresh independent native terminal marker is required")
    return copy.deepcopy(value)


def focus_owned(process: subprocess.Popen, title: str, environment: dict[str, str]) -> None:
    from runtime.verify_native_computer_use import _fixed_command

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Owned gesture fixture exited ({process.returncode}); inspect fixture-stderr.log")
        try:
            candidates = _fixed_command(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)], environment).splitlines()
            matches = [window for window in candidates if window.isdecimal() and
                       _fixed_command(["xdotool", "getwindowname", window], environment) == title]
            if len(matches) == 1:
                _fixed_command(["xdotool", "windowactivate", "--sync", matches[0]], environment)
                return
        except RuntimeError:
            pass
        time.sleep(.1)
    raise RuntimeError("Owned primary gesture window was not uniquely mapped")


def main() -> int:
    import httpx

    from portal.desktop import desktop_subprocess_environment
    from runtime.verify_native_computer_use import observe_owned

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8920")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--model")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--executor-probe", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--speech", action="store_true")
    args = parser.parse_args()
    if not args.executor_probe and (not args.model or not args.token_file):
        parser.error("Full model tests require the exact --model and --token-file")
    marker = "NATIVE-GESTURE-" + secrets.token_hex(4).upper()
    title = "Owned native gestures " + marker
    root = REPO / "runtime-data" / ("native-gestures-" + secrets.token_hex(6))
    root.mkdir(parents=True)
    required = PROBE_FLAGS if args.executor_probe else FULL_FLAGS
    (root / "config.json").write_text(json.dumps({"title": title, "marker": marker, "required": required}) + "\n")
    environment = desktop_subprocess_environment()
    stderr_log = (root / "fixture-stderr.log").open("wb")
    process = subprocess.Popen(["/usr/bin/python3", str(Path(__file__).resolve()), "--gtk-fixture", str(root)],
        env=environment, stdout=subprocess.DEVNULL, stderr=stderr_log, start_new_session=True)  # noqa: S603
    report: dict[str, Any] = {"schema": "robit.omni.native-gestures-live.v1", "status": "failed",
        "mode": "executor_only_no_models" if args.executor_probe else "generated_tts_audio" if args.speech else "real_model_plan"}
    adapter = None
    try:
        focus_owned(process, title, environment)
        if args.executor_probe:
            from portal.native_ui import NativeUiAdapter
            from qwen_omni_adapters.computer_use import RapidUiController

            adapter = NativeUiAdapter("owned-gesture-executor-probe")
            controller = RapidUiController(adapter, authorize=lambda *_args: True)
            initial = controller.observe()
            plan = baseline_plan(initial, marker, title, False)
            result = controller.run_workflow(plan["steps"], final_predicates=plan["final_predicates"])
        else:
            with httpx.Client(headers={"Authorization": "Bearer " + args.token_file.read_text().strip()},
                              cookies={"omni_portal_session": secrets.token_urlsafe(24)}, timeout=600) as client:
                initial = observe_owned(client, args.endpoint, title)
                reference = baseline_plan(initial, marker, title, True)
                instruction = ("In the owned native gesture window, click Single target once, double click Double target, "
                    "right click Context target then invoke Accept context action. Select characters zero through six "
                    "in Selectable sentence. Scroll Scroll area down. Drag Drag source onto Drop target. Click the amber "
                    "circle in the visual canvas using grounded vision. Invoke the Continue button that saves the local "
                    "draft, not the Continue that discards it. Switch to Owned gesture secondary, then back to " + title +
                    ". Verify every observed effect and the fresh complete marker.")
                context = {"ui": initial, "terminal_criteria": reference["final_predicates"]}
                if args.speech:
                    from runtime.computer_use_speech import spoken_plan

                    compiled, evidence = spoken_plan(client, args.endpoint, args.model, instruction, context, root)
                    report["speech_evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
                else:
                    response = client.post(args.endpoint + "/api/chat", json={"model": args.model, "stream": False,
                        "think": False, "portal_auto_tools": False, "format": "json", "speech_mode": "never",
                        "response_modalities": ["text"], "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
                        "messages": [{"role": "system", "content": "Return only a bounded JSON workflow with steps and final_predicates. "
                            "Each step has id,verb,target:{role,name},args,expect. Use the typed native contract and exact observed effects. "
                            + json.dumps({"contract": reference, "current_ui": initial})}, {"role": "user", "content": instruction}]})
                    response.raise_for_status()
                    body = response.json()
                    if body.get("model") != args.model:
                        raise RuntimeError("The native planner model identity changed")
                    compiled = json.loads(body["message"]["content"])
                plan = validate_plan(compiled, reference)
                response = client.post(args.endpoint + "/api/tools/native_ui/call", json={"arguments": {"operation": "workflow", **plan}})
                response.raise_for_status()
                result = response.json()["result"]
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        (root / "workflow-evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        report["action_statuses"] = [outcome["status"] for outcome in result.get("outcomes", [])]
        report["outcome_reasons"] = [outcome.get("reason") for outcome in result.get("outcomes", [])]
        report["workflow_evidence_sha256"] = hashlib.sha256((root / "workflow-evidence.json").read_bytes()).hexdigest()
        report["policy_receipts"] = result.get("policy_receipts", [])
        report["point_receipts"] = result.get("point_receipts", [])
        receipt_path = root / "gesture-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        report["fixture_flags"] = receipt["flags"]
        report["fixture_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        report["fixture_events"] = receipt["events"]
        if result.get("status") != "complete" or not receipt["complete"] or receipt["wrong_choice"]:
            raise RuntimeError("Native gesture effects incomplete: " + str(result.get("reason", result.get("status"))))
        if not args.executor_probe and (not report["policy_receipts"] or not report["point_receipts"]):
            raise RuntimeError("Full native acceptance requires actual policy and visual grounding receipts")
        report["status"] = "complete"
    except Exception as error:
        report["error"] = str(error)[:1000]
    finally:
        if adapter is not None:
            adapter.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        report["fixture_process_exit"] = process.returncode
        stderr_log.close()
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "artifact_directory": str(root)}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--gtk-fixture":
        gtk_fixture(Path(sys.argv[2]))
    else:
        sys.path.insert(0, str(REPO))
        raise SystemExit(main())
