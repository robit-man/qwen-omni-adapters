#!/usr/bin/env python3
"""Verify real browser gestures against an independently recorded local fixture.

The deterministic mode tests controller/Chromium mechanics. --speech additionally
requires deployed TTS -> actual audio ASR -> a model-produced canonical plan.
Neither a dispatched action nor a unit test is evidence of a live gesture pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from qwen_omni_adapters.computer_use import ComputerUseError, parse_predicates, parse_step

PHASES = ("double", "context", "drop", "selection", "shortcut", "scroll", "finish")
CONFIRMATIONS = (
    "Double click verified", "Context menu verified", "Drop verified",
    "Selection verified", "Shortcut verified", "Scroll verified", "All gestures verified",
)
SELECTION_TEXT = "Alpha bravo charlie delta"
INSTRUCTION = (
    "In the gesture fixture, double-click Double target and wait for Double click verified. "
    "Right-click Context target and wait for Context menu verified. "
    "Drag Drag source to Drop area and wait for Drop verified. "
    "Select the first five characters in Selection text, from offset zero to offset five, "
    "and wait for Selection verified. Press Control and K in Shortcut target and wait for Shortcut verified. "
    "Scroll down in Scroll panel and wait for Scroll verified. "
    "Click Finish gestures, then verify All gestures verified."
)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class GestureState:
    """Require the exact ordered browser event receipts, then write an artifact."""

    def __init__(self, artifact: Path | None = None) -> None:
        self.nonce = secrets.token_hex(16)
        self.marker = "BROWSER-GESTURES-PASS-" + secrets.token_hex(8).upper()
        self.artifact = artifact
        self.events: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.completed = False

    def _valid_event(self, phase: str, payload: dict[str, Any]) -> bool:
        if payload.get("nonce") != self.nonce or payload.get("trusted") is not True:
            return False
        if phase == "double":
            return (payload.get("event") == "dblclick" and type(payload.get("detail")) is int
                    and payload["detail"] == 2 and payload.get("button") == 0
                    and payload.get("click_details") == [1, 2])
        if phase == "context":
            return payload.get("event") == "contextmenu" and payload.get("button") == 2
        if phase == "drop":
            return (payload.get("event") == "drop" and payload.get("source") == "drag-source"
                    and payload.get("destination") == "drop-area" and payload.get("drag_nonce") == self.nonce
                    and payload.get("dragstart_trusted") is True)
        if phase == "selection":
            return (payload.get("event") == "select" and payload.get("focused") is True
                    and type(payload.get("start")) is int and payload["start"] == 0
                    and type(payload.get("end")) is int and payload["end"] == 5
                    and payload.get("selected_text") == "Alpha" and payload.get("value") == SELECTION_TEXT)
        if phase == "shortcut":
            return (payload.get("event") == "keydown" and payload.get("focused") is True
                    and payload.get("key") == "k" and payload.get("ctrl") is True
                    and all(payload.get(key) is False for key in ("alt", "shift", "meta")))
        if phase == "scroll":
            top, height, content = payload.get("top"), payload.get("height"), payload.get("content_height")
            return (payload.get("event") == "scroll" and type(top) in {int, float}
                    and type(height) in {int, float} and type(content) in {int, float}
                    and all(math.isfinite(number) for number in (top, height, content))
                    and height > 0 and content > height and 0 < top <= content - height)
        return phase == "finish" and payload.get("event") == "click" and payload.get("button") == 0

    def submit(self, phase: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            expected = PHASES[len(self.events)] if len(self.events) < len(PHASES) else ""
            accepted = not self.completed and phase == expected and self._valid_event(phase, payload)
            # Bound malformed/repeated traffic without obscuring accepted evidence.
            if len(self.attempts) < 100:
                self.attempts.append({"phase": phase, "accepted": accepted})
            if accepted:
                self.events.append({"sequence": len(self.events) + 1, "phase": phase,
                                    "observed_at_ns": time.monotonic_ns(), "event": copy.deepcopy(payload)})
                self.completed = phase == "finish"
                if self.completed and self.artifact is not None:
                    self.artifact.parent.mkdir(parents=True, exist_ok=True)
                    self.artifact.write_bytes(_bytes(self._summary()))
            return {"accepted": accepted,
                    "confirmation": CONFIRMATIONS[PHASES.index(phase)] if accepted else "Gesture rejected",
                    "marker": self.marker if self.completed else ""}

    def _summary(self) -> dict[str, Any]:
        return {"schema": "robit.omni.browser-gesture-server.v1", "completed": self.completed,
                "phase": "complete" if self.completed else PHASES[len(self.events)],
                "marker": self.marker if self.completed else "", "events": copy.deepcopy(self.events),
                "attempts": copy.deepcopy(self.attempts)}

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return self._summary()

    def verify_artifact(self) -> dict[str, Any]:
        """Read the file independently of both browser and controller responses."""
        if self.artifact is None or not self.completed or not self.artifact.is_file():
            return {"verified": False}
        raw = self.artifact.read_bytes()
        saved = json.loads(raw)
        verified = (saved.get("completed") is True and saved.get("marker") == self.marker
                    and [item.get("phase") for item in saved.get("events", [])] == list(PHASES)
                    and saved.get("events") == self.summary()["events"])
        return {"verified": verified, "path": str(self.artifact.resolve()),
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture_page(state: GestureState) -> str:
    return """<!doctype html><meta charset="utf-8"><title>Owned browser gesture fixture</title>
<style>body{font:16px sans-serif;max-width:900px;margin:16px auto;background:#f1f5f9;color:#17212f}
h1{font-size:23px}button,input,textarea{font:16px sans-serif;padding:10px}section{margin:14px 0}
.row{display:flex;gap:28px;align-items:center}.box{padding:20px;border:2px solid #456;background:white;min-width:170px}
#drag-source{cursor:grab}#drop-area{border-style:dashed}label{display:flex;gap:14px;align-items:center}
#scroll-panel{height:110px;overflow:auto;border:2px solid #456;background:white;width:600px}
#scroll-content{height:1000px;padding:12px}#status{min-height:24px;font-weight:bold}</style>
<h1>Browser gesture controls</h1><section class="row">
<button id="double-target">Double target</button><button id="context-target">Context target</button></section>
<section class="row"><div id="drag-source" role="button" aria-label="Drag source" tabindex="0" draggable="true" class="box">Drag source</div>
<div id="drop-area" role="region" aria-label="Drop area" tabindex="0" class="box">Drop area</div></section>
<section><label>Selection text<textarea id="selection-text" rows="1" cols="34">Alpha bravo charlie delta</textarea></label></section>
<section><label>Shortcut target<input id="shortcut-target" autocomplete="off" value="Shortcut ready"></label></section>
<section><div id="scroll-panel" role="region" aria-label="Scroll panel" tabindex="0"><div id="scroll-content">Scroll content</div></div></section>
<button id="finish" disabled>Finish gestures</button><div id="status" role="status">Ready for gestures</div><div id="marker"></div>
<script>
const nonce=__NONCE__, statusNode=document.getElementById('status');let queue=Promise.resolve();
function send(phase,details){queue=queue.then(async()=>{
 const response=await fetch('/event/'+phase,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({nonce,...details})});const result=await response.json();
 statusNode.textContent=result.confirmation;
 if(result.accepted&&phase==='scroll')document.getElementById('finish').disabled=false;
 if(result.marker)document.getElementById('marker').textContent=result.marker;
});return queue;}
let clicks=[];const doubleTarget=document.getElementById('double-target');
doubleTarget.addEventListener('click',event=>{clicks.push(event.detail);clicks=clicks.slice(-2);});
doubleTarget.addEventListener('dblclick',event=>send('double',{event:event.type,trusted:event.isTrusted,
 detail:event.detail,button:event.button,click_details:clicks}));
document.getElementById('context-target').addEventListener('contextmenu',event=>{
 event.preventDefault();send('context',{event:event.type,trusted:event.isTrusted,button:event.button});});
let dragTrusted=false;const source=document.getElementById('drag-source'),drop=document.getElementById('drop-area');
source.addEventListener('dragstart',event=>{dragTrusted=event.isTrusted;event.dataTransfer.effectAllowed='move';
 event.dataTransfer.setData('text/plain',nonce);});
drop.addEventListener('dragover',event=>{event.preventDefault();event.dataTransfer.dropEffect='move';});
drop.addEventListener('drop',event=>{event.preventDefault();send('drop',{event:event.type,trusted:event.isTrusted,
 source:source.id,destination:drop.id,drag_nonce:event.dataTransfer.getData('text/plain'),dragstart_trusted:dragTrusted});});
const selection=document.getElementById('selection-text');
selection.addEventListener('select',event=>send('selection',{event:event.type,trusted:event.isTrusted,
 focused:document.activeElement===selection,start:selection.selectionStart,end:selection.selectionEnd,
 selected_text:selection.value.slice(selection.selectionStart,selection.selectionEnd),value:selection.value}));
const shortcut=document.getElementById('shortcut-target');
shortcut.addEventListener('keydown',event=>{if(event.ctrlKey&&event.key.toLowerCase()==='k'){
 event.preventDefault();send('shortcut',{event:event.type,trusted:event.isTrusted,focused:document.activeElement===shortcut,
 key:event.key.toLowerCase(),ctrl:event.ctrlKey,alt:event.altKey,shift:event.shiftKey,meta:event.metaKey});}});
let scrolled=false;const panel=document.getElementById('scroll-panel');
panel.addEventListener('scroll',event=>{if(!scrolled&&panel.scrollTop>0){scrolled=true;
 send('scroll',{event:event.type,trusted:event.isTrusted,top:panel.scrollTop,height:panel.clientHeight,content_height:panel.scrollHeight});}});
document.getElementById('finish').addEventListener('click',event=>send('finish',{
 event:event.type,trusted:event.isTrusted,button:event.button}));
</script>""".replace("__NONCE__", json.dumps(state.nonce))


def fixture_handler(state: GestureState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            raw = fixture_page(state).encode() if self.path == "/" else b"Not found"
            self.send_response(200 if self.path == "/" else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not self.path.startswith("/event/") or not 0 < length <= 8192:
                    raise ValueError("Bounded event required")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Event must be an object")
                result = state.submit(self.path.removeprefix("/event/"), payload)
            except (ValueError, TypeError):
                result = {"accepted": False, "confirmation": "Gesture rejected", "marker": ""}
            raw = _bytes(result)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format: str, *_args: object) -> None:
            pass
    return Handler


def drop_identity(observation: dict[str, Any]) -> str:
    matches = [node for node in observation.get("nodes", [])
               if node.get("role") == "region" and node.get("name") == "Drop area"
               and node.get("visible") is True and node.get("enabled") is True]
    if len(matches) != 1 or not isinstance(matches[0].get("id"), str):
        raise ComputerUseError("Fixture requires one observed eligible Drop area")
    return matches[0]["id"]


def prescribed_plan(destination_id: str) -> dict[str, Any]:
    operations = [
        ("double_click", "button", "Double target", {}),
        ("right_click", "button", "Context target", {}),
        ("drag_drop", "button", "Drag source", {"to_node_id": destination_id}),
        ("select_text", "textbox", "Selection text", {"start": 0, "end": 5}),
        ("key_chord", "textbox", "Shortcut target", {"keys": ["CTRL", "K"]}),
        ("scroll", "region", "Scroll panel", {"direction": "down"}),
        ("invoke", "button", "Finish gestures", {}),
    ]
    steps = []
    for phase, confirmation, (verb, role, name, args) in zip(PHASES, CONFIRMATIONS, operations, strict=True):
        effects = [{"kind": "text", "text": confirmation}]
        if verb == "select_text":
            effects.append({"kind": "text_selection", "node_id": "$target", "start": 0, "end": 5})
        steps.append({"id": phase, "verb": verb, "target": {"role": role, "name": name},
                      "args": args, "expect": effects})
    return {"steps": steps, "final_predicates": [{"kind": "text", "text": CONFIRMATIONS[-1]}]}


def validate_plan(value: Any, destination_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"steps", "final_predicates"}:
        raise ComputerUseError("Gesture plan requires only steps and final_predicates")
    if not isinstance(value["steps"], list) or len(value["steps"]) != len(PHASES):
        raise ComputerUseError("Gesture plan requires the seven requested operations within the 25-step bound")
    steps = [parse_step(item) for item in value["steps"]]
    if len({step["id"] for step in steps}) != len(steps):
        raise ComputerUseError("Gesture step identities must be unique")
    expected = prescribed_plan(destination_id)
    for step, baseline in zip(steps, expected["steps"], strict=True):
        matching_verb = step["verb"] == baseline["verb"] or (
            baseline["verb"] == "invoke" and step["verb"] == "click")
        if not matching_verb or step["target"] != baseline["target"]:
            raise ComputerUseError("Plan changed the spoken gesture order or exact observed target")
        args = step.get("args", {})
        if step["verb"] == "key_chord":
            keys = [key.upper() for key in args.get("keys", [])]
            if len(keys) != 2 or keys[0] not in {"CTRL", "CONTROL"} or keys[1] != "K":
                raise ComputerUseError("Shortcut must match the spoken Control and K")
        elif args != baseline["args"]:
            raise ComputerUseError("Gesture arguments must match the spoken task and observed destination")
        if baseline["expect"][0] not in step["expect"]:
            raise ComputerUseError("Each gesture must wait for its independent server confirmation")
        if step["verb"] == "select_text" and baseline["expect"][1] not in step["expect"]:
            raise ComputerUseError("Text selection must verify the observed character range")
    terminal = parse_predicates(value["final_predicates"], terminal=True)
    if terminal != expected["final_predicates"]:
        raise ComputerUseError("Terminal predicate must match the spoken completion criterion")
    return {"steps": steps, "final_predicates": terminal}


def _token(path: Path) -> str:
    token = path.read_text().strip()
    if token.startswith("{"):
        saved = json.loads(token)
        token = str(saved.get("access_token") or saved.get("token") or "")
        if not token:
            token = parse_qs(urlsplit(str(saved.get("access_url") or "")).fragment).get("access", [""])[0]
    if len(token) < 24 or "\n" in token:
        raise ValueError("No valid token in protected token file")
    return token


def run(args: argparse.Namespace) -> int:
    artifact = args.output.parent / (args.output.stem + "-server.json")
    if artifact.exists():
        raise ValueError("Use a fresh output path; an earlier server artifact already exists")
    state = GestureState(artifact)
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(state))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    started, owned = time.monotonic(), False
    report: dict[str, Any] = {"schema": "robit.omni.browser-gestures.v1", "model": args.model,
                              "mode": "tts_asr_comprehension_actions" if args.speech else "deterministic_baseline",
                              "planner_calls": 0, "passed": False, "step_bound": 25}
    try:
        origin = urlsplit(args.endpoint)
        if (origin.scheme not in {"http", "https"} or not origin.netloc or origin.path not in {"", "/"}
                or origin.query or origin.fragment or origin.username or origin.password):
            raise ValueError("Endpoint must be an HTTP origin without credentials, query, or fragment")
        with httpx.Client(timeout=300, headers={"Authorization": "Bearer " + _token(args.token_file)},
                          cookies={"omni_portal_session": secrets.token_hex(32)}) as client:
            def tool(name: str, arguments: dict[str, Any], *, allow_failure: bool = False) -> dict[str, Any]:
                response = client.post(args.endpoint.rstrip("/") + f"/api/tools/{name}/call", json={"arguments": arguments})
                response.raise_for_status()
                result = response.json().get("result")
                if not isinstance(result, dict) or ("error" in result and not allow_failure):
                    raise RuntimeError(f"{name} did not return a successful result")
                return result
            try:
                owned = True
                tool("browser_interact", {"action": "navigate", "url": f"http://127.0.0.1:{server.server_port}/"})
                observation = tool("computer_use", {"operation": "observe", "surface": "browser"})
                observed = observation.get("state", {})
                report["observed_state"] = observed
                destination = drop_identity(observed)
                plan = prescribed_plan(destination)
                if args.speech:
                    from runtime.computer_use_speech import spoken_plan

                    report["planner_calls"] = 1
                    speech_root = args.output.parent / (args.output.stem + "-speech")
                    report["speech_artifacts"] = str(speech_root.resolve())
                    plan, report["speech_evidence"] = spoken_plan(
                        client, args.endpoint, args.model, INSTRUCTION, {"state": observed},
                        speech_root)
                report["plan"] = plan
                plan = validate_plan(plan, destination)
                result = tool("computer_use", {"operation": "workflow", "surface": "browser", **plan}, allow_failure=True)
                outcomes = result.get("outcomes", [])
                report.update(workflow=result, status=result.get("status"), planned_steps=len(plan["steps"]),
                              verified_steps=sum(item.get("status") == "verified" for item in outcomes),
                              outcomes=outcomes, terminal_receipt=result.get("terminal_receipt"), reason=result.get("reason"))
                final = tool("computer_use", {"operation": "observe", "surface": "browser"})
                marker_seen = state.marker in str(final.get("state", {}).get("text", ""))
                receipt = state.verify_artifact()
                report.update(authoritative_artifact=receipt, marker_observed=marker_seen)
                report["passed"] = (not result.get("error") and result.get("status") == "complete" and bool(result.get("terminal_receipt"))
                                    and len(outcomes) == len(PHASES)
                                    and all(item.get("status") == "verified" for item in outcomes)
                                    and receipt["verified"] and marker_seen)
            finally:
                if owned:
                    try:
                        tool("browser_interact", {"action": "close"})
                        report["owned_browser_closed"] = True
                    except Exception as exc:
                        report["cleanup_error"] = {"type": type(exc).__name__, "detail": str(exc)[:500]}
                        report["passed"] = False
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "detail": str(exc)[:500]}
        report["passed"] = False
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
        report.update(authoritative_server=state.summary(), elapsed_s=time.monotonic() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(_bytes(report))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--speech", action="store_true", help="Use deployed TTS audio as the only task goal to real ASR/planning")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
