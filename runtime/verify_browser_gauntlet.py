#!/usr/bin/env python3
"""Run a state-authoritative form, visual-grid, and drag browser gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

ACTIVE_TASK_STATUSES = {"pending", "running", "waiting", "paused"}
_FIXTURE_OBJECTIVE_MARKER = "Complete the dispatch form, every image-selection round"
_FORM_EXPECTED = {
    "recipient": "Avery Morgan",
    "email": "avery.morgan@example.test",
    "dispatch_code": "ORIN-4827",
    "notes": "Two ruggedized actuators; keep upright.",
    "speed": "expedited",
    "confirmed": "yes",
}
_GRID_SPECS = (
    ("traffic_light", "traffic lights", 2, 1),
    ("bicycle", "bicycles", 2, 0),
    ("crosswalk", "crosswalks", 2, 0),
)
_DISTRACTORS = (
    "bus",
    "car",
    "fire_hydrant",
    "mountain",
    "boat",
    "bench",
    "stop_sign",
    "palm_tree",
    "motorcycle",
)


def _is_stale_browser_gauntlet(task: dict[str, Any]) -> bool:
    """Identify only abandoned tasks created by this fixture."""

    objective = str(task.get("objective") or "")
    criteria = str(task.get("completion_criteria") or "")
    return (
        task.get("status") in ACTIVE_TASK_STATUSES
        and objective.startswith("Open http://127.0.0.1:")
        and _FIXTURE_OBJECTIVE_MARKER in objective
        and "BROWSER-GAUNTLET-PASS-" in criteria
    )


class GauntletState:
    """Server-owned truth for all stages; page reloads cannot manufacture a pass."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._random: random.Random = (
            random.Random(seed) if seed is not None else random.SystemRandom()
        )
        self.lock = threading.Lock()
        self.marker = f"BROWSER-GAUNTLET-PASS-{secrets.token_hex(5).upper()}"
        self.form_attempts: list[dict[str, Any]] = []
        self.form_passed = False
        self.grid_round = 0
        self.grid_generation = 0
        self.grid_hits: list[dict[str, Any]] = []
        self.grid_misses: list[dict[str, Any]] = []
        self.grid_verifications: list[dict[str, Any]] = []
        self.drop_attempts: list[dict[str, Any]] = []
        self.final_attempts: list[dict[str, Any]] = []
        self.final_passed = False
        self._replacement_remaining = 0
        self._tiles: list[str] = []
        self._reset_grid_locked()

    @property
    def expected_grid_hits(self) -> int:
        return sum(targets + replacements for _, _, targets, replacements in _GRID_SPECS)

    def _reset_grid_locked(self) -> None:
        if self.grid_round >= len(_GRID_SPECS):
            self._tiles = []
            self._replacement_remaining = 0
            return
        target, _, target_count, replacements = _GRID_SPECS[self.grid_round]
        distractors = list(_DISTRACTORS)
        self._random.shuffle(distractors)
        self._tiles = [target] * target_count + distractors[: 9 - target_count]
        self._random.shuffle(self._tiles)
        self._replacement_remaining = replacements
        self.grid_generation += 1

    def submit_form(self, values: dict[str, str]) -> bool:
        with self.lock:
            accepted = values == _FORM_EXPECTED
            self.form_attempts.append({"values": dict(values), "accepted": accepted})
            if accepted:
                self.form_passed = True
            return accepted

    def grid_view(self) -> dict[str, Any]:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS):
                return {"complete": True, "round": self.grid_round}
            target, label, _, _ = _GRID_SPECS[self.grid_round]
            return {
                "complete": False,
                "round": self.grid_round,
                "round_count": len(_GRID_SPECS),
                "instruction": f"Select every image containing {label}",
                "target": target,
                "generation": self.grid_generation,
                "tiles": list(self._tiles),
            }

    def click_grid(self, index: int) -> dict[str, Any]:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS) or not 0 <= index < 9:
                return {"accepted": False, "error": "invalid tile"}
            target = _GRID_SPECS[self.grid_round][0]
            kind = self._tiles[index]
            receipt = {
                "round": self.grid_round,
                "generation": self.grid_generation,
                "index": index,
                "kind": kind,
            }
            if kind != target:
                self.grid_misses.append(receipt)
                return {"accepted": False, "generation": self.grid_generation}
            self.grid_hits.append(receipt)
            if self._replacement_remaining:
                # One dynamic replacement remains a target and requires a new visual
                # inspection. Its generation and rendered scenery change immediately.
                self._replacement_remaining -= 1
            else:
                alternatives = [kind for kind in _DISTRACTORS if kind not in self._tiles]
                self._tiles[index] = self._random.choice(alternatives or list(_DISTRACTORS))
            self.grid_generation += 1
            return {"accepted": True, "generation": self.grid_generation}

    def verify_grid(self) -> bool:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS):
                return True
            target = _GRID_SPECS[self.grid_round][0]
            accepted = target not in self._tiles
            self.grid_verifications.append(
                {"round": self.grid_round, "accepted": accepted}
            )
            if accepted:
                self.grid_round += 1
                self._reset_grid_locked()
            return accepted

    def record_drop(self, crate: str) -> bool:
        with self.lock:
            accepted = crate == "purple" and self.grid_round == len(_GRID_SPECS)
            self.drop_attempts.append({"crate": crate, "accepted": accepted})
            return accepted

    def submit_final(self, seal: str) -> bool:
        with self.lock:
            dropped = any(item["accepted"] for item in self.drop_attempts)
            prerequisites = self.form_passed and self.grid_round == len(_GRID_SPECS)
            accepted = prerequisites and dropped and seal == "K7-MARS"
            self.final_attempts.append({"seal": seal, "accepted": accepted})
            if accepted:
                self.final_passed = True
            return accepted

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "marker": self.marker if self.final_passed else "",
                "complete": self.final_passed,
                "form_passed": self.form_passed,
                "form_attempts": list(self.form_attempts),
                "grid_round": self.grid_round,
                "grid_hits": list(self.grid_hits),
                "grid_misses": list(self.grid_misses),
                "grid_verifications": list(self.grid_verifications),
                "drop_attempts": list(self.drop_attempts),
                "final_attempts": list(self.final_attempts),
            }


_STYLE = """
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172033;font:16px system-ui}
main{max-width:1080px;margin:18px auto;padding:20px;background:white;border-radius:14px;
box-shadow:0 8px 30px #24324a22}h1{margin:0 0 10px;color:#173b72}h2{margin:14px 0 8px}
label{display:block;margin:8px 0 4px;font-weight:650}input,textarea{font:inherit;padding:9px;
border:1px solid #8795aa;border-radius:6px}input[type=text],input[type=email],textarea{width:100%}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}.choice{display:flex;gap:22px;margin:9px 0}
.choice label{font-weight:500}.choice input{margin-right:7px}button{font:700 16px system-ui;padding:10px 18px;
border:0;border-radius:7px;background:#1769d2;color:white;cursor:pointer}.error{color:#a20d22;font-weight:700}
"""


def _document(title: str, body: str, *, extra_style: str = "", script: str = "") -> bytes:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_STYLE}{extra_style}</style></head>"
        f"<body>{body}{script}</body></html>"
    ).encode()


def _form_page(*, error: bool = False) -> bytes:
    message = "<p class='error'>Values did not match the dispatch brief.</p>" if error else ""
    body = f"""
<main><h1>Orin actuator dispatch</h1>
<p>Enter the dispatch brief exactly, choose Expedited, confirm the handling note, and submit.</p>
{message}<form method="post" action="/submit-form">
<div class="row"><div><label for="recipient">Recipient</label><input id="recipient" name="recipient" type="text"></div>
<div><label for="email">Email</label><input id="email" name="email" type="email"></div></div>
<label for="dispatch">Dispatch code</label><input id="dispatch" name="dispatch_code" type="text">
<label for="notes">Handling notes</label><textarea id="notes" name="notes" rows="3"></textarea>
<fieldset><legend>Delivery speed</legend><div class="choice">
<label><input type="radio" name="speed" value="standard">Standard</label>
<label><input type="radio" name="speed" value="expedited">Expedited</label></div></fieldset>
<label><input type="checkbox" name="confirmed" value="yes"> I confirm the handling note is exact</label>
<p><button type="submit">Submit dispatch</button></p></form></main>"""
    return _document("Dispatch form", body)


def _grid_page(*, error: bool = False) -> bytes:
    message = "<p class='error'>Some matching images remain. Inspect the fresh grid.</p>" if error else ""
    body = f"""
<main class="verify"><h1>Image selection checkpoint</h1>
<p id="instruction">Loading image grid…</p>{message}
<canvas id="grid" width="720" height="510" aria-label="Nine unlabeled image tiles"></canvas>
<p><button id="verify" type="button">Verify selection</button></p></main>"""
    style = """
.verify{max-width:780px;margin-top:8px;padding:12px 28px}.verify h1{font-size:22px}.verify p{margin:7px 0}
canvas{display:block;width:720px;height:510px;background:#d7dee7;border:2px solid #27364c}
"""
    script = r"""<script>
const canvas=document.getElementById('grid'),ctx=canvas.getContext('2d');let view=null;
function road(x,y,w,h,v){let g=ctx.createLinearGradient(x,y,x,y+h);g.addColorStop(0,v%2?'#9fc4df':'#d9b889');g.addColorStop(.5,'#90a3a9');g.addColorStop(1,'#343a40');ctx.fillStyle=g;ctx.fillRect(x,y,w,h);
ctx.fillStyle='#555d63';ctx.beginPath();ctx.moveTo(x+w*.3,y+h);ctx.lineTo(x+w*.46,y+h*.45);ctx.lineTo(x+w*.58,y+h*.45);ctx.lineTo(x+w*.78,y+h);ctx.fill();}
function trafficLight(x,y,w,h,v){road(x,y,w,h,v);ctx.strokeStyle='#20252b';ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(x+w*.61,y+h*.88);ctx.lineTo(x+w*.61,y+h*.29);ctx.stroke();ctx.fillStyle='#16191d';ctx.fillRect(x+w*.49,y+h*.14,w*.24,h*.42);
['#e83b32','#f0bd2d','#35b45a'].forEach((c,i)=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(x+w*.61,y+h*(.22+i*.11),w*.045,0,7);ctx.fill()});}
function bicycle(x,y,w,h,v){road(x,y,w,h,v);ctx.strokeStyle='#f5f7fa';ctx.lineWidth=5;let cy=y+h*.67,r=w*.13;for(const cx of [x+w*.3,x+w*.7]){ctx.beginPath();ctx.arc(cx,cy,r,0,7);ctx.stroke()}
ctx.beginPath();ctx.moveTo(x+w*.3,cy);ctx.lineTo(x+w*.48,y+h*.45);ctx.lineTo(x+w*.6,cy);ctx.lineTo(x+w*.3,cy);ctx.lineTo(x+w*.52,cy);ctx.lineTo(x+w*.7,cy);ctx.moveTo(x+w*.48,y+h*.45);ctx.lineTo(x+w*.62,y+h*.4);ctx.stroke();}
function crosswalk(x,y,w,h,v){road(x,y,w,h,v);ctx.fillStyle='#f5f3e8';for(let i=0;i<6;i++){ctx.save();ctx.translate(x+w*.15+i*w*.13,y+h*.64);ctx.rotate(-.18);ctx.fillRect(0,0,w*.08,h*.28);ctx.restore()}}
function objectScene(kind,x,y,w,h,v){road(x,y,w,h,v);ctx.lineWidth=4;ctx.strokeStyle='#17202b';
if(kind==='bus'||kind==='car'||kind==='motorcycle'){ctx.fillStyle=kind==='bus'?'#f2b632':kind==='car'?'#da4054':'#273bb0';ctx.fillRect(x+w*.18,y+h*.43,w*(kind==='bus'?.65:.55),h*(kind==='bus'?.3:.2));ctx.fillStyle='#bde3f2';ctx.fillRect(x+w*.25,y+h*.47,w*.18,h*.1);ctx.fillRect(x+w*.49,y+h*.47,w*.18,h*.1);ctx.fillStyle='#111';for(const q of [x+w*.3,x+w*.7]){ctx.beginPath();ctx.arc(q,y+h*.75,w*.07,0,7);ctx.fill()}}
else if(kind==='fire_hydrant'){ctx.fillStyle='#d92d34';ctx.fillRect(x+w*.42,y+h*.42,w*.18,h*.38);ctx.fillRect(x+w*.34,y+h*.5,w*.34,h*.12);ctx.beginPath();ctx.arc(x+w*.51,y+h*.42,w*.12,3.14,6.29);ctx.fill()}
else if(kind==='mountain'){ctx.fillStyle='#506f62';ctx.beginPath();ctx.moveTo(x+w*.08,y+h*.8);ctx.lineTo(x+w*.45,y+h*.22);ctx.lineTo(x+w*.9,y+h*.8);ctx.fill();ctx.fillStyle='white';ctx.beginPath();ctx.moveTo(x+w*.34,y+h*.39);ctx.lineTo(x+w*.45,y+h*.22);ctx.lineTo(x+w*.58,y+h*.39);ctx.fill()}
else if(kind==='boat'){ctx.fillStyle='#3999bf';ctx.fillRect(x,y+h*.62,w,h*.38);ctx.fillStyle='#f3f0dd';ctx.beginPath();ctx.moveTo(x+w*.2,y+h*.57);ctx.lineTo(x+w*.82,y+h*.57);ctx.lineTo(x+w*.7,y+h*.76);ctx.lineTo(x+w*.3,y+h*.76);ctx.fill();ctx.strokeStyle='#5a3a1c';ctx.beginPath();ctx.moveTo(x+w*.5,y+h*.2);ctx.lineTo(x+w*.5,y+h*.6);ctx.stroke()}
else if(kind==='bench'){ctx.fillStyle='#8c5a32';ctx.fillRect(x+w*.22,y+h*.46,w*.58,h*.1);ctx.fillRect(x+w*.22,y+h*.61,w*.58,h*.1);ctx.fillRect(x+w*.27,y+h*.7,w*.06,h*.18);ctx.fillRect(x+w*.69,y+h*.7,w*.06,h*.18)}
else if(kind==='stop_sign'){ctx.strokeStyle='#333';ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(x+w*.5,y+h*.38);ctx.lineTo(x+w*.5,y+h*.9);ctx.stroke();ctx.fillStyle='#d7242d';ctx.beginPath();for(let i=0;i<8;i++){let a=Math.PI/8+i*Math.PI/4;ctx.lineTo(x+w*.5+Math.cos(a)*w*.17,y+h*.29+Math.sin(a)*w*.17)}ctx.fill()}
else{ctx.strokeStyle='#69451f';ctx.lineWidth=11;ctx.beginPath();ctx.moveTo(x+w*.5,y+h*.88);ctx.lineTo(x+w*.5,y+h*.35);ctx.stroke();ctx.strokeStyle='#2c8a45';ctx.lineWidth=17;for(let i=-2;i<3;i++){ctx.beginPath();ctx.moveTo(x+w*.5,y+h*.35);ctx.lineTo(x+w*(.5+i*.14),y+h*.18+Math.abs(i)*8);ctx.stroke()}}}
function draw(){ctx.clearRect(0,0,720,510);view.tiles.forEach((kind,i)=>{let col=i%3,row=Math.floor(i/3),x=col*240+4,y=row*170+4,w=232,h=162,v=view.generation+i;ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();if(kind==='traffic_light')trafficLight(x,y,w,h,v);else if(kind==='bicycle')bicycle(x,y,w,h,v);else if(kind==='crosswalk')crosswalk(x,y,w,h,v);else objectScene(kind,x,y,w,h,v);ctx.restore();ctx.strokeStyle='#fff';ctx.lineWidth=7;ctx.strokeRect(x,y,w,h)});}
async function load(){view=await(await fetch('/grid-state',{cache:'no-store'})).json();document.getElementById('instruction').textContent='Round '+(view.round+1)+' of '+view.round_count+' — '+view.instruction;draw()}
canvas.addEventListener('click',async e=>{const r=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-r.left)/(r.width/3)),row=Math.floor((e.clientY-r.top)/(r.height/3)),index=row*3+col;const answer=await(await fetch('/grid-click?index='+index,{cache:'no-store'})).json();if(answer.accepted)await load()});
document.getElementById('verify').addEventListener('click',()=>location.href='/verify-grid');load();
</script>"""
    return _document("Image selection checkpoint", body, extra_style=style, script=script)


def _final_page(*, error: bool = False) -> bytes:
    message = "<p class='error'>The crate or derived seal is not yet correct.</p>" if error else ""
    body = f"""
<main><h1>Mars transfer manifest</h1>
<p>Find destination <strong>MRS-204</strong> with mass <strong>46 kg</strong>. Drag its staging crate into the truck bay, then enter that row's seal.</p>{message}
<table><thead><tr><th>Destination</th><th>Mass</th><th>Staging</th><th>Seal</th></tr></thead><tbody>
<tr><td>LUNA-118</td><td>46 kg</td><td>Blue crate</td><td>Q2-LUNA</td></tr>
<tr><td>MRS-204</td><td>31 kg</td><td>Amber crate</td><td>T4-DUST</td></tr>
<tr><td>MRS-204</td><td>46 kg</td><td>Purple crate</td><td>K7-MARS</td></tr>
<tr><td>MRS-240</td><td>46 kg</td><td>Green crate</td><td>N8-ARES</td></tr></tbody></table>
<section class="yard"><div class="crates"><div class="crate blue" tabindex="0" data-crate="blue">BLUE CRATE</div>
<div class="crate purple" tabindex="0" data-crate="purple">PURPLE CRATE</div><div class="crate amber" tabindex="0" data-crate="amber">AMBER CRATE</div></div>
<div id="bay" class="bay">TRUCK BAY<br><small>DROP MATCHING CRATE HERE</small></div></section>
<form method="post" action="/submit-final"><label for="seal">Derived seal</label><input id="seal" name="seal" type="text">
<button type="submit">Confirm transfer</button></form></main>"""
    style = """
table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #9aa7b8;padding:7px;text-align:left}
th{background:#e5edf8}.yard{height:250px;background:#d8dee6;display:flex;align-items:center;justify-content:space-around;
border:2px solid #50637d;margin:10px 0}.crates{display:flex;gap:14px}.crate{width:118px;height:88px;color:white;font-weight:800;
display:grid;place-items:center;border:5px ridge #eee;cursor:grab;text-align:center}.blue{background:#2871ca}.purple{background:#7b42b5}
.amber{background:#b87316}.bay{width:260px;height:170px;border:6px dashed #1d304d;background:#f7fbff;display:grid;
place-items:center;text-align:center;font-size:24px;font-weight:850}.bay small{font-size:12px}form{display:flex;align-items:end;gap:12px}
form label{flex:1}form input{display:block;margin-top:4px}
"""
    script = r"""<script>
let active='';document.querySelectorAll('.crate').forEach(crate=>crate.addEventListener('mousedown',()=>{active=crate.dataset.crate}));
document.addEventListener('mouseup',async e=>{if(!active)return;const b=document.getElementById('bay').getBoundingClientRect();
if(e.clientX>=b.left&&e.clientX<=b.right&&e.clientY>=b.top&&e.clientY<=b.bottom){const r=await(await fetch('/drop?crate='+encodeURIComponent(active),{cache:'no-store'})).json();if(r.accepted)document.getElementById('bay').innerHTML='CRATE ACCEPTED<br><small>ENTER THE MATCHING SEAL</small>'}active=''});
</script>"""
    return _document("Mars transfer manifest", body, extra_style=style, script=script)


def _complete_page(marker: str) -> bytes:
    return _document(
        "Browser gauntlet complete",
        f"<main><h1>ALL BROWSER GATES PASSED</h1><p>Exact marker:</p>"
        f"<pre>{marker}</pre></main>",
        extra_style="pre{font:800 28px monospace;color:#126b35}",
    )


def _handler(state: GauntletState):
    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            body: bytes = b"",
            *,
            status: int = HTTPStatus.OK,
            content_type: str = "text/html; charset=utf-8",
            location: str = "",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _form_values(self) -> dict[str, str]:
            try:
                length = min(65536, int(self.headers.get("Content-Length", "0")))
            except ValueError:
                length = 0
            parsed = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            return {key: values[0] for key, values in parsed.items() if values}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send(_form_page())
            elif parsed.path == "/grid":
                self._send(_grid_page())
            elif parsed.path == "/grid-state":
                self._send(
                    json.dumps(state.grid_view()).encode(),
                    content_type="application/json",
                )
            elif parsed.path == "/grid-click":
                query = parse_qs(parsed.query)
                try:
                    result = state.click_grid(int(query["index"][0]))
                except (KeyError, ValueError, IndexError):
                    result = {"accepted": False, "error": "invalid index"}
                self._send(json.dumps(result).encode(), content_type="application/json")
            elif parsed.path == "/verify-grid":
                if state.verify_grid():
                    destination = (
                        "/final"
                        if state.grid_view().get("complete")
                        else "/grid"
                    )
                    self._send(status=HTTPStatus.SEE_OTHER, location=destination)
                else:
                    self._send(_grid_page(error=True))
            elif parsed.path == "/final":
                self._send(_final_page())
            elif parsed.path == "/drop":
                crate = parse_qs(parsed.query).get("crate", [""])[0]
                result = {"accepted": state.record_drop(crate)}
                self._send(json.dumps(result).encode(), content_type="application/json")
            elif parsed.path == "/complete":
                snapshot = state.snapshot()
                if snapshot["complete"]:
                    self._send(_complete_page(state.marker))
                else:
                    self._send(b"not complete", status=HTTPStatus.CONFLICT)
            elif parsed.path == "/state":
                self._send(
                    json.dumps(state.snapshot()).encode(),
                    content_type="application/json",
                )
            else:
                self._send(b"not found", status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            values = self._form_values()
            if parsed.path == "/submit-form":
                if state.submit_form(values):
                    self._send(status=HTTPStatus.SEE_OTHER, location="/grid")
                else:
                    self._send(_form_page(error=True), status=HTTPStatus.BAD_REQUEST)
            elif parsed.path == "/submit-final":
                if state.submit_final(values.get("seal", "")):
                    self._send(status=HTTPStatus.SEE_OTHER, location="/complete")
                else:
                    self._send(_final_page(error=True), status=HTTPStatus.BAD_REQUEST)
            else:
                self._send(b"not found", status=HTTPStatus.NOT_FOUND)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def _access_token(status_path: Path) -> str:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    fragment = urlsplit(str(status.get("access_url") or "")).fragment
    token = parse_qs(fragment).get("access", [""])[0]
    if len(token) < 24:
        raise RuntimeError(f"No portal access token is available in {status_path}")
    return token


def _tool(
    client: httpx.Client, portal: str, token: str, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    response = client.post(
        f"{portal.rstrip('/')}/api/tools/{name}/call",
        headers={"Authorization": f"Bearer {token}"},
        json={"arguments": arguments},
    )
    response.raise_for_status()
    body = response.json()
    result = body.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} returned no result object")
    return result


def _browser_action(action: dict[str, Any]) -> str:
    try:
        arguments = json.loads(str(action.get("arguments") or "{}"))
    except ValueError:
        return ""
    return str(arguments.get("action") or "")


def _validate_result(task: dict[str, Any], state: GauntletState) -> dict[str, Any]:
    snapshot = state.snapshot()
    actions = task.get("actions") if isinstance(task.get("actions"), list) else []
    tools = {
        str(action.get("tool") or "")
        for action in actions
        if isinstance(action, dict)
    }
    forbidden = tools.intersection({"shell", "web_fetch", "web_search", "gui_interact"})
    browser_actions = [
        _browser_action(action)
        for action in actions
        if isinstance(action, dict) and action.get("tool") == "browser_interact"
    ]
    allowed = {
        "navigate",
        "snapshot",
        "click",
        "visual_click",
        "drag",
        "type",
        "scroll",
        "back",
    }
    invalid = [action for action in browser_actions if action not in allowed]
    required = {"navigate", "click", "visual_click", "drag", "type"}
    missing = required.difference(browser_actions)
    if task.get("status") != "completed":
        raise RuntimeError(f"Browser gauntlet ended with status {task.get('status')}")
    if forbidden:
        raise RuntimeError(f"Browser gauntlet bypassed visible execution: {sorted(forbidden)}")
    if invalid:
        raise RuntimeError(f"Browser gauntlet used invalid browser actions: {invalid}")
    if missing:
        raise RuntimeError(f"Browser gauntlet skipped action classes: {sorted(missing)}")
    if not snapshot["complete"] or snapshot["marker"] != state.marker:
        raise RuntimeError("Task completed without the fixture-owned terminal state")
    if len(snapshot["form_attempts"]) != 1 or not snapshot["form_passed"]:
        raise RuntimeError("The exact dispatch form was not submitted once")
    if len(snapshot["grid_hits"]) != state.expected_grid_hits:
        raise RuntimeError("The task did not select every expected visual target")
    if snapshot["grid_misses"]:
        raise RuntimeError("The task clicked a non-target visual-grid tile")
    if len(snapshot["grid_verifications"]) != len(_GRID_SPECS) or not all(
        item["accepted"] for item in snapshot["grid_verifications"]
    ):
        raise RuntimeError("The task did not cleanly verify every visual-grid round")
    if snapshot["drop_attempts"] != [{"crate": "purple", "accepted": True}]:
        raise RuntimeError("The task did not make exactly one correct DOM drag")
    if snapshot["final_attempts"] != [{"seal": "K7-MARS", "accepted": True}]:
        raise RuntimeError("The task did not submit the uniquely derived seal once")
    return {
        "passed": True,
        "task_id": str(task.get("task_id") or ""),
        "form_attempts": 1,
        "grid_rounds": len(_GRID_SPECS),
        "visual_hits": len(snapshot["grid_hits"]),
        "visual_misses": 0,
        "drag_attempts": 1,
        "actions": len(actions),
    }


def run(args: argparse.Namespace) -> int:
    state = GauntletState()
    server = ThreadingHTTPServer((args.bind, 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    challenge_url = f"http://{args.bind}:{server.server_port}/"
    token = _access_token(args.status_path)
    portal_session_id = hashlib.sha256(
        b"omni-voice-session\0" + token.encode("utf-8")
    ).hexdigest()
    client = httpx.Client(
        timeout=30,
        follow_redirects=False,
        cookies={"omni_portal_session": portal_session_id},
    )
    task_id = ""
    owns_browser = False
    try:
        listed = _tool(client, args.portal, token, "background_task", {"action": "list"})
        active = [
            task
            for task in listed.get("tasks", [])
            if isinstance(task, dict) and task.get("status") in ACTIVE_TASK_STATUSES
        ]
        stale = [task for task in active if _is_stale_browser_gauntlet(task)]
        for task in stale:
            stale_id = str(task.get("task_id") or "")
            if stale_id:
                _tool(
                    client,
                    args.portal,
                    token,
                    "background_task",
                    {"action": "cancel", "task_id": stale_id},
                )
        if stale:
            _tool(client, args.portal, token, "browser_interact", {"action": "close"})
        active = [task for task in active if task not in stale]
        if active and not args.allow_concurrent:
            raise RuntimeError(
                "An unfinished background task already owns the desktop; finish or "
                "cancel it before running this browser gauntlet."
            )
        owns_browser = not active
        if owns_browser:
            _tool(client, args.portal, token, "browser_interact", {"action": "close"})
        started = _tool(
            client,
            args.portal,
            token,
            "background_task",
            {
                "action": "start",
                "independent": bool(active),
                "objective": (
                    f"Open {challenge_url} in the visible Chromium window. Complete the "
                    "dispatch form, every image-selection round, and the manifest transfer. "
                    "Use browser_interact only. Use DOM element_id actions for controls and "
                    "the draggable crate. Use fresh screenshots plus normalized_1000 "
                    "visual_click with a precise target phrase only for canvas image tiles. "
                    "Image tiles can be replaced after a correct click, so re-inspect every "
                    "fresh grid. Do not inspect source, use shell/web tools, or use whole-"
                    "desktop gui_interact. Dispatch values: recipient Avery Morgan; email "
                    "avery.morgan@example.test; code ORIN-4827; notes `Two ruggedized "
                    "actuators; keep upright.`; Expedited; handling confirmation checked."
                ),
                "completion_criteria": (
                    f"The final rendered page shows exact marker {state.marker}. All form "
                    "values, image choices, the DOM drag, and the derived seal must be "
                    "accepted by the site. Cite the fresh terminal browser evidence."
                ),
            },
        )
        if started.get("accepted") is not True:
            raise RuntimeError(f"Background task was not accepted: {started.get('error')}")
        task_id = str(started.get("task_id") or "")
        deadline = time.monotonic() + args.timeout_s
        task: dict[str, Any] = {}
        while time.monotonic() < deadline:
            time.sleep(args.poll_s)
            status = _tool(
                client,
                args.portal,
                token,
                "background_task",
                {"action": "status", "task_id": task_id},
            )
            value = status.get("task")
            if isinstance(value, dict):
                task = value
            if task.get("status") in {"completed", "blocked", "cancelled"}:
                break
        else:
            _tool(
                client,
                args.portal,
                token,
                "background_task",
                {"action": "cancel", "task_id": task_id},
            )
            raise RuntimeError("Timed out waiting for the browser gauntlet")
        result = _validate_result(task, state)
        result["task_id"] = task_id
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if task_id:
            try:
                _tool(
                    client,
                    args.portal,
                    token,
                    "background_task",
                    {"action": "cancel", "task_id": task_id},
                )
            except (httpx.HTTPError, RuntimeError):
                pass
        if owns_browser:
            try:
                _tool(client, args.portal, token, "browser_interact", {"action": "close"})
            except (httpx.HTTPError, RuntimeError):
                pass
        client.close()
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal", default="http://127.0.0.1:8920")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--poll-s", type=float, default=2)
    parser.add_argument("--allow-concurrent", action="store_true")
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("runtime-data/state/daemon-status.json"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
