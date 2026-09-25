#!/usr/bin/env python3
"""Run a state-authoritative form, visual-grid, and drag browser gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import html
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
from urllib.request import Request, urlopen

import httpx

ACTIVE_TASK_STATUSES = {"pending", "running", "waiting", "paused"}
_FIXTURE_OBJECTIVE_MARKERS = (
    "Complete the dispatch form, every image-selection round",
    "Complete the dispatch control matrix, the deeply nested routing console",
)
_FORM_EXPECTED = {
    "recipient": "Avery Morgan",
    "email": "avery.morgan@example.test",
    "password": "Saffron!27",
    "search": "thermal actuator",
    "telephone": "+1-206-555-0148",
    "reference_url": "https://example.test/orin",
    "dispatch_code": "ORIN-4827",
    "notes": "Two ruggedized actuators; keep upright.",
    "units": "2",
    "delivery_date": "2026-10-06",
    "delivery_time": "14:35",
    "delivery_datetime": "2026-10-06T14:35",
    "billing_month": "2026-10",
    "service_week": "2026-W41",
    "torque": "73",
    "marker_color": "#6f42c1",
    "region": "west",
    "channels": ["vision", "voice"],
    "speed": "expedited",
    "confirmed": "yes",
    "workflow_nonce": "owned-real-photo-v2",
    "attachment_name": "gauntlet-packing-note.txt",
    "attachment_text": "ORIN-4827 packing note: keep upright.\n",
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
    "palm_tree",
    "cat",
    "dog",
)
_PHOTO_MANIFEST = Path(__file__).with_name("browser_gauntlet_photos.json")
_PHOTO_RECORDS = tuple(json.loads(_PHOTO_MANIFEST.read_text(encoding="utf-8"))["photos"])
_PHOTO_BY_ID = {str(item["id"]): item for item in _PHOTO_RECORDS}
_PHOTO_IDS_BY_KIND: dict[str, tuple[str, ...]] = {}
for _kind in {str(item["kind"]) for item in _PHOTO_RECORDS}:
    _PHOTO_IDS_BY_KIND[_kind] = tuple(
        str(item["id"]) for item in _PHOTO_RECORDS if item["kind"] == _kind
    )


def _is_stale_browser_gauntlet(task: dict[str, Any]) -> bool:
    """Identify only abandoned tasks created by this fixture."""

    objective = str(task.get("objective") or "")
    criteria = str(task.get("completion_criteria") or "")
    return (
        task.get("status") in ACTIVE_TASK_STATUSES
        and objective.startswith("Open http://127.0.0.1:")
        and any(marker in objective for marker in _FIXTURE_OBJECTIVE_MARKERS)
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
        self.interface_events: list[str] = []
        self.interface_passed = False
        self.grid_round = 0
        self.grid_generation = 0
        self.grid_hits: list[dict[str, Any]] = []
        self.grid_misses: list[dict[str, Any]] = []
        self.grid_verifications: list[dict[str, Any]] = []
        self.drop_attempts: list[dict[str, Any]] = []
        self.final_attempts: list[dict[str, Any]] = []
        self.final_passed = False
        self._replacement_assets: list[str] = []
        self._tiles: list[str] = []
        self._reset_grid_locked()

    @property
    def expected_grid_hits(self) -> int:
        return sum(targets + replacements for _, _, targets, replacements in _GRID_SPECS)

    def _reset_grid_locked(self) -> None:
        if self.grid_round >= len(_GRID_SPECS):
            self._tiles = []
            self._replacement_assets = []
            return
        target, _, target_count, replacements = _GRID_SPECS[self.grid_round]
        target_assets = list(_PHOTO_IDS_BY_KIND[target])
        self._random.shuffle(target_assets)
        needed = target_count + replacements
        if len(target_assets) < needed:
            raise RuntimeError(f"photo manifest lacks {needed} assets for {target}")
        distractor_kinds = list(_DISTRACTORS)
        self._random.shuffle(distractor_kinds)
        distractor_assets = [
            self._random.choice(_PHOTO_IDS_BY_KIND[kind])
            for kind in distractor_kinds[: 9 - target_count]
        ]
        self._tiles = target_assets[:target_count] + distractor_assets
        self._random.shuffle(self._tiles)
        self._replacement_assets = target_assets[target_count:needed]
        self.grid_generation += 1

    def submit_form(self, values: dict[str, Any]) -> bool:
        with self.lock:
            accepted = values == _FORM_EXPECTED
            self.form_attempts.append({"values": dict(values), "accepted": accepted})
            if accepted:
                self.form_passed = True
            return accepted

    def interface_event(self, event: str) -> bool:
        sequence = ["operations", "routing", "orbital", "gale", "advanced", "delta-7"]
        with self.lock:
            next_index = len(self.interface_events)
            accepted = next_index < len(sequence) and event == sequence[next_index]
            if accepted:
                self.interface_events.append(event)
                self.interface_passed = len(self.interface_events) == len(sequence)
            return accepted

    def grid_view(self) -> dict[str, Any]:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS):
                return {"complete": True, "round": self.grid_round}
            _, label, _, _ = _GRID_SPECS[self.grid_round]
            return {
                "complete": False,
                "round": self.grid_round,
                "round_count": len(_GRID_SPECS),
                "instruction": f"Select every image containing {label}",
                "generation": self.grid_generation,
                "tiles": list(self._tiles),
            }

    def answer_indices(self) -> list[int]:
        """Return server-private answer indices for unit tests, never HTTP clients."""

        with self.lock:
            if self.grid_round >= len(_GRID_SPECS):
                return []
            target = _GRID_SPECS[self.grid_round][0]
            return [
                index
                for index, asset_id in enumerate(self._tiles)
                if _PHOTO_BY_ID[asset_id]["kind"] == target
            ]

    def click_grid(self, index: int) -> dict[str, Any]:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS) or not 0 <= index < 9:
                return {"accepted": False, "error": "invalid tile"}
            target = _GRID_SPECS[self.grid_round][0]
            asset_id = self._tiles[index]
            kind = str(_PHOTO_BY_ID[asset_id]["kind"])
            receipt = {
                "round": self.grid_round,
                "generation": self.grid_generation,
                "index": index,
                "asset": asset_id,
            }
            if kind != target:
                self.grid_misses.append(receipt)
                return {"accepted": False, "generation": self.grid_generation}
            self.grid_hits.append(receipt)
            if self._replacement_assets:
                # A fresh real photograph remains a target and requires another
                # visual inspection, matching dynamic reCAPTCHA replacement behavior.
                self._tiles[index] = self._replacement_assets.pop()
            else:
                visible_kinds = {str(_PHOTO_BY_ID[item]["kind"]) for item in self._tiles}
                alternatives = [kind for kind in _DISTRACTORS if kind not in visible_kinds]
                next_kind = self._random.choice(alternatives or list(_DISTRACTORS))
                self._tiles[index] = self._random.choice(_PHOTO_IDS_BY_KIND[next_kind])
            self.grid_generation += 1
            return {"accepted": True, "generation": self.grid_generation}

    def verify_grid(self) -> bool:
        with self.lock:
            if self.grid_round >= len(_GRID_SPECS):
                return True
            target = _GRID_SPECS[self.grid_round][0]
            accepted = all(
                _PHOTO_BY_ID[asset_id]["kind"] != target for asset_id in self._tiles
            )
            self.grid_verifications.append(
                {"round": self.grid_round, "accepted": accepted}
            )
            if accepted:
                self.grid_round += 1
                self._reset_grid_locked()
            return accepted

    def record_drop(self, crate: str) -> bool:
        with self.lock:
            accepted = (
                crate == "purple"
                and self.interface_passed
                and self.grid_round == len(_GRID_SPECS)
            )
            self.drop_attempts.append({"crate": crate, "accepted": accepted})
            return accepted

    def submit_final(self, seal: str) -> bool:
        with self.lock:
            dropped = any(item["accepted"] for item in self.drop_attempts)
            prerequisites = (
                self.form_passed
                and self.interface_passed
                and self.grid_round == len(_GRID_SPECS)
            )
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
                "interface_events": list(self.interface_events),
                "interface_passed": self.interface_passed,
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


def _form_page(
    *, error: bool = False, upload_path: str = "runtime-data/browser-uploads/gauntlet-packing-note.txt"
) -> bytes:
    message = "<p class='error'>Values did not match the dispatch brief.</p>" if error else ""
    body = f"""
<main><h1>Orin actuator dispatch — control matrix</h1>
<p>Populate every control from this mission card. Native pickers and dropdowns must retain their exact values.</p>
<details open><summary>Mission card</summary><table class="spec"><tbody>
<tr><th>Recipient</th><td>Avery Morgan</td><th>Email</th><td>avery.morgan@example.test</td></tr>
<tr><th>Passphrase</th><td>Saffron!27</td><th>Search</th><td>thermal actuator</td></tr>
<tr><th>Telephone</th><td>+1-206-555-0148</td><th>Reference URL</th><td>https://example.test/orin</td></tr>
<tr><th>Dispatch code</th><td>ORIN-4827</td><th>Units</th><td>2</td></tr>
<tr><th>Date / time</th><td>2026-10-06 / 14:35</td><th>Date-time</th><td>2026-10-06T14:35</td></tr>
<tr><th>Month / week</th><td>2026-10 / 2026-W41</td><th>Torque / color</th><td>73 / #6f42c1</td></tr>
<tr><th>Region</th><td>West staging</td><th>Channels</th><td>Voice and Vision</td></tr>
<tr><th>Speed</th><td>Expedited</td><th>Notes</th><td>Two ruggedized actuators; keep upright.</td></tr>
<tr><th>Attachment</th><td colspan="3">{html.escape(upload_path)} — choose gauntlet-packing-note.txt</td></tr>
</tbody></table></details>{message}
<form id="dispatch-form">
<input type="hidden" name="workflow_nonce" value="owned-real-photo-v2">
<div class="row"><div><label for="recipient">Recipient — text</label><input id="recipient" name="recipient" type="text"></div>
<div><label for="email">Email</label><input id="email" name="email" type="email"></div></div>
<div class="row"><div><label for="password">Passphrase — password</label><input id="password" name="password" type="password"></div>
<div><label for="search">Inventory query — search</label><input id="search" name="search" type="search"></div></div>
<div class="row"><div><label for="telephone">Telephone</label><input id="telephone" name="telephone" type="tel"></div>
<div><label for="reference-url">Reference URL</label><input id="reference-url" name="reference_url" type="url"></div></div>
<div class="row"><div><label for="dispatch">Dispatch code</label><input id="dispatch" name="dispatch_code" type="text"></div>
<div><label for="units">Units — number</label><input id="units" name="units" type="number" min="1" max="12"></div></div>
<div class="row"><div><label for="delivery-date">Delivery date</label><input id="delivery-date" name="delivery_date" type="date"></div>
<div><label for="delivery-time">Delivery time</label><input id="delivery-time" name="delivery_time" type="time"></div></div>
<div class="row"><div><label for="delivery-datetime">Delivery date-time</label><input id="delivery-datetime" name="delivery_datetime" type="datetime-local"></div>
<div><label for="billing-month">Billing month</label><input id="billing-month" name="billing_month" type="month"></div></div>
<div class="row"><div><label for="service-week">Service week</label><input id="service-week" name="service_week" type="week"></div>
<div><label for="marker-color">Manifest marker color</label><input id="marker-color" name="marker_color" type="color" value="#000000"></div></div>
<label for="torque">Torque reserve — range (73)</label><input id="torque" name="torque" type="range" min="0" max="100" value="0"><output id="torque-value">0</output>
<div class="row"><div><label for="region">Region — dropdown</label><select id="region" name="region"><option value="">Choose…</option><option value="north">North staging</option><option value="west">West staging</option><option value="orbital">Orbital staging</option></select></div>
<div><label for="channels">Channels — multi-select</label><select id="channels" name="channels" multiple size="3"><option value="voice">Voice</option><option value="vision">Vision</option><option value="telemetry">Telemetry</option></select></div></div>
<label for="notes">Handling notes — textarea</label><textarea id="notes" name="notes" rows="3"></textarea>
<fieldset><legend>Delivery speed — radio</legend><div class="choice">
<label><input type="radio" name="speed" value="standard">Standard</label>
<label><input type="radio" name="speed" value="expedited">Expedited</label></div></fieldset>
<label><input type="checkbox" name="confirmed" value="yes"> I confirm the handling note is exact</label>
<label for="attachment">Packing note — file</label><input id="attachment" name="attachment" type="file" accept="text/plain">
<div class="button-row"><input id="preview" type="button" value="Preview route"><input type="reset" value="Reset form"><input type="submit" value="Submit dispatch"></div>
<p id="preview-state" aria-live="polite"></p><p id="form-error" class="error"></p></form>
<form id="reference-form"><input type="image" alt="Open packing reference image control" title="Reference image control" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='36'%3E%3Crect width='160' height='36' rx='5' fill='%23566b86'/%3E%3Ctext x='80' y='23' text-anchor='middle' fill='white' font-size='13'%3EReference image%3C/text%3E%3C/svg%3E"></form>
</main>"""
    style = """
.spec{border-collapse:collapse;width:100%;font-size:14px}.spec th,.spec td{border:1px solid #abb7c7;padding:6px;text-align:left}.spec th{background:#e7eef7}
fieldset{margin:10px 0}.button-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}select,input[type=date],input[type=time],input[type=datetime-local],input[type=month],input[type=week],input[type=number],input[type=search],input[type=tel],input[type=url],input[type=password],input[type=file]{width:100%;font:inherit;padding:8px}.button-row input{font:700 15px system-ui;padding:9px 14px}
"""
    script = r"""<script>
const form=document.getElementById('dispatch-form');
document.getElementById('torque').addEventListener('input',e=>document.getElementById('torque-value').textContent=e.target.value);
document.getElementById('preview').addEventListener('click',()=>document.getElementById('preview-state').textContent='Route preview ready; values are not submitted.');
document.getElementById('reference-form').addEventListener('submit',e=>{e.preventDefault();document.getElementById('preview-state').textContent='Reference image control activated.'});
form.addEventListener('submit',async e=>{e.preventDefault();const fd=new FormData(form),file=document.getElementById('attachment').files[0];const values={};for(const [key,value] of fd.entries())if(key!=='attachment'&&key!=='channels')values[key]=String(value);values.channels=[...document.getElementById('channels').selectedOptions].map(o=>o.value).sort();values.attachment_name=file?file.name:'';values.attachment_text=file?await file.text():'';const response=await fetch('/submit-form',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});if(response.ok)location.href='/interface';else document.getElementById('form-error').textContent='Values did not match the mission card.'});
</script>"""
    return _document("Dispatch control matrix", body, extra_style=style, script=script)


def _interface_page(*, error: bool = False) -> bytes:
    message = (
        "<p class='error'>The menu path was attempted out of order. Start from Operations.</p>"
        if error
        else ""
    )
    body = f"""
<main class="console"><h1>Routing console</h1>
<p>Traverse the nested command menu in order: <strong>Operations → Routing → Orbital → Gale Crater</strong>. Then expand Advanced clearance, open the approval dialog, and choose <strong>Delta-7</strong>.</p>{message}
<nav class="menubar"><button id="operations" type="button">Operations ▾</button>
<section id="operations-menu" class="menu" hidden><button id="routing" type="button">Routing ▸</button><button type="button">Diagnostics</button></section>
<section id="routing-menu" class="menu nested" hidden><button type="button">Ground</button><button id="orbital" type="button">Orbital ▸</button></section>
<section id="orbital-menu" class="menu deep" hidden><button type="button">Lunar relay</button><button id="gale" type="button">Gale Crater</button><button type="button">Utopia Planitia</button></section></nav>
<section id="selection" class="selection" hidden><h2>Gale Crater route selected</h2>
<details id="advanced"><summary>Advanced clearance</summary><div class="drawer"><p>Approval is inside the modal control below.</p><button id="open-approval" type="button">Open approval dialog</button></div></details></section>
<dialog id="approval"><form method="dialog"><h2>Orbital approval</h2><p>Choose the required clearance badge.</p>
<label for="badge">Clearance badge</label><select id="badge"><option value="">Choose…</option><option value="sigma-3">Sigma-3</option><option value="delta-7">Delta-7</option><option value="nova-2">Nova-2</option></select>
<button id="accept-approval" type="button" disabled>Accept clearance</button></form></dialog>
<p id="status" aria-live="polite"></p></main>"""
    style = """
.console{min-height:700px}.menubar{position:relative;background:#16263d;padding:10px;border-radius:8px;min-height:52px}.menubar>button{background:#edf3fb;color:#172033}.menu{position:absolute;z-index:3;top:58px;left:20px;width:210px;padding:8px;background:white;border:1px solid #6d7e95;box-shadow:0 8px 22px #17203344}.menu button{display:block;width:100%;margin:4px 0;text-align:left;background:#315c92}.menu.nested{left:220px;top:72px}.menu.deep{left:420px;top:114px}.selection{margin-top:220px;border:1px solid #91a0b4;padding:14px;border-radius:8px}.drawer{height:220px;overflow:auto;padding:120px 10px 10px;background:#edf2f7}dialog{border:0;border-radius:10px;box-shadow:0 12px 48px #0008;padding:24px;min-width:420px}dialog::backdrop{background:#172033aa}dialog select{width:100%;padding:9px;margin:8px 0 20px}
"""
    script = r"""<script>
async function step(name){const r=await fetch('/interface-event?event='+encodeURIComponent(name),{cache:'no-store'});return await r.json()}
document.getElementById('operations').addEventListener('click',async()=>{if((await step('operations')).accepted)document.getElementById('operations-menu').hidden=false});
document.getElementById('routing').addEventListener('click',async()=>{if((await step('routing')).accepted)document.getElementById('routing-menu').hidden=false});
document.getElementById('orbital').addEventListener('click',async()=>{if((await step('orbital')).accepted)document.getElementById('orbital-menu').hidden=false});
document.getElementById('gale').addEventListener('click',async()=>{if((await step('gale')).accepted){document.getElementById('selection').hidden=false;document.querySelectorAll('.menu').forEach(el=>el.hidden=true)}});
document.getElementById('open-approval').addEventListener('click',async()=>{if((await step('advanced')).accepted)document.getElementById('approval').showModal()});
document.getElementById('badge').addEventListener('change',async e=>{if(e.target.value!=='delta-7')return;const result=await step('delta-7');document.getElementById('accept-approval').disabled=!result.accepted});
document.getElementById('accept-approval').addEventListener('click',()=>location.href='/grid');
</script>"""
    return _document("Nested routing console", body, extra_style=style, script=script)


def _grid_page(*, error: bool = False) -> bytes:
    message = "<p class='error'>Some matching images remain. Inspect the fresh grid.</p>" if error else ""
    body = f"""
<main class="verify"><section class="challenge-head"><span>Select all squares with</span><strong id="instruction">loading…</strong><small>If there are none, click verify</small></section>{message}
<canvas id="grid" width="510" height="510" aria-label="Nine unlabeled real photograph tiles"></canvas>
<footer><button id="reload" type="button" title="Reload current challenge">↻</button><button id="verify" type="button">VERIFY</button></footer></main>"""
    style = """
.verify{max-width:546px;margin:4px auto;padding:14px 18px;border-radius:2px}.challenge-head{height:112px;background:#4a90e2;color:white;padding:14px 20px;display:flex;flex-direction:column;justify-content:center}.challenge-head span{font-size:17px}.challenge-head strong{font-size:31px;line-height:1.05}.challenge-head small{font-size:14px;margin-top:6px}canvas{display:block;width:510px;height:510px;background:#d7dee7;border:0}footer{height:62px;border:1px solid #d5d9df;border-top:0;display:flex;align-items:center;justify-content:space-between;padding:8px 12px}footer button{border-radius:2px}#reload{background:white;color:#68788c;font-size:27px;padding:2px 12px}#verify{background:#4a90e2}.error{margin:4px 0}
"""
    script = r"""<script>
const canvas=document.getElementById('grid'),ctx=canvas.getContext('2d');let view=null;
function photo(asset){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src='/photo/'+encodeURIComponent(asset)+'.jpg?g='+view.generation})}
function tile(image,index){const x=(index%3)*170,y=Math.floor(index/3)*170,w=170,h=170,scale=Math.max(w/image.naturalWidth,h/image.naturalHeight),sw=w/scale,sh=h/scale,sx=(image.naturalWidth-sw)/2,sy=(image.naturalHeight-sh)/2;ctx.drawImage(image,sx,sy,sw,sh,x,y,w,h);ctx.strokeStyle='#fff';ctx.lineWidth=5;ctx.strokeRect(x,y,w,h)}
async function draw(){ctx.fillStyle='#d7dee7';ctx.fillRect(0,0,510,510);const images=await Promise.all(view.tiles.map(photo));images.forEach(tile)}
async function load(){view=await(await fetch('/grid-state',{cache:'no-store'})).json();document.getElementById('instruction').textContent=view.instruction.replace('Select every image containing ','');await draw()}
canvas.addEventListener('click',async e=>{const r=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-r.left)/(r.width/3)),row=Math.floor((e.clientY-r.top)/(r.height/3)),index=row*3+col;const answer=await(await fetch('/grid-click?index='+index,{cache:'no-store'})).json();if(answer.accepted)await load()});
document.getElementById('verify').addEventListener('click',()=>location.href='/verify-grid');load();
document.getElementById('reload').addEventListener('click',load);
</script>"""
    return _document("reCAPTCHA-style photo checkpoint", body, extra_style=style, script=script)


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


def _load_photo_assets(cache_dir: Path) -> dict[str, bytes]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, bytes] = {}
    for record in _PHOTO_RECORDS:
        asset_id = str(record["id"])
        expected = str(record["sha256"])
        path = cache_dir / f"{asset_id}.jpg"
        try:
            payload = path.read_bytes()
        except OSError:
            payload = b""
        if hashlib.sha256(payload).hexdigest() != expected:
            request = Request(
                str(record["url"]),
                headers={
                    "User-Agent": (
                        "qwen-omni-browser-gauntlet/1.0 "
                        "(https://github.com/robit-man/qwen-omni-adapters)"
                    )
                },
            )
            with urlopen(request, timeout=30) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
            if not payload or len(payload) > 4 * 1024 * 1024:
                raise RuntimeError(f"photo asset {asset_id} has an invalid size")
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f"photo asset {asset_id} failed its SHA-256 pin")
            partial = path.with_suffix(".jpg.part")
            partial.write_bytes(payload)
            partial.replace(path)
        assets[asset_id] = payload
    return assets


def _handler(
    state: GauntletState,
    photo_assets: dict[str, bytes],
    upload_path: Path,
):
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

        def _json_values(self) -> dict[str, Any]:
            try:
                length = min(131072, int(self.headers.get("Content-Length", "0")))
                value = json.loads(self.rfile.read(length).decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return dict(value) if isinstance(value, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send(_form_page(upload_path=str(upload_path)))
            elif parsed.path == "/interface":
                self._send(_interface_page())
            elif parsed.path == "/interface-event":
                event = parse_qs(parsed.query).get("event", [""])[0]
                self._send(
                    json.dumps({"accepted": state.interface_event(event)}).encode(),
                    content_type="application/json",
                )
            elif parsed.path == "/grid":
                self._send(_grid_page())
            elif parsed.path.startswith("/photo/") and parsed.path.endswith(".jpg"):
                asset_id = parsed.path.removeprefix("/photo/").removesuffix(".jpg")
                payload = photo_assets.get(asset_id)
                if payload is None:
                    self._send(b"not found", status=HTTPStatus.NOT_FOUND)
                else:
                    self._send(payload, content_type="image/jpeg")
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
            if parsed.path == "/submit-form":
                values = self._json_values()
                if state.submit_form(values):
                    self._send(status=HTTPStatus.SEE_OTHER, location="/interface")
                else:
                    self._send(b"form rejected", status=HTTPStatus.BAD_REQUEST)
            elif parsed.path == "/submit-final":
                values = self._form_values()
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


def _browser_arguments(action: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(action.get("arguments") or "{}"))
    except ValueError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


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
        "set_value",
        "select",
        "upload",
        "scroll",
        "back",
    }
    invalid = [action for action in browser_actions if action not in allowed]
    required = {
        "navigate",
        "click",
        "visual_click",
        "drag",
        "type",
        "set_value",
        "select",
        "upload",
    }
    missing = required.difference(browser_actions)
    navigations = [
        _browser_arguments(action)
        for action in actions
        if isinstance(action, dict)
        and action.get("tool") == "browser_interact"
        and _browser_action(action) == "navigate"
    ]
    if task.get("status") != "completed":
        raise RuntimeError(f"Browser gauntlet ended with status {task.get('status')}")
    if forbidden:
        raise RuntimeError(f"Browser gauntlet bypassed visible execution: {sorted(forbidden)}")
    if invalid:
        raise RuntimeError(f"Browser gauntlet used invalid browser actions: {invalid}")
    if missing:
        raise RuntimeError(f"Browser gauntlet skipped action classes: {sorted(missing)}")
    if len(navigations) != 1 or not str(navigations[0].get("url") or "").startswith(
        "http://127.0.0.1:"
    ):
        raise RuntimeError(
            "Browser gauntlet navigated outside its single initial fixture entry"
        )
    if not snapshot["complete"] or snapshot["marker"] != state.marker:
        raise RuntimeError("Task completed without the fixture-owned terminal state")
    if len(snapshot["form_attempts"]) != 1 or not snapshot["form_passed"]:
        raise RuntimeError("The exact dispatch form was not submitted once")
    if snapshot["interface_events"] != [
        "operations",
        "routing",
        "orbital",
        "gale",
        "advanced",
        "delta-7",
    ]:
        raise RuntimeError("The nested interface path was not completed exactly once")
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
        "interface_steps": 6,
        "grid_rounds": len(_GRID_SPECS),
        "visual_hits": len(snapshot["grid_hits"]),
        "visual_misses": 0,
        "drag_attempts": 1,
        "actions": len(actions),
    }


def run(args: argparse.Namespace) -> int:
    state = GauntletState()
    photo_assets = _load_photo_assets(args.photo_cache)
    args.upload_root.mkdir(parents=True, exist_ok=True)
    upload_path = (args.upload_root / "gauntlet-packing-note.txt").resolve()
    upload_path.write_text(_FORM_EXPECTED["attachment_text"], encoding="utf-8")
    server = ThreadingHTTPServer(
        (args.bind, 0), _handler(state, photo_assets, upload_path)
    )
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
                    "dispatch control matrix, the deeply nested routing console, every real-"
                    "photo image-selection round, and the manifest transfer. Read the exact "
                    "values from each rendered page. Use browser_interact only: type for "
                    "ordinary text, set_value for native date/time/month/week/number/range/"
                    "color controls, select for dropdowns, and upload for the displayed file "
                    "path. Use DOM element_id actions for controls and the draggable crate. "
                    "Use fresh screenshots plus normalized_1000 visual_click with a precise "
                    "target phrase only for the canvas photo tiles. A correct tile may be "
                    "replaced by another real photograph, so re-inspect every fresh grid. Do "
                    "not inspect source, use shell/web tools, or use whole-desktop gui_interact."
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
        try:
            upload_path.unlink()
        except OSError:
            pass


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
    parser.add_argument(
        "--photo-cache",
        type=Path,
        default=Path("runtime-data/browser-gauntlet/photos"),
    )
    parser.add_argument(
        "--upload-root",
        type=Path,
        default=Path("runtime-data/browser-uploads"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
