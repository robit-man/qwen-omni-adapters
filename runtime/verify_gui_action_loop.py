#!/usr/bin/env python3
"""Run a visible, model-driven GUI coordinate challenge through the live harness."""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

ACTIVE_TASK_STATUSES = {"pending", "running", "waiting", "paused"}


class ChallengeState:
    """Authoritative state is outside the page so reloads cannot fake success."""

    expected_points = ((710, 265), (630, 380), (275, 505))

    def __init__(self) -> None:
        self.stage = 0
        self.hits: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.marker = f"GUI-ACTION-PASS-{secrets.token_hex(4).upper()}"

    def attempt(self, x: int, y: int) -> dict[str, Any]:
        with self.lock:
            if self.stage >= len(self.expected_points):
                return {"accepted": True, "stage": self.stage, "marker": self.marker}
            expected_x, expected_y = self.expected_points[self.stage]
            accepted = abs(x - expected_x) <= 34 and abs(y - expected_y) <= 34
            self.hits.append(
                {"stage": self.stage, "x": x, "y": y, "accepted": accepted}
            )
            if accepted:
                self.stage += 1
            return {
                "accepted": accepted,
                "stage": self.stage,
                "marker": self.marker if self.stage == len(self.expected_points) else "",
            }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "stage": self.stage,
                "complete": self.stage == len(self.expected_points),
                "marker": self.marker if self.stage == len(self.expected_points) else "",
                "hits": list(self.hits),
            }


def _page(marker: str) -> bytes:
    # All selectable targets live inside one canvas. Browser DOM element maps can
    # navigate here, but only visual grounding plus GUI coordinates can solve it.
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Omni GUI action-loop evaluation</title>
<style>
html,body{{margin:0;background:#0d1117;color:#f0f6fc;font:20px system-ui}}
main{{width:940px;margin:16px auto}} canvas{{background:#161b22;border:3px solid #58a6ff}}
</style></head><body><main><canvas id="board" width="920" height="620"></canvas></main>
<script>
const c=document.getElementById('board'),x=c.getContext('2d'); let stage=0;
const colors={{blue:'#2386e8',green:'#2da44e',amber:'#d29922',purple:'#a371f7',red:'#f85149'}};
function shape(kind,cx,cy,color){{x.fillStyle=color;x.strokeStyle='#fff';x.lineWidth=3;x.beginPath();
 if(kind==='circle')x.arc(cx,cy,25,0,Math.PI*2);
 else if(kind==='triangle'){{x.moveTo(cx,cy-29);x.lineTo(cx+29,cy+25);x.lineTo(cx-29,cy+25);x.closePath();}}
 else if(kind==='diamond'){{x.moveTo(cx,cy-29);x.lineTo(cx+29,cy);x.lineTo(cx,cy+29);x.lineTo(cx-29,cy);x.closePath();}}
 else{{for(let i=0;i<10;i++){{let a=-Math.PI/2+i*Math.PI/5,r=i%2?12:30;x.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r)}}x.closePath();}}
 x.fill();x.stroke();}}
function label(text){{x.fillStyle='#f0f6fc';x.font='bold 28px system-ui';x.fillText(text,34,48);}}
function draw(){{x.clearRect(0,0,c.width,c.height);x.fillStyle='#161b22';x.fillRect(0,0,c.width,c.height);
 if(stage===0){{label('Stage 1 of 3 — click the BLUE TRIANGLE');
  const items=[['red','circle'],['green','triangle'],['purple','diamond'],['amber','star'],['blue','circle'],['red','diamond'],['green','star'],['blue','triangle'],['amber','circle'],['purple','triangle'],['blue','star'],['green','diamond']];
  items.forEach((v,i)=>shape(v[1],125+(i%4)*195,105+Math.floor(i/4)*160,colors[v[0]]));}}
 else if(stage===1){{label('Stage 2 of 3 — in the modal, click the AMBER STAR');
  x.fillStyle='#05070a99';x.fillRect(0,70,920,550);x.fillStyle='#f0f6fc';x.fillRect(160,100,600,430);
  x.strokeStyle='#a371f7';x.lineWidth=5;x.strokeRect(160,100,600,430);x.fillStyle='#24292f';x.font='bold 25px system-ui';x.fillText('Visual modal — choose one image',205,150);
  [['green','circle',270,255],['purple','diamond',450,255],['blue','triangle',630,255],['red','star',270,380],['green','diamond',450,380],['amber','star',630,380]].forEach(v=>shape(v[1],v[2],v[3],colors[v[0]]));}}
 else if(stage===2){{label('Stage 3 of 3 — click the PURPLE DIAMOND in the lower strip');
  [['amber','circle',115],['blue','star',275],['green','triangle',435],['red','diamond',595],['purple','circle',755]].forEach(v=>shape(v[1],v[2],245,colors[v[0]]));
  x.fillStyle='#21262d';x.fillRect(65,445,790,120);
  [['blue','circle',115],['purple','diamond',275],['amber','triangle',435],['green','star',595],['red','circle',755]].forEach(v=>shape(v[1],v[2],505,colors[v[0]]));}}
 else{{x.fillStyle='#2da44e';x.fillRect(45,170,830,250);x.fillStyle='white';x.font='bold 35px system-ui';x.fillText('ALL VISUAL ACTION GATES PASSED',105,265);x.font='bold 28px monospace';x.fillText('{marker}',145,335);}}
}}
c.addEventListener('click',async e=>{{const r=c.getBoundingClientRect(),sx=c.width/r.width,sy=c.height/r.height;
 const px=Math.round((e.clientX-r.left)*sx),py=Math.round((e.clientY-r.top)*sy);
 const response=await fetch('/hit?x='+px+'&y='+py,{{cache:'no-store'}});const result=await response.json();
 if(result.accepted)stage=result.stage;draw();}});draw();
</script></body></html>""".encode()


def _handler(state: ChallengeState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                body = _page(state.marker)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/hit":
                query = parse_qs(parsed.query)
                try:
                    result = state.attempt(int(query["x"][0]), int(query["y"][0]))
                except (KeyError, ValueError, IndexError):
                    result = {"accepted": False, "error": "invalid coordinates"}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/state":
                body = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = b"not found"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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


def run(args: argparse.Namespace) -> int:
    state = ChallengeState()
    server = ThreadingHTTPServer((args.bind, 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    challenge_url = f"http://{args.bind}:{server.server_port}/"
    token = _access_token(args.status_path)
    client = httpx.Client(timeout=30, follow_redirects=False)
    task_id = ""
    try:
        listed = _tool(client, args.portal, token, "background_task", {"action": "list"})
        active = [
            task
            for task in listed.get("tasks", [])
            if isinstance(task, dict) and task.get("status") in ACTIVE_TASK_STATUSES
        ]
        if active and not args.allow_concurrent:
            raise RuntimeError(
                "An unfinished background task already owns the desktop; finish or "
                "cancel it before running this end-to-end GUI gate."
            )
        started = _tool(
            client,
            args.portal,
            token,
            "background_task",
            {
                "action": "start",
                "independent": bool(active),
                "objective": (
                    f"Open {challenge_url} in the visible Chromium window. Complete all "
                    "three instructions drawn inside its canvas using fresh gui_interact "
                    "screenshots and coordinate clicks. browser_interact may be used only "
                    "to navigate to the URL; do not use shell, fetched page text, DOM click "
                    "targets, or source inspection."
                ),
                "completion_criteria": (
                    f"The visible canvas shows the exact marker {state.marker}. Cite the "
                    "fresh GUI evidence and only then complete the task."
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
            if task.get("status") in {"complete", "blocked", "cancelled", "failed"}:
                break
        else:
            _tool(
                client,
                args.portal,
                token,
                "background_task",
                {"action": "cancel", "task_id": task_id},
            )
            raise RuntimeError("Timed out waiting for the GUI task to finish")

        challenge = state.snapshot()
        actions = task.get("actions") if isinstance(task.get("actions"), list) else []
        forbidden = {
            str(action.get("tool") or "")
            for action in actions
            if isinstance(action, dict)
        }.intersection({"shell", "web_fetch", "web_search"})
        if task.get("status") != "complete":
            raise RuntimeError(f"GUI task ended with status {task.get('status')}")
        if not challenge["complete"] or challenge["marker"] != state.marker:
            raise RuntimeError("The task completed without passing the canvas hit gates")
        if forbidden:
            raise RuntimeError(f"GUI task bypassed visual execution with {sorted(forbidden)}")
        if len([hit for hit in challenge["hits"] if hit["accepted"]]) != 3:
            raise RuntimeError("The challenge did not record exactly three accepted hits")
        print(
            json.dumps(
                {
                    "passed": True,
                    "task_id": task_id,
                    "accepted_hits": 3,
                    "misses": len(
                        [hit for hit in challenge["hits"] if not hit["accepted"]]
                    ),
                    "actions": len(actions),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal", default="http://127.0.0.1:8920")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--timeout-s", type=float, default=600)
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
