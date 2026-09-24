"""Loopback-only browser view of the horizontally stitched camera set."""

from __future__ import annotations

import base64
import logging
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness.camera import CameraSet

logger = logging.getLogger(__name__)


class CameraLiveView:
    """Serve fresh stitched frames on an unguessable loopback URL."""

    def __init__(
        self,
        cameras: CameraSet,
        *,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.cameras = cameras
        self.enabled = enabled or (lambda: True)
        self._token = secrets.token_urlsafe(18)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        with self._lock:
            server = self._server
        if server is None:
            return ""
        return f"http://127.0.0.1:{server.server_port}/{self._token}/"

    def start(self) -> str:
        """Start lazily and return the local capability URL."""

        with self._lock:
            if self._server is not None:
                return f"http://127.0.0.1:{self._server.server_port}/{self._token}/"
            owner = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, _format: str, *_args: object) -> None:
                    return

                def _send(self, status: int, content_type: str, data: bytes) -> None:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                    root = f"/{owner._token}/"
                    if self.path.split("?", 1)[0] == root:
                        self._send(200, "text/html; charset=utf-8", _VIEW_HTML)
                        return
                    if self.path.split("?", 1)[0] != f"{root}frame.jpg":
                        self._send(404, "text/plain; charset=utf-8", b"Not found\n")
                        return
                    if not owner.enabled():
                        self._send(403, "text/plain; charset=utf-8", b"Cameras disabled\n")
                        return
                    frame = owner.cameras.snapshot()
                    if not frame or frame.get("mime_type") != "image/jpeg":
                        self._send(503, "text/plain; charset=utf-8", b"No camera frame\n")
                        return
                    try:
                        data = base64.b64decode(str(frame.get("data") or ""), validate=True)
                    except ValueError:
                        data = b""
                    if not data:
                        self._send(503, "text/plain; charset=utf-8", b"No camera frame\n")
                        return
                    self._send(200, "image/jpeg", data)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever,
                name="omni-camera-live-view",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/{self._token}/"
        logger.info("local stitched camera view started on loopback")
        return url

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)


_VIEW_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Omni stitched camera view</title>
<style>
html,body{margin:0;min-height:100%;background:#050608;color:#e9eef5;font:14px sans-serif}
main{display:grid;min-height:100vh;place-items:center}img{display:block;max-width:100vw;max-height:100vh}
#state{position:fixed;left:12px;bottom:10px;padding:6px 9px;background:#050608cc;border-radius:5px}
</style></head><body><main><img id="view" alt="Horizontally stitched live camera view"></main>
<div id="state">Opening cameras…</div><script>
const image=document.getElementById('view'), state=document.getElementById('state');
function next(){image.src='frame.jpg?t='+Date.now()}
image.onload=()=>{state.textContent='Live · cameras left to right';setTimeout(next,120)};
image.onerror=()=>{state.textContent='Waiting for camera frames…';setTimeout(next,900)};
next();
</script></body></html>""".encode()
