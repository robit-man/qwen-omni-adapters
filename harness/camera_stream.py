"""Local persistent camera owner and authenticated portal transport bridge."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import os
import secrets
import socketserver
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import httpx

from harness.camera import _camera_identifier, _camera_label

logger = logging.getLogger(__name__)


def camera_socket() -> Path:
    # The system portal and desktop harness may have different environments.
    uid = os.getuid()
    runtime = Path(f"/run/user/{uid}")
    return (runtime if runtime.is_dir() else Path("/dev/shm")) / f"omni-camera-{uid}" / "control.sock"


def persistent_camera_enabled(repo: Path) -> bool:
    override = os.environ.get("OMNI_CALL_CAMERA_STREAM")
    return override == "1" if override is not None else (repo / "runtime-data/state/camera-stream.enabled").is_file()


def gmsl_devices() -> list[str]:
    devices = []
    for path in sorted(Path("/sys/class/video4linux").glob("video[0-9]*")):
        try:
            name = (path / "name").read_text(encoding="utf-8")
        except OSError:
            continue
        if "ecam_gmsl" in name:
            devices.append(f"/dev/{path.name}")
    return devices


class CameraStreamClient:
    """A same-user Unix socket connects the portal to the native capture owner."""

    def __init__(self, socket_path: Path | None = None):
        self.socket_path = socket_path

    def request(self, action: str, **args):
        try:
            path = self.socket_path or camera_socket()
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(path)), timeout=12) as client:
                return client.post(f"http://camera.local/{action}", json=args).json()
        except (httpx.HTTPError, ValueError):
            return {"error": "Persistent camera service is unavailable"}


def _local_server(address, hub):
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
        request_queue_size = 8

        def server_bind(self):
            super().server_bind()
            Path(self.server_address).chmod(0o600)

    server = Server(str(address), _Request)
    server.hub = hub
    return server


class _Request(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - HTTP handler API
        self.connection.settimeout(3)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                self.send_error(400)
                return
            args = json.loads(self.rfile.read(size))
            if not isinstance(args, dict):
                self.send_error(400)
                return
            response = self.server.hub.request(self.path, args)
        except (ValueError, OSError):
            self.send_error(400)
            return
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        pass


class CameraStreamHub:
    def __init__(self, devices, *, orientation=None, test_source=False, socket_path: Path | None = None):
        import fcntl

        self.address = socket_path or camera_socket()
        self.root = self.address.parent
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.owner_file = (self.root / "owner.lock").open("a")
        try:
            fcntl.flock(self.owner_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.address.unlink(missing_ok=True)
            self.server = _local_server(self.address, self)
        except OSError:
            self.owner_file.close()
            raise
        self.devices = devices
        self.test_source = test_source
        self.feeds = {}
        self.peers = {}
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run, name="camera-webrtc", daemon=True)
        self.loop_thread.start()
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.orientation = orientation
        self.stop_orientation = threading.Event()
        self.orientation_thread = None
        if self.orientation is not None:
            self.orientation_thread = threading.Thread(target=self._observe_orientation, name="camera-roll", daemon=True)
            self.orientation_thread.start()
        self.server_thread.start()

    def _observe_orientation(self):
        while not self.stop_orientation.wait(.4):
            for device in self.available():
                now = time.monotonic()
                if not self.orientation.due(device, now):
                    continue
                try:
                    if frame := self.jpeg(device, 640):
                        self.orientation.observe(device, frame, now)
                except (ValueError, OSError):
                    logger.debug("Camera orientation could not read a frame")

    def _orientation_status(self, device):
        return self.orientation.status(device) if self.orientation is not None else {"state": "unavailable"}

    def _run(self):
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.wait(pending))
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    def set_enabled(self, enabled: bool):
        from harness.camera_capture import CameraFeed

        with self.lock:
            if enabled and not self.feeds:
                for device in self.devices:
                    path = self.root / f"{Path(device).name}.sock"
                    path.unlink(missing_ok=True)
                    feed = CameraFeed(device, path, test_source=self.test_source)
                    self.feeds[device] = feed
                    feed.start()
            elif not enabled and self.feeds:
                feeds, self.feeds = self.feeds, {}
                self._async(self._close_peers())
                for feed in feeds.values():
                    feed.close()

    def available(self):
        return [device for device, feed in list(self.feeds.items()) if feed.available]

    def jpeg(self, device, width, *, grayscale=False, roll=0):
        feed = self.feeds.get(device)
        sample = feed.latest() if feed else None
        return sample.jpeg(width, grayscale=grayscale, roll=roll) if sample else None

    def clip(self, device, path, seconds, *, roll=0):
        """Use existing JPEG samples for explicit clips without reopening V4L2."""
        import av

        started = time.monotonic()
        deadline = started + min(10, max(.5, seconds))
        index = 0
        with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("libx264", rate=8, options={"preset": "ultrafast"})
            while time.monotonic() < deadline:
                if jpeg := self.jpeg(device, 512, roll=roll):
                    with av.open(io.BytesIO(jpeg)) as image:
                        frame = next(image.decode(video=0))
                        if index == 0:
                            stream.width, stream.height = frame.width // 2 * 2, frame.height // 2 * 2
                        frame = frame.reformat(width=stream.width, height=stream.height, format="yuv420p")
                        frame.pts, frame.time_base = index, Fraction(1, 8)
                        output.mux(stream.encode(frame))
                    index += 1
                time.sleep(max(.01, started + index / 8 - time.monotonic()) if index else .1)
            if index:
                output.mux(stream.encode(None))
        return path if index else None

    def _async(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=8)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {"error": "Camera signaling timed out"}

    def request(self, action, args):
        devices = self.available()
        if action == "/list":
            return {"cameras": [{"id": _camera_identifier(device), "label": _camera_label(device, index)}
                                for index, device in enumerate(devices)]}
        if action == "/status":
            return {"feeds": [{"id": _camera_identifier(device), "available": feed.available,
                               "analysis_frames": feed.analysed, "width": feed.quality.width,
                               "fps": feed.quality.fps, "bitrate_kbps": feed.quality.bitrate_kbps,
                               "orientation": self._orientation_status(device)}
                              for device, feed in list(self.feeds.items())]}
        owner = args.get("owner")
        if action == "/close" and isinstance(owner, str):
            return self._async(self._close_peer(args.get("peer"), owner))
        if action == "/frames":
            selected = args.get("camera_ids")
            frames = []
            for device in devices:
                identity = _camera_identifier(device)
                if selected is not None and identity not in selected:
                    continue
                if jpeg := self.jpeg(device, 960):
                    frames.append({"id": identity, "mime_type": "image/jpeg", "encoding": "base64",
                                   "data": base64.b64encode(jpeg).decode("ascii")})
            return {"frames": frames}
        if action != "/offer" or not isinstance(owner, str):
            return {"error": "Unknown camera request"}
        feed = next((self.feeds.get(device) for device in devices
                     if _camera_identifier(device) == args.get("camera")), None)
        if feed is None:
            return {"error": "Camera is unavailable"}
        sdp = args.get("sdp")
        if not isinstance(sdp, str) or not 0 < len(sdp) < 16000:
            return {"error": "Invalid video offer"}
        return self._async(self._offer(feed, sdp, owner))

    async def _offer(self, feed, sdp, owner):
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription

        if len(self.peers) >= 6:
            return {"error": "Camera viewer capacity reached"}
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        identity = secrets.token_urlsafe(24)
        track = _video_track(feed)
        self.peers[identity] = (peer, track, owner)

        @peer.on("connectionstatechange")
        async def changed():
            if peer.connectionState in {"failed", "closed"}:
                await self._close_peer(identity, owner)

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
            transceiver = next(item for item in peer.getTransceivers() if item.kind == "video")
            transceiver.sender.replaceTrack(track)
            transceiver.direction = "sendonly"
            transceiver.setCodecPreferences([codec for codec in RTCRtpSender.getCapabilities("video").codecs
                                            if codec.mimeType == "video/H264"
                                            and codec.parameters.get("profile-level-id", "").lower() == "42e01f"])
            await peer.setLocalDescription(await peer.createAnswer())
        except (ValueError, StopIteration, RuntimeError):
            await self._close_peer(identity, owner)
            return {"error": "The camera requires an H.264 video receiver"}
        except asyncio.CancelledError:
            await self._close_peer(identity, owner)
            raise

        async def expire_unconnected():
            await asyncio.sleep(20)
            if peer.connectionState != "connected":
                await self._close_peer(identity, owner)

        asyncio.create_task(expire_unconnected())
        return {"peer": identity, "type": "answer", "sdp": peer.localDescription.sdp}

    async def _close_peer(self, identity, owner):
        entry = self.peers.get(identity) if isinstance(identity, str) else None
        if entry is not None and entry[2] == owner:
            peer, track, _ = self.peers.pop(identity)
            track.stop()
            await peer.close()
            await asyncio.to_thread(track.close_encoder)
        return {"closed": bool(entry and entry[2] == owner)}

    async def _close_peers(self):
        for identity, (_, _, owner) in list(self.peers.items()):
            await self._close_peer(identity, owner)

    def close(self):
        self.stop_orientation.set()
        if self.orientation_thread is not None:
            self.orientation_thread.join(timeout=2)
        self.set_enabled(False)
        self.server.shutdown()
        self.server.server_close()
        self.address.unlink(missing_ok=True)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=2)
        self.owner_file.close()


def _video_track(feed):
    import av
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError

    from harness.camera_capture import CameraEncoder

    class EncodedTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self.encoder = None
            self.profile = None
            self.origin = time.monotonic()
            self.due = self.origin
            self.changed = self.origin - 2
            self.encoder_lock = threading.Lock()

        def next_packet(self, profile):
            with self.encoder_lock:
                sample = feed.latest()
                if sample is None or self.readyState != "live":
                    return None
                if self.profile != profile:
                    if self.encoder is not None:
                        self.encoder.close()
                    self.encoder = CameraEncoder(feed, profile)
                    self.profile = profile
                    self.changed = time.monotonic()
                return self.encoder.encode(sample)

        def close_encoder(self):
            with self.encoder_lock:
                if self.encoder is not None:
                    self.encoder.close()
                    self.encoder = None

        async def recv(self):
            while self.readyState == "live" and feed.available:
                now = time.monotonic()
                proposed = feed.quality
                # Rate and spatial changes apply to a fresh hardware encoder.
                # Escalations can burst; hold downgrades for stable SPS/PPS.
                profile = self.profile
                if profile is None or now - self.changed >= 1 or proposed.fps > profile.fps:
                    profile = proposed
                if now >= self.due or profile != self.profile:
                    data = await asyncio.to_thread(self.next_packet, profile)
                    self.due = now + 1 / profile.fps
                    if data:
                        packet = av.Packet(data)
                        packet.pts = packet.dts = round((now - self.origin) * 90000)
                        packet.time_base = Fraction(1, 90000)
                        return packet
                await asyncio.sleep(.01)
            raise MediaStreamError

    return EncodedTrack()
