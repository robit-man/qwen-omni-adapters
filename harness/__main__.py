"""Run the always-listening call harness against a local omni adapter.

Ships with the adapter so a host that has the model has the conversation too:
microphone in, speakers out, state in the top bar, no browser involved.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from harness.audio import require_tools
from harness.call import CallConfig, TurnResult, run_call_loop
from harness.camera import CameraSet
from harness.indicator import ThreadedIndicator, build_indicator
from harness.residency import SpeechResidency
from harness.respeaker import find_source

logger = logging.getLogger("omni.harness")

DEFAULT_PORTAL = "http://127.0.0.1:8920"
DEFAULT_TOKEN_FILE = "runtime-data/state/access-token.txt"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _published_access_url(repo_root: Path) -> str:
    """Read the URL owned by the currently live daemon and tunnel only."""

    status_path = repo_root / "runtime-data" / "state" / "daemon-status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(status, dict) or status.get("state") != "ready":
        return ""
    try:
        daemon_pid = int(status.get("pid") or 0)
    except (TypeError, ValueError):
        return ""
    if not _pid_alive(daemon_pid):
        return ""
    children = status.get("children")
    tunnel_pid = 0
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict) and child.get("name") == "cloudflared":
                try:
                    tunnel_pid = int(child.get("pid") or 0)
                except (TypeError, ValueError):
                    tunnel_pid = 0
                break
    if not _pid_alive(tunnel_pid):
        return ""
    access_url = str(status.get("access_url") or "").strip()
    if not re.fullmatch(
        r"https://[-a-z0-9]+\.trycloudflare\.com/#access=[A-Za-z0-9_-]+",
        access_url,
    ):
        return ""
    return access_url


def _repo_root() -> Path:
    return Path(os.environ.get("OMNI_REPO_ROOT") or Path(__file__).resolve().parents[1])


def _read_token(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    env = os.environ.get("OMNI_PORTAL_TOKEN", "").strip()
    if env:
        return env
    path = _repo_root() / DEFAULT_TOKEN_FILE
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise SystemExit(
            f"no portal token: pass --token, set OMNI_PORTAL_TOKEN, or start the "
            f"daemon so it writes {path} ({error})"
        ) from error


def _wait_for_portal(
    url: str, token: str, timeout_s: float, reader: Any = None
) -> tuple[dict[str, Any], str]:
    """Block until the adapter answers, so the first utterance is not lost.

    The token is read again on every attempt. The daemon mints a fresh one
    each time it starts, so a harness that waits on the token it had at boot
    polls forever against a key that died when the adapter restarted -- which
    looks, from the outside, exactly like a harness that never came up.
    """

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if reader is not None:
            try:
                fresh = (reader() or "").strip()
                if fresh and fresh != token:
                    logger.info("portal token changed while waiting; using the new one")
                    token = fresh
            except Exception:  # noqa: BLE001 - keep waiting rather than give up
                pass
        try:
            response = httpx.get(
                f"{url.rstrip('/')}/api/status",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json(), token
            last = f"HTTP {response.status_code}"
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
        time.sleep(3)
    raise SystemExit(f"the omni adapter never became ready at {url} ({last})")


def _unused_frame_grabber(device: str) -> Any:
    """Capture one JPEG from the camera, or nothing if it cannot be read.

    A frame is attached to every spoken turn so "what am I holding" needs no
    special mode. The live-call prompt tells the model to use it only when it
    is relevant, so an unused frame costs a little comprehension time and
    nothing else.
    """

    if shutil.which("ffmpeg") is None:
        return None

    def grab() -> dict[str, Any] | None:
        try:
            completed = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "v4l2", "-i", device,
                    "-frames:v", "1", "-vf", "scale=768:-2",
                    "-f", "image2", "-c:v", "mjpeg", "-",
                ],
                capture_output=True,
                timeout=6,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            logger.debug("camera frame unavailable: %s", error)
            return None
        if completed.returncode != 0 or not completed.stdout:
            logger.debug("camera frame unavailable: %s", completed.stderr[:120])
            return None
        return {
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "data": base64.b64encode(completed.stdout).decode("ascii"),
        }

    return grab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omni-call", description=__doc__)
    parser.add_argument("--portal", default=os.environ.get("OMNI_PORTAL_URL", DEFAULT_PORTAL))
    parser.add_argument("--token", default=None)
    parser.add_argument("--model", default=os.environ.get("OMNI_MODEL", ""))
    parser.add_argument("--input-device", default=os.environ.get("OMNI_CALL_INPUT") or None)
    parser.add_argument("--output-device", default=os.environ.get("OMNI_CALL_OUTPUT") or None)
    parser.add_argument("--input-channels", type=int, default=int(os.environ.get("OMNI_CALL_INPUT_CHANNELS", "1")))
    parser.add_argument("--input-channel", type=int, default=int(os.environ.get("OMNI_CALL_INPUT_CHANNEL", "0")))
    parser.add_argument("--camera-device", default=os.environ.get("OMNI_CALL_CAMERA", "/dev/video0"))
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--camera-device-only",
        action="store_true",
        help="use only --camera-device instead of every camera found",
    )
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--reasoning", action="store_true", help="leave reasoning on (slower to first word)")
    parser.add_argument("--no-indicator", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--memory-path",
        default=os.environ.get("OMNI_CALL_MEMORY")
        or str(Path(os.environ.get("OMNI_REPO_ROOT") or ".") / "runtime-data/memory.sqlite3"),
    )
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    require_tools()

    token = _read_token(args.token)
    status, token = _wait_for_portal(
        args.portal, token, args.ready_timeout, lambda: _read_token(args.token)
    )
    model = args.model or str(status.get("model") or "")
    if not model:
        raise SystemExit("the adapter did not report a model tag")

    # Pick the microphone rather than assuming one. Asking a two-channel
    # laptop microphone for the array's six channels is how a harness that
    # hard-codes the ReSpeaker fails on every other machine.
    device, channels, channel = args.input_device, args.input_channels, args.input_channel
    if device is None:
        detected, detected_channels, detected_channel = find_source()
        if detected is not None:
            device, channels, channel = detected, detected_channels, detected_channel
        else:
            channels, channel = 1, 0
    logger.info("microphone: %s (%d channel(s), using %d)", device or "default", channels, channel)

    camera_on = bool(args.camera_device) and not args.no_camera
    eviction_unit = os.environ.get("OMNI_CALL_SPEECH_EVICT_UNIT", "").strip()
    residency = (
        SpeechResidency(
            eviction_unit,
            os.environ.get(
                "OMNI_CALL_COMPREHENSION_HEALTH",
                "http://127.0.0.1:8901/health",
            ),
        )
        if eviction_unit
        else None
    )
    config = CallConfig(
        portal_url=args.portal,
        token=token,
        model=model,
        input_device=device,
        output_device=args.output_device,
        input_channels=max(1, channels),
        input_channel=max(0, channel),
        tools_enabled=not args.no_tools,
        reasoning_enabled=bool(args.reasoning),
        camera_enabled=camera_on,
        camera_device=args.camera_device,
        # The daemon mints a fresh token every time it starts, so the harness
        # has to be able to go and look again rather than holding a dead key.
        token_reader=lambda: _read_token(args.token),
        memory_path="" if args.no_memory else args.memory_path,
        memory_calibration_path=(
            ""
            if args.no_memory
            else os.environ.get("OMNI_COMPREHENSION_CALIBRATION_FILE")
            or str(_repo_root() / "runtime-data/state/comprehension-memory.json")
        ),
        background_task_path=(
            ""
            if args.no_tools
            else os.environ.get("OMNI_BACKGROUND_TASKS")
            or str(_repo_root() / "runtime-data/state/background-tasks.json")
        ),
        prepare_speech=residency.prepare_speech if residency else None,
        restore_after_speech=residency.restore if residency else None,
        await_comprehension=residency.await_ready if residency else None,
        comprehension_ready=residency.ready_now if residency else None,
    )
    logger.info(
        "call harness ready: model=%s tools=%s reasoning=%s camera=%s speech_eviction=%s",
        model,
        config.tools_enabled,
        config.reasoning_enabled,
        config.camera_device if camera_on else "off",
        eviction_unit or "off",
    )

    stop = threading.Event()
    muted = threading.Event()
    cameras = CameraSet.discover(args.camera_device if args.camera_device_only else None)

    def set_tools(value: bool) -> None:
        config.tools_enabled = value
        logger.info("tools %s", "enabled" if value else "disabled")

    def set_reasoning(value: bool) -> None:
        config.reasoning_enabled = value
        logger.info("reasoning %s", "enabled" if value else "disabled")

    def set_camera(value: bool) -> None:
        config.camera_enabled = value
        logger.info("cameras %s", "enabled" if value else "disabled")

    def public_link() -> str:
        """The tunnel's URL with its key, as the daemon published it."""

        return _published_access_url(_repo_root())

    indicator = (
        build_indicator(
            on_mute=lambda value: muted.set() if value else muted.clear(),
            on_quit=stop.set,
            on_tools=set_tools,
            on_reasoning=set_reasoning,
            on_camera=set_camera,
            tools_enabled=config.tools_enabled,
            reasoning_enabled=config.reasoning_enabled,
            camera_enabled=config.camera_enabled,
            endpoint=public_link,
        )
        if not args.no_indicator
        else None
    )

    def on_state(state: str, detail: str) -> None:
        if indicator is not None:
            indicator.set_state(state, detail)

    def on_turn(result: TurnResult) -> None:
        if result.error:
            logger.warning("turn failed: %s", result.error)
            return
        logger.info(
            "turn: heard=%r reply=%r first_audio=%s total=%.0fms%s%s",
            result.transcript[:60],
            result.reply[:60],
            f"{result.first_audio_ms:.0f}ms" if result.first_audio_ms else "none",
            result.total_ms,
            f" tools={','.join(result.tools_used)}" if result.tools_used else "",
            " (interrupted)" if result.interrupted else "",
        )

    def guarded_frame(motion: bool = False) -> dict[str, Any] | None:
        """Every camera at once: a still, or a clip when the question is about time."""

        if muted.is_set() or not config.camera_enabled or not cameras.available:
            return None
        return cameras.clip() if motion else cameras.snapshot()

    def worker() -> None:
        while not stop.is_set():
            try:
                run_call_loop(
                    config,
                    on_state=lambda state, detail: on_state(
                        "muted" if muted.is_set() else state, detail
                    ),
                    on_turn=on_turn,
                    frame_grabber=guarded_frame,
                    stop=stop,
                )
            except Exception as error:  # noqa: BLE001 - keep listening
                logger.warning("call loop restarting after: %s", error)
                on_state("offline", str(error)[:60])
                if stop.wait(5):
                    return

    if indicator is None:
        worker()
        return 0

    runner = ThreadedIndicator(indicator, worker)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        runner.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
