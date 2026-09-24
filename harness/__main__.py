"""Run the always-listening call harness against a local omni adapter.

Ships with the adapter so a host that has the model has the conversation too:
microphone in, speakers out, state in the top bar, no browser involved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from harness.audio import probe_audio_server, require_tools
from harness.call import CallConfig, TurnResult, run_call_loop
from harness.camera import CameraSet
from harness.camera_view import CameraLiveView
from harness.indicator import ThreadedIndicator, build_indicator, probe_indicator
from harness.location import BrowserLocationProvider
from harness.models import IndicatorModelManager
from harness.residency import SpeechResidency
from harness.respeaker import find_source
from harness.update import RepositoryUpdateManager
from portal.background_tasks import BackgroundTaskStore

logger = logging.getLogger("omni.harness")

DEFAULT_PORTAL = "http://127.0.0.1:8920"
DEFAULT_TOKEN_FILE = "runtime-data/state/access-token.txt"
HARNESS_STATUS_FILE = "runtime-data/state/harness-status.json"
_STATUS_LOCK = threading.Lock()


def _env_enabled(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _available_token(explicit: str | None) -> str:
    """Return the current portal token, or empty while the daemon mints it."""

    if explicit:
        return explicit.strip()
    env = os.environ.get("OMNI_PORTAL_TOKEN", "").strip()
    if env:
        return env
    path = _repo_root() / DEFAULT_TOKEN_FILE
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_token(explicit: str | None) -> str:
    """Read a required token for callers that are not allowed to wait."""

    token = _available_token(explicit)
    if token:
        return token
    path = _repo_root() / DEFAULT_TOKEN_FILE
    raise SystemExit(
        f"no portal token: pass --token, set OMNI_PORTAL_TOKEN, or start the "
        f"daemon so it writes {path}"
    )


def _write_harness_status(
    repo_root: Path,
    *,
    state: str,
    indicator_backend: str,
    detail: str = "",
) -> None:
    """Publish a privacy-bounded liveness record for deployment and support."""

    path = repo_root / HARNESS_STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "robit.omni-call-harness.status.v1",
        "pid": os.getpid(),
        "state": state,
        "indicator_backend": indicator_backend,
        "updated_at": time.time(),
    }
    if detail:
        value["detail"] = detail[:160]
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with _STATUS_LOCK:
        partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        partial.chmod(0o600)
        os.replace(partial, path)


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
    token = token.strip()
    while time.monotonic() < deadline:
        if reader is not None:
            try:
                fresh = (reader() or "").strip()
                if fresh and fresh != token:
                    logger.info("portal token changed while waiting; using the new one")
                    token = fresh
            except Exception:  # noqa: BLE001 - keep waiting rather than give up
                pass
        if not token:
            last = "waiting for the daemon to publish its access token"
            time.sleep(3)
            continue
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify desktop indicator and audio prerequisites, then exit",
    )
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
    if args.check:
        backend = probe_indicator()
        audio = probe_audio_server()
        print(
            json.dumps(
                {"ok": True, "indicator_backend": backend, "audio": audio},
                sort_keys=True,
            )
        )
        return 0

    indicator_required = _env_enabled("OMNI_REQUIRE_INDICATOR")
    if indicator_required and args.no_indicator:
        raise SystemExit("OMNI_REQUIRE_INDICATOR=1 conflicts with --no-indicator")
    repo_root = _repo_root()
    _write_harness_status(
        repo_root,
        state="starting",
        indicator_backend="required" if indicator_required else "pending",
    )
    # This is a browser-side HTTPS lookup whose sanitized result is later
    # attached by the local client. Start it while the model stack is loading
    # so the first spoken location request does not race the lookup.
    client_location = BrowserLocationProvider()
    if not args.no_tools:
        client_location.start()

    # The desktop service intentionally starts in parallel with the core
    # service. The first boot therefore often precedes access-token creation;
    # absence is a normal startup state, not a reason to crash-loop.
    token = _available_token(args.token)
    status, token = _wait_for_portal(
        args.portal, token, args.ready_timeout, lambda: _available_token(args.token)
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

    def content_trace(event: str, text: str, details: dict[str, Any]) -> None:
        logger.info(
            "conversation_trace %s",
            json.dumps(
                {"event": event, "text": text, **details},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
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
        token_reader=lambda: _available_token(args.token),
        client_location_reader=client_location.get,
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
        content_trace=(
            content_trace if _env_enabled("OMNI_CALL_LOG_CONTENT") else None
        ),
    )
    logger.info(
        "call harness ready: model=%s tools=%s reasoning=%s camera=%s "
        "speech_eviction=%s content_trace=%s",
        model,
        config.tools_enabled,
        config.reasoning_enabled,
        config.camera_device if camera_on else "off",
        eviction_unit or "off",
        config.content_trace is not None,
    )

    stop = threading.Event()
    reload_requested = threading.Event()
    model_switch_requested = threading.Event()
    muted = threading.Event()
    cameras = CameraSet.discover(args.camera_device if args.camera_device_only else None)
    camera_view = CameraLiveView(cameras, enabled=lambda: config.camera_enabled)
    indicator_holder: dict[str, Any] = {}

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

    def request_reload() -> None:
        logger.info("indicator requested a clean voice-service reload")
        reload_requested.set()
        stop.set()
        live_indicator = indicator_holder.get("value")
        if live_indicator is not None:
            live_indicator.stop()

    update_manager = (
        RepositoryUpdateManager(repo_root, on_installed=request_reload)
        if not args.no_indicator
        else None
    )

    model_manager = IndicatorModelManager(repo_root, model)

    def model_action(tag: str, action: str) -> tuple[bool, str]:
        ok, detail = model_manager.action(tag, action)
        logger.info("indicator model action %s %s: %s", action, tag, detail)
        if ok and action == "activate":
            # The manager atomically updates .env and asks the system daemon to
            # restart. Exit this user service too, so systemd relaunches it with
            # the selected OMNI_MODEL rather than retaining stale environment.
            model_switch_requested.set()
            stop.set()
        return ok, detail

    indicator_task_store = (
        BackgroundTaskStore(config.background_task_path)
        if config.background_task_path
        else None
    )
    task_archive = _repo_root() / "runtime-data" / "state" / "background-task-archive.log"

    def clear_finished_tasks() -> int:
        if indicator_task_store is None:
            return 0
        count = indicator_task_store.archive_terminal(task_archive)
        logger.info("archived %d finished background task(s) to %s", count, task_archive)
        return count

    def cancel_task(task_id: str) -> bool:
        if indicator_task_store is None:
            return False
        task = indicator_task_store.cancel(task_id)
        cancelled = bool(task is not None and task.get("status") == "cancelled")
        logger.info("indicator cancelled background task %s: %s", task_id, cancelled)
        return cancelled

    def clear_task(task_id: str) -> bool:
        if indicator_task_store is None:
            return False
        removed = indicator_task_store.remove(task_id)
        logger.info("indicator cleared background task %s: %s", task_id, removed)
        return removed

    def open_task_archive() -> None:
        task_archive.parent.mkdir(parents=True, exist_ok=True)
        task_archive.touch(exist_ok=True)
        try:
            subprocess.Popen(  # noqa: S603 - fixed local desktop opener
                ["xdg-open", str(task_archive)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            logger.warning("could not open task archive %s: %s", task_archive, error)

    def open_camera_view() -> tuple[bool, str]:
        if not config.camera_enabled:
            return False, "Enable cameras before opening the live view"
        try:
            url = camera_view.start()
            subprocess.Popen(  # noqa: S603 - fixed local desktop opener
                ["xdg-open", url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            return False, f"Could not open camera view: {error}"
        return True, "Opened stitched live camera view"

    indicator = (
        build_indicator(
            on_mute=lambda value: muted.set() if value else muted.clear(),
            on_quit=stop.set,
            on_reload=request_reload,
            on_clear_tasks=clear_finished_tasks,
            on_cancel_task=cancel_task,
            on_clear_task=clear_task,
            on_open_archive=open_task_archive,
            on_tools=set_tools,
            on_reasoning=set_reasoning,
            on_camera=set_camera,
            on_open_camera_view=open_camera_view,
            tools_enabled=config.tools_enabled,
            reasoning_enabled=config.reasoning_enabled,
            camera_enabled=config.camera_enabled,
            endpoint=public_link,
            tasks=indicator_task_store.list if indicator_task_store is not None else None,
            models=model_manager.views,
            on_model_action=model_action,
            updates=update_manager.view if update_manager is not None else None,
            on_update=update_manager.install if update_manager is not None else None,
            required=indicator_required,
        )
        if not args.no_indicator
        else None
    )
    if indicator is not None:
        indicator_holder["value"] = indicator

    indicator_backend = (
        str(getattr(indicator, "backend", "unknown")) if indicator is not None else "disabled"
    )
    _write_harness_status(
        repo_root,
        # The indicator exists, but deployment readiness waits for the call
        # loop to read a real microphone frame and publish ``listening``.
        state="indicator-ready",
        indicator_backend=indicator_backend,
    )

    def on_state(state: str, detail: str) -> None:
        _write_harness_status(
            repo_root,
            state=state,
            indicator_backend=indicator_backend,
        )
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

        if muted.is_set() or not config.camera_enabled:
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
        camera_view.close()
        if update_manager is not None:
            update_manager.close()
        _write_harness_status(
            repo_root,
            state="stopped",
            indicator_backend=indicator_backend,
        )
    if reload_requested.is_set():
        arguments = list(argv) if argv is not None else sys.argv[1:]
        os.execv(sys.executable, [sys.executable, "-m", "harness", *arguments])
    if model_switch_requested.is_set():
        logger.info("model activation requested; waiting for service managers to relaunch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
