"""Portable foreground TEXT/native task runner. Importing starts nothing.

Use --prepare-only without a token for offline declared-input setup. Live work
requires a host readiness pack. Model completion remains pending independent
review; --evaluate consumes a separate host review bound to the captured bundle.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from portal.desktop import desktop_subprocess_environment
from portal.native_ui import native_task_scope_sha256, normalize_native_task_scope
from runtime.acceptance_native_support import (
    CapabilityBlocked,
    NativeEvidenceObserver,
    command,
    process_start,
)
from runtime.computer_use_speech import _SpeechDeadline
from runtime.long_horizon_amendments import (
    AmendmentRelease,
    prepare_capability,
    probe_calendar_document,
)
from runtime.long_horizon_amendments import supported as amendment_case
from runtime.long_horizon_amendments import verified_event as verified_amendment_event
from runtime.long_horizon_cases import (
    bounded_bytes,
    capture_source_proofs,
    evidence_manifest,
    file_evidence,
    freeze_evidence,
    independent_verdict,
    load_case,
    materialize,
    read_json,
    readiness,
    readiness_requirements,
    safe_path,
    sha,
    source_readiness,
    verified_media_inputs,
)
from runtime.long_horizon_desktop import TaskDesktop, managed_inventory, unrelated_preservation
from runtime.long_horizon_owned_close import (
    OwnedCloseEvent,
    probe_owned_close,
)
from runtime.long_horizon_owned_close import (
    capability as owned_close_capability,
)
from runtime.long_horizon_owned_close import (
    supported as owned_close_case,
)
from runtime.long_horizon_owned_close import (
    verified_event as verified_owned_close_event,
)
from runtime.owned_window_close import request_owned_window_close
from runtime.passive_focus_guard import FocusStopped, FocusUnavailable, sample_owned_focus
from runtime.verify_real_desktop import (
    dispatched_actions,
    physical_pointer_action,
    structured,
)
from runtime.verify_real_desktop import (
    outcomes_from_events as canonical_outcomes_from_events,
)

REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("native_ui", "memory_write", "memory_search")
MAX_EVENT_BYTES = 256 * 1024 * 1024
MAX_STREAM_BYTES = 512 * 1024 * 1024


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    path.chmod(0o600)


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("Use the existing loopback portal endpoint without credentials/query")
    return value.rstrip("/")


def task_payload(model: str, goal: str, *, think: bool = False, recall: bool = False, native_scope=None):
    if type(think) is not bool:
        raise ValueError("think must be a native boolean")
    if recall and native_scope is not None:
        raise ValueError("Memory recall has no native task scope")
    return {"model": model, "stream": True, "think": think,
            "messages": [{"role": "user", "content": goal}],
            "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
            "portal_auto_tools": True, "portal_tool_allowlist": ["memory_search"] if recall else list(ALLOWED),
            **({} if recall else {"portal_execution_profile": "native_task"}),
            **({} if native_scope is None else {"portal_native_task_scope": normalize_native_task_scope(native_scope)}),
            "portal_shell_bridge": False, "portal_background_bridge": False, "portal_camera_bridge": False,
            "response_modalities": ["text"], "speech_mode": "never",
            "options": {"temperature": 0, "num_predict": 8192}}


def verify_registered_native_scope(desktop, scope):
    """Recheck exact setup identities without adding same-process windows."""
    normalized = normalize_native_task_scope(scope)
    try:
        for window in normalized["windows"]:
            verifier = getattr(desktop, "verify_registered_window", None)
            verified = verifier(window) if callable(verifier) else None
            if verified is not None:
                if verified is not True:
                    raise CapabilityBlocked("owned_native_task_scope_identity_changed")
                continue
            pid = str(window["process_id"])
            if (process_start(pid) != window["process_start"]
                    or command(["xdotool", "getwindowpid", window["window_id"]], desktop.environment) != pid
                    or process_start(pid) != window["process_start"]):
                raise CapabilityBlocked("owned_native_task_scope_identity_changed")
    except CapabilityBlocked:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        raise CapabilityBlocked("owned_native_task_scope_identity_unavailable") from None


def registered_native_scope(desktop):
    """Freeze only the windows registered by completed fresh app setup."""
    scope = normalize_native_task_scope({"schema": "robit.omni.native-task-scope.v1", "windows": [
        {"window_id": window.window_id, "process_id": int(window.pid), "process_start": window.process_start}
        for window in tuple(desktop.windows)]})
    verify_registered_native_scope(desktop, scope)
    return scope


def idle(client, base: str) -> bool:
    response = client.get(base + "/api/activity", timeout=5)
    response.raise_for_status()
    activity = response.json()
    return all(activity.get(key) == 0 for key in ("active", "inflight", "queued"))


def drain(client, base: str, seconds: float, *, clock=time.monotonic, sleep=time.sleep) -> bool:
    deadline = clock() + seconds
    while True:
        try:
            if idle(client, base):
                return True
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        if clock() >= deadline:
            return False
        sleep(min(1, max(0, deadline - clock())))


def cancel_owned(client, base: str) -> dict:
    """Cancel this cookie's controller once; never cancel another session."""
    try:
        response = client.post(base + "/api/tools/native_ui/call",
            json={"arguments": {"operation": "cancel"}}, timeout=5)
        return {"attempted": True, "http_status": response.status_code}
    except httpx.HTTPError as exc:
        return {"attempted": True, "error_type": type(exc).__name__}


class Guard:
    def __init__(self, desktop, deadline, root):
        self.desktop, self.deadline, self.root = desktop, deadline, root
        self.stop, self.records, self.positions = threading.Event(), [], []
        self.failure = None
        self.shutdown = {"requested": False, "joined": None}
        self.focus_samples, self.focus_sample_count, self.focus_samples_omitted = [], 0, 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def _remember_focus(self, receipt):
        self.focus_sample_count += 1
        # Keep every exceptional/transition sample and changed stable proof;
        # repeating stable samples add only a count, never false progress.
        if not self.focus_samples or self.focus_samples[-1] != receipt:
            if len(self.focus_samples) < 8192:
                self.focus_samples.append(receipt)
            else:
                self.focus_samples[-1] = receipt
                self.focus_samples_omitted += 1

    def inspect(self):
        scope = {"schema": "robit.omni.native-task-scope.v1", "windows": [
            {"window_id": row.window_id, "process_id": int(row.pid), "process_start": row.process_start}
            for row in self.desktop.windows]}
        try:
            sample = sample_owned_focus(self.desktop.environment, scope,
                stop_requested=self.stop.is_set)
        except (FocusUnavailable, FocusStopped) as error:
            self._remember_focus(error.evidence)
            raise
        self._remember_focus(sample.receipt())
        return sample.identity

    def run(self):
        with (self.root / "pointer-observations.jsonl").open("w") as journal:
            while not self.stop.is_set() and len(self.positions) < 8192:
                try:
                    identity = self.inspect()
                    if self.stop.is_set():
                        return
                    if identity is None:
                        self.stop.wait(.25)
                        continue  # An owned transition grants no pointer sample.
                    if not self.records or self.records[-1] != identity:
                        self.records.append(identity)
                    result = subprocess.run(["xdotool", "getmouselocation", "--shell"],
                        env=self.desktop.environment, capture_output=True, text=True, check=False, timeout=1)  # noqa: S603
                    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                    if result.returncode == 0 and all(values.get(key, "").lstrip("-").isdecimal() for key in ("X", "Y")):
                        record = {"at_unix": time.time(), "x": int(values["X"]), "y": int(values["Y"]),
                                  "phase": "model_task", "owned_foreground": True}
                        self.positions.append(record)
                        journal.write(json.dumps(record) + "\n")
                        journal.flush()
                except FocusStopped:
                    return  # Normal shutdown is not a foreign-focus failure.
                except Exception as exc:
                    self.failure = {**safe_error(exc, stage="owned_foreground_guard"),
                                    "reason": "owned_foreground_read_or_identity_unavailable"}
                    self.deadline._expire()
                    return
                self.stop.wait(.25)

    def close(self):
        self.shutdown["requested"] = True
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
        self.shutdown["joined"] = not self.thread.is_alive()
        if not self.shutdown["joined"]:
            raise RuntimeError("owned_foreground_guard_shutdown_unconfirmed")


def guard_evidence(guard):
    """Copy passive state after shutdown on every terminal path, including failure."""
    return copy.deepcopy({"failure": guard.failure, "records": guard.records,
        "focus_samples": getattr(guard, "focus_samples", []),
        "focus_sample_count": getattr(guard, "focus_sample_count", None),
        "focus_samples_omitted": getattr(guard, "focus_samples_omitted", None),
        "shutdown": getattr(guard, "shutdown", None)})


def stream_turn(client, base, payload, root, seconds, *, desktop=None, observer=None, milestone=None):
    events, final, error = [], None, None
    deadline = _SpeechDeadline(seconds)  # Reuse bounded network interruption only; no speech API is called.
    guard = Guard(desktop, deadline, root) if desktop else None
    started = time.monotonic()
    try:
        if guard:
            guard.inspect()
        with deadline, client.stream("POST", base + "/api/chat/stream", json=payload,
                timeout=httpx.Timeout(seconds, connect=10), extensions={"trace": deadline.trace}) as response:
            deadline.bind(response)
            response.raise_for_status()
            if guard:
                guard.thread.start()
            pending, total = b"", 0
            with (root / "events.ndjson").open("wb") as journal:
                for chunk in response.iter_bytes():
                    deadline.check()
                    if total + len(chunk) > MAX_STREAM_BYTES:
                        raise ValueError("Stream evidence exceeded its bound")
                    journal.write(chunk)
                    journal.flush()
                    total += len(chunk)
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if len(line) > MAX_EVENT_BYTES:
                            raise ValueError("Stream event exceeded its bound")
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        events.append(event)
                        with (root / "receive-timing.jsonl").open("a") as timing:
                            timing.write(json.dumps({"at_unix": time.time(), "elapsed_s": time.monotonic() - started,
                                "event_index": len(events) - 1, "line_sha256": sha(line), "type": event.get("type"),
                                "phase": event.get("phase"), "round": event.get("round")}) + "\n")
                        if event.get("type") == "error":
                            raise RuntimeError("portal_stream_error")
                        if event.get("type") == "tool" and event.get("phase") == "complete" and milestone:
                            milestone(event, observer.fresh())
                        if event.get("type") == "final":
                            final = event.get("response")
                    if len(pending) > MAX_EVENT_BYTES:
                        raise ValueError("Stream event exceeded its bound")
            if pending.strip():
                raise ValueError("Unterminated stream event")
    except (Exception, KeyboardInterrupt) as exc:
        error = {**safe_error(exc, stage="text_request"), "reason": "text_request_or_owned_evidence_incomplete"}
    return {"events": events, "final": final, "error": error,
            "elapsed_s": time.monotonic() - started}, guard


def native_outcomes_from_events(events):
    """Chronological typed boundary views; raw dispatch records stay unchanged."""
    outcomes, seen = [], set()
    for event in events:
        if event.get("type") != "tool" or event.get("phase") != "complete":
            continue
        for call in event.get("tools", []):
            if call.get("name") not in {"native_ui", "computer_use"}:
                continue
            result = structured(call.get("result"))
            values = canonical_outcomes_from_events([{"tools": [call]}])
            boundary, record = result.get("ui_boundary_receipt"), result.get("dispatch_record")
            if (isinstance(boundary, dict) and boundary.get("schema") == "robit.omni.ui-boundary.v1"
                    and boundary.get("verified") is False and boundary.get("task_progress") is False
                    and type(boundary.get("new_effect_count")) is int and boundary["new_effect_count"] == 0
                    and isinstance(result.get("effect_receipt"), dict) and isinstance(record, dict)):
                action = record.get("action")
                if (isinstance(action, dict) and action.get("verb") in {"invoke", "click", "double_click", "right_click"}
                        and all(isinstance(action.get(key), str) and action[key] for key in ("id", "node_id", "generation"))
                        and action.get("args") == {}):
                    view = copy.deepcopy(record)
                    view["ui_boundary_receipt"] = copy.deepcopy(boundary)
                    view["effect_receipt"] = copy.deepcopy(result.get("effect_receipt", {}))
                    values.append(view)
            for item in values:
                key = item.get("action", {}).get("id")
                if isinstance(key, str) and key and key not in seen:
                    seen.add(key)
                    outcomes.append(item)
    return outcomes


def _new_dispatch(item):
    if item.get("effect_receipt", {}).get("effect_state") == "replayed":
        return False
    boundary = item.get("ui_boundary_receipt")
    return (boundary.get("cached") is False and boundary.get("input_attempted") is True
            and boundary.get("adapter_called") is True) if isinstance(boundary, dict) else True


def method_evidence(events, guard) -> dict:
    outcomes = [item for item in dispatched_actions(native_outcomes_from_events(events)) if _new_dispatch(item)]
    unique = {item["action"].get("idempotency_key", item["action"].get("id")): item for item in outcomes}
    values = list(unique.values())
    expected_xids = {item["window_id"] for item in guard.records} if guard else set()
    scoped = bool(values) and all(item.get(side, {}).get("scope", {}).get("window_id") in expected_xids
                                  for item in values for side in ("before", "after"))
    keyboard = any(item["action"].get("verb") == "type_text" and item.get("receipt", {}).get("backend") == "x11_guarded_xtest"
                   for item in values)
    pointer = any(item.get("receipt", {}).get("backend") == "x11_guarded_xtest"
                  and physical_pointer_action(item) for item in values)
    movement = bool(guard and len({(point["x"], point["y"]) for point in guard.positions}) >= 2)
    calls = [tool for event in events if event.get("type") == "tool" and event.get("phase") == "complete"
             for tool in event.get("tools", [])]
    allowed = bool(calls) and all(tool.get("name") in {*ALLOWED, "tool_search"} for tool in calls)
    return {"verified": bool(guard and scoped and keyboard and pointer and movement and allowed and not guard.failure),
            "unique_dispatched_inputs": len(values), "same_owned_scope": scoped,
            "actual_keyboard": keyboard, "actual_pointer": pointer, "actual_pointer_movement": movement,
            "tools_within_scope": allowed, "gui_code_or_macro_audit": "pending_independent_review"}


def turn_summary(result):
    summary = {key: value for key, value in result.items() if key not in {"events", "final"}}
    final = result.get("final")
    if isinstance(final, dict):
        summary["final"] = {**final, "portal": {key: value for key, value in final.get("portal", {}).items()
                                              if key != "safe_tools_executed"}}
    else:
        summary["final"] = final
    summary["raw_final_and_full_cumulative_trace"] = "events.ndjson"
    return summary


def native_outcome_index(events):
    return [{**{key: item[key] for key in ("status", "reason", "action", "receipt", "effect_receipt", "ui_boundary_receipt") if key in item},
             **{side + "_sha256": sha(json.dumps(item.get(side), sort_keys=True, allow_nan=False).encode())
                for side in ("before", "after")}, "full_native_state_source": "events.ndjson"}
            for item in native_outcomes_from_events(events)]


def memory_written(events) -> bool:
    return any(tool.get("name") == "memory_write" and tool.get("ok") is True
        and any(structured(tool.get("result")).get(key) is True for key in ("stored", "updated", "unchanged"))
        for event in events if event.get("type") == "tool" and event.get("phase") == "complete"
        for tool in event.get("tools", []))


def measured_metrics(events, *, elapsed_s, method) -> dict:
    completed = [event for event in events if event.get("type") == "tool" and event.get("phase") == "complete"]
    final = next((event.get("response", {}) for event in reversed(events) if event.get("type") == "final"), {})
    native = final.get("portal", {}).get("native_task", {})
    receipts = {"policy": {}, "point": {}}
    phases, seen_inputs = [], set()
    for event in completed:
        new_inputs = 0
        for tool in event.get("tools", []):
            result = structured(tool.get("result"))
            outcomes = native_outcomes_from_events([{"type": "tool", "phase": "complete", "tools": [tool]}])
            for item in dispatched_actions(outcomes):
                action = item["action"]
                key = action.get("idempotency_key") or action.get("id")
                if key and key not in seen_inputs and _new_dispatch(item):
                    seen_inputs.add(key)
                    new_inputs += 1
            for family in receipts:
                candidates = [result.get(family + "_receipt"), *result.get(family + "_receipts", [])]
                candidates.extend(item.get(family + "_receipt") or item.get("receipt", {}).get(family + "_receipt")
                                  for item in outcomes)
                for receipt in candidates:
                    if not isinstance(receipt, dict) or not receipt:
                        continue
                    digest = sha(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                    receipts[family][digest] = {"digest": digest, **{key: receipt[key] for key in
                        ("schema", "id", "status", "model", "revision", "model_confirmed", "reply_sha256", "frame_sha256", "candidate_id")
                        if key in receipt}}
        phases.append({"phase": event.get("round"), "new_unique_dispatched_inputs": new_inputs})
    delegation = {family: {"unique_receipts": len(values), "receipts": list(values.values())}
                  for family, values in receipts.items()}
    return {"request_wall_s": elapsed_s, "completed_tool_phases": len(completed),
            "native_phase_count": native.get("phases"), "native_terminal_status": native.get("status"),
            "unique_dispatched_inputs": method["unique_dispatched_inputs"],
            "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
            "token_metric_status": "not_exposed_by_native_task_public_stream",
            "resource_metric_status": "worker_identity_only; no_memory_or_residency_claim",
            "request_options": {"temperature": 0, "num_predict": 8192}, "server_limits_overridden": False,
            "delegation": delegation, "inner_inputs_per_phase": phases,
            "semantic_unique_count": None, "upstream_planner_request_count": None,
            "receipt_count_is_not_configured_availability_or_inferred_model_calls": True}


def source_state() -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    names = ["portal/app.py", "portal/ui_task.py", "portal/computer_use.py", "portal/native_ui.py",
             "src/qwen_omni_adapters/computer_use.py", "runtime/adapter_server.py",
             "runtime/verify_long_horizon_native.py", "runtime/long_horizon_cases.py",
             "runtime/long_horizon_desktop.py", "tests/live/native_long_horizon/manifest.json"]
    return {"revision": revision, "hashes": {name: sha((REPO / name).read_bytes()) for name in names}}


def worker_snapshot() -> dict:
    """Read standard daemon identities only; missing proof stays unknown."""
    path = REPO / "runtime-data/state/daemon-status.json"
    if not path.is_file():
        return {"available": False, "reason": "standard_daemon_identity_file_absent"}
    try:
        state = read_json(path)
        roles = {"supervisor": state["pid"]}
        roles.update({child["name"]: child["pid"] for child in state["children"]
                      if child["name"] in {"comprehension", "pointing", "tts", "adapter", "portal"}})
        return {"available": True, "workers": {role: {"pid": pid, "start_ticks":
            int(Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[19])} for role, pid in roles.items()}}
    except (OSError, ValueError, KeyError, IndexError):
        return {"available": False, "reason": "standard_worker_identity_unavailable"}


def cleanup_owned(desktop, *, request_started: bool, settled: bool) -> dict:
    if request_started and not settled:
        return {"owned_windows_closed": False, "status": "cleanup_deferred", "owned_windows_retained": True}
    receipts = desktop.close()
    return {"cleanup": receipts, "owned_windows_closed": all(item.get("closed") is True for item in receipts)}


def host_editor_condition(case, editor, include, environment):
    """Explicit host setup only; never change a packaged task or MIME default."""
    supplement = include and "gedit" not in case["fixture"]["blank_apps"]
    count = sum(case["fixture"]["blank_app_window_counts"].values()) + int(supplement)
    if count > 16:
        raise CapabilityBlocked("owned_native_setup_window_limit")
    default_id = None
    if include:
        expected = {"gedit": "org.gnome.gedit.desktop",
                    "gnome-text-editor": "org.gnome.TextEditor.desktop"}.get(editor)
        if expected is None:
            raise CapabilityBlocked("unsupported_default_text_editor")
        try:
            actual = command(["xdg-mime", "query", "default", "text/plain"], environment)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            raise CapabilityBlocked("default_text_editor_handler_unavailable") from None
        if actual != expected:
            raise CapabilityBlocked("default_text_editor_handler_mismatch")
        default_id = actual
    return {"include_default_text_editor": include, "selected_editor": editor if include else None,
            "text_plain_default_desktop_id": default_id, "default_handler_verified": include,
            "supplemental_blank_editor_planned": supplement, "planned_owned_window_count": count,
            "setup_completed": False, "condition": "explicit_host_supplement_no_task_or_MIME_changes"}


def prepare_blank_apps(desktop, observer, case, work, root, browser, *, supplement_editor=False):
    """Fresh native proof for every new owned window, before model submission."""
    receipts = []
    plan = [("gedit", 0, True)] if supplement_editor and "gedit" not in case["fixture"]["blank_apps"] else []
    plan.extend((app, number, False) for app in case["fixture"]["blank_apps"]
                for number in range(case["fixture"]["blank_app_window_counts"][app]))
    for app, number, supplement in plan:
        log_name = "default-text-editor.log" if supplement else app + str(number) + ".log"
        window = desktop.open_blank(app, work, browser=browser, log_path=root / log_name)
        state = observer.fresh()
        receipt = {"application": app, "window_id": window.window_id, "process_id": window.pid,
                   "process_start": window.process_start, "native": state,
                   "declared_application": None if supplement else app,
                   "setup_origin": "host_default_text_editor" if supplement else "declared_fixture",
                   "effective_application": getattr(desktop, "editor", "gedit") if app == "gedit" else app,
                   **({"editor_condition": "explicit_editor_choice_no_MIME_or_preference_changes"}
                      if app == "gedit" else {})}
        receipts.append(receipt)
        save(root / "setup-native-observations.json", receipts)
        native = state.get("state", {})
        if (state.get("outside_owned_foreground") or state.get("window_id") != window.window_id
                or native.get("scope", {}).get("window_id") != window.window_id or not native.get("nodes")):
            raise CapabilityBlocked("owned_native_setup_observation_unavailable:" + app)
        if amendment_case(case) and app == "calc":
            # The declared blank Calc is foreground now; never activate or
            # pre-save it to manufacture a calendar capability later.
            receipt["calendar_document_probe"] = probe_calendar_document(
                {"window_id": window.window_id, "process_id": int(window.pid),
                 "process_start": window.process_start}, desktop.environment)
            save(root / "setup-native-observations.json", receipts)
    return receipts


def safe_error(exc, *, stage):
    """Retain typed causes without echoing transport URLs, headers or secrets."""
    result = {"type": type(exc).__name__, "stage": stage}
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.replace("_", "").isalnum() and len(code) <= 100:
        result["code"] = code
    if isinstance(exc, CapabilityBlocked):
        # These fixed setup messages contain only app labels/identity failures.
        result["code"] = str(exc)[:200]
    elif str(exc) in {"existing_portal_not_healthy_idle", "portal_stream_error",
                       "foreground_identity_outside_owned_scope", "Unterminated stream event",
                       "Stream evidence exceeded its bound"}:
        result["code"] = str(exc)
    return result


def finalize_run(report, root, case, *, desktop, observer, guard, client, request_started, settled, started):
    errors = []
    def attempt(stage, fn):
        try:
            return True, fn()
        except Exception as exc:
            errors.append(safe_error(exc, stage=stage))
            return False, None
    stop_ok = True
    for stage, resource in (("guard_shutdown", guard), ("observer_shutdown", observer)):
        if resource:
            ok, _ = attempt(stage, resource.close)
            stop_ok &= ok
    if guard:
        report["ownership_guard"] = guard_evidence(guard)
    if desktop:
        attempt("ownership_journal", lambda: save(root / "window-ownership.json", desktop.launches))
        if stop_ok:
            ok, cleanup = attempt("owned_window_cleanup", lambda: cleanup_owned(desktop,
                request_started=request_started, settled=settled))
            if ok:
                report.update(cleanup)
            else:
                report.update(owned_windows_closed=False, owned_windows_retained=True)
        else:
            report.update(owned_windows_closed=False, owned_windows_retained=True)
        ok, after = attempt("preservation_snapshot", lambda: managed_inventory(desktop.environment))
        report["unrelated_windows_after"] = after if ok else {"available": False}
        report["unrelated_window_preservation"] = unrelated_preservation(report.get("unrelated_windows_before", {}),
                                                                      report["unrelated_windows_after"])
    if client:
        attempt("client_shutdown", client.close)
    ok, after = attempt("source_snapshot", source_state)
    report["source_after"] = after if ok else {"available": False}
    ok, after = attempt("worker_snapshot", worker_snapshot)
    report["workers_after"] = after if ok else {"available": False}
    report["workers_preserved"] = (report.get("workers_before") == report["workers_after"]
                                   if report.get("workers_before", {}).get("available") else None)
    report["source_unchanged"] = report.get("source_before") == report["source_after"]
    report["elapsed_s"] = time.monotonic() - started
    if errors:
        report["teardown_errors"] = errors
        if report.get("status") != "cleanup_deferred":
            report["status"] = "failed_cleanup_or_evidence" if request_started else "blocked_precondition"
    def journals():
        save(root / "summary.json", report)
        save(root / "after-test-note.json", {key: report.get(key) for key in
            ("case_id", "model", "think", "status", "passed", "portal_turn_attempts", "readiness_blockers",
             "authoritative_final_received", "method", "downstream_idle", "owned_windows_closed", "error",
             "teardown_errors", "elapsed_s", "source_unchanged", "editor", "host_fixture_condition",
             "native_task_scope")})
    journals()
    try:
        freeze_evidence(root, case)
    except (OSError, ValueError, TypeError) as exc:
        report["evidence_freeze_error"] = safe_error(exc, stage="immutable_evidence_manifest")
        report["status"] = "failed_evidence_freeze"
        journals()


def evaluate(args) -> int:
    root = args.output.resolve()
    if not root.is_relative_to(REPO / "runtime-data"):
        raise ValueError("Evidence must remain in repository runtime-data")
    report = read_json(root / "summary.json")
    case = load_case(report["case_id"])
    current = file_evidence(root / "Workbench", report["initial_input_hashes"], case)
    if args.review_file and args.review_file.resolve().is_relative_to(root):
        raise ValueError("Independent review must remain outside the immutable run bundle")
    review = json.loads(bounded_bytes(args.review_file)) if args.review_file else None
    captured = read_json(root / "evidence-manifest.json")
    actual = evidence_manifest(root, case["case_id"], case["task_signature_sha256"])
    bundle = captured["bundle_sha256"]
    immutable = captured == actual and report.get("task_signature_sha256") == case["task_signature_sha256"]
    result = independent_verdict(case, current, method_verified=report.get("method", {}).get("verified") is True
        and report.get("downstream_idle") is True and report.get("owned_windows_closed") is True
        and report.get("source_unchanged") is True
        and verified_owned_close_event(case, root, report)
        and verified_amendment_event(case, root, report)
        and report.get("unrelated_window_preservation", {}).get("verified") is True
        and current == report["final_file_evidence"] and immutable, review=review, bundle_sha256=bundle)
    save(root / "independent-verdict.json", {**result, "bundle_sha256": bundle, "immutable_evidence_verified": immutable})
    return 0 if result["passed"] else 4


def run(args) -> int:
    if args.evaluate:
        return evaluate(args)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800 or not 0 <= args.drain_timeout <= 600:
        raise ValueError("Expected bounded task/drain deadlines")
    base = endpoint(args.endpoint)
    case = load_case(args.case)
    root = args.output.absolute()
    if not root.resolve().is_relative_to(REPO / "runtime-data") or root.exists() or root.is_symlink():
        raise ValueError("Use a new owned output under this checkout's runtime-data")
    safe_path(REPO / "runtime-data", root.relative_to(REPO / "runtime-data").as_posix())
    os.umask(0o077)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    work = root / "Workbench"
    report = {"schema": "robit.omni.long-native-run.v1", "case_id": case["case_id"], "model": args.model,
              "task_signature_sha256": case["task_signature_sha256"],
              "think": args.think, "source": "text_goal", "physical_microphone": False, "tts_or_asr_used": False,
              "editor": {"configured_application": args.editor,
                         "setup_slot_present": "gedit" in case["fixture"]["blank_apps"],
                         "declared_application": "gedit" if "gedit" in case["fixture"]["blank_apps"] else None,
                         "effective_application": args.editor if "gedit" in case["fixture"]["blank_apps"] else None,
                         "condition": "explicit_editor_choice_no_MIME_or_preference_changes"},
              "host_fixture_condition": {"include_default_text_editor": args.include_default_text_editor,
                                         "default_handler_verified": False, "setup_completed": False},
              "passed": False, "portal_turn_attempts": 0, "upstream_model_request_count": None,
              "authoritative_final_received": False,
              "owned_windows_closed": True, "status": "blocked_precondition"}
    report.update(task_deadline_seconds=args.timeout, drain_deadline_seconds=args.drain_timeout,
                  native_input_ceiling_enforced=False, whole_task_verification="independent_review_required")
    desktop = observer = client = guard = None
    request_started, settled = False, False
    started = time.monotonic()
    try:
        report["source_before"] = source_state()
        report["workers_before"] = worker_snapshot()
        fixture_options = {"owned_media_cache": args.owned_media_cache,
                           "owned_media_manifest_sha256": args.owned_media_manifest_sha256}
        initial = materialize(case, work, **fixture_options)
        report["initial_input_hashes"] = initial
        if case["fixture"].get("additional_asset_requirements") and args.owned_media_cache is not None:
            report["owned_media_capability"] = verified_media_inputs(case, work,
                args.owned_media_cache, args.owned_media_manifest_sha256)
            save(root / "owned-media-capability.json", report["owned_media_capability"])
        save(root / "readiness-requirements.json", {"case_id": case["case_id"],
            "task_signature_sha256": case["task_signature_sha256"], "source_revision": report["source_before"]["revision"],
            "required_host_artifact_ids": sorted(readiness_requirements(case)),
            "app_readiness": "fresh_owned_observation_of_each_blank_window_before_submission",
            "rubric_readiness": "predeclared_criteria_complete; grading_remains_post_run",
            "materialized_inputs_only": True, "solver_must_not_receive_this_file": True})
        proof = read_json(args.readiness_file) if args.readiness_file else None
        # Source preflight permits blank owned setup only. Final readiness for
        # controlled cases still requires fresh native/document/protocol evidence
        # before the first solver/model submission.
        setup_capability_required = owned_close_case(case) or amendment_case(case)
        preflight = {"revision": report["source_before"]["revision"], "work": work,
                     "evidence_root": args.readiness_file.parent.resolve() if args.readiness_file else root}
        blockers = (source_readiness(case, proof, **preflight) if setup_capability_required and not args.prepare_only
                    else readiness(case, proof, **preflight, **fixture_options))
        report["readiness_blockers"] = blockers
        if args.prepare_only or blockers:
            report["status"] = "prepared_inputs" if args.prepare_only else "blocked_precondition"
            return 2 if blockers else 0
        if not setup_capability_required:
            report["captured_source_provenance"] = capture_source_proofs(case, args.readiness_file,
                root / "host-source-evidence", revision=report["source_before"]["revision"], work=work,
                **fixture_options)
        if not args.model or not args.token_file:
            raise ValueError("Live evaluation requires exact model and token file")
        if args.token_file.is_symlink() or not 0 < args.token_file.stat().st_size <= 4096:
            raise ValueError("Invalid access-token file")
        environment = desktop_subprocess_environment()
        report["unrelated_windows_before"] = managed_inventory(environment)
        report["host_fixture_condition"] = host_editor_condition(
            case, args.editor, args.include_default_text_editor, environment)
        desktop = TaskDesktop(environment, editor=args.editor)
        client = httpx.Client(headers={"Authorization": "Bearer " + args.token_file.read_text().strip()},
            cookies={"omni_portal_session": secrets.token_hex(32)}, trust_env=False, timeout=10)
        health = client.get(base + "/healthz")
        if health.status_code != 200 or health.json().get("ok") is not True or not idle(client, base):
            raise RuntimeError("existing_portal_not_healthy_idle")
        status = client.get(base + "/api/status")
        status.raise_for_status()
        deployed = status.json()
        report["deployed_model"] = deployed.get("model")
        report["loaded_python_source_adoption"] = "not_proven_by_checkout_hash"
        if deployed.get("model") != args.model:
            raise CapabilityBlocked("exact_deployed_model_mismatch")
        observer = NativeEvidenceObserver(root, desktop)
        setup = prepare_blank_apps(desktop, observer, case, work, root, args.browser,
            supplement_editor=report["host_fixture_condition"]["supplemental_blank_editor_planned"])
        report["host_fixture_condition"].update(setup_completed=True, supplemental_window_count=sum(
            item["setup_origin"] == "host_default_text_editor" for item in setup))
        native_scope = registered_native_scope(desktop)
        close_event = None
        amendment_event = None
        if owned_close_case(case):
            verify_registered_native_scope(desktop, native_scope)
            close_capability_path = root / "owned-close-capability.json"
            save(close_capability_path, owned_close_capability(case, root / "setup-native-observations.json", work,
                revision=report["source_before"]["revision"],
                probe=lambda target: probe_owned_close(target, environment)))
            fixture_options["owned_close_capability"] = close_capability_path
        elif amendment_case(case):
            verify_registered_native_scope(desktop, native_scope)
            calc = [row for row in setup if row["application"] == "calc"]
            if len(calc) != 1:
                raise CapabilityBlocked("exact_registered_calendar_window_unavailable")
            row = calc[0]
            amendment = prepare_capability(case, work=work, revision=report["source_before"]["revision"],
                initial_hashes=initial, calc_window={"window_id": row["window_id"],
                    "process_id": int(row["process_id"]), "process_start": row["process_start"]},
                document_probe=row["calendar_document_probe"])
            fixture_options["amendment_capability"] = root / "amendment-capability.json"
            save(fixture_options["amendment_capability"], amendment)
        if setup_capability_required:
            blockers = readiness(case, proof, **preflight, **fixture_options)
            report["readiness_blockers"] = blockers
            if blockers:
                raise CapabilityBlocked("controlled_fixture_capability_unverified")
            report["captured_source_provenance"] = capture_source_proofs(case, args.readiness_file,
                root / "host-source-evidence", revision=report["source_before"]["revision"], work=work,
                **fixture_options)
        if owned_close_case(case):
            close_event = OwnedCloseEvent(case, close_capability_path, work, initial,
                revision=report["source_before"]["revision"], journal=root / "owned-close-events.jsonl",
                probe=lambda target: probe_owned_close(target, environment),
                request_close=lambda target: request_owned_window_close(target["window_id"],
                    target["process_id"], target["process_start"], environment),
                window_status=desktop.controlled_window_observation,
                reserve_close=desktop.reserve_controlled_close)
            report["controlled_perturbation"] = close_event.snapshot()
        elif amendment_case(case):
            amendment_event = AmendmentRelease(case, capability=amendment, work=work,
                revision=report["source_before"]["revision"], initial_hashes=initial,
                event_root=root / "amendment-events", environment=environment)
            report["controlled_perturbation"] = amendment_event.snapshot()
        report["native_task_scope"] = {"schema": native_scope["schema"],
            "sha256": native_task_scope_sha256(native_scope), "base_window_count": len(native_scope["windows"]),
            "authority": "fresh_registered_setup_windows", "model_visible": False}
        before = setup[-1]["native"]
        save(root / "initial-native.json", before)
        observer.start()
        observer.task_started.set()
        goal = case["goal_text"].replace("WORKBENCH", str(work))
        (root / "goal.txt").write_text(goal + "\n")
        payload = task_payload(args.model, goal, think=args.think, native_scope=native_scope)
        save(root / "request-payload.json", payload)
        def milestone(event, state):
            record = {"at_unix": time.time(), "round": event.get("round"), "native": state,
                      "files": file_evidence(work, initial, case)}
            with (root / "milestones.jsonl").open("a") as journal:
                journal.write(json.dumps(record) + "\n")
            if close_event is not None:
                close_event.observe(state)
                report["controlled_perturbation"] = close_event.snapshot()
            elif amendment_event is not None:
                amendment_event.observe(state)
                report["controlled_perturbation"] = amendment_event.snapshot()
                release = report["controlled_perturbation"].get("release")
                report["released_input_hashes"] = {release["path"]: release["sha256"]} if release else {}
        verify_registered_native_scope(desktop, payload["portal_native_task_scope"])
        request_started, report["portal_turn_attempts"] = True, 1
        result, guard = stream_turn(client, base, payload,
            root, args.timeout, desktop=desktop, observer=observer, milestone=milestone)
        report["task"] = turn_summary(result)
        report["authoritative_final_received"] = isinstance(result["final"], dict)
        if result["error"] or not result["final"] or guard.failure:
            report["cancellation"] = cancel_owned(client, base)
            report["cancellation_authority"] = "store_cancel_requested; native_goal_stop_not_proven; require_actual_drain"
        settled = drain(client, base, args.drain_timeout)
        report["downstream_idle"] = settled
        if not settled:
            report["status"] = "cleanup_deferred"
            return 3
        guard.close()
        report["method"] = method_evidence(result["events"], guard)
        save(root / "metrics.json", measured_metrics(result["events"], elapsed_s=result.get("elapsed_s"),
                                                     method=report["method"]))
        report["ownership_guard"] = guard_evidence(guard)
        save(root / "native-outcomes-index.json", native_outcome_index(result["events"]))
        save(root / "final-native.json", observer.fresh())
        report["final_file_evidence"] = file_evidence(work, initial, case)
        report["independent_result"] = independent_verdict(case, report["final_file_evidence"],
            method_verified=report["method"]["verified"] and verified_owned_close_event(case, root, report)
            and verified_amendment_event(case, root, report))
        if case["memory_followup"]["required"] and result["final"] and not result["error"] and memory_written(result["events"]):
            recall_root = root / "recall"
            recall_root.mkdir()
            settled = False
            report["portal_turn_attempts"] += 1
            recall, _ = stream_turn(client, base, task_payload(args.model, case["memory_followup"]["goal"],
                think=args.think, recall=True), recall_root, min(180, args.timeout))
            report["recall"] = turn_summary(recall)
            settled = drain(client, base, args.drain_timeout)
            report["downstream_idle"] = settled
            if not settled:
                report["status"] = "cleanup_deferred"
                return 3
        report["status"] = "failed_model_request" if result["error"] or not result["final"] else report["independent_result"]["status"]
        return 1 if report["status"].startswith("failed") else 4
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = safe_error(exc, stage="task" if request_started else "prerequisite")
        if request_started:
            report["status"] = "failed_model_request"
            if "cancellation" not in report:
                report["cancellation"] = cancel_owned(client, base)
                report["cancellation_authority"] = "store_cancel_requested; native_goal_stop_not_proven; require_actual_drain"
            settled = drain(client, base, args.drain_timeout)
        else:
            report["readiness_blockers"] = ["actual_setup_or_fixture_precondition_failed",
                                           report["error"].get("code", type(exc).__name__)]
        return 3 if request_started and not settled else 1 if request_started else 2
    finally:
        finalize_run(report, root, case, desktop=desktop, observer=observer, guard=guard, client=client,
            request_started=request_started, settled=settled, started=started)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", default="LHN-001")
    result.add_argument("--model")
    result.add_argument("--endpoint", default="http://127.0.0.1:8920")
    result.add_argument("--token-file", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--readiness-file", type=Path)
    result.add_argument("--owned-media-cache", type=Path,
                        help="Host-owned frozen declared MP4 input cache, outside Workbench")
    result.add_argument("--owned-media-manifest-sha256",
                        help="Independent external pin for the exact owned-media cache manifest")
    result.add_argument("--timeout", type=float, default=1800)
    result.add_argument("--drain-timeout", type=float, default=300)
    result.add_argument("--think", action="store_true")
    result.add_argument("--browser", choices=["firefox", "chromium", "chromium-snap", "flatpak-firefox", "flatpak-chromium"], default="firefox")
    result.add_argument("--editor", choices=["gedit", "gnome-text-editor"], default="gedit")
    result.add_argument("--include-default-text-editor", action="store_true",
                        help="Own one blank selected text/plain handler when the fixture has no editor slot")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--evaluate", action="store_true")
    result.add_argument("--review-file", type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
