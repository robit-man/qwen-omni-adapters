"""Read-only acceptance focus sampling, separate from strict native dispatch.

One positively owned focus transition can be observed without cancelling the
request. It grants no input, pointer, method or task-progress authority. Every
observed endpoint and its direct registered ancestry must be positively bound;
missing identities and repeated movement remain failures, never retry hints.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal import native_ui

SCHEMA = "robit.omni.passive-owned-focus.v1"


@dataclass(frozen=True)
class OwnerBinding:
    window_id: str
    process_id: int
    process_start: str
    uid: int
    namespace_binding_sha256: str | None = None

    def __post_init__(self):
        if (not isinstance(self.window_id, str) or re.fullmatch(r"[1-9][0-9]{0,9}", self.window_id) is None
                or int(self.window_id) >= 2**32 or type(self.process_id) is not int
                or not 0 < self.process_id < 2**31 or not isinstance(self.process_start, str)
                or re.fullmatch(r"[1-9][0-9]{0,19}", self.process_start) is None
                or type(self.uid) is not int or not 0 <= self.uid < 2**32
                or self.namespace_binding_sha256 is not None and (not isinstance(self.namespace_binding_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", self.namespace_binding_sha256) is None)):
            raise ValueError("Expected a complete typed host owner binding")

    @property
    def identity(self) -> dict:
        return {"window_id": self.window_id, "process_id": self.process_id}


@dataclass(frozen=True)
class OwnershipProof:
    binding: OwnerBinding
    kind: str
    registered_base: OwnerBinding
    transient_parent_window_id: str | None = None


@dataclass(frozen=True)
class FocusSample:
    status: str
    before: OwnershipProof
    after: OwnershipProof
    stable_after: OwnershipProof | None = None

    @property
    def identity(self) -> dict | None:
        return self.after.binding.identity if self.status == "stable" else None

    def receipt(self) -> dict:
        return {"schema": SCHEMA, **asdict(self), "input_authority": False,
                "pointer_credit": False, "method_credit": False, "task_progress": False}


class FocusStopped(RuntimeError):
    """The owner stopped passive sampling; no positive focus evidence is returned."""


class FocusUnavailable(native_ui.NativeUiUnavailable):
    def __init__(self, code: str, evidence: dict):
        super().__init__(code)
        self.evidence = evidence


class X11FocusReader:
    """Fixed read-only queries and existing positive native owner resolution."""

    def __init__(self, environment: Mapping[str, str], *, strict_foreground=None, stop_requested=lambda: False):
        self.environment = dict(environment)
        self.stop_requested = stop_requested
        if strict_foreground is not None and strict_foreground is not native_ui._x11_foreground:
            raise ValueError("Passive helper requires the unchanged strict native reader")
        if self.environment.get("XDG_SESSION_TYPE") == "wayland" or self.environment.get("WAYLAND_DISPLAY"):
            raise native_ui.NativeUiUnavailable("native_foreground_wayland_unsupported")
        if not self.environment.get("DISPLAY"):
            raise native_ui.NativeUiUnavailable("native_foreground_display_unavailable")

    def _query(self, arguments: list[str], deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise native_ui.NativeUiUnavailable("native_observation_budget_exceeded")
        result = subprocess.run(arguments, capture_output=True, text=True,
            timeout=min(1, remaining), check=False, env={**self.environment, "LC_ALL": "C"})  # noqa: S603
        if result.returncode or len(result.stdout) > 4096:
            raise native_ui.NativeUiUnavailable("native_foreground_unmapped")
        return result.stdout.strip()

    def active(self, deadline: float) -> str:
        value = self._query(["xdotool", "getactivewindow"], deadline)
        if re.fullmatch(r"[1-9][0-9]{0,9}", value) is None or int(value) >= 2**32:
            raise native_ui.NativeUiUnavailable("native_foreground_unmapped")
        return value

    @staticmethod
    def _binding(value: dict) -> OwnerBinding:
        pid, xid = value.get("process_id"), value.get("window_id")
        if (type(pid) is not int or not 0 < pid < 2**31 or not isinstance(xid, str)
                or re.fullmatch(r"[1-9][0-9]{0,9}", xid) is None or int(xid) >= 2**32):
            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
        host = native_ui._host_process_identity(pid)
        start, uid = host.get("start"), host.get("uid")
        if (type(host.get("pid")) is not int or host["pid"] != pid or not isinstance(start, str)
                or re.fullmatch(r"[1-9][0-9]{0,19}", start) is None
                or type(uid) is not int or uid != os.getuid()):
            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
        namespace = value.get("_namespace_owner_binding")
        digest = hashlib.sha256(json.dumps(namespace, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest() if namespace is not None else None
        return OwnerBinding(xid, pid, start, uid, digest)

    def _identity(self, operation: str, xid: str | None, deadline: float) -> OwnerBinding:
        # Existing native fallback queries have fixed nested timeouts. A
        # dedicated read-only child bounds their TOTAL runtime without changing
        # dispatch code, dropping namespace proof or using a smaller proof cap.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise native_ui.NativeUiUnavailable("native_observation_budget_exceeded")
        request = {"operation": operation, "window_id": xid, "deadline": deadline}
        process = subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--read-identity"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**self.environment, "CUDA_VISIBLE_DEVICES": ""}, start_new_session=True)  # noqa: S603
        stdout, stderr = bytearray(), bytearray()
        try:
            process.stdin.write(json.dumps(request).encode() + b"\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, stdout)
                selector.register(process.stderr, selectors.EVENT_READ, stderr)
                while b"\n" not in stdout:
                    if self.stop_requested():
                        raise FocusStopped()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired("private passive identity helper", 0)
                    for key, _ in selector.select(min(.05, remaining)):
                        chunk = os.read(key.fileobj.fileno(), 4097)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            if key.data is stdout:
                                raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
                        key.data.extend(chunk)
                        if len(key.data) > 4096:
                            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
            if self.stop_requested():
                raise FocusStopped()
            if time.monotonic() >= deadline:
                raise native_ui.NativeUiUnavailable("native_observation_budget_exceeded")
        finally:
            # The one-request helper keeps its leader alive after replying until
            # stdin closes. We have not reaped it: this exact new process group
            # remains ours. Always clean descendants, even after an inner
            # native helper timed out but the leader returned an error reply.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=.2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=.1)
        if len(stdout) > 4096 or len(stderr) > 4096:
            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
        reply = json.loads(stdout)
        if (not isinstance(reply, dict) or reply.get("ok") is not True):
            code = reply.get("code") if isinstance(reply, dict) else None
            if not isinstance(code, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) is None:
                code = "guard_focus_owner_unavailable"
            raise native_ui.NativeUiUnavailable(code)
        if set(reply) != {"ok", "binding"} or not isinstance(reply["binding"], dict):
            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
        binding = OwnerBinding(**reply["binding"])
        # Rebind host lifetime independently after the helper returns. Never
        # accept a valid old child response after PID generation or UID churn.
        host = native_ui._host_process_identity(binding.process_id)
        if (type(host.get("pid")) is not int or host["pid"] != binding.process_id
                or host.get("start") != binding.process_start or type(host.get("uid")) is not int
                or host["uid"] != binding.uid or binding.uid != os.getuid()
                or time.monotonic() >= deadline):
            raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
        return binding

    def owner(self, xid: str, deadline: float) -> OwnerBinding:
        return self._identity("owner", xid, deadline)

    def stable(self, deadline: float) -> OwnerBinding:
        return self._identity("stable", None, deadline)

    def parent(self, xid: str, deadline: float) -> str:
        value = self._query(["xprop", "-id", xid, "WM_TRANSIENT_FOR"], deadline)
        match = re.fullmatch(r"WM_TRANSIENT_FOR\(WINDOW\): window id # (0x[0-9a-fA-F]+)", value)
        if match is None or not 0 < int(match[1], 16) < 2**32:
            raise native_ui.NativeUiUnavailable("guard_focus_transient_unproven")
        return str(int(match[1], 16))


def sample_owned_focus(environment: Mapping[str, str], registered: dict, *,
                       reader=None, strict_foreground=None, stop_requested: Callable[[], bool] = lambda: False,
                       budget_s: float = 2.5, clock: Callable[[], float] = time.monotonic) -> FocusSample:
    """Observe at most one transition; never discard a foreign or unknown endpoint."""
    if not math.isfinite(budget_s) or not 0 < budget_s <= 2.5:
        raise ValueError("Passive focus budget must fit within guard shutdown")
    deadline = clock() + budget_s
    evidence: dict[str, Any] = {"schema": SCHEMA, "status": "unavailable", "active_ids": [],
        "owner_bindings": [], "ownership_proofs": [], "input_authority": False,
        "pointer_credit": False, "method_credit": False, "task_progress": False}

    def check():
        if stop_requested():
            raise FocusStopped()
        if clock() >= deadline:
            raise native_ui.NativeUiUnavailable("native_observation_budget_exceeded")

    try:
        scope = native_ui.normalize_native_task_scope(registered)
        backend = reader or X11FocusReader(environment, strict_foreground=strict_foreground, stop_requested=stop_requested)

        def read(method, *arguments):
            check()
            try:
                value = method(*arguments, deadline)
            except Exception as error:
                if stop_requested():
                    raise FocusStopped() from error
                raise
            check()
            return value

        def active():
            xid = read(backend.active)
            evidence["active_ids"].append(xid)
            return xid

        def owner(xid):
            binding = read(backend.owner, xid)
            if (not isinstance(binding, OwnerBinding) or binding.window_id != xid
                    or type(binding.process_id) is not int or not 0 < binding.process_id < 2**31
                    or not isinstance(binding.process_start, str)
                    or re.fullmatch(r"[1-9][0-9]{0,19}", binding.process_start) is None
                    or type(binding.uid) is not int or binding.uid != os.getuid()):
                raise native_ui.NativeUiUnavailable("guard_focus_owner_unavailable")
            evidence["owner_bindings"].append(asdict(binding))
            return binding

        def ownership(binding):
            bases = [row for row in scope["windows"] if row["process_id"] == binding.process_id
                     and row["process_start"] == binding.process_start]
            if not bases:
                raise native_ui.NativeUiUnavailable("guard_focus_outside_owned_scope")
            exact = [row for row in bases if row["window_id"] == binding.window_id]
            if exact:
                proof = OwnershipProof(binding, "registered_base", binding)
            else:
                parent = read(backend.parent, binding.window_id)
                selected = [row for row in bases if row["window_id"] == parent]
                if len(selected) != 1:
                    raise native_ui.NativeUiUnavailable("guard_focus_transient_unproven")
                base = owner(parent)
                expected = selected[0]
                if any(getattr(base, key) != expected[key] for key in expected) or base.uid != binding.uid:
                    raise native_ui.NativeUiUnavailable("guard_focus_transient_unproven")
                # This is the same exact-base/direct-parent policy as
                # TaskDesktop. Its boolean helper has a fixed3-second command
                # timeout, so this passive proof uses deadline-bound reads.
                if read(backend.parent, binding.window_id) != parent or owner(binding.window_id) != binding:
                    raise native_ui.NativeUiUnavailable("guard_focus_transient_changed")
                proof = OwnershipProof(binding, "direct_transient", base, parent)
            evidence["ownership_proofs"].append(asdict(proof))
            return proof

        before = ownership(owner(active()))
        next_id = active()
        if next_id == before.binding.window_id:
            current = ownership(owner(next_id))
            if current != before:
                raise native_ui.NativeUiUnavailable("guard_focus_owner_changed")
            next_id = active()
            if next_id == before.binding.window_id:
                return FocusSample("stable", before, current)
        after = ownership(owner(next_id))
        stable = read(backend.stable)
        if not isinstance(stable, OwnerBinding) or stable != after.binding:
            raise native_ui.NativeUiUnavailable("guard_focus_repeated_transition")
        stable_proof = ownership(stable)
        if stable_proof != after or active() != after.binding.window_id:
            raise native_ui.NativeUiUnavailable("guard_focus_repeated_transition")
        return FocusSample("owned_transition", before, after, stable_proof)
    except FocusStopped as error:
        evidence.update(status="stopped", code="guard_focus_stopped")
        error.evidence = evidence
        raise
    except Exception as error:
        code = error.code if isinstance(error, native_ui.NativeUiUnavailable) else "guard_focus_observation_unavailable"
        evidence.update(code=code, error_type=type(error).__name__)
        raise FocusUnavailable(code, evidence) from error


def _identity_main():
    """Private one-request CPU helper; never exposes an action API."""
    try:
        raw = sys.stdin.buffer.readline(4097)
        request = json.loads(raw)
        if (len(raw) > 4096 or not isinstance(request, dict)
                or set(request) != {"operation", "window_id", "deadline"}
                or request["operation"] not in {"owner", "stable"}
                or type(request["deadline"]) not in {int, float}
                or not math.isfinite(request["deadline"])
                or not time.monotonic() < request["deadline"] <= time.monotonic() + 2.5):
            raise ValueError("Invalid private read request")
        environment = dict(os.environ)
        if request["operation"] == "owner":
            value = native_ui._x11_window_owner(request["window_id"], environment,
                include_binding=True, deadline=request["deadline"])
        else:
            if request["window_id"] is not None:
                raise ValueError("Stable read has no supplied window")
            value = native_ui._x11_foreground(environment, include_binding=True, deadline=request["deadline"])
        binding = X11FocusReader._binding(value)
        if time.monotonic() >= request["deadline"]:
            raise native_ui.NativeUiUnavailable("native_observation_budget_exceeded")
        reply = {"ok": True, "binding": asdict(binding)}
        status = 0
    except Exception as error:
        code = error.code if isinstance(error, native_ui.NativeUiUnavailable) else "guard_focus_owner_unavailable"
        reply, status = {"ok": False, "code": code}, 1
    sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    # Keep the leader unreaped while the parent cleans its exact process group.
    sys.stdin.buffer.read(1)
    return status


if __name__ == "__main__":
    if sys.argv[1:] != ["--read-identity"]:
        raise SystemExit(2)
    raise SystemExit(_identity_main())
