"""Case-bound, one-shot interruptions for the frozen owned-close cases.

Only the live runner supplies native setup/save evidence and effect callbacks.
This module never interprets a planner's Save wording or creates task outputs.
Readiness is a capability receipt; an actual close needs a separate journal.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_omni_adapters.ui_effects import _document
from runtime.long_horizon_cases import bounded_bytes, safe_path
from runtime.owned_window_close import _identity_matches, _valid_identity, _xlib

KINDS = {"LHN-089": "owned_editor_close", "LHN-090": "owned_browser_close"}
SCHEMA = "robit.omni.long-native-owned-close-capability.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def supported(case: dict) -> bool:
    kind = (case.get("controlled_perturbation") or {}).get("kind")
    return kind is not None and kind == KINDS.get(case.get("case_id"))


def identity(row: dict) -> dict:
    result = {key: str(row[key]) for key in ("window_id", "process_id", "process_start")}
    if not _valid_identity(result["window_id"], result["process_id"], result["process_start"]):
        raise ValueError("Invalid owned close identity")
    return result


def probe_owned_close(target: dict, environment: dict) -> dict:
    """Read-only isolated X11 protocol/identity probe; never sends WM_DELETE."""
    target = identity(target)
    try:
        result = subprocess.run([sys.executable, "-I", "-u", str(Path(__file__).resolve()),
            "--probe", target["window_id"], target["process_id"], target["process_start"]],
            env=dict(environment), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=3, check=False)
        observed = json.loads(result.stdout) if result.returncode == 0 and len(result.stdout) <= 4096 else None
        if (isinstance(observed, dict) and observed.get("identity") == target
                and observed.get("status") == "supported" and observed.get("input_attempted") is False):
            return observed
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"identity": target, "status": "unavailable", "input_attempted": False}


def _protocol_probe(target: dict) -> dict:
    xid, pid, start = (target[key] for key in ("window_id", "process_id", "process_start"))
    result = {"identity": target, "status": "unavailable", "input_attempted": False}
    if not _identity_matches(xid, pid, start):
        return result
    library = _xlib()
    display = library.XOpenDisplay(None)
    if not display:
        return result
    protocols = ctypes.POINTER(ctypes.c_ulong)()
    try:
        atom = library.XInternAtom(display, b"WM_DELETE_WINDOW", True)
        count = ctypes.c_int()
        if (atom and library.XGetWMProtocols(display, int(xid), ctypes.byref(protocols), ctypes.byref(count))
                and protocols and 0 < count.value <= 64
                and atom in [protocols[index] for index in range(count.value)]
                and _identity_matches(xid, pid, start)):
            result["status"] = "supported"
        return result
    finally:
        if protocols:
            library.XFree(protocols)
        library.XCloseDisplay(display)


def _setup_windows(case: dict, setup: list) -> tuple[list, list]:
    if not supported(case) or not isinstance(setup, list) or not 1 <= len(setup) <= 8:
        raise ValueError("Unsupported owned close setup")
    if len(setup) != sum(case["fixture"]["blank_app_window_counts"].values()):
        raise ValueError("Owned close setup differs from frozen total window count")
    windows, seen = [], set()
    for row in setup:
        current = identity(row)
        if current["window_id"] in seen or row.get("setup_origin") != "declared_fixture":
            raise ValueError("Owned close requires distinct declared setup windows")
        seen.add(current["window_id"])
        observed = row.get("native", {})
        native = observed.get("state", {})
        if (observed.get("outside_owned_foreground") is not False or observed.get("window_id") != current["window_id"]
                or native.get("scope", {}).get("window_id") != current["window_id"] or not native.get("nodes")):
            raise ValueError("Missing positive native setup evidence")
        windows.append({"application": row["application"], **current})
    for app, count in case["fixture"]["blank_app_window_counts"].items():
        if sum(row["application"] == app for row in windows) != count:
            raise ValueError("Owned close setup differs from frozen window counts")
    editors = [row for row in windows if row["application"] == "gedit"]
    targets = editors if KINDS[case["case_id"]] == "owned_editor_close" else [
        row for row in windows if row["application"] == "browser"]
    if not editors or not targets:
        raise ValueError("Owned close setup has no trigger editor or target")
    return windows, targets


def capability(case: dict, setup_path: Path, work: Path, *, revision: str, probe) -> dict:
    setup = json.loads(bounded_bytes(setup_path))
    windows, targets = _setup_windows(case, setup)
    probes = [probe(identity(row)) for row in targets]
    for row, check in zip(targets, probes, strict=True):
        if check != {"identity": identity(row), "status": "supported", "input_attempted": False}:
            raise ValueError("Owned close protocol capability was not positively observed")
    return {"schema": SCHEMA, "authority": "host_independent_evaluator", "status": "verified",
        "case_id": case["case_id"], "task_signature_sha256": case["task_signature_sha256"],
        "source_revision": revision, "work_root": str(work.resolve()), "kind": KINDS[case["case_id"]],
        "setup_path": setup_path.name, "setup_sha256": hashlib.sha256(bounded_bytes(setup_path)).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "close_helper_sha256": hashlib.sha256(Path(__file__).with_name("owned_window_close.py").read_bytes()).hexdigest(),
        "windows": windows, "targets": targets, "protocol_probes": probes,
        "input_attempted": False, "event_executed": False, "solver_progress_credit": False}


def validate_capability(case: dict, path: Path, work: Path, *, revision: str) -> dict:
    if path.resolve().is_relative_to(work.resolve()):
        raise ValueError("Solver workbench cannot confer close capability")
    value = json.loads(bounded_bytes(path))
    if (not supported(case) or value.get("schema") != SCHEMA
            or value.get("authority") != "host_independent_evaluator" or value.get("status") != "verified"
            or value.get("case_id") != case["case_id"]
            or value.get("task_signature_sha256") != case["task_signature_sha256"]
            or value.get("source_revision") != revision or value.get("work_root") != str(work.resolve())
            or value.get("kind") != KINDS[case["case_id"]]
            or value.get("input_attempted") is not False or value.get("event_executed") is not False
            or value.get("solver_progress_credit") is not False):
        raise ValueError("Unbound owned close capability")
    setup_path = safe_path(path.parent, value["setup_path"])
    if hashlib.sha256(bounded_bytes(setup_path)).hexdigest() != value["setup_sha256"]:
        raise ValueError("Owned close setup evidence changed")
    expected = capability(case, setup_path, work, revision=revision,
                          probe=lambda target: next((p for p in value["protocol_probes"]
                              if p.get("identity") == target), None))
    if value != expected:
        raise ValueError("Owned close capability does not match its setup and implementation")
    return value


def saved_draft(case: dict, value: dict, work: Path, initial: dict, observed: dict) -> dict | None:
    """Prove a current owned editor buffer equals stable newly saved file bytes.

    No filename-only trigger or completion requirement is added to the frozen
    first independently confirmed saved-draft trigger.
    """
    active = str(observed.get("window_id", ""))
    editors = [row for row in value["windows"] if row["application"] == "gedit" and row["window_id"] == active]
    state = observed.get("state", {})
    if (len(editors) != 1 or observed.get("outside_owned_foreground") is not False
            or state.get("scope", {}).get("window_id") != active):
        return None
    candidates = []
    for node in state.get("nodes", []):
        text, coverage = node.get("value"), node.get("text_observation", {})
        if (node.get("visible") is not True or node.get("enabled") is not True
                or node.get("sensitive") is not False or node.get("scope_id") != state["scope"].get("id")
                or node.get("role") != "textbox" or not isinstance(text, str) or not text
                or coverage.get("truncated") is not False
                or type(coverage.get("character_count")) is not int
                or coverage["character_count"] != len(text)
                or coverage.get("captured_range") != {"start": 0, "end": len(text)}):
            continue
        candidates.append(text.encode())
    matches = []
    for number, path in enumerate(sorted(work.rglob("*"))):
        if number >= 1024:
            raise ValueError("Saved draft inventory exceeded its bound")
        if not path.is_file():
            continue
        relative = path.relative_to(work).as_posix()
        safe_path(work, relative)
        if relative in initial:
            continue
        content = bounded_bytes(path)
        if content in candidates:
            matches.append({"relative_path": relative, "sha256": hashlib.sha256(content).hexdigest(),
                            "bytes": len(content), "editor": identity(editors[0]),
                            "native_sha256": digest(observed)})
    return matches[0] if len(matches) == 1 else None


def research_browser(case: dict, value: dict, observed: dict) -> dict | None:
    """Bind research identity to a committed declared-source native document.

    Window order, titles, address-bar strings and planner source claims do not
    establish which of the two owned browsers the solver used for research.
    """
    active = str(observed.get("window_id", ""))
    windows = [row for row in value["windows"] if row["application"] == "browser"
               and row["window_id"] == active]
    state = observed.get("state", {})
    if (len(windows) != 1 or observed.get("outside_owned_foreground") is not False
            or state.get("scope", {}).get("window_id") != active):
        return None
    try:
        document = _document(state)
    except (KeyError, TypeError, ValueError):
        return None
    if document is None:
        return None
    try:
        actual = urlsplit(document[0])
        actual_port = actual.port
    except ValueError:
        return None
    for source in case["sources_to_acquire"]:
        declared = urlsplit(source["url"])
        prefix = declared.path.rstrip("/")
        if (actual.scheme == declared.scheme == "https" and actual.hostname == declared.hostname
                and actual_port == declared.port and not actual.username and not actual.password
                and (actual.path == prefix or actual.path.startswith(prefix + "/"))):
            return {"target": identity(windows[0]), "declared_source_id": source["source_id"],
                "declared_source_url": source["url"], "document_evidence": state["document_evidence"],
                "native_sha256": digest(observed), "authority": "historical_host_native_document",
                "source_acquisition_credit": False}
    return None


def window_observation(target: dict, evidence: Any) -> tuple[str, dict]:
    """A send acknowledgement cannot substitute for current window absence."""
    if (not isinstance(evidence, dict) or evidence.get("identity") != target
            or not isinstance(evidence.get("inventory"), dict)
            or evidence["inventory"].get("available") is not True
            or not isinstance(evidence["inventory"].get("windows"), dict)
            or len(evidence["inventory"]["windows"]) > 512):
        return "unknown", {"status": "unknown", "reason": "window_inventory_unavailable"}
    windows = evidence["inventory"]["windows"]
    current = windows.get(target["window_id"])
    status = ("absent" if target["window_id"] not in windows else "present_same"
        if isinstance(current, dict) and current.get("pid") == target["process_id"]
        and current.get("start_ticks") == target["process_start"]
        else "changed_or_unknown")
    return status, {**evidence, "status": status}


class OwnedCloseEvent:
    """One conditional event; uncertain delivery is never blindly replayed."""

    def __init__(self, case: dict, capability_path: Path, work: Path, initial: dict, *, revision: str,
                 journal: Path, probe, request_close, window_status, reserve_close,
                 clock=time.monotonic, sleep=time.sleep):
        self.case, self.work, self.initial = case, work, dict(initial)
        self.value = validate_capability(case, capability_path, work, revision=revision)
        self.journal, self.probe, self.request_close = journal, probe, request_close
        self.window_status, self.reserve_close = window_status, reserve_close
        self.clock, self.sleep = clock, sleep
        self.research_observations: dict[str, dict] = {}
        self.receipt = {"schema": "robit.omni.long-native-owned-close-event.v1",
            "case_id": case["case_id"], "task_signature_sha256": case["task_signature_sha256"],
            "source_revision": revision, "capability_sha256": hashlib.sha256(bounded_bytes(capability_path)).hexdigest(),
            "status": "armed", "trigger_verified": False, "dispatch_attempted": False,
            "window_closed_observed": False, "preservation_verified": False,
            "input_authority": False, "solver_progress_credit": False}
        with journal.open("x", encoding="utf-8") as output:
            output.write(json.dumps(self.receipt, sort_keys=True) + "\n")

    def _write(self):
        with self.journal.open("a", encoding="utf-8") as output:
            output.write(json.dumps(self.receipt, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        return dict(self.receipt)

    def observe(self, observed: dict) -> dict:
        if self.receipt["status"] != "armed":
            return dict(self.receipt)
        if self.value["kind"] == "owned_browser_close":
            research = research_browser(self.case, self.value, observed)
            if research is not None:
                self.research_observations[research["target"]["window_id"]] = research
        proof = saved_draft(self.case, self.value, self.work, self.initial, observed)
        if proof is None:
            return dict(self.receipt)
        self.receipt.update(trigger_verified=True, saved_draft=proof)
        if self.value["kind"] == "owned_browser_close":
            self.receipt["research_observations"] = list(self.research_observations.values())
            if len(self.research_observations) != 1:
                self.receipt.update(status="blocked_before_input", reason="owned_research_browser_unproven_or_ambiguous")
                return self._write()
            target = next(iter(self.research_observations.values()))["target"]
        else:
            target = proof["editor"]
        self.receipt.update(trigger_verified=True, saved_draft=proof, target=target)
        protected = {**self.initial, proof["relative_path"]: proof["sha256"]}
        try:
            before_status, before_windows = window_observation(target, self.window_status(target))
            self.receipt["window_before"] = before_windows
            survivor_app = "gedit" if self.value["kind"] == "owned_editor_close" else "browser"
            survivors = [identity(row) for row in self.value["windows"] if identity(row) != target]
            standby = [row for row in self.value["windows"]
                       if row["application"] == survivor_app and identity(row) != target]
            if (self.probe(target) != {"identity": target, "status": "supported", "input_attempted": False}
                    or before_status != "present_same" or not standby
                    or any(window_observation(other, {**before_windows, "identity": other})[0] != "present_same"
                           for other in survivors)
                    or any(hashlib.sha256(bounded_bytes(safe_path(self.work, name))).hexdigest() != expected
                           for name, expected in protected.items())):
                self.receipt["status"] = "blocked_before_input"
                return self._write()
            self.reserve_close(target)  # Cleanup must not retry a possibly delivered close.
            self.receipt.update(status="request_pending", dispatch_attempted=True)
            self._write()  # Durable before the only effect callback.
            self.receipt["request_reported_sent"] = bool(self.request_close(target))
            deadline = self.clock() + 3
            for _ in range(61):
                status, observation = window_observation(target, self.window_status(target))
                self.receipt["window_observation"] = observation
                if status == "absent":
                    self.receipt["window_closed_observed"] = True
                    break
                if status != "present_same" or self.clock() >= deadline:
                    break
                self.sleep(.05)
            self.receipt["preservation_verified"] = all(
                hashlib.sha256(bounded_bytes(safe_path(self.work, name))).hexdigest() == expected
                for name, expected in protected.items())
            self.receipt["surviving_owned_window_verified"] = all(
                window_observation(other, {**self.receipt.get("window_observation", {}), "identity": other})[0]
                == "present_same" for other in survivors)
            self.receipt["status"] = ("observed_closed" if self.receipt["request_reported_sent"]
                and self.receipt["window_closed_observed"] and self.receipt["preservation_verified"]
                and self.receipt["surviving_owned_window_verified"] else "unknown")
        except Exception as error:
            self.receipt.update(status="unknown" if self.receipt["dispatch_attempted"] else "blocked_before_input",
                                error_type=type(error).__name__)
        return self._write()

    def snapshot(self) -> dict:
        return {**self.receipt, "journal_sha256": hashlib.sha256(bounded_bytes(self.journal)).hexdigest()}


def verified_event(case: dict, root: Path, report: dict) -> bool:
    """Require the actual frozen post-trigger journal, not readiness alone."""
    if not supported(case):
        return True
    try:
        value = report["controlled_perturbation"]
        body = bounded_bytes(root / "owned-close-events.jsonl")
        records = [json.loads(line) for line in body.splitlines()]
        final = records[-1]
        capability_path = root / "owned-close-capability.json"
        cap = validate_capability(case, capability_path, root / "Workbench",
            revision=report["source_before"]["revision"])
        status, _ = window_observation(final["target"], final["window_observation"])
        before, _ = window_observation(final["target"], final["window_before"])
        survivors = [identity(row) for row in cap["windows"] if identity(row) != final["target"]]
        preserved_windows = all(window_observation(other,
            {**final[side], "identity": other})[0] == "present_same"
            for other in survivors for side in ("window_before", "window_observation"))
        milestones = [json.loads(line) for line in bounded_bytes(root / "milestones.jsonl").splitlines()]
        proof = final["saved_draft"]
        triggers = [index for index, row in enumerate(milestones)
            if digest(row["native"]) == proof["native_sha256"]
            and row.get("files", {}).get("files", {}).get(proof["relative_path"], {}).get("sha256") == proof["sha256"]
            and row.get("files", {}).get("inputs_unchanged") is True]
        trigger_verified = bool(triggers) and proof["editor"] in [
            identity(row) for row in cap["windows"] if row["application"] == "gedit"]
        if cap["kind"] == "owned_editor_close":
            trigger_verified = trigger_verified and proof["editor"] == final["target"]
        research_verified = True
        if cap["kind"] == "owned_browser_close":
            sources = final.get("research_observations", [])
            research_verified = (trigger_verified and len(sources) == 1 and sources[0]["target"] == final["target"]
                and any(research_browser(case, cap, row["native"]) == sources[0]
                        for row in milestones[:triggers[0]]))
        return (2 <= len(records) <= 4 and value["journal_sha256"] == hashlib.sha256(body).hexdigest()
            and {k: v for k, v in value.items() if k != "journal_sha256"} == final
            and final["case_id"] == case["case_id"]
            and final["task_signature_sha256"] == case["task_signature_sha256"]
            and final["capability_sha256"] == hashlib.sha256(bounded_bytes(capability_path)).hexdigest()
            and final["status"] == "observed_closed" and final["trigger_verified"] is True
            and final["dispatch_attempted"] is True and final["request_reported_sent"] is True
            and final["preservation_verified"] is True and final["surviving_owned_window_verified"] is True
            and status == "absent" and before == "present_same" and preserved_windows
            and trigger_verified and research_verified
            and final["target"] in [identity(row) for row in cap["targets"]]
            and final["input_authority"] is False and final["solver_progress_credit"] is False)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--probe":
        raise SystemExit(2)
    try:
        print(json.dumps(_protocol_probe(identity(dict(zip(
            ("window_id", "process_id", "process_start"), sys.argv[2:], strict=True))))))
    except Exception:
        raise SystemExit(1) from None
