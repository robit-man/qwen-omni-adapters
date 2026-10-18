"""One-shot frozen calendar amendment fixture; never a solver or an output writer.

The optional system-GI probe only reads the exact registered Calc window. A
capability receipt is not an event receipt: a release still requires a current
Document.DocURL, a new valid persisted workbook and two matching native probes.
Importing this module performs no desktop, model, network or CUDA operations.
"""
from __future__ import annotations

import fcntl
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import ZIP_STORED, BadZipFile, ZipFile

from runtime.long_horizon_cases import MAX_FILE, bounded_bytes, load_case, safe_path, sha

SCHEMA = "robit.omni.long-native-calendar-amendment.v1"
TRIGGER = "independent first successful calendar workbook save"
RELEASE_PATH = "Input/Amendments/revision-B.json"
MIME = b"application/vnd.oasis.opendocument.spreadsheet"
NS = {"office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
      "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
      "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}


def supported(case):
    return (case.get("case_id") in {f"LHN-{i:03}" for i in range(77, 83)}
            and (case.get("controlled_perturbation") or {}).get("kind") == "predeclared_input_release")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _identity(window):
    if (not isinstance(window, dict) or set(window) != {"window_id", "process_id", "process_start"}
            or not isinstance(window["window_id"], str) or not window["window_id"].isdecimal()
            or not 0 < int(window["window_id"]) < 2**32
            or type(window["process_id"]) is not int or not 0 < window["process_id"] < 2**31
            or not isinstance(window["process_start"], str) or not window["process_start"].isdecimal()):
        raise ValueError("Expected exact registered Calc XID/PID/start")
    return dict(window)


def _root(work):
    if work.is_symlink() or not work.is_dir() or work.stat().st_uid != os.getuid():
        raise ValueError("Workbench must be an owner-owned real directory")
    work = work.resolve(strict=True)
    info = work.stat()
    return {"path": str(work), "device": info.st_dev, "inode": info.st_ino, "owner": info.st_uid}


def _declared(case):
    if (not isinstance(case, dict) or case.get("case_id") not in {f"LHN-{i:03}" for i in range(77, 83)}
            or case != load_case(case["case_id"])):
        raise ValueError("Expected unchanged frozen timed-amendment case")
    event = case["controlled_perturbation"]
    if (event.get("kind") != "predeclared_input_release" or event.get("trigger") != TRIGGER
            or event.get("payload", {}).get("path") != RELEASE_PATH):
        raise ValueError("Frozen amendment trigger or payload changed")
    return (json.dumps(event["payload"]["content"], sort_keys=True, indent=2,
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def inventory(work):
    """Stable bounded owner-owned file hashes; never follows links."""
    result = {}
    for index, path in enumerate(sorted(work.rglob("*"))):
        if index >= 1024:
            raise ValueError("Workbench inventory exceeds bound")
        name = path.relative_to(work).as_posix()
        safe_path(work, name)
        if path.is_dir():
            if path.stat().st_uid != os.getuid():
                raise ValueError("Foreign owned directory")
            continue
        content = bounded_bytes(path)
        result[name] = {"sha256": sha(content), "bytes": len(content)}
    return result


def _probe_shape(probe, window, *, saved=False):
    if (not isinstance(probe, dict) or probe.get("status") != "observed"
            or probe.get("authority") != "host_read_only_atspi"
            or probe.get("source") != "atspi.Document.DocURL"
            or probe.get("window") != window or probe.get("role") != "document spreadsheet"
            or probe.get("docurl_query_supported") is not True
            or not isinstance(probe.get("provider_identity_sha256"), str)
            or len(probe["provider_identity_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in probe["provider_identity_sha256"])
            or not isinstance(probe.get("url"), str)
            or saved and not probe["url"]):
        raise ValueError("Independent exact spreadsheet Document evidence unavailable")
    return probe


def _setup(case, work):
    path = work.parent / "setup-native-observations.json"
    body = bounded_bytes(path)
    rows = json.loads(body)
    counts = case["fixture"]["blank_app_window_counts"]
    if not isinstance(rows, list) or len(rows) != sum(counts.values()):
        raise ValueError("Missing exact declared blank-app setup")
    windows, calc = [], []
    for row in rows:
        window = _identity({"window_id": row["window_id"], "process_id": int(row["process_id"]),
                            "process_start": row["process_start"]})
        native = row.get("native", {})
        if (row.get("setup_origin") != "declared_fixture" or native.get("phase") != "setup"
                or native.get("error") or native.get("outside_owned_foreground") is not False
                or native.get("window_id") != window["window_id"]
                or native.get("state", {}).get("scope", {}).get("window_id") != window["window_id"]
                or not native.get("state", {}).get("nodes")):
            raise ValueError("Missing exact registered native setup evidence")
        windows.append((row["application"], window))
        if row["application"] == "calc":
            calc.append((window, row.get("calendar_document_probe")))
    if (len({w["window_id"] for _, w in windows}) != len(windows) or len(calc) != 1
            or any(sum(app == key for app, _ in windows) != count for key, count in counts.items())):
        raise ValueError("Setup window identities or application counts differ")
    _probe_shape(calc[0][1], calc[0][0])
    return body, calc[0]


def prepare_capability(case, *, work, revision, initial_hashes, calc_window, document_probe):
    """Bind actual fresh blank-Calc capability; no amendment is released here."""
    payload = _declared(case)
    window = _identity(calc_window)
    _probe_shape(document_probe, window)
    setup_bytes, (setup_window, setup_probe) = _setup(case, work)
    if setup_window != window or setup_probe != document_probe:
        raise ValueError("Calendar probe is not bound to fresh registered setup")
    declared = {item["path"]: item["content_sha256"] for item in case["fixture"]["input_files"]}
    observed = inventory(work)
    if initial_hashes != declared or {name: item["sha256"] for name, item in observed.items()} != declared:
        raise ValueError("Capability setup must contain exactly unchanged declared inputs")
    if safe_path(work, RELEASE_PATH).exists():
        raise ValueError("Amendment must be absent at setup")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected frozen source revision")
    return {"schema": SCHEMA, "kind": "injector_capability", "authority": "host_independent_evaluator",
            "case_id": case["case_id"], "task_signature_sha256": case["task_signature_sha256"],
            "source_revision": revision, "implementation_sha256": sha(Path(__file__).read_bytes()),
            "workbench": _root(work), "initial_hashes": declared, "calc_window": window,
            "setup_path": "setup-native-observations.json", "setup_sha256": sha(setup_bytes),
            "blank_document_probe": document_probe, "trigger": TRIGGER, "release_path": RELEASE_PATH,
            "payload_sha256": sha(payload), "status": "armed_capability_only", "event_occurred": False,
            "whole_task_success_credit": False}


def validate_capability(case, capability, *, work, revision, initial_hashes):
    """Check source/case/workroot/probe bindings; never proves a future event."""
    try:
        payload = _declared(case)
        if (capability.get("schema") != SCHEMA or capability.get("kind") != "injector_capability"
                or capability.get("authority") != "host_independent_evaluator"
                or capability.get("case_id") != case["case_id"]
                or capability.get("task_signature_sha256") != case["task_signature_sha256"]
                or capability.get("source_revision") != revision
                or capability.get("implementation_sha256") != sha(Path(__file__).read_bytes())
                or capability.get("workbench") != _root(work)
                or capability.get("initial_hashes") != initial_hashes
                or capability.get("initial_hashes") != {i["path"]: i["content_sha256"] for i in case["fixture"]["input_files"]}
                or capability.get("trigger") != TRIGGER or capability.get("release_path") != RELEASE_PATH
                or capability.get("payload_sha256") != sha(payload)
                or capability.get("status") != "armed_capability_only" or capability.get("event_occurred") is not False):
            return False
        _probe_shape(capability["blank_document_probe"], _identity(capability["calc_window"]))
        setup_bytes, (window, probe) = _setup(case, work)
        return (capability.get("setup_path") == "setup-native-observations.json"
                and capability.get("setup_sha256") == sha(setup_bytes)
                and capability["calc_window"] == window and capability["blank_document_probe"] == probe)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def persisted_workbook(content):
    """Format/first-save proof only: one cell is enough; correctness is ungraded."""
    try:
        with ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if (not 1 <= len(infos) <= 256 or len(set(names)) != len(names)
                    or sum(i.file_size for i in infos) > MAX_FILE
                    or infos[0].filename != "mimetype" or infos[0].compress_type != ZIP_STORED
                    or any(i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16)
                           or "\\" in i.filename or i.filename.startswith("/")
                           or any(p in {".", ".."} for p in i.filename.rstrip("/").split("/")) for i in infos)
                    or archive.read("mimetype") != MIME or archive.testzip() is not None):
                return False
            manifest = archive.read("META-INF/manifest.xml")
            xml = archive.read("content.xml")
            if any(len(part) > 2 * 1024 * 1024 or b"<!DOCTYPE" in part.upper() or b"<!ENTITY" in part.upper()
                   for part in (xml, manifest)):
                return False
            manifest_ns = "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
            entries = ElementTree.fromstring(manifest).findall("{" + manifest_ns + "}file-entry")
            if not any(entry.get("{" + manifest_ns + "}full-path") == "/"
                       and entry.get("{" + manifest_ns + "}media-type") == MIME.decode() for entry in entries):
                return False
            root = ElementTree.fromstring(xml)
            sheets = root.findall("office:body/office:spreadsheet/table:table", NS)
            return any(cell.get("{" + NS["office"] + "}value") is not None
                       or cell.get("{" + NS["office"] + "}date-value") is not None
                       or cell.get("{" + NS["office"] + "}boolean-value") is not None
                       or any("".join(p.itertext()).strip() for p in cell.findall("text:p", NS))
                       for sheet in sheets for cell in sheet.findall(".//table:table-cell", NS))
    except (BadZipFile, KeyError, OSError, RuntimeError, ValueError, ElementTree.ParseError):
        return False


def _saved_path(work, probe):
    url = urlsplit(probe["url"])
    if (url.scheme != "file" or url.netloc not in {"", "localhost"} or url.query or url.fragment
            or url.username or url.password or "%00" in url.path.lower()
            or any(ord(c) < 32 or ord(c) == 127 for c in probe["url"])):
        raise ValueError("Saved document must be an exact local owned file URI")
    path = Path(unquote(url.path))
    if not path.is_absolute():
        raise ValueError("Saved document URI must be absolute")
    name = path.relative_to(work.resolve()).as_posix()
    if name.startswith("Input/"):
        raise ValueError("Declared inputs cannot be the authored saved calendar")
    return safe_path(work, name), name


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(_json(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _release(work, payload):
    """Publish complete bytes without overwriting even a concurrent file/link."""
    root_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    input_fd = None
    try:
        input_fd = os.open("Input", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            os.mkdir("Amendments", 0o700, dir_fd=input_fd)
        except FileExistsError:
            pass
        directory = os.open("Amendments", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=input_fd)
    finally:
        if input_fd is not None:
            os.close(input_fd)
        os.close(root_fd)
    temporary = ".release-" + os.urandom(16).hex()
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o444)
        # Hard-link publication is atomic and O_EXCL-like; unlike replace(), it
        # cannot overwrite a solver/racing writer. The temporary has no alias
        # after this call and the final file remains an ordinary single link.
        os.link(temporary, "revision-B.json", src_dir_fd=directory,
                dst_dir_fd=directory, follow_symlinks=False)
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
        content = bounded_bytes(safe_path(work, RELEASE_PATH))
        if content != payload:
            raise ValueError("Released amendment readback differs")
        return {"path": RELEASE_PATH, "sha256": sha(content), "bytes": len(content), "readback_verified": True}
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


class AmendmentRelease:
    """Durable one-attempt state; uncertainty is sticky across object restarts."""

    def __init__(self, case, *, capability, work, revision, initial_hashes, event_root,
                 environment=None, probe=None):
        self.case, self.work, self.capability = case, work, capability
        if not validate_capability(case, capability, work=work, revision=revision, initial_hashes=initial_hashes):
            raise ValueError("Invalid amendment injector capability")
        event_root.mkdir(mode=0o700, exist_ok=True)
        if (event_root.is_symlink() or event_root.resolve().is_relative_to(work.resolve())
                or event_root.stat().st_uid != os.getuid()):
            raise ValueError("Event journal must be an independent owned directory")
        self.root, self.environment = event_root, environment
        self.probe = probe or probe_calendar_document
        self.state_path = event_root / "amendment-event.json"
        self.journal_path = event_root / "amendment-events.jsonl"
        self.binding = {"schema": SCHEMA, "case_id": case["case_id"],
                        "task_signature_sha256": case["task_signature_sha256"],
                        "capability_sha256": sha(_json(capability)), "workbench": _root(work)}
        if not self.journal_path.exists():
            descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            state = {**self.binding, "status": "armed", "release_attempts": 0}
            with os.fdopen(descriptor, "wb") as output:
                output.write(_json(state) + b"\n")
                output.flush()
                os.fsync(output.fileno())
            _atomic_json(self.state_path, state)

    def _persist(self, state):
        descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "ab") as output:
            output.write(_json(state) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        _atomic_json(self.state_path, state)

    def _records(self):
        body = bounded_bytes(self.journal_path)
        rows = [json.loads(line) for line in body.splitlines()]
        states = [row.get("status") for row in rows]
        if (states not in (["armed"], ["armed", "attempted"], ["armed", "attempted", "released"],
                           ["armed", "attempted", "unknown"])
                or any(any(row.get(key) != value for key, value in self.binding.items()) for row in rows)):
            raise ValueError("Durable amendment journal binding or transition changed")
        return body, rows

    def snapshot(self):
        body, records = self._records()
        return {**records[-1], "journal_sha256": sha(body)}

    def _freeze_first(self, content):
        path = self.root / "first-saved-workbook.ods"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            if bounded_bytes(path) != content:
                raise ValueError("First-save evidence already differs; never substitute a later save") from None
        else:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
                os.fchmod(output.fileno(), 0o444)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        if bounded_bytes(path) != content:
            raise ValueError("First-save evidence readback changed")

    def observe(self, native):
        """Called after each real native phase; never interpret action prose."""
        lock = os.open(self.root / "amendment.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, records = self._records()
            state = records[-1]
            if _root(self.work) != self.capability["workbench"]:
                raise ValueError("Workbench identity changed")
            if state.get("status") != "armed":
                return None
            window = self.capability["calc_window"]
            if (not isinstance(native, dict) or native.get("phase") != "model_task"
                    or native.get("window_id") != window["window_id"]
                    or native.get("state", {}).get("scope", {}).get("window_id") != window["window_id"]
                    or native.get("error") or native.get("outside_owned_foreground") is not False):
                return None
            try:
                first = _probe_shape(self.probe(window, self.environment), window, saved=True)
                path, name = _saved_path(self.work, first)
                content = bounded_bytes(path)
                if not persisted_workbook(content):
                    return None
                before = inventory(self.work)
                if (name not in before or before[name]["sha256"] != sha(content)
                        or any(before.get(p, {}).get("sha256") != h for p, h in self.capability["initial_hashes"].items())):
                    return None
                second = _probe_shape(self.probe(window, self.environment), window, saved=True)
                if first != second or bounded_bytes(path) != content:
                    return None
                self._freeze_first(content)  # Evaluator-only original for first-save/final ODF diff.
            except (OSError, TypeError, ValueError, subprocess.SubprocessError):
                return None
            state.update(status="attempted", release_attempts=1, trigger=TRIGGER,
                         native_sha256=sha(_json(native)), document_probe=first,
                         first_saved_workbook={"path": name, "sha256": sha(content), "bytes": len(content),
                                               "evidence_path": "first-saved-workbook.ods"},
                         files_before=before, whole_task_success_credit=False)
            self._persist(state)  # Journal attempt before publication; stale snapshots cannot permit retries.
            try:
                release = _release(self.work, _declared(self.case))
                after = inventory(self.work)
                unchanged = {p: v for p, v in after.items() if p != RELEASE_PATH} == before
                if not unchanged or bounded_bytes(path) != content:
                    raise ValueError("Output or prior input changed during release")
                state.update(status="released", release=release, files_after=after,
                             host_solver_output_mutation=False, unchanged_prior_files=True)
            except (OSError, ValueError) as exc:
                state.update(status="unknown", error_type=type(exc).__name__, retry_permitted=False)
            self._persist(state)
            return state
        finally:
            os.close(lock)


def verified_event(case, root, report):
    """Require actual release, trigger milestone and final preservation, never capability alone."""
    if not supported(case):
        return True
    try:
        work = root / "Workbench"
        capability = json.loads(bounded_bytes(root / "amendment-capability.json"))
        if not validate_capability(case, capability, work=work, revision=report["source_before"]["revision"],
                                   initial_hashes=report["initial_input_hashes"]):
            return False
        body = bounded_bytes(root / "amendment-events/amendment-events.jsonl")
        records = [json.loads(line) for line in body.splitlines()]
        if len(records) != 3 or [r.get("status") for r in records] != ["armed", "attempted", "released"]:
            return False
        final = records[-1]
        if (report["controlled_perturbation"] != {**final, "journal_sha256": sha(body)}
                or json.loads(bounded_bytes(root / "amendment-events/amendment-event.json")) != final
                or any(r.get("capability_sha256") != sha(_json(capability)) for r in records)
                or any(r.get("case_id") != case["case_id"]
                       or r.get("task_signature_sha256") != case["task_signature_sha256"]
                       or r.get("workbench") != capability["workbench"] for r in records)
                or any(final.get(key) != value for key, value in records[1].items()
                       if key not in {"status"})
                or [r.get("release_attempts") for r in records] != [0, 1, 1] or final.get("trigger") != TRIGGER
                or final.get("whole_task_success_credit") is not False
                or final.get("host_solver_output_mutation") is not False
                or final.get("unchanged_prior_files") is not True):
            return False
        _probe_shape(final["document_probe"], capability["calc_window"], saved=True)
        _, name = _saved_path(work, final["document_probe"])
        saved = final["first_saved_workbook"]
        frozen = bounded_bytes(safe_path(root / "amendment-events", saved["evidence_path"]))
        payload = _declared(case)
        expected = {"path": RELEASE_PATH, "sha256": sha(payload), "bytes": len(payload), "readback_verified": True}
        before, after = final["files_before"], final["files_after"]
        if (saved.get("path") != name or saved.get("evidence_path") != "first-saved-workbook.ods"
                or sha(frozen) != saved["sha256"] or len(frozen) != saved["bytes"] or not persisted_workbook(frozen)
                or before.get(name) != {"sha256": saved["sha256"], "bytes": saved["bytes"]}
                or RELEASE_PATH in before or final.get("release") != expected
                or after.get(RELEASE_PATH) != {"sha256": sha(payload), "bytes": len(payload)}
                or {p: v for p, v in after.items() if p != RELEASE_PATH} != before
                or bounded_bytes(safe_path(work, RELEASE_PATH)) != payload
                or any(sha(bounded_bytes(safe_path(work, p))) != h for p, h in report["initial_input_hashes"].items())):
            return False
        milestones = [json.loads(line) for line in bounded_bytes(root / "milestones.jsonl").splitlines()]
        matches = [m for m in milestones if sha(_json(m.get("native"))) == final["native_sha256"]]
        return any(m.get("native", {}).get("phase") == "model_task"
                   and m.get("native", {}).get("outside_owned_foreground") is False
                   and m.get("native", {}).get("window_id") == capability["calc_window"]["window_id"]
                   and m.get("files", {}).get("files") == before
                   and m.get("files", {}).get("inputs_unchanged") is True for m in matches)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def probe_calendar_document(window, environment):
    """System-GI subprocess: exact-scope read only; never activates or types."""
    request = _identity(window)
    repo = Path(__file__).resolve().parents[1]
    child = {**(os.environ if environment is None else environment),
             "PYTHONPATH": os.pathsep.join((str(repo), str(repo / "src"))), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(["/usr/bin/python3", "-m", "runtime.long_horizon_amendments", "--probe"],
        input=_json(request), capture_output=True, env=child, cwd=repo, timeout=9, check=False)  # noqa: S603
    if result.returncode or len(result.stdout) > 32768:
        raise ValueError("Exact read-only calendar probe unavailable")
    return _probe_shape(json.loads(result.stdout), window)


def _probe_main():
    """No action API is called; GI bridge guards window/PID/start around reads."""
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi, GLib

    from portal.native_ui import _AtspiBridge, _process_start

    window = _identity(json.loads(sys.stdin.buffer.read(32769)))
    scope = {"schema": "robit.omni.native-task-scope.v1", "windows": [window]}
    bridge = _AtspiBridge(Atspi, GLib, "independent-calendar-save", window["process_id"], window_scope=scope)
    if _process_start(window["process_id"]) != window["process_start"]:
        raise ValueError("Calc process generation changed")
    observed = bridge.observe()
    if observed["scope"]["window_id"] != window["window_id"]:
        raise ValueError("Calc document is not current")
    matches = [(node, item) for node, item, _ in bridge.handles.values()
               if item["scope_id"] == observed["scope"]["id"]
               and item["role"] == "document_spreadsheet" and item["visible"] and not item["sensitive"]]
    if len(matches) != 1:
        raise ValueError("One exact native spreadsheet document is required")
    node, _ = matches[0]
    def read():
        bridge._clear_cache(node)
        state = bridge._read(node.get_state_set)
        if (bridge._read(node.get_role_name) != "document spreadsheet"
                or not all(state.contains(s) for s in (Atspi.StateType.VISIBLE, Atspi.StateType.SHOWING))
                or state.contains(Atspi.StateType.DEFUNCT) or "Document" not in bridge._read(node.get_interfaces)):
            raise ValueError("Current native spreadsheet Document is unavailable")
        interface = bridge._read(node.get_document_iface)
        attributes = bridge._read(lambda: Atspi.Document.get_document_attributes(interface))
        url = bridge._read(lambda: Atspi.Document.get_document_attribute_value(interface, "DocURL"))
        # An untitled Calc document can omit DocURL from its attribute map;
        # successful interface/query support with an empty URL is setup only.
        # The later event still requires a nonempty exact owned file URI.
        if (not isinstance(attributes, dict) or not isinstance(url, str)
                or "DocURL" in attributes and attributes["DocURL"] != url):
            raise ValueError("Document does not expose a supported DocURL query")
        raw = bridge._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
        return {"status": "observed", "authority": "host_read_only_atspi", "source": "atspi.Document.DocURL",
                "window": window, "role": "document spreadsheet", "url": url,
                "docurl_query_supported": True, "provider_identity_sha256": sha(raw.encode())}
    first, second = read(), read()
    if (first != second or bridge.foreground() != {k: window[k] for k in ("window_id", "process_id")}
            or _process_start(window["process_id"]) != window["process_start"]):
        raise ValueError("Document/window changed during observation")
    sys.stdout.buffer.write(_json(first))


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit("Only the private read-only --probe mode is supported")
    _probe_main()
