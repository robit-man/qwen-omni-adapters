from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from portal import native_identity as identity


def direct(function):
    return function()


def valid_sample():
    uid = os.getuid()

    def process(pid, ppid, namespace):
        return {"pid": pid, "ppid": ppid, "start": str(pid * 10), "uid": uid,
                "nspid": [pid, 2], "namespaces": {"pid": namespace, "user": "user:[1]", "mnt": "mnt:[2]"},
                "cgroup": ["0::/app-flatpak-owned.scope"]}

    host, child = process(20, 19, "pid:[9]"), process(19, 18, "pid:[9]")
    peer, wrapper = process(17, 16, "pid:[1]"), process(18, 15, "pid:[1]")
    metadata = {"application": "org.mozilla.firefox", "instance": "133", "sha256": "f" * 64,
                "file": {"device": 1, "inode": 2, "uid": uid, "mode": stat.S_IFREG | 0o666,
                         "size": 120, "mtime_ns": 7, "ctime_ns": 8},
                "root": {"device": 1, "inode": 3, "uid": uid}, "read_only": False}
    values = {"host": host, "child": child, "peer": peer, "wrapper": wrapper,
              "lineage": [host, child], "bus_name": ":1.237",
              "credentials": {"pid": 17, "uid": uid}, "owner": {"pid": 20, "start": "200"},
              "registry": {"instance": "133", "wrapper": 18, "child": 19, "application": "org.mozilla.firefox"},
              "installed_proxy": {"device": 1, "inode": 4, "uid": 0, "mode": stat.S_IFREG | 0o755,
                                  "size": 123, "mtime_ns": 9, "ctime_ns": 10},
              "metadata": {key: copy.deepcopy(metadata) for key in ("host", "peer", "instance")}}
    values["metadata"]["host"]["read_only"] = True
    values["running_proxy"] = copy.deepcopy(values["installed_proxy"])
    return values


def mutate(value, path, replacement):
    current = value
    keys = path.split(".")
    for key in keys[:-1]:
        current = current[int(key)] if isinstance(current, list) else current[key]
    if isinstance(current, list):
        current[int(keys[-1])] = replacement
    else:
        current[keys[-1]] = replacement


def test_complete_portal_equivalent_identity_allows_rw_proxy_without_proxy_ancestry():
    value = valid_sample()
    assert value["peer"]["ppid"] not in {value["host"]["pid"], value["wrapper"]["pid"]}
    assert value["metadata"]["peer"]["read_only"] is False
    identity._validate(value)


@pytest.mark.parametrize(("path", "replacement"), [
    ("host.uid", -1), ("peer.uid", -1), ("child.uid", -1), ("wrapper.uid", -1),
    ("credentials.pid", 999), ("credentials.uid", -1), ("owner.pid", 999), ("owner.start", "201"),
    ("running_proxy.inode", 999), ("installed_proxy.uid", 1),
    ("installed_proxy.mode", stat.S_IFREG | 0o777), ("installed_proxy.mode", stat.S_IFLNK | 0o755),
    ("peer.cgroup", ["0::/other.scope"]), ("child.cgroup", []), ("wrapper.cgroup", []),
    ("child.ppid", 999), ("host.namespaces.pid", "pid:[3]"),
    ("lineage", []), ("registry.child", 999), ("registry.wrapper", 999),
    ("metadata.host.read_only", False), ("metadata.peer.sha256", "e" * 64),
    ("metadata.instance.instance", "134"), ("metadata.peer.application", "org.other.App"),
    ("metadata.host.file.uid", -1), ("metadata.peer.file.mode", stat.S_IFIFO | 0o600),
    ("metadata.instance.file.size", identity.MAX_METADATA + 1),
])
def test_hash_or_cgroup_alone_never_authorizes_identity(path, replacement):
    value = valid_sample()
    mutate(value, path, replacement)
    with pytest.raises(identity.IdentityUnavailable):
        identity._validate(value)


def test_host_lineage_must_have_every_positive_parent_edge():
    value = valid_sample()
    value["host"]["ppid"] = 123
    with pytest.raises(identity.IdentityUnavailable, match="lineage"):
        identity._validate(value)


def test_provider_samples_all_before_after_values_and_binds_digest(monkeypatch):
    value = valid_sample()
    samples = []

    def sample(*args, **kwargs):
        samples.append((args, kwargs))
        return copy.deepcopy(value)

    monkeypatch.setattr(identity, "_sample", sample)
    result = identity.provider_proof({"window_id": "80", "process_id": 20}, ":1.237", 17,
                                     credentials=lambda _bus: {}, read=direct, deadline=time.monotonic() + 1)
    assert len(samples) == 2
    assert result["values"] == value
    assert result["digest"] == hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert result["reported_pid"] == 17 and result["identity"]["process_id"] == 20


@pytest.mark.parametrize("path", ["host.start", "peer.start", "wrapper.start", "child.start",
                                  "metadata.peer.file.inode", "metadata.peer.root.inode", "peer.namespaces.mnt"])
def test_any_before_after_identity_change_fails(monkeypatch, path):
    before, after = valid_sample(), valid_sample()
    mutate(after, path, 999 if path.endswith("inode") else "changed")
    samples = iter([before, after])
    monkeypatch.setattr(identity, "_sample", lambda *args, **kwargs: next(samples))
    with pytest.raises(identity.IdentityUnavailable, match="identity_changed"):
        identity.provider_proof({"window_id": "80", "process_id": 20}, ":1.237", 17,
                                credentials=lambda _bus: {}, read=direct, deadline=time.monotonic() + 1)


def test_ordinary_non_flatpak_provider_never_queries_registry_or_credentials(monkeypatch):
    monkeypatch.setattr(identity, "_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(identity, "_registry", lambda **kwargs: pytest.fail("registry not applicable"))
    assert identity.provider_proof({"window_id": "80", "process_id": 20}, ":1.237", 17,
        credentials=lambda _bus: pytest.fail("credentials not applicable"), read=direct,
        deadline=time.monotonic() + 1) is None


def test_metadata_pins_root_reads_regular_owned_file_and_retains_only_allowed_ids(tmp_path):
    raw = b"[Application]\nname=org.mozilla.firefox\n[Instance]\ninstance-id=133\n[Private]\nvalue=never-retained\n"
    (tmp_path / "info").write_bytes(raw)
    result = identity._metadata(tmp_path, "info", read=direct, deadline=time.monotonic() + 1, nofollow_root=True)
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "never-retained" not in json.dumps(result)
    assert result["file"]["inode"] == (tmp_path / "info").stat().st_ino


@pytest.mark.parametrize("kind", ["symlink", "root_symlink", "oversize", "directory", "empty", "wrong_uid", "changed"])
def test_metadata_rejects_unsafe_or_changed_file(tmp_path, monkeypatch, kind):
    directory = tmp_path / "real"
    directory.mkdir()
    file = directory / "info"
    file.write_text("[Application]\nname=org.mozilla.firefox\n[Instance]\ninstance-id=133\n")
    root = directory
    if kind == "symlink":
        file.rename(directory / "target")
        file.symlink_to("target")
    elif kind == "root_symlink":
        root = tmp_path / "link"
        root.symlink_to(directory, target_is_directory=True)
    elif kind == "oversize":
        file.write_bytes(b"x" * (identity.MAX_METADATA + 1))
    elif kind == "directory":
        file.unlink()
        file.mkdir()
    elif kind == "empty":
        file.write_bytes(b"")
    elif kind == "wrong_uid":
        monkeypatch.setattr(identity.os, "getuid", lambda: -1)
    elif kind == "changed":
        original = identity.os.read

        def read(fd, size):
            result = original(fd, size)
            file.write_bytes(file.read_bytes() + b"x")
            return result

        monkeypatch.setattr(identity.os, "read", read)
    with pytest.raises((OSError, identity.IdentityUnavailable)):
        identity._metadata(root, "info", read=direct, deadline=time.monotonic() + 1, nofollow_root=True)


def test_optional_metadata_absence_is_not_positive_flatpak_proof(tmp_path):
    assert identity._metadata(tmp_path, "missing", read=direct, deadline=time.monotonic() + 1, optional=True) is None


def test_deadline_rejects_before_any_io():
    with pytest.raises(identity.IdentityUnavailable, match="budget_exceeded"):
        identity._io(lambda: pytest.fail("must not read"), direct, time.monotonic() - 1)


def test_read_budget_error_propagates_unchanged():
    error = RuntimeError("read budget exhausted")

    def denied(_function):
        raise error

    with pytest.raises(RuntimeError) as caught:
        identity._io(lambda: None, denied, time.monotonic() + 1)
    assert caught.value is error


@pytest.mark.parametrize("stdout", ["1\t2\t3\torg.example.App\n" * 129, "invalid", "1\t2\tthree\torg.example.App"])
def test_registry_requires_complete_bounded_typed_rows(monkeypatch, stdout):
    monkeypatch.setattr(identity.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    with pytest.raises(identity.IdentityUnavailable):
        identity._registry(read=direct, deadline=time.monotonic() + 1)


def test_xres_child_fixed_argv_explicit_environment_no_global_mutation(monkeypatch):
    captured = []
    parent = dict(os.environ)

    def run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"pid":20,"start":"200"}')

    monkeypatch.setattr(identity.subprocess, "run", run)
    assert identity.xres_owner("80", {"DISPLAY": ":9"}) == {"pid": 20, "start": "200"}
    command, options = captured[0]
    assert command == [identity.sys.executable, "-I", str(Path(identity.__file__).resolve()), "--xres-owner", "80"]
    assert options["env"] == {"DISPLAY": ":9"} and options["timeout"] <= 1
    assert dict(os.environ) == parent


@pytest.mark.parametrize("stdout", ['{"pid":true,"start":"1"}', '{"pid":20,"start":"bad"}',
                                   '{"pid":20,"start":"1","extra":1}', '[]', 'not json'])
def test_xres_child_bad_output_never_resolves_owner(monkeypatch, stdout):
    monkeypatch.setattr(identity.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    with pytest.raises((identity.IdentityUnavailable, ValueError, TypeError)):
        identity.xres_owner("80", {"DISPLAY": ":9"})


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.mark.parametrize(("fault", "passed"), [(None, True), ("status", False), ("count", False),
    ("mask", False), ("length", False), ("pid", False), ("extension", False), ("version", False),
    ("owner_change", False), ("start_change", False)])
def test_xres_requires_success_zero_one_pid_resource_and_stable_start(monkeypatch, fault, passed):
    retained, closed, destroyed = [], [], []
    queries = 0

    def version(_display, major, minor):
        ctypes.cast(major, ctypes.POINTER(ctypes.c_int))[0] = 1
        ctypes.cast(minor, ctypes.POINTER(ctypes.c_int))[0] = 1 if fault == "version" else 2
        return 1

    def query(_display, count, spec, result_count, result_values):
        nonlocal queries
        queries += 1
        spec_type = spec._obj.__class__
        value_type = result_values._obj._type_
        assert count == 1 and spec._obj.client == 80 and spec._obj.mask == 2
        value = value_type(spec_type(80, 1 if fault == "mask" else 2), 8 if fault == "length" else 4, None)
        retained.append(value)
        ctypes.cast(result_count, ctypes.POINTER(ctypes.c_long))[0] = 2 if fault == "count" else 1
        ctypes.cast(result_values, ctypes.POINTER(ctypes.POINTER(value_type)))[0] = ctypes.pointer(value)
        return 1 if fault == "status" else 0

    library = SimpleNamespace(XOpenDisplay=Function(lambda _name: 123), XCloseDisplay=Function(lambda _display: closed.append(True)))
    extension = SimpleNamespace(XResQueryExtension=Function(lambda *args: fault != "extension"),
        XResQueryVersion=Function(version), XResQueryClientIds=Function(query),
        XResGetClientPid=Function(lambda _value: -1 if fault == "pid" else 21 if fault == "owner_change" and queries > 1 else 20),
        XResClientIdsDestroy=Function(lambda *args: destroyed.append(True)))
    monkeypatch.setattr(identity.ctypes, "CDLL", lambda name: library if name == "libX11.so.6" else extension)
    starts = iter(["200", "201" if fault == "start_change" else "200"])
    monkeypatch.setattr(identity.Path, "read_text", lambda _path: "20 (owner) " + " ".join(["0"] * 19 + [next(starts)]))
    if passed:
        assert identity._xres("80") == {"pid": 20, "start": "200"}
        assert len(destroyed) == 2
    else:
        with pytest.raises(identity.IdentityUnavailable):
            identity._xres("80")
    assert closed == [True]


@pytest.mark.parametrize("fault", [None, "duplicate_instance", "wrong_peer_credentials", "wrong_lineage", "wrong_metadata", "wrong_executable"])
def test_collector_combines_registry_authenticated_peer_namespace_and_xres(monkeypatch, fault):
    value = valid_sample()
    calls = []
    process = {value[key]["pid"]: value[key] for key in ("host", "peer", "child", "wrapper")}
    if fault == "wrong_lineage":
        process[20]["ppid"] = 20
    if fault == "wrong_metadata":
        value["metadata"]["instance"]["sha256"] = "b" * 64
    original_stat = identity.Path.stat
    stamp = value["installed_proxy"]
    installed = SimpleNamespace(**{"st_" + key: replacement for key, replacement in {
        "dev": stamp["device"], "ino": stamp["inode"], "uid": stamp["uid"], "mode": stamp["mode"],
        "size": stamp["size"], "mtime_ns": stamp["mtime_ns"], "ctime_ns": stamp["ctime_ns"]}.items()})
    running = copy.copy(installed)
    if fault == "wrong_executable":
        running.st_ino += 1
    monkeypatch.setattr(identity, "PROXY", SimpleNamespace(stat=lambda: installed))
    monkeypatch.setattr(identity.Path, "stat", lambda path, **kwargs: running if str(path) == "/proc/17/exe" else original_stat(path, **kwargs))

    def metadata(root, name, **kwargs):
        calls.append((str(root), name, kwargs))
        key = "host" if str(root) == "/proc/20/root" else "peer" if str(root) == "/proc/17/root" else "instance"
        return copy.deepcopy(value["metadata"][key])

    monkeypatch.setattr(identity, "_metadata", metadata)
    monkeypatch.setattr(identity, "_process", lambda pid, **kwargs: copy.deepcopy(process[pid]))
    monkeypatch.setattr(identity, "_registry", lambda **kwargs: [value["registry"]] * (2 if fault == "duplicate_instance" else 1))
    monkeypatch.setattr(identity, "xres_owner", lambda xid, environment, **kwargs: value["owner"])
    kwargs = {"credentials": lambda bus: {"pid": 999 if fault == "wrong_peer_credentials" else 17, "uid": os.getuid()},
              "read": direct, "deadline": time.monotonic() + 1}
    if fault == "wrong_executable":
        assert identity._sample({"window_id": "80", "process_id": 20}, ":1.237", 17, **kwargs) is None
    elif fault is not None:
        with pytest.raises(identity.IdentityUnavailable):
            identity._sample({"window_id": "80", "process_id": 20}, ":1.237", 17, **kwargs)
    else:
        result = identity._sample({"window_id": "80", "process_id": 20}, ":1.237", 17, **kwargs)
        assert result["metadata"]["peer"]["read_only"] is False
        assert result["credentials"] == {"pid": 17, "uid": os.getuid()}
        assert calls[-1][2]["nofollow_root"] is True
        assert calls[0][2]["optional"] is True


@pytest.mark.parametrize("window", ["0", "-1", "4294967296", "１", "17; command", "", None])
def test_xres_rejects_unbounded_or_noncanonical_window_before_loading_library(monkeypatch, window):
    monkeypatch.setattr(identity.ctypes, "CDLL", lambda name: pytest.fail("invalid XID must not reach Xlib"))
    with pytest.raises(identity.IdentityUnavailable, match="window_invalid"):
        identity._xres(window)


@pytest.mark.parametrize("raw", [b"\xff", b"[Private]\nname=not-app\n", b"[Application]\nname=org.App\nname=org.Other\n"])
def test_metadata_parse_failure_is_typed_and_retains_no_raw_content(tmp_path, raw):
    (tmp_path / "info").write_bytes(raw)
    with pytest.raises(identity.IdentityUnavailable, match="metadata_invalid"):
        identity._metadata(tmp_path, "info", read=direct, deadline=time.monotonic() + 1)
