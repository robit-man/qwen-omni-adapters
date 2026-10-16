from __future__ import annotations

import copy
import hashlib
import io
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image
from test_native_manipulation import _visual_fixture

from portal import native_ui
from qwen_omni_adapters.computer_use import RapidUiController

_IDENTITY = {"window_id": "17", "process_id": 99}
_ROOT = {"window_id": "1", "geometry": {"x": 0, "y": 0, "width": 1920, "height": 1200}}
_CLIENT_QUERY = '''
xwininfo: Window id: 0x11 "Owned client"
  Root window id: 0x1 (the root window) (has no name)
  Parent window id: 0x2 "Frame"
     0 children.
'''
_ROOT_QUERY = '''
xwininfo: Window id: 0x1 (the root window) (has no name)
  Absolute upper-left X: 0
  Absolute upper-left Y: 0
  Width: 1920
  Height: 1200
'''


def _png(width: int, height: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_capture_root_proves_client_root_matches_default_screen_and_preserves_environment(monkeypatch):
    commands = []
    environment = {"DISPLAY": ":fixture", "LC_ALL": "fr_FR"}

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"capture_output": True, "text": True, "timeout": 2, "check": False,
                          "env": {"DISPLAY": ":fixture", "LC_ALL": "C"}}
        return SimpleNamespace(returncode=0, stdout=_CLIENT_QUERY if "-children" in command else _ROOT_QUERY)

    monkeypatch.setattr(native_ui.subprocess, "run", run)
    assert native_ui._x11_capture_root(_IDENTITY, environment) == _ROOT
    assert commands == [["xwininfo", "-id", "17", "-children"], ["xwininfo", "-root"]]
    assert environment == {"DISPLAY": ":fixture", "LC_ALL": "fr_FR"}


@pytest.mark.parametrize(("client", "root"), [
    (_CLIENT_QUERY.replace("Window id: 0x11", "Window id: 0x12"), _ROOT_QUERY),
    (_CLIENT_QUERY + 'xwininfo: Window id: 0x11 "Duplicate"\n', _ROOT_QUERY),
    (_CLIENT_QUERY.replace("Root window id:", "Missing root:"), _ROOT_QUERY),
    (_CLIENT_QUERY + "  Root window id: 0x1\n", _ROOT_QUERY),
    (_CLIENT_QUERY.replace("Root window id: 0x1", "Root window id: 0x0"), _ROOT_QUERY),
    (_CLIENT_QUERY.replace("Root window id: 0x1", "Root window id: 0x100000000"), _ROOT_QUERY),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("Window id: 0x1", "Window id: 0x2")),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("Width: 1920", "Width: 0")),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("Height: 1200", "Height: 16385")),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("upper-left X: 0", "upper-left X: 1")),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("upper-left Y: 0", "upper-left Y: -1")),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("Width: 1920", "Width: １９２０")),
    (_CLIENT_QUERY, _ROOT_QUERY + "  Width: 1920\n"),
    (_CLIENT_QUERY, _ROOT_QUERY.replace("  Height: 1200\n", "")),
])
def test_unknown_wrong_screen_ambiguous_or_invalid_root_never_binds_capture(monkeypatch, client, root):
    monkeypatch.setattr(native_ui.subprocess, "run", lambda command, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=client if "-children" in command else root))
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._x11_capture_root(_IDENTITY, {})
    assert exc.value.code == "native_capture_root_unavailable"


@pytest.mark.parametrize("failure", ["exit", "oversized", "oserror", "timeout"])
def test_root_query_failures_are_bounded_and_fail_closed(monkeypatch, failure):
    def run(command, **_kwargs):
        if failure == "oserror":
            raise OSError("Unavailable X11 query")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 2)
        return SimpleNamespace(returncode=int(failure == "exit"),
                               stdout="x" * 65537 if failure == "oversized" else _CLIENT_QUERY)

    monkeypatch.setattr(native_ui.subprocess, "run", run)
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._x11_capture_root(_IDENTITY, {})
    assert exc.value.code == "native_capture_root_unavailable"


def _capture_fixture(monkeypatch, surface, dimensions):
    state = {"identity": dict(_IDENTITY), "geometry": dict(surface), "root": copy.deepcopy(_ROOT),
             "process_start": "1234", "raw": _png(*dimensions), "returncode": 0, "after_capture": None}
    commands = []
    monkeypatch.setattr(native_ui, "_x11_foreground", lambda _env: copy.deepcopy(state["identity"]))
    monkeypatch.setattr(native_ui, "_x11_geometry", lambda *_args: copy.deepcopy(state["geometry"]))
    monkeypatch.setattr(native_ui, "_x11_capture_root", lambda *_args: copy.deepcopy(state["root"]))
    monkeypatch.setattr(native_ui, "_process_start", lambda _pid: state["process_start"])

    def run(command, **kwargs):
        commands.append(command)
        assert command == ["import", "+frame", "+border", "-screen", "-window", "17", "png:-"]
        assert kwargs == {"capture_output": True, "timeout": 5, "check": False, "env": {"DISPLAY": ":fixture"}}
        if state["after_capture"]:
            state["after_capture"]()
        return SimpleNamespace(returncode=state["returncode"], stdout=state["raw"])

    monkeypatch.setattr(native_ui.subprocess, "run", run)
    return state, commands


@pytest.mark.parametrize(("surface", "image", "offset"), [
    ({"x": 100, "y": 200, "width": 600, "height": 400},
     {"x": 100, "y": 200, "width": 600, "height": 400}, {"x": 0, "y": 0}),
    ({"x": -17, "y": 10, "width": 972, "height": 1184},
     {"x": 0, "y": 10, "width": 955, "height": 1184}, {"x": 17, "y": 0}),
    ({"x": 10, "y": -23, "width": 972, "height": 1184},
     {"x": 10, "y": 0, "width": 972, "height": 1161}, {"x": 0, "y": 23}),
    ({"x": 1700, "y": 10, "width": 500, "height": 200},
     {"x": 1700, "y": 10, "width": 220, "height": 200}, {"x": 0, "y": 0}),
    ({"x": 10, "y": 1100, "width": 500, "height": 200},
     {"x": 10, "y": 1100, "width": 500, "height": 100}, {"x": 0, "y": 0}),
    ({"x": -17, "y": -23, "width": 972, "height": 1184},
     {"x": 0, "y": 0, "width": 955, "height": 1161}, {"x": 17, "y": 23}),
    ({"x": -100, "y": -100, "width": 2200, "height": 1400},
     {"x": 0, "y": 0, "width": 1920, "height": 1200}, {"x": 100, "y": 100}),
])
def test_screen_capture_binds_exact_root_intersection_without_resizing_surface(monkeypatch, surface, image, offset):
    state, commands = _capture_fixture(monkeypatch, surface, (image["width"], image["height"]))
    captured = native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert captured == {"image": state["raw"], "sha256": hashlib.sha256(state["raw"]).hexdigest(),
                        "geometry": surface, "identity": _IDENTITY, "process_start": "1234",
                        "capture_source": "x11_root_screen", "capture_root": _ROOT,
                        "image_geometry": image, "pixel_offset": offset}
    assert len(commands) == 1


@pytest.mark.parametrize("surface", [
    {"x": -100, "y": 10, "width": 100, "height": 100},
    {"x": 1920, "y": 10, "width": 100, "height": 100},
    {"x": 10, "y": -100, "width": 100, "height": 100},
    {"x": 10, "y": 1200, "width": 100, "height": 100},
])
def test_fully_offscreen_window_is_rejected_before_any_capture(monkeypatch, surface):
    _state, commands = _capture_fixture(monkeypatch, surface, (1, 1))
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert exc.value.code == "native_capture_window_offscreen"
    assert commands == []


@pytest.mark.parametrize("changed", ["root_id", "root_width", "surface", "window", "pid", "process_start"])
def test_after_capture_binding_changes_reject_image_without_retry(monkeypatch, changed):
    state, commands = _capture_fixture(monkeypatch, {"x": 10, "y": 20, "width": 100, "height": 80}, (100, 80))

    def mutate():
        if changed == "root_id":
            state["root"]["window_id"] = "2"
        elif changed == "root_width":
            state["root"]["geometry"]["width"] += 1
        elif changed == "surface":
            state["geometry"]["x"] += 1
        elif changed == "window":
            state["identity"]["window_id"] = "18"
        elif changed == "pid":
            state["identity"]["process_id"] = 100
        else:
            state["process_start"] = "1235"

    state["after_capture"] = mutate
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert exc.value.code == "native_capture_window_changed"
    assert len(commands) == 1


def test_wrong_foreground_is_rejected_without_capturing_unowned_screen_pixels(monkeypatch):
    state, commands = _capture_fixture(monkeypatch, {"x": 10, "y": 20, "width": 100, "height": 80}, (100, 80))
    state["identity"]["window_id"] = "18"
    with pytest.raises(native_ui.NativeUiUnavailable):
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert commands == []


def test_pid_reuse_during_capture_preparation_rejects_before_import(monkeypatch):
    _state, commands = _capture_fixture(monkeypatch, {"x": 10, "y": 20, "width": 100, "height": 80}, (100, 80))
    starts = iter(["1234", "1235"])
    monkeypatch.setattr(native_ui, "_process_start", lambda _pid: next(starts))
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert exc.value.code == "native_capture_window_changed"
    assert commands == []


@pytest.mark.parametrize("invalid", ["exit", "empty", "signature", "chunk", "dimensions", "oversized"])
def test_capture_requires_success_and_exact_bounded_ihdr_without_retry(monkeypatch, invalid):
    state, commands = _capture_fixture(monkeypatch, {"x": 10, "y": 20, "width": 100, "height": 80}, (100, 80))
    if invalid == "exit":
        state["returncode"] = 1
    elif invalid == "empty":
        state["raw"] = b""
    elif invalid == "signature":
        state["raw"] = b"notPNG!!" + state["raw"][8:]
    elif invalid == "chunk":
        state["raw"] = state["raw"][:12] + b"JUNK" + state["raw"][16:]
    elif invalid == "dimensions":
        state["raw"] = _png(101, 80)
    else:
        state["raw"] += b"x" * (8 * 1024 * 1024)
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert exc.value.code == ("native_capture_geometry_mismatch" if invalid == "dimensions"
                              else "native_window_capture_unavailable")
    assert len(commands) == 1


def test_unbounded_clipped_rectangle_is_rejected_before_capture(monkeypatch):
    state, commands = _capture_fixture(monkeypatch, {"x": 0, "y": 0, "width": 5000, "height": 80}, (1, 1))
    state["root"]["geometry"]["width"] = 6000
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        native_ui._capture_window(_IDENTITY, {"DISPLAY": ":fixture"})
    assert exc.value.code == "native_capture_geometry_mismatch"
    assert commands == []


def test_visual_normalized_point_uses_clipped_image_but_dispatch_keeps_full_surface(monkeypatch):
    _fixture, adapter, frame, _requests = _visual_fixture(monkeypatch)
    frame["geometry"] = {"x": -17, "y": -23, "width": 972, "height": 1184}
    frame["image_geometry"] = {"x": 0, "y": 0, "width": 955, "height": 1161}
    frame["pixel_offset"] = {"x": 17, "y": 23}
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    observed = controller.observe()
    action = next(item for item in controller.enumerate(adapter._step, observed) if item["kind"] == "action")
    dispatched = []
    monkeypatch.setattr(native_ui, "_pointer_dispatch", lambda identity, geometry, point, verb, **_kwargs:
                        dispatched.append((identity, geometry, point, verb)) or {"status": "dispatched"})
    assert adapter.perform(action)["status"] == "dispatched"
    assert dispatched == [(frame["identity"], frame["geometry"], (238, 580), "click")]
    receipt = adapter.point_receipt
    for field in ("geometry", "image_geometry", "pixel_offset", "capture_source", "capture_root", "identity", "process_start"):
        assert receipt[field] == frame[field]
    assert receipt["frame_sha256"] == frame["sha256"]


@pytest.mark.parametrize("changed", ["geometry", "image_geometry", "pixel_offset", "capture_root", "capture_source", "process_start", "sha256"])
def test_visual_frame_key_binds_capture_origin_root_surface_process_and_pixels(monkeypatch, changed):
    fixture, adapter, frame, _requests = _visual_fixture(monkeypatch)
    monkeypatch.setattr(native_ui, "_capture_window", lambda *_args: copy.deepcopy(frame))
    monkeypatch.setattr(native_ui, "_x11_foreground", lambda _env: dict(fixture.foreground))
    monkeypatch.setattr("portal.desktop.desktop_subprocess_environment", lambda: {})
    # Exercise the real key producer rather than the fixture's fixed key stub.
    monkeypatch.setattr(adapter, "_visual_capture", native_ui.NativeUiAdapter._visual_capture.__get__(adapter))
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    observed = controller.observe()
    action = next(item for item in controller.enumerate(adapter._step, observed) if item["kind"] == "action")
    if changed in {"geometry", "image_geometry", "pixel_offset"}:
        frame[changed]["x"] += 1
    elif changed == "capture_root":
        frame[changed]["window_id"] = "2"
    else:
        frame[changed] += "changed"
    dispatched = []
    monkeypatch.setattr(native_ui, "_pointer_dispatch", lambda *_args, **_kwargs: dispatched.append(True))
    assert adapter.perform(action)["reason"] == "native_visual_frame_changed"
    assert dispatched == []
