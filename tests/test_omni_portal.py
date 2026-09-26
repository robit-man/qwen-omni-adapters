from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote_plus

import httpx
import pytest
from PIL import Image

from harness.background_agent import (
    _background_tool_contract,
    _task_system_prompt,
    _task_virtual_query,
)
from portal.app import (
    DEFAULT_MODEL,
    PortalConfig,
    _model_tool_result,
    create_app,
    load_voice_profile,
)
from portal.browser import (
    _SNAPSHOT_SCRIPT,
    BrowserAutomationError,
    BrowserAutomationStore,
    _navigation_error_metadata,
)
from portal.documents import SessionDocumentStore, extract_document
from portal.gui import GuiAutomation, GuiAutomationError
from portal.tools import (
    DISCOVERY_TOOLS,
    SAFE_TOOLS,
    PortalToolHarness,
    discover_tool_names,
)
from qwen_omni_adapters.context import context_catalog
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPolicy, MemoryPressure

TOKEN = "portal-test-token-with-more-than-24-characters"


class _FakeSession:
    def __init__(self) -> None:
        self.process = _StaticProc()
        self.last_seen = time.monotonic()

    def poll_replacement(self) -> int | None:
        return None


class _StaticProc:
    def __init__(self) -> None:
        self.pid = 4242

    def poll(self) -> int | None:
        return None


def _memory_governor(available: float) -> MemoryGovernor:
    return MemoryGovernor(
        MemoryPolicy(
            enabled=True,
            soft_floor_gib=3.0,
            hard_floor_gib=2.0,
            operation_reserve_gib=1.0,
            poll_interval_s=0.01,
        ),
        sampler=lambda: available,
    )


def test_browser_uses_the_generic_runtime_memory_governor() -> None:
    governor = _memory_governor(2.5)
    store = BrowserAutomationStore(memory_governor=governor)

    assert store.memory_governor is governor
    try:
        governor.require("visible browser")
    except MemoryPressure as error:
        assert error.label == "visible browser"
    else:
        raise AssertionError("expected generic memory admission to defer the browser")


def test_browser_can_use_a_declarative_measured_launch_reserve() -> None:
    governor = _memory_governor(2.6)
    store = BrowserAutomationStore(
        memory_governor=governor,
        launch_reserve_gib=0.5,
    )

    governor.require_capacity("visible browser", store.launch_reserve_gib)


def test_browser_snapshot_exposes_collapsible_reasoning_and_tool_summaries() -> None:
    assert "select,summary," in _SNAPSHOT_SCRIPT
    assert "el.labels" in _SNAPSHOT_SCRIPT
    assert "type.toLowerCase() === 'password' ? ''" in _SNAPSHOT_SCRIPT
    assert "[...el.options]" in _SNAPSHOT_SCRIPT
    assert "ids: new WeakMap()" in _SNAPSHOT_SCRIPT
    assert "identity.ids.get(el)" in _SNAPSHOT_SCRIPT
    assert "removeAttribute('data-omni-id')" not in _SNAPSHOT_SCRIPT


def test_browser_drag_emits_a_pressed_mouse_path() -> None:
    class Cdp:
        calls: list[tuple[str, dict[str, Any]]] = []

        def call(self, method: str, arguments: dict[str, Any]) -> None:
            self.calls.append((method, arguments))

    cdp = Cdp()
    BrowserAutomationStore()._drag(
        cdp,  # type: ignore[arg-type]
        {"x": 10, "y": 20, "width": 40, "height": 10},
        120,
        5,
    )

    events = [arguments["type"] for _method, arguments in cdp.calls]
    assert events[0:2] == ["mouseMoved", "mousePressed"]
    assert events[-1] == "mouseReleased"
    assert cdp.calls[-1][1]["x"] == 150
    assert cdp.calls[-1][1]["y"] == 30


def test_browser_native_form_actions_validate_control_types(tmp_path: Path) -> None:
    class Store(BrowserAutomationStore):
        expressions: list[str] = []

        def _evaluate(self, _cdp, expression):
            self.expressions.append(expression)
            return (
                {"ok": True, "value": "2026-10-06"}
                if "el.value =" in expression
                else {"ok": True}
            )

    store = Store(upload_roots=[tmp_path])
    store._set_value(
        object(),  # type: ignore[arg-type]
        {"id": "e1", "tag": "input", "type": "date"},
        "2026-10-06",
    )
    store._select_values(
        object(),  # type: ignore[arg-type]
        {"id": "e2", "tag": "select", "type": ""},
        ["voice", "vision"],
    )

    assert "2026-10-06" in store.expressions[0]
    assert "voice" in store.expressions[1]
    with pytest.raises(BrowserAutomationError, match="limited to input types"):
        store._set_value(
            object(),  # type: ignore[arg-type]
            {"id": "e3", "tag": "input", "type": "text"},
            "bypass",
        )


def test_browser_network_error_page_is_failed_visual_evidence() -> None:
    result = _navigation_error_metadata(
        "chrome-error://chromewebdata/",
        "This site can't be reached 127.0.0.1 refused to connect ERR_CONNECTION_REFUSED",
    )

    assert result == {
        "error": "browser_navigation_error",
        "message": (
            "This site can't be reached 127.0.0.1 refused to connect "
            "ERR_CONNECTION_REFUSED"
        ),
        "failure_scope": "target_state",
        "task_blocked": False,
        "retryable": False,
    }
    assert _navigation_error_metadata("https://example.test/", "Loaded") == {}


def test_browser_upload_is_root_scoped_and_uses_cdp(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    allowed = upload_root / "note.txt"
    allowed.write_text("owned fixture", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("not allowed", encoding="utf-8")

    class Cdp:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def call(self, method: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((method, arguments))
            if method == "Runtime.evaluate":
                return {"result": {"objectId": "node-7"}}
            return {}

    store = BrowserAutomationStore(upload_roots=[upload_root])
    cdp = Cdp()
    store._upload_files(
        cdp,  # type: ignore[arg-type]
        {"id": "e7", "tag": "input", "type": "file"},
        [str(allowed)],
    )

    set_files = next(arguments for method, arguments in cdp.calls if method == "DOM.setFileInputFiles")
    assert set_files == {"files": [str(allowed)], "objectId": "node-7"}
    with pytest.raises(BrowserAutomationError, match="outside"):
        store._resolve_upload_files([str(outside)])


def test_browser_visual_click_maps_normalized_point_into_exact_viewport() -> None:
    class Store(BrowserAutomationStore):
        def _evaluate(self, _cdp, _expression):
            return {"url": "https://example.test/", "width": 1010, "height": 619}

    class Cdp:
        def __init__(self, screenshot: str) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.screenshot = screenshot

        def call(self, method: str, arguments: dict[str, Any]) -> dict[str, str]:
            self.calls.append((method, arguments))
            return {"data": self.screenshot} if method == "Page.captureScreenshot" else {}

    buffer = io.BytesIO()
    Image.new("RGB", (1010, 619), "white").save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode()
    _width, _height, sample = BrowserAutomationStore._screenshot_details(screenshot)

    session = SimpleNamespace(
        visual_frame={
            "url": "https://example.test/",
            "css_width": 1010,
            "css_height": 619,
            "root_css_width": 1010,
            "root_css_height": 619,
            "origin_css_x": 0,
            "origin_css_y": 0,
            "width": 1010,
            "height": 619,
            "pixel_origin_x": 0,
            "pixel_origin_y": 0,
            "refinement_depth": 1,
        },
        visual_sample=sample,
    )
    cdp = Cdp(screenshot)

    outcome = Store()._visual_click(
        session,  # type: ignore[arg-type]
        cdp,  # type: ignore[arg-type]
        {
            "x": 750,
            "y": 500,
            "coordinate_unit": "normalized_1000",
        },
    )

    events = [arguments for method, arguments in cdp.calls if method == "Input.dispatchMouseEvent"]
    assert outcome == "clicked"
    assert len(events) == 2
    assert events[0]["x"] == pytest.approx(756.75)
    assert events[0]["y"] == pytest.approx(309.0)


def test_browser_first_visual_point_returns_a_refinement_crop_without_clicking() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (1010, 619), "white").save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode()
    _width, _height, sample = BrowserAutomationStore._screenshot_details(screenshot)
    session = SimpleNamespace(
        visual_frame={
            "url": "https://example.test/",
            "refinement_depth": 0,
        },
        visual_sample=sample,
    )

    outcome = BrowserAutomationStore()._visual_click(
        session,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {"x": 820, "y": 475, "coordinate_unit": "normalized_1000"},
    )

    assert outcome == "refine"


def test_browser_point_head_maps_bounded_region_back_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "points": [{"x": 0.25, "y": 0.5}],
                    "model": "point-test",
                    "revision": "pinned",
                }
            ).encode()

    def open_request(request, **_kwargs):
        payload = json.loads(request.data)
        point_image = Image.open(
            io.BytesIO(base64.b64decode(payload["image"]))
        )
        captured["size"] = point_image.size
        captured["target"] = payload["target"]
        return Response()

    monkeypatch.setattr("portal.browser.urlopen", open_request)

    x, y, receipt = BrowserAutomationStore(
        pointing_url="http://127.0.0.1:8940"
    )._point_target(
        Image.new("RGB", (1000, 700), "white"),
        "blue triangle",
        800,
        300,
    )

    assert captured == {"size": (400, 300), "target": "blue triangle"}
    assert (x, y) == (699, 300)
    assert receipt["grounding_region"] == {
        "origin_x": 599,
        "origin_y": 60,
        "width": 400,
        "height": 300,
        "parent_width": 1000,
        "parent_height": 700,
    }


def test_browser_point_head_searches_same_band_after_bad_x_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Response:
        def __init__(self, points: list[dict[str, float]]) -> None:
            self.points = points

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "points": self.points,
                    "model": "point-test",
                    "revision": "pinned",
                }
            ).encode()

    def open_request(_request, **_kwargs):
        nonlocal calls
        calls += 1
        return Response([{"x": 0.5, "y": 0.5}] if calls == 2 else [])

    monkeypatch.setattr("portal.browser.urlopen", open_request)

    x, y, receipt = BrowserAutomationStore(
        pointing_url="http://127.0.0.1:8940"
    )._point_target(
        Image.new("RGB", (1000, 700), "white"),
        "purple diamond",
        800,
        300,
    )

    assert calls == 4
    assert (x, y) == (200, 300)
    assert receipt["fallback_search"] is True
    assert receipt["searched_region_count"] == 4
    assert receipt["grounding_region"]["origin_x"] == 0
    assert receipt["grounding_region"]["origin_y"] == 60


def test_browser_point_head_rejects_a_targetless_visual_click() -> None:
    session = SimpleNamespace(visual_frame={"url": "https://example.test"}, visual_sample=b"x")

    with pytest.raises(BrowserAutomationError, match="requires a concise target"):
        BrowserAutomationStore(
            pointing_url="http://127.0.0.1:8940"
        )._visual_click(
            session,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            {"x": 500, "y": 500, "coordinate_unit": "normalized_1000"},
        )


def test_browser_visual_observer_reads_the_exact_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "observation": "ALL VISUAL ACTION GATES PASSED; PASS-42",
                    "model": "visual-test",
                    "revision": "pinned",
                }
            ).encode()

    def open_request(request, **_kwargs):
        payload = json.loads(request.data)
        captured["url"] = request.full_url
        captured["keys"] = sorted(payload)
        captured["size"] = Image.open(
            io.BytesIO(base64.b64decode(payload["image"]))
        ).size
        return Response()

    monkeypatch.setattr("portal.browser.urlopen", open_request)
    store = BrowserAutomationStore(pointing_url="http://127.0.0.1:8940")
    frame = io.BytesIO()
    Image.new("RGB", (1000, 700), "white").save(frame, format="PNG")
    result = {
        "screenshot": {
            "data": base64.b64encode(frame.getvalue()).decode()
        }
    }

    store._attach_verified_frame_observation(result)

    assert captured == {
        "url": "http://127.0.0.1:8940/observe",
        "keys": ["image"],
        "size": (1000, 700),
    }
    assert result["verified_visual_observation"] == {
        "provenance": "current_browser_snapshot_visual_model",
        "observation": "ALL VISUAL ACTION GATES PASSED; PASS-42",
        "model": "visual-test",
        "revision": "pinned",
    }


def test_browser_dedicated_point_head_executes_full_viewport_target_directly() -> None:
    class Store(BrowserAutomationStore):
        def _evaluate(self, _cdp, _expression):
            return {"url": "https://example.test/", "width": 1000, "height": 700}

        def _point_target(self, image, target, proposed_x, proposed_y):
            assert image.size == (1000, 700)
            assert target == "blue triangle"
            assert (proposed_x, proposed_y) == (150, 150)
            return 760, 390, {
                "source": "dedicated_point_head",
                "target": target,
                "executed": {"x": 760, "y": 390},
            }

    class Cdp:
        def __init__(self, screenshot: str) -> None:
            self.screenshot = screenshot
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def call(self, method: str, arguments: dict[str, Any]) -> dict[str, str]:
            self.calls.append((method, arguments))
            return {"data": self.screenshot} if method == "Page.captureScreenshot" else {}

    buffer = io.BytesIO()
    Image.new("RGB", (1000, 700), "white").save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode()
    _width, _height, sample = BrowserAutomationStore._screenshot_details(screenshot)
    session = SimpleNamespace(
        visual_frame={
            "url": "https://example.test/",
            "css_width": 1000,
            "css_height": 700,
            "root_css_width": 1000,
            "root_css_height": 700,
            "origin_css_x": 0,
            "origin_css_y": 0,
            "width": 1000,
            "height": 700,
            "pixel_origin_x": 0,
            "pixel_origin_y": 0,
            "refinement_depth": 0,
        },
        visual_sample=sample,
        visual_grounding={},
    )
    cdp = Cdp(screenshot)

    outcome = Store(pointing_url="http://127.0.0.1:8940")._visual_click(
        session,  # type: ignore[arg-type]
        cdp,  # type: ignore[arg-type]
        {
            "x": 150,
            "y": 150,
            "target": "blue triangle",
            "coordinate_unit": "normalized_1000",
        },
    )

    events = [args for method, args in cdp.calls if method == "Input.dispatchMouseEvent"]
    assert outcome == "clicked"
    assert events[0]["x"] == pytest.approx(759.24)
    assert events[0]["y"] == pytest.approx(272.61)
    assert session.visual_grounding["source"] == "dedicated_point_head"
    assert session.visual_grounding["frame_changed_during_inference"] == 0


def test_completed_visual_click_returns_observation_and_exact_action_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cdp:
        def __init__(self, _url: str, _timeout_s: float) -> None:
            pass

        def call(self, _method: str, _arguments=None) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    class Store(BrowserAutomationStore):
        def __init__(self) -> None:
            super().__init__(pointing_url="http://127.0.0.1:8940")
            self.session = SimpleNamespace(
                port=9222,
                target_id="page",
                page_socket="ws://127.0.0.1/page/page",
                visual_grounding={},
            )
            self.observed = False

        def _session_locked(self, _session_id: str):
            return self.session

        def _page_socket(self, _port: int, _target_id: str = "") -> str:
            return "ws://127.0.0.1/page/page"

        def _visual_click(self, session, _cdp, _arguments):
            session.visual_grounding = {
                "source": "dedicated_point_head",
                "target": "BLUE TRIANGLE",
                "executed": {"x": 734, "y": 468},
            }
            return "clicked"

        def _wait_rendered(self, _cdp, _wait_ms: int) -> None:
            pass

        def _snapshot(self, _session, _cdp):
            return {"screenshot": {"data": "current-frame"}}

        def _attach_verified_frame_observation(self, result):
            self.observed = True
            result["verified_visual_observation"] = {
                "provenance": "current_browser_snapshot_visual_model",
                "observation": "Stage 2 of 3",
            }

    monkeypatch.setattr("portal.browser._Cdp", Cdp)
    store = Store()

    result = store.act(
        "session",
        {
            "action": "visual_click",
            "target": "BLUE TRIANGLE",
            "x": 800,
            "y": 470,
        },
    )

    assert store.observed is True
    assert result["verified_visual_observation"]["observation"] == "Stage 2 of 3"
    assert result["action_receipt"] == {
        "action": "visual_click",
        "target": "BLUE TRIANGLE",
        "executed": {"x": 734, "y": 468},
        "coordinate_unit": "normalized_1000",
    }


def test_browser_dom_action_returns_semantic_control_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cdp:
        def __init__(self, _url: str, _timeout_s: float) -> None:
            pass

        def call(self, _method: str, _arguments=None) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    class Store(BrowserAutomationStore):
        def __init__(self) -> None:
            super().__init__()
            self.session = SimpleNamespace(
                port=9222,
                target_id="page",
                page_socket="ws://127.0.0.1/page/page",
                visual_grounding={},
            )

        def _session_locked(self, _session_id: str):
            return self.session

        def _page_socket(self, _port: int, _target_id: str = "") -> str:
            return "ws://127.0.0.1/page/page"

        def _refresh_element(self, _session, _cdp, _element_id):
            return {
                "id": "e4",
                "tag": "input",
                "type": "text",
                "name": "recipient",
                "label": "Recipient — text",
            }

        def _click(self, _cdp, _element) -> None:
            pass

        def _wait_rendered(self, _cdp, _wait_ms: int) -> None:
            pass

        def _snapshot(self, _session, _cdp):
            return {"screenshot": {"data": "current-frame"}}

    monkeypatch.setattr("portal.browser._Cdp", Cdp)
    result = Store().act(
        "session",
        {"action": "type", "element_id": "e4", "text": "Avery Morgan"},
    )

    assert result["action_receipt"] == {
        "action": "type",
        "target": "Recipient — text",
    }

    navigated = Store().act(
        "session",
        {"action": "navigate", "url": "https://example.test/form"},
    )
    assert navigated["action_receipt"] == {
        "action": "navigate",
        "target": "https://example.test/form",
    }


def test_browser_refinement_crop_preserves_parent_viewport_transform() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (1010, 619), "white").save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode()
    session = SimpleNamespace(
        visual_frame={
            "url": "https://example.test/",
            "revision": 4,
            "root_css_width": 1010,
            "root_css_height": 619,
        },
        visual_sample=b"",
    )
    result = {"screenshot": {"data": screenshot}}

    refined = BrowserAutomationStore()._refine_visual_result(
        session,  # type: ignore[arg-type]
        result,
        {"x": 820, "y": 475},
    )

    assert refined["action_executed"] is False
    assert refined["visual_refinement_required"] is True
    assert refined["coordinate_space"] == {
        "name": "browser_viewport_region",
        "origin_x": 610,
        "origin_y": 144,
        "width": 400,
        "height": 300,
        "coordinate_units": ["normalized_1000"],
        "revision": 4,
        "parent": "browser_viewport",
    }
    crop = Image.open(
        io.BytesIO(base64.b64decode(refined["screenshot"]["data"]))
    )
    assert crop.size == (400, 300)
    assert session.visual_frame["origin_css_x"] == pytest.approx(610.0)
    assert session.visual_frame["origin_css_y"] == pytest.approx(144.0)
    assert session.visual_frame["refinement_depth"] == 1


def test_browser_dom_action_revalidates_live_box_and_hit_target() -> None:
    class Store(BrowserAutomationStore):
        def _evaluate(self, _cdp, _expression):
            return {"ok": True, "x": 90.0, "y": 40.0, "width": 120.0, "height": 30.0}

    session = SimpleNamespace(
        elements={
            "e3": {
                "id": "e3",
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 10,
            }
        }
    )

    refreshed = Store()._refresh_element(
        session,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "e3",
    )

    assert refreshed["x"] == 90.0
    assert refreshed["y"] == 40.0
    assert session.elements["e3"]["width"] == 120.0


def test_browser_dom_action_scrolls_offscreen_control_then_revalidates() -> None:
    class Store(BrowserAutomationStore):
        def __init__(self) -> None:
            super().__init__()
            self.results = iter(
                [
                    {"ok": False, "reason": "not_visible"},
                    True,
                    {"ok": True, "x": 80.0, "y": 200.0, "width": 160.0, "height": 32.0},
                ]
            )
            self.expressions: list[str] = []

        def _evaluate(self, _cdp, expression):
            self.expressions.append(expression)
            return next(self.results)

    session = SimpleNamespace(
        elements={
            "e6": {
                "id": "e6",
                "x": 80,
                "y": 900,
                "width": 160,
                "height": 32,
            }
        }
    )
    store = Store()

    refreshed = store._refresh_element(
        session,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "e6",
    )

    assert refreshed["y"] == 200.0
    assert len(store.expressions) == 3
    assert "scrollIntoView" in store.expressions[1]


def test_gui_drag_uses_one_bounded_xdotool_gesture() -> None:
    class Gui(GuiAutomation):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[list[str]] = []

        def _run(self, command: list[str]) -> str:
            self.commands.append(command)
            return ""

        def _desktop_state(self) -> tuple[int, int, dict[str, Any]]:
            return (
                1920,
                1080,
                {
                    "id": "42",
                    "title": "Browser",
                    "bounds": {"x": 50, "y": 30, "width": 1000, "height": 700},
                },
            )

        def _snapshot(self, coordinate_space: str = "active_window") -> dict[str, Any]:
            assert coordinate_space == "active_window"
            return {
                "rendered": True,
                "coordinate_space": {
                    "name": "active_window",
                    "width": 1000,
                    "height": 700,
                },
                "active_window": {"id": "42"},
            }

    gui = Gui()
    gui._visual_states["session"] = {
        "identity": ("42", "active_window", 1000, 700),
        "sample": b"\0" * 64 * 64 * 3,
    }
    result = gui.act(
        "session",
        {"action": "drag", "x": 100, "y": 200, "to_x": 500, "to_y": 205},
    )

    assert result["rendered"] is True
    assert result["visual_change"]["comparable"] is False
    assert gui.commands == [
        [
            "xdotool",
            "mousemove",
            "--sync",
            "150",
            "230",
            "mousedown",
            "1",
            "mousemove",
            "--sync",
            "--duration",
            "600",
            "550",
            "235",
            "mouseup",
            "1",
        ]
    ]


def test_gui_screen_coordinates_remain_absolute() -> None:
    class Gui(GuiAutomation):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[list[str]] = []

        def _run(self, command: list[str]) -> str:
            self.commands.append(command)
            return ""

        def _desktop_state(self) -> tuple[int, int, dict[str, Any]]:
            return (
                1920,
                1080,
                {
                    "id": "42",
                    "title": "Browser",
                    "bounds": {"x": 50, "y": 30, "width": 1000, "height": 700},
                },
            )

        def _snapshot(self, coordinate_space: str = "active_window") -> dict[str, Any]:
            assert coordinate_space == "screen"
            return {
                "rendered": True,
                "coordinate_space": {
                    "name": "screen",
                    "width": 1920,
                    "height": 1080,
                },
                "active_window": {"id": "42"},
            }

    gui = Gui()
    gui._visual_states["session"] = {
        "identity": ("42", "screen", 1920, 1080),
        "sample": b"\0" * 64 * 64 * 3,
    }
    gui.act(
        "session",
        {
            "action": "click",
            "x": 100,
            "y": 200,
            "coordinate_space": "screen",
            "wait_ms": 0,
        },
    )

    assert gui.commands == [
        ["xdotool", "mousemove", "--sync", "100", "200", "click", "1"]
    ]


def test_gui_omitted_space_reuses_the_last_screen_snapshot() -> None:
    class Gui(GuiAutomation):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[list[str]] = []

        def _run(self, command: list[str]) -> str:
            self.commands.append(command)
            return ""

        def _desktop_state(self) -> tuple[int, int, dict[str, Any]]:
            return (
                1920,
                1080,
                {
                    "id": "42",
                    "title": "Browser",
                    "bounds": {"x": 50, "y": 30, "width": 1000, "height": 700},
                },
            )

        def _snapshot(self, coordinate_space: str = "active_window") -> dict[str, Any]:
            return {
                "rendered": True,
                "coordinate_space": {
                    "name": coordinate_space,
                    "width": 1920,
                    "height": 1080,
                },
                "active_window": {"id": "42"},
                "visual_fingerprint": self._visual_fingerprint(
                    Image.new("RGB", (32, 32), "white")
                ),
            }

    gui = Gui()
    gui.act("session", {"action": "snapshot", "coordinate_space": "screen"})
    gui.act("session", {"action": "snapshot", "coordinate_space": "screen"})
    gui.act("session", {"action": "click", "x": 380, "y": 62, "wait_ms": 0})

    assert [
        "xdotool",
        "mousemove",
        "--sync",
        "380",
        "62",
        "click",
        "1",
    ] in gui.commands


def test_gui_rejects_pointer_action_before_any_gui_snapshot() -> None:
    gui = GuiAutomation()

    with pytest.raises(GuiAutomationError, match="fresh GUI snapshot"):
        gui.act("session", {"action": "click", "x": 100, "y": 100})


def test_gui_first_screen_request_is_scoped_to_the_active_window() -> None:
    class Gui(GuiAutomation):
        def _snapshot(self, coordinate_space: str = "active_window") -> dict[str, Any]:
            assert coordinate_space == "active_window"
            return {
                "rendered": True,
                "coordinate_space": {
                    "name": "active_window",
                    "width": 1000,
                    "height": 700,
                },
                "active_window": {"id": "42"},
                "visual_fingerprint": self._visual_fingerprint(
                    Image.new("RGB", (32, 32), "white")
                ),
            }

    result = Gui().act(
        "session", {"action": "snapshot", "coordinate_space": "screen"}
    )

    assert result["coordinate_space"]["name"] == "active_window"
    assert result["scope_adjustment"] == "initial_snapshot_scoped_to_active_window"


def test_gui_rejects_switching_coordinate_frames_without_observing_it() -> None:
    gui = GuiAutomation()
    gui._visual_states["session"] = {
        "identity": ("42", "active_window", 1000, 700),
        "sample": b"\0" * 64 * 64 * 3,
    }

    with pytest.raises(GuiAutomationError, match="differs from the observed image"):
        gui.act(
            "session",
            {
                "action": "click",
                "coordinate_space": "screen",
                "x": 100,
                "y": 100,
            },
        )


def test_gui_rejects_stale_active_window_coordinates_after_focus_changes() -> None:
    gui = GuiAutomation()
    gui._visual_states["session"] = {
        "identity": ("seen-window", "active_window", 1000, 700),
        "sample": b"\0" * 1024,
    }
    gui._desktop_state = lambda: (  # type: ignore[method-assign]
        1920,
        1080,
        {
            "id": "different-window",
            "title": "Terminal",
            "bounds": {"x": 0, "y": 0, "width": 900, "height": 600},
        },
    )

    with pytest.raises(GuiAutomationError, match="active window changed"):
        gui.act("session", {"action": "click", "x": 100, "y": 100})


def test_gui_snapshot_crops_to_active_window_and_reports_its_frame(
    monkeypatch: Any,
) -> None:
    class Gui(GuiAutomation):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[list[str]] = []

        def _desktop_state(self) -> tuple[int, int, dict[str, Any]]:
            return (
                1920,
                1080,
                {
                    "id": "42",
                    "title": "Browser",
                    "bounds": {
                        "x": 54,
                        "y": 37,
                        "width": 1042,
                        "height": 800,
                        "screen": 0,
                    },
                },
            )

        def _run(self, command: list[str]) -> str:
            self.commands.append(command)
            if command[0] == "gnome-screenshot":
                Image.new("RGB", (1920, 1080), "white").save(command[-1])
            return ""

    monkeypatch.setattr("portal.gui.shutil.which", lambda _name: "/usr/bin/tool")
    gui = Gui()
    result = gui._snapshot()

    assert gui.commands == [["gnome-screenshot", "-f", gui.commands[0][-1]]]
    assert result["coordinate_space"] == {
        "name": "active_window",
        "origin_x": 54,
        "origin_y": 37,
        "width": 1042,
        "height": 800,
        "coordinate_units": ["pixels", "normalized_1000"],
    }
    assert result["active_window"]["bounds"]["x"] == 54
    with Image.open(io.BytesIO(base64.b64decode(result["screenshot"]["data"]))) as image:
        assert image.size == (1042, 800)


def test_gui_reports_when_an_action_does_not_materially_change_the_frame() -> None:
    gui = GuiAutomation()
    frame = Image.new("RGB", (100, 80), "white")

    def result(image: Image.Image) -> dict[str, Any]:
        return {
            "coordinate_space": {
                "name": "active_window",
                "width": image.width,
                "height": image.height,
            },
            "active_window": {"id": "42"},
            "visual_fingerprint": gui._visual_fingerprint(image),
        }

    first = gui._annotate_visual_change("session", result(frame))
    unchanged = gui._annotate_visual_change("session", result(frame.copy()))
    changed_frame = frame.copy()
    for x in range(50):
        for y in range(40):
            changed_frame.putpixel((x, y), (0, 0, 0))
    changed = gui._annotate_visual_change("session", result(changed_frame))

    assert first["visual_change"] == {
        "comparable": False,
        "materially_changed": None,
    }
    assert unchanged["visual_change"] == {
        "comparable": True,
        "changed_sample_count": 0,
        "changed_sample_fraction": 0.0,
        "materially_changed": False,
    }
    assert changed["visual_change"]["materially_changed"] is True


def test_existing_browser_executor_is_not_readmitted_at_the_soft_floor() -> None:
    class ExistingBrowser:
        calls = 0

        def act(self, _session_id: str, _arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            return {"rendered": True, "title": "existing"}

        def clear(self, _session_id: str) -> None:
            pass

    browser = ExistingBrowser()
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        browser_automation=browser,
        memory_governor=_memory_governor(2.5),
    )

    result = harness.execute("session", "browser_interact", {"action": "snapshot"})

    assert result["title"] == "existing"
    assert browser.calls == 1


def test_browser_close_remains_available_below_the_hard_memory_floor() -> None:
    class ExistingBrowser:
        calls = 0

        def act(self, _session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            assert arguments == {"action": "close"}
            return {"closed": True, "rendered": False}

        def clear(self, _session_id: str) -> None:
            pass

    browser = ExistingBrowser()
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        browser_automation=browser,
        memory_governor=_memory_governor(0.5),
    )

    result = harness.execute("session", "browser_interact", {"action": "close"})

    assert result == {"closed": True, "rendered": False}
    assert browser.calls == 1


def test_browser_close_reaps_lost_runtime_profile() -> None:
    class Store(BrowserAutomationStore):
        reaped = 0

        def _reap_orphan_browsers(self) -> None:
            self.reaped += 1

    store = Store()

    result = store.act("missing-session", {"action": "close"})

    assert result == {"closed": True, "rendered": False}
    assert store.reaped == 1


def test_browser_admit_refuses_a_second_live_window() -> None:
    store = BrowserAutomationStore()
    store._sessions["a"] = _FakeSession()  # type: ignore[assignment]
    try:
        store._admit_single_window()
    except BrowserAutomationError as error:
        assert "single rendered browser" in str(error)
    else:
        raise AssertionError("expected a BrowserAutomationError")


def test_browser_emergency_closes_sessions_when_unified_memory_collapses(
) -> None:
    class Store(BrowserAutomationStore):
        def __init__(self) -> None:
            super().__init__(memory_governor=_memory_governor(0.6))
            self.terminated: list[str] = []

        def _terminate(self, session: Any) -> None:
            self.terminated.append("terminated")

    store = Store()
    store._sessions["a"] = _FakeSession()  # type: ignore[assignment]
    store._expire_locked()
    assert store.terminated == ["terminated"]


def test_initial_tool_contract_stays_tiny() -> None:
    serialized = json.dumps(DISCOVERY_TOOLS, separators=(",", ":"))

    assert {item["function"]["name"] for item in DISCOVERY_TOOLS} == {"tool_search"}
    assert len(serialized) < 600


def test_browser_and_gui_tools_expose_drag_recovery_actions() -> None:
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in SAFE_TOOLS
    }

    assert "drag" in schemas["browser_interact"]["properties"]["action"]["enum"]
    assert "visual_click" in schemas["browser_interact"]["properties"]["action"]["enum"]
    assert {"set_value", "select", "upload"} <= set(
        schemas["browser_interact"]["properties"]["action"]["enum"]
    )
    assert {"value", "values", "paths"} <= set(
        schemas["browser_interact"]["properties"]
    )
    assert schemas["browser_interact"]["properties"]["coordinate_unit"]["enum"] == [
        "normalized_1000"
    ]
    assert schemas["browser_interact"]["properties"]["target"]["maxLength"] == 240
    assert {"delta_x", "delta_y"} <= set(
        schemas["browser_interact"]["properties"]
    )
    assert "drag" in schemas["gui_interact"]["properties"]["action"]["enum"]
    assert {"x", "y", "to_x", "to_y"} <= set(
        schemas["gui_interact"]["properties"]
    )
    assert schemas["gui_interact"]["properties"]["coordinate_space"]["enum"] == [
        "active_window",
        "screen",
    ]
    assert schemas["gui_interact"]["properties"]["coordinate_unit"]["enum"] == [
        "pixels",
        "normalized_1000",
    ]
    assert schemas["web_fetch"]["properties"]["format"]["enum"] == [
        "text",
        "raw_html",
    ]
    assert schemas["web_crawl"]["properties"]["extract"]["enum"] == [
        "text",
        "links",
        "all",
    ]


def test_portal_defers_perceptual_laya_routing_until_after_comprehension() -> None:
    requests: list[dict[str, Any]] = []

    class Plane:
        shadow_mode = True
        config = {"tool_families": {"browser": ["browser_interact"]}}
        calls = 0

        def evaluate(self, **_kwargs: Any) -> None:
            self.calls += 1

        def health(self) -> dict[str, Any]:
            return {"ready": True, "shadow_mode": True}

    plane = Plane()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ready."}},
        )

    app = create_app(
        _config(),
        httpx.Client(transport=httpx.MockTransport(handler)),
        decision_plane=plane,  # type: ignore[arg-type]
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {
                    "role": "user",
                    "content": "The attached audio contains the current request.",
                    "audios": [
                        {
                            "mime_type": "audio/wav",
                            "encoding": "base64",
                            "data": base64.b64encode(b"not-decoded-by-portal").decode(),
                        }
                    ],
                }
            ],
            portal_auto_tools=True,
            portal_background_bridge=True,
        ),
    )

    assert response.status_code == 200
    assert plane.calls == 0
    names = {
        item["function"]["name"] for item in requests[0]["tools"]
    }
    assert "browser_interact" in names
    assert "shell" in names
    assert "background_task" in names
    assert "request_camera_view" not in names
    assert requests[0]["omni"]["tool_routing"] == "relevant"


def test_text_request_gets_relevant_concrete_schema_without_laya_fast_path() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ready."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[{"role": "user", "content": "Open the web browser."}],
            portal_auto_tools=True,
            portal_background_bridge=True,
        ),
    )

    assert response.status_code == 200
    names = {item["function"]["name"] for item in requests[0]["tools"]}
    assert "browser_interact" in names
    assert "background_task" in names
    assert "tool_search" in names


def test_virtual_context_shadow_indexes_and_traces_without_replacing_live_prompt(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "cobalt-771"}},
        )

    app = create_app(
        _config(
            virtual_context_mode="shadow",
            virtual_context_root=tmp_path / "virtual-context",
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {"role": "user", "content": "The actuator code is cobalt-771."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "What was the actuator code?"},
            ]
        ),
    )

    assert response.status_code == 200
    assert [message["role"] for message in requests[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    virtual = response.json["portal"]["virtual_context"]
    assert virtual["mode"] == "shadow"
    assert virtual["documents"] == 3
    assert virtual["working_tokens"] <= 16_384
    assert virtual["evidence_chunk_ids"]
    assert any(event["operation"] == "PAGE_IN" for event in virtual["trace"])


def test_live_audio_bypasses_query_retrieval_until_speech_is_transcribed(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "I heard you."}},
        )

    virtual_root = tmp_path / "virtual-context"
    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=virtual_root,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    body = _request(
        messages=[
            {"role": "user", "content": "We were discussing weather."},
            {"role": "assistant", "content": "It may rain."},
            {
                "role": "user",
                "content": "The attached audio contains the latest spoken turn.",
                "audios": [
                    {
                        "mime_type": "audio/wav",
                        "encoding": "base64",
                        "data": base64.b64encode(b"current-speech").decode(),
                    }
                ],
            },
        ],
        omni={
            "schema": "robit.ollama.omni-adapter.v1",
            "task": "chat",
            "require_speech": True,
        },
    )

    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 200
    # Policy adds one system message, but active virtual memory must not replace
    # the bounded dialogue with a retrieval pack derived from the placeholder.
    assert requests[0]["messages"][-len(body["messages"]) :] == body["messages"]
    assert not list(virtual_root.glob("*.sqlite3"))
    assert response.json["portal"]["virtual_context"] == {
        "mode": "bypass",
        "reason": "untranscribed_audio_input",
    }


def test_streaming_live_audio_bypasses_query_retrieval_until_transcribed(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []
    wire = (
        b'{"type":"delta","message":{"content":"I heard you."}}\n'
        b'{"type":"final","response":{"message":{"content":"I heard you."}}}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    virtual_root = tmp_path / "virtual-context"
    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=virtual_root,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    body = _request(
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "The attached audio contains the latest spoken turn.",
                "audios": [
                    {
                        "mime_type": "audio/wav",
                        "encoding": "base64",
                        "data": base64.b64encode(b"current-speech").decode(),
                    }
                ],
            }
        ],
        omni={
            "schema": "robit.ollama.omni-adapter.v1",
            "task": "chat",
            "require_speech": True,
        },
    )

    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert requests[0]["messages"][-len(body["messages"]) :] == body["messages"]
    assert not list(virtual_root.glob("*.sqlite3"))
    assert events[-1]["response"]["portal"]["virtual_context"] == {
        "mode": "bypass",
        "reason": "untranscribed_audio_input",
    }


def test_virtual_context_active_repacks_each_tool_followup(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_portal_capabilities",
                                    "arguments": {},
                                },
                            }
                        ],
                    }
                },
            )
        assert [message["role"] for message in body["messages"]] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assert body["messages"][-2]["tool_calls"][0]["function"]["name"] == (
            "get_portal_capabilities"
        )
        assert body["messages"][-1]["tool_name"] == "get_portal_capabilities"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Capability list ready."}},
        )

    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=tmp_path / "virtual-context",
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = _request(portal_auto_tools=True)
    request["messages"][-1]["content"] = "What can you do?"
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=request,
    )

    assert response.status_code == 200
    assert len(requests) == 2
    assert len(requests[1]["messages"]) == 4
    assert requests[1]["messages"][0]["role"] == "system"
    assert requests[1]["messages"][1]["role"] == "user"
    assert requests[1]["messages"][2]["role"] == "assistant"
    assert requests[1]["messages"][3]["role"] == "tool"
    assert "get_portal_capabilities" in requests[1]["messages"][3]["content"]
    assert "What can you do?" in requests[1]["messages"][1]["content"]
    assert response.json["portal"]["virtual_context"]["working_tokens"] <= 16_384


def test_virtual_context_active_overflow_preserves_native_request(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Still responsive."}},
        )

    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=tmp_path / "virtual-context",
            virtual_context_physical_tokens=4096,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {"role": "system", "content": "pinned contract " * 1200},
                {"role": "user", "content": "Answer this foreground turn."},
            ]
        ),
    )

    assert response.status_code == 200
    assert requests[0]["messages"][-1]["content"] == "Answer this foreground turn."
    virtual = response.json["portal"]["virtual_context"]
    assert virtual["mode"] == "active"
    assert virtual["working_set_fallback"] == "native_bounded_prompt"
    assert "system contract and current query" in virtual["overflow"]


def test_4k_background_action_stays_inside_active_virtual_context(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "", "tool_calls": []}},
        )

    task = {
        "objective": "Build a researched field-service application in place. " * 31,
        "completion_criteria": "The app exists, tests pass, and the result is verified. " * 18,
        "actions": [
            {
                "call_id": "current-file",
                "tool": "workspace_file",
                "arguments": json.dumps(
                    {"action": "read", "path": "/tmp/app/src/app/page.tsx"}
                ),
                "outcome": json.dumps(
                    {"content": "export default function Page() { return null }"}
                ),
                "ok": True,
            }
        ],
    }
    schemas = _background_tool_contract(
        ["browser_interact"],
        recovery_required=False,
        phase_boundary=False,
        expand_available=True,
        can_checkpoint=True,
        include_discovery=True,
        resident_context_tokens=4_096,
    )
    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=tmp_path / "virtual-context",
            virtual_context_physical_tokens=4096,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {
                    "role": "system",
                    "content": _task_system_prompt(
                        task,
                        resident_context_tokens=4_096,
                        expand_available=True,
                    ),
                },
                {"role": "user", "content": "Continue from the retained frontier."},
            ],
            tools=schemas,
            tool_choice="required",
            think=False,
            portal_background_worker=True,
            portal_virtual_query=_task_virtual_query(
                task, resident_context_tokens=4_096
            ),
        ),
    )

    assert response.status_code == 200
    virtual = response.json["portal"]["virtual_context"]
    assert virtual["mode"] == "active"
    assert "working_set_fallback" not in virtual
    assert virtual["working_tokens"] <= 4096
    assert len(requests) == 1
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["messages"][-1]["role"] == "user"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == [
        "browser_interact",
        "tool_search",
        "task_checkpoint",
    ]


def test_background_overflow_fails_visible_instead_of_using_native_fifo(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "unexpected"}},
        )

    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=tmp_path / "virtual-context",
            virtual_context_physical_tokens=4096,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {"role": "system", "content": "oversized pinned contract " * 1600},
                {"role": "user", "content": "Continue the durable task."},
            ],
            portal_background_worker=True,
            portal_virtual_query="continue durable task",
        ),
    )

    assert response.status_code == 502
    assert "background virtual working set overflow" in response.json["error"]
    assert requests == []


def test_synthesis_bypasses_virtual_conversation_context(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "That task hit a blocker.",
                }
            },
        )

    virtual_root = tmp_path / "virtual-context"
    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=virtual_root,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {"role": "user", "content": "That task hit a blocker."},
            ],
            omni={
                "schema": "robit.ollama.omni-adapter.v1",
                "task": "synthesize",
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
            portal_auto_tools=True,
        ),
    )

    assert response.status_code == 200
    assert requests[0]["messages"] == [
        {"role": "user", "content": "That task hit a blocker."}
    ]
    assert "tools" not in requests[0]
    assert not list(virtual_root.glob("*.sqlite3"))
    assert response.json["portal"]["virtual_context"] == {
        "mode": "bypass",
        "reason": "synthesis_only",
    }


def test_streaming_synthesis_bypasses_virtual_conversation_context(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []
    wire = (
        b'{"type":"delta","message":{"content":"That task hit a blocker."}}\n'
        b'{"type":"final","response":{"message":{"content":"That task hit a blocker."}}}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    virtual_root = tmp_path / "virtual-context"
    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=virtual_root,
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            stream=True,
            messages=[
                {"role": "user", "content": "That task hit a blocker."},
            ],
            omni={
                "schema": "robit.ollama.omni-adapter.v1",
                "task": "synthesize",
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
            portal_auto_tools=True,
        ),
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert requests[0]["messages"] == [
        {"role": "user", "content": "That task hit a blocker."}
    ]
    assert "tools" not in requests[0]
    assert not list(virtual_root.glob("*.sqlite3"))
    assert events[-1]["response"]["portal"]["virtual_context"] == {
        "mode": "bypass",
        "reason": "synthesis_only",
    }


def test_social_text_does_not_gain_an_unrelated_leaf_tool() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Good morning."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[{"role": "user", "content": "Good morning."}],
            portal_auto_tools=True,
            portal_background_bridge=True,
        ),
    )

    assert response.status_code == 200
    assert {
        item["function"]["name"] for item in requests[0]["tools"]
    } == {"tool_search", "background_task"}


def test_model_facing_subagent_handoff_is_reference_only() -> None:
    schema = next(
        item for item in SAFE_TOOLS if item["function"]["name"] == "subagent_delegate"
    )["function"]["parameters"]

    assert "context" not in schema["properties"]
    assert schema["properties"]["context_source"]["enum"] == [
        "none",
        "current_user_message",
        "latest_non_discovery_tool_result",
    ]
    assert set(schema["required"]) == {"objective", "context_source"}


def _config(**overrides) -> PortalConfig:
    values = {
        "adapter_url": "http://adapter/api/chat",
        "adapter_health_url": "http://adapter/healthz",
        "comprehension_health_url": "http://comprehension/health",
        "tts_health_url": "http://tts/healthz",
        "ollama_health_url": "http://ollama/api/tags",
        "model": DEFAULT_MODEL,
        "access_token": TOKEN,
        "timeout_s": 30,
        "max_body_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return PortalConfig(**values)


def _request(**overrides):
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
        "response_modalities": ["text"],
        "speech_mode": "never",
        "think": True,
        "stream": False,
    }
    body.update(overrides)
    return body


def test_portal_index_has_mobile_security_headers_and_no_token() -> None:
    app = create_app(
        _config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))
    )
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Omni Chat" in response.data
    assert b"ROBIT" not in response.data
    assert b'id="waveform-canvas"' in response.data
    assert b'id="speak-toggle"' in response.data
    assert b'id="think-toggle"' in response.data
    assert b'id="call-button"' in response.data
    assert b'id="camera-button"' in response.data
    assert b'id="camera-video"' in response.data
    assert b'id="share-button"' in response.data
    assert b'id="share-dialog"' in response.data
    assert b'id="share-qr"' in response.data
    assert b'id="voice-button"' in response.data
    assert b'id="voice-clone-enabled"' in response.data
    assert b'id="voice-clone-toggle"' in response.data
    assert b'id="voice-preset-toggle"' in response.data
    assert b'id="voice-preset-options"' in response.data
    assert b'id="voice-reference-input"' in response.data
    assert b'id="active-user-count"' in response.data
    assert b'class="audio-observation-output"' in response.data
    assert b'class="tool-output"' in response.data
    assert b"Sounds heard" in response.data
    assert b"application/pdf" in response.data
    assert b"image/gif" in response.data
    assert b"multiple" in response.data
    assert response.data.index(b"/assets/call_vad.js") < response.data.index(
        b"/assets/call_queue.js"
    )
    assert response.data.index(b"/assets/call_queue.js") < response.data.index(
        b"/assets/call_playback.js"
    )
    assert response.data.index(b"/assets/call_playback.js") < response.data.index(
        b"/assets/session_cache.js"
    )
    assert response.data.index(b"/assets/session_cache.js") < response.data.index(
        b"/assets/qr_code.js"
    )
    assert response.data.index(b"/assets/qr_code.js") < response.data.index(
        b"/assets/portal.js"
    )
    cache_scope = re.search(rb'data-session-scope="([a-f0-9]{64})"', response.data)
    assert cache_scope is not None
    assert b'href="/assets/favicon.svg"' in response.data
    assert response.data.index(b'id="camera-button"') < response.data.index(b'id="call-button"')
    assert b'aria-pressed="false"' in response.data
    assert b"maximum-scale=1" in response.data
    assert b"user-scalable=no" in response.data
    assert TOKEN.encode() not in response.data
    assert "microphone=(self)" in response.headers["Permissions-Policy"]
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "connect-src 'self' https://ipwho.is" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    cookie = response.headers["Set-Cookie"]
    assert "omni_portal_session=" in cookie
    assert "Secure" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie

    asset = client.get("/assets/portal.js")
    assert asset.status_code == 200
    assert asset.headers["Cache-Control"] == "no-store"
    qr_asset = client.get("/assets/qr_code.js")
    assert qr_asset.status_code == 200
    assert qr_asset.headers["Cache-Control"] == "no-store"


def test_portal_assets_include_markdown_call_flow_and_neutral_composer() -> None:
    javascript = Path("portal/static/portal.js").read_text()
    css = Path("portal/static/portal.css").read_text()
    context = context_catalog()

    assert "function renderMarkdown" in javascript
    assert "function renderToolTrace" in javascript
    assert "function mergeToolTrace" in javascript
    assert "function appendToolJsonRows" in javascript
    assert "MAX_TOOL_TRACE_ITEMS" not in javascript
    assert ".tool-json-row" in css
    assert ".tool-json-branch" in css
    assert "portal_auto_tools: toolUseEnabled()" in javascript
    assert "CLIENT_LOCATION_ENDPOINT = \"https://ipwho.is/\"" in javascript
    assert "CLIENT_LOCATION_TIMEOUT_MS = 6000" in javascript
    assert "clientLocationRetryAt" in javascript
    assert "function sanitizeClientLocation" in javascript
    assert "portal_client_location" in javascript
    assert 'document.getElementById("tool-toggle")' in javascript
    assert "function portalShareUrl" in javascript
    assert "function renderShareQr" in javascript
    assert "fragment.set(\"access\", state.token)" in javascript
    assert "const preserveStreamedAssistant" in javascript
    assert "const recordTurnHistory" in javascript
    assert "if (retention.interrupted) assistant.node.classList.add(\"interrupted\")" in javascript
    assert "function markdownTableSpec" in javascript
    assert "function renderMarkdownTable" in javascript
    assert ".markdown-table-wrap" in css
    assert ".markdown-table .align-right" in css
    assert "function startCall" in javascript
    assert "function submitCallUtterance" in javascript
    assert "function enqueueCallUtterance" in javascript
    assert "function flushPendingCallUtterances" in javascript
    assert "function rememberCallAudioContext" in javascript
    assert "callQueue.classifyObservation" in javascript
    assert "require_speech: true" in javascript
    assert 'content: frame ? "Camera audio context" : "Audio context"' in javascript
    assert '"empty_assistant_response"' in javascript
    assert '"speech_addressed_elsewhere"' in javascript
    assert 'Call · observation retained · listening' in javascript
    assert "CALL_PENDING_MAX_SECONDS = 45" in javascript
    assert "A submitted turn is immutable" in javascript
    assert "function abortActiveCallTurns" not in javascript
    assert "preserveUnanswered" not in javascript
    assert "inputRequeued" not in javascript
    assert "responseStarted" not in javascript
    assert "|| call.inflight" in javascript
    assert "function supersedeCallAudio" in javascript
    assert "callPlayback.canStart(call, turn)" in javascript
    assert "supersedeCallAudio(call, call.nextSequence)" in javascript
    assert "if (requestSequence !== state.requestSequence) return" in javascript
    assert "function startCameraCapture" in javascript
    assert "function stopCameraCapture" in javascript
    assert "function streamChat" in javascript
    assert "function streamChatAttempt" in javascript
    assert "function retryableClientStreamError" in javascript
    assert "CLIENT_STREAM_RETRY_SAFE_TOOLS" in javascript
    assert 'reason: "browser_network_retry"' in javascript
    assert "replayUnsafe" in javascript
    assert javascript.count('event.type === "reset"') == 2
    assert "Retrying interrupted model stream" in javascript
    assert "BARGE_VAD_OPTIONS" in javascript
    assert "think: wantsThinking" in javascript
    assert "think: showThinking" in javascript
    assert "built.wantsThinking" in javascript
    assert "num_predict" not in javascript
    assert "max_frames: 24" in javascript
    assert "function voicePayload" in javascript
    assert "function startVoiceReferenceRecording" in javascript
    assert 'elements.prompt.value = ""' in javascript
    assert ".composer textarea:focus" in css
    assert "box-shadow: none" in css
    assert "user-select: none" in css
    assert "-webkit-touch-callout: none" in css
    assert "event.transcript" in javascript
    assert "event.audio_observation" in javascript
    assert "(data.adapter || {}).input_transcript" in javascript
    assert "(data.adapter || {}).audio_observation" in javascript
    assert '(data.adapter || {}).observation || "Voice message"' not in javascript
    assert 'task = "transcribe"' not in javascript
    assert "replaceUserWithTranscript: !typed && audioOnly" in javascript
    assert "function audioEvidenceHistory" in javascript
    assert "audioObservation: inputTranscript ? inputAudioObservation" in javascript
    assert "soundOnly: !inputTranscript && Boolean(inputAudioObservation)" in javascript
    assert "parts.push(`[Sounds heard: ${sounds}]`)" in javascript
    assert 'audioObservation: String(record.audioObservation || "")' in javascript
    assert "soundOnly: Boolean(record.soundOnly)" in javascript
    assert "content: built.display" in javascript
    assert "mediaSummary" not in javascript
    assert "use both its speech and non-speech sounds as" in javascript
    assert "assistant.node.hidden = true" in javascript
    assert ".message[hidden]" in css
    assert 'addFile(file, "video", "camera")' in javascript
    assert 'value.source !== "camera"' in javascript
    assert "function loopingVideo" in javascript
    assert "video.loop = true" in javascript
    assert "media: sentMedia" in javascript
    assert ".message-video-preview" in css
    assert "const hasMedia = state.attachments.length > 0" in javascript
    assert "MEDIA_CONVERSATION_SYSTEM_PROMPT" in javascript
    assert "LIVE_CALL_SYSTEM_PROMPT" in javascript
    assert "Do not echo" in context["prompts"]["live_call_system"]
    assert (
        "Only the media attached to the latest"
        in context["prompts"]["media_conversation_system"]
    )
    assert 'document.getElementById("omni-context")' in javascript
    assert 'task = "describe"' not in javascript
    assert "if (built.hasMedia) state.history = []" not in javascript
    assert "if (item.frame) state.history = []" not in javascript
    assert '{ role: "system", content: LIVE_CALL_SYSTEM_PROMPT }' in javascript
    assert '{ role: "system", content: MEDIA_CONVERSATION_SYSTEM_PROMPT }' in javascript
    assert "function scrollConversationToBottom" in javascript
    assert "function handleConversationScroll" in javascript
    assert "function resumeConversationAutoFollow" in javascript
    assert "state.conversationScrollGesture && movedUp" in javascript
    assert "smooth: false, force: true" in javascript
    assert 'elements.scrollLatest.addEventListener("click"' in javascript
    assert ".scroll-latest-button[hidden]" in css
    assert 'behavior: "smooth"' in javascript
    assert "elements.conversation.scrollTop = elements.conversation.scrollHeight" in javascript
    assert "new window.ResizeObserver" in javascript
    assert 'composer: document.querySelector(".composer")' in javascript
    assert "layoutResizeObserver.observe(node)" in javascript
    assert "function copyAssistantMarkdown" in javascript
    assert "navigator.clipboard.writeText(markdown)" in javascript
    assert 'String(record.content || "")' in javascript
    assert ".message-copy-button" in css
    assert "function generationMetricsFromResponse" in javascript
    assert "eval_count" in javascript
    assert "eval_duration" in javascript
    assert "1_000_000_000" in javascript
    assert "new Intl.DateTimeFormat" in javascript
    assert ".message-generation-metrics" in css
    assert "Streamed reply · replay with the player" not in javascript
    assert ".audio-note" not in css
    assert "scroll-behavior: smooth" not in css
    assert "controllers: new Set()" in javascript
    assert "call.controllers.add(turn.controller)" in javascript
    assert "for (const controller of call.controllers) controller.abort()" in javascript
    assert "pendingUtterance" not in javascript
    assert "call.busy" not in javascript
    assert "callVad.processFrame" in javascript
    assert "setVadActive(call, true)" in javascript
    assert ".waveform.calling.vad-active" in css
    assert "function refreshActivity" in javascript
    assert "function reportDiagnostic" in javascript
    assert "function clearSessionDiagnostics" in javascript
    assert "const PCM_INITIAL_BUFFER_SECONDS = 0.08" in javascript
    assert "const PCM_CROSSFADE_SECONDS = 0.003" in javascript
    assert "nextTime: context.currentTime + PCM_INITIAL_BUFFER_SECONDS" in javascript
    assert "controller.context.currentTime + PCM_RESCHEDULE_FLOOR_SECONDS" in javascript
    assert "controller.context.createGain()" in javascript
    assert "controller.nextTime - crossfade >= playbackFloor" in javascript
    assert "linearRampToValueAtTime" in javascript
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert ".audio-observation-output" in css
    assert ".message.user.sound-only .message-content" in css
    assert "opacity: .5" in css
    assert (
        'const documents = state.attachments.filter(item => item.kind === "document")' in javascript
    )
    assert "message.documents" in javascript
    assert 'item.mime === "image/gif"' in javascript
    assert "function restoreBrowserSession" in javascript
    assert "function persistBrowserSessionOnLeave" in javascript
    assert "function clearBrowserSessionCache" in javascript
    assert "window.OmniSessionCache.clear(state.cacheScope)" in javascript
    assert "state.cacheDeleted = true" in javascript
    assert "!state.cacheDeleted" in javascript
    assert "robit.omni.browser-session.v1" in javascript
    assert 'transientComposerStatus("Press and hold to record voice clip")' in javascript
    assert 'throw new Error("The microphone clip contained no samples")' not in javascript


def test_browser_session_cache_harness_restores_expires_and_clears() -> None:
    completed = subprocess.run(
        ["node", "portal/session_cache_harness.mjs"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result == {"status": "passed", "ttl_ms": 300_000}


def test_mock_call_vad_harness_rejects_noise_and_accepts_confirmed_events() -> None:
    completed = subprocess.run(
        ["node", "portal/vad_harness.mjs"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["status"] == "passed"
    assert result["remote_requests"] == {
        "calibrated_quiet": 0,
        "transient_click": 0,
        "elevated_room_noise": 0,
        "sustained_speech": 1,
        "sustained_alarm": 1,
        "quiet_speech": 1,
        "immediate_speech": 1,
        "continued_speech_segments": 2,
    }


def test_mock_call_queue_consolidates_segments_and_bounds_pending_audio() -> None:
    completed = subprocess.run(
        ["node", "portal/call_queue_harness.mjs"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "status": "passed",
        "consolidated_segments": 3,
        "consolidated_samples": 1100,
        "bounded_samples": 10,
        "single_flight_contract": "one active inference plus one bounded pending turn",
        "submitted_samples_requeued": 0,
        "next_turn_segments": 2,
        "sound_only_aborts_reply": True,
        "bounded_audio_contexts": 6,
    }


def test_mock_call_playback_harness_rejects_stale_audio() -> None:
    completed = subprocess.run(
        ["node", "portal/playback_harness.mjs"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "status": "passed",
        "stale_pending_audio_suppressed": True,
        "active_playback_interrupted": True,
        "newest_turn_owns_playback": True,
        "settled_transcript_retained": True,
        "partial_transcript_retained_and_marked": True,
    }


def test_portal_api_requires_bearer_token() -> None:
    app = create_app(
        _config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))
    )
    client = app.test_client()

    assert client.get("/api/status").status_code == 401
    assert client.get("/api/activity").status_code == 401
    assert client.get("/api/diagnostics").status_code == 401
    assert client.post("/api/chat", json=_request()).status_code == 401


def test_portal_status_probes_all_internal_stages() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, json={"ok": True})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    assert response.json["ok"] is True
    assert set(seen) == {"adapter", "comprehension", "tts", "ollama"}
    assert response.json["model"] == DEFAULT_MODEL
    assert response.json["voice_profile"]["clone_mode"] == "speaker_embedding"
    assert response.json["audio_understanding"] == {
        "speech_transcription": True,
        "environmental_sound_analysis": True,
        "evidence_field": "adapter.audio_observation",
    }
    assert response.json["documents"]["retrieval"] == ("session-isolated hashed lexical embeddings")
    assert {item["function"]["name"] for item in response.json["safe_tools"]} == {
        item["function"]["name"] for item in SAFE_TOOLS
    }
    assert response.json["memory"]["scope"] == "browser_session"
    assert response.json["memory"]["entries"] == 0
    assert response.json["location"] == {
        "scope": "browser_session",
        "delivery": "get_user_location tool",
        "source": "browser HTTPS IP geolocation",
        "precision": "approximate",
        "raw_ip_retained": False,
        "available": False,
    }
    assert response.json["runtime_environment"] == {
        "delivery": "tool_only",
        "tool": "get_system_snapshot",
        "refreshed_each_call": True,
        "includes": [
            "date/time",
            "CPU/load",
            "RAM",
            "NVIDIA GPUs",
            "network counters",
        ],
        "excludes": [
            "hostnames",
            "addresses",
            "processes",
            "credentials",
            "session content",
        ],
    }
    assert response.json["tool_execution"] == {
        "automatic": True,
        "streaming": True,
        "client_opt_in": True,
        "default_enabled": False,
        "round_limit": None,
        "call_limit": None,
        "termination": [
            "model_final",
            "repeated_nonproductive_rounds",
            "request_timeout",
            "client_disconnect",
        ],
    }
    assert response.json["subagents"] == {
        "scope": "browser_session",
        "execution": "synchronous_isolated_text_only",
        "tools_available_to_helper": False,
        "tasks": 0,
    }
    assert response.json["web"] == {
        "discovery": "duckduckgo_html",
        "search_api": False,
        "index_scope": "browser_session",
        "indexed_pages": 0,
        "indexed_chars": 0,
    }
    assert response.json["voice_profile"]["client_reference_wav"] is True
    assert response.json["requests"] == {
        "users": 0,
        "inflight": 0,
        "active": 0,
        "queued": 0,
        "slots": 1,
        "limit": 4,
    }


def test_document_store_retrieves_per_session_and_clears() -> None:
    store = SessionDocumentStore(ttl_s=300)
    first = {
        "name": "alpha.txt",
        "mime_type": "text/plain",
        "encoding": "base64",
        "data": base64.b64encode(b"Orchid launch code is seven.").decode("ascii"),
    }
    second = {
        "name": "beta.txt",
        "mime_type": "text/plain",
        "encoding": "base64",
        "data": base64.b64encode(b"Marigold launch code is nine.").decode("ascii"),
    }

    first_context, _ = store.prepare("session-one", [first], "orchid code")
    second_context, _ = store.prepare("session-two", [second], "marigold code")

    assert "alpha.txt" in first_context and "seven" in first_context
    assert "beta.txt" not in first_context and "nine" not in first_context
    assert "beta.txt" in second_context and "nine" in second_context
    assert "alpha.txt" not in second_context and "seven" not in second_context
    store.clear("session-one")
    assert store.stats("session-one") == {"documents": 0, "chunks": 0, "chars": 0}
    assert store.stats("session-two")["documents"] == 1


def test_pdf_extraction_is_bounded_through_pdftotext(monkeypatch) -> None:
    def run(command, **_kwargs):
        Path(command[-1]).write_text("Extracted PDF evidence.")
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("portal.documents.subprocess.run", run)

    assert extract_document("brief.pdf", "application/pdf", b"%PDF-1.7\n") == (
        "Extracted PDF evidence."
    )


def test_portal_indexes_documents_and_sends_only_retrieved_text() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "message": {"role": "assistant", "content": "Found it."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    document = {
        "name": "notes.md",
        "mime_type": "text/markdown",
        "encoding": "base64",
        "data": base64.b64encode(b"The copper switch enables the archive.").decode("ascii"),
    }

    response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {
                    "role": "user",
                    "content": "Which switch enables the archive?",
                    "documents": [document],
                }
            ]
        ),
    )

    assert response.status_code == 200
    upstream_message = seen[0]["messages"][-1]
    assert "documents" not in upstream_message
    assert "portal_document_context" in upstream_message["content"]
    assert "notes.md" in upstream_message["content"]
    assert "copper switch" in upstream_message["content"]
    assert response.json["portal"]["documents_indexed"][0]["name"] == "notes.md"


def test_safe_tools_search_fetch_memory_and_block_private_networks() -> None:
    def web_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "html.duckduckgo.com":
            destination = "https://example.com/guide"
            redirect = f"https://duckduckgo.com/l/?uddg={quote_plus(destination)}"
            return httpx.Response(
                200,
                text=(
                    f'<a class="result__a" href="{redirect}">Example guide</a>'
                    f'<a class="result__snippet" href="{redirect}">Copper guide.</a>'
                ),
                headers={"content-type": "text/html"},
            )
        if request.url == "https://example.com/redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        assert request.url == "https://example.com/guide"
        return httpx.Response(
            200,
            text="<html><script>ignore()</script><h1>Verified guide</h1><p>Copper fact.</p></html>",
            headers={"content-type": "text/html"},
        )

    documents = SessionDocumentStore(ttl_s=300)
    harness = PortalToolHarness(
        documents,
        web_client=httpx.Client(transport=httpx.MockTransport(web_handler)),
        resolver=lambda _hostname: ["93.184.216.34"],
    )

    search = harness.execute("one", "web_search", {"query": "example guide"})
    assert search["provider"] == "duckduckgo"
    assert search["transport"] == "direct_html"
    assert search["provenance"]["authority"] == "discovery_only"
    assert search["provenance"]["citation_ready"] is False
    assert search["alternative_tools"] == ["web_fetch", "browser_interact"]
    assert search["results"][0]["url"] == "https://example.com/guide"
    fetched = harness.execute(
        "one", "web_fetch", {"url": search["results"][0]["url"]}
    )
    assert "Verified guide" in fetched["content"]
    assert "ignore()" not in fetched["content"]
    assert fetched["provenance"]["source_url"] == "https://example.com/guide"
    assert fetched["provenance"]["citation_ready"] is True
    assert "does not prove the user's location" in fetched["claim_limits"]
    recalled = harness.execute(
        "one", "web_search", {"query": "copper verified", "mode": "session"}
    )
    assert recalled["provider"] == "session_local_index"
    assert recalled["results"][0]["url"] == "https://example.com/guide"
    assert harness.execute(
        "two", "web_search", {"query": "copper", "mode": "session"}
    )["results"] == []
    blocked = harness.execute("one", "web_fetch", {"url": "http://127.0.0.1/admin"})
    assert "error" in blocked
    redirect = harness.execute(
        "one", "web_fetch", {"url": "https://example.com/redirect"}
    )
    assert "error" in redirect

    harness.execute(
        "one",
        "memory_write",
        {"topic": "research", "key": "copper", "value": "Copper fact."},
    )
    assert harness.execute("one", "memory_search", {"query": "copper"})["results"]
    harness.execute(
        "one",
        "memory_write",
        {"topic": "demo", "key": "launch_color", "value": "ultraviolet"},
    )
    assert harness.execute("one", "memory_search", {"query": "launch color"})[
        "results"
    ][0]["value"] == "ultraviolet"
    assert harness.execute("two", "memory_search", {"query": "copper"})["results"] == []
    harness.clear("one")
    assert harness.execute("one", "memory_search", {"query": "copper"})["results"] == []


def test_user_location_tool_is_sanitized_session_scoped_and_clearable() -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    supplied = harness.set_client_location(
        "one",
        {
            "ip": "203.0.113.42",
            "city": "Seattle",
            "region": "Washington",
            "region_code": "wa",
            "country": "United States",
            "country_code": "us",
            "latitude": 47.60621,
            "longitude": -122.33207,
            "timezone": {
                "id": "America/Los_Angeles",
                "abbreviation": "PDT",
                "utc_offset": "-07:00",
            },
            "connection": {"isp": "must not be retained"},
        },
    )

    result = harness.execute("one", "get_user_location", {})
    serialized = json.dumps(result)
    assert supplied == result
    assert result["city"] == "Seattle"
    assert result["region_code"] == "WA"
    assert result["latitude"] == 47.61
    assert result["longitude"] == -122.33
    assert result["raw_ip_included"] is False
    assert result["provenance"] == {
        "tool": "get_user_location",
        "source_type": "browser_ip_geolocation",
        "evidence_type": "tool_data_not_visual_perception",
        "authority": "approximate_network_area",
        "device_gps": False,
        "street_level": False,
    }
    assert "exact address" in result["claim_limits"]["unsupported"]
    assert "203.0.113.42" not in serialized
    assert "connection" not in result
    assert harness.execute("two", "get_user_location", {})["available"] is False
    harness.clear("one")
    assert harness.execute("one", "get_user_location", {})["available"] is False


def test_document_search_tool_is_session_isolated() -> None:
    documents = SessionDocumentStore(ttl_s=300)
    harness = PortalToolHarness(documents)
    envelope = {
        "name": "script.py",
        "mime_type": "text/x-python",
        "encoding": "base64",
        "data": base64.b64encode(b"def launch_orchid(): return 'amber'").decode(),
    }
    documents.prepare("one", [envelope], "launch orchid")

    own = harness.execute("one", "document_search", {"query": "orchid"})
    other = harness.execute("two", "document_search", {"query": "orchid"})

    assert own["results"][0]["document"] == "script.py"
    assert "launch_orchid" in own["results"][0]["content"]
    assert other["results"] == []


def test_tool_search_discovers_allowlisted_tools_only() -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    result = harness.execute("one", "tool_search", {"query": "OCR scanned PDF"})
    assert result["allowlisted_only"] is True
    assert "task_complete" not in result
    assert result["results"][0]["name"] == "ocr_pdf"
    assert result["suggested_tools"][0] == "ocr_pdf"
    assert "Invoke the smallest relevant one now" in result["next_action"]
    assert {item["name"] for item in result["results"]} <= {item["function"]["name"] for item in SAFE_TOOLS}

    assert discover_tool_names("delegate a fresh isolated critic subagent") == [
        "subagent_delegate"
    ]
    assert discover_tool_names("list completed subagent tasks") == ["subagent_list"]
    assert discover_tool_names("retrieve a subagent result") == ["subagent_result"]
    assert discover_tool_names("forget a delegated helper") == ["subagent_forget"]


def test_tool_discovery_keeps_web_lookup_and_physical_vision_distinct() -> None:
    news = discover_tool_names("search the web for current breaking news")
    camera = discover_tool_names("fresh camera view of what I am physically holding")

    assert news[0] == "web_search"
    assert "request_camera_view" not in news
    assert camera[0] == "request_camera_view"

    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    result = harness.execute(
        "one", "request_camera_view", {"mode": "motion"}
    )
    assert result["camera_capture_requested"] is True
    assert result["mode"] == "motion"


def test_capability_and_research_requests_do_not_collapse_to_weather() -> None:
    capabilities = discover_tool_names("what can you do and what abilities are available")
    research = discover_tool_names("research the newest robotics papers and cite sources")

    assert capabilities[0] == "get_portal_capabilities"
    assert "get_user_location" not in capabilities
    assert "web_search" in research
    assert "get_user_location" not in research
    assert discover_tool_names(
        "best field service dispatch SaaS software 2025 technician scheduling jobs"
    ) == ["web_search"]


def test_actionable_text_match_requires_a_structured_tool_call() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            names = {item["function"]["name"] for item in body["tools"]}
            assert body["tool_choice"] == "required"
            assert "get_portal_capabilities" in names
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_portal_capabilities",
                                    "arguments": {},
                                },
                            }
                        ],
                    }
                },
            )
        assert "tool_choice" not in body
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Capabilities ready."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    request = _request(portal_auto_tools=True)
    request["messages"][-1]["content"] = "What can you do?"
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=request,
    )

    assert response.status_code == 200
    assert response.json["message"]["content"] == "Capabilities ready."


def test_successful_discovery_requires_a_concrete_leaf_before_final_answer() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "discover-calculator",
                                "type": "function",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": {"query": "calculator arithmetic"},
                                },
                            }
                        ],
                    }
                },
            )
        if len(requests) == 2:
            assert body["tool_choice"] == "required"
            names = {item["function"]["name"] for item in body["tools"]}
            assert "safe_math_eval" in names
            assert body["messages"][-1]["role"] == "tool"
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "calculate",
                                "type": "function",
                                "function": {
                                    "name": "safe_math_eval",
                                    "arguments": {"expression": "173 * 419"},
                                },
                            }
                        ],
                    }
                },
            )
        assert "tool_choice" not in body
        assert body["messages"][-1]["role"] == "tool"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "72487"}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {
                    "role": "user",
                    "content": "Use a calculator to compute 173 multiplied by 419.",
                }
            ],
            portal_auto_tools=True,
        ),
    )

    assert response.status_code == 200
    assert response.json["message"]["content"] == "72487"
    assert [
        item["name"] for item in response.json["portal"]["safe_tools_executed"]
    ] == ["tool_search", "safe_math_eval"]


def test_rendered_browser_tool_is_discoverable_and_session_scoped() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    cleared: list[str] = []

    class Browser:
        def act(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            calls.append((session_id, arguments))
            return {
                "rendered": True,
                "url": "http://example.test/",
                "elements": [{"id": "e1", "text": "Details"}],
                "screenshot": {
                    "mime_type": "image/png",
                    "encoding": "base64",
                    "data": "image-data",
                },
            }

        def clear(self, session_id: str) -> None:
            cleared.append(session_id)

    browser = Browser()
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), browser_automation=browser
    )

    assert discover_tool_names("visually navigate and click a rendered webpage")[0] == (
        "browser_interact"
    )
    result = harness.execute(
        "voice-session",
        "browser_interact",
        {"action": "click", "element_id": "e1"},
    )
    harness.clear("voice-session")

    assert result["rendered"] is True
    assert result["screenshot"]["data"] == "image-data"
    assert calls == [
        ("voice-session", {"action": "click", "element_id": "e1"})
    ]
    assert cleared == ["voice-session"]


def test_desktop_gui_tool_is_discoverable_and_returns_visual_evidence() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Gui:
        def act(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            calls.append((session_id, arguments))
            return {
                "rendered": True,
                "active_window": {"title": "Chromium"},
                "screenshot": {
                    "mime_type": "image/png",
                    "encoding": "base64",
                    "data": "desktop-image",
                },
            }

        def clear(self, _session_id: str) -> None:
            pass

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), gui_automation=Gui()
    )

    assert discover_tool_names("look at and control the desktop workspace")[0] == (
        "gui_interact"
    )
    result = harness.execute(
        "voice-session", "gui_interact", {"action": "click", "x": 20, "y": 30}
    )

    assert result["screenshot"]["data"] == "desktop-image"
    assert calls == [
        ("voice-session", {"action": "click", "x": 20, "y": 30})
    ]


def test_tool_screenshot_is_native_media_not_base64_tool_text() -> None:
    encoded = base64.b64encode(b"current-png-bytes" * 10_000).decode()

    content, images = _model_tool_result(
        {
            "rendered": True,
            "screenshot": {
                "mime_type": "image/png",
                "encoding": "base64",
                "data": encoded,
            },
            "visual_fingerprint": {
                "algorithm": "rgb64-q16-v1",
                "digest": "cobalt-17",
                "sample": "large-binary-sample",
            },
        }
    )

    receipt = json.loads(content)
    assert receipt["screenshot"]["data"] == "attached_as_current_tool_image"
    assert receipt["visual_fingerprint"]["sample"] == "omitted_binary_sample"
    assert encoded not in content
    assert images == [
        {
            "mime_type": "image/png",
            "encoding": "base64",
            "data": encoded,
        }
    ]


def test_shell_tool_returns_command_context(tmp_path: Path) -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    result = harness.execute(
        "one",
        "shell",
        {
            "command": "printf shell-out; printf shell-err >&2; exit 7",
            "cwd": str(tmp_path),
        },
    )

    assert result["cwd"] == str(tmp_path)
    assert result["stdout"] == "shell-out"
    assert result["stderr"] == "shell-err"
    assert result["exit_code"] == 7
    assert result["timed_out"] is False
    assert result["stdout_truncated"] is False


def test_shell_stdin_writes_generated_content_without_shell_quoting(tmp_path: Path) -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    content = "# Plan\nquote: 'single' and \"double\"\n$dollar `backtick`\n"

    result = harness.execute(
        "one",
        "shell",
        {
            "command": "tee plan.md >/dev/null && wc -c < plan.md",
            "cwd": str(tmp_path),
            "stdin": content,
        },
    )

    assert result["exit_code"] == 0
    assert result["stdin_bytes"] == len(content.encode())
    assert int(result["stdout"].strip()) == len(content.encode())
    assert (tmp_path / "plan.md").read_text(encoding="utf-8") == content


def test_workspace_file_compacts_create_read_replace_and_list(tmp_path: Path) -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    source = tmp_path / "app" / "main.py"

    created = harness.execute(
        "one",
        "workspace_file",
        {
            "action": "write",
            "path": str(source),
            "content": "VALUE = 'old'\n",
        },
    )
    assert created["created"] is True
    assert created["validation"] == "python_ast_ok"
    assert "content" not in created

    read = harness.execute(
        "one", "workspace_file", {"action": "read", "path": str(source)}
    )
    assert read["content"] == "VALUE = 'old'\n"
    assert read["sha256"] == created["sha256"]

    replaced = harness.execute(
        "one",
        "workspace_file",
        {
            "action": "replace",
            "path": str(source),
            "old_text": "'old'",
            "new_text": "'new'",
            "expected_sha256": read["sha256"],
        },
    )
    assert replaced["validation"] == "python_ast_ok"
    assert source.read_text(encoding="utf-8") == "VALUE = 'new'\n"

    listed = harness.execute(
        "one",
        "workspace_file",
        {"action": "list", "path": str(tmp_path), "depth": 2},
    )
    assert {item["path"] for item in listed["entries"]} == {"app", "app/main.py"}


def test_workspace_file_rejects_invalid_source_before_overwrite(tmp_path: Path) -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    source = tmp_path / "main.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    rejected = harness.execute(
        "one",
        "workspace_file",
        {"action": "write", "path": str(source), "content": "def broken(:\n"},
    )

    assert rejected["error"] == "ToolInputError"
    assert "content not written" in rejected["message"]
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_shell_is_found_without_putting_its_schema_in_the_first_pass() -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    result = harness.execute(
        "one", "tool_search", {"query": "run a raw bash shell command"}
    )

    assert result["available_tools"][0] == "shell"

    file_result = harness.execute(
        "one", "tool_search", {"query": "write a file and encode it with ffmpeg"}
    )
    assert file_result["available_tools"] == ["workspace_file", "shell"]

    workspace_result = harness.execute(
        "one", "tool_search", {"query": "write docs/plan.md"}
    )
    assert workspace_result["available_tools"] == ["workspace_file"]


def test_failed_web_fetch_routes_back_to_discovery_instead_of_guessing_hosts() -> None:
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        resolver=lambda _hostname: (_ for _ in ()).throw(OSError("dns failed")),
    )

    result = harness.execute(
        "one", "web_fetch", {"url": "https://not-a-real-source.invalid/page"}
    )

    assert result["error"] == "ToolInputError"
    assert result["failure_scope"] == "arguments"
    assert result["task_blocked"] is False
    assert result["disposition"] == "change_capability"
    assert result["alternative_tools"] == ["web_search", "browser_interact"]


def test_challenged_web_fetch_prefers_the_rendered_browser() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(403))
    )
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        web_client=client,
        resolver=lambda _hostname: ["93.184.216.34"],
    )

    result = harness.execute(
        "one", "web_fetch", {"url": "https://example.com/challenged"}
    )

    assert result["error"] == "ToolInputError"
    assert result["disposition"] == "change_capability"
    assert result["alternative_tools"] == ["browser_interact", "web_search"]
    client.close()


def test_safe_math_eval_computes_without_code_execution() -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    computed = harness.execute("one", "safe_math_eval", {"expression": "sqrt(81) + 2 ** 3"})
    blocked = harness.execute("one", "safe_math_eval", {"expression": "__import__('os').system('id')"})
    assert computed == {"expression": "sqrt(81) + 2 ** 3", "result": 17.0, "engine": "bounded_ast"}
    assert blocked["error"] == "ToolInputError"


def test_structured_read_queries_attached_json_and_yaml() -> None:
    documents = SessionDocumentStore(ttl_s=300)
    harness = PortalToolHarness(documents)
    uploads = [
        {"name": "people.json", "mime_type": "application/json", "encoding": "base64", "data": base64.b64encode(b'{"people":[{"name":"Ada","score":9},{"name":"Lin","score":8}]}').decode()},
        {"name": "settings.yaml", "mime_type": "application/yaml", "encoding": "base64", "data": base64.b64encode(b"voice:\n  preset: female\n  speed: 1.1\n").decode()},
    ]
    _context, accepted = documents.prepare("one", uploads, "people and voice")
    json_result = harness.execute("one", "structured_read", {"document_id": accepted[0]["id"], "path": "people[1]"})
    yaml_result = harness.execute("one", "structured_read", {"document_id": accepted[1]["id"], "path": "voice"})
    assert json_result["data"] == {"name": "Lin", "score": 8}
    assert yaml_result["data"] == {"preset": "female", "speed": 1.1}


def test_web_crawl_is_bounded_same_origin_and_indexed() -> None:
    requested = []
    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path == "/":
            return httpx.Response(200, text='<html><title>Root</title><a href="/guide">Guide</a><a href="https://elsewhere.example/private">Elsewhere</a></html>', headers={"content-type": "text/html"})
        return httpx.Response(200, text="<html><title>Guide</title><p>Orchid crawl evidence.</p></html>", headers={"content-type": "text/html"})
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300), web_client=httpx.Client(transport=httpx.MockTransport(handler)), resolver=lambda _hostname: ["93.184.216.34"])
    result = harness.execute("one", "web_crawl", {"url": "https://example.com/", "max_pages": 2, "max_depth": 1})
    recalled = harness.execute("one", "web_search", {"query": "orchid evidence", "mode": "session"})
    assert result["pages_fetched"] == 2
    assert result["strategy"] == "direct_http"
    assert result["pages"][0]["links"] == [
        {"url": "https://example.com/guide", "text": "Guide"}
    ]
    assert result["pages"][0]["receipt"]["link_status"] == "retrieved"
    assert requested == ["https://example.com/", "https://example.com/guide"]
    assert recalled["results"][0]["url"] == "https://example.com/guide"


def test_ocr_pdf_is_attachment_scoped_and_indexes_recognized_text() -> None:
    documents = SessionDocumentStore(ttl_s=300, document_extractor=lambda _name, _mime, _raw: "", ocr_runner=lambda raw, language, pages: f"OCR orchid evidence from {len(raw)} bytes in {language} across {pages} pages.")
    harness = PortalToolHarness(documents)
    _context, accepted = documents.prepare("one", [{"name": "scan.pdf", "mime_type": "application/pdf", "encoding": "base64", "data": base64.b64encode(b"%PDF-1.7\nscanned").decode()}], "orchid")
    result = harness.execute("one", "ocr_pdf", {"document_id": accepted[0]["id"], "language": "eng", "max_pages": 3})
    recalled = harness.execute("one", "document_search", {"query": "orchid"})
    isolated = harness.execute("two", "ocr_pdf", {"document_id": accepted[0]["id"]})
    assert result["indexed_for_session_recall"] is True
    assert "OCR orchid evidence" in recalled["results"][0]["content"]
    assert isolated["error"] == "DocumentError"


def test_working_notes_and_task_list_are_session_scoped() -> None:
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    note = harness.execute("one", "working_notes", {"action": "add", "category": "finding", "content": "Copper relay is active."})
    task = harness.execute("one", "task_list", {"action": "upsert", "content": "Verify copper relay", "status": "in_progress"})
    assert note["added"] is True
    assert harness.execute("one", "working_notes", {"action": "search", "content": "copper"})["notes"][0]["content"] == "Copper relay is active."
    assert task["status"] == "in_progress"
    assert harness.execute("one", "task_list", {"action": "list"})["tasks"]
    assert harness.execute("two", "working_notes", {"action": "list"})["notes"] == []
    assert harness.execute("two", "task_list", {"action": "list"})["tasks"] == []


def test_subagent_tools_are_synchronous_and_session_scoped() -> None:
    calls: list[tuple[str, str, str]] = []

    def runner(objective: str, role: str, context: str) -> dict[str, Any]:
        calls.append((objective, role, context))
        return {
            "content": f"Reviewed: {objective}",
            "provenance": {"tools_available": False},
        }

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), subagent_runner=runner
    )
    delegated = harness.execute(
        "one",
        "subagent_delegate",
        {
            "objective": "Critique the proposed interface.",
            "role": "critic",
            "context": "The interface has one primary action.",
        },
    )

    assert calls == [
        (
            "Critique the proposed interface.",
            "critic",
            "The interface has one primary action.",
        )
    ]
    assert delegated["status"] == "completed"
    assert delegated["result"]["content"].startswith("Reviewed:")
    task_id = delegated["task_id"]
    assert harness.execute("one", "subagent_list", {})["tasks"][0]["task_id"] == task_id
    assert harness.execute("one", "subagent_result", {"task_id": task_id})[
        "result"
    ] == delegated["result"]
    assert harness.execute("two", "subagent_list", {})["tasks"] == []
    assert harness.execute("two", "subagent_result", {"task_id": task_id})[
        "error"
    ] == "ToolInputError"
    harness.clear("one")
    assert harness.execute("one", "subagent_list", {})["tasks"] == []


def test_audio_analyze_and_video_scan_use_only_observed_session_media() -> None:
    def media_runner(raw: bytes, mime_type: str, kind: str) -> dict[str, Any]:
        return {"kind": kind, "mime_type": mime_type, "duration": len(raw) / 10, "streams": [{"codec_type": kind, "codec_name": "test"}]}
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300), media_runner=media_runner)
    harness.observe_request("one", {"messages": [{"role": "user", "content": "Analyze these.", "audios": [{"mime_type": "audio/wav", "data": base64.b64encode(b"audio-bytes").decode()}], "videos": [{"mime_type": "video/mp4", "data": base64.b64encode(b"video-bytes").decode()}]}]})
    audio = harness.execute("one", "audio_analyze", {})
    video = harness.execute("one", "video_scan", {})
    assert audio["analysis"]["streams"][0]["codec_type"] == "audio"
    assert video["analysis"]["streams"][0]["codec_type"] == "video"
    assert harness.execute("two", "audio_analyze", {})["found"] is False
    assert harness.execute("two", "video_scan", {})["found"] is False


def test_session_search_federates_conversation_memory_notes_tasks_documents_and_web() -> None:
    documents = SessionDocumentStore(ttl_s=300)
    harness = PortalToolHarness(documents)
    harness.observe_request("one", {"messages": [{"role": "user", "content": "The orchid conversation marker."}]})
    harness.execute("one", "memory_write", {"topic": "orchid", "key": "memory", "value": "Orchid memory marker."})
    harness.execute("one", "working_notes", {"action": "add", "content": "Orchid note marker."})
    harness.execute("one", "task_list", {"action": "upsert", "content": "Orchid task marker."})
    documents.prepare("one", [{"name": "orchid.txt", "mime_type": "text/plain", "encoding": "base64", "data": base64.b64encode(b"Orchid document marker.").decode()}], "orchid")
    result = harness.execute("one", "session_search", {"query": "orchid marker", "max_results": 20})
    sources = {item["source"] for item in result["results"]}
    assert {"conversation", "memory", "working_note", "task", "document"} <= sources
    assert harness.execute("two", "session_search", {"query": "orchid marker", "max_results": 20})["results"] == []


def test_portal_queues_concurrent_sessions_without_context_bleed() -> None:
    release_first = threading.Event()
    first_entered = threading.Event()
    lock = threading.Lock()
    upstream_active = 0
    max_upstream_active = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_active, max_upstream_active
        body = json.loads(request.content)
        marker = body["messages"][-1]["content"]
        with lock:
            upstream_active += 1
            max_upstream_active = max(max_upstream_active, upstream_active)
        try:
            if marker == "session-one":
                first_entered.set()
                assert release_first.wait(5)
            return httpx.Response(
                200,
                json={
                    "model": DEFAULT_MODEL,
                    "message": {"role": "assistant", "content": marker},
                    "adapter": {"route": ["language"]},
                },
            )
        finally:
            with lock:
                upstream_active -= 1

    app = create_app(
        _config(timeout_s=5, max_inflight_requests=3),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    clients = [app.test_client(), app.test_client()]
    for client in clients:
        client.get("/")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    results: dict[str, Any] = {}

    def send(index: int, marker: str) -> None:
        results[marker] = clients[index].post(
            "/api/chat",
            headers=headers,
            json=_request(messages=[{"role": "user", "content": marker}]),
        )

    first = threading.Thread(target=send, args=(0, "session-one"))
    second = threading.Thread(target=send, args=(1, "session-two"))
    first.start()
    assert first_entered.wait(2)
    second.start()

    activity = None
    observer = app.test_client()
    for _attempt in range(50):
        activity = observer.get("/api/activity", headers=headers).json
        if activity["inflight"] == 2:
            break
        time.sleep(0.02)
    assert activity == {
        "users": 2,
        "inflight": 2,
        "active": 1,
        "queued": 1,
        "slots": 1,
        "limit": 3,
    }

    release_first.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert max_upstream_active == 1
    assert results["session-one"].status_code == 200
    assert results["session-two"].status_code == 200
    assert results["session-one"].json["message"]["content"] == "session-one"
    assert results["session-two"].json["message"]["content"] == "session-two"


def test_session_diagnostics_are_isolated_redacted_clearable_and_expiring(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "message": {
                    "role": "assistant",
                    "content": body["messages"][-1]["content"],
                },
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(
        _config(session_log_dir=tmp_path, session_log_ttl_s=0.5),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = app.test_client()
    second = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}
    first.get("/")
    second.get("/")

    first_response = first.post(
        "/api/chat",
        headers=headers,
        json=_request(
            messages=[{"role": "user", "content": "secret-one"}],
            portal_auto_tools=True,
        ),
    )
    second.post(
        "/api/chat",
        headers=headers,
        json=_request(messages=[{"role": "user", "content": "secret-two"}]),
    )
    first_log = first.get("/api/diagnostics", headers=headers).json
    second_log = second.get("/api/diagnostics", headers=headers).json

    assert first_response.headers["X-Omni-Request-ID"]
    assert first_log["events"]
    assert second_log["events"]
    assert first_log != second_log
    assert "secret-one" not in json.dumps(first_log)
    assert "secret-two" not in json.dumps(second_log)
    assert first_log["events"][0]["tools_requested"] is True
    assert second_log["events"][0]["tools_requested"] is False
    assert len(list(tmp_path.glob("*.json"))) == 2

    request_id = first_response.headers["X-Omni-Request-ID"]
    telemetry = first.post(
        "/api/diagnostics",
        headers=headers,
        json={
            "event": "client_stream_timing",
            "request_id": request_id,
            "first_audio_delta_ms": 123.456,
            "content": "must not persist",
        },
    )
    assert telemetry.json == {"accepted": True}
    assert "must not persist" not in json.dumps(first.get("/api/diagnostics", headers=headers).json)

    assert first.delete("/api/diagnostics", headers=headers).status_code == 204
    assert first.get("/api/diagnostics", headers=headers).json["events"] == []
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert first.post(
        "/api/diagnostics",
        headers=headers,
        json={
            "event": "client_stream_timing",
            "request_id": request_id,
            "complete_ms": 999,
        },
    ).json == {"accepted": False}

    deadline = time.monotonic() + 1
    while list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert list(tmp_path.glob("*.json")) == []


def test_portal_pins_model_and_proxies_normal_response() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "message": {"role": "assistant", "content": "Hello back."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    bad = client.post("/api/chat", headers=headers, json=_request(model="other"))
    good = client.post("/api/chat", headers=headers, json=_request())

    assert bad.status_code == 400
    assert good.status_code == 200
    assert good.json["message"]["content"] == "Hello back."
    assert good.json["portal"]["safe_tools_executed"] == []
    assert seen[0]["model"] == DEFAULT_MODEL


def test_portal_defaults_reasoning_off_and_requires_boolean() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Answer."}})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}
    without_think = _request()
    without_think.pop("think")

    defaulted = client.post("/api/chat", headers=headers, json=without_think)
    invalid = client.post("/api/chat", headers=headers, json=_request(think="yes"))

    assert defaulted.status_code == 200
    assert seen[0]["think"] is False
    assert invalid.status_code == 400


def test_voice_profile_resolves_relative_speaker_and_validates_language(
    tmp_path,
) -> None:
    speaker = tmp_path / "reference.wav"
    speaker.write_bytes(b"RIFF")
    profile_path = tmp_path / "voice.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "name": "studio",
                "language": "en",
                "speaker_file": "reference.wav",
                "temperature": 0.5,
                "seed": 7,
            }
        )
    )

    profile = load_voice_profile(profile_path)

    assert profile["speaker_file"] == str(speaker.resolve())
    assert profile["seed"] == 7

    profile_path.write_text(
        json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "language": "unsupported",
            }
        )
    )
    try:
        load_voice_profile(profile_path)
    except RuntimeError as exc:
        assert "language must be one of" in str(exc)
    else:
        raise AssertionError("unsupported TTS language was accepted")


def test_bundled_voice_presets_are_metadata_free_pcm() -> None:
    profile_path = Path("portal/voice-profile.json")
    profile = load_voice_profile(profile_path)
    assert [
        (preset["id"], preset["label"], preset["default"]) for preset in profile["presets"]
    ] == [
        ("female", "Female", True),
        ("male", "Male", False),
    ]


    assert Path(profile["speaker_file"]).name == "female_voice.wav"

    for preset in profile["presets"]:
        voice_path = Path(preset["speaker_file"])
        raw = voice_path.read_bytes()
        assert raw[:4] == b"RIFF"
        assert raw[8:12] == b"WAVE"
        assert raw[12:16] == b"fmt "
        assert raw[36:40] == b"data"
        with wave.open(str(voice_path), "rb") as wav:
            assert wav.getcomptype() == "NONE"
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            duration_ms = round(wav.getnframes() * 1000 / wav.getframerate())
            assert 500 <= duration_ms <= 30000


def test_omnius_derived_search_uses_duckduckgo_html_and_decodes_results() -> None:
    destination = "https://example.com/guide"
    redirect = f"https://duckduckgo.com/l/?uddg={quote_plus(destination)}"
    result_html = (
        '<a href="https://duckduckgo.com/settings">Settings</a>'
        f'<a class="result__a" href="{redirect}">Example guide title</a>'
        f'<a class="result__snippet" href="{redirect}">A useful guide snippet.</a>'
        '<a class="result__a" href="http://127.0.0.1/private">Private</a>'
        '<a class="result__a" href="https://user:pass@example.com/secret">Secret</a>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        assert request.url.path == "/html/"
        assert request.url.params["q"] == "example guide"
        return httpx.Response(
            200,
            text=result_html,
            headers={"content-type": "text/html; charset=UTF-8"},
        )

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        web_client=httpx.Client(transport=httpx.MockTransport(handler)),
        resolver=lambda _hostname: ["93.184.216.34"],
    )

    result = harness.execute("one", "web_search", {"query": "example guide"})
    assert result["provider"] == "duckduckgo"
    assert result["transport"] == "direct_html"
    assert result["results"] == [
        {
            "title": "Example guide title",
            "url": destination,
            "snippet": "A useful guide snippet.",
            "link_status": "unverified_search_result",
        }
    ]

def test_web_search_has_no_browser_or_provider_fallback() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200,
            text="<html><body>No matching results.</body></html>",
            headers={"content-type": "text/html"},
        )

    result = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        web_client=httpx.Client(transport=httpx.MockTransport(handler)),
        resolver=lambda _hostname: ["93.184.216.34"],
    ).execute("one", "web_search", {"query": "unlikely query"})

    assert result["results"] == []
    assert len(requested) == 1
    assert requested[0].startswith("https://html.duckduckgo.com/html/")


def test_web_fetch_returns_receipt_raw_html_cache_and_binary_refusal() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/binary":
            return httpx.Response(
                200,
                content=b"%PDF-1.7\n",
                headers={"content-type": "text/plain"},
            )
        return httpx.Response(
            200,
            text=(
                '<html><head><title>Endpoint clues</title></head><body>'
                '<form action="/search"></form>'
                '<script>fetch("/views/ajax")</script><p>Visible text.</p></body></html>'
            ),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        web_client=httpx.Client(transport=httpx.MockTransport(handler)),
        resolver=lambda _hostname: ["93.184.216.34"],
    )
    raw = harness.execute(
        "one",
        "web_fetch",
        {"url": "https://example.com/page", "format": "raw_html"},
    )
    cached_text = harness.execute(
        "one",
        "web_fetch",
        {"url": "https://example.com/page", "format": "text"},
    )
    binary = harness.execute(
        "one", "web_fetch", {"url": "https://example.com/binary"}
    )

    assert raw["format"] == "raw_html"
    assert '<form action="/search">' in raw["content"]
    assert 'fetch("/views/ajax")' in raw["content"]
    assert raw["receipt"]["schema"] == "robit.omni.web-fetch-receipt.v1"
    assert raw["receipt"]["response_sha256"]
    assert cached_text["cached"] is True
    assert "Visible text." in cached_text["content"]
    assert "fetch(" not in cached_text["content"]
    assert len(calls) == 2
    assert binary["error"] == "ToolInputError"
    assert "PDF document" in binary["message"]


def test_foreground_static_fetch_handoff_requires_rendered_browser_recovery(
    monkeypatch,
) -> None:
    requests: list[dict[str, Any]] = []
    original_execute = PortalToolHarness.execute

    def execute(self, session_id, name, arguments):
        if name == "web_search":
            return {
                "error": "rendered_page_required",
                "retryable": False,
                "failure_scope": "capability",
                "task_blocked": False,
                "disposition": "change_capability",
                "alternative_tools": ["browser_interact", "gui_interact"],
            }
        if name == "browser_interact":
            return {
                "rendered": True,
                "url": "https://example.com/source",
                "title": "Verified source",
            }
        return original_execute(self, session_id, name, arguments)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        names = {item["function"]["name"] for item in body.get("tools", [])}
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": {"query": "current public web research"},
                                },
                            }
                        ],
                    }
                },
            )
        if len(requests) == 2:
            assert "web_search" in names
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": {
                                        "query": "current topic",
                                        "mode": "discover",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        if len(requests) == 3:
            assert {"browser_interact", "gui_interact"} <= names
            assert any(
                "<tool_recovery>" in str(message.get("content") or "")
                for message in body["messages"]
            )
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "I cannot continue because the page needs rendering.",
                    }
                },
            )
        if len(requests) == 4:
            assert {"browser_interact", "gui_interact"} <= names
            assert "Do not give up" in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "browser_interact",
                                    "arguments": {
                                        "action": "navigate",
                                        "url": "https://example.com/source",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Verified."}},
        )

    monkeypatch.setattr(PortalToolHarness, "execute", execute)
    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert response.json["message"]["content"] == "Verified."
    assert [
        item["name"] for item in response.json["portal"]["safe_tools_executed"]
    ] == ["tool_search", "web_search", "browser_interact"]


def test_successful_search_exposes_source_reading_on_the_next_round(
    monkeypatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def execute(_self, _session_id, name, _arguments):
        if name == "web_search":
            return {
                "results": [
                    {
                        "title": "Forecast",
                        "url": "https://example.com/forecast",
                    }
                ],
                "alternative_tools": ["web_fetch", "browser_interact"],
                "provenance": {
                    "authority": "discovery_only",
                    "citation_ready": False,
                },
            }
        if name == "web_fetch":
            return {
                "content": "A complete seven-day forecast.",
                "provenance": {
                    "source_url": "https://example.com/forecast",
                    "citation_ready": True,
                },
            }
        raise AssertionError(f"unexpected tool: {name}")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        names = {item["function"]["name"] for item in body.get("tools", [])}
        if len(requests) == 1:
            assert "web_search" in names
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": {"query": "seven day forecast"},
                                },
                            }
                        ],
                    }
                },
            )
        if len(requests) == 2:
            assert {"web_search", "web_fetch", "browser_interact"} <= names
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "web_fetch",
                                    "arguments": {
                                        "url": "https://example.com/forecast"
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Forecast ready."}},
        )

    monkeypatch.setattr(PortalToolHarness, "execute", execute)
    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    request = _request(portal_auto_tools=True)
    request["messages"][-1]["content"] = (
        "Search the web for Portland weather this week."
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=request,
    )

    assert response.status_code == 200
    assert response.json["message"]["content"] == "Forecast ready."
    assert [
        item["name"] for item in response.json["portal"]["safe_tools_executed"]
    ] == ["web_search", "web_fetch"]


def test_portal_enforces_server_voice_profile() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello."}},
        )

    profile = {
        "name": "fixed-voice",
        "language": "en",
        "speaker_file": "/srv/voices/fixed.wav",
        "temperature": 0.4,
        "top_k": 20,
        "top_p": 0.8,
        "seed": 42,
        "max_frames": 512,
    }
    app = create_app(
        _config(voice_profile=profile),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(speech={"speaker_file": "/tmp/client-choice.wav", "seed": -1}),
    )

    assert response.status_code == 200
    assert seen[0]["speech"] == {key: value for key, value in profile.items() if key != "name"}


def test_portal_accepts_safe_client_voice_clone_and_controls() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello."}})

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    reference = {
        "mime_type": "audio/wav",
        "encoding": "base64",
        "data": base64.b64encode(wav_bytes.getvalue()).decode("ascii"),
    }
    app = create_app(
        _config(voice_profile={"name": "default", "language": "en", "seed": 42}),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_voice={
                "clone_enabled": True,
                "speaker_audio": reference,
                "language": "ja",
                "temperature": 0.55,
                "top_k": 24,
                "top_p": 0.8,
                "seed": 7,
                "max_frames": 384,
            },
            speech={"speaker_file": "/tmp/untrusted.wav"},
        ),
    )

    assert response.status_code == 200
    assert seen[0]["speech"]["language"] == "ja"
    assert seen[0]["speech"]["temperature"] == 0.55
    assert seen[0]["speech"]["seed"] == 7
    assert "speaker_file" not in seen[0]["speech"]
    assert seen[0]["speech"]["speaker_audio"]["data"] == reference["data"]
    assert "portal_voice" not in seen[0]


def test_portal_resolves_only_allowlisted_voice_presets() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello."}})

    profile = {
        "name": "presets",
        "language": "en",
        "speaker_file": "/srv/voices/female.wav",
        "presets": [
            {
                "id": "female",
                "label": "Female",
                "speaker_file": "/srv/voices/female.wav",
                "default": True,
            },
            {
                "id": "male",
                "label": "Male",
                "speaker_file": "/srv/voices/male.wav",
                "default": False,
            },
        ],
    }
    app = create_app(
        _config(voice_profile=profile),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client = app.test_client()
    selected = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_voice={"clone_enabled": True, "preset": "male"}),
    )
    rejected = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_voice={"clone_enabled": True, "preset": "unknown"}),
    )

    assert selected.status_code == 200
    assert seen[0]["speech"]["speaker_file"] == "/srv/voices/male.wav"
    assert "preset" not in seen[0]["speech"]
    assert rejected.status_code == 400
    assert "unknown voice preset" in rejected.json["error"]


def test_portal_rejects_invalid_voice_clone_before_proxy() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_voice={
                "clone_enabled": True,
                "speaker_audio": {
                    "mime_type": "audio/wav",
                    "encoding": "base64",
                    "data": base64.b64encode(b"not a wav").decode("ascii"),
                },
            }
        ),
    )

    assert response.status_code == 400
    assert "invalid voice reference" in response.json["error"]
    assert calls == 0


def test_portal_routes_sanitized_browser_location_through_session_tool() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_user_location",
                                    "arguments": {},
                                },
                            }
                        ],
                    },
                    "adapter": {"route": ["language"]},
                },
            )
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "You are near Seattle."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_auto_tools=True,
            portal_client_location={
                "ip": "203.0.113.42",
                "city": "Seattle",
                "region": "Washington",
                "country": "United States",
                "latitude": 47.606,
                "longitude": -122.332,
            },
        ),
    )

    assert response.status_code == 200
    assert "portal_client_location" not in requests[0]
    tool_result = json.loads(requests[1]["messages"][-1]["content"])
    assert tool_result["city"] == "Seattle"
    assert tool_result["raw_ip_included"] is False
    assert "203.0.113.42" not in json.dumps(requests)
    assert response.json["portal"]["safe_tools_executed"][0]["name"] == "get_user_location"


def test_portal_keeps_only_the_active_discovered_schema() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        names = {item["function"]["name"] for item in body.get("tools", [])}
        if len(requests) == 1:
            assert names == {"tool_search"}
            call = {"name": "tool_search", "arguments": {"query": "current time"}}
        elif len(requests) == 2:
            assert "get_current_time" in names
            assert len(names) <= 4  # discovery plus at most three matches
            call = {"name": "get_current_time", "arguments": {}}
        else:
            assert names == {"tool_search", "get_current_time"}
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "It is test time."}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": call}],
                }
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 3
    assert [
        item["name"] for item in response.json["portal"]["safe_tools_executed"]
    ] == ["tool_search", "get_current_time"]


def test_embodied_client_gets_compact_physical_shell_and_background_bridges() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ready."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_auto_tools=True,
            portal_camera_bridge=True,
            portal_shell_bridge=True,
            portal_background_bridge=True,
        ),
    )

    assert response.status_code == 200
    assert {
        item["function"]["name"] for item in requests[0]["tools"]
    } == {"tool_search", "request_camera_view", "shell", "background_task"}
    assert "portal_camera_bridge" not in requests[0]
    assert "portal_shell_bridge" not in requests[0]
    assert "portal_background_bridge" not in requests[0]


def test_background_only_voice_profile_cannot_rediscover_or_call_foreground_shell(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert {
                item["function"]["name"] for item in body["tools"]
            } == {"tool_search", "background_task"}
            call = {
                "name": "tool_search",
                "arguments": {"query": "raw shell file creation"},
            }
        elif len(requests) == 2:
            discovery = json.loads(body["messages"][-1]["content"])
            assert "shell" not in discovery["available_tools"]
            assert "shell" not in {
                item["function"]["name"] for item in body["tools"]
            }
            # Even a fabricated unadvertised call is rejected server-side.
            call = {"name": "shell", "arguments": {"command": "touch forbidden"}}
        elif len(requests) == 3:
            rejected = json.loads(body["messages"][-1]["content"])
            assert rejected["error"] == "tool_not_available_in_this_execution_profile"
            call = {
                "name": "background_task",
                "arguments": {
                    "action": "start",
                    "objective": "Create and verify the requested file.",
                },
            }
        else:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "I started it."}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": call}],
                }
            },
        )

    app = create_app(
        _config(background_task_path=tmp_path / "tasks.json"),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_auto_tools=True,
            portal_shell_bridge=False,
            portal_background_bridge=True,
        ),
    )

    assert response.status_code == 200
    assert response.json["message"]["content"] == "I started it."


def test_live_tools_allow_plain_reply_or_execute_selected_tool(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert "tool_choice" not in body
            assert "respond_to_user" not in {
                item["function"]["name"] for item in body["tools"]
            }
            response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "background_task",
                                "arguments": {
                                    "action": "start",
                                    "objective": "Carry out and verify the request.",
                                },
                            },
                        }
                    ],
                }
            }
        else:
            assert "tool_choice" not in body
            response = {
                "message": {"role": "assistant", "content": "Started."}
            }
        wire = json.dumps({"type": "final", "response": response}) + "\n"
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(
        _config(background_task_path=tmp_path / "tasks.json"),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            stream=True,
            portal_auto_tools=True,
            portal_background_bridge=True,
        ),
    )
    events = [json.loads(line) for line in response.data.splitlines()]

    assert response.status_code == 200
    assert len(requests) == 2
    assert events[-1]["response"]["message"]["content"] == "Started."
    assert events[-1]["response"]["portal"]["safe_tools_executed"][0][
        "name"
    ] == "background_task"


def test_portal_executes_only_allowlisted_tool_and_strips_media_on_followup() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "tool needed",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_current_time",
                                    "arguments": {},
                                },
                            }
                        ],
                    },
                    "adapter": {"route": ["comprehension", "language"]},
                },
            )
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "It is now test time."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    body = _request(
        messages=[
            {
                "role": "user",
                "content": "Use the clock.",
                "images": [{"mime_type": "image/png", "data": "unused-in-mock"}],
            }
        ],
        portal_auto_tools=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 200
    assert len(requests) == 2
    assert "portal_auto_tools" not in requests[0]
    assert "<portal_tools>" in requests[0]["messages"][0]["content"]
    assert (
        "Call an already exposed matching tool directly"
        in requests[0]["messages"][0]["content"]
    )
    assert (
        "search again only for a different unmet capability"
        in requests[0]["messages"][0]["content"]
    )
    assert "images" not in requests[1]["messages"][0]
    tool_result = requests[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_name"] == "get_current_time"
    assert response.json["portal"]["safe_tools_executed"][0]["name"] == "get_current_time"
    assert response.json["portal"]["safe_tools_executed"][0]["result"]
    diagnostic_events = client.get(
        "/api/diagnostics", headers={"Authorization": f"Bearer {TOKEN}"}
    ).json["events"]
    tool_events = [
        event for event in diagnostic_events if event["event"].startswith("tool_call_")
    ]
    assert [event["event"] for event in tool_events] == [
        "tool_call_started",
        "tool_call_completed",
    ]
    assert {event["tool_name"] for event in tool_events} == {"get_current_time"}
    assert tool_events[-1]["tool_ok"] is True
    media_events = [
        event for event in diagnostic_events if event["event"] == "media_observed"
    ]
    assert len(media_events) == 1
    assert media_events[0]["media_id"].startswith("image-")


def test_portal_subagent_delegation_uses_fresh_tool_free_text_context() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        messages = body.get("messages") or []
        if messages and "isolated, read-only helper" in str(messages[0].get("content")):
            return httpx.Response(
                200,
                json={
                    "model": DEFAULT_MODEL,
                    "message": {
                        "role": "assistant",
                        "content": "The delegated review found one clear risk.",
                    },
                },
            )
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "subagent_delegate",
                                    "arguments": {
                                        "objective": "Review the interface risk.",
                                        "role": "critic",
                                        "context": "One action is destructive.",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "I found one clear risk."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True, think=False),
    )

    assert response.status_code == 200
    assert len(requests) == 3
    delegated = requests[1]
    assert delegated["think"] is False
    assert delegated["stream"] is False
    assert delegated["response_modalities"] == ["text"]
    assert delegated["speech_mode"] == "never"
    assert "tools" not in delegated
    assert [message["role"] for message in delegated["messages"]] == ["system", "user"]
    tool_result = json.loads(requests[2]["messages"][-1]["content"])
    assert tool_result["status"] == "completed"
    assert tool_result["result"]["provenance"] == {
        "source_type": "isolated_model_completion",
        "role": "critic",
        "tools_available": False,
        "media_available": False,
        "reasoning_enabled": False,
    }
    assert response.json["portal"]["safe_tools_executed"][0]["name"] == "subagent_delegate"


def test_subagent_can_handoff_current_user_context_without_regenerating_it() -> None:
    requests: list[dict[str, Any]] = []
    parent_context = "Audit this large evidence block: " + ("receipt-constraint " * 300)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        messages = body.get("messages") or []
        if messages and "isolated, read-only helper" in str(messages[0].get("content")):
            return httpx.Response(
                200,
                json={
                    "model": DEFAULT_MODEL,
                    "message": {"role": "assistant", "content": "Context audited."},
                },
            )
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "subagent_delegate",
                                    "arguments": {
                                        "objective": "Audit all supplied evidence.",
                                        "role": "critic",
                                        "context_source": "current_user_message",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Audit complete."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[{"role": "user", "content": parent_context}],
            portal_auto_tools=True,
        ),
    )

    assert response.status_code == 200
    assert len(requests) == 3
    delegated_user = requests[1]["messages"][-1]["content"]
    assert parent_context.strip() in delegated_user
    parent_followup = requests[2]
    tool_call_arguments = parent_followup["messages"][-2]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert tool_call_arguments == {
        "objective": "Audit all supplied evidence.",
        "role": "critic",
        "context_source": "current_user_message",
    }
    trace = response.json["portal"]["safe_tools_executed"][0]
    assert trace["arguments"]["context_source"] == "current_user_message"
    assert "context" not in trace["arguments"]


def test_portal_tool_chain_has_no_legacy_fifty_call_cap() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        call_number = len(requests)
        if call_number <= 55:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "memory_write",
                                    "arguments": {
                                        "topic": "chain-test",
                                        "key": f"step-{call_number}",
                                        "value": str(call_number),
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Chain complete."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 56
    assert len(response.json["portal"]["safe_tools_executed"]) == 55


def test_portal_tool_round_has_no_legacy_fifty_call_cap() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "memory_write",
                                    "arguments": {
                                        "topic": "batch-test",
                                        "key": f"item-{index}",
                                        "value": str(index),
                                    },
                                },
                            }
                            for index in range(55)
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Batch complete."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 2
    assert len(response.json["portal"]["safe_tools_executed"]) == 55


def test_portal_returns_duplicate_errors_then_stops_if_nothing_changes() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "memory_search",
                                "arguments": {"query": "unchanged"},
                            },
                        }
                    ],
                }
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 502
    assert "without actionable progress" in response.json["error"]
    assert len(requests) == 3


def test_duplicate_failure_is_returned_so_the_model_can_correct_it() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) <= 2:
            call = {"name": "shell", "arguments": {"command": "exit 7"}}
        elif len(requests) == 3:
            duplicate = json.loads(body["messages"][-1]["content"])
            assert duplicate["error"] == "duplicate_tool_call"
            call = {"name": "shell", "arguments": {"command": "printf fixed"}}
        else:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Fixed."}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": call}],
                }
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 4
    trace = response.json["portal"]["safe_tools_executed"]
    assert [item["ok"] for item in trace] == [False, False, True]


def test_portal_stops_varying_tool_calls_that_never_make_progress() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "tool_search",
                                "arguments": {
                                    "query": f"missing capability {len(requests)}"
                                },
                            },
                        }
                    ],
                }
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 502
    assert "without actionable progress" in response.json["error"]
    assert len(requests) == 3


def test_active_tool_can_be_called_again_without_rediscovery() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        names = {item["function"]["name"] for item in body.get("tools", [])}
        if len(requests) == 1:
            assert names == {"tool_search"}
            call = {"name": "tool_search", "arguments": {"query": "arithmetic"}}
        elif len(requests) == 2:
            assert "safe_math_eval" in names
            call = {"name": "safe_math_eval", "arguments": {"expression": "6 * 7"}}
        elif len(requests) == 3:
            assert names == {"tool_search", "safe_math_eval"}
            call = {"name": "safe_math_eval", "arguments": {"expression": "7 * 8"}}
        else:
            assert names == {"tool_search", "safe_math_eval"}
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "42 and 56."}},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": call}],
                }
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 4
    assert [
        item["name"] for item in response.json["portal"]["safe_tools_executed"]
    ] == ["tool_search", "safe_math_eval", "safe_math_eval"]


def test_portal_parses_omnius_style_text_tool_call_fallback() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"memory_write","arguments":'
                            '{"topic":"demo","key":"shape","value":"circle"}}'
                            "</tool_call>"
                        ),
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Remembered."}},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(portal_auto_tools=True),
    )

    assert response.status_code == 200
    assert len(requests) == 2
    assistant = requests[1]["messages"][-2]
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["function"]["name"] == "memory_write"
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert response.json["portal"]["safe_tools_executed"][0]["name"] == "memory_write"


def test_portal_rejects_streaming_before_proxy() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True),
    )

    assert response.status_code == 400
    assert calls == 0


def test_portal_stream_route_pins_profile_and_relays_ndjson() -> None:
    seen = []
    wire = (
        b'{"type":"delta","message":{"content":"Hi"}}\n'
        b'{"type":"final","response":{"message":{"content":"Hi"}}}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    profile = {"name": "fixed", "language": "en", "seed": 42}
    app = create_app(
        _config(voice_profile=profile),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True, speech={"seed": -1}),
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[0] == {"type": "delta", "message": {"content": "Hi"}}
    assert events[-1]["type"] == "final"
    assert events[-1]["response"]["message"]["content"] == "Hi"
    assert events[-1]["response"]["portal"]["safe_tools_executed"] == []
    assert seen[0][0] == "/api/chat/stream"
    assert seen[0][1]["stream"] is True
    assert seen[0][1]["think"] is True
    assert seen[0][1]["speech"] == {"language": "en", "seed": 42}
    assert response.headers["X-Omni-Request-ID"]


def test_portal_retries_one_network_stream_failure_and_resets_partial_text() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            wire = (
                b'{"type":"delta","message":{"content":"Partial"}}\n'
                b'{"type":"error","error":"network error"}\n'
            )
        else:
            wire = (
                b'{"type":"delta","message":{"content":"Recovered"}}\n'
                b'{"type":"final","response":{"message":{"content":"Recovered"}}}\n'
            )
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    response = client.post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True),
    )

    events = [json.loads(line) for line in response.data.splitlines()]
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert [event["type"] for event in events] == [
        "delta",
        "reset",
        "delta",
        "final",
    ]
    assert events[1] == {
        "type": "reset",
        "reason": "upstream_network_retry",
        "attempt": 1,
    }
    diagnostics = client.get(
        "/api/diagnostics",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ).json["events"]
    assert any(
        event["event"] == "upstream_stream_error"
        and event["outcome"] == "network_error"
        for event in diagnostics
    )
    assert any(event["event"] == "upstream_stream_retry" for event in diagnostics)
    assert diagnostics[-1]["event"] == "request_complete"
    assert diagnostics[-1]["status"] == 200


def test_portal_retries_an_incomplete_httpx_response_body() -> None:
    requests = 0

    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"type":"delta","message":{"content":"Partial"}}\n'
            raise httpx.ReadError("peer connection reset")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                200,
                stream=BrokenBody(),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(
            200,
            content=(
                b'{"type":"delta","message":{"content":"Recovered"}}\n'
                b'{"type":"final","response":{"message":{"content":"Recovered"}}}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True),
    )

    events = [json.loads(line) for line in response.data.splitlines()]
    assert requests == 2
    assert [event["type"] for event in events] == [
        "delta",
        "reset",
        "delta",
        "final",
    ]


def test_portal_does_not_retry_a_stream_after_audio_has_started() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            content=(
                b'{"type":"audio_start","audio":{"codec":"pcm_s16le"}}\n'
                b'{"type":"error","error":"network error"}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    response = client.post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True),
    )

    events = [json.loads(line) for line in response.data.splitlines()]
    assert requests == 1
    assert [event["type"] for event in events] == ["audio_start", "error"]
    diagnostics = client.get(
        "/api/diagnostics",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ).json["events"]
    assert diagnostics[-1]["event"] == "request_complete"
    assert diagnostics[-1]["status"] == 502


def test_mock_live_call_stream_defaults_native_reasoning_off() -> None:
    seen = []
    wire = (
        b'{"type":"delta","message":{"content":"Final answer."}}\n'
        b'{"type":"final","response":{"message":{"content":"Final answer."}}}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    body = _request(
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "Listen and reply naturally.",
                "audios": [{"data": "UklGRg=="}],
            }
        ],
        response_modalities=["text", "audio"],
        speech_mode="always",
    )
    body.pop("think")

    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert events[0]["message"]["content"] == "Final answer."
    assert events[-1]["response"]["message"]["content"] == "Final answer."
    assert seen[0]["think"] is False
    assert seen[0]["messages"][1:] == body["messages"]
    environment = seen[0]["messages"][0]
    assert environment["role"] == "system"
    assert "<runtime_environment>" not in environment["content"]
    assert "natural participant" in environment["content"]
    assert "explicit system-snapshot tool" in environment["content"]
    assert "fact that can change after training" in environment["content"]
    assert "explicitly asks you to check or verify" in environment["content"]
    assert "Tool results" in environment["content"]
    assert "only a current visual observation" in environment["content"]
    assert "not GPS, street position, or a visible scene" in environment["content"]
    assert "<live_system>" not in environment["content"]
    assert "Available offline:" not in environment["content"]


def test_battery_state_reader_rejects_stale_service_data(tmp_path: Path) -> None:
    from portal.environment import _battery_facts

    state = tmp_path / "battery.json"
    state.write_text(
        json.dumps(
            {
                "schema": "robit.egg.battery.v1",
                "available": True,
                "updated_at": time.time() - 120,
                "stale_after_seconds": 15,
                "percentage": 80,
                "voltage_v": 28.0,
            }
        ),
        encoding="utf-8",
    )

    assert _battery_facts(state)["available"] is False


def test_runtime_environment_snapshot_is_bounded_and_omits_sensitive_network_data(
    monkeypatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="0, NVIDIA Test GPU, 81920, 2048, 37, 52, 120.5, 700.0\n",
        stderr="",
    )
    monkeypatch.setattr("portal.environment.subprocess.run", lambda *args, **kwargs: completed)
    monkeypatch.setattr("portal.environment.is_tegra", lambda: False)

    snapshot = PortalToolHarness(SessionDocumentStore(ttl_s=300)).execute(
        "one", "get_system_snapshot", {}
    )

    assert snapshot["captured_at"]
    assert snapshot["utc_time"]
    assert snapshot["cpu"]["logical_cpus"] > 0
    assert snapshot["memory"]["total_gib"] > 0
    assert snapshot["gpus"][0] == {
        "index": 0,
        "name": "NVIDIA Test GPU",
        "vram_total_mib": 81920.0,
        "vram_used_mib": 2048.0,
        "utilization_percent": 37.0,
        "temperature_c": 52.0,
        "power_w": 120.5,
        "power_limit_w": 700.0,
    }
    serialized = json.dumps(snapshot)
    assert "address" in snapshot["privacy"].lower()
    assert '"ip"' not in serialized.lower()
    assert '"mac"' not in serialized.lower()


def test_runtime_environment_snapshot_reports_the_tegra_integrated_gpu(monkeypatch) -> None:
    """nvidia-smi answers [N/A] on Tegra, so sysfs is the only real evidence."""

    def refuse(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("nvidia-smi must not be queried for GPU facts on Tegra")

    monkeypatch.setattr("portal.environment.subprocess.run", refuse)
    monkeypatch.setattr("portal.environment.is_tegra", lambda: True)
    monkeypatch.setattr(
        "portal.environment.tegra_gpu_facts",
        lambda: [
            {
                "index": 0,
                "name": "NVIDIA Tegra integrated GPU (tegra234)",
                "vram_total_mib": 30698.0,
                "vram_used_mib": 9001.0,
                "utilization_percent": 42.0,
                "temperature_c": 51.0,
                "power_w": None,
                "power_limit_w": None,
                "memory_model": "unified",
                "frequency_mhz": 1300.5,
            }
        ],
    )

    snapshot = PortalToolHarness(SessionDocumentStore(ttl_s=300)).execute(
        "one", "get_system_snapshot", {}
    )

    assert snapshot["gpus"][0]["name"] == "NVIDIA Tegra integrated GPU (tegra234)"
    assert snapshot["gpus"][0]["memory_model"] == "unified"
    assert snapshot["accelerator"]["tegra"] is True


def test_compact_system_policy_merges_without_eager_host_snapshot() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Done."}})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            messages=[
                {"role": "system", "content": "Answer naturally."},
                {"role": "user", "content": "Hello."},
            ]
        ),
    )

    assert response.status_code == 200
    messages = seen[0]["messages"]
    assert [item["role"] for item in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("Answer naturally.")
    assert "<runtime_environment>" not in messages[0]["content"]
    assert "natural participant" in messages[0]["content"]
    assert "Tool results" in messages[0]["content"]
    assert "untrusted data" in messages[0]["content"]
    assert "<portal_tools>" not in messages[0]["content"]


def test_internal_background_worker_uses_the_compact_policy_envelope(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Done."}}
        )

    app = create_app(
        _config(
            virtual_context_mode="active",
            virtual_context_root=tmp_path / "virtual-context",
        ),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    task_query = "Advance and verify the pinned task. Objective: Build it."
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_background_worker=True,
            portal_virtual_query=task_query,
            messages=[
                {"role": "system", "content": "<current_task>Build it.</current_task>"},
                {
                    "role": "assistant",
                    "content": "Imaginary prior artifact mercury-884 passed every test.",
                },
                {
                    "role": "user",
                    "content": "<retained_checkpoint>Old failed probe.</retained_checkpoint>",
                },
            ],
        ),
    )

    assert response.status_code == 200
    payload = seen[0]
    assert "portal_background_worker" not in payload
    assert "portal_virtual_query" not in payload
    content = payload["messages"][0]["content"]
    assert "authenticated internal durable-task worker" in content
    assert "Tool results" in content
    assert "natural participant" not in content
    assert "mercury-884" not in json.dumps(payload)
    assert f"<current_query>\n{task_query}\n</current_query>" in payload["messages"][1][
        "content"
    ]


def test_portal_stream_route_requires_auth_and_chains_session_tools() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "memory_write",
                                "arguments": {
                                    "topic": "demo",
                                    "key": "color",
                                    "value": "violet",
                                },
                            },
                        }
                    ],
                }
            }
        elif len(requests) == 2:
            response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "memory_search",
                                "arguments": {"query": "color"},
                            },
                        }
                    ],
                }
            }
        else:
            response = {"message": {"role": "assistant", "content": "Violet."}}
        wire = json.dumps({"type": "final", "response": response}) + "\n"
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()

    assert client.post("/api/chat/stream", json=_request(stream=True)).status_code == 401
    response = client.post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True, portal_auto_tools=True),
    )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.splitlines()]
    assert len(requests) == 3
    assert {item["function"]["name"] for item in requests[0]["tools"]} == {
        "tool_search"
    }
    assert requests[1]["messages"][-1]["tool_name"] == "memory_write"
    assert requests[2]["messages"][-1]["tool_name"] == "memory_search"
    assert [
        item["name"]
        for item in events[-1]["response"]["portal"]["safe_tools_executed"]
    ] == ["memory_write", "memory_search"]
    complete_events = [
        event
        for event in events
        if event.get("type") == "tool" and event.get("phase") == "complete"
    ]
    start_events = [
        event
        for event in events
        if event.get("type") == "tool" and event.get("phase") == "start"
    ]
    assert [event["tools"][0]["id"] for event in start_events] == [
        event["tools"][0]["id"] for event in complete_events
    ]
    assert [event["round"] for event in start_events] == [1, 2]
    assert [event["round"] for event in complete_events] == [1, 2]
    assert complete_events[0]["tools"][0]["status"] == "complete"
    assert complete_events[0]["tools"][0]["result"]
    assert events[-1]["response"]["message"]["content"] == "Violet."


def test_portal_stream_tool_chain_has_no_legacy_fifty_call_cap() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        call_number = len(requests)
        if call_number <= 55:
            response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "memory_write",
                                "arguments": {
                                    "topic": "stream-chain",
                                    "key": f"step-{call_number}",
                                    "value": str(call_number),
                                },
                            },
                        }
                    ],
                }
            }
        else:
            response = {"message": {"role": "assistant", "content": "Complete."}}
        wire = json.dumps({"type": "final", "response": response}) + "\n"
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True, portal_auto_tools=True),
    )
    events = [json.loads(line) for line in response.data.splitlines()]

    assert response.status_code == 200
    assert len(requests) == 56
    assert events[-1]["type"] == "final"
    assert len(events[-1]["response"]["portal"]["safe_tools_executed"]) == 55


def test_portal_exposes_its_tool_suite_for_external_loops() -> None:
    """A caller driving its own tool loop needs to run one tool at a time.

    The portal executes these inside its own agentic loop, which suits a
    browser session. An embodied runtime mixes them with tools of its own --
    cameras, memory -- so it has to keep control of the conversation and
    execute a single tool. Without this it would have to keep a second
    service alive purely to run tools, which is the cost the single weights
    package exists to remove.
    """

    app = create_app(
        _config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))
    )
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    listed = client.get("/api/tools", headers=headers)
    assert listed.status_code == 200
    names = {item["function"]["name"] for item in listed.get_json()["tools"]}
    assert {"web_search", "web_fetch", "memory_write", "safe_math_eval"} <= names

    # A tool with no network dependence proves the execution path end to end.
    executed = client.post(
        "/api/tools/safe_math_eval/call",
        json={"arguments": {"expression": "6 * 7"}},
        headers=headers,
    )
    assert executed.status_code == 200
    body = executed.get_json()
    assert body["tool"] == "safe_math_eval"
    assert "42" in json.dumps(body["result"])


def test_portal_tool_endpoint_is_guarded() -> None:
    app = create_app(
        _config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))
    )
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    assert client.get("/api/tools").status_code == 401
    assert client.post("/api/tools/web_search/call", json={}).status_code == 401
    # Only the published suite may be invoked.
    unknown = client.post("/api/tools/rm_rf/call", json={}, headers=headers)
    assert unknown.status_code == 404
