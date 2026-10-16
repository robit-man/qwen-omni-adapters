"""Document reading is captured knowledge, never fresh input or UI verification."""
from __future__ import annotations

import base64
import copy
import hashlib
import io

import pytest
from PIL import Image
from test_native_visual_inspection import _inspection_receipt, _reader
from test_ui_task import Fixture

from portal.ui_task import _accepted_visual_reading, _compact_result


def png(color, *, changed=None):
    image = Image.new("RGB", (16, 12), color)
    if changed:
        image.putpixel(changed, (255, 0, 0))
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def frame(capture, raw):
    capture.update(image=raw, sha256=hashlib.sha256(raw).hexdigest(),
                   source_frame_sha256=hashlib.sha256(raw).hexdigest(), key=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("inside", [False, True])
def test_exact_document_crop_and_pixel_drift_have_snapshot_only_authority(monkeypatch, inside):
    adapter, state, capture, counts, calls, changed, _response = _reader(monkeypatch)
    state["nodes"][0]["bounds"] = {"x": 14, "y": 24, "width": 8, "height": 6}
    frame(capture, png("white"))
    before = copy.deepcopy(capture)
    changed["capture"] = lambda value: frame(value, png("white", changed=(3, 3) if inside else (0, 0)))
    adapter._visual, adapter._step, adapter.point_receipt = {"point": "untouched"}, {"step": "untouched"}, {"receipt": "untouched"}
    result = adapter.inspect_visual()
    receipt = result["visual_inspection_receipt"]
    assert result["status"] == ("captured_snapshot" if inside else "observed")
    assert _accepted_visual_reading(result) and _compact_result(result)["reading"] == result["reading"]
    assert receipt["snapshot_only"] is True and receipt["temporal_scope"] == "captured_snapshot"
    assert receipt["current_ui_verified"] is False and receipt["input_authority"] is False
    assert result["dispatch_status"] == "not_dispatched" and result["task_progress"] is False
    assert receipt["post_capture_pixels_match"] is not inside
    assert (receipt["frame_sha256"] != receipt["checked_frame_sha256"]) is inside
    assert receipt["source_frame_sha256"] != receipt["checked_source_frame_sha256"]
    assert receipt["captured_at_unix"] <= receipt["checked_at_unix"]
    assert receipt["document_region"] == {"node_id": "document", "generation": "1", "bounds": state["nodes"][0]["bounds"]}
    assert receipt["image_geometry"] == state["nodes"][0]["bounds"]
    assert receipt["source_image_geometry"] == before["image_geometry"]
    raw = base64.b64decode(calls[0][1]["json"]["image"])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (8, 6) and set(image.convert("RGB").getdata()) == {(255, 255, 255)}
    assert hashlib.sha256(raw).hexdigest() == receipt["frame_sha256"]
    assert capture == before and counts == {"native": 4, "capture": 2, "http": 1}
    assert adapter._visual == {"point": "untouched"} and adapter._step == {"step": "untouched"}
    assert adapter.point_receipt == {"receipt": "untouched"}


@pytest.mark.parametrize("field,value", [
    ("role", "panel"), ("generation", "2"), ("scope_id", "foreign"), ("visible", False),
    ("enabled", False), ("sensitive", True), ("bounds", None),
    ("bounds", {"x": True, "y": 22, "width": 16, "height": 12}),
    ("bounds", {"x": 12, "y": 22, "width": 0, "height": 12}),
    ("bounds", {"x": 9, "y": 22, "width": 16, "height": 12})])
def test_document_region_requires_positive_exact_native_binding(monkeypatch, field, value):
    adapter, state, _capture, counts, _calls, _changed, _response = _reader(monkeypatch)
    state["nodes"][0][field] = value
    result = adapter.inspect_visual()
    assert result["status"] == "unavailable" and "reading" not in result and counts["http"] == 0


@pytest.mark.parametrize("change", ["duplicate", "missing", "provider", "provider_digest", "window", "url"])
def test_unproven_document_region_never_falls_back_to_source_authority(monkeypatch, change):
    adapter, state, _capture, counts, _calls, _changed, _response = _reader(monkeypatch)
    if change == "duplicate":
        state["nodes"].append(copy.deepcopy(state["nodes"][0]))
    elif change == "missing":
        state["nodes"].clear()
    else:
        field = {"provider": "provider", "provider_digest": "provider_identity_sha256", "window": "window_id", "url": "url"}[change]
        state["document_evidence"][field] = "wrong"
    result = adapter.inspect_visual()
    assert result["status"] == "unavailable" and "reading" not in result and counts["http"] == 0


@pytest.mark.parametrize("field,value", [
    ("process_start", "4202"), ("identity", {"window_id": "17", "process_id": 43}),
    ("capture_source", "other"), ("capture_root", {"window_id": "2", "geometry": {"x": 0, "y": 0, "width": 100, "height": 100}}),
    ("geometry", {"x": 11, "y": 20, "width": 20, "height": 20}),
    ("source_image_geometry", {"x": 12, "y": 23, "width": 16, "height": 12})])
def test_nonpixel_capture_changes_withhold_even_valid_reader_text(monkeypatch, field, value):
    adapter, _state, _capture, counts, _calls, changed, _response = _reader(monkeypatch)
    changed["capture"] = lambda capture: capture.update({field: value})
    result = adapter.inspect_visual()
    assert result["status"] in {"stale", "unavailable"} and "reading" not in result and counts["http"] == 1


@pytest.mark.parametrize("at_native", [3, 4])
def test_document_bounds_or_protected_change_withholds_post_reply(monkeypatch, at_native):
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    def mutation(state):
        state["nodes"].append({"visible": True, "sensitive": True})
    changed.update(at_native=at_native, native=mutation)
    result = adapter.inspect_visual()
    assert result["status"] == "stale" and "reading" not in result


def test_changed_document_bounds_reject_even_with_unchanged_revision(monkeypatch):
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    changed.update(at_native=3, native=lambda state: state["nodes"][0]["bounds"].update(x=13))
    result = adapter.inspect_visual()
    assert result["status"] == "stale" and "reading" not in result


def test_non_document_reading_keeps_exact_key_rejection(monkeypatch):
    adapter, state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    state["document_evidence"] = None
    changed["capture"] = lambda capture: capture.update(key="changed")
    assert adapter.inspect_visual()["status"] == "stale"


def test_snapshot_handoff_preserves_knowledge_without_task_or_mutation_credit():
    fixture = Fixture([{"kind": "visual_inspection"}, {"kind": "complete", "message": "Only a claim"}])
    result = {"status": "captured_snapshot", "reading": "Fact from the captured frame.",
              "visual_inspection_receipt": _inspection_receipt(changed=True)}
    fixture.adapter.inspect_visual = lambda: copy.deepcopy(result)
    final = list(fixture.run())[-1]["response"]
    assert fixture.packets[1]["last_result"]["reading"] == result["reading"]
    assert fixture.packets[1]["last_result"]["visual_inspection_receipt"]["current_ui_verified"] is False
    assert final["portal"]["safe_tools_executed"][0]["ok"] is True
    assert final["portal"]["native_task"]["task_verified"] is False and fixture.adapter.calls == []
    assert "effect_receipt" not in fixture.packets[1]["last_result"]


@pytest.mark.parametrize("field,value", [("snapshot_only", False), ("current_ui_verified", True),
    ("input_authority", True), ("post_capture_pixels_match", True), ("temporal_scope", "current"),
    ("schema", "other"), ("frame_sha256", "invalid"), ("document_region", None)])
def test_untyped_or_current_authority_snapshot_is_not_planner_knowledge(field, value):
    result = {"status": "captured_snapshot", "reading": "Must be withheld", "dispatch_status": "not_dispatched",
              "evidence_authority": "inspection", "task_progress": False,
              "visual_inspection_receipt": _inspection_receipt(changed=True)}
    result["visual_inspection_receipt"][field] = value
    assert not _accepted_visual_reading(result) and "reading" not in _compact_result(result)


@pytest.mark.parametrize("field,value", [("status", []), ("task_progress", True),
    ("dispatch_status", "dispatched"), ("evidence_authority", "verification")])
def test_reading_never_grants_other_tool_authority(field, value):
    result = {"status": "captured_snapshot", "reading": "Must be withheld", "dispatch_status": "not_dispatched",
              "evidence_authority": "inspection", "task_progress": False,
              "visual_inspection_receipt": _inspection_receipt(changed=True)}
    result[field] = value
    assert not _accepted_visual_reading(result) and "reading" not in _compact_result(result)
