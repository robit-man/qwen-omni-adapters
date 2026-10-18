"""Exact rejection-time native evidence remains host-only and never grants input."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture, Node

from portal.native_ui import NativeUiAdapter, NativeUiUnavailable, _JsonBridge

REJECTED = {"status": "rejected", "reason": "stale_native_revision_or_scope"}
SCOPE = {"schema": "robit.omni.native-task-scope.v1", "windows": [
    {"window_id": "17", "process_id": 42, "process_start": "420"}]}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def scoped(monkeypatch, session="freshness-session"):
    fixture = Fixture(session)
    monkeypatch.setattr("portal.native_ui._process_start", lambda _pid: "420")
    monkeypatch.setattr("portal.native_ui._x11_window_owner",
                        lambda *_args, **_kwargs: {"window_id": "17", "process_id": 42})
    fixture.adapter.set_window_scope(SCOPE)
    fixture.bridge.window_scope = copy.deepcopy(SCOPE)
    return fixture


def stale(fixture):
    action = fixture.action()
    # The requested target is unchanged; a transient sibling disappears.
    notice = Node("/ephemeral", "label", "Transient notice")
    fixture.window.children.append(notice)
    action = fixture.action()
    fixture.window.children.remove(notice)
    return action


def test_real_worker_refusal_retains_exact_comparison_snapshot_without_another_read(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    observe = fixture.bridge.observe
    snapshots = []

    def capture(**kwargs):
        state = observe(**kwargs)
        snapshots.append(copy.deepcopy(state))
        return state

    fixture.bridge.observe = capture
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert len(snapshots) == 1
    assert diagnostic["status"] == "retained"
    assert diagnostic["stage"] == "worker_perform_entry_stale_guard"
    assert diagnostic["snapshot"] == snapshots[0]
    assert diagnostic["snapshot_sha256"] == hashlib.sha256(encoded(snapshots[0])).hexdigest()
    assert diagnostic["action_sha256"] == hashlib.sha256(encoded(action)).hexdigest()
    assert diagnostic["snapshot"]["revision"] != action["revision"]
    assert diagnostic["input_attempted"] is diagnostic["input_authority"] is diagnostic["task_progress"] is False
    protected = next(node for node in diagnostic["snapshot"]["nodes"] if node["sensitive"])
    assert protected["name"] == "[protected]" and protected["operations"] == []
    assert "never-export-this" not in encoded(diagnostic).decode()
    assert fixture.button.events == []
    assert fixture.adapter.take_freshness_rejection() is None


@pytest.mark.parametrize("change,reason", [("scope", "scope_changed"), ("unscoped", "host_window_scope_unavailable")])
def test_changed_scope_or_unrestricted_adapter_does_not_export_current_view(monkeypatch, change, reason):
    fixture = scoped(monkeypatch) if change == "scope" else Fixture()
    action = stale(fixture)
    if change == "scope":
        action["scope_id"] = "different-issued-window"
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable" and diagnostic["unavailable_reason"] == reason
    assert diagnostic["snapshot"] is diagnostic["snapshot_sha256"] is None
    assert diagnostic["input_attempted"] is False
    assert "Transient notice" not in encoded(diagnostic).decode()
    assert fixture.button.events == []


@pytest.mark.parametrize("change", ["foreign-session", "malformed-session", "missing-target", "generation", "disabled"])
def test_other_refusals_have_no_entry_stale_observation(monkeypatch, change):
    fixture = scoped(monkeypatch)
    action = fixture.action()
    if change in {"foreign-session", "malformed-session"}:
        action["session_id"] = "foreign" if change == "foreign-session" else None
    elif change == "missing-target":
        action["node_id"] = "missing"
    elif change == "generation":
        action["generation"] = "999"
    else:
        fixture.button.state.values.remove("SENSITIVE")
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    if result["reason"] == REJECTED["reason"]:
        # A real disabled-state change can first trip the unchanged revision guard.
        assert fixture.adapter.take_freshness_rejection()["status"] == "retained"
    else:
        assert fixture.adapter.take_freshness_rejection() is None
    assert fixture.button.events == []


@pytest.mark.parametrize("change", ["action", "session", "scope", "revision", "digest", "false-as-zero", "extra",
                                    "sensitive", "source-shape"])
def test_malformed_or_foreign_private_envelope_cannot_publish_exact_state(monkeypatch, change):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    request = fixture.request

    def corrupt(value):
        result = request(value)
        if value["op"] != "perform":
            return result
        result = copy.deepcopy(result)
        diagnostic = result["_freshness_rejection"]
        if change == "action":
            diagnostic["action_sha256"] = "0" * 64
        elif change in {"session", "scope", "revision"}:
            key = {"session": "session_id", "scope": "scope_id", "revision": "revision"}[change]
            diagnostic["observed_source"][key] = "foreign"
        elif change == "digest":
            diagnostic["snapshot_sha256"] = "0" * 64
        elif change == "false-as-zero":
            diagnostic["input_attempted"] = 0
        elif change == "extra":
            diagnostic["private_environment"] = "do-not-retain"
        elif change == "source-shape":
            diagnostic["snapshot"]["scope"]["environment"] = "do-not-retain"
            diagnostic["snapshot_sha256"] = hashlib.sha256(encoded(diagnostic["snapshot"])).hexdigest()
        else:
            protected = next(node for node in diagnostic["snapshot"]["nodes"] if node["sensitive"])
            protected.update(name="do-not-retain", value="do-not-retain", operations=["invoke"])
            diagnostic["snapshot_sha256"] = hashlib.sha256(encoded(diagnostic["snapshot"])).hexdigest()
        return result

    fixture.adapter._runner = corrupt
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["input_attempted"] is None  # Invalid worker evidence supplies no attestation.
    assert "do-not-retain" not in encoded(diagnostic).decode()
    assert fixture.button.events == []


@pytest.mark.parametrize("value", [None, "x" * 513, "界" * 1_000_000],
                         ids=["null", "513chars", "million-CJK-chars"])
def test_unbounded_or_malformed_requested_revision_cannot_expand_fallback(monkeypatch, value):
    fixture = scoped(monkeypatch)
    action = fixture.action()
    action["revision"] = value
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable" and len(encoded(diagnostic)) < 4096
    assert fixture.button.events == []


def test_ascii_wire_expansion_overflow_becomes_unavailable_without_truncated_exact_claim(monkeypatch):
    fixture = scoped(monkeypatch)
    # Names are individually within the existing500char limit; ensure_ascii
    # turns each non-BMP scalar into12bytes, overflowing the unchanged2MiB cap.
    action = fixture.action()
    fixture.window.children.extend(Node(f"/group-{group}", "panel", "Group", children=[
        Node(f"/cjk-{group}-{i}", "label", "😀" * 500) for i in range(125)]) for group in range(3))
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["unavailable_reason"] == "snapshot_too_large"
    assert diagnostic["snapshot"] is diagnostic["snapshot_sha256"] is None
    assert len(encoded(diagnostic)) < 4096 and diagnostic["input_attempted"] is False
    assert fixture.button.events == []


def test_private_artifact_survives_actual_jsonbridge_framing_but_not_public_receipt(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    worker_result = fixture.bridge.perform(action)
    monkeypatch.setattr("portal.native_ui.uuid.uuid4", lambda: SimpleNamespace(hex="request"))
    bridge = _JsonBridge.__new__(_JsonBridge)
    bridge.session_id, bridge.timeout_s, bridge.closed = fixture.bridge.session_id, 1, False
    bridge.process_id, bridge.window_scope = None, copy.deepcopy(SCOPE)
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb") as writer:
        bridge.process = SimpleNamespace(stdin=io.BytesIO(), stdout=reader)
        envelope = {"schema": "robit.omni.native-ui.v1", "id": "request", "ok": True, "result": worker_result}
        raw = json.dumps(envelope, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        assert len(raw) < 2 * 1024 * 1024
        writer.write(raw)
        writer.flush()
        adapter = NativeUiAdapter(fixture.bridge.session_id, bridge_runner=bridge.request)
        adapter.set_window_scope(SCOPE)
        assert adapter.perform(action) == REJECTED
        assert adapter.take_freshness_rejection()["snapshot"] == worker_result["_freshness_rejection"]["snapshot"]


@pytest.mark.parametrize("next_step", ["dispatch", "invalid", "close", "cancel"])
def test_slot_is_not_reused_after_another_attempt_close_or_cancellation(monkeypatch, next_step):
    fixture = scoped(monkeypatch)
    assert fixture.adapter.perform(stale(fixture)) == REJECTED
    if next_step == "dispatch":
        assert fixture.adapter.perform(fixture.action())["status"] == "dispatched"
    elif next_step == "invalid":
        assert fixture.adapter.perform({"verb": "unknown"})["status"] == "rejected"
    elif next_step == "close":
        fixture.adapter.close()
    else:
        next_action = fixture.action()
        fixture.adapter._cancelled = lambda: True
        with pytest.raises(NativeUiUnavailable):
            fixture.adapter.perform(next_action)
    assert fixture.adapter.take_freshness_rejection() is None


def test_partial_input_failure_keeps_public_uncertainty_and_has_no_stale_artifact(monkeypatch):
    fixture = scoped(monkeypatch)
    action = fixture.action()
    fixture.button.fail_action = True
    result = fixture.adapter.perform(action)
    assert result == {"status": "dispatched", "reason": "native_pattern_effect_unresolved"}
    assert fixture.button.events == ["invoke"]
    assert fixture.adapter.take_freshness_rejection() is None


def test_actual_controller_outcome_and_next_planner_feedback_never_receive_private_snapshot(monkeypatch):
    from portal.ui_task import _compact_result
    from qwen_omni_adapters.computer_use import RapidUiController

    fixture = scoped(monkeypatch)
    notice = Node("/changing-status", "label", "Transient notice")
    fixture.window.children.append(notice)
    controller = RapidUiController(fixture.adapter, authorize=lambda *_args: True)
    issued = controller.observe()
    actions = controller.enumerate({"id": "activate-preview", "verb": "invoke", "target": {"name": "Preview"},
                                    "args": {}, "expect": [{"kind": "text", "text": "Verified"}]})
    selected = next(action for action in actions if action["kind"] == "action")
    perform = fixture.adapter.perform

    def race(action):
        fixture.window.children.remove(notice)
        return perform(action)

    fixture.adapter.perform = race
    result = controller.execute(selected["id"])
    assert result["status"] == "rejected" and result["receipt"] == REJECTED
    assert "after" not in result and controller._state == issued
    compact = _compact_result({"outcomes": [result]})
    assert "_freshness_rejection" not in encoded(compact).decode()
    assert "snapshot" not in encoded(compact).decode()
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "retained"
    assert diagnostic["snapshot"]["revision"] != controller._state["revision"]
    assert fixture.button.events == []


def test_retained_copy_is_not_changed_by_mutating_worker_reply_after_return(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    request, replies = fixture.request, []

    def capture(value):
        result = request(value)
        if value["op"] == "perform":
            replies.append(result)
        return result

    fixture.adapter._runner = capture
    assert fixture.adapter.perform(action) == REJECTED
    raw = replies[0]["_freshness_rejection"]
    original = copy.deepcopy(raw)
    raw["snapshot"]["nodes"][0]["name"] = "changed after rejection"
    assert fixture.adapter.take_freshness_rejection() == original


def test_unknown_observation_after_prior_refusal_clears_old_private_artifact(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    assert fixture.adapter.perform(action) == REJECTED

    def unavailable():
        raise PermissionError("provider-private-content")

    fixture.button.get_state_set = unavailable
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.perform(action)
    assert error.value.code == "native_observation_unavailable"
    assert "provider-private-content" not in str(error.value)
    assert fixture.adapter.take_freshness_rejection() is None
    assert fixture.button.events == []


def test_nonserializable_worker_metadata_does_not_replace_stale_refusal(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    observe = fixture.bridge.observe

    def malformed(**kwargs):
        result = observe(**kwargs)
        result["window_visibility"].append({"malformed": object()})
        return result

    fixture.bridge.observe = malformed
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["unavailable_reason"] == "invalid_snapshot"
    assert diagnostic["snapshot"] is None and diagnostic["input_attempted"] is False
    assert fixture.button.events == []


@pytest.mark.parametrize("location", ["bounds", "structure", "ancestor", "text_observation", "captured_range",
                                      "window_binding", "document_evidence", "coordinate_binding", "surface",
                                      "frame_extents", "content", "open_popups", "window_visibility"])
def test_nested_unknown_fields_cannot_be_rehashed_into_exact_host_evidence(monkeypatch, location):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    request = fixture.request

    def corrupt(value):
        result = request(value)
        if value["op"] != "perform":
            return result
        result = copy.deepcopy(result)
        diagnostic = result["_freshness_rejection"]
        snapshot = diagnostic["snapshot"]
        node = next(node for node in snapshot["nodes"] if node["name"] == "Title")
        coordinate = {"space": "window", "window_id": "17",
                      "surface": {"x": -15, "y": -15, "width": 800, "height": 600},
                      "frame_extents": {"left": 15, "right": 15, "top": 15, "bottom": 15},
                      "content": {"x": 0, "y": 0, "width": 770, "height": 570}}
        if location == "bounds":
            node["bounds"] = {"x": 0, "y": 0, "width": 40, "height": 20}
            target = node["bounds"]
        elif location in {"structure", "ancestor"}:
            target = node["structure"] if location == "structure" else node["structure"]["ancestors"][0]
        elif location in {"text_observation", "captured_range"}:
            target = node["text_observation"] if location == "text_observation" else node["text_observation"]["captured_range"]
        elif location == "window_binding":
            node["window_binding"] = {"window_id": "17", "process_id": 42, "process_start": "420"}
            target = node["window_binding"]
        elif location == "document_evidence":
            target = snapshot["document_evidence"]
        elif location in {"coordinate_binding", "surface", "frame_extents", "content"}:
            snapshot["coordinate_binding"] = coordinate
            target = coordinate if location == "coordinate_binding" else coordinate[location]
        elif location == "open_popups":
            snapshot["open_popups"] = [{"controller_id": node["id"], "authority": "native_popup_relation"}]
            target = snapshot["open_popups"][0]
        else:
            target = snapshot["window_visibility"][0]
        target["environment"] = "nested-private-sentinel"
        diagnostic["snapshot_sha256"] = hashlib.sha256(encoded(snapshot)).hexdigest()
        return result

    fixture.adapter._runner = corrupt
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable" and diagnostic["input_attempted"] is None
    assert "nested-private-sentinel" not in encoded(diagnostic).decode()
    assert fixture.button.events == []


def test_closed_producer_shapes_preserve_real_document_metadata_and_negative_window_origin(monkeypatch):
    fixture = scoped(monkeypatch)
    document = Node("/document", "document web", "Committed page", interfaces=["Document"])
    document.get_document_iface = lambda: document
    fixture.api.Document = SimpleNamespace(get_document_attribute_value=lambda _node, _key: "https://example.org/document")
    fixture.window.children.append(document)
    coordinate = {"space": "window", "window_id": "17",
                  "surface": {"x": -15, "y": -15, "width": 800, "height": 600},
                  "frame_extents": {"left": 15, "right": 15, "top": 15, "bottom": 15},
                  "content": {"x": 0, "y": 0, "width": 770, "height": 570}}
    fixture.window.interfaces.append("Component")
    fixture.bridge._coordinate_for = lambda *_args, **_kwargs: copy.deepcopy(coordinate)
    fixture.bridge._rect = lambda _node: {"x": -5, "y": 0, "width": 40, "height": 20}
    action = stale(fixture)
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "retained"
    snapshot = diagnostic["snapshot"]
    assert snapshot["coordinate_binding"] == coordinate
    assert snapshot["document_evidence"]["status"] == "observed"
    assert snapshot["scope"]["url"] == "https://example.org/document"
    assert fixture.button.events == []


@pytest.mark.parametrize("location", ["bounds-bool", "character-count-bool", "selection-string", "visibility-string", "document-reason-object"])
def test_nested_wrong_types_cannot_be_rehashed_into_exact_evidence(monkeypatch, location):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    request = fixture.request

    def corrupt(value):
        result = request(value)
        if value["op"] != "perform":
            return result
        result = copy.deepcopy(result)
        diagnostic = result["_freshness_rejection"]
        snapshot = diagnostic["snapshot"]
        node = next(node for node in snapshot["nodes"] if node["name"] == "Title")
        if location == "bounds-bool":
            node["bounds"] = {"x": 0, "y": 0, "width": True, "height": 20}
        elif location == "character-count-bool":
            node["text_observation"]["character_count"] = True
        elif location == "selection-string":
            node["text_selection"] = {"start": "0", "end": 2}
        elif location == "visibility-string":
            snapshot["window_visibility"][0]["active"] = "true"
        else:
            snapshot["document_evidence"]["reason"] = {"content": "not-a-code"}
        diagnostic["snapshot_sha256"] = hashlib.sha256(encoded(snapshot)).hexdigest()
        return result

    fixture.adapter._runner = corrupt
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "unavailable" and diagnostic["input_attempted"] is None
    assert fixture.button.events == []


def test_requested_action_cannot_inject_rejection_time_observation(monkeypatch):
    fixture = scoped(monkeypatch)
    action = stale(fixture)
    action["_freshness_rejection"] = {"snapshot": "model-authored-substitute"}
    assert fixture.adapter.perform(action) == REJECTED
    diagnostic = fixture.adapter.take_freshness_rejection()
    assert diagnostic["status"] == "retained"
    assert "model-authored-substitute" not in encoded(diagnostic).decode()
    assert diagnostic["action_sha256"] == hashlib.sha256(encoded(action)).hexdigest()
    assert fixture.button.events == []
