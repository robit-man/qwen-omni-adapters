"""Read-only visual inspection is explicit producing-tool snapshot evidence."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import threading

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_ui import namespace_fixture
from test_portal_native_task_routes import Adapter, _config, _install_store, _post, _request
from test_ui_task import Fixture

from portal.app import _model_tool_result, create_app
from portal.native_ui import NativeUiAdapter, NativeUiUnavailable
from portal.tools import tool_schemas
from portal.ui_planner import planner_call_error, planner_tool_schemas
from portal.ui_task import _compact_result, phase_response_schema
from qwen_omni_adapters.computer_use import ComputerUseError
from runtime.pointing_server import PHOTON_MODEL, PHOTON_REVISION


def _inspection_receipt(*, changed=False):
    return {"schema": "robit.omni.native-visual-inspection.v1", "authority": "untrusted_tool_observation",
            "snapshot_only": True, "temporal_scope": "captured_snapshot", "current_ui_verified": False,
            "input_authority": False, "post_capture_pixels_match": not changed,
            "source_url": "https://docs.example/page", "frame_sha256": "a" * 64,
            "checked_frame_sha256": ("b" if changed else "a") * 64, "model": PHOTON_MODEL,
            "document_evidence": {"status": "observed", "source": "atspi.Document.DocURL", "url": "https://docs.example/page"},
            "document_region": {"node_id": "document", "generation": "1", "bounds": {"x": 12, "y": 22, "width": 16, "height": 12}}}


def test_native_phase_contract_can_explicitly_request_visual_reading():
    Draft202012Validator(phase_response_schema({"native_ui"})).validate({"kind": "visual_inspection"})


def test_store_delegates_explicit_inspection_without_input_or_policy():
    fixture = Fixture([])
    calls = []

    def inspect():
        calls.append("read")
        return {"status": "observed", "reading": "Visible source fact.",
                "visual_inspection_receipt": {"authority": "untrusted_tool_observation"}}

    fixture.adapter.inspect_visual = inspect
    result = fixture.store.act("session", {"surface": "desktop", "operation": "inspect_visual"})
    assert result["reading"] == "Visible source fact."
    assert result["evidence_authority"] == "inspection" and result["task_progress"] is False
    assert calls == ["read"] and fixture.adapter.calls == []


def _reader(monkeypatch, *, truncated=False):
    from PIL import Image

    state = {"session_id": "session", "revision": "semantic-revision", "truncated": truncated,
        "scope": {"id": "active", "kind": "desktop", "window_id": "17", "url": "https://docs.example/page"},
        "document_evidence": {"status": "observed", "source": "atspi.Document.DocURL", "provider": "atspi2",
            "scope_id": "active", "window_id": "17", "node_id": "document", "generation": "1",
            "provider_identity_sha256": "d" * 64, "url": "https://docs.example/page"},
        "nodes": [{"id": "document", "generation": "1", "scope_id": "active", "role": "document_web",
                   "name": "Document", "visible": True, "enabled": True, "sensitive": False,
                   "operations": [], "value": "\ufffc" * 5,
                   "bounds": {"x": 12, "y": 22, "width": 16, "height": 12}}]}
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "white").save(output, format="PNG")
    png = output.getvalue()
    capture = {"identity": {"window_id": "17", "process_id": 42}, "process_start": "4201",
        "geometry": {"x": 10, "y": 20, "width": 20, "height": 20},
        "image_geometry": {"x": 12, "y": 22, "width": 16, "height": 12},
        "capture_root": {"window_id": "1", "geometry": {"x": 0, "y": 0, "width": 100, "height": 100}},
        "pixel_offset": {"x": 2, "y": 2}, "capture_source": "verified-content", "image": png,
        "sha256": hashlib.sha256(png).hexdigest(), "key": "exact-content-frame-key",
        "source_frame_sha256": "outside-clock-source-one"}
    adapter = NativeUiAdapter("session", pointing_url="http://127.0.0.1:8940",
                              pointing_model=PHOTON_MODEL, pointing_revision=PHOTON_REVISION)
    counts = {"native": 0, "capture": 0, "http": 0}
    calls, changed, response = [], {}, {"model": PHOTON_MODEL, "revision": PHOTON_REVISION,
                                       "observation": "Visible fact: source says exactly this."}

    def observe():
        counts["native"] += 1
        value = copy.deepcopy(state)
        if counts["native"] >= changed.get("at_native", 100):
            changed["native"](value)
        return value

    def screenshot(_state):
        counts["capture"] += 1
        value = copy.deepcopy(capture)
        if counts["capture"] == 2 and "capture" in changed:
            changed["capture"](value)
        return value

    def post(url, **kwargs):
        counts["http"] += 1
        calls.append((url, kwargs))
        if "http" in changed:
            return changed["http"](url, kwargs)
        return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    adapter.planning_observe = observe
    adapter._visual_capture = screenshot
    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(adapter, "perform", lambda _action: pytest.fail("Visual reading never inputs"))
    return adapter, state, capture, counts, calls, changed, response


def test_pinned_reading_has_fresh_snapshot_source_and_no_input_or_attachment(monkeypatch):
    adapter, _state, capture, counts, calls, _changed, _response = _reader(monkeypatch)
    adapter._step, adapter._visual, adapter.point_receipt = {"untouched": "step"}, {"untouched": "point"}, {"point": "old"}
    result = adapter.inspect_visual()
    assert result["status"] == "observed" and result["task_progress"] is False
    receipt = result["visual_inspection_receipt"]
    assert receipt["source_url"] == "https://docs.example/page"
    assert receipt["model"] == PHOTON_MODEL and receipt["revision"] == PHOTON_REVISION
    assert receipt["frame_sha256"] == capture["sha256"]
    assert receipt["document_evidence"]["generation"] == "1"
    assert receipt["authority"] == "untrusted_tool_observation" and receipt["absence_claims_supported"] is False
    assert counts == {"native": 4, "capture": 2, "http": 1}
    assert calls[0][0].endswith("/observe") and set(calls[0][1]["json"]) == {"image"}
    assert adapter._step == {"untouched": "step"} and adapter._visual == {"untouched": "point"}
    assert adapter.point_receipt == {"point": "old"}
    textual, images = _model_tool_result(result, tool_name="native_ui")
    assert images == [] and json.loads(textual)["reading"] == result["reading"]
    assert all(key not in json.loads(textual) for key in ("screenshot", "images", "current_visual_input", "input_transcript", "visual_observation"))


@pytest.mark.parametrize("field", ["url", "document_generation", "provider", "revision", "scope"])
def test_changed_native_or_document_binding_withholds_inferred_text(monkeypatch, field):
    adapter, _state, _capture, counts, _calls, changed, _response = _reader(monkeypatch)

    def mutate(state):
        if field == "url":
            state["scope"]["url"] = "https://other.example/"
        elif field == "document_generation":
            state["document_evidence"]["generation"] = "2"
        elif field == "provider":
            state["document_evidence"]["provider_identity_sha256"] = "other"
        elif field == "revision":
            state["revision"] = "changed"
        else:
            state["scope"]["id"] = "other-window"

    changed.update(at_native=3, native=mutate)
    result = adapter.inspect_visual()
    assert result["status"] == "stale" and "reading" not in result
    assert counts["http"] == 1 and "source_url" not in result


def test_document_change_during_final_capture_is_not_source_authority(monkeypatch):
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    changed.update(at_native=4, native=lambda state: state["document_evidence"].update(generation="2"))
    result = adapter.inspect_visual()
    assert result["status"] == "stale" and "reading" not in result


def test_invalid_pixel_digest_fails_but_outside_source_clock_metadata_is_provenance(monkeypatch):
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    changed["capture"] = lambda capture: capture.update(key="changed-content", sha256="changed-pixels")
    assert adapter.inspect_visual()["status"] == "unavailable"
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)
    changed["capture"] = lambda capture: capture.update(source_frame_sha256="different-outside-clock")
    assert adapter.inspect_visual()["status"] == "observed"


@pytest.mark.parametrize("condition", ["missing_bounds", "overlap", "off_image", "hidden"])
def test_observed_visible_protected_controls_are_not_read(monkeypatch, condition):
    adapter, state, _capture, counts, _calls, _changed, _response = _reader(monkeypatch)
    node = {"visible": condition != "hidden", "sensitive": True, "value": "private-sensitive-native-value"}
    if condition != "missing_bounds":
        node["bounds"] = {"x": 13 if condition == "overlap" else 90, "y": 23, "width": 3, "height": 3}
    state["nodes"].append(node)
    result = adapter.inspect_visual()
    expected = condition in {"off_image", "hidden"}
    assert (result["status"] == "observed") is expected
    assert counts["http"] == int(expected)
    assert "private-sensitive-native-value" not in json.dumps(result)


def test_truncated_native_graph_permits_explicit_read_with_unknown_coverage(monkeypatch):
    adapter, _state, _capture, _counts, _calls, _changed, _response = _reader(monkeypatch, truncated=True)
    receipt = adapter.inspect_visual()["visual_inspection_receipt"]
    assert receipt["coverage"] == "unknown" and receipt["source_truncated"] is True
    assert receipt["absence_claims_supported"] is False


@pytest.mark.parametrize("bad", ["model", "revision", "empty", "long", "nontext", "extra"])
def test_wrong_identity_or_invalid_reader_reply_never_becomes_evidence(monkeypatch, bad):
    adapter, _state, _capture, _counts, _calls, _changed, response = _reader(monkeypatch)
    if bad in {"model", "revision"}:
        response[bad] = "wrong"
    elif bad == "extra":
        response["image"] = "not accepted"
    else:
        response["observation"] = {"empty": " ", "long": "x" * 6001, "nontext": []}[bad]
    result = adapter.inspect_visual()
    assert result["status"] == "unavailable" and "reading" not in result


@pytest.mark.parametrize("when", ["before", "during_native", "after_http"])
def test_cancelled_inspection_never_exposes_late_reading(monkeypatch, when):
    adapter, _state, _capture, counts, _calls, changed, response = _reader(monkeypatch)
    event = threading.Event()
    adapter._cancelled = event.is_set
    if when == "before":
        event.set()
    elif when == "during_native":
        changed.update(at_native=2, native=lambda _state: event.set())
    else:
        def post(url, _kwargs):
            event.set()
            return httpx.Response(200, json=response, request=httpx.Request("POST", url))
        changed["http"] = post
    result = adapter.inspect_visual()
    assert result["status"] == "cancelled" and "reading" not in result
    assert counts["http"] == int(when == "after_http")


@pytest.mark.parametrize("document", [None, {"status": "unavailable"}, {"status": "observed", "source": "addressbar", "url": "https://wrong.example/"}])
def test_missing_or_non_document_url_never_invents_source_binding(monkeypatch, document):
    adapter, state, _capture, _counts, _calls, _changed, _response = _reader(monkeypatch)
    state["document_evidence"] = document
    result = adapter.inspect_visual()
    assert result["status"] == "observed" and "source_url" not in result["visual_inspection_receipt"]
    assert result["resource"] == "active"


def test_unrequested_observation_and_reader_failures_do_not_synthesize_success(monkeypatch):
    adapter, _state, _capture, counts, _calls, _changed, _response = _reader(monkeypatch)
    adapter._semantic_observe = lambda **_kwargs: {"unrequested": True}
    assert NativeUiAdapter.planning_observe(adapter) == {"unrequested": True} and counts["http"] == 0
    adapter.pointing_url = ""
    assert adapter.inspect_visual()["status"] == "unavailable" and counts["http"] == 0


def test_catalog_and_model_projection_keep_closed_desktop_only_operation():
    schemas = tool_schemas(["native_ui", "computer_use"])
    for item in planner_tool_schemas(schemas):
        fn = item["function"]
        arguments = {"operation": "inspect_visual"}
        if fn["name"] == "computer_use":
            arguments["surface"] = "desktop"
        validator = Draft202012Validator(fn["parameters"])
        assert validator.is_valid(arguments) and planner_call_error(fn["name"], arguments) is None
        assert not validator.is_valid({**arguments, "query": "arbitrary question"})
        if fn["name"] == "computer_use":
            assert not validator.is_valid({**arguments, "surface": "browser"})
    fixture = Fixture([])
    with pytest.raises(ComputerUseError, match="desktop"):
        fixture.store.act("session", {"operation": "inspect_visual", "surface": "browser"})


@pytest.mark.parametrize("status", ["stale", "unavailable"])
def test_unaccepted_reading_does_not_enter_compacted_planner_evidence(status):
    result = {"status": status, "reading": "Unaccepted text must stay raw only.",
              "visual_inspection_receipt": {"authority": "untrusted_tool_observation"}}
    assert "reading" not in _compact_result(result)


def test_explicit_compiler_reading_handoff_is_bounded_tool_data_without_mutation():
    fixture = Fixture([{"kind": "visual_inspection"}, {"kind": "complete", "message": "Unverified claim"}])
    calls = []
    reading = "Visible source text from the captured frame."
    receipt = _inspection_receipt()

    def inspect():
        calls.append("read")
        return {"status": "observed", "reading": reading, "visual_inspection_receipt": receipt}

    fixture.adapter.inspect_visual = inspect
    response = list(fixture.run())[-1]["response"]
    assert fixture.executions == [("native_ui", {"operation": "inspect_visual"})]
    assert calls == ["read"] and fixture.adapter.calls == []
    second = fixture.packets[1]["last_result"]
    assert second["reading"] == reading and second["visual_inspection_receipt"] == receipt
    assert second["evidence_authority"] == "inspection" and "effect_receipt" not in second
    assert response["portal"]["safe_tools_executed"][0]["ok"] is True
    assert response["portal"]["native_task"]["task_verified"] is False


@pytest.mark.parametrize("status", ["unavailable", "stale", "observed"])
def test_unaccepted_or_empty_reader_output_cannot_reset_failure_count(status):
    fixture = Fixture([{"kind": "visual_inspection"}] * 3)

    def inspect():
        return {"status": status, "reading": "", "visual_inspection_receipt": {"authority": "untrusted_tool_observation"}}

    fixture.adapter.inspect_visual = inspect
    response = list(fixture.run())[-1]["response"]
    assert len(fixture.executions) == 3 and fixture.adapter.calls == []
    assert response["portal"]["native_task"]["status"] == "blocked"
    assert all(trace["ok"] is False for trace in response["portal"]["safe_tools_executed"])


@pytest.mark.parametrize("code", ["native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"])
def test_budget_failure_remains_fatal_and_arbitrary_exception_detail_is_not_logged(monkeypatch, code):
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)

    def exhausted():
        raise NativeUiUnavailable(code)

    adapter.planning_observe = exhausted
    with pytest.raises(NativeUiUnavailable) as error:
        adapter.inspect_visual()
    assert error.value.code == code
    adapter, _state, _capture, _counts, _calls, changed, _response = _reader(monkeypatch)

    def unavailable(_url, _kwargs):
        raise httpx.ReadTimeout("private endpoint or request details")

    changed["http"] = unavailable
    result = adapter.inspect_visual()
    assert result["status"] == "unavailable" and "private" not in json.dumps(result)


def test_store_late_cancellation_discards_completed_reading():
    fixture = Fixture([])
    owned = fixture.store._owned("session", "desktop")

    def late():
        owned.cancel.set()
        return {"status": "observed", "reading": "Late text must not enter planner context"}

    fixture.adapter.inspect_visual = late
    result = fixture.store.act("session", {"surface": "desktop", "operation": "inspect_visual"})
    assert result["status"] == "cancelled" and "reading" not in result
    assert fixture.adapter.calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_real_portal_profile_hands_reader_tool_evidence_to_next_compiler_only(tmp_path, monkeypatch, stream):
    _store, adapters = _install_store(monkeypatch)
    reads, requests = [], []

    def inspect(_adapter):
        reads.append("read")
        return {"status": "observed", "reading": "Visible source text.",
                "visual_inspection_receipt": _inspection_receipt()}

    monkeypatch.setattr(Adapter, "inspect_visual", inspect, raising=False)

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "tools" not in payload and "tool_choice" not in payload
        assert all(not any(key in message for key in ("images", "audios", "videos")) for message in payload["messages"])
        packet = json.loads(payload["messages"][1]["content"])
        assert any(branch["properties"]["kind"]["const"] == "visual_inspection" for branch in packet["response_schema"]["oneOf"])
        if len(requests) == 1:
            plan = {"kind": "visual_inspection"}
        else:
            assert packet["last_result"]["reading"] == "Visible source text."
            assert packet["last_result"]["visual_inspection_receipt"]["authority"] == "untrusted_tool_observation"
            assert "current_visual_input" not in packet["last_result"] and "input_transcript" not in packet["last_result"]
            plan = {"kind": "complete", "message": "Unverified task claim"}
        return httpx.Response(200, json={"message": {"content": json.dumps(plan)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _events = _post(app.test_client(), _request(stream), stream)
    assert reads == ["read"] and len(requests) == 2 and all(adapter.calls == [] for adapter in adapters)
    trace = result["portal"]["safe_tools_executed"][0]
    assert trace["arguments"] == {"operation": "inspect_visual"} and trace["ok"] is True
    assert result["portal"]["native_task"]["task_verified"] is False


@pytest.mark.parametrize("change", [False, True])
def test_namespaced_provider_proof_stays_private_and_changed_proof_withholds_read(monkeypatch, change):
    _reader_adapter, _state, capture, counts, _http_calls, changed, response = _reader(monkeypatch)
    fixture, proof, provider_calls, geometry = namespace_fixture(monkeypatch)
    fixture.window.children.remove(fixture.secret)
    adapter = fixture.adapter
    adapter.pointing_url, adapter.pointing_model, adapter.pointing_revision = (
        "http://127.0.0.1:8940", PHOTON_MODEL, PHOTON_REVISION)

    def screenshot(state):
        value = copy.deepcopy(capture)
        value["geometry"] = geometry
        value["key"] = hashlib.sha256((state["revision"] + value["sha256"]).encode()).hexdigest()
        return value

    adapter._visual_capture = screenshot
    if change:
        def post(url, _kwargs):
            proof["digest"] = "b" * 64
            return httpx.Response(200, json=response, request=httpx.Request("POST", url))
        changed["http"] = post
    result = adapter.inspect_visual()
    assert result["status"] == ("stale" if change else "observed")
    assert counts["http"] == 1 and len(provider_calls) >= 6
    assert "never-model-facing" not in json.dumps(result) and "_namespace_provider_digest" not in json.dumps(result)
    if change:
        assert "reading" not in result
    else:
        assert len(provider_calls) == 8 and adapter._provider_digest == proof["digest"]
