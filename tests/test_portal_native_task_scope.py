"""Portal-only scope transport, epoch lifecycle and portable ABI controls."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_task_scope import SCOPE
from test_portal_native_task_routes import (
    HEADERS,
    Adapter,
    _config,
    _phase,
    _post,
    _request,
    _working_reply,
)

import portal.tools as tools_module
from portal.app import SESSION_COOKIE_NAME, create_app
from portal.computer_use import ComputerUseStore
from portal.native_ui import native_task_scope_sha256, normalize_native_task_scope
from qwen_omni_adapters.computer_use import ComputerUseError
from qwen_omni_adapters.contract import OmniAdapterError, parse_adapter_request
from runtime.adapter_server import create_app as adapter_app


class ScopedAdapter(Adapter):
    def __init__(self, session):
        super().__init__(session)
        self.scope, self.closed = None, False

    def set_window_scope(self, value):
        self.scope = normalize_native_task_scope(value)

    def _state(self, include_windows=False):
        state = super()._state(include_windows)
        if self.scope is not None:
            state.update(window_scope_sha256=native_task_scope_sha256(self.scope),
                         window_scope_count=len(self.scope["windows"]))
        return state

    def close(self):
        self.closed = True


def _store(monkeypatch):
    adapters = []

    def factory(session, _surface):
        adapter = ScopedAdapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)
    return store, adapters


@pytest.mark.parametrize("stream", [False, True])
def test_scope_reaches_owned_epoch_and_model_gets_only_digest_count(tmp_path, monkeypatch, stream):
    store, adapters = _store(monkeypatch)
    packets, events = [], []
    session = "scoped-native-route-session-123456"
    body = _request(stream, think=True)
    body["portal_native_task_scope"] = copy.deepcopy(SCOPE)

    def upstream(request):
        payload = json.loads(request.content)
        packets.append(payload)
        assert payload["think"] is True and payload["options"] == body["options"]
        assert "portal_native_task_scope" not in payload
        assert "process_id" not in request.content.decode() and "process_start" not in request.content.decode()
        current = json.loads(payload["messages"][1]["content"])["current_state"]["state"]
        assert current["window_scope_sha256"] == native_task_scope_sha256(SCOPE)
        assert current["window_scope_count"] == 1
        events.append(store._owned(session, "desktop").cancel)
        plan = _phase() if len(packets) == 1 else {"kind": "blocked",
            "reason": "Scoped commit observed; independent task verification is still needed."}
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(request, plan))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, session)
    result, _ = _post(client, body, stream)
    assert len(packets) == 2 and len(adapters[0].calls) == 2
    assert [action["verb"] for action in adapters[0].calls] == ["set_value", "invoke"]
    assert adapters[0].value == adapters[0].saved == "Reference note"
    trace = result["portal"]["safe_tools_executed"][0]
    assert trace["arguments"]["operation"] == "verified_prefix"
    receipt = json.loads(trace["result"])
    assert receipt["status"] == "observed_unverified"
    assert receipt["phase_handoff"]["terminal_verified"] is False
    assert receipt["phase_handoff"]["goal_verified"] is False
    assert receipt["task_progress"] is False
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert result["portal"]["native_task"]["window_scope_sha256"] == native_task_scope_sha256(SCOPE)
    assert result["portal"]["native_task"]["task_verified"] is False
    assert adapters[0].scope == SCOPE and adapters[0].closed and all(event.is_set() for event in events)
    assert (session, "desktop") not in store._sessions
    assert "portal_native_task_scope" not in "".join(path.read_text() for path in (tmp_path / "sessions").rglob("*.json*"))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad", [None, {}, {"schema": "wrong", "windows": []}, {**SCOPE, "query": "not authority"}])
def test_invalid_scope_rejected_before_adapter_or_native_allocation(tmp_path, monkeypatch, stream, bad):
    _owned, adapters = _store(monkeypatch)
    requests = []
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request))))
    body = _request(stream)
    body["portal_native_task_scope"] = bad
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", json=body, headers=HEADERS)
    assert response.status_code == 400 and adapters == requests == []


@pytest.mark.parametrize("stream", [False, True])
def test_orphan_scope_rejected_in_ordinary_portal_turn(tmp_path, monkeypatch, stream):
    _owned, adapters = _store(monkeypatch)
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(lambda _request: pytest.fail("No upstream"))))
    body = _request(stream)
    body.pop("portal_execution_profile")
    body["portal_native_task_scope"] = SCOPE
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", json=body, headers=HEADERS)
    assert response.status_code == 400 and adapters == []


def test_cancelled_scope_cannot_be_renewed_by_direct_observe_or_planning(monkeypatch):
    store, adapters = _store(monkeypatch)
    event = store.begin_task("session", window_scope=SCOPE)
    store.act("session", {"operation": "cancel", "surface": "desktop"})
    assert event.is_set()
    assert store.act("session", {"operation": "observe", "surface": "desktop"}) == {"status": "cancelled"}
    assert store.planning_observation("session") == {"status": "cancelled"}
    assert len(adapters) == 1 and store._sessions[("session", "desktop")].window_scope == SCOPE
    with pytest.raises(ComputerUseError):
        store.act("session", {"operation": "observe", "surface": "desktop", "window_scope": SCOPE})
    store.end_task("session", "desktop", event)
    assert adapters[0].closed
    assert store.act("session", {"operation": "observe", "surface": "desktop"}).get("state")
    assert len(adapters) == 2 and adapters[1].scope is None


def test_old_task_end_never_cancels_new_epoch_or_browser(monkeypatch):
    store, adapters = _store(monkeypatch)
    browser = store._owned("session", "browser")
    first = store.begin_task("session", window_scope=SCOPE)
    second = store.begin_task("session", window_scope=SCOPE)
    store.end_task("session", "desktop", first)
    assert first.is_set() and not second.is_set()
    assert store._owned("session", "desktop").cancel is second
    assert store._owned("session", "browser") is browser and not browser.cancel.is_set()
    assert adapters[1].closed and not adapters[2].closed


def test_scoped_stream_disconnect_releases_exact_epoch_before_any_planning(tmp_path, monkeypatch):
    store, adapters = _store(monkeypatch)
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(lambda _request: pytest.fail("No planner call"))))
    client = app.test_client()
    session = "scope-disconnect-session-123456"
    client.set_cookie(SESSION_COOKIE_NAME, session)
    body = _request(True)
    body["portal_native_task_scope"] = SCOPE
    response = client.post("/api/chat/stream", json=body, headers=HEADERS, buffered=False)
    event = store._owned(session, "desktop").cancel
    assert json.loads(next(iter(response.response)))["type"] == "progress"
    response.close()
    assert event.is_set() and adapters[0].closed and (session, "desktop") not in store._sessions
    activity = client.get("/api/activity", headers=HEADERS).json
    assert activity["active"] == activity["inflight"] == activity["queued"] == 0


def test_scope_ttl_and_trash_latch_old_epoch_and_never_resume_it(monkeypatch):
    store, adapters = _store(monkeypatch)
    event = store.begin_task("session", window_scope=SCOPE)
    store._sessions[("session", "desktop")].last_seen -= 1000
    with store.task_scope("session", "desktop", event):
        assert store.planning_observation("session") == {"status": "cancelled"}
        assert store.act("session", {"operation": "observe", "surface": "desktop"}) == {"status": "cancelled"}
    assert event.is_set() and adapters[0].closed
    newer = store.begin_task("session", window_scope=SCOPE)
    store.clear("session")
    with store.task_scope("session", "desktop", newer):
        assert store.planning_observation("session") == {"status": "cancelled"}
    assert newer.is_set() and ("session", "desktop") not in store._sessions


@pytest.mark.parametrize("stream", [False, True])
def test_portable_adapter_rejects_reserved_scope_before_transport(stream):
    body = _request(stream)
    for key in ("portal_auto_tools", "portal_execution_profile", "portal_tool_allowlist"):
        body.pop(key)
    body["portal_native_task_scope"] = SCOPE
    app = adapter_app(client=httpx.Client(transport=httpx.MockTransport(lambda _request: pytest.fail("No model call"))))
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", json=body)
    assert response.status_code == 400 and "portal-only" in response.json["error"]
    body["stream"] = False
    with pytest.raises(OmniAdapterError, match="portal-only"):
        parse_adapter_request(body)
    schema = json.loads(Path("docs/schema/request-v1.schema.json").read_text())
    assert not Draft202012Validator(schema).is_valid(body)
