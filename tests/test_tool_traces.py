from __future__ import annotations

import json

import httpx

from portal.app import PortalConfig, _record_tool_diagnostics, create_app
from portal.tool_traces import SessionToolTraces


def test_call_contents_are_scoped_and_late_start_cannot_replace_output():
    store = SessionToolTraces()
    store.begin("alice", "request")
    start = {"id": "call", "name": "mcp_call", "arguments": {"query": "sample"}, "status": "running"}
    store.record("alice", "request", 1, [start])
    assert store.get("alice", "request", 1, "call")["ok"] is None
    store.record("alice", "request", 1, [{**start, "status": "complete", "ok": False, "result": '{"error":"observed failure"}'}])
    store.record("alice", "request", 1, [start])
    result = store.get("alice", "request", 1, "call")
    assert result["ok"] is False and "observed failure" in result["output"]
    assert json.loads(result["arguments"]) == {"query": "sample"}
    assert store.get("bob", "request", 1, "call") is None


def test_credentials_media_and_large_results_have_explicit_bounded_inspection():
    store = SessionToolTraces(redactions=("actual-access-secret",))
    store.begin("alice", "request")
    store.record("alice", "request", 1, [{"id": "call", "name": "capture_view", "status": "complete", "ok": True,
        "arguments": {"api_key": "private", "text": "actual-access-secret"},
        "result": json.dumps({"screenshot": {"mime_type": "image/png", "data": "raw-media"}, "text": "x" * 30000})}])
    result = store.get("alice", "request", 1, "call")
    encoded = json.dumps(result)
    assert "private" not in encoded and "actual-access-secret" not in encoded and "raw-media" not in encoded
    assert result["output_truncated"] is True and len(result["output"]) == 24000


def test_ttl_capacity_and_trash_do_not_retain_or_resurrect_call_contents():
    stamp = [0.0]
    store = SessionToolTraces(ttl_s=5, max_calls=1, clock=lambda: stamp[0])
    store.begin("alice", "request")
    store.record("alice", "request", 1, [{"id": "first", "name": "first"}, {"id": "second", "name": "second"}])
    assert store.get("alice", "request", 1, "first") is None
    stamp[0] = 6
    assert store.get("alice", "request", 1, "second") is None
    store.clear("alice")
    store.record("alice", "request", 2, [{"id": "late", "name": "late"}])
    assert store.get("alice", "request", 2, "late") is None
    store.begin("alice", "new-request")
    store.record("alice", "new-request", 1, [{"id": "new", "name": "new"}])
    assert store.get("alice", "new-request", 1, "new") is not None


def test_live_inspection_api_is_authenticated_isolated_and_diagnostics_stay_content_free(tmp_path):
    token = "fixture-token-that-is-not-a-deployment-secret"
    config = PortalConfig(access_token=token, model="fixture", remote_camera_enabled=False,
        adapter_url="http://adapter/api/chat", adapter_health_url="", comprehension_health_url="",
        tts_health_url="", ollama_health_url="",
        background_task_path=tmp_path / "background.json", virtual_context_root=tmp_path / "context")
    app = create_app(config, httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"ok":True}))))
    store = app.extensions["tool_traces"]
    class Diagnostics:
        events = []
        def record(self, _session, event, fields, **_kwargs):
            self.events.append({"event":event, **fields})
    diagnostics = Diagnostics()
    store.begin("browser-inspection-alice", "request")
    _record_tool_diagnostics(diagnostics, "browser-inspection-alice", "request", 2, "completed", [{"id":"call", "name":"mcp_call", "arguments":{"query":"private input"}, "result":"private output", "ok":True}], details_store=store)
    assert "private" not in json.dumps(diagnostics.events)
    assert diagnostics.events[0]["tool_call_id"] == "call"
    client = app.test_client()
    client.set_cookie("omni_portal_session", "browser-inspection-alice")
    path = "/api/observatory/tool-call?request_id=request&round=2&call_id=call"
    assert client.get(path).status_code == 401
    headers = {"Authorization":"Bearer "+token}
    response = client.get(path, headers=headers)
    assert response.json["output"] == "private output" and response.headers["Cache-Control"] == "no-store"
    client.set_cookie("omni_portal_session", "browser-inspection-bob")
    assert client.get(path, headers=headers).status_code == 404
    client.set_cookie("omni_portal_session", "browser-inspection-alice")
    assert client.delete("/api/diagnostics", headers=headers).status_code == 204
    assert client.get(path, headers=headers).status_code == 404
