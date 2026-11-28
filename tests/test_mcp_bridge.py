from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from portal.mcp_bridge import MCPBridgeError, RemoteMCPRegistry


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(405)
        self.end_headers()

    def do_DELETE(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        auth = self.server.expected_auth
        if auth and self.headers.get(auth[0]) != auth[1]:
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "id" not in value:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        method, arguments = value["method"], value.get("params", {})
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {"listChanged": True}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "synthetic-fixture", "version": "1"}}
        elif method == "tools/list":
            tool = {"name": "actual-name/with punctuation", "description": "Store a supplied label in the fixture.",
                "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False},
                "annotations": {"readOnlyHint": False}, "outputSchema": {"type": "object"}}
            if self.server.changed:
                tool["inputSchema"]["required"].append("version")
                tool["inputSchema"]["properties"]["version"] = {"type": "integer"}
            result = {"tools": [tool], "nextCursor": "page-two"} if not arguments.get("cursor") else {"tools": [
                {"name": "inspect", "description": "Inspect the fixture", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            self.server.effects.append(arguments.get("arguments"))
            result = {"content": [{"type": "text", "text": json.dumps({"stored": arguments["arguments"], "echo": "fixture-key"})}],
                "structuredContent": {"stored": arguments["arguments"]}, "isError": self.server.tool_error}
        elif method == "resources/read":
            result = {"contents": [{"uri": arguments["uri"], "mimeType": "text/plain", "text": "actual resource body"}]}
        elif method == "resources/list":
            result = {"resources": [{"uri": "fixture://body", "name": "body"}]}
        elif method == "resources/templates/list":
            result = {"resourceTemplates": []}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "inspect", "arguments": []}]}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": "untrusted remote prompt"}}]}
        else:
            result = {}
        raw = json.dumps({"jsonrpc": "2.0", "id": value["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "session-fixture")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.expected_auth, server.effects, server.changed, server.tool_error = None, [], False, False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}/mcp"
    server.shutdown()
    server.server_close()
    thread.join(2)


@pytest.mark.parametrize("auth", [None, ("Authorization", "Bearer fixture-key"), ("x-api-key", "fixture-key")])
def test_endpoint_intake_discovers_every_page_calls_exact_tool_and_reads_resource(tmp_path, endpoint, auth):
    server, url = endpoint
    server.expected_auth = auth
    registry = RemoteMCPRegistry(tmp_path)
    try:
        registered = registry.register("owner", url, key="fixture-key" if auth else "")
        identity = registered["server"]["id"]
        assert registered["tools"] == 2
        catalog = registry.catalog("owner", identity)
        assert catalog["total"] == 2 and catalog["items"][0]["outputSchema"] == {"type": "object"}
        result = registry.call("owner", identity, "actual-name/with punctuation", {"label": "verified label"})
        assert result["ok"] is True and server.effects == [{"label": "verified label"}]
        if auth:
            assert "fixture-key" not in json.dumps(result)
            assert "fixture-key" not in (tmp_path / "servers.json").read_text()
        assert registry.resource("owner", identity, uri="fixture://body")["result"]["contents"][0]["text"] == "actual resource body"
        assert registry.resource("owner", identity, prompt="inspect")["result"]["messages"][0]["content"]["text"] == "untrusted remote prompt"
        assert registry.list("stranger") == []
        registry.remove("owner", identity)
        assert registry.list("owner") == []
        if auth:
            assert json.loads((tmp_path / "credentials.json").read_text()) == {}
        for path in tmp_path.glob("*.json"):
            assert path.stat().st_mode & 0o777 == 0o600
    finally:
        registry.close()


def test_schema_changes_require_rediscovery_and_remote_errors_remain_failures(tmp_path, endpoint):
    server, url = endpoint
    registry = RemoteMCPRegistry(tmp_path)
    try:
        identity = registry.register("owner", url)["server"]["id"]
        with pytest.raises(MCPBridgeError, match="Discover"):
            registry.call("owner", identity, "actual-name/with punctuation", {"label": "missing discovery"})
        registry.catalog("owner", identity)
        with pytest.raises(MCPBridgeError, match="input schema"):
            registry.call("owner", identity, "actual-name/with punctuation", {"wrong": True})
        assert server.effects == []
        server.changed = True
        with pytest.raises(MCPBridgeError, match="Discover"):
            registry.call("owner", identity, "actual-name/with punctuation", {"label": "old", "version": 2})
        registry.catalog("owner", identity, name="actual-name/with punctuation")
        server.tool_error = True
        assert registry.call("owner", identity, "actual-name/with punctuation", {"label": "new", "version": 2})["ok"] is False
    finally:
        registry.close()


def test_persistence_refresh_trash_and_device_scope_preserve_caller_boundaries(tmp_path, endpoint):
    _server, url = endpoint
    registry = RemoteMCPRegistry(tmp_path)
    own = registry.register("one", url)["server"]["id"]
    shared = registry.register("one", url, scope="device")["server"]["id"]
    registry.catalog("one", shared)
    assert [item["id"] for item in registry.list("two")] == [shared]
    with pytest.raises(MCPBridgeError):
        registry.call("two", shared, "actual-name/with punctuation", {"label": "foreign discovery"})
    registry.clear("one")
    assert [item["id"] for item in registry.list("one")] == [shared]
    with pytest.raises(MCPBridgeError):
        registry.catalog("one", own)
    registry.close()
    restored = RemoteMCPRegistry(tmp_path)
    try:
        assert restored.refresh("two", shared)["total"] == 2
        assert restored.call("two", shared, "actual-name/with punctuation", {"label": "fresh"})["ok"] is True
    finally:
        restored.close()


@pytest.mark.parametrize("url", ["file:///tmp/socket", "https://user:key@example.test/mcp", "https://example.test/mcp?key=secret", "https://example.test/mcp#secret", "http://bad\nname/mcp"])
def test_credentials_cannot_enter_endpoint_display_or_query(tmp_path, url):
    registry = RemoteMCPRegistry(tmp_path)
    with pytest.raises(MCPBridgeError):
        registry.register("owner", url)
    assert registry.list("owner") == []
    registry.close()


def test_authenticated_api_registers_discovers_executes_and_trashes_connection_without_key_leak(tmp_path, endpoint):
    import httpx

    from portal.app import PortalConfig, create_app
    server, url = endpoint
    server.expected_auth = ("Authorization", "Bearer fixture-key")
    token = "fixture-portal-token-long-enough-for-auth"
    config = PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="", tts_health_url="http://tts/healthz", ollama_health_url="http://ollama/api/tags",
        model="fixture", access_token=token, remote_camera_enabled=False, mcp_enabled=True, mcp_root=tmp_path / "mcp",
        background_task_path=tmp_path / "state" / "background.json", virtual_context_root=tmp_path / "virtual")
    app = create_app(config, httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"ok": True}))))
    client = app.test_client()
    client.set_cookie("omni_portal_session", "browser-connection-one")
    headers = {"Authorization": "Bearer " + token}
    assert client.post("/api/mcp/servers", json={"endpoint": url}, headers={}).status_code == 401
    response = client.post("/api/mcp/servers", json={"endpoint": url, "key": "fixture-key"}, headers=headers)
    assert response.status_code == 201
    identity = response.json["server"]["id"]
    assert "fixture-key" not in response.text
    def call(name, arguments):
        response = client.post(f"/api/tools/{name}/call", json={"arguments": arguments}, headers=headers)
        assert response.status_code == 200
        return response.json["result"]
    assert call("mcp_catalog", {"server_id": identity, "kind": "tools"})["total"] == 2
    result = call("mcp_call", {"server_id": identity, "name": "actual-name/with punctuation", "arguments": {"label": "actual API effect"}})
    assert result["ok"] is True and server.effects[-1] == {"label": "actual API effect"}
    assert "fixture-key" not in json.dumps(result)
    other = app.test_client()
    other.set_cookie("omni_portal_session", "browser-connection-two")
    assert other.get("/api/mcp/servers", headers=headers).json["servers"] == []
    assert client.delete("/api/diagnostics", headers=headers).status_code == 204
    assert client.get("/api/mcp/servers", headers=headers).json["servers"] == []
    app.extensions["remote_mcp"].close()


def test_transport_rejects_cross_origin_before_forwarding_authentication():
    import asyncio

    import httpx

    from portal.mcp_bridge import _BoundedTransport
    async def check():
        transport = _BoundedTransport("https://registered.example/mcp")
        try:
            with pytest.raises(MCPBridgeError, match="different origin"):
                await transport.handle_async_request(httpx.Request("POST", "https://foreign.example/steal", headers={"Authorization": "Bearer fixture-key"}))
        finally:
            await transport.aclose()
    asyncio.run(check())


def test_camera_transport_pins_checked_address_and_rejects_private_dns(monkeypatch):
    import asyncio
    import socket

    import httpx

    from portal.mcp_bridge import _BoundedTransport

    async def check():
        origin = "https://camera-connector.example/mcp"
        transport = _BoundedTransport(origin, public_only=True)
        observed = []
        address = "1.1.1.1"

        async def addresses(host, port, *, type):
            assert (host, port, type) == ("camera-connector.example", 443, socket.SOCK_STREAM)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

        async def send(request):
            observed.append((str(request.url), request.headers["host"], request.extensions["sni_hostname"]))
            return httpx.Response(200, stream=httpx.ByteStream(b""))

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", addresses)
        monkeypatch.setattr(transport.wrapped, "handle_async_request", send)
        try:
            await transport.handle_async_request(httpx.Request("POST", origin))
            assert observed == [("https://1.1.1.1/mcp", "camera-connector.example", "camera-connector.example")]
            address = "127.0.0.1"
            with pytest.raises(MCPBridgeError, match="Private camera"):
                await transport.handle_async_request(httpx.Request("POST", origin))
            assert len(observed) == 1
        finally:
            await transport.aclose()

    asyncio.run(check())


def test_camera_registration_is_private_persistent_and_only_discovers(tmp_path, endpoint, monkeypatch):
    import httpx

    from portal.app import PortalConfig, create_app

    server, url = endpoint
    server.expected_auth = ("x-api-key", "fixture-key")
    token = "fixture-portal-token-long-enough-for-camera"
    config = PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="", tts_health_url="http://tts/healthz", ollama_health_url="http://ollama/api/tags",
        model="fixture", access_token=token, remote_camera_enabled=False, mcp_enabled=True, mcp_root=tmp_path / "mcp",
        background_task_path=tmp_path / "state" / "background.json", virtual_context_root=tmp_path / "virtual")
    app = create_app(config, httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"ok": True}))))
    registry = app.extensions["remote_mcp"]
    owner = hashlib.sha256(b"omni-voice-session\0" + token.encode()).hexdigest()
    headers = {"Authorization": "Bearer " + token}
    client = app.test_client()
    camera = "camera-0123456789abcdef"
    payload = {"camera_id": camera, "payload": json.dumps({"endpoint": url, "key": "fixture-key"})}
    try:
        response = client.post("/api/voice/camera-qr", json=payload, headers=headers)
        assert response.status_code == 403
        client.set_cookie("omni_portal_session", owner)
        assert client.post("/api/voice/camera-qr", json=payload, headers=headers).json["kind"] == "mcp_failed"
        assert registry.list(owner) == []
        assert not (tmp_path / "mcp" / "credentials.json").exists()
        monkeypatch.setenv("OMNI_CALL_QR_TRUSTED_MCP_ORIGINS", url.removesuffix("/mcp"))
        response = client.post("/api/voice/camera-qr", json=payload, headers=headers)
        assert response.json["kind"] == "mcp_ready" and "fixture-key" not in response.text
        assert server.effects == []
        assert client.post("/api/voice/camera-qr", json=payload, headers=headers).json["kind"] == "mcp_known"
        assert client.post("/api/voice/camera-qr", json={"camera_id": camera, "payload": "A hello from the camera"}, headers=headers).json["kind"] == "text"
        receipt = client.post("/api/tools/camera_qr/call", json={"arguments": {}}, headers=headers)
        assert receipt.status_code == 200 and "A hello from the camera" in receipt.text
        assert "fixture-key" not in receipt.text
        server_id = registry.list(owner)[0]["id"]
        assert registry.list("another-browser") == []
        assert registry.catalog(owner, server_id)["total"] == 2
        assert registry.call(owner, server_id, "actual-name/with punctuation", {"label": "requested"})["ok"]
        assert server.effects == [{"label": "requested"}]
        assert (tmp_path / "mcp" / "credentials.json").stat().st_mode & 0o777 == 0o600
        assert "fixture-key" not in (tmp_path / "mcp" / "servers.json").read_text()
    finally:
        registry.close()
    restored = RemoteMCPRegistry(tmp_path / "mcp", native_session="rotated-owner")
    try:
        assert restored.list(owner) == []
        assert restored.list("rotated-owner")[0]["id"] == server_id
        assert restored.catalog("rotated-owner", server_id)["total"] == 2
    finally:
        restored.close()


def test_legacy_sse_transport_is_real_get_endpoint_and_preserves_resource_text(tmp_path):
    import queue
    class Legacy(_Handler):
        def do_GET(self):
            self.server.opened.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(b"event: endpoint\r\ndata: /messages\r\n\r\n")
                self.wfile.flush()
                while not self.server.done.is_set():
                    try:
                        value = self.server.responses.get(timeout=0.1)
                        data = b"event: message\r\ndata: " + json.dumps(value).encode() + b"\r\n\r\n"
                    except queue.Empty:
                        data = b": heartbeat\r\n\r\n"
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != "/messages":
                self.send_response(405)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "id" in value:
                if value["method"] == "initialize":
                    result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}, "resources": {}}, "serverInfo": {"name": "legacy-fixture", "version": "1"}}
                elif value["method"] == "resources/read":
                    result = {"contents": [{"uri": value["params"]["uri"], "text": "legacy resource body", "mimeType": "text/plain"}]}
                else:
                    result = {"tools": []}
                self.server.responses.put({"jsonrpc": "2.0", "id": value["id"], "result": result})
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Legacy)
    server.responses, server.done, server.opened = queue.Queue(), threading.Event(), threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    registry = RemoteMCPRegistry(tmp_path)
    try:
        result = registry.register("owner", f"http://127.0.0.1:{server.server_port}/sse")
        assert result["server"]["transport"] == "sse" and server.opened.is_set()
        content = registry.resource("owner", result["server"]["id"], uri="fixture://legacy")
        assert content["source_body_received"] is True
        assert content["result"]["contents"][0]["text"] == "legacy resource body"
    finally:
        registry.close()
        server.done.set()
        server.shutdown()
        server.server_close()
