"""Private, bounded MCP registration and per-caller protocol sessions."""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import ipaddress
import json
import os
import queue
import secrets
import socket
import threading
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from referencing import Registry

MAX_RESPONSE = 8 * 1024 * 1024
MAX_TOOLS = 1024
CAMERA_OWNER = "native-camera"


class MCPBridgeError(ValueError):
    """A credential-free protocol or registration failure."""


class _BoundedStream(httpx.AsyncByteStream):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    async def __aiter__(self):
        total = 0
        async for chunk in self.wrapped:
            total += len(chunk)
            if total > MAX_RESPONSE:
                raise MCPBridgeError("MCP response exceeds the byte limit")
            yield chunk

    async def aclose(self):
        await self.wrapped.aclose()


class _BoundedTransport(httpx.AsyncBaseTransport):
    def __init__(self, endpoint, *, public_only=False):
        self.origin = httpx.URL(endpoint).copy_with(path="/", query=None, fragment=None)
        self.public_only = public_only
        self.wrapped = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        if request.url.copy_with(path="/", query=None, fragment=None) != self.origin:
            raise MCPBridgeError("MCP transport cannot send credentials to a different origin")
        if self.public_only:
            hostname = request.url.raw_host.decode("ascii")
            port = request.url.port or (443 if request.url.scheme == "https" else 80)
            try:
                records = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
                    hostname, port, type=socket.SOCK_STREAM), 3)
                addresses = [ipaddress.ip_address(record[4][0]) for record in records]
            except (OSError, ValueError, TimeoutError):
                raise MCPBridgeError("MCP camera endpoint could not be resolved") from None
            if not addresses or any(not address.is_global or address.is_multicast for address in addresses):
                raise MCPBridgeError("Private camera MCP origins require local configuration")
            # Connect to exactly the address just checked. Preserve the original
            # HTTP Host and TLS certificate identity through httpcore's SNI hook.
            request = httpx.Request(request.method, request.url.copy_with(host=str(addresses[0])),
                headers=request.headers, stream=request.stream,
                extensions={**request.extensions, "sni_hostname": hostname})
        response = await self.wrapped.handle_async_request(request)
        return httpx.Response(response.status_code, headers=response.headers,
            stream=_BoundedStream(response.stream), extensions=response.extensions)

    async def aclose(self):
        await self.wrapped.aclose()


class _Peer:
    """One thread owns a persistent SDK session and its task groups."""
    def __init__(self, config: Mapping[str, Any], key: str = "", *, timeout: float = 60):
        self.config, self.key, self.timeout = dict(config), key, timeout
        self.commands: queue.Queue = queue.Queue(maxsize=8)
        self.ready: concurrent.futures.Future = concurrent.futures.Future()
        self.loop = None
        self.task = None
        self.closed = False
        self.dirty = True
        self.last_used = time.monotonic()
        self.thread = threading.Thread(target=self._thread, name="omni-mcp-session", daemon=True)
        self.thread.start()

    def _thread(self):
        try:
            asyncio.run(self._run())
        except BaseException:
            if not self.ready.done():
                self.ready.set_exception(MCPBridgeError("MCP handshake or authentication failed"))
        finally:
            self.closed = True
            while True:
                try:
                    _operation, _arguments, future = self.commands.get_nowait()
                    if not future.done():
                        future.set_exception(MCPBridgeError("MCP connection closed; rediscover before continuing"))
                except queue.Empty:
                    break

    async def _run(self):
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client

        self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
        headers = {}
        if self.key:
            headers = {"Authorization": f"Bearer {self.key}"} if self.config["auth"] == "bearer" else {"x-api-key": self.key}
        timeout = httpx.Timeout(self.timeout, connect=min(5, self.timeout))
        def client_for_endpoint(**_kwargs):
            return httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False,
                transport=_BoundedTransport(self.config["endpoint"], public_only=self.config.get("public_only", False)), trust_env=False)

        async with client_for_endpoint() as client:
            transport = (sse_client(self.config["endpoint"], headers=headers, timeout=min(5, self.timeout),
                sse_read_timeout=self.timeout, httpx_client_factory=client_for_endpoint)
                if self.config["transport"] == "sse" else streamable_http_client(self.config["endpoint"], http_client=client))
            async with transport as streams:
                async def notification(message):
                    try:
                        value = message.model_dump(mode="json")
                    except (AttributeError, TypeError):
                        return
                    if "notifications/tools/list_changed" in json.dumps(value):
                        self.dirty = True
                async with ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=self.timeout),
                        message_handler=notification) as session:
                    initialization = await asyncio.wait_for(session.initialize(), min(10, self.timeout))
                    self.info = initialization.model_dump(mode="json", by_alias=True, exclude_none=True)
                    self.ready.set_result(True)
                    while True:
                        try:
                            command = self.commands.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.02)
                            continue
                        if command is None:
                            return
                        operation, arguments, future = command
                        self.last_used = time.monotonic()
                        if future.cancelled():
                            continue
                        try:
                            result = await asyncio.wait_for(self._execute(session, operation, arguments), self.timeout)
                            if not future.done():
                                future.set_result(result)
                        except BaseException as error:
                            if not future.done():
                                future.set_exception(MCPBridgeError("MCP operation failed; no automatic action retry was made"))
                            if isinstance(error, (asyncio.CancelledError, asyncio.TimeoutError)):
                                raise

    async def _execute(self, session, operation, arguments):
        if operation == "call":
            result = await session.call_tool(arguments["name"], arguments.get("arguments", {}),
                read_timeout_seconds=timedelta(seconds=self.timeout))
        elif operation == "read":
            result = await session.read_resource(arguments["uri"])
        elif operation == "prompt":
            result = await session.get_prompt(arguments["name"], arguments.get("arguments"))
        else:
            methods = {"tools": session.list_tools, "resources": session.list_resources,
                "templates": session.list_resource_templates, "prompts": session.list_prompts}
            method = methods[operation]
            items, cursor, cursors = [], None, set()
            field = {"tools": "tools", "resources": "resources", "templates": "resourceTemplates", "prompts": "prompts"}[operation]
            for _ in range(32):
                page = (await method(cursor=cursor)).model_dump(mode="json", by_alias=True, exclude_none=True)
                items.extend(page.get(field, []))
                if len(items) > MAX_TOOLS:
                    raise MCPBridgeError("MCP catalog exceeds 1024 entries; use a narrower server catalog")
                cursor = page.get("nextCursor")
                if not cursor:
                    self.dirty = False
                    return {"items": items, "complete": True}
                if cursor in cursors:
                    raise MCPBridgeError("MCP catalog repeats its pagination cursor")
                cursors.add(cursor)
            raise MCPBridgeError("MCP catalog exceeds 32 pages")
        value = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_RESPONSE:
            raise MCPBridgeError("MCP result exceeds the byte limit")
        return value

    def request(self, operation: str, arguments: Mapping[str, Any] | None = None):
        if self.closed:
            raise MCPBridgeError("MCP connection closed; rediscover before continuing")
        try:
            self.ready.result(timeout=self.timeout + 1)
            future: concurrent.futures.Future = concurrent.futures.Future()
            self.commands.put_nowait((operation, dict(arguments or {}), future))
            return future.result(timeout=self.timeout + 1)
        except (concurrent.futures.TimeoutError, queue.Full):
            self.close()
            raise MCPBridgeError("MCP operation timed out; action outcome is unverified") from None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.loop is not None and self.task is not None:
            self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=2)


class RemoteMCPRegistry:
    def __init__(self, root: Path, *, peer_factory=_Peer, idle_ttl: float = 300, native_session: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.peer_factory, self.idle_ttl = peer_factory, idle_ttl
        self._lock = threading.RLock()
        self._camera_lock = threading.Lock()
        self.native_session = native_session
        self._peers: dict[tuple[str, str], _Peer] = {}
        self._seen: dict[tuple[str, str], dict[str, str]] = {}
        self._config = self._load("servers.json", {"servers": []})
        self._secrets = self._load("credentials.json", {})

    def _load(self, name, default):
        path = self.root / name
        if not path.exists():
            return default
        if path.stat().st_size > 2 * MAX_RESPONSE:
            raise MCPBridgeError("MCP configuration exceeds its size bound")
        path.chmod(0o600)
        return json.loads(path.read_text())

    def _write(self, name, value):
        target = self.root / name
        temporary = self.root / f".{name}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _redact(self, value):
        with self._lock:
            tokens = [token for token in self._secrets.values() if token]
        def clean(item):
            if isinstance(item, str):
                for token in tokens:
                    item = item.replace(token, "[credential redacted]")
                return item
            if isinstance(item, list):
                return [clean(child) for child in item]
            if isinstance(item, Mapping):
                return {clean(key): clean(child) for key, child in item.items()}
            return item
        return clean(value)

    @staticmethod
    def _owner(session):
        return hashlib.sha256(session.encode()).hexdigest()

    def _visible_owners(self, session):
        owners = {"device", self._owner(session)}
        if self.native_session and secrets.compare_digest(session, self.native_session):
            owners.add(CAMERA_OWNER)
        return owners

    def _server(self, session, identity):
        owners = self._visible_owners(session)
        with self._lock:
            current = next((item for item in self._config["servers"] if item["id"] == identity and item["owner"] in owners), None)
        if current is None:
            raise MCPBridgeError("MCP connection is not registered in this scope")
        return dict(current)

    @staticmethod
    def _endpoint(endpoint):
        if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 2048 or any(char.isspace() or ord(char) < 32 for char in endpoint):
            raise MCPBridgeError("Provide an HTTP or HTTPS MCP endpoint")
        try:
            parts = urlsplit(endpoint)
            _port = parts.port
        except ValueError:
            raise MCPBridgeError("Provide a valid HTTP or HTTPS MCP endpoint") from None
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment):
            raise MCPBridgeError("Endpoint must be HTTP(S), without userinfo, query credentials or fragments; provide the key separately")
        return parts

    def register_camera(self, session: str, endpoint: str, *, key: str = "", trusted_origins=()):
        if CAMERA_OWNER not in self._visible_owners(session):
            raise MCPBridgeError("Camera registration belongs to the native voice owner")
        parts = self._endpoint(endpoint)
        try:
            endpoint = str(httpx.URL(endpoint))
            origin = str(httpx.URL(endpoint).copy_with(path="/"))
        except httpx.InvalidURL:
            raise MCPBridgeError("Provide a valid HTTP or HTTPS MCP endpoint") from None
        trusted = origin in trusted_origins
        if not trusted and (parts.scheme != "https" or parts.hostname == "localhost" or parts.hostname.endswith((".local", ".localhost"))):
            raise MCPBridgeError("Camera MCP requires public HTTPS or a configured trusted origin")
        try:
            address = ipaddress.ip_address(parts.hostname)
        except ValueError:
            address = None
        if not trusted and address is not None and (not address.is_global or address.is_multicast):
            raise MCPBridgeError("Private camera MCP origins require local configuration")
        if not self._camera_lock.acquire(blocking=False):
            raise MCPBridgeError("A camera MCP registration is already in progress")
        try:
            with self._lock:
                previous = next((s for s in self._config["servers"] if s["owner"] == CAMERA_OWNER and s["endpoint"] == endpoint), None)
                if previous is not None:
                    if key and not secrets.compare_digest(key.encode(), self._secrets.get(previous["secret_ref"], "").encode()):
                        raise MCPBridgeError("An existing camera connection has different authentication; remove it to replace the key")
                    return {"registered": True, "existing": True, "server": self._redact(self._public(previous))}
            return self._register(session, endpoint, key=key, name=parts.hostname,
                                  owner=CAMERA_OWNER, public_only=not trusted, timeout=8)
        finally:
            self._camera_lock.release()

    def register(self, session: str, endpoint: str, *, key: str = "", name: str = "", scope: str = "session"):
        if scope not in {"session", "device"}:
            raise MCPBridgeError("Invalid connection scope or API key")
        owner = "device" if scope == "device" else self._owner(session)
        return self._register(session, endpoint, key=key, name=name, owner=owner)

    def _register(self, session, endpoint, *, key, name, owner, public_only=False, timeout=60):
        parts = self._endpoint(endpoint)
        if not isinstance(key, str) or len(key) > 8192 or any(c in key for c in "\r\n"):
            raise MCPBridgeError("Invalid connection scope or API key")
        key = key.removeprefix("Bearer ").strip()
        name = str(name or parts.hostname)[:100]
        identity, secret_ref = secrets.token_hex(12), secrets.token_hex(16) if key else None
        with self._lock:
            if len(self._config["servers"]) >= 32:
                raise MCPBridgeError("At most 32 MCP connections may be registered")
        candidate, peer, catalog = None, None, None
        try:
            for transport in ("http", "sse"):
                for auth in (("bearer", "api_key") if key else ("none",)):
                    candidate = {"id": identity, "name": name, "endpoint": endpoint, "transport": transport,
                        "auth": auth, "secret_ref": secret_ref, "owner": owner}
                    if public_only:
                        candidate["public_only"] = True
                    peer = self.peer_factory(candidate, key, timeout=timeout)
                    try:
                        catalog = peer.request("tools")
                        break
                    except Exception:
                        peer.close()
                        peer = None
                if peer is not None:
                    break
            if peer is None:
                raise MCPBridgeError("MCP handshake/discovery failed; check endpoint and authentication")
            self._validate_tools(catalog["items"])
            with self._lock:
                if len(self._config["servers"]) >= 32:
                    raise MCPBridgeError("MCP registration capacity reached")
                if secret_ref:
                    self._secrets[secret_ref] = key
                    self._write("credentials.json", self._secrets)
                replacement = {"servers": [*self._config["servers"], candidate]}
                self._write("servers.json", replacement)
                self._config = replacement
                self._peers[(self._owner(session), identity)] = peer
            return {"registered": True, "server": self._redact(self._public(candidate)), "tools": len(catalog["items"])}
        except Exception:
            if peer:
                peer.close()
            with self._lock:
                if secret_ref and self._secrets.pop(secret_ref, None) is not None:
                    self._write("credentials.json", self._secrets)
            raise MCPBridgeError("MCP registration failed; no usable connection was installed") from None

    @staticmethod
    def _validate_tools(items):
        names = set()
        for item in items:
            name, schema = item.get("name"), item.get("inputSchema")
            if not isinstance(name, str) or not 1 <= len(name) <= 128 or name in names or not isinstance(schema, dict):
                raise MCPBridgeError("MCP tool definitions are invalid or ambiguous")
            if len(json.dumps(item).encode()) > 131072:
                raise MCPBridgeError("MCP tool definition exceeds its size bound")
            try:
                Draft202012Validator.check_schema(schema)
            except Exception:
                raise MCPBridgeError("MCP tool advertises an invalid JSON schema") from None
            names.add(name)

    @staticmethod
    def _public(server):
        return {"id": server["id"], "name": server["name"], "transport": server["transport"],
            "authenticated": server["auth"] != "none", "scope": {"device": "device", CAMERA_OWNER: "camera"}.get(server["owner"], "session")}

    def list(self, session):
        owners = self._visible_owners(session)
        with self._lock:
            return self._redact([self._public(item) for item in self._config["servers"] if item["owner"] in owners])

    def _peer(self, session, identity):
        current = self._server(session, identity)
        cache_key = (self._owner(session), identity)
        with self._lock:
            for key, peer in list(self._peers.items()):
                if peer.closed or time.monotonic() - peer.last_used > self.idle_ttl:
                    peer.close()
                    self._peers.pop(key)
                    self._seen.pop(key, None)
            peer = self._peers.get(cache_key)
            if peer is None:
                if len(self._peers) >= 32:
                    raise MCPBridgeError("MCP session capacity reached; close an idle connection")
                peer = self.peer_factory(current, self._secrets.get(current["secret_ref"], ""), timeout=60)
                self._peers[cache_key] = peer
        return peer

    def catalog(self, session, identity, *, kind="tools", cursor=0, limit=8, name=None):
        peer = self._peer(session, identity)
        if kind not in {"tools", "resources", "templates", "prompts"}:
            raise MCPBridgeError("Unknown MCP catalog kind")
        items = peer.request(kind)["items"]
        if kind == "tools":
            self._validate_tools(items)
        start, limit = max(0, int(cursor)), max(1, min(8, int(limit)))
        selected = [item for item in items if item.get("name") == name] if name else items[start:start + limit]
        if len(json.dumps(selected).encode()) > 60000:
            raise MCPBridgeError("Selected schemas exceed the working-context budget; request one tool by exact name")
        revisions = {item["name"]: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() for item in selected if kind == "tools"}
        with self._lock:
            self._seen.setdefault((self._owner(session), identity), {}).update(revisions)
        return self._redact({"server_id": identity, "kind": kind, "items": selected, "schema_revisions": revisions,
            "next_cursor": start + len(selected) if not name and start + len(selected) < len(items) else None,
            "total": len(items), "trust": "untrusted_remote_catalog", "capabilities": peer.info.get("capabilities", {})})

    def call(self, session, identity, name, arguments):
        peer = self._peer(session, identity)
        items = peer.request("tools")["items"]
        self._validate_tools(items)
        definition = next((item for item in items if item["name"] == name), None)
        if definition is None:
            raise MCPBridgeError("MCP tool is not currently advertised")
        revision = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        with self._lock:
            seen = self._seen.get((self._owner(session), identity), {}).get(name)
        if seen != revision:
            raise MCPBridgeError("Discover this exact tool schema with mcp_catalog before calling it")
        if not isinstance(arguments, Mapping) or len(json.dumps(arguments).encode()) > 65536:
            raise MCPBridgeError("MCP arguments must be a bounded object")
        try:
            Draft202012Validator(definition["inputSchema"], registry=Registry()).validate(dict(arguments))
        except Exception:
            raise MCPBridgeError("MCP arguments do not satisfy the advertised input schema") from None
        result = peer.request("call", {"name": name, "arguments": arguments})
        return self._redact({"server_id": identity, "tool": name, "schema_revision": revision,
            "ok": result.get("isError") is not True, "result": result, "trust": "untrusted_mcp_tool_result",
            "claim_limits": "Tool return success is not proof of the requested task; read linked resources through mcp_resource and verify actual effects."})

    def resource(self, session, identity, *, uri=None, prompt=None, arguments=None):
        peer = self._peer(session, identity)
        if prompt:
            result = peer.request("prompt", {"name": prompt, "arguments": arguments or {}})
        elif isinstance(uri, str) and 1 <= len(uri) <= 2048:
            result = peer.request("read", {"uri": uri})
        else:
            raise MCPBridgeError("Provide an MCP resource URI or prompt name")
        received = any(item.get("text") or item.get("blob") for item in result.get("contents", [])) or bool(result.get("messages"))
        return self._redact({"server_id": identity, "result": result, "trust": "untrusted_mcp_resource", "source_body_received": received})

    def refresh(self, session, identity):
        self._server(session, identity)
        cache_key = (self._owner(session), identity)
        with self._lock:
            peer = self._peers.pop(cache_key, None)
            self._seen.pop(cache_key, None)
        if peer:
            peer.close()
        return self.catalog(session, identity)

    def remove(self, session, identity):
        server = self._server(session, identity)
        with self._lock:
            self._config["servers"] = [item for item in self._config["servers"] if item["id"] != identity]
            self._write("servers.json", self._config)
            if server["secret_ref"]:
                self._secrets.pop(server["secret_ref"], None)
                self._write("credentials.json", self._secrets)
            for key in [key for key in self._peers if key[1] == identity]:
                self._peers.pop(key).close()
                self._seen.pop(key, None)
        return {"removed": True}

    def clear(self, session):
        owner = self._owner(session)
        for item in self.list(session):
            if item["scope"] == "session":
                self.remove(session, item["id"])
        with self._lock:
            for key in [key for key in self._peers if key[0] == owner]:
                self._peers.pop(key).close()
                self._seen.pop(key, None)

    def close(self):
        with self._lock:
            peers, self._peers = list(self._peers.values()), {}
        for peer in peers:
            peer.close()
