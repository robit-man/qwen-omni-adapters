"""Interpret a camera payload before anything can enter a model or a log."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

MAX_CODE = 4096
SECRET_TEXT = re.compile(r"(?:\b(?:api[_ -]?key|password|secret|access[_ -]?token|authorization)\b|\bbearer\s|-----BEGIN|\b(?:sk|ghp|gho|xoxb|xoxp)[_-][a-z0-9_-]+)", re.I)
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_./+=-]{16,}\Z")
MCP_PATH = re.compile(r"/(?:mcp|sse)(?:/sse)?/?\Z", re.I)
ANNOUNCEMENTS = {
    "text": "I scanned a text QR code.",
    "link": "I scanned a website QR code.",
    "contact": "I scanned a contact QR code.",
    "private": "I scanned a private code. I kept its contents private.",
    "structured": "I scanned a QR code containing structured data.",
    "unsupported": "I scanned a code that I couldn't interpret.",
    "invalid_mcp": "I scanned an MCP code with an invalid connection format.",
    "unsupported_mcp": "I scanned an MCP format that this camera connection doesn't support.",
    "mcp_ready": "I connected the MCP server from the QR code. Its tools are available on request.",
    "mcp_known": "That MCP connection is already registered.",
    "mcp_failed": "I scanned the MCP connection, but discovery did not complete. It may need a key or an allowed origin.",
}


@dataclass(frozen=True)
class CameraCode:
    kind: str
    text: str = ""
    endpoint: str = field(default="", repr=False)
    key: str = field(default="", repr=False)


def _mcp(value: dict) -> CameraCode:
    kind = value.get("type", value.get("transport", "http"))
    if kind not in ("http", "sse", "streamable-http", "streamableHttp", "mcp") or "command" in value or "args" in value:
        return CameraCode("unsupported_mcp")
    endpoint = value.get("endpoint", value.get("url"))
    headers = value.get("headers", {})
    if not isinstance(endpoint, str) or not isinstance(headers, dict):
        return CameraCode("invalid_mcp")
    candidates = [value[name] for name in ("key", "apiKey", "api_key", "token") if name in value]
    for name, header in headers.items():
        if name.lower() not in {"authorization", "x-api-key"}:
            return CameraCode("unsupported_mcp")
        if name.lower() == "authorization" and (not isinstance(header, str) or not header.lower().startswith("bearer ")):
            return CameraCode("unsupported_mcp")
        candidates.append(header)
    if any(not isinstance(key, str) or len(key) > MAX_CODE or any(ord(c) < 32 for c in key) for key in candidates):
        return CameraCode("invalid_mcp")
    keys = {re.sub(r"^bearer ", "", key, flags=re.I).strip() for key in candidates}
    if len(keys) > 1:
        return CameraCode("invalid_mcp")
    key = next(iter(keys), "")
    if "${" in key or "${" in endpoint:
        return CameraCode("unsupported_mcp")
    return CameraCode("mcp", endpoint=endpoint, key=key)


def classify_camera_code(raw: str) -> CameraCode:
    if not isinstance(raw, str) or not 0 < len(raw.encode("utf-8", "replace")) <= MAX_CODE:
        return CameraCode("unsupported")
    value = raw.strip()
    if not value:
        return CameraCode("unsupported")
    if value.startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return CameraCode("structured")
        if isinstance(decoded, dict):
            if "mcpServers" in decoded:
                servers = decoded["mcpServers"]
                if isinstance(servers, dict) and len(servers) == 1:
                    server = next(iter(servers.values()))
                    if isinstance(server, dict):
                        return _mcp(server)
                return CameraCode("unsupported_mcp")
            if ("endpoint" in decoded or decoded.get("type") == "mcp" or "url" in decoded
                    and ("headers" in decoded or decoded.get("type") in ("http", "sse", "streamable-http", "streamableHttp"))):
                return _mcp(decoded)
        return CameraCode("structured")
    if value[:6].lower() in {"mcp+ht", "mcp://"}:
        endpoint = value[4:] if value.lower().startswith("mcp+") else "https://" + value[6:]
        return _mcp({"endpoint": endpoint})
    try:
        url = urlsplit(value)
    except ValueError:
        return CameraCode("unsupported")
    if not any(c.isspace() for c in value) and url.scheme.lower() in {"http", "https"} and url.hostname:
        if MCP_PATH.search(url.path):
            return _mcp({"endpoint": value})
        return CameraCode("link", text=url.hostname[:160])
    if url.scheme.lower() in {"wifi", "otpauth", "otpauth-migration", "bitcoin", "ethereum"} or SECRET_TEXT.search(value) or OPAQUE_TOKEN.fullmatch(value):
        return CameraCode("private")
    if value.upper().startswith(("MECARD:", "BEGIN:VCARD")):
        return CameraCode("contact")
    if url.scheme and not any(c.isspace() for c in value):
        return CameraCode("structured")
    # Untrusted plain text is available in the bounded owner inbox. Never read
    # it aloud automatically or turn it into a user instruction.
    if any(ord(c) < 32 and c not in "\r\n\t" for c in value):
        return CameraCode("unsupported")
    return CameraCode("text", text=" ".join(value.split())[:500])


class CameraQRInbox:
    """A short-lived inbox for observations derived from a physical camera."""

    def __init__(self):
        self._items = deque(maxlen=16)
        self._lock = threading.Lock()

    def append(self, kind: str, camera: str, observed_text: str = "", **fields):
        public = {"kind": kind, "camera_id": camera, "captured_at": time.time(), **fields}
        if observed_text:
            public["observed_text"] = observed_text
        with self._lock:
            self._items.append(public)
        return {"kind": kind, "spoken": ANNOUNCEMENTS[kind]}

    def recent(self):
        cutoff = time.time() - 600
        with self._lock:
            return {"observations": [dict(item) for item in reversed(self._items) if item["captured_at"] >= cutoff],
                    "trust": "untrusted_qr_observation", "current_visual_input": False}
