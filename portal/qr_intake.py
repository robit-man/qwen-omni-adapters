"""Interpret a camera payload before anything can enter a model or a log."""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

MAX_CODE = 4096
SECRET_TEXT = re.compile(r"(?:\b(?:api[_ -]?key|password|secret|access[_ -]?token|authorization)\b|\bbearer\s|-----BEGIN|\b(?:sk|ghp|gho|xoxb|xoxp)[_-][a-z0-9_-]+)", re.I)
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_./+=-]{16,}\Z")
MCP_PATH = re.compile(r"/(?:mcp|sse)(?:/sse)?/?\Z", re.I)
PRINTED_ENDPOINT = re.compile(r"(?:\bmcp(?:\s+(?:endpoint|server|url))?\s*:?\s*)?(https?://[^\s<>]+)", re.I)
PRINTED_KEY = re.compile(r"(?mi)^[ \t]*(?:api[-_ ]?key|x-api-key|token|authorization)\s*[:=]\s*(?:bearer\s+)?([!-~]+)\s*$")


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


def classify_camera_code(raw: str, *, source: str = "qr") -> CameraCode:
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
    if source == "ocr" and (printed := PRINTED_ENDPOINT.search(value)):
        endpoint = printed[1].rstrip(".,;")
        try:
            is_mcp = bool(MCP_PATH.search(urlsplit(endpoint).path) or re.search(r"\bmcp\b", printed[0], re.I))
        except ValueError:
            return CameraCode("unsupported")
        if is_mcp:
            keys = PRINTED_KEY.findall(value)
            # An unreadable credential label is insufficient for registration.
            if len(keys) > 1 or (SECRET_TEXT.search(value) and not keys):
                return CameraCode("invalid_mcp")
            return _mcp({"endpoint": endpoint, "key": keys[0] if keys else ""})
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
        self._items = {"qr": deque(maxlen=8), "ocr": deque(maxlen=8)}
        self._lock = threading.Lock()

    def observe(self, code: CameraCode, camera: str, *, source: str, confidence: float | None = None):
        now = time.time()
        public = {"id": secrets.token_hex(8), "kind": code.kind, "source": source,
                  "camera_id": camera, "captured_at": now}
        if code.text:
            public["observed_text"] = code.text
        if confidence is not None:
            public["confidence"] = round(confidence, 1)
        if code.kind == "mcp":
            try:
                public["host"] = (urlsplit(code.endpoint).hostname or "")[:160]
            except ValueError:
                public["host"] = ""
            public["credential_present"] = bool(code.key)
        with self._lock:
            self._expire(now)
            history = self._items[source]
            for prior, value in reversed(history):
                if value == code and prior["camera_id"] == camera and prior["source"] == source:
                    prior["captured_at"] = now
                    return dict(prior)
            history.append((public, code))
        return dict(public)

    def _expire(self, now):
        for history in self._items.values():
            for item in list(history):
                if item[0]["captured_at"] < now - 600:
                    history.remove(item)

    def _entries(self):
        return [row for history in self._items.values() for row in history]

    def connector(self, identity: str | None) -> CameraCode | None:
        with self._lock:
            self._expire(time.time())
            rows = [(entry, code) for entry, code in self._entries() if code.kind == "mcp"]
            if identity:
                return next((code for entry, code in rows if entry["id"] == identity), None)
            if len({(code.endpoint, code.key) for _, code in rows}) == 1:
                return rows[-1][1]
            return None

    def recent(self):
        with self._lock:
            self._expire(time.time())
            items = [dict(entry) for entry, _ in sorted(self._entries(), key=lambda row: row[0]["captured_at"], reverse=True)]
            return {"observations": items, "trust": "untrusted_camera_observation", "current_visual_input": False}

    def prompt(self) -> str:
        candidates = self.recent()["observations"]
        rows = candidates[:1]
        if rows:
            # The newest observation and the newest evidence of the other
            # type keep a busy OCR scene from burying a presented code.
            other = next((row for row in candidates if row["source"] != rows[0]["source"]), None)
            if other:
                rows.append(other)
            elif len(candidates) > 1:
                rows.append(candidates[1])
        if not rows:
            return ""
        data = [{key: value[:100] if isinstance(value, str) else value
                 for key, value in row.items() if key in {"id", "kind", "source", "host", "credential_present", "observed_text", "confidence"}}
                for row in rows]
        return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
