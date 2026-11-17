"""Bounded, volatile tool-call inspection, separate from content-free diagnostics."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict


class SessionToolTraces:
    def __init__(self, *, ttl_s=300, max_sessions=32, max_calls=64, max_bytes=512_000, redactions=(), clock=time.monotonic):
        self.ttl_s, self.max_sessions, self.max_calls, self.max_bytes = ttl_s, max_sessions, max_calls, max_bytes
        self.redactions, self.clock = tuple(value for value in redactions if value), clock
        self._sessions = OrderedDict()
        self._active = {}
        self._lock = threading.RLock()

    def begin(self, session, request_id):
        with self._lock:
            if len(self._active) < 64:
                self._active[(session, request_id)] = True

    def finish(self, session, request_id):
        with self._lock:
            self._active.pop((session, request_id), None)

    def _clean(self, value, depth=0):
        if depth > 16:
            return "[nested value omitted]"
        if isinstance(value, dict):
            media = str(value.get("mimeType") or value.get("mime_type") or "").startswith(("image/", "audio/", "video/")) or str(value.get("type") or "") in {"image", "audio"}
            return {str(key): "[credential redacted]" if str(key).lower() in {"authorization", "password", "api_key", "access_token", "secret", "token"}
                else "[media bytes omitted]" if (media and key in {"data", "blob", "base64"}) or key == "images"
                else self._clean(item, depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clean(item, depth + 1) for item in value[:256]] + (["[additional items omitted]"] if len(value) > 256 else [])
        if isinstance(value, str):
            for secret in self.redactions:
                value = value.replace(secret, "[credential redacted]")
        return value

    def _text(self, value, limit):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        cleaned = self._clean(value)
        text = cleaned if isinstance(cleaned, str) else json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
        return text[:limit], len(text) > limit

    def _expire(self):
        now = self.clock()
        for session, (stamp, _calls) in list(self._sessions.items()):
            if now - stamp >= self.ttl_s:
                self._sessions.pop(session, None)

    def record(self, session, request_id, round_index, items):
        with self._lock:
            if self._active.get((session, request_id)) is not True:
                return
            self._expire()
            _stamp, calls = self._sessions.pop(session, (0, OrderedDict()))
            for item in items:
                identity = (request_id, round_index, str(item.get("id") or "")[:80])
                old = calls.get(identity, {})
                arguments, args_cut = self._text(item.get("arguments") or {}, 12_000)
                output, output_cut = self._text(item.get("result") or "", 24_000)
                complete = item.get("status") != "running"
                # A delayed start notification cannot overwrite a returned call.
                if old.get("status") == "complete" and not complete:
                    continue
                calls[identity] = {"request_id": request_id, "round": round_index, "call_id": identity[2],
                    "name": str(item.get("name") or "unknown")[:80], "status": "complete" if complete else "running",
                    "ok": item.get("ok") is True if complete else None, "arguments": arguments,
                    "arguments_truncated": args_cut, "output": output, "output_truncated": output_cut,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "retention": f"This session only; expires after {self.ttl_s:g} seconds of tool inactivity or Trash"}
                calls.move_to_end(identity)
            while len(calls) > self.max_calls or sum(len(json.dumps(item).encode()) for item in calls.values()) > self.max_bytes:
                calls.popitem(last=False)
            self._sessions[session] = (self.clock(), calls)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def get(self, session, request_id, round_index, call_id):
        with self._lock:
            self._expire()
            _stamp, calls = self._sessions.get(session, (0, {}))
            return copy.deepcopy(calls.get((request_id, round_index, call_id)))

    def clear(self, session):
        with self._lock:
            self._sessions.pop(session, None)
            for key in self._active:
                if key[0] == session:
                    self._active[key] = False
