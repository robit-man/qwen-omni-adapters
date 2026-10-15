"""Semantic CDP adapter for the finite computer-use controller.

The state/dispatch separation follows Omnius rapid-computer-use at 0ae974bd.
This adapter uses this runtime's existing visible Chromium ownership and input
checks; the policy never supplies JavaScript or executable coordinates.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def semantic_state(raw: dict[str, Any], session_id: str, target_id: str) -> dict[str, Any]:
    generation = str(raw.get("document_generation") or "")
    scope_id = f"{target_id}:{generation}"
    nodes = []
    for item in raw.get("elements", []):
        tag, kind = item.get("tag"), str(item.get("type") or "").lower()
        operations = ["focus", "click", "double_click", "right_click", "scroll", "key_chord"]
        if item.get("draggable") or (tag == "input" and kind == "range"):
            operations.append("drag_drop")
        if item.get("text_selectable") and tag in {"input", "textarea"}:
            operations.append("select_text")
        if tag in {"a", "button", "summary"} or item.get("role") in {"button", "link"}:
            operations.append("invoke")
        if not item.get("read_only") and tag in {"input", "textarea"} and kind not in {
            "checkbox", "radio", "button", "submit", "reset", "file", "hidden",
        }:
            operations.append("set_value")
        if tag == "input" and kind == "checkbox":
            operations.append("set_checked")
        if tag == "select":
            operations.append("select")
        sensitive = bool(item.get("sensitive") or kind == "password")
        node = {
            "id": str(item["id"]), "generation": generation, "scope_id": scope_id,
            "role": str(item.get("role") or "generic"),
            "name": str(item.get("label") or item.get("text") or item.get("name") or "")[:240],
            "visible": True, "enabled": not bool(item.get("disabled")),
            "sensitive": sensitive, "operations": [] if sensitive else operations,
            "value": "" if sensitive else str(item.get("value") or ""),
            "checked": bool(item.get("checked")), "focused": bool(item.get("focused")),
            "selected": list(item.get("selected") or []),
            "bounds": {key: item.get(key, 0) for key in ("x", "y", "width", "height")},
        }
        if not sensitive and isinstance(item.get("text_selection"), dict):
            node["text_selection"] = dict(item["text_selection"])
        nodes.append(node)
    state = {
        "session_id": hashlib.sha256(session_id.encode()).hexdigest(),
        "scope": {"id": scope_id, "kind": "browser", "url": str(raw.get("url") or ""),
                  "title": str(raw.get("title") or ""), "viewport": raw.get("viewport", {})},
        "nodes": nodes, "text": str(raw.get("visible_text") or "")[:12000],
        "truncated": bool(raw.get("truncated")),
    }
    state["revision"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return state


class BrowserUiAdapter:
    def __init__(self, browser: Any, session_id: str) -> None:
        self.browser = browser
        self.session_id = session_id
        self._visual: dict[str, Any] | None = None

    def prepare_step(self, step: dict[str, Any]) -> None:
        self._visual = None
        state = self.browser.semantic_observe(self.session_id)
        target = step["target"]
        matching = [node for node in state["nodes"] if all(
            node.get({"node_id": "id", "scope_id": "scope_id"}.get(key, key)) == value
            for key, value in target.items())]
        # Native controls always win. A canvas target may become one opaque
        # invoke candidate only after the local point worker grounds this frame.
        if (not matching and step["verb"] == "invoke" and target.get("name")
                and not target.get("node_id") and self.browser.pointing_url):
            self._visual = self.browser.semantic_ground(
                self.session_id, target["name"], target.get("role", "button"),
            )

    def observe(self) -> dict[str, Any]:
        state = self.browser.semantic_observe(self.session_id)
        if self._visual and self.browser.semantic_ground_is_current(self.session_id, self._visual, state):
            state["nodes"].append(copy.deepcopy(self._visual["node"]))
            state["revision"] = self._visual["revision"]
        return state

    def perform(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._visual and action.get("node_id") == self._visual["node"]["id"]:
            return self.browser.semantic_visual_perform(self.session_id, action, self._visual)
        return self.browser.semantic_perform(self.session_id, action)
