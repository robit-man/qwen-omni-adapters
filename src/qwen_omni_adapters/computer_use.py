"""Host-authorized finite computer actions with observed effect verification.

Design provenance: the adjacent Omnius rapid-computer-use contracts/controller
at revision 0ae974bd and its 2026-10-12 controlled browser/desktop evidence.
This is an original Python implementation of those architectural contracts;
it neither imports that runtime nor starts model services. Adapters own bounded
observation/input transports and must revalidate identity at actual dispatch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

VERBS = frozenset({"invoke", "focus", "set_value", "set_checked", "select", "scroll", "key_chord",
                   "click", "double_click", "right_click", "drag_drop", "select_text", "switch_window", "type_text"})
_BOUNDARY_VERBS = frozenset({"invoke", "click", "double_click", "right_click"})
_ARGUMENTS = {
    "invoke": set(), "focus": set(), "set_value": {"value"},
    "set_checked": {"checked"}, "select": {"option"},
    "scroll": {"direction"}, "key_chord": {"keys"},
    "click": set(), "double_click": set(), "right_click": set(),
    "drag_drop": {"to_node_id"}, "select_text": {"start", "end"}, "switch_window": set(),
    "type_text": {"text"},
}
_PREDICATE_FIELDS = {
    "value": {"node_id", "value"}, "checked": {"node_id", "checked"},
    "selected": {"node_id", "values"}, "focused": {"node_id"},
    "visible": {"role", "name", "visible"}, "text": {"text"}, "url": {"url"},
    "text_selection": {"node_id", "start", "end"}, "active_window": {"node_id"},
}


class ComputerUseError(ValueError):
    """A bounded computer-use contract is invalid."""


class UiAdapter(Protocol):
    def observe(self) -> dict[str, Any]: ...

    def perform(self, action: dict[str, Any]) -> dict[str, Any]: ...


class _Cancelled(Exception):
    pass


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ComputerUseError(f"Invalid {label} fields")
    return value


def _string(value: Any, label: str, limit: int = 4096, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value):
        raise ComputerUseError(f"Invalid {label}")
    return value


def _strings(value: Any, label: str, limit: int = 50) -> None:
    if not isinstance(value, list) or len(value) > limit:
        raise ComputerUseError(f"Invalid {label}")
    for item in value:
        _string(item, label, empty=True)


def _predicates(value: Any, *, terminal: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 20:
        raise ComputerUseError("Expected 1–20 effect predicates")
    for raw in value:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str) or raw["kind"] not in _PREDICATE_FIELDS:
            raise ComputerUseError("Unsupported effect predicate")
        kind = raw["kind"]
        predicate = _object(raw, {"kind"} | _PREDICATE_FIELDS[kind], "predicate")
        if kind in {"value", "checked", "selected", "focused", "text_selection", "active_window"}:
            _string(predicate.get("node_id"), "predicate node identity", 512)
            if terminal and predicate["node_id"] == "$target":
                raise ComputerUseError("Terminal predicates require an independently resolved node")
        if kind == "value":
            _string(predicate.get("value"), "expected value", empty=True)
        if kind in {"checked", "visible"} and type(predicate.get(kind)) is not bool:
            raise ComputerUseError(f"Expected boolean {kind}")
        if kind == "selected":
            _strings(predicate.get("values"), "expected selection")
        if kind == "text_selection":
            _selection_range(predicate)
        if kind == "visible":
            if not {"role", "name"} & predicate.keys():
                raise ComputerUseError("Visibility requires a semantic target")
            for key in ("role", "name"):
                if key in predicate:
                    _string(predicate[key], key, empty=key == "name")
        if kind in {"text", "url"}:
            _string(predicate.get(kind), f"expected {kind}", 20000)
    return copy.deepcopy(value)


def parse_predicates(value: Any, *, terminal: bool = False) -> list[dict[str, Any]]:
    """Public bounded predicate validator shared by canonical verification."""
    return _predicates(value, terminal=terminal)


def _selection_range(value: Mapping[str, Any]) -> None:
    start, end = value.get("start"), value.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= 1_000_000:
        raise ComputerUseError("Expected bounded character selection offsets")


def _parse_intent(value: Any, *, effects: bool) -> dict[str, Any]:
    step = _object(value, {"id", "verb", "target", "args"} | ({"expect"} if effects else set()), "step")
    _string(step.get("id"), "step identity", 200)
    verb = step.get("verb")
    if not isinstance(verb, str) or verb not in VERBS:
        raise ComputerUseError("Unsupported UI verb")
    target = _object(step.get("target"), {"node_id", "role", "name", "scope_id"}, "target")
    if not {"node_id", "role", "name"} & target.keys():
        raise ComputerUseError("A UI target is required")
    for key, item in target.items():
        _string(item, f"target {key}", 512, empty=key == "name")
    args = step.get("args", {})
    if not isinstance(args, dict) or set(args) != _ARGUMENTS[verb]:
        required = ", ".join(sorted(_ARGUMENTS[verb])) or "none (omit args or use {})"
        raise ComputerUseError(f"{verb} args must contain exactly these fields: {required}")
    if verb in {"set_value", "select"}:
        key = "value" if verb == "set_value" else "option"
        _string(args.get(key), key, empty=True)
    if verb == "type_text":
        text = _string(args.get("text"), "typed text", 2048)
        if "\x00" in text:
            raise ComputerUseError("Typed text cannot contain NUL")
    if verb == "set_checked" and type(args.get("checked")) is not bool:
        raise ComputerUseError("Expected a boolean checked state")
    if verb == "select_text":
        _selection_range(args)
    if verb == "drag_drop":
        _string(args.get("to_node_id"), "observed drop target", 512)
    if verb == "scroll" and (not isinstance(args.get("direction"), str) or args["direction"] not in {"up", "down", "left", "right"}):
        raise ComputerUseError("Invalid scroll direction")
    if verb == "key_chord":
        keys = args.get("keys")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 4:
            raise ComputerUseError("Expected 1–4 bounded key names")
        for key in keys:
            _string(key, "key", 30)
            if not all(char.isascii() and (char.isalnum() or char in "+_-") for char in key):
                raise ComputerUseError("Invalid key name")
    return {**copy.deepcopy(step), "args": copy.deepcopy(args)}


def parse_step(value: Any) -> dict[str, Any]:
    """Validate the unchanged public operation and its required predicates."""
    step = _parse_intent(value, effects=True)
    return {**step, "expect": _predicates(step.get("expect"))}


def parse_intent(value: Any) -> dict[str, Any]:
    """Private native-task boundary: one pointer/activation intent, no effects."""
    intent = _parse_intent(value, effects=False)
    if intent["verb"] not in _BOUNDARY_VERBS:
        raise ComputerUseError("Unsupported observation-boundary verb")
    return intent


def _text_observation(node: dict[str, Any]) -> None:
    """Validate optional native text knowledge; omission never means zero."""
    observation = _object(node["text_observation"],
                          {"character_count", "selection_count", "captured_range", "truncated"},
                          "text observation")
    if not observation:
        raise ComputerUseError("Empty text observation")
    for key in ("character_count", "selection_count"):
        if key in observation and (type(observation[key]) is not int or not 0 <= observation[key] < 2**31):
            raise ComputerUseError("Invalid observed text count")
    if "captured_range" in observation or "truncated" in observation:
        if not {"character_count", "captured_range", "truncated"} <= observation.keys():
            raise ComputerUseError("Incomplete observed text coverage")
        captured = _object(observation["captured_range"], {"start", "end"}, "captured text range")
        if (set(captured) != {"start", "end"} or type(captured["start"]) is not int or captured["start"] != 0
                or type(captured["end"]) is not int or not 0 <= captured["end"] <= observation["character_count"]
                or not isinstance(node.get("value"), str) or len(node["value"]) != captured["end"]
                or type(observation["truncated"]) is not bool
                or observation["truncated"] != (captured["end"] < observation["character_count"])):
            raise ComputerUseError("Invalid observed text coverage")
    selection = node.get("text_selection")
    if selection is not None and (observation.get("selection_count") == 0
            or "character_count" in observation and selection["end"] > observation["character_count"]):
        raise ComputerUseError("Contradictory observed text selection")


def _snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComputerUseError("Invalid UI snapshot")
    state = copy.deepcopy(value)
    for key in ("session_id", "revision"):
        _string(state.get(key), key, 512)
    scope = state.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("kind"), str) or scope["kind"] not in {"browser", "native", "desktop"}:
        raise ComputerUseError("Invalid UI scope")
    _string(scope.get("id"), "scope identity", 512)
    if "url" in scope:
        _string(scope["url"], "scope URL", 20000)
    nodes = state.get("nodes")
    if not isinstance(nodes, list) or len(nodes) > 600:
        raise ComputerUseError("Invalid UI node budget")
    ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ComputerUseError("Invalid UI node")
        for key in ("id", "generation", "scope_id", "role", "name"):
            _string(node.get(key), f"node {key}", 512, empty=key == "name")
        if node["id"] in ids:
            raise ComputerUseError("Duplicate UI node identity")
        ids.add(node["id"])
        for key in ("visible", "enabled", "sensitive"):
            if type(node.get(key)) is not bool:
                raise ComputerUseError(f"Invalid node {key}")
        operations = node.get("operations")
        if not isinstance(operations, list) or len(operations) > len(VERBS) or any(
            not isinstance(op, str) or op not in VERBS for op in operations
        ) or len(set(operations)) != len(operations):
            raise ComputerUseError("Invalid node operations")
        for key in ("focused", "checked"):
            if key in node and type(node[key]) is not bool:
                raise ComputerUseError(f"Invalid node {key}")
        if "value" in node:
            _string(node["value"], "observed value", empty=True)
        if "selected" in node:
            _strings(node["selected"], "observed selection")
        if "text_selection" in node:
            selection = _object(node["text_selection"], {"start", "end"}, "observed text selection")
            _selection_range(selection)
        if node["sensitive"]:
            node.pop("value", None)
            node.pop("selected", None)
            node.pop("text_selection", None)
            node.pop("text_observation", None)
            node["operations"] = []
        elif "text_observation" in node:
            _text_observation(node)
    if "text" in state:
        _string(state["text"], "observed text", 40000, empty=True)
    if "truncated" in state and type(state["truncated"]) is not bool:
        raise ComputerUseError("Invalid observation coverage")
    return state


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def predicate_satisfied(predicate: Mapping[str, Any], state: Mapping[str, Any], receipt: Any = None) -> bool:
    """Evaluate typed observed state; a dispatch receipt alone proves no effect."""
    kind = predicate.get("kind")
    nodes = state.get("nodes", [])
    node = next((item for item in nodes if item.get("id") == predicate.get("node_id") and not item.get("sensitive")), None)
    if kind == "value":
        if node is None or "value" not in node or node["value"] != predicate.get("value"):
            return False
        coverage = node.get("text_observation", {})
        return (coverage.get("truncated") is not True
                and coverage.get("character_count", len(node["value"])) == len(node["value"]))
    if kind == "checked":
        return node is not None and "checked" in node and node["checked"] is predicate.get("checked")
    if kind == "selected":
        return node is not None and "selected" in node and node["selected"] == predicate.get("values")
    if kind == "focused":
        return node is not None and node.get("focused") is True
    if kind == "text_selection":
        return node is not None and node.get("text_selection") == {"start": predicate.get("start"), "end": predicate.get("end")}
    if kind == "active_window":
        return node is not None and node.get("role") == "window" and state.get("scope", {}).get("id") == node.get("id")
    if kind == "visible":
        matched = any(item.get("visible") is True and not item.get("sensitive") and all(
            key not in predicate or item.get(key) == predicate[key] for key in ("role", "name")
        ) for item in nodes)
        if predicate.get("visible") is False and state.get("truncated") is True:
            return False
        return matched is predicate.get("visible")
    if kind == "text":
        text = predicate.get("text")
        return isinstance(text, str) and bool(text) and (text in state.get("text", "") or any(
            not item.get("sensitive") and text in item.get("name", "") for item in nodes
        ))
    if kind == "url":
        return state.get("scope", {}).get("url") == predicate.get("url")
    return False


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise _Cancelled("Computer-use task cancelled")


def _verified_mutation(outcome: Mapping[str, Any]) -> bool:
    before, after = outcome.get("before", {}), outcome.get("after", {})
    first = before.get("base_semantic_revision", before.get("revision"))
    last = after.get("base_semantic_revision", after.get("revision"))
    return bool(outcome.get("status") == "verified"
                and outcome.get("receipt", {}).get("status") == "dispatched"
                and before.get("session_id") == after.get("session_id")
                and isinstance(first, str) and first and isinstance(last, str) and last and first != last)


class RapidUiController:
    """Own candidate authority and serialize side effects for one adapter session."""

    def __init__(self, adapter: UiAdapter, *, authorize: Callable[[dict, dict], bool] | None = None,
                 max_observations: int = 20, verification_timeout_s: float = 2.5,
                 observation_interval_s: float = 0.1) -> None:
        if type(max_observations) is not int or not 1 <= max_observations <= 20:
            raise ComputerUseError("Invalid observation budget")
        if not isinstance(verification_timeout_s, (int, float)) or not 0 < verification_timeout_s <= 30:
            raise ComputerUseError("Invalid verification deadline")
        if not isinstance(observation_interval_s, (int, float)) or not 0 < observation_interval_s <= 1:
            raise ComputerUseError("Invalid observation interval")
        if authorize is not None and not callable(authorize):
            raise ComputerUseError("Invalid host authorizer")
        self.adapter = adapter
        self.authorize = authorize
        self.max_observations = max_observations
        self.verification_timeout_s = verification_timeout_s
        self.observation_interval_s = observation_interval_s
        self._state: dict[str, Any] | None = None
        self._actions: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._execution_epoch = 0
        self._completion_epochs: dict[str, int] = {}
        self._offers: dict[str, tuple[int, str]] = {}
        self._binding_keys: dict[str, str] = {}
        self._recorded_ids: dict[str, str] = {}
        self._native_task_protocol = False
        self._intent_keys: dict[str, str] = {}
        self._boundary_offers: set[str] = set()
        self._metadata = threading.RLock()
        self._input = threading.Lock()

    def observe(self, *, planning: bool = False) -> dict[str, Any]:
        with self._metadata:
            observer = (getattr(self.adapter, "planning_observe", self.adapter.observe)
                        if planning else self.adapter.observe)
            self._state = _snapshot(observer())
            return copy.deepcopy(self._state)

    def enumerate(self, step: dict[str, Any], state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._enumerate(parse_step(step), state)

    def enable_native_task_protocol(self) -> None:
        """Trusted host setup only; never change an existing client epoch."""
        with self._metadata:
            if self._actions or self._outcomes:
                raise ComputerUseError("Native task protocol requires a fresh controller")
            self._native_task_protocol = True

    @staticmethod
    def _intent_key(state: Mapping[str, Any], action: Mapping[str, Any]) -> str:
        return _digest({"session_id": state["session_id"], "scope": state["scope"],
            "semantic_revision": state.get("base_semantic_revision", state["revision"]),
            **{key: action[key] for key in ("node_id", "generation", "verb", "args")}})

    def _enumerate(self, step: dict[str, Any], state: dict[str, Any] | None = None,
                   *, boundary: bool = False) -> list[dict[str, Any]]:
        with self._metadata:
            if self._state is None or (state is not None and state != self._state):
                raise ComputerUseError("Observe the current UI before enumerating actions")
            current = self._state
            offers = {}
            common = {"session_id": current["session_id"], "scope_id": current["scope"]["id"], "revision": current["revision"]}
            actions = [{**common, "id": uuid.uuid4().hex, "kind": kind, "expect": [],
                        "idempotency_key": uuid.uuid4().hex, "description": kind}
                       for kind in ("observe", "abstain")]
            for node in current["nodes"]:
                if not node["visible"] or not node["enabled"] or node["sensitive"] or step["verb"] not in node["operations"]:
                    continue
                if any(node.get("id" if key == "node_id" else key) != item for key, item in step["target"].items()):
                    continue
                if step["verb"] == "drag_drop":
                    destination = next((item for item in current["nodes"] if item["id"] == step["args"]["to_node_id"]), None)
                    if destination is None or not destination["visible"] or not destination["enabled"] or destination["sensitive"] or destination["scope_id"] != node["scope_id"]:
                        continue
                expect = [{**predicate, "node_id": node["id"]} if predicate.get("node_id") == "$target" else copy.deepcopy(predicate)
                          for predicate in step.get("expect", [])]
                # The host owns the operation's effect. Planner predicates can
                # add requirements, but focus or unrelated text cannot verify
                # a value write. Bind the required effect into the offered ID
                # so cached receipts retain the same verification contract.
                if step["verb"] == "set_value":
                    required = {"kind": "value", "node_id": node["id"], "value": step["args"]["value"]}
                    if required not in expect:
                        if len(expect) >= 20:
                            raise ComputerUseError("Leave room for the mandatory set_value effect within 20 predicates")
                        expect.append(required)
                binding = {"session_id": current["session_id"], **common, "step_id": step["id"],
                           "node_id": node["id"], "generation": node["generation"],
                           "verb": step["verb"], "args": step["args"], "expect": expect}
                legacy = _digest(binding)
                if boundary:
                    legacy = _digest({"boundary_intent": binding})
                previous_key = self._binding_keys.get(legacy)
                previous = self._outcomes.get(previous_key, {})
                # Unknown/no-op intents retain their original lock even after
                # unrelated verified mutations. Only a proved applied intent
                # may renew after a separate verified mutation beyond its own
                # completion. Missing completion evidence keeps the old lock.
                may_renew = (_verified_mutation(previous) and previous_key in self._completion_epochs
                             and self._execution_epoch > self._completion_epochs[previous_key])
                key = (previous_key if previous_key is not None and not may_renew
                       else _digest({"binding": binding, "execution_epoch": self._execution_epoch}))
                actions.append({**common, "id": uuid.uuid4().hex, "kind": "action", "node_id": node["id"],
                                "generation": node["generation"], "verb": step["verb"], "args": copy.deepcopy(step["args"]),
                                "expect": expect, "idempotency_key": key,
                                "description": f"{step['verb']} {node['role']} {node['name']}"[:500]})
                offers[actions[-1]["id"]] = (self._execution_epoch, legacy)
            if len(actions) > 50:
                raise ComputerUseError("Ambiguous UI target; narrow to at most 48 eligible controls")
            self._actions = {action["id"]: copy.deepcopy(action) for action in actions}
            self._offers = offers
            self._boundary_offers = {action["id"] for action in actions if boundary and action["kind"] == "action"}
            return copy.deepcopy(actions)

    def candidate(self, action_id: str) -> dict[str, Any] | None:
        with self._metadata:
            return copy.deepcopy(self._actions.get(action_id)) if isinstance(action_id, str) else None

    def _record(self, outcome: dict[str, Any], legacy_binding: str | None = None) -> dict[str, Any]:
        with self._metadata:
            action = outcome["action"]
            key = action["idempotency_key"]
            if _verified_mutation(outcome) and not _verified_mutation(self._outcomes.get(key, {})):
                self._execution_epoch += 1
                self._completion_epochs[key] = self._execution_epoch
            self._outcomes[key] = copy.deepcopy(outcome)
            self._recorded_ids[action["id"]] = key
            if legacy_binding is not None:
                self._binding_keys[legacy_binding] = key
        return copy.deepcopy(outcome)

    def execute(self, action_id: str, cancel: threading.Event | None = None) -> dict[str, Any]:
        action = self.candidate(action_id)
        if action is None:
            with self._metadata:
                historical = self._outcomes.get(self._recorded_ids.get(action_id)) if isinstance(action_id, str) else None
                if historical is not None:
                    return copy.deepcopy(historical)
            return {"status": "rejected", "reason": "Unknown candidate; observe and enumerate again"}
        with self._metadata:
            previous = copy.deepcopy(self._outcomes.get(action["idempotency_key"]))
            offer = self._offers.get(action_id)
            boundary_offer = action_id in self._boundary_offers
        if previous is not None:
            return previous
        if not self._input.acquire(blocking=False):
            return {"status": "unresolved", "action": action, "reason": "Input is already in progress; effect state is unknown"}
        outcome: dict[str, Any] = {"status": "unresolved", "action": action}
        def record(value: dict[str, Any]) -> dict[str, Any]:
            return self._record(value, offer[1] if offer is not None else None)
        try:
            with self._metadata:
                previous = copy.deepcopy(self._outcomes.get(action["idempotency_key"]))
                if len(self._outcomes) >= 1000 and previous is None:
                    return {"status": "rejected", "reason": "Session action budget exhausted"}
            if previous is not None:
                return previous
            with self._metadata:
                if action["kind"] == "action" and (offer is None or offer[0] != self._execution_epoch):
                    return {**outcome, "status": "rejected", "reason": "Execution epoch changed; observe and enumerate again"}
            _check_cancel(cancel)
            if action["kind"] == "abstain":
                return record({**outcome, "status": "abstained"})
            before = self.observe()
            outcome["before"] = before
            _check_cancel(cancel)
            if action["kind"] == "observe":
                return record({**outcome, "after": before, "reason": "Observed; no input requested"})
            node = next((item for item in before["nodes"] if item["id"] == action["node_id"]), None)
            if before["session_id"] != action["session_id"] or before["scope"]["id"] != action["scope_id"] or before["revision"] != action["revision"] or node is None or node["generation"] != action["generation"] or not node["enabled"] or not node["visible"] or node["sensitive"] or action["verb"] not in node["operations"]:
                return record({**outcome, "status": "rejected", "reason": "UI identity or revision changed; observe again"})
            if self.authorize is None or self.authorize(copy.deepcopy(action), copy.deepcopy(before)) is not True:
                return record({**outcome, "status": "rejected", "reason": "Host authority did not admit this exact action"})
            _check_cancel(cancel)
            if self._native_task_protocol:
                with self._metadata:
                    intent = self._intent_key(before, action)
                    locked_key = self._intent_keys.get(intent)
                    locked = self._outcomes.get(locked_key)
                    may_renew = (locked is not None and _verified_mutation(locked)
                                 and locked_key in self._completion_epochs
                                 and self._execution_epoch > self._completion_epochs[locked_key])
                    if locked is not None and not may_renew:
                        self._recorded_ids[action_id] = locked_key
                        if offer is not None:
                            self._binding_keys[offer[1]] = locked_key
                        return copy.deepcopy(locked)
                    if intent not in self._intent_keys and len(self._intent_keys) >= 1000:
                        return record({**outcome, "status": "rejected", "reason": "Native intent budget exhausted"})
                    self._intent_keys[intent] = action["idempotency_key"]
                outcome["dispatch_attempted"] = True
            record({**outcome, "reason": "Dispatch pending; effects remain unverified"})
            receipt = self.adapter.perform(copy.deepcopy(action))
            if not isinstance(receipt, dict) or not isinstance(receipt.get("status"), str) or receipt["status"] not in {"dispatched", "rejected", "already_satisfied"}:
                raise ComputerUseError("Invalid input dispatch receipt")
            outcome["receipt"] = copy.deepcopy(receipt)
            if receipt["status"] == "rejected":
                return record({**outcome, "status": "rejected", "reason": str(receipt.get("reason", "Input rejected"))[:1000]})
            already_satisfied = receipt["status"] == "already_satisfied"
            if already_satisfied and not (
                receipt.get("input_attempted") is False and action["verb"] == "switch_window"
                and action["args"] == {} and node["role"] == "window" and node["id"] == before["scope"]["id"]
                and action["expect"] == [{"kind": "active_window", "node_id": node["id"]}]
            ):
                return record({**outcome, "status": "rejected", "reason": "Invalid already-satisfied window receipt"})
            record({**outcome, "reason": "No input attempted; active window remains unverified" if already_satisfied
                          else "Dispatched; effects remain unverified"})
            started = time.monotonic()
            deadline = started + self.verification_timeout_s
            if boundary_offer:
                _check_cancel(cancel)
                observer = getattr(self.adapter, "boundary_observe", None)
                if not callable(observer):
                    raise ComputerUseError("Pure semantic boundary observation unavailable")
                with self._metadata:
                    self._state = _snapshot(observer())
                    outcome["after"] = copy.deepcopy(self._state)
                _check_cancel(cancel)
                if time.monotonic() > deadline or outcome["after"]["session_id"] != before["session_id"]:
                    return record({**outcome, "reason": "Boundary observation binding or deadline changed"})
                return record({**outcome, "status": "observed_unverified",
                               "reason": "One input observed; no effect predicate or progress verified"})
            for index in range(self.max_observations):
                _check_cancel(cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if index:
                    wait = min(self.observation_interval_s, remaining)
                    if cancel is not None:
                        cancel.wait(wait)
                    else:
                        time.sleep(wait)
                    _check_cancel(cancel)
                    if time.monotonic() >= deadline:
                        break
                after = self.observe()
                outcome["after"] = after
                outcome["verification"] = {"observations": index + 1, "elapsed_s": time.monotonic() - started,
                                           "max_observations": self.max_observations, "timeout_s": self.verification_timeout_s}
                _check_cancel(cancel)
                if time.monotonic() > deadline:
                    break
                if already_satisfied:
                    fresh = next((item for item in after["nodes"] if item["id"] == action["node_id"]), None)
                    if (after["session_id"] != before["session_id"] or after["scope"] != before["scope"]
                            or after["revision"] != before["revision"] or fresh is None
                            or fresh["generation"] != action["generation"] or not fresh["visible"]
                            or not fresh["enabled"] or fresh["sensitive"] or fresh["role"] != "window"
                            or "switch_window" not in fresh["operations"]):
                        return record({**outcome, "status": "rejected",
                                             "reason": "Already-satisfied window binding changed during verification"})
                if all(predicate_satisfied(item, after, receipt) for item in action["expect"]):
                    return record({**outcome, "status": "verified"})
            return record({**outcome, "reason": "Expected effects were not observed within the verification budget"})
        except _Cancelled as exc:
            return record({**outcome, "status": "cancelled", "reason": str(exc)})
        except Exception as exc:
            return record({**outcome, "reason": f"Input/effect observation uncertain: {exc}"[:1000]})
        finally:
            self._input.release()

    def run_step(self, step: dict[str, Any], provider: Any = None,
                 cancel: threading.Event | None = None) -> dict[str, Any]:
        step = parse_step(step)
        return self._run_intent(step, provider, cancel)

    def run_boundary(self, intent: dict[str, Any], provider: Any = None,
                     cancel: threading.Event | None = None) -> dict[str, Any]:
        if not self._native_task_protocol:
            raise ComputerUseError("Observation boundaries require a trusted native task")
        parsed = parse_intent(intent)
        try:
            return self._run_intent(parsed, provider, cancel, boundary=True)
        finally:
            finish = getattr(self.adapter, "finish_intent", None)
            if callable(finish):
                finish()

    def _run_intent(self, step: dict[str, Any], provider: Any,
                    cancel: threading.Event | None, *, boundary: bool = False) -> dict[str, Any]:
        try:
            _check_cancel(cancel)
            prepare = getattr(self.adapter, "prepare_intent" if boundary else "prepare_step", None)
            if boundary and not callable(prepare):
                raise ComputerUseError("Boundary intent preparation unavailable")
            if prepare is not None:
                prepare(copy.deepcopy(step))
            _check_cancel(cancel)
            state = self.observe()
            actions = self._enumerate(step, state, boundary=True) if boundary else self.enumerate(step, state)
            if provider is None:
                executable = [action for action in actions if action["kind"] == "action"]
                choice = executable[0]["id"] if len(executable) == 1 else next(action["id"] for action in actions if action["kind"] == "abstain")
            else:
                decide = getattr(provider, "decide", provider)
                answer = decide(copy.deepcopy(state), copy.deepcopy(actions))
                choice = answer.get("choice") if isinstance(answer, Mapping) else answer
                if not isinstance(choice, str) or not any(action["id"] == choice for action in actions):
                    raise ComputerUseError("Provider selected an unknown finite candidate")
            _check_cancel(cancel)
            return self.execute(choice, cancel)
        except _Cancelled as exc:
            return {"status": "cancelled", "reason": str(exc)}
        except Exception as exc:
            return {"status": "unresolved", "reason": str(exc)[:1000]}

    def run_workflow(self, steps: Sequence[dict[str, Any]], *, final_predicates: list[dict[str, Any]],
                     provider: Any = None, cancel: threading.Event | None = None) -> dict[str, Any]:
        return run_workflow(self, steps, final_predicates=final_predicates, provider=provider, cancel=cancel)


def run_workflow(controller: RapidUiController, steps: Sequence[dict[str, Any]], *,
                 final_predicates: list[dict[str, Any]], provider: Any = None,
                 cancel: threading.Event | None = None) -> dict[str, Any]:
    """Run a bounded host plan, then independently observe host terminal criteria."""
    if not isinstance(steps, (list, tuple)) or not 1 <= len(steps) <= 25:
        raise ComputerUseError("Expected 1–25 workflow steps")
    parsed = [parse_step(step) for step in steps]
    final = _predicates(final_predicates, terminal=True)
    outcomes: list[dict[str, Any]] = []
    for index, step in enumerate(parsed):
        outcome = controller.run_step(step, provider, cancel)
        outcomes.append(outcome)
        if outcome["status"] != "verified":
            return {"status": "cancelled" if outcome["status"] == "cancelled" else "needs_replan",
                    "outcomes": outcomes, "reason": outcome.get("reason", outcome["status"])}
        # A new run_step enumerates fresh candidate IDs. A returned historical
        # ID proves cache reuse, not that its prerequisite still holds now.
        if index + 1 < len(parsed) and controller.candidate(outcome["action"]["id"]) is None:
            try:
                _check_cancel(cancel)
                current = controller.observe()
                _check_cancel(cancel)
                after, action = outcome["after"], outcome["action"]
                previous_nodes = {node["id"]: node for node in after["nodes"]}
                current_nodes = {node["id"]: node for node in current["nodes"]}
                effect_nodes = {item["node_id"] for item in action["expect"] if "node_id" in item}
                verified = (current["session_id"] == after["session_id"] and current["scope"] == after["scope"]
                            and all(node_id in previous_nodes and node_id in current_nodes
                                    and current_nodes[node_id]["generation"] == previous_nodes[node_id]["generation"]
                                    and current_nodes[node_id]["scope_id"] == previous_nodes[node_id]["scope_id"]
                                    and not current_nodes[node_id]["sensitive"] for node_id in effect_nodes)
                            and all(predicate_satisfied(item, current) for item in action["expect"]))
                outcome["current_prefix_verification"] = {
                    "schema": "robit.omni.ui-verification.v1", "status": "verified" if verified else "unresolved",
                    "session_id": current["session_id"], "scope_id": current["scope"]["id"],
                    "revision": current["revision"], "content_sha256": _digest(current),
                    "predicates_sha256": _digest(action["expect"]), "input_attempted": False,
                    "task_progress": False,
                }
                if not verified:
                    outcome.update(status="unresolved", reason="Historical workflow prefix did not satisfy its predicates in the current UI")
                    return {"status": "needs_replan", "outcomes": outcomes, "terminal": current, "reason": outcome["reason"]}
            except _Cancelled as exc:
                return {"status": "cancelled", "outcomes": outcomes, "reason": str(exc)}
            except Exception as exc:
                outcome.update(status="unresolved", reason="Historical workflow prefix could not be freshly verified",
                               current_prefix_verification={"schema": "robit.omni.ui-verification.v1",
                                   "status": "unresolved", "error_type": type(exc).__name__,
                                   "input_attempted": False, "task_progress": False})
                return {"status": "needs_replan", "outcomes": outcomes, "reason": outcome["reason"]}
    try:
        _check_cancel(cancel)
        terminal = controller.observe()
        _check_cancel(cancel)
        latest = outcomes[-1]["after"]
        if terminal["session_id"] != latest["session_id"] or terminal["scope"]["id"] != latest["scope"]["id"]:
            return {"status": "needs_replan", "outcomes": outcomes,
                    "reason": "Terminal observation belongs to another UI session or scope"}
        if not all(predicate_satisfied(item, terminal) for item in final):
            return {"status": "needs_replan", "outcomes": outcomes, "terminal": terminal,
                    "reason": "Independent terminal observation did not satisfy host criteria"}
        receipt = {"schema": "robit.omni.ui-verification.v1", "id": uuid.uuid4().hex,
                   "status": "verified", "session_id": terminal["session_id"],
                   "scope_id": terminal["scope"]["id"], "revision": terminal["revision"],
                   "predicates_sha256": _digest(final), "verified_at_unix": time.time(),
                   "action_ids": [outcome["action"]["id"] for outcome in outcomes]}
        return {"status": "complete", "outcomes": outcomes, "terminal": terminal, "terminal_receipt": receipt}
    except _Cancelled as exc:
        return {"status": "cancelled", "outcomes": outcomes, "reason": str(exc)}
    except Exception as exc:
        return {"status": "needs_replan", "outcomes": outcomes, "reason": str(exc)[:1000]}
