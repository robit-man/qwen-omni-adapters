"""Offline decision replay scores offered, chosen and resolved separately."""
import json
from pathlib import Path

import pytest

from runtime import replay_planner_decisions as replay

CASES = Path(__file__).parent / "live" / "native_decision_cases"


def case(name):
    return json.loads((CASES / f"{name}.json").read_text())


def direct_reply(case_value, steps):
    return {"assessment": {"source_epoch": "x", "source_phase": 1, "decision": "start", "summary": "s"},
            "phase": {"kind": "ui", "steps": steps, "final_predicates": []}}


class Client:
    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def post(self, url, json, timeout):  # noqa: A002 - mirrors httpx.Client.post
        self.requests.append(json)
        content = self.reply if isinstance(self.reply, str) else __import__("json").dumps(self.reply)
        return type("Response", (), {"json": lambda _self: {"message": {"content": content}}})()


def test_manifest_hashes_match_frozen_cases():
    import hashlib
    manifest = json.loads((CASES / "manifest.json").read_text())
    for row in manifest["cases"]:
        assert hashlib.sha256((CASES / row["file"]).read_bytes()).hexdigest() == row["sha256"]


def test_projected_input_is_offered_and_exact_double_click_scores_chosen():
    value = case("run007-p1-input-projected")
    target = value["label"]["target_node"]
    active = value["packet"]["current_state"]["state"]["scope"]["id"]
    steps = [{"id": "a", "verb": "switch_window", "target": {"node_id": active}, "args": {}},
             {"id": "b", "verb": "double_click", "target": {"node_id": target}, "args": {}}]
    row = replay.run_case(value, "direct", adapter_url="x", model="m", client=Client(direct_reply(value, steps)))
    assert row["offered"] and row["chosen"] and row["resolved"] and row["error"] is None


@pytest.mark.parametrize("target,verb,chosen,resolved", [
    ("window", "double_click", False, True), ("role_button", "focus", False, False), ("input", "click", False, True)])
def test_wrong_target_ambiguous_role_or_wrong_verb_is_not_chosen(target, verb, chosen, resolved):
    value = case("run007-p1-input-projected")
    active = value["packet"]["current_state"]["state"]["scope"]["id"]
    selector = {"window": {"node_id": active}, "role_button": {"role": "button", "scope_id": active},
                "input": {"node_id": value["label"]["target_node"]}}[target]
    row = replay.run_case(value, "direct", adapter_url="x", model="m",
                          client=Client(direct_reply(value, [{"id": "b", "verb": verb, "target": selector, "args": {}}])))
    assert (row["chosen"], row["resolved"]) == (chosen, resolved)


def test_unprojected_target_is_reported_not_offered():
    value = case("run006-p1-input-not-projected")
    row = replay.score(value, None)
    assert row["offered"] is False and row["chosen"] is False


def test_negative_control_requires_no_input_and_invalid_replies_are_recorded():
    value = case("derived-run007-p1-no-input-goal")
    blocked = {"assessment": {}, "phase": {"kind": "blocked", "reason": "nothing to do"}}
    assert replay.run_case(value, "direct", adapter_url="x", model="m", client=Client(blocked))["chosen"] is True
    row = replay.run_case(value, "direct", adapter_url="x", model="m", client=Client("not json"))
    assert row["error"] and row["chosen"] is False


def test_compact_request_is_rebuilt_from_the_frozen_packet_and_is_smaller():
    value = case("run007-p1-input-projected")
    direct, _ = replay.build_request(value, "direct", "m")
    compact, binding = replay.build_request(value, "compact", "m")
    assert binding is not None
    assert sum(len(m["content"]) for m in compact["messages"]) * 2 < sum(len(m["content"]) for m in direct["messages"])
