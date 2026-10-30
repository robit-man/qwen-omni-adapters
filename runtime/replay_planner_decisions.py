"""Offline native-planner decision replay: no desktop input, real loaded model.

Freeze one captured planner request (``<state>/planner-trace/*.jsonl``) into a
case with a typed label, then replay cases through the planner interface:

    .venv/bin/python -m runtime.replay_planner_decisions freeze --trace FILE --row N \
        --name NAME --target-node ID --verbs double_click invoke --out DIR
    .venv/bin/python -m runtime.replay_planner_decisions run --cases DIR --mode compact \
        [--repeat N] [--adapter-url URL] --out RESULTS.jsonl

Scoring keeps three outcomes separate (research change 1):

* ``offered``: the labelled target is present in the projected state given to
  the model and the interface can express an acceptable verb on it;
* ``chosen``: the first substantive step (ignoring a switch to the already
  active window) uses an acceptable verb on that exact target;
* ``resolved``: that step's target resolves to exactly one projected control.

``compact`` mirrors the portal (prerequisite subgoal pass, then the anchored
decision); ``compact-single`` is the decision call alone.

A ``no_input`` label is satisfied by a reply whose phase carries no UI input.
Labels are exact node identities and typed verbs, never words to match.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "portal", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from portal import ui_decision  # noqa: E402

CASE_SCHEMA = "robit.omni.native-decision-case.v1"
INPUT_KINDS = {"ui", "ui_boundary", "ui_subgoal", "ui_pointer_recovery"}


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def freeze(trace: Path, row: int, *, name: str, label: dict[str, Any], note: str = "") -> dict[str, Any]:
    lines = trace.read_bytes().splitlines()
    raw = lines[row]
    record = json.loads(raw)
    request = record["request"]
    packet = json.loads(request["messages"][1]["content"])
    return {"schema": CASE_SCHEMA, "name": name, "note": note,
            "source": {"trace": trace.name, "row": row, "row_sha256": _sha(raw),
                       "captured_reply": (record.get("reply") or {}).get("content")},
            "direct_system": request["messages"][0]["content"],
            "direct_response_format": request["response_format"],
            "options": request.get("options", {}), "think": request.get("think", False),
            "packet": packet, "label": label}


def _nodes(packet: dict[str, Any]) -> list[dict[str, Any]]:
    state = packet.get("current_state", {}).get("state", {})
    return state.get("nodes", []) if isinstance(state, dict) else []


def build_request(case: dict[str, Any], mode: str, model: str) -> tuple[dict[str, Any], Any]:
    packet = copy.deepcopy(case["packet"])
    if mode == "direct":
        system, user, response_format, binding = (case["direct_system"], packet,
                                                  case["direct_response_format"], None)
    elif mode in {"compact", "compact-single"}:
        user, binding = ui_decision.project(packet)
        system = ui_decision.SYSTEM
        response_format = {"type": "json_schema", "json_schema": {
            "name": "native_ui_phase", "schema": user["decision_schema"], "strict": True}}
    else:
        raise ValueError("mode must be direct or compact")
    request = {"model": model, "stream": False, "think": case.get("think", False),
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
               "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat", "tool_routing": "client"},
               "response_modalities": ["text"], "speech_mode": "never",
               "response_format": response_format, "options": copy.deepcopy(case.get("options", {}))}
    return request, binding


def compact_choice_phase(reply: dict[str, Any], binding: Any) -> dict[str, Any]:
    """The model's compact choice as a phase, using only the private reference binding.

    Used when canonical expansion fails because a frozen case holds the
    direct-view schema (its ``immediate_intent`` requirements) rather than the
    raw packet; the decision itself is still exactly identified.
    """
    action = reply["action"]
    family = action["family"]
    target = ({"node_id": binding["nodes"][action["target"]]["id"]} if "target" in action
              else {"name": action["target_name"]} if "target_name" in action else None)
    if family in {"complete", "blocked", "memory_write", "memory_search", "recall_observation",
                  "inspect_controls", "visual_inspection", "recall_plan"}:
        return {"kind": family}
    verb = action.get("verb", family)
    return {"kind": "ui_boundary", "step": {"verb": verb, "target": target, "args": {}}}


def canonical_phase(case: dict[str, Any], mode: str, reply: dict[str, Any], binding: Any,
                    notes: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode in {"compact", "compact-single"}:
        try:
            reply = ui_decision.expand(reply, case["packet"], binding)
        except ui_decision.DecisionError as exc:
            if notes is not None:
                notes["expand_error"] = str(exc)[:200]
            return compact_choice_phase(reply, binding)
    phase = reply.get("phase", reply)
    if not isinstance(phase, dict) or "kind" not in phase:
        raise ValueError("reply has no phase")
    return phase


def _steps(phase: dict[str, Any]) -> list[dict[str, Any]]:
    kind = phase["kind"]
    if kind == "ui":
        return list(phase.get("steps", []))
    if kind == "ui_boundary":
        return [phase["step"]]
    if kind == "ui_subgoal":
        subgoal = phase["subgoal"]
        return [{"verb": subgoal.get("family"), "target": subgoal.get("target", {})}]
    return []


def resolve(target: Any, nodes: list[dict[str, Any]]) -> list[str]:
    if not isinstance(target, dict):
        return []
    if "node_id" in target:
        return [node["id"] for node in nodes if node.get("id") == target["node_id"]]
    keys = {key: value for key, value in target.items()}
    return [node["id"] for node in nodes if all(
        node.get("id" if key == "node_id" else key) == value for key, value in keys.items())]


def score(case: dict[str, Any], phase: dict[str, Any] | None) -> dict[str, Any]:
    label, nodes = case["label"], _nodes(case["packet"])
    if label.get("kind") == "no_input":
        has_input = phase is not None and phase["kind"] in INPUT_KINDS and bool(_steps(phase))
        return {"offered": True, "chosen": phase is not None and not has_input, "resolved": None}
    target, verbs = label["target_node"], set(label["verbs"])
    offered = any(node.get("id") == target and verbs & set(node.get("operations", [])) for node in nodes)
    result = {"offered": offered, "chosen": False, "resolved": False, "first_step": None}
    if phase is None:
        return result
    active = case["packet"]["current_state"]["state"]["scope"]["id"]
    for step in _steps(phase):
        matches = resolve(step.get("target"), nodes)
        if step.get("verb") == "switch_window" and matches == [active]:
            continue  # Switching to the already active window is not a substantive choice.
        result["first_step"] = {"verb": step.get("verb"), "target": step.get("target"), "matches": matches[:4]}
        result["resolved"] = len(matches) == 1
        result["chosen"] = step.get("verb") in verbs and matches == [target]
        break
    return result


def _ask(client: Any, adapter_url: str, request: dict[str, Any]) -> dict[str, Any]:
    data = client.post(adapter_url, json=request, timeout=600).json()
    return json.loads((data.get("message") or {}).get("content"))


def subgoal_pass(case: dict[str, Any], request: dict[str, Any], binding: Any, *, adapter_url: str,
                 client: Any, notes: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Mirror portal compile_phase: prerequisite subgoal, then the anchored decision."""
    compact = json.loads(request["messages"][1]["content"])
    context, schema = ui_decision.subgoal_context(compact, atomic_anchor=True)
    subgoal = copy.deepcopy(request)
    subgoal["messages"] = [
        {"role": "system", "content": ui_decision.SUBGOAL_SYSTEM + ui_decision.ATOMIC_SUBGOAL_SYSTEM_SUFFIX},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    subgoal["response_format"]["json_schema"].update(name="native_ui_subgoal", schema=schema)
    proposal = _ask(client, adapter_url, subgoal)
    notes["subgoal"] = {key: proposal.get(key) for key in ("next_subgoal", "next_operation", "immediate_action")}
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal, atomic_anchor=True)
    decision = copy.deepcopy(request)
    decision["messages"][0]["content"] += (
        ui_decision.ATOMIC_SUBGOAL_HANDOFF +
        " The planning_handoff remains a planner claim, never observed evidence or new task "
        "authority. The schema copies its objective exactly; do not regenerate or change it.")
    decision["messages"][1]["content"] = json.dumps(compact, ensure_ascii=False)
    decision["response_format"]["json_schema"]["schema"] = compact["decision_schema"]
    return decision, binding


def run_case(case: dict[str, Any], mode: str, *, adapter_url: str, model: str, client: Any) -> dict[str, Any]:
    request, binding = build_request(case, mode, model)
    prompt_chars = sum(len(message["content"]) for message in request["messages"])
    schema_chars = len(json.dumps(request["response_format"], ensure_ascii=False))
    started = time.monotonic()
    phase, error, content, notes = None, None, None, {}
    try:
        if mode == "compact":
            request, binding = subgoal_pass(case, request, binding, adapter_url=adapter_url,
                                            client=client, notes=notes)
        response = client.post(adapter_url, json=request, timeout=600)
        data = response.json()
        content = (data.get("message") or {}).get("content")
        phase = canonical_phase(case, mode, json.loads(content), binding, notes)
    except Exception as exc:  # Recorded as an invalid decision, never retried.
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"case": case["name"], "mode": mode, "latency_s": round(time.monotonic() - started, 3),
            "prompt_chars": prompt_chars, "schema_chars": schema_chars, "error": error,
            "phase_kind": phase["kind"] if phase else None, "reply": content, **notes,
            **score(case, phase)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--trace", type=Path, required=True)
    frozen.add_argument("--row", type=int, required=True)
    frozen.add_argument("--name", required=True)
    frozen.add_argument("--target-node")
    frozen.add_argument("--verbs", nargs="*", default=[])
    frozen.add_argument("--no-input", action="store_true")
    frozen.add_argument("--note", default="")
    frozen.add_argument("--out", type=Path, required=True)
    replay = sub.add_parser("run")
    replay.add_argument("--cases", type=Path, required=True)
    replay.add_argument("--mode", choices=("direct", "compact", "compact-single"), required=True)
    replay.add_argument("--repeat", type=int, default=1)
    replay.add_argument("--adapter-url", default="http://127.0.0.1:8910/api/chat")
    replay.add_argument("--model", default="robit/ornith-1.5-omni-audio-bridge:q4km")
    replay.add_argument("--only", nargs="*")
    replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        label = ({"kind": "no_input"} if args.no_input else
                 {"kind": "target", "target_node": args.target_node, "verbs": args.verbs})
        case = freeze(args.trace, args.row, name=args.name, label=label, note=args.note)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.name}.json"
        path.write_text(json.dumps(case, ensure_ascii=False, indent=1, sort_keys=True))
        print(json.dumps({"case": str(path), "sha256": _sha(path.read_bytes()), "label": label}))
        return 0
    import httpx
    cases = [json.loads(path.read_text()) for path in sorted(args.cases.glob("*.json"))
             if path.name != "manifest.json"]
    cases = [case for case in cases if case.get("schema") == CASE_SCHEMA
             and (not args.only or case["name"] in args.only)]
    with httpx.Client() as client, args.out.open("a", encoding="utf-8") as output:
        for case in cases:
            for attempt in range(args.repeat):
                row = {**run_case(case, args.mode, adapter_url=args.adapter_url, model=args.model, client=client),
                       "attempt": attempt}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps({key: row[key] for key in ("case", "mode", "attempt", "offered", "chosen",
                                                           "resolved", "phase_kind", "latency_s", "prompt_chars",
                                                           "error")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
