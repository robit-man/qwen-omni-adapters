"""Bounded host-verified native prefixes; opaque input always yields to planning."""
from __future__ import annotations

import copy
from collections.abc import Callable

from qwen_omni_adapters.computer_use import ComputerUseError, parse_step


def run_verified_prefix(steps, *, execute_step: Callable[[dict, int], dict],
                        cancelled: Callable[[], bool] = lambda: False,
                        strategy_ledger=None, source_phase=None) -> dict:
    """Compile/read back through the trusted callback, never use model oracles.

    ``execute_step`` owns fresh observations, operation compilation, authority,
    deadlines and the guarded native controller. The entire plan is parsed
    before input. No model terminal predicate is evaluated by this helper.
    """
    if not isinstance(steps, list) or not 1 <= len(steps) <= 25:
        raise ComputerUseError("Expected one to 25 bounded native prefix steps")
    prepared = [parse_step(step) for step in steps]
    outcomes, contexts, policies, selections, points = [], [], [], [], []
    last, verified_count = {}, 0
    attempted_count, stop = 0, "verified_prefix_exhausted"
    for index, step in enumerate(prepared):
        if cancelled():
            stop = "cancelled"
            break
        try:
            result = execute_step(copy.deepcopy(step), index)
        except Exception as exc:
            # A callback failure cannot establish that no input happened.
            result = {"status": "needs_replan", "error": "ui_prefix_callback_failed",
                "reason": "Immediate callback failed: " + type(exc).__name__,
                "input_attempted": None, "effect_receipt": {"effect_state": "unknown",
                    "environment_may_have_changed": True}, "task_progress": False}
        if not isinstance(result, dict):
            raise ComputerUseError("Prefix callback returned no typed result")
        continuation = result.get("prefix_continuation")
        if (index > 0 and isinstance(continuation, dict)
                and continuation.get("schema") == "robit.omni.native-prefix-continuation.v1"
                and continuation.get("input_authority") is False
                and result.get("status") == "needs_replan" and result.get("dispatch_status") == "not_dispatched"
                and result.get("input_attempted") is False and result.get("task_progress") is False
                and type(result.get("new_effect_count")) is int and result["new_effect_count"] == 0
                and result.get("immediate_effect_contract", {}).get("mode") == "observation_boundary"
                and result.get("effect_receipt", {}).get("effect_state") == "absent"
                and result.get("effect_receipt", {}).get("environment_may_have_changed") is False
                and not any(result.get(key) for key in
                    ("outcomes", "dispatch_record", "receipt", "policy_receipts", "point_receipts", "pointer_recovery"))):
            last = copy.deepcopy(result)
            last["prefix_continuation"].update(step_index=index, step=copy.deepcopy(steps[index]))
            stop = "unannotated_observation_boundary"
            break
        attempted_count += 1
        # This must happen before another internal step; the outer task also
        # records the returned historical receipts, without new input authority.
        if strategy_ledger is not None:
            strategy_ledger.record(result, source_phase=source_phase)
        last = copy.deepcopy(result)
        raw = result.get("outcomes")
        if not isinstance(raw, list):
            raw = [result["dispatch_record"]] if isinstance(result.get("dispatch_record"), dict) else []
        if len(raw) > 1:
            raise ComputerUseError("An immediate prefix operation returned multiple inputs")
        outcomes.extend(copy.deepcopy(raw))
        contexts.append({"step_index": index, "status": result.get("status"),
            "outcome_count": len(raw), **{key: copy.deepcopy(result[key]) for key in
                ("immediate_effect_contract", "immediate_transition_receipt", "action_handoff",
                 "action_handoff_revalidation", "verification_only_receipt") if key in result}})
        for key, destination in (("policy_receipts", policies), ("selection_receipts", selections),
                                 ("point_receipts", points)):
            destination.extend(copy.deepcopy(result.get(key, [])))
        if cancelled() or result.get("status") == "cancelled":
            stop = "cancelled"
            break
        mode = result.get("immediate_effect_contract", {}).get("mode")
        if mode == "observation_boundary":
            stop = "opaque_input_boundary"
            break
        if (mode != "verified_step" or result.get("status") != "complete"
                or result.get("error") or result.get("phase_handoff", {}).get("terminal_verified") is not True
                or not (len(raw) == 1 and raw[0].get("status") == "verified"
                    or not raw and result.get("verification_only_receipt", {}).get("status") == "verified"
                    and result["verification_only_receipt"].get("input_attempted") is False)):
            stop = "unverified_operation"
            break
        verified_count += 1
    result = last
    result.update(outcomes=outcomes, policy_receipts=policies, selection_receipts=selections,
        point_receipts=points, task_progress=False)
    if stop == "cancelled":
        result["status"] = "cancelled"
    elif stop == "verified_prefix_exhausted":
        result["status"] = "complete"
    elif stop == "unverified_operation" and result.get("status") == "complete":
        result.update(status="needs_replan", error="ui_prefix_immediate_contract_unverified")
    prefix_verified = stop == "verified_prefix_exhausted" and verified_count == len(prepared)
    result["verified_prefix_receipt"] = {"schema": "robit.omni.native-verified-prefix.v1",
        "input_authority": False, "task_progress": False, "goal_verified": False,
        "model_terminal_predicates_verified": False, "stop": stop,
        "planned_step_count": len(prepared), "attempted_step_count": attempted_count,
        "verified_prefix_count": verified_count, "unattempted_step_count": len(prepared) - attempted_count,
        "all_immediate_readbacks_verified": prefix_verified, "step_contexts": contexts}
    if stop == "unannotated_observation_boundary":
        result["verified_prefix_receipt"]["continuation"] = result.pop("prefix_continuation")
    result["phase_handoff"] = {"schema": "robit.omni.ui-phase-handoff.v1", "executor": "host_verified_prefix",
        "terminal_verified": False, "goal_verified": False, "immediate_prefix_verified": prefix_verified,
        "new_effect_count": sum(row.get("effect_receipt", {}).get("effect_state") == "applied" for row in outcomes),
        "next": "plan_next_phase" if prefix_verified else "replan"}
    return result
