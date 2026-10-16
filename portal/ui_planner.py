"""Model-facing phase planning over the existing native/rapid UI controller.

This is an automatic-tool-loop profile, not a new executor or browser. Direct
clients retain the canonical observe(step)/execute protocol. The server also
admits the profile at dispatch: a model schema alone is not an authority check.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

UI_TOOLS = frozenset({"native_ui", "computer_use"})
PLANNER_OPERATIONS = frozenset({"observe", "workflow", "verify", "cancel"})

_DESCRIPTION = (
    "Plan bounded semantic phases for the existing UI controller. First use "
    "observe with no step to inspect current controls. Then submit workflow "
    "with 1–25 typed steps and independent final_predicates for one causal "
    "phase. The controller observes fresh state for each step, resolves legal "
    "targets, takes a unique semantic choice directly, delegates ambiguity to "
    "the configured fast policy, and uses configured visual grounding when "
    "needed. It guards input and verifies each effect before advancing. It "
    "stops at the first unverified effect and returns a phase_handoff for "
    "replanning. You plan phases; never enumerate, invent or execute action "
    "IDs. There is no per-action execute operation in this agent profile. "
    "Use semantic role/name targets for controls in later phase states; copy "
    "a fresh full node_id only when necessary to disambiguate an already "
    "observed control. Preserve the canonical typed verb/args and expect "
    "contracts. Each expect must describe the actual effect, not merely "
    "focus. click requests physical mouse input; invoke is an accessibility "
    "action. type_text inserts into an already focused editable control; "
    "it never focuses, selects all or replaces text implicitly. Plan those "
    "prerequisites explicitly. Use individual key tokens such as "
    "[\"Control_L\",\"a\"]. When a different control shows the effect, bind "
    "expect to that observed effect control rather than $target. "
    "Final predicates verify the phase independently; $target is "
    "forbidden there. A complete phase is not completion of the whole user "
    "goal. Plan the next unmet phase after checking the handoff and freshest "
    "state. Use verify for an independent current-state check, or cancel to "
    "stop. No code, raw coordinates, input fallback or permission may be "
    "supplied by the planner."
)


def planner_tool_schemas(schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clone canonical schemas and remove per-action authority from the model.

    Update both root properties and oneOf alternatives. Some decoder paths
    inspect root properties while others consume only the oneOf branches.
    Keep the nested step definitions used by workflow unchanged.
    """
    projected = copy.deepcopy(list(schemas))
    for schema in projected:
        function = schema.get("function", {})
        name = function.get("name")
        if name not in UI_TOOLS:
            continue
        function["description"] = (
            "Operate the user's actual desktop applications and existing browser "
            "profile through native accessibility and guarded input. "
            if name == "native_ui" else
            "Operate the requested browser or desktop surface through its "
            "existing guarded controller. "
        ) + _DESCRIPTION
        parameters = function["parameters"]
        properties = parameters["properties"]
        properties.pop("action_id", None)
        properties.pop("step", None)
        properties["operation"]["enum"] = [
            operation for operation in properties["operation"]["enum"]
            if operation in PLANNER_OPERATIONS
        ]
        alternatives = []
        for alternative in parameters["oneOf"]:
            operation = alternative.get("properties", {}).get("operation", {}).get("const")
            if operation not in PLANNER_OPERATIONS:
                continue
            if operation == "observe":
                alternative["properties"].pop("step", None)
            alternatives.append(alternative)
        parameters["oneOf"] = alternatives
    return projected


def planner_call_error(name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Typed admission for model calls; never reinterpret or execute a bad plan."""
    if name not in UI_TOOLS:
        return None
    operation = arguments.get("operation", "observe")
    if (isinstance(operation, str) and operation in PLANNER_OPERATIONS
            and not (operation == "observe" and "step" in arguments)):
        return None
    return {
        "error": "ui_planner_phase_required",
        "dispatch_status": "not_dispatched",
        "message": (
            "This automatic agent plans phases. Inspect with observe without "
            "step, then submit workflow with typed steps and independent "
            "final_predicates. The controller owns action selection and execution."
        ),
        "allowed_operations": sorted(PLANNER_OPERATIONS),
    }
