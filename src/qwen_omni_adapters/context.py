"""Load the single observable catalog of model-facing context and tools."""

from __future__ import annotations

import copy
import getpass
import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

CONTEXT_SCHEMA = "robit.omni.context.v1"
CONTEXT_FILE_ENV = "OMNI_CONTEXT_FILE"


class ContextConfigError(RuntimeError):
    """The model-facing context catalog is missing or malformed."""


def _default_path() -> Path:
    return Path(str(files("qwen_omni_adapters").joinpath("context.json")))


def load_context(path: Path | str | None = None) -> dict[str, Any]:
    """Read and validate one context catalog without retaining mutable state."""

    selected = Path(path or os.environ.get(CONTEXT_FILE_ENV) or _default_path()).expanduser()
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContextConfigError(f"could not read context catalog {selected}: {exc}") from exc
    except ValueError as exc:
        raise ContextConfigError(f"context catalog {selected} is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != CONTEXT_SCHEMA:
        raise ContextConfigError(f"context catalog {selected} must use schema {CONTEXT_SCHEMA}")
    for section in ("prompts", "directives", "control_tools"):
        if not isinstance(value.get(section), dict):
            raise ContextConfigError(f"context catalog section {section} must be an object")
    tools = value.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ContextConfigError("context catalog tools must be a non-empty array")
    names: set[str] = set()
    for entry in tools:
        schema = entry.get("schema") if isinstance(entry, Mapping) else None
        function = schema.get("function") if isinstance(schema, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name or name in names:
            raise ContextConfigError("every context tool must have a unique function name")
        admission = entry.get("memory_admission", "standard")
        if admission not in {"standard", "bounded", "executor", "control"}:
            raise ContextConfigError(
                f"context tool {name} has invalid memory_admission {admission!r}"
            )
        reserve = entry.get("memory_reserve_gib")
        if reserve is not None and (
            isinstance(reserve, bool)
            or not isinstance(reserve, (int, float))
            or not 0.0 <= float(reserve) <= 64.0
        ):
            raise ContextConfigError(
                f"context tool {name} has invalid memory_reserve_gib {reserve!r}"
            )
        if admission == "executor" and reserve is None:
            raise ContextConfigError(
                f"context executor tool {name} must declare memory_reserve_gib"
            )
        names.add(name)
    families = value.get("tool_families")
    if not isinstance(families, Mapping) or not families:
        raise ContextConfigError("context catalog tool_families must be a non-empty object")
    assigned: dict[str, str] = {}
    for raw_family, raw_definition in families.items():
        family = str(raw_family)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", family) or family == "uncertain":
            raise ContextConfigError(f"invalid tool family name {family!r}")
        if not isinstance(raw_definition, Mapping):
            raise ContextConfigError(f"tool family {family} must be an object")
        description = raw_definition.get("description")
        members = raw_definition.get("tools")
        if not isinstance(description, str) or not description.strip():
            raise ContextConfigError(f"tool family {family} requires a description")
        if not isinstance(members, list) or not members or len(members) > 4:
            raise ContextConfigError(
                f"tool family {family} must contain between one and four tools"
            )
        for raw_member in members:
            member = str(raw_member)
            if member not in names or member == "tool_search":
                raise ContextConfigError(
                    f"tool family {family} contains unknown or gateway tool {member!r}"
                )
            if member in assigned:
                raise ContextConfigError(
                    f"tool {member} belongs to both {assigned[member]} and {family}"
                )
            assigned[member] = family
    unassigned = sorted(names - {"tool_search"} - set(assigned))
    if unassigned:
        raise ContextConfigError(
            f"context tools missing a typed family: {', '.join(unassigned)}"
        )
    return value


@lru_cache(maxsize=1)
def context_catalog() -> dict[str, Any]:
    """Return the process-wide immutable-by-convention context catalog."""

    return load_context()


def context_text(section: str, name: str) -> str:
    value = context_catalog().get(section)
    text = value.get(name) if isinstance(value, Mapping) else None
    if not isinstance(text, str) or not text.strip():
        raise ContextConfigError(f"missing context text {section}.{name}")
    return text.strip()


def live_call_system_prompt() -> str:
    """Compose the spoken-turn policy with its machine-only response control."""

    return "\n\n".join(
        (
            context_text("prompts", "live_call_system"),
            context_text("directives", "live_response_control"),
        )
    )


def context_value(section: str, name: str) -> Any:
    value = context_catalog().get(section)
    item = value.get(name) if isinstance(value, Mapping) else None
    if item is None:
        raise ContextConfigError(f"missing context value {section}.{name}")
    return copy.deepcopy(item)


_NUMERIC_COORDINATE_TUPLE = re.compile(
    r"\(\s*-?\d{1,5}(?:\.\d+)?\s*,\s*-?\d{1,5}(?:\.\d+)?"
    r"(?:\s*,\s*-?\d{1,5}(?:\.\d+)?){0,2}\s*\)"
)


def without_parent_frame_coordinates(value: str, *, max_chars: int = 2000) -> str:
    """Retain visual identity/orientation text but remove stale geometry."""

    bounded = " ".join(str(value).split())[: max(0, int(max_chars))]
    return _NUMERIC_COORDINATE_TUPLE.sub(
        "[parent-frame coordinates omitted]", bounded
    )


def runtime_agent_name() -> str:
    """Resolve the conversational identity from configuration or OS account."""

    configured = os.environ.get("OMNI_AGENT_NAME", "").strip()
    if configured:
        raw = configured
    else:
        try:
            import pwd

            raw = pwd.getpwuid(os.getuid()).pw_name
        except (ImportError, KeyError, OSError):
            raw = getpass.getuser()
    # This value becomes prompt text. Keep a human-readable account label, but
    # never allow control characters or an environment value to add policy.
    normalized = " ".join(str(raw).split())
    normalized = re.sub(r"[^\w .@+-]", "", normalized, flags=re.UNICODE)[:64]
    return normalized or "local-agent"


def runtime_identity_context() -> str:
    """Return bounded self-state grounding for the language system prompt."""

    return context_text("directives", "runtime_identity").format(
        agent_name=runtime_agent_name()
    )


def configured_tools() -> list[dict[str, Any]]:
    return [copy.deepcopy(entry) for entry in context_catalog()["tools"]]


def configured_tool_families() -> dict[str, dict[str, Any]]:
    """Return the explicit capability-family contract used for tool paging."""

    return copy.deepcopy(context_catalog()["tool_families"])


def retained_tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return configured routing gateways present in a supplied tool set."""

    configured = {
        str(entry["schema"]["function"]["name"]): entry
        for entry in configured_tools()
        if isinstance(entry, Mapping)
    }
    supplied: set[str] = set()
    for schema in tools:
        function = schema.get("function") if isinstance(schema, Mapping) else None
        if isinstance(function, Mapping) and function.get("name"):
            supplied.add(str(function["name"]))
    return {
        name
        for name in supplied
        if str(configured.get(name, {}).get("routing_role") or "") == "gateway"
    }
