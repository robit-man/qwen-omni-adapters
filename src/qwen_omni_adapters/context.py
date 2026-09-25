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


_TOOL_TOKEN_PATTERN = re.compile(r"[\w][\w'-]{1,}", re.UNICODE)


def _ordered_tool_tokens(value: str) -> list[str]:
    return _TOOL_TOKEN_PATTERN.findall(value.casefold())


def _tool_relevance_score(query: str, document: str) -> float:
    """Score a compact tool descriptor without invoking another model."""

    query_terms = _ordered_tool_tokens(query)
    document_terms = _ordered_tool_tokens(document)
    if not query_terms or not document_terms:
        return 0.0
    document_set = set(document_terms)
    document_bigrams = {
        f"{document_terms[index]} {document_terms[index + 1]}"
        for index in range(len(document_terms) - 1)
    }
    matched = 0.0
    weight = float(len(query_terms))
    for term in query_terms:
        if term in document_set:
            matched += 1.0
        elif any(
            len(term) >= 4
            and len(candidate) >= 4
            and (candidate.startswith(term) or term.startswith(candidate))
            for candidate in document_set
        ):
            matched += 0.5
    for index in range(len(query_terms) - 1):
        weight += 2.0
        if f"{query_terms[index]} {query_terms[index + 1]}" in document_bigrams:
            matched += 2.0
    phrase = " ".join(query_terms)
    if len(query_terms) > 1 and phrase in " ".join(document_terms):
        matched += 1.0
        weight += 1.0
    return min(1.0, matched / max(1.0, weight))


def rank_tool_names(
    query: str,
    tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    limit: int = 3,
) -> list[str]:
    """Return a bounded relevant subset of supplied tool schemas.

    This is reversible prompt selection, not action selection: the language
    model still chooses a tool and constructs its arguments, and policy still
    authorizes execution. Catalog hints are observable configuration. Unknown
    client-owned schemas participate through their name and description.
    """

    if re.search(
        r"\b(?:what can (?:you|this (?:portal|system|assistant)) do|"
        r"(?:your|portal|system) (?:capabilities|abilities))\b",
        query.casefold(),
    ):
        query = f"{query} portal capabilities abilities available actions"
    entries = configured_tools()
    metadata = {
        str(entry["schema"]["function"]["name"]): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    candidates: Sequence[Mapping[str, Any]] = (
        tools
        if tools is not None
        else [entry["schema"] for entry in entries if isinstance(entry, Mapping)]
    )
    ranked: list[tuple[float, str]] = []
    for schema in candidates:
        function = schema.get("function") if isinstance(schema, Mapping) else None
        name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
        if not name or name == "tool_search":
            continue
        entry = metadata.get(name, {})
        hints = str(entry.get("discovery_hints") or "")
        description = str(function.get("description") or "")
        positive = _tool_relevance_score(query, f"{name} {hints}")
        descriptive = _tool_relevance_score(query, description)
        score = positive * 2.0 + descriptive * 0.25
        if score >= 0.06:
            ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        return []
    minimum = max(0.06, ranked[0][0] * 0.6)
    bounded = max(1, min(8, int(limit)))
    return [name for score, name in ranked if score >= minimum][:bounded]


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
