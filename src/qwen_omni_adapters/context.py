"""Load the single observable catalog of model-facing context and tools."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
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

    selected = Path(
        path or os.environ.get(CONTEXT_FILE_ENV) or _default_path()
    ).expanduser()
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContextConfigError(f"could not read context catalog {selected}: {exc}") from exc
    except ValueError as exc:
        raise ContextConfigError(f"context catalog {selected} is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != CONTEXT_SCHEMA:
        raise ContextConfigError(
            f"context catalog {selected} must use schema {CONTEXT_SCHEMA}"
        )
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


def context_value(section: str, name: str) -> Any:
    value = context_catalog().get(section)
    item = value.get(name) if isinstance(value, Mapping) else None
    if item is None:
        raise ContextConfigError(f"missing context value {section}.{name}")
    return copy.deepcopy(item)


def configured_tools() -> list[dict[str, Any]]:
    return [copy.deepcopy(entry) for entry in context_catalog()["tools"]]
