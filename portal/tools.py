"""Allowlisted tool harness for the authenticated Omni demonstration portal.

The schemas, chained execution model, no-key DuckDuckGo HTML search, verified
fetch receipts, bounded crawl, and lexical memory ranking are derived from the
adjacent Omnius runtime. Interactive/JavaScript browsing remains a separate
rendered Chromium capability. Web/document results stay untrusted and every
stateful object is partitioned by opaque portal session rather than model-global
state.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import html
import ipaddress
import json
import math
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlsplit

import httpx

from qwen_omni_adapters.context import (
    configured_tool_families,
    configured_tools,
    context_text,
)
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPressure

try:
    from portal.background_tasks import TERMINAL_STATUSES, BackgroundTaskStore
    from portal.browser import (
        BrowserAutomationError,
        BrowserAutomationStore,
        BrowserDesktopUnavailable,
    )
    from portal.documents import DocumentError, SessionDocumentStore
    from portal.environment import runtime_environment_snapshot
    from portal.gui import GuiAutomation, GuiAutomationError
except ModuleNotFoundError:  # Direct script execution from portal/.
    from background_tasks import TERMINAL_STATUSES, BackgroundTaskStore
    from browser import (
        BrowserAutomationError,
        BrowserAutomationStore,
        BrowserDesktopUnavailable,
    )
    from documents import DocumentError, SessionDocumentStore
    from environment import runtime_environment_snapshot
    from gui import GuiAutomation, GuiAutomationError

MAX_SEARCH_RESULTS = 8
MAX_SEARCH_QUERY_CHARS = 500
MAX_SEARCH_BYTES = 2 * 1024 * 1024
MAX_FETCH_BYTES = 5 * 1024 * 1024
MAX_FETCH_CHARS = 12_000
MAX_REDIRECTS = 4
FETCH_CACHE_TTL_S = 60.0
MAX_WEB_INDEX_ENTRIES = 48
MAX_WEB_INDEX_CHARS = 128_000
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
MAX_MEMORY_ENTRIES = 64
MAX_MEMORY_ENTRY_CHARS = 4_096
MAX_MEMORY_SESSION_CHARS = 32_768
MAX_SHELL_OUTPUT_BYTES = 64 * 1024
MAX_WORKSPACE_TEXT_CHARS = 65_536
MAX_SHELL_EFFECT_PATHS = 16
MAX_SHELL_EFFECT_ENTRIES = 2_048
MAX_SHELL_EFFECT_FILE_BYTES = 8 * 1024 * 1024
TOKEN_PATTERN = re.compile(r"[\w][\w'-]{1,}", re.UNICODE)


_CONFIGURED_TOOL_ENTRIES = configured_tools()
SAFE_TOOLS = [entry["schema"] for entry in _CONFIGURED_TOOL_ENTRIES]
TOOL_FAMILIES = configured_tool_families()
_TOOL_MEMORY_ADMISSION = {
    entry["schema"]["function"]["name"]: str(
        entry.get("memory_admission") or "standard"
    )
    for entry in _CONFIGURED_TOOL_ENTRIES
}
_TOOL_MEMORY_RESERVE_GIB = {
    entry["schema"]["function"]["name"]: float(entry["memory_reserve_gib"])
    for entry in _CONFIGURED_TOOL_ENTRIES
    if entry.get("memory_reserve_gib") is not None
}

_TOOL_SCHEMAS_BY_NAME = {item["function"]["name"]: item for item in SAFE_TOOLS}
# This is the entire contract sent on the first language pass. The complete
# allowlist remains available through /api/tools for clients that own their
# own loop, but the model sees one tiny discovery schema until it asks.
DISCOVERY_TOOLS = [_TOOL_SCHEMAS_BY_NAME["tool_search"]]


def tool_schemas(names: Sequence[str]) -> list[dict[str, Any]]:
    """Return allowlisted concrete schemas in requested order, once each."""

    found: list[dict[str, Any]] = []
    seen: set[str] = {"tool_search"}
    for raw_name in names:
        name = str(raw_name)
        schema = _TOOL_SCHEMAS_BY_NAME.get(name)
        if schema is None or name in seen:
            continue
        seen.add(name)
        found.append(schema)
    return found


class ToolInputError(ValueError):
    """A bounded error safe to return to the model as a tool result."""


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ToolInputError(f"{name} is required")
    if len(text) > maximum:
        raise ToolInputError(f"{name} exceeds {maximum} characters")
    return text


def _bounded_integer(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolInputError("numeric argument must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("numeric argument must be an integer") from exc
    return max(minimum, min(maximum, number))


def _shell_effect_paths(value: Any, working_directory: str) -> list[Path]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ToolInputError("mutation_paths must be an array")
    if len(value) > MAX_SHELL_EFFECT_PATHS:
        raise ToolInputError(
            f"mutation_paths contains more than {MAX_SHELL_EFFECT_PATHS} paths"
        )
    paths: list[Path] = []
    for raw in value:
        text = _bounded_text(raw, "mutation path", 4096)
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = Path(working_directory) / candidate
        paths.append(candidate.resolve(strict=False))
    return list(dict.fromkeys(paths))


def _shell_path_snapshot(path: Path) -> dict[str, Any]:
    """Return a bounded structural fingerprint for one declared effect path."""

    try:
        root = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    except OSError as exc:
        return {"exists": None, "error": type(exc).__name__}
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:
            target = ""
        return {
            "exists": True,
            "kind": "symlink",
            "target": target,
            "mtime_ns": root.st_mtime_ns,
        }
    if path.is_file():
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                remaining = MAX_SHELL_EFFECT_FILE_BYTES
                while remaining > 0 and (
                    chunk := source.read(min(1024 * 1024, remaining))
                ):
                    digest.update(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            return {
                "exists": True,
                "kind": "file",
                "bytes": root.st_size,
                "mtime_ns": root.st_mtime_ns,
                "error": type(exc).__name__,
            }
        return {
            "exists": True,
            "kind": "file",
            "bytes": root.st_size,
            "sha256": digest.hexdigest(),
            "content_hash_truncated": root.st_size > MAX_SHELL_EFFECT_FILE_BYTES,
        }
    if not path.is_dir():
        return {
            "exists": True,
            "kind": "other",
            "mode": root.st_mode,
            "mtime_ns": root.st_mtime_ns,
        }

    digest = hashlib.sha256()
    entries = 0
    truncated = False
    try:
        for current, directories, files in os.walk(path):
            directories.sort()
            files.sort()
            current_path = Path(current)
            for name in [*directories, *files]:
                candidate = current_path / name
                try:
                    stat = candidate.lstat()
                    relative = candidate.relative_to(path).as_posix()
                    kind = (
                        "l"
                        if candidate.is_symlink()
                        else "d"
                        if candidate.is_dir()
                        else "f"
                    )
                    digest.update(
                        f"{relative}\0{kind}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
                    )
                except OSError as exc:
                    digest.update(
                        f"{candidate.name}\0error\0{type(exc).__name__}\n".encode()
                    )
                entries += 1
                if entries >= MAX_SHELL_EFFECT_ENTRIES:
                    truncated = True
                    break
            if truncated:
                break
    except OSError as exc:
        return {
            "exists": True,
            "kind": "directory",
            "error": type(exc).__name__,
        }
    return {
        "exists": True,
        "kind": "directory",
        "entries": entries,
        "truncated": truncated,
        "digest": digest.hexdigest(),
    }


def _run_shell(
    command: Any,
    cwd: Any = None,
    timeout_seconds: Any = None,
    stdin: Any = None,
    memory_governor: MemoryGovernor | None = None,
    *,
    intent: Any = None,
    mutation_paths: Any = None,
) -> dict[str, Any]:
    """Run Bash and return a typed, executor-observed effect receipt."""

    source = _bounded_text(command, "command", 32_768)
    working_directory = str(cwd or "").strip() or str(Path.cwd())
    if len(working_directory) > 4096:
        raise ToolInputError("cwd exceeds 4096 characters")
    if not Path(working_directory).is_dir():
        raise ToolInputError(f"cwd is not a directory: {working_directory}")
    step_intent = str(intent or "inspect").strip().lower()
    if step_intent not in {
        "inspect",
        "mutate_filesystem",
        "mutate_runtime",
        "verify",
    }:
        raise ToolInputError(
            "intent must be inspect, mutate_filesystem, mutate_runtime, or verify"
        )
    effect_paths = _shell_effect_paths(mutation_paths, working_directory)
    if step_intent == "mutate_filesystem" and not effect_paths:
        raise ToolInputError(
            "mutate_filesystem requires at least one declared mutation_path"
        )
    if step_intent != "mutate_filesystem" and effect_paths:
        raise ToolInputError(
            "mutation_paths is valid only when intent is mutate_filesystem"
        )
    before_effects = {
        str(path): _shell_path_snapshot(path) for path in effect_paths
    }
    timeout = _bounded_integer(
        timeout_seconds,
        default=120,
        minimum=1,
        maximum=900,
    )
    if stdin is None:
        stdin_data: bytes | None = None
    elif not isinstance(stdin, str):
        raise ToolInputError("stdin must be a string")
    else:
        stdin_data = stdin.encode("utf-8")
        if len(stdin_data) > 65_536:
            raise ToolInputError("stdin exceeds 65536 UTF-8 bytes")
    if memory_governor is not None:
        memory_governor.require_capacity(
            "shell", _TOOL_MEMORY_RESERVE_GIB.get("shell", 0.5)
        )
    try:
        process = subprocess.Popen(
            ["/bin/bash", "-lc", source],
            cwd=working_directory,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolInputError(f"could not start shell: {exc}") from exc

    captured = {"stdout": bytearray(), "stderr": bytearray()}
    totals = {"stdout": 0, "stderr": 0}

    def drain(name: str, stream: Any) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            totals[name] += len(chunk)
            room = MAX_SHELL_OUTPUT_BYTES - len(captured[name])
            if room > 0:
                captured[name].extend(chunk[:room])

    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    done = threading.Event()
    watcher = None
    tripped = threading.Event()

    def cancel_for_pressure() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if memory_governor is not None:
        watcher, tripped = memory_governor.watch(
            "shell", cancel_for_pressure, done
        )
    writer: threading.Thread | None = None
    if stdin_data is not None and process.stdin is not None:

        def feed_stdin() -> None:
            try:
                process.stdin.write(stdin_data)
                process.stdin.close()
            except (BrokenPipeError, ValueError):
                pass

        writer = threading.Thread(target=feed_stdin, daemon=True)
        writer.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            cancel_for_pressure()
            process.wait()
    finally:
        done.set()
        if watcher is not None:
            watcher.join(timeout=1.0)
    for thread in threads:
        thread.join(timeout=2.0)
    if writer is not None:
        writer.join(timeout=2.0)

    if tripped.is_set() and memory_governor is not None:
        raise MemoryPressure(
            "shell",
            memory_governor.available_gib(),
            memory_governor.policy.hard_floor_gib,
        )

    after_effects = {
        str(path): _shell_path_snapshot(path) for path in effect_paths
    }
    changed_paths = [
        path for path in before_effects if before_effects[path] != after_effects[path]
    ]
    succeeded = process.returncode == 0 and not timed_out
    if step_intent == "mutate_filesystem" and succeeded and changed_paths:
        authority = "mutation"
        task_progress = True
    elif step_intent == "verify" and succeeded:
        authority = "verification"
        task_progress = False
    elif step_intent == "inspect":
        authority = "inspection"
        task_progress = False
    else:
        authority = "unverified_effect"
        task_progress = False
    return {
        "command": source,
        "cwd": working_directory,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdin_bytes": len(stdin_data or b""),
        "stdout": captured["stdout"].decode("utf-8", errors="replace"),
        "stderr": captured["stderr"].decode("utf-8", errors="replace"),
        "stdout_truncated": totals["stdout"] > MAX_SHELL_OUTPUT_BYTES,
        "stderr_truncated": totals["stderr"] > MAX_SHELL_OUTPUT_BYTES,
        "intent": step_intent,
        "task_progress": task_progress,
        "evidence_authority": authority,
        "effect_receipt": {
            "declared_paths": [str(path) for path in effect_paths],
            "changed_paths": changed_paths,
            "filesystem_change_verified": bool(changed_paths),
        },
    }


def _validate_workspace_text(path: Path, content: str) -> str:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".py":
            ast.parse(content, filename=str(path))
            return "python_ast_ok"
        if suffix == ".json":
            json.loads(content)
            return "json_parse_ok"
    except (SyntaxError, ValueError) as exc:
        raise ToolInputError(f"content not written: {exc}") from exc
    return "text"


def _workspace_file(
    action: Any,
    path: Any,
    *,
    content: Any = None,
    old_text: Any = None,
    new_text: Any = None,
    expected_occurrences: Any = None,
    expected_sha256: Any = None,
    max_chars: Any = None,
    offset_chars: Any = None,
    depth: Any = None,
) -> dict[str, Any]:
    """Perform one compact file operation with a bounded, verifiable receipt."""

    operation = str(action or "").strip().lower()
    if operation not in {"list", "read", "mkdir", "write", "replace"}:
        raise ToolInputError("action must be list, read, mkdir, write, or replace")
    raw_path = _bounded_text(path, "path", 4096)
    target = Path(raw_path).expanduser().resolve(strict=False)

    if operation == "mkdir":
        existed = target.is_dir()
        target.mkdir(parents=True, exist_ok=True)
        return {
            "action": operation,
            "path": str(target),
            "created": not existed,
            "is_directory": True,
        }

    if operation == "list":
        if not target.is_dir():
            raise ToolInputError(f"path is not a directory: {target}")
        maximum_depth = _bounded_integer(depth, default=2, minimum=1, maximum=5)
        entries: list[dict[str, Any]] = []
        for candidate in sorted(target.rglob("*")):
            relative = candidate.relative_to(target)
            if len(relative.parts) > maximum_depth:
                continue
            item: dict[str, Any] = {
                "path": str(relative),
                "kind": "directory" if candidate.is_dir() else "file",
            }
            if candidate.is_file():
                try:
                    item["bytes"] = candidate.stat().st_size
                except OSError:
                    item["bytes"] = None
            entries.append(item)
            if len(entries) >= 200:
                break
        return {
            "action": operation,
            "path": str(target),
            "depth": maximum_depth,
            "entries": entries,
            "truncated": len(entries) >= 200,
        }

    if not target.is_file() and operation in {"read", "replace"}:
        raise ToolInputError(f"path is not a file: {target}")

    if operation == "read":
        limit = _bounded_integer(
            max_chars, default=12_000, minimum=1, maximum=MAX_WORKSPACE_TEXT_CHARS
        )
        offset = _bounded_integer(
            offset_chars, default=0, minimum=0, maximum=100_000_000
        )
        try:
            source = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolInputError("path is not UTF-8 text") from exc
        segment = source[offset : offset + limit]
        return {
            "action": operation,
            "path": str(target),
            "content": segment,
            "offset_chars": offset,
            "next_offset_chars": offset + len(segment),
            "total_chars": len(source),
            "truncated": offset + len(segment) < len(source),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }

    if operation == "write":
        if not isinstance(content, str):
            raise ToolInputError("content must be a string")
        source = content
        existed = target.exists()
    else:
        if not isinstance(old_text, str) or not old_text:
            raise ToolInputError("old_text must be a non-empty string")
        if not isinstance(new_text, str):
            raise ToolInputError("new_text must be a string")
        source = target.read_text(encoding="utf-8")
        expected = _bounded_integer(
            expected_occurrences, default=1, minimum=1, maximum=1000
        )
        actual = source.count(old_text)
        if actual != expected:
            raise ToolInputError(
                f"old_text occurrence mismatch: expected {expected}, found {actual}"
            )
        source = source.replace(old_text, new_text)
        existed = True

    if len(source) > MAX_WORKSPACE_TEXT_CHARS:
        raise ToolInputError(
            f"resulting content exceeds {MAX_WORKSPACE_TEXT_CHARS} characters"
        )
    if expected_sha256 not in (None, ""):
        expected_hash = str(expected_sha256).strip().casefold()
        current_hash = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else "missing"
        )
        if current_hash != expected_hash:
            raise ToolInputError(
                f"sha256 mismatch: expected {expected_hash}, found {current_hash}"
            )
    validation = _validate_workspace_text(target, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(source)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {
        "action": operation,
        "path": str(target),
        "created": not existed,
        "chars": len(source),
        "bytes": len(source.encode("utf-8")),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "validation": validation,
    }


_MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_MATH_FUNCTIONS: dict[str, Callable[..., float | int]] = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


def _safe_math_eval(expression: Any) -> dict[str, Any]:
    source = _bounded_text(expression, "expression", 500)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(f"invalid arithmetic expression: {exc.msg}") from exc
    if sum(1 for _node in ast.walk(tree)) > 100:
        raise ToolInputError("arithmetic expression is too complex")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolInputError("only numeric constants are allowed")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in _MATH_CONSTANTS:
                raise ToolInputError(f"unknown math constant: {node.id}")
            return _MATH_CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > 100:
                    raise ToolInputError("power exponent exceeds 100")
                result = left**right
            else:
                raise ToolInputError("unsupported arithmetic operator")
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _MATH_FUNCTIONS.get(node.func.id)
            if function is None or node.keywords or len(node.args) > 8:
                raise ToolInputError("unsupported math function call")
            values = [evaluate(argument) for argument in node.args]
            if node.func.id == "pow" and len(values) >= 2 and abs(values[1]) > 100:
                raise ToolInputError("power exponent exceeds 100")
            return function(*values)
        raise ToolInputError("expression contains a forbidden operation")

    try:
        result = evaluate(tree)
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        raise ToolInputError(f"math evaluation failed: {exc}") from exc
    if isinstance(result, complex) or not math.isfinite(float(result)):
        raise ToolInputError("math result is not a finite real number")
    if abs(float(result)) > 1e100:
        raise ToolInputError("math result exceeds the magnitude limit")
    return {"expression": source, "result": result, "engine": "bounded_ast"}


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in TOKEN_PATTERN.findall(value):
        normalized = raw.lower()
        tokens.add(normalized)
        tokens.update(part for part in re.split(r"[_-]+", normalized) if len(part) > 1)
    return tokens


def _ordered_tokens(value: str) -> list[str]:
    return [
        part
        for raw in TOKEN_PATTERN.findall(value.lower())
        for part in re.split(r"[_-]+", raw)
        if len(part) > 1
    ]


def _term_match_score(query: str, document: str) -> float:
    """Small zero-dependency ranker adapted from Omnius memory retrieval."""

    query_terms = _ordered_tokens(query)
    document_terms = _ordered_tokens(document)
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
        elif any(candidate.startswith(term) or term.startswith(candidate) for candidate in document_set):
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


def discover_tool_names(family: str, limit: int = 4) -> list[str]:
    """Resolve an exact typed family without interpreting natural language."""

    definition = TOOL_FAMILIES.get(str(family))
    if not isinstance(definition, Mapping):
        return []
    members = definition.get("tools")
    if not isinstance(members, list):
        return []
    bounded = max(1, min(4, int(limit)))
    return [str(name) for name in members[:bounded]]


@dataclass(frozen=True)
class _MemoryEntry:
    topic: str
    key: str
    value: str
    saved_at: str


@dataclass
class _SessionMemory:
    entries: dict[tuple[str, str], _MemoryEntry] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)


class SessionMemoryStore:
    """Small in-memory recall store partitioned by the browser session cookie."""

    def __init__(self, *, ttl_s: float = 300.0) -> None:
        self.ttl_s = max(1.0, ttl_s)
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionMemory] = {}

    def _expire_locked(self, now: float) -> None:
        for key in [
            key
            for key, value in self._sessions.items()
            if now - value.last_seen >= self.ttl_s
        ]:
            self._sessions.pop(key, None)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_session_key(session_id), None)

    def write(self, session_id: str, topic: Any, key: Any, value: Any) -> dict[str, Any]:
        normalized_topic = _bounded_text(topic, "topic", 64)
        normalized_key = _bounded_text(key, "key", 128)
        normalized_value = _bounded_text(value, "value", MAX_MEMORY_ENTRY_CHARS)
        now = time.monotonic()
        session_key = _session_key(session_id)
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.setdefault(session_key, _SessionMemory())
            previous = session.entries.get((normalized_topic, normalized_key))
            projected = sum(len(item.value) for item in session.entries.values())
            if previous:
                projected -= len(previous.value)
            projected += len(normalized_value)
            if not previous and len(session.entries) >= MAX_MEMORY_ENTRIES:
                raise ToolInputError("session memory already contains 64 entries")
            if projected > MAX_MEMORY_SESSION_CHARS:
                raise ToolInputError("session memory exceeds 32768 characters")
            entry = _MemoryEntry(
                topic=normalized_topic,
                key=normalized_key,
                value=normalized_value,
                saved_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            session.entries[(normalized_topic, normalized_key)] = entry
            session.last_seen = now
        return {
            "stored": previous is None,
            "updated": previous is not None and previous.value != normalized_value,
            "unchanged": previous is not None and previous.value == normalized_value,
            "topic": normalized_topic,
            "key": normalized_key,
            "characters": len(normalized_value),
            "scope": "browser_session",
        }

    def read(self, session_id: str, topic: Any, key: Any) -> dict[str, Any]:
        normalized_topic = _bounded_text(topic, "topic", 64)
        normalized_key = _bounded_text(key, "key", 128)
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"found": False, "topic": normalized_topic, "key": normalized_key}
            session.last_seen = now
            entry = session.entries.get((normalized_topic, normalized_key))
            candidates = list(session.entries.values())
        if entry is None:
            suggestions = [
                {"topic": candidate.topic, "key": candidate.key}
                for candidate in candidates
                if _term_match_score(
                    f"{normalized_topic} {normalized_key}",
                    f"{candidate.topic} {candidate.key}",
                )
                > 0
            ][:8]
            return {
                "found": False,
                "topic": normalized_topic,
                "key": normalized_key,
                "related_keys": suggestions,
            }
        return {"found": True, **entry.__dict__, "scope": "browser_session"}

    def search(self, session_id: str, query: Any, max_results: Any = None) -> dict[str, Any]:
        normalized_query = _bounded_text(query, "query", 500)
        limit = _bounded_integer(max_results, default=5, minimum=1, maximum=8)
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                entries: list[_MemoryEntry] = []
            else:
                session.last_seen = now
                entries = list(session.entries.values())
        ranked: list[tuple[float, _MemoryEntry]] = []
        for entry in entries:
            score = _term_match_score(
                normalized_query,
                f"{entry.topic} {entry.key} {entry.value}",
            )
            if score > 0:
                ranked.append((score, entry))
        ranked.sort(key=lambda item: (item[0], item[1].saved_at), reverse=True)
        return {
            "query": normalized_query,
            "scope": "browser_session",
            "results": [
                {
                    "topic": entry.topic,
                    "key": entry.key,
                    "value": entry.value,
                    "saved_at": entry.saved_at,
                    "relevance": round(score, 4),
                }
                for score, entry in ranked[:limit]
            ],
        }

    def stats(self, session_id: str) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"entries": 0, "chars": 0}
            session.last_seen = now
            return {
                "entries": len(session.entries),
                "chars": sum(len(item.value) for item in session.entries.values()),
            }


def _default_resolver(hostname: str) -> list[str]:
    return sorted(
        {
            item[4][0]
            for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    )


def _validate_public_url(
    raw_url: Any,
    resolver: Callable[[str], Sequence[str]],
) -> str:
    value = _bounded_text(raw_url, "url", 4_096)
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolInputError("url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ToolInputError("credentialed URLs are blocked")
    hostname = parsed.hostname.rstrip(".").lower()
    if (
        hostname == "localhost"
        or hostname.endswith((".localhost", ".local"))
        or hostname in {"metadata", "metadata.google.internal"}
    ):
        raise ToolInputError("private or local URL hosts are blocked")
    try:
        addresses = [hostname] if _is_ip(hostname) else list(resolver(hostname))
    except OSError as exc:
        raise ToolInputError(f"could not resolve URL host: {hostname}") from exc
    if not addresses:
        raise ToolInputError(f"URL host resolved to no addresses: {hostname}")
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ToolInputError("URL host resolved to an invalid address") from exc
        if not parsed_address.is_global:
            raise ToolInputError("private, local, reserved, or metadata addresses are blocked")
    return value


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _strip_html(value: str) -> str:
    text = re.sub(
        r"<script\b[^>]*>[\s\S]*?</script>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<style\b[^>]*>[\s\S]*?</style>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"</(?:p|div|section|article|li|h[1-6]|tr|br)\s*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class _WebIndexEntry:
    url: str
    title: str
    snippet: str
    content: str
    indexed_at: float


@dataclass(frozen=True)
class _FetchReceipt:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    response_bytes: int
    response_sha256: str
    fetched_at: str

    def export(self) -> dict[str, Any]:
        return {
            "schema": "robit.omni.web-fetch-receipt.v1",
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "status": self.status,
            "content_type": self.content_type,
            "response_bytes": self.response_bytes,
            "response_sha256": self.response_sha256,
            "fetched_at": self.fetched_at,
            "link_status": "retrieved",
        }


@dataclass(frozen=True)
class _FetchedPage:
    receipt: _FetchReceipt
    mime_type: str
    body: bytes


@dataclass(frozen=True)
class _WebFetchCacheEntry:
    saved_at: float
    title: str
    text: str
    raw_html: str
    is_html: bool
    receipt: _FetchReceipt


@dataclass
class _WebSession:
    fetch_cache: dict[str, _WebFetchCacheEntry] = field(default_factory=dict)
    index: dict[str, _WebIndexEntry] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str, dict[str, str]]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self._attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        values = {key.lower(): value for key, value in attrs}
        if normalized_tag != "a" or self._href is not None:
            return
        self._href = str(values.get("href") or "").strip()
        self._attrs = {
            key: str(value or "")
            for key, value in values.items()
        }
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag != "a" or self._href is None:
            return
        self.links.append((self._href, " ".join(self._text), self._attrs))
        self._href = None
        self._text = []
        self._attrs = {}


def _binary_payload_kind(body: bytes, mime_type: str) -> str | None:
    """Classify known binary bytes before they can enter model context."""

    signatures = (
        (b"%PDF-", "PDF document"),
        (b"PK\x03\x04", "ZIP archive"),
        (b"PK\x05\x06", "ZIP archive"),
        (b"PK\x07\x08", "ZIP archive"),
        (b"SQLite format 3\x00", "SQLite database"),
        (b"\x1f\x8b", "gzip archive"),
        (b"BZh", "bzip2 archive"),
        (b"\xfd7zXZ\x00", "xz archive"),
        (b"7z\xbc\xaf'\x1c", "7-Zip archive"),
        (b"\x7fELF", "ELF executable"),
        (b"\x89PNG", "PNG image"),
        (b"\xff\xd8\xff", "JPEG image"),
        (b"GIF8", "GIF image"),
    )
    for signature, label in signatures:
        if body.startswith(signature):
            return label
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio file"
    if mime_type.startswith("video/"):
        return "video file"
    if mime_type.startswith("font/"):
        return "font file"
    if mime_type in {
        "application/octet-stream",
        "application/pdf",
        "application/zip",
        "application/x-zip-compressed",
    }:
        return "binary file"
    return None


class WebToolSuite:
    """Omnius-derived public search/fetch/crawl with session-local recall."""

    def __init__(
        self,
        *,
        ttl_s: float = 300.0,
        client: httpx.Client | None = None,
        resolver: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.ttl_s = max(1.0, ttl_s)
        self.client = client or httpx.Client(timeout=15.0, follow_redirects=False)
        self.resolver = resolver or _default_resolver
        self._lock = threading.Lock()
        self._sessions: dict[str, _WebSession] = {}

    def _expire_locked(self, now: float) -> None:
        for key in [
            key
            for key, value in self._sessions.items()
            if now - value.last_seen >= self.ttl_s
        ]:
            self._sessions.pop(key, None)

    def _session_locked(self, session_id: str, now: float) -> _WebSession:
        self._expire_locked(now)
        session = self._sessions.setdefault(_session_key(session_id), _WebSession())
        session.last_seen = now
        return session

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_session_key(session_id), None)

    def stats(self, session_id: str) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"indexed_pages": 0, "indexed_chars": 0}
            session.last_seen = now
            return {
                "indexed_pages": len(session.index),
                "indexed_chars": sum(len(item.content) for item in session.index.values()),
            }

    def _request(
        self, raw_url: str, *, max_bytes: int = MAX_FETCH_BYTES
    ) -> _FetchedPage:
        requested = _validate_public_url(raw_url, self.resolver)
        current = requested
        for redirect in range(MAX_REDIRECTS + 1):
            with self.client.stream(
                "GET",
                current,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RobitOmniPortal/1.0)",
                    "Accept": "text/html,application/xhtml+xml,text/plain,application/json,application/xml",
                },
            ) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolInputError("web redirect omitted its destination")
                    if redirect >= MAX_REDIRECTS:
                        raise ToolInputError("web request exceeded the redirect limit")
                    current = _validate_public_url(urljoin(current, location), self.resolver)
                    continue
                if response.status_code >= 400:
                    raise ToolInputError(f"web request returned HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").strip()
                mime_type = content_type.split(";", 1)[0].strip().lower()
                allowed = (
                    mime_type.startswith("text/")
                    or mime_type
                    in {
                        "application/json",
                        "application/ld+json",
                        "application/problem+json",
                        "application/xml",
                        "application/xhtml+xml",
                        "application/rss+xml",
                        "application/atom+xml",
                        "application/javascript",
                        "application/x-javascript",
                        "application/graphql",
                        "application/yaml",
                        "application/x-yaml",
                        "application/x-www-form-urlencoded",
                    }
                    or mime_type.endswith(("+json", "+xml"))
                    or not mime_type
                )
                if not allowed:
                    raise ToolInputError(f"unsupported web content type: {mime_type}")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise ToolInputError("web response exceeds its size limit")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise ToolInputError("web response exceeds its size limit")
                raw_body = bytes(body)
                binary_kind = _binary_payload_kind(raw_body, mime_type)
                if binary_kind:
                    raise ToolInputError(
                        f"web response is a {binary_kind}, not textual page content"
                    )
                try:
                    raw_body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ToolInputError("web response is not valid UTF-8 text") from exc
                return _FetchedPage(
                    receipt=_FetchReceipt(
                        requested_url=requested,
                        final_url=current,
                        status=response.status_code,
                        content_type=content_type or "unknown",
                        response_bytes=len(raw_body),
                        response_sha256=hashlib.sha256(raw_body).hexdigest(),
                        fetched_at=datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                    ),
                    mime_type=mime_type,
                    body=raw_body,
                )
        raise ToolInputError("web request exceeded the redirect limit")

    @staticmethod
    def _result_url(raw_url: str, base_url: str, search_host: str) -> str:
        candidate = html.unescape(raw_url).strip()
        if not candidate or candidate.startswith(("#", "javascript:", "mailto:")):
            return ""
        candidate = urljoin(base_url, candidate)
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme not in {"http", "https"} or not hostname:
            return ""
        if parsed.username or parsed.password:
            return ""
        provider_redirect = (
            hostname == search_host
            or hostname == "duckduckgo.com"
            or hostname.endswith(".duckduckgo.com")
        )
        if provider_redirect:
            query = parse_qs(parsed.query)
            redirected = next(
                (
                    values[0]
                    for key in ("uddg", "url", "u", "target")
                    if (values := query.get(key))
                ),
                "",
            )
            if not redirected:
                return ""
            candidate = html.unescape(redirected)
            parsed = urlsplit(candidate)
            hostname = (parsed.hostname or "").rstrip(".").lower()
            if parsed.scheme not in {"http", "https"} or not hostname:
                return ""
            if parsed.username or parsed.password:
                return ""
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            return ""
        if _is_ip(hostname) and not ipaddress.ip_address(hostname).is_global:
            return ""
        return candidate

    def _search_duckduckgo(
        self, query: str, limit: int
    ) -> list[dict[str, str]]:
        """Port of Omnius WebSearchTool's no-key DuckDuckGo HTML path."""

        search_url = f"{DUCKDUCKGO_HTML_URL}?q={quote_plus(query)}"
        page = self._request(search_url, max_bytes=MAX_SEARCH_BYTES)
        dom = page.body.decode("utf-8")
        collector = _AnchorCollector()
        collector.feed(dom)
        results: list[dict[str, str]] = []
        by_url: dict[str, dict[str, str]] = {}
        seen: set[str] = set()
        for raw_url, raw_title, attrs in collector.links:
            classes = attrs.get("class", "").split()
            if "result__snippet" in classes:
                snippet_url = self._result_url(
                    raw_url, search_url, "html.duckduckgo.com"
                )
                if snippet_url in by_url:
                    by_url[snippet_url]["snippet"] = re.sub(
                        r"\s+", " ", html.unescape(raw_title)
                    ).strip()[:1_200]
                continue
            if "result__a" not in classes:
                continue
            url = self._result_url(
                raw_url, search_url, "html.duckduckgo.com"
            )
            title = re.sub(r"\s+", " ", html.unescape(raw_title)).strip()
            if not url or len(title) < 3 or url in seen:
                continue
            seen.add(url)
            result = {
                "title": title[:300],
                "url": url,
                "snippet": "",
                "link_status": "unverified_search_result",
            }
            results.append(result)
            by_url[url] = result
            if len(results) >= limit:
                # Keep consuming only until the matching result's following
                # snippet has had a chance to appear.
                continue
        return results[:limit]

    def _index_entries(self, session_id: str, entries: Sequence[_WebIndexEntry]) -> None:
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            for entry in entries:
                session.index[entry.url] = entry
            while len(session.index) > MAX_WEB_INDEX_ENTRIES:
                oldest = min(session.index.values(), key=lambda item: item.indexed_at)
                session.index.pop(oldest.url, None)
            while sum(len(item.content) for item in session.index.values()) > MAX_WEB_INDEX_CHARS:
                with_content = [item for item in session.index.values() if item.content]
                if not with_content:
                    break
                oldest = min(with_content, key=lambda item: item.indexed_at)
                session.index[oldest.url] = _WebIndexEntry(
                    url=oldest.url,
                    title=oldest.title,
                    snippet=oldest.snippet,
                    content="",
                    indexed_at=oldest.indexed_at,
                )

    def search(
        self,
        session_id: str,
        query: Any,
        num_results: Any = None,
        mode: Any = None,
    ) -> dict[str, Any]:
        normalized_query = _bounded_text(query, "query", MAX_SEARCH_QUERY_CHARS)
        limit = _bounded_integer(num_results, default=5, minimum=1, maximum=MAX_SEARCH_RESULTS)
        normalized_mode = str(mode or "discover").strip().lower()
        if normalized_mode not in {"discover", "session"}:
            raise ToolInputError("mode must be discover or session")
        if normalized_mode == "discover":
            results = self._search_duckduckgo(normalized_query, limit)
            indexed_at = time.monotonic()
            self._index_entries(
                session_id,
                [
                    _WebIndexEntry(
                        url=item["url"],
                        title=item["title"],
                        snippet=item["snippet"],
                        content="",
                        indexed_at=indexed_at,
                    )
                    for item in results
                ],
            )
            return {
                "trust": "untrusted_web_results",
                "provider": "duckduckgo",
                "transport": "direct_html",
                "mode": "discover",
                "query": normalized_query,
                # Discovery metadata is not source evidence. Keep the concrete
                # source-reading routes in the next bounded model round so it
                # can inspect a selected result instead of issuing variations
                # of the same search merely to rediscover web_fetch.
                "alternative_tools": ["web_fetch", "browser_interact"],
                "provenance": {
                    "tool": "web_search",
                    "source_type": "search_result_metadata",
                    "evidence_type": "tool_data_not_visual_perception",
                    "authority": "discovery_only",
                    "citation_ready": False,
                },
                "results": results,
            }
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            indexed = list(session.index.values())
        ranked = [
            (
                _term_match_score(
                    normalized_query,
                    f"{entry.title} {entry.url} {entry.snippet} {entry.content}",
                ),
                entry,
            )
            for entry in indexed
        ]
        ranked = [item for item in ranked if item[0] > 0]
        ranked.sort(key=lambda item: (item[0], item[1].indexed_at), reverse=True)
        return {
            "trust": "untrusted_web_results",
            "provider": "session_local_index",
            "mode": "session",
            "query": normalized_query,
            "alternative_tools": ["web_fetch", "browser_interact"],
            "provenance": {
                "tool": "web_search",
                "source_type": "session_index_metadata",
                "evidence_type": "tool_data_not_visual_perception",
                "authority": "discovery_or_recall_only",
                "citation_ready": False,
            },
            "results": [
                {
                    "title": entry.title,
                    "url": entry.url,
                    "snippet": (entry.snippet or entry.content[:800]),
                    "relevance": round(score, 4),
                }
                for score, entry in ranked[:limit]
            ],
        }

    def fetch(
        self,
        session_id: str,
        url: Any,
        max_length: Any = None,
        response_format: Any = None,
    ) -> dict[str, Any]:
        normalized_url = _bounded_text(url, "url", 4_096)
        limit = _bounded_integer(max_length, default=6_000, minimum=500, maximum=MAX_FETCH_CHARS)
        normalized_format = str(response_format or "text").strip().lower()
        if normalized_format not in {"text", "raw_html"}:
            raise ToolInputError("format must be text or raw_html")
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            for cached_url, entry in list(session.fetch_cache.items()):
                if now - entry.saved_at >= FETCH_CACHE_TTL_S:
                    session.fetch_cache.pop(cached_url, None)
            cached = session.fetch_cache.get(normalized_url)
        if cached:
            entry = cached
            from_cache = True
        else:
            page = self._request(normalized_url)
            raw_text = page.body.decode("utf-8")
            is_html = page.mime_type in {"text/html", "application/xhtml+xml"} or (
                "<html" in raw_text[:500].lower()
            )
            title = ""
            if is_html:
                title_match = re.search(
                    r"<title[^>]*>([\s\S]*?)</title>", raw_text, re.IGNORECASE
                )
                title = _strip_html(title_match.group(1))[:300] if title_match else ""
                text = _strip_html(raw_text)
            else:
                text = raw_text
            entry = _WebFetchCacheEntry(
                saved_at=now,
                title=title,
                text=text,
                raw_html=raw_text,
                is_html=is_html,
                receipt=page.receipt,
            )
            with self._lock:
                session = self._session_locked(session_id, now)
                if len(session.fetch_cache) >= 16:
                    oldest_url = min(
                        session.fetch_cache,
                        key=lambda item: session.fetch_cache[item].saved_at,
                    )
                    session.fetch_cache.pop(oldest_url, None)
                session.fetch_cache[normalized_url] = entry
            from_cache = False
        content = (
            entry.raw_html
            if normalized_format == "raw_html" and entry.is_html
            else entry.text
        )
        format_used = (
            "raw_html"
            if normalized_format == "raw_html" and entry.is_html
            else "text"
        )
        self._index_entries(
            session_id,
            [
                _WebIndexEntry(
                    url=entry.receipt.final_url,
                    title=entry.title or entry.receipt.final_url,
                    snippet=entry.text[:800],
                    content=entry.text[:MAX_FETCH_CHARS],
                    indexed_at=now,
                )
            ],
        )
        return {
            "trust": "untrusted_web_content",
            "url": entry.receipt.final_url,
            "provenance": {
                "tool": "web_fetch",
                "source_type": "retrieved_public_page",
                "source_url": entry.receipt.final_url,
                "evidence_type": "tool_data_not_visual_perception",
                "authority": "page_content_only",
                "citation_ready": True,
            },
            "claim_limits": (
                "Attribute material claims to source_url; the page does not prove "
                "the user's location, current surroundings, or anything visually observed."
            ),
            "receipt": entry.receipt.export(),
            "cached": from_cache,
            "format": format_used,
            "content": content[:limit],
            "truncated": len(content) > limit,
            "rendering_hint": (
                "The extracted HTML text is very short; use browser_interact for "
                "JavaScript-rendered content."
                if format_used == "text" and entry.is_html and len(entry.text) < 200
                else ""
            ),
            "indexed_for_session_recall": True,
        }

    def crawl(
        self,
        session_id: str,
        url: Any,
        max_pages: Any = None,
        max_depth: Any = None,
        max_length: Any = None,
        extract: Any = None,
    ) -> dict[str, Any]:
        start_url = _validate_public_url(url, self.resolver)
        page_limit = _bounded_integer(max_pages, default=3, minimum=1, maximum=8)
        depth_limit = _bounded_integer(max_depth, default=1, minimum=0, maximum=2)
        char_limit = _bounded_integer(max_length, default=12_000, minimum=1_000, maximum=20_000)
        extract_mode = str(extract or "all").strip().lower()
        if extract_mode not in {"text", "links", "all"}:
            raise ToolInputError("extract must be text, links, or all")
        origin = (urlsplit(start_url).hostname or "").rstrip(".").lower()
        queue: list[tuple[str, int]] = [(start_url, 0)]
        queued = {start_url}
        pages: list[dict[str, Any]] = []
        indexed: list[_WebIndexEntry] = []
        used_chars = 0
        while queue and len(pages) < page_limit and used_chars < char_limit:
            current, depth = queue.pop(0)
            fetched = self._request(current)
            final_url = fetched.receipt.final_url
            final_origin = (urlsplit(final_url).hostname or "").rstrip(".").lower()
            if final_origin != origin:
                raise ToolInputError(
                    "web crawl redirected outside its same-origin boundary"
                )
            raw_text = fetched.body.decode("utf-8")
            is_html = fetched.mime_type in {
                "text/html",
                "application/xhtml+xml",
            } or "<html" in raw_text[:500].lower()
            title = ""
            collector = _AnchorCollector()
            if is_html:
                title_match = re.search(
                    r"<title[^>]*>([\s\S]*?)</title>", raw_text, re.IGNORECASE
                )
                title = _strip_html(title_match.group(1))[:300] if title_match else ""
                text = _strip_html(raw_text)
                collector.feed(raw_text)
            else:
                text = raw_text
            remaining = char_limit - used_chars
            excerpt = text[:remaining] if extract_mode in {"text", "all"} else ""
            used_chars += len(excerpt)
            public_links: list[dict[str, str]] = []
            for raw_link, raw_title, _attrs in collector.links:
                candidate = urljoin(final_url, html.unescape(raw_link).strip())
                parsed = urlsplit(candidate)
                hostname = (parsed.hostname or "").rstrip(".").lower()
                if (
                    parsed.scheme not in {"http", "https"}
                    or hostname != origin
                    or parsed.username
                    or parsed.password
                ):
                    continue
                candidate = parsed._replace(fragment="").geturl()
                public_links.append(
                    {
                        "url": candidate,
                        "text": re.sub(r"\s+", " ", raw_title).strip()[:300],
                    }
                )
            page_result: dict[str, Any] = {
                "url": final_url,
                "title": title or final_url,
                "depth": depth,
                "receipt": fetched.receipt.export(),
            }
            if extract_mode in {"text", "all"}:
                page_result.update(
                    {"content": excerpt, "truncated": len(text) > len(excerpt)}
                )
            if extract_mode in {"links", "all"}:
                page_result["links"] = public_links[:50]
            pages.append(page_result)
            indexed.append(_WebIndexEntry(url=final_url, title=title or final_url, snippet=text[:800], content=text[:MAX_FETCH_CHARS], indexed_at=time.monotonic()))
            if depth >= depth_limit or not is_html:
                continue
            for link in public_links:
                candidate = link["url"]
                if candidate in queued:
                    continue
                queued.add(candidate)
                queue.append((candidate, depth + 1))
        self._index_entries(session_id, indexed)
        return {
            "trust": "untrusted_web_content",
            "start_url": start_url,
            "provenance": {
                "tool": "web_crawl",
                "source_type": "retrieved_public_pages",
                "source_url": start_url,
                "evidence_type": "tool_data_not_visual_perception",
                "authority": "page_content_only",
                "citation_ready": True,
            },
            "same_origin": origin,
            "strategy": "direct_http",
            "extract": extract_mode,
            "pages": pages,
            "pages_fetched": len(pages),
            "characters": used_chars,
            "indexed_for_session_recall": True,
        }


@dataclass(frozen=True)
class _WorkspaceNote:
    note_id: str
    category: str
    content: str
    created_at: str


@dataclass
class _WorkspaceTask:
    task_id: str
    content: str
    status: str
    updated_at: str


@dataclass(frozen=True)
class _ObservedMedia:
    media_id: str
    kind: str
    mime_type: str
    bytes: int
    observed_at: str
    analysis: Mapping[str, Any]


@dataclass
class _WorkspaceSession:
    conversation: list[dict[str, str]] = field(default_factory=list)
    conversation_hashes: set[str] = field(default_factory=set)
    notes: dict[str, _WorkspaceNote] = field(default_factory=dict)
    tasks: dict[str, _WorkspaceTask] = field(default_factory=dict)
    media: dict[str, _ObservedMedia] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)


def _probe_media_bytes(raw: bytes, mime_type: str, kind: str) -> dict[str, Any]:
    if len(raw) > 96 * 1024 * 1024:
        raise ToolInputError("media exceeds the 96 MiB analysis limit")
    suffixes = {"audio/wav": ".wav", "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "video/mp4": ".mp4", "video/webm": ".webm", "image/gif": ".gif"}
    with tempfile.TemporaryDirectory(prefix="robit-omni-media-") as temp_dir:
        source = Path(temp_dir) / f"input{suffixes.get(mime_type, '.bin')}"
        source.write_bytes(raw)
        try:
            completed = subprocess.run([os.environ.get("FFPROBE_BIN", "ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)], check=False, capture_output=True, timeout=float(os.environ.get("OMNI_MEDIA_PROBE_TIMEOUT_S", "15")))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolInputError(f"media probe failed: {exc}") from exc
        if completed.returncode != 0:
            raise ToolInputError(f"media probe failed: {completed.stderr.decode('utf-8', errors='replace')[-500:]}")
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolInputError("media probe returned invalid JSON") from exc
        streams = []
        for stream in list(payload.get("streams") or [])[:16]:
            if isinstance(stream, Mapping):
                streams.append({key: stream[key] for key in ("index", "codec_name", "codec_long_name", "codec_type", "sample_rate", "channels", "channel_layout", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate", "duration", "nb_frames") if key in stream})
        format_data = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
        result: dict[str, Any] = {"kind": kind, "mime_type": mime_type, "format": {key: format_data[key] for key in ("format_name", "format_long_name", "duration", "size", "bit_rate") if key in format_data}, "streams": streams}
        if kind == "audio":
            try:
                volume = subprocess.run([os.environ.get("FFMPEG_BIN", "ffmpeg"), "-hide_banner", "-nostdin", "-i", str(source), "-af", "volumedetect", "-f", "null", "-"], check=False, capture_output=True, timeout=float(os.environ.get("OMNI_MEDIA_PROBE_TIMEOUT_S", "15")))
                diagnostic = volume.stderr.decode("utf-8", errors="replace")
                measurements = {label: match.group(1).strip()[:80] for label in ("mean_volume", "max_volume") if (match := re.search(rf"{label}:\s*([^\r\n]+)", diagnostic))}
                if measurements:
                    result["volume"] = measurements
            except (OSError, subprocess.TimeoutExpired):
                result["volume"] = {"available": False}
    return result


class SessionWorkspaceStore:
    """Session-only notes, tasks, conversation recall, and media observations."""

    def __init__(self, *, ttl_s: float = 300.0, media_runner: Callable[[bytes, str, str], Mapping[str, Any]] | None = None) -> None:
        self.ttl_s = max(1.0, ttl_s)
        self.media_runner = media_runner or _probe_media_bytes
        self._lock = threading.Lock()
        self._sessions: dict[str, _WorkspaceSession] = {}

    def _expire_locked(self, now: float) -> None:
        for key in [key for key, value in self._sessions.items() if now - value.last_seen >= self.ttl_s]:
            self._sessions.pop(key, None)

    def _session_locked(self, session_id: str, now: float) -> _WorkspaceSession:
        self._expire_locked(now)
        session = self._sessions.setdefault(_session_key(session_id), _WorkspaceSession())
        session.last_seen = now
        return session

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_session_key(session_id), None)

    def observe_conversation(self, session_id: str, messages: Sequence[Any]) -> None:
        additions = []
        for message in messages:
            if not isinstance(message, Mapping) or str(message.get("role") or "") not in {"user", "assistant"}:
                continue
            role = str(message.get("role"))
            content = str(message.get("content") or "").strip()[:4_000]
            if content:
                additions.append((hashlib.sha256(f"{role}\0{content}".encode()).hexdigest(), {"role": role, "content": content}))
        with self._lock:
            session = self._session_locked(session_id, time.monotonic())
            for fingerprint, item in additions:
                if fingerprint not in session.conversation_hashes:
                    session.conversation_hashes.add(fingerprint)
                    session.conversation.append(item)
            while len(session.conversation) > 64:
                removed = session.conversation.pop(0)
                session.conversation_hashes.discard(hashlib.sha256(f"{removed['role']}\0{removed['content']}".encode()).hexdigest())

    def observe_media(self, session_id: str, messages: Sequence[Any]) -> list[str]:
        observed = []
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            for media_field, kind in (("audios", "audio"), ("videos", "video")):
                values = message.get(media_field) or []
                if not isinstance(values, list):
                    continue
                for envelope in values[-4:]:
                    if not isinstance(envelope, Mapping) or not str(envelope.get("data") or "").strip():
                        continue
                    try:
                        raw = base64.b64decode(str(envelope.get("data")).strip(), validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ToolInputError("media observation contains invalid base64") from exc
                    mime_type = str(envelope.get("mime_type") or "application/octet-stream").lower()
                    media_id = hashlib.sha256(raw).hexdigest()[:16]
                    with self._lock:
                        session = self._session_locked(session_id, time.monotonic())
                        if media_id in session.media:
                            observed.append(media_id)
                            continue
                    try:
                        analysis = dict(self.media_runner(raw, mime_type, kind))
                    except ToolInputError as exc:
                        analysis = {"available": False, "error": str(exc)[:500]}
                    entry = _ObservedMedia(media_id=media_id, kind=kind, mime_type=mime_type, bytes=len(raw), observed_at=datetime.now().astimezone().isoformat(timespec="seconds"), analysis=analysis)
                    with self._lock:
                        session = self._session_locked(session_id, time.monotonic())
                        session.media[media_id] = entry
                        while len(session.media) > 16:
                            session.media.pop(next(iter(session.media)), None)
                    observed.append(media_id)
        return observed

    def _media_list(self, session_id: str) -> list[_ObservedMedia]:
        with self._lock:
            session = self._sessions.get(_session_key(session_id))
            return list(session.media.values()) if session else []

    def media(self, session_id: str, kind: str, media_id: Any = None) -> dict[str, Any]:
        requested = str(media_id or "").strip()
        with self._lock:
            session = self._session_locked(session_id, time.monotonic())
            candidates = [item for item in session.media.values() if item.kind == kind and (not requested or item.media_id == requested)]
        if not candidates:
            return {"found": False, "kind": kind, "available": [item.media_id for item in self._media_list(session_id) if item.kind == kind]}
        item = candidates[-1]
        return {"found": True, "media_id": item.media_id, "kind": item.kind, "mime_type": item.mime_type, "bytes": item.bytes, "observed_at": item.observed_at, "analysis": dict(item.analysis)}

    def notes(self, session_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip().lower()
        with self._lock:
            session = self._session_locked(session_id, time.monotonic())
            if action == "add":
                content = _bounded_text(arguments.get("content"), "content", 4_096)
                category = str(arguments.get("category") or "finding").strip()[:64] or "finding"
                note_id = hashlib.sha256(f"{time.time_ns()}\0{category}\0{content}".encode()).hexdigest()[:12]
                if len(session.notes) >= 100:
                    raise ToolInputError("working notes already contain 100 entries")
                session.notes[note_id] = _WorkspaceNote(note_id, category, content, datetime.now().astimezone().isoformat(timespec="seconds"))
                return {"added": True, "note_id": note_id, "scope": "browser_session"}
            if action == "remove":
                return {"removed": session.notes.pop(_bounded_text(arguments.get("note_id"), "note_id", 64), None) is not None}
            if action == "clear":
                removed = len(session.notes)
                session.notes.clear()
                return {"cleared": removed}
            notes = list(session.notes.values())
        if action == "search":
            query = _bounded_text(arguments.get("content"), "content", 500)
            notes = [item for item in notes if _term_match_score(query, f"{item.category} {item.content}") > 0]
        elif action != "list":
            raise ToolInputError("working_notes action must be add, list, search, remove, or clear")
        return {"scope": "browser_session", "notes": [item.__dict__ for item in notes[-100:]]}

    def task_list(self, session_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip().lower()
        with self._lock:
            session = self._session_locked(session_id, time.monotonic())
            if action == "upsert":
                content = _bounded_text(arguments.get("content"), "content", 1_000)
                status = str(arguments.get("status") or "pending").strip().lower()
                if status not in {"pending", "in_progress", "completed", "blocked"}:
                    raise ToolInputError("task status is invalid")
                task_id = str(arguments.get("task_id") or "").strip()[:64] or hashlib.sha256(f"{time.time_ns()}\0{content}".encode()).hexdigest()[:12]
                if task_id not in session.tasks and len(session.tasks) >= 100:
                    raise ToolInputError("task list already contains 100 entries")
                session.tasks[task_id] = _WorkspaceTask(task_id, content, status, datetime.now().astimezone().isoformat(timespec="seconds"))
                return {"upserted": True, **session.tasks[task_id].__dict__}
            if action == "remove":
                return {"removed": session.tasks.pop(_bounded_text(arguments.get("task_id"), "task_id", 64), None) is not None}
            if action == "clear":
                removed = len(session.tasks)
                session.tasks.clear()
                return {"cleared": removed}
            if action != "list":
                raise ToolInputError("task_list action must be upsert, list, remove, or clear")
            tasks = [item.__dict__ for item in session.tasks.values()]
        return {"scope": "browser_session", "tasks": tasks}

    def search(self, session_id: str, query: str, max_results: int) -> list[dict[str, Any]]:
        with self._lock:
            session = self._session_locked(session_id, time.monotonic())
            candidates = [("conversation", item["role"], item["content"]) for item in session.conversation]
            candidates.extend(("working_note", item.category, item.content) for item in session.notes.values())
            candidates.extend(("task", item.status, item.content) for item in session.tasks.values())
        ranked = [(_term_match_score(query, f"{label} {content}"), source, label, content) for source, label, content in candidates]
        ranked = sorted((item for item in ranked if item[0] > 0), key=lambda item: item[0], reverse=True)
        return [{"source": source, "label": label, "content": content, "relevance": round(score, 4)} for score, source, label, content in ranked[:max_results]]

    def stats(self, session_id: str) -> dict[str, int]:
        with self._lock:
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"notes": 0, "tasks": 0, "conversation_turns": 0, "media": 0}
            session.last_seen = time.monotonic()
            return {"notes": len(session.notes), "tasks": len(session.tasks), "conversation_turns": len(session.conversation), "media": len(session.media)}


@dataclass
class _SessionLocation:
    value: dict[str, Any]
    last_seen: float = field(default_factory=time.monotonic)


class SessionLocationStore:
    """Sanitized browser-reported IP geolocation, isolated by session cookie."""

    def __init__(self, *, ttl_s: float = 300.0) -> None:
        self.ttl_s = max(1.0, ttl_s)
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionLocation] = {}

    def _expire_locked(self, now: float) -> None:
        for key in [
            key
            for key, value in self._sessions.items()
            if now - value.last_seen >= self.ttl_s
        ]:
            self._sessions.pop(key, None)

    @staticmethod
    def _text(value: Any, maximum: int) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:maximum]

    @staticmethod
    def _coordinate(value: Any, *, minimum: float, maximum: float) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(coordinate) or not minimum <= coordinate <= maximum:
            return None
        return round(coordinate, 2)

    def set(self, session_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ToolInputError("portal_client_location must be an object")
        timezone = payload.get("timezone")
        timezone = timezone if isinstance(timezone, Mapping) else {}
        value: dict[str, Any] = {
            "available": True,
            "source": "browser_ip_geolocation",
            "precision": "ip_approximate",
            "scope": "browser_session",
            "raw_ip_included": False,
            "provenance": {
                "tool": "get_user_location",
                "source_type": "browser_ip_geolocation",
                "evidence_type": "tool_data_not_visual_perception",
                "authority": "approximate_network_area",
                "device_gps": False,
                "street_level": False,
            },
            "claim_limits": {
                "supported": "approximate city, region, country, timezone, and nearby search seed",
                "unsupported": "exact address, current street, device position, or visible surroundings",
            },
            "city": self._text(payload.get("city"), 120),
            "region": self._text(payload.get("region"), 120),
            "region_code": self._text(payload.get("region_code"), 16).upper(),
            "country": self._text(payload.get("country"), 120),
            "country_code": self._text(payload.get("country_code"), 8).upper(),
            "continent": self._text(payload.get("continent"), 120),
            "continent_code": self._text(payload.get("continent_code"), 8).upper(),
            "latitude": self._coordinate(payload.get("latitude"), minimum=-90.0, maximum=90.0),
            "longitude": self._coordinate(payload.get("longitude"), minimum=-180.0, maximum=180.0),
            "timezone": {
                "id": self._text(timezone.get("id"), 120),
                "abbreviation": self._text(timezone.get("abbreviation"), 24),
                "utc_offset": self._text(timezone.get("utc_offset"), 16),
            },
            "caveat": "IP geolocation is approximate and may reflect a VPN or carrier gateway.",
        }
        if not (
            value["city"]
            or value["region"]
            or value["country"]
            or value["latitude"] is not None
            or value["longitude"] is not None
        ):
            raise ToolInputError("portal_client_location contains no usable location")
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            self._sessions[_session_key(session_id)] = _SessionLocation(value=value, last_seen=now)
        return dict(value)

    def get(self, session_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {
                    "available": False,
                    "scope": "browser_session",
                    "raw_ip_included": False,
                    "reason": "The browser has not supplied approximate location data.",
                    "next_action": context_text(
                        "directives", "location_unavailable_next_action"
                    ),
                }
            session.last_seen = now
            return dict(session.value)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_session_key(session_id), None)

    def stats(self, session_id: str) -> dict[str, bool]:
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"available": False}
            session.last_seen = now
            return {"available": True}


@dataclass(frozen=True)
class _SubagentRecord:
    task_id: str
    objective: str
    role: str
    created_at: str
    result: dict[str, Any]


@dataclass
class _SubagentSession:
    last_seen: float
    tasks: dict[str, _SubagentRecord] = field(default_factory=dict)


class SessionSubagentStore:
    """Run and retain isolated one-shot helper completions for one browser session."""

    def __init__(
        self,
        *,
        ttl_s: float,
        runner: Callable[[str, str, str], Mapping[str, Any]] | None,
    ) -> None:
        self.ttl_s = max(1.0, float(ttl_s))
        self.runner = runner
        self._lock = threading.RLock()
        self._sessions: dict[str, _SubagentSession] = {}

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            key
            for key, value in self._sessions.items()
            if now - value.last_seen >= self.ttl_s
        ]
        for key in expired:
            self._sessions.pop(key, None)

    def _session_locked(self, session_id: str, now: float) -> _SubagentSession:
        self._cleanup_locked(now)
        key = _session_key(session_id)
        session = self._sessions.get(key)
        if session is None:
            session = _SubagentSession(last_seen=now)
            self._sessions[key] = session
        session.last_seen = now
        return session

    @staticmethod
    def _export(record: _SubagentRecord, *, include_result: bool) -> dict[str, Any]:
        value: dict[str, Any] = {
            "task_id": record.task_id,
            "status": "completed",
            "role": record.role,
            "objective": record.objective,
            "created_at": record.created_at,
            "scope": "browser_session",
        }
        if include_result:
            value["result"] = copy_mapping(record.result)
        return value

    def delegate(self, session_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.runner is None:
            raise ToolInputError("sub-agent runner is unavailable in this deployment")
        objective = _bounded_text(arguments.get("objective"), "objective", 12_000)
        role = str(arguments.get("role") or "general").strip().lower()
        if role not in {"general", "researcher", "planner", "critic"}:
            raise ToolInputError("sub-agent role is invalid")
        context = str(arguments.get("context") or "").strip()
        context_source = str(arguments.get("context_source") or "inline").strip()
        if context_source not in {
            "inline",
            "none",
            "current_user_message",
            "latest_non_discovery_tool_result",
        }:
            raise ToolInputError("sub-agent context_source is invalid")
        if context_source in {
            "current_user_message",
            "latest_non_discovery_tool_result",
        } and not context:
            raise ToolInputError(
                "referenced sub-agent context is available only inside an automatic chat tool loop"
            )
        if len(context) > 24_000:
            raise ToolInputError("sub-agent context exceeds 24000 characters")

        raw_result = self.runner(objective, role, context)
        if not isinstance(raw_result, Mapping):
            raise ToolInputError("sub-agent runner returned an invalid result")
        result = copy_mapping(raw_result)
        task_id = hashlib.sha256(
            f"{_session_key(session_id)}\0{time.time_ns()}\0{role}\0{objective}".encode()
        ).hexdigest()[:16]
        record = _SubagentRecord(
            task_id=task_id,
            objective=objective,
            role=role,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            result=result,
        )
        now = time.monotonic()
        with self._lock:
            self._session_locked(session_id, now).tasks[task_id] = record
        return self._export(record, include_result=True)

    def list(self, session_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            tasks = [
                self._export(record, include_result=False)
                for record in session.tasks.values()
            ]
        return {"scope": "browser_session", "tasks": tasks}

    def result(self, session_id: str, task_id: Any) -> dict[str, Any]:
        normalized = _bounded_text(task_id, "task_id", 80)
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            record = session.tasks.get(normalized)
            if record is None:
                raise ToolInputError("sub-agent task was not found in this browser session")
            return self._export(record, include_result=True)

    def forget(self, session_id: str, task_id: Any) -> dict[str, Any]:
        normalized = _bounded_text(task_id, "task_id", 80)
        now = time.monotonic()
        with self._lock:
            session = self._session_locked(session_id, now)
            removed = session.tasks.pop(normalized, None) is not None
        return {"task_id": normalized, "forgotten": removed, "scope": "browser_session"}

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_session_key(session_id), None)

    def stats(self, session_id: str) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            session = self._sessions.get(_session_key(session_id))
            if session is None:
                return {"tasks": 0}
            session.last_seen = now
            return {"tasks": len(session.tasks)}


def copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe defensive copy for helper results exposed back to the model."""

    try:
        copied = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError) as exc:
        raise ToolInputError("sub-agent result could not be serialized") from exc
    if not isinstance(copied, dict):
        raise ToolInputError("sub-agent result must be an object")
    return copied


class PortalToolHarness:
    """Execute exactly the schemas in ``SAFE_TOOLS`` for one portal session."""

    def __init__(
        self,
        documents: SessionDocumentStore,
        *,
        ttl_s: float = 300.0,
        web_client: httpx.Client | None = None,
        resolver: Callable[[str], Sequence[str]] | None = None,
        media_runner: Callable[[bytes, str, str], Mapping[str, Any]] | None = None,
        subagent_runner: Callable[[str, str, str], Mapping[str, Any]] | None = None,
        background_tasks: BackgroundTaskStore | None = None,
        browser_automation: Any | None = None,
        gui_automation: Any | None = None,
        memory_governor: MemoryGovernor | None = None,
    ) -> None:
        self.documents = documents
        self.memory = SessionMemoryStore(ttl_s=ttl_s)
        self.web = WebToolSuite(
            ttl_s=ttl_s,
            client=web_client,
            resolver=resolver,
        )
        self.workspace = SessionWorkspaceStore(ttl_s=ttl_s, media_runner=media_runner)
        self.subagents = SessionSubagentStore(ttl_s=ttl_s, runner=subagent_runner)
        self.location = SessionLocationStore(ttl_s=ttl_s)
        self.background_tasks = background_tasks
        self.memory_governor = memory_governor
        self.browser = browser_automation or BrowserAutomationStore(
            ttl_s=max(900.0, ttl_s),
            memory_governor=memory_governor,
            launch_reserve_gib=_TOOL_MEMORY_RESERVE_GIB.get("browser_interact"),
        )
        self.gui = gui_automation or GuiAutomation()

    def clear(self, session_id: str) -> None:
        self.memory.clear(session_id)
        self.web.clear(session_id)
        self.workspace.clear(session_id)
        self.subagents.clear(session_id)
        self.location.clear(session_id)
        self.browser.clear(session_id)
        self.gui.clear(session_id)

    def memory_stats(self, session_id: str) -> dict[str, int]:
        return self.memory.stats(session_id)

    def web_stats(self, session_id: str) -> dict[str, int]:
        return self.web.stats(session_id)

    def workspace_stats(self, session_id: str) -> dict[str, int]:
        return self.workspace.stats(session_id)

    def location_stats(self, session_id: str) -> dict[str, bool]:
        return self.location.stats(session_id)

    def subagent_stats(self, session_id: str) -> dict[str, int]:
        return self.subagents.stats(session_id)

    def set_client_location(self, session_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.location.set(session_id, payload)

    def client_location(self, session_id: str) -> dict[str, Any]:
        """Return only the TTL-bounded, sanitized location for this session."""

        return self.location.get(session_id)

    def observe_request(self, session_id: str, payload: Mapping[str, Any]) -> list[str]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return []
        self.workspace.observe_conversation(session_id, messages)
        return self.workspace.observe_media(session_id, messages)

    def execute(
        self,
        session_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            # Resource admission is declarative tool metadata, not a growing
            # list of tool-name special cases. "standard" work may establish
            # new residency and must clear the soft floor; tightly "bounded"
            # work may run inside the soft-to-hard safety band; an "executor"
            # owns dynamic admission; and "control" remains available so work
            # can be inspected or cancelled under pressure.
            memory_admission = _TOOL_MEMORY_ADMISSION.get(name, "standard")
            task_control = memory_admission == "control"
            # Teardown must remain callable at the memory floor: closing the
            # rendered browser is itself a pressure-relief operation.  Treat
            # it like task cancellation rather than replacing its successful
            # receipt with a retryable resource-pressure error afterwards.
            release_only = (
                name == "browser_interact"
                and str(arguments.get("action") or "").strip().lower() == "close"
            )
            if self.memory_governor is not None:
                if memory_admission == "standard" and not release_only:
                    self.memory_governor.require(f"tool {name}")
                elif memory_admission == "bounded" and not release_only:
                    self.memory_governor.require_hard_floor(f"tool {name}")
            if name == "get_current_time":
                now = datetime.now().astimezone()
                result: dict[str, Any] = {
                    "date": now.date().isoformat(),
                    "time": now.isoformat(timespec="seconds"),
                    "utc_offset": now.strftime("%z"),
                    "timezone": str(now.tzinfo),
                }
            elif name == "get_system_snapshot":
                result = runtime_environment_snapshot()
            elif name == "get_user_location":
                result = self.location.get(session_id)
            elif name == "get_portal_capabilities":
                result = {
                    "input": ["text", "microphone", "wav", "image", "video", "gif", "pdf", "docx", "utf-8 text/code"],
                    "output": ["text", "thinking", "tool_calls", "audio/wav"],
                    "tasks": ["chat", "transcribe", "describe", "synthesize"],
                    "safe_tools": [item["function"]["name"] for item in SAFE_TOOLS],
                    "memory_scope": "browser_session",
                    "web_access": "duckduckgo_html_discovery_fetch_crawl_and_session_index",
                }
            elif name == "request_camera_view":
                mode = str(arguments.get("mode") or "still").strip()
                if mode not in {"still", "motion"}:
                    raise ToolInputError("mode must be still or motion")
                result = {
                    "camera_capture_requested": True,
                    "mode": mode,
                    "next_action": context_text(
                        "directives", "camera_next_action"
                    ),
                }
            elif name == "tool_search":
                family = _bounded_text(arguments.get("family"), "family", 64)
                if family == "uncertain":
                    names: list[str] = []
                else:
                    names = discover_tool_names(family)
                    if not names:
                        raise ToolInputError(f"unknown tool family: {family}")
                result = {
                    "family": family,
                    "allowlisted_only": True,
                    "suggested_tools": names,
                    "available_tools": names,
                    "next_action": (
                        context_text("directives", "tool_family_uncertain_next_action")
                        if family == "uncertain"
                        else context_text("directives", "tool_search_next_action")
                    ),
                    "results": [{"name": discovered} for discovered in names],
                }
                if family == "uncertain":
                    result["available_families"] = [
                        {
                            "name": family_name,
                            "description": str(definition.get("description") or ""),
                        }
                        for family_name, definition in TOOL_FAMILIES.items()
                    ]
            elif name == "safe_math_eval":
                result = _safe_math_eval(arguments.get("expression"))
            elif name == "web_search":
                result = self.web.search(
                    session_id,
                    arguments.get("query"),
                    arguments.get("num_results"),
                    arguments.get("mode"),
                )
            elif name == "web_fetch":
                result = self.web.fetch(
                    session_id,
                    arguments.get("url"),
                    arguments.get("max_length"),
                    arguments.get("format"),
                )
            elif name == "browser_interact":
                result = self.browser.act(session_id, dict(arguments))
            elif name == "gui_interact":
                result = self.gui.act(session_id, dict(arguments))
            elif name == "web_crawl":
                result = self.web.crawl(
                    session_id,
                    arguments.get("url"),
                    arguments.get("max_pages"),
                    arguments.get("max_depth"),
                    arguments.get("max_length"),
                    arguments.get("extract"),
                )
            elif name == "document_search":
                result = {
                    "trust": "untrusted_document_content",
                    "query": _bounded_text(arguments.get("query"), "query", 500),
                    "results": self.documents.search(
                        session_id,
                        arguments.get("query"),
                        max_results=_bounded_integer(
                            arguments.get("max_results"), default=5, minimum=1, maximum=8
                        ),
                    ),
                }
            elif name == "structured_read":
                result = self.documents.structured_read(session_id, arguments.get("document_id"), arguments.get("path"), _bounded_integer(arguments.get("max_rows"), default=50, minimum=1, maximum=200))
            elif name == "ocr_pdf":
                result = self.documents.ocr_pdf(session_id, arguments.get("document_id"), arguments.get("language"), _bounded_integer(arguments.get("max_pages"), default=20, minimum=1, maximum=50), arguments.get("force") is True)
            elif name == "memory_write":
                result = self.memory.write(
                    session_id,
                    arguments.get("topic"),
                    arguments.get("key"),
                    arguments.get("value"),
                )
            elif name == "memory_read":
                result = self.memory.read(session_id, arguments.get("topic"), arguments.get("key"))
            elif name == "memory_search":
                result = self.memory.search(session_id, arguments.get("query"), arguments.get("max_results"))
            elif name == "working_notes":
                result = self.workspace.notes(session_id, arguments)
            elif name == "task_list":
                result = self.workspace.task_list(session_id, arguments)
            elif name == "workspace_file":
                result = _workspace_file(
                    arguments.get("action"),
                    arguments.get("path"),
                    content=arguments.get("content"),
                    old_text=arguments.get("old_text"),
                    new_text=arguments.get("new_text"),
                    expected_occurrences=arguments.get("expected_occurrences"),
                    expected_sha256=arguments.get("expected_sha256"),
                    max_chars=arguments.get("max_chars"),
                    offset_chars=arguments.get("offset_chars"),
                    depth=arguments.get("depth"),
                )
            elif name == "shell":
                result = _run_shell(
                    arguments.get("command"),
                    arguments.get("cwd"),
                    arguments.get("timeout_seconds"),
                    arguments.get("stdin"),
                    self.memory_governor,
                    intent=arguments.get("intent"),
                    mutation_paths=arguments.get("mutation_paths"),
                )
            elif name == "background_task":
                if self.background_tasks is None:
                    raise ToolInputError("background task worker is not configured")
                action = str(arguments.get("action") or "").strip()
                if action == "start":
                    active = [
                        task
                        for task in self.background_tasks.list()
                        if task.get("status") not in TERMINAL_STATUSES
                    ]
                    if active and arguments.get("independent") is not True:
                        result = {
                            "accepted": False,
                            "error": "unfinished_task_requires_relationship",
                            "active_tasks": [
                                {
                                    "task_id": task.get("task_id"),
                                    "status": task.get("status"),
                                    "objective": task.get("objective"),
                                    "current_stage": task.get("current_stage"),
                                }
                                for task in active[:8]
                            ],
                            "next_action": context_text(
                                "directives", "background_relationship_next_action"
                            ),
                        }
                        return result
                    completion_criteria = str(
                        arguments.get("completion_criteria") or ""
                    ).strip()
                    if len(completion_criteria) > 2000:
                        raise ToolInputError(
                            "completion_criteria exceeds 2000 characters"
                        )
                    result = self.background_tasks.create(
                        _bounded_text(arguments.get("objective"), "objective", 6000),
                        completion_criteria,
                    )
                    result["accepted"] = True
                    result["next_action"] = context_text(
                        "directives", "background_started_next_action"
                    )
                elif action == "update":
                    task_id = _bounded_text(arguments.get("task_id"), "task_id", 80)
                    guidance = _bounded_text(
                        arguments.get("guidance"), "guidance", 4000
                    )
                    task = self.background_tasks.add_guidance(task_id, guidance)
                    result = {
                        "found": task is not None,
                        "accepted": bool(
                            task is not None
                            and task.get("status")
                            not in {"completed", "blocked", "cancelled"}
                        ),
                        "task": task,
                    }
                elif action == "list":
                    result = {"tasks": self.background_tasks.list()}
                elif action == "status":
                    task_id = _bounded_text(arguments.get("task_id"), "task_id", 80)
                    task = self.background_tasks.get(task_id)
                    result = {"found": task is not None, "task": task}
                elif action == "cancel":
                    task_id = _bounded_text(arguments.get("task_id"), "task_id", 80)
                    task = self.background_tasks.cancel(task_id)
                    result = {"found": task is not None, "task": task}
                else:
                    raise ToolInputError(
                        "background_task action must be start, update, status, list, or cancel"
                    )
            elif name == "audio_analyze":
                result = self.workspace.media(session_id, "audio", arguments.get("media_id"))
            elif name == "video_scan":
                result = self.workspace.media(session_id, "video", arguments.get("media_id"))
            elif name == "subagent_delegate":
                result = self.subagents.delegate(session_id, arguments)
            elif name == "subagent_list":
                result = self.subagents.list(session_id)
            elif name == "subagent_result":
                result = self.subagents.result(session_id, arguments.get("task_id"))
            elif name == "subagent_forget":
                result = self.subagents.forget(session_id, arguments.get("task_id"))
            elif name == "session_search":
                query = _bounded_text(arguments.get("query"), "query", 500)
                limit = _bounded_integer(arguments.get("max_results"), default=10, minimum=1, maximum=20)
                buckets = [
                    self.workspace.search(session_id, query, limit),
                    [{"source": "memory", **item} for item in self.memory.search(session_id, query, limit)["results"]],
                    [{"source": "document", **item} for item in self.documents.search(session_id, query, max_results=limit)],
                    [{"source": "web", **item} for item in self.web.search(session_id, query, limit, "session")["results"]],
                ]
                combined: list[dict[str, Any]] = []
                for index in range(limit):
                    added = False
                    for bucket in buckets:
                        if index < len(bucket):
                            combined.append(bucket[index])
                            added = True
                            if len(combined) >= limit:
                                break
                    if len(combined) >= limit or not added:
                        break
                result = {"query": query, "scope": "browser_session", "results": combined[:limit]}
            else:
                result = {
                    "error": "tool_not_allowed",
                    "allowed": [item["function"]["name"] for item in SAFE_TOOLS],
                }
            if (
                self.memory_governor is not None
                and not task_control
                and not release_only
                and self.memory_governor.under_hard_pressure()
            ):
                raise MemoryPressure(
                    f"tool {name}",
                    self.memory_governor.available_gib(),
                    self.memory_governor.policy.hard_floor_gib,
                )
        except MemoryPressure:
            result = {
                "error": "resource_pressure",
                "retryable": True,
                "message": "The runtime deferred this operation to preserve memory headroom.",
            }
        except BrowserDesktopUnavailable as exc:
            result = {
                "error": type(exc).__name__,
                "message": str(exc)[:500],
                "failure_scope": "capability",
                "task_blocked": False,
                "disposition": "change_capability",
                "alternative_tools": ["gui_interact"],
            }
        except ToolInputError as exc:
            result = {"error": type(exc).__name__, "message": str(exc)[:500]}
            if name == "web_fetch":
                # A guessed, stale, blocked, or non-renderable URL does not
                # prove that web research is blocked. Route back to discovery
                # (or the rendered browser) instead of letting an agent vary
                # hostnames inside the same failed fetch capability.
                rendered_first = bool(
                    re.search(r"\bHTTP\s+(?:401|403|429)\b", str(exc), re.IGNORECASE)
                )
                result.update(
                    {
                        "failure_scope": "arguments",
                        "task_blocked": False,
                        "disposition": "change_capability",
                        "alternative_tools": (
                            ["browser_interact", "web_search"]
                            if rendered_first
                            else ["web_search", "browser_interact"]
                        ),
                    }
                )
        except (
            BrowserAutomationError,
            GuiAutomationError,
            DocumentError,
            httpx.HTTPError,
        ) as exc:
            result = {"error": type(exc).__name__, "message": str(exc)[:500]}
        return result


def tool_result_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tool_use_instructions() -> str:
    """Trusted, compact procedure injected only when the user enables tools."""

    return context_text("directives", "tool_use")
