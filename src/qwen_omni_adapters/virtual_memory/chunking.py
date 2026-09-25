"""Structure-aware source chunking with exact source offsets.

Chunks are views over immutable source text.  Generated summaries never enter
this layer, and every chunk can therefore be reconstructed byte-for-byte from
its document plus offsets.
"""

from __future__ import annotations

import ast
import bisect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\S+")
MARKDOWN_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$")
CONVERSATION_TURN_RE = re.compile(
    r"(?mi)^(user|assistant|system|tool|human|agent)\s*:\s*"
)
LOG_EVENT_RE = re.compile(
    r"(?m)^(?=(?:\d{4}-\d{2}-\d{2}[T ][^\n ]+|"
    r"\[[A-Z][A-Z0-9_-]{1,15}\]|(?:ERROR|WARN|INFO|DEBUG|TRACE)\b))"
)
SYMBOL_RE = re.compile(
    r"(?m)^\s*(?:async\s+)?(?:def|class|function|interface|type|struct|enum)\s+"
    r"([A-Za-z_$][\w$]*)"
)


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    byte_start: int
    byte_end: int
    parent_kind: str | None = None
    parent_name: str | None = None
    symbols: tuple[tuple[str, str], ...] = ()
    code_edges: tuple[tuple[str, str, str], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class _Offsets:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_starts = [match.start() for match in TOKEN_RE.finditer(text)]
        self.byte_prefix = [0]
        total = 0
        for character in text:
            total += len(character.encode("utf-8"))
            self.byte_prefix.append(total)

    def draft(
        self,
        start: int,
        end: int,
        *,
        parent_kind: str | None = None,
        parent_name: str | None = None,
        symbols: tuple[tuple[str, str], ...] = (),
        code_edges: tuple[tuple[str, str, str], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> ChunkDraft | None:
        while start < end and self.text[start].isspace():
            start += 1
        while end > start and self.text[end - 1].isspace():
            end -= 1
        if start >= end:
            return None
        return ChunkDraft(
            text=self.text[start:end],
            char_start=start,
            char_end=end,
            token_start=bisect.bisect_left(self.token_starts, start),
            token_end=bisect.bisect_left(self.token_starts, end),
            byte_start=self.byte_prefix[start],
            byte_end=self.byte_prefix[end],
            parent_kind=parent_kind,
            parent_name=parent_name,
            symbols=symbols,
            code_edges=code_edges,
            metadata=dict(metadata or {}),
        )


class StructureAwareChunker:
    """Prefer source structure, with a bounded overlapping fallback."""

    def __init__(
        self,
        *,
        target_tokens: int = 1024,
        max_tokens: int = 2048,
        overlap_tokens: int = 128,
    ) -> None:
        if not 64 <= target_tokens <= max_tokens:
            raise ValueError("target_tokens must be between 64 and max_tokens")
        if not 0 <= overlap_tokens < target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        text: str,
        *,
        source: str = "",
        media_type: str | None = None,
        kind: str | None = None,
    ) -> list[ChunkDraft]:
        if not text:
            return []
        selected_kind = kind or self._kind(source, media_type)
        if selected_kind == "python":
            chunks = self._python(text)
        elif selected_kind == "code":
            chunks = self._sections(text, SYMBOL_RE, "symbol_block")
        elif selected_kind == "markdown":
            chunks = self._sections(text, MARKDOWN_HEADING_RE, "section")
        elif selected_kind == "conversation":
            chunks = self._sections(text, CONVERSATION_TURN_RE, "turn")
        elif selected_kind == "json":
            chunks = self._json(text)
        elif selected_kind == "log":
            chunks = self._sections(text, LOG_EVENT_RE, "event")
        else:
            chunks = []
        return chunks or self._fallback(text)

    @staticmethod
    def _kind(source: str, media_type: str | None) -> str:
        suffix = Path(source).suffix.lower()
        mime = str(media_type or "").lower()
        if suffix == ".py" or mime == "text/x-python":
            return "python"
        if suffix in {
            ".c",
            ".cc",
            ".cpp",
            ".cs",
            ".go",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".jsx",
            ".mjs",
            ".rs",
            ".ts",
            ".tsx",
        }:
            return "code"
        if suffix in {".md", ".markdown", ".rst"} or "markdown" in mime:
            return "markdown"
        if suffix in {".json", ".jsonl"} or mime == "application/json":
            return "json"
        if suffix in {".log", ".out"}:
            return "log"
        return "text"

    def _fallback_range(
        self,
        text: str,
        offsets: _Offsets,
        start: int,
        end: int,
        *,
        parent_kind: str | None = None,
        parent_name: str | None = None,
        symbols: tuple[tuple[str, str], ...] = (),
        code_edges: tuple[tuple[str, str, str], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkDraft]:
        token_matches = list(TOKEN_RE.finditer(text, start, end))
        token_starts = [match.start() for match in token_matches]
        if len(token_matches) <= self.max_tokens:
            item = offsets.draft(
                start,
                end,
                parent_kind=parent_kind,
                parent_name=parent_name,
                symbols=symbols,
                code_edges=code_edges,
                metadata=metadata,
            )
            return [item] if item is not None else []
        result: list[ChunkDraft] = []
        index = 0
        while index < len(token_matches):
            stop = min(len(token_matches), index + self.target_tokens)
            char_start = token_matches[index].start()
            char_end = token_matches[stop - 1].end()
            if stop < len(token_matches):
                paragraph = text.rfind("\n\n", char_start, char_end)
                if paragraph > char_start + (char_end - char_start) // 2:
                    char_end = paragraph
                    stop = bisect.bisect_left(token_starts, paragraph)
            item = offsets.draft(
                char_start,
                char_end,
                parent_kind=parent_kind,
                parent_name=parent_name,
                symbols=symbols,
                code_edges=code_edges,
                metadata=metadata,
            )
            if item is not None:
                result.append(item)
            if stop >= len(token_matches):
                break
            index = max(index + 1, stop - self.overlap_tokens)
        return result

    def _fallback(self, text: str) -> list[ChunkDraft]:
        return self._fallback_range(text, _Offsets(text), 0, len(text))

    def _sections(
        self,
        text: str,
        pattern: re.Pattern[str],
        parent_kind: str,
    ) -> list[ChunkDraft]:
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        offsets = _Offsets(text)
        starts = [0] if matches[0].start() else []
        starts.extend(match.start() for match in matches)
        starts = sorted(set(starts))
        result: list[ChunkDraft] = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            header = next((match for match in matches if match.start() == start), None)
            name = header.group(header.lastindex or 0).strip()[:240] if header else None
            symbols = tuple(
                (match.group(1), "symbol")
                for match in SYMBOL_RE.finditer(text, start, end)
            )
            result.extend(
                self._fallback_range(
                    text,
                    offsets,
                    start,
                    end,
                    parent_kind=parent_kind,
                    parent_name=name,
                    symbols=symbols,
                )
            )
        return result

    def _python(self, text: str) -> list[ChunkDraft]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        lines = text.splitlines(keepends=True)
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))
        offsets = _Offsets(text)
        boundaries: list[
            tuple[
                int,
                int,
                str,
                str,
                tuple[tuple[str, str], ...],
                tuple[tuple[str, str, str], ...],
            ]
        ] = []
        for node in tree.body:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            start = line_starts[node.lineno - 1]
            end = line_starts[min(node.end_lineno, len(lines))]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                name = node.name
            elif isinstance(node, ast.ClassDef):
                kind = "class"
                name = node.name
            else:
                kind = "module_block"
                name = type(node).__name__
            symbols: list[tuple[str, str]] = []
            edges: list[tuple[str, str, str]] = []
            for child in ast.walk(node):
                if isinstance(child, ast.ClassDef):
                    symbols.append((child.name, "class"))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append((child.name, "function"))
                if isinstance(child, ast.Call):
                    try:
                        target = ast.unparse(child.func)
                    except (AttributeError, ValueError):
                        target = ""
                    if target:
                        edges.append((name, "calls", target))
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    module = getattr(child, "module", None)
                    imported = [alias.name for alias in child.names]
                    for target in imported:
                        qualified = f"{module}.{target}" if module else target
                        edges.append((name, "imports", qualified))
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    try:
                        target = ast.unparse(base)
                    except (AttributeError, ValueError):
                        target = ""
                    if target:
                        edges.append((name, "inherits", target))
            boundaries.append(
                (
                    start,
                    end,
                    kind,
                    name,
                    tuple(dict.fromkeys(symbols)),
                    tuple(dict.fromkeys(edges)),
                )
            )
        if not boundaries:
            return []
        result: list[ChunkDraft] = []
        first_start = boundaries[0][0]
        if first_start:
            result.extend(
                self._fallback_range(
                    text,
                    offsets,
                    0,
                    first_start,
                    parent_kind="module_preamble",
                    parent_name=Path("module").name,
                )
            )
        cursor = first_start
        for start, end, kind, name, symbols, code_edges in boundaries:
            if start > cursor:
                result.extend(
                    self._fallback_range(
                        text,
                        offsets,
                        cursor,
                        start,
                        parent_kind="module_gap",
                        parent_name="module",
                    )
                )
            result.extend(
                self._fallback_range(
                    text,
                    offsets,
                    start,
                    end,
                    parent_kind=kind,
                    parent_name=name,
                    symbols=symbols,
                    code_edges=code_edges,
                )
            )
            cursor = max(cursor, end)
        if cursor < len(text):
            result.extend(
                self._fallback_range(
                    text,
                    offsets,
                    cursor,
                    len(text),
                    parent_kind="module_tail",
                    parent_name="module",
                )
            )
        return result

    def _json(self, text: str) -> list[ChunkDraft]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, (dict, list)):
            return []
        # Preserve exact source slices.  Top-level line/object boundaries are
        # used as structural hints; the complete source remains in documents.
        offsets = _Offsets(text)
        if "\n" not in text or len(list(TOKEN_RE.finditer(text))) <= self.max_tokens:
            item = offsets.draft(
                0,
                len(text),
                parent_kind="json_root",
                parent_name="$",
                metadata={"json_type": type(parsed).__name__},
            )
            return [item] if item is not None else []
        result: list[ChunkDraft] = []
        cursor = 0
        depth = 0
        in_string = False
        escaped = False
        for index, character in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                depth += 1
            elif character in "]}":
                depth -= 1
            elif character == "," and depth == 1:
                result.extend(
                    self._fallback_range(
                        text,
                        offsets,
                        cursor,
                        index + 1,
                        parent_kind="json_node",
                        parent_name="$",
                    )
                )
                cursor = index + 1
        if cursor < len(text):
            result.extend(
                self._fallback_range(
                    text,
                    offsets,
                    cursor,
                    len(text),
                    parent_kind="json_node",
                    parent_name="$",
                )
            )
        return result
