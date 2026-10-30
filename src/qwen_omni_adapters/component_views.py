"""Verified reuse of derived GGUF files; never a heterogeneous graph loader."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from qwen_omni_adapters.single_gguf import (
    COMPONENT_BY_NAME,
    ComponentViewIndex,
    SingleGGUFError,
)

INDEX_SCHEMA = "robit.omni.component-view-index.v1"
EXTRACTOR_VERSION = 1
INDEX_FILENAME = "component-view-index.json"
MAX_INDEX_BYTES = 1024 * 1024


def _source_identity(path: Path) -> dict[str, int]:
    info = path.stat()
    return {name: getattr(info, name) for name in (
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
    )}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_INDEX_BYTES + 1)
        if len(raw) > MAX_INDEX_BYTES:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("binding") != binding:
            return {}
        if data.get("schema") != INDEX_SCHEMA or not isinstance(data.get("views"), dict):
            return {}
        return data["views"]
    except (OSError, ValueError):
        return {}


def _verified_report(entry: Any, output: Path) -> dict[str, Any] | None:
    if not isinstance(entry, dict) or entry.get("filename") != output.name:
        return None
    report = entry.get("report")
    if not isinstance(report, dict) or not output.is_file() or output.is_symlink():
        return None
    try:
        if output.stat().st_size != report.get("output_size_bytes"):
            return None
        if _sha256(output) != entry.get("sha256"):
            return None
    except OSError:
        return None
    return {**report, "output": str(output), "reused": True}


def _write_index(path: Path, data: dict[str, Any]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
            suffix=".partial", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_component_views(
    *, bundle: Path, bundle_digest: str, output_dir: Path, views: tuple[str, ...],
    overwrite: bool, reuse: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Use one source inventory and reuse only source-bound, SHA-verified views.

    The digest is the descriptor supplied by the existing Ollama resolver.
    Source inode/timestamps also invalidate receipts if that installed blob is
    changed in place. Output hashes are checked on every reuse, even at the same
    size. A receipt does not prove the source descriptor's authenticity or that
    a heterogeneous GGUF can be loaded directly by stock llama.cpp.
    """
    source = bundle.expanduser().resolve()
    if not views or set(views) - {"base", *COMPONENT_BY_NAME}:
        raise SingleGGUFError("component preparation requires known, nonempty views")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity = _source_identity(source)
    binding = {"bundle_digest": bundle_digest, "source": str(source),
               "source_identity": identity, "extractor_version": EXTRACTOR_VERSION}
    receipt_path = destination / INDEX_FILENAME
    entries = _read_index(receipt_path, binding)
    outputs: dict[str, dict[str, Any]] = {}
    paths = {view: destination / (view.replace("_", "-") + ".gguf")
             for view in dict.fromkeys(views)}
    # Check conflicts before constructing any new file. The standalone unpack
    # operation and preparation without reuse keep their existing refusal.
    for view, output in paths.items():
        if output.is_symlink():
            raise SingleGGUFError(f"component output must not be a symlink: {output}")
        cached = _verified_report(entries.get(view), output) if reuse else None
        if cached is not None:
            outputs[view] = cached
        elif output.exists() and not overwrite:
            raise SingleGGUFError(
                f"output already exists; pass overwrite=True to replace it: {output}"
            )
    index = None
    for view, output in paths.items():
        if view in outputs:
            continue
        if index is None:
            index = ComponentViewIndex(source)
        report = index.materialize(view=view, out_gguf=output, overwrite=overwrite)
        entries[view] = {"filename": output.name, "sha256": _sha256(output),
                         "report": report}
        outputs[view] = {**report, "reused": False}
    if _source_identity(source) != identity:
        raise SingleGGUFError("bundle changed during component preparation; retry from a stable blob")
    if index is not None:
        _write_index(receipt_path, {"schema": INDEX_SCHEMA, "binding": binding, "views": entries})
    reused = [view for view, report in outputs.items() if report["reused"]]
    rebuilt = [view for view in outputs if view not in reused]
    return outputs, {
        "schema": INDEX_SCHEMA, "path": str(receipt_path),
        "reused_views": reused, "materialized_views": rebuilt,
        "reused_bytes": sum(outputs[view]["output_size_bytes"] for view in reused),
        "materialized_bytes": sum(outputs[view]["output_size_bytes"] for view in rebuilt),
    }
