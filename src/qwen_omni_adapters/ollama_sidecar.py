from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from qwen_omni_adapters.single_gguf import (
    COMPREHENSION_PROJECTOR_COMPONENT,
    LIGHTWEIGHT_AUDIO_BRIDGE_SCHEMA,
    SPLIT_AUDIO_BRIDGE_SCHEMA,
    _add_reader_tensor,
    _require_gguf,
    inspect_monolithic_gguf,
    materialize_component_view,
)

OMNI_LAYER_MEDIA_TYPE = "application/vnd.robit.ollama.omni.bundle.v1+gguf"
OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
OLLAMA_PROJECTOR_MEDIA_TYPE = "application/vnd.ollama.image.projector"
DEFAULT_REGISTRY = "registry.ollama.ai"
RUNTIME_VIEWS = (
    "comprehension_model",
    "comprehension_projector",
    "tts_model",
    "tts_projector",
)


class OllamaSidecarError(RuntimeError):
    """Raised when an Omni sidecar cannot be resolved or attached safely."""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models_dir(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None,
        Path("/srv/ollama/models"),
        # The official Linux install script runs ollama as its own `ollama`
        # system user with this home, so this -- not ~/.ollama -- is where a
        # standard service-managed host keeps manifests and blobs.
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/.ollama/models"),
        Path.home() / ".ollama" / "models",
    ]
    # Prefer a store that actually holds manifests. An empty ~/.ollama/models
    # is created by any `ollama` client invocation, and taking the first
    # directory that merely exists let it shadow the real service store --
    # producing a "manifest not found" for a tag that is plainly installed.
    populated: list[Path] = []
    existing: list[Path] = []
    unreadable: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser()
        try:
            if not resolved.is_dir():
                continue
        except PermissionError:
            unreadable.append(resolved)
            continue
        resolved = resolved.resolve()
        existing.append(resolved)
        if any((resolved / folder).is_dir() for folder in ("manifests-v2", "manifests")):
            populated.append(resolved)
    for choice in (populated, existing):
        if choice:
            return choice[0]
    if unreadable:
        raise OllamaSidecarError(
            f"permission denied reading the Ollama models directory {unreadable[0]}; "
            "join the ollama group (sudo usermod -aG ollama $USER) and start a new login session"
        )
    raise OllamaSidecarError("could not locate the Ollama models directory")


def manifest_path(model: str, models_dir: Path) -> Path:
    reference = model.strip()
    if not reference:
        raise OllamaSidecarError("model reference cannot be empty")
    if ":" in reference.rsplit("/", 1)[-1]:
        repository, tag = reference.rsplit(":", 1)
    else:
        repository, tag = reference, "latest"
    parts = repository.split("/")
    if "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        registry = parts.pop(0)
    else:
        registry = DEFAULT_REGISTRY
    if len(parts) == 1:
        parts.insert(0, "library")
    if len(parts) < 2 or not all(parts) or not tag:
        raise OllamaSidecarError(f"invalid Ollama model reference: {model!r}")
    v2_registry = "ollama.com" if registry == DEFAULT_REGISTRY else registry
    v2_path = models_dir / "manifests-v2" / v2_registry / Path(*parts) / tag
    if v2_path.is_symlink() or v2_path.is_file():
        return v2_path
    return models_dir / "manifests" / registry / Path(*parts) / tag


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OllamaSidecarError(f"Ollama manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaSidecarError(f"cannot read Ollama manifest {path}: {exc}") from exc
    if not isinstance(manifest.get("layers"), list):
        raise OllamaSidecarError(f"Ollama manifest has no layer list: {path}")
    return manifest


def _resolve_standard_layers(
    manifest: dict[str, Any],
    root: Path,
) -> dict[str, dict[str, Any]]:
    media_types = {
        "language_model": OLLAMA_MODEL_MEDIA_TYPE,
        "projector": OLLAMA_PROJECTOR_MEDIA_TYPE,
    }
    resolved: dict[str, dict[str, Any]] = {}
    for name, media_type in media_types.items():
        matches = [
            layer for layer in manifest["layers"] if layer.get("mediaType") == media_type
        ]
        if len(matches) > 1:
            raise OllamaSidecarError(
                f"expected at most one {name} layer, found {len(matches)}"
            )
        if not matches:
            continue
        layer = matches[0]
        digest = str(layer.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise OllamaSidecarError(f"{name} layer has no SHA-256 digest")
        blob = root / "blobs" / digest.replace(":", "-", 1)
        available = blob.is_file() and blob.stat().st_size == layer.get("size")
        resolved[name] = {
            "layer": layer,
            "path": str(blob),
            "available": available,
        }
    return resolved


def _install_blob(source: Path, destination: Path) -> str:
    digest = _sha256(source)
    expected = destination / f"sha256-{digest}"
    if expected.exists():
        if expected.stat().st_size != source.stat().st_size or _sha256(expected) != digest:
            raise OllamaSidecarError(f"existing Ollama blob failed digest validation: {expected}")
        return digest
    expected.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, expected)
    except OSError:
        partial = expected.with_name(expected.name + ".partial")
        try:
            with source.open("rb") as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if _sha256(partial) != digest:
                raise OllamaSidecarError("copied sidecar blob failed digest validation")
            os.replace(partial, expected)
        finally:
            if partial.exists():
                partial.unlink()
    return digest


def _write_manifest(path: Path, root: Path, manifest: dict[str, Any]) -> None:
    contents = json.dumps(manifest, separators=(",", ":"), sort_keys=True)
    with TemporaryDirectory(prefix=".omni-manifest-", dir=path.parent) as directory:
        partial = Path(directory) / "manifest"
        partial.write_text(contents, encoding="utf-8")
        if root / "manifests-v2" in path.parents:
            # V2 tags point to content-addressed manifests that other tags can share.
            partial.chmod(0o644)
            digest = _install_blob(partial, root / "blobs")
            blob = root / "blobs" / f"sha256-{digest}"
            pointer = Path(directory) / "tag"
            pointer.symlink_to(os.path.relpath(blob, start=path.parent))
            partial = pointer
        os.replace(partial, path)


def _verify_split_layer_bindings(
    inspection: dict[str, Any], standard_layers: dict[str, dict[str, Any]],
) -> None:
    digests = inspection["manifest"]["standard_layers"]["digests"]
    for name, digest in digests.items():
        actual = standard_layers.get(name) or {}
        if not actual.get("available") or actual["layer"].get("digest") != digest:
            raise OllamaSidecarError(f"split audio sidecar requires the bound {name} layer {digest}")


def attach_ollama_sidecar(
    *,
    model: str,
    bundle_gguf: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach a custom Omni GGUF layer to an existing runnable Ollama tag."""
    bundle = bundle_gguf.expanduser().resolve()
    inspection = inspect_monolithic_gguf(bundle)
    if not inspection["valid"]:
        raise OllamaSidecarError(f"invalid Omni bundle: {inspection['errors']}")
    root = resolve_models_dir(models_dir)
    path = manifest_path(model, root)
    manifest = _read_manifest(path)
    if inspection["manifest"]["schema"] == SPLIT_AUDIO_BRIDGE_SCHEMA:
        _verify_split_layer_bindings(inspection, _resolve_standard_layers(manifest, root))
    digest = _install_blob(bundle, root / "blobs")
    layer = {
        "mediaType": OMNI_LAYER_MEDIA_TYPE,
        "digest": f"sha256:{digest}",
        "size": bundle.stat().st_size,
        "annotations": {
            "org.opencontainers.image.title": bundle.name,
            "io.robit.omni.schema": inspection["manifest"]["schema"],
        },
    }
    previous_sidecars = sum(
        existing.get("mediaType") == OMNI_LAYER_MEDIA_TYPE for existing in manifest["layers"]
    )
    retained = [
        existing
        for existing in manifest["layers"]
        if existing.get("mediaType") != OMNI_LAYER_MEDIA_TYPE
    ]
    manifest["layers"] = [*retained, layer]
    _write_manifest(path, root, manifest)
    return {
        "model": model,
        "manifest": str(path),
        "models_dir": str(root),
        "layer": layer,
        "replaced_existing_sidecar": previous_sidecars > 0,
        "bundle": str(bundle),
        "bundle_inspection": {
            "valid": inspection["valid"],
            "tensor_count": inspection["tensor_count"],
            "view_tensor_counts": inspection["view_tensor_counts"],
        },
    }


def resolve_ollama_sidecar(
    *,
    model: str,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate the Omni sidecar layer for a local Ollama tag."""
    root = resolve_models_dir(models_dir)
    path = manifest_path(model, root)
    manifest = _read_manifest(path)
    layers = [
        layer for layer in manifest["layers"] if layer.get("mediaType") == OMNI_LAYER_MEDIA_TYPE
    ]
    if len(layers) != 1:
        raise OllamaSidecarError(
            f"expected exactly one Omni sidecar layer for {model}, found {len(layers)}"
        )
    layer = layers[0]
    digest = str(layer.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise OllamaSidecarError("Omni sidecar layer has no SHA-256 digest")
    blob = root / "blobs" / digest.replace(":", "-", 1)
    if not blob.is_file() or blob.stat().st_size != layer.get("size"):
        raise OllamaSidecarError(f"Omni sidecar blob is missing or has the wrong size: {blob}")
    inspection = inspect_monolithic_gguf(blob)
    if not inspection["valid"]:
        raise OllamaSidecarError(f"Omni sidecar failed inspection: {inspection['errors']}")
    schema = (inspection.get("manifest") or {}).get("schema")
    split_projector = schema == SPLIT_AUDIO_BRIDGE_SCHEMA
    profile = (
        "trained-audio-bridge"
        if schema in {LIGHTWEIGHT_AUDIO_BRIDGE_SCHEMA, SPLIT_AUDIO_BRIDGE_SCHEMA}
        else "legacy-cascade"
    )
    standard_layers = _resolve_standard_layers(manifest, root)
    if split_projector:
        _verify_split_layer_bindings(inspection, standard_layers)
    if profile == "trained-audio-bridge":
        required = {"language_model", "projector"}
        missing = sorted(
            name
            for name in required
            if name not in standard_layers or not standard_layers[name]["available"]
        )
        if missing:
            raise OllamaSidecarError(
                "trained audio bridge requires standard Ollama layers: "
                + ", ".join(missing)
            )
    return {
        "model": model,
        "manifest": str(path),
        "layer": layer,
        "bundle": str(blob),
        "inspection": inspection,
        "profile": profile,
        "split_projector": split_projector,
        "standard_layers": standard_layers,
    }


def _materialize_split_projector(
    *, bundle: Path, vision: Path, output: Path, overwrite: bool,
) -> dict[str, Any]:
    gguf, np = _require_gguf()
    if output.resolve() in {bundle.resolve(), vision.resolve()}:
        raise OllamaSidecarError("runtime projector cannot overwrite its source layers")
    if output.exists() and not overwrite:
        raise OllamaSidecarError(f"runtime projector already exists: {output}")
    image = gguf.GGUFReader(str(vision))
    audio = gguf.GGUFReader(str(bundle))
    audio_prefix = COMPREHENSION_PROJECTOR_COMPONENT.metadata_prefix
    tensor_prefix = COMPREHENSION_PROJECTOR_COMPONENT.tensor_prefix
    audio_fields = {
        name.removeprefix(audio_prefix): field
        for name, field in audio.fields.items()
        if name.startswith(audio_prefix + "clip.audio.")
        or name == audio_prefix + "clip.has_audio_encoder"
    }
    audio_tensors = {
        tensor.name.removeprefix(tensor_prefix): tensor
        for tensor in audio.tensors
        if tensor.name.startswith(tensor_prefix)
    }
    if (
        image.fields["general.architecture"].contents() != "clip"
        or not image.get_field("clip.has_vision_encoder")
        or not image.fields["clip.has_vision_encoder"].contents()
        or (image.get_field("clip.has_audio_encoder") and image.fields["clip.has_audio_encoder"].contents())
        or any(tensor.name.startswith(("a.", "mm.a.")) for tensor in image.tensors)
    ):
        raise OllamaSidecarError("split audio requires an image-only CLIP layer")
    vision_width = image.get_field("clip.vision.projection_dim")
    audio_width = audio_fields.get("clip.audio.projection_dim")
    audio_enabled = audio_fields.get("clip.has_audio_encoder")
    audio_type = audio_fields.get("clip.audio.projector_type")
    weight = audio_tensors.get("mm.a.mlp.2.weight")
    bias = audio_tensors.get("mm.a.mlp.2.bias")
    if (
        not vision_width or not audio_width or vision_width.contents() != audio_width.contents()
        or not audio_enabled or not audio_enabled.contents()
        or not audio_type or audio_type.contents() != "qwen3a"
        or weight is None or tuple(weight.shape) != (1280, audio_width.contents())
        or bias is None or tuple(bias.shape) != (audio_width.contents(),)
        or any(not name.startswith(("a.", "mm.a.")) for name in audio_tensors)
    ):
        raise OllamaSidecarError("split audio and image projector geometry does not match")

    excluded = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "clip.projector_type"}
    fields = {name: field for name, field in image.fields.items() if name not in excluded}
    if "clip.vision.projector_type" not in fields:
        projector_type = image.get_field("clip.projector_type")
        if not projector_type:
            raise OllamaSidecarError("image projector type is missing")
        fields["clip.vision.projector_type"] = projector_type
    fields.update(audio_fields)

    with NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as tmp:
        partial = Path(tmp.name)
    writer = gguf.GGUFWriter(str(partial), arch="clip")
    try:
        try:
            for name, field in fields.items():
                value = field.contents()
                if not isinstance(value, list) or value:
                    writer.add_key_value(name, value, field.types[0])
            for tensor in image.tensors:
                _add_reader_tensor(writer, tensor, tensor.name, np)
            for name, tensor in audio_tensors.items():
                _add_reader_tensor(writer, tensor, name, np)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "tensor_count": len(image.tensors) + len(audio_tensors),
        "sha256": _sha256(output),
    }


def prepare_ollama_sidecar(
    *,
    model: str,
    output_dir: Path,
    views: tuple[str, ...] | None = None,
    models_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize disposable runtime views directly from an installed Ollama tag."""
    resolved = resolve_ollama_sidecar(model=model, models_dir=models_dir)
    default_views = (
        ("tts_model", "tts_projector")
        if resolved["profile"] == "trained-audio-bridge"
        else RUNTIME_VIEWS
    )
    if resolved["split_projector"]:
        default_views = ("comprehension_projector", *default_views)
    selected_views = default_views if views is None else views
    unknown = sorted(set(selected_views) - set(RUNTIME_VIEWS))
    if unknown:
        raise OllamaSidecarError(f"unsupported runtime views: {', '.join(unknown)}")
    if not selected_views:
        raise OllamaSidecarError("at least one runtime view is required")
    declared = {
        str(component.get("name"))
        for component in (resolved["inspection"].get("manifest") or {}).get(
            "components", []
        )
        if isinstance(component, dict)
    }
    unavailable = sorted(set(selected_views) - declared)
    if unavailable:
        raise OllamaSidecarError(
            f"sidecar profile {resolved['profile']} has no runtime views: "
            + ", ".join(unavailable)
        )
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for view in selected_views:
        output = destination / (view.replace("_", "-") + ".gguf")
        if resolved["split_projector"] and view == "comprehension_projector":
            outputs[view] = _materialize_split_projector(
                bundle=Path(resolved["bundle"]),
                vision=Path(resolved["standard_layers"]["projector"]["path"]),
                output=output,
                overwrite=overwrite,
            )
        else:
            outputs[view] = materialize_component_view(
                bundle_gguf=Path(resolved["bundle"]),
                view=view,
                out_gguf=output,
                overwrite=overwrite,
            )
    if resolved["split_projector"] and set(default_views).issubset(outputs):
        marker = destination / "split-audio.sha256"
        temporary = marker.with_suffix(".partial")
        temporary.write_text(resolved["layer"]["digest"] + "\n", encoding="utf-8")
        temporary.replace(marker)
    return {
        "model": model,
        "bundle": resolved["bundle"],
        "bundle_digest": resolved["layer"]["digest"],
        "output_dir": str(destination),
        "disposable_cache": True,
        "profile": resolved["profile"],
        "standard_layers": resolved["standard_layers"],
        "views": outputs,
    }
