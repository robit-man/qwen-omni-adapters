from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from qwen_omni_adapters.contract import adapter_contract
from qwen_omni_adapters.ollama_sidecar import (
    OllamaSidecarError,
    attach_ollama_sidecar,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from qwen_omni_adapters.single_gguf import (
    SingleGGUFError,
    inspect_monolithic_gguf,
    materialize_component_view,
    pack_monolithic_gguf,
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _run(command: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, detail[-2000:]


def _doctor(args: argparse.Namespace) -> int:
    system = platform.system()
    commands = ["cmake", "curl", "ffmpeg", "git", "ollama", "openssl"]
    if system == "Linux":
        commands.extend(["docker", "jq", "nvidia-smi", "ss"])
    elif system == "Windows":
        commands.append("nvidia-smi")
    if not args.no_tunnel:
        commands.append("cloudflared")
    checks: dict[str, Any] = {
        "commands": {name: shutil.which(name) for name in commands},
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
    }
    repo_root = Path(__file__).resolve().parents[2]
    binaries = {}
    for name in ("llama-server", "llama-tts"):
        candidates = [
            repo_root / "vendor" / "llama.cpp" / "build" / "bin" / name,
            repo_root / "vendor" / "llama.cpp" / "build" / "bin" / f"{name}.exe",
            repo_root / "vendor" / "llama.cpp" / "build" / "bin" / "Release" / f"{name}.exe",
        ]
        binaries[name] = next((path for path in candidates if path.is_file()), candidates[0])
    checks["llama_cpp"] = {
        name: {"path": str(path), "executable": path.is_file() and path.stat().st_mode & 0o111 != 0}
        for name, path in binaries.items()
    }
    if shutil.which("ollama"):
        checks["models"] = {}
        for model in (args.model, args.language_model):
            ok, detail = _run(["ollama", "show", model])
            checks["models"][model] = {"installed": ok, "detail": detail if not ok else ""}
    if args.deployment and system == "Linux" and shutil.which("docker"):
        ok, detail = _run(["docker", "gpu", "discover"])
        checks["gpu_broker"] = {"available": ok, "detail": detail}

    missing_commands = [name for name, path in checks["commands"].items() if not path]
    missing_binaries = [
        name for name, value in checks["llama_cpp"].items() if not value["executable"]
    ]
    missing_models = [
        name for name, value in checks.get("models", {}).items() if not value["installed"]
    ]
    broker_failed = (
        args.deployment
        and system == "Linux"
        and not checks.get("gpu_broker", {}).get("available", False)
    )
    checks["ok"] = not (missing_commands or missing_binaries or missing_models or broker_failed)
    checks["remediation"] = {
        "missing_commands": missing_commands,
        "missing_llama_cpp_binaries": missing_binaries,
        "missing_models": missing_models,
        "bootstrap": "./scripts/bootstrap.sh",
    }
    _print_json(checks)
    return 0 if checks["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-omni",
        description="Inspect, attach, resolve, and run logical Ollama Omni bundles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("contract", help="Print the adapter v1 wire contract")

    pack = sub.add_parser("pack", help="Pack executable component GGUFs into one sidecar")
    pack.add_argument("--base", "--base-gguf", dest="base", type=Path, required=True)
    pack.add_argument("--base-projector", "--base-projector-gguf", dest="base_projector", type=Path)
    pack.add_argument(
        "--comprehension",
        "--comprehension-gguf",
        dest="comprehension",
        type=Path,
        required=True,
    )
    pack.add_argument(
        "--comprehension-projector",
        "--comprehension-projector-gguf",
        dest="comprehension_projector",
        type=Path,
    )
    pack.add_argument("--tts", "--tts-gguf", dest="tts", type=Path, required=True)
    pack.add_argument("--tts-projector", "--tts-projector-gguf", dest="tts_projector", type=Path)
    pack.add_argument("--out", type=Path, required=True)
    pack.add_argument("--base-source")
    pack.add_argument("--base-projector-source")
    pack.add_argument("--comprehension-source")
    pack.add_argument("--comprehension-projector-source")
    pack.add_argument("--tts-source")
    pack.add_argument("--tts-projector-source")
    pack.add_argument("--overwrite", action="store_true")

    inspect = sub.add_parser("inspect", help="Validate and describe a sidecar GGUF")
    inspect.add_argument("bundle", type=Path)

    materialize = sub.add_parser("materialize", help="Extract one executable GGUF view")
    materialize.add_argument("bundle", type=Path)
    materialize.add_argument("view")
    materialize.add_argument("--out", type=Path, required=True)
    materialize.add_argument("--overwrite", action="store_true")

    attach = sub.add_parser("attach", help="Attach a sidecar to an installed Ollama tag")
    attach.add_argument("model")
    attach.add_argument("bundle", type=Path)
    attach.add_argument("--models-dir", type=Path)

    resolve = sub.add_parser("resolve", help="Resolve and validate a tag's Omni sidecar")
    resolve.add_argument("model")
    resolve.add_argument("--models-dir", type=Path)

    prepare = sub.add_parser("prepare", help="Materialize disposable runtime component views")
    prepare.add_argument("model")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument(
        "--view",
        dest="views",
        action="append",
        choices=(
            "comprehension_model",
            "comprehension_projector",
            "tts_model",
            "tts_projector",
        ),
        help="View to materialize; repeat to select a subset",
    )
    prepare.add_argument("--models-dir", type=Path)
    prepare.add_argument("--overwrite", action="store_true")

    doctor = sub.add_parser("doctor", help="Check the host before deployment")
    doctor.add_argument(
        "--model",
        default="robit/qwen3.8-27b-e03-obliterated-omni:q4km",
    )
    doctor.add_argument(
        "--language-model",
        default="robit/qwen3.8-27b-obliterated-e03:27b",
    )
    doctor.add_argument("--deployment", action="store_true", help="Also query the GPU broker")
    doctor.add_argument("--no-tunnel", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "contract":
            result = adapter_contract()
        elif args.command == "pack":
            result = pack_monolithic_gguf(
                base_gguf=args.base,
                base_projector_gguf=args.base_projector,
                comprehension_gguf=args.comprehension,
                comprehension_projector_gguf=args.comprehension_projector,
                tts_gguf=args.tts,
                tts_projector_gguf=args.tts_projector,
                out_gguf=args.out,
                base_source=args.base_source,
                base_projector_source=args.base_projector_source,
                comprehension_source=args.comprehension_source,
                comprehension_projector_source=args.comprehension_projector_source,
                tts_source=args.tts_source,
                tts_projector_source=args.tts_projector_source,
                overwrite=args.overwrite,
            )
        elif args.command == "inspect":
            result = inspect_monolithic_gguf(args.bundle)
        elif args.command == "materialize":
            result = materialize_component_view(
                bundle_gguf=args.bundle,
                view=args.view,
                out_gguf=args.out,
                overwrite=args.overwrite,
            )
        elif args.command == "attach":
            result = attach_ollama_sidecar(
                model=args.model,
                bundle_gguf=args.bundle,
                models_dir=args.models_dir,
            )
        elif args.command == "resolve":
            result = resolve_ollama_sidecar(model=args.model, models_dir=args.models_dir)
        elif args.command == "prepare":
            views = (
                tuple(args.views)
                if args.views
                else (
                    "comprehension_model",
                    "comprehension_projector",
                    "tts_model",
                    "tts_projector",
                )
            )
            result = prepare_ollama_sidecar(
                model=args.model,
                output_dir=args.out,
                views=views,
                models_dir=args.models_dir,
                overwrite=args.overwrite,
            )
        elif args.command == "doctor":
            return _doctor(args)
        else:  # pragma: no cover - argparse enforces a command
            raise AssertionError(args.command)
    except (OllamaSidecarError, SingleGGUFError, ValueError) as exc:
        print(f"qwen-omni: {exc}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0
