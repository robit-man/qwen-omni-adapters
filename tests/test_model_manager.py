from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from harness.indicator import model_views
from harness.models import IndicatorModelManager, _ollama_api_tags, _replace_env
from qwen_omni_adapters.model_catalog import MANAGED_MODELS


def _manager(tmp_path: Path, active: str, installed: set[str]) -> IndicatorModelManager:
    manager = IndicatorModelManager.__new__(IndicatorModelManager)
    manager.repo_root = tmp_path
    manager.active_model = active
    manager.ollama_url = "http://127.0.0.1:11434"
    manager._lock = threading.Lock()
    manager._operations = {}
    manager._cached_inventory = (time.monotonic(), installed, set())
    manager._inventory_refreshing = False
    return manager


def test_catalog_contains_compact_and_legacy_models_but_not_rejected_ornith() -> None:
    tags = [model.tag for model in MANAGED_MODELS]

    assert tags == [
        "robit/ornith-1.5-omni-audio-bridge:q4km",
        "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km",
        "robit/inkling-2-mini-omni:iq4-xs",
        "robit/inkling-2-mini-omni:iq3-xs",
        "robit/ornith-1.5-omni:q4km",
        "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
    ]
    assert not any("ornith-1.5-obliterated" in tag for tag in tags)
    assert [model.max_context_tokens for model in MANAGED_MODELS[:2]] == [
        65_536,
        16_384,
    ]
    assert [model.max_context_tokens for model in MANAGED_MODELS[2:4]] == [8_192, 8_192]
    assert [model.language_disable_thinking for model in MANAGED_MODELS] == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_ollama_inventory_is_parsed_as_exact_tags() -> None:
    payload = {"models": [{"name": "robit/a:q4"}, {"model": "robit/b:q8"}, {}, None]}

    assert _ollama_api_tags(payload) == {"robit/a:q4", "robit/b:q8"}
    with pytest.raises(ValueError, match="did not include a model list"):
        _ollama_api_tags({"models": None})


def test_environment_replacement_is_atomic_and_preserves_unrelated_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "KEEP_ME=yes\nOMNI_MODEL=old\nOMNI_CALL_SPEECH_EVICT_UNIT=legacy.service\n",
        encoding="utf-8",
    )

    _replace_env(
        path,
        {"OMNI_MODEL": "new", "OMNI_LANGUAGE_MODEL": "new"},
        {"OMNI_CALL_SPEECH_EVICT_UNIT"},
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "KEEP_ME=yes",
        "OMNI_MODEL=new",
        "OMNI_LANGUAGE_MODEL=new",
    ]


def test_activation_selects_one_trunk_and_requests_a_managed_restart(
    tmp_path: Path,
) -> None:
    selected = MANAGED_MODELS[0]
    manager = _manager(tmp_path, MANAGED_MODELS[1].tag, {selected.tag})

    ok, detail = manager.activate(selected.tag)

    assert ok is True
    assert "Activating" in detail
    environment = dict(
        line.split("=", 1) for line in (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    )
    assert environment["OMNI_MODEL"] == selected.tag
    assert environment["OMNI_LANGUAGE_MODEL"] == selected.tag
    assert environment["OMNI_COMPREHENSION_CONTEXT_TOKENS"] == "65536"
    assert environment["OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS"] == "65536"
    assert environment["OMNI_STARTUP_SMOKE"] == "0"
    request = json.loads(
        (tmp_path / "runtime-data/state/restart.request").read_text(encoding="utf-8")
    )
    assert request["model"] == selected.tag


def test_destructive_and_load_actions_fail_closed(tmp_path: Path) -> None:
    selected = MANAGED_MODELS[0]
    manager = _manager(tmp_path, selected.tag, set())

    assert manager.delete(selected.tag) == (False, "Model is not downloaded")
    assert manager.load(selected.tag) == (
        False,
        "Active model is already loaded by the daemon",
    )
    assert manager.load(MANAGED_MODELS[1].tag) == (
        False,
        "Download the model before loading it",
    )
    assert manager.action("unmanaged/model:latest", "delete") == (
        False,
        "model is not in the managed allowlist",
    )


@pytest.mark.parametrize(("profile", "quant"), [("inkling2-mini", "iq4-xs"), ("inkling2-mini-iq3-xs", "iq3-xs")])
def test_inkling_activation_requires_its_binaries_and_uses_8k(
    tmp_path: Path, profile: str, quant: str,
) -> None:
    tag = f"robit/inkling-2-mini-omni:{quant}"
    manager = _manager(tmp_path, MANAGED_MODELS[0].tag, {tag})

    ok, detail = manager.activate(tag)
    assert not ok and f"./deploy.sh {profile} before" in detail
    assert not (tmp_path / ".env").exists()

    binary_dir = tmp_path / "vendor/llama.cpp-inkling/build/bin"
    binary_dir.mkdir(parents=True)
    for name in ("llama-server", "llama-tts"):
        (binary_dir / name).touch(mode=0o700)

    ok, detail = manager.activate(tag)
    assert ok, detail
    environment = (tmp_path / ".env").read_text()
    assert f"OMNI_PROFILE={profile}\n" in environment
    assert f"OMNI_MODEL={tag}\n" in environment
    assert f"OMNI_LANGUAGE_MODEL={tag}\n" in environment
    assert "OMNI_COMPREHENSION_CONTEXT_TOKENS=8192\n" in environment


def test_indicator_model_rows_show_download_active_loaded_and_progress() -> None:
    views = model_views(
        [
            {
                "tag": "robit/missing:q4",
                "label": "Missing",
                "installed": False,
                "loaded": False,
                "active": False,
                "size_gib": 8.0,
                "generation": "compact",
            },
            {
                "tag": "robit/active:q4",
                "label": "Active",
                "installed": True,
                "loaded": True,
                "active": True,
                "size_gib": 18.0,
                "generation": "compact",
            },
            {
                "tag": "robit/pulling:q4",
                "label": "Pulling",
                "installed": False,
                "loaded": False,
                "active": False,
                "operation": {
                    "state": "running",
                    "action": "download",
                    "progress": 47,
                },
            },
        ]
    )

    assert "download required" in views[0]["label"]
    assert "active, Ollama loaded" in views[1]["label"]
    assert "download 47%" in views[2]["label"]


def test_active_daemon_residency_and_context_are_visible(tmp_path: Path) -> None:
    selected = MANAGED_MODELS[0]
    manager = _manager(tmp_path, selected.tag, {selected.tag})
    state = tmp_path / "runtime-data/state"
    state.mkdir(parents=True)
    (state / "daemon-status.json").write_text(
        json.dumps(
            {
                "state": "ready",
                "pid": os.getpid(),
                "model": selected.tag,
                "comprehension_context_tokens": 16_384,
                "comprehension_context_ceiling": 16_384,
            }
        ),
        encoding="utf-8",
    )

    view = manager.views()[0]

    assert view["service_loaded"] is True
    assert view["loaded"] is True
    assert view["context_tokens"] == 16_384
    assert "16,384 ctx" in model_views([view])[0]["label"]
