from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = REPO_ROOT / "deploy.sh"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DEPLOY), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_guided_deployer_lists_only_the_two_release_audio_bridges() -> None:
    completed = _run("--list-models")

    assert completed.returncode == 0
    assert "robit/ornith-1.5-omni-audio-bridge:q4km" in completed.stdout
    assert "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km" in completed.stdout
    assert "ornith-1.5-obliterated" not in completed.stdout
    assert len(completed.stdout.strip().splitlines()) == 2


def test_noninteractive_dry_run_uses_one_bridge_tag_for_both_stages() -> None:
    completed = _run(
        "--profile",
        "ornith15",
        "--action",
        "deploy",
        "--no-update",
        "--no-harness",
        "--yes",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    tag = "robit/ornith-1.5-omni-audio-bridge:q4km"
    assert f"OMNI_MODEL={tag}" in completed.stdout
    assert f"OMNI_LANGUAGE_MODEL={tag}" in completed.stdout
    assert "scripts/bootstrap.sh --refresh-models" in completed.stdout
    assert "services/linux/install.sh --auto --no-enable" in completed.stdout
    assert "inventory owners of ports 8892,8901,8910,8920,8930" in completed.stdout
    assert "unload only the selected, prior-configured" in completed.stdout
    assert "systemctl start qwen-omni-adapters.service" in completed.stdout
    assert "wait up to 30 minutes for state=ready" in completed.stdout


def test_noninteractive_mode_fails_closed_without_a_profile() -> None:
    completed = _run("--action", "deploy", "--dry-run")

    assert completed.returncode != 0
    assert "non-interactive use requires --profile" in completed.stderr


def test_cutover_stops_and_unloads_the_old_runtime_before_installing() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    deploy_body = source.split("deploy_service() {", 1)[1].split("\n}\n\nwhile (($#))", 1)[0]

    assert deploy_body.index("prepare_runtime_handoff") < deploy_body.index(
        "install_environment"
    )
    assert "unload_relevant_ollama_models" in source
    assert "handoff_command admit" in source
    assert "OMNI_DEPLOY_MEMORY_RESERVE_MIB:-6144" in source
    assert "nvidia-smi" not in source


def test_readiness_wait_accepts_activation_and_prints_the_journal_on_failure() -> None:
    source = DEPLOY.read_text(encoding="utf-8")

    assert "active|activating|reloading" in source
    assert "NRestarts" in source
    assert 'journalctl -u "$SERVICE_NAME"' in source


def test_arrow_key_menu_selects_qwen_bridge() -> None:
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            str(DEPLOY),
            "--no-update",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    selected_action = False
    selected_model = False
    selected_service = False
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                if not selected_action and b"Choose what to do" in output:
                    os.write(master, b"\r")
                    selected_action = True
                    output.clear()
                elif not selected_model and b"Select the logical Omni model" in output:
                    os.write(master, b"\x1b[B\r")
                    selected_model = True
                    output.clear()
                elif not selected_service and b"Select the service deployment" in output:
                    os.write(master, b"\r")
                    selected_service = True
            if process.poll() is not None:
                break
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)

    rendered = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output.decode(errors="replace"))
    assert selected_action, rendered
    assert selected_model, rendered
    assert selected_service, rendered
    assert process.returncode == 0, rendered
    assert "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km" in rendered
