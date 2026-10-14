from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
    assert "start the desktop indicator immediately" in completed.stdout
    assert "do not run generation smoke" in completed.stdout


def test_fresh_host_installs_missing_prerequisites_before_bootstrap(tmp_path: Path) -> None:
    fresh_bin = tmp_path / "bin"
    fresh_bin.mkdir()
    for directory in (Path("/usr/bin"), Path("/bin")):
        for tool in directory.iterdir():
            if tool.name in {"cloudflared", "ffmpeg", "node", "ollama"} or (fresh_bin / tool.name).exists():
                continue
            (fresh_bin / tool.name).symlink_to(tool)
    completed = subprocess.run(
        [
            str(DEPLOY),
            "--profile",
            "ornith15",
            "--action",
            "deploy",
            "--no-update",
            "--no-harness",
            "--yes",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PATH": str(fresh_bin)},
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "Installing missing deployment prerequisites:" in output
    assert "apt-get -o DPkg::Lock::Timeout=600 install -y ffmpeg" in output
    assert "deb.nodesource.com/setup_22.x" in output
    assert "ollama.com/install.sh" in output
    assert "cloudflared/releases/latest/download/cloudflared-linux-" in output
    assert output.index("ollama.com/install.sh") < output.index("scripts/bootstrap.sh")


def test_dry_run_plans_camera_stack_and_automatic_updates() -> None:
    completed = _run(
        "--profile", "ornith15", "--action", "deploy", "--no-update",
        "--no-harness", "--yes", "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "install omni-auto-update.timer" in completed.stdout
    source = DEPLOY.read_text(encoding="utf-8")
    assert '"$ECAM_DIR/install.sh" --yes --if-needed' in source

    opted_out = _run(
        "--profile", "ornith15", "--action", "deploy", "--no-update", "--no-harness",
        "--no-camera", "--no-auto-update", "--yes", "--dry-run",
    )
    assert opted_out.returncode == 0, opted_out.stderr
    assert "omni-auto-update" not in opted_out.stdout
    assert "jetson-ecam-gmsl" not in opted_out.stdout
    assert "Camera stack" not in opted_out.stdout


def test_camera_stack_runs_before_the_runtime_handoff() -> None:
    completed = _run(
        "--profile", "ornith15", "--action", "deploy", "--no-update",
        "--no-harness", "--yes", "--dry-run",
    )
    assert completed.returncode == 0, completed.stderr
    source = DEPLOY.read_text(encoding="utf-8")
    main_flow = source.rsplit('"${bootstrap[@]}"\n', 1)[1]
    assert main_flow.index("install_camera_stack") < main_flow.index("deploy_service")


def test_unreadable_port_owner_is_identified_with_sudo_before_refusing() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    handoff = source.split("prepare_runtime_handoff() {", 1)[1].split("\n}\n", 1)[0]
    assert "sudo \"$REPO_ROOT/.venv/bin/python\" -m qwen_omni_adapters.deployment_handoff" in handoff
    assert handoff.index("handoff_command targets") < handoff.index("sudo \"$REPO_ROOT/.venv/bin/python\"")
    assert 'sudo kill -0 "$pid"' in source


def test_desktop_selection_bootstraps_and_waits_for_the_real_indicator() -> None:
    completed = _run(
        "--profile",
        "ornith15",
        "--action",
        "deploy",
        "--no-update",
        "--with-harness",
        "--yes",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "scripts/bootstrap.sh --refresh-models --with-harness" in completed.stdout
    assert "services/linux/install.sh --auto --no-enable --with-harness" in completed.stdout
    assert "wait up to 120 seconds for a live GTK/AppIndicator harness status" in completed.stdout
    source = DEPLOY.read_text(encoding="utf-8")
    assert "listening|hearing|thinking|speaking|muted" in source
    assert "indicator-ready" not in source.split("wait_for_harness()", 1)[1].split(
        "wait_for_service()", 1
    )[0]


def test_yes_path_defaults_to_the_visible_desktop_indicator() -> None:
    completed = _run(
        "--profile",
        "ornith15",
        "--action",
        "deploy",
        "--no-update",
        "--yes",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Services:   core daemon + always-listening harness" in completed.stdout
    assert "scripts/bootstrap.sh --refresh-models --with-harness" in completed.stdout
    assert "services/linux/install.sh --auto --no-enable --with-harness" in completed.stdout


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
    assert "egg-omni-*.service" in source
    assert "LEGACY_USER_UNITS" in source


def test_guided_bridge_disables_legacy_eviction_and_blocking_smoke() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    install_body = source.split("install_environment() {", 1)[1].split(
        "\n}\n\nrestore_environment", 1
    )[0]

    assert "OMNI_CALL_SPEECH_EVICT_UNIT" in install_body
    assert "OMNI_ENABLE_COMPREHENSION=1" in install_body
    assert "context_tokens=65536" in install_body
    assert "OMNI_COMPREHENSION_CONTEXT_TOKENS=%s" in install_body
    assert "if [[ $PROFILE == ornith15 ]] && is_tegra" in install_body
    assert "cache_type=q8_0" in install_body
    assert "OMNI_COMPREHENSION_CACHE_TYPE_K=%s" in install_body
    assert "OMNI_COMPREHENSION_CACHE_TYPE_V=%s" in install_body
    assert "background_residency_mode=action" in install_body
    assert "!/^OMNI_BACKGROUND_RESIDENCY_MODE=/" in install_body
    assert "OMNI_BACKGROUND_RESIDENCY_MODE=%s" in install_body
    assert "OMNI_VIRTUAL_CONTEXT_MODE=active" in install_body
    assert "OMNI_VIRTUAL_CONTEXT_PHYSICAL_TOKENS=%s" in install_body
    assert "OMNI_VIRTUAL_CONTEXT_TOKENIZE_URL=http://127.0.0.1:8901/tokenize" in install_body
    assert "OMNI_VIRTUAL_CONTEXT_RECURRENT_TOKENS=512" in install_body
    assert "OMNI_VIRTUAL_CONTEXT_RECURRENT_SOURCE_CHUNKS=200" in install_body
    assert "OMNI_STARTUP_SMOKE=0" in install_body
    assert "OMNI_TTS_PERSISTENT=1" in install_body
    assert "co_resident_stack == 1" not in source


def test_indicator_service_starts_before_the_core_readiness_wait() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    deploy_body = source.split("deploy_service() {", 1)[1].split(
        "\n}\n\nwhile (($#))", 1
    )[0]

    assert deploy_body.index(
        "systemctl --user start omni-call-harness.service"
    ) < deploy_body.index("wait_for_service")


def test_tegra_desktop_install_whitelists_only_the_read_only_lock_query() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    rule = (
        REPO_ROOT / "services/linux/49-qwen-omni-package-lock-query.pkla"
    ).read_text(encoding="utf-8")

    assert "install_desktop_package_lock_policy" in source
    assert "com.ubuntu.update-notifier.pkexec.package-system-locked" in rule
    assert "Identity=unix-group:sudo" in rule
    assert "ResultAny=yes" in rule
    assert "apt-get" not in rule
    assert "update-manager" not in rule


def test_readiness_wait_accepts_activation_and_prints_the_journal_on_failure() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    harness_wait = source.split("wait_for_harness() {", 1)[1].split(
        "\n}\n\nwait_for_service()", 1
    )[0]

    assert "active|activating|reloading" in source
    assert "NRestarts" in source
    assert 'journalctl -u "$SERVICE_NAME"' in source
    assert "restarts -ge" not in harness_wait
    assert "Only the deadline is terminal" in harness_wait


def _run_readiness_wait(tmp_path: Path, *, broker: bool, direct_pid: int = 456):
    state_dir = tmp_path / "runtime-data" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "daemon-status.json").write_text(
        '{"state":"ready","model":"robit/test:q4km","updated_at":101,"pid":'
        + str(direct_pid) + '}', encoding="utf-8",
    )
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    source = DEPLOY.read_text(encoding="utf-8")
    wait = "wait_for_service() {" + source.split("wait_for_service() {", 1)[1].split(
        "\n}\n\nwarn()", 1
    )[0] + "\n}\n"
    script = r'''
set -e
REPO_ROOT=$1
SERVICE_NAME=qwen-omni-adapters.service
OMNI_MODEL=robit/test:q4km
SERVICE_START_EPOCH=100
configured_value() { return 0; }
sudo() {
  case $5 in
    ActiveState) printf 'active\n' ;;
    SubState) printf 'running\n' ;;
    NRestarts) printf '0\n' ;;
    MainPID) printf '456\n' ;;
    *) exit 90 ;;
  esac
}
handoff_command() {
  [[ $TEST_BROKER == 1 ]] || return 1
  printf '{"state":"ready","mode":"broker","model":"robit/test:q4km","pid":456}\n'
}
sleep() { printf 'still-pending\n'; exit 23; }
report_service_failure() { printf 'failed\n'; }
die() { exit 24; }
'''
    return subprocess.run(
        ["bash", "-c", script + wait + '\nwait_for_service\nprintf "mode=%s\\n" "$SERVICE_READY_MODE"\n',
         "readiness-test", str(tmp_path)],
        env={**os.environ, "TEST_BROKER": str(int(broker)), "OMNI_PORTAL_PORT": "8920"},
        text=True, capture_output=True, timeout=10, check=False,
    )


def test_guided_readiness_accepts_broker_supervisor_without_direct_status(tmp_path: Path) -> None:
    completed = _run_readiness_wait(tmp_path, broker=True, direct_pid=999)

    assert completed.returncode == 0, completed.stderr
    assert "mode=broker" in completed.stdout


def test_guided_readiness_preserves_direct_daemon_status(tmp_path: Path) -> None:
    completed = _run_readiness_wait(tmp_path, broker=False)

    assert completed.returncode == 0, completed.stderr
    assert "mode=direct" in completed.stdout


def test_guided_readiness_rejects_another_direct_daemons_status(tmp_path: Path) -> None:
    completed = _run_readiness_wait(tmp_path, broker=False, direct_pid=999)

    assert completed.returncode == 23, completed.stderr
    assert "still-pending" in completed.stdout


def test_guided_deploy_prints_broker_proof_without_direct_daemon_control() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    final_status = source.split("  printf '\\nDeployment ready.\\n'", 1)[1].split(
        "  DEPLOY_COMPLETE=1", 1
    )[0]

    assert '"$SERVICE_READY_STATUS"' in final_status
    assert "qwen-omni-daemon" not in final_status  # direct status includes its access credential
    assert "--status" not in final_status  # launcher status would print the access URL


@pytest.mark.parametrize("current_model,keys", [
    ("", b"\x1b[B\r"),
    ("robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km", b"\x1b[B\x1b[B\r"),
])
def test_arrow_key_menu_selects_qwen_bridge(tmp_path: Path, current_model: str, keys: bytes) -> None:
    # Isolate saved configuration; exercising zero-index wrap catches errexit
    # failures independently of the developer machine's currently chosen model.
    deploy = tmp_path / "deploy.sh"
    deploy.write_bytes(DEPLOY.read_bytes())
    deploy.chmod(0o755)
    (tmp_path / ".env").write_text(f"OMNI_MODEL={current_model}\n")
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            str(deploy),
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
                    os.write(master, keys)
                    selected_model = True
                    output.clear()
                elif not selected_service and b"Select the service deployment" in output:
                    os.write(master, b"\r")
                    selected_service = True
            if process.poll() is not None:
                break
        process.wait(timeout=5)
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
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
    assert "Services:   core daemon + always-listening harness" in rendered
    assert "scripts/bootstrap.sh --refresh-models --with-harness" in rendered


def test_automatic_updates_are_on_by_default_and_opt_out_revokes_them() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    sudo = source.split("ensure_unattended_sudo() {", 1)[1].split("\n}\n", 1)[0]
    assert "select_menu" not in sudo  # no consent prompt; enabled by default
    assert 'sudo install -m 0440 "$rule" "$AUTO_UPDATE_SUDOERS"' in sudo
    opt_out = source.split("install_auto_update() {", 1)[1].split("\n}\n", 1)[0]
    assert "systemctl --user disable --now omni-auto-update.timer" in opt_out
    assert 'sudo rm -f "$AUTO_UPDATE_SUDOERS"' in opt_out
    assert "WITH_AUTO_UPDATE=1\n" in source


def test_passwordless_sudo_is_configured_before_setup_steps() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    main_flow = source.split("\nconfirm_plan\n", 1)[1]
    assert main_flow.index("ensure_unattended_sudo") < main_flow.index("install_deploy_prerequisites")


def test_services_prefer_a_bundled_capable_ffmpeg() -> None:
    bootstrap = (REPO_ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "ffmpeg-n8.1-latest-${ffmpeg_arch}-gpl-8.1.tar.xz" in bootstrap
    assert "ffmpeg_is_capable" in bootstrap
    for unit in ("qwen-omni-adapters.service.in", "omni-call-harness.service.in"):
        text = (REPO_ROOT / "services" / "linux" / unit).read_text(encoding="utf-8")
        assert "Environment=PATH=@REPO_ROOT@/vendor/ffmpeg/bin:" in text, unit
