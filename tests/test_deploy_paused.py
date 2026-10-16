"""Execute the installer function with private files/locks and mocked services."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy.sh"


def deployment(tmp_path, *, paused=True, prior_active=False, core="inactive", fault="", marker_kind="regular",
               control_umask=None):
    state = tmp_path / "runtime-data/state"
    state.mkdir(parents=True)
    marker = state / "voice-service.paused"
    if paused and control_umask is None:
        if marker_kind == "directory":
            marker.mkdir()
        elif marker_kind == "symlink":
            target = state / "other"
            target.write_text("not a pause marker")
            marker.symlink_to(target)
        else:
            marker.write_text("Paused by operator; resume with qwen-omni service resume.\n")
            marker.chmod(0o644 if marker_kind == "mode" else 0o600)
    lock = state / "voice-service-control.lock"
    if control_umask is not None:
        subprocess.run([sys.executable, "-c", '''
import os
import sys
from pathlib import Path
import qwen_omni_adapters.service_control as control_module
from qwen_omni_adapters.service_control import VoiceServiceControl
assert Path(control_module.__file__).resolve() == Path(os.environ["TEST_ISOLATE"]) / "src/qwen_omni_adapters/service_control.py"
os.umask(int(sys.argv[2]))
control = VoiceServiceControl(Path(sys.argv[1]))
def simulated_systemctl(action):
    assert action == "stop"
def forbidden_doctor():
    raise AssertionError("Private pause creation must not run deployment doctor")
control._systemctl = simulated_systemctl
control._doctor = forbidden_doctor
assert control.set_paused(True) == (True, "Voice paused")
''', str(tmp_path), str(control_umask)], check=True, timeout=10,
            env={**os.environ, "PYTHONPATH": str(DEPLOY.parent / "src") + os.pathsep + str(DEPLOY.parent),
                 "TEST_ISOLATE": str(DEPLOY.parent)})
    elif fault != "missing-lock":
        if fault == "lock-directory":
            lock.mkdir()
        elif fault == "lock-symlink":
            target = state / "other-lock"
            target.write_bytes(b"")
            target.chmod(0o600)
            lock.symlink_to(target)
        else:
            lock.write_bytes(b"")
            lock.chmod(0o666 if fault == "lock-world-write" else 0o400 if fault == "lock-readonly" else 0o600)
    if paused and fault == "read-atime":
        os.utime(marker, ns=(1, marker.stat().st_mtime_ns))
    before = marker.stat() if marker.exists() else None
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    commands = tmp_path / "commands.log"
    (tmp_path / "harness-state").write_text("active" if prior_active else "inactive")
    for name, body in (
        (".venv/bin/qwen-omni", 'printf "doctor\\n" >>"$TEST_LOG"\n[[ $TEST_FAULT != doctor ]] || exit 43\n'),
        ("scripts/validate.sh", 'printf "validate\\n" >>"$TEST_LOG"\n[[ $TEST_FAULT != validate ]] || exit 44\n'),
        ("services/linux/install.sh", 'printf "install\\n" >>"$TEST_LOG"\n'
         '[[ $TEST_FAULT != install ]] || exit 41\n'
         'if [[ $TEST_FAULT == marker-drift ]]; then printf changed >>"$TEST_ROOT/runtime-data/state/voice-service.paused"; fi\n'
         'if [[ $TEST_FAULT == marker-remove ]]; then rm -- "$TEST_ROOT/runtime-data/state/voice-service.paused"; fi\n'
         'if [[ $TEST_FAULT == marker-replace ]]; then cp -- "$TEST_ROOT/runtime-data/state/voice-service.paused" "$TEST_ROOT/replacement"; mv -- "$TEST_ROOT/replacement" "$TEST_ROOT/runtime-data/state/voice-service.paused"; fi\n'
         'if [[ $TEST_FAULT == lock-replace ]]; then mv -- "$TEST_ROOT/runtime-data/state/voice-service-control.lock" "$TEST_ROOT/old-lock"; : >"$TEST_ROOT/runtime-data/state/voice-service-control.lock"; chmod 600 "$TEST_ROOT/runtime-data/state/voice-service-control.lock"; fi\n'
         'if [[ $TEST_FAULT == unexpected-harness ]]; then printf active >"$TEST_ROOT/harness-state"; fi\n'),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -e\n" + body)
        path.chmod(0o755)
    install = tmp_path / "services/linux/install.sh"
    with install.open("a") as target:
        target.write(r'''
if [[ $TEST_FAULT == read-atime ]]; then cat "$TEST_ROOT/runtime-data/state/voice-service.paused" >/dev/null; fi
if [[ $TEST_FAULT == shared-control ]]; then
  "$TEST_ROOT/.venv/bin/python" - <<'PYCONTROL'
import os
from pathlib import Path
import qwen_omni_adapters.service_control as control_module
from qwen_omni_adapters.service_control import VoiceServiceControl
assert Path(control_module.__file__).resolve() == Path(os.environ["TEST_ISOLATE"]) / "src/qwen_omni_adapters/service_control.py"
control = VoiceServiceControl(Path(os.environ["TEST_ROOT"]))
def forbidden(*args):
    raise AssertionError("Private control must not reach doctor or systemctl while locked")
control._doctor = control._systemctl = forbidden
for paused in (False, True):
    result = control.set_paused(paused)
    assert result[0] is False and control.paused()
with Path(os.environ["TEST_LOG"]).open("a") as log:
    log.write("shared-control-busy\n")
PYCONTROL
fi
''')
    functions = DEPLOY.read_text().split("\nwhile (($#)); do\n", 1)[0]
    script = tmp_path / "functions.sh"
    script.write_text(functions + r'''
PROFILE=ornith15
OMNI_MODEL=robit/test:q4km
OMNI_LANGUAGE_MODEL=$OMNI_MODEL
WITH_HARNESS=1
HARNESS_UNIT=$REPO_ROOT/owned-harness.service
is_tegra() { return 1; }
prepare_runtime_handoff() {
  printf 'handoff\n' >>"$TEST_LOG"
  HANDOFF_STARTED=1
  if [[ $TEST_PRIOR_ACTIVE == 1 ]]; then STOPPED_USER_UNITS=("$HARNESS_NAME"); fi
  if [[ $TEST_FAULT == retire-harness-state ]]; then LEGACY_USER_UNITS=(egg-omni-test.service); fi
  printf inactive >"$TEST_ROOT/harness-state"
  [[ $TEST_FAULT != handoff ]] || return 42
}
id() {
  if [[ $1 == -u ]]; then
    if [[ $TEST_FAULT == owner ]]; then printf '%s\n' "$((TEST_UID + 1))"; else printf '%s\n' "$TEST_UID"; fi
  else command id "$@"; fi
}
sudo() {
  printf 'sudo %s\n' "$*" >>"$TEST_LOG"
  if [[ $1 == systemctl && $2 == show ]]; then
    case $5 in
      ActiveState) printf '%s\n' "$TEST_CORE" ;;
      SubState) if [[ $TEST_CORE == inactive ]]; then printf 'dead\n'; elif [[ $TEST_CORE == active ]]; then printf 'running\n'; else printf 'failed\n'; fi ;;
      NRestarts) printf '0\n' ;;
      MainPID) if [[ $TEST_CORE == active ]]; then printf '999\n'; else printf '0\n'; fi ;;
      *) return 90 ;;
    esac
  elif [[ $1 == test ]]; then return 1
  elif [[ $1 == systemctl ]]; then return 0
  else return 91; fi
}
systemctl() {
  printf 'user %s\n' "$*" >>"$TEST_LOG"
  if [[ $* == *is-enabled* ]]; then return 1; fi
  if [[ $* == *is-active* ]]; then [[ $(cat "$TEST_ROOT/harness-state") == active ]]; return; fi
  if [[ $2 == show ]]; then cat "$TEST_ROOT/harness-state"; return 0; fi
  if [[ $2 == stop ]]; then printf inactive >"$TEST_ROOT/harness-state"; fi
  if [[ $2 == disable && $TEST_FAULT == retire-harness-state ]]; then
    if [[ $TEST_PRIOR_ACTIVE == 1 ]]; then printf inactive >"$TEST_ROOT/harness-state"; else printf active >"$TEST_ROOT/harness-state"; fi
  fi
  if [[ $2 == start ]]; then
    [[ $TEST_FAULT != harness-start ]] || return 45
    if [[ $TEST_FAULT != false-harness ]]; then printf active >"$TEST_ROOT/harness-state"; fi
  fi
}
flock() {
  command flock "$@" || return $?
  if [[ $TEST_FAULT == opened-lock-replace ]]; then
    mv -- "$TEST_ROOT/runtime-data/state/voice-service-control.lock" "$TEST_ROOT/old-lock"
    : >"$TEST_ROOT/runtime-data/state/voice-service-control.lock"
    chmod 600 "$TEST_ROOT/runtime-data/state/voice-service-control.lock"
  fi
}
handoff_command() { printf 'health\n' >>"$TEST_LOG"; return 1; }
sleep() { SECONDS=$((SECONDS + 61)); }
report_service_failure() { printf 'service-failure\n' >>"$TEST_LOG"; }
wait_for_harness() { printf 'listening-wait\n' >>"$TEST_LOG"; }
deploy_service
''')
    holder = None
    if fault == "busy-lock":
        holder = lock.open("rb")
        import fcntl
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = subprocess.run(["bash", str(script)], cwd=tmp_path, text=True, capture_output=True, timeout=10,
            env={**os.environ, "TEST_LOG": str(commands), "TEST_ROOT": str(tmp_path),
                 "PYTHONPATH": str(DEPLOY.parent / "src") + os.pathsep + str(DEPLOY.parent),
                 "TEST_ISOLATE": str(DEPLOY.parent),
                 "TEST_UID": str(os.getuid()), "TEST_CORE": core, "TEST_FAULT": fault,
                 "TEST_PRIOR_ACTIVE": str(int(prior_active))})
    finally:
        if holder:
            holder.close()
    log = commands.read_text().splitlines() if commands.exists() else []
    return result, log, marker, lock, before


@pytest.mark.parametrize("prior_active", [False, True])
def test_valid_paused_install_is_held_without_core_start_or_health(tmp_path, prior_active):
    result, log, marker, lock, before = deployment(tmp_path, prior_active=prior_active)
    assert result.returncode == 0, result.stderr
    assert "HELD_PAUSED" in result.stdout and "Deployment ready." not in result.stdout
    assert {"doctor", "validate", "handoff", "install"} <= set(log)
    assert not any("systemctl start qwen-omni" in line or line == "health" or line == "listening-wait" for line in log)
    assert sum(line == "user --user start omni-call-harness.service" for line in log) == int(prior_active)
    after = marker.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_size, before.st_mode) == (
        after.st_ino, after.st_mtime_ns, after.st_size, after.st_mode)
    assert marker.read_text() == "Paused by operator; resume with qwen-omni service resume.\n"
    assert lock.read_bytes() == b""
    import fcntl
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_absent_pause_preserves_unpaused_inactive_failure(tmp_path):
    result, log, *_ = deployment(tmp_path, paused=False)
    assert result.returncode != 0 and "HELD_PAUSED" not in result.stdout
    assert "sudo systemctl start qwen-omni-adapters.service" in log
    assert "service-failure" in log


@pytest.mark.parametrize("fault,kind,core", [
    ("owner", "regular", "inactive"), ("", "symlink", "inactive"),
    ("", "directory", "inactive"), ("", "mode", "inactive"),
    ("missing-lock", "regular", "inactive"), ("busy-lock", "regular", "inactive"),
    ("marker-drift", "regular", "inactive"), ("", "regular", "active"),
    ("", "regular", "failed"), ("", "regular", "unknown"),
    ("install", "regular", "inactive"), ("handoff", "regular", "inactive"),
    ("doctor", "regular", "inactive"), ("validate", "regular", "inactive"),
    ("marker-remove", "regular", "inactive"), ("marker-replace", "regular", "inactive"),
    ("lock-replace", "regular", "inactive"), ("opened-lock-replace", "regular", "inactive"),
    ("unexpected-harness", "regular", "inactive"),
    ("lock-symlink", "regular", "inactive"), ("lock-directory", "regular", "inactive"),
    ("lock-world-write", "regular", "inactive"), ("lock-readonly", "regular", "inactive"),
])
def test_uncertain_paused_install_never_becomes_held(tmp_path, fault, kind, core):
    result, log, *_ = deployment(tmp_path, fault=fault, marker_kind=kind, core=core)
    assert result.returncode != 0 and "HELD_PAUSED" not in result.stdout
    assert not any("systemctl start qwen-omni" in line for line in log)
    if core != "inactive":
        assert "handoff" not in log and "install" not in log


@pytest.mark.parametrize("fault", ["harness-start", "false-harness"])
def test_failed_prior_active_harness_restore_does_not_claim_preservation(tmp_path, fault):
    result, log, *_ = deployment(tmp_path, prior_active=True, fault=fault)
    assert result.returncode != 0 and "HELD_PAUSED" not in result.stdout
    assert not any("systemctl start qwen-omni" in line for line in log)


@pytest.mark.parametrize("prior_active", [False, True])
def test_harness_activity_is_verified_after_prior_unit_retirement(tmp_path, prior_active):
    result, log, *_ = deployment(tmp_path, prior_active=prior_active, fault="retire-harness-state")
    assert result.returncode != 0 and "HELD_PAUSED" not in result.stdout
    assert "prior harness activity was not preserved" in result.stderr
    assert "user --user disable egg-omni-test.service" in log


@pytest.mark.parametrize("fault", ["install", "handoff"])
def test_existing_rollback_runs_for_failure_after_handoff_without_clearing_pause(tmp_path, fault):
    result, log, marker, lock, before = deployment(tmp_path, prior_active=True, fault=fault)
    assert result.returncode != 0 and "restoring the prior configuration" in result.stderr
    assert "sudo systemctl stop qwen-omni-adapters.service" in log
    assert "sudo systemctl daemon-reload" in log
    assert "user --user start omni-call-harness.service" in log
    assert not (tmp_path / ".env").exists()
    assert marker.stat().st_ino == before.st_ino and marker.stat().st_mtime_ns == before.st_mtime_ns
    assert lock.read_bytes() == b""


def test_reading_marker_access_time_does_not_invalidate_its_immutable_binding(tmp_path):
    result, _, marker, _, before = deployment(tmp_path, fault="read-atime")
    assert result.returncode == 0, result.stderr
    assert marker.stat().st_atime_ns != before.st_atime_ns
    assert marker.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("umask,expected_mode", [(0o002, 0o664), (0o022, 0o644), (0o077, 0o600)])
def test_existing_api_normal_umask_lock_preserves_paused_install(tmp_path, umask, expected_mode):
    result, log, marker, lock, before = deployment(tmp_path, control_umask=umask)
    assert lock.stat().st_mode & 0o7777 == expected_mode
    assert result.returncode == 0, result.stderr
    assert "HELD_PAUSED" in result.stdout
    assert not any("systemctl start qwen-omni" in line or line == "health" or line == "listening-wait" for line in log)
    assert marker.stat().st_ino == before.st_ino and marker.stat().st_mtime_ns == before.st_mtime_ns
    assert marker.stat().st_mode & 0o7777 == 0o600
    import fcntl
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("control_umask", [None, 0o002])
def test_existing_pause_and_resume_api_observe_the_same_actual_kernel_lock(tmp_path, control_umask):
    result, log, _, lock, _ = deployment(tmp_path, fault="shared-control", control_umask=control_umask)
    assert lock.stat().st_mode & 0o7777 == (0o600 if control_umask is None else 0o664)
    assert result.returncode == 0, result.stderr
    assert "shared-control-busy" in log
