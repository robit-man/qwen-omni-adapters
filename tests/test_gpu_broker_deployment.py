"""Exercise the real probe and shell entry points without changing host services."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)


def host(tmp_path: Path, state: str) -> dict[str, str]:
    # Only the installation-marker lookup is isolated from the development
    # machine. The actual subprocess probe, parser and shell paths execute.
    executable(tmp_path / ".venv/bin/python", f"#!{sys.executable}\n" + f"""
import os
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from qwen_omni_adapters import gpu_broker
assert sys.argv[1:3] == ['-m', 'qwen_omni_adapters.gpu_broker']
gpu_broker._installation_marker = lambda: '/fixture/docker-gpu' if os.environ['TEST_STATE'] == 'broken-plugin' else ''
if os.environ['TEST_STATE'] == 'no-docker':
    gpu_broker.shutil.which = lambda _: None
raise SystemExit(gpu_broker.main(sys.argv[3:]))
""")
    executable(tmp_path / ".venv/bin/qwen-omni-daemon", "#!/bin/sh\nexit 91\n")
    executable(tmp_path / "bin/docker", f"#!{sys.executable}\n" + """
import os
import sys
assert sys.argv[1:] == ['gpu', 'discover']
state = os.environ['TEST_STATE']
if state == 'healthy':
    print('{"available":true,"active_leases":[{"owner":"neighbor"}],"warnings":["coordinate"]}')
elif state == 'malformed':
    print('invalid broker JSON')
elif state in ('missing-plugin', 'broken-plugin'):
    print("docker: unknown command: docker gpu\\n\\nRun 'docker --help' for more information", file=sys.stderr)
    sys.exit(1)
else:
    print('GPU broker socket permission denied', file=sys.stderr)
    sys.exit(1)
""")
    executable(tmp_path / "bin/sudo", f"#!{sys.executable}\n" + """
import os
import sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['TEST_ROOT'])
with (root / 'sudo.log').open('a') as log:
    log.write(' '.join(args) + '\\n')
if args[:3] == ['install', '-m', '0644'] and args[4:] == ['/etc/systemd/system/qwen-omni-adapters.service']:
    (root / 'rendered.service').write_bytes(Path(args[3]).read_bytes())
elif args != ['systemctl', 'daemon-reload']:
    raise SystemExit('Unexpected privileged action')
""")
    (tmp_path / "scratch").mkdir()
    return {
        **os.environ, "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "TEST_ROOT": str(tmp_path), "TEST_STATE": state, "TMPDIR": str(tmp_path / "scratch"),
    }


@pytest.mark.parametrize("state,option,mode", [
    ("missing-plugin", "--auto", "direct"),
    ("no-docker", "--auto", "direct"),
    ("missing-plugin", "--direct", "direct"),
    ("healthy", "--auto", "broker"),
    ("healthy", "--broker", "broker"),
    ("healthy", "--direct", None),
    ("missing-plugin", "--broker", None),
    ("broken-plugin", "--auto", None),
    ("unavailable", "--auto", None),
    ("unavailable", "--direct", None),
    ("malformed", "--auto", None),
])
def test_linux_installer_uses_safe_lifecycle(tmp_path, state, option, mode):
    env = host(tmp_path, state)
    installer = tmp_path / "services/linux/install.sh"
    installer.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "services/linux/install.sh", installer)
    shutil.copyfile(ROOT / "services/linux/qwen-omni-adapters.service.in", installer.with_name(
        "qwen-omni-adapters.service.in"
    ))
    result = subprocess.run(
        ["bash", str(installer), option, "--no-enable"], env=env,
        text=True, capture_output=True, timeout=10, check=False,
    )
    if mode is None:
        assert result.returncode != 0, result.stdout
        assert not (tmp_path / "rendered.service").exists()
        assert not (tmp_path / "runtime-data").exists()
        assert not (tmp_path / "sudo.log").exists()
        return
    assert result.returncode == 0, result.stderr
    rendered = (tmp_path / "rendered.service").read_text()
    if mode == "direct":
        assert f"ExecStart={tmp_path}/.venv/bin/qwen-omni-daemon serve --allow-direct-gpu" in rendered
    else:
        assert f"ExecStart={tmp_path}/portal/start.sh --foreground" in rendered
    assert f"({mode} mode)" in result.stdout
    assert "systemctl start" not in (tmp_path / "sudo.log").read_text()


@pytest.mark.parametrize("output", ["", "unknown", "direct\nunexpected"])
def test_installer_rejects_an_invalid_probe_mode_before_mutation(tmp_path, output):
    env = host(tmp_path, "missing-plugin")
    executable(tmp_path / ".venv/bin/python", f"#!{sys.executable}\nprint({output!r})\n")
    installer = tmp_path / "services/linux/install.sh"
    installer.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "services/linux/install.sh", installer)
    result = subprocess.run(
        ["bash", str(installer), "--no-enable"], env=env,
        text=True, capture_output=True, timeout=10, check=False,
    )
    assert result.returncode != 0 and "Invalid GPU lifecycle probe result" in result.stderr
    assert not (tmp_path / "runtime-data").exists()
    assert not (tmp_path / "sudo.log").exists()


@pytest.mark.parametrize("state,can_inspect", [
    ("missing-plugin", True), ("no-docker", True), ("healthy", True),
    ("broken-plugin", False), ("unavailable", False), ("malformed", False),
])
def test_handoff_probes_before_touching_live_runtime(tmp_path, state, can_inspect):
    env = host(tmp_path, state)
    function = "prepare_runtime_handoff() {" + (ROOT / "deploy.sh").read_text().split(
        "prepare_runtime_handoff() {", 1
    )[1].split("\n}\n", 1)[0] + "\n}\n"
    script = r'''
set -Eeuo pipefail
REPO_ROOT=$1
DEPLOYMENT_PORTS=8892,8901,8910,8920,8930,8940
die() { printf 'refused: %s\n' "$*" >&2; exit 64; }
handoff_command() {
  [[ $1 == inventory ]] || exit 92
  printf 'reached-read-only-inventory\n'
  exit 73
}
sudo() { exit 93; }
systemctl() { exit 94; }
'''
    result = subprocess.run(
        ["bash", "-c", script + function + "\nprepare_runtime_handoff\n", "handoff-test", str(tmp_path)],
        env=env, text=True, capture_output=True, timeout=10, check=False,
    )
    assert result.returncode == (73 if can_inspect else 64), result.stderr
    assert ("reached-read-only-inventory" in result.stdout) == can_inspect
    if not can_inspect:
        assert not list((tmp_path / "scratch").iterdir())
    if state == "healthy":
        assert '"owner":"neighbor"' in result.stdout and '"coordinate"' in result.stdout
