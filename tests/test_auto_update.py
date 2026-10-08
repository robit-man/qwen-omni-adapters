from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATER = REPO_ROOT / "scripts" / "auto_update.sh"


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _publish(origin_work: Path, marker: str) -> str:
    (origin_work / "marker.txt").write_text(marker)
    _git(origin_work, "add", "-A")
    _git(origin_work, "commit", "-qm", marker)
    _git(origin_work, "push", "-q", "origin", "main")
    return _git(origin_work, "rev-parse", "HEAD")


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    work = tmp_path / "origin-work"
    _git(tmp_path, "clone", "-q", str(bare), str(work))
    (work / "scripts").mkdir()
    shutil.copy2(UPDATER, work / "scripts" / "auto_update.sh")
    (work / ".gitignore").write_text(".env\n.deployed-commit\n.venv/\n")
    _publish(work, "first")

    # A tarball install: the published files without any git metadata.
    install = tmp_path / "install"
    shutil.copytree(work, install, ignore=shutil.ignore_patterns(".git"))
    (install / ".env").write_text("OMNI_PROFILE=ornith15\n")
    (install / ".venv").mkdir()
    (install / ".venv" / "keep").write_text("runtime state")

    calls = tmp_path / "deploy-calls.txt"
    fake_deploy = tmp_path / "fake-deploy.sh"
    fake_deploy.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >>{calls}\n'
        'exit "${FAKE_DEPLOY_STATUS:-0}"\n'
    )
    fake_deploy.chmod(0o755)
    env = {
        **os.environ,
        "OMNI_UPDATE_REPO": str(bare),
        "OMNI_UPDATE_DEPLOY": str(fake_deploy),
        "OMNI_UPDATE_SKIP_SUDO_CHECK": "1",
        "XDG_RUNTIME_DIR": str(tmp_path),
    }
    return install, work, calls, env


def _update(install: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(install / "scripts" / "auto_update.sh")],
        cwd=install,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_tarball_install_converts_to_git_and_redeploys_the_published_head(tmp_path: Path) -> None:
    install, work, calls, env = _setup(tmp_path)
    head = _publish(work, "second")

    completed = _update(install, env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (install / ".git").is_dir()
    assert (install / "marker.txt").read_text() == "second"
    assert (install / ".venv" / "keep").read_text() == "runtime state"
    assert (install / ".deployed-commit").read_text().strip() == head
    deploy_args = calls.read_text().split()
    assert deploy_args[:6] == ["--profile", "ornith15", "--action", "deploy", "--yes", "--no-update"]
    assert deploy_args[6:] in (["--with-harness"], ["--no-harness"])

    # Nothing new on main: no redeploy.
    calls.unlink()
    assert _update(install, env).returncode == 0
    assert not calls.exists()


def test_failed_redeploy_is_retried_and_local_edits_block_updates(tmp_path: Path) -> None:
    install, work, calls, env = _setup(tmp_path)
    head = _publish(work, "second")

    failed = _update(install, {**env, "FAKE_DEPLOY_STATUS": "1"})
    assert failed.returncode != 0
    assert not (install / ".deployed-commit").exists()

    retried = _update(install, env)
    assert retried.returncode == 0, retried.stdout + retried.stderr
    assert (install / ".deployed-commit").read_text().strip() == head
    assert len(calls.read_text().splitlines()) == 2

    _publish(work, "third")
    (install / "marker.txt").write_text("local edit")
    blocked = _update(install, env)
    assert blocked.returncode != 0
    assert "local changes" in blocked.stdout
    assert (install / "marker.txt").read_text() == "local edit"


def test_camera_stack_change_reinstalls_it_without_an_omni_redeploy(tmp_path: Path) -> None:
    install, work, calls, env = _setup(tmp_path)
    assert _update(install, env).returncode == 0  # first deploy of the Omni runtime
    calls.unlink()

    camera_bare = tmp_path / "camera.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(camera_bare))
    camera_work = tmp_path / "camera-work"
    _git(tmp_path, "clone", "-q", str(camera_bare), str(camera_work))
    installs = tmp_path / "camera-installs.txt"
    installer = camera_work / "install.sh"
    installer.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >>{installs}\n')
    installer.chmod(0o755)
    _publish(camera_work, "camera-1")
    _git(tmp_path, "clone", "-q", str(camera_bare), str(install / "vendor" / "jetson-ecam-gmsl"))
    camera_env = {**env, "OMNI_UPDATE_SUDO": "env"}

    assert _update(install, camera_env).returncode == 0
    assert not installs.exists()  # camera checkout already current

    head = _publish(camera_work, "camera-2")
    completed = _update(install, camera_env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert installs.read_text().split() == ["--yes", "--if-needed"]
    assert _git(install / "vendor" / "jetson-ecam-gmsl", "rev-parse", "HEAD") == head
    assert not calls.exists()  # the Omni runtime was not redeployed
