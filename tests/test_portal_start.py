from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_shutdown_finishes_after_pending_signal_and_non_cuda_child_failure(tmp_path: Path) -> None:
    source = Path("portal/start.sh").read_text()
    cleanup = "cleanup() {" + source.split("cleanup() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    evidence = tmp_path / "cleanup.log"
    script = r'''
set -e
CLEANING_UP=0
TUNNEL_PID=tunnel
SMOKE_PID= PORTAL_PID= ADAPTER_PID= POINTING_PID= TTS_PID= LAYA_PID=
COMP_PID=worker HEARTBEAT_PID=heartbeat
LEASE_TOKEN=private-test-lease
TTS_ACTIVE_PID_FILE= SUPERVISOR_PID_FILE= ACCESS_URL_FILE= TOKEN_FILE= GPU_FILE= GPU_LEASE_FILE=
log() { :; }
terminate_child() {
  [[ -n $1 ]] || return 0
  printf 'stopping:%s\n' "$1" >> "$EVIDENCE"
  if [[ $1 == tunnel ]]; then
    sleep 0.2
    return 143
  fi
}
terminate_pid_file() { :; }
terminate_process_group() { printf 'stopped:%s\n' "$1" >> "$EVIDENCE"; }
docker() { printf 'lease:%s\n' "$2" >> "$EVIDENCE"; }
cleanup_session_logs() { :; }
cleanup_cache() { :; }
'''
    script += cleanup + r'''
trap cleanup EXIT
trap 'cleanup; exit 0' TERM
printf 'ready\n' >> "$EVIDENCE"
while true; do sleep 0.05; done
'''
    process = subprocess.Popen(["bash", "-c", script], env={**os.environ, "EVIDENCE": str(evidence)})
    try:
        deadline = time.monotonic() + 3
        while not evidence.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        while "stopping:tunnel" not in evidence.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert evidence.read_text().splitlines() == [
        "ready", "stopping:tunnel", "stopping:worker", "stopped:heartbeat", "lease:release"]


@pytest.mark.skipif(shutil.which("jq") is None, reason="broker launcher requires jq")
@pytest.mark.parametrize("duration", [None, "7200"])
def test_broker_launcher_supplies_required_lease_metadata(tmp_path: Path, duration: str | None) -> None:
    portal = tmp_path / "portal"
    portal.mkdir()
    launcher = portal / "start.sh"
    shutil.copy2(Path("portal/start.sh"), launcher)
    cache = tmp_path / "components"
    cache.mkdir()
    for name in ("comprehension-model", "comprehension-projector", "tts-model", "tts-projector"):
        (cache / f"{name}.gguf").touch()

    # Exercise the real startup path with a broker that enforces its CLI
    # contract, then refuses admission so no CUDA worker can be launched.
    driver = tmp_path / "fake-host"
    driver.write_text(
        f"#!{sys.executable}\n"
        """import argparse
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
if name in {"ss", "ollama"}:
    sys.exit(0)
if name == "python":
    if sys.argv[1:2] == ["-"]:
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    cache = Path(os.environ["OMNI_COMPONENT_CACHE"])
    print(json.dumps({
        "profile": "trained-audio-bridge",
        "standard_layers": {
            "language_model": {"path": str(cache / "comprehension-model.gguf")},
            "projector": {"path": str(cache / "comprehension-projector.gguf")},
        },
    }))
    sys.exit(0)
if name == "docker" and sys.argv[1] == "gpu":
    action = sys.argv[2]
    if action == "discover":
        print(json.dumps({"gpus": [{
            "uuid": "GPU-test", "selected_for_ollama": True,
            "total_mib": 81920, "free_mib": 80000,
        }]}))
        sys.exit(0)
    if action == "status":
        print('{"leases": []}')
        sys.exit(0)
    if action == "acquire":
        parser = argparse.ArgumentParser()
        parser.add_argument("--owner", required=True)
        parser.add_argument("--justification", required=True)
        parser.add_argument("--expected-duration", required=True, type=float)
        parser.add_argument("--vram-mib", required=True, type=int)
        parser.add_argument("--gpu", required=True)
        parser.add_argument("--token-only", action="store_true")
        args = parser.parse_args(sys.argv[3:])
        Path(os.environ["LEASE_REQUEST"]).write_text(json.dumps(vars(args)))
        sys.exit(73)
raise SystemExit("unexpected host operation: " + repr(sys.argv))
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("docker", "ss", "ollama", "nvidia-smi", "cloudflared"):
        (fake_bin / name).symlink_to(driver)
    for name in (".venv/bin/python", "vendor/llama.cpp/build/bin/llama-server",
                 "vendor/llama.cpp/build/bin/llama-tts"):
        executable = tmp_path / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.symlink_to(driver)
    request_file = tmp_path / "lease-request.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("OMNI_")}
    env.update({
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "OMNI_MODEL": "test/omni-bridge:q4km",
        "OMNI_COMPONENT_CACHE": str(cache),
        "OMNI_DECISION_PLANE_ENABLED": "0",
        "LEASE_REQUEST": str(request_file),
    })
    if duration is not None:
        env["OMNI_GPU_LEASE_EXPECTED_DURATION"] = duration
        env["OMNI_COMPREHENSION_GPU_UUID"] = "GPU-test"
    completed = subprocess.run(
        ["bash", str(launcher), "--foreground"],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )

    assert completed.returncode == 73, completed.stdout + completed.stderr
    request = json.loads(request_file.read_text(encoding="utf-8"))
    assert request["owner"] == "robit-omni-phone-portal"
    assert "test/omni-bridge:q4km" in request["justification"]
    assert request["expected_duration"] == float(duration or "86400")
    assert request["vram_mib"] == 45000
    assert request["gpu"] == "GPU-test"
    assert request["token_only"] is True
    assert "starting CUDA comprehension" not in completed.stdout


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_broker_pointing_child_inherits_exact_scope_and_portal_url(tmp_path: Path, enabled: str) -> None:
    worker = tmp_path / "point-python"
    record = tmp_path / "worker.json"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import json, os\nfrom pathlib import Path\n"
        "Path(os.environ['POINT_WORKER_RECORD']).write_text(json.dumps({key: os.environ.get(key) "
        "for key in ('CUDA_VISIBLE_DEVICES', 'OLLAMA_UNIFY_GPU_LEASE', 'OMNI_POINTING_GPU_UUID', "
        "'OMNI_POINTING_HOST', 'OMNI_POINTING_PORT')}))\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    source = Path("portal/start.sh").read_text(encoding="utf-8")
    block = source.split('  local pointing_url=""', 1)[1].split(
        '  log "starting broker-coordinated CUDA TTS wrapper"', 1
    )[0]
    script = r'''
set -eu
REPO_ROOT=$1
LOG_DIR=$1
POINTING_PYTHON=$2
POINTING_PORT=8940
gpu_uuid=GPU-reserved
LEASE_TOKEN=private-test-lease
log() { :; }
wait_http() { wait "$4"; }
start_pointing() {
  local pointing_url=""
'''
    completed = subprocess.run(
        ["bash", "-c", script + block + '\nprintf "portal-pointing-url=%s\\n" "$pointing_url"\n}\nstart_pointing\n',
         "pointing-launch-test", str(tmp_path), str(worker)],
        env={**os.environ, "POINTING_ENABLED": enabled, "POINT_WORKER_RECORD": str(record)},
        text=True, capture_output=True, check=False, timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "private-test-lease" not in completed.stdout + completed.stderr
    if enabled == "1":
        assert json.loads(record.read_text()) == {
            "CUDA_VISIBLE_DEVICES": "GPU-reserved",
            "OLLAMA_UNIFY_GPU_LEASE": "private-test-lease",
            "OMNI_POINTING_GPU_UUID": "GPU-reserved",
            "OMNI_POINTING_HOST": "127.0.0.1",
            "OMNI_POINTING_PORT": "8940",
        }
        assert "portal-pointing-url=http://127.0.0.1:8940" in completed.stdout
    else:
        assert not record.exists()
        assert "portal-pointing-url=\n" in completed.stdout
    portal_start = source.split('  log "starting authenticated phone portal"', 1)[1]
    assert 'OMNI_POINTING_URL="$pointing_url"' in portal_start


def test_broker_cleanup_stops_pointing_before_releasing_gpu_scope(tmp_path: Path) -> None:
    source = Path("portal/start.sh").read_text(encoding="utf-8")
    cleanup = "cleanup() {" + source.split("cleanup() {", 1)[1].split(
        "\n}\n\nservice_is_running()", 1
    )[0] + "\n}\n"
    script = r'''
set -eu
CLEANING_UP=0
POINTING_PID=1234
TUNNEL_PID= SMOKE_PID= PORTAL_PID= ADAPTER_PID= TTS_PID= LAYA_PID= COMP_PID= HEARTBEAT_PID=
LEASE_TOKEN=private-test-lease
TTS_ACTIVE_PID_FILE=$1/missing
SUPERVISOR_PID_FILE=$1/missing ACCESS_URL_FILE=$1/missing TOKEN_FILE=$1/missing
GPU_FILE=$1/missing GPU_LEASE_FILE=$1/missing
terminate_child() { [[ -z $1 ]] || printf 'stop %s\n' "$2"; return 0; }
terminate_pid_file() { :; }
terminate_process_group() { :; }
cleanup_session_logs() { :; }
cleanup_cache() { :; }
log() { printf '%s\n' "$*"; }
docker() { [[ $1 == gpu && $2 == release ]]; }
'''
    completed = subprocess.run(
        ["bash", "-c", script + cleanup + "\ncleanup\n", "pointing-cleanup-test", str(tmp_path)],
        text=True, capture_output=True, check=False, timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.index("stop CUDA pointing worker") < completed.stdout.index("releasing scoped GPU lease")


def test_pointing_bootstrap_provisions_x86_without_loading_cuda(tmp_path: Path) -> None:
    driver = tmp_path / "python"
    record = tmp_path / "bootstrap.jsonl"
    driver.write_text(
        f"#!{sys.executable}\n"
        """import json, os, sys
from pathlib import Path
if Path(sys.argv[0]).name == 'uname':
    print('Linux' if sys.argv[-1] == '-s' else 'x86_64')
    raise SystemExit(0)
with Path(os.environ['BOOTSTRAP_RECORD']).open('a') as output:
    output.write(json.dumps({'args': sys.argv[1:], 'stdin': sys.stdin.read() if sys.argv[1:2] == ['-'] else ''}) + '\\n')
if sys.argv[1:3] == ['-m', 'venv']:
    python = Path(sys.argv[3]) / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(__file__).resolve())
elif sys.argv[1:2] == ['-c']:
    print('3.12')
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    (tmp_path / "uname").symlink_to(driver)
    completed = subprocess.run(
        ["bash", "scripts/bootstrap_pointing.sh"],
        env={
            **os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "PYTHON": str(driver), "OMNI_POINTING_VENV": str(tmp_path / "point-venv"),
            "BOOTSTRAP_RECORD": str(record),
        },
        text=True, capture_output=True, check=False, timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert any(call["args"] == ["-m", "pip", "install", "torch"] for call in calls)
    download = next(call for call in calls if "snapshot_download" in call["stdin"])
    assert download["args"] == ["-", "vikhyatk/moondream2", "9a7d4024050840e001defacec2b00727e89149e6"]
    assert not any("from_pretrained" in call["stdin"] or "cuda.is_available" in call["stdin"] for call in calls)


@pytest.mark.skipif(shutil.which("jq") is None, reason="broker profile resolver requires jq")
@pytest.mark.parametrize("logical_tag,profile,language_default,overrides", [
    ("robit/ornith-1.5-omni-audio-bridge:q4km", "trained-audio-bridge", "1", {}),
    ("robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km", "trained-audio-bridge", "0", {}),
    ("robit/ornith-1.5-omni:q4km", "legacy-cascade", "1", {}),
    ("robit/qwen3.8-27b-e03-obliterated-omni:q4km", "legacy-cascade", "0", {}),
    ("test/unknown-ornith-bridge:q4km", "trained-audio-bridge", "0", {}),
    ("robit/ornith-1.5-omni-audio-bridge:q4km", "trained-audio-bridge", "1", {
        "OMNI_COMPREHENSION_DISABLE_THINKING": "0", "OMNI_COMPREHENSION_REPEAT_PENALTY": "1.7",
        "OMNI_LANGUAGE_DISABLE_THINKING": "0"}),
    ("robit/qwen3.8-27b-e03-obliterated-omni:q4km", "legacy-cascade", "0", {
        "OMNI_COMPREHENSION_DISABLE_THINKING": "1", "OMNI_COMPREHENSION_REPEAT_PENALTY": "1.4",
        "OMNI_LANGUAGE_DISABLE_THINKING": "1"}),
    ("robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km", "trained-audio-bridge", "0", {
        "OMNI_LANGUAGE_DISABLE_THINKING": "1"}),
    ("robit/ornith-1.5-omni-audio-bridge:q4km", "trained-audio-bridge", "1", {
        "OMNI_LANGUAGE_DISABLE_THINKING": ""}),
    ("robit/ornith-1.5-omni-audio-bridge:q4km", "trained-audio-bridge", "1", {
        "OMNI_LANGUAGE_LLAMA_CONTEXT_PROBE": "0"}),
])
def test_broker_resolved_profile_reaches_actual_adapter_environment_and_audio_chat_prompt(
    tmp_path: Path, logical_tag: str, profile: str, language_default: str, overrides: dict[str, str],
) -> None:
    """Execute the real resolver/child assignment and adapter payload builder.

    Only sidecar filesystem resolution and the long-lived child process are
    substituted. The actual catalog, environment parsing, and audio-chat
    comprehension contract run without any CUDA imports or host services.
    """
    source = Path("portal/start.sh").read_text()
    library = tmp_path / "portal" / "start-library.sh"
    library.parent.mkdir()
    library.write_text(source.rsplit("\ncase ${1:---daemon} in", 1)[0] + "\n")
    cache = tmp_path / "components"
    cache.mkdir()
    for name in ("comprehension-model", "comprehension-projector"):
        (cache / f"{name}.gguf").touch()
    driver = tmp_path / "python"
    record = tmp_path / "adapter-environment.json"
    driver.write_text(f"#!{sys.executable}\n" + r'''
import base64
import io
import json
import os
from pathlib import Path
import sys
import wave

if sys.argv[1:4] == ['-m', 'qwen_omni_adapters', 'resolve']:
    cache = Path(os.environ['OMNI_COMPONENT_CACHE'])
    print(json.dumps({'profile': os.environ['FIXTURE_PROFILE'], 'standard_layers': {
        'language_model': {'path': str(cache / 'comprehension-model.gguf')},
        'projector': {'path': str(cache / 'comprehension-projector.gguf')},
    }}))
elif sys.argv[1:2] == ['-']:
    # Execute the launcher's actual MODEL_BY_TAG lookup, not a fixture boolean.
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
elif sys.argv[1].endswith('/runtime/adapter_server.py'):
    from qwen_omni_adapters.contract import ADAPTER_SCHEMA, parse_adapter_request
    from runtime.adapter_server import Config, TRAINED_AUDIO_SYSTEM_PROMPT, build_comprehension_payload
    config = Config.from_environment()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\x01\x00' * 160)
    parsed = parse_adapter_request({
        'model': os.environ['OMNI_MODEL'], 'stream': False, 'think': False,
        'omni': {'schema': ADAPTER_SCHEMA, 'task': 'chat'},
        'messages': [{'role': 'user', 'content': 'Process the attached spoken request.',
                      'audios': [{'data': base64.b64encode(buffer.getvalue()).decode()}]}],
    })
    payload = build_comprehension_payload(parsed, config)
    names = ('OMNI_COMPREHENSION_MODEL', 'OMNI_COMPREHENSION_DISABLE_THINKING',
             'OMNI_COMPREHENSION_REPEAT_PENALTY', 'OMNI_LANGUAGE_MODEL',
             'OMNI_LANGUAGE_DISABLE_THINKING', 'OMNI_LANGUAGE_API',
             'OMNI_LANGUAGE_LLAMA_CONTEXT_PROBE')
    Path(os.environ['ADAPTER_ENV_RECORD']).write_text(json.dumps({
        'environment': {name: os.environ[name] for name in names},
        'config': {'comprehension_disable_thinking': config.comprehension_disable_thinking,
                   'comprehension_repeat_penalty': config.comprehension_repeat_penalty,
                   'language_disable_thinking': config.language_disable_thinking},
        'audio_chat_payload': {'trained_prompt': payload['messages'][0]['content'] == TRAINED_AUDIO_SYSTEM_PROMPT,
                               'chat_template_kwargs': payload.get('chat_template_kwargs'),
                               'repeat_penalty': payload.get('repeat_penalty', 1.0),
                               'cache_prompt': payload['cache_prompt']},
        'gpu_modules_imported': [name for name in ('torch', 'moondream', 'kestrel') if name in sys.modules],
    }))
else:
    raise SystemExit('Unexpected fixture interpreter operation')
''')
    driver.chmod(0o755)
    block = source.split('  log "starting unified Omni adapter"', 1)[1].split(
        '  local portal_token=', 1)[0]
    script = r'''
set -Eeuo pipefail
source "$1"
mkdir -p "$LOG_DIR"
wait_http() { wait "$4"; }
launch_adapter_fixture() {
  local decision_ready=0
'''
    script += block + "\n}\nresolve_runtime_profile\nlaunch_adapter_fixture\n"
    root = Path.cwd()
    env = {key: value for key, value in os.environ.items() if not key.startswith("OMNI_")}
    env.update({"OMNI_MODEL": logical_tag, "OMNI_PYTHON_BIN": str(driver),
                "OMNI_COMPONENT_CACHE": str(cache), "FIXTURE_PROFILE": profile,
                "ADAPTER_ENV_RECORD": str(record), "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}",
                **overrides})
    completed = subprocess.run(["bash", "-c", script, "broker-adapter-profile-test", str(library)],
                               env=env, text=True, capture_output=True, check=False, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr + (
        (tmp_path / "runtime-data/logs/adapter.log").read_text()
        if (tmp_path / "runtime-data/logs/adapter.log").exists() else "")
    evidence = json.loads(record.read_text())
    expected_comp = overrides.get("OMNI_COMPREHENSION_DISABLE_THINKING", "1" if profile == "trained-audio-bridge" else "0")
    expected_repeat = float(overrides.get("OMNI_COMPREHENSION_REPEAT_PENALTY", "1.1" if profile == "trained-audio-bridge" else "1.0"))
    expected_language = overrides.get("OMNI_LANGUAGE_DISABLE_THINKING", language_default)
    assert evidence["environment"]["OMNI_COMPREHENSION_DISABLE_THINKING"] == expected_comp
    assert evidence["environment"]["OMNI_LANGUAGE_DISABLE_THINKING"] == expected_language
    assert evidence["environment"]["OMNI_LANGUAGE_LLAMA_CONTEXT_PROBE"] == overrides.get(
        "OMNI_LANGUAGE_LLAMA_CONTEXT_PROBE", "1" if profile == "trained-audio-bridge" else "0")
    assert evidence["config"] == {"comprehension_disable_thinking": expected_comp == "1",
                                  "comprehension_repeat_penalty": expected_repeat,
                                  "language_disable_thinking": expected_language == "1"}
    assert evidence["audio_chat_payload"] == {
        "trained_prompt": expected_comp == "1", "repeat_penalty": expected_repeat,
        "chat_template_kwargs": {"enable_thinking": False} if expected_comp == "1" else None,
        "cache_prompt": False,
    }
    assert evidence["gpu_modules_imported"] == []
    if profile == "trained-audio-bridge":
        assert evidence["environment"]["OMNI_LANGUAGE_MODEL"] == "local-audio-bridge"
        assert evidence["environment"]["OMNI_LANGUAGE_API"] == "openai"
