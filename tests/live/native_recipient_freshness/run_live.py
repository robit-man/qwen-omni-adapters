"""Live #3 comparison on an owned GTK3 TextView through the real native worker.

Unrelated overlay drift must admit focused input exactly once; a changed private
caret or changed recipient geometry must still reject before input.

Usage: .venv/bin/python -I tests/live/native_recipient_freshness/run_live.py "$PWD"
(requires an X11 desktop with AT-SPI, xdotool and /usr/bin/python3 GTK3 bindings).
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

sys.path[:0] = [sys.argv[1], sys.argv[1] + "/src"]
from portal import native_ui  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL = tempfile.mkdtemp(prefix="recipient-ctl-")
SEQUENCE = [0]


def command(value):
    SEQUENCE[0] += 1
    name = f"{SEQUENCE[0]:03d}"
    with open(os.path.join(CONTROL, "cmd-" + name), "w") as handle:
        handle.write(value)
    path = os.path.join(CONTROL, "ack-" + name)
    for _ in range(100):
        if os.path.exists(path):
            with open(path) as handle:
                return ast.literal_eval(handle.read())
        time.sleep(0.05)
    raise SystemExit("fixture did not acknowledge " + value)


def window_of(process, title):
    for _ in range(100):
        found = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid), "--name", title],
                               capture_output=True, text=True, check=False).stdout.split()
        if found:
            return str(int(found[0]))
        time.sleep(0.1)
    raise SystemExit("fixture window did not appear")


def arm(name, verb, args, drift, expected):
    title = f"Omni Recipient Fixture {uuid.uuid4().hex[:8]}"
    process = subprocess.Popen(["/usr/bin/python3", os.path.join(HERE, "fixture_app.py"), CONTROL, title])
    try:
        xid = window_of(process, title)
        subprocess.run(["xdotool", "windowactivate", "--sync", xid], check=True)
        time.sleep(0.5)
        app_before = command("dump")
        session = f"live-recipient-{name}"
        adapter = native_ui.NativeUiAdapter(session, process_id=process.pid)
        adapter.set_window_scope({"schema": "robit.omni.native-task-scope.v1", "windows": [
            {"window_id": xid, "process_id": process.pid, "process_start": native_ui._process_start(process.pid)}]})
        state = adapter.observe()
        node = next(item for item in state["nodes"]
                    if item.get("focused") and verb in item["operations"] and item["role"] == "textbox")
        action = {"session_id": session, "kind": "action", "id": "live", "node_id": node["id"],
                  "generation": node["generation"], "scope_id": state["scope"]["id"], "revision": state["revision"],
                  "verb": verb, "args": args, "expect": [], "idempotency_key": "live-" + name}
        for value in drift:
            command(value)
        time.sleep(0.3)
        result = adapter.perform(action)
        time.sleep(0.5)
        app_after = command("dump")
        after = adapter.observe()
        adapter.close()
        later = next((item for item in after["nodes"] if item["id"] == node["id"]), {})
        return {"arm": name, "verb": verb, "args": args, "drift_cmds": drift, "issued_revision": state["revision"],
                "result": result, "app_before": app_before, "app_after": app_after,
                "atspi_readback": later.get("value"), "expected_status": expected,
                "public_node_diff": {key: (node.get(key), later.get(key)) for key in set(node) | set(later)
                                     if node.get(key) != later.get(key)},
                "status_ok": result["status"] == expected}
    finally:
        process.terminate()
        process.wait(5)
        absent = subprocess.run(["xdotool", "search", "--pid", str(process.pid)],
                                capture_output=True, text=True, check=False).stdout.strip() == ""
        print(json.dumps({"cleanup_window_absent": absent, "pid": process.pid}), flush=True)


results = [
    arm("drift_type", "type_text", {"text": "[x"}, ["drift"], "dispatched"),
    arm("drift_shift_return", "key_chord", {"keys": ["shift", "Return"]}, ["drift"], "dispatched"),
    arm("drift_caret_changed", "type_text", {"text": "[x"}, ["drift", "caret0"], "rejected"),
    arm("drift_recipient_resized", "type_text", {"text": "[x"}, ["drift_resize"], "rejected"),
    arm("no_drift_control", "type_text", {"text": "[x"}, [], "dispatched"),
]
print(json.dumps(results, indent=1))
