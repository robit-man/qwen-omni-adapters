"""Live #4 check: generated typing compiles exact nonempty insertion from the observed caret.

Drives ComputerUseStore.task_scope -> immediate_transition -> compiler -> controller ->
NativeUiAdapter -> real AT-SPI worker -> XTEST against an owned GTK3 TextView, with no
hand-written expected predicate. The fixture's own buffer readback is independent evidence.

Usage: .venv/bin/python -I tests/live/native_text_insertion/run_live.py "$PWD"
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
from portal.computer_use import ComputerUseStore  # noqa: E402

APP = os.path.join(sys.argv[1], "tests/live/native_recipient_freshness/fixture_app.py")
CONTROL = tempfile.mkdtemp(prefix="insertion-ctl-")
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


def arm(name, text, setup, inserted, expected):
    title = f"Omni Insertion Fixture {uuid.uuid4().hex[:8]}"
    process = subprocess.Popen(["/usr/bin/python3", APP, CONTROL, title])
    try:
        for _ in range(100):
            found = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid), "--name", title],
                                   capture_output=True, text=True, check=False).stdout.split()
            if found:
                break
            time.sleep(0.1)
        xid = str(int(found[0]))
        subprocess.run(["xdotool", "windowactivate", "--sync", xid], check=True)
        time.sleep(0.5)
        command("text:" + text)
        app_before = [command(value) for value in setup][-1]
        session = f"live-insertion-{name}"
        adapter = native_ui.NativeUiAdapter(session, process_id=process.pid)
        adapter.set_window_scope({"schema": "robit.omni.native-task-scope.v1", "windows": [
            {"window_id": xid, "process_id": process.pid, "process_start": native_ui._process_start(process.pid)}]})
        state = adapter.observe()
        node = next(item for item in state["nodes"] if item.get("focused") and "type_text" in item["operations"])
        store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
        cancel = store.begin_task(session, private_boundaries=True)
        with store.task_scope(session, "desktop", cancel):
            result = store.immediate_transition(session, {"id": name, "verb": "type_text",
                "target": {"node_id": node["id"]}, "args": {"text": inserted}},
                goal="Edit the owned fixture document", objective="Insert the requested text")
        time.sleep(0.3)
        app_after = command("dump")
        adapter.close()
        contract = result.get("immediate_effect_contract", {})
        return {"arm": name, "initial": text, "setup": setup, "inserted": inserted,
                "observed_insertion": node.get("text_insertion"), "observed_selection": node.get("text_selection"),
                "compiled_mode": contract.get("mode"), "compiled_predicates": contract.get("predicates"),
                "status": result.get("status"),
                "outcome_status": (result.get("outcomes") or [{}])[-1].get("status"),
                "app_before": app_before, "app_after": app_after, "expected": expected,
                "ok": app_after[0] == expected and contract.get("predicates") == [
                    {"kind": "value", "node_id": "$target", "value": expected}]}
    finally:
        process.terminate()
        process.wait(5)
        absent = subprocess.run(["xdotool", "search", "--pid", str(process.pid)],
                                capture_output=True, text=True, check=False).stdout.strip() == ""
        print(json.dumps({"cleanup_window_absent": absent, "pid": process.pid}), flush=True)


results = [
    arm("middle", "Hello world", ["caret6"], "brave ", "Hello brave world"),
    arm("end", "Hello world", ["caret11"], "!", "Hello world!"),
    arm("unicode_middle", "café 中 é", ["caret5"], "Ελ", "café Ελ中 é"),
    arm("selection_replace", "Hello world", ["select:6:11"], "there", "Hello there"),
]
print(json.dumps(results, indent=1, ensure_ascii=False))
