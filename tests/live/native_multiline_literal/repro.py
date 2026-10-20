"""Reproduce #5: multiline type_text into an owned standalone gedit/GtkSourceView with auto-indent.

Usage: .venv/bin/python -I tests/live/native_multiline_literal/repro.py "$PWD"
Readback: AT-SPI value plus the file the editor saves (Ctrl+S) for independent bytes.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path[:0] = [sys.argv[1], sys.argv[1] + "/src"]
from portal import native_ui  # noqa: E402

TEXT = "def f(x):\n    if x:\n        return 1\n    return 2\n\nclass A:\n    pass\n"


def main():
    directory = tempfile.mkdtemp(prefix="omni-multiline-")
    path = os.path.join(directory, "literal.py")
    process = subprocess.Popen(["gedit", "--standalone", "--new-window", path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            found = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)],
                                   capture_output=True, text=True, check=False).stdout.split()
            if found:
                break
            time.sleep(0.1)
        xid = str(int(found[-1]))
        subprocess.run(["xdotool", "windowactivate", "--sync", xid], check=True)
        time.sleep(1.5)
        session = "live-multiline"
        adapter = native_ui.NativeUiAdapter(session, process_id=process.pid)
        adapter.set_window_scope({"schema": "robit.omni.native-task-scope.v1", "windows": [
            {"window_id": xid, "process_id": process.pid, "process_start": native_ui._process_start(process.pid)}]})
        state = adapter.observe()
        node = next(item for item in state["nodes"] if item.get("focused") and "type_text" in item["operations"])
        action = {"session_id": session, "kind": "action", "id": "type", "node_id": node["id"],
                  "generation": node["generation"], "scope_id": state["scope"]["id"], "revision": state["revision"],
                  "verb": "type_text", "args": {"text": TEXT}, "expect": [], "idempotency_key": "multiline"}
        result = adapter.perform(action)
        time.sleep(1)
        after = adapter.observe()
        value = next(item.get("value") for item in after["nodes"] if item["id"] == node["id"])
        save = next(item for item in after["nodes"] if item["id"] == node["id"])
        adapter.perform({**action, "id": "save", "verb": "key_chord", "args": {"keys": ["ctrl", "s"]},
                         "generation": save["generation"], "revision": after["revision"], "idempotency_key": "save"})
        time.sleep(1)
        saved = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                saved = handle.read()
        adapter.close()
        print(json.dumps({"result": result, "intended_len": len(TEXT), "atspi_len": len(value or ""),
                          "atspi_exact": value == TEXT, "saved_len": None if saved is None else len(saved),
                          "saved_exact": saved == TEXT, "atspi_value": value, "saved": saved,
                          "focused_node": {k: node.get(k) for k in ("role", "text_insertion", "text_observation")}},
                         indent=1))
    finally:
        process.terminate()
        process.wait(10)


main()
