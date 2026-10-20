"""Live #5 check: multiline type_text is literal in autoindenting and plain editors.

Generated path: ComputerUseStore.task_scope -> immediate_transition -> compiler ->
controller -> NativeUiAdapter -> AT-SPI worker -> EditableText literal insertion.
gedit (GtkSourceView, user auto-indent on) also saves its buffer for independent bytes.
A GTK4 TextView (gtk4_textview.py) exposes EditableText too, but on this Egg's window
manager its client-side-decorated window has no frame extents, so the worker refuses
observation (native_window_frame_extents_unavailable) before any input; tracked separately.

Usage: .venv/bin/python -I tests/live/native_multiline_literal/run_live.py "$PWD"
"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "def f(x):\n    if x:\n        return 1\n    return 2\n\nclass A:\n    pass\n"


def window(process, title=None):
    for _ in range(150):
        argv = ["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)] + (["--name", title] if title else [])
        found = subprocess.run(argv, capture_output=True, text=True, check=False).stdout.split()
        if found:
            return str(int(found[-1]))
        time.sleep(0.1)
    raise SystemExit("window did not appear")


def arm(name, argv, inserted, expected, *, title=None, saved_path=None):
    process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        xid = window(process, title)
        subprocess.run(["xdotool", "windowactivate", "--sync", xid], check=True)
        time.sleep(1.5)
        session = "live-literal-" + name
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
                goal="Write the requested literal text into the owned document", objective="Insert the text")
        after = adapter.observe()
        value = next(item.get("value") for item in after["nodes"] if item["id"] == node["id"])
        saved = None
        if saved_path:
            current = next(item for item in after["nodes"] if item["id"] == node["id"])
            adapter.perform({"session_id": session, "kind": "action", "id": "save", "node_id": node["id"],
                             "generation": current["generation"], "scope_id": after["scope"]["id"],
                             "revision": after["revision"], "verb": "key_chord", "args": {"keys": ["ctrl", "s"]},
                             "expect": [], "idempotency_key": "save"})
            time.sleep(1)
            with open(saved_path, encoding="utf-8") as handle:
                saved = handle.read()
        adapter.close()
        contract = result.get("immediate_effect_contract", {})
        receipt = (result.get("outcomes") or [{}])[-1].get("receipt", {})
        return {"arm": name, "initial": node.get("value"), "observed_insertion": node.get("text_insertion"),
                "inserted": inserted, "expected": expected, "compiled_predicates": contract.get("predicates"),
                "status": result.get("status"), "outcome_status": (result.get("outcomes") or [{}])[-1].get("status"),
                "backend": receipt.get("backend"), "atspi_after": value, "saved": saved,
                # gedit (ensure-trailing-newline) strips one final newline on load
                # and appends one on save; the buffer itself must be exact.
                "ok": value == expected and (saved_path is None or saved == expected + "\n")
                      and contract.get("predicates") == [{"kind": "value", "node_id": "$target", "value": expected}]}
    finally:
        process.terminate()
        process.wait(10)
        absent = subprocess.run(["xdotool", "search", "--pid", str(process.pid)],
                                capture_output=True, text=True, check=False).stdout.strip() == ""
        print(json.dumps({"cleanup_window_absent": absent, "pid": process.pid}), flush=True)


def gedit(name, initial, inserted, expected):
    path = os.path.join(tempfile.mkdtemp(prefix="omni-literal-"), "literal.py")
    if initial is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(initial)
    return arm(name, ["gedit", "--standalone", "--new-window", path], inserted, expected, saved_path=path)


def gtk3(name, inserted, expected):
    # The owned GTK3 TextView fixture starts with "alpha" and the caret at its end.
    title = "Omni Literal " + uuid.uuid4().hex[:8]
    app = os.path.join(HERE, "..", "native_recipient_freshness", "fixture_app.py")
    return arm(name, ["/usr/bin/python3", app, tempfile.mkdtemp(prefix="omni-literal-ctl-"), title],
               inserted, expected, title=title)


results = [
    gedit("gedit_autoindent_empty", None, CODE, CODE),
    gedit("gedit_autoindent_prefix", "# header\n", "    a = [\n        1,\n    ]\n", "    a = [\n        1,\n    ]\n# header"),
    gedit("gedit_unicode", None, "café:\n    中 → é\n\tx\n", "café:\n    中 → é\n\tx\n"),
    gtk3("gtk3_textview_end", "\n  beta\n    γ", "alpha\n  beta\n    γ"),
]
print(json.dumps(results, indent=1, ensure_ascii=False))
