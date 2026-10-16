from __future__ import annotations

import copy
from types import SimpleNamespace

import httpx
import pytest
from test_native_ui import Fixture, Node

from portal.native_ui import NativeUiAdapter, NativeUiUnavailable, _x11_frame_extents, _x11_geometry
from qwen_omni_adapters.computer_use import RapidUiController, parse_predicates, parse_step

_REPARENTED_GEOMETRY = '''
xwininfo: Window id: 0x11 "Owned window"
  Absolute upper-left X:  44
  Absolute upper-left Y:  69
  Relative upper-left X:  14
  Relative upper-left Y:  49
  Width: 880
  Height: 720
'''


def test_x11_geometry_uses_absolute_client_origin_without_adding_reparent_offsets(monkeypatch) -> None:
    def run(command, **kwargs):
        assert command == ["xwininfo", "-id", "17"]
        assert kwargs["env"]["LC_ALL"] == "C"
        return SimpleNamespace(returncode=0, stdout=_REPARENTED_GEOMETRY)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert _x11_geometry({"window_id": "17"}) == {"x": 44, "y": 69, "width": 880, "height": 720}


@pytest.mark.parametrize("bad_geometry", [
    _REPARENTED_GEOMETRY.replace("0x11", "0x12"),
    _REPARENTED_GEOMETRY + "  Absolute upper-left X: 900\n",
    _REPARENTED_GEOMETRY.replace("Width: 880", "Width: 0"),
    _REPARENTED_GEOMETRY.replace("Absolute upper-left Y:  69", "Absolute upper-left Y: unknown"),
])
def test_x11_geometry_rejects_different_identity_ambiguous_or_invalid_extents(monkeypatch, bad_geometry) -> None:
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=bad_geometry))
    with pytest.raises(NativeUiUnavailable):
        _x11_geometry({"window_id": "17"})


def test_x11_frame_margins_are_read_only_for_exact_client_and_keep_asymmetric_offsets(monkeypatch) -> None:
    def run(command, **kwargs):
        assert command == ["xprop", "-id", "17", "_GTK_FRAME_EXTENTS"]
        assert kwargs["env"]["LC_ALL"] == "C"
        return SimpleNamespace(returncode=0, stdout="_GTK_FRAME_EXTENTS(CARDINAL) = 61, 61, 55, 67\n")
    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert _x11_frame_extents({"window_id": "17"}) == {"left": 61, "right": 61, "top": 55, "bottom": 67}


@pytest.mark.parametrize("value", ["_GTK_FRAME_EXTENTS: not found.", "_GTK_FRAME_EXTENTS(CARDINAL) = -1, 0, 0, 0",
    "_GTK_FRAME_EXTENTS(CARDINAL) = 4097, 0, 0, 0", "_GTK_FRAME_EXTENTS(CARDINAL) = 1, 2, 3",
    "_GTK_FRAME_EXTENTS(CARDINAL) = 1.5, 2, 3, 4", "_GTK_FRAME_EXTENTS(CARDINAL) = １, 2, 3, 4",
    "_GTK_FRAME_EXTENTS(CARDINAL) = 1, 2, 3, 4\n_GTK_FRAME_EXTENTS(CARDINAL) = 1, 2, 3, 4"])
def test_unknown_ambiguous_negative_or_unbounded_frame_margins_never_authorize_translation(monkeypatch, value) -> None:
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=value))
    with pytest.raises(NativeUiUnavailable) as exc:
        _x11_frame_extents({"window_id": "17"})
    assert exc.value.code == "native_window_frame_extents_unavailable"


def _window_coordinate_fixture(monkeypatch):
    fixture = Fixture()
    fixture.api.CoordType = SimpleNamespace(SCREEN="screen", WINDOW="window")
    surface = {"x": 50, "y": 7, "width": 482, "height": 613}
    margins = {"left": 61, "right": 61, "top": 55, "bottom": 67}
    positions = {fixture.window: {"x": 0, "y": 0, "width": 360, "height": 491},
                 fixture.button: {"x": 16, "y": 245, "width": 64, "height": 44},
                 fixture.title: {"x": 16, "y": 389, "width": 64, "height": 44}}
    for node in positions:
        if "Component" not in node.interfaces:
            node.interfaces.append("Component")
        node.get_extents = lambda space, item=node: SimpleNamespace(**(
            positions[item] if space == "window" else {**positions[item], "x": 0, "y": 0}))
        node.get_parent = lambda: fixture.window
    label = Node("/button-label", "label", "Preview")
    label.get_parent = lambda: fixture.button
    hits = []
    def hit(x, y, space):
        hits.append((x, y, space))
        assert space == "window", "Collapsed SCREEN positions cannot prove native target ownership"
        return label
    fixture.window.get_accessible_at_point = hit
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: dict(surface))
    monkeypatch.setattr("portal.native_ui._x11_frame_extents", lambda *_args, **_kwargs: dict(margins))
    commands, location = [], {"x": 0, "y": 0}
    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "mousemove":
            location.update(x=surface["x"]+int(command[-2]), y=surface["y"]+int(command[-1]))
        stdout = f"X={location['x']}\nY={location['y']}\nWINDOW=17\n" if command[1] == "getmouselocation" else ""
        return SimpleNamespace(returncode=0, stdout=stdout)
    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    return fixture, commands, hits, positions, surface, margins, label


def test_collapsed_screen_mapping_uses_exact_csd_dimensions_native_window_hit_and_label_ancestry(monkeypatch) -> None:
    fixture, commands, hits, _positions, _surface, _margins, _label = _window_coordinate_fixture(monkeypatch)
    state = fixture.adapter.observe()
    target = next(node for node in state["nodes"] if node["name"] == "Preview")
    assert target["bounds"] == {"x": 127, "y": 307, "width": 64, "height": 44}
    assert state["coordinate_binding"]["content"] == {"x": 111, "y": 62, "width": 360, "height": 491}
    assert fixture.adapter.perform(fixture.action("Preview", "click"))["status"] == "dispatched"
    assert hits == [(48, 267, "window")]
    assert ["xdotool", "mousemove", "--sync", "--window", "17", "109", "322"] in commands
    assert commands[-1] == ["xdotool", "click", "--repeat", "1", "--delay", "70", "1"]


@pytest.mark.parametrize("invalid", ["dimension", "local-root-origin", "root-size", "unknown-margins", "foreground"])
def test_collapsed_coordinates_require_exact_current_mapping_before_any_pointer_input(monkeypatch, invalid) -> None:
    fixture, commands, _hits, positions, surface, _margins, _label = _window_coordinate_fixture(monkeypatch)
    if invalid == "dimension":
        surface["width"] += 1
    elif invalid == "local-root-origin":
        positions[fixture.window]["x"] = 1
    elif invalid == "root-size":
        positions[fixture.window]["height"] -= 1
    elif invalid == "unknown-margins":
        def unavailable(*_args, **_kwargs):
            raise NativeUiUnavailable("native_window_frame_extents_unavailable")
        monkeypatch.setattr("portal.native_ui._x11_frame_extents", unavailable)
    else:
        def changed(*_args, **_kwargs):
            fixture.foreground["window_id"] = "18"
            return {"left": 61, "right": 61, "top": 55, "bottom": 67}
        monkeypatch.setattr("portal.native_ui._x11_frame_extents", changed)
    with pytest.raises(NativeUiUnavailable):
        fixture.adapter.observe()
    assert commands == []


def test_clipped_child_withholds_only_its_pointer_geometry_and_keeps_semantics_and_siblings(monkeypatch) -> None:
    fixture, commands, _hits, positions, _surface, _margins, _label = _window_coordinate_fixture(monkeypatch)
    panel = Node("/clipped-panel", "panel", "Header decoration", interfaces=["Component", "Action"])
    panel.get_extents = lambda space: SimpleNamespace(x=-1 if space == "window" else 0, y=0, width=362, height=46)
    fixture.window.children.append(panel)
    state = fixture.adapter.observe()
    clipped = next(node for node in state["nodes"] if node["name"] == "Header decoration")
    sibling = next(node for node in state["nodes"] if node["name"] == "Preview")
    assert "bounds" not in clipped and clipped["operations"] == ["invoke"]
    assert sibling["bounds"] == {"x": 127, "y": 307, "width": 64, "height": 44}
    positions[fixture.button]["x"] = 359
    state = fixture.adapter.observe()
    clipped_target = next(node for node in state["nodes"] if node["name"] == "Preview")
    assert "bounds" not in clipped_target and "invoke" in clipped_target["operations"]
    assert not {"click", "double_click", "right_click", "drag_drop", "scroll"}.intersection(clipped_target["operations"])
    assert fixture.adapter.perform(fixture.action("Preview", "click"))["reason"] == "native_target_not_eligible"
    assert commands == []


@pytest.mark.parametrize("invalid", ["entry-margins", "final-margins", "final-surface", "target-bounds", "occluded", "protected"])
def test_native_window_coordinate_binding_is_rechecked_before_xtest_without_hit_bypass(monkeypatch, invalid) -> None:
    fixture, commands, _hits, positions, surface, margins, label = _window_coordinate_fixture(monkeypatch)
    action = fixture.action("Preview", "click")
    reads = 0
    def extents(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        if reads >= (2 if invalid == "entry-margins" else 3) and invalid in {"entry-margins", "final-margins"}:
            return {**margins, "left": 62, "right": 60}
        return dict(margins)
    monkeypatch.setattr("portal.native_ui._x11_frame_extents", extents)
    if invalid == "target-bounds":
        positions[fixture.button]["x"] += 1
    elif invalid == "final-surface":
        original = fixture.window.get_accessible_at_point
        def changed_surface(*args):
            value = original(*args)
            surface["x"] += 1
            return value
        fixture.window.get_accessible_at_point = changed_surface
    if invalid in {"occluded", "protected"}:
        label.get_parent = lambda: fixture.secret if invalid == "protected" else fixture.title
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert commands == []


def test_coordinate_dispatch_observation_budget_error_propagates_before_pointer_input(monkeypatch) -> None:
    fixture, commands, _hits, _positions, _surface, _margins, _label = _window_coordinate_fixture(monkeypatch)
    action = fixture.action("Preview", "click")
    original, calls = fixture.bridge._coordinate_for, 0
    def bounded(window):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        return original(window)
    fixture.bridge._coordinate_for = bounded
    with pytest.raises(NativeUiUnavailable) as exc:
        fixture.adapter.perform(action)
    assert exc.value.code == "native_observation_budget_exceeded"
    assert commands == []


def test_existing_screen_coordinate_path_does_not_query_or_apply_window_margins(monkeypatch) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Noncollapsed SCREEN coordinates retain the existing path")
    monkeypatch.setattr("portal.native_ui._x11_frame_extents", unexpected)
    assert fixture.adapter.perform(fixture.action("Preview", "click"))["status"] == "dispatched"
    assert commands[-1][1] == "click"


def _pointer_fixture(monkeypatch):
    fixture = Fixture()
    fixture.api.CoordType = SimpleNamespace(SCREEN="screen")
    geometry = {"x": 100, "y": 200, "width": 600, "height": 400}
    positions = {fixture.window: geometry, fixture.button: {"x": 150, "y": 250, "width": 100, "height": 60},
                 fixture.title: {"x": 350, "y": 250, "width": 180, "height": 60}}
    for node, bounds in positions.items():
        if "Component" not in node.interfaces:
            node.interfaces.append("Component")
        node.get_extents = lambda _space, rect=bounds: SimpleNamespace(**rect)
        node.get_parent = lambda: fixture.window
    fixture.window.get_accessible_at_point = lambda x, _y, _space: fixture.button if x < 300 else fixture.title
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: dict(geometry))
    commands, location = [], {"x": 0, "y": 0}

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "mousemove":
            location.update(x=100+int(command[-2]), y=200+int(command[-1]))
        stdout = f"X={location['x']}\nY={location['y']}\nWINDOW=17\n" if command[1] == "getmouselocation" else ""
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    return fixture, commands


@pytest.mark.parametrize(("verb", "button", "count"), [("click", "1", "1"), ("double_click", "1", "2"), ("right_click", "3", "1")])
def test_explicit_pointer_clicks_derive_native_geometry_and_target_one_window(monkeypatch, verb, button, count) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    action = fixture.action("Preview", verb)
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert commands[0] == ["xdotool", "getmouselocation", "--shell"]
    assert commands[1] == ["xdotool", "mousemove", "--sync", "--window", "17", "100", "80"]
    assert commands[-1] == ["xdotool", "click", "--repeat", count, "--delay", "70", button]
    assert "x" not in action["args"] and "y" not in action["args"]


@pytest.mark.parametrize(("verb", "button", "count"), [("click", "1", "1"), ("double_click", "1", "2"), ("right_click", "3", "1")])
def test_already_at_owned_target_skips_no_motion_sync_but_posts_requested_click(monkeypatch, verb, button, count) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)

    def run(command, **_kwargs):
        commands.append(command)
        assert command[1] != "mousemove", "A redundant --sync move would wait without a movement event"
        return SimpleNamespace(returncode=0, stdout="X=200\nY=280\nWINDOW=17\n")

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert fixture.adapter.perform(fixture.action("Preview", verb))["status"] == "dispatched"
    assert commands == [["xdotool", "getmouselocation", "--shell"],
                        ["xdotool", "getmouselocation", "--shell"],
                        ["xdotool", "click", "--repeat", count, "--delay", "70", button]]


@pytest.mark.parametrize("changed", ["foreground", "pointer-window", "geometry", "occluded"])
def test_already_at_target_cannot_bypass_live_pointer_binding_or_native_hit(monkeypatch, changed) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    action = fixture.action("Preview", "right_click")
    reads = 0

    def run(command, **_kwargs):
        nonlocal reads
        commands.append(command)
        if command[1] == "getmouselocation":
            reads += 1
            if changed == "foreground":
                fixture.foreground["window_id"] = "18"
            if changed == "geometry":
                monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs:
                                    {"x": 101, "y": 200, "width": 600, "height": 400})
        window = "18" if changed == "pointer-window" else "17"
        return SimpleNamespace(returncode=0, stdout=f"X=200\nY=280\nWINDOW={window}\n")

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    if changed == "occluded":
        fixture.window.get_accessible_at_point = lambda *_args: fixture.title
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert all(command[1] != "click" for command in commands)
    assert reads == 0 if changed == "occluded" else reads > 0


def test_native_pointer_refuses_occlusion_without_posting_a_click(monkeypatch) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    action = fixture.action("Preview", "double_click")
    fixture.window.get_accessible_at_point = lambda *_args: fixture.title
    assert fixture.adapter.perform(action)["reason"] == "native_pointer_target_occluded"
    assert commands == []


def test_native_pointer_descends_real_gtk_filler_hit_containers_before_target(monkeypatch) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    outer, inner = Node("/outer", "filler", interfaces=["Component"]), Node("/inner", "filler", interfaces=["Component"])
    outer.get_parent = lambda: fixture.window
    inner.get_parent = lambda: outer
    outer.get_accessible_at_point = lambda *_args: inner
    inner.get_accessible_at_point = lambda *_args: fixture.button
    fixture.window.get_accessible_at_point = lambda *_args: outer
    action = fixture.action("Preview", "click")
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert commands[-1][1] == "click"


def _notebook_fixture(monkeypatch):
    fixture, commands = _pointer_fixture(monkeypatch)
    page = Node("/selected-page", "page tab", interfaces=["Component"], children=[fixture.button])
    page.state.values.add("SELECTED")
    other = Node("/inactive-page", "page tab", interfaces=["Component"])
    notebook = Node("/notebook", "page tab list", interfaces=["Component", "Selection"], children=[page, other])
    notebook.selected = [0]
    notebook.get_parent = lambda: fixture.window
    page.get_parent = lambda: notebook
    other.get_parent = lambda: notebook
    fixture.button.get_parent = lambda: page
    fixture.window.children.remove(fixture.button)
    fixture.window.children.append(notebook)
    fixture.window.get_accessible_at_point = lambda *_args: notebook
    notebook.get_accessible_at_point = lambda *_args: None
    page.get_accessible_at_point = lambda *_args: fixture.button
    return fixture, commands, notebook, page, other


def test_gtk_notebook_missing_hit_uses_only_native_uniquely_selected_direct_page(monkeypatch) -> None:
    fixture, commands, _notebook, _page, _other = _notebook_fixture(monkeypatch)
    assert fixture.adapter.perform(fixture.action("Preview", "double_click"))["status"] == "dispatched"
    assert commands[-1] == ["xdotool", "click", "--repeat", "2", "--delay", "70", "1"]


@pytest.mark.parametrize("invalid", ["ambiguous", "inactive", "unrelated-page", "not-direct", "no-selection", "occluded", "protected"])
def test_notebook_selected_page_bridge_never_bypasses_ambiguity_inactivity_or_obstruction(monkeypatch, invalid) -> None:
    fixture, commands, notebook, page, other = _notebook_fixture(monkeypatch)
    if invalid == "ambiguous":
        notebook.selected = [0, 1]
    elif invalid == "inactive":
        page.state.values.remove("SELECTED")
    elif invalid == "unrelated-page":
        notebook.selected = [1]
        other.state.values.add("SELECTED")
    elif invalid == "not-direct":
        page.get_parent = lambda: fixture.window
    elif invalid == "no-selection":
        notebook.interfaces.remove("Selection")
    elif invalid == "occluded":
        page.get_accessible_at_point = lambda *_args: fixture.title
    else:
        page.get_accessible_at_point = lambda *_args: fixture.secret
    assert fixture.adapter.perform(fixture.action("Preview", "double_click"))["reason"] == "native_pointer_target_occluded"
    assert commands == []


def test_native_drag_and_scroll_use_observed_destination_and_bounded_input(monkeypatch) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    state = fixture.adapter.observe()
    destination = next(node for node in state["nodes"] if node["name"] == "Title")
    action = fixture.action("Preview", "drag_drop", {"to_node_id": destination["id"]})
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert [command[1] for command in commands] == ["getmouselocation", "mousemove", "getmouselocation", "mousedown", "mousemove", "getmouselocation", "mouseup"]
    assert commands[-3] == ["xdotool", "mousemove", "--sync", "--window", "17", "340", "80"]
    commands.clear()
    action = fixture.action("Preview", "scroll", {"direction": "down"})
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert commands[-1] == ["xdotool", "click", "--repeat", "3", "--delay", "30", "5"]


@pytest.mark.parametrize("changed", ["foreground", "pointer-window", "pointer-position"])
def test_guarded_xtest_posts_no_click_when_final_pointer_or_foreground_binding_changes(monkeypatch, changed) -> None:
    fixture, commands = _pointer_fixture(monkeypatch)
    action = fixture.action("Preview", "click")
    original = __import__("portal.native_ui", fromlist=["subprocess"]).subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if command[1] == "getmouselocation":
            if changed == "foreground":
                fixture.foreground["window_id"] = "18"
            elif changed == "pointer-window":
                result.stdout = result.stdout.replace("WINDOW=17", "WINDOW=18")
            else:
                result.stdout = result.stdout.replace("X=200", "X=201")
        return result

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert all(command[1] != "click" for command in commands)


def test_native_text_selection_observes_actual_character_range() -> None:
    fixture = Fixture()
    fixture.title.value = "Select these characters"
    fixture.title.selection_range = None
    text = fixture.api.Text
    text.get_n_selections = lambda node: int(node.selection_range is not None)
    text.get_selection = lambda node, _index: SimpleNamespace(start_offset=node.selection_range[0], end_offset=node.selection_range[1])
    text.add_selection = lambda node, start, end: setattr(node, "selection_range", (start, end)) or True
    text.set_selection = lambda node, _index, start, end: setattr(node, "selection_range", (start, end)) or True
    action = fixture.action("Title", "select_text", {"start": 0, "end": 6})
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    observed = next(node for node in fixture.adapter.observe()["nodes"] if node["name"] == "Title")
    assert observed["text_selection"] == {"start": 0, "end": 6}
    action = fixture.action("Title", "select_text", {"start": 0, "end": 999})
    assert fixture.adapter.perform(action)["reason"] == "native_text_selection_out_of_range"


def test_native_window_switch_uses_observed_mapping_and_verifies_active_scope(monkeypatch) -> None:
    fixture = Fixture()
    other = Node("/other-window", "frame", "Owned other window")
    other.process_id = 99
    application = Node("/other-app", "application", children=[other])
    application.process_id = 99
    fixture.desktop.children.append(application)
    fixture.bridge._mapped_window = lambda node, _pid: "21" if node is other else "17"
    monkeypatch.setattr("portal.native_ui._process_start", lambda pid: f"{pid}01")

    def run(command, **_kwargs):
        assert command == ["xdotool", "windowactivate", "--sync", "21"]
        fixture.foreground.update(window_id="21", process_id=99)
        fixture.window.state.values.remove("ACTIVE")
        other.state.values.add("ACTIVE")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    controller = RapidUiController(fixture.adapter, authorize=lambda *_args: True)
    result = controller.run_step({"id": "switch", "verb": "switch_window", "target": {"role": "window", "name": "Owned other window"},
                                  "args": {}, "expect": [{"kind": "active_window", "node_id": "$target"}]})
    assert result["status"] == "verified"
    assert result["after"]["scope"]["window_id"] == "21"


def _visual_fixture(monkeypatch, **adapter_options):
    fixture = Fixture()
    adapter = NativeUiAdapter("one", bridge_runner=fixture.request, pointing_url="http://127.0.0.1:8940",
                              pointing_model="fixture/moondream", pointing_revision="pinned", **adapter_options)
    frame = {"key": "bound-frame", "sha256": "a"*64, "image": b"fixture-png",
             "geometry": {"x": 100, "y": 200, "width": 600, "height": 400},
             "identity": dict(fixture.foreground), "environment": {}}
    monkeypatch.setattr(adapter, "_visual_capture", lambda _state: copy.deepcopy(frame))
    requests = []

    def point(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(200, json={"model": "fixture/moondream", "revision": "pinned", "points": [{"x": .25, "y": .5}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.post", point)
    adapter.prepare_step({"id": "visual", "verb": "click", "target": {"role": "button", "name": "Canvas target"},
                          "args": {}, "expect": [{"kind": "text", "text": "Done"}]})
    return fixture, adapter, frame, requests


@pytest.mark.parametrize(("options", "expected"), [({}, 60), ({"pointing_timeout_s": 90}, 90), ({"pointing_timeout_s": 999}, 120)])
def test_native_point_http_budget_is_distinct_from_bounded_accessibility_bridge(monkeypatch, options, expected) -> None:
    _fixture, adapter, _frame, requests = _visual_fixture(monkeypatch, **options)
    adapter.observe()
    assert requests[0][1]["timeout"] == expected
    assert adapter.timeout_s == 10


def test_native_visual_fallback_creates_finite_candidate_and_reuses_one_point_request(monkeypatch) -> None:
    _fixture, adapter, _frame, requests = _visual_fixture(monkeypatch)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    step = copy.deepcopy(adapter._step)
    candidate = next(action for action in controller.enumerate(step, state) if action["kind"] == "action")
    assert candidate["node_id"].startswith("visual:")
    assert controller.observe() == state
    dispatched = []
    monkeypatch.setattr("portal.native_ui._pointer_dispatch", lambda identity, geometry, point, verb, **_kwargs:
                        dispatched.append((identity, geometry, point, verb)) or {"status": "dispatched"})
    assert adapter.perform(candidate)["status"] == "dispatched"
    assert len(requests) == 1
    assert dispatched[0][2:] == ((250, 400), "click")
    assert adapter.point_receipt["frame_sha256"] == "a"*64
    assert adapter.point_receipt["model"] == "fixture/moondream"


def test_named_word_grounding_dispatches_existing_double_click_without_new_coordinates(monkeypatch) -> None:
    _fixture, adapter, _frame, requests = _visual_fixture(monkeypatch)
    adapter.prepare_step({"id": "word", "verb": "double_click", "target": {"role": "text", "name": "the spoken word at document start"},
                          "args": {}, "expect": [{"kind": "text_selection", "node_id": "document", "start": 0, "end": 5}]})
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    action = next(item for item in controller.enumerate(adapter._step, state) if item["kind"] == "action")
    assert action["args"] == {} and action["node_id"].startswith("visual:")
    dispatched = []
    monkeypatch.setattr("portal.native_ui._pointer_dispatch", lambda _identity, _geometry, _point, verb, **_kwargs:
                        dispatched.append(verb) or {"status": "dispatched"})
    assert adapter.perform(action)["status"] == "dispatched"
    assert dispatched == ["double_click"] and len(requests) == 1


def test_native_visual_stale_frame_and_hidden_semantic_targets_cannot_dispatch(monkeypatch) -> None:
    fixture, adapter, frame, requests = _visual_fixture(monkeypatch)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    action = next(item for item in controller.enumerate(adapter._step, state) if item["kind"] == "action")
    frame["key"] = "changed-pixels"
    assert adapter.perform(action)["reason"] == "native_visual_frame_changed"
    assert len(requests) == 1
    fixture.button.state.values.remove("SHOWING")
    adapter.prepare_step({"id": "protected", "verb": "click", "target": {"name": "Preview"},
                          "args": {}, "expect": [{"kind": "text", "text": "Done"}]})
    assert not any(node["id"].startswith("visual:") for node in adapter.observe()["nodes"])
    assert len(requests) == 1


@pytest.mark.parametrize("args", [{"start": -1, "end": 2}, {"start": 2, "end": 2}, {"start": True, "end": 3}, {"start": 0, "end": 1_000_001}])
def test_character_selection_contract_rejects_invalid_ranges(args) -> None:
    with pytest.raises(ValueError):
        parse_step({"id": "select", "verb": "select_text", "target": {"name": "Document"}, "args": args,
                    "expect": [{"kind": "text_selection", "node_id": "$target", **args}]})


def test_public_predicate_validator_supports_active_window_and_observed_selection() -> None:
    predicates = [{"kind": "active_window", "node_id": "window"},
                  {"kind": "text_selection", "node_id": "document", "start": 0, "end": 4}]
    assert parse_predicates(predicates, terminal=True) == predicates
