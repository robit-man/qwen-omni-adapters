from __future__ import annotations

import copy
import hashlib
import io

import httpx
import pytest
from PIL import Image, PngImagePlugin

from portal import native_ui
from qwen_omni_adapters.computer_use import RapidUiController


def _png(size, changes=(), *, palette=False, metadata=''):
    image = Image.new('RGB', size, 'white')
    for x, y, color in changes:
        image.putpixel((x, y), color)
    if palette:
        image = image.quantize(colors=16)
    output = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text('source-time', metadata)
    image.save(output, format='PNG', pnginfo=info)
    return output.getvalue()


def _fixture(monkeypatch, *, geometry=None, margins=None, root=None):
    identity = {'window_id': '17', 'process_id': 42}
    geometry = geometry or {'x': -17, 'y': -23, 'width': 972, 'height': 1184}
    margins = margins or {'left': 61, 'right': 61, 'top': 55, 'bottom': 67}
    root = root or {'window_id': '492', 'geometry': {'x': 0, 'y': 0, 'width': 1440, 'height': 2500}}
    image = {'x': max(0, geometry['x']), 'y': max(0, geometry['y']),
             'width': min(root['geometry']['width'], geometry['x'] + geometry['width']) - max(0, geometry['x']),
             'height': min(root['geometry']['height'], geometry['y'] + geometry['height']) - max(0, geometry['y'])}
    frame = {'image': _png((image['width'], image['height'])), 'identity': identity, 'geometry': geometry,
             'capture_root': root, 'capture_source': 'x11_root_screen', 'process_start': '101',
             'image_geometry': image, 'pixel_offset': {'x': image['x'] - geometry['x'], 'y': image['y'] - geometry['y']}}
    frame['sha256'] = hashlib.sha256(frame['image']).hexdigest()
    binding = {'space': 'window', 'window_id': '17', 'surface': copy.deepcopy(geometry), 'frame_extents': margins,
               'content': {'x': geometry['x'] + margins['left'], 'y': geometry['y'] + margins['top'],
                           'width': geometry['width'] - margins['left'] - margins['right'],
                           'height': geometry['height'] - margins['top'] - margins['bottom']}}
    state = {'session_id': 'one', 'revision': 'semantic', 'scope': {'id': 'native', 'kind': 'desktop', 'window_id': '17'},
             'coordinate_binding': binding, 'nodes': []}
    monkeypatch.setattr(native_ui, '_x11_foreground', lambda _env: dict(identity))
    monkeypatch.setattr(native_ui, '_capture_window', lambda *_args: copy.deepcopy(frame))
    monkeypatch.setattr('portal.desktop.desktop_subprocess_environment', lambda: {})
    return native_ui.NativeUiAdapter('one'), frame, state


def _replace_png(frame, changes=(), *, palette=False, metadata=''):
    size = frame['image_geometry']['width'], frame['image_geometry']['height']
    frame['image'] = _png(size, changes, palette=palette, metadata=metadata)
    frame['sha256'] = hashlib.sha256(frame['image']).hexdigest()


def test_actual_shaped_outside_content_pixel_changes_keep_identical_transmitted_frame_key(monkeypatch):
    adapter, frame, state = _fixture(monkeypatch)
    first = adapter._visual_capture(state)
    _replace_png(frame, [(261, 10, (255, 0, 0))], metadata='later-source-time')
    second = adapter._visual_capture(state)
    assert first['key'] == second['key']
    assert first['image'] == second['image']
    assert first['source_frame_sha256'] != second['source_frame_sha256']
    assert second['image_geometry'] == {'x': 44, 'y': 32, 'width': 850, 'height': 1062}
    assert second['pixel_offset'] == {'x': 61, 'y': 55}
    assert second['geometry'] == frame['geometry']


def test_changed_content_pixel_changes_exact_transmitted_frame_and_key(monkeypatch):
    adapter, frame, state = _fixture(monkeypatch)
    first = adapter._visual_capture(state)
    _replace_png(frame, [(50, 50, (0, 0, 0))])
    second = adapter._visual_capture(state)
    assert first['key'] != second['key'] and first['sha256'] != second['sha256']
    with Image.open(io.BytesIO(second['image'])) as image:
        assert image.mode == 'RGB' and image.getpixel((6, 18)) == (0, 0, 0)


def test_palette_and_global_png_metadata_cannot_change_identical_decoded_content_binding(monkeypatch):
    adapter, frame, state = _fixture(monkeypatch)
    _replace_png(frame, palette=True, metadata='first')
    first = adapter._visual_capture(state)
    _replace_png(frame, [(261, 10, (0, 0, 255))], palette=True, metadata='second')
    second = adapter._visual_capture(state)
    assert first['source_frame_sha256'] != second['source_frame_sha256']
    assert first['sha256'] == second['sha256'] and first['key'] == second['key']
    with Image.open(io.BytesIO(second['image'])) as image:
        assert image.mode == 'RGB' and image.info == {}


@pytest.mark.parametrize('fault', ['none', 'extra', 'missing', 'space', 'window', 'surface', 'content',
                                  'margin', 'bool_margin', 'negative_margin', 'root_origin', 'source_origin', 'offset'])
def test_present_unproven_or_inconsistent_mapping_fails_closed_without_full_frame_fallback(monkeypatch, fault):
    adapter, frame, state = _fixture(monkeypatch)
    binding = state['coordinate_binding']
    if fault == 'none':
        state['coordinate_binding'] = None
    elif fault == 'extra':
        binding['claimed_valid'] = True
    elif fault == 'missing':
        del binding['frame_extents']
    elif fault == 'space':
        binding['space'] = 'screen'
    elif fault == 'window':
        binding['window_id'] = '18'
    elif fault in {'surface', 'content'}:
        binding[fault]['x'] += 1
    elif fault in {'margin', 'bool_margin', 'negative_margin'}:
        binding['frame_extents']['left'] = 62 if fault == 'margin' else True if fault == 'bool_margin' else -1
    elif fault == 'root_origin':
        frame['capture_root']['geometry']['x'] = 1
    elif fault == 'source_origin':
        frame['image_geometry']['x'] = 1
    else:
        frame['pixel_offset']['x'] += 1
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        adapter._visual_capture(state)
    assert exc.value.code == 'native_content_capture_binding_invalid'


@pytest.mark.parametrize('changed', ['root', 'surface', 'content_and_margins', 'semantic_revision', 'process_start'])
def test_consistent_positive_binding_changes_still_change_action_frame_key(monkeypatch, changed):
    adapter, frame, state = _fixture(monkeypatch)
    first = adapter._visual_capture(state)
    if changed == 'root':
        frame['capture_root']['window_id'] = '493'
    elif changed == 'surface':
        frame['geometry']['x'] -= 1
        frame['geometry']['width'] += 1
        frame['pixel_offset']['x'] += 1
        state['coordinate_binding']['surface'] = copy.deepcopy(frame['geometry'])
        state['coordinate_binding']['frame_extents']['left'] += 1
    elif changed == 'content_and_margins':
        state['coordinate_binding']['frame_extents']['left'] += 1
        state['coordinate_binding']['content']['x'] += 1
        state['coordinate_binding']['content']['width'] -= 1
    elif changed == 'semantic_revision':
        state['revision'] = 'changed-semantic'
    else:
        frame['process_start'] = '102'
    second = adapter._visual_capture(state)
    assert first['key'] != second['key']


@pytest.mark.parametrize(('geometry', 'margins', 'expected'), [
    ({'x': -100, 'y': -100, 'width': 300, 'height': 300},
     {'left': 10, 'right': 10, 'top': 10, 'bottom': 10}, {'x': 0, 'y': 0, 'width': 190, 'height': 190}),
    ({'x': 1380, 'y': 2450, 'width': 100, 'height': 100},
     {'left': 5, 'right': 5, 'top': 5, 'bottom': 5}, {'x': 1385, 'y': 2455, 'width': 55, 'height': 45}),
])
def test_content_crop_intersects_root_at_all_clipped_edges_without_resizing(monkeypatch, geometry, margins, expected):
    adapter, frame, state = _fixture(monkeypatch, geometry=geometry, margins=margins)
    capture = adapter._visual_capture(state)
    assert capture['image_geometry'] == expected and capture['geometry'] == geometry
    assert capture['source_image_geometry'] == frame['image_geometry']
    assert capture['pixel_offset'] == {'x': expected['x'] - geometry['x'], 'y': expected['y'] - geometry['y']}
    with Image.open(io.BytesIO(capture['image'])) as image:
        assert image.size == (expected['width'], expected['height'])


def test_content_entirely_outside_root_withholds_capture(monkeypatch):
    adapter, _frame, state = _fixture(monkeypatch, geometry={'x': -200, 'y': 0, 'width': 220, 'height': 100},
                                    margins={'left': 10, 'right': 30, 'top': 10, 'bottom': 10})
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        adapter._visual_capture(state)
    assert exc.value.code == 'native_content_capture_offscreen'


@pytest.mark.parametrize('fault', ['invalid', 'truncated', 'wrong_dimensions', 'jpeg', 'sha', 'oversize'])
def test_malformed_or_unbound_png_is_not_normalized_into_trusted_content(monkeypatch, fault):
    adapter, frame, state = _fixture(monkeypatch)
    if fault == 'invalid':
        frame['image'] = b'not-a-png' * 10
    elif fault == 'truncated':
        frame['image'] = frame['image'][:40]
    elif fault == 'wrong_dimensions':
        frame['image'] = _png((1, 1))
    elif fault == 'jpeg':
        output = io.BytesIO()
        Image.new('RGB', (955, 1161), 'white').save(output, format='JPEG')
        frame['image'] = output.getvalue()
    elif fault == 'oversize':
        frame['image'] = b'x' * (8 * 1024 * 1024 + 1)
    frame['sha256'] = hashlib.sha256(frame['image']).hexdigest() if fault != 'sha' else 'unbound'
    with pytest.raises(native_ui.NativeUiUnavailable) as exc:
        adapter._visual_capture(state)
    assert exc.value.code == 'native_content_capture_image_invalid'


def test_no_native_coordinate_binding_preserves_existing_full_frame_path(monkeypatch):
    adapter, frame, state = _fixture(monkeypatch)
    del state['coordinate_binding']
    first = adapter._visual_capture(state)
    assert first['image'] == frame['image'] and first['sha256'] == frame['sha256']
    assert first['image_geometry'] == frame['image_geometry']
    assert not {'content_region', 'source_frame_sha256', 'source_image_geometry'}.intersection(first)
    _replace_png(frame, [(261, 10, (255, 0, 0))])
    assert adapter._visual_capture(state)['key'] != first['key']


@pytest.mark.parametrize('inside', [False, True])
def test_bound_content_point_uses_global_crop_origin_and_full_surface_guard(monkeypatch, inside):
    adapter, frame, state = _fixture(monkeypatch)
    adapter._runner = lambda _request: copy.deepcopy(state)
    adapter.pointing_url = 'http://127.0.0.1:8940'
    adapter.pointing_model, adapter.pointing_revision = 'fixture/moondream', 'pinned'
    requests = []
    def point(url, **kwargs):
        import base64
        with Image.open(io.BytesIO(base64.b64decode(kwargs['json']['image']))) as image:
            assert image.size == (850, 1062) and image.mode == 'RGB'
        requests.append(kwargs)
        return httpx.Response(200, json={'model': 'fixture/moondream', 'revision': 'pinned', 'points': [{'x': .5, 'y': .5}]},
                              request=httpx.Request('POST', url))
    monkeypatch.setattr('httpx.post', point)
    step = {'id': 'canvas', 'verb': 'click', 'target': {'name': 'Canvas'}, 'args': {}, 'expect': [{'kind': 'text', 'text': 'Done'}]}
    adapter.prepare_step(step)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    observed = controller.observe()
    action = next(item for item in controller.enumerate(step, observed) if item['kind'] == 'action')
    original_source_sha = frame['sha256']
    _replace_png(frame, [(50, 50, (0, 0, 0))] if inside else [(261, 10, (0, 0, 0))])
    dispatched = []
    monkeypatch.setattr(native_ui, '_pointer_dispatch', lambda identity, geometry, pixel, verb, **_kwargs:
                        dispatched.append((identity, geometry, pixel, verb)) or {'status': 'dispatched'})
    result = adapter.perform(action)
    assert len(requests) == 1
    if inside:
        assert result['reason'] == 'native_visual_frame_changed' and dispatched == []
    else:
        assert result['status'] == 'dispatched'
        assert dispatched[0][1:] == (frame['geometry'], (469, 563), 'click')
        assert adapter.point_receipt['source_frame_sha256'] == original_source_sha != frame['sha256']
        assert adapter.point_receipt['source_image_geometry'] == frame['image_geometry']
        assert adapter.point_receipt['content_region'] == state['coordinate_binding']
