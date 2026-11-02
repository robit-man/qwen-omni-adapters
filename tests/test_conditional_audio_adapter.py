from __future__ import annotations

import base64
import hashlib
import io
import json
import wave

import gguf
import numpy as np
import pytest

from qwen_omni_adapters.audio_adapter import AUDIO_ADAPTER_MEDIA_TYPE, pack_audio_adapter
from qwen_omni_adapters.audio_prefill import (
    BOUNDARIES_KEY,
    PREFILL_ABI,
    PREFILL_KEY,
    PROMPT_KEY,
    TRAINED_AUDIO_PROMPT_SHA256,
)
from qwen_omni_adapters.contract import parse_adapter_request
from qwen_omni_adapters.indexed_sidecar import pack_indexed_tts_sidecar
from qwen_omni_adapters.ollama_sidecar import (
    INDEXED_BINDING_ANNOTATION,
    INDEXED_BINDING_MEDIA_TYPE,
    OMNI_LAYER_MEDIA_TYPE,
    OllamaSidecarError,
    attach_ollama_sidecar,
    manifest_path,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from runtime.adapter_server import (
    Config,
    _separate_bridge_perception,
    build_comprehension_payload,
    build_language_payload,
)


def _write(path, arch, metadata, tensors):
    writer = gguf.GGUFWriter(str(path), arch=arch)
    for key, value in metadata.items():
        kind = (gguf.GGUFValueType.BOOL if type(value) is bool else
                gguf.GGUFValueType.UINT32 if type(value) is int else
                gguf.GGUFValueType.FLOAT32 if type(value) is float else gguf.GGUFValueType.STRING)
        writer.add_key_value(key, value, kind)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture
def bundle(tmp_path, request):
    blocks = getattr(request, 'param', 8)
    model = _write(tmp_path / 'base.gguf', 'qwen35', {
        'qwen35.block_count': blocks, 'qwen35.embedding_length': 8,
        'qwen35.feed_forward_length': 16,
    }, {'token_embd.weight': np.ones((16, 8), dtype=np.float32)})
    projector = _write(tmp_path / 'projector.gguf', 'clip', {
        PREFILL_KEY: PREFILL_ABI, PROMPT_KEY: TRAINED_AUDIO_PROMPT_SHA256,
        BOUNDARIES_KEY: False,
    }, {'mm.weight': np.ones((8, 16), dtype=np.float32)})
    tts = _write(tmp_path / 'tts.gguf', 'llama', {}, {'tts.weight': np.ones((8, 8), dtype=np.float32)})
    codec = _write(tmp_path / 'codec.gguf', 'clip', {}, {'codec.weight': np.ones((8, 8), dtype=np.float32)})
    sidecar = tmp_path / 'sidecar.gguf'
    pack_indexed_tts_sidecar(tts_gguf=tts, tts_projector_gguf=codec, out_gguf=sidecar)
    tensors = {}
    for layer in range(blocks - 8, blocks):
        for module, inputs, outputs in [('down', 16, 8), ('gate', 8, 16), ('up', 8, 16)]:
            stem = f'blk.{layer}.ffn_{module}.weight.'
            tensors[stem + 'lora_a'] = np.full((8, inputs), .01, dtype=np.float32)
            tensors[stem + 'lora_b'] = np.full((outputs, 8), .02, dtype=np.float32)
    source = _write(tmp_path / 'source.gguf', 'qwen35', {
        'general.type': 'adapter', 'adapter.type': 'lora', 'adapter.lora.alpha': 8.0,
    }, tensors)
    bound = tmp_path / 'bound.gguf'
    report = pack_audio_adapter(source=source, base_model=model, projector=projector, output=bound)
    root = tmp_path / 'models'
    layers = []
    for role, path in [('model', model), ('projector', projector)]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        blob = root / 'blobs' / ('sha256-' + digest)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(path.read_bytes())
        layers.append({'mediaType': 'application/vnd.ollama.image.' + role,
                       'digest': 'sha256:' + digest, 'size': path.stat().st_size})
    path = manifest_path('robit/test:v3', root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schemaVersion': 2, 'layers': layers}))
    return root, path, sidecar, bound, source, report


def _full_ffn_export(bundle, tmp_path, *, missing=False):
    root, path, _, _, _, _ = bundle
    layers = json.loads(path.read_text())['layers']
    base, projector = [root / 'blobs' / layer['digest'].replace(':', '-', 1)
                       for layer in layers]
    blocks = int(gguf.GGUFReader(str(base)).fields['qwen35.block_count'].contents())
    tensors = {}
    for layer in range(blocks):
        for module, inputs, outputs in [('down', 16, 8), ('gate', 8, 16), ('up', 8, 16)]:
            stem = f'blk.{layer}.ffn_{module}.weight.'
            tensors[stem + 'lora_a'] = np.full((8, inputs), (layer + 1) / 1000,
                                             dtype=np.float32)
            tensors[stem + 'lora_b'] = np.full((outputs, 8), -(layer + 1) / 1000,
                                             dtype=np.float32)
    if missing:
        tensors.pop('blk.0.ffn_up.weight.lora_b')
    source = _write(tmp_path / 'full-source.gguf', 'qwen35', {
        'general.type': 'adapter', 'adapter.type': 'lora', 'adapter.lora.alpha': 8.0,
    }, tensors)
    return base, projector, source, blocks


@pytest.mark.parametrize('bundle', [32, 64], indirect=True)
def test_full_ffn_binding_preserves_all_layers_and_survives_descriptor_roundtrip(bundle, tmp_path):
    root, path, sidecar, _, _, _ = bundle
    base, projector, source, blocks = _full_ffn_export(bundle, tmp_path)
    output = tmp_path / 'full-bound.gguf'
    report = pack_audio_adapter(source=source, base_model=base, projector=projector,
                                output=output, scope='full-ffn')
    assert report['abi'] == 'robit.omni.conditional-audio-lora.v2'
    assert report['tensors'] == blocks * 6
    assert report['all_tensor_values_unchanged'] and not report['release_qualified']
    original, bound = gguf.GGUFReader(str(source)), gguf.GGUFReader(str(output))
    assert bound.fields['robit.audio_adapter.scope'].contents() == 'full-ffn'
    assert all(a.name == b.name and np.array_equal(a.data, b.data)
               for a, b in zip(original.tensors, bound.tensors, strict=True))
    attach_ollama_sidecar(model='robit/test:v3', bundle_gguf=sidecar,
                          models_dir=root, audio_adapter_gguf=output)
    manifest = json.loads(path.read_text())
    manifest['layers'] = [{k: v for k, v in layer.items() if k != 'annotations'}
                          for layer in manifest['layers']]
    path.write_text(json.dumps(manifest))
    resolved = resolve_ollama_sidecar(model='robit/test:v3', models_dir=root)
    assert resolved['audio_adapter']['binding']['abi'] == report['abi']
    assert resolved['audio_adapter']['binding']['tensors'] == blocks * 6
    prepared = prepare_ollama_sidecar(model='robit/test:v3', models_dir=root,
                                     output_dir=tmp_path / 'views', prefer_direct=True)
    assert prepared['audio_adapter']['path'] == resolved['audio_adapter']['path']
    assert set(prepared['views']) == {'tts_model', 'tts_projector'}


@pytest.mark.parametrize('bundle', [32], indirect=True)
def test_full_ffn_is_not_implicitly_enabled_by_tensor_count(bundle, tmp_path):
    base, projector, source, _ = _full_ffn_export(bundle, tmp_path)
    output = tmp_path / 'never.gguf'
    with pytest.raises(ValueError, match='last8'):
        pack_audio_adapter(source=source, base_model=base, projector=projector, output=output)
    assert not output.exists()


@pytest.mark.parametrize('bundle', [32], indirect=True)
def test_full_ffn_rejects_an_incomplete_early_block(bundle, tmp_path):
    base, projector, source, _ = _full_ffn_export(bundle, tmp_path, missing=True)
    output = tmp_path / 'never.gguf'
    with pytest.raises(ValueError, match='full.*FFN'):
        pack_audio_adapter(source=source, base_model=base, projector=projector,
                           output=output, scope='full-ffn')
    assert not output.exists() and not list(tmp_path.glob('never.gguf.*.partial'))


@pytest.mark.parametrize('bundle', [32], indirect=True)
@pytest.mark.parametrize('abi,scope', [
    ('robit.omni.conditional-audio-lora.v2', None),
    ('robit.omni.conditional-audio-lora.v1', 'full-ffn'),
    ('robit.omni.conditional-audio-lora.v2', 'last8-ffn'),
])
def test_component_abi_and_scope_cannot_disagree(bundle, tmp_path, abi, scope):
    from qwen_omni_adapters.audio_adapter import inspect_audio_adapter
    base, projector, source, _ = _full_ffn_export(bundle, tmp_path)
    output = tmp_path / 'full-bound.gguf'
    report = pack_audio_adapter(source=source, base_model=base, projector=projector,
                                output=output, scope='full-ffn')
    reader = gguf.GGUFReader(str(output))
    metadata = {key: field.contents() for key, field in reader.fields.items()
                if not key.startswith('GGUF.') and key != 'general.architecture'}
    metadata['robit.audio_adapter.abi'] = abi
    if scope is None:
        metadata.pop('robit.audio_adapter.scope')
    else:
        metadata['robit.audio_adapter.scope'] = scope
    damaged = _write(tmp_path / 'damaged.gguf', 'qwen35', metadata,
                     {t.name: t.data.copy() for t in reader.tensors})
    with pytest.raises(ValueError, match='ABI|scope'):
        inspect_audio_adapter(damaged, base_model=base,
                              base_digest=report['base_digest'],
                              projector_digest=report['projector_digest'])


@pytest.mark.parametrize('bundle', [32], indirect=True)
def test_full_ffn_descriptor_bound_covers_qwen_size_and_keeps_v1_limit(bundle, tmp_path):
    from qwen_omni_adapters.audio_adapter import inspect_audio_adapter
    root, path, sidecar, legacy, _, legacy_report = bundle
    base, projector, source, _ = _full_ffn_export(bundle, tmp_path)
    output = tmp_path / 'full-bound.gguf'
    pack_audio_adapter(source=source, base_model=base, projector=projector,
                       output=output, scope='full-ffn')
    # Sparse padding exercises real descriptor/file size admission without
    # retaining a second set of large neural tensors in the unit-test cache.
    with output.open('r+b') as stream:
        stream.truncate(132 * 1024**2 + 65536)
    attach_ollama_sidecar(model='robit/test:v3', bundle_gguf=sidecar,
                          models_dir=root, audio_adapter_gguf=output)
    resolved = resolve_ollama_sidecar(model='robit/test:v3', models_dir=root)
    assert resolved['audio_adapter']['binding']['size'] > 64 * 1024**2
    with legacy.open('r+b') as stream:
        stream.truncate(64 * 1024**2 + 1)
    with pytest.raises(ValueError, match='size|limit'):
        inspect_audio_adapter(legacy, base_model=base,
                              base_digest=legacy_report['base_digest'],
                              projector_digest=legacy_report['projector_digest'])
    with output.open('r+b') as stream:
        stream.truncate(256 * 1024**2 + 1)
    with pytest.raises(ValueError, match='size|limit'):
        inspect_audio_adapter(output, base_model=base,
                              base_digest=legacy_report['base_digest'],
                              projector_digest=legacy_report['projector_digest'])


@pytest.mark.parametrize('bundle', [32], indirect=True)
def test_cli_can_explicitly_bind_full_ffn(bundle, tmp_path, capsys):
    from qwen_omni_adapters.cli import main
    base, projector, source, _ = _full_ffn_export(bundle, tmp_path)
    output = tmp_path / 'cli-full.gguf'
    assert main(['bind-audio-adapter', str(source), '--base-model', str(base),
                 '--projector', str(projector), '--out', str(output),
                 '--scope', 'full-ffn']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['tensors'] == 192
    assert result['abi'] == 'robit.omni.conditional-audio-lora.v2'


def test_binding_survives_stock_descriptor_roundtrip_and_reuses_tts(bundle, tmp_path):
    root, path, sidecar, bound, source, report = bundle
    original = attach_ollama_sidecar(model='robit/test:v3', bundle_gguf=sidecar, models_dir=root)
    attached = attach_ollama_sidecar(model='robit/test:v3', bundle_gguf=sidecar,
                                    models_dir=root, audio_adapter_gguf=bound)
    assert attached['layer']['digest'] == original['layer']['digest']
    assert report['all_tensor_values_unchanged'] is True
    before, after = gguf.GGUFReader(str(source)), gguf.GGUFReader(str(bound))
    assert all(np.array_equal(a.data, b.data) for a, b in zip(before.tensors, after.tensors, strict=True))
    manifest = json.loads(path.read_text())
    manifest['layers'] = [{k: v for k, v in layer.items() if k != 'annotations'} for layer in manifest['layers']]
    path.write_text(json.dumps(manifest))
    resolved = resolve_ollama_sidecar(model='robit/test:v3', models_dir=root)
    assert resolved['audio_adapter']['binding']['digest'] == report['digest']
    prepared = prepare_ollama_sidecar(model='robit/test:v3', models_dir=root,
                                     output_dir=tmp_path / 'views', prefer_direct=True)
    assert prepared['audio_adapter']['path'] == resolved['audio_adapter']['path']
    assert set(prepared['views']) == {'tts_model', 'tts_projector'}
    assert sum(layer['mediaType'] == OMNI_LAYER_MEDIA_TYPE for layer in manifest['layers']) == 1
    # Old v2 resolvers compare against this exact expected object, so a required
    # v3 component cannot be silently ignored by an older runtime.
    binding_layer = next(x for x in manifest['layers'] if x['mediaType'] == INDEXED_BINDING_MEDIA_TYPE)
    binding = json.loads((root / 'blobs' / binding_layer['digest'].replace(':', '-', 1)).read_text())
    assert binding['schema'] == 'robit.omni.layer-binding.v3'
    assert binding['audio_adapter']['execution_scope'] == 'audio-perception'


@pytest.mark.parametrize('damage', ['remove', 'duplicate', 'bytes', 'size', 'base', 'projector', 'binding'])
def test_required_component_damage_and_pair_swaps_fail_closed(bundle, damage):
    root, path, sidecar, bound, *_ = bundle
    attach_ollama_sidecar(model='robit/test:v3', bundle_gguf=sidecar, models_dir=root, audio_adapter_gguf=bound)
    manifest = json.loads(path.read_text())
    layer = next(x for x in manifest['layers'] if x['mediaType'] == AUDIO_ADAPTER_MEDIA_TYPE)
    if damage == 'remove':
        manifest['layers'].remove(layer)
    elif damage == 'duplicate':
        manifest['layers'].append(dict(layer))
    elif damage == 'bytes':
        blob = root / 'blobs' / layer['digest'].replace(':', '-', 1)
        with blob.open('r+b') as stream:
            stream.seek(-4, 2)
            stream.write(b'XXXX')
    elif damage == 'size':
        layer['size'] += 1
    elif damage in {'base', 'projector'}:
        role = 'model' if damage == 'base' else 'projector'
        standard = next(x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.' + role)
        original = root / 'blobs' / standard['digest'].replace(':', '-', 1)
        raw = original.read_bytes() + b'padding'
        digest = hashlib.sha256(raw).hexdigest()
        (root / 'blobs' / ('sha256-' + digest)).write_bytes(raw)
        standard.update(digest='sha256:' + digest, size=len(raw))
    else:
        manifest['layers'] = [x for x in manifest['layers'] if x['mediaType'] != INDEXED_BINDING_MEDIA_TYPE]
        next(x for x in manifest['layers'] if x['mediaType'] == OMNI_LAYER_MEDIA_TYPE)['annotations'].pop(INDEXED_BINDING_ANNOTATION)
    path.write_text(json.dumps(manifest))
    with pytest.raises(OllamaSidecarError):
        resolve_ollama_sidecar(model='robit/test:v3', models_dir=root)


@pytest.mark.parametrize('damage', ['architecture', 'alpha', 'missing', 'rank', 'nan'])
def test_incompatible_source_never_creates_a_bound_output(bundle, tmp_path, damage):
    root, path, _, _, source, _ = bundle
    layers = json.loads(path.read_text())['layers']
    base, projector = [root / 'blobs' / layer['digest'].replace(':', '-', 1) for layer in layers]
    reader = gguf.GGUFReader(str(source))
    tensors = {t.name: t.data.copy() for t in reader.tensors}
    metadata = {'general.type': 'adapter', 'adapter.type': 'lora', 'adapter.lora.alpha': 8.0}
    arch = 'qwen35'
    if damage == 'architecture':
        arch = 'llama'
    elif damage == 'alpha':
        metadata['adapter.lora.alpha'] = 16.0
    elif damage == 'missing':
        tensors.pop(next(iter(tensors)))
    elif damage == 'rank':
        tensors[next(iter(tensors))] = np.ones((4, 16), dtype=np.float32)
    else:
        tensors[next(iter(tensors))].flat[0] = np.nan
    bad = _write(tmp_path / 'bad.gguf', arch, metadata, tensors)
    output = tmp_path / 'never.gguf'
    with pytest.raises(ValueError):
        pack_audio_adapter(source=bad, base_model=base, projector=projector, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob('never.gguf.*.partial'))


def test_cannot_rebind_weights_or_overwrite_an_existing_export(bundle, tmp_path):
    root, path, _, bound, source, _ = bundle
    layers = json.loads(path.read_text())['layers']
    base, projector = [root / 'blobs' / layer['digest'].replace(':', '-', 1) for layer in layers]
    output = tmp_path / 'output.gguf'
    with pytest.raises(ValueError, match='already bound'):
        pack_audio_adapter(source=bound, base_model=base, projector=projector, output=output)
    output.write_bytes(b'retained previous artifact')
    with pytest.raises(ValueError, match='already exists'):
        pack_audio_adapter(source=source, base_model=base, projector=projector, output=output)
    assert output.read_bytes() == b'retained previous artifact'


def _config(**changes):
    fields = dict(comprehension_url='http://127.0.0.1:8901/v1/chat/completions',
                  comprehension_model='primary', language_url='http://127.0.0.1:8901/v1/chat/completions',
                  tts_url='unused', timeout_s=10, language_api='openai',
                  comprehension_disable_thinking=True, conditional_audio_adapter=True)
    return Config(**{**fields, **changes})


def _request(*, audio=False, image=False):
    message = {'role': 'user', 'content': 'Describe this.'}
    if audio:
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x01\x00' * 1600)
        message['audios'] = [base64.b64encode(stream.getvalue()).decode()]
    if image:
        # Valid 1x1 PNG from a real image encoder.
        from PIL import Image

        stream = io.BytesIO()
        Image.new('RGB', (1, 1), 'red').save(stream, format='PNG')
        message['images'] = [base64.b64encode(stream.getvalue()).decode()]
    return parse_adapter_request({'model': 'primary', 'stream': False, 'messages': [message]})


@pytest.mark.parametrize('audio,image', [(True, False), (False, True), (True, True)])
def test_backend_scales_are_pinned_by_actual_media_and_language_always_disables(audio, image):
    config, parsed = _config(), _request(audio=audio, image=image)
    payload = build_comprehension_payload(parsed, config)
    visual, acoustic = _separate_bridge_perception(payload, config)
    if audio and image:
        assert visual['lora'] == [{'id': 0, 'scale': 0.0}]
        assert acoustic['lora'] == [{'id': 0, 'scale': 1.0}]
        assert not any(p['type'] == 'input_audio' for m in visual['messages'][1:] for p in m['content'])
    else:
        assert visual['lora'] == [{'id': 0, 'scale': float(audio)}]
    assert visual['cache_prompt'] is False
    language = build_language_payload(parsed, '<audio_observation>sound</audio_observation>',
                                      language_api='openai', config=config)
    assert language['lora'] == [{'id': 0, 'scale': 0.0}]
    assert build_language_payload(_request(), None, language_api='openai', config=config)['lora'] == [{'id': 0, 'scale': 0.0}]


@pytest.mark.parametrize('changes', [dict(language_api='ollama'), dict(comprehension_disable_thinking=False),
                                     dict(language_url='https://example.org'),
                                     dict(comprehension_url='http://example.org', language_url='http://example.org')])
def test_conditional_mode_refuses_a_second_or_uncontrolled_trunk(changes):
    with pytest.raises(ValueError, match='one local trained-bridge trunk'):
        _config(**changes)
