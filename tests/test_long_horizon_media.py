"""Real CPU-generated input fixtures, never model/task-completion evidence."""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from runtime import long_horizon_cases as cases
from runtime import long_horizon_media as media


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('Actual CPU media tools are unavailable; media readiness remains blocked')
    case = cases.load_case('LHN-083')
    root = tmp_path_factory.mktemp('owned-media')
    cache = root / 'cache-a'
    receipt = media.build_media_cache(case, cache)
    return case, cache, receipt


def clone_cache(source, target):
    shutil.copytree(source, target, copy_function=shutil.copy2)
    return target


def test_generate_six_clips_twice_on_pinned_cpu_toolchain_is_byte_identical(generated, tmp_path):
    case, first, receipt = generated
    second = tmp_path / 'cache-b'
    again = media.build_media_cache(cases.load_case('LHN-088'), second)
    assert again['asset_hashes'] == receipt['asset_hashes']
    for path in receipt['asset_hashes']:
        assert (first / Path(path).name).read_bytes() == (second / Path(path).name).read_bytes()
    a = json.loads((first / 'media-cache.json').read_bytes())
    b = json.loads((second / 'media-cache.json').read_bytes())
    assert a['recipe_sha256'] == b['recipe_sha256'] and a['toolchain'] == b['toolchain']
    assert a['recipe']['font'] == json.loads(json.dumps(media.FONT))
    assert a['recipe']['threads'] == 1 and a['recipe']['cpu_flags'] == 0
    assert a['recipe']['video_codec'] == 'libx264'
    assert receipt['model_calls'] == receipt['GUI_inputs'] == 0 and receipt['task_progress'] is False


def test_actual_decoded_frames_ids_colors_durations_and_audio_match_declared_inputs(generated):
    case, cache, receipt = generated
    capability = media.verify_media_cache(case, cache, expected_manifest_sha256=receipt['cache_manifest_sha256'])
    assert capability['status'] == 'verified' and capability['input_only'] is True
    manifest = json.loads((cache / 'media-cache.json').read_bytes())
    assert [a['specification']['duration_seconds'] for a in manifest['assets']] == [12, 14, 16, 18, 20, 22]
    for item, spec in zip(manifest['assets'], media.specifications(case), strict=True):
        actual = media.inspect_media(cache / item['file'], spec)
        assert actual['sha256'] == item['readback']['sha256']
        assert actual['container_duration_seconds'] == spec['duration_seconds']
        assert actual['video_frames'] == spec['duration_seconds'] * media.FPS
        assert {f['decoded_clip_id'] for f in actual['frames']} == {spec['clip_id']}
        assert actual['audio_peak_abs_pcm16'] == 0
        assert len(actual['frames']) == 3
    assert capability['solver_method_or_completion_evidence'] is False


@pytest.mark.parametrize('case_id', [f'LHN-{index:03d}' for index in range(83, 89)])
def test_exact_media_copies_enter_initial_inventory_without_outputs_or_spec_mutation(generated, tmp_path, case_id):
    _, cache, receipt = generated
    case = cases.load_case(case_id)
    before = copy.deepcopy(case)
    work = tmp_path / 'Workbench'
    initial = cases.materialize(case, work)
    added, proof = media.materialize_media(case, work, cache,
        expected_manifest_sha256=receipt['cache_manifest_sha256'])
    assert case == before
    assert added == receipt['asset_hashes'] and len({**initial, **added}) == 9
    assert {p.relative_to(work).as_posix() for p in work.rglob('*') if p.is_file()} == set(initial) | set(added)
    for path, digest in added.items():
        target, source = work / path, cache / Path(path).name
        assert cases.sha(target.read_bytes()) == cases.sha(source.read_bytes()) == digest
        assert target.stat().st_nlink == 1 and source.stat().st_nlink == 1
        assert (target.stat().st_dev, target.stat().st_ino) != (source.stat().st_dev, source.stat().st_ino)
        assert target.stat().st_mode & 0o777 == 0o444
    assert not any((work / path).exists() for path in case['required_outputs'])
    declared = json.loads((work / 'Input/clip-manifest.json').read_bytes())
    assert all(item['materialized'] is False for item in declared)
    assert proof['task_progress'] is False and proof['model_calls'] == proof['GUI_inputs'] == 0
    assert all(item['independent_regular_copy'] for item in proof['materialized_inputs'])
    archive_inputs = {item['member_path'] for item in case['archive_contract']['entries']} & set(added)
    assert archive_inputs == set(added)


@pytest.mark.parametrize('fault', ['path', 'id', 'duration', 'color', 'audio', 'missing', 'duplicate', 'materialized'])
def test_undeclared_or_mutated_public_specs_fail_without_output(tmp_path, fault):
    case = cases.load_case('LHN-083')
    item = next(i for i in case['fixture']['input_files'] if i['path'] == 'Input/clip-manifest.json')
    specs = json.loads(item['content_utf8'])
    if fault in {'path', 'id', 'duration', 'materialized'}:
        specs[0][{'id': 'clip_id', 'duration': 'duration_seconds'}.get(fault, fault)] = {
            'path': '../outside.mp4', 'id': 'OTHER', 'duration': 1, 'materialized': True}[fault]
    elif fault in {'audio', 'color'}:
        specs[0]['generation_spec'][fault] = 'different'
    elif fault == 'missing':
        specs.pop()
    else:
        specs[-1] = copy.deepcopy(specs[0])
    item['content_utf8'] = json.dumps(specs)
    item['content_sha256'] = cases.sha(item['content_utf8'].encode())
    with pytest.raises(ValueError, match='frozen clip declaration'):
        media.build_media_cache(case, tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('fault', ['missing', 'changed', 'manifest_rewritten', 'partial', 'symlink', 'directory', 'wrong_pin'])
def test_cache_changes_never_repair_themselves_or_confer_readiness(generated, tmp_path, fault):
    case, source, receipt = generated
    cache = clone_cache(source, tmp_path / 'cache')
    target = cache / 'clip-1.mp4'
    before = target.read_bytes()
    pin = receipt['cache_manifest_sha256']
    if fault == 'missing':
        target.unlink()
    elif fault in {'changed', 'manifest_rewritten'}:
        target.chmod(0o600)
        target.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
        if fault == 'manifest_rewritten':
            manifest = cache / 'media-cache.json'
            data = json.loads(manifest.read_bytes())
            data['assets'][0]['readback']['sha256'] = cases.sha(target.read_bytes())
            manifest.chmod(0o600)
            manifest.write_text(json.dumps(data))
    elif fault == 'partial':
        (cache / 'clip-1.mp4.partial').write_bytes(b'incomplete')
    elif fault in {'symlink', 'directory'}:
        target.unlink()
        target.symlink_to(source / target.name) if fault == 'symlink' else target.mkdir()
    else:
        pin = '0' * 64
    with pytest.raises((ValueError, OSError)):
        media.verify_media_cache(case, cache, expected_manifest_sha256=pin)
    assert (source / 'clip-1.mp4').read_bytes() == before


def test_hardlinked_cache_and_symlink_root_are_rejected(generated, tmp_path):
    case, source, receipt = generated
    cache = clone_cache(source, tmp_path / 'cache')
    alias = tmp_path / 'external-hardlink'
    os.link(cache / 'clip-1.mp4', alias)
    with pytest.raises(ValueError, match='bytes changed'):
        media.verify_media_cache(case, cache, expected_manifest_sha256=receipt['cache_manifest_sha256'])
    alias.unlink()
    symlink = tmp_path / 'linked-cache'
    symlink.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        media.verify_media_cache(case, symlink, expected_manifest_sha256=receipt['cache_manifest_sha256'])


def test_existing_destination_partial_file_is_preserved_and_no_other_clips_are_written(generated, tmp_path):
    case, cache, receipt = generated
    work = tmp_path / 'Workbench'
    cases.materialize(case, work)
    clips = work / 'Input/Clips'
    clips.mkdir()
    (clips / 'clip-1.mp4').write_bytes(b'existing user-owned bytes')
    with pytest.raises(ValueError, match='overwrite'):
        media.materialize_media(case, work, cache, expected_manifest_sha256=receipt['cache_manifest_sha256'])
    assert list(clips.iterdir()) == [clips / 'clip-1.mp4']
    assert (clips / 'clip-1.mp4').read_bytes() == b'existing user-owned bytes'


def test_wrong_declared_workbench_manifest_and_existing_cache_are_preserved(generated, tmp_path):
    case, cache, receipt = generated
    with pytest.raises(ValueError, match='exclusive'):
        media.build_media_cache(case, cache)
    work = tmp_path / 'Workbench'
    cases.materialize(case, work)
    (work / 'Input/clip-manifest.json').write_bytes(b'changed public inputs')
    with pytest.raises(ValueError, match='unchanged declared'):
        media.materialize_media(case, work, cache, expected_manifest_sha256=receipt['cache_manifest_sha256'])
    assert not (work / 'Input/Clips').exists()


def test_media_source_does_not_consume_goal_programme_order_or_evaluator_answers():
    case = cases.load_case('LHN-083')
    minimal = {'fixture': {'input_files': [next(i for i in case['fixture']['input_files']
        if i['path'] == 'Input/clip-manifest.json')]}}
    assert media.specifications(minimal) == media.specifications(case)


@pytest.fixture
def portable_tools(tmp_path, monkeypatch):
    """Filesystem identities are real; fake listings never confer media readiness."""
    source = tmp_path / 'checkout/runtime/long_horizon_media.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fixture module location')
    monkeypatch.setattr(media, '__file__', str(source))
    bundled = source.parents[1] / 'vendor/ffmpeg/bin'
    path_bin = tmp_path / 'path-bin'
    explicit = tmp_path / 'explicit-bin'
    for folder in (bundled, path_bin, explicit):
        folder.mkdir(parents=True)
        for name in ('ffmpeg', 'ffprobe'):
            target = folder / name
            target.write_bytes((str(folder) + '/' + name).encode())
            target.chmod(0o700)
    monkeypatch.setenv('PATH', str(path_bin))
    state = {'encoders': b'Encoders:\n V....D libx264 CPU H.264 encoder\n',
             'ldd': b'libc.so.6 => /not-a-codec/libc.so.6 (0x1)\n', 'calls': []}

    def run(command, **kwargs):
        state['calls'].append(command)
        if command[0] == 'ldd':
            return state['ldd']
        if command[1:] == ['-version']:
            return (Path(command[0]).name + ' fixture-version\nconfiguration: CPU test').encode()
        if command[1:] == ['-hide_banner', '-encoders']:
            return state['encoders']
        raise AssertionError('Unexpected command; fake tools must never generate assets')

    monkeypatch.setattr(media, '_run', run)
    return bundled, path_bin, explicit, state


def test_default_media_pair_prefers_bundle_over_path_but_preserves_explicit_pair(portable_tools):
    bundled, path_bin, explicit, _ = portable_tools
    default = media._toolchain('ffmpeg', 'ffprobe')
    chosen = media._toolchain(str(explicit / 'ffmpeg'), str(explicit / 'ffprobe'))
    for name in ('ffmpeg', 'ffprobe'):
        assert default[name]['path'] == str(bundled / name)
        assert default[name]['sha256'] == media.sha((bundled / name).read_bytes())
        assert chosen[name]['path'] == str(explicit / name)
        assert chosen[name]['path'] != str(path_bin / name)


@pytest.mark.parametrize('availability', ['absent', 'not_executable'])
def test_unavailable_bundled_tool_uses_path_without_hiding_explicit_failure(portable_tools, availability):
    bundled, path_bin, explicit, _ = portable_tools
    for name in ('ffmpeg', 'ffprobe'):
        if availability == 'absent':
            (bundled / name).unlink()
        else:
            (bundled / name).chmod(0o600)
        assert media._tool(name)['path'] == str(path_bin / name)
    missing = explicit / 'missing-ffmpeg'
    assert media._media_tool(str(missing)) == str(missing)
    with pytest.raises(ValueError, match='Missing CPU media tool'):
        media._tool(str(missing))


@pytest.mark.parametrize('encoders', [b' V....D h264_nvenc GPU encoder\n',
    b' V....D libx264rgb RGB encoder\n', b' V....D h264 encoder description mentions libx264\n'])
def test_missing_exact_cpu_libx264_rejects_before_generation(portable_tools, tmp_path, encoders):
    _, _, _, state = portable_tools
    state['encoders'] = encoders
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError, match='libx264 CPU encoder'):
        media.build_media_cache(cases.load_case('LHN-083'), cache)
    assert not cache.exists()
    assert not any(command[0] == 'ldd' for command in state['calls'])
    assert all(command[1:] in (['-version'], ['-hide_banner', '-encoders']) for command in state['calls'])


@pytest.mark.parametrize('linkage', ['shared_x264', 'embedded_avcodec', 'embedded_ffmpeg'])
def test_encoder_provenance_pins_actual_shared_or_embedding_container(portable_tools, tmp_path, linkage):
    bundled, _, _, state = portable_tools
    codec = tmp_path / 'libavcodec.so.60'
    codec.write_bytes(b'actual codec fixture identity')
    x264 = tmp_path / 'libx264.so.164'
    x264.write_bytes(b'actual encoder fixture identity')
    lines = []
    if linkage != 'embedded_ffmpeg':
        lines.append(f'libavcodec.so.60 => {codec} (0x1)')
    if linkage == 'shared_x264':
        lines.append(f'libx264.so.164 => {x264} (0x2)')
    # Successful ldd with codecs embedded, as in the Egg bundle. This does not
    # simulate fully static ELF binaries, whose ldd may instead fail.
    state['ldd'] = ('\n'.join(lines) + '\n').encode()
    chain = media._toolchain('ffmpeg', 'ffprobe')
    encoder = chain['cpu_encoder']
    expected = x264 if linkage == 'shared_x264' else codec if linkage == 'embedded_avcodec' else bundled / 'ffmpeg'
    assert encoder['name'] == 'libx264'
    assert encoder['capability_sha256'] == media.sha(state['encoders'])
    assert encoder['containers'] == [{'path': str(expected), 'sha256': media.sha(expected.read_bytes())}]
    assert encoder['linkage'] == ('shared_library' if linkage == 'shared_x264' else 'embedded_in_pinned_container')
    before = encoder['containers'][0]['sha256']
    expected.write_bytes(expected.read_bytes() + b' changed binary')
    again = media._toolchain('ffmpeg', 'ffprobe')
    assert again['cpu_encoder']['containers'][0]['sha256'] != before
    assert again['cpu_encoder']['capability_sha256'] == encoder['capability_sha256']


def test_actual_generation_provenance_and_readback_use_same_resolved_pair(generated, monkeypatch):
    case, cache, _ = generated
    manifest = json.loads((cache / 'media-cache.json').read_bytes())
    expected = {name: Path(tool['path']).resolve() for name, tool in manifest['toolchain'].items()
                if name in {'ffmpeg', 'ffprobe'}}
    calls = []
    actual_run = media._run

    def record(command, **kwargs):
        calls.append(command)
        return actual_run(command, **kwargs)

    monkeypatch.setattr(media, '_run', record)
    spec = media.specifications(case)[0]
    readback = media.inspect_media(cache / 'clip-1.mp4', spec)
    assert readback['sha256'] == manifest['assets'][0]['readback']['sha256']
    assert Path(calls[0][0]).resolve() == expected['ffprobe']
    assert len(calls) == 3 and all(Path(command[0]).resolve() == expected['ffmpeg'] for command in calls[1:])
    assert manifest['toolchain']['cpu_encoder']['name'] == 'libx264'
