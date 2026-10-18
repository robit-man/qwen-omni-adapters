"""Owned CC0 screening inputs; CPU generation/readback never counts as solver work."""
from __future__ import annotations

import array
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from runtime.long_horizon_cases import MAX_FILE, bounded_bytes, safe_path, sha

DECLARED_SHA256 = 'd2d382c37daf53dfcbaeb215cbb2092f54e1e0fe30032a60bbb00a86e4296539'
WIDTH, HEIGHT, FPS, RATE = 640, 360, 25, 48000
SCALE, LEFT, TOP = 8, 180, 152
COLORS = {'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255),
          'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255)}
# Original fixed bitmap glyphs, dedicated with the generated assets under CC0.
FONT = {
    'C': ('01110','10001','10000','10000','10000','10001','01110'),
    'L': ('10000','10000','10000','10000','10000','10000','11111'),
    'I': ('11111','00100','00100','00100','00100','00100','11111'),
    'P': ('11110','10001','10001','11110','10000','10000','10000'),
    '-': ('00000','00000','00000','11111','00000','00000','00000'),
    '0': ('01110','10001','10011','10101','11001','10001','01110'),
    '1': ('00100','01100','00100','00100','00100','00100','01110'),
    '2': ('01110','10001','00001','00010','00100','01000','11111'),
    '3': ('11110','00001','00001','01110','00001','00001','11110'),
    '4': ('00010','00110','01010','10010','11111','00010','00010'),
    '5': ('11111','10000','10000','11110','00001','00001','11110'),
    '6': ('01110','10000','10000','11110','10001','10001','01110'),
    '7': ('11111','00001','00010','00100','01000','01000','01000'),
    '8': ('01110','10001','10001','01110','10001','10001','01110'),
    '9': ('01110','10001','10001','01111','00001','00001','01110'),
}
RECIPE = {'schema': 'robit.omni.owned-media-recipe.v1', 'license': 'CC0-1.0',
    'width': WIDTH, 'height': HEIGHT, 'fps': FPS, 'audio_rate': RATE, 'channels': 1,
    'font': FONT, 'font_scale': SCALE, 'text_origin': [LEFT, TOP], 'colors_rgb': COLORS,
    'video_codec': 'libx264', 'profile': 'main', 'pixel_format': 'yuv420p', 'qp': 18,
    'audio_codec': 'aac', 'audio_bitrate': '64k', 'threads': 1, 'cpu_flags': 0,
    'x264_parameters': 'threads=1:lookahead_threads=1:sliced_threads=0:asm=0',
    'creation_time': '1970-01-01T00:00:00Z', 'video_track_timescale': 1000,
    'byte_identity_scope': 'same pinned generator/toolchain; reuse frozen bytes on other hosts'}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def specifications(case: dict) -> list[dict]:
    """Consume only the unchanged public clip declaration, never evaluator data."""
    files = [item for item in case['fixture']['input_files'] if item['path'] == 'Input/clip-manifest.json']
    if len(files) != 1:
        raise ValueError('Expected the one declared owned clip manifest')
    item = files[0]
    raw = item['content_utf8'].encode()
    if sha(raw) != DECLARED_SHA256 or item['content_sha256'] != DECLARED_SHA256:
        raise ValueError('The frozen clip declaration changed')
    specs = json.loads(raw)
    if not isinstance(specs, list) or len(specs) != 6:
        raise ValueError('Expected six unique declared clips')
    for index, (spec, color) in enumerate(zip(specs, COLORS, strict=True), 1):
        if (set(spec) != {'clip_id', 'path', 'duration_seconds', 'license', 'generation_spec', 'materialized'}
                or spec['clip_id'] != f'CLIP-{index}' or spec['path'] != f'Input/Clips/clip-{index}.mp4'
                or spec['duration_seconds'] != 10 + index * 2 or spec['materialized'] is not False
                or spec['license'] != 'CC0 owned synthetic evaluation asset'
                or spec['generation_spec'] != {'container': 'mp4',
                    'video': 'deterministic solid color with burned-in owned clip ID',
                    'audio': 'silence', 'color': color}):
            raise ValueError('Unsupported, duplicate or undeclared media specification')
    return copy.deepcopy(specs)


def _run(command: list[str], *, limit: int = MAX_FILE, timeout: int = 90) -> bytes:
    result = subprocess.run(command, capture_output=True, timeout=timeout,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'AV_LOG_FORCE_NOCOLOR': '1'}, check=False)
    if result.returncode or len(result.stdout) > limit or len(result.stderr) > 1024 * 1024:
        raise ValueError('Bounded CPU media command failed: ' + Path(command[0]).name)
    return result.stdout


def _tool(name: str) -> dict:
    found = shutil.which(name)
    if found is None:
        raise ValueError('Missing CPU media tool: ' + name)
    path = Path(found).resolve(strict=True)
    version = _run([str(path), '-version'], limit=1024 * 1024)
    return {'path': str(path), 'sha256': sha(path.read_bytes()),
            'version': version.decode().splitlines()[0], 'build_sha256': sha(version)}


def _toolchain(ffmpeg: str, ffprobe: str) -> dict:
    tools = {'ffmpeg': _tool(ffmpeg), 'ffprobe': _tool(ffprobe)}
    output = _run(['ldd', tools['ffmpeg']['path']], limit=1024 * 1024).decode()
    selected = ('libavcodec.', 'libavformat.', 'libavutil.', 'libswscale.', 'libswresample.', 'libx264.')
    libraries = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == '=>' and parts[0].startswith(selected):
            path = Path(parts[2]).resolve(strict=True)
            libraries[path.name] = {'path': str(path), 'sha256': sha(path.read_bytes())}
    if not any(name.startswith('libx264.') for name in libraries):
        raise ValueError('The CPU H.264 encoder library could not be pinned')
    return {**tools, 'codec_libraries': libraries}


def _frame(spec: dict) -> bytes:
    rgb = bytes(COLORS[spec['generation_spec']['color']])
    pixels = bytearray(rgb * (WIDTH * HEIGHT))
    for y in range(TOP - 16, TOP + 7 * SCALE + 16):
        for x in range(LEFT - 16, LEFT + 35 * SCALE + 16):
            at = (y * WIDTH + x) * 3
            pixels[at:at + 3] = b'\0\0\0'
    for character, letter in enumerate(spec['clip_id']):
        for row, pattern in enumerate(FONT[letter]):
            for col, bit in enumerate(pattern):
                if bit == '1':
                    for y in range(TOP + row * SCALE, TOP + (row + 1) * SCALE):
                        x = LEFT + character * 6 * SCALE + col * SCALE
                        at = (y * WIDTH + x) * 3
                        pixels[at:at + SCALE * 3] = b'\xff' * (SCALE * 3)
    return f'P6\n{WIDTH} {HEIGHT}\n255\n'.encode() + pixels


def _identity(frame: bytes) -> str:
    inverse = {tuple(pattern): letter for letter, pattern in FONT.items()}
    result = ''
    for character in range(6):
        pattern = []
        for row in range(7):
            bits = ''
            for col in range(5):
                x = LEFT + character * 6 * SCALE + col * SCALE + SCALE // 2
                y = TOP + row * SCALE + SCALE // 2
                rgb = frame[(y * WIDTH + x) * 3:(y * WIDTH + x) * 3 + 3]
                if min(rgb) >= 210:
                    bits += '1'
                elif max(rgb) <= 40:
                    bits += '0'
                else:
                    raise ValueError('Burned-in identity pixels are ambiguous')
            pattern.append(bits)
        if tuple(pattern) not in inverse:
            raise ValueError('Burned-in identity does not match the pinned owned font')
        result += inverse[tuple(pattern)]
    return result


def inspect_media(path: Path, spec: dict, *, ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe') -> dict:
    """Independent decoder readback of actual streams, color, glyphs and silence."""
    before = bounded_bytes(path)
    probe = json.loads(_run([ffprobe, '-v', 'error', '-count_frames', '-show_streams',
                            '-show_format', '-of', 'json', str(path)], limit=1024 * 1024))
    streams = probe['streams']
    video = [item for item in streams if item['codec_type'] == 'video']
    audio = [item for item in streams if item['codec_type'] == 'audio']
    seconds = spec['duration_seconds']
    if len(streams) != 2 or len(video) != 1 or len(audio) != 1:
        raise ValueError('Expected exactly one video and one silent audio stream')
    v, a = video[0], audio[0]
    if (v['codec_name'] != 'h264' or v['pix_fmt'] != 'yuv420p'
            or (v['width'], v['height']) != (WIDTH, HEIGHT)
            or Fraction(v['avg_frame_rate']) != FPS or int(v['nb_read_frames']) != seconds * FPS
            or Fraction(v['time_base']) * int(v['duration_ts']) != seconds
            or a['codec_name'] != 'aac' or int(a['sample_rate']) != RATE or a['channels'] != 1
            or Fraction(a['time_base']) * int(a['duration_ts']) != seconds
            or abs(float(probe['format']['duration']) - seconds) > 0.000001
            or 'mp4' not in probe['format']['format_name'].split(',')):
        raise ValueError('Actual media codec, duration or frame contract differs')
    indices = [0, seconds * FPS // 2, seconds * FPS - 1]
    select = '+'.join(f'eq(n\\,{index})' for index in indices)
    frames = _run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-i', str(path),
        '-map', '0:v:0', '-vf', 'select=' + select, '-vsync', '0', '-pix_fmt', 'rgb24',
        '-f', 'rawvideo', '-threads', '1', 'pipe:1'])
    size = WIDTH * HEIGHT * 3
    if len(frames) != 3 * size:
        raise ValueError('Actual first/middle/last frame readback is incomplete')
    readings = []
    for index, start in zip(indices, range(0, len(frames), size), strict=True):
        frame = frames[start:start + size]
        identity = _identity(frame)
        corner = tuple(frame[(8 * WIDTH + 8) * 3:(8 * WIDTH + 8) * 3 + 3])
        expected = COLORS[spec['generation_spec']['color']]
        if identity != spec['clip_id'] or any(abs(x - y) > 8 for x, y in zip(corner, expected, strict=True)):
            raise ValueError('Decoded clip identity or color differs from its public declaration')
        readings.append({'frame_index': index, 'decoded_clip_id': identity, 'rgb': corner,
                         'rgb_sha256': sha(frame)})
    pcm = _run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-i', str(path),
        '-map', '0:a:0', '-ac', '1', '-ar', str(RATE), '-f', 's16le', 'pipe:1'])
    samples = array.array('h', pcm)
    if sys.byteorder != 'little':
        samples.byteswap()
    peak = max(map(abs, samples), default=-1)
    if peak != 0 or not seconds * RATE <= len(samples) <= seconds * RATE + 1023:
        raise ValueError('Actual decoded audio is not bounded digital silence')
    if bounded_bytes(path) != before:
        raise ValueError('Media changed during independent readback')
    return {'sha256': sha(before), 'size_bytes': len(before), 'container_duration_seconds': seconds,
        'video_frames': seconds * FPS, 'frames': readings, 'audio_samples_decoded': len(samples),
        'audio_peak_abs_pcm16': peak, 'audio_pcm_sha256': sha(pcm),
        'audio_padding_note': 'AAC decoder padding bounded by one1024-sample frame; declared duration verified separately',
        'ffprobe': probe}


def build_media_cache(case: dict, cache_root: Path, *, ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe') -> dict:
    """Create a NEW frozen cache; never repair or overwrite a prior cache silently."""
    specs = specifications(case)
    if cache_root.exists() or cache_root.is_symlink():
        raise ValueError('Generate into a new exclusive owned media cache')
    cache_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_path(cache_root.parent, cache_root.name)
    toolchain = _toolchain(ffmpeg, ffprobe)
    with tempfile.TemporaryDirectory(prefix='.owned-media-', dir=cache_root.parent) as temporary:
        staging = Path(temporary)
        assets = []
        for spec in specs:
            ppm = staging / (spec['clip_id'] + '.ppm')
            ppm.write_bytes(_frame(spec))
            target = staging / Path(spec['path']).name
            args = [toolchain['ffmpeg']['path'], '-v', 'error', '-nostdin', '-n', '-cpuflags', '0',
                '-threads', '1', '-filter_threads', '1', '-loop', '1', '-framerate', str(FPS), '-i', str(ppm),
                '-f', 'lavfi', '-i', f'anullsrc=r={RATE}:cl=mono', '-t', str(spec['duration_seconds']),
                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-profile:v', 'main',
                '-pix_fmt', 'yuv420p', '-qp', '18', '-g', str(FPS), '-bf', '0', '-sc_threshold', '0',
                '-x264-params', RECIPE['x264_parameters'], '-threads:v', '1', '-threads:a', '1',
                '-c:a', 'aac', '-b:a', '64k', '-fflags', '+bitexact', '-flags:v', '+bitexact',
                '-flags:a', '+bitexact', '-map_metadata', '-1', '-metadata', 'encoder=',
                '-metadata', 'creation_time=' + RECIPE['creation_time'], '-metadata:s:v', 'encoder=',
                '-metadata:s:a', 'encoder=', '-movflags', '+faststart', '-video_track_timescale', '1000', str(target)]
            _run(args)
            readback = inspect_media(target, spec, ffmpeg=ffmpeg, ffprobe=ffprobe)
            assets.append({'specification': spec, 'file': target.name, 'source_frame_sha256': sha(ppm.read_bytes()),
                           'readback': readback})
            ppm.unlink()
        manifest = {'schema': 'robit.omni.owned-media-cache.v1', 'declared_manifest_sha256': DECLARED_SHA256,
            'recipe': RECIPE, 'recipe_sha256': sha(_json(RECIPE)), 'toolchain': toolchain, 'assets': assets,
            'input_only': True, 'solver_method_or_completion_evidence': False}
        content = _json(manifest)
        (staging / 'media-cache.json').write_bytes(content)
        for path in staging.iterdir():
            path.chmod(0o444)
            with path.open('rb') as f:
                os.fsync(f.fileno())
        # The parent is owner-owned and the final path was required absent; an
        # exclusive directory reservation prevents replacing another cache.
        cache_root.mkdir(mode=0o700, exist_ok=False)
        for path in staging.iterdir():
            os.rename(path, cache_root / path.name)
        descriptor = os.open(cache_root, os.O_RDONLY)
        os.fsync(descriptor)
        os.close(descriptor)
    return {'schema': 'robit.omni.owned-media-generation.v1', 'cache_root': str(cache_root.absolute()),
            'cache_manifest_sha256': sha(content), 'asset_hashes': {a['specification']['path']: a['readback']['sha256'] for a in assets},
            'input_only': True, 'task_progress': False, 'GUI_inputs': 0, 'model_calls': 0}


def verify_media_cache(case: dict, cache_root: Path, *, expected_manifest_sha256: str,
                       ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe') -> dict:
    specs = specifications(case)
    safe_path(cache_root.parent, cache_root.name)
    if not cache_root.is_dir() or cache_root.stat().st_uid != os.getuid():
        raise ValueError('Expected an owned regular cache directory')
    expected_files = {'media-cache.json', *(Path(s['path']).name for s in specs)}
    if {p.name for p in cache_root.iterdir()} != expected_files:
        raise ValueError('Missing, duplicate, partial or undeclared cache assets')
    content = bounded_bytes(cache_root / 'media-cache.json')
    if sha(content) != expected_manifest_sha256:
        raise ValueError('Frozen media cache manifest changed')
    manifest = json.loads(content)
    if (manifest.get('schema') != 'robit.omni.owned-media-cache.v1'
            or manifest.get('declared_manifest_sha256') != DECLARED_SHA256
            or manifest.get('recipe_sha256') != sha(_json(RECIPE)) or _json(manifest.get('recipe')) != _json(RECIPE)
            or len(manifest.get('assets', [])) != 6):
        raise ValueError('Frozen media recipe or asset inventory changed')
    verified = {}
    for spec, item in zip(specs, manifest['assets'], strict=True):
        path = safe_path(cache_root, Path(spec['path']).name)
        if (item.get('specification') != spec or item.get('file') != path.name
                or path.stat().st_nlink != 1 or sha(bounded_bytes(path)) != item['readback']['sha256']):
            raise ValueError('Frozen media specification or bytes changed')
        actual = inspect_media(path, spec, ffmpeg=ffmpeg, ffprobe=ffprobe)
        if actual['sha256'] != item['readback']['sha256']:
            raise ValueError('Readback is not bound to frozen media bytes')
        verified[spec['path']] = {'sha256': actual['sha256'], 'size_bytes': actual['size_bytes'],
            'duration_seconds': actual['container_duration_seconds'], 'decoded_clip_ids': [f['decoded_clip_id'] for f in actual['frames']],
            'audio_peak_abs_pcm16': actual['audio_peak_abs_pcm16']}
    return {'schema': 'robit.omni.owned-media-capability.v1', 'case_id': case['case_id'],
        'task_signature_sha256': case['task_signature_sha256'], 'cache_root': str(cache_root.absolute()),
        'cache_manifest_sha256': expected_manifest_sha256, 'declared_manifest_sha256': DECLARED_SHA256,
        'assets': verified, 'status': 'verified', 'input_only': True, 'task_progress': False,
        'solver_method_or_completion_evidence': False, 'GUI_inputs': 0, 'model_calls': 0}


def materialize_media(case: dict, work: Path, cache_root: Path, *, expected_manifest_sha256: str,
                      ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe') -> tuple[dict[str, str], dict]:
    """Copy exact verified inputs exclusively; cache/evaluator receipts stay outside Workbench."""
    capability = verify_media_cache(case, cache_root, expected_manifest_sha256=expected_manifest_sha256,
                                    ffmpeg=ffmpeg, ffprobe=ffprobe)
    if (not work.is_dir() or work.is_symlink() or work.stat().st_uid != os.getuid()
            or sha(bounded_bytes(safe_path(work, 'Input/clip-manifest.json'))) != DECLARED_SHA256):
        raise ValueError('Media inputs require the unchanged declared owned Workbench')
    destinations = [safe_path(work, spec['path']) for spec in specifications(case)]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise ValueError('Never overwrite an existing or partial media fixture')
    inventory, copied = {}, []
    for spec, target in zip(specifications(case), destinations, strict=True):
        source = safe_path(cache_root, target.name)
        content = bounded_bytes(source)
        expected = capability['assets'][spec['path']]['sha256']
        if sha(content) != expected:
            raise ValueError('Source media changed after capability verification')
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        target.chmod(0o444)
        if (sha(bounded_bytes(target)) != expected or target.stat().st_nlink != 1
                or (source.stat().st_dev, source.stat().st_ino) == (target.stat().st_dev, target.stat().st_ino)):
            raise ValueError('Copied media is not an independent exact regular file')
        inventory[spec['path']] = expected
        copied.append({'path': spec['path'], 'sha256': expected, 'independent_regular_copy': True})
    return inventory, {**capability, 'materialized_inputs': copied}
