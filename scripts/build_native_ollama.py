#!/usr/bin/env python3
"""Build the pinned experimental native Omni Ollama runner in an isolated tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '04088e7c9d41cba5d33bafbf788d87a465c79d5b'
ARCHIVE_SHA256 = '17c24d7268a286722d8e02d0f1d940a94922ecb73c20fb4377d2ebd7b1f9640c'
TOOLCHAIN = 'go1.26.10'
VERSION = '0.40.13-robit-omni.5'

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, default=ROOT / 'runtime-data/native-ollama')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--cuda-backend', choices=['cuda_v12', 'cuda_v13'], default='cuda_v13')
    a = p.parse_args()
    work = a.build_dir.absolute()
    # Build payloads/caches are derived. Source, tests and exact patch live in
    # the checkout. This is a CPU build; it must never initialize CUDA.
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 14 * 2**30:
        raise SystemExit('native build requires4GiB forecast plus10GiB disk reserve')
    archive = work / 'source.tar.gz'
    if not archive.exists():
        temporary = work / 'source.tar.gz.partial'
        try:
            with urllib.request.urlopen(
                    f'https://codeload.github.com/ollama/ollama/tar.gz/{COMMIT}', timeout=60) as source, temporary.open('xb') as output:
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 2**20:
                        raise ValueError('source archive exceeded100MiB forecast')
                    output.write(chunk)
            if sha(temporary) != ARCHIVE_SHA256:
                raise ValueError('official pinned source archive SHA256 mismatch')
            temporary.rename(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError('cached source archive SHA256 mismatch')
    source = work / 'source'
    if not source.exists():
        source.mkdir()
        expanded = 0
        with tarfile.open(archive) as tar:
            for entry in tar.getmembers():
                parts = Path(entry.name).parts
                if not parts or '..' in parts or Path(entry.name).is_absolute():
                    raise ValueError('unsafe source archive path')
                if entry.issym() or entry.islnk():
                    continue
                relative = Path(*parts[1:])
                target = source / relative
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    expanded += entry.size
                    if expanded > 512 * 2**20:
                        raise ValueError('source expansion exceeded512MiB forecast')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(entry) as reader, target.open('xb') as output:
                        shutil.copyfileobj(reader, output)
    patch = ROOT / 'native/ollama/ollama-0.40.13.patch'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1',
           'GOTOOLCHAIN': TOOLCHAIN, 'GOMODCACHE': str(work / 'gomodcache'),
           'GOCACHE': str(work / 'gocache'),
           'GIT_CEILING_DIRECTORIES': str(work)}
    def apply(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(['git', 'apply', '--unidiff-zero', *args, str(patch)], cwd=source,
                              env=env, check=check, capture_output=True, text=True)
    fresh = apply('--check', check=False)
    if fresh.returncode == 0:
        apply()
    elif apply('--reverse', '--check', check=False).returncode != 0:
        raise ValueError('source does not match pinned upstream or the exact current overlay')
    for file in (ROOT / 'native/ollama').glob('*/*.go'):
        shutil.copyfile(file, source / file.parent.name / file.name)
    for file in (ROOT / 'native/ollama').glob('*/testdata/*'):
        if file.is_file():
            target = source / file.relative_to(ROOT / 'native/ollama')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    receipt = {'upstream_commit': COMMIT, 'source_archive_sha256': ARCHIVE_SHA256,
               'patch_sha256': sha(patch), 'native_files': {
                   str(file.relative_to(ROOT)): sha(file)
                   for file in [*(ROOT / 'native/ollama').glob('*/*.go'),
                                *(ROOT / 'native/ollama').glob('*/testdata/*')] if file.is_file()},
               'CUDA_started': False, 'release_qualified': False}
    if not a.prepare_only:
        subprocess.run(['go', 'test', './llm', './server', '-count=1'], cwd=source,
                       env=env, check=True)
        binary = work / 'ollama-native'
        subprocess.run(['go', 'build', '-ldflags',
                        f'-X github.com/ollama/ollama/version.Version={VERSION}',
                        '-o', str(binary), '.'], cwd=source, env=env, check=True)
        receipt.update(binary_sha256=sha(binary), binary=str(binary),
                       native_Go_tests_passed=True)
        engine = ROOT / 'vendor/llama.cpp/build/bin'
        if not (engine / 'llama-server').is_file():
            raise ValueError('build the pinned native engines with scripts/build_llama_cpp.sh first')
        payload = work / 'lib/ollama'
        payload.mkdir(parents=True, exist_ok=True)
        for original in [engine / 'llama-server', *engine.glob('*.so*')]:
            link = payload / original.name
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != original.resolve():
                    raise ValueError('existing native payload conflicts with the pinned engines')
            else:
                link.symlink_to(original.absolute())
        backend = payload / a.cuda_backend
        if not backend.exists():
            backend.symlink_to(engine.absolute(), target_is_directory=True)
    (work / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
