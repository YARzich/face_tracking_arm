#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Freeze package sources and installed libraries outside the package tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {
    'include', 'src', 'msg', 'srv', 'action', 'launch', 'python', 'scripts',
    'config', 'description', 'test', 'tools', 'worlds', 'docs', 'docker',
}
SOURCE_FILES = {
    'CMakeLists.txt', 'package.xml', 'README.md', 'LICENSE', 'setup.py',
    'setup.cfg', 'pytest.ini', 'pyproject.toml', 'AGENTS.md',
}
EXCLUDED_PARTS = {
    '.git', '.idea', '.pytest_cache', '__pycache__', 'build', 'install',
    'log', 'cmake-build-debug', 'cmake-build-release',
}


def run(command: list[str], *, cwd: Path = REPO) -> str:
    """Run a checked command and return its trimmed standard output."""
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> list[str]:
    """Return tracked and non-ignored source paths, excluding build output."""
    listed = run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard']
    ).splitlines()
    selected = []
    for name in listed:
        path = Path(name)
        if not path.parts or any(
                part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.parts[0] in SOURCE_ROOTS or name in SOURCE_FILES:
            if (REPO / path).is_file():
                selected.append(name)
    return sorted(set(selected))


def copy_runtime_libraries(output: Path) -> dict[str, str]:
    """Copy installed package libraries with their original loader names."""
    library_dir = REPO.parent.parent / 'install/face_tracking_arm/lib'
    copied = {}
    if not library_dir.is_dir():
        return copied
    destination_dir = output / 'runtime-libs'
    for library in sorted(library_dir.glob('*.so*')):
        if not library.is_file():
            continue
        destination = destination_dir / library.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, destination)
        copied[library.name] = sha256(destination)
    return copied


def main() -> int:
    """Capture selected source files and metadata into a new directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True,
                        help='new, empty directory outside the source tree')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(
            'output already exists; refusing to overwrite: {}'.format(output))
    if output == REPO or REPO in output.parents:
        parser.error('baseline output must be outside the source tree')

    requested = source_paths()
    present = [
        relative for relative in requested if (
            REPO / relative).is_file()]
    missing = sorted(set(requested) - set(present))
    before = {relative: sha256(REPO / relative) for relative in present}
    output.mkdir(parents=True)
    for relative in present:
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    after = {relative: sha256(REPO / relative) for relative in present}
    changed_during_capture = sorted(
        relative for relative in present if before[relative] != after[relative]
    )

    cpu_model = 'unknown'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    osqp_header = Path('/opt/ros/jazzy/include/osqp/constants.h')
    osqp_version = None
    if osqp_header.exists():
        match = re.search(r'#\s*define\s+OSQP_VERSION\s+\("([^"]+)"\)',
                          osqp_header.read_text())
        osqp_version = match.group(1) if match else None

    runtime_library_hashes = copy_runtime_libraries(output)
    metadata = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(REPO),
        'git_head': run(['git', 'rev-parse', 'HEAD']),
        'git_status': run(['git', 'status', '--short']),
        'platform': platform.platform(),
        'kernel': run(['uname', '-a']),
        'cpu_model': cpu_model,
        'compiler': run(['c++', '--version']).splitlines()[0],
        'osqp_version': osqp_version,
        'source_sha256_before': before,
        'source_sha256_after': after,
        'runtime_library_sha256': runtime_library_hashes,
        'missing_sources': missing,
        'changed_during_capture': changed_during_capture,
        'capture_complete': not missing and not changed_during_capture,
    }
    (output / 'metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n'
    )
    (output / 'git-diff.patch').write_text(
        run(['git', 'diff', '--binary', '--', *present]) + '\n')
    (output / 'README.md').write_text(
        '# Frozen package baseline\n\n'
        'This directory snapshots package sources before a measured change.\n'
        'It includes sources, hashes, Git state and installed libraries.\n'
        'A capture with `capture_complete: false` is not a valid baseline.\n'
        'Do not overwrite a capture; create a new directory for each state.\n'
        'Record limits, scene, scenario, build command and timing.\n'
        'This script does not start ROS or contact a robot.\n'
    )
    print(output)
    print('capture_complete={}'.format(metadata['capture_complete']))
    print(
        'source_files={} missing={} changed={}'.format(
            len(present), len(missing), len(changed_during_capture)))
    return 0 if metadata['capture_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
