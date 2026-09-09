#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Prepare pinned official metric Small source; weights require an explicit flag."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCE_REVISION = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
MODEL_REVISION = '3bc65d4e14a6786a61acec16453c50e12bf5f338'
MODEL_REPO = 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Small'
WEIGHTS = 'depth_anything_v2_metric_hypersim_vits.pth'
WEIGHTS_SHA256 = 'b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545'


def download(url, destination, sha256=None):
    if destination.is_file():
        if sha256 is None or hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
            return
        raise RuntimeError(f'Existing file has a different checksum: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if sha256 and digest.hexdigest() != sha256:
        raise RuntimeError(f'Download checksum mismatch: {temporary}')
    temporary.replace(destination)
    print(f'Ready: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache/face_tracking_arm')
    parser.add_argument('--download-weights', action='store_true',
                        help='Download the 99 MB Apache-2.0 Small checkpoint')
    args = parser.parse_args()
    source = args.cache_dir / 'depth_anything_metric'
    raw = f'https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/{SOURCE_REVISION}/'
    tree_url = ('https://api.github.com/repos/DepthAnything/Depth-Anything-V2/git/trees/'
                f'{SOURCE_REVISION}?recursive=1')
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        tree = json.load(response)
    for item in tree['tree']:
        path = item['path']
        if item['type'] == 'blob' and path.startswith('metric_depth/depth_anything_v2/'):
            download(raw + path, source / path.removeprefix('metric_depth/'))
    download(raw + 'LICENSE', source / 'LICENSE')
    download(raw + 'metric_depth/README.md', source / 'UPSTREAM_README.md')
    (source / 'SOURCE.json').write_text(json.dumps({
        'source_revision': SOURCE_REVISION, 'model_repo': MODEL_REPO,
        'model_revision': MODEL_REVISION, 'weights_sha256': WEIGHTS_SHA256,
        'license': 'Apache-2.0 (Small only)', 'modifications': 'none'}, indent=2) + '\n')
    if args.download_weights:
        download(f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{WEIGHTS}',
                 args.cache_dir / 'models' / WEIGHTS, WEIGHTS_SHA256)
    else:
        print('Weights were not downloaded. Add --download-weights to request the 99 MB file.')


if __name__ == '__main__':
    main()
