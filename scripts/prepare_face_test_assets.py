#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Fetch only the static person and its referenced textures, at a pinned revision."""

import argparse
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


REVISION = '8163eb4b5e7e21985c6591d1c0bfb56468c0093f'
SOURCE = f'https://raw.githubusercontent.com/osrf/gazebo_models/{REVISION}/'


def fetch(relative, destination):
    """Keep existing cached files; publish a complete download atomically."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE + relative, timeout=60) as response:
        data = response.read()
    temporary = destination.with_suffix(destination.suffix + '.part')
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f'{relative}: {len(data)} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cache-dir', type=Path,
        default=Path.home() / '.cache/face_tracking_arm')
    args = parser.parse_args()
    root = args.cache_dir / 'gazebo_models'
    for relative in ('model.config', 'model.sdf', 'meshes/standing.dae'):
        fetch('person_standing/' + relative, root / 'person_standing' / relative)
    mesh = root / 'person_standing/meshes/standing.dae'
    tree = ET.parse(mesh)
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    for element in tree.findall('.//c:library_images/c:image/c:init_from', ns):
        filename = Path(element.text).name
        relative = 'person_standing/materials/textures/' + filename
        fetch(relative, root / relative)
    fetch('LICENSE', root / 'person_standing/LICENSE')
    (root / 'person_standing/ATTRIBUTION.txt').write_text(
        'Standing person by Marina Kollmitz; created with MakeHuman.\n'
        f'Source: https://github.com/osrf/gazebo_models/tree/{REVISION}/person_standing\n'
        'CC BY 3.0: https://creativecommons.org/licenses/by/3.0/\n'
        'Downloaded without modifications.\n')
    print(f'Ready: {root}')


if __name__ == '__main__':
    main()
