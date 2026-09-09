# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Check exact surface preservation by the build-time collision mesh cleanup."""

import importlib.util
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    'deduplicate_collision_mesh',
    PROJECT_ROOT / 'scripts' / 'deduplicate_collision_mesh.py',
)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def _record(vertices):
    return struct.pack('<12fH', 0.0, 0.0, 1.0, *sum(vertices, ()), 0)


def _stl(records):
    header = b'preserved header'.ljust(80, b'\0')
    return header + struct.pack('<I', len(records)) + b''.join(records)


def test_duplicate_winding_and_vertex_order_preserve_first_record():
    vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    first = _record(vertices)
    data = _stl([first, _record(tuple(reversed(vertices))), first])
    assert GENERATOR.deduplicate(data) == _stl([first])


def test_nearby_faces_are_not_rounded_or_removed():
    first = _record(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    second = _record(((1.0e-10, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    data = _stl([first, second])
    assert GENERATOR.deduplicate(data) == data


@pytest.mark.parametrize('data', [b'', b'header'.ljust(80, b'\0'), _stl([]) + b'extra'])
def test_invalid_binary_stl_is_rejected(data):
    with pytest.raises(ValueError):
        GENERATOR.deduplicate(data)


def test_non_finite_geometry_is_rejected():
    data = _stl([_record(((float('nan'), 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))])
    with pytest.raises(ValueError):
        GENERATOR.deduplicate(data)


@pytest.mark.parametrize('link', ['link4', 'link5'])
def test_installed_mesh_has_identical_triangle_set_and_fewer_faces(link):
    share = Path(get_package_share_directory('face_tracking_arm'))
    source = share / 'description' / 'ufactory_lite6' / 'meshes' / 'lite6' / 'collision'
    generated = share / 'description' / 'generated' / 'collision'
    original_records = list(GENERATOR.triangle_records((source / f'{link}.stl').read_bytes()))
    generated_records = list(GENERATOR.triangle_records((generated / f'{link}.stl').read_bytes()))
    assert {key for _, key in original_records} == {key for _, key in generated_records}
    assert len(generated_records) < 0.65 * len(original_records)
    assert GENERATOR.deduplicate((source / f'{link}.stl').read_bytes()) == (
        generated / f'{link}.stl'
    ).read_bytes()


def test_robot_uses_generated_collision_meshes_and_original_visuals():
    share = Path(get_package_share_directory('face_tracking_arm'))
    description = subprocess.run(
        ['xacro', str(share / 'description' / 'lite6_table.urdf.xacro')],
        check=True, capture_output=True, text=True,
    ).stdout
    robot = ET.fromstring(description)
    for name in ('link4', 'link5'):
        link = robot.find(f"./link[@name='{name}']")
        collision = link.find('./collision/geometry/mesh').attrib['filename']
        visual = link.find('./visual/geometry/mesh').attrib['filename']
        assert Path(collision.removeprefix('file://')).resolve() == (
            share / 'description' / 'generated' / 'collision' / f'{name}.stl'
        ).resolve()
        assert '/ufactory_lite6/meshes/lite6/visual/' in visual
