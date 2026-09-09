# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate the simulated monitor geometry and fixed-frame contract."""

import math
from pathlib import Path
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest


MONITOR_THICKNESS = 0.03
MONITOR_WIDTH = 0.30
MONITOR_HEIGHT = 0.20
MONITOR_MASS = 0.50
ABS_TOL = 1.0e-9


def _vector(text):
    return tuple(float(value) for value in text.split())


def _matmul(matrix, vector):
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _add(left, right):
    return tuple(left[index] + right[index] for index in range(3))


def _rpy_matrix(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _assert_vector_close(actual, expected, tolerance=ABS_TOL):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=tolerance,
        )


def _binary_stl_vertices(path):
    vertices = []
    with path.open('rb') as mesh_file:
        mesh_file.read(80)
        triangle_count = struct.unpack('<I', mesh_file.read(4))[0]
        for _ in range(triangle_count):
            triangle = struct.unpack('<12fH', mesh_file.read(50))
            vertices.extend(
                (
                    (triangle[3], triangle[4], triangle[5]),
                    (triangle[6], triangle[7], triangle[8]),
                    (triangle[9], triangle[10], triangle[11]),
                )
            )
    return vertices


@pytest.fixture(scope='module')
def package_share():
    return Path(get_package_share_directory('face_tracking_arm'))


@pytest.fixture(scope='module')
def expanded_robot_description(package_share):
    xacro_path = package_share / 'description' / 'lite6_table.urdf.xacro'
    result = subprocess.run(
        ['xacro', str(xacro_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture(scope='module')
def robot(expanded_robot_description):
    return ET.fromstring(expanded_robot_description)


def test_expanded_description_passes_check_urdf(expanded_robot_description):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf') as urdf_file:
        urdf_file.write(expanded_robot_description)
        urdf_file.flush()
        subprocess.run(
            ['check_urdf', urdf_file.name],
            check=True,
            capture_output=True,
            text=True,
        )


def test_monitor_visual_and_collision_match(robot):
    monitor_link = robot.find("./link[@name='monitor_link']")
    assert monitor_link is not None

    expected_size = (MONITOR_THICKNESS, MONITOR_WIDTH, MONITOR_HEIGHT)
    visual_box = monitor_link.find('./visual/geometry/box')
    collision_box = monitor_link.find('./collision/geometry/box')
    assert visual_box is not None
    assert collision_box is not None
    _assert_vector_close(_vector(visual_box.attrib['size']), expected_size)
    _assert_vector_close(_vector(collision_box.attrib['size']), expected_size)
    _assert_vector_close(
        _vector(monitor_link.find('./visual/origin').attrib['xyz']),
        (0.0, 0.0, 0.0),
    )
    _assert_vector_close(
        _vector(monitor_link.find('./collision/origin').attrib['xyz']),
        (0.0, 0.0, 0.0),
    )


def test_monitor_inertia_matches_solid_box(robot):
    monitor_link = robot.find("./link[@name='monitor_link']")
    mass_element = monitor_link.find('./inertial/mass')
    inertia_element = monitor_link.find('./inertial/inertia')
    assert mass_element is not None
    assert inertia_element is not None
    _assert_vector_close(
        _vector(monitor_link.find('./inertial/origin').attrib['xyz']),
        (0.0, 0.0, 0.0),
    )

    mass = float(mass_element.attrib['value'])
    assert math.isclose(mass, MONITOR_MASS, rel_tol=0.0, abs_tol=ABS_TOL)

    expected_moments = (
        mass * (MONITOR_WIDTH**2 + MONITOR_HEIGHT**2) / 12.0,
        mass * (MONITOR_THICKNESS**2 + MONITOR_HEIGHT**2) / 12.0,
        mass * (MONITOR_THICKNESS**2 + MONITOR_WIDTH**2) / 12.0,
    )
    actual_moments = tuple(
        float(inertia_element.attrib[name]) for name in ('ixx', 'iyy', 'izz')
    )
    _assert_vector_close(actual_moments, expected_moments)
    assert all(moment > 0.0 for moment in actual_moments)
    assert all(
        math.isclose(
            float(inertia_element.attrib[name]),
            0.0,
            rel_tol=0.0,
            abs_tol=ABS_TOL,
        )
        for name in ('ixy', 'ixz', 'iyz')
    )


def test_monitor_mount_and_control_frame_contract(robot):
    mount_joint = robot.find("./joint[@name='monitor_mount_joint']")
    control_joint = robot.find("./joint[@name='monitor_control_joint']")
    assert mount_joint is not None
    assert control_joint is not None
    assert mount_joint.attrib['type'] == 'fixed'
    assert control_joint.attrib['type'] == 'fixed'
    assert mount_joint.find('parent').attrib['link'] == 'link6'
    assert mount_joint.find('child').attrib['link'] == 'monitor_link'
    assert control_joint.find('parent').attrib['link'] == 'monitor_link'
    assert control_joint.find('child').attrib['link'] == 'monitor_control_frame'

    mount_origin = mount_joint.find('origin')
    mount_translation = _vector(mount_origin.attrib['xyz'])
    mount_rotation = _rpy_matrix(*_vector(mount_origin.attrib['rpy']))
    control_translation = _vector(control_joint.find('origin').attrib['xyz'])

    _assert_vector_close(
        _matmul(mount_rotation, (1.0, 0.0, 0.0)),
        (0.0, 0.0, 1.0),
    )
    _assert_vector_close(
        _matmul(mount_rotation, (0.0, 0.0, 1.0)),
        (1.0, 0.0, 0.0),
    )
    _assert_vector_close(
        _matmul(mount_rotation, (0.0, 1.0, 0.0)),
        (0.0, -1.0, 0.0),
    )
    _assert_vector_close(
        _vector(control_joint.find('origin').attrib['rpy']),
        (0.0, 0.0, 0.0),
    )

    rear_surface = _add(
        mount_translation,
        _matmul(mount_rotation, (-MONITOR_THICKNESS / 2.0, 0.0, 0.0)),
    )
    front_surface = _add(
        mount_translation,
        _matmul(mount_rotation, control_translation),
    )
    _assert_vector_close(rear_surface, (0.0, 0.0, 0.0))
    _assert_vector_close(front_surface, (0.0, 0.0, MONITOR_THICKNESS))


def test_monitor_starts_outside_link6_collision_volume(package_share, robot):
    mesh_path = (
        package_share
        / 'description'
        / 'ufactory_lite6'
        / 'meshes'
        / 'lite6'
        / 'collision'
        / 'link6.stl'
    )
    link6_vertices = _binary_stl_vertices(mesh_path)
    minimum_z = min(vertex[2] for vertex in link6_vertices)
    maximum_z = max(vertex[2] for vertex in link6_vertices)
    assert minimum_z < -ABS_TOL
    assert maximum_z <= ABS_TOL

    mount_joint = robot.find("./joint[@name='monitor_mount_joint']")
    mount_translation = _vector(mount_joint.find('origin').attrib['xyz'])
    mount_rotation = _rpy_matrix(
        *_vector(mount_joint.find('origin').attrib['rpy'])
    )
    monitor_z_values = []
    for x_value in (-MONITOR_THICKNESS / 2.0, MONITOR_THICKNESS / 2.0):
        for y_value in (-MONITOR_WIDTH / 2.0, MONITOR_WIDTH / 2.0):
            for z_value in (-MONITOR_HEIGHT / 2.0, MONITOR_HEIGHT / 2.0):
                corner = _add(
                    mount_translation,
                    _matmul(mount_rotation, (x_value, y_value, z_value)),
                )
                monitor_z_values.append(corner[2])

    assert min(monitor_z_values) >= -ABS_TOL
    assert math.isclose(
        max(monitor_z_values),
        MONITOR_THICKNESS,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )


def test_monitor_has_clearance_from_link5(package_share, robot):
    mesh_path = (
        package_share
        / 'description'
        / 'ufactory_lite6'
        / 'meshes'
        / 'lite6'
        / 'collision'
        / 'link5.stl'
    )
    joint6 = robot.find("./joint[@name='joint6']")
    joint_translation = _vector(joint6.find('origin').attrib['xyz'])
    joint_rotation = _rpy_matrix(*_vector(joint6.find('origin').attrib['rpy']))
    inverse_rotation = tuple(zip(*joint_rotation))

    link5_vertices_in_link6 = (
        _matmul(
            inverse_rotation,
            tuple(
                coordinate - translation
                for coordinate, translation in zip(vertex, joint_translation)
            ),
        )
        for vertex in _binary_stl_vertices(mesh_path)
    )
    maximum_link5_z = max(vertex[2] for vertex in link5_vertices_in_link6)
    assert maximum_link5_z < -ABS_TOL
