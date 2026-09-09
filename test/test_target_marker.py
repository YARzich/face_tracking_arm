# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate the Gazebo face-target marker and its ROS adapter contract."""

import ast
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml


MARKER_RADIUS_M = 0.04
HIDDEN_Z_M = -10.0
ABS_TOL = 1.0e-9
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call, name):
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _string_literals(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


@pytest.fixture(scope='module')
def package_share():
    return Path(get_package_share_directory('face_tracking_arm'))


@pytest.fixture(scope='module')
def world(package_share):
    return ET.parse(package_share / 'worlds' / 'lite6_table.sdf').getroot()


def test_marker_is_hidden_visual_only_red_sphere(world):
    marker = world.find("./world/model[@name='face_target_marker']")
    assert marker is not None
    assert marker.findtext('./static') == 'true'

    pose = tuple(float(value) for value in marker.findtext('./pose').split())
    assert len(pose) == 6
    assert math.isclose(pose[2], HIDDEN_Z_M, rel_tol=0.0, abs_tol=ABS_TOL)

    link = marker.find("./link[@name='marker_link']")
    assert link is not None
    assert link.findall('./collision') == []
    assert link.findall('./sensor') == []
    assert link.findall('./plugin') == []

    visuals = link.findall('./visual')
    assert len(visuals) == 1
    sphere = visuals[0].find('./geometry/sphere')
    assert sphere is not None
    assert math.isclose(
        float(sphere.findtext('./radius')),
        MARKER_RADIUS_M,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert visuals[0].findtext('./cast_shadows') == 'false'

    material = visuals[0].find('./material')
    for element_name in ('ambient', 'diffuse', 'emissive'):
        color = tuple(
            float(value) for value in material.findtext(element_name).split()
        )
        assert len(color) == 4
        assert color[0] > 0.0
        assert color[1] == 0.0
        assert color[2] == 0.0
        assert color[3] == 1.0


def test_marker_is_not_part_of_robot_or_moveit_scene(package_share):
    robot_description = (
        package_share / 'description' / 'lite6_table.urdf.xacro'
    ).read_text(encoding='utf-8')
    semantic_description = (
        package_share / 'config' / 'moveit' / 'lite6.srdf'
    ).read_text(encoding='utf-8')
    assert 'face_target_marker' not in robot_description
    assert 'face_target_marker' not in semantic_description


def test_marker_configuration_reuses_tracking_frame_and_freshness(package_share):
    config = _load_yaml(package_share / 'config' / 'tracking.yaml')
    tracking = config['face_tracking_controller']['ros__parameters']
    marker = config['face_target_marker']['ros__parameters']

    assert marker['planning_frame'] == tracking['planning_frame'] == 'world'
    assert math.isclose(
        marker['face_target_freshness_timeout_sec'],
        tracking['face_target_freshness_timeout_sec'],
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert marker['marker_entity_name'] == 'face_target_marker'
    assert marker['set_pose_service'] == '/world/lite6_table/set_pose/blocking'
    assert 0.0 < marker['update_rate_hz'] <= 30.0


def test_tracking_launch_bridges_and_starts_marker_by_default(package_share):
    launch_path = package_share / 'launch' / 'tracking_sim.launch.py'
    syntax_tree = ast.parse(launch_path.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Call)]

    declared_arguments = [
        call
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'visualize_target'
    ]
    assert len(declared_arguments) == 1
    default_value = _keyword(declared_arguments[0], 'default_value')
    assert isinstance(default_value, ast.Constant)
    assert default_value.value == 'true'

    marker_nodes = []
    for call in calls:
        if _call_name(call) != 'Node':
            continue
        literals = _string_literals(call)
        if 'face_target_marker' in literals:
            marker_nodes.append(call)
    assert len(marker_nodes) == 1
    marker_condition = _keyword(marker_nodes[0], 'condition')
    assert isinstance(marker_condition, ast.Call)
    assert _call_name(marker_condition) == 'IfCondition'
    assert 'use_sim_time' in _string_literals(marker_nodes[0])

    launch_literals = _string_literals(syntax_tree)
    assert any(
        'set_pose/blocking' in literal
        for literal in launch_literals
    )
    assert any(
        'ros_gz_interfaces/srv/SetEntityPose' in literal
        for literal in launch_literals
    )


def test_marker_adapter_is_nonblocking_latest_only_and_cleans_timeouts(package_share):
    del package_share
    source_path = PROJECT_ROOT / 'src' / 'face_target_marker_node.cpp'
    assert source_path.exists()
    source = source_path.read_text(encoding='utf-8')

    assert 'KeepLast(1)' in source
    assert '.best_effort()' in source
    assert 'service_is_ready()' in source
    assert 'wait_for_service' not in source
    assert 'async_send_request' in source
    assert 'request_pending_' in source
    assert 'remove_pending_request' in source
    assert 'steady_clock' in source
    assert 'tf_buffer_.transform' in source
    assert 'durationFromSec(0.0)' in source
    assert 'stamp.nanoseconds() <= 0' in source
    assert 'std::chrono::seconds(6)' in source
    assert 'age_sec < 0.0' in source
    assert 'age_sec >= freshness_timeout_sec_' in source
    assert 'observe_clock' in source
    assert 'current_time < *last_observed_time_' in source
    assert 'last_commanded_position_.reset()' in source
