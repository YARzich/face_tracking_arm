# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Appearance timing, attached camera geometry and preservation of the table world."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import xacro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from face_tracking_perception.appearance_motion import AppearanceMotion  # noqa: E402, I100
from face_tracking_perception.appearance_motion import companion_center  # noqa: E402
from face_tracking_perception.appearance_sequence import AppearanceSequence  # noqa: E402
from face_tracking_perception.arm_scene import write_vision_world  # noqa: E402


def test_appearance_waits_for_readiness_rest_and_pose_acknowledgement():
    seq = AppearanceSequence(3, visible_sec=12, rest_sec=8)
    assert seq.action(0, False, False) is None
    assert seq.action(8, True, False) is None
    assert seq.action(9, True, True) is None
    assert seq.action(9.5, True, True) == 'show'
    assert seq.phase == 'WAIT_REST'  # Unacknowledged/failed service does not advance time.
    seq.acknowledge('show', 10)
    assert seq.action(21.9, True, False) is None
    assert seq.action(22, True, False) == 'hide'
    seq.acknowledge('hide', 22)
    assert seq.index == 1
    assert seq.action(31, True, False) is None
    assert seq.action(32, True, True) is None
    assert seq.action(32.5, True, True) == 'show'


def test_sequence_finishes_hidden_and_resets_after_clock_jump():
    seq = AppearanceSequence(1, visible_sec=1, rest_sec=0, cycles=1)
    seq.action(0, True, True)
    assert seq.action(.5, True, True) == 'show'
    seq.acknowledge('show', .5)
    assert seq.action(1.5, True, False) == 'hide'
    seq.acknowledge('hide', 1.5)
    assert seq.phase == 'COMPLETE'
    assert seq.action(100, True, True) is None
    assert seq.action(0, True, True) == 'hide'
    seq.acknowledge('hide', 0)
    assert seq.phase == 'WAIT_REST'
    assert seq.appearances == 0


def test_safety_readiness_interruption_restarts_rest_confirmation():
    seq = AppearanceSequence(2, rest_sec=0)
    seq.action(0, True, True)
    assert seq.action(.4, False, True) is None
    assert seq.action(.5, True, True) is None
    assert seq.action(.9, True, True) is None
    assert seq.action(1.0, True, True) == 'show'


@pytest.mark.parametrize('visible,rest', [(0, 1), (1, -1), (math.nan, 1)])
def test_invalid_timing_is_rejected(visible, rest):
    with pytest.raises(ValueError):
        AppearanceSequence(3, visible, rest)


def test_person_motion_starts_at_rest_and_stays_within_configured_amplitudes():
    center = np.array([2.25, -.35])
    motion = AppearanceMotion()
    initial = np.array(motion.pose(center, 0))
    np.testing.assert_allclose(initial[:2], center)
    # No instantaneous velocity at appearance, and no corner after the start ramp.
    assert np.linalg.norm((np.array(motion.pose(center, 1e-4)) - initial) / 1e-4) < 1e-6
    left = (np.array(motion.pose(center, 1)) - motion.pose(center, .9999)) / .0001
    right = (np.array(motion.pose(center, 1.0001)) - motion.pose(center, 1)) / .0001
    np.testing.assert_allclose(left, right, atol=1e-4)
    poses = np.array([motion.pose(center, t) for t in np.linspace(0, 12, 361)])
    radial = center / np.linalg.norm(center)
    lateral = np.array([-radial[1], radial[0]])
    offsets = poses[:, :2] - center
    assert np.max(np.abs(offsets @ radial)) <= motion.depth_m + 1e-12
    assert np.max(np.abs(offsets @ lateral)) <= motion.lateral_m + 1e-12
    assert np.ptp(offsets @ lateral) > .27  # It actually moves in both directions.
    facing = np.arctan2(poses[:, 1], poses[:, 0]) - math.pi / 2
    assert np.max(np.abs(poses[:, 2] - facing)) <= motion.yaw_rad + 1e-12
    assert np.max(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)) < .008


def test_zero_motion_amplitudes_preserve_stationary_targets():
    motion = AppearanceMotion(0, 0, 0)
    np.testing.assert_allclose(motion.pose((2.35, 0), 0), motion.pose((2.35, 0), 123))


def test_companion_approaches_smoothly_and_becomes_unambiguously_closer():
    center = np.array([2.35, 0])
    assert np.linalg.norm(companion_center(center, 0)) > np.linalg.norm(center) + .5
    assert np.linalg.norm(companion_center(center, 8)) < np.linalg.norm(center) - .5
    for endpoint in (3, 8):
        before = np.array(companion_center(center, endpoint - .0001))
        after = np.array(companion_center(center, endpoint + .0001))
        assert np.linalg.norm(after - before) / .0002 < 1e-6


def test_acknowledged_motion_does_not_extend_visibility_or_change_person():
    from face_tracking_perception.appearance_node import FaceAppearances

    sequence = AppearanceSequence(3, visible_sec=12)
    sequence.acknowledge('show', 0)
    futures = [SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True))]
    node = SimpleNamespace(sequence=sequence, pending=futures, seconds=lambda: 11.9)
    FaceAppearances.on_pose_result(node, futures, 'move')
    assert node.pending is None
    assert sequence.changed_at == 0
    assert sequence.index == 0 and sequence.appearances == 1
    assert sequence.action(12, True, False) == 'hide'


def test_search_scenario_waits_for_discovery_and_does_not_restart_motion():
    sequence = AppearanceSequence(1, visible_sec=12, rest_sec=8, await_face=True)
    sequence.acknowledge('show', 10)
    assert sequence.phase == 'ACQUIRING'
    assert sequence.action(30, True, False) is None
    assert sequence.action(31, True, False, tracking_face=True) is None
    assert sequence.phase == 'VISIBLE'
    assert sequence.shown_at == 10
    assert sequence.action(42.9, True, False) is None
    assert sequence.action(43, True, False) == 'hide'


def test_search_scenario_reports_a_person_never_discovered():
    sequence = AppearanceSequence(1, await_face=True)
    sequence.acknowledge('show', 10)
    assert sequence.action(70, True, False) == 'hide'
    sequence.acknowledge('hide', 70)
    assert sequence.discovery_timeouts == 1


@pytest.mark.parametrize('options', [
    {'lateral_m': -1}, {'depth_m': math.nan}, {'period_sec': 0}, {'yaw_rad': math.inf}])
def test_invalid_motion_is_rejected(options):
    with pytest.raises(ValueError):
        AppearanceMotion(**options)


def fixed_pose(robot, parent, child):
    joint = next(j for j in robot.findall('joint') if j.find('child').get('link') == child)
    origin = joint.find('origin')
    transform = np.eye(4)
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        transform[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    joint_parent = joint.find('parent').get('link')
    if joint_parent == parent:
        return transform
    return fixed_pose(robot, parent, joint_parent) @ transform


@pytest.mark.parametrize('mode', ['disabled', 'mono_cpu', 'stereo'])
def test_camera_optical_axes_height_and_baseline_match_sensor_calibration(mode):
    robot = ET.fromstring(xacro.process_file(
        str(ROOT / 'description/lite6_table.urdf.xacro'), mappings={'camera_mode': mode}).toxml())
    sensors = robot.findall('gazebo/sensor')
    assert len(sensors) == {'disabled': 0, 'mono_cpu': 1, 'stereo': 2}[mode]
    if mode == 'disabled':
        assert robot.find("link[@name='monitor_camera_bar']") is None
        return
    left = fixed_pose(robot, 'monitor_link', 'face_test_camera_optical_frame')
    assert left[2, 3] > .10  # Above the top edge of the 20 cm screen.
    np.testing.assert_allclose(left[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(left[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-9)
    if mode == 'stereo':
        right = fixed_pose(robot, 'monitor_link', 'face_test_right_optical_frame')
        np.testing.assert_allclose((np.linalg.inv(left) @ right)[:3, 3], [.08, 0, 0], atol=1e-9)
        calibration = sensors[1].find('camera/lens/projection')
        assert -float(calibration.findtext('tx')) / float(calibration.findtext('p_fx')) == .08


@pytest.mark.parametrize('people_count', [1, 2])
def test_world_preserves_robot_table_and_hides_person_initially(tmp_path, people_count):
    base = ROOT / 'worlds/lite6_table.sdf'
    world_path = tmp_path / 'world.sdf'
    write_vision_world(base, world_path, people_count)
    original, result = ET.parse(base).getroot(), ET.parse(world_path).getroot()
    assert result.find('world').get('name') == 'lite6_table'
    for name in ['round_table', 'ground_plane']:
        query = f"world/model[@name='{name}']"
        assert ET.tostring(original.find(query)) == ET.tostring(result.find(query))
    assert result.findtext('world/include/pose').split()[2] == '-10'
    assert result.findtext('world/include/uri') == 'model://person_standing'
    people = result.findall('world/include')
    assert len(people) == people_count
    assert len({person.findtext('name') for person in people}) == people_count
    assert all(person.findtext('pose').split()[2] == '-10' for person in people)
