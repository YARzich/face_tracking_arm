# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Fast target motion, nearest-person rank and the three-person Gazebo world."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from face_tracking_perception.arm_scene import person_entity_name  # noqa: E402, I100
from face_tracking_perception.arm_scene import write_vision_world  # noqa: E402
from face_tracking_perception.fast_stress_motion import FastStressMotion  # noqa: E402


def test_all_three_people_start_visible_and_background_stays_stationary():
    motion = FastStressMotion()
    for elapsed in (-1, 0, 9.99):
        sample = motion.sample(elapsed)
        assert sample.poses == motion.initial_poses
        assert sample.phase == 'START_HOLD' and not sample.complete
    for elapsed in np.linspace(0, motion.duration_sec, 1001):
        sample = motion.sample(elapsed)
        assert len(sample.poses) == 3
        assert sample.poses[1:] == motion.initial_poses[1:]
        assert all(pose[2] > -.2 for pose in sample.poses)
    with pytest.raises(FrozenInstanceError):
        sample.complete = True


def test_radius_height_and_quick_direction_changes():
    motion = FastStressMotion()
    poses = np.array([motion.sample(t).poses[0] for t in np.linspace(0, 96, 9601)])
    radius = np.linalg.norm(poses[:, :2], axis=1)
    assert radius.min() == pytest.approx(1.4)
    assert radius.max() == pytest.approx(3.6)
    assert (poses[:, 2] + motion.face_reference_height_m).min() == pytest.approx(1.60)
    assert (poses[:, 2] + motion.face_reference_height_m).max() == pytest.approx(1.90)
    assert motion.sample(11).poses[0][1] == pytest.approx(.4)
    assert motion.sample(12.2).poses[0][1] == pytest.approx(-.4)
    assert motion.sample(13.4).poses[0][1] == pytest.approx(.4)
    velocity = np.diff(poses[:, :3], axis=0) / .01
    # 0.8 m over 1.2 s with quintic easing peaks at 1.25 m/s.
    assert np.max(np.linalg.norm(velocity, axis=1)) == pytest.approx(1.25, abs=.002)
    assert np.max(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)) < .013


@pytest.mark.parametrize('boundary', [10, 11, 12.2, 13.4, 14.4, 15.4, 16.6, 17.6,
                                      20, 26, 34, 44, 56, 66, 76, 82, 96])
def test_segment_joins_have_continuous_position_velocity_and_acceleration(boundary):
    motion = FastStressMotion()
    step = .0001
    values = [np.array(motion.sample(boundary + offset * step).poses[0])
              for offset in (-2, -1, 0, 1, 2)]
    before, left, center, right, after = values
    np.testing.assert_allclose(left, center, atol=1e-9)
    np.testing.assert_allclose(right, center, atol=1e-9)
    np.testing.assert_allclose((center - left) / step, 0, atol=1e-5)
    np.testing.assert_allclose((right - center) / step, 0, atol=1e-5)
    np.testing.assert_allclose((center - 2 * left + before) / step**2, 0, atol=.004)
    np.testing.assert_allclose((after - 2 * right + center) / step**2, 0, atol=.004)


def test_all_people_face_the_robot_base():
    motion = FastStressMotion()
    for elapsed in np.linspace(0, 96, 501):
        for x, y, _, yaw in motion.sample(elapsed).poses:
            radius = math.hypot(x, y)
            np.testing.assert_allclose([math.sin(yaw), -math.cos(yaw)],
                                       [-x / radius, -y / radius], atol=1e-12)


@pytest.mark.parametrize('start,end,phase,nearest', [
    (0, 10, 'START_HOLD', 0), (26, 34, 'NEAR_HOLD', 0), (44, 56, 'FAR_HOLD', 1),
    (66, 76, 'NEAR_RETURN_HOLD', 0), (82, 96, 'FINAL_HOLD', 0)])
def test_control_pauses_make_nearest_face_change_with_a_clear_margin(start, end, phase, nearest):
    motion = FastStressMotion()
    sample = motion.sample(start)
    assert end - start >= 5
    assert sample.phase == phase
    assert sample.poses == motion.sample(end - .001).poses
    # The selection frame is the robot base, 0.75 m above the floor.
    faces = np.array([(x, y, z + motion.face_reference_height_m - .75)
                      for x, y, z, _ in sample.poses])
    distances = np.linalg.norm(faces, axis=1)
    assert distances.argmin() == nearest
    assert np.sort(distances)[1] - distances[nearest] > .35


@pytest.mark.parametrize('elapsed', [0, 26, 44, 66, 82])
@pytest.mark.parametrize('camera_x', [.2, .65])
def test_control_pauses_separate_primary_body_from_background_faces(elapsed, camera_x):
    poses = FastStressMotion().sample(elapsed).poses
    main = np.array(poses[0][:2]) - [camera_x, 0]
    main_angle = math.atan2(main[1], main[0])
    # A conservative 0.5 m main-body width must not cover a background face.
    main_half_width = math.asin(.25 / np.linalg.norm(main))
    for pose in poses[1:]:
        background = np.array(pose[:2]) - [camera_x, 0]
        background_angle = math.atan2(background[1], background[0])
        background_half_width = math.asin(.09 / np.linalg.norm(background))
        assert abs(main_angle - background_angle) > main_half_width + background_half_width


def test_repeat_and_finite_completion_leave_people_visible_without_a_pose_jump():
    motion = FastStressMotion()
    for elapsed in (0, 10, 26, 44, 75, 95):
        assert motion.sample(elapsed) == motion.sample(elapsed + motion.duration_sec)
    assert not motion.sample(96, cycles=2).complete
    assert not motion.sample(191.999, cycles=2).complete
    for elapsed in (192, 1000):
        sample = motion.sample(elapsed, cycles=2)
        assert sample.complete and sample.phase == 'COMPLETE'
        assert sample.poses == motion.initial_poses


@pytest.mark.parametrize('elapsed', [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_time(elapsed):
    with pytest.raises(ValueError, match='time'):
        FastStressMotion().sample(elapsed)


@pytest.mark.parametrize('cycles', [-1, .5, True])
def test_rejects_invalid_cycles(cycles):
    with pytest.raises(ValueError, match='cycles'):
        FastStressMotion().sample(0, cycles=cycles)


def test_world_uses_unique_person_names_and_initializes_all_three_poses(tmp_path):
    destination = tmp_path / 'fast.sdf'
    motion = FastStressMotion()
    write_vision_world(ROOT / 'worlds/lite6_table.sdf', destination, 3,
                       initial_person_poses=motion.initial_poses)
    people = ET.parse(destination).getroot().findall('world/include')
    assert [person.findtext('name') for person in people] == [
        'face_test_person', 'face_test_person_2', 'face_test_person_3']
    for index, (person, pose) in enumerate(zip(people, motion.initial_poses)):
        x, y, z, yaw = pose
        assert person.findtext('name') == person_entity_name(index)
        np.testing.assert_allclose(np.fromstring(person.findtext('pose'), sep=' '),
                                   [x, y, z, 0, 0, yaw])


@pytest.mark.parametrize('options', [
    {'initial_person_pose': (1, 0, 0, 0), 'initial_person_poses': [(1, 0, 0, 0)]},
    {'initial_person_poses': [(1, 0, 0, 0)]},
    {'initial_person_poses': [(1, 0, 0, 0), (2, 0, math.inf, 0), (3, 0, 0, 0)]},
    {'initial_person_poses': [(1, 0, 0, 0), None, (3, 0, 0, 0)]},
])
def test_world_rejects_conflicting_or_invalid_initial_poses(tmp_path, options):
    with pytest.raises(ValueError):
        write_vision_world(ROOT / 'worlds/lite6_table.sdf', tmp_path / 'bad.sdf', 3, **options)


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_world_rejects_invalid_people_count(tmp_path, count):
    with pytest.raises(ValueError, match='count'):
        write_vision_world(ROOT / 'worlds/lite6_table.sdf', tmp_path / 'bad.sdf', count)


@pytest.mark.parametrize('index', [-1, .5, True])
def test_entity_name_rejects_invalid_index(index):
    with pytest.raises(ValueError, match='index'):
        person_entity_name(index)
