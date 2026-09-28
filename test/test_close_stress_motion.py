# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Close-person geometry, retreat stimulus and continuity of the test route."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from face_tracking_perception.close_stress_motion import CloseStressMotion  # noqa: E402, I100


def test_close_route_stays_visible_at_human_height_and_faces_the_robot():
    motion = CloseStressMotion()
    samples = [motion.sample(t) for t in np.linspace(0, motion.duration_sec, 12001)]
    assert all(sample.visible for sample in samples)
    poses = np.array([sample.pose for sample in samples])
    radius = np.linalg.norm(poses[:, :2], axis=1)
    assert radius.min() == pytest.approx(.75)
    assert radius.max() == pytest.approx(2.1)
    height = poses[:, 2] + motion.face_reference_height_m
    assert height.min() == pytest.approx(1.40)
    assert height.max() == pytest.approx(1.75)
    bearing = np.arctan2(poses[:, 1], poses[:, 0])
    assert math.degrees(bearing.min()) == pytest.approx(-30)
    assert math.degrees(bearing.max()) == pytest.approx(30)
    face_forward = np.column_stack((np.sin(poses[:, 3]), -np.cos(poses[:, 3])))
    np.testing.assert_allclose(face_forward, -poses[:, :2] / radius[:, None], atol=1e-12)
    velocity = np.diff(poses[:, :3], axis=0) / .01
    assert np.max(np.linalg.norm(velocity, axis=1)) < .22


def test_second_approach_requires_a_settled_screen_to_retreat():
    motion = CloseStressMotion()
    preferred_distance = .4
    first_face = np.array(motion.sample(22).pose[:3])
    # Approximate face centre in the standing mesh, facing towards the base.
    first_face += [-.21, 0, motion.face_reference_height_m + .015]
    # A frozen ideal screen at the first stop makes the close-approach stimulus
    # explicit. Actual distance and robot retreat still require a runtime test.
    face_direction = first_face - [0, 0, 1.017]
    frozen_screen = first_face - (
        preferred_distance * face_direction / np.linalg.norm(face_direction))
    close_faces = np.array([motion.sample(t).pose[:3] for t in np.linspace(28, 32, 401)])
    close_faces += [-.21, 0, motion.face_reference_height_m + .015]
    distances = np.linalg.norm(close_faces - frozen_screen, axis=1)
    assert distances[0] == pytest.approx(preferred_distance)
    assert np.all(np.diff(distances) <= 0)
    assert distances[-1] == pytest.approx(.241, abs=.002)
    assert distances[-1] < preferred_distance - .15


@pytest.mark.parametrize('boundary', [10, 22, 28, 32, 42, 50, 60, 68, 76, 84,
                                      94, 100, 110, 120])
def test_segment_and_cycle_joins_have_continuous_position_velocity_and_acceleration(boundary):
    motion = CloseStressMotion()
    step = .001
    values = [np.array(motion.sample(boundary + offset * step).pose)
              for offset in (-2, -1, 0, 1, 2)]
    before, left, center, right, after = values
    np.testing.assert_allclose(left, center, atol=1e-8)
    np.testing.assert_allclose(right, center, atol=1e-8)
    np.testing.assert_allclose((center - left) / step, 0, atol=1e-5)
    np.testing.assert_allclose((right - center) / step, 0, atol=1e-5)
    np.testing.assert_allclose((center - 2 * left + before) / step**2, 0, atol=.001)
    np.testing.assert_allclose((after - 2 * right + center) / step**2, 0, atol=.001)


@pytest.mark.parametrize('start,end,phase', [
    (0, 10, 'START_HOLD'), (22, 28, 'NEAR_HOLD'), (32, 42, 'CLOSE_HOLD'),
    (76, 84, 'LOW_CLOSE_HOLD'), (94, 100, 'RECOVERY_HOLD'), (110, 120, 'FINAL_HOLD')])
def test_stops_leave_time_to_settle_before_another_approach_or_departure(start, end, phase):
    motion = CloseStressMotion()
    assert end - start >= 6
    beginning = motion.sample(start)
    ending = motion.sample(end - .001)
    assert beginning.pose == ending.pose
    assert beginning.phase == ending.phase == phase
    assert beginning.visible and ending.visible


def test_completion_and_repetition_preserve_the_initial_pose():
    motion = CloseStressMotion()
    assert motion.sample(-1).pose == motion.initial_pose
    for elapsed in (0, 10, 22, 32, 60, 94, 119):
        assert motion.sample(elapsed) == motion.sample(elapsed + motion.duration_sec)
    assert not motion.sample(120, cycles=2).complete
    assert not motion.sample(239.999, cycles=2).complete
    for elapsed in (240, 1000):
        sample = motion.sample(elapsed, cycles=2)
        assert sample.complete and sample.visible and sample.phase == 'COMPLETE'
        assert sample.pose == motion.initial_pose
