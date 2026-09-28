# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Geometry, timing and smoothness of the rendered-person stress scenario."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from face_tracking_perception.arm_scene import write_vision_world  # noqa: E402, I100
from face_tracking_perception.stress_motion import StressMotion  # noqa: E402


def test_starts_visible_and_stationary_without_waiting_for_rest():
    motion = StressMotion()
    for elapsed in (-1, 0, 5, 19.999):
        sample = motion.sample(elapsed)
        assert sample.pose == motion.initial_pose
        assert sample.visible and not sample.complete
        assert sample.phase == 'START_HOLD'
    with pytest.raises(FrozenInstanceError):
        sample.visible = False


def test_visits_full_height_radius_and_side_range_while_facing_robot():
    motion = StressMotion()
    samples = [motion.sample(t) for t in np.linspace(0, motion.duration_sec, 4021)]
    poses = np.array([sample.pose for sample in samples if sample.visible])
    radius = np.linalg.norm(poses[:, :2], axis=1)
    assert radius.min() == pytest.approx(1.4, abs=1e-5)
    assert radius.max() == pytest.approx(2.6, abs=1e-5)
    height = poses[:, 2] + motion.face_reference_height_m
    assert height.min() == pytest.approx(1.25, abs=1e-5)
    assert height.max() == pytest.approx(1.85, abs=1e-5)
    azimuth = np.arctan2(poses[:, 1], poses[:, 0])
    assert math.degrees(azimuth.min()) == pytest.approx(-100, abs=.001)
    assert math.degrees(azimuth.max()) == pytest.approx(110, abs=.001)
    face_forward = np.column_stack((np.sin(poses[:, 3]), -np.cos(poses[:, 3])))
    np.testing.assert_allclose(face_forward, -poses[:, :2] / radius[:, None], atol=1e-12)
    assert motion.sample(75).pose[3] < motion.sample(66).pose[3]  # Reverses the side arc.


@pytest.mark.parametrize('boundary', [20, 32, 46, 66, 82, 92, 100, 108, 122, 134])
def test_phase_boundaries_have_continuous_position_velocity_and_acceleration(boundary):
    motion = StressMotion()
    step = .001
    values = [np.array(motion.sample(boundary + offset * step).pose)
              for offset in (-2, -1, 0, 1, 2)]
    before, left, center, right, after = values
    left_velocity, right_velocity = (center - left) / step, (right - center) / step
    left_acceleration = (center - 2 * left + before) / step**2
    right_acceleration = (after - 2 * right + center) / step**2
    np.testing.assert_allclose(left, center, atol=1e-8)
    np.testing.assert_allclose(right, center, atol=1e-8)
    np.testing.assert_allclose(left_velocity, 0, atol=1e-5)
    np.testing.assert_allclose(right_velocity, 0, atol=1e-5)
    np.testing.assert_allclose(left_acceleration, 0, atol=.001)
    np.testing.assert_allclose(right_acceleration, 0, atol=.001)


def test_pauses_disappearance_and_reappearance_have_fixed_simulation_times():
    motion = StressMotion()
    for start, end, phase in ((82, 92, 'SETTLE_HOLD'), (100, 108, 'REAPPEAR_HOLD'),
                              (122, 134, 'FINAL_HOLD')):
        beginning = motion.sample(start)
        ending = motion.sample(end - .001)
        assert beginning.pose == ending.pose
        assert beginning.phase == ending.phase == phase
        assert beginning.visible and ending.visible
    assert motion.sample(91.999).visible
    for t in (92, 95, 99.999):
        sample = motion.sample(t)
        assert sample.phase == 'HIDDEN' and not sample.visible
    assert motion.sample(100).visible
    assert math.degrees(motion.sample(100).pose[3] + math.pi / 2) == pytest.approx(-70)


def test_repetition_preserves_route_and_finite_run_holds_final_pose():
    motion = StressMotion()
    for t in (0, 20, 40, 82, 93, 110, 133):
        assert motion.sample(t) == motion.sample(t + motion.duration_sec)
    assert not motion.sample(motion.duration_sec, cycles=2).complete
    assert not motion.sample(2 * motion.duration_sec - .001, cycles=2).complete
    for t in (2 * motion.duration_sec, 1000):
        sample = motion.sample(t, cycles=2)
        assert sample.complete and sample.visible and sample.phase == 'COMPLETE'
        assert sample.pose == motion.initial_pose


@pytest.mark.parametrize('elapsed', [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_time(elapsed):
    with pytest.raises(ValueError, match='time'):
        StressMotion().sample(elapsed)


@pytest.mark.parametrize('cycles', [-1, .5, True])
def test_rejects_invalid_cycle_count(cycles):
    with pytest.raises(ValueError, match='cycles'):
        StressMotion().sample(0, cycles=cycles)


def test_world_starts_with_visible_stress_person_and_keeps_other_people_hidden(tmp_path):
    base = ROOT / 'worlds/lite6_table.sdf'
    destination = tmp_path / 'stress.sdf'
    write_vision_world(base, destination, 2, initial_person_pose=StressMotion.initial_pose)
    result = ET.parse(destination).getroot()
    people = result.findall('world/include')
    first = np.fromstring(people[0].findtext('pose'), sep=' ')
    np.testing.assert_allclose(first, [2.1, 0, 0, 0, 0, -math.pi / 2])
    assert people[1].findtext('pose') == '0 0 -10 0 0 0'
    original = ET.parse(base).getroot()
    for name in ('round_table', 'ground_plane'):
        query = f"world/model[@name='{name}']"
        assert ET.tostring(original.find(query)) == ET.tostring(result.find(query))


@pytest.mark.parametrize('pose', [(1, 2, 3), (1, 2, 3, 4, 5), (1, 2, math.nan, 0)])
def test_world_rejects_invalid_initial_pose(tmp_path, pose):
    with pytest.raises(ValueError, match='Initial person pose'):
        write_vision_world(ROOT / 'worlds/lite6_table.sdf', tmp_path / 'bad.sdf',
                           initial_person_pose=pose)
