# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Manufacturer coordinates and active reduced envelopes need no robot or ROS."""

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup.robot_profile import (  # noqa: E402, I100
    get_robot_profile, identify_robot, intersect_joint_limits, validate_joint_positions)


def test_xarm6_and_lite6_are_different_bounded_models():
    arm, lite = get_robot_profile(), get_robot_profile('lite6')
    assert arm['model'] == 'xarm6'
    assert arm['planning_group'] == 'xarm6'
    assert arm['payload_kg'] == 5.0
    assert arm['position_limits_rad'][2] == [-3.92699, .191986]
    assert lite['position_limits_rad'][2][0] == pytest.approx(math.radians(-3.5))
    for profile in (arm, lite):
        assert profile['position_limits_rad'][0] == [-2 * math.pi, 2 * math.pi]
        validate_joint_positions([0] * 6, profile['position_limits_rad'])


def test_revision_is_identified_from_actual_identity():
    old = identify_robot('XI13001234B123', 6, 6)
    new = identify_robot('XI13051234B123', 6, 6)
    assert old['position_limits_rad'][1] == [-2.059488, 2.094395]
    assert new['position_limits_rad'][1] == [-2.042035, 2.024581]
    assert new['position_limits_rad'][2] == [-3.822271, .174532]
    assert new['model_num'] == 1305


@pytest.mark.parametrize('serial,axis,kind', [
    ('XI13051234B123', 7, 7), ('XI12501234B123', 6, 8),
    ('XI13801234B123', 6, 11), ('LX13001234B123', 6, 9),
    ('XIunknown', 6, 6), ('XI00001234B123', 6, 6), ('LX13051234B123', 6, 6),
])
def test_unknown_or_wrong_arm_never_falls_back_to_an_arbitrary_model(serial, axis, kind):
    with pytest.raises(ValueError):
        identify_robot(serial, axis, kind)


def test_boundaries_are_valid_start_positions_without_unwrapping():
    limits = get_robot_profile()['position_limits_rad']
    angles = [2 * math.pi, limits[1][0], 0, -2 * math.pi, math.pi, 5.7]
    original = angles.copy()
    validate_joint_positions(angles, limits)
    assert angles == original
    angles[0] += .01
    with pytest.raises(ValueError, match='joint1'):
        validate_joint_positions(angles, limits)


def test_reduced_envelope_intersects_but_never_expands_factory_limits():
    factory = get_robot_profile()['position_limits_rad']
    reduced = [[-10, 10] for _ in range(6)]
    reduced[0] = [-1, 2]
    assert intersect_joint_limits(factory, reduced) == [[-1, 2], *factory[1:]]
    assert factory[0] == [-2 * math.pi, 2 * math.pi]
    reduced[0] = [8, 9]
    with pytest.raises(ValueError, match='overlap'):
        intersect_joint_limits(factory, reduced)


@pytest.mark.parametrize('value', [math.inf, math.nan])
def test_invalid_angles_and_reduced_ranges_fail(value):
    limits = get_robot_profile()['position_limits_rad']
    with pytest.raises(ValueError):
        validate_joint_positions([value, 0, 0, 0, 0, 0], limits)
    with pytest.raises(ValueError):
        intersect_joint_limits(limits, [[value, 1], *limits[1:]])
