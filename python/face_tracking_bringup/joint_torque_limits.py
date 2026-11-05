# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Apply optional joint torque limits to an expanded robot description."""

import math
import xml.etree.ElementTree as ET


_SETTING = 'compact_maneuver.joint_torque_limits_nm'


def apply_joint_torque_limits(robot_description, servo_config):
    """Apply the shared torque override map, preserving the XML when it is empty."""
    if not isinstance(servo_config, dict):
        raise ValueError('collision-aware servo config: expected a YAML mapping')
    compact = servo_config.get('compact_maneuver', {})
    if not isinstance(compact, dict):
        raise ValueError('compact_maneuver: expected a YAML mapping')
    overrides = compact.get('joint_torque_limits_nm', {})
    if not isinstance(overrides, dict):
        raise ValueError(f'{_SETTING}: expected a joint-name to Nm mapping')
    if not overrides:
        return robot_description

    try:
        robot = ET.fromstring(robot_description)
    except (ET.ParseError, TypeError) as error:
        raise ValueError('Cannot parse robot_description for torque-limit overrides') from error

    torque_joints = {}
    for joint in robot.findall('./joint'):
        limit = joint.find('limit')
        if joint.get('type') in ('revolute', 'continuous') and limit is not None:
            torque_joints[joint.get('name')] = limit

    normalized = {}
    for joint_name, value in overrides.items():
        if not isinstance(joint_name, str) or joint_name not in torque_joints:
            raise ValueError(f'{_SETTING}: unknown torque-limited robot joint {joint_name!r}')
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric_value = math.nan
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(numeric_value) or numeric_value <= 0):
            raise ValueError(f'{_SETTING}.{joint_name}: expected a finite positive Nm value')
        normalized[joint_name] = numeric_value

    for joint_name, value in normalized.items():
        torque_joints[joint_name].set('effort', format(value, '.17g'))
    return ET.tostring(robot, encoding='unicode')
