# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""
Finite manufacturer joint ranges in the robot's calibrated coordinates.

Source: xArm-Python-SDK, xarm/core/config/x_config.py at
d911319cebc45142613e18086aabb8067b18ab7f. Reduced-mode limits are separate.
Model number follows the manufacturer's serial-number slice SN[2:6].
"""

import math


_TURN = (-2.0 * math.pi, 2.0 * math.pi)
_XARM6 = (_TURN, (-2.059488, 2.094395), (-3.92699, .191986),
          _TURN, (-1.692969, math.pi), _TURN)
_XARM6_1305 = (_TURN, (-2.042035, 2.024581), (-3.822271, .174532),
               _TURN, (-1.692969, math.pi), _TURN)
_LITE6 = (_TURN, (-2.6179938779914944, 2.6179938779914944),
          (-.061086523819801536, 5.235987755982989), _TURN,
          (-2.1642082724729685, 2.1642082724729685), _TURN)


def get_robot_profile(model='xarm6', model_num=0):
    """Return a fresh profile; model_num=0 is the offline/default model only."""
    if isinstance(model_num, bool) or not isinstance(model_num, int):
        raise ValueError('robot.model_num must be an integer')
    if model_num < 0 or model_num >= 8500:
        raise ValueError('Unsupported robot model number')
    if model == 'xarm6':
        if model_num in (1250, 1380):
            raise ValueError('This xArm variant requires a different robot description')
        bounds = _XARM6_1305 if model_num >= 1305 else _XARM6
        group, payload = 'xarm6', 5.0
    elif model == 'lite6':
        bounds, group, payload = _LITE6, 'lite6_arm', .6
    else:
        raise ValueError('robot.model must be xarm6 or lite6')
    return {
        'model': model, 'model_num': model_num, 'planning_group': group,
        'controller_name': 'lite6_arm_controller', 'payload_kg': payload,
        'position_limits_rad': [list(pair) for pair in bounds],
        'urdf_file': f'description/{model}_hardware.urdf.xacro',
        'srdf_file': f'config/moveit/{model}.srdf',
    }


def identify_robot(serial_number, axis, device_type, expected_model='xarm6'):
    """Reject unknown identities instead of fitting a plausible six-axis arm."""
    model = {6: 'xarm6', 9: 'lite6'}.get(device_type) if axis == 6 else None
    if model is None or model != expected_model:
        raise ValueError(f'Connected robot type {device_type}/{axis} does not match '
                         f'robot.model={expected_model}')
    prefix = 'XI' if model == 'xarm6' else 'L'
    if not isinstance(serial_number, str) or not serial_number.startswith(prefix):
        raise ValueError('Robot serial number does not match its reported model')
    number = serial_number[2:6]
    if len(number) != 4 or not number.isdigit() or int(number) == 0:
        raise ValueError('Cannot identify revision from the robot serial number')
    return get_robot_profile(model, int(number))


def intersect_joint_limits(factory_limits, reduced_limits):
    """Apply an active reduced envelope without widening manufacturer bounds."""
    if len(factory_limits) != 6 or len(reduced_limits) != 6:
        raise ValueError('Expected six joint-limit pairs')
    result = []
    for factory, reduced in zip(factory_limits, reduced_limits):
        if len(reduced) != 2 or not all(math.isfinite(v) for v in reduced):
            raise ValueError('Invalid reduced joint limits')
        low, high = max(factory[0], reduced[0]), min(factory[1], reduced[1])
        if low >= high:
            raise ValueError('Reduced joint limits do not overlap the robot profile')
        result.append([low, high])
    return result


def validate_joint_positions(positions, limits, label='joint positions', tolerance=1e-5):
    """Check actual coordinates inclusively: startup needs no interior margin."""
    if len(positions) != 6:
        raise ValueError(f'{label}: expected six joint angles in radians')
    for i, (value, (low, high)) in enumerate(zip(positions, limits), start=1):
        if not math.isfinite(value) or not low - tolerance <= value <= high + tolerance:
            raise ValueError(f'{label}: joint{i} is outside the selected robot limits')
