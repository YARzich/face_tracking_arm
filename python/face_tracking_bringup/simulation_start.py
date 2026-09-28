# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Select and validate simulation startup angles in manufacturer coordinates."""

from pathlib import Path

from face_tracking_bringup.robot_profile import get_robot_profile, validate_joint_positions
import yaml


START_POSES = ('rest', 'zero', 'folded', 'joint1_limit', 'incident')
_JOINT_NAMES = tuple(f'joint{i}' for i in range(1, 7))


def resolve_simulation_start(package_share, *, robot_model='xarm6', start_pose='rest',
                             initial_positions_file=''):
    """
    Return a checked YAML path; an explicit file takes precedence over a preset.

    Only joint ranges are checked here. The controller validates collision geometry
    after loading the complete robot and scene. Angles are neither clamped nor wrapped.
    """
    profile = get_robot_profile(robot_model)
    if start_pose not in START_POSES:
        raise ValueError(f'Unknown start_pose: {start_pose}')
    control = Path(package_share) / 'config' / 'control'
    if initial_positions_file:
        path = Path(initial_positions_file).expanduser()
    elif start_pose == 'rest':
        filename = ('initial_positions.yaml' if robot_model == 'xarm6'
                    else 'lite6_initial_positions.yaml')
        path = control / filename
    elif start_pose == 'zero':
        path = control / 'start_poses' / 'zero.yaml'
    else:
        if start_pose == 'incident' and robot_model != 'lite6':
            raise ValueError('start_pose=incident is the recorded Lite6 pose; '
                             'use robot_model=lite6 or an explicit initial_positions_file')
        if start_pose == 'folded' and robot_model != 'xarm6':
            raise ValueError('start_pose=folded is an adapted xArm6 pose; '
                             'use robot_model=xarm6 or start_pose=incident for Lite6')
        path = control / 'start_poses' / f'{robot_model}_{start_pose}.yaml'
    path = path.resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'Cannot read initial_positions_file {path}: {error}') from error
    if not isinstance(data, dict) or set(data) != {'initial_positions'}:
        raise ValueError(f'{path}: expected an initial_positions mapping')
    positions = data['initial_positions']
    if not isinstance(positions, dict) or set(positions) != set(_JOINT_NAMES):
        raise ValueError(f'{path}: initial_positions must contain exactly joint1..joint6')
    angles = [positions[name] for name in _JOINT_NAMES]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in angles):
        raise ValueError(f'{path}: joint angles must be numbers in radians')
    validate_joint_positions(
        angles, profile['position_limits_rad'], label=str(path), tolerance=0.0)
    return path
