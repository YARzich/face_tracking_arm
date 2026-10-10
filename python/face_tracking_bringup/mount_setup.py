# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""State and configuration operations used by the separate mount setup mode."""

import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import yaml

from .mount_geometry import (
    _inverse, _multiply, migrate_mount_config, rpy_transform, transform_xyz_rpy)


def _matvec(matrix, vector):
    return [sum(matrix[row][column] * vector[column] for column in range(3))
            for row in range(3)]


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _normalize(vector):
    magnitude = math.sqrt(_dot(vector, vector))
    if magnitude < 1e-8:
        raise ValueError('Measured mount direction is undefined for this flange pose')
    return [value / magnitude for value in vector]


def forward_kinematics_from_urdf(robot_description, joint_positions_rad,
                                 root_frame='world', tip_frame='link6'):
    """Compute one configured URDF chain pose for read-only top remembering."""
    if len(joint_positions_rad) != 6 or not all(
            math.isfinite(value) for value in joint_positions_rad):
        raise ValueError('Forward kinematics requires six finite measured joint angles')
    robot = ET.fromstring(robot_description)
    by_parent = {}
    for joint in robot.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is not None and child is not None:
            by_parent.setdefault(parent.get('link'), []).append(joint)

    chain = []

    def find_chain(frame, visited):
        if frame == tip_frame:
            return True
        if frame in visited:
            return False
        visited.add(frame)
        for joint in by_parent.get(frame, []):
            child = joint.find('child').get('link')
            chain.append(joint)
            if find_chain(child, visited):
                return True
            chain.pop()
        return False

    if not find_chain(root_frame, set()):
        raise ValueError(f'URDF has no chain from {root_frame} to {tip_frame}')

    result = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    joint_values = {f'joint{index}': joint_positions_rad[index - 1]
                    for index in range(1, 7)}
    for joint in chain:
        origin = joint.find('origin')
        xyz = [float(value) for value in origin.get('xyz', '0 0 0').split()] \
            if origin is not None else [0.0, 0.0, 0.0]
        rpy = [math.degrees(float(value))
               for value in origin.get('rpy', '0 0 0').split()] \
            if origin is not None else [0.0, 0.0, 0.0]
        transform = rpy_transform(xyz, rpy)
        kind = joint.get('type')
        if kind in ('revolute', 'continuous', 'prismatic'):
            name = joint.get('name')
            if name not in joint_values:
                raise ValueError(f'URDF chain has an unsupported moving joint: {name}')
            axis_element = joint.find('axis')
            axis = ([float(value) for value in axis_element.get('xyz', '1 0 0').split()]
                    if axis_element is not None else [1.0, 0.0, 0.0])
            axis = _normalize(axis)
            value = joint_values[name]
            if kind == 'prismatic':
                motion = [[1.0, 0.0, 0.0, axis[0] * value],
                          [0.0, 1.0, 0.0, axis[1] * value],
                          [0.0, 0.0, 1.0, axis[2] * value],
                          [0.0, 0.0, 0.0, 1.0]]
            else:
                x, y, z = axis
                sine, cosine = math.sin(value), math.cos(value)
                complement = 1.0 - cosine
                rotation = [
                    [cosine + x * x * complement,
                     x * y * complement - z * sine,
                     x * z * complement + y * sine],
                    [y * x * complement + z * sine,
                     cosine + y * y * complement,
                     y * z * complement - x * sine],
                    [z * x * complement - y * sine,
                     z * y * complement + x * sine,
                     cosine + z * z * complement],
                ]
                motion = [rotation[row] + [0.0] for row in range(3)] + [
                    [0.0, 0.0, 0.0, 1.0]]
            transform = _multiply(transform, motion)
        elif kind != 'fixed':
            raise ValueError(f'URDF chain has an unsupported joint type: {kind}')
        result = _multiply(result, transform)
    return result


def remember_top_direction(base_from_flange, joint_positions_rad):
    """Store flange-local world-up direction and all six measured start angles."""
    if len(joint_positions_rad) != 6 or not all(math.isfinite(x) for x in joint_positions_rad):
        raise ValueError('Remember-top requires six finite measured joint angles')
    if len(base_from_flange) != 4:
        raise ValueError('Remember-top requires the complete base-to-flange transform')
    rotation = [row[:3] for row in base_from_flange[:3]]
    world_z_in_flange = [rotation[2][column] for column in range(3)]
    # J6 spins about flange-local Z. Remove this component, then normalize the
    # selected horizontal side. A near-vertical flange has no usable top angle.
    direction = [world_z_in_flange[0], world_z_in_flange[1], 0.0]
    if math.hypot(direction[0], direction[1]) < math.sin(math.radians(5)):
        raise ValueError('Flange axis is vertical; change the starting pose before remembering '
                         'top')
    return {
        'preferred_top_flange': _normalize(direction),
        'remembered_joint_positions_deg': [math.degrees(value) for value in joint_positions_rad],
    }


def _measurement_frame_flange(preferred_top_flange):
    """Build a right-handed forward/left/up frame in flange coordinates."""
    forward = [0.0, 0.0, 1.0]  # Outward normal of the seating plane.
    up = _normalize(preferred_top_flange)
    up = [up[i] - _dot(up, forward) * forward[i] for i in range(3)]
    up = _normalize(up)
    side = _normalize([up[1] * forward[2] - up[2] * forward[1],
                       up[2] * forward[0] - up[0] * forward[2],
                       up[0] * forward[1] - up[1] * forward[0]])
    # Columns are forward, side (left), and up in flange axes.
    return [[forward[row], side[row], up[row]] for row in range(3)]


def manual_measurement_top(config):
    """Use a saved top direction, or derive it from the configured monitor pose."""
    setup = config.get('mount_setup', {})
    preferred = setup.get('preferred_top_flange')
    source = 'saved mount_setup.preferred_top_flange'
    if preferred is None:
        monitor_rotation = rpy_transform([0.0, 0.0, 0.0],
                                         config['monitor']['mount_rpy_deg'])
        preferred = [monitor_rotation[row][2] for row in range(3)]
        source = 'configured monitor up axis'
    if (len(preferred) != 3 or not all(math.isfinite(float(value)) for value in preferred)):
        raise ValueError('The selected top direction must be three finite flange coordinates')
    side_direction = [float(preferred[0]), float(preferred[1]), 0.0]
    if math.hypot(*side_direction[:2]) < math.sin(math.radians(5)):
        raise ValueError('The top direction is parallel to flange forward; select another side')
    return _normalize(side_direction), source


def pose_in_measurement_frame(preferred_top_flange, xyz_flange_m, rpy_flange_deg):
    """Express a stored flange-relative pose as forward/left/up measurements."""
    basis = _measurement_frame_flange(preferred_top_flange)
    flange_from_measurement = [basis[row] + [0.0] for row in range(3)]
    flange_from_measurement.append([0.0, 0.0, 0.0, 1.0])
    measurement_from_object = _multiply(
        _inverse(flange_from_measurement), rpy_transform(xyz_flange_m, rpy_flange_deg))
    return transform_xyz_rpy(measurement_from_object)


def measured_pose_in_flange(preferred_top_flange, xyz_forward_side_up_m, rpy_deg):
    """
    Convert tape-measure coordinates into T_flange_object.

    Distances and angles are entered in the shown measurement frame whose origin
    is the flange seating-plane center and whose axes are forward, left, up.
    """
    basis = _measurement_frame_flange(preferred_top_flange)
    measurement_pose = rpy_transform(xyz_forward_side_up_m, rpy_deg)
    result = [[sum(basis[row][k] * measurement_pose[k][column] for k in range(3))
               for column in range(3)] for row in range(3)]
    translation = _matvec(basis, xyz_forward_side_up_m)
    for row in range(3):
        result[row].append(translation[row])
    return result + [[0.0, 0.0, 0.0, 1.0]]


def top_joint6_goal(base_from_flange, current_joint_positions_rad,
                    preferred_top_flange, joint6_limits_rad):
    """Find the J6-only angle that points the saved side nearest world up."""
    if len(current_joint_positions_rad) != 6:
        raise ValueError('Set-top requires six measured joint angles')
    low, high = joint6_limits_rad
    current = current_joint_positions_rad[5]
    if not (math.isfinite(low) and math.isfinite(high) and low < high and low <= current <= high):
        raise ValueError('Current J6 or its limits are invalid')
    rotation = [row[:3] for row in base_from_flange[:3]]
    top = _normalize(preferred_top_flange)
    z_row = rotation[2]
    cosine_term = z_row[0] * top[0] + z_row[1] * top[1]
    sine_term = -z_row[0] * top[1] + z_row[1] * top[0]
    ideal_delta = math.atan2(sine_term, cosine_term)
    candidates = [low, high]
    first_turn = math.ceil((low - current - ideal_delta) / (2 * math.pi))
    last_turn = math.floor((high - current - ideal_delta) / (2 * math.pi))
    candidates.extend(ideal_delta + turn * 2 * math.pi + current
                      for turn in range(first_turn, last_turn + 1))
    best_angle, best_alignment = current, -math.inf
    for candidate in candidates:
        delta = candidate - current
        c, s = math.cos(delta), math.sin(delta)
        rotated_top = [c * top[0] - s * top[1], s * top[0] + c * top[1], top[2]]
        world_top = _matvec(rotation, rotated_top)
        alignment = world_top[2]
        if (alignment > best_alignment + 1e-12 or
                (abs(alignment - best_alignment) <= 1e-12 and
                 abs(candidate - current) < abs(best_angle - current))):
            best_angle, best_alignment = candidate, alignment
    return best_angle


def top_alignment_error_deg(base_from_flange, joint_positions_rad, preferred_top_flange,
                            joint6_goal_rad):
    """Return the remaining top tilt after the J6-only alignment."""
    rotation = [row[:3] for row in base_from_flange[:3]]
    top = _normalize(preferred_top_flange)
    delta = joint6_goal_rad - joint_positions_rad[5]
    c, s = math.cos(delta), math.sin(delta)
    rotated_top = [c * top[0] - s * top[1], s * top[0] + c * top[1], top[2]]
    world_top = _matvec(rotation, rotated_top)
    return math.degrees(math.acos(max(-1.0, min(1.0, world_top[2]))))


def apply_optical_calibration(config, mode, flange_from_optical):
    """
    Apply a calibrated optical pose while preserving optical/body offset.

    OpenCV estimates T_flange_optical. The YAML model stores T_flange_body, so
    the known T_body_optical is removed exactly once.
    """
    from .hand_eye_calibration import validate_rigid_transform

    flange_from_optical = validate_rigid_transform(flange_from_optical, 'flange_from_optical')
    result = migrate_mount_config(config)
    camera = result[mode]['left']
    body_from_optical = rpy_transform(camera['optical_xyz_m'],
                                      camera['optical_rpy_deg'])
    flange_from_body = _multiply(flange_from_optical, _inverse(body_from_optical))
    xyz, rpy = transform_xyz_rpy(flange_from_body)
    result['camera_mount']['xyz_m'] = xyz
    result['camera_mount']['rpy_deg'] = rpy
    result['mount_geometry_version'] = 2
    return result


def apply_manual_geometry(config, preferred_top_flange, monitor_xyz, monitor_rpy,
                          camera_xyz, camera_rpy):
    """Store tape-measured poses entered as forward/left/up from flange center."""
    result = migrate_mount_config(config)
    monitor = measured_pose_in_flange(preferred_top_flange, monitor_xyz, monitor_rpy)
    camera = measured_pose_in_flange(preferred_top_flange, camera_xyz, camera_rpy)
    result['monitor']['mount_xyz_m'], result['monitor']['mount_rpy_deg'] = \
        transform_xyz_rpy(monitor)
    result['camera_mount']['xyz_m'], result['camera_mount']['rpy_deg'] = \
        transform_xyz_rpy(camera)
    result['mount_geometry_version'] = 2
    return result


def save_yaml_atomic(path, config):
    """Write the complete validated configuration atomically in its directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent,
                                     prefix=destination.name + '.', suffix='.tmp',
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
    temporary.replace(destination)
