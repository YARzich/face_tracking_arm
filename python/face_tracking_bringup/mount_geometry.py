# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Rigid transforms and migration for the camera/monitor mount configuration."""

import copy
import math


def _multiply(left, right):
    return [[sum(left[row][k] * right[k][column] for k in range(4))
             for column in range(4)] for row in range(4)]


def _inverse(transform):
    rotation = [[transform[row][column] for column in range(3)] for row in range(3)]
    translation = [transform[row][3] for row in range(3)]
    inverse_rotation = [[rotation[column][row] for column in range(3)]
                        for row in range(3)]
    inverse_translation = [-sum(inverse_rotation[row][column] * translation[column]
                                for column in range(3)) for row in range(3)]
    return [inverse_rotation[row] + [inverse_translation[row]] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]]


def rpy_transform(xyz_m, rpy_deg):
    """Return T_parent_child using REP-103 fixed-axis roll, pitch, yaw."""
    roll, pitch, yaw = [math.radians(value) for value in rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [rotation[row] + [float(xyz_m[row])] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]]


def transform_xyz_rpy(transform):
    """Convert a homogeneous transform to REP-103 XYZ and fixed-axis RPY."""
    rotation = [row[:3] for row in transform[:3]]
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return ([float(transform[row][3]) for row in range(3)],
            [float(math.degrees(angle)) for angle in (roll, pitch, yaw)])


def migrate_mount_config(config):
    """
    Return v2 flange-relative camera geometry, migrating unversioned YAML.

    Before v2, monitor pose was relative to link6 and camera body pose was
    relative to monitor_link. The product preserves the represented TF exactly.
    """
    result = copy.deepcopy(config)
    version = result.get('mount_geometry_version', 1)
    if version == 2:
        return result
    if version != 1:
        raise ValueError(f'Unsupported mount_geometry_version: {version}')
    if not isinstance(result.get('monitor'), dict) or not isinstance(
            result.get('camera_mount'), dict):
        # Camera-only use may intentionally omit the robot/mount sections.
        result['mount_geometry_version'] = 2
        return result
    monitor = result['monitor']
    camera = result['camera_mount']
    flange_to_monitor = rpy_transform(monitor['mount_xyz_m'], monitor['mount_rpy_deg'])
    monitor_to_camera = rpy_transform(camera['xyz_m'], camera['rpy_deg'])
    camera_xyz, camera_rpy = transform_xyz_rpy(
        _multiply(flange_to_monitor, monitor_to_camera))
    camera['xyz_m'] = camera_xyz
    camera['rpy_deg'] = camera_rpy
    result['mount_geometry_version'] = 2
    return result
