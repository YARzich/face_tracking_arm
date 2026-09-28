# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Prepare the existing table world for real rendered face observations."""

import math
import xml.etree.ElementTree as ET


def person_entity_name(index):
    """Return the shared Gazebo entity name for a zero-based person index."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('Person index must be a nonnegative integer')
    return 'face_test_person' + (f'_{index + 1}' if index else '')


def _validated_person_pose(pose):
    """Read one world x/y/z/yaw pose without accepting nonfinite geometry."""
    try:
        pose = tuple(float(v) for v in pose)
    except (TypeError, ValueError) as error:
        raise ValueError('Initial person pose must contain finite x, y, z and yaw') from error
    if len(pose) != 4 or not all(map(math.isfinite, pose)):
        raise ValueError('Initial person pose must contain finite x, y, z and yaw')
    return pose


def write_vision_world(base_world, destination, people_count=1, initial_person_pose=None,
                       initial_person_poses=None):
    """
    Create the camera world, optionally showing the first person immediately.

    ``initial_person_pose`` shows the first person, leaving other people hidden.
    Alternatively, ``initial_person_poses`` sets exactly one pose per person.
    Each pose contains world x/y/z in metres and yaw in radians.
    """
    if not isinstance(people_count, int) or isinstance(people_count, bool) or people_count < 1:
        raise ValueError('People count must be a positive integer')
    if initial_person_pose is not None and initial_person_poses is not None:
        raise ValueError('Use either initial_person_pose or initial_person_poses, not both')
    poses = [None] * people_count
    if initial_person_pose is not None:
        poses[0] = _validated_person_pose(initial_person_pose)
    if initial_person_poses is not None:
        poses = [_validated_person_pose(pose) for pose in initial_person_poses]
        if len(poses) != people_count:
            raise ValueError('initial_person_poses must contain one pose per person')
    tree = ET.parse(base_world)
    world = tree.getroot().find('world')
    plugin = ET.SubElement(world, 'plugin', {
        'filename': 'gz-sim-sensors-system', 'name': 'gz::sim::systems::Sensors'})
    ET.SubElement(plugin, 'render_engine').text = 'ogre2'
    # Front illumination and sufficient ambient light for the existing face texture.
    world.find('scene/ambient').text = '0.65 0.65 0.65 1'
    world.find('scene/shadows').text = 'false'
    world.find("light[@name='sun']/direction").text = '0.5 0 -1'
    for index in range(people_count):
        include = ET.SubElement(world, 'include')
        ET.SubElement(include, 'uri').text = 'model://person_standing'
        ET.SubElement(include, 'name').text = person_entity_name(index)
        ET.SubElement(include, 'static').text = 'true'
        pose = '0 0 -10 0 0 0'
        if poses[index] is not None:
            x, y, z, yaw = poses[index]
            pose = f'{x} {y} {z} 0 0 {yaw}'
        ET.SubElement(include, 'pose').text = pose
    tree.write(destination, encoding='utf-8', xml_declaration=True)
