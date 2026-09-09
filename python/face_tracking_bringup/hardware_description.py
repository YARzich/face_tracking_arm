# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Build one consistent robot model, named poses and controller limits from the hardware YAML."""

import math
import xml.etree.ElementTree as ET

import xacro
import yaml


def load_yaml(path):
    return yaml.safe_load(path.read_text())


def robot_parameters(config, mode, share, config_path, mock):
    xml = xacro.process_file(str(share / 'description/lite6_hardware.urdf.xacro'), mappings={
        'config_file': str(config_path), 'camera_mode': mode,
        'mock_hardware': str(mock).lower()}).toxml()
    robot = ET.fromstring(xml)
    names = [link.get('name') for link in robot.findall('link')]
    if len(names) != len(set(names)):
        raise ValueError('Camera frame names collide with the robot links')
    semantic = ET.parse(share / 'config/moveit/lite6.srdf').getroot()
    for name in ('rest', 'search'):
        state = semantic.find(f"group_state[@name='{name}']")
        values = config['motion'][name + '_joints_deg']
        for i, value in enumerate(values, start=1):
            rad = math.radians(value)
            limit = robot.find(f"joint[@name='joint{i}']/limit")
            low, high = float(limit.get('lower')), float(limit.get('upper'))
            if i == 5:
                low, high = max(low, -1.60), min(high, 1.60)
            if not low + .1 <= rad <= high - .1:
                raise ValueError(f'{name}_joints_deg: joint{i} is outside allowed margins')
            state.find(f"joint[@name='joint{i}']").set('value', str(rad))
    ET.SubElement(semantic, 'disable_collisions', {
        'link1': 'monitor_link', 'link2': config['camera_mount']['body_frame'],
        'reason': 'Adjacent'})
    limits = load_yaml(share / 'config/moveit/joint_limits.yaml')
    scale = config['motion']['speed_scale']
    for values in limits['joint_limits'].values():
        for key in ('max_velocity', 'max_acceleration', 'max_jerk'):
            values[key] *= scale
    return [
        {'robot_description': xml},
        {'robot_description_semantic': ET.tostring(semantic, encoding='unicode')},
        {'robot_description_kinematics': load_yaml(share / 'config/moveit/kinematics.yaml')},
        {'robot_description_planning': limits}], limits


def control_parameters(config, share, limits):
    servo = load_yaml(share / 'config/moveit/servo.yaml')
    servo.update(publish_period=.01, max_expected_latency=.05, check_collisions=False,
                 use_smoothing=False, publish_joint_positions=True,
                 publish_joint_velocities=False, publish_joint_accelerations=False)
    motion = load_yaml(share / 'config/moveit/collision_aware_servo.yaml')
    joints = limits['joint_limits']
    motion['joint_names'] = list(joints)
    for field, key in (('max_joint_velocity_rad_s', 'max_velocity'),
                       ('max_joint_acceleration_rad_s2', 'max_acceleration'),
                       ('max_joint_jerk_rad_s3', 'max_jerk')):
        motion[field] = [v[key] for v in joints.values()]
    motion.update({k: v for k, v in config['motion'].items() if k.startswith('search_')
                   and k != 'search_joints_deg'})
    tracking = load_yaml(share / 'config/tracking.yaml')[
        'face_tracking_controller']['ros__parameters']
    tracking.update(config['tracking'])
    table = config['table']
    tracking.update(table_shape=table['shape'], table_dimensions_m=table['dimensions_m'],
                    table_center_m=table['center_m'], table_yaw_rad=math.radians(table['yaw_deg']))
    controllers = load_yaml(share / 'config/control/controllers.yaml')
    for node in controllers.values():
        node['ros__parameters']['use_sim_time'] = False
    return servo, motion, tracking, controllers
