# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Build one consistent robot model, named poses and controller limits from the hardware YAML."""

import math
import xml.etree.ElementTree as ET

import xacro
import yaml

from .hardware_config import point_filter_parameters
from .mount_collisions import allow_fixed_mount_collisions
from .robot_profile import get_robot_profile


def load_yaml(path):
    return yaml.safe_load(path.read_text())


def perception_parameters(config, mode, *, camera_only=False):
    """Assemble detector parameters from the current external hardware configuration."""
    settings = config[mode]
    params = {k: v for k, v in config['perception'].items() if k != 'python_executable'}
    params.update(point_filter_parameters(config, mode))
    params.update(mode=mode, use_sim_time=False, rectification=settings['rectification'],
                  sync_slop_sec=settings.get('sync_slop_sec', 0.0),
                  selection_frame=settings['left']['frame_id'] if camera_only else 'link_base',
                  world_frame='' if camera_only else 'world',
                  input_left_frame=settings['left']['frame_id'])
    if mode == 'stereo':
        params.update(stereo_baseline_m=settings['baseline_m'],
                      stereo_num_disparities=settings.get('num_disparities', 128),
                      stereo_block_size=settings.get('block_size', 5),
                      input_right_frame=settings['right']['frame_id'])
    return params


def robot_parameters(config, mode, share, config_path, mock, probe_result=None):
    profile = (probe_result['profile'] if probe_result else
               get_robot_profile(config['robot']['model'], config['robot']['model_num']))
    bounds = (probe_result['effective_position_limits_rad'] if probe_result else
              profile['position_limits_rad'])
    mappings = {'config_file': str(config_path), 'camera_mode': mode,
                'mock_hardware': str(mock).lower(), 'robot_model': profile['model'],
                'model_num': str(profile['model_num'])}
    for i, (low, high) in enumerate(bounds, start=1):
        mappings[f'joint{i}_lower_limit'] = str(low)
        mappings[f'joint{i}_upper_limit'] = str(high)
    xml = xacro.process_file(str(share / profile['urdf_file']), mappings=mappings).toxml()
    robot = ET.fromstring(xml)
    names = [link.get('name') for link in robot.findall('link')]
    if len(names) != len(set(names)):
        raise ValueError('Camera frame names collide with the robot links')
    semantic = ET.parse(share / profile['srdf_file']).getroot()
    for name in ('rest', 'search'):
        state = semantic.find(f"group_state[@name='{name}']")
        values = config['motion'][name + '_joints_deg']
        for i, value in enumerate(values, start=1):
            rad = math.radians(value)
            limit = robot.find(f"joint[@name='joint{i}']/limit")
            low, high = float(limit.get('lower')), float(limit.get('upper'))
            if not low <= rad <= high:
                raise ValueError(f'{name}_joints_deg: joint{i} is outside robot limits')
            state.find(f"joint[@name='joint{i}']").set('value', str(rad))
    allow_fixed_mount_collisions(robot, semantic)
    limits = {'joint_limits': {}}
    scale = config['motion']['speed_scale']
    for i in range(6):
        joint = f'joint{i + 1}'
        velocity = config['limits']['max_velocity_rad_s'][i] * scale
        maximum = float(robot.find(f"joint[@name='{joint}']/limit").get('velocity'))
        if probe_result and probe_result.get('reduced_max_joint_speed_rad_s') is not None:
            maximum = min(maximum, probe_result['reduced_max_joint_speed_rad_s'])
        if velocity > maximum:
            raise ValueError(f'limits: {joint} requests {velocity:g} rad/s after speed_scale, '
                             f'above the robot limit {maximum:g} rad/s')
        limits['joint_limits'][joint] = {
            'has_velocity_limits': True, 'max_velocity': velocity,
            'has_acceleration_limits': True,
            'max_acceleration': config['limits']['max_acceleration_rad_s2'][i] * scale,
            'has_jerk_limits': True,
            'max_jerk': config['limits']['max_jerk_rad_s3'][i] * scale,
        }
    solvers = load_yaml(share / 'config/moveit/kinematics.yaml')
    kinematics = {profile['planning_group']: solvers[profile['planning_group']]}
    return [
        {'robot_description': xml},
        {'robot_description_semantic': ET.tostring(semantic, encoding='unicode')},
        {'robot_description_kinematics': kinematics},
        {'robot_description_planning': limits}], limits


def control_parameters(config, share, limits, mode='mono_cpu'):
    profile = get_robot_profile(config['robot']['model'], config['robot']['model_num'])
    servo = load_yaml(share / 'config/moveit/servo.yaml')
    servo.update(publish_period=.01, max_expected_latency=.05, check_collisions=False,
                 use_smoothing=False, publish_joint_positions=True,
                 publish_joint_velocities=False, publish_joint_accelerations=False,
                 move_group_name=profile['planning_group'],
                 command_out_topic='/' + profile['controller_name'] + '/joint_trajectory')
    motion = load_yaml(share / 'config/moveit/collision_aware_servo.yaml')
    motion.update(config['controller'])
    motion['table_collision_enabled'] = any(config['table']['dimensions_m'])
    motion['planning_group_name'] = profile['planning_group']
    motion['gaze_frame'] = (config[mode]['left']['frame_id']
                            if config['camera_mount']['publish_optical_tf'] else
                            'tracking_gaze_optical')
    planner = motion['recovery_planner']
    group_settings = next(v for k, v in planner.items() if k != 'planner_configs')
    motion['recovery_planner'] = {
        'planner_configs': planner['planner_configs'], profile['planning_group']: group_settings}
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
