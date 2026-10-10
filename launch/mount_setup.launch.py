# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Isolated camera/mount setup, with no face-tracking controller running."""

import copy
from pathlib import Path
import shutil
import sys
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
sys.path.insert(0, str(scripts))

from face_tracking_bringup.hardware_config import validate_config  # noqa: E402, I100
from face_tracking_bringup.hardware_description import (  # noqa: E402, I100
    load_yaml, robot_parameters)
from face_tracking_bringup.hardware_launch import (  # noqa: E402, I100
    camera_actions, mount_setup_robot_actions)  # noqa: E402, I100
from face_tracking_bringup.hardware_probe import (  # noqa: E402, I100
    probe_robot, sample_stationary_joint_positions)  # noqa: E402, I100
from face_tracking_bringup.mount_setup import (  # noqa: E402, I100
    forward_kinematics_from_urdf, remember_top_direction, save_yaml_atomic)  # noqa: E402, I100
from face_tracking_bringup.robot_profile import get_robot_profile  # noqa: E402, I100


def _setup(context):
    values = {name: LaunchConfiguration(name).perform(context) for name in (
        'config_file', 'mode', 'operation', 'marker_side_mm', 'output_file',
        'mock_hardware', 'show_image', 'use_sim_time', 'external_sim')}
    mode = values['mode']
    mock = values['mock_hardware'] == 'true'
    use_sim_time = values['use_sim_time'] == 'true'
    external_sim = values['external_sim'] == 'true'
    path = Path(values['config_file']).expanduser().resolve()
    share = Path(get_package_share_directory('face_tracking_arm'))
    operation = values['operation']
    if external_sim and operation != 'calibrate':
        raise ValueError('external_sim supports only operation=calibrate')
    if external_sim and not use_sim_time:
        raise ValueError('external_sim requires use_sim_time:=true')
    if external_sim and mode != 'rgbd':
        raise ValueError('external_sim requires the rendered rgbd camera mode')
    raw_config = load_yaml(path)
    if external_sim:
        raw_config = copy.deepcopy(raw_config)
        raw_config['rgbd']['source'] = 'ros'
        raw_config['rgbd']['left'].update(
            image_topic='/face_test/image_raw',
            info_topic='/face_test/camera_info',
            depth_topic='/face_test/depth/image_raw',
            frame_id='face_test_camera_optical_frame')
    config = validate_config(
        raw_config, mode, mock=(mock or external_sim),
        driver_only=(operation == 'view'),
        setup_only=(operation in ('remember-top', 'set-top')),
        directory=path.parent)
    probe = None
    if not mock and not external_sim and operation != 'view':
        probe = probe_robot(
            config['robot']['ip'], expected_model=config['robot']['model'],
            require_ready=(operation != 'remember-top'))
        config['robot']['model_num'] = probe['profile']['model_num']
        print(f"Setup connected: {config['robot']['model']} {config['robot']['model_num']}; "
              f"reduced mode {probe['reduced_mode']}", flush=True)
    profile = None
    limits = []
    if operation != 'view':
        profile = (probe['profile'] if probe else get_robot_profile(
            config['robot']['model'], config['robot']['model_num']))
        limits = (probe['effective_position_limits_rad'] if probe else
                  profile['position_limits_rad'])
    if operation == 'remember-top':
        if mock or external_sim:
            raise ValueError('remember-top requires read-only measurements from the real robot')
        joints = sample_stationary_joint_positions(
            config['robot']['ip'], expected_model=config['robot']['model'])
        robot_model, _ = robot_parameters(
            config, mode, share, path, False, probe)
        world_from_flange = forward_kinematics_from_urdf(
            robot_model[0]['robot_description'], joints)
        remembered = remember_top_direction(world_from_flange, joints)
        config.setdefault('mount_setup', {}).update(remembered)
        save_yaml_atomic(values['output_file'], config)
        print('Запомнено read-only: робот не запускал controller и не двигался; '
              'полная FK поза рассчитана из шести измеренных суставов.', flush=True)
        return []
    config_path = Path(tempfile.mkdtemp(prefix='face_mount_setup_')) / 'resolved.yaml'
    config_path.write_text(yaml.safe_dump(config))
    setup_arguments = [
        '--operation', values['operation'], '--mode', mode,
        '--config-file', str(config_path), '--output-file', values['output_file'],
        '--marker-side-mm', values['marker_side_mm'],
        '--show-image', values['show_image']]
    if profile:
        setup_arguments.extend(['--controller', profile['controller_name']])
    node_parameters = [{'use_sim_time': use_sim_time}]
    if operation != 'view':
        node_parameters[0].update(
            joint_limits_rad=[bound for pair in limits for bound in pair],
            planning_group=profile['planning_group'])
    setup_node = Node(
        package='face_tracking_arm', executable='mount_setup_node.py', output='screen',
        arguments=setup_arguments,
        parameters=node_parameters)

    def cleanup_temp(event, context):
        shutil.rmtree(config_path.parent, ignore_errors=True)
        return []

    cleanup = RegisterEventHandler(OnShutdown(on_shutdown=cleanup_temp))
    actions = [cleanup]
    if operation in ('view', 'calibrate') and not external_sim:
        actions.extend(camera_actions(config, mode, driver_only=(operation == 'view')))
    if operation == 'view':
        actions.append(setup_node)
    elif external_sim:

        def finish_external_setup(event, context):
            if context.is_shutdown:
                return []
            if event.returncode != 0:
                raise RuntimeError(f'Gazebo mount calibration failed ({event.returncode})')
            return [EmitEvent(event=Shutdown(reason='Gazebo mount calibration completed'))]

        finish = RegisterEventHandler(OnProcessExit(
            target_action=setup_node, on_exit=finish_external_setup))
        actions.extend([finish, setup_node])
    else:
        actions.extend(mount_setup_robot_actions(
            config, mode, share, config_path, config_path.parent, mock, probe, setup_node,
            use_sim_time=use_sim_time))
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory('face_tracking_arm'))
    arguments = [
        DeclareLaunchArgument('config_file', default_value=str(share / 'config/hardware.yaml')),
        DeclareLaunchArgument('mode', default_value='mono_cpu',
                              choices=['mono_cpu', 'stereo', 'rgbd']),
        DeclareLaunchArgument('operation', default_value='calibrate',
                              choices=['remember-top', 'set-top', 'calibrate', 'view']),
        DeclareLaunchArgument('marker_side_mm', default_value='45'),
        DeclareLaunchArgument('output_file', default_value='/output/hardware.yaml'),
        DeclareLaunchArgument('mock_hardware', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('show_image', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('external_sim', default_value='false', choices=['true', 'false']),
    ]
    setup = OpaqueFunction(function=_setup)
    return LaunchDescription([*arguments, setup])
