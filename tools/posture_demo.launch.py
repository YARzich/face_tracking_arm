#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Start a visible Gazebo posture/range demonstration without rebuilding."""

import math
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import yaml


GOOD_FIXTURE = (0.0, 0.0, -1.5, 0.0, -0.07079632679219806,
                3.6732051034624226e-06)
BAD_FIXTURES = {
    6: (0.0, 1.0, -3.8, 0.0, 1.2292036732078009, 3.67320510346244e-06),
    7: (0.0, 1.2, -3.65, 0.0, 0.8792036732078009, 3.6732051034623506e-06),
    8: (0.0, 1.4, -3.5, 0.0, 0.5292036732078017, 3.673205103462396e-06),
    9: (0.0, 1.6, -3.65, 0.0, 0.4792036732078016, 3.6732051034623964e-06),
    10: (0.0, 1.8, -3.8, 0.0, 0.42920367604058574,
         3.6732051031411426e-06),
}
SCENARIOS = ('posture', 'posture_good', 'crossing', 'range', 'boundaries',
             'obstacle')


def _temporary_configs(scenario, minimum, maximum, fixture_q):
    """Copy installed configs and apply demo-only safety/runtime settings."""
    share = Path(get_package_share_directory('face_tracking_arm'))
    servo_source = share / 'config/moveit/collision_aware_servo.yaml'
    tracking_source = share / 'config/tracking.yaml'
    servo = yaml.safe_load(servo_source.read_text(encoding='utf-8'))
    tracking = yaml.safe_load(tracking_source.read_text(encoding='utf-8'))

    recovery = servo.setdefault('recovery', {})
    if not isinstance(recovery, dict):
        raise ValueError('Servo YAML recovery must be a mapping')
    recovery['mode'] = 'disabled'
    tracking_params = tracking['face_tracking_controller']['ros__parameters']
    tracking_params['return_to_rest_delay_sec'] = 3600.0
    tracking_params['face_distance_min_m'] = minimum
    tracking_params['face_distance_max_m'] = maximum

    temporary = tempfile.TemporaryDirectory(prefix='face_posture_demo_')
    directory = Path(temporary.name)
    servo_path = directory / 'collision_aware_servo.yaml'
    tracking_path = directory / 'tracking.yaml'
    positions_path = directory / 'initial_positions.yaml'
    rviz_path = directory / 'posture_demo.rviz'
    servo_path.write_text(
        yaml.safe_dump(servo, sort_keys=False), encoding='utf-8')
    tracking_path.write_text(
        yaml.safe_dump(tracking, sort_keys=False), encoding='utf-8')
    if fixture_q is not None:
        initial_positions = {
            f'joint{index + 1}': value
            for index, value in enumerate(fixture_q)}
        positions_path.write_text(
            yaml.safe_dump({'initial_positions': initial_positions}),
            encoding='utf-8')
    rviz_config = {
        'Panels': [{'Class': 'rviz_common/Displays', 'Name': 'Displays'}],
        'Visualization Manager': {
            'Class': '',
            'Displays': [
                {'Class': 'rviz_default_plugins/RobotModel',
                 'Name': 'RobotModel', 'Enabled': True, 'Alpha': 1,
                 'Description Source': 'Topic',
                 'Description Topic': {
                     'Value': '/robot_description', 'Depth': 1,
                     'Reliability Policy': 'Reliable',
                     'Durability Policy': 'Transient Local',
                     'History Policy': 'Keep Last'},
                 'Visual Enabled': True, 'Collision Enabled': False},
                {'Class': 'rviz_default_plugins/Grid', 'Name': 'Floor',
                 'Enabled': True, 'Cell Size': 0.25,
                 'Plane Cell Count': 20, 'Reference Frame': 'world'},
                {'Class': 'rviz_default_plugins/MarkerArray',
                 'Name': 'POSTURE-1C geometry (S/E/W/B/F)', 'Enabled': True,
                 'Value': True, 'Topic': {
                     'Value': '/posture_demo/markers', 'Depth': 10,
                     'Reliability Policy': 'Reliable',
                     'Durability Policy': 'Volatile',
                     'History Policy': 'Keep Last'}},
                {'Class': 'rviz_default_plugins/TF', 'Name': 'TF',
                 'Enabled': True}],
            'Enabled': True,
            'Global Options': {'Fixed Frame': 'world', 'Frame Rate': 30},
            'Name': 'root',
            'Transformation': {'Current': {
                'Class': 'rviz_default_plugins/TF'}},
            'Views': {'Current': {
                'Class': 'rviz_default_plugins/Orbit', 'Distance': 2.8,
                'Focal Point': {'X': 0.1, 'Y': 0.0, 'Z': 1.3},
                'Pitch': 0.4, 'Yaw': 0.7, 'Target Frame': 'world'}}},
        'Window Geometry': {'Width': 1200, 'Height': 750},
        'Preferences': {'PromptSaveOnExit': False},
    }
    rviz_path.write_text(yaml.safe_dump(rviz_config, sort_keys=False),
                         encoding='utf-8')
    return (temporary, directory, servo_path, tracking_path, positions_path,
            rviz_path)


def _launch_demo(context):
    scenario = context.launch_configurations['scenario']
    if scenario not in SCENARIOS:
        raise ValueError(f'scenario must be one of: {", ".join(SCENARIOS)}')
    minimum = float(context.launch_configurations['face_distance_min_m'])
    maximum = float(context.launch_configurations['face_distance_max_m'])
    fixture_pose_id = int(context.launch_configurations['posture_fixture_id'])
    if (not math.isfinite(minimum) or not math.isfinite(maximum) or
            minimum <= 0.0 or minimum >= maximum):
        raise ValueError(
            'face distance bounds must be finite and satisfy 0 < min < max')

    if scenario == 'posture' and fixture_pose_id not in BAD_FIXTURES:
        raise ValueError('posture_fixture_id must be one of 6, 7, 8, 9, 10')
    fixture_q = (BAD_FIXTURES[fixture_pose_id] if scenario == 'posture' else
                 GOOD_FIXTURE if scenario == 'posture_good' else None)
    temporary, _directory, servo_path, tracking_path, positions_path, rviz_path = (
        _temporary_configs(scenario, minimum, maximum, fixture_q))
    share = Path(get_package_share_directory('face_tracking_arm'))
    launch_args = {
        'headless': 'false',
        'robot_model': 'xarm6',
        'start_pose': 'rest',
        'initial_positions_file': (
            str(positions_path) if fixture_q is not None else ''),
        'camera_mode': 'disabled',
        'idle_behavior': 'rest',
        'require_perception_status': 'false',
        'test_face_scenario': 'disabled',
        'collision_aware_servo_config_file': str(servo_path),
        'tracking_config_file': str(tracking_path),
    }
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(share / 'launch/tracking_sim.launch.py')),
        launch_arguments=launch_args.items())
    publisher_cmd = [
        'python3', str(Path(__file__).with_name('posture_demo_target.py')),
        '--scenario', scenario,
        '--minimum', str(minimum),
        '--maximum', str(maximum),
        '--srdf-path', str(share / 'config/moveit/xarm6.srdf'),
    ]
    if fixture_q is not None:
        publisher_cmd.extend([
            '--fixture-q', *(str(value) for value in fixture_q)])
    if scenario == 'posture':
        publisher_cmd.extend(['--fixture-id', str(fixture_pose_id)])
    publisher = ExecuteProcess(cmd=publisher_cmd, output='screen')
    show_markers = context.launch_configurations['show_markers'].lower() in (
        'true', '1', 'yes')
    marker_view = Node(
        package='rviz2', executable='rviz2', name='posture_demo_rviz',
        arguments=['-d', str(rviz_path)], parameters=[{'use_sim_time': True}],
        output='screen') if show_markers else None

    def cleanup_temp(_context):
        temporary.cleanup()
        return []

    cleanup = RegisterEventHandler(
        OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup_temp)]))
    actions = [cleanup, simulation, publisher]
    if marker_view is not None:
        actions.append(marker_view)
    return actions


def generate_launch_description():
    """Generate the demonstration launch, with conservative default bounds."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'scenario', default_value='posture',
            description=(
                'One of: posture, posture_good, crossing, range, boundaries, '
                'obstacle.'),
        ),
        DeclareLaunchArgument('face_distance_min_m', default_value='0.37'),
        DeclareLaunchArgument('face_distance_max_m', default_value='0.43'),
        DeclareLaunchArgument(
            'posture_fixture_id', default_value='10',
            description='BAD reference pose id 6..10 for scenario=posture.'),
        DeclareLaunchArgument(
            'show_markers', default_value='true',
            description='Open a compact RViz view for B/W/F posture markers.'),
        OpaqueFunction(function=_launch_demo),
    ])
