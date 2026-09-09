# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Standalone RGB face test in Gazebo Harmonic; all outputs are image-space data."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess
from launch.actions import OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def validate_files(context):
    """Fail before starting Gazebo if the separate environment/assets are absent."""
    python = Path(LaunchConfiguration('python_executable').perform(context)).expanduser()
    assets = Path(LaunchConfiguration('assets_dir').perform(context)).expanduser()
    if not python.is_file() or not (assets / 'person_standing/model.sdf').is_file():
        raise RuntimeError('Prepare the face test first: see docs/face_detection_test.md')
    return []


def generate_launch_description():
    share = Path(get_package_share_directory('face_tracking_arm'))
    prefix = Path(get_package_prefix('face_tracking_arm'))
    cache = Path.home() / '.cache/face_tracking_arm'
    declarations = [
        DeclareLaunchArgument('detector', default_value='yunet',
                              choices=['yunet', 'yolov5n_face', 'yolo_facev2n']),
        DeclareLaunchArgument('model_dir', default_value=str(cache / 'models')),
        DeclareLaunchArgument('assets_dir', default_value=str(cache / 'gazebo_models')),
        DeclareLaunchArgument('python_executable',
                              default_value=str(cache / 'perception_venv/bin/python')),
        DeclareLaunchArgument('confidence', default_value='0.6'),
        DeclareLaunchArgument('threads', default_value='2'),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('show_image', default_value='true', choices=['true', 'false']),
    ]
    resources = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [LaunchConfiguration('assets_dir'), os.pathsep,
         os.environ.get('GZ_SIM_RESOURCE_PATH', '')])
    world = str(share / 'worlds/face_detection_test.sdf')
    gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world], output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')))
    gazebo_server = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '--headless-rendering', world], output='screen',
        condition=IfCondition(LaunchConfiguration('headless')))
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='face_test_bridge', output='screen',
        parameters=[{'config_file': str(share / 'config/face_detection_bridge.yaml'),
                     'use_sim_time': True}])
    detector = Node(
        executable=LaunchConfiguration('python_executable'),
        arguments=[str(prefix / 'lib/face_tracking_arm/face_detection_node.py')],
        name='face_detection_test', output='screen',
        parameters=[{
            'use_sim_time': True,
            'detector': LaunchConfiguration('detector'),
            'model_dir': LaunchConfiguration('model_dir'),
            'confidence': ParameterValue(LaunchConfiguration('confidence'), value_type=float),
            'threads': ParameterValue(LaunchConfiguration('threads'), value_type=int),
        }])
    viewer = Node(
        package='rviz2', executable='rviz2',
        name='face_test_image_view',
        arguments=['-d', str(share / 'config/face_detection_view.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('show_image'), "' == 'true' and '",
            LaunchConfiguration('headless'), "' == 'false'"])))
    camera_pose = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='face_test_camera_pose', parameters=[{'use_sim_time': True}],
        arguments=['--x', '0', '--y', '-1.2', '--z', '1.65', '--yaw', '1.57079632679',
                   '--frame-id', 'world', '--child-frame-id', 'face_test_camera_link'])
    optical_pose = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='face_test_optical_pose', parameters=[{'use_sim_time': True}],
        arguments=['--roll', '-1.57079632679', '--yaw', '-1.57079632679',
                   '--frame-id', 'face_test_camera_link',
                   '--child-frame-id', 'face_test_camera_optical_frame'])
    # An unavailable model or a closed simulator must not leave orphan processes.
    shutdown_handlers = []
    for process in (detector, gazebo_gui, gazebo_server):
        handler = RegisterEventHandler(OnProcessExit(
            target_action=process,
            on_exit=[EmitEvent(event=Shutdown(reason='Face test process exited'))]))
        shutdown_handlers.append(handler)
    description = LaunchDescription(declarations)
    description.add_action(OpaqueFunction(function=validate_files))
    description.add_action(resources)
    for handler in shutdown_handlers:
        description.add_action(handler)
    for action in (gazebo_gui, gazebo_server, bridge, detector, camera_pose, optical_pose, viewer):
        description.add_action(action)
    return description
