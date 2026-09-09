# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Assemble one depth experiment; the three public launch files select its mode."""

import os
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction
from launch.actions import RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
import yaml


def assemble(context):
    share = Path(get_package_share_directory('face_tracking_arm'))
    prefix = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'

    def value(name):
        return LaunchConfiguration(name).perform(context)

    mode = value('mode')
    headless, reference = value('headless') == 'true', value('reference') == 'true'
    baseline = float(value('baseline'))
    if not 0.02 <= baseline <= 0.25:
        raise ValueError('baseline must be 0.02..0.25 meters')
    python = str(Path(value('python_executable')).expanduser())
    assets = Path(value('assets_dir')).expanduser()
    if not Path(python).is_file() or not (assets / 'person_standing/model.sdf').is_file():
        raise RuntimeError('Prepare the 2D test environment/assets first')
    if mode == 'mono_gpu' and not Path(value('depth_weights')).expanduser().is_file():
        raise RuntimeError('Metric depth weights missing; see docs/face_depth_tests.md')
    directory = tempfile.TemporaryDirectory(prefix='face_depth_')
    world = Path(directory.name) / 'world.sdf'
    world.write_text(xacro.process_file(
        str(share / 'worlds/face_depth_test.sdf.xacro'), mappings={
            'stereo': str(mode == 'stereo').lower(), 'reference': str(reference).lower(),
            'baseline': str(baseline)}).toxml())
    bridges = yaml.safe_load((share / 'config/face_detection_bridge.yaml').read_text())
    extra = []
    if mode == 'stereo':
        extra += [('/face_test/right/image_raw', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
                  ('/face_test/right/camera_info', 'sensor_msgs/msg/CameraInfo',
                   'gz.msgs.CameraInfo')]
    if reference:
        extra.append(('/face_test/reference/depth', 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
    for topic, ros_type, gz_type in extra:
        bridges.append({'ros_topic_name': topic, 'gz_topic_name': topic,
                        'ros_type_name': ros_type, 'gz_type_name': gz_type,
                        'direction': 'GZ_TO_ROS', 'qos_profile': 'SENSOR_DATA'})
    bridge_file = Path(directory.name) / 'bridge.yaml'
    bridge_file.write_text(yaml.safe_dump(bridges))
    resource_path = str(assets) + os.pathsep + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    environment = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path)
    gazebo = ExecuteProcess(cmd=['gz', 'sim', '-r'] + (
        ['-s', '--headless-rendering'] if headless else []) + [str(world)], output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  name='face_depth_bridge', parameters=[{'use_sim_time': True,
                                                        'config_file': str(bridge_file)}])
    parameters = {'use_sim_time': True, 'mode': mode, 'detector': value('detector'),
                  'model_dir': value('model_dir'), 'threads': int(value('threads')),
                  'confidence': float(value('confidence')),
                  'face_width_m': float(value('face_width_m')),
                  'depth_weights': value('depth_weights'), 'depth_source': value('depth_source'),
                  'depth_input_size': int(value('depth_input_size')),
                  'depth_scale': float(value('depth_scale'))}
    estimator = Node(executable=python, arguments=[str(prefix / 'face_position_node.py')],
                     name='face_position', parameters=[parameters], output='screen')
    nodes = [gazebo, bridge, estimator]
    transforms = [
        ('camera_pose', ['--x', '0', '--y', '-1.2', '--z', '1.65', '--yaw', '1.5707963267948966',
                         '--frame-id', 'world', '--child-frame-id', 'face_test_camera_link']),
        ('optical_pose', ['--roll', '-1.5707963267948966', '--yaw', '-1.5707963267948966',
                          '--frame-id', 'face_test_camera_link',
                          '--child-frame-id', 'face_test_camera_optical_frame'])]
    if mode == 'stereo':
        transforms.append(('right_pose', ['--x', str(baseline), '--frame-id',
                                          'face_test_camera_optical_frame', '--child-frame-id',
                                          'face_test_right_optical_frame']))
    for name, arguments in transforms:
        nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
                          name='face_depth_' + name, arguments=arguments,
                          parameters=[{'use_sim_time': True}]))
    if reference:
        nodes.append(Node(executable=python, arguments=[str(prefix / 'face_depth_reference.py')],
                          name='face_depth_reference', parameters=[{'use_sim_time': True}],
                          output='screen'))
    if not headless and value('show_image') == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', name='face_depth_view',
                          arguments=['-d', str(share / 'config/face_depth_view.rviz')],
                          parameters=[{'use_sim_time': True}]))
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process, on_exit=[EmitEvent(event=Shutdown(reason='Depth test exited'))]))
                for process in (gazebo, estimator)]
    cleanup = RegisterEventHandler(OnShutdown(on_shutdown=lambda event, ctx: directory.cleanup()))
    return [environment, cleanup, *handlers, *nodes]


def generate_launch_description():
    cache = Path.home() / '.cache/face_tracking_arm'
    defaults = {
        'mode': 'mono_cpu', 'detector': 'yunet', 'headless': 'false', 'show_image': 'true',
        'reference': 'true', 'baseline': '0.08', 'threads': '2', 'confidence': '0.6',
        'face_width_m': '0.16', 'depth_input_size': '280', 'depth_scale': '1.0',
        'model_dir': str(cache / 'models'), 'assets_dir': str(cache / 'gazebo_models'),
        'python_executable': str(cache / 'perception_venv/bin/python'),
        'depth_weights': str(cache / 'models/depth_anything_v2_metric_hypersim_vits.pth'),
        'depth_source': str(cache / 'depth_anything_metric'),
    }
    choices = {'mode': ['mono_cpu', 'mono_gpu', 'stereo'],
               'detector': ['yunet', 'yolov5n_face', 'yolo_facev2n']}
    choices.update({name: ['true', 'false'] for name in ('headless', 'show_image', 'reference')})
    declarations = [DeclareLaunchArgument(name, default_value=default, choices=choices.get(name))
                    for name, default in defaults.items()]
    action = OpaqueFunction(function=assemble)
    return LaunchDescription([*declarations, action])
