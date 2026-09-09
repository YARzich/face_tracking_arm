# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Connect monitor cameras and measured faces to the existing tracking scenario."""

import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def assemble(context):
    share = Path(get_package_share_directory('face_tracking_arm'))
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    sys.path.insert(0, str(scripts))
    from face_tracking_perception.arm_scene import write_vision_world

    def value(name):
        return LaunchConfiguration(name).perform(context)

    mode = value('mode')
    assets = Path(value('assets_dir')).expanduser()
    python = str(Path(value('python_executable')).expanduser())
    if not (assets / 'person_standing/model.sdf').is_file() or not Path(python).is_file():
        raise RuntimeError('Prepare the face test first; see docs/face_detection_test.md')
    directory = tempfile.TemporaryDirectory(prefix='tracking_vision_')
    world_path = Path(directory.name) / 'world.sdf'
    write_vision_world(share / 'worlds/lite6_table.sdf', world_path, int(value('people_count')))
    world = ET.parse(world_path)
    world.getroot().find('world').append(
        ET.parse(share / 'config/tracking_vision_gazebo.xml').getroot())
    world.write(world_path, encoding='utf-8', xml_declaration=True)
    bridges = yaml.safe_load((share / 'config/face_detection_bridge.yaml').read_text())
    bridges = [item for item in bridges if item['ros_topic_name'] != '/clock']
    if mode == 'stereo':
        for topic, ros_type, gz_type in (
                ('image_raw', 'Image', 'Image'), ('camera_info', 'CameraInfo', 'CameraInfo')):
            bridges.append({'ros_topic_name': '/face_test/right/' + topic,
                            'gz_topic_name': '/face_test/right/' + topic,
                            'ros_type_name': 'sensor_msgs/msg/' + ros_type,
                            'gz_type_name': 'gz.msgs.' + gz_type,
                            'direction': 'GZ_TO_ROS', 'qos_profile': 'SENSOR_DATA'})
    bridge_path = Path(directory.name) / 'bridge.yaml'
    bridge_path.write_text(yaml.safe_dump(bridges))
    resource_path = str(assets) + os.pathsep + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    environment = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path)
    tracking = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(share / 'launch/tracking_sim.launch.py')),
        launch_arguments={'headless': value('headless'), 'test_face_scenario': 'disabled',
                          'visualize_target': 'false', 'servo_backend': 'collision_aware',
                          'camera_mode': mode, 'camera_baseline': value('baseline'),
                          'idle_behavior': value('idle_behavior'),
                          'world_file': str(world_path)}.items())
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  name='monitor_camera_bridge', parameters=[{'use_sim_time': True,
                                                            'config_file': str(bridge_path)}])
    detector = Node(
        executable=python, arguments=[str(scripts / 'face_position_node.py')],
        name='face_position', output='screen', parameters=[{
            'use_sim_time': True, 'mode': mode, 'detector': value('detector'),
            'selection_frame': 'link_base',
            'model_dir': value('model_dir'), 'threads': int(value('threads')),
            'face_width_m': float(value('face_width_m')),
            'confidence': float(value('confidence'))}],
        remappings=[('/face_test/center_3d', '/face/center')])
    srdf = ET.parse(share / 'config/moveit/lite6.srdf').getroot()
    rest = srdf.find("group_state[@name='rest']")
    joints = {joint.get('name'): float(joint.get('value')) for joint in rest}
    appearances = Node(
        executable=python, arguments=[str(scripts / 'face_appearances.py')],
        name='face_appearances', output='screen', parameters=[
            str(share / 'config/face_appearances.yaml'), {
                'use_sim_time': True, 'visible_sec': float(value('visible_sec')),
                'rest_sec': float(value('rest_sec')), 'cycles': int(value('cycles')),
                'idle_behavior': value('idle_behavior'),
                'people_count': int(value('people_count')),
                'rest_joints': [joints[f'joint{i}'] for i in range(1, 7)]}])
    nodes = [bridge, detector, appearances]
    if value('headless') == 'false' and value('show_image') == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', name='face_camera_view',
                          arguments=['-d', str(share / 'config/tracking_vision_view.rviz')],
                          parameters=[{'use_sim_time': True}]))
    guards = [RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=[EmitEvent(event=Shutdown(reason='Vision process exited'))]))
              for node in (bridge, detector, appearances)]
    cleanup = RegisterEventHandler(OnShutdown(on_shutdown=lambda event, ctx: directory.cleanup()))
    return [environment, cleanup, *guards, tracking, *nodes]


def generate_launch_description():
    cache = Path.home() / '.cache/face_tracking_arm'
    defaults = {
        'mode': 'mono_cpu', 'headless': 'false', 'show_image': 'true',
        'idle_behavior': 'rest', 'people_count': '1',
        'detector': 'yunet', 'baseline': '0.08', 'threads': '2',
        'confidence': '0.6', 'face_width_m': '0.16', 'visible_sec': '12.0',
        'rest_sec': '8.0', 'cycles': '0', 'model_dir': str(cache / 'models'),
        'assets_dir': str(cache / 'gazebo_models'),
        'python_executable': str(cache / 'perception_venv/bin/python')}
    choices = {'mode': ['mono_cpu', 'stereo'], 'headless': ['true', 'false'],
               'show_image': ['true', 'false'],
               'idle_behavior': ['rest', 'search_sweep', 'search_local_then_sweep'],
               'people_count': ['1', '2'],
               'detector': ['yunet', 'yolov5n_face', 'yolo_facev2n']}
    declarations = [DeclareLaunchArgument(name, default_value=default, choices=choices.get(name))
                    for name, default in defaults.items()]
    setup = OpaqueFunction(function=assemble)
    return LaunchDescription([*declarations, setup])
