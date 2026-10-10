# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Consume a generic ROS color/depth pair and publish a stamped 3D face point."""

from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    defaults = {
        'image_topic': '/camera/camera/color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'color_frame': 'camera_color_optical_frame',
        'rectification': 'raw', 'rgbd_sync_slop_sec': '0.02',
        'max_frame_age_sec': '0.20', 'source_timeout_sec': '0.25',
        'detector': 'yunet', 'model_dir': '/opt/face_tracking_models',
        'confidence': '0.6', 'threads': '2',
    }
    declarations = [DeclareLaunchArgument(name, default_value=default)
                    for name, default in defaults.items()]
    estimator = Node(
        executable=str(scripts / 'face_position_node.py'), name='face_position',
        output='screen', parameters=[{
            'use_sim_time': False, 'mode': 'rgbd', 'detector': LaunchConfiguration('detector'),
            'model_dir': LaunchConfiguration('model_dir'),
            'confidence': LaunchConfiguration('confidence'),
            'threads': LaunchConfiguration('threads'),
            'rectification': LaunchConfiguration('rectification'),
            'rgbd_sync_slop_sec': LaunchConfiguration('rgbd_sync_slop_sec'),
            'max_frame_age_sec': LaunchConfiguration('max_frame_age_sec'),
            'source_timeout_sec': LaunchConfiguration('source_timeout_sec'),
            'input_left_frame': LaunchConfiguration('color_frame'),
            'selection_frame': LaunchConfiguration('color_frame'), 'world_frame': '',
            'rgbd_depth_topic': '/face_test/depth/image_raw'}],
        remappings=[('/face_test/image_raw', LaunchConfiguration('image_topic')),
                    ('/face_test/camera_info', LaunchConfiguration('info_topic')),
                    ('/face_test/depth/image_raw', LaunchConfiguration('depth_topic'))])
    return LaunchDescription([*declarations, estimator])
