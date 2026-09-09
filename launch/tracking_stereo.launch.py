# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run monitor-mounted face detection with the existing arm controller."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('face_tracking_arm')
    scenario = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(share + '/launch/tracking_vision_common.launch.py'),
        launch_arguments={'mode': 'stereo'}.items())
    return LaunchDescription([scenario])
