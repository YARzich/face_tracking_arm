# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Standalone stereo face depth scenario."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = Path(get_package_share_directory('face_tracking_arm'))
    scenario = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(share / 'launch/face_depth_common.launch.py')),
        launch_arguments={'mode': 'stereo'}.items())
    return LaunchDescription([scenario])
