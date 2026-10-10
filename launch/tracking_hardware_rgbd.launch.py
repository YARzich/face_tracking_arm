# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run the configured RGB-D camera and face-tracking arm without Gazebo."""

from pathlib import Path
import sys

from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    sys.path.insert(0, str(scripts))
    from face_tracking_bringup.hardware_launch import generate_hardware_launch
    return generate_hardware_launch('rgbd')
