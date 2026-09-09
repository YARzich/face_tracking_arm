# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Assemble the configured hardware scenario without Gazebo or synthetic targets."""

from pathlib import Path
import sys

from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    sys.path.insert(0, str(scripts))
    from face_tracking_bringup.hardware_launch import generate_hardware_launch
    return generate_hardware_launch(driver_only=True)
