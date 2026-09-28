# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Zero-size equipment omits collision shapes while preserving the tracking frames."""

from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.hardware_description import (  # noqa: E402, I100
    control_parameters, robot_parameters)


@pytest.mark.parametrize('mode', ['mono_cpu', 'stereo'])
@pytest.mark.parametrize('external_optical_tf', [False, True])
def test_zero_monitor_preserves_camera_and_control_frames(tmp_path, mode, external_optical_tf):
    config = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    config['monitor']['size_m'] = [0., 0., 0.]
    config['monitor']['mass_kg'] = 0.
    config['camera_mount']['publish_optical_tf'] = not external_optical_tf
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    share = Path(get_package_share_directory('face_tracking_arm'))
    params, _ = robot_parameters(config, mode, share, path, True)
    xml = params[0]['robot_description']
    robot = ET.fromstring(xml)
    monitor = robot.find("link[@name='monitor_link']")
    assert monitor is not None
    assert monitor.find('visual') is None
    assert monitor.find('collision') is None
    assert monitor.find('inertial') is None
    assert robot.find("link[@name='monitor_control_frame']") is not None
    control_origin = robot.find("joint[@name='monitor_control_joint']/origin")
    assert [float(v) for v in control_origin.get('xyz').split()] == [0., 0., 0.]
    mount_origin = robot.find("joint[@name='monitor_mount_joint']/origin")
    assert [float(v) for v in mount_origin.get('xyz').split()] == config['monitor']['mount_xyz_m']
    camera_origin = robot.find("joint[@name='hardware_camera_mount']/origin")
    assert [float(v) for v in camera_origin.get('xyz').split()] == config['camera_mount']['xyz_m']
    camera = robot.find(f"link[@name='{config['camera_mount']['body_frame']}']")
    assert camera.find('collision/geometry/box') is not None
    frames = (['tracking_gaze_optical'] if external_optical_tf else
              [config[mode][side]['frame_id'] for side in
               (('left', 'right') if mode == 'stereo' else ('left',))])
    for name in frames:
        assert robot.find(f"link[@name='{name}']") is not None
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text(xml)
    subprocess.run(['check_urdf', str(urdf)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('shape,dimensions', [('cylinder', [0., 0.]), ('box', [0., 0., 0.])])
def test_zero_table_reaches_both_tracking_and_servo_readiness(tmp_path, shape, dimensions):
    config = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    share = Path(get_package_share_directory('face_tracking_arm'))
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    _, limits = robot_parameters(config, 'mono_cpu', share, path, True)
    _, before, _, _ = control_parameters(config, share, limits)
    assert before['table_collision_enabled'] is True
    config['table'].update(shape=shape, dimensions_m=dimensions)
    _, motion, tracking, _ = control_parameters(config, share, limits)
    assert motion['table_collision_enabled'] is False
    assert tracking['table_shape'] == shape
    assert tracking['table_dimensions_m'] == dimensions
    assert tracking['table_center_m'] == config['table']['center_m']
