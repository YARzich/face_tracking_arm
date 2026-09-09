# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Hardware configuration and robot assembly checks without camera, arm or network."""

import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.hardware_config import validate_config  # noqa: E402, I100
from face_tracking_bringup.hardware_description import (  # noqa: E402
    control_parameters, robot_parameters)


@pytest.fixture
def config():
    c = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    c['perception']['python_executable'] = sys.executable
    c['mono_cpu']['source'] = 'ros'
    return c


def test_real_address_is_required_but_camera_only_never_needs_a_robot(config):
    with pytest.raises(ValueError, match='robot.ip'):
        validate_config(config, 'mono_cpu')
    validate_config(config, 'mono_cpu', camera_only=True)
    validate_config(config, 'mono_cpu', mock=True)
    config['robot']['ip'] = '192.168.1.161'
    validate_config(config, 'stereo')


@pytest.mark.parametrize('section,key,value', [
    ('motion', 'speed_scale', 0), ('motion', 'speed_scale', 1.01),
    ('stereo', 'sync_slop_sec', .5), ('stereo', 'baseline_m', -.08),
    ('perception', 'max_frame_age_sec', 4), ('camera_mount', 'mass_kg', .9),
    ('table', 'dimensions_m', [0, .5]), ('tracking', 'idle_behavior', 'circle')])
def test_invalid_settings_fail_before_starting_drivers(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config, 'stereo', mock=True)


def test_stereo_cannot_accidentally_use_the_same_input_twice(config):
    config['stereo']['right']['image_topic'] = config['stereo']['left']['image_topic']
    with pytest.raises(ValueError, match='different cameras'):
        validate_config(config, 'stereo', mock=True)


@pytest.mark.parametrize('path,value', [
    (('camera_mount',), None), (('mono_cpu', 'left'), []),
    (('mono_cpu', 'left', 'image_topic'), None),
    (('camera_mount', 'body_frame'), ['camera']),
    (('perception', 'model_dir'), None),
])
def test_malformed_yaml_reports_the_setting_instead_of_a_python_exception(config, path, value):
    owner = config
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    with pytest.raises(ValueError, match=path[-1]):
        validate_config(config, 'mono_cpu', mock=True)


def test_missing_nested_setting_names_its_full_path(config):
    del config['motion']['rest_joints_deg']
    with pytest.raises(ValueError, match=r'motion.rest_joints_deg'):
        validate_config(config, 'mono_cpu', mock=True)


def test_relative_paths_use_the_callers_current_directory(config, tmp_path, monkeypatch):
    config['perception']['model_dir'] = 'models'
    original = copy.deepcopy(config)
    monkeypatch.chdir(tmp_path)
    result = validate_config(config, 'mono_cpu', mock=True)
    assert result['perception']['model_dir'] == str(tmp_path / 'models')
    assert config == original


@pytest.mark.parametrize('frame', ['/camera', 'camera\tlink', 'camera\nlink'])
def test_body_frame_obeys_the_same_rules_as_the_optical_frames(config, frame):
    config['camera_mount']['body_frame'] = frame
    with pytest.raises(ValueError, match='frame'):
        validate_config(config, 'mono_cpu', mock=True)


def test_integer_yaml_values_become_the_ros_parameter_types(config):
    config['perception']['max_processing_fps'] = 15
    config['tracking']['return_to_rest_delay_sec'] = 3
    config['table']['shape'] = 'box'
    config['table']['dimensions_m'] = [1, 1, .05]
    result = validate_config(config, 'mono_cpu', mock=True)
    assert type(result['perception']['max_processing_fps']) is float
    assert type(result['tracking']['return_to_rest_delay_sec']) is float
    assert all(type(x) is float for x in result['table']['dimensions_m'])
    assert type(result['perception']['threads']) is int


def test_usb_requires_calibration_matching_capture_resolution(config, tmp_path):
    camera = config['mono_cpu']['left']
    config['mono_cpu']['source'] = 'usb'
    validate_config(config, 'mono_cpu', driver_only=True)
    with pytest.raises(ValueError, match='calibration_file'):
        validate_config(config, 'mono_cpu', camera_only=True)
    data = {'image_width': 320, 'image_height': 480,
            'camera_matrix': {'data': [500., 0, 320, 0, 500., 240, 0, 0, 1]},
            'rectification_matrix': {'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            'projection_matrix': {'data': [500., 0, 320, 0, 0, 500., 240, 0, 0, 0, 1, 0]}}
    path = tmp_path / 'camera.yaml'
    path.write_text(yaml.safe_dump(data))
    camera['calibration_file'] = str(path)
    with pytest.raises(ValueError, match='resolution'):
        validate_config(config, 'mono_cpu', camera_only=True)
    data['image_width'] = 640
    path.write_text(yaml.safe_dump(data))
    validate_config(config, 'mono_cpu', camera_only=True)
    # A nonpositive vertical focal length is invalid even when horizontal fx is valid.
    data['camera_matrix']['data'][4] = 0
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='focal length'):
        validate_config(config, 'mono_cpu', camera_only=True)
    for malformed in ('', '[]', 'camera_matrix: ['):
        path.write_text(malformed)
        with pytest.raises(ValueError, match='camera.yaml'):
            validate_config(config, 'mono_cpu', camera_only=True)


@pytest.mark.parametrize('mode', ['mono_cpu', 'stereo'])
def test_hardware_model_has_configured_frames_limits_and_no_gazebo(config, tmp_path, mode):
    share = Path(get_package_share_directory('face_tracking_arm'))
    config['robot']['base_xyz_m'] = [.1, .2, .82]
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    model, limits = robot_parameters(config, mode, share, path, True)
    root = ET.fromstring(model[0]['robot_description'])
    assert root.find('gazebo') is None
    assert root.findtext('ros2_control/hardware/plugin') == 'mock_components/GenericSystem'
    assert root.find("joint[@name='world_joint']/origin").get('xyz') == '0.1 0.2 0.82'
    for side in ('left', 'right') if mode == 'stereo' else ('left',):
        assert root.find(f"link[@name='{config[mode][side]['frame_id']}']") is not None
    # A velocity reduction changes working limits, never the manufacturer's position limits.
    upper = float(root.find("joint[@name='joint1']/limit").get('upper'))
    assert upper == pytest.approx(2 * 3.14159265)
    assert limits['joint_limits']['joint1']['max_velocity'] == .25
    servo, motion, _, controllers = control_parameters(config, share, limits)
    assert not servo['publish_joint_velocities'] and not servo['publish_joint_accelerations']
    assert motion['max_joint_jerk_rad_s3'] == [5.] * 6
    assert all(not c['ros__parameters']['use_sim_time'] for c in controllers.values())
    assert controllers['controller_manager']['ros__parameters']['update_rate'] == 100
    bad = copy.deepcopy(config)
    bad['motion']['rest_joints_deg'][2] = -10
    with pytest.raises(ValueError, match='joint3'):
        robot_parameters(bad, mode, share, path, True)
