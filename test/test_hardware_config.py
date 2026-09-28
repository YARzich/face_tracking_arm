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
    control_parameters, perception_parameters, robot_parameters)


@pytest.fixture
def config():
    c = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    c['perception']['python_executable'] = sys.executable
    c['mono_cpu']['source'] = 'ros'
    return c


@pytest.mark.parametrize('mode,noise,acceleration', [
    ('mono_cpu', .03, 1.0), ('stereo', .02, 1.5)])
def test_hardware_template_enables_the_selected_kalman_profile(config, mode, noise, acceleration):
    validated = validate_config(config, mode, mock=True)
    params = perception_parameters(validated, mode)
    assert params['point_smoothing'] is True
    assert params['point_smoothing_method'] == 'kalman'
    assert params['point_kalman_measurement_std_m'] == noise
    assert params['point_kalman_acceleration_std_mps2'] == acceleration
    assert params['point_smoothing_reset_after_sec'] == .25
    assert params['selection_frame'] == 'link_base'
    assert params['world_frame'] == 'world'
    assert params['input_left_frame'] == config[mode]['left']['frame_id']
    assert params['use_sim_time'] is False
    assert 'python_executable' not in params
    if mode == 'stereo':
        assert params['input_right_frame'] == config[mode]['right']['frame_id']
        assert params['sync_slop_sec'] == config[mode]['sync_slop_sec']


def test_old_hardware_yaml_does_not_silently_enable_smoothing(config):
    del config['point_filter']
    validated = validate_config(config, 'mono_cpu', mock=True)
    assert perception_parameters(validated, 'mono_cpu')['point_smoothing'] is False


@pytest.mark.parametrize('mode', ['mono_cpu', 'stereo'])
def test_filter_tuning_is_read_again_from_the_external_file(config, mode, tmp_path):
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))

    def read_parameters():
        current = validate_config(yaml.safe_load(path.read_text()), mode, mock=True)
        return perception_parameters(current, mode, camera_only=True)

    before = read_parameters()
    config['point_filter'][mode] = {'measurement_std_m': .07, 'acceleration_std_mps2': 2}
    config['point_filter']['reset_after_sec'] = .4
    path.write_text(yaml.safe_dump(config))
    after = read_parameters()
    assert before['point_kalman_measurement_std_m'] != after['point_kalman_measurement_std_m']
    assert after['point_kalman_measurement_std_m'] == .07
    assert after['point_kalman_acceleration_std_mps2'] == 2.0
    assert type(after['point_kalman_acceleration_std_mps2']) is float
    assert after['point_smoothing_reset_after_sec'] == .4
    assert after['selection_frame'] == config[mode]['left']['frame_id']
    assert after['world_frame'] == ''


def test_filter_can_be_disabled_or_changed_without_changing_the_detector(config):
    original = copy.deepcopy(config['perception'])
    config['point_filter'] = {
        'enabled': False, 'method': 'one_euro',
        'one_euro': {'min_cutoff_hz': 2, 'beta': 0, 'derivative_cutoff_hz': 3}}
    validated = validate_config(config, 'mono_cpu', mock=True)
    params = perception_parameters(validated, 'mono_cpu')
    assert params['point_smoothing'] is False
    assert params['point_smoothing_method'] == 'one_euro'
    assert params['point_smoothing_min_cutoff_hz'] == 2.0
    assert params['point_smoothing_beta'] == 0.0
    assert params['point_smoothing_derivative_cutoff_hz'] == 3.0
    assert params['detector'] == original['detector']
    assert config['perception'] == original


@pytest.mark.parametrize('settings,path', [
    (None, 'point_filter'),
    ({'enable': True}, 'point_filter'),
    ({'enabled': 'true'}, 'point_filter.enabled'),
    ({'method': 'Kalman'}, 'point_filter.method'),
    ({'reset_after_sec': 0}, 'point_filter.reset_after_sec'),
    ({'reset_after_sec': float('nan')}, 'point_filter.reset_after_sec'),
    ({'mono_cpu': []}, 'point_filter.mono_cpu'),
    ({'stereo': {'measurement_std_m': -1}}, 'point_filter.stereo.measurement_std_m'),
    ({'mono_cpu': {'measurement_std_m': 0}}, 'point_filter.mono_cpu.measurement_std_m'),
    ({'mono_cpu': {'acceleration_std_mps2': True}}, 'point_filter.mono_cpu.acceleration_std_mps2'),
    ({'stereo': {'acceleration_std_mps2': float('inf')}},
     'point_filter.stereo.acceleration_std_mps2'),
    ({'stereo': {'measurement_std': .02}}, 'point_filter.stereo'),
    ({'one_euro': {'beta': -.1}}, 'point_filter.one_euro.beta'),
    ({'one_euro': {'min_cutoff_hz': 0}}, 'point_filter.one_euro.min_cutoff_hz'),
    ({'one_euro': {'derivative_cutoff_hz': 0}}, 'point_filter.one_euro.derivative_cutoff_hz'),
])
def test_bad_filter_settings_fail_before_drivers_even_when_disabled(config, settings, path):
    config['point_filter'] = settings
    with pytest.raises(ValueError, match=path):
        validate_config(config, 'mono_cpu', mock=True)


def test_real_address_is_required_but_camera_only_never_needs_a_robot(config):
    with pytest.raises(ValueError, match='robot.ip'):
        validate_config(config, 'mono_cpu')
    validate_config(config, 'mono_cpu', camera_only=True)
    validate_config(config, 'mono_cpu', mock=True)
    config['robot']['ip'] = '192.168.1.161'
    validate_config(config, 'stereo')


@pytest.mark.parametrize('section,key,value', [
    ('motion', 'speed_scale', 0), ('motion', 'speed_scale', float('nan')),
    ('stereo', 'sync_slop_sec', -.5), ('stereo', 'baseline_m', -.08),
    ('perception', 'max_frame_age_sec', 0), ('camera_mount', 'mass_kg', 5),
    ('table', 'dimensions_m', [0, .5]), ('tracking', 'idle_behavior', 'circle')])
def test_invalid_settings_fail_before_starting_drivers(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config, 'stereo', mock=True)


@pytest.mark.parametrize('shape,dimensions', [('cylinder', [0, 0]), ('box', [0, 0, 0])])
@pytest.mark.parametrize('mass', [0, .692])
def test_zero_size_equipment_is_valid_without_losing_mount_coordinates(
        config, shape, dimensions, mass):
    config['table'].update(shape=shape, dimensions_m=dimensions)
    config['monitor'].update(size_m=[0, 0, 0], mass_kg=mass)
    result = validate_config(config, 'mono_cpu', mock=True)
    assert result['table']['dimensions_m'] == dimensions
    assert result['monitor']['size_m'] == [0., 0., 0.]
    assert result['monitor']['mass_kg'] == mass
    assert result['monitor']['mount_xyz_m'] == config['monitor']['mount_xyz_m']
    assert result['camera_mount'] == config['camera_mount']
    assert result['robot']['base_xyz_m'] == config['robot']['base_xyz_m']


@pytest.mark.parametrize('section,key,value', [
    ('monitor', 'size_m', [0, .3, .2]), ('monitor', 'size_m', [-.1, -.1, -.1]),
    ('monitor', 'size_m', [0, 0]), ('monitor', 'size_m', [0, float('nan'), 0]),
    ('monitor', 'size_m', [False, 0, 0]), ('monitor', 'mass_kg', 0),
    ('table', 'dimensions_m', [0, .2]), ('table', 'dimensions_m', [-1, -1]),
    ('table', 'dimensions_m', [0, 0, 0]), ('camera_mount', 'size_m', [0, 0, 0]),
    ('perception', 'confidence', 0), ('perception', 'confidence', 1),
])
def test_degenerate_geometry_and_detector_thresholds_are_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError, match=section + '.' + key):
        validate_config(config, 'mono_cpu', mock=True)


def test_hidden_monitor_does_not_bypass_the_configured_payload_limit(config):
    config['monitor'].update(size_m=[0, 0, 0], mass_kg=10)
    with pytest.raises(ValueError, match='payload'):
        validate_config(config, 'mono_cpu', mock=True)


def test_camera_tuning_reaches_perception_without_modifying_program_defaults(config):
    config['mono_cpu']['sync_slop_sec'] = .007
    config['stereo'].update(sync_slop_sec=.004, num_disparities=256, block_size=7)
    mono = perception_parameters(validate_config(config, 'mono_cpu', mock=True), 'mono_cpu')
    stereo = perception_parameters(validate_config(config, 'stereo', mock=True), 'stereo')
    assert mono['sync_slop_sec'] == .007
    assert stereo['sync_slop_sec'] == .004
    assert stereo['stereo_num_disparities'] == 256
    assert stereo['stereo_block_size'] == 7
    assert type(stereo['stereo_num_disparities']) is int


def test_old_yaml_keeps_exact_mono_sync_and_original_stereo_search(config):
    del config['mono_cpu']['sync_slop_sec']
    del config['stereo']['num_disparities']
    del config['stereo']['block_size']
    mono = perception_parameters(validate_config(config, 'mono_cpu', mock=True), 'mono_cpu')
    stereo = perception_parameters(validate_config(config, 'stereo', mock=True), 'stereo')
    assert mono['sync_slop_sec'] == 0.0
    assert stereo['stereo_num_disparities'] == 128
    assert stereo['stereo_block_size'] == 5


@pytest.mark.parametrize('mode,key,value', [
    ('mono_cpu', 'sync_slop_sec', -.1), ('mono_cpu', 'sync_slop_sec', float('inf')),
    ('stereo', 'num_disparities', 0), ('stereo', 'num_disparities', 130),
    ('stereo', 'num_disparities', True), ('stereo', 'num_disparities', 128.),
    ('stereo', 'block_size', 0), ('stereo', 'block_size', 4),
    ('stereo', 'block_size', True), ('stereo', 'block_size', '5'),
])
def test_invalid_camera_tuning_names_the_setting(config, mode, key, value):
    config[mode][key] = value
    with pytest.raises(ValueError, match=mode + '.' + key):
        validate_config(config, mode, mock=True)


@pytest.mark.parametrize('num_disparities,block_size', [(640, 5), (128, 481), (624, 33)])
def test_usb_stereo_search_must_fit_the_configured_images(config, num_disparities, block_size):
    config['stereo'].update(source='usb', num_disparities=num_disparities, block_size=block_size)
    with pytest.raises(ValueError, match='capture dimensions'):
        validate_config(config, 'stereo', driver_only=True)


@pytest.mark.parametrize('explicit', [False, True])
def test_return_delay_must_exceed_face_freshness_before_any_driver_starts(config, explicit):
    config['tracking']['return_to_rest_delay_sec'] = .5
    if not explicit:
        del config['tracking']['face_target_freshness_timeout_sec']
    with pytest.raises(ValueError, match='must be greater than'):
        validate_config(config, 'mono_cpu', mock=True)


def test_implicit_freshness_is_applied_to_the_controller_as_validated(config):
    del config['tracking']['face_target_freshness_timeout_sec']
    result = validate_config(config, 'mono_cpu', mock=True)
    assert result['tracking']['face_target_freshness_timeout_sec'] == .5


def test_legacy_geometry_reserve_is_not_silently_presented_as_effective_tuning(config):
    config['controller']['monitor_near_distance_lipschitz_m_per_rad'] = 1.0
    with pytest.warns(UserWarning, match='derived from robot geometry'):
        validate_config(config, 'mono_cpu', mock=True)


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
    servo, motion, _, controllers = control_parameters(config, share, limits, mode)
    assert motion['gaze_frame'] == config[mode]['left']['frame_id']
    assert motion['planning_group_name'] == 'xarm6'
    assert not servo['publish_joint_velocities'] and not servo['publish_joint_accelerations']
    assert motion['max_joint_jerk_rad_s3'] == [5.] * 6
    assert all(not c['ros__parameters']['use_sim_time'] for c in controllers.values())
    assert controllers['controller_manager']['ros__parameters']['update_rate'] == 100
    bad = copy.deepcopy(config)
    bad['motion']['rest_joints_deg'][2] = 30
    with pytest.raises(ValueError, match='joint3'):
        robot_parameters(bad, mode, share, path, True)


@pytest.mark.parametrize('path', [
    ('unexpected',), ('robot', 'modell'), ('stereo', 'baselin_m'),
    ('stereo', 'right', 'devic'), ('controller', 'maximum_velocity'),
    ('limits', 'max_velocity'), ('tracking', 'idle_behaviour')])
def test_unknown_settings_are_not_silently_ignored(config, path):
    owner = config
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = 1
    with pytest.raises(ValueError, match='unknown settings'):
        validate_config(config, 'mono_cpu', mock=True)


def test_finite_positive_values_are_not_capped_by_the_previous_lite6_preset(config):
    config['motion']['speed_scale'] = 1.2
    config['tracking']['safe_reach_radius_m'] = 1.5
    config['perception']['threads'] = 16
    config['perception']['max_frame_age_sec'] = 4
    config['camera_mount']['mass_kg'] = .9
    config['controller']['maximum_angular_reference_radps'] = 2
    config['stereo']['sync_slop_sec'] = .5
    result = validate_config(config, 'stereo', mock=True)
    assert result['motion']['speed_scale'] == 1.2
    assert result['perception']['threads'] == 16
    assert result['controller']['maximum_angular_reference_radps'] == 2


@pytest.mark.parametrize('values', [[1] * 5, [1] * 5 + [0], [True] * 6, [float('inf')] * 6])
def test_joint_limits_require_six_finite_positive_values(config, values):
    config['limits']['max_velocity_rad_s'] = values
    with pytest.raises(ValueError, match='limits.max_velocity_rad_s'):
        validate_config(config, 'mono_cpu', mock=True)


def test_reserved_internal_gaze_frame_is_rejected(config):
    config['mono_cpu']['left']['frame_id'] = 'tracking_gaze_optical'
    with pytest.raises(ValueError, match='reserved'):
        validate_config(config, 'mono_cpu', mock=True)


def test_configured_velocity_is_not_silently_clamped_to_factory_limit(config, tmp_path):
    share = Path(get_package_share_directory('face_tracking_arm'))
    config['limits']['max_velocity_rad_s'] = [100.] * 6
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='above the robot limit'):
        robot_parameters(config, 'mono_cpu', share, path, True)


def test_external_optical_tf_uses_a_separate_internal_gaze_frame(config, tmp_path):
    share = Path(get_package_share_directory('face_tracking_arm'))
    config['camera_mount']['publish_optical_tf'] = False
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    model, limits = robot_parameters(config, 'mono_cpu', share, path, True)
    root = ET.fromstring(model[0]['robot_description'])
    assert root.find("link[@name='tracking_gaze_optical']") is not None
    assert root.find("link[@name='face_camera_optical_frame']") is None
    _, motion, _, _ = control_parameters(config, share, limits)
    assert motion['gaze_frame'] == 'tracking_gaze_optical'
