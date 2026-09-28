# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate hardware settings before starting any driver or motion process."""

import copy
import ipaddress
import math
from pathlib import Path
import warnings

import yaml

from .camera_controls import validate_control_values, validate_driver_parameters
from .hardware_schema import (
    CAMERA_FIELDS, CONTROLLER_FIELDS, KALMAN_FIELDS, MODE_OPTIONAL, ONE_EURO_FIELDS,
    POINT_FILTER_FIELDS, SECTIONS, TRACKING_OPTIONAL)
from .robot_profile import get_robot_profile


MODES = ('mono_cpu', 'stereo')


def mapping(value, name, required=()):
    """Report incomplete YAML at its boundary, with the full configuration path."""
    if not isinstance(value, dict):
        raise ValueError(f'{name}: expected a YAML mapping')
    for key in required:
        if key not in value:
            raise ValueError(f'{name}.{key}: required setting is missing')
    return value


def text(value, name, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{name}: expected a {"possibly empty " if allow_empty else ""}string')
    return value


def known_keys(value, name, allowed):
    mapping(value, name)
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{name}: unknown settings {sorted(map(str, unknown))}')


def validate_structure(config, mode):
    """Reject typos at every level, including the inactive camera configuration."""
    mapping(config, 'hardware', SECTIONS)
    known_keys(config, 'hardware', set(SECTIONS) | {'point_filter'})
    for section, fields in SECTIONS.items():
        required = fields.split()
        allowed = set(required)
        if section == 'tracking':
            allowed |= TRACKING_OPTIONAL
        elif section == 'controller':
            allowed |= CONTROLLER_FIELDS
        allowed |= MODE_OPTIONAL.get(section, set())
        mapping(config[section], section, required)
        known_keys(config[section], section, allowed)
    for camera_mode in MODES:
        settings = config[camera_mode]
        for side in ('left', 'right') if camera_mode == 'stereo' else ('left',):
            name = f'{camera_mode}.{side}'
            camera = mapping(settings[side], name, (
                'frame_id', 'image_topic', 'info_topic', 'optical_xyz_m', 'optical_rpy_deg'))
            known_keys(camera, name, CAMERA_FIELDS)
            if settings['source'] == 'usb':
                mapping(camera, name, ('device', 'width', 'height', 'fps', 'pixel_format',
                                       'camera_name', 'calibration_file'))
                for key in ('device', 'pixel_format', 'camera_name', 'calibration_file'):
                    text(camera[key], name + '.' + key, allow_empty=key == 'calibration_file')
            validate_driver_parameters(camera.get('driver_parameters', {}),
                                       name + '.driver_parameters')
            validate_control_values(camera.get('control_values', {}), name + '.control_values')


def number(value, name, low=-math.inf, high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{name}: expected a finite number in {low}..{high}')
    return float(value)


def positive(value, name):
    value = number(value, name, 0)
    if value == 0:
        raise ValueError(f'{name}: expected a positive number')
    return value


def point_filter_parameters(config, mode):
    """Validate the optional filter section and translate it to perception parameters."""
    if mode not in MODES:
        raise ValueError('Unknown camera mode: ' + str(mode))
    settings = config.get('point_filter', {})
    known_keys(settings, 'point_filter', POINT_FILTER_FIELDS)
    enabled = settings.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('point_filter.enabled: expected true or false')
    method = settings.get('method', 'kalman')
    if method not in ('kalman', 'one_euro'):
        raise ValueError('point_filter.method: expected kalman or one_euro')
    params = {
        'point_smoothing': enabled,
        'point_smoothing_method': method,
        'point_smoothing_reset_after_sec': positive(
            settings.get('reset_after_sec', .25), 'point_filter.reset_after_sec'),
    }
    for camera_mode, defaults in (('mono_cpu', (.03, 1.0)), ('stereo', (.02, 1.5))):
        name = 'point_filter.' + camera_mode
        values = settings.get(camera_mode, {})
        known_keys(values, name, KALMAN_FIELDS)
        for key, default in zip(('measurement_std_m', 'acceleration_std_mps2'), defaults):
            value = positive(values.get(key, default), name + '.' + key)
            if camera_mode == mode:
                params['point_kalman_' + key] = value
    values = settings.get('one_euro', {})
    known_keys(values, 'point_filter.one_euro', ONE_EURO_FIELDS)
    for key, default in (('min_cutoff_hz', 1.5), ('beta', 32.0),
                         ('derivative_cutoff_hz', 1.0)):
        name = 'point_filter.one_euro.' + key
        value = values.get(key, default)
        params['point_smoothing_' + key] = (
            number(value, name, 0) if key == 'beta' else positive(value, name))
    return params


def vector(value, name, length=3, low=-math.inf, high=math.inf):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{name}: expected {length} numbers')
    return [number(x, name, low, high) for x in value]


def positive_vector(value, name, length=3):
    result = vector(value, name, length)
    return [positive(x, name) for x in result]


def optional_geometry_size(value, name, length=3):
    """Accept a solid object or an explicit all-zero absence, never degenerate geometry."""
    result = vector(value, name, length, low=0)
    if any(x == 0 for x in result) and any(x != 0 for x in result):
        raise ValueError(f'{name}: use positive dimensions or all zeros to omit the object')
    return result


def expanded_path(value, directory, name='path'):
    path = Path(text(value, name)).expanduser()
    return str(path if path.is_absolute() else directory / path)


def validate_config(config, mode, *, camera_only=False, mock=False, driver_only=False,
                    directory=None):
    """Return a validated copy; require calibration only for actual perception."""
    c = copy.deepcopy(config)
    if mode not in MODES or not isinstance(c, dict):
        raise ValueError('Expected a hardware YAML and mono_cpu or stereo')
    validate_structure(c, mode)
    point_filter_parameters(c, mode)
    directory = Path.cwd() if directory is None else Path(directory)
    robot, table, monitor, mount = (c[k] for k in ('robot', 'table', 'monitor', 'camera_mount'))
    if not camera_only and not driver_only and not mock:
        try:
            address = ipaddress.ip_address(text(robot['ip'], 'robot.ip'))
            if address.is_unspecified or address.is_loopback or address.is_multicast:
                raise ValueError('Not a robot address')
        except ValueError as error:
            raise ValueError(
                'Fill robot.ip with the robot address from UFACTORY Studio') from error
    profile = get_robot_profile(robot['model'], robot['model_num'])
    if robot['report_type'] not in ('normal', 'rich', 'dev'):
        raise ValueError('robot.report_type must be normal, rich or dev')
    for owner, key in ((robot, 'base_xyz_m'), (table, 'center_m'),
                       (monitor, 'mount_xyz_m'), (mount, 'xyz_m')):
        owner[key] = vector(owner[key], key)
    for owner, key in ((robot, 'base_rpy_deg'), (monitor, 'mount_rpy_deg'), (mount, 'rpy_deg')):
        owner[key] = vector(owner[key], key)
    if table['shape'] not in ('box', 'cylinder'):
        raise ValueError('table.shape must be box or cylinder')
    table['dimensions_m'] = optional_geometry_size(
        table['dimensions_m'], 'table.dimensions_m', 3 if table['shape'] == 'box' else 2)
    table['yaw_deg'] = number(table['yaw_deg'], 'table.yaw_deg')
    monitor['size_m'] = optional_geometry_size(monitor['size_m'], 'monitor.size_m')
    monitor['mass_kg'] = (positive(monitor['mass_kg'], 'monitor.mass_kg')
                          if any(monitor['size_m']) else
                          number(monitor['mass_kg'], 'monitor.mass_kg', 0))
    mount['size_m'] = positive_vector(mount['size_m'], 'camera_mount.size_m')
    mount['mass_kg'] = positive(mount['mass_kg'], 'camera_mount.mass_kg')
    if monitor['mass_kg'] + mount['mass_kg'] > profile['payload_kg']:
        raise ValueError(f'{robot["model"]} payload exceeds {profile["payload_kg"]} kg, '
                         'including the mounting hardware')
    if not isinstance(mount['publish_optical_tf'], bool):
        raise ValueError('camera_mount.publish_optical_tf must be true or false')
    p = c['perception']
    p['python_executable'] = expanded_path(
        p['python_executable'], directory, 'perception.python_executable')
    p['model_dir'] = expanded_path(p['model_dir'], directory, 'perception.model_dir')
    if not driver_only and not Path(p['python_executable']).is_file():
        raise ValueError('perception.python_executable not found; prepare the venv first')
    if p['detector'] not in ('yunet', 'yolov5n_face', 'yolo_facev2n'):
        raise ValueError('Unknown perception.detector')
    p['confidence'] = number(p['confidence'], 'perception.confidence', 0, 1)
    if p['confidence'] in (0, 1):
        raise ValueError('perception.confidence must be strictly between 0 and 1')
    for key in ('face_width_m', 'preview_fps', 'max_frame_age_sec'):
        p[key] = positive(p[key], 'perception.' + key)
    for key in ('max_processing_fps', 'switch_margin_m', 'switch_delay_sec'):
        p[key] = number(p[key], 'perception.' + key, 0)
    if type(p['threads']) is not int or p['threads'] <= 0:
        raise ValueError('perception.threads must be a positive integer')
    settings = c[mode]
    if settings['source'] not in ('usb', 'ros'):
        raise ValueError(f'{mode}.source must be usb or ros')
    if settings['rectification'] not in ('raw', 'rectified'):
        raise ValueError(f'{mode}.rectification must be raw or rectified')
    settings['sync_slop_sec'] = number(
        settings.get('sync_slop_sec', 0), mode + '.sync_slop_sec', 0)
    cameras = [settings['left']] + ([settings['right']] if mode == 'stereo' else [])
    frames = [text(mount['body_frame'], 'camera_mount.body_frame')]
    for camera in cameras:
        frame = camera['frame_id']
        if not isinstance(frame, str) or not frame or frame.startswith('/') or ' ' in frame:
            raise ValueError('Camera frame_id must be a nonempty TF name without a leading /')
        frames.append(frame)
        camera['optical_xyz_m'] = vector(camera['optical_xyz_m'], 'optical_xyz_m')
        camera['optical_rpy_deg'] = vector(
            camera['optical_rpy_deg'], 'optical_rpy_deg')
        for key in ('image_topic', 'info_topic'):
            if not text(camera[key], key).startswith('/'):
                raise ValueError(f'{key} must be an absolute ROS topic')
        if settings['source'] == 'usb':
            for key in ('width', 'height'):
                positive(camera[key], key)
                if type(camera[key]) is not int:
                    raise ValueError(f'{key} must be an integer')
            camera['fps'] = positive(camera['fps'], 'fps')
            if camera['calibration_file']:
                camera['calibration_file'] = expanded_path(camera['calibration_file'], directory)
            if not driver_only:
                check_calibration(camera)
    if any(f.startswith('/') or any(char.isspace() for char in f) for f in frames):
        raise ValueError('Camera frame names must not have whitespace or a leading /')
    if len(set(frames)) != len(frames):
        raise ValueError('Camera body and optical frame names must be different')
    if 'tracking_gaze_optical' in frames:
        raise ValueError('tracking_gaze_optical is reserved for the internal camera model')
    if mode == 'stereo':
        settings['baseline_m'] = number(settings['baseline_m'], 'stereo.baseline_m', 0)
        for key, default, divisor, remainder in (
                ('num_disparities', 128, 16, 0), ('block_size', 5, 2, 1)):
            value = settings.get(key, default)
            if type(value) is not int or value <= 0 or value % divisor != remainder:
                requirement = 'a positive multiple of 16' if key == 'num_disparities' else (
                    'a positive odd integer')
                raise ValueError(f'stereo.{key}: expected {requirement}')
            settings[key] = value
        if settings['source'] == 'usb':
            for camera in cameras:
                if (camera['width'] <= settings['num_disparities'] + settings['block_size'] // 2
                        or min(camera['width'], camera['height']) < settings['block_size']):
                    raise ValueError(
                        'stereo: num_disparities/block_size exceed capture dimensions')
        for key in ('image_topic', 'info_topic'):
            if cameras[0][key] == cameras[1][key]:
                raise ValueError(f'Stereo {key} must refer to different cameras')
        if settings['source'] == 'usb' and cameras[0]['device'] == cameras[1]['device']:
            raise ValueError('Stereo USB devices must be different')
    t, m = c['tracking'], c['motion']
    if t['idle_behavior'] not in ('rest', 'search_sweep', 'search_local_then_sweep'):
        raise ValueError('Unknown tracking.idle_behavior')
    for key in ('minimum_face_distance_m', 'safe_reach_radius_m', 'return_to_rest_delay_sec'):
        t[key] = positive(t[key], 'tracking.' + key)
    t['face_target_freshness_timeout_sec'] = positive(
        t.get('face_target_freshness_timeout_sec', .5),
        'tracking.face_target_freshness_timeout_sec')
    if t['return_to_rest_delay_sec'] <= t['face_target_freshness_timeout_sec']:
        raise ValueError('tracking.return_to_rest_delay_sec must be greater than '
                         'tracking.face_target_freshness_timeout_sec')
    if 'rest_position_m' in t:
        t['rest_position_m'] = vector(t['rest_position_m'], 'tracking.rest_position_m')
    m['speed_scale'] = positive(m['speed_scale'], 'motion.speed_scale')
    for key in ('search_speed_rad_s', 'search_sweep_half_range_rad',
                'search_local_half_range_rad'):
        m[key] = positive(m[key], 'motion.' + key)
    m['search_local_duration_sec'] = number(
        m['search_local_duration_sec'], 'motion.search_local_duration_sec', 0)
    for key in ('rest_joints_deg', 'search_joints_deg'):
        m[key] = vector(m[key], 'motion.' + key, 6)
    for key in c['limits']:
        c['limits'][key] = positive_vector(c['limits'][key], 'limits.' + key, 6)
    for key, value in c['controller'].items():
        if key == 'monitor_guard_joint_name':
            text(value, 'controller.' + key, allow_empty=True)
        elif key in ('maximum_collision_constraints', 'solver_max_iterations',
                     'segment_validation_substeps'):
            if type(value) is not int or value <= 0:
                raise ValueError(f'controller.{key}: expected a positive integer')
        else:
            c['controller'][key] = number(value, 'controller.' + key)
            if key not in ('monitor_guard_min_position_rad', 'monitor_guard_max_position_rad'):
                if value < 0:
                    raise ValueError(f'controller.{key}: expected a nonnegative number')
    if 'monitor_near_distance_lipschitz_m_per_rad' in c['controller']:
        warnings.warn(
            'controller.monitor_near_distance_lipschitz_m_per_rad is derived from robot geometry; '
            'this legacy setting does not change the result', UserWarning, stacklevel=2)
    return c


def check_calibration(camera):
    """Catch a missing or wrong-resolution USB calibration before opening the arm."""
    path = Path(camera['calibration_file'])
    if not camera['calibration_file'] or not path.is_file():
        raise ValueError('Fill calibration_file with the camera YAML; see docs/hardware.md')
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f'{path}: invalid calibration YAML') from error
    mapping(data, str(path), ('image_width', 'image_height', 'camera_matrix',
                              'rectification_matrix', 'projection_matrix'))
    if (data['image_width'], data['image_height']) != (camera['width'], camera['height']):
        raise ValueError(f'{path}: calibration resolution differs from width/height')
    for key, length in (('camera_matrix', 9), ('rectification_matrix', 9),
                        ('projection_matrix', 12)):
        matrix = mapping(data[key], f'{path}: {key}', ('data',))
        vector(matrix['data'], f'{path}: {key}.data', length)
    if any(data[key]['data'][index] <= 0 for key, index in (
            ('camera_matrix', 0), ('camera_matrix', 4),
            ('projection_matrix', 0), ('projection_matrix', 5))):
        raise ValueError(f'{path}: calibration focal length must be positive')
