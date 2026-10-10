# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate hardware settings before starting any driver or motion process."""

import copy
import ipaddress
import math
from pathlib import Path

import yaml

from .camera_controls import validate_control_values, validate_driver_parameters
from .hardware_schema import (
    CAMERA_FIELDS, COMPACT_MANEUVER_FIELDS, CONTROLLER_FIELDS, KALMAN_FIELDS,
    MODE_OPTIONAL, MOUNT_SETUP_FIELDS, ONE_EURO_FIELDS, PERCEPTION_OPTIONAL,
    POINT_FILTER_FIELDS, ROOT_FIELDS, SECTIONS, TRACKING_OPTIONAL)
from .mount_geometry import migrate_mount_config
from .robot_profile import get_robot_profile, validate_joint_positions


MODES = ('mono_cpu', 'stereo', 'rgbd')
MAX_SEARCH_SPEED_RAD_S = 0.50
MIN_COLLISION_CONSTRAINT_ROWS = 24
MAX_COLLISION_CONSTRAINT_ROWS = 64
MAX_SOLVER_TIME_SEC = 0.003


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


def validate_structure(config, mode, *, camera_only=False, driver_only=False, setup_only=False):
    """Check names everywhere, requiring values only for active components."""
    mapping(config, 'hardware')
    known_keys(config, 'hardware', set(SECTIONS) | {'point_filter'} | ROOT_FIELDS)
    if 'mount_setup' in config:
        known_keys(config['mount_setup'], 'mount_setup', MOUNT_SETUP_FIELDS)
    if 'compact_maneuver' in config:
        known_keys(config['compact_maneuver'], 'compact_maneuver',
                   COMPACT_MANEUVER_FIELDS)
    if driver_only:
        active = {mode}
    elif camera_only:
        active = {mode, 'perception'}
    else:
        active = set(SECTIONS) - set(MODES) - ({'perception'} if setup_only else set()) | {mode}
    for section, fields in SECTIONS.items():
        required = fields.split()
        allowed = set(required)
        if section == 'motion':
            tracking = config.get('tracking', {})
            if isinstance(tracking, dict) and tracking.get('idle_behavior') == 'rest':
                required = [key for key in required if not key.startswith('search_')]
        if section == 'tracking':
            allowed |= TRACKING_OPTIONAL
        elif section == 'perception':
            allowed |= PERCEPTION_OPTIONAL
        elif section == 'controller':
            allowed |= CONTROLLER_FIELDS
        allowed |= MODE_OPTIONAL.get(section, set())
        if section not in config:
            if section in active:
                raise ValueError(f'hardware.{section}: required setting is missing')
            continue
        mapping(config[section], section, required if section in active else ())
        known_keys(config[section], section, allowed)
    for camera_mode in MODES:
        if camera_mode not in config:
            continue
        settings = config[camera_mode]
        selected = camera_mode == mode
        for side in ('left', 'right') if camera_mode == 'stereo' else ('left',):
            name = f'{camera_mode}.{side}'
            if side not in settings:
                if selected:
                    raise ValueError(f'{name}: required camera configuration is missing')
                continue
            camera = mapping(settings[side], name, (
                'frame_id', 'image_topic', 'info_topic', 'optical_xyz_m', 'optical_rpy_deg'
            ) if selected else ())
            known_keys(camera, name, CAMERA_FIELDS)
            if selected and settings.get('source') == 'usb':
                mapping(camera, name, ('device', 'width', 'height', 'fps', 'pixel_format',
                                       'camera_name', 'calibration_file'))
                for key in ('device', 'pixel_format', 'camera_name', 'calibration_file'):
                    text(camera[key], name + '.' + key, allow_empty=key == 'calibration_file')
            if selected:
                validate_driver_parameters(camera.get('driver_parameters', {}),
                                           name + '.driver_parameters')
                validate_control_values(camera.get('control_values', {}),
                                        name + '.control_values')


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
    _validate_point_filter_structure(config)
    settings = config.get('point_filter', {})
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
            settings.get('reset_after_sec', .25), 'point_filter.reset_after_sec')
        if enabled else .25,
    }
    if enabled and method == 'kalman':
        defaults = (.03, 1.0) if mode == 'mono_cpu' else (.02, 1.5)
        values = settings.get(mode, {})
        for key, default in zip(('measurement_std_m', 'acceleration_std_mps2'), defaults):
            params['point_kalman_' + key] = positive(
                values.get(key, default), 'point_filter.' + mode + '.' + key)
    values = settings.get('one_euro', {})
    if enabled and method == 'one_euro':
        for key, default in (('min_cutoff_hz', 1.5), ('beta', 32.0),
                             ('derivative_cutoff_hz', 1.0)):
            name = 'point_filter.one_euro.' + key
            value = values.get(key, default)
            params['point_smoothing_' + key] = (
                number(value, name, 0) if key == 'beta' else positive(value, name))
    else:
        for key, default in (('min_cutoff_hz', 1.5), ('beta', 32.0),
                             ('derivative_cutoff_hz', 1.0)):
            params['point_smoothing_' + key] = default
    return params


def _validate_point_filter_structure(config):
    """Keep misspelled filter keys visible even when perception is not started."""
    settings = config.get('point_filter', {})
    known_keys(settings, 'point_filter', POINT_FILTER_FIELDS)
    for camera_mode in MODES:
        known_keys(settings.get(camera_mode, {}),
                   'point_filter.' + camera_mode, KALMAN_FIELDS)
    known_keys(settings.get('one_euro', {}), 'point_filter.one_euro', ONE_EURO_FIELDS)


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
                    setup_only=False, directory=None):
    """Return a validated copy; require calibration only for actual perception."""
    c = copy.deepcopy(config)
    if mode not in MODES or not isinstance(c, dict):
        raise ValueError('Expected a hardware YAML and mono_cpu or stereo')
    c = migrate_mount_config(c)
    validate_structure(c, mode, camera_only=camera_only, driver_only=driver_only,
                       setup_only=setup_only)
    if not driver_only:
        point_filter_parameters(c, mode)
    else:
        _validate_point_filter_structure(c)
    directory = Path.cwd() if directory is None else Path(directory)
    full_robot = not camera_only and not driver_only
    if full_robot:
        _validate_robot_settings(c, mock=mock)
        if 'mount_setup' in c:
            settings = c['mount_setup']
            if 'preferred_top_flange' in settings:
                direction = vector(settings['preferred_top_flange'],
                                   'mount_setup.preferred_top_flange')
                norm = math.sqrt(sum(value * value for value in direction))
                if not .999 <= norm <= 1.001 or abs(direction[2]) > .001:
                    raise ValueError('mount_setup.preferred_top_flange must be a unit vector '
                                     'in the flange XY plane')
            if 'remembered_joint_positions_deg' in settings:
                settings['remembered_joint_positions_deg'] = vector(
                    settings['remembered_joint_positions_deg'],
                    'mount_setup.remembered_joint_positions_deg', 6)
    if not driver_only and not setup_only:
        _validate_perception_settings(c, directory)
    settings = c[mode]
    allowed_sources = (('usb', 'ros') if mode in ('mono_cpu', 'stereo')
                       else ('ros', 'realsense'))
    if settings['source'] not in allowed_sources:
        raise ValueError(f'{mode}.source must be one of {allowed_sources}')
    if settings['rectification'] not in ('raw', 'rectified'):
        raise ValueError(f'{mode}.rectification must be raw or rectified')
    settings['sync_slop_sec'] = number(
        settings.get('sync_slop_sec', 0), mode + '.sync_slop_sec', 0)
    cameras = [settings['left']] + ([settings['right']] if mode == 'stereo' else [])
    frames = ([text(c['camera_mount']['body_frame'], 'camera_mount.body_frame')]
              if full_robot else [])
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
        if mode == 'rgbd' or settings['source'] == 'usb':
            for key in ('width', 'height'):
                positive(camera[key], key)
                if type(camera[key]) is not int:
                    raise ValueError(f'{key} must be an integer')
            camera['fps'] = positive(camera['fps'], 'fps')
            if mode != 'rgbd' and camera.get('calibration_file'):
                camera['calibration_file'] = expanded_path(camera['calibration_file'], directory)
            if mode != 'rgbd' and not driver_only and not setup_only:
                check_calibration(camera)
        if mode == 'rgbd':
            if not text(camera.get('depth_topic'), 'rgbd.left.depth_topic').startswith('/'):
                raise ValueError('rgbd.left.depth_topic must be an absolute ROS topic')
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
    if mode == 'rgbd':
        for key in ('serial_no', 'camera_namespace', 'camera_name',
                    'color_profile', 'depth_profile'):
            text(settings[key], 'rgbd.' + key,
                 allow_empty=key in ('serial_no', 'camera_namespace'))
        if (settings['source'] == 'realsense' and
                settings['left']['frame_id'] != settings['camera_name'] +
                '_color_optical_frame'):
            raise ValueError('For rgbd.source=realsense, left.frame_id must match '
                             'camera_name + _color_optical_frame; use camera_name to set '
                             'the frame prefix')
    return c


def _validate_perception_settings(config, directory):
    p = config['perception']
    p.setdefault('timing_mode', 'auto')
    p.setdefault('source_timeout_sec', .50)
    p.setdefault('mono_partial_depth_max_age_sec', .30)
    rgbd = config.get('rgbd', {})
    p.setdefault('rgbd_depth_topic', rgbd.get('left', {}).get(
        'depth_topic', '/camera/camera/aligned_depth_to_color/image_raw'))
    p.setdefault('rgbd_sync_slop_sec', rgbd.get('sync_slop_sec', .02))
    p.setdefault('rgbd_16u_scale_m', .001)
    p.setdefault('rgbd_32f_scale_m', 1.0)
    if p['timing_mode'] not in ('auto', 'manual'):
        raise ValueError('perception.timing_mode must be auto or manual')
    p['source_timeout_sec'] = positive(p['source_timeout_sec'],
                                       'perception.source_timeout_sec')
    p['mono_partial_depth_max_age_sec'] = positive(
        p['mono_partial_depth_max_age_sec'], 'perception.mono_partial_depth_max_age_sec')
    p['rgbd_depth_topic'] = text(p['rgbd_depth_topic'], 'perception.rgbd_depth_topic')
    if not p['rgbd_depth_topic'].startswith('/'):
        raise ValueError('perception.rgbd_depth_topic must be an absolute ROS topic')
    p['rgbd_sync_slop_sec'] = number(p['rgbd_sync_slop_sec'],
                                     'perception.rgbd_sync_slop_sec', 0)
    p['rgbd_16u_scale_m'] = positive(p['rgbd_16u_scale_m'], 'perception.rgbd_16u_scale_m')
    p['rgbd_32f_scale_m'] = positive(p['rgbd_32f_scale_m'], 'perception.rgbd_32f_scale_m')
    p['python_executable'] = expanded_path(
        p['python_executable'], directory, 'perception.python_executable')
    p['model_dir'] = expanded_path(p['model_dir'], directory, 'perception.model_dir')
    if not Path(p['python_executable']).is_file():
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


def _validate_robot_settings(config, *, mock):
    robot, table, monitor, mount = (config[k] for k in (
        'robot', 'table', 'monitor', 'camera_mount'))
    if not mock:
        try:
            address = ipaddress.ip_address(text(robot['ip'], 'robot.ip'))
            if address.is_unspecified or address.is_loopback or address.is_multicast:
                raise ValueError('Not a robot address')
        except ValueError as error:
            raise ValueError(
                'Fill robot.ip with the robot address from UFACTORY Studio') from error
    profile = get_robot_profile(robot['model'], robot['model_num'])
    compact_settings = config.get('compact_maneuver', {})
    if 'speed_scale' in compact_settings:
        compact_settings['speed_scale'] = positive(
            compact_settings['speed_scale'], 'compact_maneuver.speed_scale')
    if 'goal_joints_rad' in compact_settings:
        goal = vector(compact_settings['goal_joints_rad'],
                      'compact_maneuver.goal_joints_rad', 6)
        validate_joint_positions(goal, profile['position_limits_rad'],
                                 'compact_maneuver.goal_joints_rad')
        compact_settings['goal_joints_rad'] = goal
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
    t, m = config['tracking'], config['motion']
    if t['idle_behavior'] not in ('rest', 'search_sweep', 'search_local_then_sweep'):
        raise ValueError('Unknown tracking.idle_behavior')
    for key in ('minimum_face_distance_m', 'safe_reach_radius_m', 'return_to_rest_delay_sec'):
        t[key] = positive(t[key], 'tracking.' + key)
    t['face_target_freshness_timeout_sec'] = positive(
        t.get('face_target_freshness_timeout_sec', .5),
        'tracking.face_target_freshness_timeout_sec')
    t['face_prediction_timeout_sec'] = number(
        t.get('face_prediction_timeout_sec', .8),
        'tracking.face_prediction_timeout_sec', 0)
    if (t['face_prediction_timeout_sec'] > 2.0 or
            0 < t['face_prediction_timeout_sec'] <= t['face_target_freshness_timeout_sec']):
        raise ValueError('tracking.face_prediction_timeout_sec must be zero or greater than '
                         'face_target_freshness_timeout_sec and no more than 2 seconds')
    if t['return_to_rest_delay_sec'] <= t['face_target_freshness_timeout_sec']:
        raise ValueError('tracking.return_to_rest_delay_sec must be greater than '
                         'tracking.face_target_freshness_timeout_sec')
    if 'rest_position_m' in t:
        t['rest_position_m'] = vector(t['rest_position_m'], 'tracking.rest_position_m')
    if not config['controller']['monitor_guard_joint_name']:
        config['controller'].pop('monitor_guard_min_position_rad', None)
        config['controller'].pop('monitor_guard_max_position_rad', None)
    m['speed_scale'] = positive(m['speed_scale'], 'motion.speed_scale')
    m['rest_joints_deg'] = vector(m['rest_joints_deg'], 'motion.rest_joints_deg', 6)
    if t['idle_behavior'] == 'rest':
        m.pop('search_joints_deg', None)
        for key in ('search_speed_rad_s', 'search_sweep_half_range_rad',
                    'search_local_half_range_rad', 'search_local_duration_sec'):
            m.pop(key, None)
    else:
        for key in ('search_speed_rad_s', 'search_sweep_half_range_rad',
                    'search_local_half_range_rad'):
            m[key] = positive(m[key], 'motion.' + key)
        if m['search_speed_rad_s'] > MAX_SEARCH_SPEED_RAD_S:
            raise ValueError('motion.search_speed_rad_s: expected a value in (0, 0.5] rad/s')
        m['search_local_duration_sec'] = number(
            m['search_local_duration_sec'], 'motion.search_local_duration_sec', 0)
        m['search_joints_deg'] = vector(m['search_joints_deg'], 'motion.search_joints_deg', 6)
    for key in config['limits']:
        config['limits'][key] = positive_vector(config['limits'][key], 'limits.' + key, 6)
    for key, value in config['controller'].items():
        if key == 'monitor_near_distance_lipschitz_m_per_rad':
            continue  # Legacy, derived value: accepted in YAML but not consumed or forwarded.
        if key == 'monitor_guard_joint_name':
            text(value, 'controller.' + key, allow_empty=True)
        elif key == 'maximum_collision_constraints':
            if (type(value) is not int or not MIN_COLLISION_CONSTRAINT_ROWS <= value <=
                    MAX_COLLISION_CONSTRAINT_ROWS):
                raise ValueError(f'controller.{key}: expected an integer in '
                                 f'{MIN_COLLISION_CONSTRAINT_ROWS}..'
                                 f'{MAX_COLLISION_CONSTRAINT_ROWS}')
        elif key in ('solver_max_iterations', 'segment_validation_substeps'):
            if type(value) is not int or value <= 0:
                raise ValueError(f'controller.{key}: expected a positive integer')
        elif key == 'position_gain':
            config['controller'][key] = positive(value, 'controller.' + key)
        elif key == 'solver_time_limit_sec':
            config['controller'][key] = positive(value, 'controller.' + key)
            if config['controller'][key] > MAX_SOLVER_TIME_SEC:
                raise ValueError(f'controller.{key}: expected a value in (0, '
                                 f'{MAX_SOLVER_TIME_SEC}] seconds')
        else:
            config['controller'][key] = number(value, 'controller.' + key)
            if key not in ('monitor_guard_min_position_rad', 'monitor_guard_max_position_rad',
                           'monitor_near_distance_lipschitz_m_per_rad'):
                if value < 0:
                    raise ValueError(f'controller.{key}: expected a nonnegative number')


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
