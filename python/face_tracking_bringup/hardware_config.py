# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate hardware settings before starting any driver or motion process."""

import copy
import ipaddress
import math
from pathlib import Path

import yaml


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


def validate_structure(config, mode):
    """Validate containers and required keys before reading any nested setting."""
    sections = {
        'robot': 'ip report_type base_xyz_m base_rpy_deg',
        'table': 'shape dimensions_m center_m yaw_deg',
        'monitor': 'size_m mass_kg mount_xyz_m mount_rpy_deg',
        'camera_mount': 'xyz_m rpy_deg size_m mass_kg body_frame publish_optical_tf',
        'perception': 'python_executable model_dir detector threads confidence face_width_m '
                      'max_processing_fps preview_fps max_frame_age_sec '
                      'switch_margin_m switch_delay_sec',
        'tracking': 'idle_behavior minimum_face_distance_m safe_reach_radius_m '
                    'return_to_rest_delay_sec',
        'motion': 'speed_scale search_speed_rad_s search_sweep_half_range_rad '
                  'search_local_half_range_rad search_local_duration_sec '
                  'rest_joints_deg search_joints_deg',
        'mono_cpu': '', 'stereo': '',
    }
    mapping(config, 'hardware', sections)
    unknown = set(config) - sections.keys()
    if unknown:
        raise ValueError(f'Unknown YAML sections: {sorted(map(str, unknown))}')
    for section, keys in sections.items():
        mapping(config[section], section, keys.split())
    settings = mapping(config[mode], mode, ('source', 'rectification', 'left'))
    sides = ('left', 'right') if mode == 'stereo' else ('left',)
    if mode == 'stereo':
        mapping(settings, mode, ('right', 'baseline_m', 'sync_slop_sec'))
    for side in sides:
        name = f'{mode}.{side}'
        camera = mapping(settings[side], name, (
            'frame_id', 'image_topic', 'info_topic', 'optical_xyz_m', 'optical_rpy_deg'))
        if settings['source'] == 'usb':
            mapping(camera, name, ('device', 'width', 'height', 'fps', 'pixel_format',
                                   'camera_name', 'calibration_file'))
            for key in ('device', 'pixel_format', 'camera_name', 'calibration_file'):
                text(camera[key], name + '.' + key, allow_empty=key == 'calibration_file')
            mapping(camera.get('driver_parameters', {}), name + '.driver_parameters')


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not low <= value <= high:
        raise ValueError(f'{name}: expected a number in {low}..{high}')
    return float(value)


def vector(value, name, length=3, low=-100, high=100):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{name}: expected {length} numbers')
    return [number(x, name, low, high) for x in value]


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
    directory = Path.cwd() if directory is None else Path(directory)
    robot, table, monitor, mount = (c[k] for k in ('robot', 'table', 'monitor', 'camera_mount'))
    if not camera_only and not driver_only and not mock:
        try:
            address = ipaddress.ip_address(text(robot['ip'], 'robot.ip'))
            if address.is_unspecified or address.is_loopback or address.is_multicast:
                raise ValueError('Not a robot address')
        except ValueError as error:
            raise ValueError(
                'Fill robot.ip with the Lite 6 address from UFACTORY Studio') from error
    if robot['report_type'] not in ('normal', 'rich', 'dev'):
        raise ValueError('robot.report_type must be normal, rich or dev')
    for owner, key in ((robot, 'base_xyz_m'), (table, 'center_m'),
                       (monitor, 'mount_xyz_m'), (mount, 'xyz_m')):
        owner[key] = vector(owner[key], key)
    for owner, key in ((robot, 'base_rpy_deg'), (monitor, 'mount_rpy_deg'), (mount, 'rpy_deg')):
        owner[key] = vector(owner[key], key, low=-360, high=360)
    if table['shape'] not in ('box', 'cylinder'):
        raise ValueError('table.shape must be box or cylinder')
    table['dimensions_m'] = vector(table['dimensions_m'], 'table.dimensions_m',
                                   3 if table['shape'] == 'box' else 2, .001, 10)
    table['yaw_deg'] = number(table['yaw_deg'], 'table.yaw_deg', -360, 360)
    for name, item in (('monitor', monitor), ('camera_mount', mount)):
        item['size_m'] = vector(item['size_m'], name + '.size_m', low=.001, high=2)
        item['mass_kg'] = number(item['mass_kg'], name + '.mass_kg', .001, 1)
    if monitor['mass_kg'] + mount['mass_kg'] > 1.0:
        raise ValueError('Lite 6 payload exceeds 1 kg, including the mounting hardware')
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
    for key, low, high in (('threads', 1, 8), ('confidence', .1, .99),
                           ('face_width_m', .08, .30), ('max_processing_fps', 1, 60),
                           ('preview_fps', 1, 30),
                           ('max_frame_age_sec', .02, .5), ('switch_margin_m', 0, 2),
                           ('switch_delay_sec', 0, 2)):
        normalized = number(p[key], 'perception.' + key, low, high)
        if key != 'threads':
            p[key] = normalized
    if not isinstance(p['threads'], int):
        raise ValueError('perception.threads must be an integer')
    settings = c[mode]
    if settings['source'] not in ('usb', 'ros'):
        raise ValueError(f'{mode}.source must be usb or ros')
    if settings['rectification'] not in ('raw', 'rectified'):
        raise ValueError(f'{mode}.rectification must be raw or rectified')
    cameras = [settings['left']] + ([settings['right']] if mode == 'stereo' else [])
    frames = [text(mount['body_frame'], 'camera_mount.body_frame')]
    for camera in cameras:
        frame = camera['frame_id']
        if not isinstance(frame, str) or not frame or frame.startswith('/') or ' ' in frame:
            raise ValueError('Camera frame_id must be a nonempty TF name without a leading /')
        frames.append(frame)
        camera['optical_xyz_m'] = vector(camera['optical_xyz_m'], 'optical_xyz_m')
        camera['optical_rpy_deg'] = vector(
            camera['optical_rpy_deg'], 'optical_rpy_deg', low=-360, high=360)
        for key in ('image_topic', 'info_topic'):
            if not text(camera[key], key).startswith('/'):
                raise ValueError(f'{key} must be an absolute ROS topic')
        if settings['source'] == 'usb':
            for key in ('width', 'height'):
                number(camera[key], key, 64, 4096)
                if not isinstance(camera[key], int):
                    raise ValueError(f'{key} must be an integer')
            camera['fps'] = number(camera['fps'], 'fps', 1, 120)
            if camera['calibration_file']:
                camera['calibration_file'] = expanded_path(camera['calibration_file'], directory)
            if not driver_only:
                check_calibration(camera)
    if any(f.startswith('/') or any(char.isspace() for char in f) for f in frames):
        raise ValueError('Camera frame names must not have whitespace or a leading /')
    if len(set(frames)) != len(frames):
        raise ValueError('Camera body and optical frame names must be different')
    if mode == 'stereo':
        settings['baseline_m'] = number(settings['baseline_m'], 'stereo.baseline_m', 0, 1)
        settings['sync_slop_sec'] = number(
            settings['sync_slop_sec'], 'stereo.sync_slop_sec', 0, .02)
        for key in ('image_topic', 'info_topic'):
            if cameras[0][key] == cameras[1][key]:
                raise ValueError(f'Stereo {key} must refer to different cameras')
        if settings['source'] == 'usb' and cameras[0]['device'] == cameras[1]['device']:
            raise ValueError('Stereo USB devices must be different')
    t, m = c['tracking'], c['motion']
    if t['idle_behavior'] not in ('rest', 'search_sweep', 'search_local_then_sweep'):
        raise ValueError('Unknown tracking.idle_behavior')
    for key, low, high in (('minimum_face_distance_m', .4, 2),
                           ('safe_reach_radius_m', .1, .42),
                           ('return_to_rest_delay_sec', .5, 30)):
        t[key] = number(t[key], key, low, high)
    m['speed_scale'] = number(m['speed_scale'], 'speed_scale', .1, 1)
    for key, low, high in (('search_speed_rad_s', .01, .5),
                           ('search_sweep_half_range_rad', .05, math.pi),
                           ('search_local_half_range_rad', .01, 1),
                           ('search_local_duration_sec', 0, 30)):
        m[key] = number(m[key], key, low, high)
    for key in ('rest_joints_deg', 'search_joints_deg'):
        m[key] = vector(m[key], key, 6, -360, 360)
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
        vector(matrix['data'], f'{path}: {key}.data', length, -1e6, 1e6)
    if any(data[key]['data'][index] <= 0 for key, index in (
            ('camera_matrix', 0), ('camera_matrix', 4),
            ('projection_matrix', 0), ('projection_matrix', 5))):
        raise ValueError(f'{path}: calibration focal length must be positive')
