# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Apply explicit hardware controls through the separately licensed cameractrls CLI."""

import os
from pathlib import Path
import re
import subprocess


CLI = os.environ.get('FACE_TRACKING_CAMERACTRLS', '/opt/cameractrls/cameractrls.py')
# These names are interpreted by usb_cam, not arbitrary V4L2 controls.
DRIVER_TYPES = {
    'brightness': int, 'contrast': int, 'saturation': int, 'sharpness': int,
    'gain': int, 'auto_white_balance': bool, 'white_balance': int,
    'autoexposure': bool, 'exposure': int, 'autofocus': bool, 'focus': int,
    'av_device_format': str, 'io_method': str,
}


def validate_driver_parameters(parameters, name):
    if not isinstance(parameters, dict):
        raise ValueError(f'{name}: expected a mapping')
    for key, value in parameters.items():
        expected = DRIVER_TYPES.get(key)
        if expected is None:
            raise ValueError(f'{name}.{key}: unsupported usb_cam parameter')
        if type(value) is not expected:
            raise ValueError(f'{name}.{key}: expected {expected.__name__}')
        if key == 'io_method' and value not in ('mmap', 'read', 'userptr'):
            raise ValueError(f'{name}.io_method: expected mmap, read or userptr')


def validate_control_values(values, name):
    if not isinstance(values, dict):
        raise ValueError(f'{name}: expected a mapping')
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', key):
            raise ValueError(f'{name}: invalid hardware control name')
        # Do not expose cameractrls software actions, format changes or persistence.
        if key.startswith(('systemd_', 'desktop_', 'preset', 'kiyo_pro_save')) or key in (
                'pixelformat', 'resolution', 'fps', 'color_preset'):
            raise ValueError(f'{name}.{key}: use the explicit capture settings instead')
        if type(value) not in (str, int, bool) or not re.fullmatch(
                r'[A-Za-z0-9_.+-]+', str(value)):
            raise ValueError(f'{name}.{key}: expected one integer, boolean or menu value')


def parse_controls(output):
    """Parse the pinned CLI's list output; buttons and informational fields are excluded."""
    controls = {}
    for line in output.splitlines():
        match = re.match(r'^\s+([a-z][a-z0-9_]*) = (\S+)\s+\(\s*(.*?)\)', line)
        if not match:
            continue
        name, value, details = match.groups()
        control = {'value': value, 'readonly': '| readonly' in line,
                   'inactive': '| inactive' in line}
        menu = re.search(r'values:\s*(.*)', details)
        bounds = re.search(r'min:\s*(-?\d+)\s+max:\s*(-?\d+)', details)
        if menu:
            control['values'] = [v.strip() for v in menu.group(1).split(',')]
        elif bounds:
            control['minimum'], control['maximum'] = map(int, bounds.groups())
            if value in ('True', 'False'):
                control['value'] = str(int(value == 'True'))
            step = re.search(r'step:\s*(\d+)', details)
            control['step'] = max(1, int(step.group(1))) if step else 1
        else:
            continue
        controls[name] = control
    return controls


def requested_value(name, value, control):
    if control['readonly'] or control['inactive']:
        raise ValueError(f'{name}: camera control is read-only or inactive')
    if 'values' in control:
        expected = str(value)
        if expected not in control['values']:
            raise ValueError(f'{name}: supported values are {control["values"]}')
        return expected
    if value == 'minimum':
        value = control['minimum']
    elif type(value) is bool:
        value = int(value)
    elif type(value) is not int:
        raise ValueError(f'{name}: expected an integer or minimum')
    if not control['minimum'] <= value <= control['maximum'] or (
            value - control['minimum']) % control['step']:
        raise ValueError(f'{name}: value does not match the camera range/step')
    return str(value)


def configure_controls(camera, *, command_runner=subprocess.run, cli=CLI):
    """Set controls and independently read them back; failure prevents robot launch."""
    desired = camera.get('control_values', {})
    if not desired:
        return {}
    validate_control_values(desired, 'control_values')
    if not Path(cli).is_file():
        raise ValueError('Camera controls need cameractrls; use the prepared Docker image')
    device = str(Path(camera['device']).resolve())
    # Keep the real videoN name inside Docker: the external tool resolves USB ID
    # and extension-unit descriptors through the matching read-only sysfs entry.
    command = ['/usr/bin/python3', cli, '-d', device]
    environment = dict(os.environ, LANG='C', LC_ALL='C')

    def run(arguments):
        try:
            result = command_runner(command + arguments, capture_output=True, text=True,
                                    env=environment, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f'{device}: camera control tool failed: {error}') from error
        if result.returncode:
            raise ValueError(f'{device}: camera control tool failed: {result.stderr.strip()}')
        return result.stdout

    before = parse_controls(run(['--list']))
    expected = {}
    for name, value in desired.items():
        if name not in before:
            raise ValueError(f'{device}: camera/USB ID does not support control {name}')
        expected[name] = requested_value(name, value, before[name])
    run(['-c', ','.join(f'{name}={value}' for name, value in expected.items())])
    actual = parse_controls(run(['--list']))
    for name, value in expected.items():
        received = actual.get(name, {}).get('value')
        if received != value:
            raise ValueError(f'{device}: {name} requested {value}, read back {received}')
    return expected
