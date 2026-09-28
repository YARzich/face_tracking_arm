# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Camera capability negotiation and independent readback, with no USB device."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.camera_controls import (  # noqa: E402, I100
    configure_controls, parse_controls, validate_control_values, validate_driver_parameters)


LISTING = """Basic / Crop
 logitech_brio_fov = 78\t( values: 65, 78, 90 )
 zoom_absolute = 125\t( default: 100 min: 100 max: 500 step: 5 )
Basic / Focus
 focus_automatic_continuous = False\t( default: True min: 0 max: 1 )
 focus_absolute = 30\t( default: 0 min: 0 max: 255 step: 5 ) | inactive
 identity = camera
 preset\t\t( buttons: save, restore )
"""


def test_only_writable_hardware_value_rows_are_parsed():
    result = parse_controls(LISTING)
    assert result['logitech_brio_fov']['values'] == ['65', '78', '90']
    assert result['zoom_absolute']['minimum'] == 100
    assert result['focus_automatic_continuous']['value'] == '0'
    assert result['focus_absolute']['inactive']
    assert 'identity' not in result and 'preset' not in result


@pytest.fixture
def camera():
    return {'device': '/dev/video7', 'control_values': {
        'logitech_brio_fov': '90', 'zoom_absolute': 'minimum'}}


def runner(tmp_path, outputs):
    cli = tmp_path / 'cameractrls.py'
    cli.write_text('# External CLI fixture, never executed\n')
    calls = []

    def run(command, **kwargs):
        assert kwargs['env']['LC_ALL'] == 'C'
        assert kwargs['timeout'] == 10
        assert command[:4] == ['/usr/bin/python3', str(cli), '-d', '/dev/video7']
        calls.append(command[4:])
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr='')

    return str(cli), calls, run


def test_fov_and_zoom_are_confirmed_by_a_separate_hardware_read(camera, tmp_path):
    after = LISTING.replace(' = 78', ' = 90').replace(' = 125', ' = 100')
    cli, calls, run = runner(tmp_path, iter([LISTING, '', after]))
    assert configure_controls(camera, command_runner=run, cli=cli) == {
        'logitech_brio_fov': '90', 'zoom_absolute': '100'}
    assert calls == [['--list'], ['-c', 'logitech_brio_fov=90,zoom_absolute=100'], ['--list']]


def test_minimum_zoom_comes_from_the_device_instead_of_a_brio_constant(camera, tmp_path):
    before = LISTING.replace('min: 100', 'min: 80')
    after = before.replace(' = 78', ' = 90').replace(' = 125', ' = 80')
    cli, _, run = runner(tmp_path, iter([before, '', after]))
    assert configure_controls(camera, command_runner=run, cli=cli)['zoom_absolute'] == '80'


@pytest.mark.parametrize('controls,match', [
    ({'unsupported_fov': 90}, 'does not support'),
    ({'logitech_brio_fov': '120'}, 'supported values'),
    ({'zoom_absolute': 102}, 'range/step'),
    ({'focus_absolute': 30}, 'inactive'),
])
def test_unsupported_requests_fail_before_any_write(camera, tmp_path, controls, match):
    camera['control_values'] = controls
    cli, calls, run = runner(tmp_path, iter([LISTING]))
    with pytest.raises(ValueError, match=match):
        configure_controls(camera, command_runner=run, cli=cli)
    assert calls == [['--list']]


def test_tool_success_does_not_hide_a_firmware_rejection(camera, tmp_path):
    cli, _, run = runner(tmp_path, iter([LISTING, '', LISTING]))
    with pytest.raises(ValueError, match='requested 90, read back 78'):
        configure_controls(camera, command_runner=run, cli=cli)


def test_unknown_readback_is_not_considered_success(camera, tmp_path):
    cli, _, run = runner(tmp_path, iter([LISTING, '', '']))
    with pytest.raises(ValueError, match='read back None'):
        configure_controls(camera, command_runner=run, cli=cli)


def test_no_controls_does_not_need_optional_cli():
    assert configure_controls({'device': '/missing'}, cli='/missing') == {}


def test_hung_control_tool_has_a_bounded_failure(camera, tmp_path):
    cli, _, _ = runner(tmp_path, iter([]))

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])

    with pytest.raises(ValueError, match='timed out'):
        configure_controls(camera, command_runner=timeout, cli=cli)


@pytest.mark.parametrize('name', ['fov', 'zoom', 'video_device', 'skip_device_check'])
def test_ros_driver_parameters_cannot_silently_ignore_or_override_capture_fields(name):
    with pytest.raises(ValueError, match='unsupported usb_cam parameter'):
        validate_driver_parameters({name: 90}, 'camera.driver_parameters')


@pytest.mark.parametrize('method', ['mmap', 'read', 'userptr'])
def test_usb_capture_io_methods_are_supported(method):
    validate_driver_parameters({'io_method': method}, 'camera.driver_parameters')


@pytest.mark.parametrize('method', ['', 'auto', 'MMAP', 0, True])
def test_invalid_usb_capture_io_methods_are_rejected(method):
    with pytest.raises(ValueError, match='io_method'):
        validate_driver_parameters({'io_method': method}, 'camera.driver_parameters')


@pytest.mark.parametrize('method', [None, 'mmap', 'read', 'userptr'])
def test_camera_launch_preserves_the_selected_io_method(method, tmp_path, monkeypatch):
    from face_tracking_bringup import hardware_launch

    nodes = []
    monkeypatch.setattr(hardware_launch, 'get_package_prefix', lambda name: str(tmp_path))
    monkeypatch.setattr(hardware_launch, 'configure_controls', lambda camera: {})
    monkeypatch.setattr(hardware_launch, 'guard', lambda node: None)
    monkeypatch.setattr(hardware_launch, 'Node', lambda **kwargs: nodes.append(kwargs))
    camera = {
        'calibration_file': str(tmp_path / 'left.yaml'), 'device': '/dev/video7',
        'width': 640, 'height': 480, 'fps': 30.0, 'pixel_format': 'mjpeg2rgb',
        'camera_name': 'left', 'frame_id': 'left_optical',
        'image_topic': '/left/image', 'info_topic': '/left/camera_info',
        'driver_parameters': {} if method is None else {'io_method': method},
    }
    hardware_launch.camera_actions({'mono_cpu': {'source': 'usb', 'left': camera}},
                                   'mono_cpu', False)
    assert nodes[0]['parameters'][0]['io_method'] == (method or 'mmap')


@pytest.mark.parametrize('controls', [
    {'preset': 'save'}, {'resolution': '640x480'}, {'systemd_cameractrlsd': 'enable'},
    {'logitech_brio_fov': '90,preset=save'}, {'zoom_absolute': 1.5},
])
def test_hardware_controls_do_not_expose_software_actions_or_extra_assignments(controls):
    with pytest.raises(ValueError):
        validate_control_values(controls, 'camera.control_values')
