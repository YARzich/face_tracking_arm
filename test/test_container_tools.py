# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Host/container boundaries without Docker, ROS, cameras or a physical arm."""

import copy
import csv
import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docker'))
import host  # noqa: E402, I100
import runtime  # noqa: E402


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())


@pytest.mark.parametrize('value', [None, [], 'not a mapping'])
def test_empty_or_malformed_yaml_has_an_actionable_error(value):
    with pytest.raises(ValueError, match='hardware.yaml'):
        runtime.describe(value, 'cpu')


def test_resource_description_rejects_non_string_model_path_before_host_mounts(config):
    config['perception']['model_dir'] = None
    with pytest.raises(ValueError, match='perception.model_dir'):
        runtime.describe(config, 'cpu')


@pytest.mark.parametrize('mode', ['cpu', 'stereo'])
def test_container_config_preserves_user_settings_and_maps_only_selected_camera(config, mode):
    config[runtime.MODES[mode]]['source'] = 'usb'
    original = copy.deepcopy(config)
    result = runtime.resolved_config(config, mode)
    assert config == original
    assert result['motion'] == original['motion']
    assert result['robot'] == original['robot']
    assert result['perception']['python_executable'] == '/opt/face_tracking_venv/bin/python'
    for i, side in enumerate(['left', 'right'] if mode == 'stereo' else ['left']):
        camera = result[runtime.MODES[mode]][side]
        assert camera['device'] == f'/dev/video{i}'
        assert camera['calibration_file'] == f'/calibration/{side}.yaml'


def test_external_ros_camera_keeps_topics_and_frames(config):
    section = config['stereo']
    section['source'] = 'ros'
    section['left']['image_topic'] = '/vendor/image'
    result = runtime.resolved_config(config, 'stereo', external_models=True)
    assert result['stereo'] == section
    assert result['perception']['model_dir'] == '/models'
    assert runtime.describe(config, 'stereo')['cameras']['left'] == section['left']


def test_calibration_does_not_try_to_load_missing_old_calibration(config):
    result = runtime.resolved_config(config, 'cpu', calibration=True)
    assert result['mono_cpu']['left']['calibration_file'] == ''


def test_mount_quotes_spaces_and_commas_without_shell_evaluation(tmp_path):
    path = tmp_path / 'camera, settings $(touch unwanted)'
    flags = host.mount(path, '/input/hardware.yaml')
    fields = next(csv.reader([flags[1]]))
    assert fields == ['type=bind', f'src={path}', 'dst=/input/hardware.yaml', 'readonly']


@pytest.fixture
def resources(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    (models / host.MODELS['yunet']).write_bytes(b'test fixture')
    calibration = tmp_path / 'calibration/mono_cpu/left.yaml'
    calibration.parent.mkdir(parents=True)
    calibration.write_text('image_width: 640\n')
    return {'source': 'usb', 'cameras': {'left': {
        'device': '/dev/null', 'calibration_file': ''}},
        'model_dir': str(models), 'detector': 'yunet'}


def test_only_selected_devices_and_files_are_shared(resources, tmp_path):
    flags = host.resources(resources, tmp_path / 'hardware.yaml', 'cpu')
    assert '/dev/null:/dev/video0:rw' in flags
    mounts = [flags[i + 1] for i, flag in enumerate(flags) if flag == '--mount']
    assert len(mounts) == 3
    assert all('readonly' in next(csv.reader([m])) for m in mounts)
    assert not any('/var/run/docker.sock' in value or '--privileged' in value for value in flags)


def test_stereo_device_aliases_cannot_capture_the_same_camera_twice(resources, tmp_path):
    alias = tmp_path / 'camera_alias'
    alias.symlink_to('/dev/null')
    resources['cameras']['right'] = {'device': str(alias), 'calibration_file': ''}
    with pytest.raises(ValueError, match='разных устройства'):
        host.resources(resources, tmp_path / 'hardware.yaml', 'stereo', calibration=True)


def test_missing_calibration_stops_before_any_robot_launch(resources, tmp_path):
    resources['cameras']['left']['calibration_file'] = 'missing.yaml'
    with pytest.raises(ValueError, match='Нет калибровки'):
        host.resources(resources, tmp_path / 'hardware.yaml', 'cpu')


def test_external_camera_does_not_require_local_usb_or_calibration(resources, tmp_path):
    resources['source'] = 'ros'
    resources['cameras']['left']['device'] = '/missing/device'
    flags = host.resources(resources, tmp_path / 'hardware.yaml', 'cpu')
    assert '--device' not in flags
    assert not any('/calibration/' in value for value in flags)


def test_custom_model_path_is_not_silently_replaced_by_bundled_weights(resources, tmp_path):
    resources['model_dir'] = str(tmp_path / 'missing-models')
    with pytest.raises(ValueError, match='model_dir'):
        host.resources(resources, tmp_path / 'hardware.yaml', 'cpu')


@pytest.mark.parametrize('directory', ['', '~/.cache/face_tracking_arm/models'])
def test_default_model_directory_selects_bundled_yunet(resources, tmp_path, directory):
    resources['source'] = 'ros'
    resources['model_dir'] = directory
    flags = host.resources(resources, tmp_path / 'hardware.yaml', 'cpu')
    assert not any('/models/' in value for value in flags)


@pytest.mark.parametrize('camera_only', [False, True])
def test_control_priority_permissions_are_only_granted_for_arm_launch(
        config, tmp_path, monkeypatch, camera_only):
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    config['mono_cpu']['source'] = 'ros'
    monkeypatch.setattr(host, 'metadata', lambda *args: runtime.describe(config, 'cpu'))
    commands = []

    def start(command, *, start_new_session):
        assert start_new_session
        commands.append(command)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(host.subprocess, 'Popen', start)
    args = SimpleNamespace(config=path, operation='cpu', show_image=False,
                           camera_only=camera_only, mock=True, domain=71)
    assert host.execute(['docker'], args) == 0
    assert ('rtprio=50' in commands[0]) == (not camera_only)
    assert '--privileged' not in commands[0]
    assert '--mock' in commands[0]


def archive(path, contents):
    with tarfile.open(path, 'w:gz') as target:
        for name, value in contents.items():
            data = value.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            target.addfile(info, io.BytesIO(data))


def test_calibration_exports_only_expected_yaml_files(tmp_path):
    source = tmp_path / 'calibration.tar.gz'
    archive(source, {'ost.yaml': 'image_width: 640\n', '../outside': 'unwanted'})
    output = tmp_path / 'output'
    output.mkdir()
    runtime.save_calibration(source, output, 'cpu')
    assert [p.name for p in output.iterdir()] == ['left.yaml']
    assert yaml.safe_load((output / 'left.yaml').read_text())['image_width'] == 640
    assert not (tmp_path / 'outside').exists()


def test_incomplete_stereo_calibration_does_not_overwrite_existing_files(tmp_path):
    source = tmp_path / 'calibration.tar.gz'
    archive(source, {'left.yaml': 'image_width: 640\n'})
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'left.yaml').write_text('original')
    with pytest.raises(KeyError):
        runtime.save_calibration(source, output, 'stereo')
    assert (output / 'left.yaml').read_text() == 'original'
