# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Host/container boundaries without Docker, ROS, cameras or a physical arm."""

import copy
import csv
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docker'))
import host  # noqa: E402, I100
import image_manifest  # noqa: E402
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
    devices = {'left': '/dev/video7', 'right': '/dev/video9'}
    result = runtime.resolved_config(config, mode, video_devices=devices)
    assert config == original
    assert result['motion'] == original['motion']
    assert result['robot'] == original['robot']
    assert result['limits'] == original['limits']
    assert result['controller'] == original['controller']
    assert result['point_filter'] == original['point_filter']
    assert result['perception']['python_executable'] == '/opt/face_tracking_venv/bin/python'
    for side in ['left', 'right'] if mode == 'stereo' else ['left']:
        camera = result[runtime.MODES[mode]][side]
        assert camera['device'] == devices[side]
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
    result = runtime.resolved_config(config, 'cpu', calibration=True,
                                     video_devices={'left': '/dev/video7'})
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
    assert '/dev/null:/dev/null:rw' in flags
    assert 'FACE_TRACKING_VIDEO_DEVICES={"left": "/dev/null"}' in flags
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


@pytest.mark.parametrize('mode', ['cpu', 'stereo'])
def test_missing_calibration_command_uses_selected_config_with_shell_quoting(
        resources, tmp_path, mode):
    config = tmp_path / "profiles/station's $(touch unwanted)/hardware.yaml"
    resources['cameras']['left']['calibration_file'] = 'missing.yaml'
    with pytest.raises(ValueError, match='Нет калибровки') as error:
        host.resources(resources, config, mode)
    command = str(error.value).split('Выполните ', 1)[1]
    assert shlex.split(command) == ['./run', 'calibrate', mode, '--config', str(config)]


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


@pytest.mark.parametrize('point_filter', [None, 'off', 'kalman', 'one_euro'])
@pytest.mark.parametrize('camera_only', [False, True])
def test_control_priority_permissions_are_only_granted_for_arm_launch(
        config, tmp_path, monkeypatch, camera_only, point_filter):
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
                           point_filter=point_filter,
                           camera_only=camera_only, mock=True, domain=71)
    assert host.execute(['docker'], args) == 0
    assert ('rtprio=50' in commands[0]) == (not camera_only)
    assert '--privileged' not in commands[0]
    assert '--mock' in commands[0]
    if point_filter is None:
        assert '--point-filter' not in commands[0]
    else:
        assert commands[0][-2:] == ['--point-filter', point_filter]


@pytest.mark.parametrize('mode', ['cpu', 'stereo'])
@pytest.mark.parametrize('method', [None, 'off', 'kalman', 'one_euro'])
def test_filter_override_preserves_tuning_and_original_config(config, mode, method):
    config[runtime.MODES[mode]]['source'] = 'ros'
    config['point_filter']['enabled'] = False
    config['point_filter']['one_euro'] = {'beta': 7.0}
    original = copy.deepcopy(config)
    result = runtime.resolved_config(config, mode, point_filter=method)
    expected = copy.deepcopy(original['point_filter'])
    if method is not None:
        expected['enabled'] = method != 'off'
        if method != 'off':
            expected['method'] = method
    assert result['point_filter'] == expected
    assert config == original


@pytest.mark.parametrize('method', ['off', 'kalman', 'one_euro'])
def test_filter_override_supports_old_config_without_filter_section(config, method):
    del config['point_filter']
    result = runtime.resolved_config(config, 'stereo', point_filter=method)
    assert result['point_filter']['enabled'] == (method != 'off')
    if method != 'off':
        assert result['point_filter']['method'] == method
    assert 'point_filter' not in config


def test_filter_override_reports_invalid_configuration(config):
    config['point_filter'] = []
    with pytest.raises(ValueError, match='point_filter'):
        runtime.resolved_config(config, 'stereo', point_filter='kalman')
    with pytest.raises(ValueError, match='point_filter'):
        runtime.resolved_config(config, 'stereo', point_filter='typo')


@pytest.mark.parametrize('mode', ['cpu', 'stereo'])
@pytest.mark.parametrize('method', ['off', 'kalman', 'one_euro'])
def test_filter_cli_reaches_runtime_config_and_launch(config, tmp_path, monkeypatch, mode, method):
    seen = []
    monkeypatch.setattr(host, 'docker_connection', lambda: ['docker'])
    monkeypatch.setattr(host, 'image_exists', lambda _: True)
    monkeypatch.setattr(host, 'execute', lambda _, args: seen.append(args.point_filter) or 0)
    monkeypatch.setattr(sys, 'argv', ['./run', mode, '--point-filter', method])
    assert host.main() == 0
    assert seen == [method]
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runtime, 'CONFIG', path)

    def write(config, selected_mode, *, point_filter):
        assert selected_mode == mode
        assert point_filter == method
        return tmp_path / 'resolved.yaml'

    commands = []
    monkeypatch.setattr(runtime, 'write_config', write)
    monkeypatch.setattr(runtime.os, 'execvp', lambda _, command: commands.append(command))
    monkeypatch.setattr(sys, 'argv', ['runtime.py', 'launch', mode, '--point-filter', method])
    runtime.main()
    assert f'tracking_hardware_{mode}.launch.py' in commands[0]
    assert f'config_file:={tmp_path / "resolved.yaml"}' in commands[0]
    assert path.read_text() == yaml.safe_dump(config)


@pytest.mark.parametrize('module,arguments', [
    (host, ['cpu', '--point-filter', 'typo']),
    (host, ['build', '--point-filter', 'kalman']),
    (runtime, ['launch', 'cpu', '--point-filter', 'typo']),
    (runtime, ['calibrate', 'cpu', '--point-filter', 'kalman']),
])
def test_filter_cli_rejects_wrong_method_or_operation(module, arguments, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run', *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


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


def test_usb_requires_an_explicit_host_device_mapping(config):
    with pytest.raises(ValueError, match='device mapping'):
        runtime.resolved_config(config, 'cpu')


def test_export_archive_version_matches_the_current_image(monkeypatch):
    monkeypatch.setattr(host, 'architecture', lambda: 'amd64')
    assert host.archive_path().name == 'face-tracking-arm-v2-amd64.tar.gz'


@pytest.fixture
def image_source(tmp_path):
    for name in image_manifest.INPUT_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    for name in image_manifest.INPUT_DIRECTORIES:
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'source.txt').write_text(name)
    return tmp_path


def test_image_manifest_covers_every_local_docker_copy():
    # config is copied as a directory but hashed only through its explicit defaults.
    covered = set(image_manifest.INPUT_FILES) | set(image_manifest.INPUT_DIRECTORIES) | {'config'}
    for line in (ROOT / 'docker/Dockerfile').read_text().splitlines():
        if line.startswith('COPY ') and not line.startswith('COPY --from='):
            for source in shlex.split(line)[1:-1]:
                assert any(source == item or source.startswith(item + '/') for item in covered)


def test_program_configuration_manifest_contains_existing_defaults():
    assert 'config' not in image_manifest.INPUT_DIRECTORIES
    assert 'config/hardware.yaml' not in image_manifest.INPUT_FILES
    assert image_manifest.PROGRAM_CONFIG_FILES
    for name in image_manifest.PROGRAM_CONFIG_FILES:
        assert name.startswith('config/')
        assert (ROOT / name).is_file()


def test_fingerprint_never_scans_user_configuration_directories(image_source, monkeypatch):
    before = image_manifest.source_fingerprint(image_source)
    profile = image_source / 'config/profiles/custom'
    profile.mkdir(parents=True)
    (profile / 'hardware.yaml').write_text('user profile')
    (profile / 'model.onnx').write_bytes(b'large external model placeholder')
    original_rglob = Path.rglob
    original_open = Path.open

    def rglob(path, pattern, *args, **kwargs):
        assert path != image_source / 'config', 'config must not be traversed'
        return original_rglob(path, pattern, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        assert not path.is_relative_to(profile), 'user assets must not be read'
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'rglob', rglob)
    monkeypatch.setattr(Path, 'open', open_file)
    assert image_manifest.source_fingerprint(image_source) == before


@pytest.mark.parametrize('name', [
    'config/hardware.yaml', 'config/another.local.yaml', 'config/custom.yaml', 'config/custom.yml',
    'config/calibration/mono_cpu/left.yaml',
    'config/profiles/desk/hardware.yaml', 'config/profiles/desk/calibration/mono_cpu/left.yaml',
    'config/profiles/desk/calibration/stereo/right.yaml', 'config/models/face.onnx',
    'config/keys/camera.key', 'config/hardware.yaml.bak',
    'config/control/custom.yaml', 'config/moveit/custom.yaml',
    'README.md', 'ROADMAP.md', 'docs/hardware.md', 'test/test_example.py',
    'python/example/__pycache__/node.cpython-312.pyc', 'launch/__pycache__/launch.pyc',
    'python/.pytest_cache/state', 'python/.cache/state', 'config/hardware.yaml~',
])
def test_runtime_settings_and_non_program_files_do_not_require_new_image(image_source, name):
    before = image_manifest.source_fingerprint(image_source)
    path = image_source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('first settings')
    assert image_manifest.source_fingerprint(image_source) == before
    path.write_text('changed settings')
    assert image_manifest.source_fingerprint(image_source) == before
    path.unlink()
    assert image_manifest.source_fingerprint(image_source) == before


@pytest.mark.parametrize('suffix', sorted(image_manifest.IGNORED_SUFFIXES))
def test_dockerignored_weights_keys_and_temporary_files_do_not_change_fingerprint(
        image_source, suffix):
    before = image_manifest.source_fingerprint(image_source)
    path = image_source / 'python/example' / ('data' + suffix)
    path.parent.mkdir()
    path.write_bytes(b'external data')
    assert image_manifest.source_fingerprint(image_source) == before
    path.write_bytes(b'changed external data')
    assert image_manifest.source_fingerprint(image_source) == before


def test_ignored_suffixes_are_excluded_from_the_docker_build_context():
    patterns = set((ROOT / '.dockerignore').read_text().splitlines())
    for suffix in image_manifest.IGNORED_SUFFIXES:
        pattern = '**/*.py[cod]' if suffix in ('.pyc', '.pyo', '.pyd') else '**/*' + suffix
        assert pattern in patterns


@pytest.mark.parametrize('name', [
    *image_manifest.PROGRAM_CONFIG_FILES,
    'docker/Dockerfile', 'package.xml', 'python/example/node.py',
    'src/controller.cpp', 'include/controller.hpp', 'launch/example.launch.py',
    'description/monitor.xacro', 'docker/runtime.py', 'docker/patches/driver.patch',
])
def test_bundled_code_and_dependency_changes_require_new_image(image_source, name):
    before = image_manifest.source_fingerprint(image_source)
    path = image_source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('first version')
    first = image_manifest.source_fingerprint(image_source)
    assert first != before
    path.write_text('second version')
    assert image_manifest.source_fingerprint(image_source) != first


def test_fingerprint_ignores_file_times_and_checkout_location(image_source, tmp_path_factory):
    before = image_manifest.source_fingerprint(image_source)
    path = image_source / 'package.xml'
    os.utime(path, (1, 1))
    assert image_manifest.source_fingerprint(image_source) == before
    target = tmp_path_factory.mktemp('relocated') / 'project'
    shutil.copytree(image_source, target)
    assert image_manifest.source_fingerprint(target) == before


def test_removing_or_renaming_bundled_source_changes_fingerprint(image_source):
    before = image_manifest.source_fingerprint(image_source)
    path = image_source / 'src/source.txt'
    moved = path.with_name('other.txt')
    path.rename(moved)
    renamed = image_manifest.source_fingerprint(image_source)
    assert renamed != before
    moved.unlink()
    assert image_manifest.source_fingerprint(image_source) not in (before, renamed)


@pytest.mark.parametrize('source_label', [None, 'old-version', 'matching'])
def test_existing_image_must_match_program_contents(image_source, monkeypatch, source_label):
    monkeypatch.setattr(host, 'ROOT', image_source)
    monkeypatch.setattr(host, 'architecture', lambda: 'amd64')
    labels = {'io.face_tracking_arm.launch-schema': '2'}
    if source_label is not None:
        labels['io.face_tracking_arm.source-sha256'] = (
            image_manifest.source_fingerprint(image_source)
            if source_label == 'matching' else source_label)
    image = {'Architecture': 'amd64', 'Config': {'Labels': labels}}
    monkeypatch.setattr(host, 'capture', lambda command: SimpleNamespace(
        returncode=0, stdout=json.dumps([image])))
    if source_label == 'matching':
        assert host.image_exists(['docker'])
    else:
        with pytest.raises(ValueError, match=r'\./run build'):
            host.image_exists(['docker'])


def test_prepare_never_automatically_rebuilds_a_stale_image(monkeypatch, tmp_path):
    def stale(docker):
        raise host.ImageCompatibilityError('stale image: ./run build')

    def unexpected_build(docker):
        pytest.fail('prepare must not rebuild an existing stale image')

    monkeypatch.setattr(host, 'image_exists', stale)
    monkeypatch.setattr(host, 'build', unexpected_build)
    monkeypatch.setattr(host, 'archive_path', lambda: tmp_path / 'missing.tar.gz')
    with pytest.raises(ValueError, match='stale image'):
        host.prepare(['docker'])


@pytest.mark.parametrize('valid_checksum', [False, True])
@pytest.mark.parametrize('compatible_archive', [False, True])
def test_prepare_can_replace_stale_image_only_with_checked_archive(
        tmp_path, monkeypatch, valid_checksum, compatible_archive):
    path = tmp_path / 'image.tar.gz'
    path.write_bytes(b'image archive fixture')
    path.with_suffix('.gz.sha256').write_text(host.digest(path) if valid_checksum else 'bad')
    commands = []
    inspected = []

    def exists(docker):
        inspected.append(True)
        if len(inspected) == 1 or not compatible_archive:
            raise host.ImageCompatibilityError('stale image')
        return True

    monkeypatch.setattr(host, 'image_exists', exists)
    monkeypatch.setattr(host, 'archive_path', lambda: path)
    monkeypatch.setattr(host.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(host, 'build', lambda docker: pytest.fail('must not build a stale image'))
    if valid_checksum and compatible_archive:
        host.prepare(['docker'])
    else:
        with pytest.raises(ValueError):
            host.prepare(['docker'])
    assert commands == ([['docker', 'image', 'load', '-i', str(path)]] if valid_checksum else [])
    assert len(inspected) == (2 if valid_checksum else 1)


def test_build_records_source_fingerprint_without_running_docker(image_source, monkeypatch):
    monkeypatch.setattr(host, 'ROOT', image_source)
    monkeypatch.setattr(host, 'architecture', lambda: 'amd64')
    commands = []
    monkeypatch.setattr(host.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    host.build(['docker'])
    argument = 'FACE_TRACKING_SOURCE_SHA256=' + image_manifest.source_fingerprint(image_source)
    assert commands[0][commands[0].index('--build-arg') + 1] == argument


@pytest.mark.parametrize('interrupted', [False, True])
@pytest.mark.parametrize('exit_code', [0, 1])
def test_calibration_reports_actual_config_directory_and_custom_path(
        config, tmp_path, monkeypatch, capsys, interrupted, exit_code):
    path = tmp_path / 'custom settings/hardware.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config))
    camera = config['mono_cpu']['left']
    camera.update(device='/dev/null', calibration_file='previous.yaml')
    monkeypatch.setattr(host, 'metadata', lambda *args: runtime.describe(config, 'cpu'))
    monkeypatch.setattr(host, 'display_options', lambda environment: [])
    monkeypatch.setattr(host, 'capture', lambda command: None)
    waits = [exit_code]
    if interrupted:
        waits.insert(0, KeyboardInterrupt())

    def wait(**kwargs):
        result = waits.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(host.subprocess, 'Popen',
                        lambda *args, **kwargs: SimpleNamespace(wait=wait))
    args = SimpleNamespace(config=path, operation='calibrate', mode='cpu', show_image=False,
                           point_filter=None,
                           camera_only=False, mock=False, domain=71, size='8x6', square=.025)
    assert host.execute(['docker'], args) == (130 if interrupted else exit_code)
    output = capsys.readouterr().out
    assert (str(path.parent / 'calibration/mono_cpu') in output) == (exit_code == 0)
    assert ('calibration_file' in output) == (exit_code == 0)
