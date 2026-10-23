# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Check that initialized MoveIt components release their workers and plugin objects."""

import os
from pathlib import Path
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.hardware_launch import (  # noqa: E402, I100
    build_artifact_manifest, robot_actions)


def test_fault_artifact_manifest_hashes_effective_launch_inputs(monkeypatch, tmp_path):
    share = tmp_path / 'share'
    (share / 'config/moveit').mkdir(parents=True)
    (share / 'config/control').mkdir(parents=True)
    for relative in ('config/moveit/servo.yaml',
                     'config/moveit/collision_aware_servo.yaml',
                     'config/moveit/kinematics.yaml',
                     'config/control/controllers.yaml',
                     'config/tracking.yaml'):
        path = share / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    config_path = tmp_path / 'resolved.yaml'
    config_path.write_text(
        'robot:\n  model: xarm6\n  ip: 192.0.2.1\n  secret: test-secret\n')
    monkeypatch.setenv('FACE_TRACKING_SOURCE_SHA256', 'f' * 64)
    monkeypatch.setenv('FACE_TRACKING_DOCKER_IMAGE_ID', 'sha256:' + 'a' * 64)
    manifest = build_artifact_manifest(
        config_path, share,
        [{'robot_description': '<robot/>'},
         {'robot_description_semantic': '<robot name="arm"/>'}],
        {'controller_manager': {'ros__parameters': {'update_rate': 100}}},
        {'model': 'xarm6', 'model_num': 1300},
        {'firmware_version': '2.4.0', 'serial_number': 'serial-test'},
        {'servo': {'publish_period': .01}, 'motion': {'gate': True},
         'tracking': {'require_perception_status': True}})

    assert manifest['source_tree_sha256'] == 'f' * 64
    assert manifest['docker_image_id'] == 'sha256:' + 'a' * 64
    assert manifest['image_digest'] is None
    assert manifest['image_digest_status'] == 'manifest_digest_unavailable'
    assert manifest['firmware_version'] == '2.4.0'
    assert manifest['robot_serial'] == 'serial-test'
    assert len(manifest['artifacts']['expanded_urdf_sha256']) == 64
    assert len(manifest['artifacts']['expanded_srdf_sha256']) == 64
    assert len(manifest['artifacts']['effective_controllers_sha256']) == 64
    assert all(len(value) == 64 for value in
               manifest['artifacts']['effective_parameters_sha256'].values())
    assert len(manifest['artifacts']['hardware_config_sha256']) == 64
    assert all(len(value) == 64 for value in manifest['artifacts']['source_yaml_sha256'].values())
    assert '192.0.2.1' not in str(manifest)
    assert 'test-secret' not in str(manifest)
    assert '<robot/>' in manifest['replay_inputs']['expanded_urdf']
    assert 'ip: <redacted>' in manifest['replay_inputs']['resolved_hardware_yaml_redacted']
    assert 'secret: <redacted>' in manifest['replay_inputs']['resolved_hardware_yaml_redacted']


@pytest.mark.parametrize('omit_monitor,table_shape', [
    (False, None), (True, None), (False, 'cylinder'), (True, 'cylinder'), (True, 'box')])
def test_initialized_components_exit_without_retained_plugin_objects(
        monkeypatch, tmp_path, omit_monitor, table_shape):
    # Use the real launch assembly with GenericSystem, without cameras or a robot connection.
    monkeypatch.setenv('ROS_DOMAIN_ID', str(101 + os.getpid() % 90))
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(ROOT / 'docker/fastdds.xml'))
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    monkeypatch.setenv('RCUTILS_COLORIZED_OUTPUT', '0')
    output, exits = [], []
    initialized = False

    def capture_output(event):
        nonlocal initialized
        output.append(event.text.decode(errors='replace'))
        if not initialized and 'Collision-aware Servo initialized:' in ''.join(output):
            initialized = True
            return [TimerAction(period=0.5, actions=[
                EmitEvent(event=Shutdown(reason='Initialized; exercise normal shutdown'))])]
        return None

    def capture_exit(event, context):
        exits.append((event.process_name, event.returncode))

    io_handler = RegisterEventHandler(OnProcessIO(
        on_stdout=capture_output, on_stderr=capture_output))
    exit_handler = RegisterEventHandler(OnProcessExit(on_exit=capture_exit))
    deadline = TimerAction(period=25.0, actions=[
        EmitEvent(event=Shutdown(reason='Controller initialization timed out'))])
    share = Path(get_package_share_directory('face_tracking_arm'))
    config = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    config['robot']['ip'] = ''
    if omit_monitor:
        config['monitor'].update(size_m=[0., 0., 0.], mass_kg=0.)
    if table_shape:
        config['table'].update(
            shape=table_shape, dimensions_m=[0.] * (3 if table_shape == 'box' else 2))
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    with directory:
        config_path = Path(directory.name) / 'hardware.yaml'
        config_path.write_text(yaml.safe_dump(config))
        nodes = robot_actions(config, 'mono_cpu', share, config_path, directory, mock=True)
        description = LaunchDescription([io_handler, exit_handler, deadline, *nodes])
        service = LaunchService(argv=[])
        service.include_launch_description(description)
        result = service.run()

    log = ''.join(output)
    assert result == 0, log
    assert initialized, log
    assert len(exits) == 5 and all(code == 0 for _, code in exits), (exits, log)
    assert 'Stopping planning scene monitor' in log, log
    assert 'SEVERE WARNING' not in log, log
