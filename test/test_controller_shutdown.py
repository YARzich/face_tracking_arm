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
from face_tracking_bringup.hardware_launch import robot_actions  # noqa: E402, I100


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
