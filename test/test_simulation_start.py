# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Simulation startup selection preserves measured angles and model boundaries."""

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup.robot_profile import get_robot_profile  # noqa: E402, I100
from face_tracking_bringup.simulation_start import resolve_simulation_start  # noqa: E402


PACKAGE = Path(__file__).resolve().parents[1]


def _angles(path):
    positions = yaml.safe_load(path.read_text())['initial_positions']
    return [positions[f'joint{i}'] for i in range(1, 7)]


def _custom_file(tmp_path, angles):
    path = tmp_path / 'custom.yaml'
    path.write_text(yaml.safe_dump({'initial_positions': {
        f'joint{i}': value for i, value in enumerate(angles, start=1)}}))
    return path


@pytest.mark.parametrize('model,pose', [
    ('xarm6', 'rest'), ('xarm6', 'zero'), ('xarm6', 'folded'),
    ('xarm6', 'joint1_limit'), ('lite6', 'rest'), ('lite6', 'zero'),
    ('lite6', 'joint1_limit'), ('lite6', 'incident'),
])
def test_presets_exist_and_fit_the_selected_model(model, pose):
    path = resolve_simulation_start(PACKAGE, robot_model=model, start_pose=pose)
    assert path.is_absolute()
    limits = get_robot_profile(model)['position_limits_rad']
    assert all(low <= value <= high for value, (low, high) in zip(_angles(path), limits))


@pytest.mark.parametrize('model', ['xarm6', 'lite6'])
def test_default_start_matches_the_models_semantic_rest(model):
    profile = get_robot_profile(model)
    xml = ET.parse(PACKAGE / profile['srdf_file']).getroot()
    rest = xml.find(f"group_state[@name='rest'][@group='{profile['planning_group']}']")
    expected = {joint.attrib['name']: float(joint.attrib['value']) for joint in rest}
    angles = _angles(resolve_simulation_start(PACKAGE, robot_model=model))
    assert angles == pytest.approx([expected[f'joint{i}'] for i in range(1, 7)], abs=1e-8)


def test_zero_means_joint_coordinates_not_a_cartesian_pose():
    assert _angles(resolve_simulation_start(PACKAGE, start_pose='zero')) == [0] * 6


@pytest.mark.parametrize('model', ['xarm6', 'lite6'])
def test_joint1_limit_keeps_turn_count_and_other_rest_angles(model):
    rest = _angles(resolve_simulation_start(PACKAGE, robot_model=model))
    edge = _angles(resolve_simulation_start(
        PACKAGE, robot_model=model, start_pose='joint1_limit'))
    assert math.degrees(edge[0]) == pytest.approx(359.9)
    assert edge[1:] == rest[1:]


def test_incident_is_exactly_the_recorded_lite6_configuration():
    angles = _angles(resolve_simulation_start(
        PACKAGE, robot_model='lite6', start_pose='incident'))
    assert list(map(math.degrees, angles)) == pytest.approx(
        [354.3, -67.5, 20.3, -100.4, -91.6, 117.8])


@pytest.mark.parametrize('model,pose', [('xarm6', 'incident'), ('lite6', 'folded')])
def test_model_specific_pose_is_not_silently_reinterpreted(model, pose):
    with pytest.raises(ValueError, match=f'start_pose={pose}'):
        resolve_simulation_start(PACKAGE, robot_model=model, start_pose=pose)


def test_explicit_file_overrides_the_preset_without_rewriting_angles(tmp_path):
    angles = [2 * math.pi, -.2, -.3, -2 * math.pi, -.9, 1.8]
    path = _custom_file(tmp_path, angles)
    before = path.read_bytes()
    resolved = resolve_simulation_start(
        PACKAGE, start_pose='incident', initial_positions_file=str(path))
    assert resolved == path
    assert _angles(resolved) == angles
    assert path.read_bytes() == before


@pytest.mark.parametrize('offset', [0.0, -1e-9])
def test_physical_boundary_needs_no_artificial_startup_margin(tmp_path, offset):
    path = _custom_file(tmp_path, [2 * math.pi + offset, 0, 0, 0, 0, 0])
    assert resolve_simulation_start(PACKAGE, initial_positions_file=path) == path


def test_even_a_small_physical_limit_violation_is_rejected(tmp_path):
    path = _custom_file(tmp_path, [2 * math.pi + 1e-9, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match='joint1'):
        resolve_simulation_start(PACKAGE, initial_positions_file=path)


def test_custom_lite6_incident_does_not_bypass_xarm6_joint_limits():
    incident = PACKAGE / 'config/control/start_poses/lite6_incident.yaml'
    with pytest.raises(ValueError, match='joint3'):
        resolve_simulation_start(PACKAGE, initial_positions_file=incident)


@pytest.mark.parametrize('value', [True, None, '0.5', float('nan'), float('inf')])
def test_custom_angles_must_be_finite_numbers(tmp_path, value):
    path = _custom_file(tmp_path, [value, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        resolve_simulation_start(PACKAGE, initial_positions_file=path)


@pytest.mark.parametrize('data', [
    {}, {'joint1': 0}, {'initial_positions': []},
    {'initial_positions': {'joint1': 0}},
    {'initial_positions': {f'joint{i}': 0 for i in range(1, 8)}},
    {'initial_positions': {f'joint{i}': 0 for i in range(1, 7)}, 'units': 'degrees'},
])
def test_yaml_requires_exactly_six_named_joint_positions(tmp_path, data):
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='initial_positions'):
        resolve_simulation_start(PACKAGE, initial_positions_file=path)


def test_invalid_yaml_and_missing_file_report_the_input_path(tmp_path):
    path = tmp_path / 'invalid.yaml'
    for content in (None, 'initial_positions: ['):
        if content is not None:
            path.write_text(content)
        with pytest.raises(ValueError, match='Cannot read initial_positions_file'):
            resolve_simulation_start(PACKAGE, initial_positions_file=path)


def test_unknown_models_and_presets_are_rejected(tmp_path):
    path = _custom_file(tmp_path, [0] * 6)
    with pytest.raises(ValueError, match='robot.model'):
        resolve_simulation_start(PACKAGE, robot_model='xarm7')
    with pytest.raises(ValueError, match='Unknown start_pose'):
        resolve_simulation_start(PACKAGE, start_pose='typo', initial_positions_file=path)


def test_launch_bootstraps_installed_helpers_without_project_pythonpath(tmp_path):
    pytest.importorskip('launch.actions')
    pytest.importorskip('launch_ros.actions')
    installed = tmp_path / 'lib/face_tracking_arm/face_tracking_bringup'
    installed.mkdir(parents=True)
    for filename in ('__init__.py', 'robot_profile.py', 'simulation_start.py'):
        shutil.copyfile(PACKAGE / 'python/face_tracking_bringup' / filename,
                        installed / filename)
    env = dict(os.environ)
    project_python = (PACKAGE / 'python').resolve()
    env['PYTHONPATH'] = os.pathsep.join(
        entry for entry in env.get('PYTHONPATH', '').split(os.pathsep)
        if entry and Path(entry).resolve() != project_python)
    script = """
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('tracking_sim', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.get_package_prefix = lambda package: sys.argv[2]
module.get_package_share_directory = lambda package: sys.argv[3]
description = module.generate_launch_description()
assert description.entities
import face_tracking_bringup.simulation_start as helper
assert Path(helper.__file__).is_relative_to(Path(sys.argv[2]))
"""
    result = subprocess.run(
        [sys.executable, '-c', script, str(PACKAGE / 'launch/tracking_sim.launch.py'),
         str(tmp_path), str(PACKAGE)], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
