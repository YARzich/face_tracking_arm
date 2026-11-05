#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Launch one bounded compact-maneuver visualization with a pinned target."""

import importlib.util
from pathlib import Path
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import yaml


def _load_joint1_working_boundary(source):
    logic_path = source / 'compact_demo_logic.py'
    spec = importlib.util.spec_from_file_location('compact_demo_logic', logic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Не удалось загрузить {logic_path}')
    logic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logic)
    return logic.joint1_working_boundary


def _limit_from_xacro(path, name):
    text = path.read_text(encoding='utf-8')
    match = re.search(
        rf'<xacro:arg\s+name="{re.escape(name)}"\s+default="([^"]+)"', text)
    if match is None:
        raise RuntimeError(f'Не найден {name} в {path}')
    return float(match.group(1))


def _simulation(context, share, source):
    side = LaunchConfiguration('side').perform(context)
    scenario = LaunchConfiguration('scenario').perform(context)
    servo_path = Path(LaunchConfiguration(
        'collision_aware_servo_config_file').perform(context))
    servo = yaml.safe_load(servo_path.read_text(encoding='utf-8')) or {}
    margin = float(servo.get('joint_position_margin_rad', 0.10))
    if side not in {'upper', 'lower'}:
        raise RuntimeError("side должен быть 'upper' или 'lower'")
    if scenario not in {'stationary', 'arc', 'after_return'}:
        raise RuntimeError("scenario должен быть 'stationary', 'arc' или 'after_return'")
    if margin < 0.0:
        raise RuntimeError('joint_position_margin_rad не может быть отрицательным')
    tracking = yaml.safe_load(
        (source.parent / 'config/tracking.yaml').read_text(encoding='utf-8'))
    face_distance = float(
        tracking['face_tracking_controller']['ros__parameters']['minimum_face_distance_m'])

    fixture_path = source.parent / f'config/fixtures/compact_demo_{side}_edge.yaml'
    fixture = yaml.safe_load(fixture_path.read_text(encoding='utf-8'))
    positions = fixture['initial_positions']
    xacro = source.parent / 'description/xarm6_hardware.urdf.xacro'
    low = _limit_from_xacro(xacro, 'joint1_lower_limit')
    high = _limit_from_xacro(xacro, 'joint1_upper_limit')
    positions['joint1'] = _load_joint1_working_boundary(source)(
        low, high, margin, side)

    with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', prefix='compact_demo_', suffix='.yaml',
            delete=False) as generated:
        yaml.safe_dump(fixture, generated, sort_keys=False)
        fixture_file = generated.name

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(share / 'launch/tracking_sim.launch.py')),
        launch_arguments={
            'robot_model': 'xarm6',
            'start_pose': 'folded',
            'initial_positions_file': fixture_file,
            'test_face_scenario': 'disabled',
            'camera_mode': 'disabled',
            'headless': LaunchConfiguration('headless'),
            'collision_aware_servo_config_file': LaunchConfiguration(
                'collision_aware_servo_config_file'),
            'visualize_target': 'true',
        }.items())
    target_runner = ExecuteProcess(
        cmd=['python3', str(source / 'compact_demo_target.py',),
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', f'scenario:={scenario}',
             '-p', f'face_distance_m:={face_distance}'],
        output='screen')
    return [simulation, target_runner]


def generate_launch_description():
    share = Path(get_package_share_directory('face_tracking_arm'))
    source = Path(__file__).resolve().parent
    headless = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo without its graphical client.')
    servo_config = DeclareLaunchArgument(
        'collision_aware_servo_config_file',
        default_value=str(source.parent / 'config/moveit/collision_aware_servo.yaml'),
        description='Collision-aware Servo YAML, including compact phase speed scales.')
    side = DeclareLaunchArgument(
        'side', default_value='upper',
        description='J1 рабочая граница: upper или lower.')
    scenario = DeclareLaunchArgument(
        'scenario', default_value='stationary',
        description='stationary, arc или after_return; arc двигает цель уже у края J1.')
    simulation = OpaqueFunction(
        function=lambda context: _simulation(context, share, source))
    return LaunchDescription([headless, servo_config, side, scenario, simulation])
