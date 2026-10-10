# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Resolve simulation descriptions and parameters without starting ROS processes."""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
import yaml

from .mount_collisions import allow_fixed_mount_collisions
from .robot_profile import get_robot_profile
from .simulation_start import resolve_simulation_start


_GAZEBO_PHYSICS_LIMIT_PADDING_RAD = 0.01


def _gazebo_physics_robot_description(robot_description):
    """
    Give bounded Gazebo revolute joints clearance from the DART hard stop.

    This copy is only used to spawn the Gazebo entity. The original robot
    description remains authoritative for robot_state_publisher, MoveIt,
    ros2_control command-interface limits, and real hardware.
    """
    try:
        root = ET.fromstring(robot_description)
    except (ET.ParseError, TypeError) as error:
        raise ValueError('Cannot parse robot_description for Gazebo spawn') from error

    for joint in root.findall('./joint[@type="revolute"]'):
        limit = joint.find('limit')
        if limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib:
            raise ValueError(
                f"Bounded revolute joint '{joint.get('name', '<unnamed>')}' "
                'must define finite lower and upper limits')
        try:
            lower = float(limit.attrib['lower'])
            upper = float(limit.attrib['upper'])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Revolute joint '{joint.get('name', '<unnamed>')}' has "
                'non-numeric position limits') from error
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"Revolute joint '{joint.get('name', '<unnamed>')}' must have "
                'finite lower < upper position limits')

        padded_lower = lower - _GAZEBO_PHYSICS_LIMIT_PADDING_RAD
        padded_upper = upper + _GAZEBO_PHYSICS_LIMIT_PADDING_RAD
        if not math.isfinite(padded_lower) or not math.isfinite(padded_upper):
            raise ValueError(
                f"Revolute joint '{joint.get('name', '<unnamed>')}' limits "
                'cannot be padded safely')
        limit.set('lower', format(padded_lower, '.17g'))
        limit.set('upper', format(padded_upper, '.17g'))

    return ET.tostring(root, encoding='unicode')


@dataclass
class SimulationConfig:
    """Prepared parameter sets shared by the simulation's launch actions."""

    robot_parameters: list
    standard_servo: dict
    collision_aware_servo: dict
    controller: dict
    tracking: dict
    marker: dict
    state_publisher: dict


def camera_mappings(*, mode, baseline, width, height, rate, horizontal_fov):
    """Validate camera launch values and convert them to xacro mappings."""
    if mode not in ('disabled', 'mono_cpu', 'stereo', 'rgbd'):
        raise ValueError('camera_mode must be disabled, mono_cpu, stereo or rgbd')
    if mode == 'disabled':
        return {
            'camera_mode': mode, 'camera_baseline': '0.08',
            'camera_width': '1280', 'camera_height': '720',
            'camera_rate': '30.0', 'camera_horizontal_fov': '1.433',
        }
    width, height = int(width), int(height)
    rate, horizontal_fov = map(float, (rate, horizontal_fov))
    if (width <= 0 or height <= 0 or not math.isfinite(rate)
            or rate <= 0 or not 0 < horizontal_fov < math.pi):
        raise ValueError(
            'Camera dimensions/rate must be positive; horizontal FOV is radians (0, pi)')
    if mode == 'stereo':
        baseline = float(baseline)
        if not 0.02 <= baseline <= 0.25:
            raise ValueError('camera_baseline must be 0.02..0.25 meters')
    else:
        baseline = .08
    return {
        'camera_mode': mode, 'camera_baseline': str(baseline),
        'camera_width': str(width), 'camera_height': str(height),
        'camera_rate': str(rate), 'camera_horizontal_fov': str(horizontal_fov),
    }


def _load_yaml(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _robot_parameters(share, profile, initial_positions_path, camera):
    mappings = {
        **camera, 'enable_ros2_control': 'true',
        'initial_positions_file': str(initial_positions_path),
        'controller_config_file': str(share / 'config/control/controllers.yaml'),
    }
    xml = xacro.process_file(
        str(share / 'description' / f'{profile["model"]}_table.urdf.xacro'),
        mappings=mappings).toxml()
    semantic = ET.parse(share / profile['srdf_file']).getroot()
    allow_fixed_mount_collisions(ET.fromstring(xml), semantic)
    return [
        {'robot_description': xml},
        {'robot_description_semantic': ET.tostring(semantic, encoding='unicode')},
        {'robot_description_kinematics': _load_yaml(share / 'config/moveit/kinematics.yaml')},
        {'robot_description_planning': _load_yaml(share / 'config/moveit/joint_limits.yaml')},
    ]


def _servo_parameters(share, planning_group, smoothing_plugin):
    standard = _load_yaml(share / 'config/moveit/servo.yaml')
    standard.update(move_group_name=planning_group,
                    smoothing_filter_plugin_name=smoothing_plugin)
    collision_aware = dict(standard)
    # Match the command and JTC grids. Position-only output preserves the exact
    # linear joint-space segments checked by the collision validator.
    collision_aware.update(
        publish_period=0.01, max_expected_latency=0.05,
        check_collisions=False, use_smoothing=False,
        publish_joint_positions=True, publish_joint_velocities=False,
        publish_joint_accelerations=False)
    return standard, collision_aware


def _controller_parameters(
        share, planning_group, camera_mode, joint_limits, idle_behavior,
        controller_config_file=None):
    config_path = (Path(controller_config_file).expanduser() if controller_config_file else
                   share / 'config/moveit/collision_aware_servo.yaml')
    controller = _load_yaml(config_path)
    compact = controller.setdefault('compact_maneuver', {})
    if not isinstance(compact, dict):
        raise ValueError('compact_maneuver must be a mapping')
    compact_scale = compact.get('speed_scale', 1.0)
    if (isinstance(compact_scale, bool) or not isinstance(compact_scale, (int, float)) or
            not math.isfinite(compact_scale) or compact_scale <= 0):
        raise ValueError('compact_maneuver.speed_scale must be finite and positive')
    compact['speed_scale'] = float(compact_scale)
    controller['planning_group_name'] = planning_group
    planner = controller['recovery_planner']
    planner[planning_group] = planner.pop('xarm6')
    if camera_mode != 'disabled':
        controller['gaze_frame'] = 'face_test_camera_optical_frame'
    controller['joint_names'] = list(joint_limits)
    controller['search_enabled'] = idle_behavior != 'rest'
    for field, limit in (
        ('max_joint_velocity_rad_s', 'max_velocity'),
        ('max_joint_acceleration_rad_s2', 'max_acceleration'),
        ('max_joint_jerk_rad_s3', 'max_jerk'),
    ):
        controller[field] = [values[limit] for values in joint_limits.values()]
    return controller


def load_simulation_config(share, *, robot_model, start_pose, initial_positions_file,
                           camera, servo_backend, smoothing_plugin, idle_behavior,
                           require_perception_status=False, controller_config_file=None):
    """Build a consistent model and controller configuration for either backend."""
    share = Path(share)
    if idle_behavior != 'rest' and servo_backend != 'collision_aware':
        raise ValueError('Search requires the collision_aware backend')
    profile = get_robot_profile(robot_model)
    initial_positions = resolve_simulation_start(
        share, robot_model=robot_model, start_pose=start_pose,
        initial_positions_file=initial_positions_file)
    robot = _robot_parameters(share, profile, initial_positions, camera)
    standard, collision_aware = _servo_parameters(
        share, profile['planning_group'], smoothing_plugin)
    controller = _controller_parameters(
        share, profile['planning_group'], camera['camera_mode'],
        robot[3]['robot_description_planning']['joint_limits'], idle_behavior,
        controller_config_file)
    tracking = _load_yaml(share / 'config/tracking.yaml')
    tracking_parameters = tracking['face_tracking_controller']['ros__parameters']
    tracking_parameters['require_perception_status'] = bool(require_perception_status)
    controller['face_target_freshness_timeout_sec'] = \
        tracking_parameters['face_target_freshness_timeout_sec']
    controller['face_prediction_timeout_sec'] = \
        tracking_parameters['face_prediction_timeout_sec']
    return SimulationConfig(
        robot_parameters=robot, standard_servo=standard,
        collision_aware_servo=collision_aware, controller=controller,
        tracking=tracking_parameters,
        marker=tracking['face_target_marker']['ros__parameters'],
        state_publisher=tracking['robot_state_publisher']['ros__parameters'])
