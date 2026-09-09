# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validate the reactive tracking configuration contracts."""

import ast
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml


JOINT_NAMES = tuple(f'joint{index}' for index in range(1, 7))
ARM_LINK_NAMES = tuple(f'link{index}' for index in range(1, 7))
ABS_TOL = 1.0e-9
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def _interface_initial_value(joint, interface_name):
    interface = joint.find(f"./state_interface[@name='{interface_name}']")
    assert interface is not None
    initial_value = interface.find("./param[@name='initial_value']")
    assert initial_value is not None
    return float(initial_value.text)


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call, name):
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _string_literals(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


@pytest.fixture(scope='module')
def package_share():
    return Path(get_package_share_directory('face_tracking_arm'))


@pytest.fixture(scope='module')
def enabled_robot_description(package_share):
    xacro_path = package_share / 'description' / 'lite6_table.urdf.xacro'
    result = subprocess.run(
        ['xacro', str(xacro_path), 'enable_ros2_control:=true'],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture(scope='module')
def robot(enabled_robot_description):
    return ET.fromstring(enabled_robot_description)


@pytest.fixture(scope='module')
def srdf(package_share):
    return ET.parse(package_share / 'config' / 'moveit' / 'lite6.srdf').getroot()


@pytest.fixture(scope='module')
def controller_config(package_share):
    return _load_yaml(package_share / 'config' / 'control' / 'controllers.yaml')


@pytest.fixture(scope='module')
def initial_positions(package_share):
    config = _load_yaml(
        package_share / 'config' / 'control' / 'initial_positions.yaml'
    )
    return config['initial_positions']


@pytest.fixture(scope='module')
def servo_config(package_share):
    return _load_yaml(package_share / 'config' / 'moveit' / 'servo.yaml')


@pytest.fixture(scope='module')
def collision_aware_config(package_share):
    return _load_yaml(
        package_share / 'config' / 'moveit' / 'collision_aware_servo.yaml'
    )


@pytest.fixture(scope='module')
def tracking_config(package_share):
    config = _load_yaml(package_share / 'config' / 'tracking.yaml')
    return config['face_tracking_controller']['ros__parameters']


def test_tcp_transform_publication_fits_tracking_freshness_budget(
    package_share, controller_config
):
    config = _load_yaml(package_share / 'config' / 'tracking.yaml')
    frequency = config['robot_state_publisher']['ros__parameters'][
        'publish_frequency'
    ]
    control_frequency = controller_config['controller_manager'][
        'ros__parameters'
    ]['update_rate']
    source = (PROJECT_ROOT / 'src' / 'face_tracking_component.cpp').read_text()
    timeout = re.search(r'kTcpPoseTimeoutNs = ([0-9\']+);', source)
    assert timeout is not None
    timeout_sec = int(timeout.group(1).replace("'", '')) * 1.0e-9
    # Leave a complete publication interval for delivery/scheduling jitter.
    assert 2.0 / frequency < timeout_sec
    assert frequency >= control_frequency


def test_enabled_xacro_has_one_complete_gazebo_control_system(
    package_share, robot, initial_positions
):
    control_systems = robot.findall('./ros2_control')
    assert len(control_systems) == 1
    control_system = control_systems[0]
    assert control_system.attrib == {
        'name': 'Lite6GazeboSystem',
        'type': 'system',
    }
    hardware_plugin = control_system.find('./hardware/plugin')
    assert hardware_plugin is not None
    assert hardware_plugin.text == 'gz_ros2_control/GazeboSimSystem'

    joints = control_system.findall('./joint')
    assert tuple(joint.attrib['name'] for joint in joints) == JOINT_NAMES
    assert set(initial_positions) == set(JOINT_NAMES)
    for joint in joints:
        joint_name = joint.attrib['name']
        command_interfaces = joint.findall("./command_interface[@name='position']")
        assert len(command_interfaces) == 1
        assert len(joint.findall('./command_interface')) == 1
        assert {
            interface.attrib['name'] for interface in joint.findall('./state_interface')
        } == {'position', 'velocity'}
        assert math.isclose(
            _interface_initial_value(joint, 'position'),
            initial_positions[joint_name],
            rel_tol=0.0,
            abs_tol=ABS_TOL,
        )
        assert math.isclose(
            _interface_initial_value(joint, 'velocity'),
            0.0,
            rel_tol=0.0,
            abs_tol=ABS_TOL,
        )

    gazebo_plugins = robot.findall('./gazebo/plugin')
    control_plugins = [
        plugin
        for plugin in gazebo_plugins
        if plugin.attrib.get('name')
        == 'gz_ros2_control::GazeboSimROS2ControlPlugin'
    ]
    assert len(control_plugins) == 1
    control_plugin = control_plugins[0]
    assert control_plugin.attrib['filename'] == 'gz_ros2_control-system'
    parameters = control_plugin.find('./parameters')
    assert parameters is not None
    assert parameters.text is not None
    assert Path(parameters.text.strip()).resolve() == (
        package_share / 'config' / 'control' / 'controllers.yaml'
    ).resolve()
    position_gain = control_plugin.find('./position_proportional_gain')
    assert position_gain is not None
    assert math.isclose(
        float(position_gain.text), 0.5, rel_tol=0.0, abs_tol=ABS_TOL
    )


def test_controller_servo_and_initial_state_are_consistent(
    controller_config, initial_positions, servo_config, srdf
):
    manager = controller_config['controller_manager']['ros__parameters']
    controller_name = 'lite6_arm_controller'
    controller = controller_config[controller_name]['ros__parameters']

    assert manager['use_sim_time'] is True
    assert manager['update_rate'] == 100
    assert manager['enforce_command_limits'] is True
    assert manager['joint_state_broadcaster']['type'] == (
        'joint_state_broadcaster/JointStateBroadcaster'
    )
    assert manager[controller_name]['type'] == (
        'joint_trajectory_controller/JointTrajectoryController'
    )
    assert controller['use_sim_time'] is True
    assert tuple(controller['joints']) == JOINT_NAMES
    assert controller['command_interfaces'] == ['position']
    assert controller['state_interfaces'] == ['position', 'velocity']
    assert controller['allow_partial_joints_goal'] is False
    assert math.isclose(controller['state_publish_rate'], 100.0)
    assert controller['allow_nonzero_velocity_at_trajectory_end'] is True
    assert controller['interpolate_from_desired_state'] is True
    assert controller['interpolation_method'] == 'splines'

    assert servo_config['command_out_topic'] == (
        f'/{controller_name}/joint_trajectory'
    )
    assert servo_config['command_out_type'] == 'trajectory_msgs/JointTrajectory'
    assert servo_config['joint_topic'] == '/joint_states'
    assert servo_config['publish_joint_positions'] is True
    assert servo_config['publish_joint_velocities'] is True
    assert servo_config['publish_joint_accelerations'] is True
    assert math.isclose(
        servo_config['publish_period'],
        1.0 / manager['update_rate'],
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert servo_config['max_expected_latency'] >= 3.0 * servo_config[
        'publish_period'
    ]
    assert servo_config['is_primary_planning_scene_monitor'] is True
    assert servo_config['check_collisions'] is True
    assert math.isclose(
        servo_config['collision_check_rate'],
        10.0,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert servo_config['smoothing_filter_plugin_name'] == (
        'online_signal_smoothing::AccelerationLimitedPlugin'
    )

    rest_state = srdf.find("./group_state[@name='rest'][@group='lite6_arm']")
    assert rest_state is not None
    srdf_positions = {
        joint.attrib['name']: float(joint.attrib['value'])
        for joint in rest_state.findall('./joint')
    }
    assert set(srdf_positions) == set(JOINT_NAMES)
    for joint_name in JOINT_NAMES:
        assert math.isclose(
            srdf_positions[joint_name],
            initial_positions[joint_name],
            rel_tol=0.0,
            abs_tol=ABS_TOL,
        )


def test_moveit_and_tracking_frames_and_limits_are_consistent(
    package_share, servo_config, tracking_config, srdf
):
    group = srdf.find("./group[@name='lite6_arm']")
    assert group is not None
    chain = group.find('./chain')
    assert chain is not None
    assert chain.attrib == {
        'base_link': 'link_base',
        'tip_link': 'monitor_control_frame',
    }

    assert servo_config['move_group_name'] == group.attrib['name']
    assert tracking_config['planning_frame'] == 'world'
    assert tracking_config['base_frame'] == chain.attrib['base_link']
    assert tracking_config['control_frame'] == chain.attrib['tip_link']
    assert 'monitor_face_distance_m' not in tracking_config
    assert math.isclose(
        tracking_config['minimum_face_distance_m'],
        0.40,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert math.isclose(
        tracking_config['safe_reach_radius_m'],
        0.42,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert tracking_config['reach_center_frame'] == 'link1'
    assert math.isclose(
        tracking_config['face_target_freshness_timeout_sec'],
        0.50,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert math.isclose(
        tracking_config['return_to_rest_delay_sec'],
        2.0,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    )
    assert tracking_config['return_to_rest_delay_sec'] > tracking_config[
        'face_target_freshness_timeout_sec'
    ]
    assert tracking_config['rest_position_m'] == [0.20, 0.00, 1.05]

    kinematics = _load_yaml(
        package_share / 'config' / 'moveit' / 'kinematics.yaml'
    )
    assert set(kinematics) == {group.attrib['name']}
    assert kinematics[group.attrib['name']]['kinematics_solver'] == (
        'kdl_kinematics_plugin/KDLKinematicsPlugin'
    )

    joint_limits = _load_yaml(
        package_share / 'config' / 'moveit' / 'joint_limits.yaml'
    )['joint_limits']
    assert set(joint_limits) == set(JOINT_NAMES)
    for limits in joint_limits.values():
        assert limits['has_velocity_limits'] is True
        assert limits['has_acceleration_limits'] is True
        assert limits['has_jerk_limits'] is True
        assert 0.0 < limits['max_velocity'] <= 1.0
        assert 0.0 < limits['max_acceleration'] <= 2.0
        assert 0.0 < limits['max_jerk'] <= 20.0


def test_collision_aware_servo_safety_profile_is_consistent(
    collision_aware_config, servo_config
):
    config = collision_aware_config
    assert math.isclose(
        config['control_period_sec'], config['trajectory_controller_period_sec'],
        abs_tol=ABS_TOL,
    )
    assert math.isclose(
        config['trajectory_controller_period_sec'], 0.01, abs_tol=ABS_TOL
    )
    assert math.isclose(servo_config['publish_period'], 0.01, abs_tol=ABS_TOL)
    assert config['planning_group_name'] == servo_config['move_group_name']
    assert config['command_frame'] == 'monitor_control_frame'
    assert math.isclose(config['position_deadband_m'], 0.03, abs_tol=ABS_TOL)
    # Small head motions must exceed the angular quiet zone independently of
    # the deliberately loose position tolerance.
    assert 0.0 < config['pointing_deadband_rad'] <= math.radians(2.0)
    assert math.isclose(
        config['primary_orientation_tolerance_radps'], 0.002,
        abs_tol=ABS_TOL,
    )
    assert math.isclose(config['hard_clearance_m'], 0.015, abs_tol=ABS_TOL)
    assert math.isclose(
        config['collision_query_distance_m'], 0.200, abs_tol=ABS_TOL
    )
    assert config['maximum_collision_constraints'] >= 24
    assert config['collision_gradient_epsilon'] <= 1.0e-8
    assert config['monitor_guard_joint_name'] == 'joint5'
    assert config['monitor_guard_min_position_rad'] >= -1.60
    assert config['monitor_guard_max_position_rad'] <= 1.60
    assert config['joint_position_margin_rad'] >= 0.10
    assert config['residual_command_latency_sec'] >= 0.020
    assert 0.0 < config['state_feedback_timeout_sec'] <= 0.040
    assert 0.006 < config['following_position_tolerance_rad'] <= 0.05
    assert 0.0 < config['following_velocity_tolerance_rad_s'] <= 0.30
    assert math.isclose(
        config['collision_tracking_error_bound_rad'], 0.006, abs_tol=ABS_TOL
    )
    assert config['collision_tracking_error_bound_rad'] <= config[
        'following_position_tolerance_rad'
    ]
    assert config['numerical_distance_reserve_m'] >= 0.0001
    assert config['default_distance_lipschitz_m_per_rad'] >= 2.6193
    assert config['monitor_near_distance_lipschitz_m_per_rad'] >= 0.4543
    assert config['solver_time_limit_sec'] < config['control_period_sec']
    assert 3 <= config['segment_validation_substeps'] <= 16
    assert config['segment_validation_distance_m'] >= config['hard_clearance_m']
    assert config['segment_validation_distance_m'] <= config[
        'collision_query_distance_m'
    ]
    assert 'maximum_consecutive_solver_failures' not in config
    assert 'collision_release_hysteresis_m' not in config
    assert 'collision_influence_distance_m' not in config
    assert 'collision_repulsion_gain' not in config
    assert 'maximum_repulsion_speed_mps' not in config


def test_custom_backend_has_explicit_mode_and_background_planning(collision_aware_config):
    assert collision_aware_config['use_tracking_target'] is True
    planner = collision_aware_config['recovery_planner']
    assert planner['planner_configs']['RRTConnectkConfigDefault']['type'] == (
        'geometric::RRTConnect'
    )
    assert planner['lite6_arm']['longest_valid_segment_fraction'] <= 0.005


def test_collision_aware_backend_keeps_braking_and_segment_safety_contracts():
    source_paths = sorted((PROJECT_ROOT / 'src').glob('collision_aware_servo_*.cpp'))
    source_paths.extend([
        PROJECT_ROOT / 'src' / 'collision_aware_servo_component.hpp',
        PROJECT_ROOT / 'src' / 'collision_aware_parameters.cpp',
    ])
    source = '\n'.join(path.read_text(encoding='utf-8') for path in source_paths)
    cmake = (PROJECT_ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
    package_xml = (PROJECT_ROOT / 'package.xml').read_text(encoding='utf-8')

    assert 'shouldValidateSegment(' in source
    assert 'collision_constraint_builder_->validateSegment(' in source
    assert 'CommandAttempt::kTracking' in source
    assert 'CommandAttempt::kBraking' in source
    assert 'enqueue_and_publish(' in source
    assert 'std::move(*accepted_command)' in source
    assert 'safe jerk-limited braking command published' in source
    assert 'braking next tick' not in source
    assert re.search(
        r'void CollisionAwareServoComponent::enter_latched_halt\(std::string message\).*'
        r'command_queue_\.clear\(\).*HALT_FOR_COLLISION',
        source,
        re.DOTALL,
    )
    assert 'minimum_observed_self_distance_m' in source
    assert 'minimum_observed_world_distance_m' in source
    assert 'screen_normal_pointing_error_p99_rad' in source
    assert 'segment_validation_count' in source
    assert 'segment_rejection_count' in source
    assert 'Fresh complete finite robot feedback is unavailable' in source
    assert 'Waiting for the first complete finite robot state' in source
    assert 'controller_mode_ == ControllerMode::kLatchedHalt' in source
    assert 'publish_emergency_hold(' not in source
    assert 'hold.points.resize(1)' not in source
    assert 'EmergencyBrakeTailGenerator' in source
    assert 'PublishedTrajectoryHistory' in source
    assert 'active_publication_id_' in source
    assert 'correctedSampleTimes(' in source
    assert 'PublicationMatchKind::kOlder' in source
    assert 'pre-published braking tail' in source
    assert 'replacement_commit_time' in source
    assert 'minimum_replacement_lead_sec' in source
    assert 'braking_tail_collision_rejection_count' in source
    assert 'expected_command_state_at(' in source
    assert 'Trajectory following error:' in source
    assert 'validateActualState(' in source
    assert 'Measured robot state is unsafe:' in source
    assert 'controller_state->speed_scaling_factor - 1.0' in source
    assert 'command_epoch_has_published_' in source
    assert 'controller_state->reference.positions' in source
    assert 'controller_state->output.positions' in source
    assert 'command_alignment_failure_count' in source
    assert 'enter_controlled_rearm(' in source
    assert 'controlled_rearm_count' in source
    assert 'last_controlled_rearm_reason' in source
    assert 'enter_startup_retry(' in source
    assert 'startup_retry_count' in source
    assert 'last_startup_retry_reason' in source
    assert 'safe_rearm_ready(' in source
    assert 'kRequiredStableRearmSamples' in source
    assert 'PlanningSceneMonitor::UPDATE_GEOMETRY' in source
    assert 'PlanningScene collision geometry changed while commands were buffered' in source
    assert 'Command buffer fell below the minimum safe lead' in source
    assert 'maximum_collision_constraints > kMaximumCollisionConstraintRows' in source
    assert 'parameters.segment_validation_substeps >' in source
    assert 'maximum_consecutive_solver_failures' not in source
    assert 'collision_release_hysteresis_m' not in source
    assert 'realtime_tools::configure_sched_fifo(' in source
    assert 'kRequiredStartupWarmupCycles' in source
    assert 'Priming collision and solver caches' in source
    assert 'Stationary checked command timeline primed' in source
    assert 'Safe re-arm is monitoring the pre-published braking tail' in source
    assert 'published_tail_rearm_pending_ ?' in source
    assert 'TimelineRecoveryStatus::kExhaustedPublishedSuffix' in source
    assert 'TimelineRecoveryStatus::kInvariantFailure' in source
    assert 'is waiting for an unmatched controller phase to settle' in source
    assert 'controller sample left the published braking-tail horizon' in source
    assert 'find_package(realtime_tools REQUIRED)' in cmake
    assert '<depend>realtime_tools</depend>' in package_xml


def test_monitor_collision_exemption_is_limited_to_direct_mount(srdf):
    disabled_pairs = {
        frozenset((entry.attrib['link1'], entry.attrib['link2']))
        for entry in srdf.findall('./disable_collisions')
    }
    monitor_pairs = {
        pair for pair in disabled_pairs if 'monitor_link' in pair
    }
    assert monitor_pairs == {frozenset(('link6', 'monitor_link'))}
    for link_name in ARM_LINK_NAMES[:-1]:
        assert frozenset((link_name, 'monitor_link')) not in disabled_pairs


def test_launch_keeps_headless_server_and_excludes_move_group_and_rviz(package_share):
    launch_path = package_share / 'launch' / 'lite6_table.launch.py'
    syntax_tree = ast.parse(launch_path.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Call)]

    headless_arguments = [
        call
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'headless'
    ]
    assert len(headless_arguments) == 1

    server_processes = []
    gui_processes = []
    for call in calls:
        if _call_name(call) != 'ExecuteProcess':
            continue
        cmd = _keyword(call, 'cmd')
        condition = _keyword(call, 'condition')
        if cmd is None or condition is None or not isinstance(condition, ast.Call):
            continue
        command_literals = _string_literals(cmd)
        if {'gz', 'sim', '-r', '-s'}.issubset(command_literals):
            server_processes.append((_call_name(condition), command_literals))
        elif {'gz', 'sim', '-r'}.issubset(command_literals):
            gui_processes.append((_call_name(condition), command_literals))

    assert len(server_processes) == 1
    assert server_processes[0][0] == 'IfCondition'
    assert len(gui_processes) == 1
    assert gui_processes[0][0] == 'UnlessCondition'
    assert '-s' not in gui_processes[0][1]

    forbidden_runtime_names = {
        'move_group',
        'moveit_ros_move_group',
        'rviz',
        'rviz2',
    }
    runtime_literals = set()
    for call in calls:
        if _call_name(call) not in {
            'Node',
            'ComposableNode',
            'ComposableNodeContainer',
            'ExecuteProcess',
        }:
            continue
        runtime_literals.update(_string_literals(call))
    assert runtime_literals.isdisjoint(forbidden_runtime_names)


def test_tracking_launch_defaults_to_gui_and_supports_headless(package_share):
    launch_path = package_share / 'launch' / 'tracking_sim.launch.py'
    launch_source = launch_path.read_text(encoding='utf-8')
    syntax_tree = ast.parse(launch_source)
    calls = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Call)]

    runtime_literals = set()
    for call in calls:
        if _call_name(call) in {
            'Node',
            'ComposableNode',
            'ComposableNodeContainer',
            'ExecuteProcess',
        }:
            runtime_literals.update(_string_literals(call))

    gazebo_server_processes = []
    gazebo_gui_processes = []
    for call in calls:
        if _call_name(call) != 'ExecuteProcess':
            continue
        cmd = _keyword(call, 'cmd')
        condition = _keyword(call, 'condition')
        if cmd is None or condition is None or not isinstance(condition, ast.Call):
            continue
        command_literals = _string_literals(cmd)
        if {'gz', 'sim', '-r', '-s'}.issubset(command_literals):
            gazebo_server_processes.append(_call_name(condition))
        elif {'gz', 'sim', '-r'}.issubset(command_literals):
            gazebo_gui_processes.append(_call_name(condition))

    assert gazebo_server_processes == ['IfCondition']
    assert gazebo_gui_processes == ['UnlessCondition']
    assert 'graceful_component_container' in runtime_literals
    assert 'moveit_servo::ServoNode' in runtime_literals
    assert 'face_tracking_arm::CollisionAwareServoComponent' in runtime_literals
    assert 'face_tracking_arm::FaceTrackingComponent' in runtime_literals
    assert 'test_face_target_publisher' in runtime_literals
    assert 'mock_target' not in launch_source
    assert 'mock_pattern' not in launch_source
    assert 'mock_face_target' not in launch_source
    assert (
        'collision_aware_moveit_servo_parameters'
        "['publish_joint_positions'] = True" in launch_source
    )
    assert (
        "collision_aware_moveit_servo_parameters['publish_period'] = 0.01"
        in launch_source
    )
    assert (
        "collision_aware_moveit_servo_parameters['max_expected_latency'] = 0.05"
        in launch_source
    )
    assert (
        'collision_aware_moveit_servo_parameters'
        "['publish_joint_velocities'] = False" in launch_source
    )
    assert "'publish_joint_accelerations'" in launch_source
    assert runtime_literals.isdisjoint(
        {'move_group', 'moveit_ros_move_group', 'rviz', 'rviz2'}
    )

    startup_edges = [call for call in calls if _call_name(call) == '_after_success']
    assert len(startup_edges) == 3

    declared_arguments = {
        call.args[0].value
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert 'use_sim_time' not in declared_arguments
    assert 'headless' in declared_arguments
    assert 'test_face_scenario' in declared_arguments
    assert 'servo_backend' in declared_arguments

    headless_arguments = [
        call
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'headless'
    ]
    assert len(headless_arguments) == 1
    headless_default = _keyword(headless_arguments[0], 'default_value')
    assert isinstance(headless_default, ast.Constant)
    assert headless_default.value == 'false'

    scenario_arguments = [
        call
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'test_face_scenario'
    ]
    assert len(scenario_arguments) == 1
    scenario_default = _keyword(scenario_arguments[0], 'default_value')
    assert isinstance(scenario_default, ast.Constant)
    assert scenario_default.value == 'people'
    scenario_choices = _keyword(scenario_arguments[0], 'choices')
    assert isinstance(scenario_choices, ast.List)
    assert [choice.value for choice in scenario_choices.elts] == [
        'disabled',
        'stationary',
        'circle',
        'people',
        'walk_around',
    ]

    backend_arguments = [
        call
        for call in calls
        if _call_name(call) == 'DeclareLaunchArgument'
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'servo_backend'
    ]
    assert len(backend_arguments) == 1
    backend_default = _keyword(backend_arguments[0], 'default_value')
    assert isinstance(backend_default, ast.Constant)
    assert backend_default.value == 'collision_aware'
    backend_choices = _keyword(backend_arguments[0], 'choices')
    assert isinstance(backend_choices, ast.List)
    assert [choice.value for choice in backend_choices.elts] == [
        'collision_aware',
        'standard',
    ]

    test_publishers = [
        call
        for call in calls
        if _call_name(call) == 'Node'
        and 'test_face_target_publisher' in _string_literals(call)
    ]
    assert len(test_publishers) == 1
    publisher = test_publishers[0]
    condition = _keyword(publisher, 'condition')
    assert isinstance(condition, ast.Call)
    assert _call_name(condition) == 'UnlessCondition'
    assert any(
        'disabled' in literal for literal in _string_literals(condition)
    )
    parameters = _keyword(publisher, 'parameters')
    assert parameters is not None
    assert {'scenario', 'use_sim_time'}.issubset(_string_literals(parameters))

    runtime_guards = [
        call for call in calls if _call_name(call) == '_shutdown_on_exit'
    ]
    assert len(runtime_guards) == 5

    components = [call for call in calls if _call_name(call) == 'ComposableNode']
    assert len(components) == 3
    for component in components:
        extra_arguments = _keyword(component, 'extra_arguments')
        assert extra_arguments is not None
        assert 'use_intra_process_comms' in _string_literals(extra_arguments)

    containers = [
        call for call in calls if _call_name(call) == 'ComposableNodeContainer'
    ]
    assert len(containers) == 1
    initially_loaded = _keyword(
        containers[0], 'composable_node_descriptions'
    )
    assert isinstance(initially_loaded, ast.List)
    assert initially_loaded.elts == []

    load_actions = [
        call for call in calls if _call_name(call) == 'LoadComposableNodes'
    ]
    assert len(load_actions) == 2
    delayed_component_sets = []
    for load_action in load_actions:
        target_container = _keyword(load_action, 'target_container')
        assert isinstance(target_container, ast.Name)
        assert target_container.id == 'servo_container'
        condition = _keyword(load_action, 'condition')
        assert isinstance(condition, ast.Call)
        assert _call_name(condition) == 'IfCondition'
        delayed_components = _keyword(
            load_action, 'composable_node_descriptions'
        )
        assert isinstance(delayed_components, ast.List)
        delayed_component_sets.append(
            [element.id for element in delayed_components.elts]
        )
    assert delayed_component_sets == [
        ['tracking_component', 'collision_aware_servo_component'],
        ['tracking_component', 'standard_servo_component'],
    ]


def test_test_face_target_publisher_preserves_ros_contract():
    source_path = PROJECT_ROOT / 'src' / 'test_face_target_publisher_node.cpp'
    source = source_path.read_text(encoding='utf-8')

    assert 'Node("test_face_target_publisher")' in source
    assert 'declare_parameter<std::string>("scenario", "people")' in source
    assert '"/face/center", rclcpp::QoS(1).best_effort()' in source
    assert '"/tracking/diagnostics", rclcpp::QoS(1).reliable()' in source
    assert 'reports_ready(status)' in source
    assert 'std::chrono::milliseconds(20)' in source
    assert 'target.header.stamp = sample_time' in source
    assert 'target.header.frame_id = "world"' in source
    assert 'test_targets::sample(scenario_, elapsed)' in source
    assert '!last_phase_ || *last_phase_ != sample.phase' in source


def test_tracking_readiness_rejects_dangerous_servo_status():
    adapter_source = (
        PROJECT_ROOT / 'src' / 'face_tracking_component.cpp'
    ).read_text(encoding='utf-8')
    publisher_source = (
        PROJECT_ROOT / 'src' / 'test_face_target_publisher_node.cpp'
    ).read_text(encoding='utf-8')

    assert 'bool infrastructure_ready(const std::int64_t now_ns) const' in (
        adapter_source
    )
    assert 'bool motion_ready(const std::int64_t now_ns) const' in adapter_source
    assert re.search(
        r'bool motion_ready\(const std::int64_t now_ns\) const\s*\{\s*'
        r'return infrastructure_ready\(now_ns\) && '
        r'!is_dangerous_servo_status\(last_servo_code_\);',
        adapter_source,
    )
    assert re.search(
        r'const bool infrastructure_ready_now = infrastructure_ready\(now_ns\);\s*'
        r'ready_ = motion_ready\(now_ns\);\s*'
        r'if \(!infrastructure_ready_now\)',
        adapter_source,
    )
    assert '"infrastructure_ready", bool_string(infrastructure_ready_now)' in (
        adapter_source
    )
    assert (
        'status.level != diagnostic_msgs::msg::DiagnosticStatus::OK'
        in publisher_source
    )
