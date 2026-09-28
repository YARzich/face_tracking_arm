# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run reactive tracking with model-specific startup poses in Gazebo Sim."""

from functools import partial
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
import xacro
import yaml


WORLD_NAME = 'lite6_table'


def _load_yaml(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def _actions_after_successful_exit(
    event, context, *, actions, prerequisite_name
):
    """Start dependants only after a successful one-shot prerequisite."""
    if context.is_shutdown:
        return []
    if event.returncode == 0:
        return list(actions)
    return [
        EmitEvent(
            event=Shutdown(
                reason=(
                    f"Prerequisite '{prerequisite_name}' failed with exit "
                    f'code {event.returncode}'
                )
            )
        )
    ]


def _after_success(target_action, actions, prerequisite_name):
    """Build one fail-closed startup edge."""
    return RegisterEventHandler(
        OnProcessExit(
            target_action=target_action,
            on_exit=partial(
                _actions_after_successful_exit,
                actions=tuple(actions),
                prerequisite_name=prerequisite_name,
            ),
        )
    )


def _shutdown_after_unexpected_exit(event, context, *, process_name):
    """Stop the scenario if a long-running process disappears."""
    if context.is_shutdown:
        return []
    return [
        EmitEvent(
            event=Shutdown(
                reason=(
                    f"Required process '{process_name}' exited with code "
                    f'{event.returncode}'
                )
            )
        )
    ]


def _shutdown_on_exit(target_action, process_name):
    """Build a fail-closed runtime process guard."""
    return RegisterEventHandler(
        OnProcessExit(
            target_action=target_action,
            on_exit=partial(
                _shutdown_after_unexpected_exit,
                process_name=process_name,
            ),
        )
    )


def _launch_setup(context):
    """Create Gazebo, controllers, and the continuously running Servo loop."""
    from face_tracking_bringup.mount_collisions import allow_fixed_mount_collisions
    from face_tracking_bringup.robot_profile import get_robot_profile
    from face_tracking_bringup.simulation_start import resolve_simulation_start

    package_share = Path(get_package_share_directory('face_tracking_arm'))
    headless = LaunchConfiguration('headless')
    servo_backend = LaunchConfiguration('servo_backend')
    test_face_scenario = LaunchConfiguration('test_face_scenario')
    smoothing_plugin = LaunchConfiguration('smoothing_plugin')
    visualize_target = LaunchConfiguration('visualize_target')
    gz_ros2_control_library_path = str(
        Path(get_package_prefix('gz_ros2_control')) / 'lib'
    )
    existing_system_plugin_path = os.environ.get(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', ''
    )
    system_plugin_path = os.pathsep.join(
        path
        for path in (
            gz_ros2_control_library_path,
            existing_system_plugin_path,
        )
        if path
    )

    robot_model = LaunchConfiguration('robot_model').perform(context)
    robot_profile = get_robot_profile(robot_model)
    planning_group = robot_profile['planning_group']
    initial_positions_path = resolve_simulation_start(
        package_share,
        robot_model=robot_model,
        start_pose=LaunchConfiguration('start_pose').perform(context),
        initial_positions_file=LaunchConfiguration('initial_positions_file').perform(context),
    )
    controller_config_path = (
        package_share / 'config' / 'control' / 'controllers.yaml'
    )
    camera_mode = LaunchConfiguration('camera_mode').perform(context)
    camera_width = int(LaunchConfiguration('camera_width').perform(context))
    camera_height = int(LaunchConfiguration('camera_height').perform(context))
    camera_rate = float(LaunchConfiguration('camera_rate').perform(context))
    camera_fov = float(LaunchConfiguration('camera_horizontal_fov').perform(context))
    if (camera_width <= 0 or camera_height <= 0 or not math.isfinite(camera_rate)
            or camera_rate <= 0 or not 0 < camera_fov < math.pi):
        raise ValueError(
            'Camera dimensions/rate must be positive; horizontal FOV is radians (0, pi)')
    idle_behavior = LaunchConfiguration('idle_behavior').perform(context)
    if idle_behavior != 'rest' and servo_backend.perform(context) != 'collision_aware':
        raise ValueError('Search requires the collision_aware backend')
    camera_baseline = float(LaunchConfiguration('camera_baseline').perform(context))
    if not 0.02 <= camera_baseline <= 0.25:
        raise ValueError('camera_baseline must be 0.02..0.25 meters')
    world_path = LaunchConfiguration('world_file').perform(context)
    robot_description_xml = xacro.process_file(
        str(package_share / 'description' / f'{robot_model}_table.urdf.xacro'),
        mappings={
            'enable_ros2_control': 'true',
            'camera_mode': camera_mode,
            'camera_baseline': str(camera_baseline),
            'camera_width': str(camera_width),
            'camera_height': str(camera_height),
            'camera_rate': str(camera_rate),
            'camera_horizontal_fov': str(camera_fov),
            'initial_positions_file': str(initial_positions_path),
            'controller_config_file': str(controller_config_path),
        },
    ).toxml()
    robot_description = {'robot_description': robot_description_xml}
    semantic = ET.parse(package_share / robot_profile['srdf_file']).getroot()
    allow_fixed_mount_collisions(ET.fromstring(robot_description_xml), semantic)
    robot_description_semantic = {
        'robot_description_semantic': ET.tostring(semantic, encoding='unicode')}
    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml(
            package_share / 'config' / 'moveit' / 'kinematics.yaml'
        )
    }
    joint_limits_config = _load_yaml(
        package_share / 'config' / 'moveit' / 'joint_limits.yaml'
    )
    robot_description_planning = {
        'robot_description_planning': joint_limits_config
    }
    standard_servo_parameters = _load_yaml(
        package_share / 'config' / 'moveit' / 'servo.yaml'
    )
    standard_servo_parameters['move_group_name'] = planning_group
    standard_servo_parameters['smoothing_filter_plugin_name'] = ParameterValue(
        smoothing_plugin, value_type=str
    )
    collision_aware_moveit_servo_parameters = dict(
        standard_servo_parameters
    )
    # Match the command and JTC grids to preserve discrete acceleration limits.
    # Build with optimization to keep FCL/QP within the 10 ms control budget.
    collision_aware_moveit_servo_parameters['publish_period'] = 0.01
    collision_aware_moveit_servo_parameters['max_expected_latency'] = 0.05
    collision_aware_moveit_servo_parameters['check_collisions'] = False
    collision_aware_moveit_servo_parameters['use_smoothing'] = False
    # Keep derivatives inside the private QP state. Position-only output makes
    # JointTrajectoryController use linear interpolation along the exact joint-
    # space segments checked by the collision validator.
    collision_aware_moveit_servo_parameters['publish_joint_positions'] = True
    collision_aware_moveit_servo_parameters['publish_joint_velocities'] = False
    collision_aware_moveit_servo_parameters[
        'publish_joint_accelerations'
    ] = False
    collision_aware_parameters = _load_yaml(
        package_share / 'config' / 'moveit' / 'collision_aware_servo.yaml'
    )
    collision_aware_parameters['planning_group_name'] = planning_group
    recovery_planner = collision_aware_parameters['recovery_planner']
    recovery_group = recovery_planner.pop('xarm6')
    recovery_planner[planning_group] = recovery_group
    if camera_mode != 'disabled':
        collision_aware_parameters['gaze_frame'] = 'face_test_camera_optical_frame'
    ordered_joint_limits = joint_limits_config['joint_limits']
    joint_names = list(ordered_joint_limits)
    collision_aware_parameters.update(
        {
            'joint_names': joint_names,
            'max_joint_velocity_rad_s': [
                ordered_joint_limits[name]['max_velocity']
                for name in joint_names
            ],
            'max_joint_acceleration_rad_s2': [
                ordered_joint_limits[name]['max_acceleration']
                for name in joint_names
            ],
            'max_joint_jerk_rad_s3': [
                ordered_joint_limits[name]['max_jerk']
                for name in joint_names
            ],
        }
    )
    tracking_config = _load_yaml(package_share / 'config' / 'tracking.yaml')
    tracking_parameters = tracking_config['face_tracking_controller'][
        'ros__parameters'
    ]
    marker_parameters = tracking_config['face_target_marker']['ros__parameters']
    state_publisher_parameters = tracking_config['robot_state_publisher'][
        'ros__parameters'
    ]

    gazebo_server_process = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            '-s',
            '--headless-rendering',
            world_path,
        ],
        additional_env={'GZ_SIM_SYSTEM_PLUGIN_PATH': system_plugin_path},
        condition=IfCondition(headless),
        output='screen',
    )
    gazebo_gui_process = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            world_path,
        ],
        additional_env={'GZ_SIM_SYSTEM_PLUGIN_PATH': system_plugin_path},
        condition=UnlessCondition(headless),
        output='screen',
    )
    clock_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        parameters=[{'use_sim_time': True}],
        arguments=[
            f'/world/{WORLD_NAME}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            (
                f'/world/{WORLD_NAME}/set_pose/blocking'
                '@ros_gz_interfaces/srv/SetEntityPose'
            ),
        ],
        remappings=[(f'/world/{WORLD_NAME}/clock', '/clock')],
        output='screen',
    )
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            robot_description, state_publisher_parameters, {'use_sim_time': True}
        ],
        output='screen',
    )
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_lite6',
        arguments=[
            '-world',
            WORLD_NAME,
            '-topic',
            '/robot_description',
            '-name',
            'lite6',
            '-allow_renaming',
            'false',
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='joint_state_broadcaster_spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='lite6_arm_controller_spawner',
        arguments=[
            'lite6_arm_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '120',
        ],
        output='screen',
    )

    standard_servo_component = ComposableNode(
        package='moveit_servo',
        plugin='moveit_servo::ServoNode',
        name='servo_node',
        parameters=[
            {'moveit_servo': standard_servo_parameters},
            {
                'update_period': 0.01,
                'planning_group_name': planning_group,
                'use_sim_time': True,
            },
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
        ],
        extra_arguments=[{'use_intra_process_comms': True}],
    )
    collision_aware_servo_component = ComposableNode(
        package='face_tracking_arm',
        plugin='face_tracking_arm::CollisionAwareServoComponent',
        name='servo_node',
        parameters=[
            {'moveit_servo': collision_aware_moveit_servo_parameters},
            collision_aware_parameters,
            {'use_sim_time': True},
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
        ],
        extra_arguments=[{'use_intra_process_comms': True}],
    )
    tracking_component = ComposableNode(
        package='face_tracking_arm',
        plugin='face_tracking_arm::FaceTrackingComponent',
        name='face_tracking_controller',
        parameters=[tracking_parameters, {'use_sim_time': True, 'idle_behavior': idle_behavior}],
        extra_arguments=[{'use_intra_process_comms': True}],
    )
    servo_container = ComposableNodeContainer(
        name='face_tracking_servo_container',
        namespace='/',
        package='face_tracking_arm',
        executable='graceful_component_container',
        composable_node_descriptions=[],
        output='screen',
    )
    load_collision_aware_servo_components = LoadComposableNodes(
        target_container=servo_container,
        # Load the fast adapter first. Servo performs substantial construction
        # before replying to the component service on MoveIt 2.12.4.
        composable_node_descriptions=[
            tracking_component,
            collision_aware_servo_component,
        ],
        condition=IfCondition(
            PythonExpression(["'", servo_backend, "' == 'collision_aware'"])
        ),
    )
    load_standard_servo_components = LoadComposableNodes(
        target_container=servo_container,
        composable_node_descriptions=[
            tracking_component,
            standard_servo_component,
        ],
        condition=IfCondition(
            PythonExpression(["'", servo_backend, "' == 'standard'"])
        ),
    )
    test_face_target_publisher_node = Node(
        package='face_tracking_arm',
        executable='test_face_target_publisher',
        name='test_face_target_publisher',
        parameters=[
            {'scenario': test_face_scenario, 'use_sim_time': True},
        ],
        condition=UnlessCondition(
            PythonExpression(["'", test_face_scenario, "' == 'disabled'"])
        ),
        output='screen',
    )
    target_marker_node = Node(
        package='face_tracking_arm',
        executable='face_target_marker',
        name='face_target_marker',
        parameters=[marker_parameters, {'use_sim_time': True}],
        condition=IfCondition(visualize_target),
        output='screen',
    )

    start_joint_state_broadcaster = _after_success(
        spawn_robot_node,
        [joint_state_broadcaster_spawner],
        'spawn_lite6',
    )
    start_arm_controller = _after_success(
        joint_state_broadcaster_spawner,
        [arm_controller_spawner],
        'joint_state_broadcaster',
    )
    start_servo_loop = _after_success(
        arm_controller_spawner,
        [
            load_collision_aware_servo_components,
            load_standard_servo_components,
            test_face_target_publisher_node,
        ],
        'lite6_arm_controller',
    )
    guard_gazebo_server = _shutdown_on_exit(gazebo_server_process, 'gz sim')
    guard_gazebo_gui = _shutdown_on_exit(gazebo_gui_process, 'gz sim GUI')
    guard_clock_bridge = _shutdown_on_exit(clock_bridge_node, 'clock bridge')
    guard_robot_state_publisher = _shutdown_on_exit(
        robot_state_publisher_node, 'robot state publisher'
    )
    guard_servo_container = _shutdown_on_exit(
        servo_container, 'Servo component container'
    )

    launch_description = LaunchDescription()
    launch_description.add_action(guard_gazebo_server)
    launch_description.add_action(guard_gazebo_gui)
    launch_description.add_action(guard_clock_bridge)
    launch_description.add_action(guard_robot_state_publisher)
    launch_description.add_action(guard_servo_container)
    launch_description.add_action(gazebo_server_process)
    launch_description.add_action(gazebo_gui_process)
    launch_description.add_action(clock_bridge_node)
    launch_description.add_action(servo_container)
    launch_description.add_action(target_marker_node)
    launch_description.add_action(robot_state_publisher_node)
    launch_description.add_action(spawn_robot_node)
    launch_description.add_action(start_joint_state_broadcaster)
    launch_description.add_action(start_arm_controller)
    launch_description.add_action(start_servo_loop)
    return list(launch_description.entities)


def generate_launch_description() -> LaunchDescription:
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    sys.path.insert(0, str(scripts))
    from face_tracking_bringup.simulation_start import START_POSES

    headless_argument = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run only the Gazebo server without its graphical client.',
    )
    servo_backend_argument = DeclareLaunchArgument(
        'servo_backend',
        default_value='collision_aware',
        choices=['collision_aware', 'standard'],
        description='Select the collision-aware controller or stock MoveIt Servo.',
    )
    test_face_scenario_argument = DeclareLaunchArgument(
        'test_face_scenario',
        default_value='people',
        choices=['disabled', 'stationary', 'circle', 'people', 'walk_around'],
        description='Deterministic test source published after tracking readiness.',
    )
    visualize_target_argument = DeclareLaunchArgument(
        'visualize_target',
        default_value='true',
        description='Show the latest fresh /face/center point in Gazebo.',
    )
    smoothing_plugin_argument = DeclareLaunchArgument(
        'smoothing_plugin',
        default_value='online_signal_smoothing::AccelerationLimitedPlugin',
        choices=[
            'online_signal_smoothing::RuckigFilterPlugin',
            'online_signal_smoothing::AccelerationLimitedPlugin',
        ],
        description='Servo smoother selected once for the entire process.',
    )

    camera_mode_argument = DeclareLaunchArgument(
        'camera_mode', default_value='disabled', choices=['disabled', 'mono_cpu', 'stereo'],
        description='Optional monitor-mounted RGB sensor assembly.')
    camera_baseline_argument = DeclareLaunchArgument(
        'camera_baseline', default_value='0.08', description='Stereo baseline in meters.')
    camera_arguments = [DeclareLaunchArgument(name, default_value=default, description=description)
                        for name, default, description in (
                            ('camera_width', '1280', 'Image width in pixels.'),
                            ('camera_height', '720', 'Image height in pixels.'),
                            ('camera_rate', '30', 'Camera frames per second.'),
                            ('camera_horizontal_fov', '1.433', 'Horizontal FOV in radians.'))]
    idle_argument = DeclareLaunchArgument(
        'idle_behavior', default_value='rest',
        choices=['rest', 'search_sweep', 'search_local_then_sweep'],
        description='Behavior without a visible face, chosen once at startup.')
    share = Path(get_package_share_directory('face_tracking_arm'))
    default_world = share / 'worlds/lite6_table.sdf'
    initial_argument = DeclareLaunchArgument(
        'initial_positions_file',
        default_value='',
        description='Optional YAML with six angles in radians; overrides start_pose.')
    robot_model_argument = DeclareLaunchArgument(
        'robot_model', default_value='xarm6', choices=['xarm6', 'lite6'],
        description='Simulated robot; lite6 reproduces the recorded incident pose.')
    start_pose_argument = DeclareLaunchArgument(
        'start_pose', default_value='rest', choices=list(START_POSES),
        description='Initial joint pose: rest, zero, xArm6 folded, joint1_limit, Lite6 incident.')
    world_argument = DeclareLaunchArgument(
        'world_file', default_value=str(default_world),
        description='Gazebo world file, retaining world name lite6_table.')
    setup = OpaqueFunction(function=_launch_setup)
    return LaunchDescription([
        headless_argument, servo_backend_argument, test_face_scenario_argument,
        visualize_target_argument, smoothing_plugin_argument, camera_mode_argument,
        camera_baseline_argument, *camera_arguments, idle_argument, world_argument,
        initial_argument, robot_model_argument, start_pose_argument, setup])
