# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Assemble Gazebo processes, control components and their startup dependencies."""

from functools import partial
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

from .simulation_config import (
    _gazebo_physics_robot_description,
    camera_mappings,
    load_simulation_config,
)


WORLD_NAME = 'lite6_table'


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


def _gazebo_processes(world_path, headless, system_plugin_path):
    gazebo_server_process = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            '-s',
            '--headless-rendering',
            world_path,
        ],
        additional_env={
            'GZ_SIM_SYSTEM_PLUGIN_PATH': system_plugin_path,
            'GZ_SIM_RESOURCE_PATH': os.pathsep.join(path for path in (
                str(Path(get_package_share_directory('face_tracking_arm')) / 'worlds' / 'models'),
                os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
            ) if path),
        },
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
        additional_env={
            'GZ_SIM_SYSTEM_PLUGIN_PATH': system_plugin_path,
            'GZ_SIM_RESOURCE_PATH': os.pathsep.join(path for path in (
                str(Path(get_package_share_directory('face_tracking_arm')) / 'worlds' / 'models'),
                os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
            ) if path),
        },
        condition=UnlessCondition(headless),
        output='screen',
    )
    return gazebo_server_process, gazebo_gui_process


def _clock_bridge():
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
    return clock_bridge_node


def _robot_nodes(config):
    original_robot_description = config.robot_parameters[0]['robot_description']
    gazebo_robot_description = _gazebo_physics_robot_description(
        original_robot_description)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            config.robot_parameters[0], config.state_publisher, {'use_sim_time': True}
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
            '-param',
            'robot_description',
            '-name',
            'lite6',
            '-allow_renaming',
            'false',
        ],
        parameters=[{'robot_description': gazebo_robot_description}],
        output='screen',
    )

    return robot_state_publisher_node, spawn_robot_node


def _controller_spawner(controller, node_name):
    return Node(
        package='controller_manager', executable='spawner', name=node_name,
        arguments=[controller, '--controller-manager', '/controller_manager',
                   '--controller-manager-timeout', '120'], output='screen')


def _servo_components(config, servo_backend, idle_behavior):
    standard_servo_component = ComposableNode(
        package='moveit_servo',
        plugin='moveit_servo::ServoNode',
        name='servo_node',
        parameters=[
            {'moveit_servo': config.standard_servo},
            {
                'update_period': 0.01,
                'planning_group_name': config.controller['planning_group_name'],
                'use_sim_time': True,
            },
            *config.robot_parameters,
        ],
        extra_arguments=[{'use_intra_process_comms': True}],
    )
    collision_aware_servo_component = ComposableNode(
        package='face_tracking_arm',
        plugin='face_tracking_arm::CollisionAwareServoComponent',
        name='servo_node',
        parameters=[
            {'moveit_servo': config.collision_aware_servo},
            config.controller,
            {'use_sim_time': True},
            *config.robot_parameters,
        ],
        extra_arguments=[{'use_intra_process_comms': True}],
    )
    tracking_component = ComposableNode(
        package='face_tracking_arm',
        plugin='face_tracking_arm::FaceTrackingComponent',
        name='face_tracking_controller',
        parameters=[config.tracking, {'use_sim_time': True, 'idle_behavior': idle_behavior}],
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
    return (servo_container, load_collision_aware_servo_components,
            load_standard_servo_components)


def _move_group(config):
    """Start MoveIt planning services for calibration, without a servo/tracker."""
    moveit_share = Path(get_package_share_directory('moveit_configs_utils'))
    import yaml

    with (moveit_share / 'default_configs/ompl_planning.yaml').open(encoding='utf-8') as stream:
        ompl = yaml.safe_load(stream)
    return Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[
            *config.robot_parameters,
            {
                'use_sim_time': True,
                'planning_pipelines': ['ompl'],
                'ompl': ompl,
                'planning_scene_monitor.publish_planning_scene': True,
                'planning_scene_monitor.publish_geometry_updates': True,
                'planning_scene_monitor.publish_state_updates': True,
                'planning_scene_monitor.publish_transforms_updates': True,
            },
        ],
    )


def _rgbd_camera_bridge():
    """Bridge only the rendered camera feeds needed by hand-eye calibration."""
    return Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='calibration_camera_bridge', output='screen',
        parameters=[{
            'use_sim_time': True,
            'config_file': str(Path(get_package_share_directory('face_tracking_arm')) /
                               'config/calibration_rgbd_bridge.yaml'),
        }],
    )


def _target_nodes(config, test_face_scenario, visualize_target):
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
        parameters=[config.marker, {'use_sim_time': True}],
        condition=IfCondition(visualize_target),
        output='screen',
    )

    return test_face_target_publisher_node, target_marker_node


def _configuration(context, share):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    camera = camera_mappings(
        mode=value('camera_mode'), baseline=value('camera_baseline'),
        width=value('camera_width'), height=value('camera_height'),
        rate=value('camera_rate'), horizontal_fov=value('camera_horizontal_fov'))
    return load_simulation_config(
        share, robot_model=value('robot_model'), start_pose=value('start_pose'),
        initial_positions_file=value('initial_positions_file'), camera=camera,
        servo_backend=value('servo_backend'), smoothing_plugin=value('smoothing_plugin'),
        idle_behavior=value('idle_behavior'),
        require_perception_status=value('require_perception_status') == 'true',
        controller_config_file=value('collision_aware_servo_config_file'))


def assemble_simulation(context):
    """Start control only after robot creation and both controller activations."""
    share = Path(get_package_share_directory('face_tracking_arm'))
    config = _configuration(context, share)
    calibration_mode = LaunchConfiguration('calibration_mode').perform(context) == 'true'
    if calibration_mode and LaunchConfiguration('camera_mode').perform(context) != 'rgbd':
        raise ValueError('calibration_mode requires camera_mode:=rgbd')
    plugin_path = os.pathsep.join(path for path in (
        str(Path(get_package_prefix('gz_ros2_control')) / 'lib'),
        os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    ) if path)
    world_file = Path(LaunchConfiguration('world_file').perform(context)).expanduser()
    if (calibration_mode and
            world_file.resolve() == (share / 'worlds/lite6_table.sdf').resolve()):
        world_file = share / 'worlds/hand_eye_calibration.sdf'
    gazebo_server_process, gazebo_gui_process = _gazebo_processes(
        str(world_file),
        LaunchConfiguration('headless'), plugin_path)
    clock_bridge_node = _clock_bridge()
    robot_state_publisher_node, spawn_robot_node = _robot_nodes(config)
    joint_state_broadcaster_spawner = _controller_spawner(
        'joint_state_broadcaster', 'joint_state_broadcaster_spawner')
    arm_controller_spawner = _controller_spawner(
        'lite6_arm_controller', 'lite6_arm_controller_spawner')
    if calibration_mode:
        move_group = _move_group(config)
        camera_bridge = _rgbd_camera_bridge()
        guard_camera_bridge = _shutdown_on_exit(camera_bridge, 'calibration RGB-D bridge')
        servo_container = None
        target_nodes = []
        start_control = _after_success(
            arm_controller_spawner, [move_group], 'lite6_arm_controller')
        guard_control = _shutdown_on_exit(move_group, 'MoveGroup calibration services')
        control_nodes = []
        guards = [guard_control, guard_camera_bridge]
    else:
        (servo_container, load_collision_aware_servo_components,
         load_standard_servo_components) = _servo_components(
            config, LaunchConfiguration('servo_backend'),
            LaunchConfiguration('idle_behavior').perform(context))
        test_face_target_publisher_node, target_marker_node = _target_nodes(
            config, LaunchConfiguration('test_face_scenario'),
            LaunchConfiguration('visualize_target'))
        target_nodes = []
        start_control = _after_success(
            arm_controller_spawner,
            [load_collision_aware_servo_components,
             load_standard_servo_components, test_face_target_publisher_node],
            'lite6_arm_controller')
        guard_control = _shutdown_on_exit(servo_container, 'Servo component container')
        control_nodes = [servo_container, target_marker_node]
        guards = [guard_control]
        camera_bridge = None

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
    guard_gazebo_server = _shutdown_on_exit(gazebo_server_process, 'gz sim')
    guard_gazebo_gui = _shutdown_on_exit(gazebo_gui_process, 'gz sim GUI')
    guard_clock_bridge = _shutdown_on_exit(clock_bridge_node, 'clock bridge')
    guard_robot_state_publisher = _shutdown_on_exit(
        robot_state_publisher_node, 'robot state publisher'
    )
    return [
        guard_gazebo_server, guard_gazebo_gui, guard_clock_bridge,
        guard_robot_state_publisher, *guards,
        gazebo_server_process, gazebo_gui_process, clock_bridge_node,
        *control_nodes, *target_nodes,
        *([camera_bridge] if camera_bridge is not None else []),
        robot_state_publisher_node, spawn_robot_node,
        start_joint_state_broadcaster, start_arm_controller, start_control,
    ]
