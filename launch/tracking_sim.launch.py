# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run reactive tracking with model-specific startup poses in Gazebo Sim."""

from pathlib import Path
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def generate_launch_description() -> LaunchDescription:
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    sys.path.insert(0, str(scripts))
    from face_tracking_bringup.simulation_launch import assemble_simulation
    from face_tracking_bringup.simulation_start import START_POSES

    headless_argument = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run only the Gazebo server without its graphical client.',
    )
    calibration_mode_argument = DeclareLaunchArgument(
        'calibration_mode', default_value='false', choices=['true', 'false'],
        description=('Run Gazebo RGB-D and MoveIt setup services without starting '
                     'the face-tracking controller.'),
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
        choices=['disabled', 'stationary', 'circle', 'people', 'walk_around', 'approach'],
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
        'camera_mode', default_value='disabled',
        choices=['disabled', 'mono_cpu', 'stereo', 'rgbd'],
        description='Optional monitor-mounted RGB, stereo or rendered RGB-D sensor assembly.')
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
    perception_status_argument = DeclareLaunchArgument(
        'require_perception_status', default_value='false', choices=['true', 'false'],
        description='Require fresh machine-readable camera health before tracking motion.')
    share = Path(get_package_share_directory('face_tracking_arm'))
    default_world = share / 'worlds/lite6_table.sdf'
    initial_argument = DeclareLaunchArgument(
        'initial_positions_file',
        default_value='',
        description='Optional YAML with six angles in radians; overrides start_pose.')
    test_world_box_argument = DeclareLaunchArgument(
        'test_world_box', default_value='false', choices=['true', 'false'],
        description=('Test-only: use a box instead of the round table in the initial MoveIt '
                     'planning-scene update.'))
    table_enabled_argument = DeclareLaunchArgument(
        'table_enabled', default_value='true', choices=['true', 'false'],
        description='Keep the tabletop in Gazebo and MoveIt planning scene.')
    robot_model_argument = DeclareLaunchArgument(
        'robot_model', default_value='xarm6', choices=['xarm6', 'lite6'],
        description='Simulated robot; lite6 reproduces the recorded incident pose.')
    start_pose_argument = DeclareLaunchArgument(
        'start_pose', default_value='rest', choices=list(START_POSES),
        description='Initial joint pose: rest, zero, xArm6 folded, joint1_limit, Lite6 incident.')
    world_argument = DeclareLaunchArgument(
        'world_file', default_value=str(default_world),
        description='Gazebo world file, retaining world name lite6_table.')
    collision_aware_servo_config_argument = DeclareLaunchArgument(
        'collision_aware_servo_config_file',
        default_value=str(share / 'config/moveit/collision_aware_servo.yaml'),
        description=('Controller YAML. Point to a source-tree copy to edit simulation settings '
                     'without rebuilding the installed package.'))
    setup = OpaqueFunction(function=assemble_simulation)
    return LaunchDescription([
        headless_argument, calibration_mode_argument, servo_backend_argument,
        test_face_scenario_argument,
        visualize_target_argument, smoothing_plugin_argument, camera_mode_argument,
        camera_baseline_argument, *camera_arguments, idle_argument,
        perception_status_argument, world_argument, collision_aware_servo_config_argument,
        initial_argument, test_world_box_argument, table_enabled_argument,
        robot_model_argument, start_pose_argument, setup])
