# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Launch the Lite 6 table scenario in Gazebo Sim."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


WORLD_NAME = 'lite6_table'


def generate_launch_description() -> LaunchDescription:
    """Create the Gazebo world, clock bridge, and Lite 6 entity."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    headless = LaunchConfiguration('headless')

    world_path = PathJoinSubstitution(
        [FindPackageShare('face_tracking_arm'), 'worlds', 'lite6_table.sdf']
    )
    robot_xacro_path = PathJoinSubstitution(
        [
            FindPackageShare('face_tracking_arm'),
            'description',
            'xarm6_table.urdf.xacro',
        ]
    )
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', robot_xacro_path]),
        value_type=str,
    )

    use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use the Gazebo simulation clock.',
    )

    headless_argument = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run only the Gazebo server without its graphical client.',
    )

    gazebo_server_process = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', world_path],
        condition=IfCondition(headless),
        output='screen',
    )
    gazebo_gui_process = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        condition=UnlessCondition(headless),
        output='screen',
    )

    clock_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=[
            f'/world/{WORLD_NAME}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        remappings=[(f'/world/{WORLD_NAME}/clock', '/clock')],
        output='screen',
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }
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

    launch_description = LaunchDescription()
    launch_description.add_action(use_sim_time_argument)
    launch_description.add_action(headless_argument)
    launch_description.add_action(gazebo_server_process)
    launch_description.add_action(gazebo_gui_process)
    launch_description.add_action(clock_bridge_node)
    launch_description.add_action(robot_state_publisher_node)
    launch_description.add_action(spawn_robot_node)
    return launch_description
