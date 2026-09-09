# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Assemble real cameras and Lite 6 through existing ROS contracts, without simulator nodes."""

from functools import partial
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
import yaml

from .hardware_config import validate_config
from .hardware_description import control_parameters, load_yaml, robot_parameters


def exit_actions(event, context, actions=()):
    if context.is_shutdown:
        return []
    if event.returncode == 0 and actions:
        return list(actions)
    return [EmitEvent(event=Shutdown(reason=f'Required process exited ({event.returncode})'))]


def guard(node, actions=()):
    return RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=partial(exit_actions, actions=actions)))


def camera_actions(config, mode, driver_only):
    settings = config[mode]
    if settings['source'] == 'ros':
        return []
    get_package_prefix('usb_cam')  # Diagnose the missing optional driver before starting the arm.
    nodes = []
    for side in ('left', 'right') if mode == 'stereo' else ('left',):
        c = settings[side]
        calibration = c['calibration_file']
        if driver_only and not calibration:
            calibration_url = ''
        else:
            calibration_url = Path(calibration).as_uri()
        node = Node(package='usb_cam', executable='usb_cam_node_exe',
                    name='face_' + side + '_camera', namespace='/face_camera/' + side,
                    output='screen', parameters=[{
                        **c.get('driver_parameters', {}),
                        'use_sim_time': False, 'video_device': c['device'],
                        'image_width': c['width'], 'image_height': c['height'],
                        'framerate': float(c['fps']), 'pixel_format': c['pixel_format'],
                        'io_method': 'mmap', 'camera_name': c['camera_name'],
                        'frame_id': c['frame_id'], 'camera_info_url': calibration_url}],
                    remappings=[('image_raw', c['image_topic']), ('camera_info', c['info_topic'])])
        nodes.extend([guard(node), node])
    return nodes


def robot_actions(config, mode, share, config_path, directory, mock):
    api_config = None
    if not mock:
        get_package_prefix('xarm_controller')
        api_config = str(Path(get_package_share_directory('xarm_api')) / 'config/xarm_params.yaml')
    model, limits = robot_parameters(config, mode, share, config_path, mock)
    servo, motion, tracking, controllers = control_parameters(config, share, limits)
    controllers_path = Path(directory.name) / 'controllers.yaml'
    controllers_path.write_text(yaml.safe_dump(controllers))
    state = Node(package='robot_state_publisher', executable='robot_state_publisher',
                 parameters=[model[0], {'use_sim_time': False, 'publish_frequency': 100.0}],
                 output='screen')
    manager_params = [model[0], str(controllers_path)]
    if api_config:
        manager_params.append(api_config)
    manager = Node(package='controller_manager', executable='ros2_control_node',
                   parameters=manager_params, output='screen',
                   remappings=[('robot_description', '/robot_description')])
    spawners = [Node(package='controller_manager', executable='spawner', output='screen',
                     arguments=[name, '--controller-manager', '/controller_manager',
                                '--controller-manager-timeout', '60'])
                for name in ('joint_state_broadcaster', 'lite6_arm_controller')]
    container = ComposableNodeContainer(
        name='face_tracking_servo_container', namespace='/', package='face_tracking_arm',
        executable='graceful_component_container', composable_node_descriptions=[],
        output='screen')
    tracker = ComposableNode(
        package='face_tracking_arm', name='face_tracking_controller',
        plugin='face_tracking_arm::FaceTrackingComponent',
        parameters=[tracking, {'use_sim_time': False}],
        extra_arguments=[{'use_intra_process_comms': True}])
    executor = ComposableNode(
        package='face_tracking_arm', name='servo_node',
        plugin='face_tracking_arm::CollisionAwareServoComponent',
        parameters=[{'moveit_servo': servo}, motion, {'use_sim_time': False}, *model],
        extra_arguments=[{'use_intra_process_comms': True}])
    load = LoadComposableNodes(target_container=container,
                               composable_node_descriptions=[tracker, executor])
    return [guard(state), guard(manager), guard(container),
            guard(spawners[0], [spawners[1]]), guard(spawners[1], [load]),
            state, manager, container, spawners[0]]


def assemble(context, *, mode, driver_only):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    if driver_only:
        mode = value('camera_mode')
    camera_only = driver_only or value('camera_only') == 'true'
    mock = not driver_only and value('mock_hardware') == 'true'
    path = Path(value('config_file')).expanduser().resolve()
    config = validate_config(load_yaml(path), mode, camera_only=camera_only, mock=mock,
                             driver_only=driver_only, directory=path.parent)
    directory = tempfile.TemporaryDirectory(prefix='face_tracking_hardware_')
    # Retain the directory for the whole launch lifetime, including child shutdown.
    config_path = Path(directory.name) / 'resolved.yaml'
    config_path.write_text(yaml.safe_dump(config))
    cleanup = RegisterEventHandler(OnShutdown(on_shutdown=lambda event, ctx: directory.cleanup()))
    share = Path(get_package_share_directory('face_tracking_arm'))
    actions = [cleanup, *camera_actions(config, mode, driver_only)]
    if driver_only:
        if config[mode]['source'] == 'ros':
            raise ValueError('source=ros: start the camera manufacturer driver separately')
        return actions
    p, settings = config['perception'], config[mode]
    scripts = Path(get_package_prefix('face_tracking_arm')) / 'lib/face_tracking_arm'
    model_name = {'yunet': 'face_detection_yunet_2023mar.onnx',
                  'yolov5n_face': 'yolov5n-face.onnx',
                  'yolo_facev2n': 'yolo-facev2n-preweight.onnx'}
    if not (Path(p['model_dir']) / model_name[p['detector']]).is_file():
        raise ValueError(f'Model missing in {p["model_dir"]}; see docs/hardware.md')
    params = {k: v for k, v in p.items() if k != 'python_executable'}
    params.update(mode=mode, use_sim_time=False, rectification=settings['rectification'],
                  selection_frame=settings['left']['frame_id'] if camera_only else 'link_base',
                  world_frame='' if camera_only else 'world',
                  input_left_frame=settings['left']['frame_id'])
    remaps = [('/face_test/image_raw', settings['left']['image_topic']),
              ('/face_test/camera_info', settings['left']['info_topic']),
              ('/face_test/center_3d', '/face/center')]
    if mode == 'stereo':
        params.update(sync_slop_sec=settings['sync_slop_sec'],
                      stereo_baseline_m=settings['baseline_m'],
                      input_right_frame=settings['right']['frame_id'])
        remaps += [('/face_test/right/image_raw', settings['right']['image_topic']),
                   ('/face_test/right/camera_info', settings['right']['info_topic'])]
    detector = Node(executable=p['python_executable'], name='face_position', output='screen',
                    arguments=[str(scripts / 'face_position_node.py')], parameters=[params],
                    remappings=remaps)
    actions += [guard(detector), detector]
    if not camera_only:
        actions += robot_actions(config, mode, share, config_path, directory, mock)
    if value('show_image') == 'true':
        viewer = Node(package='image_view', executable='image_view',
                      remappings=[('image', '/face_test/debug_image')],
                      parameters=[{'use_sim_time': False, 'image_transport': 'compressed',
                                   'window_name': 'Face tracking'}])
        actions.append(viewer)
    return actions


def generate_hardware_launch(mode='mono_cpu', *, driver_only=False):
    share = Path(get_package_share_directory('face_tracking_arm'))
    arguments = [DeclareLaunchArgument('config_file',
                 default_value=str(share / 'config/hardware.yaml'),
                 description='Absolute path to the editable hardware YAML.')]
    if driver_only:
        argument = DeclareLaunchArgument('camera_mode', default_value=mode,
                                         choices=['mono_cpu', 'stereo'])
        arguments.append(argument)
    else:
        for name, default, description in (
                ('camera_only', 'false', 'Run perception without connecting to the robot.'),
                ('show_image', 'false', 'Show a lightweight camera preview.'),
                ('mock_hardware', 'false', 'Development check without a real Lite 6 connection.')):
            argument = DeclareLaunchArgument(name, default_value=default,
                                             choices=['true', 'false'], description=description)
            arguments.append(argument)
    setup = OpaqueFunction(function=partial(assemble, mode=mode, driver_only=driver_only))
    return LaunchDescription([*arguments, setup])
