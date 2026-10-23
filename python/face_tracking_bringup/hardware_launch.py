# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Assemble the configured robot and cameras through their ROS contracts."""

from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.event_handlers import OnProcessExit, OnProcessStart, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node, SetRemap
from launch_ros.descriptions import ComposableNode

import yaml

from .camera_controls import configure_controls
from .execution_budget_profile import (
    validate_active_hardware_profile)
from .hardware_config import validate_config
from .hardware_description import (
    control_parameters, load_yaml, perception_parameters, robot_parameters)
from .hardware_probe import probe_robot
from .robot_profile import get_robot_profile


def exit_actions(event, context, actions=()):
    if context.is_shutdown:
        return []
    if event.returncode == 0 and actions:
        return list(actions)
    return [EmitEvent(event=Shutdown(reason=f'Required process exited ({event.returncode})'))]


def guard(node, actions=()):
    return RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=partial(exit_actions, actions=actions)))


def build_artifact_manifest(config_path, share, model, controllers, profile,
                            probe_result=None, effective_parameters=None):
    """Hash runtime artifacts and retain redacted launch inputs for offline review."""
    def digest_bytes(value):
        return hashlib.sha256(value).hexdigest()

    def digest_file(path):
        try:
            return digest_bytes(Path(path).read_bytes())
        except OSError:
            return None

    def redact(value, key=''):
        if isinstance(value, dict):
            return {name: redact(item, str(name).lower()) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [redact(item, key) for item in value]
        if any(part in key for part in ('password', 'token', 'secret', 'credential', 'api_key')):
            return '<redacted>'
        if key in ('ip', 'robot_ip'):
            return '<redacted>'
        return value

    descriptions = {
        'expanded_urdf_sha256': digest_bytes(
            model[0]['robot_description'].encode()),
        'expanded_srdf_sha256': digest_bytes(
            model[1]['robot_description_semantic'].encode()),
        'effective_controllers_sha256': digest_bytes(
            yaml.safe_dump(controllers, sort_keys=True).encode()),
        'effective_parameters_sha256': {
            name: digest_bytes(yaml.safe_dump(value, sort_keys=True).encode())
            for name, value in (effective_parameters or {}).items()},
        'hardware_config_sha256': digest_file(config_path),
        'source_yaml_sha256': {
            name: digest_file(Path(share) / name) for name in (
                'config/moveit/servo.yaml',
                'config/moveit/collision_aware_servo.yaml',
                'config/moveit/kinematics.yaml',
                'config/control/controllers.yaml',
                'config/tracking.yaml')},
    }
    source_sha256 = os.environ.get('FACE_TRACKING_SOURCE_SHA256')
    if (not source_sha256 or len(source_sha256) != 64
            or any(character not in '0123456789abcdef' for character in source_sha256.lower())):
        source_sha256 = None
    effective_parameters = effective_parameters or {}
    resolved_config = yaml.safe_dump(redact(yaml.safe_load(Path(config_path).read_text())),
                                     sort_keys=True)
    controllers_yaml = yaml.safe_dump(controllers, sort_keys=True)
    effective_parameters_yaml = {
        name: yaml.safe_dump(value, sort_keys=True)
        for name, value in effective_parameters.items()}
    return {
        'schema': 1,
        'robot_model': profile['model'],
        'model_num': profile['model_num'],
        'firmware_version': (probe_result or {}).get('firmware_version'),
        'robot_serial': (probe_result or {}).get('serial_number'),
        'source_tree_sha256': source_sha256,
        'docker_image_id': os.environ.get('FACE_TRACKING_DOCKER_IMAGE_ID'),
        'image_digest': None,
        'image_digest_status': 'manifest_digest_unavailable',
        'replay_inputs': {
            'resolved_hardware_yaml_redacted': resolved_config,
            'expanded_urdf': model[0]['robot_description'],
            'expanded_srdf': model[1]['robot_description_semantic'],
            'effective_controllers_yaml': controllers_yaml,
            'effective_parameters_yaml': effective_parameters_yaml,
        },
        'artifacts': descriptions,
    }


def _monitor_actions(config, probe_result, *, artifact_manifest=None,
                     servo_execution_context_topic='/servo_node/execution_context',
                     use_sim_time=False, after_ready=()):
    """Arm the read-only hardware monitor and gate downstream control on readiness."""
    profile = get_robot_profile(config['robot']['model'], config['robot']['model_num'])
    namespace = 'xarm' if profile['model'] == 'xarm6' else 'ufactory'
    snapshot = {key: probe_result[key] for key in (
        'serial_number', 'reduced_mode', 'reduced_joint_limits_rad',
        'reduced_tcp_boundary_mm', 'reduced_tcp_max_speed_mmps',
        'effective_position_limits_rad', 'reduced_max_joint_speed_rad_s',
        'safety_boundary_enabled', 'collision_rebound_enabled')}
    monitor = Node(
        package='face_tracking_arm', executable='hardware_monitor_node.py',
        name='hardware_monitor', output='screen',
        parameters=[{'robot_ip': config['robot']['ip'], 'expected_mode': 1,
                     'robot_state_topic': f'/{namespace}/robot_states',
                     'set_state_service': f'/{namespace}/set_state',
                     'hardware_inhibit_service': f'/{namespace}/_inhibit_servoj',
                     'servo_execution_context_topic': servo_execution_context_topic,
                     'face_transform_provenance_topic':
                         '/tracking/face_transform_provenance',
                     'servo_trajectory_publications_topic':
                         '/servo_node/trajectory_publications',
                     'preflight_snapshot_json': json.dumps(snapshot),
                     'artifact_manifest_json': json.dumps(
                         artifact_manifest or {}, sort_keys=True),
                     'sdk_request_timeout_sec': .5,
                     'sdk_state_timeout_sec': .75,
                     'sdk_poll_period_sec': .5,
                     'startup_timeout_sec': 20.0,
                     'report_timeout_sec': .5,
                     'use_sim_time': use_sim_time}])
    gate = Node(package='face_tracking_arm', executable='hardware_monitor_gate.py',
                name='hardware_monitor_gate', output='screen',
                parameters=[{'use_sim_time': use_sim_time}])

    def require_live_monitor(event, context):
        if context.is_shutdown:
            return []
        raise RuntimeError(f'Hardware monitor exited unexpectedly ({event.returncode})')

    def require_ready_gate(event, context, actions=()):
        if context.is_shutdown:
            return []
        if event.returncode != 0:
            raise RuntimeError(f'Hardware monitor did not become ready ({event.returncode})')
        return list(actions)

    return [RegisterEventHandler(OnProcessExit(
                target_action=monitor, on_exit=require_live_monitor)),
            RegisterEventHandler(OnProcessExit(
                target_action=gate,
                on_exit=partial(require_ready_gate, actions=after_ready))),
            monitor, gate]


def camera_actions(config, mode, driver_only):
    settings = config[mode]
    if settings['source'] == 'ros':
        return []
    if mode == 'rgbd' and settings['source'] == 'realsense':
        share = Path(get_package_share_directory('realsense2_camera'))
        camera = settings['left']
        namespace = settings['camera_namespace'].strip('/')
        name = settings['camera_name']
        prefix = '/' + '/'.join(part for part in (namespace, name) if part)
        default_topics = {
            'image_topic': prefix + '/color/image_raw',
            'info_topic': prefix + '/color/camera_info',
            'depth_topic': prefix + '/aligned_depth_to_color/image_raw',
        }
        remappings = [(default_topics[key], camera[key]) for key in default_topics
                      if default_topics[key] != camera[key]]
        params = {
            'camera_namespace': settings['camera_namespace'], 'camera_name': name,
            'serial_no': settings['serial_no'],
            # Do not inherit the driver's external YAML config_file launch arg.
            'config_file': '',
            'rgb_camera.color_profile': settings['color_profile'],
            'depth_module.depth_profile': settings['depth_profile'],
            'enable_color': True, 'enable_depth': True, 'enable_sync': True,
            'align_depth.enable': True, 'publish_tf': False,
            'enable_infra': False, 'enable_gyro': False, 'enable_accel': False,
            'pointcloud.enable': False, 'enable_rgbd': False,
            'diagnostics_period': 1.0,
        }
        driver = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/rs_launch.py')),
            launch_arguments={key: str(value).lower() if isinstance(value, bool)
                              else str(value) for key, value in params.items()}.items(),
        )
        if remappings:
            return [GroupAction([*(SetRemap(src=source, dst=target)
                                   for source, target in remappings), driver])]
        return [driver]
    get_package_prefix('usb_cam')  # Diagnose the missing optional driver before starting the arm.
    nodes = []
    for side in ('left', 'right') if mode == 'stereo' else ('left',):
        c = settings[side]
        applied = configure_controls(c)
        if applied:
            print(f'Camera {side}: verified hardware controls {applied}', flush=True)
        calibration = c['calibration_file']
        if driver_only and not calibration:
            calibration_url = ''
        else:
            calibration_url = Path(calibration).as_uri()
        node = Node(package='usb_cam', executable='usb_cam_node_exe',
                    name='face_' + side + '_camera', namespace='/face_camera/' + side,
                    output='screen', parameters=[{
                        'io_method': 'mmap',
                        **c.get('driver_parameters', {}),
                        'use_sim_time': False, 'video_device': c['device'],
                        'image_width': c['width'], 'image_height': c['height'],
                        'framerate': float(c['fps']), 'pixel_format': c['pixel_format'],
                        'camera_name': c['camera_name'],
                        'frame_id': c['frame_id'], 'camera_info_url': calibration_url}],
                    remappings=[('image_raw', c['image_topic']), ('camera_info', c['info_topic'])])
        nodes.extend([guard(node), node])
    return nodes


def robot_actions(config, mode, share, config_path, directory, mock, probe_result=None):
    api_config = None
    if not mock:
        get_package_prefix('xarm_controller')
        api_config = str(Path(get_package_share_directory('xarm_api')) / 'config/xarm_params.yaml')
    model, limits = robot_parameters(config, mode, share, config_path, mock, probe_result)
    servo, motion, tracking, controllers = control_parameters(config, share, limits, mode)
    profile = get_robot_profile(config['robot']['model'], config['robot']['model_num'])
    if not mock:
        profile = probe_result['profile']
        budget, fingerprint, controller_mode = validate_active_hardware_profile(
            config, probe_result, profile, servo, motion, controllers,
            model[0]['robot_description'], model[1]['robot_description_semantic'])
        motion.update(
            hardware_execution_budget_source=2,
            hardware_execution_budget_profile_id=budget['profile_id'],
            hardware_execution_budget_robot_model=budget['robot_model'],
            hardware_execution_budget_model_num=budget['model_num'],
            hardware_execution_budget_firmware=budget['firmware_version'],
            hardware_execution_budget_controller_mode=budget['controller_mode'],
            hardware_execution_budget_conditions_fingerprint=fingerprint,
            hardware_execution_budget_evidence_reference=budget['evidence_reference'],
            hardware_execution_budget_reduced_mode_known=True,
            hardware_execution_budget_reduced_mode=budget['reduced_mode'],
            hardware_execution_error_bound_rad=budget['execution_error_bound_rad'],
            hardware_reaction_threshold_rad=budget['reaction_threshold_rad'],
            hardware_post_threshold_growth_rad=budget['post_threshold_growth_rad'],
            hardware_execution_expected_profile_id=budget['profile_id'],
            hardware_execution_expected_robot_model=probe_result['profile']['model'],
            hardware_execution_expected_model_num=probe_result['profile']['model_num'],
            hardware_execution_expected_firmware=probe_result['firmware_version'],
            hardware_execution_expected_controller_mode=controller_mode,
            hardware_execution_expected_conditions_fingerprint=fingerprint,
            hardware_execution_expected_reduced_mode=probe_result['reduced_mode'])
    hw_namespace = 'xarm' if profile['model'] == 'xarm6' else 'ufactory'
    motion.update(
        require_hardware_io_status=not mock,
        hardware_io_status_topic=f'/{hw_namespace}/hardware_io_status',
        hardware_inhibit_service=f'/{hw_namespace}/_inhibit_servoj',
        servo_execution_context_topic='/servo_node/execution_context')
    if not mock:
        motion['hardware_factory_position_limits_rad'] = [
            value for pair in profile['position_limits_rad'] for value in pair]
        motion['hardware_position_limits_rad'] = [
            value for pair in probe_result['effective_position_limits_rad'] for value in pair]
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
                for name in ('joint_state_broadcaster', get_robot_profile(
                    config['robot']['model'], config['robot']['model_num'])['controller_name'])]
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
    artifact_manifest = build_artifact_manifest(
        config_path, share, model, controllers, profile, probe_result,
        {'servo': servo, 'motion': motion, 'tracking': tracking})
    activation = ([load] if mock else _monitor_actions(
        config, probe_result, artifact_manifest=artifact_manifest,
        servo_execution_context_topic='/servo_node/execution_context', after_ready=[load]))
    return [guard(state), guard(manager), guard(container),
            guard(spawners[0], [spawners[1]]), guard(spawners[1], activation),
            state, manager, container, spawners[0]]


def mount_setup_robot_actions(config, mode, share, config_path, directory, mock,
                              probe_result, setup_node, *, use_sim_time=False):
    """Start feedback and the trajectory controller without tracking components."""
    api_config = None
    if not mock:
        get_package_prefix('xarm_controller')
        api_config = str(Path(get_package_share_directory('xarm_api')) /
                         'config/xarm_params.yaml')
    model, limits = robot_parameters(config, mode, share, config_path, mock, probe_result)
    controllers = control_parameters(config, share, limits, mode)[3]
    for node in controllers.values():
        node['ros__parameters']['use_sim_time'] = use_sim_time
    directory_path = (Path(directory.name) if hasattr(directory, 'name')
                      else Path(directory))
    controller_path = directory_path / 'mount_setup_controllers.yaml'
    controller_path.write_text(yaml.safe_dump(controllers))
    state = Node(package='robot_state_publisher', executable='robot_state_publisher',
                 parameters=[model[0], {'use_sim_time': use_sim_time,
                                        'publish_frequency': 100.0}],
                 output='screen')
    manager_params = [model[0], str(controller_path)]
    if api_config:
        manager_params.append(api_config)
    manager = Node(package='controller_manager', executable='ros2_control_node',
                   parameters=[*manager_params, {'use_sim_time': use_sim_time}], output='screen',
                   remappings=[('robot_description', '/robot_description')])
    profile = get_robot_profile(config['robot']['model'], config['robot']['model_num'])
    controller = profile['controller_name']
    broadcaster = Node(package='controller_manager', executable='spawner', output='screen',
                       arguments=['joint_state_broadcaster', '--controller-manager',
                                  '/controller_manager', '--controller-manager-timeout', '60'])
    spawner = Node(package='controller_manager', executable='spawner', output='screen',
                   arguments=[controller, '--controller-manager', '/controller_manager',
                              '--controller-manager-timeout', '60'])
    moveit_share = Path(get_package_share_directory('moveit_configs_utils'))
    ompl = load_yaml(moveit_share / 'default_configs/ompl_planning.yaml')
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[*model, {'use_sim_time': use_sim_time,
                             'planning_pipelines': ['ompl'], 'ompl': ompl,
                             'planning_scene_monitor.publish_planning_scene': True,
                             'planning_scene_monitor.publish_geometry_updates': True,
                             'planning_scene_monitor.publish_state_updates': True,
                             'planning_scene_monitor.publish_transforms_updates': True}])

    def finish_setup(event, context):
        if context.is_shutdown:
            return []
        if event.returncode != 0:
            raise RuntimeError(f'Mount setup failed with exit code {event.returncode}')
        return [EmitEvent(event=Shutdown(
            reason=f'Mount setup completed with exit code {event.returncode}'))]

    start_setup = RegisterEventHandler(OnProcessStart(
        target_action=move_group, on_start=lambda event, context: [setup_node]))
    finish = RegisterEventHandler(OnProcessExit(target_action=setup_node,
                                                on_exit=finish_setup))
    after_controller = ([move_group] if mock else _monitor_actions(
        config, probe_result, use_sim_time=use_sim_time, after_ready=[move_group]))
    return [guard(state), guard(manager), guard(broadcaster, [spawner]),
            guard(spawner, after_controller), guard(move_group), start_setup, finish,
            *after_controller, state, manager, broadcaster, spawner]


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
    probe_result = None
    if not camera_only and not mock:
        pose_names = ['rest']
        if config['tracking']['idle_behavior'] != 'rest':
            pose_names.append('search')
        probe_result = probe_robot(
            config['robot']['ip'], expected_model=config['robot']['model'],
            joint_targets={name: [math.radians(v) for v in config['motion'][name + '_joints_deg']]
                           for name in pose_names})
        config['robot']['model_num'] = probe_result['profile']['model_num']
        print(f'Robot verified: {config["robot"]["model"]}, '
              f'variant {config["robot"]["model_num"]}, '
              f'firmware {probe_result["firmware_version"]}, '
              f'reduced mode {probe_result["reduced_mode"]}', flush=True)
        if probe_result.get('warning_code'):
            print(f'Robot warning code: {probe_result["warning_code"]}; check Studio', flush=True)
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
    params = perception_parameters(config, mode, camera_only=camera_only)
    remaps = [('/face_test/image_raw', settings['left']['image_topic']),
              ('/face_test/camera_info', settings['left']['info_topic']),
              ('/face_test/center_3d', '/face/center')]
    if mode == 'rgbd':
        remaps.append(('/face_test/depth/image_raw', settings['left']['depth_topic']))
    if mode == 'stereo':
        remaps += [('/face_test/right/image_raw', settings['right']['image_topic']),
                   ('/face_test/right/camera_info', settings['right']['info_topic'])]
    detector = Node(executable=p['python_executable'], name='face_position', output='screen',
                    arguments=[str(scripts / 'face_position_node.py')], parameters=[params],
                    remappings=remaps)
    actions += [guard(detector), detector]
    if not camera_only:
        actions += robot_actions(config, mode, share, config_path, directory, mock, probe_result)
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
                ('mock_hardware', 'false', 'Development check without a robot connection.')):
            argument = DeclareLaunchArgument(name, default_value=default,
                                             choices=['true', 'false'], description=description)
            arguments.append(argument)
    setup = OpaqueFunction(function=partial(assemble, mode=mode, driver_only=driver_only))
    return LaunchDescription([*arguments, setup])
