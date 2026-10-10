# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Check simulation configuration and startup decisions without starting processes."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.substitutions import LaunchConfiguration
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.simulation_config import (  # noqa: E402, I100
    _gazebo_physics_robot_description,
    camera_mappings,
    load_simulation_config,
)
from face_tracking_bringup import simulation_launch as sim  # noqa: E402, I100


def _camera(**overrides):
    values = {'mode': 'disabled', 'baseline': '0.08', 'width': '1280', 'height': '720',
              'rate': '30', 'horizontal_fov': '1.433'}
    return camera_mappings(**{**values, **overrides})


def _config(**overrides):
    settings = {
        'robot_model': 'xarm6', 'start_pose': 'rest', 'initial_positions_file': '',
        'camera': _camera(), 'servo_backend': 'collision_aware',
        'smoothing_plugin': 'online_signal_smoothing::AccelerationLimitedPlugin',
        'idle_behavior': 'rest'}
    return load_simulation_config(ROOT, **{**settings, **overrides})


def test_gazebo_physics_description_pads_only_top_level_revolute_bounds():
    source = """<robot name='test'>
      <link name="base"/><link name="arm"/><link name="tool"/>
      <joint name="joint1" type="revolute"><parent link="base"/>
        <child link="arm"/><limit lower="-1" upper="2" effort="5" velocity="3"/>
      </joint>
      <joint name="joint2" type="continuous"><parent link="arm"/>
        <child link="tool"/><limit effort="5" velocity="3"/>
      </joint>
      <joint name="fixed_joint" type="fixed"><parent link="base"/>
        <child link="tool"/></joint>
      <ros2_control name="arm_control" type="system">
        <joint name="joint1"><command_interface name="position">
          <param name="min">-1</param><param name="max">2</param>
        </command_interface></joint>
      </ros2_control>
    </robot>"""
    original_tree = ET.fromstring(source)

    result = _gazebo_physics_robot_description(source)

    assert source == """<robot name='test'>
      <link name="base"/><link name="arm"/><link name="tool"/>
      <joint name="joint1" type="revolute"><parent link="base"/>
        <child link="arm"/><limit lower="-1" upper="2" effort="5" velocity="3"/>
      </joint>
      <joint name="joint2" type="continuous"><parent link="arm"/>
        <child link="tool"/><limit effort="5" velocity="3"/>
      </joint>
      <joint name="fixed_joint" type="fixed"><parent link="base"/>
        <child link="tool"/></joint>
      <ros2_control name="arm_control" type="system">
        <joint name="joint1"><command_interface name="position">
          <param name="min">-1</param><param name="max">2</param>
        </command_interface></joint>
      </ros2_control>
    </robot>"""
    result_tree = ET.fromstring(result)
    original_joint = original_tree.find('./joint[@name="joint1"]/limit')
    result_joint = result_tree.find('./joint[@name="joint1"]/limit')
    assert float(result_joint.attrib['lower']) == pytest.approx(-1.01)
    assert float(result_joint.attrib['upper']) == pytest.approx(2.01)
    assert {key: value for key, value in result_joint.attrib.items()
            if key not in ('lower', 'upper')} == {
                key: value for key, value in original_joint.attrib.items()
                if key not in ('lower', 'upper')}
    assert result_tree.find('./joint[@name="joint2"]/limit').attrib == {
        'effort': '5', 'velocity': '3'}
    assert result_tree.find('./joint[@name="fixed_joint"]') is not None
    assert result_tree.find('./ros2_control/joint/command_interface/param[@name="min"]') \
        .text == '-1'
    assert result_tree.find('./ros2_control/joint/command_interface/param[@name="max"]') \
        .text == '2'
    result_tree.find('./joint[@name="joint1"]/limit').attrib.update(
        lower=original_joint.attrib['lower'], upper=original_joint.attrib['upper'])
    assert ET.tostring(result_tree) == ET.tostring(original_tree)


@pytest.mark.parametrize('limit', [
    '<limit lower="nan" upper="1"/>',
    '<limit lower="0" upper="inf"/>',
    '<limit lower="2" upper="1"/>',
    '<limit lower="0"/>',
])
def test_gazebo_physics_description_rejects_invalid_bounded_revolute_limits(limit):
    description = (
        '<robot name="test"><joint name="joint1" type="revolute">'
        '<parent link="base"/><child link="tool"/>' + limit + '</joint></robot>')
    with pytest.raises(ValueError, match='joint1'):
        _gazebo_physics_robot_description(description)


def test_spawn_receives_physics_copy_while_state_publisher_keeps_oem_description(
    monkeypatch,
):
    config = _config()
    original_description = config.robot_parameters[0]['robot_description']
    captured_nodes = []
    monkeypatch.setattr(
        sim, 'Node', lambda **kwargs: captured_nodes.append(kwargs) or kwargs)

    state_publisher, spawn = sim._robot_nodes(config)

    assert state_publisher['parameters'][0]['robot_description'] == original_description
    assert config.robot_parameters[0]['robot_description'] == original_description
    assert '-param' in spawn['arguments']
    assert 'robot_description' in spawn['arguments']
    spawned_description = spawn['parameters'][0]['robot_description']
    assert spawned_description != original_description
    assert captured_nodes[0]['name'] == 'robot_state_publisher'
    assert captured_nodes[1]['name'] == 'spawn_lite6'


def _assemble_with_stub_actions(monkeypatch, *, calibration):
    context = LaunchContext()
    context.launch_configurations.update({
        'calibration_mode': 'true' if calibration else 'false',
        'camera_mode': 'rgbd' if calibration else 'disabled',
        'world_file': str(ROOT / 'worlds/lite6_table.sdf'),
        'headless': 'true', 'robot_model': 'xarm6', 'start_pose': 'rest',
        'initial_positions_file': '', 'camera_baseline': '0.08',
        'camera_width': '1280', 'camera_height': '720', 'camera_rate': '30',
        'camera_horizontal_fov': '1.433', 'servo_backend': 'collision_aware',
        'smoothing_plugin': 'online_signal_smoothing::AccelerationLimitedPlugin',
        'idle_behavior': 'rest', 'require_perception_status': 'false',
        'test_face_scenario': 'people', 'visualize_target': 'true',
    })
    camera = _camera(mode='rgbd') if calibration else _camera()
    config = _config(camera=camera)
    monkeypatch.setattr(sim, 'get_package_share_directory', lambda _: str(ROOT))
    monkeypatch.setattr(sim, 'get_package_prefix', lambda _: '/test/prefix')
    monkeypatch.setattr(sim, '_configuration', lambda *_: config)
    monkeypatch.setattr(
        sim, '_gazebo_processes',
        lambda world, *_: (SimpleNamespace(tag='gz-server', world=world),
                           SimpleNamespace(tag='gz-gui')))
    monkeypatch.setattr(sim, '_clock_bridge', lambda: SimpleNamespace(tag='clock'))
    monkeypatch.setattr(sim, '_robot_nodes', lambda *_: (
        SimpleNamespace(tag='robot-state-publisher'), SimpleNamespace(tag='spawn-robot')))
    monkeypatch.setattr(sim, '_controller_spawner', lambda _, name: SimpleNamespace(tag=name))
    monkeypatch.setattr(
        sim, '_shutdown_on_exit',
        lambda _, name: SimpleNamespace(tag='guard:' + name))

    def after_success(target, dependants, prerequisite):
        return SimpleNamespace(
            tag='after:' + prerequisite, target=target, dependants=tuple(dependants))

    monkeypatch.setattr(sim, '_after_success', after_success)
    if calibration:
        monkeypatch.setattr(sim, '_move_group', lambda *_: SimpleNamespace(tag='move-group'))
        monkeypatch.setattr(sim, '_rgbd_camera_bridge', lambda: SimpleNamespace(tag='rgbd-bridge'))
        monkeypatch.setattr(sim, '_servo_components', lambda *_: pytest.fail(
            'calibration must not start Servo/tracking components'))
        monkeypatch.setattr(sim, '_target_nodes', lambda *_: pytest.fail(
            'calibration must not start face targets'))
    else:
        monkeypatch.setattr(sim, '_move_group', lambda *_: pytest.fail(
            'ordinary tracking must not start calibration MoveGroup'))
        monkeypatch.setattr(sim, '_rgbd_camera_bridge', lambda: pytest.fail(
            'ordinary simulation must not start the calibration bridge'))
        monkeypatch.setattr(sim, '_servo_components', lambda *_: (
            SimpleNamespace(tag='servo-container'), SimpleNamespace(tag='collision-aware'),
            SimpleNamespace(tag='standard')))
        monkeypatch.setattr(sim, '_target_nodes', lambda *_: (
            SimpleNamespace(tag='face-target-source'), SimpleNamespace(tag='target-marker')))
    return sim.assemble_simulation(context)


@pytest.mark.parametrize('model', ['xarm6', 'lite6'])
@pytest.mark.parametrize('mode', ['disabled', 'mono_cpu', 'stereo', 'rgbd'])
def test_model_camera_and_controller_parameters_agree(model, mode):
    config = _config(robot_model=model, camera=_camera(mode=mode))
    robot = ET.fromstring(config.robot_parameters[0]['robot_description'])
    semantic = ET.fromstring(config.robot_parameters[1]['robot_description_semantic'])
    group = config.controller['planning_group_name']
    assert semantic.find(f'group[@name="{group}"]') is not None
    assert group == config.standard_servo['move_group_name']
    assert group in config.robot_parameters[2]['robot_description_kinematics']
    assert set(config.controller['recovery_planner']) == {'planner_configs', group}
    if mode == 'disabled':
        assert not config.controller.get('gaze_frame')
    else:
        assert config.controller['gaze_frame'] == 'face_test_camera_optical_frame'
        assert robot.find('link[@name="face_test_camera_optical_frame"]') is not None
    limits = config.robot_parameters[3]['robot_description_planning']['joint_limits']
    assert config.controller['joint_names'] == list(limits)
    assert config.controller['face_target_freshness_timeout_sec'] == \
        config.tracking['face_target_freshness_timeout_sec']
    assert config.controller['face_prediction_timeout_sec'] == \
        config.tracking['face_prediction_timeout_sec']
    for parameter, key in (
        ('max_joint_velocity_rad_s', 'max_velocity'),
        ('max_joint_acceleration_rad_s2', 'max_acceleration'),
        ('max_joint_jerk_rad_s3', 'max_jerk'),
    ):
        assert config.controller[parameter] == [entry[key] for entry in limits.values()]
    if mode != 'disabled':
        excluded = [{item.get('link1'), item.get('link2')}
                    for item in semantic.findall('disable_collisions')]
        assert {'monitor_camera_bar', 'link6'} in excluded
        assert {'monitor_camera_bar', 'link5'} not in excluded


def test_backends_keep_independent_interpolation_and_smoothing_contracts():
    smoother = 'online_signal_smoothing::RuckigFilterPlugin'
    config = _config(smoothing_plugin=smoother)
    original = yaml.safe_load((ROOT / 'config/moveit/servo.yaml').read_text())
    assert config.standard_servo['smoothing_filter_plugin_name'] == smoother
    for field in ('publish_period', 'max_expected_latency', 'check_collisions',
                  'use_smoothing', 'publish_joint_positions', 'publish_joint_velocities',
                  'publish_joint_accelerations'):
        assert config.standard_servo[field] == original[field]
    assert config.collision_aware_servo['publish_period'] == .01
    assert config.collision_aware_servo['max_expected_latency'] == .05
    assert config.collision_aware_servo['publish_joint_positions'] is True
    for field in ('check_collisions', 'use_smoothing', 'publish_joint_velocities',
                  'publish_joint_accelerations'):
        assert config.collision_aware_servo[field] is False
    config.controller['joint_names'].clear()
    assert _config().controller['joint_names'] == [f'joint{i}' for i in range(1, 7)]


@pytest.mark.parametrize('field,value,mode', [
    ('width', '0', 'mono_cpu'), ('height', '-1', 'mono_cpu'),
    ('rate', 'nan', 'mono_cpu'), ('rate', 'inf', 'mono_cpu'), ('rate', '0', 'mono_cpu'),
    ('horizontal_fov', '0', 'mono_cpu'), ('horizontal_fov', '3.15', 'mono_cpu'),
    ('horizontal_fov', 'nan', 'mono_cpu'), ('baseline', '.01', 'stereo'),
    ('baseline', '.26', 'stereo'), ('baseline', 'nan', 'stereo'),
])
def test_invalid_active_camera_values_are_rejected_before_actions(field, value, mode):
    with pytest.raises(ValueError):
        _camera(mode=mode, **{field: value})


def test_inactive_simulation_camera_values_are_not_validated_or_forwarded():
    disabled = _camera(width='0', height='bad', rate='nan', horizontal_fov='0', baseline='nan')
    assert disabled['camera_mode'] == 'disabled'
    assert disabled['camera_baseline'] == '0.08'
    mono = _camera(mode='mono_cpu', baseline='nan')
    assert mono['camera_baseline'] == '0.08'


@pytest.mark.parametrize('behavior', ['search_sweep', 'search_local_then_sweep'])
def test_search_requires_the_backend_that_implements_it(behavior):
    with pytest.raises(ValueError, match='Search requires'):
        _config(servo_backend='standard', idle_behavior=behavior)
    assert _config(idle_behavior=behavior).controller['search_enabled'] is True


def test_rest_simulation_disables_collision_aware_search():
    config = _config(idle_behavior='rest')
    assert config.controller['search_enabled'] is False


def test_compact_speed_scale_loads_from_an_external_controller_yaml(tmp_path):
    source_config = ROOT / 'config/moveit/collision_aware_servo.yaml'
    external_config = tmp_path / 'collision_aware_servo.yaml'
    external_config.write_text(source_config.read_text().replace(
        '  speed_scale: 1.0', '  speed_scale: 0.4'), encoding='utf-8')

    configured = _config(controller_config_file=external_config)

    assert configured.controller['compact_maneuver']['speed_scale'] == pytest.approx(.4)
    assert _config().controller['compact_maneuver']['speed_scale'] == pytest.approx(1.0)


@pytest.mark.parametrize('mode', ['compact', 'disabled'])
def test_recovery_mode_loads_from_an_external_controller_yaml(tmp_path, mode):
    source_config = ROOT / 'config/moveit/collision_aware_servo.yaml'
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load(source_config.read_text())
    data['recovery']['mode'] = mode
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    configured = _config(controller_config_file=external_config)

    assert configured.controller['recovery.mode'] == mode
    assert 'recovery' not in configured.controller
    assert _config().controller['recovery.mode'] == 'compact'


def test_old_controller_yaml_defaults_to_compact_recovery(tmp_path):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    data.pop('recovery')
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    configured = _config(controller_config_file=external_config)

    assert configured.controller['recovery.mode'] == 'compact'


def test_flat_recovery_mode_is_supported_without_nested_mapping(tmp_path):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    data.pop('recovery')
    data['recovery.mode'] = 'disabled'
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    configured = _config(controller_config_file=external_config)

    assert configured.controller['recovery.mode'] == 'disabled'


def test_simulation_rejects_flat_and_nested_recovery_mode_together(tmp_path):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    data['recovery.mode'] = 'disabled'
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    with pytest.raises(ValueError, match='either the flat key or nested recovery mapping'):
        _config(controller_config_file=external_config)


@pytest.mark.parametrize('mode', ['search', '', None, True])
def test_simulation_rejects_unknown_recovery_mode(tmp_path, mode):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    data['recovery']['mode'] = mode
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    with pytest.raises(ValueError, match='recovery.mode must be compact or disabled'):
        _config(controller_config_file=external_config)


def test_simulation_rejects_unknown_recovery_settings(tmp_path):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    data = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    data['recovery']['future_mode'] = 'search'
    external_config.write_text(yaml.safe_dump(data), encoding='utf-8')

    with pytest.raises(ValueError, match='recovery: unknown settings'):
        _config(controller_config_file=external_config)


@pytest.mark.parametrize('scale', [0, -0.1, float('nan'), float('inf'), True])
def test_compact_speed_scale_must_be_finite_and_positive(tmp_path, scale):
    external_config = tmp_path / 'collision_aware_servo.yaml'
    config = yaml.safe_load((ROOT / 'config/moveit/collision_aware_servo.yaml').read_text())
    config['compact_maneuver']['speed_scale'] = scale
    external_config.write_text(yaml.safe_dump(config), encoding='utf-8')

    with pytest.raises(ValueError, match='compact_maneuver.speed_scale'):
        _config(controller_config_file=external_config)


def test_perception_status_gate_is_explicit_and_defaults_off_for_point_targets():
    assert _config().tracking['require_perception_status'] is False
    camera_config = _config(require_perception_status=True)
    assert camera_config.tracking['require_perception_status'] is True


def test_hand_eye_world_uses_a_static_rendered_marker_and_rgbd_sensor():
    world = ET.parse(ROOT / 'worlds/hand_eye_calibration.sdf').getroot().find('world')
    assert world.find("plugin[@filename='gz-sim-sensors-system']") is not None
    marker = world.find("include[name='hand_eye_calibration_marker']")
    assert marker is not None
    assert marker.findtext('uri') == 'model://face_tracking_arm_calibration_marker'
    marker_pose = [float(value) for value in marker.findtext('pose').split()]
    assert marker_pose[:3] == pytest.approx([0.75, 0.0, 1.275])
    assert marker_pose[3:5] == pytest.approx([0.0, -math.pi / 2], abs=1e-9)
    assert math.radians(15) <= abs(marker_pose[5]) <= math.radians(25)

    model = ET.parse(
        ROOT / 'worlds/models/face_tracking_arm_calibration_marker/model.sdf').getroot()
    assert model.find('.//static').text == 'true'
    assert model.find('.//collision') is None
    texture = ROOT / (
        'worlds/models/face_tracking_arm_calibration_marker/materials/textures/aruco_0.png')
    assert texture.is_file()

    bridges = yaml.safe_load((ROOT / 'config/calibration_rgbd_bridge.yaml').read_text())
    assert {(item['ros_topic_name'], item['gz_topic_name']) for item in bridges} == {
        ('/face_test/image_raw', '/face_test/rgbd/image'),
        ('/face_test/camera_info', '/face_test/rgbd/camera_info'),
        ('/face_test/depth/image_raw', '/face_test/rgbd/depth_image'),
    }


def test_ordinary_assembly_keeps_face_source_and_servo_container(monkeypatch):
    actions = _assemble_with_stub_actions(monkeypatch, calibration=False)
    tags = {getattr(action, 'tag', None) for action in actions}
    assert {'servo-container', 'target-marker'} <= tags
    assert 'move-group' not in tags
    assert 'rgbd-bridge' not in tags
    startup = next(action for action in actions if action.tag == 'after:lite6_arm_controller')
    assert [action.tag for action in startup.dependants].count('face-target-source') == 1


def test_calibration_assembly_uses_rendered_world_and_omits_tracking(monkeypatch):
    actions = _assemble_with_stub_actions(monkeypatch, calibration=True)
    tagged = {getattr(action, 'tag', None): action for action in actions}
    assert 'rgbd-bridge' in tagged
    assert 'servo-container' not in tagged
    assert 'face-target-source' not in tagged
    assert 'target-marker' not in tagged
    assert tagged['gz-server'].world == str(ROOT / 'worlds/hand_eye_calibration.sdf')
    startup = tagged['after:lite6_arm_controller']
    assert [action.tag for action in startup.dependants] == ['move-group']


@pytest.mark.parametrize('headless', ['true', 'false'])
def test_exactly_one_gazebo_process_is_selected(headless):
    context = LaunchContext()
    context.launch_configurations['headless'] = headless
    server, gui = sim._gazebo_processes('scene.sdf', LaunchConfiguration('headless'), '/plugins')
    assert server.condition.evaluate(context) is (headless == 'true')
    assert gui.condition.evaluate(context) is (headless == 'false')


@pytest.mark.parametrize('scenario', ['disabled', 'people', 'walk_around'])
@pytest.mark.parametrize('visible', ['true', 'false'])
def test_target_source_and_marker_switch_independently(scenario, visible):
    context = LaunchContext()
    context.launch_configurations.update(test_face_scenario=scenario, visualize_target=visible)
    publisher, marker = sim._target_nodes(
        _config(), LaunchConfiguration('test_face_scenario'),
        LaunchConfiguration('visualize_target'))
    assert publisher.condition.evaluate(context) is (scenario != 'disabled')
    assert marker.condition.evaluate(context) is (visible == 'true')


@pytest.mark.parametrize('code', [0, 1, -15])
@pytest.mark.parametrize('shutdown', [False, True])
def test_startup_advances_only_after_success_and_never_during_shutdown(code, shutdown):
    context = SimpleNamespace(is_shutdown=shutdown)
    event = SimpleNamespace(returncode=code)
    following = object()
    actions = sim._actions_after_successful_exit(
        event, context, actions=[following], prerequisite_name='robot creation')
    if shutdown:
        assert actions == []
    elif code == 0:
        assert actions == [following]
    else:
        assert len(actions) == 1
        assert 'robot creation' in actions[0].event.reason


@pytest.mark.parametrize('code', [0, 1, -15])
def test_required_process_exit_always_stops_scenario_unless_already_stopping(code):
    event = SimpleNamespace(returncode=code)
    running = SimpleNamespace(is_shutdown=False)
    actions = sim._shutdown_after_unexpected_exit(event, running, process_name='Gazebo')
    assert len(actions) == 1 and 'Gazebo' in actions[0].event.reason
    assert sim._shutdown_after_unexpected_exit(
        event, SimpleNamespace(is_shutdown=True), process_name='Gazebo') == []
