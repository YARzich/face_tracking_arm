# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Only fixed tool bodies may ignore contact with each other."""

import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import xacro
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup.hardware_description import robot_parameters  # noqa: E402, I100
from face_tracking_bringup.mount_collisions import allow_fixed_mount_collisions  # noqa: E402
from face_tracking_bringup.robot_profile import get_robot_profile  # noqa: E402


def _pairs(semantic):
    return {frozenset((pair.get('link1'), pair.get('link2')))
            for pair in semantic.findall('disable_collisions')}


def _assert_mount_pairs(semantic, camera, *, monitor=True):
    pairs = _pairs(semantic)
    assert frozenset(('link6', camera)) in pairs
    if monitor:
        assert frozenset(('monitor_link', camera)) in pairs
    for body in ('monitor_link', camera):
        for other in ('link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'table'):
            assert frozenset((body, other)) not in pairs


@pytest.mark.parametrize('omit_monitor', [False, True])
@pytest.mark.parametrize('model', ['xarm6', 'lite6'])
@pytest.mark.parametrize('mode', ['mono_cpu', 'stereo'])
def test_hardware_mount_accepts_camera_next_to_flange(tmp_path, omit_monitor, model, mode):
    config = yaml.safe_load((ROOT / 'config/hardware.yaml').read_text())
    config['robot']['model'] = model
    profile = get_robot_profile(model)
    poses = ET.parse(ROOT / profile['srdf_file']).getroot()
    for name in ('rest', 'search'):
        config['motion'][name + '_joints_deg'] = [
            math.degrees(float(joint.get('value')))
            for joint in poses.find(f"group_state[@name='{name}']").findall('joint')]
    config['camera_mount'].update(body_frame='custom_camera', xyz_m=[0., 0., 0.])
    if omit_monitor:
        config['monitor']['size_m'] = [0., 0., 0.]
    path = tmp_path / 'hardware.yaml'
    path.write_text(yaml.safe_dump(config))
    share = Path(get_package_share_directory('face_tracking_arm'))
    params, _ = robot_parameters(config, mode, share, path, True)
    semantic = ET.fromstring(params[1]['robot_description_semantic'])
    _assert_mount_pairs(semantic, 'custom_camera', monitor=not omit_monitor)


@pytest.mark.parametrize('model', ['xarm6', 'lite6'])
@pytest.mark.parametrize('mode', ['disabled', 'mono', 'stereo'])
def test_simulation_uses_the_same_rigid_mount_boundary(model, mode):
    share = Path(get_package_share_directory('face_tracking_arm'))
    profile = get_robot_profile(model)
    robot = ET.fromstring(xacro.process_file(
        str(share / 'description' / f'{model}_table.urdf.xacro'),
        mappings={'camera_mode': mode}).toxml())
    semantic = ET.parse(share / profile['srdf_file']).getroot()
    before = _pairs(semantic)
    allow_fixed_mount_collisions(robot, semantic)
    if mode == 'disabled':
        assert _pairs(semantic) == before
    else:
        _assert_mount_pairs(semantic, 'monitor_camera_bar')


def test_fixed_paths_cross_empty_frames_but_stop_at_moving_joints():
    robot = ET.fromstring("""<robot name="assembly">
      <link name="wrist"><collision/></link>
      <link name="link6"><collision/></link>
      <link name="empty_mount"/>
      <link name="camera"><collision/></link>
      <link name="tilting_tool"><collision/></link>
      <link name="other_frame"/>
      <joint name="wrist_rotation" type="revolute">
        <parent link="wrist"/><child link="link6"/>
      </joint>
      <joint name="mount" type="fixed">
        <parent link="link6"/><child link="empty_mount"/>
      </joint>
      <joint name="camera_mount" type="fixed">
        <parent link="empty_mount"/><child link="camera"/>
      </joint>
      <joint name="tilt" type="revolute">
        <parent link="camera"/><child link="tilting_tool"/>
      </joint>
      <joint name="optical" type="fixed">
        <parent link="camera"/><child link="other_frame"/>
      </joint>
    </robot>""")
    semantic = ET.fromstring("""<robot name="assembly">
      <disable_collisions link1="wrist" link2="link6" reason="Adjacent"/>
    </robot>""")
    allow_fixed_mount_collisions(robot, semantic)
    expected = {frozenset(('wrist', 'link6')), frozenset(('link6', 'camera'))}
    assert _pairs(semantic) == expected
    first = ET.tostring(semantic)
    allow_fixed_mount_collisions(robot, semantic)
    assert ET.tostring(semantic) == first
    assert semantic.find('disable_collisions').get('reason') == 'Adjacent'


def test_missing_flange_reports_model_mismatch():
    empty = ET.fromstring('<robot name="empty"/>')
    with pytest.raises(ValueError, match='link6'):
        allow_fixed_mount_collisions(empty, empty)
