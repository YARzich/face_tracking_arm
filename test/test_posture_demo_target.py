# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Check the posture demo's measured side metric against simple geometry."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / 'tools/posture_demo_target.py'
TARGET_SPEC = importlib.util.spec_from_file_location(
    'posture_demo_target', TARGET_PATH)
target = importlib.util.module_from_spec(TARGET_SPEC)
TARGET_SPEC.loader.exec_module(target)
LAUNCH_PATH = ROOT / 'tools/posture_demo.launch.py'
LAUNCH_SPEC = importlib.util.spec_from_file_location(
    'posture_demo_launch', LAUNCH_PATH)
launch = importlib.util.module_from_spec(LAUNCH_SPEC)
LAUNCH_SPEC.loader.exec_module(launch)


@pytest.mark.parametrize(('face_y', 'expected_s'), [
    (0.4, -0.1),
    (-0.4, 0.1),
])
def test_side_metric_changes_sign_when_face_crosses_base_wrist_line(
        face_y, expected_s):
    # The elbow is +Y from BW. A face on -Y is the preferred side (s > 0).
    c, signed_clearance, radius = target._side_metrics(
        (0.0, 0.0, 0.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0),
        (0.5, face_y, 0.0))

    assert c == pytest.approx(0.1 * face_y)
    assert signed_clearance == pytest.approx(expected_s)
    assert radius == pytest.approx(abs(face_y))


@pytest.mark.parametrize(('quaternion', 'expected_axis'), [
    (SimpleNamespace(x=math.sqrt(0.5), y=0.0, z=0.0, w=math.sqrt(0.5)),
     (0.0, -1.0, 0.0)),
    (SimpleNamespace(x=0.0, y=math.sqrt(0.5), z=0.0, w=math.sqrt(0.5)),
     (1.0, 0.0, 0.0)),
])
def test_rotate_z_returns_world_rotation_matrix_column(
        quaternion, expected_axis):
    assert target._rotate_z(quaternion) == pytest.approx(expected_axis)


def test_side_metric_reports_undefined_when_target_lies_on_base_wrist_line():
    c, signed_clearance, radius = target._side_metrics(
        (0.0, 0.0, 0.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0),
        (0.5, 0.0, 0.0))

    assert c == pytest.approx(0.0)
    assert math.isnan(signed_clearance)
    assert radius == pytest.approx(0.0)


def test_crossing_window_returns_offsets_on_opposite_oracle_sides():
    screen_perpendicular = (0.0, 0.2, 0.0)
    elbow_side = (0.0, 1.0, 0.0)
    amplitude = target._crossing_window(
        screen_perpendicular, elbow_side, radius=0.5, near=0.005)
    values = []
    for side_step in (amplitude, -amplitude):
        face = (0.5, side_step, 0.0)
        c, signed_clearance, _ = target._side_metrics(
            (0.0, 0.0, 0.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0), face)
        values.append((c, signed_clearance))

    assert amplitude == pytest.approx(0.12)
    assert values[0][0] * values[1][0] < 0.0
    assert values[0][1] * values[1][1] < 0.0


def test_crossing_window_reports_when_both_oracle_sides_do_not_fit():
    with pytest.raises(ValueError, match='both sides'):
        target._crossing_window((0.0, 0.497, 0.0), (0.0, 1.0, 0.0),
                                radius=0.5, near=0.005)


def test_crossing_targets_flip_measured_side_and_stay_on_screen_distance():
    node = target.DemoTarget.__new__(target.DemoTarget)
    positions = {
        'link_base': (0.0, 0.0, 0.0),
        'link3': (0.5, 0.1, 0.0),
        'link4': (1.0, 0.0, 0.0),
    }

    def lookup(_world, frame, _time):
        x, y, z = positions[frame]
        translation = SimpleNamespace(x=x, y=y, z=z)
        return SimpleNamespace(transform=SimpleNamespace(
            translation=translation))

    node.tf = SimpleNamespace(lookup_transform=lookup)
    screen = (0.5, 0.2, 0.0)
    node._lookup_screen = lambda: (screen, (1.0, 0.0, 0.0))
    node.crossing_distance = 0.5
    node.crossing_near = 0.005
    node.crossing_beta_sign = 1.0

    positive = node._crossing_point(0.1)
    negative = node._crossing_point(-0.1)
    c_positive, s_positive, _ = target._side_metrics(
        positions['link_base'], positions['link3'], positions['link4'], positive)
    c_negative, s_negative, _ = target._side_metrics(
        positions['link_base'], positions['link3'], positions['link4'], negative)

    assert c_positive * c_negative < 0.0
    assert s_positive * s_negative < 0.0
    assert target._distance(screen, positive) == pytest.approx(0.5)
    assert target._distance(screen, negative) == pytest.approx(0.5)


@pytest.mark.parametrize(('available', 'message'), [
    (True, 'следует за текущей геометрией BW'),
    (False, 'зафиксирована в world и не следует за экраном'),
])
def test_crossing_schedule_log_matches_target_behavior(capsys,
                                                       available, message):
    node = target.DemoTarget.__new__(target.DemoTarget)
    node.args = SimpleNamespace(scenario='crossing')
    node.crossing_available = available

    node._begin_schedule(1_000_000_000)

    assert message in capsys.readouterr().out


def test_fixture_matching_flags_seed_mismatch_without_mutating_expected_seed():
    fixture = (0.0, 1.8, -3.8, 0.0, 0.4292, 0.0)
    close_state = (0.0, 1.8, -3.8, 0.0, 0.4292, 0.019)
    wrong_state = (0.0, 1.8, -3.8, 0.0, 0.4292, 0.021)

    assert target._fixture_matches(close_state, fixture)
    assert not target._fixture_matches(wrong_state, fixture)
    assert fixture == (0.0, 1.8, -3.8, 0.0, 0.4292, 0.0)


def test_bad_demo_fixtures_are_exact_approved_reference_poses():
    reference = json.loads(
        (ROOT / 'docs/reference/posture1c/poses.json').read_text())
    poses = {pose['id']: pose for pose in reference['poses']}
    for pose_id, actual in launch.BAD_FIXTURES.items():
        expected = tuple(poses[pose_id]['joints'][f'joint{i}']
                         for i in range(1, 7))
        assert actual == pytest.approx(expected, abs=1e-12)
        assert poses[pose_id]['label'] == 'BAD'


def test_hinge_comfort_matches_model_named_pose_branch_and_human_fixtures():
    reference = json.loads(
        (ROOT / 'docs/reference/posture1c/poses.json').read_text())
    poses = {pose['id']: pose for pose in reference['poses']}

    def geometry(pose):
        chain = pose['mainchain']
        rotation = pose['frames_at_q']['link3']
        axis = [rotation[row][2] for row in range(3)]
        return target._hinge_geometry(
            chain['S_link2_m'], chain['E_link3_m'], chain['W_link4_m'], axis)

    srdf = target._named_joint3_positions(ROOT / 'config/moveit/xarm6.srdf')
    profile_path = ROOT / 'python/face_tracking_bringup/robot_profile.py'
    profile_spec = importlib.util.spec_from_file_location(
        'robot_profile', profile_path)
    profile = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile)
    limits = profile.get_robot_profile('xarm6')['position_limits_rad'][2]
    reference_pose = poses[10]
    offset = geometry(reference_pose)[0] - reference_pose['joints']['joint3']
    sign, minimum_sine, lower, upper = target._joint3_comfort(
        srdf['rest'], srdf['search'], offset, limits[0], limits[1])
    calibration = {
        'angle_offset': offset,
        'preferred_sign': sign,
        'minimum_sine': minimum_sine,
        'joint3_lower': lower,
        'joint3_upper': upper,
    }

    assert sign == 1.0
    assert minimum_sine == pytest.approx(0.703, abs=0.01)
    for pose_id, pose in poses.items():
        metrics = target._hinge_bend_metrics(
            pose['mainchain']['S_link2_m'],
            pose['mainchain']['E_link3_m'],
            pose['mainchain']['W_link4_m'],
            [pose['frames_at_q']['link3'][row][2] for row in range(3)],
            pose['joints']['joint3'], calibration)
        assert metrics['calibration_consistent']
        assert metrics['comfortable'] is (pose['label'] == 'GOOD'), pose_id


def test_startup_calibration_rejects_joint_and_latest_tf_from_different_ticks():
    # Startup q3=-3.8 at t=20ms; a latest-TF lookup 1 tick later would add
    # the observed 0.42rad transient to the calibrated angle offset.
    q3 = -3.8
    correct_offset = 2.7331842691
    synchronized_angle = q3 + correct_offset
    stale_q3 = q3
    later_tf_angle = synchronized_angle + 0.4204

    with pytest.raises(ValueError, match='stamps differ'):
        target._synchronized_angle_offset(
            later_tf_angle, stale_q3, 20_000_000, 30_000_000)

    assert target._synchronized_angle_offset(
        synchronized_angle, q3, 20_000_000, 20_000_000) == pytest.approx(
            correct_offset)


def test_demo_calibration_looks_up_all_fk_at_joint_state_timestamp(monkeypatch):
    node = target.DemoTarget.__new__(target.DemoTarget)
    node.latest_joint_stamp_ns = 20_000_000
    node.joint_state_history = target.deque(maxlen=32)
    node.args = SimpleNamespace(
        srdf_path=str(ROOT / 'config/moveit/xarm6.srdf'))
    node.bend_calibration = None
    requested = []
    q3 = -3.8
    expected_offset = 2.7331842691
    angle = q3 + expected_offset
    translation = {
        'link2': (0.0, 0.0, 0.0),
        'link3': (1.0, 0.0, 0.0),
        'link4': (1.0 + math.cos(angle), math.sin(angle), 0.0),
    }

    def lookup(_world, frame, stamp):
        requested.append(stamp.nanoseconds)
        position = translation[frame]
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(
                sec=0, nanosec=20_000_000)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(
                    x=position[0], y=position[1], z=position[2]),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    node.tf = SimpleNamespace(lookup_transform=lookup)
    joints = (0.0, 1.8, q3, 0.0, 0.4292, 0.0)
    node.joint_state_history.append((20_000_000, joints))
    monkeypatch.setattr(target, '_physical_joint3_limits',
                        lambda: (-3.92699, 0.191986))
    sample = node._synchronized_hinge_sample()
    node._calibrate_bend(sample)

    assert requested == [20_000_000, 20_000_000, 20_000_000]
    assert node.bend_calibration['angle_offset'] == pytest.approx(
        expected_offset)


def test_synchronized_sample_falls_back_from_future_latest_to_previous_tf():
    node = target.DemoTarget.__new__(target.DemoTarget)
    node.joint_state_history = target.deque(maxlen=32)
    previous = (0.0, 1.8, -3.8, 0.0, 0.4292, 0.0)
    latest = (0.0, 1.7, -3.7, 0.0, 0.4292, 0.0)
    node.joint_state_history.extend(((20_000_000, previous),
                                     (30_000_000, latest)))
    requested = []
    positions = {'link2': (0.0, 0.0, 0.0),
                 'link3': (1.0, 0.0, 0.0),
                 'link4': (1.0, 0.0, 1.0)}

    def lookup(_world, frame, stamp):
        requested.append((frame, stamp.nanoseconds))
        if stamp.nanoseconds == 30_000_000:
            raise RuntimeError('latest JointState is ahead of TF')
        x, y, z = positions[frame]
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(
                sec=0, nanosec=20_000_000)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=x, y=y, z=z),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    node.tf = SimpleNamespace(lookup_transform=lookup)
    sample = node._synchronized_hinge_sample()

    assert sample[0] == 20_000_000
    assert sample[1] == previous
    assert requested[0] == ('link2', 30_000_000)
    assert requested[-3:] == [('link2', 20_000_000),
                              ('link3', 20_000_000),
                              ('link4', 20_000_000)]


def test_joint_state_sample_history_is_bounded():
    node = target.DemoTarget.__new__(target.DemoTarget)
    node.joint_state_history = target.deque(
        maxlen=target.JOINT_STATE_HISTORY_SIZE)
    joints = (0.0,) * 6

    for stamp_ns in range(target.JOINT_STATE_HISTORY_SIZE + 3):
        node.joint_state_history.append((stamp_ns, joints))

    assert len(node.joint_state_history) == target.JOINT_STATE_HISTORY_SIZE
    assert node.joint_state_history[0][0] == 3


def test_fk_shape_remains_visible_when_joint_state_consistency_warns():
    angle = math.asin(0.8)
    metrics = target._hinge_bend_metrics(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + math.cos(angle), math.sin(angle), 0.0),
        (0.0, 0.0, 1.0), -1.0,
        {'preferred_sign': 1.0, 'angle_offset': 2.7331842691,
         'minimum_sine': 0.7, 'joint3_lower': -1.95,
         'joint3_upper': -0.37})

    assert metrics['normal_branch']
    assert metrics['comfortable']
    assert not metrics['calibration_consistent']
    text = target._bend_status_text(metrics, -1.0)
    assert 'обычная/в комфорте/невыпрямленная' in text
    assert 'sin=+0.800' in text
    assert 'q3/TF mismatch' in text


def test_seed_label_preserves_human_reference_after_robot_moves():
    assert target._seed_label(10) == 'эталон BAD10'
    assert target._seed_label(5) == 'эталон GOOD5'


def test_seed_mismatch_keeps_a_world_target_available():
    node = target.DemoTarget.__new__(target.DemoTarget)
    node.args = SimpleNamespace(scenario='posture', minimum=0.37, maximum=0.43)
    node.fixture = (0.0, 1.8, -3.8, 0.0, 0.4292, 0.0)
    node.tf = SimpleNamespace(
        lookup_transform=lambda *_args: (_ for _ in ()).throw(
            RuntimeError('test has no TF')))
    node.failed = False
    node.initial_position = None
    node.initial_normal = None
    node.previous_target = None
    node.phase_target = None

    node._seed_target(
        (0.0, 1.8, -3.8, 0.0, 0.4292, 0.03),
        (0.4, 0.0, 1.0), (1.0, 0.0, 0.0))

    assert not node.failed
    assert node.initial_position == (0.4, 0.0, 1.0)
    assert node.phase_target == pytest.approx((0.8, 0.0, 1.0))
