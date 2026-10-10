# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Mount migration and hand-eye math checks without ROS, a camera or robot."""

from dataclasses import replace
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from face_tracking_bringup import mount_setup_node  # noqa: E402, I100
from face_tracking_bringup.hand_eye_calibration import (  # noqa: E402, I100
    CalibrationSample, detect_square_marker, fixed_marker_consistency,
    planned_joint_offsets_deg, solve_hand_eye, validate_held_out,
    validate_rigid_transform, validate_training_samples)
from face_tracking_bringup.hardware_description import load_yaml  # noqa: E402
from face_tracking_bringup.mount_geometry import (  # noqa: E402
    _inverse, _multiply, migrate_mount_config, rpy_transform)
from face_tracking_bringup.mount_setup import (  # noqa: E402
    apply_manual_geometry, apply_optical_calibration, forward_kinematics_from_urdf,
    manual_measurement_top, measured_pose_in_flange, pose_in_measurement_frame,
    remember_top_direction, save_yaml_atomic, top_alignment_error_deg, top_joint6_goal)


def test_legacy_mount_migration_preserves_camera_pose_and_monitor_pose():
    old = {
        'monitor': {'mount_xyz_m': [0.03, -0.04, 0.12], 'mount_rpy_deg': [10, 20, -30]},
        'camera_mount': {'xyz_m': [0.02, 0.01, 0.08], 'rpy_deg': [4, -8, 13]},
    }
    migrated = migrate_mount_config(old)
    old_camera = _multiply(
        rpy_transform(old['monitor']['mount_xyz_m'], old['monitor']['mount_rpy_deg']),
        rpy_transform(old['camera_mount']['xyz_m'], old['camera_mount']['rpy_deg']))
    new_camera = rpy_transform(migrated['camera_mount']['xyz_m'],
                               migrated['camera_mount']['rpy_deg'])
    assert migrated['mount_geometry_version'] == 2
    assert np.allclose(old_camera, new_camera, atol=1e-12)
    assert old['camera_mount']['xyz_m'] == [0.02, 0.01, 0.08]


def _synthetic_samples():
    flange_from_camera = rpy_transform([.042, -.027, .115], [8, -11, 17])
    base_from_marker = rpy_transform([.30, .0, .90], [178, 2, -6])
    samples = []
    offsets = ((0, 0), (10, 0), (-10, 0), (0, 15), (0, -15), (10, 15), (-10, -15))
    for index, (joint5, joint6) in enumerate(offsets):
        base_from_flange = rpy_transform(
            [.2 + index * .015, -.1 + index * .007, .42 + index * .004],
            [joint5 * .75, joint6 * .8, joint5 * .2 - joint6 * .15])
        camera_from_marker = _multiply(
            _inverse(_multiply(base_from_flange, flange_from_camera)), base_from_marker)
        positions = [0.0, -.3, .4, .1, math.radians(joint5), math.radians(joint6)]
        samples.append(CalibrationSample(
            tuple(positions), tuple(map(tuple, base_from_flange)),
            tuple(map(tuple, camera_from_marker))))
    return samples, flange_from_camera


def test_hand_eye_solver_recovers_static_marker_camera_extrinsic():
    samples, expected = _synthetic_samples()
    actual = np.asarray(solve_hand_eye(samples, .045))
    assert np.allclose(actual, expected, atol=1e-6)
    report = fixed_marker_consistency(samples, actual)
    assert report['position_rms_m'] < 1e-8
    assert report['orientation_rms_deg'] < 1e-3


@pytest.mark.parametrize('samples,side,error', [
    ([], .045, 'six training poses'),
    ([], 0.0, 'side'),
])
def test_hand_eye_refuses_insufficient_or_invalid_marker_data(samples, side, error):
    with pytest.raises(ValueError, match=error):
        validate_training_samples(samples, side)


def test_training_plan_separates_held_out_poses_from_solver_samples():
    training, held_out = planned_joint_offsets_deg()
    assert len(training) == 7
    assert len(held_out) == 4
    assert set(training).isdisjoint(held_out)
    assert all(abs(j5) <= 10 and abs(j6) <= 15 for j5, j6 in training + held_out)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_interruption_handler_is_idempotent_and_restores_handlers(signum):
    """Signals unwind once; later signals cannot interrupt teardown."""
    shutdown = mount_setup_node._DeferredSignalShutdown()
    original_handlers = {
        item: signal.getsignal(item) for item in shutdown._SIGNALS}
    shutdown.install()
    try:
        handler = signal.getsignal(signum)
        with pytest.raises(SystemExit) as error:
            handler(signum, None)
        assert error.value.code == 128 + signum

        for cleanup_signal in shutdown._SIGNALS:
            signal.getsignal(cleanup_signal)(cleanup_signal, None)
        shutdown.begin_cleanup()
        for cleanup_signal in shutdown._SIGNALS:
            signal.getsignal(cleanup_signal)(cleanup_signal, None)
    finally:
        shutdown.begin_cleanup()
        shutdown.restore()

    restored_handlers = {
        item: signal.getsignal(item) for item in shutdown._SIGNALS}
    assert restored_handlers == original_handlers


def _fake_mount_setup_node():
    node = object.__new__(mount_setup_node.MountSetupNode)
    node.active_goal = None
    node.pending_goal_future = None
    node.goal_submission_unknown = False
    return node


def test_pending_acceptance_timeout_is_not_reported_as_stopped(monkeypatch):
    """A missing acceptance response must remain an unknown arm state."""
    node = _fake_mount_setup_node()
    pending = Mock()
    pending.done.return_value = False
    node.pending_goal_future = pending
    logger = Mock()
    monkeypatch.setattr(mount_setup_node.rclpy, 'spin_until_future_complete',
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mount_setup_node.MountSetupNode, 'get_logger', lambda _self: logger)

    assert not node._cancel_active_motion()
    assert node.pending_goal_future is pending
    logger.error.assert_called_once()


def test_motion_stores_goal_acceptance_future_before_waiting():
    """Cleanup sees the request when interrupted during acceptance."""
    node = _fake_mount_setup_node()
    node.args = SimpleNamespace(operation='calibrate')
    node.limits = [(-1.0, 1.0)] * 6
    node.joint_names = [f'joint{i}' for i in range(1, 7)]
    node._joint_snapshot = Mock(return_value=(None, [0.0] * 6, None))
    node._validate_path = Mock()
    node.action = Mock()
    node.action.wait_for_server.return_value = True
    pending = Mock()
    shutdown = mount_setup_node._DeferredSignalShutdown()
    node.signal_shutdown = shutdown
    shutdown.install()
    try:
        def interrupt_inside_send(_goal):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(.05)
            return pending

        node.action.send_goal_async.side_effect = interrupt_inside_send
        with pytest.raises(SystemExit) as error:
            node._move([0.0] * 6)
        assert error.value.code == 128 + signal.SIGINT
    finally:
        shutdown.begin_cleanup()
        shutdown.restore()

    assert node.pending_goal_future is pending


@pytest.mark.parametrize('delayed_info,sequence,previous_sequence', [
    (False, 4, 4),  # A dropped post-motion frame leaves only the old sequence.
    (True, 5, 4),   # CameraInfo arrives outside the image synchronization window.
])
def test_capture_times_out_without_a_new_synchronized_frame(
        monkeypatch, delayed_info, sequence, previous_sequence):
    """Dropped or delayed inputs must not reuse a pre-motion image."""
    node = object.__new__(mount_setup_node.MountSetupNode)
    node.lock = threading.Lock()

    def stamp(nanosec):
        return SimpleNamespace(sec=1, nanosec=nanosec)

    image = SimpleNamespace(
        header=SimpleNamespace(frame_id='camera', stamp=stamp(0)),
        width=640, height=480)
    info = SimpleNamespace(
        header=SimpleNamespace(frame_id='camera', stamp=stamp(60_000_000 if delayed_info else 0)),
        width=640, height=480)
    joints = SimpleNamespace(
        header=SimpleNamespace(frame_id='', stamp=stamp(0)))
    node.latest_image = image
    node.latest_info = info
    node.latest_joint_state = joints
    node.image_sequence = sequence
    node.camera = {'frame_id': 'camera'}
    node._spin_for = Mock()

    clock = [0.0]

    def monotonic():
        now = clock[0]
        clock[0] += 1.0
        return now

    monkeypatch.setattr(mount_setup_node.time, 'monotonic', monotonic)
    monkeypatch.setattr(mount_setup_node.rclpy, 'ok', lambda: True)

    with pytest.raises(ValueError, match='not visible in synchronized fresh frames'):
        node._capture_one(.045, previous_sequence)

    assert clock[0] >= 16.0
    node._spin_for.assert_called()


def test_near_limit_calibration_target_is_rejected_before_motion():
    """A planned J5 offset beyond the active limit must not call the mover."""
    node = object.__new__(mount_setup_node.MountSetupNode)
    node.limits = [(-1.0, 1.0)] * 6
    node.calibration_start = [0.0, 0.0, 0.0, 0.0, 1.0 - math.radians(5), 0.0]
    target = node.calibration_start.copy()
    target[4] += math.radians(10)
    node._joint_snapshot = Mock()
    node._move = Mock()

    with pytest.raises(ValueError, match='outside the active joint limits'):
        node._capture_at(target, .045, 0)

    node._joint_snapshot.assert_not_called()
    node._move.assert_not_called()


def test_unknown_goal_submission_is_not_reported_as_stopped(monkeypatch):
    node = _fake_mount_setup_node()
    node.goal_submission_unknown = True
    logger = Mock()
    monkeypatch.setattr(mount_setup_node.MountSetupNode, 'get_logger',
                        lambda _self: logger)

    assert not node._cancel_active_motion()
    logger.error.assert_called_once()


@pytest.mark.parametrize('stopped,expected', [(True, True), (False, False)])
def test_rejected_pending_goal_requires_fresh_stopped_feedback(
        monkeypatch, stopped, expected):
    """A rejected action is clean only with fresh stationary joint feedback."""
    node = _fake_mount_setup_node()
    pending = Mock()
    pending.done.return_value = True
    pending.result.return_value = SimpleNamespace(accepted=False)
    node.pending_goal_future = pending
    monkeypatch.setattr(mount_setup_node.rclpy, 'spin_until_future_complete',
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mount_setup_node.MountSetupNode, '_wait_until_still',
                        lambda _self: stopped)
    monkeypatch.setattr(mount_setup_node.MountSetupNode, 'get_logger',
                        lambda _self: Mock())

    assert node._cancel_active_motion() is expected
    assert node.pending_goal_future is None


def test_accepted_pending_goal_is_cancelled_and_stopped(monkeypatch):
    """Promote and cancel an accepted pending action."""
    from action_msgs.msg import GoalStatus

    node = _fake_mount_setup_node()
    cancel_future = Mock()
    cancel_future.done.return_value = True
    cancel_future.result.return_value = object()
    result_future = Mock()
    result_future.done.return_value = True
    result_future.result.return_value = SimpleNamespace(status=GoalStatus.STATUS_CANCELED)
    goal_handle = Mock(accepted=True)
    goal_handle.cancel_goal_async.return_value = cancel_future
    goal_handle.get_result_async.return_value = result_future
    pending = Mock()
    pending.done.return_value = True
    pending.result.return_value = goal_handle
    node.pending_goal_future = pending
    monkeypatch.setattr(mount_setup_node.rclpy, 'spin_until_future_complete',
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mount_setup_node.MountSetupNode, '_wait_until_still',
                        lambda _self: True)

    assert node._cancel_active_motion()
    assert node.pending_goal_future is None
    assert node.active_goal is None
    goal_handle.cancel_goal_async.assert_called_once_with()


def test_manual_measurements_use_named_forward_left_up_axes():
    mount = measured_pose_in_flange([1, 0, 0], [.2, .03, .1], [0, 0, 0])
    # With saved top along flange +X, forward is +Z and left is -Y.
    assert np.allclose([mount[0][3], mount[1][3], mount[2][3]], [.1, -.03, .2])
    assert np.allclose([mount[0][2], mount[1][2], mount[2][2]], [1, 0, 0])


def test_manual_frame_defaults_to_configured_monitor_top_and_round_trips_pose():
    config = {'monitor': {'mount_rpy_deg': [180.0, -90.0, 0.0]}}
    top, source = manual_measurement_top(config)
    assert top == pytest.approx([1.0, 0.0, 0.0], abs=1e-8)
    assert source == 'configured monitor up axis'

    original_xyz = [.12525, 0.0, .005]
    original_rpy = [180.0, -90.0, 0.0]
    measured_xyz, measured_rpy = pose_in_measurement_frame(
        top, original_xyz, original_rpy)
    reconstructed = measured_pose_in_flange(top, measured_xyz, measured_rpy)
    assert np.allclose(reconstructed, rpy_transform(original_xyz, original_rpy), atol=1e-9)


def test_fk_from_urdf_includes_world_placement_and_joint_orientation():
    urdf = """<robot name="test">
      <link name="world"/><link name="base"/><link name="arm"/><link name="link6"/>
      <joint name="world_joint" type="fixed"><parent link="world"/>
        <child link="base"/><origin xyz="1 2 0" rpy="0 0 0"/></joint>
      <joint name="joint1" type="revolute"><parent link="base"/>
        <child link="arm"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="joint2" type="fixed"><parent link="arm"/>
        <child link="link6"/><origin xyz="1 0 0"/></joint>
    </robot>"""
    result = forward_kinematics_from_urdf(urdf, [math.pi / 2, 0, 0, 0, 0, 0])
    expected = rpy_transform([1, 3, 0], [0, 0, 90])
    assert np.allclose(result, expected, atol=1e-12)


def test_marker_detection_accepts_camera_info_lists_from_ros_message():
    cv2 = pytest.importorskip('cv2')
    aruco = getattr(cv2, 'aruco', None)
    if aruco is None:
        pytest.skip('OpenCV ArUco marker generator is unavailable')
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, 'generateImageMarker'):
        marker = aruco.generateImageMarker(dictionary, 0, 180)
    elif hasattr(aruco, 'drawMarker'):
        marker = aruco.drawMarker(dictionary, 0, 180)
    else:
        pytest.skip('OpenCV ArUco marker generator is unavailable')
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    image[150:330, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    result = detect_square_marker(
        image, [[600, 0, 320], [0, 600, 240], [0, 0, 1]], [0, 0, 0, 0, 0], .045)
    assert result is not None
    assert np.asarray(result[0])[2, 3] > 0


def test_marker_detection_refines_rendered_gray_panel_to_physical_square():
    cv2 = pytest.importorskip('cv2')
    aruco = getattr(cv2, 'aruco', None)
    if (aruco is None or not hasattr(aruco, 'CORNER_REFINE_APRILTAG') or
            not hasattr(aruco, 'estimatePoseSingleMarkers')):
        pytest.skip('OpenCV ArUco AprilTag refinement is unavailable')

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, 'generateImageMarker'):
        marker = aruco.generateImageMarker(dictionary, 0, 900)
    elif hasattr(aruco, 'drawMarker'):
        marker = aruco.drawMarker(dictionary, 0, 900)
    else:
        pytest.skip('OpenCV ArUco marker generator is unavailable')

    # Match the Gazebo asset: a 45 mm marker on a 60 mm gray panel, rendered
    # with a 12.5% white quiet zone on all sides.
    texture = np.full((1200, 1200), 100, dtype=np.uint8)
    marker_region = texture[150:1050, 150:1050]
    marker_region[marker == 0] = 0
    K = np.array([[734.8766486413382, 0, 640],
                  [0, 734.8766486413382, 360], [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros(5, dtype=np.float64)
    camera_from_marker = np.eye(4)
    face_camera_rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                                    dtype=np.float64)
    marker_tilt = cv2.Rodrigues(np.array([0, math.radians(20), 0]))[0]
    camera_from_marker[:3, :3] = face_camera_rotation @ marker_tilt
    camera_from_marker[:3, 3] = [0, 0, .42]
    rotation, _ = cv2.Rodrigues(camera_from_marker[:3, :3])
    plane_points = np.array([[-.03, .03, 0], [.03, .03, 0],
                             [.03, -.03, 0], [-.03, -.03, 0]], dtype=np.float64)
    panel_corners, _ = cv2.projectPoints(
        plane_points, rotation, camera_from_marker[:3, 3], K, distortion)
    source_corners = np.array([[0, 0], [1199, 0], [1199, 1199], [0, 1199]],
                              dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_corners,
                                            panel_corners.reshape(4, 2).astype(np.float32))
    image = cv2.warpPerspective(
        cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR), transform, (1280, 720),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(218, 218, 218))

    detected = detect_square_marker(image, K, distortion, .045)
    assert detected is not None
    estimated, corners = detected
    parameters = (aruco.DetectorParameters() if hasattr(aruco, 'ArucoDetector') else
                  aruco.DetectorParameters_create())
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    if hasattr(aruco, 'ArucoDetector'):
        detected_corners, ids, _ = aruco.ArucoDetector(dictionary, parameters).detectMarkers(
            image)
    else:
        detected_corners, ids, _ = aruco.detectMarkers(
            image, dictionary, parameters=parameters)
    assert ids is not None
    index = int(np.flatnonzero(ids.flatten() == 0)[0])
    expected_refined_corners = np.ascontiguousarray(
        detected_corners[index].reshape(1, 4, 2), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001)
    cv2.cornerSubPix(gray, expected_refined_corners, (5, 5), (-1, -1), criteria)
    assert np.allclose(corners, expected_refined_corners.reshape(4, 2), atol=1e-4)
    refined_pose_rvec, refined_pose_tvec, _ = aruco.estimatePoseSingleMarkers(
        [np.asarray(corners, dtype=np.float32)], .045, K, distortion)
    expected_pose_rotation, _ = cv2.Rodrigues(refined_pose_rvec[0, 0])
    expected_pose = np.eye(4)
    expected_pose[:3, :3] = expected_pose_rotation
    expected_pose[:3, 3] = refined_pose_tvec[0, 0]
    assert np.allclose(estimated, expected_pose, atol=1e-7)
    marker_points = np.array([[-.0225, .0225, 0], [.0225, .0225, 0],
                              [.0225, -.0225, 0], [-.0225, -.0225, 0]],
                             dtype=np.float64)
    expected_corners, _ = cv2.projectPoints(
        marker_points, rotation, camera_from_marker[:3, 3], K, distortion)
    corner_error = np.linalg.norm(
        np.asarray(corners) - expected_corners.reshape(4, 2), axis=1)
    assert max(corner_error) < 2.0
    assert np.linalg.norm(np.asarray(estimated)[:3, 3] - camera_from_marker[:3, 3]) < .005


def test_setup_node_config_loader_accepts_launch_string_path(tmp_path):
    config_path = tmp_path / 'resolved.yaml'
    config_path.write_text('operation: calibrate\n')
    assert load_yaml(str(config_path)) == {'operation': 'calibrate'}


def test_manual_geometry_persists_flange_relative_measurements():
    config = {'monitor': {}, 'camera_mount': {}, 'mount_geometry_version': 2}
    updated = apply_manual_geometry(config, [1, 0, 0], [.2, .03, .1], [0, 0, 0],
                                    [.4, -.02, .15], [0, 0, 0])
    assert updated['monitor']['mount_xyz_m'] == pytest.approx([.1, -.03, .2])
    assert updated['camera_mount']['xyz_m'] == pytest.approx([.15, .02, .4])


def test_remember_top_uses_full_pose_and_set_top_changes_only_j6():
    base_from_flange = rpy_transform([.2, 0, .4], [0, 90, math.radians(35)])
    top = remember_top_direction(base_from_flange, [0, 0, 0, 0, .2, .6])
    assert top['remembered_joint_positions_deg'] == pytest.approx(
        [0, 0, 0, 0, math.degrees(.2), math.degrees(.6)])
    assert np.allclose(top['preferred_top_flange'], [-1, 0, 0])
    target = top_joint6_goal(base_from_flange, [0, 0, 0, 0, .2, .6],
                             top['preferred_top_flange'], (-math.pi, math.pi))
    assert target == pytest.approx(.6)


def test_remember_top_rejects_vertical_flange_axis():
    with pytest.raises(ValueError, match='vertical'):
        remember_top_direction(rpy_transform([0, 0, 0], [0, 0, 0]), [0] * 6)


def test_set_top_keeps_best_j6_alignment_even_when_tilt_remains():
    flange_pose = rpy_transform([0, 0, 0], [60, 0, 0])
    top = [1, 0, 0]
    joints = [0, 0, 0, 0, 0, 0]
    goal = top_joint6_goal(flange_pose, joints, top, (-math.pi, math.pi))
    assert -math.pi <= goal <= math.pi
    assert top_alignment_error_deg(flange_pose, joints, top, goal) > 15


def test_hand_eye_result_updates_camera_body_after_removing_optical_offset():
    config = {
        'monitor': {'mount_xyz_m': [0, 0, .1], 'mount_rpy_deg': [0, 0, 0]},
        'camera_mount': {'xyz_m': [0, 0, .2], 'rpy_deg': [0, 0, 0]},
        'mono_cpu': {'left': {'optical_xyz_m': [.01, 0, 0],
                              'optical_rpy_deg': [-90, 0, -90]}},
    }
    flange_from_optical = rpy_transform([.2, .1, .3], [20, -30, 40])
    updated = apply_optical_calibration(config, 'mono_cpu', flange_from_optical)
    body_from_optical = rpy_transform([.01, 0, 0], [-90, 0, -90])
    recovered = _multiply(
        rpy_transform(updated['camera_mount']['xyz_m'],
                      updated['camera_mount']['rpy_deg']), body_from_optical)
    assert np.allclose(recovered, flange_from_optical, atol=1e-12)
    assert updated['monitor'] == config['monitor']


def test_hand_eye_numpy_result_round_trips_as_plain_yaml_scalars(tmp_path):
    config = {
        'mount_geometry_version': 2,
        'camera_mount': {'xyz_m': [0.1, 0.0, 0.0], 'rpy_deg': [0.0, 0.0, 0.0]},
        'mono_cpu': {'left': {'optical_xyz_m': [0.01, 0.0, 0.0],
                              'optical_rpy_deg': [-90.0, 0.0, -90.0]}},
    }
    flange_from_optical = np.eye(4, dtype=np.float64)
    flange_from_optical[:3, 3] = np.array([0.12757581538334134, 0.02, 0.03])

    updated = apply_optical_calibration(config, 'mono_cpu', flange_from_optical)
    assert all(type(value) is float for value in updated['camera_mount']['xyz_m'])
    assert all(type(value) is float for value in updated['camera_mount']['rpy_deg'])

    output_path = tmp_path / 'calibrated.yaml'
    save_yaml_atomic(output_path, updated)
    reloaded = load_yaml(str(output_path))
    assert reloaded['camera_mount']['xyz_m'] == pytest.approx(
        updated['camera_mount']['xyz_m'])
    assert reloaded['camera_mount']['rpy_deg'] == pytest.approx(
        updated['camera_mount']['rpy_deg'])


def test_hand_eye_requires_rigid_pose_j5_j6_only_and_synchronized_stamps():
    samples, _ = _synthetic_samples()
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match=r'SO\(3\)'):
        validate_rigid_transform(reflection, 'test')
    moved_other_joint = list(samples)
    first = samples[1]
    positions = list(first.joint_positions_rad)
    positions[0] += math.radians(1)
    moved_other_joint[1] = CalibrationSample(
        tuple(positions), first.base_from_flange, first.camera_from_marker)
    with pytest.raises(ValueError, match='J1-J4'):
        validate_training_samples(moved_other_joint, .045)
    stale_pair = list(samples)
    first = samples[1]
    stale_pair[1] = CalibrationSample(
        first.joint_positions_rad, first.base_from_flange, first.camera_from_marker,
        image_stamp_sec=1.0, joint_state_stamp_sec=1.06)
    with pytest.raises(ValueError, match='stamps'):
        validate_training_samples(stale_pair, .045)


def test_held_out_acceptance_checks_static_pose_and_marker_projection():
    import cv2

    samples, flange_from_camera = _synthetic_samples()
    intrinsics = np.asarray([[520, 0, 320], [0, 520, 240], [0, 0, 1]], dtype=float)
    distortion = np.zeros(5)
    object_points = np.asarray([
        [-.0225, .0225, 0], [.0225, .0225, 0],
        [.0225, -.0225, 0], [-.0225, -.0225, 0]], dtype=float)
    held_out = []
    held_out_offsets = ((7.5, 0), (-7.5, 0), (0, 7.5), (0, -7.5))
    for sample, (j5_deg, j6_deg) in zip(samples[-4:], held_out_offsets):
        rotation, _ = cv2.Rodrigues(np.asarray(sample.camera_from_marker)[:3, :3])
        corners, _ = cv2.projectPoints(object_points, rotation,
                                       np.asarray(sample.camera_from_marker)[:3, 3],
                                       intrinsics, distortion)
        joints = list(samples[0].joint_positions_rad)
        joints[4] += math.radians(j5_deg)
        joints[5] += math.radians(j6_deg)
        held_out.append(CalibrationSample(
            tuple(joints), sample.base_from_flange,
            sample.camera_from_marker, tuple(map(tuple, corners.reshape(4, 2)))))
    report = validate_held_out(samples, held_out, flange_from_camera,
                               intrinsics, distortion, .045)
    assert report['accepted'] is True
    assert report['position_rms_m'] < 1e-8
    assert report['reprojection_rms_px'] < 1e-8


def test_held_out_reprojection_uses_training_marker_reference():
    import cv2

    samples, flange_from_camera = _synthetic_samples()
    intrinsics = np.asarray([[520, 0, 320], [0, 520, 240], [0, 0, 1]], dtype=float)
    object_points = np.asarray([
        [-.0225, .0225, 0], [.0225, .0225, 0],
        [.0225, -.0225, 0], [-.0225, -.0225, 0]], dtype=float)
    held_out = []
    base_from_marker = rpy_transform([.30, .0, .90], [178, 2, -6])
    for index, (j5, j6) in enumerate(((7.5, 0), (-7.5, 0), (0, 7.5), (0, -7.5))):
        base_from_flange = rpy_transform(
            [.22 + index * .013, -.08 + index * .009, .44 + index * .006],
            [j5 * .75, j6 * .8, j5 * .2 - j6 * .15])
        marker = np.asarray(_multiply(
            _inverse(_multiply(base_from_flange, flange_from_camera)), base_from_marker))
        rotation, _ = cv2.Rodrigues(marker[:3, :3])
        corners, _ = cv2.projectPoints(object_points, rotation, marker[:3, 3],
                                       intrinsics, np.zeros(5))
        joints = list(samples[0].joint_positions_rad)
        joints[4] += math.radians(j5)
        joints[5] += math.radians(j6)
        held_out.append(replace(samples[0], joint_positions_rad=tuple(joints),
                                base_from_flange=tuple(map(tuple, base_from_flange)),
                                camera_from_marker=tuple(map(tuple, marker)),
                                marker_corners_px=tuple(map(tuple, corners.reshape(4, 2)))))
    biased = np.asarray(flange_from_camera).copy()
    biased[0, 3] += .10
    with pytest.raises(ValueError, match='reprojection RMS'):
        validate_held_out(samples, held_out, biased, intrinsics, np.zeros(5), .045,
                          max_position_rms_m=10.0, max_position_m=10.0,
                          max_orientation_rms_deg=180.0)


def test_held_out_refuses_camera_info_change_and_nonfinite_joint_values():
    samples, flange_from_camera = _synthetic_samples()
    intrinsics = np.asarray([[520, 0, 320], [0, 520, 240], [0, 0, 1]], dtype=float)
    held = replace(samples[1], camera_matrix=(1.0,))
    with pytest.raises(ValueError, match='Camera calibration'):
        validate_held_out(samples, [held, samples[2]], flange_from_camera,
                          intrinsics, np.zeros(5), .045)
    invalid = list(samples)
    positions = list(invalid[1].joint_positions_rad)
    positions[5] = float('nan')
    invalid[1] = replace(invalid[1], joint_positions_rad=tuple(positions))
    with pytest.raises(ValueError, match='finite'):
        validate_training_samples(invalid, .045)
