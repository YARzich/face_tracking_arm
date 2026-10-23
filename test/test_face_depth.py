# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Depth geometry and stereo checks without models, GPU or a running ROS graph."""

from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from cv_bridge import CvBridge  # noqa: E402
from face_tracking_arm.msg import PerceptionStatus  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from rclpy.clock import ClockType  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from tf2_ros import TransformException  # noqa: E402
import xacro  # noqa: E402
from face_tracking_perception.depth_backends import (  # noqa: E402, I100
    AlignedRGBDDepth, StereoDepth)
from face_tracking_perception.depth_geometry import (  # noqa: E402, I100
    Camera, partial_face_center, sample_face_depth, sample_face_measurement, size_depth,
    stereo_baseline, visible_face)
from face_tracking_perception.depth_messages import (  # noqa: E402
    point_marker, stamped_point, status_message)
from face_tracking_perception.detectors import Face  # noqa: E402
from face_tracking_perception.face_selection import NearestFaceSelector  # noqa: E402
from face_tracking_perception.kalman_point_filter import KalmanPointFilter  # noqa: E402
from face_tracking_perception.depth_node import FacePositionNode  # noqa: E402, I100
from face_tracking_perception.node import FaceDetectionNode  # noqa: E402
from face_tracking_perception.point_filter import PointFilter  # noqa: E402
from face_tracking_perception.rectification import Rectification  # noqa: E402


def camera_info():
    return CameraInfo(
        header=Header(frame_id='left_optical'), width=640, height=480,
        p=[500., 0., 320., 0., 0., 500., 240., 0., 0., 0., 1., 0.],
        r=[1., 0., 0., 0., 1., 0., 0., 0., 1.], d=[0.] * 5)


def test_bad_image_encoding_drops_the_observation_and_next_frame_is_processed():
    observations, detected = [], []
    invalid_cadence_observations = []
    previews, points = [], []
    discard = SimpleNamespace(publish=lambda message: None)

    def detect(image):
        detected.append(image.shape)
        return []

    def publish(selected, reason):
        observations.append((selected, reason, node.pending_observation))
        FacePositionNode.publish_observation(node, selected, reason)

    node = SimpleNamespace(
        options={'max_processing_fps': 0, 'input_left_frame': '', 'rectification': 'none',
                 'selection_frame': 'world', 'preview_fps': 5.},
        mode='mono_cpu', last_processed_at=None, pending_observation=None, bridge=CvBridge(),
        detector=SimpleNamespace(detect=detect),
        selector=SimpleNamespace(select=lambda *args: None),
        rectifiers=[SimpleNamespace(apply=lambda image, info, mode: (
            image, Camera.from_info(info), np.eye(3)))],
        publish_observation=publish,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(clock_type=ClockType.ROS_TIME)),
        detection_pub=discard, marker_pub=discard, status_pub=discard,
        cadence=SimpleNamespace(observe=lambda *args: invalid_cadence_observations.append(args)),
        pair_cadence=SimpleNamespace(
            observe=lambda *args: invalid_cadence_observations.append(args),
            bounded_gap_limit=lambda maximum: maximum),
        point_pub=SimpleNamespace(publish=points.append),
        image_pub=SimpleNamespace(publish=previews.append, get_subscription_count=lambda: 1),
        compressed_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        pending_world_point=None, world_point_drops=0, observation_drops=0,
        last_preview_at=-float('inf'), last_report=float('inf'))
    node.flush_observation = lambda: FacePositionNode.flush_observation(node)
    node.publish_world_point = lambda: FacePositionNode.publish_world_point(node)
    invalid = Image(header=camera_info().header, width=640, height=480, encoding='unknown')
    FacePositionNode.on_frame(node, invalid, camera_info())
    assert not detected
    assert observations[0][0] is None
    assert observations[0][2][1] is None  # No image exists to annotate.
    assert observations[0][2][3] == []   # No 3D point can escape a failed conversion.
    assert invalid_cadence_observations == []
    assert not hasattr(node, 'last_pair_receipt')
    assert not previews and not points
    valid = node.bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8), 'bgr8')
    valid.header = camera_info().header
    FacePositionNode.on_frame(node, valid, camera_info())
    assert detected == [(480, 640, 3)]
    assert len(invalid_cadence_observations) == 2
    assert observations[1][1] == 'NO_FACE'
    assert len(previews) == 1 and not points


def test_standalone_detector_reports_bad_frame_without_exiting():
    results, warnings = [], []
    node = SimpleNamespace(
        bridge=CvBridge(), results=SimpleNamespace(publish=results.append),
        get_logger=lambda: SimpleNamespace(warning=lambda text, **kwargs: warnings.append(text)))
    message = Image(header=camera_info().header, encoding='unknown')
    FaceDetectionNode.on_image(node, message)
    assert results[0].header == message.header
    assert not results[0].detections
    assert len(warnings) == 1


def test_unprojection_uses_optical_z_and_meters():
    camera = Camera.from_info(camera_info())
    assert camera.unproject(420, 190, 2) == pytest.approx((0.4, -0.2, 2))
    assert camera.unproject(320, 240, 1) == pytest.approx((0, 0, 1))
    for z in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            camera.unproject(320, 240, z)


def test_calibration_rejects_distorted_missing_or_unrectified_input():
    for field, value in [('d', [0.1] * 5), ('d', [np.nan] * 5), ('r', [0.] * 9), ('p', [0.] * 12),
                         ('p', [np.nan] * 12), ('width', 0)]:
        info = camera_info()
        setattr(info, field, value)
        with pytest.raises(ValueError):
            Camera.from_info(info)


def test_width_distance_scales_and_rejects_clipped_faces():
    camera = Camera.from_info(camera_info())
    assert size_depth(Face(200, 100, 80, 100, 0.9), camera, .16) == pytest.approx(1)
    assert size_depth(Face(200, 100, 40, 50, 0.9), camera, .16) == pytest.approx(2)
    assert visible_face(Face(0, 100, 80, 100, 0.9), camera)
    with pytest.raises(ValueError, match='truncated'):
        size_depth(Face(0, 100, 80, 100, 0.9), camera, .16)
    with pytest.raises(ValueError):
        size_depth(Face(200, 100, 5, 5, 0.9), camera, .16)


def test_depth_median_ignores_background_outliers_and_invalid_pixels():
    depth = np.full((480, 640), 7, dtype=np.float32)
    face = Face(200, 100, 80, 100, 0.9)
    depth[120:180, 220:260] = 1.2
    depth[148:151, 235:245] = np.nan
    depth[152:155, 235:245] = 6
    assert sample_face_depth(depth, face) == pytest.approx(1.2)
    depth[:] = np.nan
    with pytest.raises(ValueError, match='Insufficient'):
        sample_face_depth(depth, face)
    depth[:] = np.random.default_rng(10).uniform(.5, 6, depth.shape)
    with pytest.raises(ValueError, match='inconsistent'):
        sample_face_depth(depth, face)


def test_partial_rgbd_face_uses_supported_pixel_bearing_not_clipped_box_center():
    depth = np.full((480, 640), np.nan, dtype=np.float32)
    face = Face(0, 100, 90, 100, .9)
    depth[125:175, 8:28] = 1.2
    z, u, v = sample_face_measurement(depth, face)
    assert z == pytest.approx(1.2)
    assert u == pytest.approx(17.5)
    assert v == pytest.approx(149.5)


def test_partial_cpu_face_direction_compensates_for_clipped_width_using_retained_range():
    camera = Camera.from_info(camera_info())
    left = Face(0, 100, 45, 100, .9)
    right = Face(595, 100, 45, 100, .9)
    # A full 16 cm face at 1 m spans 80 px; the visible 45 px fragment is not
    # treated as the full face center.
    assert partial_face_center(left, camera, 1.0, .16)[0] == pytest.approx(40)
    assert partial_face_center(right, camera, 1.0, .16)[0] == pytest.approx(600)


def test_rgbd_replay_stops_when_selected_visible_face_loses_depth_even_if_background_is_valid():
    selected = Face(100, 100, 80, 100, .95)
    background = Face(420, 110, 90, 110, .99)
    header = Header(frame_id='color_optical')
    header.stamp.sec = 10
    published = []
    node = SimpleNamespace(
        mode='rgbd',
        last_selected_face_box=(selected.x, selected.y, selected.width, selected.height),
        pending_observation=(SimpleNamespace(header=header), None,
                             [selected, background], [(1, (1., 0., 2.))], None, 0., 'OK'),
        options={'selection_frame': 'world', 'max_frame_age_sec': .2},
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=10.05)),
        publish_observation=lambda index, reason: published.append((index, reason)),
        publish_world_point=lambda: None)
    FacePositionNode.flush_observation(node)
    assert published == [(None, 'NO_VALID_FACE_DEPTH')]


def test_stereo_rejects_invalid_required_right_calibration_even_without_detections():
    bridge = CvBridge()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left_info, right_info = camera_info(), camera_info()
    left_info.header.stamp.sec = right_info.header.stamp.sec = 10
    right_info.header.frame_id = 'right_optical'
    right_info.p[3] = -50.
    invalid_right = camera_info()
    invalid_right.header = right_info.header
    invalid_right.p = [0.] * 12
    messages = [bridge.cv2_to_imgmsg(frame, 'bgr8'), bridge.cv2_to_imgmsg(frame, 'bgr8')]
    messages[0].header, messages[1].header = left_info.header, right_info.header
    node = SimpleNamespace(
        mode='stereo', options={
            'max_frame_age_sec': .2, 'input_left_frame': '', 'input_right_frame': '',
            'rectification': 'none', 'selection_frame': 'left_optical',
            'stereo_baseline_m': 0., 'timing_mode': 'auto', 'source_timeout_sec': .25,
            'point_smoothing': False, 'face_width_m': .16},
        bridge=bridge, rectifiers=[Rectification(), Rectification()],
        detector=SimpleNamespace(detect=lambda image: []),
        pending_observation=None, last_processed_at=None, last_capture_sec=None,
        last_pair_receipt=None, flush_observation=lambda: None,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=10.05)),
        pending_since=None, observation_drops=0,
        publish_observation=lambda *args: None,
        status_pub=SimpleNamespace(publish=lambda msg: None))
    FacePositionNode.on_frame(node, messages[0], left_info, messages[1], invalid_right)
    assert node.pending_observation[6].startswith('INVALID_INPUT: CameraInfo')


def make_rgbd_frame_node(depth_message, detected_faces):
    bridge = CvBridge()
    header = Header(frame_id='color_optical', stamp=Time(seconds=10).to_msg())
    image = bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8), 'bgr8')
    image.header = header
    info = camera_info()
    info.header = header
    transform = TransformStamped()
    transform.header.frame_id = 'color_optical'
    transform.transform.rotation.w = 1.
    perception_states = []
    published_points, published_markers = [], []
    empty_pub = SimpleNamespace(publish=lambda _message: None)
    detector_calls = []

    node = SimpleNamespace(
        mode='rgbd', options={
            'max_frame_age_sec': .2, 'input_left_frame': 'color_optical',
            'rectification': 'none', 'rgbd_sync_slop_sec': .02,
            'rgbd_depth_topic': '/depth', 'timing_mode': 'auto', 'source_timeout_sec': .25,
            'selection_frame': 'color_optical', 'point_smoothing': False,
            'point_smoothing_method': 'one_euro', 'preview_fps': 5.},
        bridge=bridge, rectifiers=[Rectification()], backend=AlignedRGBDDepth(),
        detector=SimpleNamespace(detect=lambda _image: detector_calls.append(True)
                                 or detected_faces), rate_gate=None,
        pending_observation=None, pending_since=None, selector=NearestFaceSelector(),
        cadence=SimpleNamespace(observe=lambda *args: None),
        pair_cadence=SimpleNamespace(observe=lambda *args: None,
                                     bounded_gap_limit=lambda maximum: maximum),
        processing_cadence=SimpleNamespace(observe=lambda *args: None),
        last_selected_face_box=None, last_reliable_fixed_point=None,
        last_reliable_depth_stamp=None, last_depth_stamp=None,
        last_target_id=-1, last_valid_input_receipt=None, last_pair_receipt=None,
        last_capture_sec=None, last_processed_at=None, observation_drops=0,
        last_preview_at=-float('inf'), last_report=time.monotonic(),
        pending_world_point=None, world_point_drops=0,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=10.05)),
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: transform),
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        publish_observation=None,
        publish_perception_state=lambda state, reason, stamp=None: perception_states.append(
            (state, reason, stamp)), publish_world_point=lambda: None,
        status_pub=empty_pub, perception_status_pub=empty_pub,
        detection_pub=empty_pub, marker_pub=SimpleNamespace(publish=published_markers.append),
        point_pub=SimpleNamespace(publish=published_points.append),
        image_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        compressed_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        depth_pub=SimpleNamespace(get_subscription_count=lambda: 0))
    node.publish_observation = lambda *args: FacePositionNode.publish_observation(node, *args)
    node.flush_observation = lambda: FacePositionNode.flush_observation(node)
    FacePositionNode.on_frame(node, image, info, depth_message)
    node.published_points = published_points
    node.published_markers = published_markers
    return node, perception_states, detector_calls


def test_rgbd_all_invalid_depth_map_without_a_face_is_a_healthy_no_face_pair():
    header = Header(frame_id='color_optical', stamp=Time(seconds=10).to_msg())
    depth = CvBridge().cv2_to_imgmsg(
        np.full((480, 640), np.inf, dtype=np.float32), '32FC1')
    depth.header = header
    node, states, detector_calls = make_rgbd_frame_node(depth, [])

    assert detector_calls == [True]
    assert node.last_pair_receipt is not None
    assert states[-1][0:2] == (PerceptionStatus.NO_FACE, 'NO_FACE')


def test_rgbd_visible_face_with_all_invalid_depth_is_face_no_depth():
    header = Header(frame_id='color_optical', stamp=Time(seconds=10).to_msg())
    depth = CvBridge().cv2_to_imgmsg(
        np.full((480, 640), np.inf, dtype=np.float32), '32FC1')
    depth.header = header
    face = Face(270, 180, 80, 90, .95)
    node, states, detector_calls = make_rgbd_frame_node(depth, [face])

    assert detector_calls == [True]
    assert node.last_pair_receipt is not None
    assert states[-1][0:2] == (PerceptionStatus.FACE_NO_DEPTH, 'NO_VALID_FACE_DEPTH')


def test_rgbd_confirming_face_is_wait_and_does_not_republish_old_3d_target():
    header = Header(frame_id='color_optical', stamp=Time(seconds=10).to_msg())
    depth_image = np.full((480, 640), 1.5, dtype=np.float32)
    depth = CvBridge().cv2_to_imgmsg(depth_image, '32FC1')
    depth.header = header
    node, states, detector_calls = make_rgbd_frame_node(
        depth, [Face(270, 180, 80, 90, .95)])

    assert detector_calls == [True]
    assert states[-1][0:2] == (PerceptionStatus.WAIT, 'CONFIRMING_FACE')
    assert not node.published_points
    assert node.published_markers[-1].action == node.published_markers[-1].DELETE


@pytest.mark.parametrize(
    'encoding,data', [('bad_encoding', bytes(4)), ('32FC1', b'')])
def test_rgbd_malformed_or_empty_depth_frame_is_invalid_input(encoding, data):
    header = Header(frame_id='color_optical', stamp=Time(seconds=10).to_msg())
    depth = Image(header=header, height=480, width=640, encoding=encoding,
                  step=640 * 4, data=data)
    node, states, detector_calls = make_rgbd_frame_node(depth, [])

    assert states[-1][0] == PerceptionStatus.INVALID_INPUT
    assert states[-1][1].startswith('INVALID_INPUT:')
    assert node.last_pair_receipt is None
    assert not detector_calls


def test_cpu_partial_target_reuses_original_depth_age_and_capture_tf():
    header = Header(frame_id='left_optical', stamp=Time(seconds=10.1).to_msg())
    camera = Camera.from_info(camera_info())
    face = Face(0, 100, 30, 100, .95)
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    selected = []

    def publish(index, reason, filtered=None):
        selected.append((index, reason, node.pending_observation[3][index]))
        node.pending_observation = None

    node = SimpleNamespace(
        mode='mono_cpu', options={
            'selection_frame': 'world', 'max_frame_age_sec': .2,
            'mono_partial_depth_max_age_sec': .30, 'face_width_m': .16,
            'point_smoothing': False},
        pending_observation=(SimpleNamespace(header=header), None, [face], [], None, 1.0,
                             'NO_VALID_FACE_DEPTH', [(0, face)], camera, np.eye(3)),
        last_selected_face_box=(0., 100., 40., 100.),
        last_reliable_fixed_point=(0., 0., 2.),
        last_reliable_depth_stamp=Time(seconds=10).to_msg(),
        selector=SimpleNamespace(select=lambda points, stamp: 0, track_id=3),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=10.15)),
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: tf),
        publish_observation=publish, publish_world_point=lambda: None,
        observation_drops=0)
    FacePositionNode.flush_observation(node)
    assert selected[0][0:2] == (0, 'OK')
    assert selected[0][2][1][0] == pytest.approx(-1.2)
    assert node.last_reliable_depth_stamp == Time(seconds=10).to_msg()


def test_aligned_rgbd_units_encodings_and_invalid_depth_are_explicit():
    backend = AlignedRGBDDepth()
    raw = np.asarray([[0, 250, 1200, 9000]], dtype=np.uint16)
    converted = backend.convert(raw, '16UC1')
    assert np.isnan(converted[0, 0])
    assert converted[0, 1] == pytest.approx(.25)
    assert converted[0, 2] == pytest.approx(1.2)
    assert np.isnan(converted[0, 3])
    float_depth = backend.convert(np.asarray([[1.0, np.inf]], dtype=np.float32), '32FC1')
    assert float_depth[0, 0] == pytest.approx(1.0)
    assert np.isnan(float_depth[0, 1])
    with pytest.raises(ValueError, match='encoding'):
        backend.convert(raw, 'mono16')


def test_stereo_requires_calibrated_positive_baseline():
    left = Camera.from_info(camera_info())
    info = camera_info()
    info.p[3] = -40
    right = Camera.from_info(info)
    assert stereo_baseline(left, right) == pytest.approx(.08)
    with pytest.raises(ValueError):
        stereo_baseline(right, left)
    with pytest.raises(ValueError):
        stereo_baseline(left, left)


@pytest.mark.parametrize('num_disparities,block_size,shift', [(128, 5, 20), (256, 7, 160)])
def test_real_sgbm_recovers_known_shift_in_rectified_pair(num_disparities, block_size, shift):
    rng = np.random.default_rng(42)
    left_image = rng.integers(0, 256, (240, 640, 3), dtype=np.uint8)
    right_image = np.zeros_like(left_image)
    right_image[:, :-shift] = left_image[:, shift:]
    left = Camera(640, 240, 500, 500, 320, 120)
    right = Camera(640, 240, 500, 500, 320, 120, -40)
    depth = StereoDepth(num_disparities, block_size).compute(
        left_image, right_image, left, right)
    expected_distance = 500 * .08 / shift
    assert np.nanmedian(depth[40:200, 300:600]) == pytest.approx(expected_distance, abs=.02)
    assert np.isnan(depth[:, :100]).all()


@pytest.mark.parametrize('num_disparities', [0, -16, 1, 127, 16.0, True])
def test_stereo_rejects_invalid_disparity_ranges(num_disparities):
    with pytest.raises(ValueError, match='stereo_num_disparities'):
        StereoDepth(num_disparities=num_disparities)


@pytest.mark.parametrize('block_size', [0, -1, 2, 4, 5.0, True])
def test_stereo_rejects_invalid_matching_blocks(block_size):
    with pytest.raises(ValueError, match='stereo_block_size'):
        StereoDepth(block_size=block_size)


@pytest.mark.parametrize('height,width,block_size', [
    (100, 128, 5), (100, 129, 5), (100, 130, 5), (4, 640, 5), (5, 640, 7)])
def test_stereo_rejects_images_too_small_for_the_selected_settings(height, width, block_size):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='Stereo image'):
        StereoDepth(block_size=block_size).compute(image, image, None, None)


def test_stereo_rejects_mismatched_image_dimensions():
    left = np.zeros((240, 640, 3), dtype=np.uint8)
    right = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='same dimensions'):
        StereoDepth().compute(left, right, None, None)


def test_stamped_points_and_expiring_markers():
    header = Header(frame_id='left_optical')
    header.stamp.sec = 9
    point = stamped_point(header, (.1, -.2, 1.3))
    assert point.header == header
    assert point.point.z == 1.3
    marker = point_marker(point, 'estimate', (0., 1., 0.))
    assert marker.header == header
    assert marker.lifetime.nanosec == 400000000
    status = status_message(header, 'depth', 'NO_FACE', {'point_valid': False}, False)
    assert status.header == header
    assert status.status[0].level == status.status[0].WARN


def test_moving_camera_waits_for_exact_tf_and_preserves_measurement_stamp():
    header = Header(frame_id='left_optical')
    header.stamp.sec = 10
    point = stamped_point(header, (0., 0., 2.))
    published = []
    transform = TransformStamped(header=Header(frame_id='world'))
    transform.header.stamp.sec = 11  # do_transform_point uses this unless restored.
    transform.transform.translation.x = 1.0
    transform.transform.rotation.w = 1.0

    def unavailable(*args):
        raise TransformException('Joint TF is not received yet')

    def sim_time(seconds):
        return Time(seconds=seconds, clock_type=ClockType.ROS_TIME)

    node = SimpleNamespace(
        pending_world_point=point, world_point_drops=0,
        get_clock=lambda: SimpleNamespace(now=lambda: sim_time(10.1)),
        world_pub=SimpleNamespace(publish=published.append),
        tf_buffer=SimpleNamespace(lookup_transform=unavailable))
    FacePositionNode.publish_world_point(node)
    assert not published
    assert node.pending_world_point is point

    def available(target, source, stamp):
        assert (target, source) == ('world', 'left_optical')
        assert stamp == sim_time(10)
        return transform

    node.tf_buffer.lookup_transform = available
    FacePositionNode.publish_world_point(node)
    assert published[0].point.x == 1.0
    assert published[0].point.z == 2.0
    assert published[0].header.stamp == header.stamp
    assert node.pending_world_point is None

    node.pending_world_point = point
    node.get_clock = lambda: SimpleNamespace(now=lambda: sim_time(10.3))
    FacePositionNode.publish_world_point(node)
    assert node.pending_world_point is None
    assert node.world_point_drops == 1
    assert len(published) == 1


def test_face_selection_waits_for_measurement_tf_and_drops_stale_observations():
    header = Header(frame_id='left_optical')
    header.stamp.sec = 10
    observation = (SimpleNamespace(header=header), None, [],
                   [(0, (0., 0., 2.)), (1, (1., 0., 1.))], None, 0, '')
    chosen, selected_points = [], []
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.rotation.w = 1.0

    def unavailable(*args):
        raise TransformException('TF still arriving')

    def select(points, stamp):
        assert stamp == 10
        selected_points.append(points)
        return 0

    def publish(index, reason):
        chosen.append((index, reason))
        node.pending_observation = None

    node = SimpleNamespace(
        pending_observation=observation, observation_drops=0,
        options={'selection_frame': 'link_base'},
        selector=SimpleNamespace(select=select),
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.1, clock_type=ClockType.ROS_TIME)),
        tf_buffer=SimpleNamespace(lookup_transform=unavailable),
        publish_observation=publish, publish_world_point=lambda: None)
    FacePositionNode.flush_observation(node)
    assert not chosen and not selected_points

    def available(target, source, stamp):
        assert (target, source) == ('link_base', 'left_optical')
        assert stamp.nanoseconds == 10_000_000_000
        return transform

    node.tf_buffer.lookup_transform = available
    FacePositionNode.flush_observation(node)
    np.testing.assert_allclose(selected_points[0], [(1, 0, 2), (2, 0, 1)])
    assert chosen == [(0, 'OK')]
    node.pending_observation = observation
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: Time(seconds=10.3, clock_type=ClockType.ROS_TIME))
    FacePositionNode.flush_observation(node)
    assert chosen[-1] == (None, 'STALE_OBSERVATION')
    assert len(selected_points) == 1 and node.observation_drops == 1


def point_publication_harness(smoothing, method='one_euro'):
    """Exercise real observation publication with explicit camera poses, without ROS spin."""
    output = SimpleNamespace(points=[], world=[], markers=[], statuses=[], selected=[], lookups=[])
    discard = SimpleNamespace(publish=lambda message: None)
    selector = SimpleNamespace(track_id=1, current=None)

    def select(points, stamp):
        output.selected.append(np.asarray(points).copy())
        selector.current = np.asarray(points[0])
        return 0

    selector.select = select
    node = SimpleNamespace(
        options={'selection_frame': 'link_base', 'point_smoothing': smoothing,
                 'point_smoothing_method': method,
                 'preview_fps': 5.}, world_frame='world', mode='mono_cpu',
        point_filter=KalmanPointFilter() if method == 'kalman' else PointFilter(beta=0.),
        selector=selector, observation_drops=0,
        pending_observation=None, pending_world_point=None, world_point_drops=0,
        get_clock=lambda: SimpleNamespace(now=lambda: node.now),
        detection_pub=discard, point_pub=SimpleNamespace(publish=output.points.append),
        world_pub=SimpleNamespace(publish=output.world.append),
        marker_pub=SimpleNamespace(publish=output.markers.append),
        status_pub=SimpleNamespace(publish=output.statuses.append),
        image_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        compressed_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        last_preview_at=-float('inf'), last_report=float('inf'))
    node.publish_observation = lambda *args: FacePositionNode.publish_observation(node, *args)
    node.publish_world_point = lambda: FacePositionNode.publish_world_point(node)

    def submit(fixed, stamp_sec, translation, yaw):
        stamp = Time(seconds=stamp_sec, clock_type=ClockType.ROS_TIME)
        header = Header(frame_id='left_optical', stamp=stamp.to_msg())
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        translation = np.asarray(translation)
        raw = rotation.T @ (np.asarray(fixed) - translation)

        def lookup(target, source, when):
            output.lookups.append((target, source, when.nanoseconds))
            assert when == stamp
            inverse = target == 'left_optical'
            assert (target, source) in (
                ('link_base', 'left_optical'), ('left_optical', 'link_base'),
                ('world', 'left_optical'))
            transform = TransformStamped(
                header=Header(frame_id=target, stamp=stamp.to_msg()), child_frame_id=source)
            offset = -rotation.T @ translation if inverse else translation
            transform.transform.translation.x = float(offset[0])
            transform.transform.translation.y = float(offset[1])
            transform.transform.translation.z = float(offset[2])
            transform.transform.rotation.z = float(np.sin((-yaw if inverse else yaw) / 2))
            transform.transform.rotation.w = float(np.cos(yaw / 2))
            return transform

        node.tf_buffer = SimpleNamespace(lookup_transform=lookup)
        node.now = Time(seconds=stamp_sec + .01, clock_type=ClockType.ROS_TIME)
        node.pending_observation = (
            Image(header=header), None, [Face(200, 100, 80, 100, .9)],
            [(0, tuple(raw))], None, 0., 'OK')
        FacePositionNode.flush_observation(node)
        assert node.pending_observation is None
        point, world, marker = output.points[-1], output.world[-1], output.markers[-1]
        assert point.header == marker.header == header
        assert world.header.frame_id == 'world' and world.header.stamp == header.stamp
        assert marker.pose.position == point.point
        values = {item.key: item.value for item in output.statuses[-1].status[0].values}
        np.testing.assert_allclose([float(values[f'raw_{axis}_m']) for axis in 'xyz'], raw)
        np.testing.assert_allclose(output.selected[-1], [fixed], atol=1e-12)
        assert values['selected_track_id'] == str(selector.track_id)
        assert values['point_smoothing'] == str(smoothing)
        assert values['point_smoothing_method'] == method
        return raw, point, world

    return node, output, submit


@pytest.mark.parametrize('method', ['one_euro', 'kalman'])
def test_disabled_point_smoothing_publishes_unmodified_measurement_and_stamp(method):
    node, output, submit = point_publication_harness(False, method)

    def unexpected(*args):
        pytest.fail('Disabled smoothing must never update filter state')

    node.point_filter = SimpleNamespace(update=unexpected)
    for index, fixed in enumerate(((1., .4, 2.), (1.08, .4, 2.))):
        raw, point, world = submit(fixed, 10. + index * .04, (.3, -.4, .2), np.pi / 2)
        assert (point.point.x, point.point.y, point.point.z) == tuple(raw)
        np.testing.assert_allclose(
            (world.point.x, world.point.y, world.point.z), fixed, atol=1e-12)
    assert not any(target == 'left_optical' for target, _, _ in output.lookups)


@pytest.mark.parametrize('method', ['one_euro', 'kalman'])
def test_point_smoothing_uses_fixed_frame_and_resets_when_selected_person_changes(method):
    node, output, submit = point_publication_harness(True, method)
    face = (1., .4, 2.)
    submit(face, 10., (0., 0., 0.), 0.)
    _, _, stationary = submit(face, 10.04, (.3, -.4, .2), np.pi / 2)
    np.testing.assert_allclose(
        (stationary.point.x, stationary.point.y, stationary.point.z), face, atol=1e-12)

    noisy_face = (1.08, .4, 2.)
    raw, point, filtered = submit(noisy_face, 10.08, (-.1, .2, -.1), -np.pi / 2)
    assert face[0] < filtered.point.x < noisy_face[0]
    np.testing.assert_allclose((filtered.point.y, filtered.point.z), face[1:], atol=1e-12)
    assert np.linalg.norm(np.array((point.point.x, point.point.y, point.point.z)) - raw) > .01

    node.selector.track_id = 2
    new_face = (1.6, .4, 2.2)
    _, _, reset = submit(new_face, 10.12, (.2, .1, .3), np.pi / 3)
    np.testing.assert_allclose((reset.point.x, reset.point.y, reset.point.z), new_face, atol=1e-12)
    assert len(output.points) == len(output.world) == 4


@pytest.mark.parametrize('age,reason,selected', [
    (.1, 'NO_FACE', True),
    (.3, 'STALE_OBSERVATION', False),
    (-.1, 'STALE_OBSERVATION', False),
])
def test_empty_frame_updates_selection_only_when_fresh_without_waiting_for_tf(
        age, reason, selected):
    observations, selection_calls = [], []
    info = camera_info()
    info.header.stamp.sec = 10

    def publish(index, status):
        observations.append((index, status))
        node.pending_observation = None

    def unexpected_tf(*args):
        pytest.fail('An empty observation must not wait for a camera transform')

    node = SimpleNamespace(
        options={'max_processing_fps': 0, 'input_left_frame': '', 'rectification': 'none'},
        mode='mono_cpu', last_processed_at=None, pending_observation=None, bridge=CvBridge(),
        detector=SimpleNamespace(detect=lambda image: []),
        rectifiers=[SimpleNamespace(apply=lambda image, info, mode: (
            image, Camera.from_info(info), np.eye(3)))],
        selector=SimpleNamespace(select=lambda points, stamp: selection_calls.append(
            (points, stamp))),
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10 + age, clock_type=ClockType.ROS_TIME)),
        tf_buffer=SimpleNamespace(lookup_transform=unexpected_tf),
        observation_drops=0, publish_observation=publish, publish_world_point=lambda: None,
        status_pub=SimpleNamespace(publish=lambda message: observations.append(
            (None, message.status[0].message))))
    node.flush_observation = lambda: FacePositionNode.flush_observation(node)
    image = node.bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8), 'bgr8')
    image.header = info.header

    FacePositionNode.on_frame(node, image, info)

    assert selection_calls == ([([], 10.)] if selected else [])
    assert observations == [(None, reason)]
    assert node.pending_observation is None
    assert node.observation_drops == int(not selected)


@pytest.mark.parametrize('stereo,reference', [(False, False), (True, True)])
def test_world_camera_spacing_and_projection_agree(stereo, reference):
    path = Path(__file__).resolve().parents[1] / 'worlds/face_depth_test.sdf.xacro'
    tree = ET.fromstring(xacro.process_file(str(path), mappings={
        'stereo': str(stereo).lower(), 'reference': str(reference).lower(),
        'baseline': '0.10'}).toxml())
    models = {model.get('name'): model for model in tree.findall('world/model')}
    assert ('right_camera' in models) == stereo
    assert ('reference_camera' in models) == reference
    if stereo:
        right = models['right_camera']
        assert float(right.findtext('pose').split()[0]) == pytest.approx(.1)
        projection = right.find('link/sensor/camera/lens/projection')
        baseline = -float(projection.findtext('tx')) / float(projection.findtext('p_fx'))
        assert baseline == pytest.approx(.1)


@pytest.mark.parametrize('stale_input', range(4))
def test_stale_stereo_input_is_dropped_before_conversion_and_keeps_pending_result(stale_input):
    messages = [Image(), CameraInfo(), Image(), CameraInfo()]
    for index, message in enumerate(messages):
        message.header.frame_id = 'left_optical' if index < 2 else 'right_optical'
        message.header.stamp.sec = 10
        message.header.stamp.nanosec = 100_000_000
    messages[stale_input].header.stamp.nanosec = 99_000_000
    statuses, flushes = [], []
    pending = object()

    def unexpected(*args):
        pytest.fail('Stale inputs must not reach conversion, detection or depth inference')

    node = SimpleNamespace(
        mode='stereo', options={'max_frame_age_sec': .2, 'max_processing_fps': 20.},
        pending_observation=pending, last_processed_at=None, observation_drops=0,
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.3, clock_type=ClockType.ROS_TIME)),
        bridge=SimpleNamespace(imgmsg_to_cv2=unexpected),
        detector=SimpleNamespace(detect=unexpected), backend=SimpleNamespace(compute=unexpected),
        flush_observation=lambda: flushes.append(node.pending_observation),
        status_pub=SimpleNamespace(publish=statuses.append))

    FacePositionNode.on_frame(node, *messages)

    assert node.last_processed_at is None  # Dropped work must not consume the FPS allowance.
    assert node.pending_observation is pending
    assert flushes == [pending]
    assert node.observation_drops == 1
    assert statuses[0].header == messages[0].header
    assert statuses[0].status[0].message == 'STALE_OBSERVATION'
    values = {v.key: v.value for v in statuses[0].status[0].values}
    assert float(values['processing_ms']) == 0
    assert float(values['oldest_input_age_ms']) == pytest.approx(201.)


def test_watchdog_does_not_reclassify_a_recent_pair_as_stale_by_observation_age():
    published = []
    now = time.monotonic()
    node = SimpleNamespace(
        last_pair_receipt=now - .25, source_gap_limit=.5,
        last_perception_state=PerceptionStatus.OK, last_perception_reason='OK',
        last_source_stamp=Time(seconds=9.7).to_msg(),
        publish_perception_state=lambda state, reason: published.append((state, reason)))
    FacePositionNode.publish_watchdog_status(node)
    assert published == [(PerceptionStatus.OK, 'OK')]


def test_watchdog_reports_stream_lost_only_after_the_bounded_pair_gap():
    published = []
    node = SimpleNamespace(
        last_pair_receipt=time.monotonic() - .51, source_gap_limit=.5,
        last_perception_state=PerceptionStatus.OK, last_perception_reason='OK',
        last_source_stamp=Time(seconds=9.7).to_msg(),
        publish_perception_state=lambda state, reason: published.append((state, reason)))
    FacePositionNode.publish_watchdog_status(node)
    assert published == [(PerceptionStatus.DEPTH_STREAM_LOST, 'camera pair watchdog expired')]


def test_pending_exact_tf_is_retried_before_next_frame_uses_its_fps_slot():
    events = []
    info = camera_info()
    info.header.stamp.sec = 10
    image = Image(header=info.header)
    node = SimpleNamespace(
        options={'max_frame_age_sec': .2, 'max_processing_fps': 20.},
        rate_gate=SimpleNamespace(accept=lambda now: False), last_processed_at=float('inf'),
        pending_observation=None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.1, clock_type=ClockType.ROS_TIME)),
        flush_observation=lambda: events.append('retry exact TF'))

    FacePositionNode.on_frame(node, image, info)

    assert events == ['retry exact TF']


def test_waiting_for_exact_tf_yields_the_executor_instead_of_replacing_the_result():
    info = camera_info()
    info.header.stamp.sec = 10
    pending = object()

    def unexpected(*args):
        pytest.fail('Inference must yield while the previous result waits for its exact TF')

    node = SimpleNamespace(
        options={'max_frame_age_sec': .2, 'max_processing_fps': 0.},
        pending_observation=pending, last_processed_at=None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.1, clock_type=ClockType.ROS_TIME)),
        flush_observation=lambda: None,
        bridge=SimpleNamespace(imgmsg_to_cv2=unexpected),
        detector=SimpleNamespace(detect=unexpected))

    FacePositionNode.on_frame(node, Image(header=info.header), info)

    assert node.pending_observation is pending
    assert node.last_processed_at is None


@pytest.mark.parametrize('pending_stamp,first_reason', [(10., 'OK'), (9.8, 'STALE_OBSERVATION')])
def test_stereo_inference_resumes_after_pending_tf_resolves_or_observation_expires(
        pending_stamp, first_reason):
    left_info, right_info = camera_info(), camera_info()
    for info in (left_info, right_info):
        info.header.stamp.sec = 10
        info.header.stamp.nanosec = 100_000_000
    right_info.header.frame_id = 'right_optical'
    right_info.p[3] = -50.
    bridge = CvBridge()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = [bridge.cv2_to_imgmsg(image, 'bgr8') for _ in range(2)]
    left.header, right.header = left_info.header, right_info.header
    old_header = Header(frame_id='left_optical', stamp=Time(seconds=pending_stamp).to_msg())
    depth_calls, published = [], []
    transform = TransformStamped()
    transform.transform.rotation.w = 1.

    def publish(selected, reason):
        published.append((node.pending_observation[0].header.stamp, reason))
        node.pending_observation = None

    def compute(*args):
        depth_calls.append('computed fresh stereo pair')
        return np.ones((480, 640), dtype=np.float32)

    rectifier = SimpleNamespace(apply=lambda image, info, mode: (
        image, Camera.from_info(info), np.eye(3)))
    node = SimpleNamespace(
        mode='stereo', options={'max_processing_fps': 0., 'max_frame_age_sec': .2,
                                'selection_frame': 'link_base', 'rectification': 'none',
                                'input_left_frame': '', 'input_right_frame': '',
                                'stereo_baseline_m': 0.},
        pending_observation=(Image(header=old_header), None, [],
                             [(0, (0., 0., 1.))], None, 0, 'CONFIRMING_FACE'),
        last_processed_at=None, observation_drops=0, bridge=bridge,
        rectifiers=[rectifier, rectifier],
        detector=SimpleNamespace(detect=lambda image: [Face(200, 100, 80, 100, .9)]),
        backend=SimpleNamespace(compute=compute),
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.1, clock_type=ClockType.ROS_TIME)),
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: transform),
        selector=SimpleNamespace(select=lambda points, stamp: 0),
        publish_observation=publish, publish_world_point=lambda: None)
    node.flush_observation = lambda: FacePositionNode.flush_observation(node)

    FacePositionNode.on_frame(node, left, left_info, right, right_info)

    assert depth_calls == ['computed fresh stereo pair']
    assert published == [(old_header.stamp, first_reason), (left.header.stamp, 'OK')]
    assert node.pending_observation is None
    assert node.last_processed_at is not None
