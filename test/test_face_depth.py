# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Depth geometry and stereo checks without models, GPU or a running ROS graph."""

from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
from rclpy.clock import ClockType
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import TransformException
import xacro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from face_tracking_perception.depth_backends import StereoDepth  # noqa: E402, I100
from face_tracking_perception.depth_geometry import (  # noqa: E402, I100
    Camera, sample_face_depth, size_depth, stereo_baseline, visible_face)
from face_tracking_perception.depth_messages import (  # noqa: E402
    point_marker, stamped_point, status_message)
from face_tracking_perception.detectors import Face  # noqa: E402
from face_tracking_perception.depth_node import FacePositionNode  # noqa: E402, I100
from face_tracking_perception.node import FaceDetectionNode  # noqa: E402


def camera_info():
    return CameraInfo(
        header=Header(frame_id='left_optical'), width=640, height=480,
        p=[500., 0., 320., 0., 0., 500., 240., 0., 0., 0., 1., 0.],
        r=[1., 0., 0., 0., 1., 0., 0., 0., 1.], d=[0.] * 5)


def test_bad_image_encoding_drops_the_observation_and_next_frame_is_processed():
    observations, detected = [], []
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
    assert not previews and not points
    valid = node.bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8), 'bgr8')
    valid.header = camera_info().header
    FacePositionNode.on_frame(node, valid, camera_info())
    assert detected == [(480, 640, 3)]
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
    assert not visible_face(Face(0, 100, 80, 100, 0.9), camera)
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


def test_real_sgbm_recovers_known_shift_in_rectified_pair():
    # A 20 px disparity at f=500 px and baseline=.08 m corresponds to Z=2 m.
    rng = np.random.default_rng(42)
    left_image = rng.integers(0, 256, (240, 400, 3), dtype=np.uint8)
    right_image = np.zeros_like(left_image)
    right_image[:, :-20] = left_image[:, 20:]
    left = Camera(400, 240, 500, 500, 200, 120)
    right = Camera(400, 240, 500, 500, 200, 120, -40)
    depth = StereoDepth().compute(left_image, right_image, left, right)
    assert np.nanmedian(depth[40:200, 150:350]) == pytest.approx(2, abs=.02)
    assert np.isnan(depth[:, :100]).all()


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
        mode='mono_cpu', last_processed_at=None, bridge=CvBridge(),
        detector=SimpleNamespace(detect=lambda image: []),
        rectifiers=[SimpleNamespace(apply=lambda image, info, mode: (
            image, Camera.from_info(info), np.eye(3)))],
        selector=SimpleNamespace(select=lambda points, stamp: selection_calls.append(
            (points, stamp))),
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10 + age, clock_type=ClockType.ROS_TIME)),
        tf_buffer=SimpleNamespace(lookup_transform=unexpected_tf),
        observation_drops=0, publish_observation=publish, publish_world_point=lambda: None)
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
