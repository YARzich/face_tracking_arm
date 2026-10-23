# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Synchronized ROS adapter for the three interchangeable 3D face experiments."""

from dataclasses import replace
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray
from face_tracking_arm.msg import PerceptionStatus
from geometry_msgs.msg import PointStamped
import message_filters
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker

from .depth_backends import AlignedRGBDDepth, MetricDepth, StereoDepth
from .depth_geometry import (fully_visible_face, partial_face_center, sample_face_measurement,
                             size_depth, stereo_baseline, visible_face)
from .depth_messages import point_marker, stamped_point, status_message
from .detectors import create_detector
from .face_selection import (NearestFaceSelector, same_visible_face,
                             selected_target_missing_depth)
from .kalman_point_filter import KalmanPointFilter
from .node import detection_message
from .point_filter import PointFilter
from .processing_rate import ProcessingRateGate, StableFrameCadence
from .rectification import Rectification


class FacePositionNode(Node):
    """Use only measured RGB/stereo data; never subscribe to simulator reference depth."""

    def __init__(self):
        super().__init__('face_position')
        defaults = {
            'mode': 'mono_cpu', 'detector': 'yunet', 'model_dir': '', 'threads': 2,
            'confidence': 0.6, 'face_width_m': 0.16, 'depth_input_size': 280,
            'depth_weights': '', 'depth_source': '', 'depth_scale': 1.0,
            'selection_frame': 'world', 'switch_margin_m': .25, 'switch_delay_sec': .4,
            'point_smoothing': False, 'point_smoothing_method': 'one_euro',
            'point_smoothing_min_cutoff_hz': 1.5,
            'point_smoothing_beta': 32.0, 'point_smoothing_derivative_cutoff_hz': 1.0,
            'point_smoothing_reset_after_sec': .25,
            'point_kalman_measurement_std_m': .02,
            'point_kalman_acceleration_std_mps2': 1.5,
            'rectification': 'none', 'sync_slop_sec': 0.0, 'stereo_baseline_m': 0.0,
            'stereo_num_disparities': 128, 'stereo_block_size': 5,
            'max_processing_fps': 0.0, 'max_frame_age_sec': .2,
            'rgbd_sync_slop_sec': .02, 'rgbd_16u_scale_m': .001,
            'rgbd_32f_scale_m': 1.0, 'source_timeout_sec': .25,
            'timing_mode': 'auto', 'tf_wait_timeout_sec': .05,
            'mono_partial_depth_max_age_sec': .30,
            'preview_fps': 5.0,
            'input_left_frame': '', 'input_right_frame': '',
            'rgbd_depth_topic': '/face_test/depth/image_raw', 'world_frame': 'world',
        }
        self.declare_parameters('', list(defaults.items()))
        self.options = {k: self.get_parameter(k).value for k in defaults}
        if self.options['timing_mode'] not in ('auto', 'manual'):
            raise ValueError('timing_mode must be auto or manual')
        for name in ('sync_slop_sec', 'stereo_baseline_m', 'max_processing_fps',
                     'max_frame_age_sec', 'preview_fps', 'rgbd_sync_slop_sec',
                     'source_timeout_sec', 'tf_wait_timeout_sec',
                     'mono_partial_depth_max_age_sec'):
            value = self.options[name]
            if (not np.isfinite(value) or value < 0 or
                    (name in ('max_frame_age_sec', 'preview_fps', 'source_timeout_sec',
                              'tf_wait_timeout_sec', 'mono_partial_depth_max_age_sec')
                     and value == 0)):
                raise ValueError(f'{name} must be finite and '
                                 'positive (zero is allowed for synchronization/FPS limits)')
        if self.options['rectification'] not in ('none', 'raw', 'rectified'):
            raise ValueError('Unknown rectification mode')
        if self.options['point_smoothing_method'] not in ('one_euro', 'kalman'):
            raise ValueError('point_smoothing_method must be one_euro or kalman')
        self.rectifiers = (Rectification(), Rectification())
        self.last_processed_at = None
        self.rate_gate = ProcessingRateGate(self.options['max_processing_fps'])
        self.cadence = StableFrameCadence()
        self.pair_cadence = StableFrameCadence()
        self.processing_cadence = StableFrameCadence()
        self.last_pair_receipt = None
        self.source_gap_limit = self.options['source_timeout_sec']
        self.last_source_stamp = None
        self.last_target_id = -1
        self.last_selected_face_box = None
        self.last_reliable_fixed_point = None
        self.last_reliable_depth_stamp = None
        self.last_depth_stamp = None
        self.last_capture_sec = None
        self.last_valid_input_receipt = None
        self.pending_since = None
        self.last_perception_state = PerceptionStatus.WAIT
        self.last_perception_reason = 'waiting for a valid synchronized camera sample'
        self.last_preview_at = -float('inf')
        self.world_frame = self.options['world_frame']
        self.mode = self.options['mode']
        if self.mode not in ('mono_cpu', 'mono_gpu', 'stereo', 'rgbd'):
            raise ValueError('Unknown 3D estimation mode')
        sizes = [self.options['face_width_m'], self.options['depth_scale']]
        if not np.isfinite(sizes).all() or min(sizes) <= 0:
            raise ValueError('Face width and metric depth scale must be positive')
        self.detector = create_detector(
            self.options['detector'], self.options['model_dir'],
            self.options['confidence'], threads=self.options['threads'])
        self.backend = None
        if self.mode == 'stereo':
            self.backend = StereoDepth(
                num_disparities=self.options['stereo_num_disparities'],
                block_size=self.options['stereo_block_size'])
        elif self.mode == 'mono_gpu':
            self.backend = MetricDepth(self.options['depth_weights'], self.options['depth_source'],
                                       self.options['depth_input_size'])
        elif self.mode == 'rgbd':
            self.backend = AlignedRGBDDepth(
                self.options['rgbd_16u_scale_m'], self.options['rgbd_32f_scale_m'])
        self.selector = NearestFaceSelector(
            self.options['switch_margin_m'], self.options['switch_delay_sec'])
        if self.options['point_smoothing_method'] == 'kalman':
            self.point_filter = KalmanPointFilter(
                measurement_std_m=self.options['point_kalman_measurement_std_m'],
                acceleration_std_mps2=self.options['point_kalman_acceleration_std_mps2'],
                reset_after_sec=self.options['point_smoothing_reset_after_sec'])
        else:
            self.point_filter = PointFilter(
                min_cutoff_hz=self.options['point_smoothing_min_cutoff_hz'],
                beta=self.options['point_smoothing_beta'],
                derivative_cutoff_hz=self.options['point_smoothing_derivative_cutoff_hz'],
                reset_after_sec=self.options['point_smoothing_reset_after_sec'])
        self.pending_observation = None
        self.observation_drops = 0
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.point_pub = self.create_publisher(PointStamped, '/face_test/center_3d', qos)
        self.world_pub = self.create_publisher(PointStamped, '/face_test/center_world', qos)
        self.detection_pub = self.create_publisher(Detection2DArray, '/face_test/detections', qos)
        self.image_pub = self.create_publisher(Image, '/face_test/debug_image', qos)
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/face_test/debug_image/compressed', qos)
        self.depth_pub = self.create_publisher(Image, '/face_test/depth_estimate', qos)
        self.marker_pub = self.create_publisher(Marker, '/face_test/markers', 1)
        self.status_pub = self.create_publisher(DiagnosticArray, '/face_test/depth_status', qos)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.perception_status_pub = self.create_publisher(
            PerceptionStatus, '/face/perception_status', status_qos)
        self.pending_world_point = None
        self.world_point_drops = 0
        self.world_timer = self.create_timer(0.01, self.flush_observation)
        self.status_timer = self.create_timer(
            .05, self.publish_watchdog_status, clock=Clock(clock_type=ClockType.STEADY_TIME))
        inputs = [(Image, '/face_test/image_raw'), (CameraInfo, '/face_test/camera_info')]
        if self.mode == 'rgbd':
            inputs.append((Image, self.options['rgbd_depth_topic']))
        if self.mode == 'stereo':
            inputs += [(Image, '/face_test/right/image_raw'),
                       (CameraInfo, '/face_test/right/camera_info')]
        # DDS keeps only the latest input while inference runs. The synchronizer's
        # separate small cache still matches images and CameraInfo arriving apart.
        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscribers = [message_filters.Subscriber(self, kind, topic, qos_profile=input_qos)
                            for kind, topic in inputs]
        slop = self.options['rgbd_sync_slop_sec'] if self.mode == 'rgbd' else (
            self.options['sync_slop_sec'])
        self.sync = (message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=2, slop=slop) if slop else
            message_filters.TimeSynchronizer(self.subscribers, queue_size=2))
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(f'{self.mode}: ready; outputs are meters, original frame stamps')
        self.last_report = time.monotonic()

    def on_frame(self, image_msg, info, auxiliary_msg=None, right_info=None):
        # A previously computed result may now have its exact TF. Give it a chance
        # before the next inference blocks this executor or replaces the result.
        self.flush_observation()
        mode = getattr(self, 'mode', 'mono_cpu')
        depth_msg = auxiliary_msg if mode == 'rgbd' else None
        right_msg = auxiliary_msg if mode == 'stereo' else None
        now = self.get_clock().now()
        inputs = (image_msg, info, depth_msg, right_msg, right_info)
        ages = [(now.nanoseconds - Time.from_msg(message.header.stamp).nanoseconds) / 1e9
                for message in inputs if message is not None]
        if any(age > self.options.get('max_frame_age_sec', .2) or age < 0 for age in ages):
            self.observation_drops += 1
            publish_state = getattr(self, 'publish_perception_state', None)
            if publish_state is not None:
                publish_state(PerceptionStatus.INVALID_INPUT,
                              'stale or future-stamped synchronized input')
            self.status_pub.publish(status_message(
                image_msg.header, 'face_depth', 'STALE_OBSERVATION',
                {'mode': self.mode, 'processing_ms': 0.0, 'frame_age_ms': ages[0] * 1000,
                 'oldest_input_age_ms': max(ages) * 1000, 'point_valid': False,
                 'observation_drops': self.observation_drops}, False))
            return
        if depth_msg is not None:
            rgb_depth_delta = abs(Time.from_msg(image_msg.header.stamp).nanoseconds
                                  - Time.from_msg(depth_msg.header.stamp).nanoseconds) / 1e9
            if rgb_depth_delta > self.options['rgbd_sync_slop_sec']:
                publish_state = getattr(self, 'publish_perception_state', None)
                if publish_state is not None:
                    publish_state(PerceptionStatus.SYNC_ERROR,
                                  'RGB and aligned depth stamps exceed sync tolerance')
                return
        capture_sec = Time.from_msg(image_msg.header.stamp).nanoseconds / 1e9
        last_capture_sec = getattr(self, 'last_capture_sec', None)
        if last_capture_sec is not None and capture_sec <= last_capture_sec:
            self.cadence.reset()
            self.selector.reset()
            self.point_filter.reset()
            self.last_selected_face_box = None
            self.last_reliable_fixed_point = None
            self.last_reliable_depth_stamp = None
            self.last_target_id = -1
            self.pair_cadence.reset()
            self.processing_cadence.reset()
            self.source_gap_limit = self.options['source_timeout_sec']
            publish_state = getattr(self, 'publish_perception_state', None)
            if publish_state is not None:
                publish_state(PerceptionStatus.INVALID_INPUT,
                              'duplicate or out-of-order RGB capture stamp')
            return
        if self.pending_observation is not None:
            # Replace pending work only after the incoming pair has passed its
            # timestamp checks; invalid packets cannot disturb a valid result.
            pending_since = getattr(self, 'pending_since', None)
            if pending_since is None:
                return
            if (time.monotonic() - pending_since >= self.options.get('tf_wait_timeout_sec', .05)):
                self.observation_drops += 1
                self.publish_observation(None, 'TF_UNAVAILABLE')
            else:
                self.pending_observation = None
                self.observation_drops += 1
        if (getattr(self, 'last_processed_at', None) is not None
                and not np.isfinite(self.last_processed_at)):
            return
        started = time.perf_counter()
        self.last_processed_at = started
        faces = []
        candidates = []
        partial_faces = []
        depth_map = None
        reason = 'NO_FACE'
        image = None
        camera = None
        to_original = None
        right_image = None
        right_camera = None
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            expected = self.options['input_left_frame']
            if expected and image_msg.header.frame_id != expected:
                raise ValueError(f'Left image frame must be {expected}')
            image, camera, to_original = self.rectifiers[0].apply(
                image, info, self.options['rectification'])
            if image_msg.header.frame_id != info.header.frame_id or image.shape[:2] != (
                    camera.height, camera.width):
                raise ValueError('Image and CameraInfo frame/size mismatch')
            if mode == 'stereo':
                expected_right = self.options['input_right_frame']
                if expected_right and right_msg.header.frame_id != expected_right:
                    raise ValueError(f'Right image frame must be {expected_right}')
                if (right_msg.header.frame_id != right_info.header.frame_id
                        or right_msg.header.frame_id == image_msg.header.frame_id):
                    raise ValueError('Invalid stereo optical frames')
                right_image = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                right_image, right_camera, _ = self.rectifiers[1].apply(
                    right_image, right_info, self.options['rectification'])
                right_size = (right_camera.width, right_camera.height)
                if right_image.shape != image.shape or right_size != (camera.width, camera.height):
                    raise ValueError('Stereo image sizes differ')
                baseline = self.options['stereo_baseline_m']
                if baseline:
                    right_camera = replace(right_camera, tx=right_camera.fx * (
                        camera.tx / camera.fx - baseline))
                stereo_baseline(camera, right_camera)
            if mode == 'rgbd':
                if (depth_msg.header.frame_id != image_msg.header.frame_id
                        or depth_msg.width != camera.width or depth_msg.height != camera.height):
                    raise ValueError('Aligned depth must match RGB dimensions and optical frame')
                try:
                    raw_depth = self.bridge.imgmsg_to_cv2(
                        depth_msg, desired_encoding='passthrough')
                except TypeError as error:
                    raise ValueError(f'Malformed aligned depth payload: {error}') from error
                if raw_depth.shape != image.shape[:2]:
                    raise ValueError('Aligned depth dimensions do not match RGB image')
                raw_depth = self.rectifiers[0].apply_depth(
                    raw_depth, self.options['rectification'])
                depth_map = self.backend.convert(raw_depth, depth_msg.encoding)
                self.last_depth_stamp = depth_msg.header.stamp
            # Only calibrated, decoded input pairs may train cadence or refresh
            # the pair watchdog. This precedes the optional inference rate gate
            # so skipped inference and empty-but-valid depth maps still reflect a
            # valid camera observation.
            valid_input_receipt = time.monotonic()
            self.last_valid_input_receipt = valid_input_receipt
            self.last_pair_receipt = valid_input_receipt
            if hasattr(self, 'pair_cadence'):
                self.pair_cadence.observe(valid_input_receipt, valid_input_receipt)
            if hasattr(self, 'cadence'):
                self.cadence.observe(capture_sec, valid_input_receipt)
            self.last_capture_sec = capture_sec
            if (hasattr(self, 'pair_cadence')
                    and self.options.get('timing_mode', 'auto') == 'auto'):
                self.source_gap_limit = self.pair_cadence.bounded_gap_limit(
                    self.options.get('source_timeout_sec', .25))
            processing_started = time.monotonic()
            rate_gate = getattr(self, 'rate_gate', None)
            if rate_gate is not None and not rate_gate.accept(processing_started):
                return
            faces = self.detector.detect(image)
            valid = [(i, face) for i, face in enumerate(faces) if visible_face(face, camera)]
            if valid and mode == 'stereo':
                depth_map = self.backend.compute(image, right_image, camera, right_camera)
            elif valid and mode == 'mono_gpu':
                depth_map = self.backend.compute(image) * self.options['depth_scale']
            elif valid and mode == 'rgbd':
                pass  # The aligned depth image is validated even when no face is visible.
            for index, face in valid:
                try:
                    if mode == 'mono_cpu':
                        if not fully_visible_face(face, camera):
                            partial_faces.append((index, face))
                            continue
                        z = size_depth(face, camera, self.options['face_width_m'])
                        u, v = face.center
                    else:
                        z, u, v = sample_face_measurement(depth_map, face)
                    if np.isfinite(z) and .25 <= z <= 8.0:
                        xyz = to_original @ camera.unproject(u, v, z)
                        candidates.append((index, tuple(float(v) for v in xyz)))
                except ValueError:
                    continue  # An invalid face must not hide other usable candidates.
            if faces:
                reason = 'CONFIRMING_FACE' if candidates else 'NO_VALID_FACE_DEPTH'
        except (CvBridgeError, ValueError, cv2.error) as error:
            candidates = []
            depth_map = None
            reason = f'INVALID_INPUT: {error}'
        if (not reason.startswith('INVALID_INPUT:') and reason != 'DEPTH_STREAM_LOST'
                and hasattr(self, 'processing_cadence')):
            self.processing_cadence.observe(processing_started, processing_started)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.pending_observation = (
            image_msg, image, faces, candidates, depth_map, elapsed_ms, reason,
            partial_faces, camera if image is not None else None,
            to_original if image is not None else None)
        if getattr(self, 'pending_since', None) is None:
            self.pending_since = time.monotonic()
        self.flush_observation()

    def flush_observation(self):
        """Retry only the newest frame against its exact camera pose, without blocking."""
        if self.pending_observation is not None:
            pending = self.pending_observation
            image_msg, _, faces, candidates, _, _, reason = pending[:7]
            partial_faces = pending[7] if len(pending) > 7 else []
            camera = pending[8] if len(pending) > 8 else None
            to_original = pending[9] if len(pending) > 9 else None
            # Associate against the previous target before dropping detections
            # whose aligned depth is unusable. Otherwise a valid background face
            # could mask a depth fault on the person already being tracked.
            candidate_indices = {index for index, _ in candidates}
            selected_still_visible_without_depth = (
                getattr(self, 'mode', '') == 'rgbd' and selected_target_missing_depth(
                    getattr(self, 'last_selected_face_box', None), faces, candidate_indices))
            if selected_still_visible_without_depth:
                self.publish_observation(None, 'NO_VALID_FACE_DEPTH')
                return
            stamp = Time.from_msg(image_msg.header.stamp)
            age = (self.get_clock().now().nanoseconds - stamp.nanoseconds) / 1e9
            if age > self.options.get('max_frame_age_sec', .2) or age < 0:
                self.observation_drops += 1
                self.publish_observation(None, 'STALE_OBSERVATION')
            else:
                can_reuse_cpu_depth = (
                    getattr(self, 'mode', '') == 'mono_cpu' and partial_faces
                    and camera is not None
                    and to_original is not None
                    and getattr(self, 'last_reliable_fixed_point', None) is not None
                    and getattr(self, 'last_reliable_depth_stamp', None) is not None
                    and any(same_visible_face(getattr(self, 'last_selected_face_box', None), face)
                            for _, face in partial_faces))
                if not candidates and not can_reuse_cpu_depth:
                    self.selector.select([], stamp.nanoseconds / 1e9)
                    self.publish_observation(None, reason)
                    self.publish_world_point()
                    return
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.options['selection_frame'], image_msg.header.frame_id, stamp)
                    smoothing = self.options.get('point_smoothing', False)
                    need_inverse = smoothing or bool(can_reuse_cpu_depth)
                    inverse = None
                    if need_inverse:
                        # Resolve both directions at capture time before state updates.
                        inverse = self.tf_buffer.lookup_transform(
                            image_msg.header.frame_id, self.options['selection_frame'], stamp)
                except TransformException:
                    pending_since = getattr(self, 'pending_since', None)
                    if (pending_since is not None and time.monotonic() - pending_since
                            >= self.options.get('tf_wait_timeout_sec', .05)):
                        self.publish_observation(None, 'TF_UNAVAILABLE')
                    return
                candidates = list(candidates)
                partial_candidate_indices = set()
                if (can_reuse_cpu_depth and self.mode == 'mono_cpu' and partial_faces
                        and camera is not None
                        and to_original is not None and inverse is not None
                        and self.last_reliable_fixed_point is not None
                        and self.last_reliable_depth_stamp is not None):
                    capture_sec = stamp.nanoseconds / 1e9
                    depth_age = capture_sec - (
                        Time.from_msg(self.last_reliable_depth_stamp).nanoseconds / 1e9)
                    if 0 <= depth_age <= self.options.get('mono_partial_depth_max_age_sec', .30):
                        previous_box = getattr(self, 'last_selected_face_box', None)
                        for index, face in partial_faces:
                            if not same_visible_face(previous_box, face):
                                continue
                            old_fixed = stamped_point(
                                Header(frame_id=self.options['selection_frame']),
                                self.last_reliable_fixed_point)
                            current_camera = do_transform_point(old_fixed, inverse).point
                            z = float(current_camera.z)
                            if not np.isfinite(z) or not .25 <= z <= 8.0:
                                continue
                            u, v = partial_face_center(
                                face, camera, z, self.options['face_width_m'])
                            xyz = to_original @ camera.unproject(u, v, z)
                            candidates.append((index, tuple(float(value) for value in xyz)))
                            partial_candidate_indices.add(index)
                            break
                self.pending_observation = (
                    *pending[:3], candidates, *pending[4:])
                if not candidates:
                    self.selector.select([], stamp.nanoseconds / 1e9)
                    self.publish_observation(None, reason)
                    self.publish_world_point()
                    return
                fixed = []
                for _, xyz in candidates:
                    point = do_transform_point(
                        stamped_point(image_msg.header, xyz), transform).point
                    fixed.append((point.x, point.y, point.z))
                selected = self.selector.select(fixed, stamp.nanoseconds / 1e9)
                reason = 'OK' if selected is not None else 'CONFIRMING_FACE'
                if selected is None:
                    self.publish_observation(None, reason)
                elif smoothing:
                    filtered = self.point_filter.update(
                        fixed[selected], stamp.nanoseconds / 1e9, self.selector.track_id)
                    fixed_point = stamped_point(transform.header, filtered)
                    point = do_transform_point(fixed_point, inverse).point
                    self.pending_selected_fixed_xyz = fixed[selected]
                    self.pending_partial_depth = (
                        candidates[selected][0] in partial_candidate_indices)
                    self.publish_observation(selected, reason, (point.x, point.y, point.z))
                else:
                    self.pending_selected_fixed_xyz = fixed[selected]
                    self.pending_partial_depth = (
                        candidates[selected][0] in partial_candidate_indices)
                    self.publish_observation(selected, reason)
        self.publish_world_point()

    def publish_observation(self, selected, reason, filtered_xyz=None):
        image_msg, image, faces, candidates, depth_map, elapsed_ms, _ = (
            self.pending_observation[:7])
        self.pending_observation = None
        self.pending_since = None
        header = image_msg.header
        partial_depth_observation = getattr(self, 'pending_partial_depth', False)
        selected_fixed_xyz = getattr(self, 'pending_selected_fixed_xyz', None)
        selected_index, raw_xyz = candidates[selected] if selected is not None else (None, None)
        xyz = filtered_xyz if filtered_xyz is not None else raw_xyz
        if xyz is not None:
            self.last_target_id = self.selector.track_id
            if selected_index is not None and selected_index < len(faces):
                face = faces[selected_index]
                self.last_selected_face_box = (
                    float(face.x), float(face.y), float(face.width), float(face.height))
            if (getattr(self, 'mode', '') == 'mono_cpu' and selected_fixed_xyz is not None
                    and not partial_depth_observation):
                self.last_reliable_fixed_point = tuple(
                    float(value) for value in selected_fixed_xyz)
                self.last_reliable_depth_stamp = header.stamp
        self.pending_selected_fixed_xyz = None
        self.pending_partial_depth = False
        self.detection_pub.publish(detection_message(header, faces))
        if xyz is not None:
            point = stamped_point(header, xyz)
            self.point_pub.publish(point)
            self.marker_pub.publish(point_marker(point, 'estimate', (0.1, 1.0, 0.2)))
            # Joint TF may arrive after this image. Retry the latest observation
            # without blocking image processing or using a pose from another time.
            self.pending_world_point = point if self.world_frame else None
            self.publish_world_point()
        else:
            self.pending_world_point = None
            self.marker_pub.publish(Marker(
                header=header, ns='estimate', id=0, action=Marker.DELETE))
        values = {'mode': self.mode, 'processing_ms': round(elapsed_ms, 3),
                  'frame_age_ms': (self.get_clock().now().nanoseconds
                                   - Time.from_msg(header.stamp).nanoseconds) / 1e6,
                  'faces': len(faces), 'point_valid': xyz is not None,
                  'world_point_pending': self.pending_world_point is not None,
                  'world_point_drops': self.world_point_drops,
                  'observation_drops': self.observation_drops, 'valid_faces': len(candidates),
                  'selected_track_id': self.selector.track_id if xyz is not None else 0,
                  'point_smoothing': self.options.get('point_smoothing', False),
                  'point_smoothing_method': self.options.get(
                      'point_smoothing_method', 'one_euro'),
                  'selection_frame': self.options['selection_frame']}
        values.update({
            'capture_fps': getattr(getattr(self, 'cadence', None), 'fps', None),
            'synchronized_pair_fps': getattr(
                getattr(self, 'pair_cadence', None), 'fps', None),
            'processing_fps': getattr(
                getattr(self, 'processing_cadence', None), 'fps', None),
            'rgb_depth_skew_ms': (abs(Time.from_msg(header.stamp).nanoseconds
                                      - Time.from_msg(self.last_depth_stamp).nanoseconds)
                                  / 1e6 if getattr(self, 'last_depth_stamp', None) is not None
                                  else None),
            'source_gap_limit_sec': getattr(self, 'source_gap_limit', None),
            'timing_mode': self.options.get('timing_mode', 'auto'),
        })
        if xyz is not None:
            values.update(dict(zip(('x_m', 'y_m', 'z_m'), xyz)))
            values.update(dict(zip(('raw_x_m', 'raw_y_m', 'raw_z_m'), raw_xyz)))
            values['point_smoothing_correction_m'] = float(
                np.linalg.norm(np.asarray(xyz) - raw_xyz))
            values['selected_distance_m'] = float(np.linalg.norm(self.selector.current))
            if (partial_depth_observation
                    and getattr(self, 'last_reliable_depth_stamp', None) is not None):
                values['retained_depth_age_sec'] = max(0.0, (
                    Time.from_msg(header.stamp).nanoseconds
                    - Time.from_msg(self.last_reliable_depth_stamp).nanoseconds) / 1e9)
        self.status_pub.publish(status_message(
            header, 'face_depth', reason, values, xyz is not None))
        if xyz is not None:
            state = PerceptionStatus.OK
        elif reason == 'TF_UNAVAILABLE':
            state = PerceptionStatus.TF_UNAVAILABLE
        elif reason == 'STALE_OBSERVATION':
            state = PerceptionStatus.INVALID_INPUT
        elif reason == 'DEPTH_STREAM_LOST':
            state = PerceptionStatus.DEPTH_STREAM_LOST
        elif reason.startswith('INVALID_INPUT:'):
            state = PerceptionStatus.INVALID_INPUT
        elif reason == 'NO_VALID_FACE_DEPTH' and faces:
            state = PerceptionStatus.FACE_NO_DEPTH
        elif reason == 'CONFIRMING_FACE':
            state = PerceptionStatus.WAIT
        elif not faces:
            state = PerceptionStatus.NO_FACE
        else:
            state = PerceptionStatus.INVALID_INPUT
        publish_state = getattr(self, 'publish_perception_state', None)
        if publish_state is not None:
            publish_state(state, reason, header.stamp)
        if depth_map is not None and self.depth_pub.get_subscription_count():
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), '32FC1')
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)
        preview_time = time.monotonic()
        raw_preview = self.image_pub.get_subscription_count()
        compressed_preview = self.compressed_pub.get_subscription_count()
        if (image is not None and (raw_preview or compressed_preview)
                and preview_time - self.last_preview_at >= 1 / self.options['preview_fps']):
            self.last_preview_at = preview_time
            annotated = image.copy()
            for index, face in enumerate(faces):
                x, y, w, h = (round(v) for v in (face.x, face.y, face.width, face.height))
                color = (0, 230, 0) if index == selected_index else (0, 180, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                if index == selected_index and xyz is not None:
                    cv2.circle(annotated, tuple(round(v) for v in face.center), 4,
                               (0, 0, 255), -1)
            caption = f'{self.mode} {elapsed_ms:.1f} ms: {reason}'
            if xyz is not None:
                caption = (f'{self.mode} {elapsed_ms:.1f} ms: '
                           f'XYZ {xyz[0]:.2f} {xyz[1]:.2f} {xyz[2]:.2f} m')
            cv2.putText(annotated, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(annotated, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            if raw_preview:
                debug = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                debug.header = header
                self.image_pub.publish(debug)
            if compressed_preview:
                ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.compressed_pub.publish(CompressedImage(
                        header=header, format='bgr8; jpeg compressed bgr8',
                        data=encoded.tobytes()))
        if time.monotonic() - self.last_report > 3:
            self.get_logger().info(f'{reason}, processing={elapsed_ms:.1f} ms, point={xyz}')
            self.last_report = time.monotonic()

    def publish_perception_state(self, state, reason, source_stamp=None):
        """Publish typed motion-gating state; only real samples update source time."""
        publisher = getattr(self, 'perception_status_pub', None)
        if publisher is None:
            return
        if source_stamp is not None:
            self.last_source_stamp = source_stamp
        message = PerceptionStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        if self.last_source_stamp is not None:
            message.source_stamp = self.last_source_stamp
        message.state = state
        message.reason = str(reason)
        if hasattr(message, 'target_id'):
            message.target_id = self.last_target_id
        publisher.publish(message)
        self.last_perception_state = state
        self.last_perception_reason = str(reason)

    def publish_watchdog_status(self):
        """Heartbeat without synchronized pairs; never refresh capture time."""
        now = time.monotonic()
        if self.last_pair_receipt is None:
            state, reason = (PerceptionStatus.WAIT,
                             'waiting for a valid synchronized camera sample')
        elif now - self.last_pair_receipt > self.source_gap_limit:
            state, reason = (PerceptionStatus.DEPTH_STREAM_LOST,
                             'camera pair watchdog expired')
        else:
            state, reason = self.last_perception_state, self.last_perception_reason
        self.publish_perception_state(state, reason)

    def publish_world_point(self):
        point = self.pending_world_point
        if point is None:
            return
        measurement_time = Time.from_msg(point.header.stamp)
        age = (self.get_clock().now().nanoseconds - measurement_time.nanoseconds) / 1e9
        maximum_age = getattr(self, 'options', {}).get('max_frame_age_sec', .2)
        if age > maximum_age or age < 0:
            self.pending_world_point = None
            self.world_point_drops += 1
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                getattr(self, 'world_frame', 'world'), point.header.frame_id, measurement_time)
        except TransformException:
            return
        world_point = do_transform_point(point, transform)
        world_point.header.stamp = point.header.stamp
        self.world_pub.publish(world_point)
        self.pending_world_point = None


def main():
    rclpy.init()
    node = None
    try:
        node = FacePositionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
