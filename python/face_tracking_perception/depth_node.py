# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Synchronized ROS adapter for the three interchangeable 3D face experiments."""

from dataclasses import replace
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PointStamped
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker

from .depth_backends import MetricDepth, StereoDepth
from .depth_geometry import sample_face_depth, size_depth, visible_face
from .depth_messages import point_marker, stamped_point, status_message
from .detectors import create_detector
from .face_selection import NearestFaceSelector
from .node import detection_message
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
            'rectification': 'none', 'sync_slop_sec': 0.0, 'stereo_baseline_m': 0.0,
            'max_processing_fps': 0.0, 'max_frame_age_sec': .2,
            'preview_fps': 5.0,
            'input_left_frame': '', 'input_right_frame': '', 'world_frame': 'world',
        }
        self.declare_parameters('', list(defaults.items()))
        self.options = {k: self.get_parameter(k).value for k in defaults}
        for name, low, high in (
                ('sync_slop_sec', 0, .02), ('stereo_baseline_m', 0, 1),
                ('max_processing_fps', 0, 120), ('max_frame_age_sec', .02, .5),
                ('preview_fps', 1, 30)):
            if not low <= self.options[name] <= high:
                raise ValueError(f'{name} must be {low}..{high}')
        if self.options['rectification'] not in ('none', 'raw', 'rectified'):
            raise ValueError('Unknown rectification mode')
        self.rectifiers = (Rectification(), Rectification())
        self.last_processed_at = None
        self.last_preview_at = -float('inf')
        self.world_frame = self.options['world_frame']
        self.mode = self.options['mode']
        if self.mode not in ('mono_cpu', 'mono_gpu', 'stereo'):
            raise ValueError('Unknown 3D estimation mode')
        sizes = [self.options['face_width_m'], self.options['depth_scale']]
        if not np.isfinite(sizes).all() or min(sizes) <= 0:
            raise ValueError('Face width and metric depth scale must be positive')
        self.detector = create_detector(
            self.options['detector'], self.options['model_dir'],
            self.options['confidence'], threads=self.options['threads'])
        self.backend = None
        if self.mode == 'stereo':
            self.backend = StereoDepth()
        elif self.mode == 'mono_gpu':
            self.backend = MetricDepth(self.options['depth_weights'], self.options['depth_source'],
                                       self.options['depth_input_size'])
        self.selector = NearestFaceSelector(
            self.options['switch_margin_m'], self.options['switch_delay_sec'])
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
        self.pending_world_point = None
        self.world_point_drops = 0
        self.world_timer = self.create_timer(0.01, self.flush_observation)
        inputs = [(Image, '/face_test/image_raw'), (CameraInfo, '/face_test/camera_info')]
        if self.mode == 'stereo':
            inputs += [(Image, '/face_test/right/image_raw'),
                       (CameraInfo, '/face_test/right/camera_info')]
        # CameraInfo and a large image can arrive at different times. Keep a small
        # matching buffer; output queues still contain only the newest result.
        input_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscribers = [message_filters.Subscriber(self, kind, topic, qos_profile=input_qos)
                            for kind, topic in inputs]
        slop = self.options['sync_slop_sec']
        self.sync = (message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=4, slop=slop) if slop else
            message_filters.TimeSynchronizer(self.subscribers, queue_size=4))
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(f'{self.mode}: ready; outputs are meters, original frame stamps')
        self.last_report = time.monotonic()

    def on_frame(self, image_msg, info, right_msg=None, right_info=None):
        started = time.perf_counter()
        fps = self.options['max_processing_fps']
        if (fps and self.last_processed_at is not None
                and started - self.last_processed_at < 1 / fps):
            return
        self.last_processed_at = started
        faces = []
        candidates = []
        depth_map = None
        reason = 'NO_FACE'
        image = None
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
            faces = self.detector.detect(image)
            valid = [(i, face) for i, face in enumerate(faces) if visible_face(face, camera)]
            if valid and self.mode == 'stereo':
                expected_right = self.options['input_right_frame']
                if expected_right and right_msg.header.frame_id != expected_right:
                    raise ValueError(f'Right image frame must be {expected_right}')
                if (right_msg.header.frame_id != right_info.header.frame_id
                        or right_msg.header.frame_id == image_msg.header.frame_id):
                    raise ValueError('Invalid stereo optical frames')
                right = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                right, right_camera, _ = self.rectifiers[1].apply(
                    right, right_info, self.options['rectification'])
                baseline = self.options['stereo_baseline_m']
                if baseline:
                    right_camera = replace(right_camera, tx=right_camera.fx * (
                        camera.tx / camera.fx - baseline))
                if right.shape != image.shape:
                    raise ValueError('Stereo image sizes differ')
                depth_map = self.backend.compute(image, right, camera, right_camera)
            elif valid and self.mode == 'mono_gpu':
                depth_map = self.backend.compute(image) * self.options['depth_scale']
            for index, face in valid:
                try:
                    z = (size_depth(face, camera, self.options['face_width_m'])
                         if self.mode == 'mono_cpu' else sample_face_depth(depth_map, face))
                    if np.isfinite(z) and .25 <= z <= 8.0:
                        xyz = to_original @ camera.unproject(*face.center, z)
                        candidates.append((index, tuple(float(v) for v in xyz)))
                except ValueError:
                    continue  # An invalid face must not hide other usable candidates.
            if faces:
                reason = 'CONFIRMING_FACE' if candidates else 'NO_VALID_FACE_DEPTH'
        except (CvBridgeError, ValueError, cv2.error) as error:
            candidates = []
            depth_map = None
            reason = str(error)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.pending_observation = (
            image_msg, image, faces, candidates, depth_map, elapsed_ms, reason)
        self.flush_observation()

    def flush_observation(self):
        """Retry only the newest frame against its exact camera pose, without blocking."""
        if self.pending_observation is not None:
            image_msg, _, _, candidates, _, _, reason = self.pending_observation
            stamp = Time.from_msg(image_msg.header.stamp)
            age = (self.get_clock().now() - stamp).nanoseconds / 1e9
            if age > self.options.get('max_frame_age_sec', .2) or age < -.05:
                self.observation_drops += 1
                self.publish_observation(None, 'STALE_OBSERVATION')
            elif not candidates:
                self.selector.select([], stamp.nanoseconds / 1e9)
                self.publish_observation(None, reason)
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.options['selection_frame'], image_msg.header.frame_id, stamp)
                except TransformException:
                    return
                fixed = []
                for _, xyz in candidates:
                    point = do_transform_point(
                        stamped_point(image_msg.header, xyz), transform).point
                    fixed.append((point.x, point.y, point.z))
                selected = self.selector.select(fixed, stamp.nanoseconds / 1e9)
                reason = 'OK' if selected is not None else 'CONFIRMING_FACE'
                self.publish_observation(selected, reason)
        self.publish_world_point()

    def publish_observation(self, selected, reason):
        image_msg, image, faces, candidates, depth_map, elapsed_ms, _ = self.pending_observation
        self.pending_observation = None
        selected_index, xyz = candidates[selected] if selected is not None else (None, None)
        header = image_msg.header
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
                  'frame_age_ms': (self.get_clock().now() - Time.from_msg(header.stamp)
                                   ).nanoseconds / 1e6,
                  'faces': len(faces), 'point_valid': xyz is not None,
                  'world_point_pending': self.pending_world_point is not None,
                  'world_point_drops': self.world_point_drops,
                  'observation_drops': self.observation_drops, 'valid_faces': len(candidates),
                  'selected_track_id': self.selector.track_id if xyz is not None else 0,
                  'selection_frame': self.options['selection_frame']}
        if xyz is not None:
            values.update(dict(zip(('x_m', 'y_m', 'z_m'), xyz)))
            values['selected_distance_m'] = float(np.linalg.norm(self.selector.current))
        self.status_pub.publish(status_message(
            header, 'face_depth', reason, values, xyz is not None))
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
                cv2.circle(annotated, tuple(round(v) for v in face.center), 4, (0, 0, 255), -1)
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

    def publish_world_point(self):
        point = self.pending_world_point
        if point is None:
            return
        measurement_time = Time.from_msg(point.header.stamp)
        age = (self.get_clock().now() - measurement_time).nanoseconds / 1e9
        maximum_age = getattr(self, 'options', {}).get('max_frame_age_sec', .2)
        if age > maximum_age or age < -0.05:
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
