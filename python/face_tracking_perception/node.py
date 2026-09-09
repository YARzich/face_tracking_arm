# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""ROS image adapter for the standalone face detector."""

import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .detectors import create_detector


def detection_message(header, faces):
    """Publish an empty array on a miss, preserving the observation's stamp/frame."""
    result = Detection2DArray(header=header)
    for face in faces:
        detection = Detection2D(header=header)
        detection.bbox.center.position.x, detection.bbox.center.position.y = face.center
        detection.bbox.size_x = face.width
        detection.bbox.size_y = face.height
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = 'face'
        hypothesis.hypothesis.score = face.confidence
        detection.results.append(hypothesis)
        result.detections.append(detection)
    return result


class FaceDetectionNode(Node):
    """Process the newest queued image in a single executor callback."""

    def __init__(self):
        super().__init__('face_detection_test')
        self.declare_parameter('detector', 'yunet')
        self.declare_parameter('model_dir', '')
        self.declare_parameter('confidence', 0.6)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('threads', 2)
        self.name = self.get_parameter('detector').value
        self.detector = create_detector(
            self.name, self.get_parameter('model_dir').value,
            self.get_parameter('confidence').value,
            self.get_parameter('nms_threshold').value,
            self.get_parameter('threads').value)
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.results = self.create_publisher(Detection2DArray, '/face_test/detections', qos)
        self.debug = self.create_publisher(Image, '/face_test/debug_image', qos)
        self.subscription = self.create_subscription(
            Image, '/face_test/image_raw', self.on_image, qos)
        self.last_report = time.monotonic()
        self.get_logger().info(f'Detector ready: {self.name} (CPU); coordinates are pixels')

    def on_image(self, message):
        started = time.perf_counter()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            faces = self.detector.detect(image)
        except (CvBridgeError, ValueError, cv2.error) as error:
            self.results.publish(detection_message(message.header, []))
            self.get_logger().warning(f'Invalid camera frame: {error}', throttle_duration_sec=3.0)
            return
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.results.publish(detection_message(message.header, faces))
        if self.debug.get_subscription_count():
            annotated = image.copy()
            for face in faces:
                first = (round(face.x), round(face.y))
                last = (round(face.x + face.width), round(face.y + face.height))
                cv2.rectangle(annotated, first, last, (0, 230, 0), 2)
                cv2.circle(annotated, tuple(round(v) for v in face.center), 4, (0, 0, 255), -1)
                cv2.putText(annotated, f'{face.confidence:.2f}', first,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
            caption = f'{self.name}  {elapsed_ms:.1f} ms  faces: {len(faces)}'
            cv2.putText(annotated, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3)
            cv2.putText(annotated, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            debug = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug.header = message.header
            self.debug.publish(debug)
        if time.monotonic() - self.last_report >= 3:
            self.get_logger().info(f'faces={len(faces)}, processing={elapsed_ms:.1f} ms')
            self.last_report = time.monotonic()


def main():
    rclpy.init()
    node = None
    try:
        node = FaceDetectionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
