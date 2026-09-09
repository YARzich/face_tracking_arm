# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Independent evaluator; simulator depth is never an input to face estimation."""

from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PointStamped
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker

from .depth_geometry import Camera
from .depth_messages import point_marker, stamped_point, status_message


class DepthReference(Node):
    """Compare estimated Z against rendered surface depth on the same face ray."""

    def __init__(self):
        super().__init__('face_depth_reference')
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        inputs = [(PointStamped, '/face_test/center_3d'),
                  (Image, '/face_test/reference/depth'),
                  (CameraInfo, '/face_test/camera_info')]
        self.subscribers = [message_filters.Subscriber(self, kind, topic, qos_profile=qos)
                            for kind, topic in inputs]
        self.sync = message_filters.TimeSynchronizer(self.subscribers, queue_size=8)
        self.sync.registerCallback(self.compare)
        self.point_pub = self.create_publisher(PointStamped, '/face_test/reference_3d', qos)
        self.error_pub = self.create_publisher(DiagnosticArray, '/face_test/depth_error', qos)
        self.marker_pub = self.create_publisher(Marker, '/face_test/markers', 1)

    def compare(self, point, depth_msg, info):
        if not (point.header.frame_id == depth_msg.header.frame_id == info.header.frame_id):
            return
        camera = Camera.from_info(info)
        if (not np.isfinite([point.point.x, point.point.y, point.point.z]).all()
                or point.point.z <= 0):
            return
        u = round(camera.fx * point.point.x / point.point.z + camera.cx)
        v = round(camera.fy * point.point.y / point.point.z + camera.cy)
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            return
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        if depth.shape != (camera.height, camera.width):
            return
        patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.25) & (patch < 8)]
        if len(valid) < 9:
            return
        truth = stamped_point(point.header, camera.unproject(u, v, float(np.median(valid))))
        error = np.linalg.norm([point.point.x - truth.point.x, point.point.y - truth.point.y,
                                point.point.z - truth.point.z])
        values = {'error_m': float(error), 'estimated_z_m': point.point.z,
                  'reference_z_m': truth.point.z, 'z_bias_m': point.point.z - truth.point.z}
        self.point_pub.publish(truth)
        self.marker_pub.publish(point_marker(truth, 'reference', (0.1, 0.4, 1.0)))
        self.error_pub.publish(status_message(
            point.header, 'face_depth_reference', 'SURFACE_DEPTH_COMPARISON', values, True))


def main():
    rclpy.init()
    node = DepthReference()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
