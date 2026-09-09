# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Small ROS conversion helpers shared by estimation and independent evaluation."""

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker


def stamped_point(header, xyz):
    return PointStamped(
        header=header, point=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])))


def point_marker(point, namespace, color):
    marker = Marker(header=point.header, ns=namespace, id=0, type=Marker.SPHERE)
    marker.pose.position = point.point
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 0.045
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 0.9
    marker.lifetime.nanosec = 400000000
    return marker


def status_message(header, name, state, values, valid):
    status = DiagnosticStatus(
        name=name, hardware_id='simulated_camera', message=state,
        level=DiagnosticStatus.OK if valid else DiagnosticStatus.WARN)
    status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
    return DiagnosticArray(header=header, status=[status])
