# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Move only a Gazebo person; never publish measured faces or robot commands."""

import math

from diagnostic_msgs.msg import DiagnosticArray
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from .appearance_motion import AppearanceMotion, companion_center
from .appearance_sequence import AppearanceSequence
from .depth_messages import status_message


class FaceAppearances(Node):
    """Wait for tracking readiness and the measured SRDF rest pose between people."""

    def __init__(self):
        super().__init__('face_appearances')
        self.declare_parameters('', [
            ('visible_sec', 12.0), ('rest_sec', 8.0), ('cycles', 0),
            ('motion_lateral_m', 0.14), ('motion_depth_m', 0.07),
            ('motion_yaw_rad', 0.07), ('motion_period_sec', 6.0),
            ('idle_behavior', 'rest'), ('people_count', 1),
            ('positions_xy', [2.35, 0.0, 2.25, -0.35, 2.60, 0.40]),
            ('rest_joints', [0.0] * 6)])
        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.positions = np.asarray(value('positions_xy')).reshape(-1, 2)
        self.rest_joints = np.asarray(value('rest_joints'))
        if (not np.isfinite(self.positions).all() or (self.positions[:, 0] <= .8).any()
                or self.rest_joints.shape != (6,) or not np.isfinite(self.rest_joints).all()):
            raise ValueError('Invalid person positions or six-joint rest configuration')
        self.searching = value('idle_behavior') != 'rest'
        self.people_count = value('people_count')
        if self.people_count not in (1, 2):
            raise ValueError('people_count must be 1 or 2')
        self.sequence = AppearanceSequence(len(self.positions), value('visible_sec'),
                                           value('rest_sec'), value('cycles'), self.searching)
        self.motion = AppearanceMotion(value('motion_lateral_m'), value('motion_depth_m'),
                                       value('motion_yaw_rad'), value('motion_period_sec'))
        self.ready = False
        self.mode = ''
        self.tracking_stamp = None
        self.joint_stamp = None
        self.servo_stamp = None
        self.search_ready = False
        self.at_rest = False
        self.pending = None
        self.retry_at = 0.0
        self.client = self.create_client(SetEntityPose, '/world/lite6_table/set_pose/blocking')
        self.tracking_sub = self.create_subscription(
            DiagnosticArray, '/tracking/diagnostics', self.on_tracking, 1)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.on_joints, qos_profile_sensor_data)
        self.servo_sub = self.create_subscription(
            DiagnosticArray, '/servo_node/diagnostics', self.on_servo, 1)
        self.publisher = self.create_publisher(DiagnosticArray, '/face_test/scenario', 1)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

    def seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_tracking(self, message):
        for status in message.status:
            if status.name == 'face_tracking_arm/tracking':
                values = {v.key: v.value for v in status.values}
                self.ready = values.get('ready') == 'true'
                self.mode = values.get('mode', '')
                self.tracking_stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9

    def on_joints(self, message):
        self.at_rest = False
        try:
            indices = [message.name.index(f'joint{i}') for i in range(1, 7)]
            q = np.array([message.position[i] for i in indices])
            v = np.array([message.velocity[i] for i in indices])
        except (ValueError, IndexError):
            return
        self.joint_stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        self.at_rest = bool(np.isfinite(q).all() and np.isfinite(v).all()
                            and np.max(np.abs(q - self.rest_joints)) < .035
                            and np.max(np.abs(v)) < .035)

    def on_servo(self, message):
        for status in message.status:
            values = {v.key: v.value for v in status.values}
            if 'motion_reference_state' in values:
                self.search_ready = values['motion_reference_state'] in (
                    'SEARCH_LOCAL', 'SEARCH_SWEEP')
                self.servo_stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9

    def tick(self):
        now = self.seconds()
        fresh = all(stamp is not None and 0 <= now - stamp < .5
                    for stamp in (self.tracking_stamp, self.joint_stamp))
        idle_ready = self.at_rest and self.mode == 'REST'
        if self.searching:
            idle_ready = (self.mode == 'SEARCH' and self.search_ready
                          and self.servo_stamp is not None and 0 <= now - self.servo_stamp < .5)
        action = self.sequence.action(now, self.ready and fresh, idle_ready,
                                      tracking_face=self.mode == 'FACE' and self.ready and fresh)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        values = {'index': self.sequence.index, 'appearances': self.sequence.appearances,
                  'ready': self.ready and fresh, 'at_rest': self.at_rest,
                  'tracking_mode': self.mode, 'pose_request_pending': self.pending is not None,
                  'discovery_timeouts': self.sequence.discovery_timeouts,
                  'people_count': self.people_count}
        self.publisher.publish(status_message(
            header, 'face_appearances', self.sequence.phase, values, fresh))
        if action is None and self.sequence.phase in ('VISIBLE', 'ACQUIRING'):
            action = 'move'
        if (not action or self.pending is not None or now < self.retry_at
                or not self.client.service_is_ready()):
            return
        futures = []
        for index in range(self.people_count):
            request = SetEntityPose.Request()
            name = 'face_test_person' + ('_2' if index else '')
            request.entity = Entity(name=name, type=Entity.MODEL)
            request.pose.orientation.w = 1.0
            request.pose.position.z = -10.0
            if action in ('show', 'move'):
                elapsed = now - self.sequence.shown_at if action == 'move' else 0.0
                center = self.positions[self.sequence.index]
                if index:
                    center = companion_center(center, elapsed)
                x, y, yaw = self.motion.pose(center, elapsed)
                request.pose.position.x, request.pose.position.y = x, y
                request.pose.position.z = 0.0
                request.pose.orientation.z, request.pose.orientation.w = (
                    math.sin(yaw / 2), math.cos(yaw / 2))
            futures.append(self.client.call_async(request))
        self.pending = futures
        for future in futures:
            future.add_done_callback(lambda _: self.on_pose_result(futures, action))

    def on_pose_result(self, futures, action):
        if self.pending is not futures or not all(future.done() for future in futures):
            return
        self.pending = None
        try:
            success = all(future.result().success for future in futures)
        except Exception as error:  # Service failures must not advance the appearance sequence.
            self.get_logger().warning(f'Gazebo pose service: {error}')
            success = False
        now = self.seconds()
        if success:
            if action == 'move':
                return  # Motion must not restart visibility time or advance the person index.
            self.sequence.acknowledge(action, now)
            self.get_logger().info(
                f'{self.sequence.phase}: appearance {self.sequence.appearances}, '
                f'next index {self.sequence.index}')
        else:
            self.retry_at = now + 1
            self.get_logger().warning('Gazebo did not move the person; retrying in one second')


def main():
    rclpy.init()
    node = FaceAppearances()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
