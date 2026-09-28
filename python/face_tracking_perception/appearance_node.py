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
from .arm_scene import person_entity_name
from .close_stress_motion import CloseStressMotion
from .depth_messages import status_message
from .fast_stress_motion import FastStressMotion
from .stress_motion import StressMotion


class FaceAppearances(Node):
    """Apply appearance or stress poses; observe the arm without commanding it."""

    def __init__(self):
        super().__init__('face_appearances')
        self.declare_parameters('', [
            ('visible_sec', 12.0), ('rest_sec', 8.0), ('cycles', 0),
            ('motion_lateral_m', 0.14), ('motion_depth_m', 0.07),
            ('motion_yaw_rad', 0.07), ('motion_period_sec', 6.0),
            ('idle_behavior', 'rest'), ('people_count', 1), ('scenario', 'appearances'),
            ('positions_xy', [2.35, 0.0, 2.25, -0.35, 2.60, 0.40]),
            ('rest_joints', [0.0] * 6)])
        value = lambda name: self.get_parameter(name).value  # noqa: E731
        self.positions = np.asarray(value('positions_xy')).reshape(-1, 2)
        self.rest_joints = np.asarray(value('rest_joints'))
        if (not np.isfinite(self.positions).all()
                or (np.linalg.norm(self.positions, axis=1) <= .8).any()
                or self.rest_joints.shape != (6,) or not np.isfinite(self.rest_joints).all()):
            raise ValueError('Invalid person positions or six-joint rest configuration')
        self.searching = value('idle_behavior') != 'rest'
        self.people_count = value('people_count')
        self.scenario = value('scenario')
        counts = {'appearances': (1, 2), 'stress': (1,), 'stress_fast': (3,), 'stress_close': (1,)}
        if self.scenario not in counts or self.people_count not in counts[self.scenario]:
            raise ValueError(
                'Use appearances with 1–2 people, stress/stress_close with 1, '
                'or stress_fast with 3')
        self.cycles = value('cycles')
        profiles = {'stress': StressMotion, 'stress_fast': FastStressMotion,
                    'stress_close': CloseStressMotion}
        self.stress = profiles.get(self.scenario, StressMotion)()
        self.stress_started_at = None
        self.stress_last_time = None
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
        self.pose_success_at = None
        self.pose_failures = 0
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
        if self.scenario in ('stress', 'stress_fast', 'stress_close'):
            self.tick_stress(now)
            return
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
        poses = []
        for index in range(self.people_count):
            pose = (0.0, 0.0, -10.0, 0.0)
            if action in ('show', 'move'):
                elapsed = now - self.sequence.shown_at if action == 'move' else 0.0
                center = self.positions[self.sequence.index]
                if index:
                    center = companion_center(center, elapsed)
                x, y, yaw = self.motion.pose(center, elapsed)
                pose = (x, y, 0.0, yaw)
            poses.append(pose)
        self.request_poses(poses, action)

    def tick_stress(self, now):
        # The person is already in the world before control starts. Run the
        # schedule from the first feedback, without waiting for REST or FACE.
        if self.stress_started_at is None and self.joint_stamp is not None:
            self.stress_started_at = self.joint_stamp
        if self.stress_last_time is not None and now < self.stress_last_time:
            self.stress_started_at = now
            self.retry_at = now
            self.pose_success_at = None
        self.stress_last_time = now
        elapsed = 0.0 if self.stress_started_at is None else now - self.stress_started_at
        sample = self.stress.sample(elapsed, cycles=self.cycles)
        if self.scenario == 'stress_fast':
            poses, visible = sample.poses, True
        else:
            poses = [sample.pose if sample.visible else (0.0, 0.0, -10.0, 0.0)]
            visible = sample.visible
        fresh = all(stamp is not None and 0 <= now - stamp < .5
                    for stamp in (self.tracking_stamp, self.joint_stamp))
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        values = {'scenario': self.scenario, 'elapsed_sec': elapsed,
                  'people_count': self.people_count,
                  'duration_sec': self.stress.duration_sec, 'visible': visible,
                  'complete': sample.complete, 'ready': self.ready and fresh,
                  'tracking_mode': self.mode, 'pose_request_pending': self.pending is not None,
                  'pose_failures': self.pose_failures,
                  'pose_ack_age_sec': (now - self.pose_success_at
                                       if self.pose_success_at is not None else -1.0)}
        self.publisher.publish(status_message(
            header, 'face_appearances', sample.phase, values, fresh))
        if (self.pending is not None or now < self.retry_at
                or not self.client.service_is_ready()):
            return
        self.request_poses(poses, 'move')

    def request_poses(self, poses, action):
        """Keep one acknowledged Gazebo request per person in flight."""
        futures = []
        for index, (x, y, z, yaw) in enumerate(poses):
            request = SetEntityPose.Request()
            name = person_entity_name(index)
            request.entity = Entity(name=name, type=Entity.MODEL)
            request.pose.position.x, request.pose.position.y, request.pose.position.z = x, y, z
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
            self.pose_success_at = now
            if action == 'move':
                return  # Motion must not restart visibility time or advance the person index.
            self.sequence.acknowledge(action, now)
            self.get_logger().info(
                f'{self.sequence.phase}: appearance {self.sequence.appearances}, '
                f'next index {self.sequence.index}')
        else:
            self.pose_failures += 1
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
