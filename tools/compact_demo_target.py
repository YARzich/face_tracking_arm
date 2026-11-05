#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Move one world-frame target briefly, then hold it through compact return."""

import time

from compact_demo_logic import (
    ARC_DURATION_SEC,
    freeze_target_on_start,
    generation_completed,
    POST_RETURN_ARC_SEC,
    post_return_target,
    STABLE_HOLD_SEC,
    STANDARD_FACE_DISTANCE_M,
    target_for_scenario,
    target_from_monitor_transform,
)
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.subscription import RCLError
from rclpy.time import Time
from rosgraph_msgs.msg import Clock as ClockMessage
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


STARTUP_TIMEOUT_SEC = 45.0
TRIGGER_TIMEOUT_SIM_SEC = 15.0
REPORT_INTERVAL_SEC = 5.0
JOINT_STATE_MAX_AGE_SEC = 0.2


class CompactDemoTarget(Node):
    def __init__(self):
        super().__init__('compact_demo_target')
        self.declare_parameter('scenario', 'stationary')
        self.declare_parameter('face_distance_m', STANDARD_FACE_DISTANCE_M)
        self.scenario = str(self.get_parameter('scenario').value)
        self.face_distance_m = float(self.get_parameter('face_distance_m').value)
        if self.scenario not in {'stationary', 'arc', 'after_return'}:
            raise ValueError(f'unknown compact demo scenario: {self.scenario}')
        if self.face_distance_m <= 0.0:
            raise ValueError('face_distance_m must be positive')
        self.target_pub = self.create_publisher(PointStamped, '/face/center', 10)
        self.create_subscription(
            DiagnosticArray, '/tracking/diagnostics', self._on_tracking, 10)
        self.create_subscription(
            DiagnosticArray, '/servo_node/diagnostics', self._on_servo, 10)
        self.create_subscription(ClockMessage, '/clock', self._on_clock, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ready = False
        self.last_clock_ns = 0
        self.joint_stamp_ns = 0
        self.joints_present = False
        self.compact_phase = 'INACTIVE'
        self.compact_generation = 0
        self.started_generation = 0
        self.frozen_target = None
        self.completed = False
        self.complete_sim_ns = None
        self.after_return_motion_started_ns = None
        self.stable_hold_reported = False
        self.after_return_motion_reported = False
        self.completion_reported = False
        self.startup_wall = time.monotonic()
        self.motion_start_sim_ns = None
        self.initial_target = None
        self.initial_target_center = None
        self.last_published_target = None
        self.initial_target_sent = False
        self.last_report_wall = 0.0
        self.last_wait_reason = None
        self.timer = self.create_timer(
            0.02, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _on_tracking(self, message):
        for status in message.status:
            if status.name == 'face_tracking_arm/tracking':
                values = {item.key: item.value for item in status.values}
                self.ready = values.get('ready') == 'true'

    def _on_servo(self, message):
        for status in message.status:
            if status.name != 'face_tracking_arm/collision_aware_servo':
                continue
            values = {item.key: item.value for item in status.values}
            phase = values.get('compact_phase', self.compact_phase)
            try:
                generation = int(values.get('compact_generation', '0'))
            except ValueError:
                generation = 0
            self.compact_phase = phase
            self.compact_generation = generation
            frozen, started_generation = freeze_target_on_start(
                phase, generation, self.last_published_target,
                self.frozen_target, self.started_generation)
            if self.frozen_target is None and frozen is not None:
                self.frozen_target = frozen
                self.started_generation = started_generation
                self.get_logger().info(
                    f'Компактный манёвр начался: generation={generation}, '
                    f'phase={phase}; фиксирую F={self.frozen_target}')
            if generation_completed(phase, generation, self.started_generation):
                self.completed = True
                if self.complete_sim_ns is None:
                    self.complete_sim_ns = self.last_clock_ns
                if not self.completion_reported:
                    self.completion_reported = True
                    self.get_logger().info(
                        'Компактный манёвр завершён; продолжаю публиковать ту же F')

    def _on_clock(self, message):
        stamp = message.clock
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns > 0:
            self.last_clock_ns = stamp_ns

    def _on_joint_state(self, message):
        required = {f'joint{i}' for i in range(1, 7)}
        self.joints_present = required.issubset(message.name)
        stamp = message.header.stamp
        self.joint_stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

    def _target_from_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', 'monitor_control_frame', Time(), timeout=Duration(seconds=0.01))
            pose = transform.transform
            self.initial_target_center = (
                pose.translation.x, pose.translation.y, pose.translation.z)
            return target_from_monitor_transform(
                (pose.translation.x, pose.translation.y, pose.translation.z),
                (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w),
                self.face_distance_m)
        except Exception:
            return None

    def _wait_reason(self, current_sim_ns):
        if current_sim_ns <= 0:
            return 'жду /clock'
        if not self.ready:
            return 'жду готовность tracking'
        if not self.joints_present or self.joint_stamp_ns <= 0:
            return 'жду /joint_states'
        joint_age_ns = current_sim_ns - self.joint_stamp_ns
        if joint_age_ns < 0 or joint_age_ns > JOINT_STATE_MAX_AGE_SEC * 1_000_000_000:
            return 'жду свежий /joint_states (не старше 0,2 с)'
        if self.initial_target is None:
            return 'жду TF world → monitor_control_frame'
        return None

    def _publish(self, target):
        x, y, z = target
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'world'
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)
        self.target_pub.publish(point)
        self.last_published_target = (float(x), float(y), float(z))

    def _tick(self):
        now_wall = time.monotonic()
        current_sim_ns = self.last_clock_ns

        if current_sim_ns > 0 and self.initial_target is None:
            self.initial_target = self._target_from_tf()
            if self.initial_target is not None:
                self._publish(self.initial_target)
                self.initial_target_sent = True
                self.get_logger().info(
                    f'Подана F по TF monitor_control_frame +X на '
                    f'{self.face_distance_m:.3f} м: {self.initial_target}')

        if not self.initial_target_sent:
            reason = ('жду положительное время симуляции /clock'
                      if current_sim_ns <= 0 else
                      'жду TF world → monitor_control_frame')
            self._report_wait(reason, now_wall)
            if now_wall - self.startup_wall > STARTUP_TIMEOUT_SEC:
                self._startup_failed(reason)
            return

        if self.motion_start_sim_ns is None:
            reason = self._wait_reason(current_sim_ns)
            if reason is not None:
                self._report_wait(reason, now_wall)
                if now_wall - self.startup_wall > STARTUP_TIMEOUT_SEC:
                    self._startup_failed(reason)
                return
            self.motion_start_sim_ns = current_sim_ns
            self.last_report_wall = now_wall
            self.get_logger().info(
                f'Симуляция готова; сценарий цели: {self.scenario}')

        elapsed_sim_sec = max(0.0, (current_sim_ns - self.motion_start_sim_ns) / 1e9)
        if self.frozen_target is None:
            target = target_for_scenario(
                self.scenario, elapsed_sim_sec, self.initial_target,
                self.initial_target_center)
            self._publish(target)
            if not self.started_generation and elapsed_sim_sec > TRIGGER_TIMEOUT_SIM_SEC:
                self.get_logger().error(
                    f'Компактный манёвр не начался за {TRIGGER_TIMEOUT_SIM_SEC:.0f} '
                    f'симуляционных секунд (phase={self.compact_phase}, '
                    f'generation={self.compact_generation})')
                rclpy.shutdown()
                return
            if (not self.started_generation and elapsed_sim_sec > ARC_DURATION_SEC and
                    now_wall - self.last_report_wall >= REPORT_INTERVAL_SEC):
                self.get_logger().info(
                    f'F удерживается; жду естественный trigger '
                    f'(phase={self.compact_phase})')
                self.last_report_wall = now_wall
        else:
            if self.completed and self.complete_sim_ns is not None:
                completed_elapsed = (current_sim_ns - self.complete_sim_ns) / 1e9
                if completed_elapsed >= STABLE_HOLD_SEC and not self.stable_hold_reported:
                    self.stable_hold_reported = True
                    self.get_logger().info(
                        f'Удержание после COMPLETE завершено: {STABLE_HOLD_SEC:.0f} с')
                if (self.scenario == 'after_return' and
                        completed_elapsed >= STABLE_HOLD_SEC):
                    if self.after_return_motion_started_ns is None:
                        self.after_return_motion_started_ns = current_sim_ns
                        self.get_logger().info(
                            '10 с устойчивого слежения завершены; начинаю короткий '
                            'прогон цели влево/вправо')
                    post_elapsed = (
                        current_sim_ns - self.after_return_motion_started_ns) / 1e9
                    if post_elapsed < POST_RETURN_ARC_SEC:
                        self._publish(post_return_target(
                            post_elapsed, self.frozen_target, self.initial_target_center))
                        return
                    if not self.after_return_motion_reported:
                        self.after_return_motion_reported = True
                        self.get_logger().info(
                            'Короткие дуги завершены; продолжаю удерживать цель')
            # Keep the exact last point from the first actual compact PATH phase,
            # including after COMPLETE, so the face-return target remains live.
            self._publish(self.frozen_target)

    def _report_wait(self, reason, now_wall):
        if (reason != self.last_wait_reason or
                now_wall - self.last_report_wall >= REPORT_INTERVAL_SEC):
            self.get_logger().info(f'Ожидание: {reason}')
            self.last_wait_reason = reason
            self.last_report_wall = now_wall

    def _startup_failed(self, reason):
        self.get_logger().error(
            f'Не удалось подготовить демо за {STARTUP_TIMEOUT_SEC:.0f} с: {reason}')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = CompactDemoTarget()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
