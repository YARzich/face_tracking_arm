#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Publish/observe FACE targets and capture source-stamped POSTURE-1D data."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

from diagnostic_msgs.msg import DiagnosticArray
from face_tracking_arm.msg import (
    ServoExecutionContext,
    ServoTrajectoryPublication,
    TrackingTarget,
)
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import ServoStatus
from posture1d_metrics import nearest_by_stamp
from posture1d_sequence import load_sequence_definition, target_at
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
import tf2_geometry_msgs  # noqa: F401
from tf2_ros import Buffer, TransformListener

FRAMES = ('monitor_control_frame', 'link3', 'link4')
MAX_PENDING_ATTEMPTS_PER_TICK = 8
TF_LOOKUP_TIMEOUT_SEC = 0.0
SEQUENCE_ARM_FRESHNESS_SEC = 0.05
SEQUENCE_FACE_MATCH_TOLERANCE_M = 1.0e-5


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def xyz(point) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def optional_finite_float(value, available: bool) -> float | None:
    """Represent unavailable or non-finite ROS measurements as JSON null."""
    number = float(value)
    return number if available and math.isfinite(number) else None


def qxyz(message: JointState) -> dict | None:
    try:
        indices = [message.name.index(f'joint{i}') for i in range(1, 7)]
        q = [float(message.position[i]) for i in indices]
        v = [float(message.velocity[i]) for i in indices]
        if not all(math.isfinite(x) for x in q + v):
            return None
        return {'stamp_sec': stamp_seconds(message.header.stamp), 'q_rad': q,
                'v_radps': v}
    except (ValueError, IndexError):
        return None


def context_record(message: ServoExecutionContext) -> dict:
    return {
        'stamp_sec': stamp_seconds(message.header.stamp),
        'tick_sequence': int(message.tick_sequence),
        'controller_mode': int(message.controller_mode),
        'control_time_wait': bool(getattr(message, 'control_time_wait', False)),
        'face_tracking_published': bool(message.face_tracking_published),
        'target_available': bool(message.target_available),
        'target_used': bool(message.target_used_for_reference),
        'target_id': int(message.target_id),
        'target_mode': int(message.target_mode),
        'target_source_stamp_sec': stamp_seconds(message.target_source_stamp),
        'face_stamp_sec': stamp_seconds(message.face_stamp),
        'target_frame': message.target_frame,
        'target_position': xyz(message.target_position),
        'face_position': xyz(message.face_position),
        'tf_provenance_available': bool(message.tf_provenance_available),
        'active_collision_rows': int(message.active_collision_rows),
        'cycle_duration_ns': int(message.cycle_duration_ns),
        'dropped_context_records': int(message.dropped_context_records),
        'dropped_publication_events': int(
            message.dropped_publication_events),
    }


def sequence_face_arming_ready(context, expected_face, now_sec):
    """Require a fresh, consumed FACE command for the frozen sequence origin."""
    if not context:
        return False
    try:
        context_stamp = float(context['stamp_sec'])
        source_stamp = float(context['target_source_stamp_sec'])
        face = [float(value) for value in context['face_position']]
        expected = [float(value) for value in expected_face]
    except (KeyError, TypeError, ValueError):
        return False
    if (len(face) != 3 or len(expected) != 3 or
            not all(math.isfinite(value) for value in face + expected)):
        return False
    return bool(
        context.get('controller_mode') == ServoExecutionContext.CONTROLLER_TRACKING and
        context.get('target_mode') == TrackingTarget.FACE and
        context.get('target_available') and context.get('target_used') and
        context.get('face_tracking_published') and source_stamp > 0.0 and
        abs(now_sec - context_stamp) <= SEQUENCE_ARM_FRESHNESS_SEC and
        0.0 <= now_sec - source_stamp <= SEQUENCE_ARM_FRESHNESS_SEC and
        math.dist(face, expected) <= SEQUENCE_FACE_MATCH_TOLERANCE_M)


def execution_context_source_record(message: ServoExecutionContext) -> dict:
    """Serialize one complete raw execution-context record for provenance."""
    record = context_record(message)
    record.update({
        'kind': 'execution_context',
        'frame_id': message.header.frame_id,
        'collision_distance_available': bool(
            message.collision_distance_available),
        'minimum_world_distance_m': optional_finite_float(
            message.minimum_world_distance_m, message.collision_distance_available),
        'closest_pair': message.closest_pair,
        'active_world_pair_available': bool(
            message.active_world_pair_available),
        'active_world_pair_distance_m': optional_finite_float(
            message.active_world_pair_distance_m,
            message.active_world_pair_available),
        'active_world_pair': message.active_world_pair,
        'hardware_epoch': int(message.hardware_epoch),
        'hardware_read_sequence': int(message.hardware_read_sequence),
        'active_publication_id': int(message.active_publication_id),
        'latest_publication_id': int(message.latest_publication_id),
        'braking_tail_publication_id': int(
            message.braking_tail_publication_id),
        'braking_tail_stationary_stamp_sec': stamp_seconds(
            message.braking_tail_stationary_stamp),
        'target_transform': {
            'header_stamp_sec': stamp_seconds(
                message.target_transform.header.stamp),
            'frame_id': message.target_transform.header.frame_id,
            'child_frame_id': message.target_transform.child_frame_id,
            'translation': xyz(
                message.target_transform.transform.translation),
            'rotation_xyzw': [
                float(message.target_transform.transform.rotation.x),
                float(message.target_transform.transform.rotation.y),
                float(message.target_transform.transform.rotation.z),
                float(message.target_transform.transform.rotation.w),
            ],
        },
    })
    return record


def write_execution_context_source(
        stream, message: ServoExecutionContext) -> dict:
    """Append a raw source record without forcing a per-callback flush."""
    record = execution_context_source_record(message)
    stream.write(
        json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
    return record


def trajectory_record(message, start_stamp: float, duration: float) -> dict | None:
    """Serialize one source-stamped publication when it falls in capture."""
    stamp = stamp_seconds(message.header.stamp)
    elapsed = stamp - start_stamp
    if not 0.0 <= elapsed <= duration:
        return None
    trajectory = message.trajectory
    return {
        'event_stamp_sec': stamp,
        'trajectory_stamp_sec': stamp_seconds(trajectory.header.stamp),
        'event_frame_id': message.header.frame_id,
        'trajectory_frame_id': trajectory.header.frame_id,
        'elapsed_sec': elapsed,
        'publication_kind': int(message.publication_kind),
        'publication_id': int(message.publication_id),
        'stationary_time_sec': stamp_seconds(message.stationary_time),
        'tail_start_index': int(message.tail_start_index),
        'hardware_epoch': int(message.hardware_epoch),
        'hardware_read_sequence': int(message.hardware_read_sequence),
        'control_tick_sequence': int(message.control_tick_sequence),
        'dropped_publication_events': int(message.dropped_publication_events),
        'joint_names': list(trajectory.joint_names),
        'points': [{
            'positions': [float(value) for value in point.positions],
            'velocities': [float(value) for value in point.velocities],
            'accelerations': [float(value) for value in point.accelerations],
            'effort': [float(value) for value in point.effort],
            'time_from_start_sec': stamp_seconds(point.time_from_start),
        } for point in trajectory.points],
    }


class Observer(Node):
    def __init__(self, args):
        super().__init__('posture1d_runtime_observer')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.args = args
        self.output = Path(args.output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / 'samples.jsonl').open('x', encoding='utf-8')
        self.target = args.target
        self.sequence = None
        self.sequence_origin = None
        self.start_sim = None
        self.started_wall = time.monotonic()
        self.end_sim = None
        self.preflight = {'clock': False, 'tracking_ready': False,
                          'joint_state': False, 'context': False, 'tf': False,
                          'servo_diagnostics': False, 'servo_status': False}
        self.joint = None
        self.last_sampled_joint_stamp = None
        self.start_joint_stamp = None
        self.pending_samples = deque()
        self.raw_joint_source = []
        self.trajectory_stream = (self.output / 'trajectory_publications.jsonl').open(
            'x', encoding='utf-8')
        self.context_source_stream = (self.output / 'execution_context_source.jsonl').open(
            'x', encoding='utf-8')
        self.capture_done = False
        self.source_complete = False
        self.callbacks = MutuallyExclusiveCallbackGroup()
        self.timer_callbacks = MutuallyExclusiveCallbackGroup()
        self.execution_context = None
        self.context_history = deque(maxlen=512)
        self.tracking = {}
        self.servo_diagnostics = None
        self.servo_status = None
        self.face_input = None
        self.face_history = deque(maxlen=256)
        self.tf = Buffer(cache_time=Duration(seconds=120), node=self)
        self.listener = TransformListener(self.tf, self, spin_thread=False)
        self.publisher = self.create_publisher(PointStamped, '/face/center', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 256,
                                 callback_group=self.callbacks)
        self.create_subscription(ServoExecutionContext,
                                 '/servo_node/execution_context', self.on_context, 512,
                                 callback_group=self.callbacks)
        self.create_subscription(ServoTrajectoryPublication,
                                 '/servo_node/trajectory_publications',
                                 self.on_trajectory_publication, 64,
                                 callback_group=self.callbacks)
        self.create_subscription(PointStamped, '/face/center', self.on_face,
                                 callback_group=self.callbacks,
                                 qos_profile=QoSProfile(
                                     depth=64,
                                     reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(DiagnosticArray, '/tracking/diagnostics',
                                 self.on_tracking, 10,
                                 callback_group=self.callbacks)
        self.create_subscription(DiagnosticArray, '/servo_node/diagnostics',
                                 self.on_servo_diagnostics, 10,
                                 callback_group=self.callbacks)
        self.create_subscription(ServoStatus, '/servo_node/status',
                                 self.on_servo_status, 10,
                                 callback_group=self.callbacks)
        self.timer = self.create_timer(
            0.02, self.tick, callback_group=self.timer_callbacks)
        self.write({'kind': 'manifest', 'scenario': args.scenario,
                    'mode': args.mode, 'fixed_target_world_m': self.target,
                    'recording_hz': 50, 'frames': list(FRAMES),
                    'joint_tf_time_basis': 'JointState.header.stamp'})

    def write(self, value):
        self.stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        self.stream.flush()

    def on_joint(self, message):
        joint = qxyz(message)
        if joint is None:
            return
        self.joint = joint
        if self.start_sim is None or self.capture_done:
            return
        stamp = joint['stamp_sec']
        elapsed = stamp - self.start_joint_stamp
        if elapsed > self.args.duration:
            self.source_complete = True
            return
        self.raw_joint_source.append(joint.copy())
        if elapsed >= self.args.duration:
            self.source_complete = True
        if (self.last_sampled_joint_stamp is None or
                stamp - self.last_sampled_joint_stamp >= 0.02 - 1e-9):
            if 0.0 <= elapsed <= self.args.duration:
                self.pending_samples.append({
                    'joint': joint.copy(), 'elapsed_sec': elapsed,
                    'queued_wall': time.monotonic(),
                })
            self.last_sampled_joint_stamp = stamp

    def on_context(self, message):
        self.execution_context = write_execution_context_source(
            self.context_source_stream, message)
        self.context_history.append(self.execution_context)

    def on_trajectory_publication(self, message):
        """Retain immutable published commands with their queue provenance."""
        if self.start_joint_stamp is None or self.capture_done:
            return
        record = trajectory_record(message, self.start_joint_stamp,
                                   self.args.duration)
        if record is None:
            return
        self.trajectory_stream.write(
            json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
        self.trajectory_stream.flush()

    def on_face(self, message):
        record = {'stamp_sec': stamp_seconds(message.header.stamp),
                  'frame_id': message.header.frame_id, 'position': xyz(message.point),
                  'world_position': None, 'transform_error': None}
        if message.header.frame_id == 'world':
            record['world_position'] = record['position']
        else:
            try:
                transformed = self.tf.transform(message, 'world',
                                                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC))
                record['world_position'] = xyz(transformed.point)
            except Exception as error:  # TF may legitimately lag a source-stamped observation.
                record['transform_error'] = type(error).__name__
        self.face_input = record
        self.face_history.append(record)

    def on_tracking(self, message):
        for status in message.status:
            if status.name == 'face_tracking_arm/tracking':
                values = {item.key: item.value for item in status.values}
                self.tracking = {'stamp_sec': stamp_seconds(message.header.stamp),
                                 'message': status.message, 'values': values}
                self.preflight['tracking_ready'] = values.get('ready') == 'true'

    def on_servo_diagnostics(self, message):
        self.servo_diagnostics = {
            'stamp_sec': stamp_seconds(message.header.stamp),
            'statuses': {
                status.name: {
                    'level': int(status.level[0]) if status.level else 0,
                    'message': status.message,
                    'values': {item.key: item.value for item in status.values},
                }
                for status in message.status
            },
        }

    def on_servo_status(self, message):
        self.servo_status = {
            'arrival_stamp_sec': self.get_clock().now().nanoseconds * 1e-9,
            'code': int(message.code), 'message': message.message,
        }

    def transforms_at_joint_stamp(self, joint=None):
        joint = joint or self.joint
        if not joint:
            return None, {'joint_state': 'unavailable'}
        stamp = Time(seconds=int(joint['stamp_sec']),
                     nanoseconds=int((joint['stamp_sec'] % 1) * 1e9))
        found, errors = {}, {}
        for frame in FRAMES:
            try:
                transform = self.tf.lookup_transform(
                    'world', frame, stamp,
                    timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC))
                t = transform.transform.translation
                found[frame] = [float(t.x), float(t.y), float(t.z)]
            except Exception as error:
                errors[frame] = type(error).__name__
        return found, errors

    def derive_inband_target(self, monitor_transform):
        # monitor_control_frame screen normal is its local +X axis.
        rotation = monitor_transform.transform.rotation
        x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
        vx = 1.0 - 2.0 * (y*y + z*z)
        vy = 2.0 * (x*y + z*w)
        vz = 2.0 * (x*z - y*w)
        t = monitor_transform.transform.translation
        return [t.x + .4 * vx, t.y + .4 * vy, t.z + .4 * vz]

    def preflight_transforms(self):
        if not self.joint:
            return None
        stamp = Time(seconds=int(self.joint['stamp_sec']),
                     nanoseconds=int((self.joint['stamp_sec'] % 1) * 1e9))
        try:
            return self.tf.lookup_transform('world', 'monitor_control_frame', stamp,
                                            timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC))
        except Exception:
            return None

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.preflight['clock'] = now > 0
        if now <= 0:
            return
        joint_age = (now - self.joint['stamp_sec']) if self.joint else math.inf
        if self.start_sim is None:
            monitor = self.preflight_transforms()
            _, tf_errors = self.transforms_at_joint_stamp()
            self.preflight['tf'] = monitor is not None and not tf_errors
        else:
            monitor = None
            tf_errors = {}
        self.preflight['joint_state'] = self.joint is not None and joint_age <= 0.05
        context_age = (abs(now - self.execution_context['stamp_sec'])
                       if self.execution_context else math.inf)
        self.preflight['context'] = (
            self.execution_context is not None and context_age <= 0.05)
        self.preflight['servo_diagnostics'] = (
            self.servo_diagnostics is not None and
            abs(now - self.servo_diagnostics['stamp_sec']) <= 0.1)
        self.preflight['servo_status'] = (
            self.servo_status is not None and
            abs(now - self.servo_status['arrival_stamp_sec']) <= 0.1)
        if not self.start_sim and all(self.preflight.values()):
            if self.args.mode in ('inband', 'sequence') and self.target is None:
                self.target = self.derive_inband_target(monitor)
            if self.args.mode == 'sequence':
                if self.sequence is None:
                    self.sequence_origin = self.target.copy()
                    self.sequence = self.load_sequence(self.sequence_origin)
                self.target = self.sequence_target(0.0)
                if not sequence_face_arming_ready(
                        self.execution_context, self.target, now):
                    target = PointStamped()
                    target.header.stamp = self.get_clock().now().to_msg()
                    target.header.frame_id = 'world'
                    target.point.x, target.point.y, target.point.z = self.target
                    self.publisher.publish(target)
                else:
                    self.start_sim = self.execution_context['stamp_sec']
                    self.start_joint_stamp = self.execution_context['stamp_sec']
            else:
                self.start_sim = now
                self.start_joint_stamp = self.joint['stamp_sec']
            if self.start_sim is not None:
                self.write({'kind': 'preflight', 'stamp_sec': self.start_sim,
                            'status': 'PASS', 'checks': self.preflight.copy(),
                            'target_world_m': self.target})
        if self.start_sim is None:
            if time.monotonic() - self.started_wall > self.args.preflight_timeout:
                self.write({'kind': 'preflight', 'status': 'FAIL',
                            'checks': self.preflight.copy()})
                rclpy.shutdown()
            return
        source_elapsed = (self.joint['stamp_sec'] - self.start_joint_stamp
                          if self.joint and self.start_joint_stamp is not None else -math.inf)
        if (self.args.mode in ('fixed', 'inband', 'sequence') and
                not self.source_complete and source_elapsed < self.args.duration):
            if self.args.mode == 'sequence':
                self.target = self.sequence_target(source_elapsed)
            target = PointStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = 'world'
            target.point.x, target.point.y, target.point.z = self.target
            self.publisher.publish(target)
        self.flush_pending(now)
        if self.source_complete:
            self.capture_done = True
            if not self.pending_samples:
                rclpy.shutdown()

    def load_sequence(self, origin):
        definition = load_sequence_definition(self.args.sequence)
        self.write({'kind': 'sequence_definition', 'definition': definition,
                    'return_target_world_m': origin,
                    'return_target_derived_once_at_preflight': True})
        return definition

    def sequence_target(self, elapsed):
        return target_at(self.sequence, self.sequence_origin, elapsed)

    def flush_pending(self, now):
        attempts = min(len(self.pending_samples), MAX_PENDING_ATTEMPTS_PER_TICK)
        for _ in range(attempts):
            pending = self.pending_samples.popleft()
            transforms, errors = self.transforms_at_joint_stamp(pending['joint'])
            queue_delay = time.monotonic() - pending['queued_wall']
            if errors and queue_delay < 0.25:
                self.pending_samples.append(pending)
                continue
            joint = pending['joint']
            context = nearest_by_stamp(self.context_history, joint['stamp_sec'])
            face_reference_stamp = (
                context['face_stamp_sec'] if context and
                context['face_stamp_sec'] > 0.0 else joint['stamp_sec'])
            face = nearest_by_stamp(self.face_history, face_reference_stamp)
            monitor_rotation = None
            try:
                monitor_tf = self.tf.lookup_transform(
                    'world', 'monitor_control_frame',
                    Time(seconds=int(joint['stamp_sec']),
                         nanoseconds=int((joint['stamp_sec'] % 1) * 1e9)),
                    timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC))
                rotation = monitor_tf.transform.rotation
                monitor_rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
            except Exception:  # A missing orientation makes the gaze metric unavailable.
                pass
            self.write({
                'kind': 'sample', 'sim_sec': now,
                'elapsed_sec': pending['elapsed_sec'], 'joint': joint,
                'context': context, 'input_face': face,
                'transforms_world_at_joint_stamp': transforms,
                'monitor_rotation_world_at_joint_stamp_xyzw': monitor_rotation,
                'transform_errors': errors, 'tracking': self.tracking,
                'servo_diagnostics': self.servo_diagnostics,
                'servo_status': self.servo_status,
                'target_world_m': self.target,
                'face_joint_stamp_skew_sec': (
                    abs(face['stamp_sec'] - joint['stamp_sec']) if face else None),
                'context_joint_stamp_skew_sec': (
                    abs(context['stamp_sec'] - joint['stamp_sec']) if context else None),
                'context_association_within_50ms': bool(
                    context and abs(context['stamp_sec'] - joint['stamp_sec']) <= 0.05),
                'context_age_at_sample_sec': (
                    abs(now - context['stamp_sec']) if context else None),
                'joint_age_sec': max(0.0, now - joint['stamp_sec']),
                'sample_queue_delay_wall_sec': queue_delay,
                'pending_queue_depth_at_write': len(self.pending_samples),
                'association': 'nearest_source_stamp_at_flush',
            })

    def close(self):
        if self.raw_joint_source:
            with (self.output / 'joint_source.jsonl').open('x', encoding='utf-8') as stream:
                for sample in self.raw_joint_source:
                    stream.write(json.dumps(sample, separators=(',', ':')) + '\n')
        end_joint = self.joint['stamp_sec'] if self.joint else None
        self.write({'kind': 'capture_end', 'sim_sec': self.get_clock().now().nanoseconds * 1e-9,
                    'elapsed_sec': (end_joint - self.start_joint_stamp)
                    if self.start_joint_stamp is not None and end_joint is not None else None,
                    'last_joint_stamp_sec': end_joint,
                    'source_complete': self.source_complete})
        self.stream.close()
        self.trajectory_stream.close()
        self.context_source_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--mode', choices=('fixed', 'inband', 'sequence', 'observe'),
                        required=True)
    parser.add_argument('--target', nargs=3, type=float)
    parser.add_argument('--duration', type=float, required=True,
                        help='simulated seconds after the preflight gate')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--preflight-timeout', type=float, default=30.0)
    args = parser.parse_args()
    if args.mode == 'fixed' and args.target is None:
        parser.error('--target is required for fixed mode')
    if args.mode in ('inband', 'sequence'):
        args.target = None
    if args.mode == 'observe':
        args.target = None
    if args.mode == 'sequence' and not args.sequence:
        parser.error('--sequence is required for sequence mode')
    rclpy.init()
    node = Observer(args)
    result = 0
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
        if node.start_sim is None:
            result = 2
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        result = 130
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
