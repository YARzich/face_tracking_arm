#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Inject perception status faults and verify that search pauses in place."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

from diagnostic_msgs.msg import DiagnosticArray
from face_tracking_arm.msg import PerceptionStatus, TrackingTarget
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock as ClockMsg


PHASE_LOCAL = 'SEARCH_LOCAL'
PHASE_SWEEP = 'SEARCH_SWEEP'
MODE_NAMES = {
    TrackingTarget.FACE: 'FACE',
    TrackingTarget.HOLD: 'HOLD',
    TrackingTarget.REST: 'REST',
    TrackingTarget.POSE: 'POSE',
    TrackingTarget.SEARCH: 'SEARCH',
}
STATE_NAMES = {
    PerceptionStatus.OK: 'OK',
    PerceptionStatus.NO_FACE: 'NO_FACE',
    PerceptionStatus.DEPTH_STREAM_LOST: 'DEPTH_STREAM_LOST',
}


def uint8_value(value):
    """Handle ROS uint8 properties represented as a Python one-byte object."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        if len(value) != 1:
            raise ValueError(f'Expected one byte for uint8, got {len(value)}')
        return value[0]
    return int(value)


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_sec(stamp):
    return stamp_ns(stamp) * 1e-9


def diagnostic_values(message, name):
    for status in message.status:
        if status.name == name:
            return {item.key: item.value for item in status.values}
    return {}


class SearchHoldProbe(Node):
    """Run the bounded synthetic sequence and persist all observed evidence."""

    def __init__(self, args):
        super().__init__('search_camera_hold_probe')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.args = args
        self.output = Path(args.output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.events_file = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        self.started_wall = time.monotonic()
        self.sim_sec = None
        self.last_tick_sim = None
        self.state = PerceptionStatus.OK
        self.last_source_stamp = None
        self.face_start_sim = None
        self.loss_start_sim = None
        self.phase = None
        self.tracking_ready = False
        self.search_target_mode = None
        self.target_count = 0
        self.last_tracking_diag = {}
        self.last_servo_diag = {}
        self.last_servo_diag_stamp_sec = None
        self.last_target = None
        self.local_active_sec = 0.0
        self.sweep_active_sec = 0.0
        self.holds = []
        self.current_hold = None
        self.return_face_seen = False
        self.return_face_target = None
        self.return_face_diag = None
        self.fresh_face_restore_sim_sec = None
        self.synthetic_face_point_world = [1.05, 1.0, 1.5]
        self.criteria = {}
        self.event_tail = deque(maxlen=32)
        self.done = False
        self.failed_reason = None

        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.status_pub = self.create_publisher(
            PerceptionStatus, '/face/perception_status', status_qos)
        self.center_pub = self.create_publisher(PointStamped, '/face/center', status_qos)
        self.create_subscription(ClockMsg, '/clock', self.on_clock, 10)
        self.create_subscription(DiagnosticArray, '/tracking/diagnostics',
                                 self.on_tracking_diagnostics, 10)
        self.create_subscription(DiagnosticArray, '/servo_node/diagnostics',
                                 self.on_servo_diagnostics, 10)
        self.create_subscription(TrackingTarget, '/servo_node/tracking_target_cmds',
                                 self.on_target, 10)
        self.timer = self.create_timer(
            0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.record('probe_started', {'max_wall_sec': args.max_wall_sec})
        self.record('synthetic_face_point', {
            'frame_id': 'world', 'xyz_m': self.synthetic_face_point_world})

    def record(self, event, fields=None):
        entry = {
            'event': event,
            'wall_elapsed_sec': round(time.monotonic() - self.started_wall, 4),
            'sim_sec': self.sim_sec,
        }
        if fields:
            entry.update(fields)
        self.events_file.write(json.dumps(entry, separators=(',', ':'), allow_nan=False) + '\n')
        self.events_file.flush()
        self.event_tail.append(entry)

    def on_clock(self, message):
        self.sim_sec = stamp_sec(message.clock)

    def on_tracking_diagnostics(self, message):
        self.last_tracking_diag = diagnostic_values(message, 'face_tracking_arm/tracking')
        self.tracking_ready = self.last_tracking_diag.get('ready') == 'true'

    def on_servo_diagnostics(self, message):
        self.last_servo_diag = diagnostic_values(
            message, 'face_tracking_arm/collision_aware_servo')
        self.last_servo_diag_stamp_sec = stamp_sec(message.header.stamp)
        phase = self.last_servo_diag.get('motion_reference_state')
        if phase and phase != self.phase:
            previous = self.phase
            self.phase = phase
            self.record('phase_changed', {'from': previous, 'to': phase})
        if (self.current_hold is not None and
                self.state == PerceptionStatus.DEPTH_STREAM_LOST and phase == 'HOLD'):
            hold = self.current_hold
            hold['servo_hold_diag_samples'] = hold.get('servo_hold_diag_samples', 0) + 1
            hold.setdefault('servo_hold_first_diag_stamp_sec', self.last_servo_diag_stamp_sec)
            hold['servo_hold_last_diag_stamp_sec'] = self.last_servo_diag_stamp_sec
            hold['servo_hold_phase'] = phase
            hold['servo_hold_tracking_phase'] = self.last_servo_diag.get('tracking_phase')
        if (self.return_face_seen and self.return_face_diag is None and
                self.return_face_target is not None and
                self.last_servo_diag_stamp_sec is not None and
                self.last_servo_diag_stamp_sec > self.return_face_target['header_stamp_sec'] and
                phase == 'FACE' and self.last_servo_diag.get('mode') == 'TRACKING'):
            self.return_face_diag = {
                'stamp_sec': self.last_servo_diag_stamp_sec,
                'motion_reference_state': phase,
                'tracking_phase': self.last_servo_diag.get('tracking_phase'),
                'mode': self.last_servo_diag.get('mode'),
            }
            self.record('servo_confirmed_face_tracking', self.return_face_diag)
            self.check_success()

    def on_target(self, message):
        self.target_count += 1
        mode = uint8_value(message.mode)
        source = stamp_sec(message.face_stamp)
        self.last_target = {
            'mode': MODE_NAMES.get(mode, f'UNKNOWN_{mode}'),
            'target_id': int(message.target_id),
            'search_pattern': message.search_pattern,
            'header_stamp_sec': stamp_sec(message.header.stamp),
            'face_stamp_sec': source,
        }
        self.search_target_mode = mode
        if (self.current_hold is not None and
                self.state == PerceptionStatus.DEPTH_STREAM_LOST and
                mode == TrackingTarget.HOLD):
            self.current_hold['hold_command_seen'] = True
            self.current_hold['hold_target_count'] = self.target_count
        if self.last_target['mode'] != getattr(self, '_last_target_mode', None):
            self._last_target_mode = self.last_target['mode']
            self.record('tracking_target_mode', dict(self.last_target))
        if (self.state == PerceptionStatus.OK and mode == TrackingTarget.FACE and
                self.loss_start_sim is not None):
            self.return_face_seen = True
            self.return_face_target = dict(self.last_target)
            self.record('fresh_face_command_after_loss', {
                **self.return_face_target,
                'fresh_ok_started_sim_sec': self.fresh_face_restore_sim_sec,
            })

    def publish_source(self):
        if self.sim_sec is None or self.sim_sec <= 0.0:
            return
        now = self.get_clock().now().to_msg()
        now_ns = stamp_ns(now)
        status = PerceptionStatus()
        status.header.stamp = now
        status.header.frame_id = 'world'
        status.state = self.state
        status.reason = 'synthetic_search_hold_probe'
        status.target_id = 1
        if self.state == PerceptionStatus.DEPTH_STREAM_LOST:
            source_ns = self.last_source_stamp or now_ns
        else:
            source_ns = now_ns
            self.last_source_stamp = now_ns
        status.source_stamp.sec = source_ns // 1_000_000_000
        status.source_stamp.nanosec = source_ns % 1_000_000_000
        self.status_pub.publish(status)
        self.record('source_status', {
            'state': STATE_NAMES.get(self.state, str(self.state)),
            'header_stamp_sec': stamp_sec(now),
            'source_stamp_sec': source_ns * 1e-9,
            'target_id': status.target_id,
        })

        if self.state == PerceptionStatus.OK:
            center = PointStamped()
            center.header.stamp = now
            center.header.frame_id = 'world'
            center.point.x, center.point.y, center.point.z = self.synthetic_face_point_world
            self.center_pub.publish(center)

    def set_state(self, new_state, event, details=None):
        old_state = STATE_NAMES.get(self.state, str(self.state))
        self.state = new_state
        if new_state == PerceptionStatus.OK and event == 'face_restored_with_fresh_measurement':
            self.fresh_face_restore_sim_sec = self.sim_sec
        self.record(event, {'from': old_state,
                            'to': STATE_NAMES.get(new_state, str(new_state)),
                            **(details or {})})

    def begin_hold(self, phase):
        self.current_hold = {
            'index': len(self.holds) + 1,
            'phase_before': phase,
            'start_sim_sec': self.sim_sec,
            'plans_start': self.plan_count(),
            'hold_command_seen': False,
            'resume_same_phase': False,
        }
        self.holds.append(self.current_hold)
        self.set_state(PerceptionStatus.DEPTH_STREAM_LOST, 'hold_injected', {
            'index': self.current_hold['index'], 'phase': phase,
            'retained_source_stamp_sec': (self.last_source_stamp or 0) * 1e-9,
            'plans_start': self.current_hold['plans_start']})

    def end_hold(self):
        hold = self.current_hold
        if hold is None:
            return
        hold['end_fault_sim_sec'] = self.sim_sec
        hold['duration_sec'] = round(self.sim_sec - hold['start_sim_sec'], 4)
        hold['plans_end_fault'] = self.plan_count()
        hold['plans_unchanged_during_fault'] = (
            hold['plans_start'] is not None and
            hold['plans_start'] == hold['plans_end_fault'])
        hold['servo_hold_diag_seen'] = hold.get('servo_hold_diag_samples', 0) > 0
        hold['release_target_count'] = self.target_count
        self.set_state(PerceptionStatus.NO_FACE, 'hold_released', {
            'index': hold['index'], 'duration_sec': hold['duration_sec'],
            'plans_end_fault': hold['plans_end_fault'],
            'plans_unchanged_during_fault': hold['plans_unchanged_during_fault']})
        hold['awaiting_resume'] = True

    def plan_count(self):
        value = self.last_servo_diag.get('plans_requested')
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    def publish_final_report(self, status):
        phases_seen = {event.get('to') for event in self.event_tail
                       if event.get('event') == 'phase_changed'}
        # The event tail is only a convenience; the complete event stream is authoritative.
        report = {
            'result': status,
            'failure_reason': self.failed_reason,
            'max_wall_sec': self.args.max_wall_sec,
            'wall_elapsed_sec': round(time.monotonic() - self.started_wall, 3),
            'synthetic_face_point_world_m': self.synthetic_face_point_world,
            'criteria': self.criteria,
            'holds': self.holds,
            'last_target': self.last_target,
            'return_face_target': self.return_face_target,
            'return_face_servo_diagnostic': self.return_face_diag,
            'last_tracking_diagnostics': self.last_tracking_diag,
            'last_servo_diagnostics': self.last_servo_diag,
            'event_tail': list(self.event_tail),
            'notes': ['Full phase and source-stamp timeline is events.jsonl.'],
        }
        with (self.output / 'report.json').open('x', encoding='utf-8') as report_file:
            report_file.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        self.record('probe_finished', {
            'result': status, 'phases_seen_in_tail': sorted(phases_seen)})
        self.events_file.close()

    def finish(self, status, reason=None):
        if self.done:
            return
        self.done = True
        self.failed_reason = reason
        self.publish_final_report(status)
        self.get_logger().info(f'{status}: report written to {self.output / "report.json"}')

    def tick(self):
        wall_elapsed = time.monotonic() - self.started_wall
        if wall_elapsed >= self.args.max_wall_sec:
            self.criteria = {
                'local_phase_observed': ('PASS' if self.local_active_sec > 0.0 else
                                         'NOT_ASSESSED'),
                'sweep_phase_observed': ('PASS' if self.sweep_active_sec > 0.0 else
                                         'NOT_ASSESSED'),
                'three_holds_and_resumes': ('PASS' if len(self.holds) == 3 and all(
                    hold.get('hold_command_seen') and hold.get('servo_hold_diag_seen') and
                    hold.get('resume_same_phase')
                    for hold in self.holds) else 'NOT_ASSESSED'),
                'local_active_time_sec': round(self.local_active_sec, 3),
                'max_wall_timeout': 'FAIL',
            }
            self.finish('FAIL', 'max_wall_timeout')
            rclpy.shutdown()
            return
        if self.sim_sec is None or self.sim_sec <= 0.0:
            return

        if self.last_tick_sim is not None and self.sim_sec >= self.last_tick_sim:
            delta = min(0.25, self.sim_sec - self.last_tick_sim)
            if self.phase == PHASE_LOCAL and self.state != PerceptionStatus.DEPTH_STREAM_LOST:
                self.local_active_sec += delta
            if self.phase == PHASE_SWEEP and self.state != PerceptionStatus.DEPTH_STREAM_LOST:
                self.sweep_active_sec += delta
        self.last_tick_sim = self.sim_sec

        self.publish_source()
        if self.state == PerceptionStatus.OK and self.face_start_sim is None:
            if self.tracking_ready and self.last_target and self.last_target['mode'] == 'FACE':
                self.face_start_sim = self.sim_sec
                self.record('initial_face_confirmed', {'target_id': self.last_target['target_id']})
        if self.loss_start_sim is None:
            if (self.face_start_sim is not None and self.sim_sec - self.face_start_sim >= 5.0):
                self.loss_start_sim = self.sim_sec
                self.set_state(PerceptionStatus.NO_FACE, 'initial_face_interval_complete')
        else:
            self.drive_holds()
        self.check_success()

    def drive_holds(self):
        if self.current_hold is not None:
            hold = self.current_hold
            if self.state == PerceptionStatus.DEPTH_STREAM_LOST:
                if self.last_target and self.last_target['mode'] == 'HOLD':
                    hold['hold_command_seen'] = True
                hold['plans_last_fault'] = self.plan_count()
                if self.sim_sec - hold['start_sim_sec'] >= 0.5:
                    self.end_hold()
                return
            if hold.get('awaiting_resume'):
                resumed = (self.phase == hold['phase_before'] and
                           hold['hold_command_seen'] and
                           self.target_count > hold.get(
                               'release_target_count', self.target_count) and
                           self.last_target and self.last_target['mode'] in ('SEARCH', 'REST'))
                if resumed:
                    hold['resume_same_phase'] = True
                    hold['resume_sim_sec'] = self.sim_sec
                    self.record('hold_resumed_same_phase', {
                        'index': hold['index'], 'phase': self.phase,
                        'target_mode': self.last_target['mode']})
                    self.current_hold = None
                return

        if self.phase == PHASE_LOCAL and len(self.holds) < 2:
            threshold = 0.1 if len(self.holds) == 0 else 1.5
            if self.local_active_sec >= threshold:
                self.begin_hold(PHASE_LOCAL)
        elif self.phase == PHASE_SWEEP and len(self.holds) == 2:
            if self.sweep_active_sec >= 0.5:
                self.begin_hold(PHASE_SWEEP)
        elif (len(self.holds) == 3 and self.current_hold is None and
              self.holds[2].get('resume_same_phase') and self.sweep_active_sec >= 1.5 and
              self.state == PerceptionStatus.NO_FACE):
            self.set_state(PerceptionStatus.OK, 'face_restored_with_fresh_measurement')

    def check_success(self):
        if not self.return_face_seen:
            return
        local_elapsed_ok = 5.0 <= self.local_active_sec <= 8.0
        self.criteria = {
            'local_phase_observed': self.local_active_sec > 0.0,
            'sweep_phase_observed': self.sweep_active_sec > 0.0,
            'three_hold_commands_seen': len(self.holds) == 3 and all(
                hold.get('hold_command_seen') for hold in self.holds),
            'servo_confirmed_hold_phase_each_time': len(self.holds) == 3 and all(
                hold.get('servo_hold_diag_seen') and hold.get('servo_hold_phase') == 'HOLD'
                for hold in self.holds),
            'same_phase_resume': len(self.holds) == 3 and all(
                hold.get('resume_same_phase') for hold in self.holds),
            'plans_unchanged_during_holds': len(self.holds) == 3 and all(
                hold.get('plans_unchanged_during_fault') for hold in self.holds),
            'hold_duration_around_0_5_sec': len(self.holds) == 3 and all(
                0.40 <= hold.get('duration_sec', 0.0) <= 0.75 for hold in self.holds),
            'local_active_time_sec': round(self.local_active_sec, 3),
            'local_active_time_5_to_8_sec': local_elapsed_ok,
            'returned_to_face_after_fresh_ok': self.last_target['mode'] == 'FACE',
            'face_measurement_after_fresh_ok': (
                self.return_face_target is not None and
                self.fresh_face_restore_sim_sec is not None and
                self.return_face_target['face_stamp_sec'] > self.fresh_face_restore_sim_sec),
            'servo_confirmed_face_tracking_after_face_command': self.return_face_diag is not None,
        }
        if all(self.criteria[key] for key in (
                'local_phase_observed', 'sweep_phase_observed',
                'three_hold_commands_seen', 'same_phase_resume',
                'servo_confirmed_hold_phase_each_time',
                'plans_unchanged_during_holds', 'hold_duration_around_0_5_sec',
                'local_active_time_5_to_8_sec', 'returned_to_face_after_fresh_ok',
                'face_measurement_after_fresh_ok',
                'servo_confirmed_face_tracking_after_face_command')):
            self.finish('PASS')
            rclpy.shutdown()
        elif self.last_target and self.last_target['mode'] == 'FACE':
            self.finish('FAIL', 'returned_to_face_but_hold_criteria_failed')
            rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='Directory for report.json and events.jsonl')
    parser.add_argument('--max-wall-sec', type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 10.0 <= args.max_wall_sec <= 120.0:
        raise SystemExit('--max-wall-sec must be between 10 and 120 seconds')
    rclpy.init()
    probe = SearchHoldProbe(args)
    try:
        rclpy.spin(probe)
    except KeyboardInterrupt:
        probe.finish('NOT_ASSESSED', 'interrupted')
    finally:
        if not probe.done:
            probe.finish('NOT_ASSESSED', 'spin_ended_before_completion')
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if probe.failed_reason or probe.criteria and not all(
            value is True for key, value in probe.criteria.items()
            if key != 'local_active_time_sec'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
