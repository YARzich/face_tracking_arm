#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run a read-only xArm status monitor and exit nonzero on detected faults."""

from collections import deque
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import queue
import signal
import sys
import time

from face_tracking_arm.msg import (
    FaceTransformProvenance, HardwareIoStatus, ServoExecutionContext,
    ServoTrajectoryPublication, TrackingTarget)

from face_tracking_bringup import hardware_monitor as hardware_monitor_module
from face_tracking_bringup.hardware_fault_recorder import (
    BoundedContextHistory, BoundedPublicationPayloadCache, HardwareFaultRecorder,
    missing_publication_payload_companions)
from face_tracking_bringup.hardware_monitor import (
    can_request_controlled_stop, face_transform_provenance_details,
    hardware_io_status_details, HardwareMonitorFault, HardwareStateMonitor,
    inhibit_future_result, inhibit_status_confirms_request, report_stamp_problem,
    robot_report, robot_report_details, RobotReport,
    servo_execution_context_details, servo_trajectory_publication_details,
    stop_future_result, StopEvidence)

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool

from std_srvs.srv import Trigger

from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import SetInt16


def _sdk_reader(stop_event, output_queue, robot_ip, request_timeout_sec, poll_period_sec):
    """Own the SDK socket in a child process so a stuck request can be killed safely."""
    client = None
    output_queue.put(('started', None))
    try:
        from face_tracking_bringup.hardware_probe import _SdkReadOnlyClient
        client = _SdkReadOnlyClient(robot_ip, request_timeout_sec)
        next_poll = time.monotonic()
        serial_sent = False
        while not stop_event.is_set():
            snapshot = client.get_reduced_states()
            output_queue.put(('snapshot', (snapshot, time.monotonic())))
            if not serial_sent:
                output_queue.put(('serial', client.get_robot_sn()))
                serial_sent = True
            next_poll += poll_period_sec
            stop_event.wait(max(0.0, next_poll - time.monotonic()))
    except BaseException as error:  # Child failures are delivered to the ROS supervisor.
        try:
            output_queue.put(('error', str(error)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException:
                pass


class HardwareMonitorNode(Node):
    """ROS adapter that reports readiness and supervises RobotMsg + SDK state."""

    def __init__(self):
        """Initialize the readiness gate, state subscriber, and SDK poll worker."""
        super().__init__('hardware_monitor')
        self.declare_parameter('robot_ip', '')
        self.declare_parameter('expected_mode', 1)
        self.declare_parameter('robot_state_topic', '/xarm/robot_states')
        self.declare_parameter('hardware_io_status_topic', '')
        self.declare_parameter('servo_execution_context_topic',
                               '/servo_node/execution_context')
        self.declare_parameter(
            'face_transform_provenance_topic', '/tracking/face_transform_provenance')
        self.declare_parameter(
            'servo_trajectory_publications_topic', '/servo_node/trajectory_publications')
        self.declare_parameter('measured_stop_stable_samples', 0)
        self.declare_parameter('measured_stop_velocity_tolerance_rad_s', -1.0)
        self.declare_parameter('measured_stop_position_tolerance_rad', -1.0)
        self.declare_parameter('measured_stop_max_feedback_age_sec', 0.0)
        self.declare_parameter('preflight_snapshot_json', '')
        self.declare_parameter('artifact_manifest_json', '{}')
        self.declare_parameter('startup_timeout_sec', 20.0)
        self.declare_parameter('report_timeout_sec', 0.5)
        self.declare_parameter('report_stamp_timeout_sec', 0.5)
        self.declare_parameter('report_future_tolerance_sec', 0.25)
        self.declare_parameter('sdk_request_timeout_sec', 0.5)
        self.declare_parameter('sdk_state_timeout_sec', 0.75)
        self.declare_parameter('sdk_poll_period_sec', 0.5)
        self.declare_parameter('stop_observation_timeout_sec', 5.0)
        self.declare_parameter('fault_record_directory', '/tmp/face_tracking_faults')
        self.declare_parameter('set_state_service', '/xarm/set_state')
        self.declare_parameter('hardware_inhibit_service', '')
        self.robot_ip = self.get_parameter('robot_ip').value
        if not self.robot_ip:
            raise ValueError('robot_ip is required for hardware monitoring')
        preflight_json = self.get_parameter('preflight_snapshot_json').value
        if not preflight_json:
            raise ValueError('preflight_snapshot_json is required for hardware monitoring')
        preflight = json.loads(preflight_json)
        self.preflight_snapshot_json = preflight_json
        manifest_json = self.get_parameter('artifact_manifest_json').value
        if not manifest_json:
            manifest_json = '{}'
        self.artifact_manifest_json = manifest_json
        self.artifact_manifest = json.loads(manifest_json)
        if not isinstance(self.artifact_manifest, dict):
            raise ValueError('artifact_manifest_json must contain a JSON object')
        self.report_timeout_sec = float(self.get_parameter('report_timeout_sec').value)
        self.report_stamp_timeout_sec = float(
            self.get_parameter('report_stamp_timeout_sec').value)
        self.report_future_tolerance_sec = float(
            self.get_parameter('report_future_tolerance_sec').value)
        self.sdk_request_timeout_sec = float(
            self.get_parameter('sdk_request_timeout_sec').value)
        self.sdk_state_timeout_sec = float(self.get_parameter('sdk_state_timeout_sec').value)
        self.sdk_poll_period_sec = float(self.get_parameter('sdk_poll_period_sec').value)
        self.stop_observation_timeout_sec = float(
            self.get_parameter('stop_observation_timeout_sec').value)
        self.fault_record_directory = str(self.get_parameter('fault_record_directory').value)
        if (not math.isfinite(self.sdk_request_timeout_sec)
                or not 0 < self.sdk_request_timeout_sec <= 0.5
                or not math.isfinite(self.sdk_poll_period_sec)
                or self.sdk_poll_period_sec != 0.5
                or not math.isfinite(self.report_timeout_sec)
                or not 0 < self.report_timeout_sec <= 0.5
                or not math.isfinite(self.report_stamp_timeout_sec)
                or not 0 < self.report_stamp_timeout_sec <= 0.5
                or not math.isfinite(self.report_future_tolerance_sec)
                or self.report_future_tolerance_sec <= 0
                or not math.isfinite(self.stop_observation_timeout_sec)
                or not 0 < self.stop_observation_timeout_sec <= 60):
            raise ValueError('SDK request timeout must be <=0.5s and polls must run at 2Hz')
        self.monitor = HardwareStateMonitor(
            preflight,
            expected_mode=int(self.get_parameter('expected_mode').value),
            startup_timeout_sec=float(self.get_parameter('startup_timeout_sec').value),
            report_timeout_sec=self.report_timeout_sec,
            sdk_timeout_sec=self.sdk_state_timeout_sec)
        ready_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_pub = self.create_publisher(
            Bool, '/face/hardware_monitor/ready', ready_qos)
        self._publish_ready(False)
        state_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        topic = self.get_parameter('robot_state_topic').value
        status_topic = self.get_parameter('hardware_io_status_topic').value
        if not status_topic:
            if topic.endswith('/robot_states'):
                status_topic = topic[:-len('/robot_states')] + '/hardware_io_status'
            else:
                status_topic = '/xarm/hardware_io_status'
        self.hardware_io_status_topic = status_topic
        self.servo_execution_context_topic = str(
            self.get_parameter('servo_execution_context_topic').value)
        self.face_transform_provenance_topic = str(
            self.get_parameter('face_transform_provenance_topic').value)
        self.servo_trajectory_publications_topic = str(
            self.get_parameter('servo_trajectory_publications_topic').value)
        self.subscription = self.create_subscription(
            RobotMsg, topic, self._on_robot_report, state_qos)
        self.hardware_io_subscription = self.create_subscription(
            HardwareIoStatus, status_topic, self._on_hardware_io_status,
            QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE))
        self.execution_context_history = BoundedContextHistory(capacity=4096)
        self.execution_context_subscription = self.create_subscription(
            ServoExecutionContext, self.servo_execution_context_topic,
            self._on_servo_execution_context,
            QoSProfile(depth=512, reliability=ReliabilityPolicy.RELIABLE))
        self.face_transform_history = BoundedContextHistory(capacity=2048)
        self._face_transform_joined = set()
        self._face_transform_join_order = deque(maxlen=4096)
        self._face_transform_seen = set()
        self._face_transform_seen_order = deque(maxlen=2048)
        self._current_face_transform_provenance = None
        self._current_face_transform_used_for_reference = False
        self.face_transform_subscription = self.create_subscription(
            FaceTransformProvenance, self.face_transform_provenance_topic,
            self._on_face_transform_provenance,
            QoSProfile(depth=128, reliability=ReliabilityPolicy.RELIABLE))
        self.publication_payload_cache = BoundedPublicationPayloadCache(capacity=256)
        self._fault_pinned_publications = None
        self._fault_missing_publications = ()
        self._trajectory_publications_seen = set()
        self._trajectory_publications_seen_order = deque(maxlen=4096)
        self._last_publication_drop_count = None
        self._last_context_publication_drop_count = None
        self._trajectory_publication_joined = set()
        self._trajectory_publication_join_order = deque(maxlen=8192)
        self.trajectory_publication_subscription = self.create_subscription(
            ServoTrajectoryPublication, self.servo_trajectory_publications_topic,
            self._on_servo_trajectory_publication,
            QoSProfile(depth=512, reliability=ReliabilityPolicy.RELIABLE))
        self.stop_client = self.create_client(
            SetInt16, self.get_parameter('set_state_service').value)
        self.hardware_inhibit_service = str(
            self.get_parameter('hardware_inhibit_service').value)
        self.inhibit_client = (self.create_client(Trigger, self.hardware_inhibit_service)
                               if self.hardware_inhibit_service else None)
        self._inhibit_required = False
        self._inhibit_status_confirmed = False
        self._inhibit_request_pending = False
        self._inhibit_future = None
        self._inhibit_deadline = None
        self._inhibit_retry_at = 0.0
        self._inhibit_epoch = None
        self._inhibit_previous_request_id = None
        self._inhibit_ack_recorded = False
        self._stop_future = None
        self._stop_deadline = None
        self.stop_requested = False
        self._stop_request_pending = False
        self._last_stop_wait_reason = None
        self.stop_acknowledged = False
        self.stop_reported = False
        self._last_stop_report_signature = None
        # This monitor has no fresh joint-position/velocity feedback source.
        # Keep measured standstill explicitly unknown instead of inferring it from ACK/state.
        self.stop_measured = None
        stop_values = (
            int(self.get_parameter('measured_stop_stable_samples').value),
            float(self.get_parameter('measured_stop_velocity_tolerance_rad_s').value),
            float(self.get_parameter('measured_stop_position_tolerance_rad').value),
            float(self.get_parameter('measured_stop_max_feedback_age_sec').value))
        if stop_values == (0, -1.0, -1.0, 0.0):
            self.stop_evidence = None
        elif (stop_values[0] < 2 or stop_values[1] < 0 or stop_values[2] < 0
              or stop_values[3] <= 0 or not all(math.isfinite(v) for v in stop_values[1:])):
            raise ValueError('measured-stop thresholds must be fully configured and valid')
        else:
            self.stop_evidence = StopEvidence(
                stable_samples=stop_values[0], velocity_tolerance=stop_values[1],
                position_tolerance=stop_values[2], max_feedback_age_sec=stop_values[3])
        self._last_inhibit_key = None
        self._last_hardware_fault_key = None
        self._measured_stop_event_recorded = False
        self._last_feedback_receipt = None
        self._last_feedback_age_at_receipt = None
        self.last_hardware_io_status = None
        self._last_hardware_io_receipt = None
        self._last_hardware_io_age_at_receipt = None
        self._last_write_in_flight = None
        self._latest_hardware_status_key = None
        self._latest_hardware_status_details = None
        self._latest_hardware_status_receipt = None
        self._joined_context_keys = set()
        self._joined_context_order = deque(maxlen=2048)
        self.last_execution_context = None
        self._latest_live_report = None
        self._latest_live_report_at = None
        self.recorder = HardwareFaultRecorder()
        self._stop_observation_deadline = None
        self._stop_window_done = False
        self._stop_ack_settled = False
        self._stop_request_expired_recorded = False
        self._fault_recorded = False
        self._mp_context = multiprocessing.get_context('spawn')
        self._stop_event = self._mp_context.Event()
        self._output = self._mp_context.Queue(maxsize=2)
        self._reader = self._mp_context.Process(
            target=_sdk_reader,
            args=(self._stop_event, self._output, self.robot_ip,
                  self.sdk_request_timeout_sec, self.sdk_poll_period_sec),
            name='xarm-read-only-monitor', daemon=True)
        self._reader.start()
        self._sdk_worker_started_at = time.monotonic()
        self._last_sdk_receipt = None
        self._last_warning = 0
        self.fault = None
        self.last_report_details = None
        self.timer = self.create_timer(0.05, self._poll_health)
        self.get_logger().info(f'Monitoring {topic}; arming after matched SERVOJ and SDK state')

    def _publish_ready(self, value):
        message = Bool()
        message.data = value
        self.ready_pub.publish(message)

    def _on_robot_report(self, message):
        parsed = None
        try:
            stamp = message.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            report = robot_report(message)
            parsed = report
            self.last_report_details = robot_report_details(message)
            self.recorder.record('robot_report', stamp_ns, report.mode, report.state,
                                 report.mt_able, report.mt_brake, report.error, report.warning)
            self.recorder.record('robot_report_full', *self.last_report_details)
            problem = report_stamp_problem(
                stamp_ns, self.get_clock().now().nanoseconds,
                self.report_stamp_timeout_sec, self.report_future_tolerance_sec)
            if problem is not None:
                if self.monitor.ready:
                    self._set_fault(problem)
                return
            report_receipt = time.monotonic()
            self._latest_live_report = report
            self._latest_live_report_at = report_receipt
            if self.fault is not None:
                self.stop_reported = self.monitor.observe_stop_report(report)
                self.recorder.record('stop_report', stamp_ns, report.state,
                                     report.error, report.mt_able, report.mt_brake)
                signature = (report.mode, report.state, report.error,
                             report.mt_able, report.mt_brake)
                if signature != self._last_stop_report_signature:
                    self.recorder.record('stop_report_transition', stamp_ns, *signature)
                    self._last_stop_report_signature = signature
                if self._stop_request_pending:
                    self._request_stop_if_safe()
                return
            self.monitor.observe_report(
                report, time.monotonic(), source_stamp_ns=stamp_ns)
            if report.warning != self._last_warning:
                if report.warning:
                    self.get_logger().warning(f'Robot warning code {report.warning}')
                self._last_warning = report.warning
        except HardwareMonitorFault as error:
            self._set_fault(str(error), parsed or self.monitor.fault_report)
        except ValueError as error:
            self._set_fault(str(error), parsed)

    def _on_hardware_io_status(self, message):
        """Record the driver's existing read/write snapshot without polling its SDK."""
        receipt = time.monotonic()
        details = hardware_io_status_details(message)
        self.last_hardware_io_status = details
        self._last_hardware_io_receipt = receipt
        status_key = (int(message.epoch), int(message.read_sequence))
        self._latest_hardware_status_key = status_key
        self._latest_hardware_status_details = details
        self._latest_hardware_status_receipt = receipt
        self._record_context_join(status_key)
        self._last_write_in_flight = int(message.write_in_flight)
        status_stamp_ns = (int(message.header.stamp.sec) * 1_000_000_000
                           + int(message.header.stamp.nanosec))
        status_now_ns = self.get_clock().now().nanoseconds
        self._last_hardware_io_age_at_receipt = (
            (status_now_ns - status_stamp_ns) / 1e9
            if status_stamp_ns > 0 and status_now_ns >= status_stamp_ns else float('inf'))
        robot_stamp_ns = (int(message.robot_msg_stamp.sec) * 1_000_000_000
                          + int(message.robot_msg_stamp.nanosec))
        robot_age_sec = int(message.robot_msg_age_us) / 1_000_000.0
        report_age_at_receipt = robot_age_sec + self._last_hardware_io_age_at_receipt
        if (int(message.robot_msg_sequence) > 0 and robot_stamp_ns > 0
                and math.isfinite(report_age_at_receipt)
                and report_age_at_receipt >= 0
                and report_age_at_receipt <= self.report_timeout_sec):
            self._latest_live_report = RobotReport(
                mode=int(message.robot_mode), state=int(message.robot_state),
                mt_able=int(message.robot_mt_able), mt_brake=int(message.robot_mt_brake),
                error=int(message.robot_error_code), warning=int(message.robot_warn_code))
            self._latest_live_report_at = receipt - report_age_at_receipt
        else:
            self._latest_live_report = None
            self._latest_live_report_at = None
        # First-fault provenance is always recorded, but its new automatic
        # reaction stays disabled until the driver policy is explicitly armed.
        if message.fault_latched:
            fault_key = (int(message.fault_epoch), int(message.fault_sequence),
                         int(message.fault_kind), int(message.fault_code))
            if fault_key != self._last_hardware_fault_key:
                if message.runtime_actions_enabled and self.fault is None:
                    reason = (f'Driver I/O fault kind={fault_key[2]} code={fault_key[3]} '
                              f'sequence={fault_key[1]} epoch={fault_key[0]}')
                    self._set_fault(reason)
                self.recorder.record('hardware_io_fault', *fault_key,
                                     int(message.fault_read_attempt_sequence),
                                     int(message.fault_stamp.sec),
                                     int(message.fault_stamp.nanosec))
                self._last_hardware_fault_key = fault_key
        if message.inhibit_latched:
            inhibit_key = (int(message.inhibit_epoch), int(message.inhibit_request_id))
            if inhibit_key != self._last_inhibit_key:
                if message.runtime_actions_enabled and self.fault is None:
                    self._set_fault(
                        f'Driver ServoJ inhibit latched at epoch {inhibit_key[0]} '
                        f'(request {inhibit_key[1]})')
                self.recorder.record('inhibit_latched', *inhibit_key)
                self._last_inhibit_key = inhibit_key
        if self._inhibit_required:
            self._observe_inhibit_status(message, receipt)
        self.recorder.record('hardware_io_status', receipt, details)
        if (message.runtime_actions_enabled and message.inhibit_latched
                and not self.stop_requested):
            self._stop_request_pending = True
            self._request_stop_if_safe()
        elif self._stop_request_pending:
            self._request_stop_if_safe()
        if self.fault is None or self.stop_evidence is None:
            return

        status_age = self._last_hardware_io_age_at_receipt
        steady_age = max(0.0, time.monotonic() - receipt)
        transport_age = max(status_age, steady_age)
        report = RobotReport(
            mode=int(message.robot_mode), state=int(message.robot_state),
            mt_able=int(message.robot_mt_able), mt_brake=int(message.robot_mt_brake),
            error=int(message.robot_error_code), warning=int(message.robot_warn_code))
        robot_stamp_ns = (int(message.robot_msg_stamp.sec) * 1_000_000_000
                          + int(message.robot_msg_stamp.nanosec))
        robot_age_sec = int(message.robot_msg_age_us) / 1_000_000.0
        robot_report_valid = (
            int(message.robot_msg_sequence) > 0 and robot_stamp_ns > 0
            and math.isfinite(robot_age_sec)
            and robot_age_sec + status_age <= self.report_timeout_sec
            and status_age <= self.report_timeout_sec)
        self.stop_evidence.observe_report(report, valid=robot_report_valid)
        previous_measured = self.stop_measured
        measured = self.stop_evidence.observe_feedback(
            message.q, message.dq,
            read_ok=bool(message.read_ok) and int(message.source_kind) == message.SOURCE_HARDWARE,
            read_sequence=int(message.read_sequence), epoch=int(message.epoch),
            read_age_sec=int(message.read_age_us) / 1_000_000.0,
            transport_age_sec=transport_age)
        self.stop_measured = bool(measured)
        self._last_feedback_receipt = receipt
        self._last_feedback_age_at_receipt = (
            int(message.read_age_us) / 1_000_000.0 + transport_age)
        if previous_measured and not measured:
            self.recorder.record('stop_measurement_lost', int(message.read_sequence),
                                 'feedback invalid, stale, replayed, or moving')
            self._measured_stop_event_recorded = False
        if measured:
            if not self._measured_stop_event_recorded:
                self.recorder.record('stop_measured', int(message.read_sequence),
                                     tuple(message.q), tuple(message.dq))
                self._measured_stop_event_recorded = True

    def _on_servo_execution_context(self, message):
        """Record executor IDs and targets without performing another TF lookup."""
        receipt = time.monotonic()
        details = servo_execution_context_details(message)
        context_values = dict(details)
        epoch = int(context_values.get('hardware_epoch', 0))
        dropped_publications = int(context_values.get('dropped_publication_events', 0))
        if (dropped_publications > 0
                and self._last_context_publication_drop_count !=
                (epoch, dropped_publications)):
            previous = (0 if self._last_context_publication_drop_count is None
                        or self._last_context_publication_drop_count[0] != epoch else
                        self._last_context_publication_drop_count[1])
            self.recorder.record('servo_trajectory_publication_overflow',
                                 epoch, previous, dropped_publications,
                                 'execution-context snapshot')
        self._last_context_publication_drop_count = (epoch, dropped_publications)
        source_stamp = message.header.stamp
        source_stamp_ns = int(source_stamp.sec) * 1_000_000_000 + int(source_stamp.nanosec)
        self.last_execution_context = details
        if not self.execution_context_history.record(source_stamp_ns, details, receipt):
            self.recorder.record('servo_execution_context_drop', receipt,
                                 int(message.tick_sequence))
        self.recorder.record('servo_execution_context', receipt, details)
        self._current_face_transform_provenance = None
        self._current_face_transform_used_for_reference = False
        self._record_context_join((int(message.hardware_epoch),
                                   int(message.hardware_read_sequence)),
                                  context=(receipt, details))
        active_ids = (dict(details).get('active_publication_id', 0),
                      dict(details).get('latest_publication_id', 0),
                      dict(details).get('braking_tail_publication_id', 0))
        self.publication_payload_cache.set_active(
            int(message.hardware_epoch), active_ids)
        self._join_face_transform_with_context(receipt, details)
        self._join_trajectory_publications_with_context(receipt, details)

    def _on_face_transform_provenance(self, message):
        """Retain the exact TF chain and join it to execution context by source stamp."""
        receipt = time.monotonic()
        details = face_transform_provenance_details(message)
        source_stamp = dict(details)['source_stamp']
        source_stamp_ns = source_stamp[0] * 1_000_000_000 + source_stamp[1]
        sequence = int(message.accepted_face_sequence)
        seen_key = (sequence, source_stamp_ns)
        if seen_key not in self._face_transform_seen:
            if len(self._face_transform_seen_order) == self._face_transform_seen_order.maxlen:
                self._face_transform_seen.discard(self._face_transform_seen_order[0])
            self._face_transform_seen_order.append(seen_key)
            self._face_transform_seen.add(seen_key)
            self.recorder.record('face_transform_provenance', receipt, details)
        if not self.face_transform_history.record(source_stamp_ns, details, receipt):
            self.recorder.record('face_transform_provenance_drop', receipt, sequence)
        match = self.execution_context_history.match_payload_stamp(
            'face_stamp', source_stamp_ns, receipt, 0.5)
        if match is not None:
            self._join_face_transform_with_context(receipt, match[0].payload, details)

    def _join_face_transform_with_context(self, receipt, context_details,
                                          provenance_details=None):
        """Record one exact-stamp target/TF association from either callback order."""
        context = dict(context_details)
        stamp = context.get('face_stamp')
        if not stamp or stamp == (0, 0):
            return
        source_stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
        if provenance_details is None:
            match = self.face_transform_history.match(
                receipt, 0.5, key=None)
            if match is None:
                return
            item, _ = match
            if dict(item.payload).get('source_stamp') != stamp:
                exact = self.face_transform_history.match_payload_stamp(
                    'source_stamp', source_stamp_ns, receipt, 0.5)
                if exact is None:
                    return
                item, _ = exact
            provenance_details = item.payload
        provenance = dict(provenance_details)
        if provenance.get('source_stamp') != stamp:
            return
        join_key = (int(provenance.get('accepted_face_sequence', 0)), source_stamp_ns,
                    int(context.get('tick_sequence', 0)))
        if join_key in self._face_transform_joined:
            return
        if len(self._face_transform_join_order) == self._face_transform_join_order.maxlen:
            self._face_transform_joined.discard(self._face_transform_join_order[0])
        self._face_transform_join_order.append(join_key)
        self._face_transform_joined.add(join_key)
        face_target_used = (
            int(context.get('target_mode', -1)) == TrackingTarget.FACE
            and bool(context.get('target_available', False))
            and bool(context.get('target_used_for_reference', False)))
        current_context = getattr(self, 'last_execution_context', None)
        if current_context is not None:
            current = dict(current_context)
            if (int(current.get('tick_sequence', -1)) == join_key[2]
                    and current.get('face_stamp') == stamp):
                self._current_face_transform_provenance = (
                    join_key, provenance_details)
                self._current_face_transform_used_for_reference = face_target_used
        self.recorder.record('face_transform_execution_context_join',
                             receipt, join_key, source_stamp_ns,
                             int(context.get('target_mode', -1)),
                             bool(context.get('target_available', False)),
                             bool(context.get('target_used_for_reference', False)))

    def _on_servo_trajectory_publication(self, message):
        """Record the complete sent trajectory once under its driver-independent ID."""
        receipt = time.monotonic()
        details = servo_trajectory_publication_details(message)
        epoch = int(message.hardware_epoch)
        publication_id = int(message.publication_id)
        seen_key = (epoch, publication_id)
        if seen_key not in self._trajectory_publications_seen:
            if (len(self._trajectory_publications_seen_order) ==
                    self._trajectory_publications_seen_order.maxlen):
                self._trajectory_publications_seen.discard(
                    self._trajectory_publications_seen_order[0])
            self._trajectory_publications_seen_order.append(seen_key)
            self._trajectory_publications_seen.add(seen_key)
            self.recorder.record('servo_trajectory_publication', receipt, details)
            if not self.publication_payload_cache.record(
                    epoch, publication_id, receipt, details):
                self.recorder.record('servo_trajectory_payload_cache_drop',
                                     epoch, publication_id)
        dropped_count = int(message.dropped_publication_events)
        if (dropped_count > 0
                and self._last_publication_drop_count != (epoch, dropped_count)):
            previous = (0 if self._last_publication_drop_count is None
                        or self._last_publication_drop_count[0] != epoch
                        else self._last_publication_drop_count[1])
            self.recorder.record('servo_trajectory_publication_overflow',
                                 epoch, previous, dropped_count)
        self._last_publication_drop_count = (epoch, dropped_count)
        match = self.execution_context_history.match_payload_id(
            ('active_publication_id', 'latest_publication_id',
             'braking_tail_publication_id'), publication_id, receipt, 0.5,
            hardware_epoch=epoch)
        if match is not None:
            self._join_trajectory_publications_with_context(
                receipt, match[0].payload, details)

    def _join_trajectory_publications_with_context(
            self, receipt, context_details, publication_details=None):
        """Associate active/latest/tail publication IDs with full trajectory events."""
        context = dict(context_details)
        epoch = int(context.get('hardware_epoch', 0))
        tick = int(context.get('tick_sequence', 0))
        ids = (int(context.get('active_publication_id', 0)),
               int(context.get('latest_publication_id', 0)),
               int(context.get('braking_tail_publication_id', 0)))
        if publication_details is not None:
            candidates = [dict(publication_details)]
        else:
            candidates = []
            for publication_id in dict.fromkeys(value for value in ids if value > 0):
                payload = self.publication_payload_cache.get_payload(epoch, publication_id)
                if payload is not None:
                    candidates.append(dict(payload))
        for publication in candidates:
            publication_id = int(publication.get('publication_id', 0))
            if (publication_id <= 0 or publication.get('hardware_epoch') != epoch
                    or publication_id not in ids):
                continue
            join_key = (epoch, publication_id, tick)
            if join_key in self._trajectory_publication_joined:
                continue
            if len(self._trajectory_publication_join_order) == (
                    self._trajectory_publication_join_order.maxlen):
                self._trajectory_publication_joined.discard(
                    self._trajectory_publication_join_order[0])
            self._trajectory_publication_join_order.append(join_key)
            self._trajectory_publication_joined.add(join_key)
            self.recorder.record('servo_trajectory_execution_context_join',
                                 receipt, join_key, publication_id)

    def _record_context_join(self, key, context=None):
        """Attach exact matching read provenance to a context event once per tick."""
        context_tick = None
        context_receipt = None
        context_details = None
        if context is not None:
            context_receipt, context_details = context
            context_tick = dict(context_details).get('tick_sequence')
        else:
            match = self.execution_context_history.match(
                time.monotonic(), 0.5, key=key)
            if match is not None:
                item, _ = match
                context_receipt, context_details = item.receipt_monotonic, item.payload
                context_tick = dict(context_details).get('tick_sequence')
        if (context_details is None or self._latest_hardware_status_key != key
                or self._latest_hardware_status_details is None):
            return
        joined_key = (int(context_tick or 0), key[0], key[1])
        if joined_key in self._joined_context_keys:
            return
        if len(self._joined_context_order) == self._joined_context_order.maxlen:
            self._joined_context_keys.discard(self._joined_context_order[0])
        self._joined_context_order.append(joined_key)
        self._joined_context_keys.add(joined_key)
        self.recorder.record(
            'hardware_io_execution_context_join',
            self._latest_hardware_status_receipt, context_receipt,
            key[0], key[1], self._latest_hardware_status_details, context_details)

    def _poll_health(self):
        if self.fault is not None:
            self._ensure_fault_recorded()
            now = time.monotonic()
            if self._inhibit_required and not self._inhibit_status_confirmed:
                self._progress_inhibit_request(now)
            if (self._stop_request_pending and not self.stop_requested
                    and self._stop_observation_deadline is not None
                    and now < self._stop_observation_deadline):
                self._request_stop_if_safe()
            if (self._stop_request_pending and not self.stop_requested
                    and self._stop_observation_deadline is not None
                    and now >= self._stop_observation_deadline
                    and not self._stop_request_expired_recorded):
                self.recorder.record('stop_request_unconfirmed_timeout',
                                     self._last_stop_wait_reason or 'no fresh safe report')
                self._stop_request_pending = False
                self._stop_request_expired_recorded = True
            if not self._stop_ack_settled:
                self._stop_ack_settled = self.settle_stop_request()
            self._expire_measured_stop_if_stale()
            deadline_reached = (self._stop_observation_deadline is not None
                                and now >= self._stop_observation_deadline)
            close_outcome = ('measured' if self.stop_measured is True
                             else ('timeout' if deadline_reached else None))
            if (not self._stop_window_done and close_outcome is not None
                    and self._stop_ack_settled):
                self._stop_window_done = self.recorder.close_window(close_outcome)
                if self._stop_window_done:
                    self._write_fault_dump()
            return
        try:
            self._drain_sdk_results()
            if self._last_sdk_receipt is None:
                if (self._sdk_worker_started_at is not None
                        and time.monotonic() - self._sdk_worker_started_at
                        > self.monitor.startup_timeout_sec):
                    raise HardwareMonitorFault(
                        'Timed out waiting for the first SDK serial and safety snapshot')
            elif time.monotonic() - self._last_sdk_receipt > self.sdk_state_timeout_sec:
                raise HardwareMonitorFault('SDK safety-state poll heartbeat expired')
            self.monitor.check_health()
            if self.monitor.ready:
                self._publish_ready(True)
        except HardwareMonitorFault as error:
            self._set_fault(str(error))

    def _expire_measured_stop_if_stale(self):
        if (self.stop_measured is not True or self.stop_evidence is None
                or self._last_feedback_receipt is None
                or self._last_feedback_age_at_receipt is None):
            return
        age = (self._last_feedback_age_at_receipt
               + max(0.0, time.monotonic() - self._last_feedback_receipt))
        if age <= self.stop_evidence.max_feedback_age_sec:
            return
        self.stop_evidence.invalidate_feedback()
        self.stop_measured = False
        self._measured_stop_event_recorded = False
        self.recorder.record('stop_measurement_lost', 'expired', age)

    def _drain_sdk_results(self):
        while True:
            try:
                kind, payload = self._output.get_nowait()
            except queue.Empty:
                break
            if kind == 'snapshot':
                settings, received_at = payload
                self.recorder.record('sdk_safety_snapshot', received_at, settings)
                self.monitor.observe_sdk(settings, received_at)
                self._last_sdk_receipt = received_at
            elif kind == 'serial':
                self.monitor.observe_serial(payload)
            elif kind == 'started':
                # The process can spend time importing the SDK or opening its bounded socket.
                # Startup readiness has its own longer deadline; steady polls remain bounded.
                pass
            elif kind == 'error':
                raise HardwareMonitorFault(f'SDK reduced-state request failed: {payload}')
        if not self._reader.is_alive() and not self._stop_event.is_set():
            raise HardwareMonitorFault('SDK read-only polling worker exited')

    def _set_fault(self, reason, report=None, *, request_stop=True):
        if self.fault is not None:
            return
        self.fault = reason
        self._stop_observation_deadline = (
            time.monotonic() + self.stop_observation_timeout_sec)
        if report is None:
            report = self.monitor.fault_report or self.monitor.last_report
        if report is None:
            self._fault_context = (reason,)
            report_context = 'RobotMsg unavailable'
        else:
            self._fault_context = (reason, report.mode, report.state, report.mt_able,
                                   report.mt_brake, report.error, report.warning,
                                   self.last_report_details)
            report_context = f'RobotMsg fields={self.last_report_details or report}'
        self._ensure_fault_recorded()
        self.get_logger().fatal(f'{reason}; {report_context}')
        self._publish_ready(False)
        if request_stop:
            status = self.last_hardware_io_status or ()
            status = dict(status)
            if status.get('runtime_actions_enabled') and status.get('hardware_active'):
                self._inhibit_required = True
                self._stop_request_pending = True
                self._begin_inhibit_request()
            else:
                # Keep legacy gate-off behavior; the new latch protocol is only
                # enabled with the matching driver runtime policy.
                self._request_stop_if_safe()

    def _begin_inhibit_request(self):
        """Start one asynchronous private inhibit request for this fault epoch."""
        if self._inhibit_status_confirmed or self._inhibit_future is not None:
            return
        if time.monotonic() < self._inhibit_retry_at:
            return
        status = dict(self.last_hardware_io_status or ())
        if not status.get('hardware_active') or not status.get('runtime_actions_enabled'):
            return
        epoch = int(status.get('epoch', 0))
        request_id = int(status.get('inhibit_request_id', 0))
        if status.get('inhibit_latched') and int(status.get('inhibit_epoch', -1)) == epoch:
            self._inhibit_epoch = epoch
            self._inhibit_previous_request_id = request_id - 1
            status_age = (self._last_hardware_io_age_at_receipt
                          + max(0.0, time.monotonic() - self._last_hardware_io_receipt))
            self._observe_inhibit_status_message(status, status_age)
            return
        if self.inhibit_client is None or not self.inhibit_client.service_is_ready():
            self._inhibit_request_pending = True
            self._inhibit_retry_at = time.monotonic() + 0.25
            self._last_stop_wait_reason = 'hardware inhibit service unavailable'
            return
        self._inhibit_epoch = epoch
        self._inhibit_previous_request_id = request_id
        try:
            self._inhibit_future = self.inhibit_client.call_async(Trigger.Request())
            self._inhibit_deadline = time.monotonic() + 0.5
            self._inhibit_request_pending = False
            self._inhibit_ack_recorded = False
            self.recorder.record('inhibit_request', epoch, request_id)
        except Exception as error:
            self._inhibit_request_pending = True
            self._inhibit_retry_at = time.monotonic() + 0.25
            self._last_stop_wait_reason = f'hardware inhibit request failed: {error}'

    def _progress_inhibit_request(self, now):
        if self._inhibit_status_confirmed:
            return
        future = self._inhibit_future
        if future is not None:
            done, accepted, detail = inhibit_future_result(
                future, now, self._inhibit_deadline or 0.0)
            if done:
                self._remove_pending_request(self.inhibit_client, future)
                self._inhibit_future = None
                self._inhibit_deadline = None
                if accepted:
                    if not self._inhibit_ack_recorded:
                        self.recorder.record('inhibit_service_ack', True)
                        self._inhibit_ack_recorded = True
                    self._inhibit_request_pending = True
                    self._inhibit_retry_at = now + 0.25
                else:
                    self.recorder.record('inhibit_service_ack', False, detail)
                    self._last_stop_wait_reason = detail
                    self._inhibit_request_pending = True
                    self._inhibit_retry_at = now + 0.25
        if self._inhibit_request_pending and now < (self._stop_observation_deadline or 0.0):
            self._begin_inhibit_request()

    def _observe_inhibit_status(self, message, receipt):
        age = self._last_hardware_io_age_at_receipt
        self._observe_inhibit_status_message(
            {
                'hardware_active': message.hardware_active,
                'runtime_actions_enabled': message.runtime_actions_enabled,
                'epoch': message.epoch,
                'inhibit_latched': message.inhibit_latched,
                'inhibit_epoch': message.inhibit_epoch,
                'inhibit_request_id': message.inhibit_request_id,
                'write_in_flight': message.write_in_flight,
            }, age)

    def _observe_inhibit_status_message(self, status, age):
        was_confirmed = self._inhibit_status_confirmed
        confirmed = inhibit_status_confirms_request(
                hardware_active=status.get('hardware_active', False),
                runtime_actions_enabled=status.get('runtime_actions_enabled', False),
                status_epoch=status.get('epoch', -1),
                inhibit_latched=status.get('inhibit_latched', False),
                inhibit_epoch=status.get('inhibit_epoch', -1),
                inhibit_request_id=status.get('inhibit_request_id', 0),
                requested_epoch=self._inhibit_epoch,
                previous_request_id=self._inhibit_previous_request_id,
                status_age_sec=age if age is not None else float('inf'),
                write_in_flight=status.get('write_in_flight', -1),
                freshness_sec=self.report_timeout_sec)
        self._inhibit_status_confirmed = confirmed
        if confirmed:
            self._inhibit_request_pending = False
            self._stop_request_pending = True
            self.recorder.record('inhibit_status_confirmed', self._inhibit_epoch,
                                 status.get('inhibit_request_id'))
            self._request_stop_if_safe()
        elif was_confirmed and not self.stop_requested:
            self._inhibit_request_pending = True
            self._inhibit_retry_at = 0.0

    def _ensure_fault_recorded(self):
        if self._fault_recorded:
            return
        if (getattr(self, '_fault_pinned_publications', None) is None
                and hasattr(self, 'publication_payload_cache')):
            self._fault_pinned_publications = self.publication_payload_cache.active_payloads()
            self._fault_missing_publications = (
                self.publication_payload_cache.missing_active_keys())
        self._fault_recorded = self.recorder.latch_fault(*self._fault_context)

    def _write_fault_dump(self):
        if not self._stop_window_done:
            return
        try:
            self._fault_dump_path = self.recorder.write_dump(
                self.fault_record_directory, self._fault_metadata())
            self.get_logger().error(f'Hardware fault record saved: {self._fault_dump_path}')
        except Exception as error:
            self.get_logger().error(f'Hardware fault record could not be saved: {error}')

    def _fault_metadata(self):
        def digest(path):
            try:
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                return None

        pre_events = self.recorder.snapshot()[0]
        retained_active_publications = [
            {'epoch': key[0], 'publication_id': key[1], 'receipt_monotonic': receipt,
             'payload': payload}
            for key, receipt, payload in missing_publication_payload_companions(
                pre_events, self._fault_pinned_publications or ())]
        return {
            'source_sha256': {
                'hardware_monitor_node.py': digest(__file__),
                'hardware_monitor.py': digest(hardware_monitor_module.__file__),
            },
            'source_tree_sha256': self.artifact_manifest.get('source_tree_sha256'),
            'preflight_snapshot_sha256': hashlib.sha256(
                self.preflight_snapshot_json.encode()).hexdigest(),
            'preflight_snapshot': self.preflight_snapshot_json,
            'artifact_manifest': self.artifact_manifest,
            'robot_state_topic': self.get_parameter('robot_state_topic').value,
            'hardware_io_status_topic': self.hardware_io_status_topic,
            'last_hardware_io_status': self.last_hardware_io_status,
            'servo_execution_context_topic': self.servo_execution_context_topic,
            'face_transform_provenance_topic': self.face_transform_provenance_topic,
            'servo_trajectory_publications_topic':
                self.servo_trajectory_publications_topic,
            'last_servo_execution_context': self.last_execution_context,
            'execution_context_join_index': {
                'capacity': self.execution_context_history.capacity,
                'dropped_events': self.execution_context_history.dropped_events,
                'overwritten_events': self.execution_context_history.overwritten_events,
                'joined_ticks_retained': len(self._joined_context_keys),
                'tf_provenance_available': (
                    None if self.last_execution_context is None else
                    dict(self.last_execution_context).get('tf_provenance_available')),
            },
            'face_transform_join_index': {
                'history_capacity': self.face_transform_history.capacity,
                'history_dropped_events': self.face_transform_history.dropped_events,
                'history_overwritten_events': self.face_transform_history.overwritten_events,
                'joined_pairs_retained': len(self._face_transform_joined),
                'tf_provenance_supported': True,
                'tf_provenance_available': (
                    getattr(self, '_current_face_transform_provenance', None) is not None
                    and getattr(self, '_current_face_transform_used_for_reference', False)),
                'tf_source_stamp_join_available': (
                    getattr(self, '_current_face_transform_provenance', None) is not None),
                'face_target_used_for_reference': getattr(
                    self, '_current_face_transform_used_for_reference', False),
                'latest_join': getattr(self, '_current_face_transform_provenance', None),
            },
            'trajectory_publication_join_index': {
                'topic': self.servo_trajectory_publications_topic,
                'payload_cache_capacity': self.publication_payload_cache.capacity,
                'payload_cache_dropped_events':
                    self.publication_payload_cache.dropped_events,
                'payload_cache_overwritten_events':
                    self.publication_payload_cache.overwritten_events,
                'joined_publication_ticks_retained':
                    len(self._trajectory_publication_joined),
                'last_dropped_publication_events': self._last_publication_drop_count,
                'retained_active_publications': retained_active_publications,
                'missing_active_publications': self._fault_missing_publications,
                'payload_cache_byte_capacity': self.publication_payload_cache.byte_capacity,
                'payload_cache_estimated_bytes_retained':
                    self.publication_payload_cache.stored_bytes,
            },
            'measured_stop_thresholds': (None if self.stop_evidence is None else {
                'stable_samples': self.stop_evidence.stable_samples,
                'velocity_tolerance_rad_s': self.stop_evidence.velocity_tolerance,
                'position_tolerance_rad': self.stop_evidence.position_tolerance,
                'max_feedback_age_sec': self.stop_evidence.max_feedback_age_sec,
            }),
            'robot_serial': self.monitor.expected_serial,
            'stop_requested': self.stop_requested,
            'stop_acknowledged': self.stop_acknowledged,
            'stop_reported': self.stop_reported,
            'stop_measured': self.stop_measured,
            'inhibit_service': self.hardware_inhibit_service,
            'inhibit_required': self._inhibit_required,
            'inhibit_request_epoch': self._inhibit_epoch,
            'inhibit_previous_request_id': self._inhibit_previous_request_id,
            'inhibit_status_confirmed': self._inhibit_proof_is_current(),
            'inhibit_request_pending': self._inhibit_request_pending,
        }

    def _request_stop_if_safe(self):
        """Request only state=4 when the last robot report supports a safe attempt."""
        if self.stop_requested:
            self._stop_request_pending = False
            return
        if self._inhibit_required and not self._inhibit_proof_is_current():
            self._stop_request_pending = True
            return
        if self._last_hardware_io_receipt is None or self._last_write_in_flight is None:
            self._stop_request_pending = True
            return
        status_age = (self._last_hardware_io_age_at_receipt
                      + max(0.0, time.monotonic() - self._last_hardware_io_receipt))
        if (status_age > self.report_timeout_sec
                or self._last_write_in_flight != 0):
            reason = ('hardware status stale' if status_age > self.report_timeout_sec
                      else 'SDK write still in flight')
            if reason != self._last_stop_wait_reason:
                self.get_logger().warning(f'Controlled stop pending: {reason}')
                self._last_stop_wait_reason = reason
            self._stop_request_pending = True
            return
        report = self._latest_live_report
        report_at = self._latest_live_report_at
        if report is None:
            report = self.monitor.last_report
            report_at = self.monitor.last_report_at
        safe_report = can_request_controlled_stop(
            report, report_at, time.monotonic(), self.report_timeout_sec)
        if not safe_report:
            self.get_logger().error(
                'Controlled stop unconfirmed: no fresh healthy RobotMsg; no state service sent')
            self._stop_request_pending = True
            return
        if not self.stop_client.service_is_ready():
            self.get_logger().error(
                'Controlled stop unconfirmed: set_state service is unavailable')
            self._stop_request_pending = True
            return
        request = SetInt16.Request()
        request.data = 4
        try:
            self._stop_future = self.stop_client.call_async(request)
            self._stop_deadline = time.monotonic() + 0.5
            self._stop_ack_settled = False
            self.stop_requested = True
            self._stop_request_pending = False
            self._last_stop_wait_reason = None
            if self.stop_evidence is not None:
                self.stop_evidence.mark_requested()
            self.recorder.record('stop_request', 4)
        except Exception as error:
            self.get_logger().error(f'Controlled stop unconfirmed: {error}')
            self._stop_request_pending = True

    def _inhibit_proof_is_current(self):
        if not self._inhibit_required or self._last_hardware_io_receipt is None:
            return False
        age = (self._last_hardware_io_age_at_receipt
               + max(0.0, time.monotonic() - self._last_hardware_io_receipt))
        status = dict(self.last_hardware_io_status or ())
        current = inhibit_status_confirms_request(
            hardware_active=status.get('hardware_active', False),
            runtime_actions_enabled=status.get('runtime_actions_enabled', False),
            status_epoch=status.get('epoch', -1),
            inhibit_latched=status.get('inhibit_latched', False),
            inhibit_epoch=status.get('inhibit_epoch', -1),
            inhibit_request_id=status.get('inhibit_request_id', 0),
            requested_epoch=self._inhibit_epoch,
            previous_request_id=self._inhibit_previous_request_id,
            status_age_sec=age if age is not None else float('inf'),
            write_in_flight=status.get('write_in_flight', -1),
            freshness_sec=self.report_timeout_sec)
        self._inhibit_status_confirmed = current
        return current

    @staticmethod
    def _remove_pending_request(client, future):
        if client is None or future is None:
            return
        try:
            client.remove_pending_request(future)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def settle_stop_request(self):
        """Finish the bounded stop acknowledgement while ROS callbacks still run."""
        future = self._stop_future
        done, accepted, detail = stop_future_result(
            future, time.monotonic(), self._stop_deadline or 0.0)
        if not done:
            return False
        if future is None:
            return True
        if accepted:
            self.stop_acknowledged = True
            if self.stop_evidence is not None:
                self.stop_evidence.mark_acknowledged(True)
            self.recorder.record('stop_ack', True)
            self._remove_pending_request(self.stop_client, future)
            self._stop_future = None
            self.get_logger().warning(
                'Controlled stop request acknowledged (state=4); physical standstill unconfirmed')
            return True
        if detail == 'set_state acknowledgement timed out':
            self.get_logger().error(
                'Controlled stop unconfirmed: set_state acknowledgement timed out')
            if self.stop_evidence is not None:
                self.stop_evidence.mark_acknowledged(False)
            self.recorder.record('stop_ack_timeout')
            self._remove_pending_request(self.stop_client, future)
            self._stop_future = None
            return True
        if detail:
            self.get_logger().error(f'Controlled stop unconfirmed: {detail}')
            if self.stop_evidence is not None:
                self.stop_evidence.mark_acknowledged(False)
            self.recorder.record('stop_ack', False, detail)
            self._remove_pending_request(self.stop_client, future)
            self._stop_future = None
            return True
        return True

    def close(self):
        """Stop or terminate the bounded SDK child and release its queue."""
        self._stop_event.set()
        if self._reader.is_alive():
            self._reader.join(timeout=1.5)
        if self._reader.is_alive():
            self._reader.terminate()
            self._reader.join(timeout=0.25)
        if self._reader.is_alive() and hasattr(self._reader, 'kill'):
            self._reader.kill()
            self._reader.join(timeout=0.25)
        self._output.close()
        self._output.cancel_join_thread()


def main(args=None):
    """Keep the supervisor alive until clean shutdown or a detected fault."""
    rclpy.init(args=args)
    node = None
    exit_code = 0
    try:
        node = HardwareMonitorNode()
        while rclpy.ok() and (node.fault is None or not (
                node._stop_window_done and node._stop_ack_settled)):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception as error:
        print(f'hardware monitor failed: {error}', file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if node is not None:
            if node.fault:
                exit_code = 1
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
