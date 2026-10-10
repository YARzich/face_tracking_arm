#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Run a read-only xArm status monitor and exit nonzero on detected faults."""

import json
import math
import multiprocessing
import queue
import signal
import sys
import time

from face_tracking_bringup.hardware_monitor import (
    can_request_controlled_stop, HardwareMonitorFault, HardwareStateMonitor,
    report_stamp_problem, robot_report)
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
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
        self.declare_parameter('preflight_snapshot_json', '')
        self.declare_parameter('startup_timeout_sec', 20.0)
        self.declare_parameter('report_timeout_sec', 0.5)
        self.declare_parameter('report_stamp_timeout_sec', 0.5)
        self.declare_parameter('report_future_tolerance_sec', 0.25)
        self.declare_parameter('sdk_request_timeout_sec', 0.5)
        self.declare_parameter('sdk_state_timeout_sec', 0.75)
        self.declare_parameter('sdk_poll_period_sec', 0.5)
        self.declare_parameter('set_state_service', '/xarm/set_state')
        self.robot_ip = self.get_parameter('robot_ip').value
        if not self.robot_ip:
            raise ValueError('robot_ip is required for hardware monitoring')
        preflight_json = self.get_parameter('preflight_snapshot_json').value
        if not preflight_json:
            raise ValueError('preflight_snapshot_json is required for hardware monitoring')
        preflight = json.loads(preflight_json)
        self.report_timeout_sec = float(self.get_parameter('report_timeout_sec').value)
        self.report_stamp_timeout_sec = float(
            self.get_parameter('report_stamp_timeout_sec').value)
        self.report_future_tolerance_sec = float(
            self.get_parameter('report_future_tolerance_sec').value)
        self.sdk_request_timeout_sec = float(
            self.get_parameter('sdk_request_timeout_sec').value)
        self.sdk_state_timeout_sec = float(self.get_parameter('sdk_state_timeout_sec').value)
        self.sdk_poll_period_sec = float(self.get_parameter('sdk_poll_period_sec').value)
        if (not math.isfinite(self.sdk_request_timeout_sec)
                or not 0 < self.sdk_request_timeout_sec <= 0.5
                or not math.isfinite(self.sdk_poll_period_sec)
                or self.sdk_poll_period_sec != 0.5
                or not math.isfinite(self.report_timeout_sec)
                or not 0 < self.report_timeout_sec <= 0.5
                or not math.isfinite(self.report_stamp_timeout_sec)
                or not 0 < self.report_stamp_timeout_sec <= 0.5
                or not math.isfinite(self.report_future_tolerance_sec)
                or self.report_future_tolerance_sec <= 0):
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
        self.subscription = self.create_subscription(
            RobotMsg, topic, self._on_robot_report, state_qos)
        self.stop_client = self.create_client(
            SetInt16, self.get_parameter('set_state_service').value)
        self._stop_future = None
        self._stop_deadline = None
        self.stop_confirmed = False
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
        self.timer = self.create_timer(0.05, self._poll_health)
        self.get_logger().info(f'Monitoring {topic}; arming after matched SERVOJ and SDK state')

    def _publish_ready(self, value):
        message = Bool()
        message.data = value
        self.ready_pub.publish(message)

    def _on_robot_report(self, message):
        try:
            stamp = message.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            problem = report_stamp_problem(
                stamp_ns, self.get_clock().now().nanoseconds,
                self.report_stamp_timeout_sec, self.report_future_tolerance_sec)
            if problem is not None:
                if self.monitor.ready:
                    self._set_fault(problem)
                return
            report = robot_report(message)
            self.monitor.observe_report(
                report, time.monotonic(), source_stamp_ns=stamp_ns)
            if report.warning != self._last_warning:
                if report.warning:
                    self.get_logger().warning(f'Robot warning code {report.warning}')
                self._last_warning = report.warning
        except HardwareMonitorFault as error:
            self._set_fault(str(error))
        except ValueError as error:
            self._set_fault(str(error))

    def _poll_health(self):
        if self.fault is not None:
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

    def _drain_sdk_results(self):
        while True:
            try:
                kind, payload = self._output.get_nowait()
            except queue.Empty:
                break
            if kind == 'snapshot':
                settings, received_at = payload
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

    def _set_fault(self, reason):
        if self.fault is not None:
            return
        self.fault = reason
        self.get_logger().fatal(reason)
        self._publish_ready(False)
        self._request_stop_if_safe()

    def _request_stop_if_safe(self):
        """Request only state=4 when the last robot report supports a safe attempt."""
        report = self.monitor.last_report
        report_at = self.monitor.last_report_at
        safe_report = can_request_controlled_stop(
            report, report_at, time.monotonic(), self.report_timeout_sec)
        if not safe_report:
            self.get_logger().error(
                'Controlled stop unconfirmed: no fresh healthy RobotMsg; no state service sent')
            return
        if not self.stop_client.service_is_ready():
            self.get_logger().error(
                'Controlled stop unconfirmed: set_state service is unavailable')
            return
        request = SetInt16.Request()
        request.data = 4
        try:
            self._stop_future = self.stop_client.call_async(request)
            self._stop_deadline = time.monotonic() + 0.5
        except Exception as error:
            self.get_logger().error(f'Controlled stop unconfirmed: {error}')

    def settle_stop_request(self):
        """Finish the bounded stop acknowledgement while ROS callbacks still run."""
        future = self._stop_future
        if future is None:
            return True
        if future.done():
            try:
                response = future.result()
                if response is not None and response.ret == 0:
                    self.stop_confirmed = True
                    self.get_logger().warning('Controlled stop request acknowledged (state=4)')
                    return True
                detail = 'empty response' if response is None else str(response.message)
            except Exception as error:
                detail = str(error)
            self.get_logger().error(f'Controlled stop unconfirmed: {detail}')
            self._stop_future = None
            return True
        if time.monotonic() >= self._stop_deadline:
            self.get_logger().error(
                'Controlled stop unconfirmed: set_state acknowledgement timed out')
            self._stop_future = None
            return True
        return False

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
        while rclpy.ok() and (node.fault is None or not node.settle_stop_request()):
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
