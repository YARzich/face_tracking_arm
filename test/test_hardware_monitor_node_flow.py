# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Exercise the monitor's own fault-to-STOP methods without ROS services."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))


def _install_message_stubs():
    arm_package = ModuleType('face_tracking_arm')
    arm_package.__path__ = []
    arm_msgs = ModuleType('face_tracking_arm.msg')

    class HardwareIoStatus:
        pass

    class ServoExecutionContext:
        pass

    class FaceTransformProvenance:
        pass

    class ServoTrajectoryPublication:
        pass

    class TrackingTarget:
        FACE = 0
        REST = 2

    arm_msgs.HardwareIoStatus = HardwareIoStatus
    arm_msgs.ServoExecutionContext = ServoExecutionContext
    arm_msgs.FaceTransformProvenance = FaceTransformProvenance
    arm_msgs.ServoTrajectoryPublication = ServoTrajectoryPublication
    arm_msgs.TrackingTarget = TrackingTarget
    arm_package.msg = arm_msgs
    sys.modules.setdefault('face_tracking_arm', arm_package)
    sys.modules.setdefault('face_tracking_arm.msg', arm_msgs)

    xarm_package = ModuleType('xarm_msgs')
    xarm_package.__path__ = []
    xarm_msgs = ModuleType('xarm_msgs.msg')
    xarm_srv = ModuleType('xarm_msgs.srv')

    class RobotMsg:
        pass

    class SetInt16:
        class Request:
            def __init__(self):
                self.data = 0

    xarm_msgs.RobotMsg = RobotMsg
    xarm_srv.SetInt16 = SetInt16
    xarm_package.msg = xarm_msgs
    xarm_package.srv = xarm_srv
    sys.modules.setdefault('xarm_msgs', xarm_package)
    sys.modules.setdefault('xarm_msgs.msg', xarm_msgs)
    sys.modules.setdefault('xarm_msgs.srv', xarm_srv)


_install_message_stubs()
_NODE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'hardware_monitor_node.py'
_SPEC = importlib.util.spec_from_file_location('hardware_monitor_node_test_module', _NODE_PATH)
_NODE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NODE_MODULE)
HardwareMonitorNode = _NODE_MODULE.HardwareMonitorNode


class _FakeFuture:
    def __init__(self, response=None, done=False):
        self._response = response
        self._done = done

    def done(self):
        return self._done

    def result(self):
        return self._response


class _FakeClient:
    def __init__(self, future=None, ready=True):
        self.future = future or _FakeFuture()
        self.ready = ready
        self.requests = []
        self.removed = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return self.future

    def remove_pending_request(self, future):
        self.removed.append(future)


class _Recorder:
    def __init__(self):
        self.events = []

    def record(self, kind, *payload):
        self.events.append((kind, payload))
        return True

    def latch_fault(self, *payload):
        self.events.append(('fault', payload))
        return True


class _Logger:
    def __init__(self):
        self.messages = []

    def fatal(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)


def _node(*, stop_ready=True):
    node = SimpleNamespace()
    now = time.monotonic()
    node.fault = None
    node.stop_requested = False
    node.stop_acknowledged = False
    node._fault_context = None
    node._fault_recorded = False
    node._stop_observation_deadline = now + 5.0
    node._stop_request_pending = False
    node._last_stop_wait_reason = None
    node._inhibit_required = False
    node._inhibit_status_confirmed = False
    node._inhibit_request_pending = False
    node._inhibit_future = None
    node._inhibit_deadline = None
    node._inhibit_retry_at = 0.0
    node._inhibit_epoch = None
    node._inhibit_previous_request_id = None
    node._inhibit_ack_recorded = False
    node._last_hardware_io_receipt = now
    node._last_hardware_io_age_at_receipt = 0.01
    node._last_write_in_flight = 0
    node._last_stop_report_signature = None
    node.stop_evidence = None
    node.report_timeout_sec = 0.5
    node._latest_live_report = SimpleNamespace(
        mode=1, state=1, error=0, warning=0, mt_able=0x3F, mt_brake=0x3F)
    node._latest_live_report_at = now
    node.monitor = SimpleNamespace(
        last_report=node._latest_live_report, last_report_at=now,
        fault_report=None, expected_serial='mock')
    node.last_report_details = ('state', '1')
    node.last_hardware_io_status = (
        ('hardware_active', True), ('runtime_actions_enabled', True),
        ('epoch', 7), ('inhibit_latched', False), ('inhibit_epoch', 0),
        ('inhibit_request_id', 10), ('write_in_flight', 0))
    node.recorder = _Recorder()
    node._inhibit_future = None
    node._stop_future = None
    node._stop_deadline = None
    node._stop_ack_settled = False
    node._last_stop_wait_reason = None
    node._stop_request_expired_recorded = False
    node._stop_window_done = False
    node.stop_measured = False
    node._stop_ack_settled = True
    node._stop_observation_timeout_sec = 5.0
    node.stop_observation_timeout_sec = 5.0
    node._fault_dump_path = None
    node.hardware_inhibit_service = '/xarm/_inhibit_servoj'
    node.inhibit_client = _FakeClient(
        _FakeFuture(SimpleNamespace(success=True, message='latched'), done=True))
    node.stop_client = _FakeClient(ready=stop_ready)
    node.get_logger = lambda: _Logger()
    node._publish_ready = lambda value: None
    node._ensure_fault_recorded = MethodType(
        HardwareMonitorNode._ensure_fault_recorded, node)
    for method_name in (
            '_begin_inhibit_request', '_progress_inhibit_request',
            '_observe_inhibit_status_message', '_request_stop_if_safe',
            '_inhibit_proof_is_current', '_poll_health', 'settle_stop_request'):
        setattr(node, method_name, MethodType(
            getattr(HardwareMonitorNode, method_name), node))
    node._remove_pending_request = HardwareMonitorNode._remove_pending_request
    node._expire_measured_stop_if_stale = lambda: None
    return node


def _set_status(node, *, epoch=7, latched=True, inhibit_epoch=7,
                request_id=11, in_flight=0, age=0.01):
    node._last_hardware_io_receipt = time.monotonic()
    node._last_hardware_io_age_at_receipt = age
    node._last_write_in_flight = in_flight
    node.last_hardware_io_status = (
        ('hardware_active', True), ('runtime_actions_enabled', True),
        ('epoch', epoch), ('inhibit_latched', latched),
        ('inhibit_epoch', inhibit_epoch), ('inhibit_request_id', request_id),
        ('write_in_flight', in_flight))


def test_own_fault_waits_for_trigger_ack_matching_status_and_write_drain():
    node = _node()
    HardwareMonitorNode._set_fault(node, 'reduced profile changed')
    assert node.fault == 'reduced profile changed'
    assert len(node.inhibit_client.requests) == 1
    assert not node.stop_client.requests

    # Trigger ACK only says the service handled the request; no status, no STOP.
    HardwareMonitorNode._progress_inhibit_request(node, time.monotonic())
    assert not node.stop_client.requests

    _set_status(node, latched=False, request_id=10)
    HardwareMonitorNode._observe_inhibit_status_message(
        node, dict(node.last_hardware_io_status), 0.01)
    assert not node.stop_client.requests

    _set_status(node, in_flight=1)
    HardwareMonitorNode._observe_inhibit_status_message(
        node, dict(node.last_hardware_io_status), 0.01)
    assert not node.stop_client.requests

    _set_status(node, in_flight=0)
    HardwareMonitorNode._observe_inhibit_status_message(
        node, dict(node.last_hardware_io_status), 0.01)
    assert len(node.stop_client.requests) == 1
    assert node.stop_client.requests[0].data == 4


def test_driver_fault_to_stop_ack_report_and_measured_standstill_chain():
    """Exercise the complete mock status→inhibit→STOP evidence sequence."""
    from face_tracking_bringup.hardware_monitor import RobotReport, StopEvidence

    gate_binary = Path(os.environ['HARDWARE_IO_WRITE_GATE_TRACE_BIN'])
    trace = subprocess.run(
        [str(gate_binary), '--emit-cross-layer-trace'], check=True,
        capture_output=True, text=True)
    events = [json.loads(line) for line in trace.stdout.splitlines() if line.strip()]
    assert [event['event'] for event in events] == [
        'write_admitted', 'read_fault_latched', 'admitted_write_drained']

    node = _node()
    node.stop_evidence = StopEvidence(
        stable_samples=3, velocity_tolerance=0.01, position_tolerance=0.001,
        max_feedback_age_sec=0.5)
    node.stop_client.future = _FakeFuture(
        SimpleNamespace(ret=0, message='state accepted'), done=True)

    fault_status = events[1]
    assert fault_status['write_in_flight'] == 1
    HardwareMonitorNode._set_fault(node, 'driver read fault')
    HardwareMonitorNode._progress_inhibit_request(node, time.monotonic())
    node._last_hardware_io_receipt = time.monotonic()
    node._last_hardware_io_age_at_receipt = 0.01
    node._last_write_in_flight = fault_status['write_in_flight']
    node.last_hardware_io_status = tuple(fault_status.items())
    HardwareMonitorNode._observe_inhibit_status_message(
        node, fault_status, 0.01)
    assert not node.stop_client.requests

    # The same gate trace reports the SDK call drained; now STOP may be requested.
    drained_status = events[2]
    assert drained_status['write_in_flight'] == 0
    node._last_hardware_io_receipt = time.monotonic()
    node._last_hardware_io_age_at_receipt = 0.01
    node._last_write_in_flight = drained_status['write_in_flight']
    node.last_hardware_io_status = tuple(drained_status.items())
    HardwareMonitorNode._observe_inhibit_status_message(
        node, drained_status, 0.01)
    assert len(node.stop_client.requests) == 1
    assert node.stop_client.requests[0].data == 4

    assert HardwareMonitorNode.settle_stop_request(node)
    assert node.stop_acknowledged
    assert node.stop_evidence.acknowledged
    assert not node.stop_evidence.stop_reported
    assert not node.stop_evidence.measured_stopped

    report = RobotReport(mode=1, state=4, mt_able=0x3F, mt_brake=0x3F,
                         error=0, warning=0)
    assert node.stop_evidence.observe_report(report)
    position = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
    velocity = (0.0,) * 6
    for sequence in (1, 2):
        assert not node.stop_evidence.observe_feedback(
            position, velocity, read_ok=True, read_sequence=sequence, epoch=7,
            read_age_sec=0.01, transport_age_sec=0.01)
    assert node.stop_evidence.observe_feedback(
        position, velocity, read_ok=True, read_sequence=3, epoch=7,
        read_age_sec=0.01, transport_age_sec=0.01)
    assert node.stop_evidence.measured_stopped


def test_driver_fault_diagnostics_do_not_start_new_stop_when_gate_is_disabled():
    """A diagnostic-only first fault must not initiate automatic STOP."""
    node = _node()
    node.fault = None
    node.stop_evidence = None
    node._last_hardware_fault_key = None
    node._last_inhibit_key = None
    node._record_context_join = lambda _key: None
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=2_000_000_000))
    node._set_fault = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('gate-off driver fault must remain diagnostic only'))
    node._request_stop_if_safe = lambda: (_ for _ in ()).throw(
        AssertionError('gate-off driver fault must not initiate STOP'))

    message = SimpleNamespace(
        epoch=7, read_sequence=2, write_in_flight=0,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=900_000_000)),
        robot_msg_stamp=SimpleNamespace(sec=1, nanosec=900_000_000),
        robot_msg_age_us=10_000, robot_msg_sequence=3,
        robot_mode=1, robot_state=1, robot_mt_able=0x3F, robot_mt_brake=0x3F,
        robot_error_code=0, robot_warn_code=0,
        fault_latched=True, fault_epoch=7, fault_sequence=4,
        fault_kind=2, fault_code=-9, fault_read_attempt_sequence=2,
        fault_stamp=SimpleNamespace(sec=1, nanosec=800_000_000),
        runtime_actions_enabled=False, inhibit_latched=False,
        inhibit_epoch=0, inhibit_request_id=0)
    for name in ('source_kind', 'hardware_active', 'read_attempt_sequence', 'read_epoch',
                 'read_ok', 'read_code', 'read_stamp', 'read_age_us', 'q', 'dq',
                 'write_sequence', 'write_stamp', 'driver_command_sequence',
                 'write_attempted', 'write_ok', 'write_code', 'command_double',
                 'command_float', 'candidate_sequence', 'candidate_double',
                 'candidate_float', 'candidate_valid',
                 'candidate_failure_joint_index', 'fault_candidate_valid',
                 'fault_candidate_joint_index', 'fault_candidate_double',
                 'fault_candidate_float'):
        setattr(message, name, 0 if name.endswith(('sequence', 'code', 'age_us', 'epoch'))
                else False if name.endswith(('valid', 'ok', 'attempted'))
                else [0.0] * 6 if name.endswith(('double', 'float', 'q', 'dq'))
                else SimpleNamespace(sec=0, nanosec=0))
    message.get_fields_and_field_types = lambda: {
        name: 'field' for name in vars(message) if not name.startswith('_')}

    HardwareMonitorNode._on_hardware_io_status(node, message)
    assert node.fault is None
    assert node.recorder.events[-1][0] == 'hardware_io_status'
    assert any(event[0] == 'hardware_io_fault' for event in node.recorder.events)
    assert not node.stop_client.requests


def test_epoch_switch_or_lost_proof_while_stop_service_waits_blocks_stop():
    """A changed status epoch invalidates earlier inhibit proof."""
    node = _node(stop_ready=False)
    HardwareMonitorNode._set_fault(node, 'reduced profile changed')
    node._inhibit_future._done = True
    HardwareMonitorNode._progress_inhibit_request(node, time.monotonic())
    _set_status(node)
    HardwareMonitorNode._observe_inhibit_status_message(
        node, dict(node.last_hardware_io_status), 0.01)
    assert not node.stop_client.requests
    assert node._inhibit_status_confirmed

    _set_status(node, epoch=8, inhibit_epoch=7, request_id=11)
    HardwareMonitorNode._request_stop_if_safe(node)
    assert not node.stop_client.requests
    assert not node._inhibit_status_confirmed

    _set_status(node, epoch=7, latched=False, inhibit_epoch=0, request_id=11)
    HardwareMonitorNode._request_stop_if_safe(node)
    assert not node.stop_client.requests


def test_poll_health_rechecks_previously_confirmed_inhibit_before_stop():
    node = _node(stop_ready=False)
    HardwareMonitorNode._set_fault(node, 'reduced profile changed')
    _set_status(node)
    HardwareMonitorNode._observe_inhibit_status_message(
        node, dict(node.last_hardware_io_status), 0.01)
    assert node._inhibit_status_confirmed
    assert not node.stop_client.requests

    # The old receipt has aged out without a callback; the poll path must not
    # treat its remembered confirmation as authorization to send STOP.
    node._last_hardware_io_receipt -= 1.0
    HardwareMonitorNode._poll_health(node)
    assert not node.stop_client.requests
    assert not node._inhibit_status_confirmed


def test_service_timeouts_remove_pending_requests_from_rclpy_clients():
    node = _node()
    inhibit_future = _FakeFuture(done=False)
    node._inhibit_future = inhibit_future
    node._inhibit_deadline = time.monotonic() - 0.1
    HardwareMonitorNode._progress_inhibit_request(node, time.monotonic())
    assert node.inhibit_client.removed == [inhibit_future]
    assert node._inhibit_future is None

    stop_future = _FakeFuture(done=False)
    node._stop_future = stop_future
    node._stop_deadline = time.monotonic() - 0.1
    assert HardwareMonitorNode.settle_stop_request(node)
    assert node.stop_client.removed == [stop_future]
    assert node._stop_future is None
    assert node.stop_acknowledged is False


def test_face_tf_callback_joins_exact_source_stamp_to_execution_context_once():
    from collections import deque

    from face_tracking_bringup.hardware_fault_recorder import BoundedContextHistory

    class Recorder:
        def __init__(self):
            self.events = []

        def record(self, kind, *payload):
            self.events.append((kind, payload))
            return True

    node = SimpleNamespace(
        execution_context_history=BoundedContextHistory(capacity=4),
        face_transform_history=BoundedContextHistory(capacity=4),
        _face_transform_seen=set(), _face_transform_seen_order=deque(maxlen=4),
        _face_transform_joined=set(), _face_transform_join_order=deque(maxlen=4),
        last_execution_context=(('tick_sequence', 8), ('face_stamp', (7, 11))),
        _current_face_transform_provenance=None,
        _current_face_transform_used_for_reference=False,
        recorder=Recorder())
    node._join_face_transform_with_context = lambda receipt, context, provenance=None: (
        HardwareMonitorNode._join_face_transform_with_context(
            node, receipt, context, provenance))
    now = time.monotonic()
    context = (('face_stamp', (7, 11)), ('tick_sequence', 8),
               ('hardware_epoch', 3), ('hardware_read_sequence', 90),
               ('target_mode', _NODE_MODULE.TrackingTarget.FACE),
               ('target_available', True), ('target_used_for_reference', True))
    assert node.execution_context_history.record(7000000011, context, now)
    stamp = SimpleNamespace(sec=7, nanosec=11)

    def point(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    def transform(parent, child):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id=parent), child_frame_id=child,
            transform=SimpleNamespace(translation=point(0, 0, 0),
                                      rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))

    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='camera'), accepted_face_sequence=42,
        source_face=point(1, 2, 3), planning_face=point(1, 2, 3),
        planning_from_face=transform('world', 'camera'),
        planning_from_base=transform('world', 'base'),
        planning_from_reach_center=transform('world', 'reach'),
        planning_base=point(0, 0, 0), planning_reach_center=point(0, 0, 0),
        face_transform_lookup_used=True, base_transform_lookup_used=True,
        reach_center_transform_lookup_used=True)
    HardwareMonitorNode._on_face_transform_provenance(node, message)
    HardwareMonitorNode._on_face_transform_provenance(node, message)
    joins = [event for event in node.recorder.events
             if event[0] == 'face_transform_execution_context_join']
    assert len(joins) == 1
    assert joins[0][1][1] == (42, 7000000011, 8)
    assert joins[0][1][2] == 7000000011
    assert node._current_face_transform_provenance is not None
    assert node._current_face_transform_used_for_reference
    node._current_face_transform_provenance = None
    HardwareMonitorNode._join_face_transform_with_context(
        node, now, (('face_stamp', (7, 12)), ('tick_sequence', 9)), message.__dict__)
    assert node._current_face_transform_provenance is None


def test_face_tf_join_distinguishes_exact_source_match_from_actual_face_use():
    from collections import deque

    from face_tracking_bringup.hardware_fault_recorder import BoundedContextHistory

    class Recorder:
        def __init__(self):
            self.events = []

        def record(self, kind, *payload):
            self.events.append((kind, payload))
            return True

    def point(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    stamp = SimpleNamespace(sec=7, nanosec=11)

    def transform(parent, child):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id=parent), child_frame_id=child,
            transform=SimpleNamespace(translation=point(0, 0, 0),
                                      rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))

    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='camera'), accepted_face_sequence=42,
        source_face=point(1, 2, 3), planning_face=point(1, 2, 3),
        planning_from_face=transform('world', 'camera'),
        planning_from_base=transform('world', 'base'),
        planning_from_reach_center=transform('world', 'reach'),
        planning_base=point(0, 0, 0), planning_reach_center=point(0, 0, 0),
        face_transform_lookup_used=True, base_transform_lookup_used=True,
        reach_center_transform_lookup_used=True)

    scenarios = (
        (_NODE_MODULE.TrackingTarget.FACE, True, True, True),
        (_NODE_MODULE.TrackingTarget.FACE, False, False, False),
        (_NODE_MODULE.TrackingTarget.REST, True, False, False),
    )
    for target_mode, target_available, target_used, expected_used in scenarios:
        for provenance_first in (False, True):
            now = time.monotonic()
            context = (
                ('face_stamp', (7, 11)), ('tick_sequence', 8), ('hardware_epoch', 3),
                ('hardware_read_sequence', 90), ('target_mode', target_mode),
                ('target_available', target_available),
                ('target_used_for_reference', target_used))
            node = SimpleNamespace(
                execution_context_history=BoundedContextHistory(capacity=4),
                face_transform_history=BoundedContextHistory(capacity=4),
                _face_transform_seen=set(), _face_transform_seen_order=deque(maxlen=4),
                _face_transform_joined=set(), _face_transform_join_order=deque(maxlen=4),
                last_execution_context=context,
                _current_face_transform_provenance=None,
                _current_face_transform_used_for_reference=False,
                recorder=Recorder())
            node._join_face_transform_with_context = (
                lambda receipt, context_details, provenance=None:
                HardwareMonitorNode._join_face_transform_with_context(
                    node, receipt, context_details, provenance))
            if provenance_first:
                HardwareMonitorNode._on_face_transform_provenance(node, message)
                context_receipt = time.monotonic()
                assert node.execution_context_history.record(
                    7000000011, context, context_receipt)
                HardwareMonitorNode._join_face_transform_with_context(
                    node, context_receipt, context)
            else:
                assert node.execution_context_history.record(7000000011, context, now)
                HardwareMonitorNode._on_face_transform_provenance(node, message)
            assert node._current_face_transform_provenance is not None
            assert node._current_face_transform_used_for_reference is expected_used
            assert len([event for event in node.recorder.events
                        if event[0] == 'face_transform_provenance']) == 1
            joins = [event for event in node.recorder.events
                     if event[0] == 'face_transform_execution_context_join']
            assert len(joins) == 1
            assert joins[0][1][3:] == (target_mode, target_available, target_used)


def test_trajectory_publication_callback_records_full_payload_once_and_joins_ids():
    from collections import deque

    from face_tracking_bringup.hardware_fault_recorder import (
        BoundedContextHistory, BoundedPublicationPayloadCache)

    class Recorder:
        def __init__(self):
            self.events = []

        def record(self, kind, *payload):
            self.events.append((kind, payload))
            return True

    node = SimpleNamespace(
        execution_context_history=BoundedContextHistory(capacity=4),
        publication_payload_cache=BoundedPublicationPayloadCache(capacity=2),
        _trajectory_publications_seen=set(),
        _trajectory_publications_seen_order=deque(maxlen=4),
        _last_publication_drop_count=None,
        _trajectory_publication_joined=set(),
        _trajectory_publication_join_order=deque(maxlen=8),
        recorder=Recorder())
    node._join_trajectory_publications_with_context = (
        lambda receipt, context, publication=None:
        HardwareMonitorNode._join_trajectory_publications_with_context(
            node, receipt, context, publication))
    now = time.monotonic()
    context = (('hardware_epoch', 3), ('tick_sequence', 8),
               ('active_publication_id', 42), ('latest_publication_id', 43),
               ('braking_tail_publication_id', 42))
    assert node.execution_context_history.record(900, context, now)
    stamp = SimpleNamespace(sec=9, nanosec=0)
    duration = SimpleNamespace(sec=0, nanosec=20_000_000)
    point = SimpleNamespace(positions=[0.1] * 6, velocities=[0.0] * 6,
                            accelerations=[0.0] * 6, effort=[], time_from_start=duration)
    trajectory = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='base'),
        joint_names=[f'joint{index}' for index in range(1, 7)], points=[point])
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='publication'),
        publication_kind=1, publication_id=42,
        stationary_time=stamp, tail_start_index=5, hardware_epoch=3,
        hardware_read_sequence=90, control_tick_sequence=8,
        dropped_publication_events=2, trajectory=trajectory)
    HardwareMonitorNode._on_servo_trajectory_publication(node, message)
    HardwareMonitorNode._on_servo_trajectory_publication(node, message)
    publications = [event for event in node.recorder.events
                    if event[0] == 'servo_trajectory_publication']
    joins = [event for event in node.recorder.events
             if event[0] == 'servo_trajectory_execution_context_join']
    overflows = [event for event in node.recorder.events
                 if event[0] == 'servo_trajectory_publication_overflow']
    assert len(publications) == 1
    assert len(joins) == 1
    assert len(overflows) == 1
    details = dict(publications[0][1][1])
    assert details['publication_id'] == 42
    assert len(details['points']) == 1
    assert len(details['points'][0][0]) == 6
    assert joins[0][1][1] == (3, 42, 8)
    assert joins[0][1][2] == 42
    assert len(joins[0][1]) == 3
