# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Read-only hardware-state interlock for an activated xArm controller."""

from dataclasses import dataclass
import math
import time


class HardwareMonitorFault(RuntimeError):
    """Raised when controller state or reduced safety settings change."""


@dataclass(frozen=True)
class ReducedSettings:
    """Decoded SDK reduced-mode settings with TCP boundaries in mm and speed in mm/s."""

    mode: bool
    tcp_boundary_mm: tuple
    tcp_max_speed_mmps: float
    joint_max_speed: float
    joint_ranges: tuple
    safety_boundary_enabled: bool
    collision_rebound_enabled: bool


@dataclass(frozen=True)
class RobotReport:
    """Machine-readable status fields from xarm_msgs/RobotMsg."""

    mode: int
    state: int
    mt_able: int
    mt_brake: int
    error: int
    warning: int

    def fault_reasons(self, expected_mode=1, active_states=(0, 1, 2), axis_mask=0x3F):
        """Return every live fault represented by this report, preserving simultaneous causes."""
        reasons = []
        if self.error:
            reasons.append(f'error code {self.error}')
        if self.mode != expected_mode:
            reasons.append(f'mode changed from SERVOJ ({expected_mode}) to {self.mode}')
        if self.state not in active_states:
            reasons.append(f'non-motion state {self.state}')
        if self.mt_brake & axis_mask != axis_mask:
            reasons.append(f'brake mask changed to {self.mt_brake:#x}')
        if self.mt_able & axis_mask != axis_mask:
            reasons.append(f'servo-enable mask is incomplete ({self.mt_able:#x})')
        return tuple(reasons)


def report_stamp_problem(stamp_ns, now_ns, max_age_sec=0.5, future_tolerance_sec=0.25):
    """Return a concrete source-stamp fault for absent, old or future reports."""
    if stamp_ns <= 0:
        return 'Robot state report has no capture timestamp'
    age_sec = (now_ns - stamp_ns) / 1e9
    if age_sec > max_age_sec:
        return f'Robot state report timestamp is stale by {age_sec:.3f}s'
    if age_sec < -future_tolerance_sec:
        return f'Robot state report timestamp is future-dated by {-age_sec:.3f}s'
    return None


def reduced_settings(values):
    """Validate and decode ``UxbusCmdTcp.get_reduced_states`` payload data."""
    if len(values) != 7:
        raise ValueError('Incomplete SDK reduced-state snapshot')
    (mode, tcp_boundary_mm, tcp_max_speed_mmps, joint_max_speed, joint_ranges,
     fence, rebound) = values
    if mode not in (0, 1) or len(tcp_boundary_mm) != 6 or len(joint_ranges) != 14:
        raise ValueError('Malformed SDK reduced-state snapshot')
    numbers = [*tcp_boundary_mm, tcp_max_speed_mmps, joint_max_speed, *joint_ranges]
    if not all(math.isfinite(float(value)) for value in numbers):
        raise ValueError('Non-finite SDK reduced-state value')
    if not all(value in (0, 1) for value in (fence, rebound)):
        raise ValueError('Invalid SDK reduced-state safety flags')
    return ReducedSettings(
        bool(mode), tuple(float(value) for value in tcp_boundary_mm),
        float(tcp_max_speed_mmps), float(joint_max_speed),
        tuple(float(value) for value in joint_ranges), bool(fence), bool(rebound))


def robot_report(message):
    """Extract the status fields used for the interlock from RobotMsg."""
    try:
        report = RobotReport(
            mode=int(message.mode), state=int(message.state),
            mt_able=int(message.mt_able), mt_brake=int(message.mt_brake),
            error=int(message.err), warning=int(message.warn))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('RobotMsg is missing valid machine status fields') from error
    if report.mode < 0 or report.state < 0 or report.mt_able < 0 or report.mt_brake < 0:
        raise ValueError('RobotMsg contains a negative status value')
    return report


def robot_report_details(message):
    """Capture every declared ROS report field in a bounded diagnostic representation."""
    fields = getattr(message, 'get_fields_and_field_types', None)
    if callable(fields):
        names = tuple(fields())
    else:
        names = ('mode', 'state', 'mt_able', 'mt_brake', 'err', 'warn')
    details = []
    for name in names:
        try:
            value = str(getattr(message, name))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            value = '<unavailable>'
        details.append((str(name), value[:256]))
    return tuple(details)


def hardware_io_status_details(message):
    """Capture a whole fixed-size hardware provenance snapshot for the fault recorder."""
    fields = getattr(message, 'get_fields_and_field_types', None)
    if not callable(fields):
        raise ValueError('HardwareIoStatus does not expose its declared fields')
    details = []
    for name in fields():
        value = getattr(message, name)
        if name in (
                'q', 'dq', 'command_double', 'command_float', 'candidate_double',
                'candidate_float', 'fault_candidate_double', 'fault_candidate_float'):
            value = tuple(value)
        elif name in ('header', 'read_stamp', 'write_stamp', 'robot_msg_stamp',
                      'fault_stamp'):
            stamp = value.stamp if name == 'header' else value
            value = (int(stamp.sec), int(stamp.nanosec))
        details.append((str(name), value))
    return tuple(details)


def servo_execution_context_details(message):
    """Decode bounded executor provenance while preserving its own clock stamps."""
    def stamp(value):
        return int(value.sec), int(value.nanosec)

    def point(value):
        return float(value.x), float(value.y), float(value.z)

    transform = message.target_transform
    return (
        ('header', stamp(message.header.stamp)),
        ('tick_sequence', int(message.tick_sequence)),
        ('hardware_epoch', int(message.hardware_epoch)),
        ('hardware_read_sequence', int(message.hardware_read_sequence)),
        ('target_available', bool(message.target_available)),
        ('target_used_for_reference', bool(message.target_used_for_reference)),
        ('target_mode', int(message.target_mode)),
        ('target_id', int(message.target_id)),
        ('target_source_stamp', stamp(message.target_source_stamp)),
        ('face_stamp', stamp(message.face_stamp)),
        ('target_frame', str(message.target_frame)),
        ('target_position', point(message.target_position)),
        ('face_position', point(message.face_position)),
        ('tf_provenance_available', bool(message.tf_provenance_available)),
        ('target_transform', (
            stamp(transform.header.stamp), str(transform.header.frame_id),
            str(transform.child_frame_id), point(transform.transform.translation),
            (float(transform.transform.rotation.x), float(transform.transform.rotation.y),
             float(transform.transform.rotation.z), float(transform.transform.rotation.w)))),
        ('active_publication_id', int(message.active_publication_id)),
        ('latest_publication_id', int(message.latest_publication_id)),
        ('braking_tail_publication_id', int(message.braking_tail_publication_id)),
        ('braking_tail_stationary_stamp', stamp(message.braking_tail_stationary_stamp)),
        ('dropped_context_records', int(message.dropped_context_records)),
        ('dropped_publication_events', int(message.dropped_publication_events)),
    )


def face_transform_provenance_details(message):
    """Decode the exact three TF results used to build one accepted face sample."""
    def stamp(value):
        return int(value.sec), int(value.nanosec)

    def point(value):
        return float(value.x), float(value.y), float(value.z)

    def transform(value):
        data = value.transform
        return (
            stamp(value.header.stamp), str(value.header.frame_id),
            str(value.child_frame_id), point(data.translation),
            (float(data.rotation.x), float(data.rotation.y),
             float(data.rotation.z), float(data.rotation.w)))

    return (
        ('source_stamp', stamp(message.header.stamp)),
        ('source_frame', str(message.header.frame_id)),
        ('accepted_face_sequence', int(message.accepted_face_sequence)),
        ('source_face', point(message.source_face)),
        ('planning_face', point(message.planning_face)),
        ('planning_from_face', transform(message.planning_from_face)),
        ('planning_from_base', transform(message.planning_from_base)),
        ('planning_from_reach_center', transform(message.planning_from_reach_center)),
        ('planning_base', point(message.planning_base)),
        ('planning_reach_center', point(message.planning_reach_center)),
        ('face_transform_lookup_used', bool(message.face_transform_lookup_used)),
        ('base_transform_lookup_used', bool(message.base_transform_lookup_used)),
        ('reach_center_transform_lookup_used',
         bool(message.reach_center_transform_lookup_used)),
    )


def servo_trajectory_publication_details(message):
    """Decode a complete immutable publication event without confusing IDs."""
    def stamp(value):
        return int(value.sec), int(value.nanosec)

    trajectory = message.trajectory
    points = tuple((
        tuple(float(value) for value in point.positions),
        tuple(float(value) for value in point.velocities),
        tuple(float(value) for value in point.accelerations),
        tuple(float(value) for value in point.effort),
        stamp(point.time_from_start)) for point in trajectory.points)
    return (
        ('event_stamp', stamp(message.header.stamp)),
        ('event_frame_id', str(message.header.frame_id)),
        ('publication_kind', int(message.publication_kind)),
        ('publication_id', int(message.publication_id)),
        ('stationary_time', stamp(message.stationary_time)),
        ('tail_start_index', int(message.tail_start_index)),
        ('hardware_epoch', int(message.hardware_epoch)),
        ('hardware_read_sequence', int(message.hardware_read_sequence)),
        ('control_tick_sequence', int(message.control_tick_sequence)),
        ('dropped_publication_events', int(message.dropped_publication_events)),
        ('trajectory_header_stamp', stamp(trajectory.header.stamp)),
        ('trajectory_frame_id', str(trajectory.header.frame_id)),
        ('joint_names', tuple(str(name) for name in trajectory.joint_names)),
        ('points', points),
    )


def can_request_controlled_stop(report, report_at, now, freshness_sec=0.5):
    """Return true only when a fresh report says the robot can accept a stop request."""
    return (report is not None and report_at is not None
            and now - report_at <= freshness_sec and report.state in (0, 1, 2)
            and report.error == 0 and report.mt_brake & 0x3F == 0x3F
            and report.mt_able & 0x3F == 0x3F)


def inhibit_status_confirms_request(
        *, hardware_active, runtime_actions_enabled, status_epoch,
        inhibit_latched, inhibit_epoch, inhibit_request_id,
        requested_epoch, previous_request_id, status_age_sec,
        write_in_flight, freshness_sec=0.5):
    """Require fresh same-epoch inhibit evidence and drained writes before STOP."""
    values = (status_age_sec, freshness_sec)
    if not all(math.isfinite(value) for value in values):
        return False
    try:
        return (bool(hardware_active) and bool(runtime_actions_enabled)
                and bool(inhibit_latched)
                and int(status_epoch) == int(requested_epoch)
                and int(inhibit_epoch) == int(requested_epoch)
                and int(inhibit_request_id) > int(previous_request_id)
                and int(write_in_flight) == 0
                and 0 <= status_age_sec <= freshness_sec)
    except (TypeError, ValueError, OverflowError):
        return False


def stop_future_result(future, now, deadline):
    """Classify an asynchronous STOP future without waiting for a late response."""
    if future is None:
        return True, None, None
    if not future.done():
        if now < deadline:
            return False, None, None
        return True, False, 'set_state acknowledgement timed out'
    try:
        response = future.result()
    except Exception as error:
        return True, False, str(error)
    if response is not None and getattr(response, 'ret', None) == 0:
        return True, True, None
    return True, False, 'empty response' if response is None else str(
        getattr(response, 'message', 'nonzero result'))


def inhibit_future_result(future, now, deadline):
    """Classify a private Trigger ACK without blocking the monitor executor."""
    if future is None:
        return True, None, None
    if not future.done():
        if now < deadline:
            return False, None, None
        return True, False, 'hardware inhibit acknowledgement timed out'
    try:
        response = future.result()
    except Exception as error:
        return True, False, str(error)
    if response is not None and bool(getattr(response, 'success', False)):
        return True, True, None
    return True, False, 'empty response' if response is None else str(
        getattr(response, 'message', 'inhibit latch rejected'))


class StopEvidence:
    """
    Track STOP requests, acknowledgements, controller state, and standstill.

    Measured standstill remains separate from service and report evidence.
    """

    def __init__(self, stable_samples, velocity_tolerance, position_tolerance,
                 max_feedback_age_sec):
        if (not isinstance(stable_samples, int) or stable_samples < 2
                or not math.isfinite(velocity_tolerance) or velocity_tolerance < 0
                or not math.isfinite(position_tolerance) or position_tolerance < 0
                or not math.isfinite(max_feedback_age_sec) or max_feedback_age_sec <= 0):
            raise ValueError('invalid measured-stop thresholds')
        self.stable_samples = stable_samples
        self.velocity_tolerance = velocity_tolerance
        self.position_tolerance = position_tolerance
        self.max_feedback_age_sec = max_feedback_age_sec
        self.requested = False
        self.acknowledged = False
        self.stop_reported = False
        self.measured_stopped = False
        self._last_sequence = None
        self._epoch = None
        self._anchor_q = None
        self._stable_count = 0

    def _reset_feedback_evidence(self):
        self.measured_stopped = False
        self._anchor_q = None
        self._stable_count = 0

    def invalidate_feedback(self):
        """Clear a prior standstill claim when the latest source is no longer fresh."""
        self._reset_feedback_evidence()

    def mark_requested(self):
        self.requested = True

    def mark_acknowledged(self, accepted):
        self.acknowledged = bool(accepted)

    def observe_report(self, report, *, valid=True):
        if not valid:
            self.stop_reported = False
            return False
        report = report if isinstance(report, RobotReport) else robot_report(report)
        self.stop_reported = report.state == 4
        return self.stop_reported

    def observe_feedback(self, q, dq, *, read_ok, read_sequence, epoch,
                         read_age_sec, transport_age_sec):
        """Require fresh reads, low velocity, and stable position to confirm stop."""
        try:
            read_age_sec = float(read_age_sec)
            transport_age_sec = float(transport_age_sec)
        except (TypeError, ValueError, OverflowError):
            self._reset_feedback_evidence()
            return False
        if (not read_ok or not math.isfinite(read_age_sec)
                or not math.isfinite(transport_age_sec) or read_age_sec < 0
                or transport_age_sec < 0
                or read_age_sec + transport_age_sec > self.max_feedback_age_sec):
            self._reset_feedback_evidence()
            return False
        try:
            positions = tuple(float(value) for value in q)
            velocities = tuple(float(value) for value in dq)
            sequence = int(read_sequence)
            epoch = int(epoch)
        except (TypeError, ValueError, OverflowError):
            self._reset_feedback_evidence()
            return False
        if (len(positions) != 6 or len(velocities) != 6 or sequence < 0
                or epoch < 0
                or not all(math.isfinite(value) for value in (*positions, *velocities))):
            self._reset_feedback_evidence()
            return False
        if self._last_sequence is not None and epoch != self._epoch:
            self._reset_feedback_evidence()
        elif self._last_sequence is not None and sequence <= self._last_sequence:
            self._reset_feedback_evidence()
            return False
        self._last_sequence = sequence
        self._epoch = epoch
        if any(abs(value) > self.velocity_tolerance for value in velocities):
            self._reset_feedback_evidence()
            return False
        if (self._anchor_q is None or any(
                abs(value - anchor) > self.position_tolerance
                for value, anchor in zip(positions, self._anchor_q))):
            self._anchor_q = positions
            self._stable_count = 1
        else:
            self._stable_count += 1
        self.measured_stopped = self._stable_count >= self.stable_samples
        return self.measured_stopped


class HardwareStateMonitor:
    """Pure state machine that arms after an expected robot report and SDK snapshot."""

    ACTIVE_STATES = frozenset((0, 1, 2))
    AXIS_MASK = 0x3F

    def __init__(self, preflight, expected_mode=1, startup_timeout_sec=20.0,
                 report_timeout_sec=0.5, sdk_timeout_sec=0.75, tolerance=1e-6,
                 clock=time.monotonic):
        """Initialize from the read-only snapshot captured before arm startup."""
        if expected_mode != 1:
            raise ValueError('The tracking controller requires xArm SERVOJ mode 1')
        limits = preflight.get('reduced_joint_limits_rad')
        if not isinstance(limits, (list, tuple)) or len(limits) != 6:
            raise ValueError('Preflight snapshot must contain six reduced joint ranges')
        if not all(isinstance(pair, (list, tuple)) and len(pair) == 2
                   and all(math.isfinite(float(value)) for value in pair) for pair in limits):
            raise ValueError('Preflight snapshot contains invalid reduced joint ranges')
        tcp_boundary = preflight.get('reduced_tcp_boundary_mm')
        tcp_speed = preflight.get('reduced_tcp_max_speed_mmps')
        if (tcp_speed is None or not isinstance(tcp_boundary, (list, tuple))
                or len(tcp_boundary) != 6
                or not all(math.isfinite(float(value)) for value in tcp_boundary)
                or not math.isfinite(float(tcp_speed))):
            raise ValueError('Preflight snapshot contains invalid TCP reduced settings')
        for name, value in (('startup_timeout_sec', startup_timeout_sec),
                            ('report_timeout_sec', report_timeout_sec),
                            ('sdk_timeout_sec', sdk_timeout_sec), ('tolerance', tolerance)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.expected_mode = int(expected_mode)
        self.expected_serial = str(preflight.get('serial_number', '')).strip()
        if not self.expected_serial:
            raise ValueError('Preflight snapshot must contain a robot serial number')
        self.serial_matches = False
        self.expected_reduced_mode = bool(preflight['reduced_mode'])
        self.expected_ranges = tuple(tuple(float(v) for v in pair) for pair in limits)
        self.expected_tcp_boundary = tuple(float(value) for value in tcp_boundary)
        self.expected_tcp_speed_mmps = float(tcp_speed)
        self.expected_joint_speed = preflight.get('reduced_max_joint_speed_rad_s')
        self.expected_fence = bool(preflight['safety_boundary_enabled'])
        self.expected_rebound = bool(preflight['collision_rebound_enabled'])
        self.startup_timeout_sec = startup_timeout_sec
        self.report_timeout_sec = report_timeout_sec
        self.sdk_timeout_sec = sdk_timeout_sec
        self.tolerance = tolerance
        self.clock = clock
        self.started_at = clock()
        self.last_report_at = None
        self.last_report_stamp_ns = None
        self.last_report = None
        self.fault_report = None
        self.last_sdk_at = None
        self.settings = None
        self.armed = False
        self.fault = None

    @property
    def ready(self):
        """Return true only after both sources match and the monitor has armed."""
        return self.armed and self.fault is None

    def _fail(self, reason):
        if self.fault is None:
            self.fault = str(reason)
        raise HardwareMonitorFault(self.fault)

    def _compare_preflight(self, settings):
        """Compare decoded SDK values with the constraints saved by preflight."""
        if settings.mode != self.expected_reduced_mode:
            self._fail('SDK reduced mode differs from preflight snapshot')
        if not self._close_nested((settings.tcp_boundary_mm,), (self.expected_tcp_boundary,)):
            self._fail('SDK reduced TCP boundary differs from preflight snapshot')
        if not self._close(settings.tcp_max_speed_mmps, self.expected_tcp_speed_mmps):
            self._fail('SDK reduced TCP speed differs from preflight snapshot')
        actual = tuple((settings.joint_ranges[2 * i], settings.joint_ranges[2 * i + 1])
                       for i in range(6))
        if not self._close_nested(actual, self.expected_ranges):
            for index, (expected, observed) in enumerate(zip(self.expected_ranges, actual), 1):
                if not self._close_nested((expected,), (observed,)):
                    self._fail(
                        f'SDK reduced joint limits changed for joint{index}: preflight '
                        f'[{expected[0]:.6g}, {expected[1]:.6g}] rad, live '
                        f'[{observed[0]:.6g}, {observed[1]:.6g}] rad')
            self._fail('SDK reduced joint limits differ from preflight snapshot')
        if (self.expected_joint_speed is not None
                and not self._close(settings.joint_max_speed, self.expected_joint_speed)):
            self._fail('SDK reduced joint speed differs from preflight snapshot')
        if settings.safety_boundary_enabled != self.expected_fence:
            self._fail('SDK safety-boundary setting differs from preflight snapshot')
        if settings.collision_rebound_enabled != self.expected_rebound:
            self._fail('SDK collision-rebound setting differs from preflight snapshot')

    def _close(self, left, right):
        return abs(float(left) - float(right)) <= self.tolerance

    def _close_nested(self, left, right):
        return (len(left) == len(right) and all(
            self._close(a, b) for left_pair, right_pair in zip(left, right)
            for a, b in zip(left_pair, right_pair)))

    def observe_report(self, report, received_at=None, source_stamp_ns=None):
        """Record RobotMsg; transient startup transitions do not arm or fault."""
        if self.fault is not None:
            raise HardwareMonitorFault(self.fault)
        received_at = self.clock() if received_at is None else received_at
        if source_stamp_ns is not None:
            if (source_stamp_ns <= 0 or (self.last_report_stamp_ns is not None
                                         and source_stamp_ns <= self.last_report_stamp_ns)):
                if self.armed:
                    self._fail('Robot state report timestamp repeated or moved backwards')
                raise ValueError('Robot state report timestamp repeated or moved backwards')
            self.last_report_stamp_ns = source_stamp_ns
        self.last_report = report if isinstance(report, RobotReport) else robot_report(report)
        self.last_report_at = received_at
        if self.armed:
            self._validate_live_report(self.last_report)
        else:
            self._try_arm()

    def observe_serial(self, serial):
        """Verify the SDK connection still points at the preflighted robot."""
        if str(serial).strip() != self.expected_serial:
            self._fail('Connected robot serial differs from preflight snapshot')
        self.serial_matches = True
        self._try_arm()

    def observe_sdk(self, raw_settings, received_at=None):
        """Validate the reduced settings snapshot and arm only with a fresh report."""
        if self.fault is not None:
            raise HardwareMonitorFault(self.fault)
        received_at = self.clock() if received_at is None else received_at
        if self.last_sdk_at is not None and received_at <= self.last_sdk_at:
            if self.armed:
                self._fail('SDK safety poll timestamp repeated or moved backwards')
            return
        if self.armed and self.clock() - received_at > self.sdk_timeout_sec:
            self._fail('SDK safety poll result arrived after its freshness timeout')
        try:
            snapshot = (raw_settings if isinstance(raw_settings, ReducedSettings)
                        else reduced_settings(raw_settings))
        except (TypeError, ValueError) as error:
            self._fail(f'Invalid SDK reduced-state response: {error}')
        if self.settings is not None:
            difference = self._settings_difference(snapshot, self.settings)
            if difference is not None:
                self._fail(difference)
        self._compare_preflight(snapshot)
        self.settings = snapshot
        self.last_sdk_at = received_at
        self._try_arm()

    def _same_settings(self, left, right):
        floats_match = (
            self._close_nested((left.tcp_boundary_mm,), (right.tcp_boundary_mm,))
            and self._close(left.tcp_max_speed_mmps, right.tcp_max_speed_mmps)
            and self._close(left.joint_max_speed, right.joint_max_speed)
            and self._close_nested((left.joint_ranges,), (right.joint_ranges,)))
        flags_match = (left.mode == right.mode
                       and left.safety_boundary_enabled == right.safety_boundary_enabled
                       and left.collision_rebound_enabled == right.collision_rebound_enabled)
        return floats_match and flags_match

    def _settings_difference(self, current, previous):
        """Describe a changed safety limit with units and its previous/current values."""
        for index in range(6):
            old = previous.joint_ranges[index * 2:index * 2 + 2]
            new = current.joint_ranges[index * 2:index * 2 + 2]
            if not self._close_nested((old,), (new,)):
                return (f'SDK reduced joint limits changed after startup for joint{index + 1}: '
                        f'old [{old[0]:.6g}, {old[1]:.6g}] rad, '
                        f'new [{new[0]:.6g}, {new[1]:.6g}] rad')
        if not self._same_settings(current, previous):
            return 'SDK reduced safety settings changed after monitor startup'
        return None

    def _try_arm(self):
        if (self.armed or not self.serial_matches or self.settings is None
                or self.last_report is None):
            return
        if self.clock() - self.last_report_at > self.report_timeout_sec:
            return
        if self.last_sdk_at is None or self.clock() - self.last_sdk_at > self.sdk_timeout_sec:
            return
        report = self.last_report
        # The controller may publish its own mode transition and STOP(5) while
        # the JTC spawner performs activation. Do not arm until its full state
        # agrees with the mode we are meant to supervise.
        if (report.mode != self.expected_mode or report.state not in self.ACTIVE_STATES
                or report.error != 0 or report.mt_brake & self.AXIS_MASK != self.AXIS_MASK
                or report.mt_able & self.AXIS_MASK != self.AXIS_MASK):
            return
        self.armed = True

    def _validate_live_report(self, report):
        reasons = report.fault_reasons(self.expected_mode, self.ACTIVE_STATES, self.AXIS_MASK)
        if reasons:
            self.fault_report = report
            self._fail('Robot report fault: ' + '; '.join(reasons))

    def observe_stop_report(self, report):
        """Keep observing reports after a latched fault; STOP state is not measured standstill."""
        report = report if isinstance(report, RobotReport) else robot_report(report)
        self.fault_report = report
        return report.state == 4

    def check_health(self, now=None):
        """Check startup, RobotMsg receipt, and SDK poll age independently."""
        now = self.clock() if now is None else now
        if self.fault is not None:
            raise HardwareMonitorFault(self.fault)
        if not self.armed:
            if now - self.started_at > self.startup_timeout_sec:
                self._fail('Timed out waiting for matching RobotMsg and SDK safety snapshot')
            return
        if self.last_report_at is None or now - self.last_report_at > self.report_timeout_sec:
            self._fail('Robot state report heartbeat expired')
        if self.last_sdk_at is None or now - self.last_sdk_at > self.sdk_timeout_sec:
            self._fail('SDK safety-state poll heartbeat expired')
        self._validate_live_report(self.last_report)
