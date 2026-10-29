#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Publish fixed-world gaze targets for the manual Gazebo posture demo."""

import argparse
from collections import deque
import importlib.util
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from diagnostic_msgs.msg import DiagnosticArray

from geometry_msgs.msg import PointStamped

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


JOINTS = tuple(f'joint{i}' for i in range(1, 7))
TRANSITION_SEC = 5.0
JOINT_STATE_HISTORY_SIZE = 32


def _rotate_x(q):
    """Return the unit vector of a transform's local +X axis."""
    return [1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            2.0 * (q.x * q.y + q.z * q.w),
            2.0 * (q.x * q.z - q.y * q.w)]


def _rotate_z(q):
    """Return the unit vector of a transform's local +Z axis."""
    return [2.0 * (q.x * q.z + q.y * q.w),
            2.0 * (q.y * q.z - q.x * q.w),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y)]


def _joint3_comfort(rest, search, angle_offset, physical_lower,
                    physical_upper):
    """Derive the normal bend branch and comfort interval from SRDF poses."""
    values = (rest, search, angle_offset, physical_lower, physical_upper)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('joint3 bend calibration values must be finite')
    if physical_lower >= physical_upper:
        raise ValueError('joint3 physical limits must have lower < upper')

    rest_angle = rest + angle_offset
    search_angle = search + angle_offset
    rest_sine = math.sin(rest_angle)
    search_sine = math.sin(search_angle)
    if (abs(rest_sine) <= 1e-9 or abs(search_sine) <= 1e-9 or
            rest_sine * search_sine <= 0.0):
        raise ValueError('SRDF rest/search poses disagree on bend branch')

    preferred_sign = 1.0 if rest_sine > 0.0 else -1.0
    minimum_sine = min(abs(rest_sine), abs(search_sine))
    sine_angle = math.asin(minimum_sine)
    transformed_rest = preferred_sign * rest_angle
    transformed_search = preferred_sign * search_angle
    center = (transformed_rest + transformed_search) / 2.0
    period_index = round((center - math.pi / 2.0) / (2.0 * math.pi))
    interval = None
    tolerance = 1e-9
    for index in range(period_index - 2, period_index + 3):
        low = sine_angle + 2.0 * math.pi * index
        high = math.pi - sine_angle + 2.0 * math.pi * index
        if (low - tolerance <= transformed_rest <= high + tolerance and
                low - tolerance <= transformed_search <= high + tolerance):
            interval = (low, high)
            break
    if interval is None:
        raise ValueError('no continuous comfort interval contains rest/search')

    low, high = interval
    if preferred_sign > 0.0:
        lower = low - angle_offset
        upper = high - angle_offset
    else:
        lower = -high - angle_offset
        upper = -low - angle_offset
    lower = max(lower, physical_lower)
    upper = min(upper, physical_upper)
    if (lower > upper or not lower - tolerance <= rest <= upper + tolerance or
            not lower - tolerance <= search <= upper + tolerance):
        raise ValueError('joint3 comfort interval misses a named pose')
    return preferred_sign, minimum_sine, lower, upper


def _hinge_geometry(shoulder, elbow, wrist, joint3_axis):
    """Return world hinge angle and normalized signed sine from FK points."""
    start = [elbow[i] - shoulder[i] for i in range(3)]
    end = [wrist[i] - elbow[i] for i in range(3)]
    start_length = math.sqrt(sum(value * value for value in start))
    end_length = math.sqrt(sum(value * value for value in end))
    axis_length = math.sqrt(sum(value * value for value in joint3_axis))
    if min(start_length, end_length, axis_length) <= 1e-12:
        return None
    axis = [value / axis_length for value in joint3_axis]
    cross = [start[1] * end[2] - start[2] * end[1],
             start[2] * end[0] - start[0] * end[2],
             start[0] * end[1] - start[1] * end[0]]
    signed_sine = sum(axis[i] * cross[i] for i in range(3)) / (
        start_length * end_length)
    cosine = sum(start[i] * end[i] for i in range(3)) / (
        start_length * end_length)
    return math.atan2(signed_sine, cosine), signed_sine


def _hinge_bend_metrics(shoulder, elbow, wrist, joint3_axis,
                        joint3_position, calibration):
    """Measure the signed S-E-W hinge bend and its comfort state."""
    geometry = _hinge_geometry(shoulder, elbow, wrist, joint3_axis)
    if geometry is None:
        return None
    angle, raw_sine = geometry
    preferred_sine = calibration['preferred_sign'] * raw_sine
    expected_sine = calibration['preferred_sign'] * math.sin(
        joint3_position + calibration['angle_offset'])
    lower = calibration['joint3_lower']
    upper = calibration['joint3_upper']
    comfortable = (
        preferred_sine >= calibration['minimum_sine'] - 1e-9 and
        lower - 1e-9 <= joint3_position <= upper + 1e-9)
    return {
        'angle_rad': angle,
        'signed_sine': raw_sine,
        'preferred_sine': preferred_sine,
        'expected_sine': expected_sine,
        'normal_branch': preferred_sine > 0.0,
        'comfortable': comfortable,
        'calibration_consistent': abs(raw_sine - math.sin(
            joint3_position + calibration['angle_offset'])) <= 0.02,
    }


def _joint_stamp_ns(stamp):
    """Return a ROS stamp in integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _synchronized_angle_offset(angle, joint3_position,
                               joint_stamp_ns, transform_stamp_ns):
    """Derive calibration only when q3 and FK share the same sample time."""
    if joint_stamp_ns != transform_stamp_ns:
        raise ValueError('joint state and FK calibration stamps differ')
    return angle - joint3_position


def _bend_status_text(metrics, joint3_position):
    """Keep FK shape visible when a sampled q3 consistency warning occurs."""
    if not metrics:
        return 'unavailable'
    branch = 'обычная' if metrics['normal_branch'] else 'обратная'
    comfort = 'в комфорте' if metrics['comfortable'] else 'вне комфорта'
    shape = ('невыпрямленная' if metrics['preferred_sine'] > 0.05 else
             'почти прямая')
    result = (f'{branch}/{comfort}/{shape} '
              f'sin={metrics["preferred_sine"]:+.3f} '
              f'q3={joint3_position:+.3f}rad')
    if not metrics['calibration_consistent']:
        result += ' [warning: q3/TF mismatch]'
    return result


def _seed_label(fixture_id):
    """Return the human reference label attached to a fixed demo seed."""
    if fixture_id is None:
        return 'без эталонной фикстуры'
    return f'эталон BAD{fixture_id}' if fixture_id >= 6 else \
        f'эталон GOOD{fixture_id}'


def _named_joint3_positions(srdf_path):
    """Read only the joint3 values for the model's named SRDF poses."""
    root = ET.parse(srdf_path).getroot()
    result = {}
    for pose_name in ('rest', 'search'):
        states = root.findall(f"./group_state[@name='{pose_name}']")
        if len(states) != 1:
            raise ValueError(f'SRDF must define one {pose_name} group state')
        joints = states[0].findall("./joint[@name='joint3']")
        if len(joints) != 1:
            raise ValueError(f'SRDF {pose_name} must define joint3 once')
        result[pose_name] = float(joints[0].attrib['value'])
    return result


def _physical_joint3_limits():
    """Load calibrated xArm6 limits from this package's installed profile."""
    from ament_index_python.packages import get_package_prefix

    profile_path = (Path(get_package_prefix('face_tracking_arm')) /
                    'lib/face_tracking_arm/face_tracking_bringup/'
                    'robot_profile.py')
    spec = importlib.util.spec_from_file_location(
        'face_tracking_arm_robot_profile', profile_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load robot profile: {profile_path}')
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    return profile.get_robot_profile('xarm6')['position_limits_rad'][2]


def _distance(a, b):
    squared = sum((a[i] - b[i]) ** 2 for i in range(3))
    return math.sqrt(squared)


def _side_metrics(base, elbow, wrist, target):
    """Return C and signed elbow clearance against the B/W/F plane."""
    base_to_wrist = [wrist[i] - base[i] for i in range(3)]
    base_to_elbow = [elbow[i] - base[i] for i in range(3)]
    base_to_target = [target[i] - base[i] for i in range(3)]
    length_squared = sum(value * value for value in base_to_wrist)
    if length_squared <= 1e-18:
        return math.nan, math.nan, math.nan
    projection = sum(base_to_elbow[i] * base_to_wrist[i]
                     for i in range(3)) / length_squared
    target_projection = sum(base_to_target[i] * base_to_wrist[i]
                            for i in range(3)) / length_squared
    elbow_perpendicular = [base_to_elbow[i] -
                           base_to_wrist[i] * projection for i in range(3)]
    target_perpendicular = [base_to_target[i] -
                            base_to_wrist[i] * target_projection
                            for i in range(3)]
    c = sum(elbow_perpendicular[i] * target_perpendicular[i]
            for i in range(3))
    target_perpendicular_length = math.sqrt(sum(
        value * value for value in target_perpendicular))
    if target_perpendicular_length <= 1e-9:
        return c, math.nan, target_perpendicular_length
    return c, -c / target_perpendicular_length, target_perpendicular_length


def _fixture_matches(joints, fixture, tolerance=0.02):
    """Report whether every measured joint matches a demo seed."""
    return len(joints) == 6 and len(fixture) == 6 and all(
        abs(joints[index] - fixture[index]) <= tolerance for index in range(6))


def _crossing_window(screen_perpendicular, elbow_side, radius, near):
    """Return the largest symmetric +/- oracle offset reachable on a sphere."""
    side_projection = sum(screen_perpendicular[i] * elbow_side[i]
                          for i in range(3))
    perpendicular = [screen_perpendicular[i] - side_projection * elbow_side[i]
                     for i in range(3)]
    orthogonal_squared = sum(value * value for value in perpendicular)
    room_squared = radius * radius - orthogonal_squared
    if room_squared <= 0.0:
        raise ValueError('screen offset leaves no crossing room on target sphere')
    amplitude = math.sqrt(room_squared) - abs(side_projection)
    if amplitude <= near:
        raise ValueError('target sphere does not reach both sides of C=0')
    return min(0.12, amplitude)


def _pointing_error(screen, normal, target):
    """Return the gaze-axis error in radians for the measured screen pose."""
    ray = [target[i] - screen[i] for i in range(3)]
    distance = math.sqrt(sum(value * value for value in ray))
    if distance <= 1e-12:
        return math.nan
    cosine = sum(normal[i] * ray[i] for i in range(3)) / distance
    return math.acos(min(1.0, max(-1.0, cosine)))


def _horizontal_roll(rotation):
    """Return absolute roll from world-up around the monitor's gaze axis."""
    gaze = _rotate_x(rotation)
    up = [2.0 * (rotation.x * rotation.z + rotation.y * rotation.w),
          2.0 * (rotation.y * rotation.z - rotation.x * rotation.w),
          1.0 - 2.0 * (rotation.x * rotation.x + rotation.y * rotation.y)]
    world_up = [0.0, 0.0, 1.0]
    reference = [world_up[i] - sum(world_up[j] * gaze[j] for j in range(3)) * gaze[i]
                 for i in range(3)]
    reference_norm = math.sqrt(sum(value * value for value in reference))
    if reference_norm <= 1e-9:
        return math.nan
    reference = [value / reference_norm for value in reference]
    projected_up = [up[i] - sum(up[j] * gaze[j] for j in range(3)) * gaze[i]
                    for i in range(3)]
    projected_norm = math.sqrt(sum(value * value for value in projected_up))
    if projected_norm <= 1e-9:
        return math.nan
    projected_up = [value / projected_norm for value in projected_up]
    dot = sum(reference[i] * projected_up[i] for i in range(3))
    dot = min(1.0, max(-1.0, dot))
    return math.acos(dot)


def _smoothstep(value):
    t = min(1.0, max(0.0, value))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _make_schedule(scenario, minimum, maximum):
    """Return phases as start, label, radius, transition, and anchor."""
    width = maximum - minimum
    middle = (minimum + maximum) / 2.0
    if scenario == 'posture':
        phase = (0.0, 'F неподвижна в world / выправление BAD-фикстуры',
                 middle, 0.0, 'initial')
        return [phase]
    if scenario == 'posture_good':
        phase = (0.0, 'GOOD-фикстура / удержание при неподвижной F',
                 middle, 0.0, 'initial')
        return [phase]
    if scenario == 'obstacle':
        return []
    if scenario == 'crossing':
        return []
    if scenario == 'range':
        inner_min = minimum + width / 6.0
        inner_max = maximum - width / 6.0
        return [
            (0.0, 'середина полосы', middle, 0.0, 'initial'),
            (10.0, 'внутренняя нижняя точка', inner_min, 5.0, 'initial'),
            (25.0, 'середина полосы', middle, 5.0, 'initial'),
            (40.0, 'внутренняя верхняя точка', inner_max, 5.0, 'initial'),
        ]
    return [
        (0.0, 'середина полосы', middle, 0.0, 'initial'),
        (5.0, 'ниже min', max(0.01, minimum - 0.05), 5.0, 'live'),
        (25.0, 'середина полосы', middle, 5.0, 'live'),
        (35.0, 'выше max', maximum + 0.05, 5.0, 'live'),
        (55.0, 'середина полосы', middle, 5.0, 'live'),
    ]


class DemoTarget(Node):
    """Publish fixed targets and print the current measured face distance."""

    def __init__(self, args):
        """Create subscriptions, target publisher, and timer."""
        super().__init__(
            'posture_demo_target',
            parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.args = args
        self.fixture = args.fixture_q
        self.schedule = _make_schedule(
            args.scenario, args.minimum, args.maximum)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(
            PointStamped, '/face/center', qos)
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/posture_demo/markers', 10)
        self.latest_joints = {}
        self.latest_joint_stamp_ns = None
        self.joint_state_history = deque(maxlen=JOINT_STATE_HISTORY_SIZE)
        self.last_joint_sample = None
        self.joint_path_rad = 0.0
        self.ready = False
        self.servo_status = 'unavailable'
        self.failed = False
        self.last_wait_report = None
        self.last_wait_report_wall = 0.0
        self.initial_position = None
        self.initial_normal = None
        self.bend_calibration = None
        self.fixture_id = args.fixture_id
        self.started_ns = None
        self.phase_index = -1
        self.phase_target = None
        self.previous_target = None
        self.crossing_foot = None
        self.crossing_axis = None
        self.crossing_radius_squared = None
        self.crossing_distance = None
        self.crossing_beta_sign = 1.0
        self.crossing_amplitude = 0.0
        self.crossing_near = 0.005
        self.crossing_phase = None
        self.crossing_available = False
        self.last_print_second = -1
        self.last_runtime_error = None
        self.last_runtime_error_ns = 0
        self.monitor_path_m = 0.0
        self.last_monitor_position = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, qos)
        self.create_subscription(
            DiagnosticArray, '/tracking/diagnostics', self._on_diagnostics, 10)
        self.create_subscription(
            DiagnosticArray, '/servo_node/diagnostics',
            self._on_servo_diagnostics, 10)
        # Keep startup diagnostics alive when simulation time has not started.
        self.timer = self.create_timer(
            1.0 / 30.0, self._tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _on_joint_state(self, message):
        joints = tuple(dict(zip(message.name, message.position)).get(name)
                       for name in JOINTS)
        if any(value is None for value in joints):
            return
        stamp_ns = _joint_stamp_ns(message.header.stamp)
        self.latest_joints = dict(zip(JOINTS, joints))
        self.latest_joint_stamp_ns = stamp_ns
        self.joint_state_history.append((stamp_ns, joints))

    def _on_diagnostics(self, message):
        for status in message.status:
            if status.name == 'face_tracking_arm/tracking':
                values = {value.key: value.value for value in status.values}
                self.ready = values.get('ready') == 'true'

    def _on_servo_diagnostics(self, message):
        for status in message.status:
            if status.name == 'face_tracking_arm/collision_aware_servo':
                values = {value.key: value.value for value in status.values}
                mode = values.get('mode', 'unknown')
                reference = values.get('motion_reference_state', 'unknown')
                code = values.get('servo_status_code', 'unknown')
                guards = ','.join(
                    f'{key}={values.get(key, "?")}' for key in (
                        'minimum_robust_headroom_m', 'active_collision_rows',
                        'outside_collision_viability_contacts',
                        'missed_control_deadlines', 'skipped_control_periods',
                        'solver_failure_count'))
                self.servo_status = (
                    f'{mode}/{reference}/servo={code} guards[{guards}]')

    def _joints(self):
        if not all(name in self.latest_joints for name in JOINTS):
            return None
        return tuple(self.latest_joints[name] for name in JOINTS)

    def _report_wait(self, reason):
        now = time.monotonic()
        if reason != self.last_wait_report or now - self.last_wait_report_wall >= 2.0:
            self.last_wait_report = reason
            self.last_wait_report_wall = now
            print(f'Ожидание старта publisher: {reason}', flush=True)

    def _fixture_matches(self, joints):
        return _fixture_matches(joints, self.fixture)

    def _lookup_screen(self):
        transform = self.tf.lookup_transform(
            'world', 'monitor_control_frame', Time())
        t = transform.transform.translation
        q = transform.transform.rotation
        position = [t.x, t.y, t.z]
        normal = _rotate_x(q)
        magnitude = math.sqrt(sum(value * value for value in normal))
        normal = [value / magnitude for value in normal]
        return position, normal

    def _synchronized_hinge_sample(self):
        """Select the newest buffered joint sample with matching chain TF."""
        last_error = None
        for stamp_ns, joints in reversed(self.joint_state_history):
            sample_time = Time(nanoseconds=stamp_ns)
            try:
                transforms = tuple(self.tf.lookup_transform(
                    'world', frame, sample_time) for frame in
                    ('link2', 'link3', 'link4'))
            except Exception as exc:
                last_error = exc
                continue
            if all(_joint_stamp_ns(tf.header.stamp) == stamp_ns
                   for tf in transforms):
                return stamp_ns, joints, transforms
        if last_error is not None:
            raise last_error
        raise ValueError('no buffered JointState has matching chain TF')

    def _calibrate_bend(self, sample=None):
        """Calibrate from FK transforms at a matched JointState timestamp."""
        if sample is None:
            sample = self._synchronized_hinge_sample()
        stamp_ns, joints, transforms = sample
        shoulder_tf, elbow_tf, wrist_tf = transforms
        shoulder = shoulder_tf.transform.translation
        elbow = elbow_tf.transform.translation
        wrist = wrist_tf.transform.translation
        angle, _bend = _hinge_geometry(
            (shoulder.x, shoulder.y, shoulder.z),
            (elbow.x, elbow.y, elbow.z),
            (wrist.x, wrist.y, wrist.z),
            _rotate_z(elbow_tf.transform.rotation))
        if angle is None:
            raise ValueError('invalid FK hinge geometry at JointState time')
        angle_offset = _synchronized_angle_offset(
            angle, joints[2], stamp_ns, _joint_stamp_ns(elbow_tf.header.stamp))
        named = _named_joint3_positions(self.args.srdf_path)
        joint_limits = _physical_joint3_limits()
        sign, minimum_sine, lower, upper = _joint3_comfort(
            named['rest'], named['search'], angle_offset,
            joint_limits[0], joint_limits[1])
        self.bend_calibration = {
            'angle_offset': angle_offset,
            'preferred_sign': sign,
            'minimum_sine': minimum_sine,
            'joint3_lower': lower,
            'joint3_upper': upper,
        }
        print('Калибровка hinge: '
              f'offset={angle_offset:+.6f}rad, '
              f'ветвь={sign:+.0f}, min_sin={minimum_sine:.3f}, '
              f'comfort_q3=[{lower:+.3f},{upper:+.3f}]rad; '
              f'SRDF rest/search={named["rest"]:+.3f}/'
              f'{named["search"]:+.3f}rad.', flush=True)

    @staticmethod
    def _point_at(position, normal, distance):
        return [position[i] + distance * normal[i] for i in range(3)]

    def _message(self, target):
        message = PointStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'world'
        message.point.x, message.point.y, message.point.z = target
        self.publisher.publish(message)

    @staticmethod
    def _vector_message(point):
        from geometry_msgs.msg import Point
        result = Point()
        result.x, result.y, result.z = point
        return result

    def _publish_markers(self, base, shoulder, elbow, wrist, target,
                         signed_clearance, bend_metrics):
        markers = MarkerArray()

        def make(marker_id, marker_type, color, scale):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'posture_demo'
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            return marker

        line = make(0, Marker.LINE_STRIP, (1.0, 0.75, 0.05, 0.95),
                    (0.012, 0.0, 0.0))
        line.points = [self._vector_message(base), self._vector_message(wrist)]
        markers.markers.append(line)
        hinge_color = ((0.1, 0.9, 0.25, 0.95)
                       if bend_metrics and bend_metrics['comfortable'] else
                       (0.95, 0.15, 0.1, 0.95))
        hinge = make(6, Marker.LINE_STRIP, hinge_color, (0.022, 0.0, 0.0))
        hinge.points = [self._vector_message(shoulder),
                        self._vector_message(elbow),
                        self._vector_message(wrist)]
        markers.markers.append(hinge)
        for marker_id, point, color in (
                (3, base, (1.0, 0.45, 0.05, 1.0)),
                (4, elbow, (0.15, 1.0, 0.2, 1.0)),
                (5, wrist, (0.8, 0.2, 1.0, 1.0))):
            joint = make(marker_id, Marker.SPHERE, color,
                         (0.035, 0.035, 0.035))
            joint.pose.position = self._vector_message(point)
            markers.markers.append(joint)
        goal = make(1, Marker.SPHERE, (0.1, 0.8, 1.0, 0.95),
                    (0.055, 0.055, 0.055))
        goal.pose.position = self._vector_message(target)
        markers.markers.append(goal)
        color = ((0.1, 0.9, 0.25, 0.95) if signed_clearance > 0.0 else
                 (0.95, 0.15, 0.1, 0.95) if math.isfinite(signed_clearance) else
                 (1.0, 0.65, 0.05, 0.95))
        direction = make(2, Marker.LINE_LIST, color, (0.018, 0.0, 0.0))
        base_to_wrist = [wrist[i] - base[i] for i in range(3)]
        length_squared = sum(value * value for value in base_to_wrist)
        if length_squared > 1e-18:
            ratio = sum((elbow[i] - base[i]) * base_to_wrist[i]
                        for i in range(3)) / length_squared
            projection = [base[i] + ratio * base_to_wrist[i]
                          for i in range(3)]
            direction.points = [self._vector_message(projection),
                                self._vector_message(elbow)]
            markers.markers.append(direction)
        if bend_metrics:
            label = make(7, Marker.TEXT_VIEW_FACING, hinge_color,
                         (0.0, 0.0, 0.055))
            label.pose.position = self._vector_message(
                [elbow[0], elbow[1], elbow[2] + 0.12])
            state = ('комфортная форма' if bend_metrics['comfortable'] else
                     'форма вне комфорта')
            label.text = (f'{state}; bend='
                          f"{bend_metrics['preferred_sine']:+.2f}")
            markers.markers.append(label)
        self.marker_publisher.publish(markers)

    def _seed_target(self, joints, position, normal):
        if (self.args.scenario in ('posture', 'posture_good') and
                not self._fixture_matches(joints)):
            print(
                'ОШИБКА seed: фактические q не совпадают с fixture; '
                'цель всё равно будет опубликована от текущего состояния. '
                'expected=[' + ', '.join(f'{q:.4f}' for q in self.fixture) +
                '] actual=[' +
                ', '.join(f'{q:.4f}' for q in joints) +
                ']', flush=True)
        self.initial_position = position
        self.initial_normal = normal
        middle = (self.args.minimum + self.args.maximum) / 2.0
        self.previous_target = self._point_at(position, normal, middle)
        if self.args.scenario == 'obstacle':
            self.previous_target = [0.95, 0.0, 1.15]
        if self.args.scenario == 'crossing':
            try:
                self.crossing_distance = middle
                self.crossing_near = 0.005
                positive_candidate = self._crossing_point(0.0)
                positive_candidate = self._crossing_point(
                    self.crossing_amplitude)
                positive_score = sum(
                    (positive_candidate[i] - position[i]) * normal[i]
                    for i in range(3))
                self.crossing_beta_sign = -1.0
                negative_candidate = self._crossing_point(
                    self.crossing_amplitude)
                negative_score = sum(
                    (negative_candidate[i] - position[i]) * normal[i]
                    for i in range(3))
                if positive_score >= negative_score:
                    self.crossing_beta_sign = 1.0
                    self.previous_target = positive_candidate
                else:
                    self.previous_target = negative_candidate
            except ValueError as exc:
                print(f'ОШИБКА crossing: {exc}; оставляю F неподвижной.',
                      flush=True)
                self.phase_target = self.previous_target
                return
            self.crossing_available = True
        self.phase_target = self.previous_target
        print(
            'Начальный state: q=[' + ', '.join(f'{q:.4f}' for q in joints) +
            '], monitor p=[' + ', '.join(f'{x:.4f}' for x in position) +
            '], normal+X=[' + ', '.join(f'{x:.4f}' for x in normal) + ']',
            flush=True)
        print(f'Эталон seed: {_seed_label(getattr(self, "fixture_id", None))}; '
              'это метка исходной человеческой позы.', flush=True)
        try:
            self._calibrate_bend()
            base = self.tf.lookup_transform(
                'world', 'link_base', Time()).transform.translation
            elbow = self.tf.lookup_transform(
                'world', 'link3', Time()).transform.translation
            wrist = self.tf.lookup_transform(
                'world', 'link4', Time()).transform.translation
            goal = self.previous_target
            cross, clearance, radius = _side_metrics(
                (base.x, base.y, base.z),
                (elbow.x, elbow.y, elbow.z),
                (wrist.x, wrist.y, wrist.z), goal)
            print(f'Стартовая сторона B/E/W/F: C={cross:+.6f}m^2 '
                  f'r={radius:.6f}m s={clearance:+.5f}m '
                  '(target preseeded in world)', flush=True)
            if self.args.scenario == 'crossing':
                print(f'Crossing path: radius={self.crossing_distance:.3f}m, '
                      f'offset={self.crossing_amplitude:.4f}m -> '
                      f'±{self.crossing_near:.4f}m near BW.', flush=True)
        except Exception as exc:
            print('Стартовая геометрия B/E/W/F пока недоступна.',
                  f' {type(exc).__name__}: {exc}', flush=True)
        print(
            f'Сразу публикую зафиксированную цель на радиусе={middle:.4f}m; '
            'расписание начнётся после ready.', flush=True)

    def _begin_schedule(self, now_ns):
        self.started_ns = now_ns
        self.phase_index = -1
        duration = {
            'posture': 'до Ctrl+C',
            'posture_good': 'до Ctrl+C',
            'crossing': '16 секунд перехода + удержание',
            'range': '55 секунд + удержание',
            'boundaries': '70 секунд + удержание',
            'obstacle': 'неподвижная цель за тестовым препятствием',
        }[self.args.scenario]
        print(f'ready получен. Расписание: {duration}; publisher до Ctrl+C.',
              flush=True)
        if self.args.scenario == 'crossing' and self.crossing_available:
            print('Цель в world следует за текущей геометрией BW; '
                  'пиксельные координаты экрана не используются.', flush=True)
        else:
            print('Целевая точка зафиксирована в world и '
                  'не следует за экраном.',
                  flush=True)

    def _activate_phase(self, index, elapsed):
        start, label, radius, _transition, anchor = self.schedule[index]
        if anchor == 'initial':
            position, normal = self.initial_position, self.initial_normal
        else:
            position, normal = self._lookup_screen()
        target = self._point_at(position, normal, radius)
        self.previous_target = self.phase_target
        self.phase_target = target
        self.phase_index = index
        print(f'Фаза: {label}; endpoint зафиксирован в world.', flush=True)
        return start

    def _crossing_point(self, side_offset):
        """Return a point on the live BW side at fixed screen distance."""
        base_tf = self.tf.lookup_transform('world', 'link_base', Time())
        elbow_tf = self.tf.lookup_transform('world', 'link3', Time())
        wrist_tf = self.tf.lookup_transform('world', 'link4', Time())
        base = base_tf.transform.translation
        elbow = elbow_tf.transform.translation
        wrist = wrist_tf.transform.translation
        b = [base.x, base.y, base.z]
        e = [elbow.x, elbow.y, elbow.z]
        w = [wrist.x, wrist.y, wrist.z]
        screen, _normal = self._lookup_screen()

        axis = [w[i] - b[i] for i in range(3)]
        length = math.sqrt(sum(value * value for value in axis))
        if length <= 1e-9:
            raise ValueError('B and W are degenerate during crossing')
        axis = [value / length for value in axis]
        screen_offset = [screen[i] - b[i] for i in range(3)]
        along = sum(screen_offset[i] * axis[i] for i in range(3))
        foot = [b[i] + along * axis[i] for i in range(3)]
        screen_perpendicular = [screen[i] - foot[i] for i in range(3)]

        elbow_offset = [e[i] - b[i] for i in range(3)]
        elbow_along = sum(elbow_offset[i] * axis[i] for i in range(3))
        side = [elbow_offset[i] - elbow_along * axis[i]
                for i in range(3)]
        side_length = math.sqrt(sum(value * value for value in side))
        if side_length <= 1e-9:
            raise ValueError('elbow side is degenerate during crossing')
        side = [value / side_length for value in side]

        self.crossing_amplitude = _crossing_window(
            screen_perpendicular, side, self.crossing_distance,
            self.crossing_near)
        side_offset = min(self.crossing_amplitude,
                          max(-self.crossing_amplitude, side_offset))
        side_projection = sum(screen_perpendicular[i] * side[i]
                              for i in range(3))
        orthogonal = [screen_perpendicular[i] - side_projection * side[i]
                      for i in range(3)]
        remaining_squared = (
            self.crossing_distance ** 2 -
            sum(value * value for value in orthogonal) -
            (side_offset - side_projection) ** 2)
        if remaining_squared <= 0.0:
            raise ValueError('target sphere has no room for the crossing point')
        beta = math.sqrt(remaining_squared)
        return [foot[i] + self.crossing_beta_sign * beta * axis[i] +
                side_offset * side[i]
                for i in range(3)]

    def _scheduled_target(self, elapsed):
        if self.args.scenario == 'obstacle':
            return self.phase_target, 'неподвижная цель у штатного стола'
        if self.args.scenario == 'crossing':
            if not self.crossing_available:
                return self.phase_target, 'crossing недоступен; F неподвижна'
            amplitude = self.crossing_amplitude
            near = self.crossing_near
            if elapsed < 7.0:
                phase = 'цель со стороны локтя'
                offset = near + (amplitude - near) * (
                    1.0 - _smoothstep(elapsed / 7.0))
            elif elapsed < 9.0:
                phase = 'плавный проход у BW (неопределённая сторона)'
                offset = near * (1.0 - 2.0 *
                                 _smoothstep((elapsed - 7.0) / 2.0))
            elif elapsed < 16.0:
                phase = 'цель с противоположной стороны'
                offset = -near - (amplitude - near) * _smoothstep(
                    (elapsed - 9.0) / 7.0)
            else:
                phase = 'удержание противоположной стороны'
                offset = -amplitude
            if phase != self.crossing_phase:
                self.crossing_phase = phase
                print(f'Фаза: {phase}; F следует за текущей BW на сфере '
                      f'{self.crossing_distance:.3f}m.', flush=True)
            try:
                target = self._crossing_point(offset)
            except Exception as exc:
                now_ns = self.get_clock().now().nanoseconds
                error = f'{type(exc).__name__}: {exc}'
                if (error != self.last_runtime_error or
                        now_ns - self.last_runtime_error_ns >=
                        2_000_000_000):
                    self.last_runtime_error = error
                    self.last_runtime_error_ns = now_ns
                    print(f'Crossing target unavailable; keep last F: {error}',
                          flush=True)
                return self.previous_target, phase
            self.previous_target = target
            self.phase_target = target
            return target, phase
        index = max(i for i, phase in enumerate(self.schedule)
                    if elapsed >= phase[0])
        if index != self.phase_index:
            try:
                start = self._activate_phase(index, elapsed)
            except Exception:
                return self.phase_target, self.schedule[self.phase_index][1]
        else:
            start = self.schedule[index][0]
        _phase_start, label, _radius, transition, _anchor = (
            self.schedule[index])
        if transition > 0.0 and elapsed < start + transition:
            blend = _smoothstep((elapsed - start) / transition)
            target = [self.previous_target[i] + blend *
                      (self.phase_target[i] - self.previous_target[i])
                      for i in range(3)]
            return target, label + ' (переход)'
        return self.phase_target, label

    def _tick(self):
        now = self.get_clock().now()
        if self.failed:
            return
        if now.nanoseconds <= 0:
            self._report_wait(
                '/clock пока равен нулю; target с корректным sim-time stamp '
                'не публикуется')
            return
        if self.initial_position is None:
            joints = self._joints()
            if joints is None:
                self._report_wait(
                    'нет полного JointState для joint1..joint6 на /joint_states')
                return
            try:
                position, normal = self._lookup_screen()
            except Exception as exc:
                self._report_wait(
                    f'нет TF world->monitor_control_frame: '
                    f'{type(exc).__name__}: {exc}')
                return
            self._seed_target(joints, position, normal)
            if self.failed:
                return
        if self.started_ns is None and self.ready:
            self._begin_schedule(now.nanoseconds)
        if self.started_ns is None:
            target, label = self.phase_target, 'ожидание ready'
        else:
            elapsed = (now.nanoseconds - self.started_ns) * 1e-9
            target, label = self._scheduled_target(elapsed)
        self._message(target)
        try:
            joints = self._joints()
            if joints is not None:
                if self.last_joint_sample is not None:
                    self.joint_path_rad += math.sqrt(sum(
                        (joints[i] - self.last_joint_sample[i]) ** 2
                        for i in range(6)))
                self.last_joint_sample = joints
            if joints is None or self.latest_joint_stamp_ns is None:
                self._report_wait('ожидание JointState перед FK-метрикой')
                return
            try:
                sample = self._synchronized_hinge_sample()
            except Exception as exc:
                self._report_wait(
                    'ожидание согласованной пары JointState/TF: '
                    f'{type(exc).__name__}: {exc}')
                return
            sample_stamp_ns, sample_joints, shape_tfs = sample
            if self.bend_calibration is None:
                self._calibrate_bend(sample)
            position, normal = self._lookup_screen()
            actual_distance = _distance(position, target)
            pointing = math.degrees(
                _pointing_error(position, normal, target))
            if self.last_monitor_position is not None:
                self.monitor_path_m += _distance(
                    self.last_monitor_position, position)
            self.last_monitor_position = position
            joint_time = Time(nanoseconds=sample_stamp_ns)
            base_tf = self.tf.lookup_transform(
                'world', 'link_base', joint_time)
            shoulder_tf, elbow_tf, wrist_tf = shape_tfs
            points = []
            for transform in (base_tf, elbow_tf, wrist_tf):
                t = transform.transform.translation
                points.append([t.x, t.y, t.z])
            base, elbow, wrist = points
            shoulder_t = shoulder_tf.transform.translation
            shoulder = [shoulder_t.x, shoulder_t.y, shoulder_t.z]
            joint3_axis = _rotate_z(elbow_tf.transform.rotation)
            bend_metrics = None
            if self.bend_calibration is not None:
                bend_metrics = _hinge_bend_metrics(
                    shoulder, elbow, wrist, joint3_axis, sample_joints[2],
                    self.bend_calibration)
            cross, signed_clearance, side_radius = _side_metrics(
                base, elbow, wrist, target)
            self._publish_markers(
                base, shoulder, elbow, wrist, target, signed_clearance,
                bend_metrics)
            rotation = self.tf.lookup_transform(
                'world', 'monitor_control_frame', Time()).transform.rotation
            horizontal_roll = math.degrees(_horizontal_roll(rotation))
            second = int(self.get_clock().now().nanoseconds * 1e-9)
            if second != self.last_print_second:
                self.last_print_second = second
                elapsed = (self.get_clock().now().nanoseconds -
                           self.started_ns) * 1e-9 if self.started_ns else 0.0
                s_text = (f'{signed_clearance:+.4f}m' if
                          math.isfinite(signed_clearance) else 'undefined')
                bend_text = _bend_status_text(
                    bend_metrics, sample_joints[2])
                print(
                    f't={elapsed:.1f}s {label}: actual d={actual_distance:.4f}m '
                    f'(band {self.args.minimum:.2f}..'
                    f'{self.args.maximum:.2f}); '
                    f'gaze={pointing:.3f}deg roll={horizontal_roll:.3f}deg; '
                    f'FACE-side C={cross:+.6f}m^2 s={s_text}; '
                    f'hinge={bend_text}; '
                    f'r(BW,F)={side_radius:.6f}m; '
                    f'B={base!r} E={elbow!r} W={wrist!r} F={target!r}; '
                    f'monitor_path={self.monitor_path_m:.4f}m; '
                    f'joint_path={self.joint_path_rad:.4f}rad; '
                    f'ready={self.ready}; servo={self.servo_status}',
                    flush=True)
        except Exception as exc:
            now_ns = self.get_clock().now().nanoseconds
            error = f'{type(exc).__name__}: {exc}'
            if (error != self.last_runtime_error or
                    now_ns - self.last_runtime_error_ns >= 2_000_000_000):
                self.last_runtime_error = error
                self.last_runtime_error_ns = now_ns
                print(f'Метрики этого тика недоступны: {error}', flush=True)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--scenario', choices=('posture', 'posture_good', 'crossing', 'range',
                               'boundaries', 'obstacle'),
        required=True)
    parser.add_argument('--minimum', type=float, default=0.37)
    parser.add_argument('--maximum', type=float, default=0.43)
    parser.add_argument('--fixture-q', type=float, nargs=6)
    parser.add_argument('--fixture-id', type=int)
    parser.add_argument('--srdf-path', required=True)
    args = parser.parse_args()
    if (not math.isfinite(args.minimum) or not math.isfinite(args.maximum) or
            args.minimum <= 0.0 or args.minimum >= args.maximum):
        parser.error('bounds must be finite and satisfy 0 < minimum < maximum')
    if args.scenario in ('posture', 'posture_good') and not args.fixture_q:
        parser.error('posture scenarios require --fixture-q')
    return args


def main():
    """Run the ROS target publisher until interrupted."""
    args = _arguments()
    rclpy.init(args=sys.argv[:1])
    node = DemoTarget(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
