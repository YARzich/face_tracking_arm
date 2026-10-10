# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""ROS adapter for isolated top alignment and camera hand-eye calibration."""

import argparse
import math
import signal
import threading
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry, CollisionObject, PlanningScene, PlanningSceneComponents, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .hand_eye_calibration import (
    CalibrationSample, detect_square_marker, planned_joint_offsets_deg,
    solve_hand_eye, validate_held_out)
from .hardware_description import load_yaml
from .mount_setup import (
    apply_optical_calibration, remember_top_direction, save_yaml_atomic,
    top_alignment_error_deg, top_joint6_goal)
from .robot_profile import get_robot_profile


def _transform_matrix(value: TransformStamped):
    t, q = value.transform.translation, value.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [rotation[0] + [t.x], rotation[1] + [t.y], rotation[2] + [t.z],
            [0.0, 0.0, 0.0, 1.0]]


def _stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class _DeferredSignalShutdown:
    """Keep ROS alive until the CLI confirms canceled motion."""

    _SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self):
        """Track prior handlers and suppress repeated termination signals."""
        self._shutdown_requested = False
        self._goal_submission_active = False
        self._deferred_signum = None
        self._previous_handlers = {}

    def install(self):
        """Install termination handlers and remember the process defaults."""
        for signum in self._SIGNALS:
            if signum not in self._previous_handlers:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)

    def begin_cleanup(self):
        self._shutdown_requested = True

    def begin_goal_submission(self):
        self._goal_submission_active = True

    def end_goal_submission(self):
        self._goal_submission_active = False
        signum = self._deferred_signum
        self._deferred_signum = None
        if signum is not None:
            self._handle(signum, None)

    def restore(self):
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle(self, signum, _frame):
        if self._shutdown_requested:
            return
        if self._goal_submission_active:
            if self._deferred_signum is None:
                self._deferred_signum = signum
            return
        self._shutdown_requested = True
        raise SystemExit(128 + signum)


class MountSetupNode(Node):
    """Thin ROS adapter; geometry and acceptance rules live in pure modules."""

    def __init__(self, args, signal_shutdown):
        super().__init__('mount_setup')
        self.args = args
        self.signal_shutdown = signal_shutdown
        self.config = load_yaml(args.config_file)
        self.mode_config = self.config[args.mode]
        self.camera = self.mode_config['left']
        self.joint_names = [f'joint{i}' for i in range(1, 7)]
        if args.operation == 'view':
            self.limits = []
            self.planning_group = ''
        else:
            self.declare_parameter('joint_limits_rad', Parameter.Type.DOUBLE_ARRAY)
            flat_limits = list(self.get_parameter('joint_limits_rad').value)
            if len(flat_limits) != 12:
                raise ValueError('Robot joint limits are required for setup motion')
            self.limits = [flat_limits[index:index + 2] for index in range(0, 12, 2)]
            self.declare_parameter('planning_group', '')
            self.planning_group = str(self.get_parameter('planning_group').value)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_joint_state = None
        self.latest_info = None
        self.latest_image = None
        self.image_sequence = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._joint_callback, 20)
        self.create_subscription(CameraInfo, self.camera['info_topic'], self._info_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.camera['image_topic'], self._image_callback,
                                 qos_profile_sensor_data)
        if args.operation == 'view':
            self.action = None
            self.validity = None
            self.apply_scene = None
            self.get_scene = None
            self.scene = None
        else:
            profile = get_robot_profile(self.config['robot']['model'],
                                        self.config['robot']['model_num'])
            controller = args.controller or profile['controller_name']
            self.action = ActionClient(self, FollowJointTrajectory,
                                       '/' + controller + '/follow_joint_trajectory')
            self.validity = self.create_client(GetStateValidity, '/check_state_validity')
            self.apply_scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
            self.get_scene = self.create_client(GetPlanningScene, '/get_planning_scene')
            self.scene = self.create_publisher(PlanningScene, '/planning_scene', 1)
        self.active_goal = None
        self.pending_goal_future = None
        self.goal_submission_unknown = False

    def _joint_callback(self, message):
        with self.lock:
            self.latest_joint_state = message

    def _info_callback(self, message):
        with self.lock:
            self.latest_info = message

    def _image_callback(self, message):
        with self.lock:
            self.latest_image = message
            self.image_sequence += 1

    def _spin_for(self, duration_sec):
        end = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.05)

    def _joint_snapshot(self):
        with self.lock:
            message = self.latest_joint_state
        if message is None:
            raise ValueError('No joint feedback received')
        by_name = {name: i for i, name in enumerate(message.name)}
        if any(name not in by_name for name in self.joint_names):
            raise ValueError('Joint feedback does not contain all six arm joints')
        if (len(message.position) != len(message.name) or
                len(message.velocity) != len(message.name)):
            raise ValueError('Joint position and velocity arrays must match joint names')
        stamp = _stamp_seconds(message.header.stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0 or stamp <= 0 or now - stamp > .10 or stamp - now > .05:
            raise ValueError('Fresh joint feedback is unavailable; setup stopped')
        positions = [message.position[by_name[name]] for name in self.joint_names]
        velocities = [message.velocity[by_name[name]] for name in self.joint_names]
        if (not all(math.isfinite(value) for value in positions) or
                not all(math.isfinite(value) for value in velocities)):
            raise ValueError('Non-finite joint feedback; setup stopped')
        for index, (position, bounds) in enumerate(zip(positions, self.limits)):
            if not bounds[0] <= position <= bounds[1]:
                raise ValueError(f'Joint{index + 1} feedback is outside active limits')
        return message, positions, velocities

    def _flange_pose(self, stamp, reference_frame='link_base'):
        capture_time = Time.from_msg(stamp)
        deadline = time.monotonic() + .5
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                if self.tf_buffer.can_transform(reference_frame, 'link6', capture_time):
                    transform = self.tf_buffer.lookup_transform(
                        reference_frame, 'link6', capture_time,
                        timeout=Duration(seconds=0.0))
                    return _transform_matrix(transform)
            except Exception:
                pass
            self._spin_for(.02)
        raise ValueError(f'No {reference_frame}→link6 transform at requested stamp')

    def _require_stopped_state(self, target):
        stable = 0
        last_stamp = -math.inf
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_for(.02)
            message, positions, velocities = self._joint_snapshot()
            stamp = _stamp_seconds(message.header.stamp)
            if stamp <= last_stamp:
                continue
            last_stamp = stamp
            if (all(abs(positions[i] - target[i]) < math.radians(.5) for i in range(6)) and
                    all(abs(value) < .01 for value in velocities)):
                stable += 1
                if stable >= 5:
                    return
            else:
                stable = 0
        raise ValueError('Arm did not settle at the requested pose; calibration stopped')

    def _wait_until_still(self, timeout_sec=5.0):
        stable = 0
        previous = None
        last_stamp = -math.inf
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_for(.05)
            message, positions, velocities = self._joint_snapshot()
            stamp = _stamp_seconds(message.header.stamp)
            if stamp <= last_stamp:
                continue
            last_stamp = stamp
            moved = (previous is not None and max(
                abs(positions[index] - previous[index]) for index in range(6)) >=
                math.radians(.1))
            if not moved and all(abs(value) < .01 for value in velocities):
                stable += 1
                if stable >= 5:
                    return True
            else:
                stable = 0
            previous = positions
        return False

    def _cancel_active_motion(self):
        """Cancel motion and confirm its result plus fresh stopped feedback."""
        if self.goal_submission_unknown and self.pending_goal_future is None:
            self.get_logger().error(
                'Trajectory send outcome is unknown; arm stop is unconfirmed')
            return False
        if self.active_goal is None and self.pending_goal_future is not None:
            try:
                pending = self.pending_goal_future
                rclpy.spin_until_future_complete(
                    self, pending, timeout_sec=2.0)
                if not pending.done():
                    self.get_logger().error(
                        'Goal acceptance is unconfirmed; arm stop is unknown')
                    return False
                accepted_goal = pending.result()
                if accepted_goal is None:
                    self.get_logger().error(
                        'Goal acceptance has no result; arm stop is unknown')
                    return False
                self.pending_goal_future = None
                if not accepted_goal.accepted:
                    if not self._wait_until_still():
                        self.get_logger().error(
                            'Rejected goal has no fresh stopped feedback')
                        return False
                    return True
                self.active_goal = accepted_goal
            except Exception as error:
                self.get_logger().error(
                    f'Goal acceptance could not be confirmed: {error}')
                return False
        if self.active_goal is None:
            return True
        try:
            cancel_future = self.active_goal.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            if not cancel_future.done() or cancel_future.result() is None:
                self.get_logger().error('Motion cancellation acknowledgement timed out')
                return False
            result_future = self.active_goal.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=4.0)
            if not result_future.done() or result_future.result() is None:
                self.get_logger().error('Motion result after cancellation is unconfirmed')
                return False
            result = result_future.result()
            self.active_goal = None
            if result.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED,
                                     GoalStatus.STATUS_SUCCEEDED):
                self.get_logger().error('Trajectory remains active after cancellation request')
                return False
            if not self._wait_until_still():
                self.get_logger().error('Fresh feedback did not confirm that the arm is still')
                return False
            return True
        except Exception as error:
            self.get_logger().error(f'Motion cancellation could not be confirmed: {error}')
            return False

    def _wait_for_joint_feedback(self, timeout_sec=10.0):
        deadline = time.monotonic() + timeout_sec
        last_error = 'joint feedback has not arrived'
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_for(.05)
            try:
                return self._joint_snapshot()
            except ValueError as error:
                last_error = str(error)
        raise ValueError(f'Fresh arm feedback is unavailable: {last_error}')

    def _publish_table(self):
        table = self.config['table']
        if not any(table['dimensions_m']):
            return
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        object_ = CollisionObject()
        object_.id = 'round_table'
        object_.header.frame_id = 'world'
        primitive = SolidPrimitive()
        pose = Pose()
        dimensions = table['dimensions_m']
        if table['shape'] == 'box':
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = dimensions
        else:
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [dimensions[0], dimensions[1]]
        object_.primitives = [primitive]
        object_.primitive_poses = [pose]
        pose.position.x, pose.position.y, pose.position.z = table['center_m']
        yaw = math.radians(table['yaw_deg']) / 2
        pose.orientation.z = math.sin(yaw)
        pose.orientation.w = math.cos(yaw)
        object_.operation = CollisionObject.ADD
        scene.world.collision_objects = [object_]
        if (not self.get_scene.wait_for_service(timeout_sec=8.0) or
                not self.apply_scene.wait_for_service(timeout_sec=8.0)):
            raise ValueError('MoveIt planning-scene services are unavailable; no motion sent')
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        scene_future = self.get_scene.call_async(scene_request)
        rclpy.spin_until_future_complete(self, scene_future, timeout_sec=3.0)
        if not scene_future.done() or scene_future.result() is None:
            raise ValueError('MoveIt did not return the collision matrix; no motion sent')
        matrix = scene_future.result().scene.allowed_collision_matrix
        names = matrix.entry_names
        for row in matrix.entry_values:
            row.enabled.extend([False] * max(0, len(names) - len(row.enabled)))
        while len(matrix.entry_values) < len(names):
            entry = AllowedCollisionEntry()
            entry.enabled = [False] * len(names)
            matrix.entry_values.append(entry)
        if 'link_base' not in names:
            names.append('link_base')
            for row in matrix.entry_values:
                row.enabled.append(False)
            entry = AllowedCollisionEntry()
            entry.enabled = [False] * len(names)
            matrix.entry_values.append(entry)
        if object_.id not in names:
            names.append(object_.id)
            for row in matrix.entry_values:
                row.enabled.append(False)
            entry = AllowedCollisionEntry()
            entry.enabled = [False] * len(names)
            matrix.entry_values.append(entry)
        base_index, table_index = names.index('link_base'), names.index(object_.id)
        matrix.entry_values[base_index].enabled[table_index] = True
        matrix.entry_values[table_index].enabled[base_index] = True
        request = ApplyPlanningScene.Request()
        request.scene = scene
        request.scene.allowed_collision_matrix = matrix
        future = self.apply_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if not future.done() or future.result() is None or not future.result().success:
            raise ValueError('MoveIt did not acknowledge the setup table collision object')

    def _validate_path(self, start, target):
        if not self.validity.wait_for_service(timeout_sec=8.0):
            raise ValueError('MoveIt collision validity service is unavailable; no motion sent')
        self._publish_table()
        steps = max(1, math.ceil(max(abs(target[i] - start[i]) for i in range(6)) /
                                 math.radians(2.0)))
        for step in range(steps + 1):
            ratio = step / steps
            state = RobotState()
            state.joint_state.name = self.joint_names
            state.joint_state.position = [start[i] + ratio * (target[i] - start[i])
                                          for i in range(6)]
            request = GetStateValidity.Request()
            request.group_name = self.planning_group
            request.robot_state = state
            future = self.validity.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                raise ValueError('MoveIt collision check timed out; no motion sent')
            response = future.result()
            if not response.valid:
                names = [contact.contact_body_1 + '/' + contact.contact_body_2
                         for contact in response.contacts]
                detail = ', '.join(names[:3]) or 'collision or invalid robot state'
                raise ValueError(f'Calibration path is not collision-free: {detail}')

    def _move(self, positions):
        if len(positions) != 6 or not all(math.isfinite(value) for value in positions):
            raise ValueError('Motion target must contain six finite joint angles')
        for index, (value, bounds) in enumerate(zip(positions, self.limits)):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'Joint{index + 1} target is outside its active limit')
        current_message, current, _ = self._joint_snapshot()
        if any(abs(positions[i] - current[i]) > math.radians(.1) for i in range(4)):
            raise ValueError('J1-J4 changed during setup; no trajectory was sent')
        # Hold J1-J4 at their measured values; set-top also holds J5. Calibration
        # may move J5 and J6, while the checks above reject any J1-J4 drift.
        positions = list(positions)
        positions[:4] = current[:4]
        if self.args.operation == 'set-top':
            positions[4] = current[4]
        self._validate_path(current, positions)
        if not self.action.wait_for_server(timeout_sec=10.0):
            raise ValueError('Joint trajectory controller is not active')
        delta = max(abs(positions[4] - current[4]), abs(positions[5] - current[5]))
        duration = max(5.0, delta / .05 + 1.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal.trajectory.points = [point]
        self.signal_shutdown.begin_goal_submission()
        try:
            future = self.action.send_goal_async(goal)
            self.pending_goal_future = future
            self.goal_submission_unknown = False
        except Exception:
            self.goal_submission_unknown = True
            raise
        finally:
            self.signal_shutdown.end_goal_submission()
        rclpy.spin_until_future_complete(self, future, timeout_sec=12.0)
        if not future.done() or future.result() is None:
            raise ValueError('Joint trajectory goal was not accepted')
        self.active_goal = future.result()
        self.pending_goal_future = None
        if not self.active_goal.accepted:
            self.active_goal = None
            raise ValueError('Joint trajectory goal was not accepted')
        result_future = self.active_goal.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 5.0)
        if not result_future.done() or result_future.result() is None:
            stopped = self._cancel_active_motion()
            if not stopped:
                raise ValueError('Motion timed out; cancellation/stopped state is unconfirmed. '
                                 'Stop the arm from UFACTORY Studio; no recovery motion was sent')
            raise ValueError('Motion timed out and was cancelled; no recovery motion was sent')
        result = result_future.result()
        self.active_goal = None
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
            if not self._wait_until_still():
                raise ValueError('Robot stopped or rejected the motion, but stopped feedback is '
                                 'unconfirmed; no reset or retry was sent')
            raise ValueError('Robot stopped or rejected the motion; no reset or retry was sent')
        self._require_stopped_state(positions)

    def _capture_one(self, side_m, previous_sequence):
        deadline = time.monotonic() + 15.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_for(.01)
            with self.lock:
                image = self.latest_image
                info = self.latest_info
                sequence = self.image_sequence
                joints = self.latest_joint_state
            if image is None or info is None or joints is None or sequence <= previous_sequence:
                continue
            if image.header.frame_id != self.camera['frame_id']:
                raise ValueError('Image frame_id differs from the configured optical camera frame')
            if info.header.frame_id != image.header.frame_id:
                raise ValueError('Image and CameraInfo frame_id values differ')
            if image.width != info.width or image.height != info.height:
                raise ValueError('Image dimensions differ from CameraInfo')
            image_stamp = _stamp_seconds(image.header.stamp)
            info_stamp = _stamp_seconds(info.header.stamp)
            joint_stamp = _stamp_seconds(joints.header.stamp)
            if info_stamp <= 0 or abs(image_stamp - info_stamp) > .05:
                continue
            if abs(image_stamp - joint_stamp) > .05:
                continue
            try:
                self._joint_snapshot()
            except ValueError:
                continue
            now = self.get_clock().now().nanoseconds * 1e-9
            if now <= 0 or image_stamp <= 0 or not -.05 <= now - image_stamp <= .5:
                continue
            by_name = {name: index for index, name in enumerate(joints.name)}
            if any(name not in by_name for name in self.joint_names):
                continue
            joints_rad = tuple(joints.position[by_name[name]] for name in self.joint_names)
            detection = detect_square_marker(
                self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8'),
                [[info.k[0], info.k[1], info.k[2]],
                 [info.k[3], info.k[4], info.k[5]],
                 [info.k[6], info.k[7], info.k[8]]], info.d, side_m)
            if detection is None:
                self._preview(image, None)
                continue
            matrix, corners = detection
            self._preview(image, corners)
            flange = self._flange_pose(image.header.stamp)
            return CalibrationSample(
                joints_rad, tuple(map(tuple, flange)), tuple(map(tuple, matrix)),
                tuple(map(tuple, corners)), image_stamp, joint_stamp,
                image.header.frame_id, 'link6', tuple(info.k), tuple(info.d),
                (info.width, info.height)), sequence
        raise ValueError('Square marker was not visible in synchronized fresh frames')

    def _preview(self, image, corners):
        if self.args.show_image != 'true':
            return
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        if corners is not None:
            cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 2)
        cv2.imshow('Mount setup: fixed square marker', frame)
        cv2.waitKey(1)

    def _capture_pose(self, side_m, target, sequence):
        self._require_stopped_state(target)
        captures = []
        for _ in range(5):
            sample, sequence = self._capture_one(side_m, sequence)
            captures.append(sample)
        # Use the medoid of repeated observations at one settled arm pose.
        scores = []
        for candidate in captures:
            position = candidate.camera_from_marker
            score = sum(sum((position[row][3] - other.camera_from_marker[row][3]) ** 2
                            for row in range(3)) for other in captures)
            scores.append(score)
        return captures[min(range(len(scores)), key=scores.__getitem__)], sequence

    def _capture_start_pose(self, side_m):
        _, current, _ = self._joint_snapshot()
        sample, sequence = self._capture_pose(side_m, current, self.image_sequence)
        return current, sample, sequence

    def run(self):
        if self.args.operation == 'view':
            print('Камера открыта. Для остановки нажмите Ctrl+C.', flush=True)
            sequence = -1
            while rclpy.ok():
                self._spin_for(.02)
                with self.lock:
                    image = self.latest_image
                    current = self.image_sequence
                if image is not None and current != sequence:
                    sequence = current
                    self._preview(image, None)
            return
        if self.args.operation == 'remember-top':
            self._wait_for_joint_feedback()
            joint_message, joints, _ = self._joint_snapshot()
            self._require_stopped_state(joints)
            joint_message, joints, _ = self._joint_snapshot()
            pose = self._flange_pose(joint_message.header.stamp, 'world')
            self.config['mount_setup'] = remember_top_direction(pose, joints)
            save_yaml_atomic(self.args.output_file, self.config)
            print('Сохранён выбранный верх и все шесть измеренных углов.', flush=True)
            return
        if self.args.operation == 'set-top':
            setup = self.config.get('mount_setup', {})
            direction = setup.get('preferred_top_flange')
            if direction is None:
                raise ValueError('Сначала выполните ./run mount-setup --action remember-top')
            joint_message, current, _ = self._wait_for_joint_feedback()
            self._require_stopped_state(current)
            joint_message, current, _ = self._joint_snapshot()
            pose = self._flange_pose(joint_message.header.stamp, 'world')
            angle = top_joint6_goal(pose, current, direction, self.limits[5])
            target = current.copy()
            target[5] = angle
            tilt = top_alignment_error_deg(pose, current, direction, angle)
            print(f'J6: {math.degrees(current[5]):.1f}° → {math.degrees(angle):.1f}°; '
                  f'J1–J5 остаются в текущей позе; остаточный наклон {tilt:.1f}°.',
                  flush=True)
            self._move(target)
            return
        self._calibrate()

    def _calibrate(self):
        if self.args.marker_side_mm <= 0:
            raise ValueError('Marker side must be measured in millimeters and be positive')
        marker_side_m = self.args.marker_side_mm / 1000.0
        print(f'Не двигайте держатель телефона. Маркер должен быть виден; размер '
              f'измеренного квадрата {marker_side_m * 1000:g} mm.', flush=True)
        self._wait_for_joint_feedback()
        start, first, sequence = self._capture_start_pose(marker_side_m)
        self.calibration_start = start.copy()
        training_offsets, held_out_offsets = planned_joint_offsets_deg()
        training = [first]
        for number, (j5_deg, j6_deg) in enumerate(training_offsets[1:], start=2):
            target = start.copy()
            target[4] += math.radians(j5_deg)
            target[5] += math.radians(j6_deg)
            training.append(self._capture_at(target, marker_side_m, sequence))
            sequence = self._last_sequence
            print(f'Собрано обучающих поз: {number}/{len(training_offsets)}', flush=True)
        flange_from_optical = solve_hand_eye(training, marker_side_m)
        held_out = []
        for number, (j5_deg, j6_deg) in enumerate(held_out_offsets, start=1):
            target = start.copy()
            target[4] += math.radians(j5_deg)
            target[5] += math.radians(j6_deg)
            held_out.append(self._capture_at(target, marker_side_m, sequence))
            sequence = self._last_sequence
            print(f'Проверочных поз: {number}/{len(held_out_offsets)}', flush=True)
        report = validate_held_out(
            training, held_out, flange_from_optical,
            [training[0].camera_matrix[index:index + 3] for index in (0, 3, 6)],
            training[0].distortion, marker_side_m)
        print(f"Проверка: RMS {report['position_rms_m'] * 1000:.1f} mm, "
              f"{report['orientation_rms_deg']:.2f}°, "
              f"проекция {report['reprojection_rms_px']:.2f} px.", flush=True)
        self._move(start)
        updated = apply_optical_calibration(self.config, self.args.mode, flange_from_optical)
        save_yaml_atomic(self.args.output_file, updated)
        print('Проверка пройдена; flange-relative camera mount сохранён.', flush=True)

    def _capture_at(self, target, side_m, sequence):
        if any(not self.limits[index][0] <= target[index] <= self.limits[index][1]
               for index in (4, 5)):
            raise ValueError('Calibration pose is outside the active joint limits')
        _, current, _ = self._joint_snapshot()
        if any(abs(current[index] - self.calibration_start[index]) > math.radians(.1)
               for index in range(4)):
            raise ValueError('J1-J4 drifted during calibration; no next motion was sent')
        target = list(target)
        target[:4] = current[:4]
        self._move(target)
        sample, self._last_sequence = self._capture_pose(side_m, target, sequence)
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=['remember-top', 'set-top', 'calibrate', 'view'],
                        required=True)
    parser.add_argument('--mode', choices=['mono_cpu', 'stereo', 'rgbd'], required=True)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--output-file', default='/output/hardware.yaml')
    parser.add_argument('--marker-side-mm', type=float, default=45)
    parser.add_argument('--controller', default='')
    parser.add_argument('--show-image', choices=['true', 'false'], default='true')
    args, _ = parser.parse_known_args()
    signal_shutdown = _DeferredSignalShutdown()
    node = None
    try:
        signal_shutdown.install()
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = MountSetupNode(args, signal_shutdown)
        node.run()
    finally:
        signal_shutdown.begin_cleanup()
        try:
            if node is not None:
                if (node.active_goal is not None or
                        node.pending_goal_future is not None or
                        node.goal_submission_unknown):
                    confirmed = node._cancel_active_motion()
                    if not confirmed:
                        print('Остановка руки не подтверждена; проверьте её в UFACTORY Studio.',
                              flush=True)
                if args.show_image == 'true':
                    cv2.destroyAllWindows()
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            signal_shutdown.restore()


if __name__ == '__main__':
    main()
