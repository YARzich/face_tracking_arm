#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""
Drive bounded FACE scenarios and retain raw cycle timing/context records.

Start the isolated GenericSystem simulator first, with
`test_face_scenario:=disabled`, then run this script in the same ROS domain.
No hardware driver is started or contacted by this recorder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from diagnostic_msgs.msg import DiagnosticArray

from face_tracking_arm.msg import (
    ServoExecutionContext,
    ServoTrajectoryPublication,
)

from geometry_msgs.msg import PointStamped

from moveit_msgs.msg import ServoStatus

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import JointState

STAGES = (
    ('stationary_face', 10.0),
    ('dynamic_face', 10.0),
    ('edge_approach_withdraw', 10.0),
    ('obstacle_approach', 10.0),
    ('loss', 3.0),
    ('return', 8.0),
)
WARMUP_TIMEOUT_SEC = 120.0


def stage_sample(
    stage: str, elapsed: float
) -> tuple[float, float, float] | None:
    """Return a deterministic world-frame target for one scenario stage."""
    if stage == 'stationary_face':
        return 0.70, 0.0, 1.68
    if stage == 'dynamic_face':
        phase = 2.0 * math.pi * elapsed / 8.0
        return (
            0.70 + 0.035 * math.cos(phase),
            0.08 * math.sin(phase),
            1.68 + 0.025 * math.sin(phase * 0.5),
        )
    if stage == 'edge_approach_withdraw':
        phase = 2.0 * math.pi * elapsed / 10.0
        radius = 0.75 + 0.10 * math.cos(math.pi * elapsed / 10.0)
        return radius * math.cos(phase), radius * math.sin(phase), 1.68
    if stage == 'obstacle_approach':
        local = elapsed if elapsed <= 5.0 else elapsed - 5.0
        fraction = min(1.0, max(0.0, local / 5.0))
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        x = 0.70 - 0.18 * smooth if elapsed <= 5.0 else 0.52 + 0.18 * smooth
        return x, 0.12, 1.68
    if stage == 'return':
        return 0.70, 0.0, 1.68
    return None


class RuntimeRecorder(Node):
    """Publish deterministic FACE targets and record control-loop evidence."""

    def __init__(self, output: Path):
        """Create streams and subscribe to simulator state and context."""
        super().__init__('face_runtime_recorder')
        self.set_parameters(
            [Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )
        self.output = output
        self.raw_path = output / 'cycles.jsonl'
        self.target_path = output / 'targets.jsonl'
        self.joint_state_path = output / 'joint_states.jsonl'
        self.trajectory_path = output / 'trajectory_publications.jsonl'
        self.summary_path = output / 'summary.json'
        self.raw_stream = self.raw_path.open('x', encoding='utf-8')
        self.target_stream = self.target_path.open('x', encoding='utf-8')
        self.joint_state_stream = self.joint_state_path.open(
            'x', encoding='utf-8'
        )
        self.trajectory_stream = self.trajectory_path.open(
            'x', encoding='utf-8'
        )
        self.target_pub = self.create_publisher(
            PointStamped, '/face/center', 1
        )
        self.create_subscription(
            DiagnosticArray,
            '/tracking/diagnostics',
            self.on_tracking_diagnostics,
            10,
        )
        self.create_subscription(
            ServoExecutionContext,
            '/servo_node/execution_context',
            self.on_context,
            512,
        )
        self.create_subscription(
            ServoTrajectoryPublication,
            '/servo_node/trajectory_publications',
            self.on_trajectory_publication,
            64,
        )
        self.create_subscription(
            JointState, '/joint_states', self.on_joint_state, 128
        )
        self.create_subscription(
            ServoStatus, '/servo_node/status', self.on_servo_status, 10
        )
        self.create_subscription(
            DiagnosticArray,
            '/servo_node/diagnostics',
            self.on_servo_diagnostics,
            10,
        )
        self.ready = False
        self.started_at_ns: int | None = None
        self.wall_start = time.monotonic()
        self.last_servo_status: dict[str, object] | None = None
        self.last_diagnostics: dict[str, object] = {}
        self.values_by_stage: dict[str, list[int]] = {
            name: [] for name, _ in STAGES
        }
        self.active_values_by_stage: dict[str, list[int]] = {
            name: [] for name, _ in STAGES
        }
        self.counts_by_stage: dict[str, int] = {name: 0 for name, _ in STAGES}
        self.active_ticks_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.controller_mode_ticks: dict[str, dict[int, int]] = {
            name: {0: 0, 1: 0, 2: 0} for name, _ in STAGES
        }
        self.target_available_ticks: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.target_used_ticks: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.publication_advances_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.last_publication_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.deadlines_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.previous_tick_sequence: int | None = None
        self.tick_sequence_gaps = 0
        self.maximum_context_drops = 0
        self.maximum_publication_drops = 0
        self._obstacle_world_distances: list[float] = []
        self._obstacle_active_rows: list[int] = []
        self._obstacle_pairs: list[str] = []
        self._active_world_pair_distances: list[float] = []
        self._active_world_pairs: list[str] = []
        self.loss_ticks_after_grace = 0
        self.loss_active_ticks_after_grace = 0
        self.joint_state_samples_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.joint_state_malformed_samples = 0
        self.trajectory_publications_by_stage: dict[str, int] = {
            name: 0 for name, _ in STAGES
        }
        self.test_world_box = False
        self.stage_offsets: list[tuple[str, float, float]] = []
        cursor = 0.0
        for name, duration in STAGES:
            self.stage_offsets.append((name, cursor, cursor + duration))
            cursor += duration
        self.total_duration = cursor
        self.timer = self.create_timer(0.02, self.publish_target)

    def on_tracking_diagnostics(self, message: DiagnosticArray) -> None:
        """Wait for tracking readiness before starting the measured cycle."""
        for status in message.status:
            if status.name == 'face_tracking_arm/tracking':
                values = {item.key: item.value for item in status.values}
                if values.get('ready') == 'true':
                    self.ready = True

    def on_servo_status(self, message: ServoStatus) -> None:
        """Keep the latest Servo status for the next recorded cycle."""
        self.last_servo_status = {
            'code': int(message.code),
            'message': str(message.message),
        }

    def on_servo_diagnostics(self, message: DiagnosticArray) -> None:
        """Keep the latest diagnostics snapshot for the next cycle record."""
        self.last_diagnostics = {
            status.name: {
                'level': (
                    int(status.level[0])
                    if isinstance(status.level, bytes)
                    else int(status.level)
                ),
                'message': str(status.message),
                'values': {item.key: item.value for item in status.values},
            }
            for status in message.status
        }

    def stage_at(self, stamp_ns: int) -> tuple[str, float] | None:
        """Map a source ROS timestamp to its stage and elapsed offset."""
        if self.started_at_ns is None:
            return None
        elapsed = (stamp_ns - self.started_at_ns) / 1e9
        if elapsed < 0.0:
            return None
        for name, start, end in self.stage_offsets:
            if start <= elapsed < end:
                return name, elapsed - start
        return (
            ('complete', elapsed) if elapsed >= self.total_duration else None
        )

    def on_context(self, message: ServoExecutionContext) -> None:
        """Record the cycle's timing, modes, and provenance identifiers."""
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if self.started_at_ns is None:
            return
        stage_info = self.stage_at(stamp_ns)
        if stage_info is None:
            return
        stage, local_elapsed = stage_info
        if stage == 'complete':
            return
        duration_ns = int(message.cycle_duration_ns)
        self.values_by_stage[stage].append(duration_ns)
        self.counts_by_stage[stage] += 1
        controller_mode = int(message.controller_mode)
        servo_active = bool(message.face_tracking_published)
        self.controller_mode_ticks[stage][controller_mode] = (
            self.controller_mode_ticks[stage].get(controller_mode, 0) + 1
        )
        if message.target_available:
            self.target_available_ticks[stage] += 1
        if message.target_used_for_reference:
            self.target_used_ticks[stage] += 1
        if servo_active:
            self.active_ticks_by_stage[stage] += 1
            self.active_values_by_stage[stage].append(duration_ns)
        if stage == 'loss' and local_elapsed >= 0.60:
            self.loss_ticks_after_grace += 1
            if servo_active:
                self.loss_active_ticks_after_grace += 1
        latest_publication = int(message.latest_publication_id)
        if latest_publication > self.last_publication_by_stage[stage]:
            self.publication_advances_by_stage[stage] += 1
            self.last_publication_by_stage[stage] = latest_publication
        if duration_ns > 10_000_000:
            self.deadlines_by_stage[stage] += 1
        tick_sequence = int(message.tick_sequence)
        if (
            self.previous_tick_sequence is not None
            and tick_sequence > self.previous_tick_sequence + 1
        ):
            self.tick_sequence_gaps += (
                tick_sequence - self.previous_tick_sequence - 1
            )
        self.previous_tick_sequence = tick_sequence
        self.maximum_context_drops = max(
            self.maximum_context_drops, int(message.dropped_context_records)
        )
        self.maximum_publication_drops = max(
            self.maximum_publication_drops,
            int(message.dropped_publication_events),
        )
        world_distance = (
            message.minimum_world_distance_m
            if message.collision_distance_available
            else None
        )
        active_rows = int(message.active_collision_rows)
        closest_pair = str(message.closest_pair)
        active_world_pair = str(message.active_world_pair)
        active_world_pair_distance = (
            message.active_world_pair_distance_m
            if message.active_world_pair_available
            else None
        )
        if stage == 'obstacle_approach':
            if world_distance not in (None, 'unavailable', 'nan'):
                self._obstacle_world_distances.append(float(world_distance))
            if active_rows not in (None, 'unavailable'):
                self._obstacle_active_rows.append(int(active_rows))
            if closest_pair not in (None, '', 'unavailable'):
                self._obstacle_pairs.append(str(closest_pair))
            if active_world_pair_distance is not None:
                self._active_world_pair_distances.append(
                    float(active_world_pair_distance)
                )
                self._active_world_pairs.append(active_world_pair)
        record = {
            'control_stamp_ns': stamp_ns,
            'tick_sequence': tick_sequence,
            'cycle_duration_ns': duration_ns,
            'stage': stage,
            'stage_elapsed_sec': local_elapsed,
            'target_available': bool(message.target_available),
            'target_used_for_reference': bool(
                message.target_used_for_reference
            ),
            'target_mode': int(message.target_mode),
            'target_id': int(message.target_id),
            'active_publication_id': int(message.active_publication_id),
            'latest_publication_id': int(message.latest_publication_id),
            'tail_publication_id': int(message.braking_tail_publication_id),
            'hardware_epoch': int(message.hardware_epoch),
            'hardware_read_sequence': int(message.hardware_read_sequence),
            'controller_mode': controller_mode,
            'face_tracking_published': bool(message.face_tracking_published),
            'servo_active_tracking_tick': servo_active,
            'active_collision_rows': active_rows,
            'minimum_world_distance_m': world_distance,
            'closest_pair': closest_pair,
            'active_world_pair': active_world_pair,
            'active_world_pair_distance_m': active_world_pair_distance,
            'servo_status': self.last_servo_status,
            'servo_diagnostics': self.last_diagnostics,
        }
        self.raw_stream.write(json.dumps(record, separators=(',', ':')) + '\n')
        if tick_sequence % 100 == 0:
            self.raw_stream.flush()

    def on_joint_state(self, message: JointState) -> None:
        """Persist the actual simulator state with its sensor clock stamp."""
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        stage_info = self.stage_at(stamp_ns)
        if stage_info is None or stage_info[0] == 'complete':
            return
        if len(message.name) != len(message.position) or (
            message.velocity and len(message.velocity) != len(message.name)
        ):
            self.joint_state_malformed_samples += 1
            return
        stage, elapsed = stage_info
        record = {
            'source_stamp_ns': stamp_ns,
            'receipt_steady_ns': time.monotonic_ns(),
            'stage': stage,
            'stage_elapsed_sec': elapsed,
            'joint_names': list(message.name),
            'position_rad': [float(value) for value in message.position],
            'velocity_rad_s': [float(value) for value in message.velocity],
        }
        self.joint_state_stream.write(
            json.dumps(record, separators=(',', ':')) + '\n'
        )
        self.joint_state_samples_by_stage[stage] += 1
        if self.joint_state_samples_by_stage[stage] % 100 == 0:
            self.joint_state_stream.flush()

    def on_trajectory_publication(
        self, message: ServoTrajectoryPublication
    ) -> None:
        """Persist the complete checked trajectory and its source IDs."""
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        stage_info = self.stage_at(stamp_ns)
        if stage_info is None or stage_info[0] == 'complete':
            return
        stage, elapsed = stage_info
        trajectory = message.trajectory
        record = {
            'event_stamp_ns': stamp_ns,
            'trajectory_stamp_ns': (
                int(trajectory.header.stamp.sec) * 1_000_000_000
                + int(trajectory.header.stamp.nanosec)
            ),
            'event_frame_id': message.header.frame_id,
            'trajectory_frame_id': trajectory.header.frame_id,
            'stage': stage,
            'stage_elapsed_sec': elapsed,
            'publication_kind': int(message.publication_kind),
            'publication_id': int(message.publication_id),
            'stationary_time_ns': (
                int(message.stationary_time.sec) * 1_000_000_000
                + int(message.stationary_time.nanosec)
            ),
            'tail_start_index': int(message.tail_start_index),
            'hardware_epoch': int(message.hardware_epoch),
            'hardware_read_sequence': int(message.hardware_read_sequence),
            'control_tick_sequence': int(message.control_tick_sequence),
            'dropped_publication_events': int(
                message.dropped_publication_events
            ),
            'joint_names': list(trajectory.joint_names),
            'points': [
                {
                    'positions': [float(value) for value in point.positions],
                    'velocities': [float(value) for value in point.velocities],
                    'accelerations': [
                        float(value) for value in point.accelerations
                    ],
                    'effort': [float(value) for value in point.effort],
                    'time_from_start_ns': (
                        int(point.time_from_start.sec) * 1_000_000_000
                        + int(point.time_from_start.nanosec)
                    ),
                }
                for point in trajectory.points
            ],
        }
        self.trajectory_stream.write(
            json.dumps(record, separators=(',', ':')) + '\n'
        )
        self.trajectory_publications_by_stage[stage] += 1
        if self.trajectory_publications_by_stage[stage] % 10 == 0:
            self.trajectory_stream.flush()

    def publish_target(self) -> None:
        """Publish the target scheduled for the current simulation time."""
        now = self.get_clock().now()
        if not self.ready:
            if time.monotonic() - self.wall_start > WARMUP_TIMEOUT_SEC:
                self.get_logger().error(
                    'Timed out waiting for tracking readiness'
                )
                rclpy.shutdown()
            return
        if self.started_at_ns is None:
            self.started_at_ns = now.nanoseconds
            self.get_logger().info(
                'Tracking ready; beginning measured FACE cycle'
            )
        elapsed = (now.nanoseconds - self.started_at_ns) / 1e9
        if elapsed >= self.total_duration:
            self.finish()
            return
        stage = next(
            (
                name
                for name, start, end in self.stage_offsets
                if start <= elapsed < end
            ),
            None,
        )
        if stage is None:
            return
        sample = stage_sample(
            stage,
            elapsed
            - next(
                start for name, start, _ in self.stage_offsets if name == stage
            ),
        )
        target_record = {
            'stamp_ns': now.nanoseconds,
            'stage': stage,
            'frame_id': 'world',
            'point': (
                None
                if sample is None
                else {'x': sample[0], 'y': sample[1], 'z': sample[2]}
            ),
        }
        self.target_stream.write(
            json.dumps(target_record, separators=(',', ':')) + '\n'
        )
        if int(elapsed * 50) % 50 == 0:
            self.target_stream.flush()
        if sample is None:
            return
        message = PointStamped()
        message.header.stamp = now.to_msg()
        message.header.frame_id = 'world'
        message.point.x, message.point.y, message.point.z = sample
        self.target_pub.publish(message)

    def finish(self) -> None:
        """Flush evidence and write stage coverage and timing summaries."""
        self.timer.cancel()
        self.raw_stream.flush()
        self.joint_state_stream.flush()
        self.trajectory_stream.flush()
        summary = {}
        for name, _ in STAGES:

            def distribution_for(samples: list[int]) -> dict[str, int] | None:
                values = sorted(samples)
                if not values:
                    return None

                def percentile(fraction: float) -> int:
                    index = min(
                        len(values) - 1, math.ceil(fraction * len(values)) - 1
                    )
                    return values[index]

                return {
                    'p50_ns': percentile(0.50),
                    'p95_ns': percentile(0.95),
                    'p99_ns': percentile(0.99),
                    'max_ns': values[-1],
                }

            summary[name] = {
                'cycle_count': self.counts_by_stage[name],
                'active_face_ticks': self.active_ticks_by_stage[name],
                'publication_advances': self.publication_advances_by_stage[
                    name
                ],
                'deadline_misses_over_10ms': self.deadlines_by_stage[name],
                'cycle_duration': distribution_for(self.values_by_stage[name]),
                'active_face_cycle_duration': distribution_for(
                    self.active_values_by_stage[name]
                ),
                'controller_mode_ticks': {
                    'tracking': self.controller_mode_ticks[name].get(0, 0),
                    'braking': self.controller_mode_ticks[name].get(1, 0),
                    'waiting': self.controller_mode_ticks[name].get(2, 0),
                },
                'target_available_ticks': self.target_available_ticks[name],
                'target_used_for_reference_ticks': self.target_used_ticks[
                    name
                ],
            }
        package_root = Path(__file__).resolve().parent.parent
        source_revision = (
            subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=package_root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or 'unavailable'
        )
        diff = subprocess.run(
            ['git', 'diff', '--binary'],
            cwd=package_root,
            check=False,
            capture_output=True,
        ).stdout
        patch_hash = hashlib.sha256(diff).hexdigest()
        component_path = (
            package_root.parent.parent
            / 'install/face_tracking_arm/lib/'
            'libcollision_aware_servo_component.so'
        )
        scene_path = package_root / 'worlds/lite6_table.sdf'

        def hash_file(path: Path) -> str:
            if not path.is_file():
                return 'unavailable'
            return hashlib.sha256(path.read_bytes()).hexdigest()

        coverage = {}
        for name, _ in STAGES:
            if name == 'loss':
                continue
            cycle_count = self.counts_by_stage[name]
            active_count = self.active_ticks_by_stage[name]
            coverage[name] = (
                cycle_count > 0
                and active_count >= 100
                and active_count / cycle_count >= 0.80
                and self.publication_advances_by_stage[name]
                >= active_count // 2
            )
        coverage['loss'] = (
            self.loss_ticks_after_grace > 0
            and self.loss_active_ticks_after_grace == 0
        )
        obstacle_evidence = any(
            distance <= 0.20 for distance in self._active_world_pair_distances
        )
        result = {
            'clock': (
                'ROS control stamp; cycle durations measured by steady clock'
            ),
            'runtime_complete': all(coverage.values())
            and obstacle_evidence
            and self.maximum_context_drops == 0
            and self.maximum_publication_drops == 0
            and self.tick_sequence_gaps == 0,
            'incomplete_reasons': [
                name for name, covered in coverage.items() if not covered
            ]
            + (
                []
                if obstacle_evidence
                else ['no active world collision constraint within 0.20 m']
            )
            + (
                ['execution context sequence gaps detected']
                if self.tick_sequence_gaps
                else []
            ),
            'total_duration_sec': self.total_duration,
            'stages': summary,
            'tick_sequence_gaps': self.tick_sequence_gaps,
            'loss_ticks_after_0_60s_grace': self.loss_ticks_after_grace,
            'loss_active_ticks_after_grace': (
                self.loss_active_ticks_after_grace
            ),
            'maximum_context_drops': self.maximum_context_drops,
            'maximum_publication_drops': self.maximum_publication_drops,
            'joint_state_samples_by_stage': self.joint_state_samples_by_stage,
            'trajectory_publications_by_stage': (
                self.trajectory_publications_by_stage
            ),
            'malformed_joint_state_samples': (
                self.joint_state_malformed_samples
            ),
            'obstacle_minimum_world_distance_m': (
                min(self._obstacle_world_distances)
                if self._obstacle_world_distances
                else None
            ),
            'obstacle_world_distance_samples': len(
                self._obstacle_world_distances
            ),
            'obstacle_active_collision_rows_max': (
                max(self._obstacle_active_rows)
                if self._obstacle_active_rows
                else None
            ),
            'obstacle_closest_pair_samples': len(self._obstacle_pairs),
            'obstacle_active_world_pair_samples': len(
                self._active_world_pairs
            ),
            'obstacle_active_world_pairs': sorted(
                set(self._active_world_pairs)
            ),
            'inputs': {
                'scenario': (
                    'stationary/dynamic/edge/table-approach/loss/return'
                ),
                'initial_test_world_box': self.test_world_box,
                'test_world_box': (
                    {
                        'planning_scene_object_id': 'round_table',
                        'shape': 'box',
                        'dimensions_m': [0.10, 0.10, 0.10],
                        'center_world_m': [0.50, 0.00, 1.15],
                        'yaw_rad': 0.0,
                    }
                    if self.test_world_box
                    else None
                ),
                'target_frame': 'world',
                'target_rate_hz': 50,
                'scene_world': str(scene_path),
                'scene_sha256': hash_file(scene_path),
                'launch': 'tracking_sim.launch.py with Gazebo GenericSystem',
                'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', 'default'),
            },
            'source': {
                'git_revision': source_revision,
                'working_tree_diff_sha256': patch_hash,
                'component_library_sha256': hash_file(component_path),
            },
            'raw_cycles': self.raw_path.name,
            'target_inputs': self.target_path.name,
            'actual_joint_states': self.joint_state_path.name,
            'trajectory_publications': self.trajectory_path.name,
        }
        self.summary_path.write_text(
            json.dumps(result, indent=2) + '\n', encoding='utf-8'
        )
        self.get_logger().info(
            'FACE runtime recording complete: {}'.format(self.summary_path)
        )
        rclpy.shutdown()

    def close(self) -> None:
        """Close all recorder output streams."""
        if not self.raw_stream.closed:
            self.raw_stream.close()
        if not self.target_stream.closed:
            self.target_stream.close()
        if not self.joint_state_stream.closed:
            self.joint_state_stream.close()
        if not self.trajectory_stream.closed:
            self.trajectory_stream.close()


def main() -> int:
    """Parse output options and run the ROS recorder node."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='new output directory for raw records and summary',
    )
    parser.add_argument(
        '--test-world-box',
        action='store_true',
        help='record the initial test-only box substituted for the table',
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('output directory already exists: {}'.format(output))
    output.mkdir(parents=True)
    rclpy.init()
    recorder = RuntimeRecorder(output)
    recorder.test_world_box = args.test_world_box
    try:
        rclpy.spin(recorder)
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
