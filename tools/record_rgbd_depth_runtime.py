#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Bounded passive RGB-D runtime capture for depth and TF investigations."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import time

from diagnostic_msgs.msg import DiagnosticArray
from face_tracking_arm.msg import PerceptionStatus
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy)
from rosgraph_msgs.msg import Clock as ClockMsg
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection2DArray


LOSS_STATES = {
    PerceptionStatus.NO_FACE, PerceptionStatus.FACE_NO_DEPTH,
    PerceptionStatus.DEPTH_STREAM_LOST, PerceptionStatus.SYNC_ERROR,
    PerceptionStatus.INVALID_INPUT, PerceptionStatus.TF_UNAVAILABLE,
}


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_sec(stamp):
    return stamp_ns(stamp) * 1e-9


def xyz(point):
    return [float(point.x), float(point.y), float(point.z)]


def json_safe(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def image_array(message):
    """View common scalar depth encodings while honoring ROS row stride."""
    types = {'16UC1': ('u2', 2), 'mono16': ('u2', 2),
             '32FC1': ('f4', 4), '64FC1': ('f8', 8)}
    if message.encoding not in types:
        return None
    dtype, size = types[message.encoding]
    dtype = np.dtype(('>' if message.is_bigendian else '<') + dtype)
    if message.step < message.width * size or len(message.data) < message.step * message.height:
        return None
    return np.ndarray((message.height, message.width), dtype=dtype,
                      buffer=message.data, strides=(message.step, size))


def detection_record(detection):
    box = detection.bbox
    return {'id': str(detection.id), 'x': float(box.center.position.x),
            'y': float(box.center.position.y), 'width': float(box.size_x),
            'height': float(box.size_y),
            'hypotheses': [{'class_id': h.hypothesis.class_id,
                            'score': float(h.hypothesis.score)}
                           for h in detection.results]}


def diagnostics_record(message):
    """Serialize DiagnosticStatus, including Jazzy's bytes-backed uint8 level."""
    return {'stamp_sec': stamp_sec(message.header.stamp),
            'statuses': [{'name': s.name, 'level': uint8_value(s.level), 'message': s.message,
                          'values': {v.key: v.value for v in s.values}}
                         for s in message.status]}


def uint8_value(value):
    """Convert a ROS uint8 scalar or its one-byte Python representation."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        if len(value) != 1:
            raise ValueError(f'Expected one byte for uint8, got {len(value)}')
        return value[0]
    return int(value)


class Recorder(Node):
    def __init__(self, args):
        super().__init__('rgbd_depth_runtime_recorder')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.args = args
        self.output = Path(args.output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.samples_dir = self.output / 'samples'
        self.samples_dir.mkdir()
        self.events = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        self.counts = Counter()
        self.losses = Counter()
        self.clock_sec = None
        self.started_sim = None
        self.started_wall = time.monotonic()
        self.last_sample_sec = None
        self.scenario_elapsed = None
        self.scenario_elapsed_first = None
        self.scenario_elapsed_last = None
        self.scenario_elapsed_gaps = []
        self.selected_target_id = None
        self.last_sampled_detection_stamp = None
        self.messages = {k: deque(maxlen=8) for k in ('rgb', 'depth', 'info', 'detections')}
        self.selected_ids = Counter()
        self.output_summary = None
        self.create_subscription(ClockMsg, '/clock', self.on_clock, 10)
        self.create_subscription(Image, '/face_test/image_raw',
                                 lambda m: self.on_image('rgb', m), qos_profile_sensor_data)
        self.create_subscription(Image, '/face_test/depth/image_raw',
                                 lambda m: self.on_image('depth', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/face_test/camera_info',
                                 lambda m: self.on_cached('info', m), qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/face_test/detections',
                                 lambda m: self.on_cached('detections', m),
                                 qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/face_test/depth_status',
                                 lambda m: self.on_diag('depth_status', m),
                                 qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/face_test/scenario',
                                 lambda m: self.on_diag('scenario', m), qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/tracking/diagnostics',
                                 lambda m: self.on_diag('tracking_diagnostics', m),
                                 qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/servo_node/diagnostics',
                                 lambda m: self.on_diag('servo_diagnostics', m),
                                 qos_profile_sensor_data)
        self.create_subscription(PerceptionStatus, '/face/perception_status',
                                 self.on_perception, 10)
        self.create_subscription(PointStamped, '/face/center', self.on_center,
                                 qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/tf', self.on_tf, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/tf_static', self.on_tf,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(.05, self.tick,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        (self.output / 'manifest.json').write_text(json.dumps({
            'tool': 'record_rgbd_depth_runtime.py', 'scenario': args.scenario,
            'duration_sim_sec': args.duration_sim_sec, 'max_wall_sec': args.max_wall_sec,
            'sample_hz': args.sample_hz,
            'capture_window_scenario_elapsed_sec': list(args.capture_window),
            'topics': {'rgb': '/face_test/image_raw',
                       'camera_info': '/face_test/camera_info',
                       'depth': '/face_test/depth/image_raw',
                       'detections': '/face_test/detections',
                       'depth_status': '/face_test/depth_status',
                       'tracking_diagnostics': '/tracking/diagnostics',
                       'servo_diagnostics': '/servo_node/diagnostics',
                       'perception': '/face/perception_status',
                       'scenario': '/face_test/scenario', 'actual_center': '/face/center',
                       'tf': ['/tf', '/tf_static']},
            'sample_format': ('compressed NPZ with original ROS image bytes and metadata; '
                              'no conversion'),
            'raw_receipt_metadata': ('one JSONL event per RGB, depth, and CameraInfo callback; '
                                     'source stamp, frame ID, arrival /clock and monotonic time'),
            'roi_stats': ('finite positive scalar depth in detection box; '
                          'input-unit scale recorded')},
            indent=2) + '\n', encoding='utf-8')

    def write(self, kind, record):
        record['kind'] = kind
        record.setdefault('arrival_sim_sec', self.clock_sec)
        self.events.write(json.dumps(record, separators=(',', ':'), default=json_safe,
                                     allow_nan=False) + '\n')
        self.events.flush()
        self.counts[kind] += 1

    def on_clock(self, message):
        self.clock_sec = stamp_sec(message.clock)
        if self.started_sim is None:
            self.started_sim = self.clock_sec

    def on_image(self, kind, message):
        self.messages[kind].append(message)
        self.counts[kind + '_received'] += 1
        self.write('raw_receipt', {
            'stream': kind, 'source_stamp_sec': stamp_sec(message.header.stamp),
            'frame_id': message.header.frame_id,
            'arrival_monotonic_sec': time.monotonic()})

    def on_cached(self, kind, message):
        self.messages[kind].append(message)
        self.counts[kind + '_received'] += 1
        if kind == 'info':
            self.write('raw_receipt', {
                'stream': kind, 'source_stamp_sec': stamp_sec(message.header.stamp),
                'frame_id': message.header.frame_id,
                'arrival_monotonic_sec': time.monotonic()})
        if kind == 'detections':
            self.write(kind, {'source_stamp_sec': stamp_sec(message.header.stamp),
                              'frame_id': message.header.frame_id,
                              'detections': [detection_record(d) for d in message.detections]})

    def on_diag(self, kind, message):
        record = diagnostics_record(message)
        if kind == 'scenario':
            for status in record['statuses']:
                if status['name'] != 'face_appearances':
                    continue
                try:
                    elapsed = float(status['values']['elapsed_sec'])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(elapsed):
                    if self.scenario_elapsed_last is not None:
                        delta = elapsed - self.scenario_elapsed_last
                        if delta > .25:
                            self.scenario_elapsed_gaps.append(delta)
                    if self.scenario_elapsed_first is None:
                        self.scenario_elapsed_first = elapsed
                    self.scenario_elapsed_last = elapsed
                    self.scenario_elapsed = elapsed
                    self.counts['scenario_source_samples'] += 1
        self.write(kind, record)

    def on_perception(self, message):
        """Record perception outcome and the selector's current identity."""
        state = int(message.state)
        state_name = {int(getattr(PerceptionStatus, name)): name for name in
                      ('WAIT', 'OK', 'NO_FACE', 'FACE_NO_DEPTH', 'DEPTH_STREAM_LOST',
                       'SYNC_ERROR', 'INVALID_INPUT', 'TF_UNAVAILABLE')}.get(state, str(state))
        self.selected_target_id = int(message.target_id)
        self.selected_ids[str(self.selected_target_id)] += 1
        self.write('perception', {'stamp_sec': stamp_sec(message.header.stamp),
                                  'source_stamp_sec': stamp_sec(message.source_stamp),
                                  'state': state, 'state_name': state_name,
                                  'reason': message.reason, 'target_id': int(message.target_id)})
        if state in LOSS_STATES:
            self.losses[f'{state_name}: {message.reason or "(без причины)"}'] += 1

    def on_center(self, message):
        """Record the actual stamped point consumed by the arm."""
        source = stamp_sec(message.header.stamp)
        self.write('actual_center', {'source_stamp_sec': source,
                                     'frame_id': message.header.frame_id,
                                     'point_m': xyz(message.point),
                                     'age_at_arrival_sec': (self.clock_sec - source)
                                     if self.clock_sec is not None else None})

    def on_tf(self, message):
        """Record each received TF edge with its source stamp and transform."""
        for transform in message.transforms:
            tr = transform.transform
            source = stamp_sec(transform.header.stamp)
            self.write('tf', {'source_stamp_sec': source,
                              'frame_id': transform.header.frame_id,
                              'child_frame_id': transform.child_frame_id,
                              'translation_m': [float(tr.translation.x), float(tr.translation.y),
                                                float(tr.translation.z)],
                              'rotation_xyzw': [float(tr.rotation.x), float(tr.rotation.y),
                                                float(tr.rotation.z), float(tr.rotation.w)],
                              'age_at_arrival_sec': (self.clock_sec - source)
                              if self.clock_sec is not None and source > 0 else None})

    @staticmethod
    def exact_stamp(items, source_ns):
        """Return a buffered message with the exact source stamp."""
        return next((m for m in reversed(items)
                     if stamp_ns(m.header.stamp) == source_ns), None)

    def sample(self, sim_sec):
        detections = sorted(self.messages['detections'],
                            key=lambda m: stamp_ns(m.header.stamp), reverse=True)
        matched = None
        for detection in detections:
            source_ns = stamp_ns(detection.header.stamp)
            if (self.last_sampled_detection_stamp is not None and
                    source_ns <= self.last_sampled_detection_stamp):
                continue
            rgb = self.exact_stamp(self.messages['rgb'], source_ns)
            depth = self.exact_stamp(self.messages['depth'], source_ns)
            info = self.exact_stamp(self.messages['info'], source_ns)
            if rgb is not None and depth is not None and info is not None:
                matched = (detection, rgb, depth, info)
                break
        if matched is None:
            has_new_detection = any(
                self.last_sampled_detection_stamp is None or
                stamp_ns(d.header.stamp) > self.last_sampled_detection_stamp
                for d in detections)
            reason = ('no_detection' if not detections else
                      'no_new_detection' if not has_new_detection else
                      'no_exact_rgb_depth_camera_info_tuple')
            self.counts[f'sample_not_synced_{reason}'] += 1
            newest_detection_ns = (stamp_ns(detections[0].header.stamp)
                                   if detections else None)
            self.write('sample_not_synced', {
                'requested_sim_sec': sim_sec, 'reason': reason,
                'detection_stamps_ns': [stamp_ns(m.header.stamp) for m in detections],
                'newest_detection_stamp_ns': newest_detection_ns,
                'available_rgb_stamps_ns': [stamp_ns(m.header.stamp)
                                            for m in self.messages['rgb']],
                'available_depth_stamps_ns': [stamp_ns(m.header.stamp)
                                              for m in self.messages['depth']],
                'available_camera_info_stamps_ns': [stamp_ns(m.header.stamp)
                                                    for m in self.messages['info']]})
            return
        detections, rgb, depth, info = matched
        source_ns = stamp_ns(detections.header.stamp)
        self.last_sampled_detection_stamp = source_ns
        rgb_delta = 0.0
        info_delta = 0.0
        ddelta = 0.0
        roi_stats = []
        depth_view = image_array(depth)
        if detections and depth_view is not None:
            for d in detections.detections:
                box = detection_record(d)
                x0 = max(0, int(math.floor(box['x'] - box['width'] / 2)))
                x1 = min(depth.width, int(math.ceil(box['x'] + box['width'] / 2)))
                y0 = max(0, int(math.floor(box['y'] - box['height'] / 2)))
                y1 = min(depth.height, int(math.ceil(box['y'] + box['height'] / 2)))
                region = depth_view[y0:y1, x0:x1]
                values = (region[np.isfinite(region) & (region > 0)]
                          if region.size else np.array([]))
                scale = .001 if depth.encoding in ('16UC1', 'mono16') else 1.0
                roi_stats.append({'id': box['id'], 'pixel_count': int(region.size),
                                  'valid_positive_count': int(values.size),
                                  'valid_fraction': float(values.size / region.size)
                                  if region.size else 0.0,
                                  'median_m': float(np.median(values) * scale)
                                  if values.size else None})
        base = f'{stamp_ns(rgb.header.stamp)}'
        path = self.samples_dir / f'{base}.npz'
        np.savez_compressed(path,
                            rgb=np.frombuffer(bytes(rgb.data), dtype=np.uint8),
                            depth=np.frombuffer(bytes(depth.data), dtype=np.uint8))
        self.write('sample', {'file': str(path.relative_to(self.output)),
                              'rgb': self.image_metadata(rgb), 'depth': self.image_metadata(depth),
                              'camera_info': self.info_metadata(info) if info else None,
                              'detection_stamp_sec': stamp_sec(detections.header.stamp),
                              'rgb_detection_delta_sec': 0.0,
                              'depth_detection_delta_sec': 0.0,
                              'camera_info_detection_delta_sec': 0.0,
                              'rgb_depth_delta_sec': rgb_delta,
                              'rgb_age_at_capture_sec': max(
                                  0.0, self.clock_sec - stamp_sec(rgb.header.stamp))
                              if self.clock_sec is not None else None,
                              'depth_age_at_capture_sec': max(
                                  0.0, self.clock_sec - stamp_sec(depth.header.stamp))
                              if self.clock_sec is not None else None,
                              'camera_info_delta_sec': info_delta,
                              'detection_delta_sec': ddelta,
                              'selected_id': self.selected_target_id, 'roi_depth': roi_stats})
        self.counts['samples_saved'] += 1

    @staticmethod
    def image_metadata(message):
        return {'stamp_sec': stamp_sec(message.header.stamp),
                'frame_id': message.header.frame_id, 'width': int(message.width),
                'height': int(message.height), 'encoding': message.encoding,
                'is_bigendian': int(message.is_bigendian), 'step': int(message.step),
                'data_bytes': len(message.data)}

    @staticmethod
    def info_metadata(message):
        return {'stamp_sec': stamp_sec(message.header.stamp),
                'frame_id': message.header.frame_id, 'width': int(message.width),
                'height': int(message.height), 'distortion_model': message.distortion_model,
                'd': list(message.d), 'k': list(message.k), 'r': list(message.r),
                'p': list(message.p), 'binning_x': int(message.binning_x),
                'binning_y': int(message.binning_y),
                'roi': {'x_offset': int(message.roi.x_offset),
                        'y_offset': int(message.roi.y_offset),
                        'height': int(message.roi.height), 'width': int(message.roi.width),
                        'do_rectify': bool(message.roi.do_rectify)}}

    def tick(self):
        wall_elapsed = time.monotonic() - self.started_wall
        sim_elapsed = (None if self.started_sim is None or self.clock_sec is None else
                       self.clock_sec - self.started_sim)
        if self.clock_sec is not None and self.scenario_elapsed is not None:
            start, end = self.args.capture_window
            in_window = start <= self.scenario_elapsed <= end
            due = (self.last_sample_sec is None or
                   self.scenario_elapsed - self.last_sample_sec >= 1.0 / self.args.sample_hz)
            if in_window and due:
                self.sample(self.clock_sec)
                self.last_sample_sec = self.scenario_elapsed
        if wall_elapsed >= self.args.max_wall_sec:
            self.finish('max_wall_sec')
        elif sim_elapsed is not None and sim_elapsed >= self.args.duration_sim_sec:
            self.finish('duration_sim_sec')

    def finish(self, reason):
        if self.output_summary is not None:
            return
        self.output_summary = True
        expected = max(0, int((self.args.capture_window[1] - self.args.capture_window[0])
                              * self.args.sample_hz) + 1)
        covered = (self.scenario_elapsed_last - self.scenario_elapsed_first
                   if self.scenario_elapsed_last is not None and
                   self.scenario_elapsed_first is not None else None)
        summary = {'stop_reason': reason,
                   'sim_sec_at_stop': self.clock_sec,
                   'sim_duration_observed_sec': (self.clock_sec - self.started_sim)
                   if self.clock_sec is not None and self.started_sim is not None else None,
                   'wall_duration_sec': time.monotonic() - self.started_wall,
                   'scenario_source': {'first_elapsed_sec': self.scenario_elapsed_first,
                                       'last_elapsed_sec': self.scenario_elapsed_last,
                                       'covered_elapsed_sec': covered,
                                       'gaps_over_250ms_sec': self.scenario_elapsed_gaps,
                                       'message_count': self.counts['scenario_source_samples'],
                                       'requested_duration_sim_sec': self.args.duration_sim_sec},
                   'expected_window_samples_approx': expected,
                   'counts': dict(self.counts),
                   'loss_status_counts_by_state_and_reason': dict(self.losses),
                   'selected_target_id_counts': dict(self.selected_ids),
                   'coverage': {'sample_completeness':
                                self.counts['samples_saved'] / expected if expected else None,
                                'rgb_received': self.counts['rgb_received'],
                                'depth_received': self.counts['depth_received'],
                                'camera_info_received': self.counts['info_received'],
                                'detection_messages': self.counts['detections']},
                   'limits': {'requested_duration_sim_sec': self.args.duration_sim_sec,
                              'requested_max_wall_sec': self.args.max_wall_sec,
                              'capture_window_scenario_elapsed_sec':
                              list(self.args.capture_window)}}
        with (self.output / 'summary.json').open('w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, default=json_safe, allow_nan=False)
            stream.write('\n')
        self.events.close()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='new output directory (normally under ignored runtime log/)')
    parser.add_argument('--scenario', default='unknown')
    parser.add_argument('--duration-sim-sec', type=float, required=True)
    parser.add_argument('--max-wall-sec', type=float, default=240.0)
    parser.add_argument('--sample-hz', type=float, default=1.0)
    parser.add_argument('--capture-depth-window', default='26:36', metavar='START:END',
                        help='scenario elapsed interval for RGB/depth capture (default: 26:36)')
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in
               (args.duration_sim_sec, args.max_wall_sec, args.sample_hz)):
        parser.error('duration, max wall time, and sample rate must be finite and positive')
    if args.sample_hz > 1.0:
        parser.error('sample rate is bounded to 1 Hz')
    try:
        args.capture_window = tuple(float(value) for value in args.capture_depth_window.split(':'))
    except ValueError:
        parser.error('capture-depth-window must be START:END in scenario seconds')
    if (len(args.capture_window) != 2 or
            not all(math.isfinite(value) and value >= 0 for value in args.capture_window) or
            args.capture_window[1] <= args.capture_window[0]):
        parser.error('capture-depth-window must be an increasing finite START:END interval')
    rclpy.init()
    node = Recorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if not node.output_summary:
            node.finish('interrupted')
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if not node.output_summary:
            node.finish('shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
