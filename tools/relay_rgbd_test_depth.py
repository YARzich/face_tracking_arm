#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""
Test-only depth relay with source-stamp HOLD drop windows.

Example: python3 tools/relay_rgbd_test_depth.py --drop-window 95:95.3 \
    --drop-window 97:97.4 --output-audit-jsonl /tmp/depth-relay.jsonl \
    --max-wall-sec 600 --end-sim-sec 150
"""

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def parse_window(value):
    """Parse a half-open START:END interval into integer nanoseconds."""
    try:
        start, end = (int(float(part) * 1_000_000_000)
                      for part in value.split(':'))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError('window must be START:END seconds') from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError('window requires 0 <= START < END')
    return start, end


def stamp_ns(message):
    """Return a ROS header stamp as integer nanoseconds."""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


class DepthRelay(Node):
    """Pass source depth messages except for configured sim-time windows."""

    def __init__(self, args):
        super().__init__('rgbd_test_depth_relay')
        self.args = args
        self.windows = args.drop_window
        self.output = Path(args.output_audit_jsonl).expanduser()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.audit = self.output.open('w', encoding='utf-8')
        self.started = time.monotonic()
        self.counts = {'passed': 0, 'dropped': 0, 'cutoff': 0}
        self.stopped = False
        self.publisher = self.create_publisher(Image, '/face_test/test_depth',
                                               qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            Image, '/face_test/depth/image_raw', self.on_image,
            qos_profile_sensor_data)
        self.timer = self.create_timer(.1, self.check_wall,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.write({'event': 'start', 'windows_ns': self.windows,
                    'end_sim_sec': args.end_sim_sec, 'max_wall_sec': args.max_wall_sec})

    def write(self, row):
        """Append and flush one audit record."""
        self.audit.write(json.dumps(row, separators=(',', ':')) + '\n')
        self.audit.flush()

    def on_image(self, message):
        """Publish or drop the original image based only on its source stamp."""
        stamp = stamp_ns(message)
        seconds = stamp / 1_000_000_000
        base = {'stamp_ns': stamp, 'stamp_sec': seconds,
                'frame_id': message.header.frame_id}
        if self.args.end_sim_sec is not None and seconds >= self.args.end_sim_sec:
            self.counts['cutoff'] += 1
            self.write(dict(base, decision='end_sim_cutoff'))
            self.stop('end_sim_sec')
            return
        dropped = any(start <= stamp < end for start, end in self.windows)
        decision = 'drop' if dropped else 'pass'
        self.counts['dropped' if dropped else 'passed'] += 1
        self.write(dict(base, decision=decision, width=message.width,
                        height=message.height, encoding=message.encoding))
        if not dropped:
            self.publisher.publish(message)

    def check_wall(self):
        """Stop after the wall-clock limit, independently of sim time."""
        if time.monotonic() - self.started >= self.args.max_wall_sec:
            self.stop('max_wall_sec')

    def stop(self, reason):
        """Request shutdown once and persist the terminal count summary."""
        if self.stopped:
            return
        self.stopped = True
        self.write({'event': 'summary', 'reason': reason, 'counts': self.counts,
                    'wall_sec': round(time.monotonic() - self.started, 3)})
        rclpy.shutdown()


def main():
    """Run the relay until a configured sim-time or wall-time cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drop-window', action='append', type=parse_window,
                        default=[], metavar='START:END',
                        help='half-open sim-time seconds interval; repeatable')
    parser.add_argument('--output-audit-jsonl', required=True)
    parser.add_argument('--max-wall-sec', type=float, default=600.0)
    parser.add_argument('--end-sim-sec', type=float)
    args = parser.parse_args()
    if args.max_wall_sec <= 0 or (args.end_sim_sec is not None and args.end_sim_sec <= 0):
        parser.error('time limits must be positive')
    rclpy.init()
    relay = DepthRelay(args)
    try:
        rclpy.spin(relay)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not relay.stopped:
            relay.write({'event': 'summary', 'reason': 'external_shutdown',
                         'counts': relay.counts,
                         'wall_sec': round(time.monotonic() - relay.started, 3)})
        relay.audit.close()
        relay.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
