# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Wait for an armed hardware monitor before enabling control components."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


class HardwareMonitorGate(Node):
    """Exit successfully only after the monitor publishes a healthy ready state."""

    def __init__(self):
        super().__init__('hardware_monitor_gate')
        self.ready = False
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription = self.create_subscription(
            Bool, '/face/hardware_monitor/ready', self._on_ready, qos)

    def _on_ready(self, message):
        self.ready = message.data


def main():
    rclpy.init()
    node = HardwareMonitorGate()
    deadline = time.monotonic() + 20.0
    try:
        while rclpy.ok() and not node.ready and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if not node.ready:
            raise RuntimeError('Hardware monitor did not report ready within 20 seconds')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
