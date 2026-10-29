# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Exercise TF delivery while a real ROS image callback is occupied."""

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import face_tracking_perception.depth_node as depth_node_module  # noqa: E402,I100,I201
from face_tracking_perception.depth_node import (  # noqa: E402,I100,I201
    create_face_position_executor,
    FacePositionNode,
)
from face_tracking_perception.detectors import Face  # noqa: E402,I100,I201


class LoadedImageNode(Node):
    """Listen for TF while an image callback holds the executor worker."""

    def __init__(self):
        """Create the listener and an intentionally busy image callback."""
        super().__init__('continuity_loaded_image')
        self.buffer = Buffer(node=self)
        self.listener = TransformListener(self.buffer, self)
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.lookup_error = None
        self.lookup_ok_during_load = False
        self.create_subscription(Image, '/continuity/image', self.on_image, 1)

    def on_image(self, image):
        """Wait for the source-stamped transform during simulated inference."""
        self.entered.set()
        deadline = time.monotonic() + .18
        while time.monotonic() < deadline:
            if self.buffer.can_transform(
                    'world', image.header.frame_id,
                    Time.from_msg(image.header.stamp)):
                self.lookup_ok_during_load = True
                break
            time.sleep(.001)
        try:
            self.buffer.lookup_transform(
                'world', image.header.frame_id,
                Time.from_msg(image.header.stamp))
        except TransformException as error:
            self.lookup_error = error
        finally:
            self.finished.set()


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Initialize ROS once for this integration-test module."""
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.mark.parametrize('arrival_delay', [.01, .045, .12])
def test_production_node_receives_exact_tf_during_inference(monkeypatch, arrival_delay):
    """Verify the owned production TF executor updates the buffer during inference."""
    detector_entered = threading.Event()
    detector_saw_tf = threading.Event()
    node_holder = {}

    def detect(_image):
        detector_entered.set()
        node = node_holder['node']
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            if node.tf_buffer.can_transform(
                    'world', 'camera_optical', node_holder['stamp']):
                detector_saw_tf.set()
                break
            time.sleep(.001)
        return [Face(270, 180, 80, 90, .95)]

    monkeypatch.setattr(
        depth_node_module, 'create_detector',
        lambda *args, **kwargs: SimpleNamespace(detect=detect))
    image_node = FacePositionNode()
    node_holder['node'] = image_node
    observed = []

    def record_observation(_selected, reason):
        observed.append(reason)

    image_node.publish_observation = record_observation
    publisher_node = Node('continuity_tf_publisher')
    image_pub = publisher_node.create_publisher(
        Image, '/face_test/image_raw', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    info_pub = publisher_node.create_publisher(
        CameraInfo, '/face_test/camera_info', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    tf_pub = publisher_node.create_publisher(
        TFMessage, '/tf', QoSProfile(depth=100))
    executor = create_face_position_executor()
    executor.add_node(image_node)
    executor.add_node(publisher_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 3.
        while (image_pub.get_subscription_count() == 0
               or info_pub.get_subscription_count() == 0
               or tf_pub.get_subscription_count() == 0) and (
                   time.monotonic() < deadline):
            time.sleep(.01)
        assert image_pub.get_subscription_count() > 0
        assert info_pub.get_subscription_count() > 0
        assert tf_pub.get_subscription_count() > 0

        stamp = Time.from_msg(image_node.get_clock().now().to_msg())
        node_holder['stamp'] = stamp
        image = Image()
        image.header.frame_id = 'camera_optical'
        image.header.stamp = stamp.to_msg()
        image.height = 480
        image.width = 640
        image.encoding = 'bgr8'
        image.step = 640 * 3
        image.data = bytes(640 * 480 * 3)
        camera = CameraInfo()
        camera.header = image.header
        camera.width = 640
        camera.height = 480
        camera.p = [500., 0., 320., 0., 0., 500., 240., 0., 0., 0., 1., 0.]
        camera.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        camera.d = [0.] * 5
        image_pub.publish(image)
        info_pub.publish(camera)
        assert detector_entered.wait(2.)

        transform = TransformStamped()
        transform.header.stamp = image.header.stamp
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'camera_optical'
        transform.transform.rotation.w = 1.
        tf_message = TFMessage(transforms=[transform])

        def publish_after_delay():
            time.sleep(arrival_delay)
            tf_pub.publish(tf_message)

        publisher_thread = threading.Thread(target=publish_after_delay)
        publisher_thread.start()

        assert detector_saw_tf.wait(2.)
        publisher_thread.join(timeout=1.)
        assert not publisher_thread.is_alive()
        assert image_node.tf_buffer.can_transform(
            'world', 'camera_optical', stamp)
        deadline = time.monotonic() + 2.
        while not any(reason in ('OK', 'CONFIRMING_FACE') for reason in observed) \
                and time.monotonic() < deadline:
            time.sleep(.005)
        assert any(reason in ('OK', 'CONFIRMING_FACE') for reason in observed)
    finally:
        executor.shutdown(timeout_sec=2.)
        spin_thread.join(timeout=2.)
        assert not spin_thread.is_alive(), (
            'executor worker remained after shutdown')
        executor.remove_node(image_node)
        executor.remove_node(publisher_node)
        image_node.tf_listener_owner.close()
        image_node.destroy_node()
        publisher_node.destroy_node()


def test_same_node_listener_cannot_dispatch_tf_during_busy_single_executor():
    """Keep a negative control for a listener sharing the frame executor."""
    image_node = LoadedImageNode()
    publisher_node = Node('continuity_negative_tf_publisher')
    image_pub = publisher_node.create_publisher(Image, '/continuity/image', 1)
    tf_pub = publisher_node.create_publisher(
        TFMessage, '/tf', QoSProfile(depth=100))
    executor = SingleThreadedExecutor()
    executor.add_node(image_node)
    executor.add_node(publisher_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 3.
        while (image_pub.get_subscription_count() == 0
               or tf_pub.get_subscription_count() == 0) and time.monotonic() < deadline:
            time.sleep(.01)
        assert image_pub.get_subscription_count() > 0
        assert tf_pub.get_subscription_count() > 0
        image = Image()
        image.header.frame_id = 'camera_optical'
        image.header.stamp = Time(seconds=1).to_msg()
        image_pub.publish(image)
        assert image_node.entered.wait(2.)
        transform = TransformStamped()
        transform.header.stamp = image.header.stamp
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'camera_optical'
        transform.transform.rotation.w = 1.
        tf_pub.publish(TFMessage(transforms=[transform]))
        assert image_node.finished.wait(2.)
        assert not image_node.lookup_ok_during_load
        assert image_node.lookup_error is not None
    finally:
        executor.shutdown(timeout_sec=2.)
        spin_thread.join(timeout=2.)
        assert not spin_thread.is_alive()
        executor.remove_node(image_node)
        executor.remove_node(publisher_node)
        image_node.destroy_node()
        publisher_node.destroy_node()


def test_ros_clock_backward_jump_resets_camera_epoch_and_clears_tf(monkeypatch):
    """Check the production node clears TF and accepts the new camera epoch."""
    monkeypatch.setattr(
        depth_node_module, 'create_detector',
        lambda *args, **kwargs: SimpleNamespace(detect=lambda _image: []))
    depth_node = FacePositionNode()
    result = depth_node.set_parameters([Parameter('use_sim_time', value=True)])
    assert result[0].successful
    publisher_node = Node('continuity_clock_publisher')
    clock_pub = publisher_node.create_publisher(Clock, '/clock', 10)
    tf_pub = publisher_node.create_publisher(
        TFMessage, '/tf', QoSProfile(depth=100))
    image_pub = publisher_node.create_publisher(Image, '/face_test/image_raw', 1)
    info_pub = publisher_node.create_publisher(
        CameraInfo, '/face_test/camera_info', 1)
    executor = create_face_position_executor()
    executor.add_node(depth_node)
    executor.add_node(publisher_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 3.
        while (clock_pub.get_subscription_count() == 0
               or tf_pub.get_subscription_count() == 0
               or image_pub.get_subscription_count() == 0
               or info_pub.get_subscription_count() == 0) and (
                   time.monotonic() < deadline):
            time.sleep(.01)
        assert clock_pub.get_subscription_count() > 0
        assert tf_pub.get_subscription_count() > 0
        assert image_pub.get_subscription_count() > 0
        assert info_pub.get_subscription_count() > 0

        clock_pub.publish(Clock(clock=Time(seconds=10).to_msg()))
        deadline = time.monotonic() + 2.
        while depth_node.get_clock().now().nanoseconds != 10_000_000_000 and (
                time.monotonic() < deadline):
            time.sleep(.005)
        assert depth_node.get_clock().now().nanoseconds == 10_000_000_000

        transform = TransformStamped()
        transform.header.stamp = Time(seconds=10).to_msg()
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'camera_optical'
        transform.transform.rotation.w = 1.
        tf_pub.publish(TFMessage(transforms=[transform]))
        query_time = Time(seconds=10)
        deadline = time.monotonic() + 2.
        while not depth_node.tf_buffer.can_transform(
                'world', 'camera_optical', query_time) and time.monotonic() < deadline:
            time.sleep(.005)
        assert depth_node.tf_buffer.can_transform(
            'world', 'camera_optical', query_time)

        states = []
        original_publish_state = depth_node.publish_perception_state

        def record_state(state, reason, source_stamp=None):
            states.append((state, reason))
            original_publish_state(state, reason, source_stamp)

        depth_node.publish_perception_state = record_state

        def send_frame(seconds):
            header = Header(frame_id='camera_optical',
                            stamp=Time(seconds=seconds).to_msg())
            image = Image(header=header, height=480, width=640,
                          encoding='bgr8', step=640 * 3,
                          data=bytes(640 * 480 * 3))
            info = CameraInfo(header=header, width=640, height=480,
                              p=[500., 0., 320., 0., 0., 500., 240., 0.,
                                 0., 0., 1., 0.],
                              r=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                              d=[0.] * 5)
            image_pub.publish(image)
            info_pub.publish(info)

        send_frame(10)
        deadline = time.monotonic() + 2.
        while depth_node.last_capture_sec != 10. and time.monotonic() < deadline:
            time.sleep(.005)
        assert depth_node.last_capture_sec == 10.
        clock_pub.publish(Clock(clock=Time(seconds=10.1).to_msg()))
        deadline = time.monotonic() + 2.
        while depth_node.get_clock().now().nanoseconds != 10_100_000_000 and (
                time.monotonic() < deadline):
            time.sleep(.005)
        assert depth_node.get_clock().now().nanoseconds == 10_100_000_000
        send_frame(10)
        deadline = time.monotonic() + 2.
        while not any('duplicate or out-of-order' in reason
                      for _, reason in states) and time.monotonic() < deadline:
            time.sleep(.005)
        assert any('duplicate or out-of-order' in reason
                   for _, reason in states)
        previous_state_count = len(states)
        send_frame(9.95)
        deadline = time.monotonic() + 2.
        while len(states) == previous_state_count and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(states) > previous_state_count
        assert 'duplicate or out-of-order' in states[-1][1]
        assert depth_node.last_capture_sec == 10.

        clock_pub.publish(Clock(clock=Time(seconds=2).to_msg()))
        deadline = time.monotonic() + 2.
        while depth_node.get_clock().now().nanoseconds != 2_000_000_000 and (
                time.monotonic() < deadline):
            time.sleep(.005)
        assert depth_node.get_clock().now().nanoseconds == 2_000_000_000
        assert not depth_node.tf_buffer.can_transform(
            'world', 'camera_optical', query_time)

        # Leave a candidate from the old epoch pending until the next input.
        depth_node.world_timer.cancel()
        old_header = Header(
            frame_id='camera_optical', stamp=Time(seconds=10).to_msg())
        depth_node.pending_observation = (
            SimpleNamespace(header=old_header), None, [], [(0, (0., 0., 1.))],
            None, 0., '', [], None, None)
        depth_node.pending_since = time.monotonic()
        depth_node.tf_missing_since = time.monotonic()
        depth_node.tf_missing_reported = True
        observed_reasons = []
        original_publish_observation = depth_node.publish_observation

        def record_observation(selected, reason, filtered_xyz=None):
            observed_reasons.append(reason)
            original_publish_observation(selected, reason, filtered_xyz)

        depth_node.publish_observation = record_observation
        send_frame(2)
        deadline = time.monotonic() + 2.
        while depth_node.last_capture_sec != 2. and time.monotonic() < deadline:
            time.sleep(.005)
        assert depth_node.last_capture_sec == 2.
        assert 'STALE_OBSERVATION' not in observed_reasons
        assert observed_reasons[-1] == 'NO_FACE'
        assert depth_node.pending_observation is None
        assert depth_node.pending_since is None
        assert depth_node.tf_missing_since is None
        assert not depth_node.tf_missing_reported
    finally:
        executor.shutdown(timeout_sec=2.)
        spin_thread.join(timeout=2.)
        assert not spin_thread.is_alive(), (
            'executor worker remained after shutdown')
        executor.remove_node(depth_node)
        executor.remove_node(publisher_node)
        depth_node.tf_listener_owner.close()
        depth_node.destroy_node()
        publisher_node.destroy_node()


def test_sigint_closes_production_executors_during_active_callback():
    """Exercise main's orderly shutdown in a real process without a model."""
    source_root = Path(__file__).resolve().parents[1]
    script = r"""
import signal
import threading
import time
from types import SimpleNamespace
import face_tracking_perception.depth_node as depth

depth.create_detector = lambda *args, **kwargs: SimpleNamespace(detect=lambda _image: [])
original_node = depth.FacePositionNode

class ActiveCallbackNode(original_node):
    def __init__(self):
        super().__init__()
        self._triggered = False
        self.create_timer(.02, self._active_callback)

    def _active_callback(self):
        if self._triggered:
            return
        self._triggered = True
        print('ACTIVE_CALLBACK', flush=True)
        threading.Timer(.05, signal.raise_signal, (signal.SIGINT,)).start()
        time.sleep(.2)

depth.FacePositionNode = ActiveCallbackNode
depth.main()
print('CLEAN_EXIT', flush=True)
"""
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(source_root / 'python'), environment.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=source_root, env=environment,
        text=True, capture_output=True, timeout=8.)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ACTIVE_CALLBACK' in result.stdout
    assert 'CLEAN_EXIT' in result.stdout
    assert 'context is invalid' not in result.stderr
    assert 'Traceback' not in result.stderr


def test_executor_shutdown_waits_for_inference_callback_to_finish(monkeypatch):
    """Shutdown joins the worker after an in-flight inference is released."""
    entered = threading.Event()
    release = threading.Event()

    def blocked_detector(_image):
        entered.set()
        assert release.wait(2.)
        return []

    monkeypatch.setattr(
        depth_node_module, 'create_detector',
        lambda *args, **kwargs: SimpleNamespace(detect=blocked_detector))
    depth_node = FacePositionNode()
    publisher_node = Node('continuity_shutdown_publisher')
    image_pub = publisher_node.create_publisher(Image, '/face_test/image_raw', 1)
    info_pub = publisher_node.create_publisher(
        CameraInfo, '/face_test/camera_info', 1)
    executor = create_face_position_executor()
    executor.add_node(depth_node)
    executor.add_node(publisher_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 3.
        while (image_pub.get_subscription_count() == 0
               or info_pub.get_subscription_count() == 0) and (
                   time.monotonic() < deadline):
            time.sleep(.01)
        stamp = depth_node.get_clock().now().to_msg()
        header = Header(frame_id='camera_optical', stamp=stamp)
        image = Image(header=header, height=480, width=640,
                      encoding='bgr8', step=640 * 3,
                      data=bytes(640 * 480 * 3))
        info = CameraInfo(header=header, width=640, height=480,
                          p=[500., 0., 320., 0., 0., 500., 240., 0.,
                             0., 0., 1., 0.],
                          r=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                          d=[0.] * 5)
        image_pub.publish(image)
        info_pub.publish(info)
        assert entered.wait(2.)

        shutdown_thread = threading.Thread(target=executor.shutdown)
        shutdown_thread.start()
        time.sleep(.02)
        release.set()
        shutdown_thread.join(timeout=2.)
        assert not shutdown_thread.is_alive()
        spin_thread.join(timeout=2.)
        assert not spin_thread.is_alive()
    finally:
        release.set()
        executor.shutdown(timeout_sec=2.)
        spin_thread.join(timeout=2.)
        executor.remove_node(depth_node)
        executor.remove_node(publisher_node)
        depth_node.tf_listener_owner.close()
        depth_node.destroy_node()
        publisher_node.destroy_node()
