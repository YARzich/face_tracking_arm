# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Image geometry and ROS contract tests without inference models or Gazebo."""

from pathlib import Path
import sys

import numpy as np
import pytest
from std_msgs.msg import Header

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from face_tracking_perception.detectors import (  # noqa: E402, I100
    create_detector, decode_yolo, Face, letterbox)
from face_tracking_perception.node import detection_message  # noqa: E402


@pytest.mark.parametrize('columns', [6, 16])
def test_yolo_restores_original_pixel_center_and_suppresses_duplicates(columns):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, transform = letterbox(image)
    assert padded.shape == (640, 640, 3)
    rows = np.zeros((1, 3, columns), dtype=np.float32)
    rows[0, 0, :5] = [320, 280, 100, 120, 0.9]
    rows[0, 0, -1] = 0.9
    rows[0, 1] = rows[0, 0]
    rows[0, 1, 4] = 0.8
    rows[0, 2, :5] = [100, 100, 30, 30, 0.9]
    rows[0, 2, -1] = 0.1  # Objectness alone must not produce a detection.
    faces = decode_yolo(rows, image.shape, transform, 0.6, 0.4)
    assert len(faces) == 1
    assert faces[0].center == pytest.approx((320, 200))
    assert faces[0].height == pytest.approx(120)
    assert faces[0].confidence == pytest.approx(0.81)


def test_non_square_resize_uses_actual_scale_and_padding():
    image = np.zeros((301, 803, 3), dtype=np.uint8)
    _, transform = letterbox(image)
    sx, sy, left, top = transform
    row = np.array([[[200 * sx + left, 150 * sy + top, 40 * sx, 70 * sy, 1, 1]]])
    face = decode_yolo(row, image.shape, transform, 0.6, 0.4)[0]
    assert face.center == pytest.approx((200, 150))
    assert (face.width, face.height) == pytest.approx((40, 70))


def test_invalid_boxes_and_empty_results():
    rows = np.array([[[0, 0, -2, 2, 1, 1], [np.nan, 0, 2, 2, 1, 1]]])
    assert decode_yolo(rows, (480, 640), (1, 1, 0, 0), 0.6, 0.4) == []
    with pytest.raises(ValueError):
        decode_yolo(np.zeros((1, 4, 85)), (480, 640), (1, 1, 0, 0), 0.6, 0.4)


def test_detection_stamp_frame_center_and_explicit_loss():
    header = Header(frame_id='camera_optical')
    header.stamp.sec = 12
    header.stamp.nanosec = 345
    result = detection_message(header, [Face(10, 20, 40, 60, 0.8)])
    assert result.header == header
    detection = result.detections[0]
    assert detection.header == header
    assert detection.bbox.center.position.x == 30
    assert detection.bbox.center.position.y == 50
    assert detection.results[0].hypothesis.score == 0.8
    assert detection_message(header, []).detections == []


def test_unknown_or_missing_model_never_silently_falls_back(tmp_path):
    with pytest.raises(ValueError):
        create_detector('unknown', tmp_path)
    with pytest.raises(FileNotFoundError):
        create_detector('yolo_facev2n', tmp_path)
