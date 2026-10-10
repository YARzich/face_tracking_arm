# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Verify calibrated rays and raw/rectified inputs without a camera or ROS runtime."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_perception.rectification import Rectification  # noqa: E402, I100


def info():
    return CameraInfo(header=Header(frame_id='left'), width=640, height=480,
                      distortion_model='plumb_bob', d=[-.08, .01, 0., 0., 0.],
                      k=[500., 0., 320., 0., 500., 240., 0., 0., 1.],
                      r=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                      p=[500., 0., 320., 0., 0., 500., 240., 0., 0., 0., 1., 0.])


def test_rectified_input_retains_raw_metadata_and_undoes_the_rotation_for_3d():
    calibration = info()
    rotation, _ = cv2.Rodrigues(np.array([0., .12, 0.]))
    calibration.r = rotation.reshape(-1).tolist()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rectifier = Rectification()
    output, camera, to_original = rectifier.apply(image, calibration, 'rectified')
    assert output is image
    measured = to_original @ camera.unproject(360, 220, 2)
    np.testing.assert_allclose(rotation @ measured, [.16, -.08, 2], atol=1e-10)
    assert calibration.d[0] == -.08  # Do not falsify the source CameraInfo.


def test_raw_input_reuses_maps_and_invalidates_them_on_calibration_change():
    rectifier, calibration = Rectification(), info()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    output, _, _ = rectifier.apply(image, calibration, 'raw')
    maps = rectifier.maps
    assert output.shape == image.shape
    rectifier.apply(image, calibration, 'raw')
    assert rectifier.maps is maps
    calibration.p[0] += 10
    rectifier.apply(image, calibration, 'raw')
    assert rectifier.maps is not maps


def test_raw_rgbd_depth_uses_the_same_nonzero_distortion_map_as_color():
    calibration = info()
    rectifier = Rectification()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rectifier.apply(rgb, calibration, 'raw')
    raw_depth = np.zeros((480, 640), dtype=np.uint16)
    raw_depth[220, 320] = 1234
    expected = cv2.remap(raw_depth, *rectifier.maps, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_output = rectifier.apply_depth(raw_depth, 'raw')
    np.testing.assert_array_equal(aligned_output, expected)
    assert np.count_nonzero(aligned_output) == 1
    actual_pixel = np.argwhere(aligned_output == 1234)[0].tolist()
    expected_pixel = np.argwhere(expected == 1234)[0].tolist()
    assert actual_pixel == expected_pixel
    assert not np.array_equal(aligned_output, raw_depth)


def test_bad_calibration_is_rejected_instead_of_guessing_focal_length():
    calibration = info()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='already rectified'):
        Rectification().apply(image, calibration, 'none')
    calibration.r = [0.] * 9
    with pytest.raises(ValueError, match='rotation'):
        Rectification().apply(image, calibration, 'raw')
