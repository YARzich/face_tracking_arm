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


def test_bad_calibration_is_rejected_instead_of_guessing_focal_length():
    calibration = info()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='already rectified'):
        Rectification().apply(image, calibration, 'none')
    calibration.r = [0.] * 9
    with pytest.raises(ValueError, match='rotation'):
        Rectification().apply(image, calibration, 'raw')
