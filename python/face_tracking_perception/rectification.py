# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Calibrated OpenCV rectification, with points returned to the original optical frame."""

import cv2
import numpy as np

from .depth_geometry import Camera


class Rectification:
    """Cache remap tables; retain the calibrated rotation instead of relabeling points."""

    def __init__(self):
        self.signature = None
        self.maps = None

    def apply(self, image, info, mode):
        if mode == 'none':
            return image, Camera.from_info(info), np.eye(3)
        if mode not in ('raw', 'rectified'):
            raise ValueError('rectification must be none, raw or rectified')
        k = np.asarray(info.k).reshape(3, 3)
        r = np.asarray(info.r).reshape(3, 3)
        p = np.asarray(info.p).reshape(3, 4)
        d = np.asarray(info.d)
        if (not info.header.frame_id or image.shape[:2] != (info.height, info.width)
                or not all(np.isfinite(a).all() for a in (k, r, p, d))
                or k[0, 0] <= 0 or k[1, 1] <= 0 or p[0, 0] <= 0 or p[1, 1] <= 0
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-5)):
            raise ValueError('Invalid CameraInfo size, calibration or rectification rotation')
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial', 'equidistant', ''):
            raise ValueError('Unsupported camera distortion model')
        camera = Camera(info.width, info.height, p[0, 0], p[1, 1], p[0, 2], p[1, 2], p[0, 3])
        if mode == 'raw':
            signature = (info.width, info.height, info.distortion_model,
                         tuple(info.k), tuple(info.d), tuple(info.r), tuple(info.p))
            if self.signature != signature:
                build = (cv2.fisheye.initUndistortRectifyMap
                         if info.distortion_model == 'equidistant'
                         else cv2.initUndistortRectifyMap)
                self.maps = build(k, d, r, p[:, :3], (info.width, info.height), cv2.CV_16SC2)
                self.signature = signature
            image = cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR)
        return image, camera, r.T
