# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Metric geometry for calibrated, rectified pinhole images (no ROS state)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole calibration for the image being processed."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    tx: float = 0.0

    @classmethod
    def from_info(cls, info):
        if info.width <= 0 or info.height <= 0 or not info.header.frame_id:
            raise ValueError('CameraInfo has no valid size/frame')
        if not np.isfinite(info.p).all() or info.p[0] <= 0 or info.p[5] <= 0:
            raise ValueError('CameraInfo has no valid projection matrix')
        if not np.isfinite(info.d).all() or np.any(np.abs(info.d) > 1e-8) or not np.allclose(
                np.asarray(info.r).reshape(3, 3), np.eye(3)):
            raise ValueError('This test requires already rectified, undistorted images')
        return cls(info.width, info.height, info.p[0], info.p[5],
                   info.p[2], info.p[6], info.p[3])

    def unproject(self, u, v, depth):
        """Depth is optical Z, not Euclidean range along the viewing ray."""
        if not np.isfinite([u, v, depth]).all() or depth <= 0:
            raise ValueError('Non-finite point or non-positive depth')
        return ((u - self.cx) * depth / self.fx,
                (v - self.cy) * depth / self.fy, float(depth))


def visible_face(face, camera):
    """Reject tiny/truncated boxes whose width cannot measure face scale."""
    values = [face.x, face.y, face.width, face.height]
    return (np.isfinite(values).all() and face.width >= 12 and face.height >= 12
            and face.x >= 2 and face.y >= 2
            and face.x + face.width < camera.width - 2
            and face.y + face.height < camera.height - 2)


def size_depth(face, camera, face_width_m):
    """Approximate Z using a user-selected physical frontal-face width."""
    if not visible_face(face, camera) or not np.isfinite(face_width_m) or face_width_m <= 0:
        raise ValueError('Face too small/truncated or invalid physical face width')
    return camera.fx * face_width_m / face.width


def sample_face_depth(depth, face, minimum=0.25, maximum=8.0):
    """Median of the central face patch; ignore invalid pixels and weak support."""
    u, v = face.center
    radius_x, radius_y = max(2, round(face.width * 0.15)), max(2, round(face.height * 0.15))
    x1, x2 = max(0, round(u) - radius_x), min(depth.shape[1], round(u) + radius_x + 1)
    y1, y2 = max(0, round(v) - radius_y), min(depth.shape[0], round(v) + radius_y + 1)
    patch = depth[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch >= minimum) & (patch <= maximum)]
    if valid.size < max(9, patch.size * 0.25):
        raise ValueError('Insufficient valid depth within the face')
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    if mad > max(0.10, median * 0.10):
        raise ValueError('Depth within face is inconsistent')
    return median


def stereo_baseline(left, right):
    """Read baseline from the right projection matrix; do not infer it from detections."""
    if (left.width, left.height) != (right.width, right.height) or not np.allclose(
            [left.fx, left.fy, left.cy], [right.fx, right.fy, right.cy]):
        raise ValueError('Stereo cameras are not a rectified pair')
    baseline = left.tx / left.fx - right.tx / right.fx
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Right CameraInfo must encode a positive stereo baseline')
    return baseline
