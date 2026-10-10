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
    """Accept a finite face box with enough visible pixels, including at image edges."""
    values = [face.x, face.y, face.width, face.height]
    if not np.isfinite(values).all() or face.width < 12 or face.height < 12:
        return False
    return (max(0.0, face.x) < min(float(camera.width), face.x + face.width)
            and max(0.0, face.y) < min(float(camera.height), face.y + face.height))


def fully_visible_face(face, camera, margin=2):
    """Mono face-size ranging needs a complete, un-clipped face width."""
    return (visible_face(face, camera) and face.x >= margin and face.y >= margin
            and face.x + face.width <= camera.width - margin
            and face.y + face.height <= camera.height - margin)


def partial_face_center(face, camera, optical_z, face_width_m):
    """Estimate full-box bearing from a clipped box and retained range."""
    if (fully_visible_face(face, camera) or not np.isfinite(
            [optical_z, face_width_m, face.width, face.height]).all()
            or optical_z <= 0 or face_width_m <= 0 or face.width <= 0 or face.height <= 0):
        raise ValueError('A finite clipped face and positive retained range are required')
    expected_width = camera.fx * face_width_m / optical_z
    expected_height = expected_width * face.height / face.width
    u, v = face.center
    if face.x <= 1:
        u = face.x + expected_width / 2
    elif face.x + face.width >= camera.width - 1:
        u = face.x + face.width - expected_width / 2
    if face.y <= 1:
        v = face.y + expected_height / 2
    elif face.y + face.height >= camera.height - 1:
        v = face.y + face.height - expected_height / 2
    return (float(np.clip(u, 0, camera.width - 1)),
            float(np.clip(v, 0, camera.height - 1)))


def size_depth(face, camera, face_width_m):
    """Approximate Z using a user-selected physical frontal-face width."""
    if (not fully_visible_face(face, camera) or not np.isfinite(face_width_m)
            or face_width_m <= 0):
        raise ValueError('Face too small/truncated or invalid physical face width')
    return camera.fx * face_width_m / face.width


def sample_face_depth(depth, face, minimum=0.25, maximum=8.0):
    """Median depth and pixel location from the visible central face patch."""
    z, _, _ = sample_face_measurement(depth, face, minimum, maximum)
    return z


def sample_face_measurement(depth, face, minimum=0.25, maximum=8.0):
    """Return robust (optical Z, u, v), using only pixels inside the clipped box."""
    u, v = face.center
    # For an edge-clipped detection the geometric center may lie outside the
    # image. Sampling uses the visible box intersection, then its measured depth
    # pixels determine the returned bearing.
    u = float(np.clip(u, max(0.0, face.x), min(float(depth.shape[1] - 1), face.x + face.width)))
    v = float(np.clip(v, max(0.0, face.y), min(float(depth.shape[0] - 1), face.y + face.height)))
    if face.x <= 2:
        u = min(float(depth.shape[1] - 1), max(0.0, face.x) + face.width * .30)
    elif face.x + face.width >= depth.shape[1] - 2:
        u = max(0.0, min(float(depth.shape[1] - 1), face.x + face.width) - face.width * .30)
    if face.y <= 2:
        v = min(float(depth.shape[0] - 1), max(0.0, face.y) + face.height * .30)
    elif face.y + face.height >= depth.shape[0] - 2:
        v = max(0.0, min(float(depth.shape[0] - 1), face.y + face.height) - face.height * .30)
    partial = (face.x <= 2 or face.y <= 2 or face.x + face.width >= depth.shape[1] - 2
               or face.y + face.height >= depth.shape[0] - 2)
    patch_fraction = 0.10 if partial else 0.25
    radius_x, radius_y = max(2, round(face.width * 0.30)), max(2, round(face.height * 0.30))
    box_x1 = max(0, int(np.floor(face.x)))
    box_x2 = min(depth.shape[1], int(np.ceil(face.x + face.width)))
    box_y1 = max(0, int(np.floor(face.y)))
    box_y2 = min(depth.shape[0], int(np.ceil(face.y + face.height)))
    x1, x2 = max(box_x1, round(u) - radius_x), min(box_x2, round(u) + radius_x + 1)
    y1, y2 = max(box_y1, round(v) - radius_y), min(box_y2, round(v) + radius_y + 1)
    patch = depth[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch >= minimum) & (patch <= maximum)]
    if valid.size < max(9, patch.size * patch_fraction):
        raise ValueError('Insufficient valid depth within the face')
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    if mad > max(0.10, median * 0.10):
        raise ValueError('Depth within face is inconsistent')
    rows, columns = np.nonzero(np.isfinite(patch) & (patch >= minimum) & (patch <= maximum))
    # Use a robust center of actual face-supported depth pixels, not the center
    # of a box that may have been clipped by the image boundary.
    return median, float(x1 + np.median(columns)), float(y1 + np.median(rows))


def stereo_baseline(left, right):
    """Read baseline from the right projection matrix; do not infer it from detections."""
    if (left.width, left.height) != (right.width, right.height) or not np.allclose(
            [left.fx, left.fy, left.cy], [right.fx, right.fy, right.cy]):
        raise ValueError('Stereo cameras are not a rectified pair')
    baseline = left.tx / left.fx - right.tx / right.fx
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Right CameraInfo must encode a positive stereo baseline')
    return baseline
