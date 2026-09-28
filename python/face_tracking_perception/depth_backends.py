# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Depth estimators using OpenCV stereo or the official metric depth network."""

from pathlib import Path
import sys

import cv2
import numpy as np

from .depth_geometry import stereo_baseline


class StereoDepth:
    """OpenCV SGBM on calibrated rectified pairs; output meters in the left frame."""

    def __init__(self, num_disparities=128, block_size=5):
        if type(num_disparities) is not int or num_disparities <= 0 or num_disparities % 16:
            raise ValueError('stereo_num_disparities must be a positive multiple of 16 pixels')
        if type(block_size) is not int or block_size < 1 or block_size % 2 != 1:
            raise ValueError('stereo_block_size must be an odd positive integer in pixels')
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2, disp12MaxDiff=1,
            uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def compute(self, left_image, right_image, left_camera, right_camera):
        if left_image.shape != right_image.shape:
            raise ValueError('Stereo images must have the same dimensions')
        height, width = left_image.shape[:2]
        if (width <= self.num_disparities + self.block_size // 2 or
                min(height, width) < self.block_size):
            raise ValueError('Stereo image width must exceed stereo_num_disparities + '
                             'stereo_block_size // 2; both dimensions must be at least '
                             'stereo_block_size pixels')
        baseline = stereo_baseline(left_camera, right_camera)
        left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        denominator = disparity - (left_camera.cx - right_camera.cx)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = (disparity > 0) & (denominator > 0)
        depth[valid] = left_camera.fx * baseline / denominator[valid]
        return depth


class MetricDepth:
    """Official Depth Anything V2 Metric Hypersim Small, loaded once on CUDA."""

    def __init__(self, weights, source_dir, input_size=280):
        import torch

        weights = Path(weights).expanduser()
        source_dir = Path(source_dir).expanduser()
        if not weights.is_file() or not (source_dir / 'depth_anything_v2/dpt.py').is_file():
            raise FileNotFoundError('Metric model missing; see docs/face_depth_tests.md')
        if not torch.cuda.is_available():
            raise RuntimeError('mono_gpu requires CUDA; use mono_cpu for a CPU-only machine')
        if input_size < 140 or input_size % 14:
            raise ValueError('depth_input_size must be a multiple of 14, at least 140')
        sys.path.insert(0, str(source_dir))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(
            encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.model = self.model.to('cuda').eval()
        self.input_size = input_size
        # Complete initialization before subscribing, so the first observation is not queued.
        self.model.infer_image(np.zeros((480, 640, 3), dtype=np.uint8), input_size=input_size)

    def compute(self, image):
        return self.model.infer_image(image, input_size=self.input_size)
