# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Single-person lateral and radial reversals for RGB-D diagnostics."""

from .fast_stress_motion import _model_pose, FastStressMotion


class ReversalStressMotion(FastStressMotion):
    """Short-pause reversal schedule with one rendered person."""

    duration_sec = 78.0
    initial_poses = (_model_pose(2.1, 0.0, 1.75),)
    _keyframes = (
        (0.0, 2.1, 0.0, 1.75, 'START_HOLD'),
        (2.0, 2.1, 0.0, 1.75, 'SLOW_LATERAL'),
        (3.5, 2.1, .4, 1.75, 'SLOW_LATERAL'),
        (6.5, 2.1, -.4, 1.75, 'SLOW_LATERAL'),
        (9.5, 2.1, .4, 1.75, 'SLOW_LATERAL'),
        (12.5, 2.1, -.4, 1.75, 'SLOW_LATERAL'),
        (14.0, 2.1, 0.0, 1.75, 'LATERAL_HOLD'),
        (15.0, 2.1, 0.0, 1.75, 'FAST_LATERAL'),
        (16.2, 2.1, .4, 1.75, 'FAST_LATERAL'),
        (17.4, 2.1, -.4, 1.75, 'FAST_LATERAL'),
        (18.6, 2.1, .4, 1.75, 'FAST_LATERAL'),
        (19.8, 2.1, -.4, 1.75, 'FAST_LATERAL'),
        (21.0, 2.1, .4, 1.75, 'FAST_LATERAL'),
        (22.2, 2.1, -.4, 1.75, 'FAST_LATERAL'),
        (23.4, 2.1, .4, 1.75, 'FAST_LATERAL'),
        (24.6, 2.1, -.4, 1.75, 'FAST_LATERAL'),
        (25.8, 2.1, .4, 1.75, 'FAST_LATERAL'),
        (27.0, 2.1, 0.0, 1.75, 'LATERAL_HOLD'),
        (28.0, 2.1, 0.0, 1.75, 'SLOW_DEPTH'),
        (30.0, 1.2, 0.0, 1.75, 'SLOW_DEPTH'),
        (34.0, 2.8, 0.0, 1.75, 'SLOW_DEPTH'),
        (38.0, 1.2, 0.0, 1.75, 'SLOW_DEPTH'),
        (40.0, 2.1, 0.0, 1.75, 'DEPTH_HOLD'),
        (41.0, 2.1, 0.0, 1.75, 'FAST_DEPTH'),
        (43.4, 1.2, 0.0, 1.75, 'FAST_DEPTH'),
        (45.8, 2.8, 0.0, 1.75, 'FAST_DEPTH'),
        (48.2, 1.2, 0.0, 1.75, 'FAST_DEPTH'),
        (50.6, 2.8, 0.0, 1.75, 'FAST_DEPTH'),
        (53.0, 1.2, 0.0, 1.75, 'FAST_DEPTH'),
        (55.4, 2.8, 0.0, 1.75, 'FAST_DEPTH'),
        (57.8, 1.2, 0.0, 1.75, 'FAST_DEPTH'),
        (60.2, 2.8, 0.0, 1.75, 'FAST_DEPTH'),
        (62.6, 1.2, 0.0, 1.75, 'FAST_DEPTH'),
        (65.0, 2.1, 0.0, 1.75, 'DEPTH_HOLD'),
        (66.0, 2.1, 0.0, 1.75, 'COMBINED'),
        (69.0, 1.2, .4, 1.75, 'COMBINED'),
        (72.0, 2.8, -.4, 1.75, 'COMBINED'),
        (75.0, 1.2, .4, 1.75, 'COMBINED'),
        (78.0, 2.1, 0.0, 1.75, 'COMPLETE'),
    )
