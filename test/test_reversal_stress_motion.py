# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Checks for the single-person reversal diagnostic schedule."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from face_tracking_perception.reversal_stress_motion import (  # noqa: E402, I100
    ReversalStressMotion)


def test_reversal_blocks_positions_pauses_and_constant_face_height():
    """Keep each profile block on its intended world axes and height."""
    motion = ReversalStressMotion()
    assert motion.duration_sec == 78
    assert len(motion.initial_poses) == 1
    assert [(motion.sample(t).phase) for t in (2, 15, 28, 41, 66)] == [
        'SLOW_LATERAL', 'FAST_LATERAL', 'SLOW_DEPTH', 'FAST_DEPTH', 'COMBINED']
    for t, y in ((3.5, .4), (6.5, -.4), (9.5, .4), (12.5, -.4),
                 (16.2, .4), (17.4, -.4), (25.8, .4), (69, .4), (72, -.4)):
        assert motion.sample(t).poses[0][1] == pytest.approx(y)
    for t, x in ((30, 1.2), (34, 2.8), (38, 1.2), (43.4, 1.2),
                 (45.8, 2.8), (62.6, 1.2)):
        assert motion.sample(t).poses[0][0] == pytest.approx(x)
    times = np.linspace(0, 78, 7801)
    points = np.array([motion.sample(t).poses[0] for t in times])
    face_height = points[:, 2] + motion.face_reference_height_m
    assert np.max(np.abs(face_height - 1.75)) < 1e-12
    lateral = points[(times >= 2) & (times <= 27), 0]
    assert np.max(np.abs(lateral - 2.1)) < 1e-12
    radial = points[(times >= 28) & (times <= 65), 1]
    assert np.max(np.abs(radial)) < 1e-12
    for t, phase, duration in ((0, 'START_HOLD', 2), (14, 'LATERAL_HOLD', 1),
                               (27, 'LATERAL_HOLD', 1), (40, 'DEPTH_HOLD', 1),
                               (65, 'DEPTH_HOLD', 1)):
        assert motion.sample(t).phase == phase
        held = motion.sample(t + duration - .001).poses
        assert held == motion.sample(t).poses


@pytest.mark.parametrize('boundary', [
    2, 3.5, 6.5, 9.5, 12.5, 14, 15, 16.2, 17.4, 25.8, 27, 28, 30, 34,
    38, 40, 41, 43.4, 45.8, 62.6, 65, 66, 69, 72, 75, 78])
def test_segment_joins_and_cycle_wrap_are_continuous(boundary):
    """Check adjacent eased segments meet without a position jump."""
    motion = ReversalStressMotion()
    step = 1e-4
    left = np.array(motion.sample(boundary - step).poses[0])
    center = np.array(motion.sample(boundary).poses[0])
    right = np.array(motion.sample(boundary + step).poses[0])
    assert np.linalg.norm(center - left) < .001
    assert np.linalg.norm(right - center) < .001


def test_endless_loop_and_finite_run_hold_last_pose():
    """Repeat endless runs and keep bounded runs at their start pose."""
    motion = ReversalStressMotion()
    for t in (0, 1.5, 3.5, 14, 42.2, 77.9):
        assert motion.sample(t).poses == motion.sample(t + 78).poses
    assert motion.sample(78, cycles=1).complete
    assert motion.sample(78, cycles=1).poses == motion.initial_poses
    assert not motion.sample(77.999, cycles=1).complete
