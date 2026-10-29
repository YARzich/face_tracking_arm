# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Tests for deterministic target waypoint transitions."""

import json
import math
from pathlib import Path
import unittest

from posture1d_sequence import target_at

ROOT = Path(__file__).resolve().parents[1]


class SequenceTest(unittest.TestCase):
    """Verify holds, smooth transitions, and live-derived return targets."""

    def setUp(self):
        """Create a two-segment sequence with a dynamic return point."""
        self.definition = {
            'waypoints': [
                {'target_world_m': [0.0, 0.0, 0.0],
                 'transition_start_sec': 0, 'transition_sec': 0},
                {'target_world_m': [10.0, 0.0, 0.0],
                 'transition_start_sec': 10, 'transition_sec': 10},
                {'target_world_m': None,
                 'transition_start_sec': 30, 'transition_sec': 10},
            ]}
        self.return_target = [2.0, 3.0, 4.0]

    def test_hold_before_transition_and_after_arrival(self):
        """Targets remain constant before motion and after transition."""
        self.assertEqual(target_at(self.definition, self.return_target, 9),
                         [0.0, 0.0, 0.0])
        self.assertEqual(target_at(self.definition, self.return_target, 25),
                         [10.0, 0.0, 0.0])
        self.assertEqual(target_at(self.definition, self.return_target, 45),
                         self.return_target)

    def test_quintic_transition_has_stationary_endpoints(self):
        """Quintic interpolation starts and ends at the waypoint values."""
        self.assertEqual(target_at(self.definition, self.return_target, 10),
                         [0.0, 0.0, 0.0])
        halfway = target_at(self.definition, self.return_target, 15)
        self.assertEqual(halfway, [5.0, 0.0, 0.0])
        self.assertEqual(target_at(self.definition, self.return_target, 20),
                         [10.0, 0.0, 0.0])

    def test_rejects_nonfinite_time(self):
        """Reject a non-finite simulation timestamp."""
        with self.assertRaises(ValueError):
            target_at(self.definition, self.return_target, float('nan'))

    def test_slow_return_manifest_respects_target_speed_limit(self):
        manifest = json.loads((ROOT / 'tools/posture1d_far_lateral_up_return_slow.json')
                              .read_text())
        points = manifest['waypoints']
        start = points[1]['target_world_m']
        return_target = [0.75, 0.0, 1.15]
        duration = points[2]['transition_sec']
        peak_speed = 1.875 * math.dist(start, return_target) / duration

        self.assertEqual(manifest['duration_sim_sec'], 205)
        self.assertLessEqual(peak_speed, manifest['return_transition_max_speed_mps'])


if __name__ == '__main__':
    unittest.main()
