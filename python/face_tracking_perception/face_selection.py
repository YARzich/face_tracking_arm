# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Keep a nearby measured face stable without face recognition or simulator truth."""

from collections import deque
from statistics import median

import numpy as np

MAX_OBSERVATION_GAP_SEC = 1.5


class NearestFaceSelector:
    """Associate positions in a fixed frame; confirm acquisition and closer challengers."""

    def __init__(self, switch_margin_m=.25, switch_delay_sec=.4):
        if (not np.isfinite([switch_margin_m, switch_delay_sec]).all()
                or switch_margin_m < 0 or switch_delay_sec < 0):
            raise ValueError('Invalid face switching thresholds')
        self.margin = switch_margin_m
        self.delay = switch_delay_sec
        self.track_id = 0
        self.reset()

    def reset(self):
        self.current = None
        self.last_seen = None
        self.last_stamp = None
        self.candidate = None
        self.since = None
        self.confirmations = 0
        self.frame_intervals = deque(maxlen=3)

    def _confirm(self, point, stamp, delay):
        if self.candidate is None or np.linalg.norm(point - self.candidate) > .5:
            self.since = stamp
            self.confirmations = 0
        self.candidate = point.copy()
        self.confirmations += 1
        return self.confirmations >= 3 and stamp - self.since >= delay

    def select(self, points, stamp):
        """Return an input index or None; points are relative to the fixed distance origin."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not np.isfinite(points).all() or not np.isfinite(stamp):
            raise ValueError('Non-finite face selection input')
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.reset()
        if self.last_stamp == stamp:
            return None
        interval = stamp - self.last_stamp if self.last_stamp is not None else None
        if interval is not None:
            if interval > MAX_OBSERVATION_GAP_SEC:
                self.reset()
            else:
                self.frame_intervals.append(interval)
        # Learn the actual cadence, including frames without faces. A median
        # rejects one delayed frame; a bounded window still expires lost tracks.
        period = median(self.frame_intervals) if self.frame_intervals else 0.0
        cadence_gap = min(MAX_OBSERVATION_GAP_SEC, 1.5 * period)
        if interval is not None and interval > max(.2, cadence_gap):
            self.candidate = None
        self.last_stamp = stamp
        if self.last_seen is not None and stamp - self.last_seen > max(.5, cadence_gap):
            self.current = None
        if len(points) == 0:
            self.candidate = None
            return None
        distances = np.linalg.norm(points, axis=1)
        nearest = int(np.argmin(distances))
        selected = None
        if self.current is not None:
            differences = np.linalg.norm(points - self.current, axis=1)
            match = int(np.argmin(differences))
            if differences[match] <= .5:
                selected = match
                margin = max(self.margin, .10 * distances[match])
                if nearest != match and distances[nearest] + margin < distances[match]:
                    if self._confirm(points[nearest], stamp, self.delay):
                        selected = nearest
                        self.track_id += 1
                        self.candidate = None
                else:
                    self.candidate = None
            else:
                self.candidate = None
                return None  # Give a briefly occluded target its bounded grace interval.
        elif self._confirm(points[nearest], stamp, .06):
            selected = nearest
            self.track_id += 1
            self.candidate = None
        if selected is not None:
            self.current = points[selected].copy()
            self.last_seen = stamp
        return selected
