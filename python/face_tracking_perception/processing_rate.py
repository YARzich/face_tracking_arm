# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Small monotonic-time helpers for bounded, non-backlogged perception."""

from collections import deque
from math import floor
from statistics import median


class ProcessingRateGate:
    """Rate cap on a persistent deadline, without alternating-frame loss."""

    def __init__(self, maximum_fps=0.0):
        if maximum_fps < 0:
            raise ValueError('maximum_fps must be nonnegative')
        self.period = 0.0 if maximum_fps == 0 else 1.0 / float(maximum_fps)
        self.next_due = None

    def accept(self, now):
        now = float(now)
        if self.period == 0:
            return True
        if self.next_due is not None and now < self.next_due:
            return False
        if self.next_due is None:
            self.next_due = now
        self.next_due += self.period
        if self.next_due <= now:
            missed = floor((now - self.next_due) / self.period + 1e-9) + 1
            self.next_due += missed * self.period
        return True

    def set_maximum_fps(self, maximum_fps):
        """Change the cap without moving an already scheduled deadline."""
        if maximum_fps < 0:
            raise ValueError('maximum_fps must be nonnegative')
        self.period = 0.0 if maximum_fps == 0 else 1.0 / float(maximum_fps)

    def reset(self):
        self.next_due = None


class StableFrameCadence:
    """Robust capture period estimate; bad stamps and pauses never train it."""

    def __init__(self, capacity=9,
                 minimum_period_sec=1.0 / 120.0, maximum_period_sec=0.25):
        self.intervals = deque(maxlen=capacity)
        self.receipt_intervals = deque(maxlen=capacity)
        self.minimum_period_sec = minimum_period_sec
        self.maximum_period_sec = maximum_period_sec
        self.previous_stamp = None
        self.previous_receipt = None

    @property
    def fps(self):
        return 1.0 / median(self.intervals) if self.intervals else None

    @property
    def stable_fps(self):
        return self.fps if len(self.intervals) >= 3 else None

    def bounded_gap_limit(self, maximum_gap_sec, typical_periods=2.5, jitter_periods=2.0):
        """Choose a bounded watchdog gap from observed inter-arrival cadence and jitter."""
        if not self.intervals or len(self.intervals) < 3:
            return float(maximum_gap_sec)
        receipt_periods = self.receipt_intervals or self.intervals
        if len(receipt_periods) < 3:
            return float(maximum_gap_sec)
        typical_period = median(receipt_periods)
        recent_slowest_period = max(receipt_periods)
        estimate = max(typical_periods * typical_period,
                       jitter_periods * recent_slowest_period)
        return min(float(maximum_gap_sec), estimate)

    def observe(self, capture_sec, receipt_sec, valid=True):
        capture_sec, receipt_sec = float(capture_sec), float(receipt_sec)
        if not valid:
            self.reset_reference()
            return self.fps
        if self.previous_stamp is not None:
            capture_delta = capture_sec - self.previous_stamp
            receipt_delta = receipt_sec - self.previous_receipt
            if (capture_delta <= 0 or receipt_delta <= 0
                    or capture_delta > self.maximum_period_sec
                    or not self.minimum_period_sec <= capture_delta <= self.maximum_period_sec):
                self.reset_reference()
                self.previous_stamp, self.previous_receipt = capture_sec, receipt_sec
                return self.fps
            self.intervals.append(capture_delta)
            self.receipt_intervals.append(receipt_delta)
        self.previous_stamp, self.previous_receipt = capture_sec, receipt_sec
        return self.fps

    def reset_reference(self):
        self.previous_stamp = None
        self.previous_receipt = None

    def reset(self):
        self.intervals.clear()
        self.receipt_intervals.clear()
        self.reset_reference()
