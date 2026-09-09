# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Simulation-time appearance schedule, independent of ROS and Gazebo services."""

import math


class AppearanceSequence:
    """Advance only after an acknowledged pose change and a confirmed rest interval."""

    def __init__(self, count, visible_sec=12.0, rest_sec=8.0, cycles=0, await_face=False):
        if (count < 1 or cycles < 0 or not math.isfinite(visible_sec + rest_sec)
                or visible_sec <= 0 or rest_sec < 0):
            raise ValueError('Invalid appearance sequence settings')
        self.count, self.visible_sec, self.rest_sec, self.cycles = (
            count, visible_sec, rest_sec, cycles)
        self.phase = 'WAIT_REST'
        self.index = 0
        self.appearances = 0
        self.changed_at = None
        self.rest_since = None
        self.last_time = None
        self.await_face = await_face
        self.shown_at = None
        self.discovery_timeouts = 0

    def action(self, now, ready, at_rest, tracking_face=False):
        if self.last_time is not None and now < self.last_time:
            self.phase = 'RESET'
        self.last_time = now
        if self.changed_at is None:
            self.changed_at = now
        if self.phase == 'RESET':
            return 'hide'
        if self.phase == 'ACQUIRING':
            if tracking_face:
                self.phase, self.changed_at = 'VISIBLE', now
            elif now - self.shown_at >= 60:
                return 'hide'
            return None
        if self.phase == 'VISIBLE':
            return 'hide' if now - self.changed_at >= self.visible_sec else None
        if self.phase == 'COMPLETE':
            return None
        if not ready or not at_rest:
            self.rest_since = None
            return None
        if self.rest_since is None:
            self.rest_since = now
        if now - self.changed_at >= self.rest_sec and now - self.rest_since >= 0.5:
            return 'show'
        return None

    def acknowledge(self, action, now):
        if action == 'show':
            self.phase = 'ACQUIRING' if self.await_face else 'VISIBLE'
            self.shown_at = now
            self.appearances += 1
        elif self.phase == 'RESET':
            self.phase, self.index, self.appearances = 'WAIT_REST', 0, 0
        else:
            if self.phase == 'ACQUIRING':
                self.discovery_timeouts += 1
            self.phase = ('COMPLETE' if self.cycles and self.appearances >=
                          self.count * self.cycles else 'WAIT_REST')
            self.index = (self.index + 1) % self.count
        self.changed_at = now
        self.rest_since = None
