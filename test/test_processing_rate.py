# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Cadence and rate gate behavior using deterministic monotonic timestamps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from face_tracking_perception.processing_rate import (  # noqa: E402
    ProcessingRateGate, StableFrameCadence)
import pytest  # noqa: E402


def test_rate_cap_uses_a_persistent_deadline_for_30_to_20_fps():
    gate = ProcessingRateGate(20)
    accepted = [frame / 30 for frame in range(90) if gate.accept(frame / 30)]
    assert len(accepted) == pytest.approx(60, abs=1)
    assert accepted[-1] - accepted[0] == pytest.approx(2.95, abs=1 / 30)


def test_rate_gate_never_catches_up_after_slow_inference():
    gate = ProcessingRateGate(20)
    assert gate.accept(1.0)
    assert not gate.accept(1.01)
    assert gate.accept(1.20)
    assert not gate.accept(1.21)
    assert gate.accept(1.26)


def test_cadence_estimator_uses_valid_capture_times_and_rejects_jumps():
    cadence = StableFrameCadence()
    for index in range(6):
        cadence.observe(index / 15, 10 + index / 15)
    assert cadence.fps == pytest.approx(15)
    cadence.observe(2.0, 11.0)  # Time jump resets only the interval reference.
    assert cadence.fps == pytest.approx(15)
    cadence.observe(2.0 + 1 / 10, 11.1)
    assert cadence.fps == pytest.approx(15)  # A single new interval cannot relearn cadence.


def test_cadence_estimator_never_learns_duplicate_or_future_receipt_order():
    cadence = StableFrameCadence()
    cadence.observe(1.0, 5.0)
    cadence.observe(1.0, 5.1)
    assert cadence.fps is None
    cadence.observe(1.1, 5.0)
    assert cadence.fps is None


def test_auto_gap_uses_bounded_valid_pair_receipt_jitter_not_requested_capture_rate():
    cadence = StableFrameCadence()
    receipts = [0.0, 0.066, 0.132, 0.264, 0.330]
    for index, receipt in enumerate(receipts):
        cadence.observe(index / 15, receipt)
    assert cadence.stable_fps == pytest.approx(15)
    assert cadence.bounded_gap_limit(.30) == pytest.approx(.264)
    assert cadence.bounded_gap_limit(.25) == pytest.approx(.25)


def test_auto_gap_warmup_stays_at_the_configured_cap():
    cadence = StableFrameCadence()
    for index in range(3):
        cadence.observe(index / 15, index / 15)
    assert cadence.bounded_gap_limit(.30) == pytest.approx(.30)
