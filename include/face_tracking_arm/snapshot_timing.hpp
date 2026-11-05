// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_
#define FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_

#include <cstdint>
#include <utility>

namespace face_tracking_arm::control
{

// Covers the observed 2 ms sim-clock lead plus one 1 ms physics step.
inline constexpr std::int64_t kSimTimeFeedbackFutureToleranceNs = 3'000'000LL;

/// Read a mutable-data snapshot before sampling the clock used to validate its stamp.
template<typename SnapshotReader, typename ClockReader>
auto readSnapshotAtCurrentTime(SnapshotReader && read_snapshot, ClockReader && read_clock)
{
  auto snapshot = std::forward<SnapshotReader>(read_snapshot)();
  auto snapshot_time = std::forward<ClockReader>(read_clock)();
  return std::make_pair(std::move(snapshot), std::move(snapshot_time));
}

/// Validate a state timestamp against the supplied reference time.
inline bool isFreshSnapshotTimestamp(
  const std::int64_t stamp_ns, const std::int64_t snapshot_time_ns,
  const std::int64_t maximum_age_ns,
  const std::int64_t maximum_future_ns = 0) noexcept
{
  if (stamp_ns <= 0 || snapshot_time_ns <= 0 || maximum_age_ns < 0 || maximum_future_ns < 0) {
    return false;
  }
  if (stamp_ns > snapshot_time_ns) {
    return stamp_ns - snapshot_time_ns <= maximum_future_ns;
  }
  return snapshot_time_ns - stamp_ns <= maximum_age_ns;
}

/// Reject a snapshot if the ROS clock rewound after the control tick began.
inline bool isInSameClockEpoch(
  const std::int64_t tick_time_ns, const std::int64_t snapshot_time_ns) noexcept
{
  return tick_time_ns >= 0 && snapshot_time_ns >= tick_time_ns;
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_
