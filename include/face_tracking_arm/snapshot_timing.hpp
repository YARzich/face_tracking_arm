// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_
#define FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_

#include <cstdint>
#include <utility>

namespace face_tracking_arm::control
{

/// Read a mutable-data snapshot before sampling the clock used to validate its stamp.
template<typename SnapshotReader, typename ClockReader>
auto readSnapshotAtCurrentTime(SnapshotReader && read_snapshot, ClockReader && read_clock)
{
  auto snapshot = std::forward<SnapshotReader>(read_snapshot)();
  auto snapshot_time = std::forward<ClockReader>(read_clock)();
  return std::make_pair(std::move(snapshot), std::move(snapshot_time));
}

/// Validate a state timestamp against the time sampled after reading its snapshot.
inline bool isFreshSnapshotTimestamp(
  const std::int64_t stamp_ns, const std::int64_t snapshot_time_ns,
  const std::int64_t maximum_age_ns) noexcept
{
  return stamp_ns > 0 && snapshot_time_ns > 0 && maximum_age_ns >= 0 &&
         stamp_ns <= snapshot_time_ns && snapshot_time_ns - stamp_ns <= maximum_age_ns;
}

/// Reject a snapshot if the ROS clock rewound after the control tick began.
inline bool isInSameClockEpoch(
  const std::int64_t tick_time_ns, const std::int64_t snapshot_time_ns) noexcept
{
  return tick_time_ns >= 0 && snapshot_time_ns >= tick_time_ns;
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__SNAPSHOT_TIMING_HPP_
