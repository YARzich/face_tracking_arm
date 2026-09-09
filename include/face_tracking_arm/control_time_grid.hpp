// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__CONTROL_TIME_GRID_HPP_
#define FACE_TRACKING_ARM__CONTROL_TIME_GRID_HPP_

#include <cstdint>
#include <optional>

namespace face_tracking_arm::control
{

enum class ControlTimeStatus : std::uint8_t
{
  kDue,
  kWait,
  kNotAdvancing,
  kRewind,
};

struct ControlTimeObservation final
{
  ControlTimeStatus status{ControlTimeStatus::kNotAdvancing};

  /// Grid boundary represented by this observation. Set only for kDue.
  std::optional<std::int64_t> scheduled_time_ns;

  /// First grid boundary strictly after the observed time.
  std::int64_t next_due_time_ns{0};

  /// Complete grid slots discarded before the boundary represented by kDue.
  std::uint64_t skipped_periods{0};

  [[nodiscard]] bool due() const noexcept
  {
    return status == ControlTimeStatus::kDue;
  }
};

/// Fixed-period scheduler driven exclusively by an externally supplied clock.
///
/// The first observation is due immediately. Subsequent due observations remain
/// on that fixed grid, so jitter cannot move the epoch. When several boundaries
/// pass between observations, only the latest one is reported and older slots
/// are counted as skipped. A rewind starts a new observation epoch and is never
/// itself reported as due.
class ControlTimeGrid final
{
public:
  explicit ControlTimeGrid(std::int64_t period_ns);

  [[nodiscard]] ControlTimeObservation observe(std::int64_t now_ns);

  /// Start a command epoch at now + initial_lead, then return stamps exactly
  /// one period apart. If the buffered timeline falls below minimum_lead,
  /// return nullopt without rebasing it: changing one interval would invalidate
  /// the acceleration and jerk guarantees of the fixed-period controller.
  [[nodiscard]] std::optional<std::int64_t> commandStamp(
    std::int64_t now_ns, std::int64_t initial_lead_ns,
    std::int64_t minimum_lead_ns);

  /// Move the active command epoch to an already-published future stamp.
  ///
  /// This is used when a late control callback adopts an immutable prefix of
  /// the previously published braking suffix. The stamp must be later than the
  /// current command stamp and lie on the same fixed-period grid. No lead check
  /// is needed because the point has already been accepted by the controller.
  [[nodiscard]] bool adoptPublishedStamp(std::int64_t stamp_ns);

  /// Start a new command-stamp epoch without changing the observation grid.
  void resetCommandEpoch() noexcept;

  [[nodiscard]] std::int64_t periodNs() const noexcept;

private:
  std::int64_t period_ns_;
  std::optional<std::int64_t> previous_observation_ns_;
  std::optional<std::int64_t> next_due_time_ns_;
  std::optional<std::int64_t> last_command_stamp_ns_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__CONTROL_TIME_GRID_HPP_
