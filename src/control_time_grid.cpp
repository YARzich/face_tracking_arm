// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/control_time_grid.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace face_tracking_arm::control
{
namespace
{

std::int64_t checkedAdd(
  const std::int64_t value, const std::int64_t positive_increment)
{
  if (positive_increment < 0) {
    throw std::invalid_argument("time increment must not be negative");
  }
  if (value > std::numeric_limits<std::int64_t>::max() - positive_increment) {
    throw std::overflow_error("nanosecond timestamp overflow");
  }
  return value + positive_increment;
}

void validateTime(const std::int64_t time_ns, const char * const label)
{
  if (time_ns < 0) {
    throw std::invalid_argument(std::string(label) + " must not be negative");
  }
}

}  // namespace

ControlTimeGrid::ControlTimeGrid(const std::int64_t period_ns)
: period_ns_(period_ns)
{
  if (period_ns <= 0) {
    throw std::invalid_argument("control period must be positive");
  }
}

ControlTimeObservation ControlTimeGrid::observe(const std::int64_t now_ns)
{
  validateTime(now_ns, "observed time");

  if (!previous_observation_ns_.has_value()) {
    const std::int64_t next_due_time_ns = checkedAdd(now_ns, period_ns_);
    previous_observation_ns_ = now_ns;
    next_due_time_ns_ = next_due_time_ns;
    return {
      ControlTimeStatus::kDue, now_ns, next_due_time_ns, 0};
  }

  if (now_ns < *previous_observation_ns_) {
    const std::int64_t next_due_time_ns = checkedAdd(now_ns, period_ns_);
    previous_observation_ns_ = now_ns;
    next_due_time_ns_ = next_due_time_ns;
    return {
      ControlTimeStatus::kRewind, std::nullopt, next_due_time_ns, 0};
  }

  if (now_ns == *previous_observation_ns_) {
    return {
      ControlTimeStatus::kNotAdvancing, std::nullopt,
      *next_due_time_ns_, 0};
  }

  if (now_ns < *next_due_time_ns_) {
    previous_observation_ns_ = now_ns;
    return {
      ControlTimeStatus::kWait, std::nullopt, *next_due_time_ns_, 0};
  }

  const std::int64_t elapsed_ns = now_ns - *next_due_time_ns_;
  const std::int64_t skipped_periods = elapsed_ns / period_ns_;
  const std::int64_t scheduled_time_ns =
    *next_due_time_ns_ + skipped_periods * period_ns_;
  const std::int64_t next_due_time_ns =
    checkedAdd(scheduled_time_ns, period_ns_);
  previous_observation_ns_ = now_ns;
  next_due_time_ns_ = next_due_time_ns;
  return {
    ControlTimeStatus::kDue, scheduled_time_ns, next_due_time_ns,
    static_cast<std::uint64_t>(skipped_periods)};
}

std::optional<std::int64_t> ControlTimeGrid::commandStamp(
  const std::int64_t now_ns, const std::int64_t initial_lead_ns,
  const std::int64_t minimum_lead_ns)
{
  validateTime(now_ns, "command time");
  validateTime(initial_lead_ns, "initial command lead");
  validateTime(minimum_lead_ns, "minimum command lead");
  if (initial_lead_ns < minimum_lead_ns) {
    throw std::invalid_argument(
            "initial command lead must not be smaller than minimum command lead");
  }

  if (!last_command_stamp_ns_.has_value()) {
    const std::int64_t requested_stamp_ns = checkedAdd(now_ns, initial_lead_ns);
    last_command_stamp_ns_ = requested_stamp_ns;
    return requested_stamp_ns;
  }

  const std::int64_t next_grid_stamp_ns =
    checkedAdd(*last_command_stamp_ns_, period_ns_);
  const std::int64_t minimum_safe_stamp_ns = checkedAdd(now_ns, minimum_lead_ns);
  if (next_grid_stamp_ns < minimum_safe_stamp_ns) {
    return std::nullopt;
  }
  last_command_stamp_ns_ = next_grid_stamp_ns;
  return next_grid_stamp_ns;
}

bool ControlTimeGrid::adoptPublishedStamp(const std::int64_t stamp_ns)
{
  validateTime(stamp_ns, "published command stamp");
  if (!last_command_stamp_ns_.has_value() || stamp_ns <= *last_command_stamp_ns_) {
    return false;
  }
  const std::int64_t difference_ns = stamp_ns - *last_command_stamp_ns_;
  if (difference_ns % period_ns_ != 0) {
    return false;
  }
  last_command_stamp_ns_ = stamp_ns;
  return true;
}

void ControlTimeGrid::resetCommandEpoch() noexcept
{
  last_command_stamp_ns_.reset();
}

std::int64_t ControlTimeGrid::periodNs() const noexcept
{
  return period_ns_;
}

}  // namespace face_tracking_arm::control
