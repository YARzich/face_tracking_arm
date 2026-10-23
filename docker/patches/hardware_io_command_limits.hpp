// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>

namespace face_tracking_arm
{
namespace hardware
{

struct PositionCommandValidation
{
  bool valid = false;
  std::size_t joint_index = 0;
};

inline PositionCommandValidation validate_finite_position_command(
  const double * requested, const float * wire, std::size_t count) noexcept
{
  if (requested == nullptr || wire == nullptr || count != 6) {
    return {false, 0};
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(requested[i]) || !std::isfinite(wire[i])) {
      return {false, i};
    }
  }
  return {true, count};
}

inline bool position_command_allowed(
  bool runtime_actions_enabled, bool limits_available,
  bool finite_candidate_valid, bool bounded_candidate_valid) noexcept
{
  return finite_candidate_valid &&
         (!runtime_actions_enabled || (limits_available && bounded_candidate_valid));
}

inline PositionCommandValidation validate_position_command(
  const double * requested, const float * wire,
  const double * minimum, const double * maximum,
  const bool * limit_valid, std::size_t count) noexcept
{
  if (requested == nullptr || wire == nullptr || minimum == nullptr ||
    maximum == nullptr || limit_valid == nullptr || count != 6)
  {
    return {false, 0};
  }
  for (std::size_t i = 0; i < count; ++i) {
    const double wire_value = static_cast<double>(wire[i]);
    if (!limit_valid[i] || !std::isfinite(minimum[i]) || !std::isfinite(maximum[i]) ||
      minimum[i] >= maximum[i] || !std::isfinite(requested[i]) ||
      !std::isfinite(wire[i]) || requested[i] < minimum[i] ||
      requested[i] > maximum[i] || wire_value<minimum[i] ||
      wire_value> maximum[i])
    {
      return {false, i};
    }
  }
  return {true, count};
}

}  // namespace hardware
}  // namespace face_tracking_arm
