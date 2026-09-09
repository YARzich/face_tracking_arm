// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/following_error_policy.hpp"

#include <cmath>

namespace face_tracking_arm::control
{

FollowingErrorAction classifyFollowingError(
  const double position_error_rad, const double velocity_error_rad_s,
  const FollowingErrorPolicy & policy) noexcept
{
  const bool valid =
    std::isfinite(position_error_rad) && position_error_rad >= 0.0 &&
    std::isfinite(velocity_error_rad_s) && velocity_error_rad_s >= 0.0 &&
    std::isfinite(policy.position_tolerance_rad) &&
    policy.position_tolerance_rad > 0.0 &&
    std::isfinite(policy.velocity_tolerance_rad_s) &&
    policy.velocity_tolerance_rad_s > 0.0 &&
    std::isfinite(policy.recoverable_position_error_rad) &&
    policy.recoverable_position_error_rad > 0.0 &&
    std::isfinite(policy.recoverable_velocity_error_rad_s) &&
    policy.recoverable_velocity_error_rad_s >= policy.velocity_tolerance_rad_s;
  if (!valid) {
    return FollowingErrorAction::kLatchedHalt;
  }

  if (position_error_rad <= policy.position_tolerance_rad &&
    velocity_error_rad_s <= policy.velocity_tolerance_rad_s)
  {
    return FollowingErrorAction::kContinue;
  }

  if (position_error_rad <= policy.recoverable_position_error_rad &&
    velocity_error_rad_s <= policy.recoverable_velocity_error_rad_s)
  {
    return FollowingErrorAction::kControlledRearm;
  }
  return FollowingErrorAction::kLatchedHalt;
}

}  // namespace face_tracking_arm::control
