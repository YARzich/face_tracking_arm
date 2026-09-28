// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__FOLLOWING_ERROR_POLICY_HPP_
#define FACE_TRACKING_ARM__FOLLOWING_ERROR_POLICY_HPP_

namespace face_tracking_arm::control
{

enum class FollowingErrorAction
{
  kContinue,
  kControlledRearm,
  kLatchedHalt,
};

struct FollowingErrorPolicy final
{
  double position_tolerance_rad{0.0};
  double velocity_tolerance_rad_s{0.0};
};

/// Classify controller lag without treating every transient as a permanent halt.
///
/// Finite lag requests braking and renewed state validation, never a permanent
/// halt. Collision and physical motion checks are performed by the caller.
[[nodiscard]] FollowingErrorAction classifyFollowingError(
  double position_error_rad, double velocity_error_rad_s,
  const FollowingErrorPolicy & policy) noexcept;

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__FOLLOWING_ERROR_POLICY_HPP_
