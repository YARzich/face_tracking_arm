// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_
#define FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_

#include <cmath>

namespace face_tracking_arm::control
{

/// A planning preference, not a replacement for the executor's braking limits.
/// Wait for an actual approach to a wide bounded joint's end. A stationary face
/// or inward motion does not justify a disruptive global posture change.
inline bool needsPostureRecovery(
  double position, double velocity, double lower, double upper, double margin,
  double face_speed)
{
  if (!std::isfinite(position) || !std::isfinite(velocity) ||
    !std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(margin) ||
    !std::isfinite(face_speed) || upper - lower <= 2.0 * M_PI ||
    margin < 0.0 || face_speed <= 0.0)
  {
    return false;
  }
  const double center = 0.5 * (lower + upper);
  const double direction = position >= center ? 1.0 : -1.0;
  const double outward_velocity = direction * velocity;
  const double remaining = direction > 0.0 ? upper - margin - position :
    position - lower - margin;
  // The existing motion estimator already removes stationary numerical noise.
  // One radian of reserve plus two seconds at the measured approach speed.
  // The close-boundary clause also handles a joint already slowed by the QP.
  return outward_velocity >= -1.0e-6 &&
         (remaining < 0.25 || (outward_velocity > 0.02 &&
         remaining < 1.0 + 2.0 * outward_velocity));
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_
