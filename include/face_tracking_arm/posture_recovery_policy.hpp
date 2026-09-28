// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_
#define FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <optional>

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

/// Detect a stalled pointing objective, including static targets and narrow
/// joints. Distance/roll errors alone do not justify a disruptive detour.
class PointingProgressMonitor
{
public:
  bool update(double error_rad, double time_sec)
  {
    if (!std::isfinite(error_rad) || !std::isfinite(time_sec) || error_rad <= 0.03) {
      reset();
      return false;
    }
    if (!since_ || time_sec < *since_ || error_rad + 0.015 < best_error_) {
      since_ = time_sec;
      best_error_ = error_rad;
      return false;
    }
    return time_sec - *since_ >= 1.5;
  }

  void reset() {since_.reset();}

private:
  std::optional<double> since_;
  double best_error_{0.0};
};

/// Two bounded attempts per level: retain pointing, then permit a modest cone,
/// finally permit a checked joint detour. This is not a retry limit.
inline unsigned int recoveryRelaxation(unsigned int failed_attempts)
{
  return std::min(2u, failed_attempts / 2u);
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__POSTURE_RECOVERY_POLICY_HPP_
