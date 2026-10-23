// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef SERVO_TYPES_HPP_
#define SERVO_TYPES_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <moveit_servo/utils/datatypes.hpp>

namespace face_tracking_arm
{
enum class ControllerMode
{
  kTracking,
  kBraking,
  kWaitingForSafeState,
};

enum class CommandAttempt
{
  kTracking,
  kBraking,
};

enum class CandidateRejectionKind
{
  kOther,
  kReferenceGuard,
  kCompactStopEnvelope,
};

inline constexpr double kFiniteEpsilon = 1.0e-12;

inline constexpr double kStoppedVelocityRadps = 1.0e-2;

inline constexpr double kSafeRearmSettleBudgetSec = 2.0;

// The JTC reconstructs the same spline with finite timestamp/float precision.
// Tens of microradians remain far inside the collision tracking reserve.

inline constexpr double kControllerCommandAlignmentToleranceRad = 1.0e-4;

inline constexpr double kControllerSpeedScalingTolerance = 1.0e-9;

/// A following-position offset may be rebased only after a stationary hold has
/// an identified owner. Before our first publication that is the controller's
/// own settled reference/output pair; afterwards it is an acknowledged checked
/// publication. Feedback freshness and safety remain separate prerequisites.
inline bool settled_rearm_state(
  const double reference_output_error, const double following_position_error,
  const double reference_velocity, const double feedback_velocity,
  const double command_alignment_tolerance, const double following_error_bound,
  const bool feedback_fresh, const bool feedback_safe,
  const bool stationary_hold_owned)
{
  if (!feedback_fresh || !feedback_safe || !std::isfinite(reference_output_error) ||
    !std::isfinite(following_position_error) || !std::isfinite(reference_velocity) ||
    !std::isfinite(feedback_velocity) || !std::isfinite(command_alignment_tolerance) ||
    !std::isfinite(following_error_bound))
  {
    return false;
  }
  return reference_output_error <= command_alignment_tolerance &&
         reference_velocity <= kStoppedVelocityRadps &&
         feedback_velocity <= kStoppedVelocityRadps &&
         (following_position_error <= following_error_bound ||
         stationary_hold_owned);
}

inline constexpr std::size_t kPublishedTrajectoryHistoryCapacity = 16;

inline const char * command_attempt_name(const CommandAttempt attempt)
{
  return attempt == CommandAttempt::kTracking ? "tracking" : "braking";
}

inline std::int64_t seconds_to_nanoseconds(const double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0) {
    throw std::invalid_argument("duration must be finite and non-negative");
  }
  constexpr double nanoseconds_per_second = 1.0e9;
  const double nanoseconds = seconds * nanoseconds_per_second;
  if (nanoseconds > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("duration cannot be represented in nanoseconds");
  }
  return static_cast<std::int64_t>(std::llround(nanoseconds));
}

inline bool finite_state(const moveit_servo::KinematicState & state, const std::size_t size)
{
  return state.joint_names.size() == size &&
         static_cast<std::size_t>(state.positions.size()) == size &&
         static_cast<std::size_t>(state.velocities.size()) == size &&
         static_cast<std::size_t>(state.accelerations.size()) == size &&
         state.positions.allFinite() && state.velocities.allFinite() &&
         state.accelerations.allFinite();
}


}  // namespace face_tracking_arm

#endif  // SERVO_TYPES_HPP_
