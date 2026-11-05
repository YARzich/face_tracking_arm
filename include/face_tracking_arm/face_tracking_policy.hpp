// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__FACE_TRACKING_POLICY_HPP_
#define FACE_TRACKING_ARM__FACE_TRACKING_POLICY_HPP_

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <optional>

namespace face_tracking_arm::control
{

/// FACE alignment guard used for recovery qualification and tracking guards.
inline constexpr double kFaceTrackingErrorGuardRad{0.08726646259971647};  // 5 deg
inline constexpr double kFaceTrackingForecastHorizonSec{0.30};

/// The committed behavior of the FACE controller.
enum class FaceTrackingPhase
{
  kTracking,
  kCompactManeuver
};

struct FaceTrackingPolicyInput
{
  /// True when a valid J1 boundary assessment reaches its working interval edge.
  bool need_to_reconfigure{false};

  /// True when the already accepted compact maneuver has reached its goal.
  bool maneuver_complete{false};
};

struct FaceTrackingErrorForecast
{
  /// Angle diagnostics: screen-normal left/up and horizontal-edge tilt, radians.
  /// When horizontal_edge_axis is true, axis 2 is not used for prediction.
  Eigen::Vector3d signed_error_rad{Eigen::Vector3d::Zero()};
  /// Target-only feed-forward for the first two angular rows, excluding gain*error.
  Eigen::Vector3d desired_feedforward_radps{Eigen::Vector3d::Zero()};
  /// Achieved QP primary velocity. The third FACE row is edge-height rate, not rad/s.
  Eigen::Vector3d achieved_primary_radps{Eigen::Vector3d::Zero()};
  /// When true, the explicit third-row forecast uses sin(tilt) (dimensionless)
  /// and its time derivative (1/s); edge_guard is sin(angle guard).
  bool horizontal_edge_axis{false};
  double signed_edge_error{0.0};
  double desired_edge_rate{0.0};
  double achieved_edge_rate{0.0};
  double edge_guard{0.0};
  double horizon_sec{0.0};
  double pointing_guard_rad{0.0};
  double roll_guard_rad{0.0};
};

/// Predict signed task-coordinate error using e_dot ~= target_feedforward -
/// achieved_relative_angular_velocity, then test normal error norm and absolute
/// roll. Do not extrapolate a closing component through zero: at alignment its
/// correction direction must be recomputed. This prevents a fast correction
/// from appearing to cause persistent loss of tracking on the opposite side.
/// Do not subtract achieved velocity from the complete requested reference: it
/// includes proportional gain*error and forecasts growth even while a stationary
/// error is being corrected. The caller qualifies valid commands and persistent
/// shortfall separately; this remains a candidate heuristic, not a feasibility
/// proof.
[[nodiscard]] inline bool trackingErrorWillCrossGuard(
  const FaceTrackingErrorForecast & forecast) noexcept
{
  if (!forecast.signed_error_rad.allFinite() ||
    !forecast.desired_feedforward_radps.allFinite() ||
    !forecast.achieved_primary_radps.allFinite() ||
    (forecast.horizontal_edge_axis && (!std::isfinite(forecast.signed_edge_error) ||
    !std::isfinite(forecast.desired_edge_rate) ||
    !std::isfinite(forecast.achieved_edge_rate) || !std::isfinite(forecast.edge_guard) ||
    forecast.edge_guard <= 0.0)) ||
    !std::isfinite(forecast.horizon_sec) || forecast.horizon_sec <= 0.0 ||
    !std::isfinite(forecast.pointing_guard_rad) || forecast.pointing_guard_rad <= 0.0 ||
    !std::isfinite(forecast.roll_guard_rad) || forecast.roll_guard_rad <= 0.0)
  {
    return false;
  }

  const Eigen::Vector3d relative_error_rate =
    forecast.desired_feedforward_radps - forecast.achieved_primary_radps;
  Eigen::Vector3d predicted_error = forecast.signed_error_rad +
    forecast.horizon_sec * relative_error_rate;
  if (!predicted_error.allFinite()) {
    return false;
  }

  // A linear horizon is useful for detecting sustained growth, but a closing
  // component cannot be assumed to keep the same sign after reaching zero.
  for (Eigen::Index axis = 0; axis < predicted_error.size(); ++axis) {
    if (forecast.signed_error_rad[axis] * relative_error_rate[axis] < 0.0 &&
      forecast.signed_error_rad[axis] * predicted_error[axis] <= 0.0)
    {
      predicted_error[axis] = 0.0;
    }
  }

  const double current_pointing_error = forecast.signed_error_rad.head<2>().norm();
  const double predicted_pointing_error = predicted_error.head<2>().norm();
  constexpr double kProgressEpsilonRad = 1.0e-9;
  const bool pointing_error_not_closing =
    predicted_pointing_error + kProgressEpsilonRad >= current_pointing_error;
  if (forecast.horizontal_edge_axis) {
    const double relative_edge_error_rate =
      forecast.desired_edge_rate - forecast.achieved_edge_rate;
    double predicted_edge_error = forecast.signed_edge_error +
      forecast.horizon_sec * relative_edge_error_rate;
    if (forecast.signed_edge_error * relative_edge_error_rate < 0.0 &&
      forecast.signed_edge_error * predicted_edge_error <= 0.0)
    {
      predicted_edge_error = 0.0;
    }
    const bool edge_error_not_closing = std::abs(predicted_edge_error) + 1.0e-9 >=
      std::abs(forecast.signed_edge_error);
    return (predicted_pointing_error >= forecast.pointing_guard_rad &&
           pointing_error_not_closing) ||
           (std::abs(predicted_edge_error) >= forecast.edge_guard &&
           edge_error_not_closing);
  }
  const double current_roll_error = std::abs(forecast.signed_error_rad[2]);
  const double predicted_roll_error = std::abs(predicted_error[2]);
  const bool roll_error_not_closing =
    predicted_roll_error + kProgressEpsilonRad >= current_roll_error;
  return (predicted_pointing_error >= forecast.pointing_guard_rad &&
         pointing_error_not_closing) ||
         (predicted_roll_error >= forecast.roll_guard_rad &&
         roll_error_not_closing);
}

enum class BaseJointDirection : int
{
  kNegative = -1,
  kNone = 0,
  kPositive = 1
};

struct J1BoundaryInput
{
  double actual_position_rad{0.0};
  double brake_min_position_rad{0.0};
  double brake_max_position_rad{0.0};
  double lower_position_rad{0.0};
  double upper_position_rad{0.0};
  double position_margin_rad{0.0};
  /// Arithmetic position uncertainty from the existing motion numerical policy.
  double position_error_rad{0.0};
};

struct J1BoundaryAssessment
{
  bool valid{false};
  BaseJointDirection edge_direction{BaseJointDirection::kNone};
  double actual_position_rad{0.0};
  double brake_min_position_rad{0.0};
  double brake_max_position_rad{0.0};
  double working_lower_rad{0.0};
  double working_upper_rad{0.0};
};

/// Classify J1's working-boundary contact from measured position and the
/// extrema of the exact scalar braking rollout supplied by the caller.
[[nodiscard]] inline J1BoundaryAssessment assessJ1Boundary(
  const J1BoundaryInput & input) noexcept
{
  J1BoundaryAssessment result;
  const double working_lower = input.lower_position_rad + input.position_margin_rad;
  const double working_upper = input.upper_position_rad - input.position_margin_rad;
  const double lower_contact = working_lower + input.position_error_rad;
  const double upper_contact = working_upper - input.position_error_rad;
  if (!std::isfinite(input.actual_position_rad) ||
    !std::isfinite(input.brake_min_position_rad) ||
    !std::isfinite(input.brake_max_position_rad) ||
    !std::isfinite(input.lower_position_rad) ||
    !std::isfinite(input.upper_position_rad) ||
    !std::isfinite(input.position_margin_rad) || input.position_margin_rad < 0.0 ||
    !std::isfinite(input.position_error_rad) || input.position_error_rad < 0.0 ||
    input.lower_position_rad >= input.upper_position_rad ||
    !std::isfinite(working_lower) || !std::isfinite(working_upper) ||
    !std::isfinite(lower_contact) || !std::isfinite(upper_contact) ||
    working_lower >= working_upper ||
    input.brake_min_position_rad > input.brake_max_position_rad)
  {
    return result;
  }

  const bool lower_reached = std::min(
    input.actual_position_rad, input.brake_min_position_rad) <= lower_contact;
  const bool upper_reached = std::max(
    input.actual_position_rad, input.brake_max_position_rad) >= upper_contact;
  if (lower_reached && upper_reached) {
    return result;
  }
  result.valid = true;
  result.actual_position_rad = input.actual_position_rad;
  result.brake_min_position_rad = input.brake_min_position_rad;
  result.brake_max_position_rad = input.brake_max_position_rad;
  result.working_lower_rad = working_lower;
  result.working_upper_rad = working_upper;
  result.edge_direction = lower_reached ? BaseJointDirection::kNegative :
    upper_reached ? BaseJointDirection::kPositive : BaseJointDirection::kNone;
  return result;
}

/// Selects between exact tracking and one committed compact maneuver.
///
/// This policy has no kinematic or planning logic. The caller supplies a
/// qualified need to reconfigure and reports completion of the accepted
/// maneuver. Face loss, target reversal, and ordinary controller rearm do not
/// cancel a committed maneuver. Explicit stop handling may call clearForStop().
class FaceTrackingPolicy final
{
public:
  [[nodiscard]] FaceTrackingPhase phase() const noexcept
  {
    return phase_;
  }

  [[nodiscard]] FaceTrackingPhase update(const FaceTrackingPolicyInput & input) noexcept
  {
    if (phase_ == FaceTrackingPhase::kCompactManeuver) {
      if (input.maneuver_complete) {
        phase_ = FaceTrackingPhase::kTracking;
      }
      return phase_;
    }

    if (input.need_to_reconfigure) {
      phase_ = FaceTrackingPhase::kCompactManeuver;
    }
    return phase_;
  }

  /// Clear policy state after an explicit stop/reset, not on face loss or rearm.
  void clearForStop() noexcept
  {
    phase_ = FaceTrackingPhase::kTracking;
  }

private:
  FaceTrackingPhase phase_{FaceTrackingPhase::kTracking};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__FACE_TRACKING_POLICY_HPP_
