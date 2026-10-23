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

/// FACE alignment guard used for recovery qualification and the yaw-resource reserve.
inline constexpr double kFaceTrackingErrorGuardRad{0.08726646259971647};  // 5 deg
inline constexpr double kFaceTrackingForecastHorizonSec{0.30};
inline constexpr double kPairYawProbeHorizonSec{1.0};
inline constexpr double kPairYawProbeDisplacementRad{kFaceTrackingErrorGuardRad};

/// The committed behavior of the FACE controller.
enum class FaceTrackingPhase
{
  kTracking,
  kCompactManeuver
};

struct FaceTrackingPolicyInput
{
  /// True only after the caller confirms exact tracking cannot continue.
  /// A small joint reserve or a single rejected control update is insufficient.
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

struct PairYawJointResourceInput
{
  /// Derivative of (face-ray azimuth - screen-normal azimuth) per joint radian.
  double yaw_error_derivative_per_rad{0.0};
  double actual_position_rad{0.0};
  /// Jerk-limited stopping position from the current command/feedback state.
  double predicted_hold_position_rad{0.0};
  double lower_position_rad{0.0};
  double upper_position_rad{0.0};
  double position_margin_rad{0.0};
  /// Existing J1 handoff latch direction; kNone when J1 is not held.
  BaseJointDirection held_direction{BaseJointDirection::kNone};
};

struct PairYawResourceInput
{
  double signed_yaw_error_rad{0.0};
  /// Feed-forward angular rate of the target ray around the tracking origin.
  double target_bearing_rate_radps{0.0};
  double orientation_gain_per_sec{0.0};
  double pointing_deadband_rad{0.0};
  double target_rate_deadband_radps{0.0};
  PairYawJointResourceInput j1;
  PairYawJointResourceInput j4;
};

struct PairYawJointResourceAssessment
{
  double yaw_error_derivative_per_rad{0.0};
  double actual_position_rad{0.0};
  double predicted_hold_position_rad{0.0};
  double lower_position_rad{0.0};
  double upper_position_rad{0.0};
  double position_margin_rad{0.0};
  BaseJointDirection required_direction{BaseJointDirection::kNone};
  double remaining_joint_travel_rad{0.0};
  double remaining_yaw_rad{0.0};
  bool held_for_required_direction{false};
  bool directionally_bound{false};
};

struct PairYawResourceAssessment
{
  bool valid{false};
  bool demand_present{false};
  /// A required J1/J4 direction is position- or handoff-bound. This is only a
  /// candidate; the counterfactual orientation-preserving probe proves causality.
  bool pair_bound_candidate{false};
  bool probe_valid{false};
  bool actual_probe_solved{false};
  bool actual_probe_infeasible{false};
  bool relaxed_probe_solved{false};
  bool pair_bound_cause{false};
  bool exhausted{false};
  double signed_yaw_error_rad{0.0};
  double target_bearing_rate_radps{0.0};
  double yaw_demand_radps{0.0};
  double probe_horizon_sec{kPairYawProbeHorizonSec};
  double actual_available_yaw_rad{0.0};
  double pair_relaxed_available_yaw_rad{0.0};
  double actual_roll_residual_rad{0.0};
  double relaxed_roll_residual_rad{0.0};
  double actual_elevation_residual_rad{0.0};
  double relaxed_elevation_residual_rad{0.0};
  PairYawJointResourceAssessment j1;
  PairYawJointResourceAssessment j4;
};

enum class PairYawResourceState
{
  kUnknown,
  kAvailable,
  kExhausted,
};

/// Classify only evidence that is sufficient to act on pair-resource state.
/// A failed probe leaves the state unknown unless a solved actual probe already
/// proves that useful yaw reserve remains.
[[nodiscard]] inline PairYawResourceState pairYawResourceState(
  const PairYawResourceAssessment & assessment) noexcept
{
  if (!assessment.valid) {
    return PairYawResourceState::kUnknown;
  }
  if (!assessment.demand_present || !assessment.pair_bound_candidate) {
    return PairYawResourceState::kAvailable;
  }
  if (assessment.probe_valid && assessment.actual_probe_solved &&
    !assessment.actual_probe_infeasible &&
    std::isfinite(assessment.actual_available_yaw_rad) &&
    assessment.actual_available_yaw_rad >= 0.80 * kPairYawProbeDisplacementRad)
  {
    return PairYawResourceState::kAvailable;
  }
  return assessment.exhausted ? PairYawResourceState::kExhausted :
         PairYawResourceState::kUnknown;
}

/// Compare the bounded reserve probes. The actual configuration must be unable
/// to provide most of a small yaw increment, while relaxing only the required
/// J1/J4 position direction must restore it without tilting the screen.
[[nodiscard]] inline bool pairYawProbeShowsPairBoundCause(
  const PairYawResourceAssessment & assessment) noexcept
{
  constexpr double kUsefulFraction{0.80};
  constexpr double kMinimumCausalImprovementRad{0.017453292519943295};  // 1 deg
  constexpr double kPoseHoldToleranceRad{0.017453292519943295};  // 1 deg / probe
  if (!assessment.valid || !assessment.demand_present ||
    !assessment.pair_bound_candidate || !assessment.probe_valid ||
    !assessment.actual_probe_solved ||
    !assessment.relaxed_probe_solved ||
    !std::isfinite(assessment.actual_available_yaw_rad) ||
    !std::isfinite(assessment.pair_relaxed_available_yaw_rad) ||
    !std::isfinite(assessment.actual_roll_residual_rad) ||
    !std::isfinite(assessment.relaxed_roll_residual_rad) ||
    !std::isfinite(assessment.actual_elevation_residual_rad) ||
    !std::isfinite(assessment.relaxed_elevation_residual_rad) ||
    assessment.actual_available_yaw_rad < 0.0 ||
    assessment.pair_relaxed_available_yaw_rad < 0.0 ||
    assessment.actual_roll_residual_rad < 0.0 ||
    assessment.relaxed_roll_residual_rad < 0.0 ||
    assessment.actual_elevation_residual_rad < 0.0 ||
    assessment.relaxed_elevation_residual_rad < 0.0 ||
    !std::isfinite(assessment.probe_horizon_sec) || assessment.probe_horizon_sec <= 0.0)
  {
    return false;
  }
  const double useful_yaw = kPairYawProbeDisplacementRad;
  return assessment.actual_available_yaw_rad < kUsefulFraction * useful_yaw &&
         assessment.pair_relaxed_available_yaw_rad >= kUsefulFraction * useful_yaw &&
         assessment.pair_relaxed_available_yaw_rad -
         assessment.actual_available_yaw_rad >= kMinimumCausalImprovementRad &&
         assessment.relaxed_roll_residual_rad <= kPoseHoldToleranceRad + 1.0e-4 &&
         assessment.relaxed_elevation_residual_rad <= kPoseHoldToleranceRad + 1.0e-4;
}

/// Mark a candidate only when yaw is demanded and at least one J1/J4 direction
/// is bound. A separate orientation-preserving QP counterfactual must prove the
/// pair bound caused the yaw shortfall before `exhausted` becomes true.
[[nodiscard]] inline PairYawResourceAssessment assessPairYawResource(
  const PairYawResourceInput & input) noexcept
{
  PairYawResourceAssessment result;
  const auto valid_joint = [](const PairYawJointResourceInput & joint) {
      return std::isfinite(joint.yaw_error_derivative_per_rad) &&
             std::isfinite(joint.actual_position_rad) &&
             std::isfinite(joint.predicted_hold_position_rad) &&
             std::isfinite(joint.lower_position_rad) &&
             std::isfinite(joint.upper_position_rad) &&
             std::isfinite(joint.position_margin_rad) &&
             joint.lower_position_rad < joint.upper_position_rad &&
             joint.position_margin_rad >= 0.0 &&
             joint.position_margin_rad * 2.0 <
             joint.upper_position_rad - joint.lower_position_rad &&
             joint.actual_position_rad >= joint.lower_position_rad - 0.01 &&
             joint.actual_position_rad <= joint.upper_position_rad + 0.01;
    };
  if (!std::isfinite(input.signed_yaw_error_rad) ||
    !std::isfinite(input.target_bearing_rate_radps) ||
    !std::isfinite(input.orientation_gain_per_sec) || input.orientation_gain_per_sec < 0.0 ||
    !std::isfinite(input.pointing_deadband_rad) || input.pointing_deadband_rad < 0.0 ||
    !std::isfinite(input.target_rate_deadband_radps) || input.target_rate_deadband_radps < 0.0 ||
    !valid_joint(input.j1) ||
    !valid_joint(input.j4))
  {
    return result;
  }

  const auto outside_deadband = [](const double value, const double deadband) {
      return std::abs(value) > deadband ? value : 0.0;
    };
  const double target_rate = outside_deadband(
    input.target_bearing_rate_radps, input.target_rate_deadband_radps);
  const double yaw_error = outside_deadband(
    input.signed_yaw_error_rad, input.pointing_deadband_rad);
  result.yaw_demand_radps = target_rate + input.orientation_gain_per_sec * yaw_error;
  if (!std::isfinite(result.yaw_demand_radps)) {
    return PairYawResourceAssessment{};
  }
  result.valid = true;
  result.signed_yaw_error_rad = input.signed_yaw_error_rad;
  result.target_bearing_rate_radps = input.target_bearing_rate_radps;
  result.demand_present = std::abs(result.yaw_demand_radps) > 1.0e-6;
  if (!result.demand_present) {
    return result;
  }

  constexpr double kDirectionalBoundToleranceRad{0.01};
  const auto assess_joint = [&result](
    const PairYawJointResourceInput & joint,
    PairYawJointResourceAssessment & assessment) {
      assessment.yaw_error_derivative_per_rad = joint.yaw_error_derivative_per_rad;
      assessment.actual_position_rad = joint.actual_position_rad;
      assessment.predicted_hold_position_rad = joint.predicted_hold_position_rad;
      assessment.lower_position_rad = joint.lower_position_rad;
      assessment.upper_position_rad = joint.upper_position_rad;
      assessment.position_margin_rad = joint.position_margin_rad;
      const double derivative = joint.yaw_error_derivative_per_rad;
      if (std::abs(derivative) <= 1.0e-9) {
        return;
      }
      const double required_velocity_sign =
        -result.yaw_demand_radps / derivative;
      assessment.required_direction = required_velocity_sign > 0.0 ?
        BaseJointDirection::kPositive : BaseJointDirection::kNegative;
      if (joint.held_direction == assessment.required_direction) {
        assessment.held_for_required_direction = true;
        assessment.directionally_bound = true;
        return;
      }
      const double allowed_lower = joint.lower_position_rad + joint.position_margin_rad;
      const double allowed_upper = joint.upper_position_rad - joint.position_margin_rad;
      assessment.remaining_joint_travel_rad = std::max(0.0,
        assessment.required_direction == BaseJointDirection::kPositive ?
        allowed_upper - joint.predicted_hold_position_rad :
        joint.predicted_hold_position_rad - allowed_lower);
      assessment.remaining_yaw_rad = std::abs(derivative) *
        assessment.remaining_joint_travel_rad;
      assessment.directionally_bound = assessment.remaining_joint_travel_rad <=
        kDirectionalBoundToleranceRad;
    };
  assess_joint(input.j1, result.j1);
  assess_joint(input.j4, result.j4);
  result.pair_bound_candidate =
    result.j1.directionally_bound || result.j4.directionally_bound;
  result.exhausted = false;
  return result;
}

/// Signed shortest change between consecutive target bearings. This wraps only
/// the target angle to keep its motion continuous across +/- pi; actual joint
/// positions are always retained as measured bounded coordinates.
[[nodiscard]] inline std::optional<double> targetBearingDeltaRad(
  const double previous_bearing_rad, const double current_bearing_rad) noexcept
{
  if (!std::isfinite(previous_bearing_rad) || !std::isfinite(current_bearing_rad)) {
    return std::nullopt;
  }
  const double difference = current_bearing_rad - previous_bearing_rad;
  return std::atan2(std::sin(difference), std::cos(difference));
}

struct BaseJointHandoffInput
{
  bool working_edge_reached{false};
  /// Measured bounded J1 position, used to validate the snapshot.
  double actual_base_position_rad{0.0};
  /// Jerk-limited stopping endpoint computed from the current q/v/a state.
  /// When already settled this equals actual_base_position_rad.
  double predicted_hold_position_rad{0.0};
  BaseJointDirection edge_direction{BaseJointDirection::kNone};
  /// Direction of the ordinary exact FACE yaw request at this working edge.
  BaseJointDirection tracking_yaw_direction{BaseJointDirection::kNone};
  /// Caller-qualified direction of fresh target-bearing motion; small/noisy
  /// deltas must be reported as kNone after existing motion filtering.
  BaseJointDirection target_bearing_motion_direction{BaseJointDirection::kNone};
  bool valid_target_changed{false};
};

/// Holds only J1 at its reached working edge while normal FACE yaw continues
/// through the remaining task Jacobian. This is a position latch, not a joint
/// velocity command. The caller must translate it into a dynamically feasible
/// per-joint velocity constraint using the existing q/v/a/jerk state and QP;
/// it must not be used only as a weighted secondary objective. A qualified
/// reverse target-bearing motion or valid target-ID change releases the latch.
class BaseJointHandoffPolicy final
{
public:
  [[nodiscard]] std::optional<double> update(const BaseJointHandoffInput & input) noexcept
  {
    if (held_position_rad_) {
      const bool reversed = input.target_bearing_motion_direction != BaseJointDirection::kNone &&
        input.target_bearing_motion_direction != held_direction_;
      if (input.valid_target_changed || reversed) {
        clear();
        return std::nullopt;
      }
      return held_position_rad_;
    }

    if (!input.working_edge_reached || !std::isfinite(input.actual_base_position_rad) ||
      !std::isfinite(input.predicted_hold_position_rad) ||
      input.edge_direction == BaseJointDirection::kNone ||
      input.tracking_yaw_direction != input.edge_direction)
    {
      return std::nullopt;
    }

    held_position_rad_ = input.predicted_hold_position_rad;
    held_direction_ = input.edge_direction;
    return held_position_rad_;
  }

  void clear() noexcept
  {
    held_position_rad_.reset();
    held_direction_ = BaseJointDirection::kNone;
  }

  [[nodiscard]] std::optional<double> heldPositionRad() const noexcept
  {
    return held_position_rad_;
  }

  [[nodiscard]] BaseJointDirection heldDirection() const noexcept
  {
    return held_direction_;
  }

private:
  std::optional<double> held_position_rad_;
  BaseJointDirection held_direction_{BaseJointDirection::kNone};
};

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
