// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "face_tracking_arm/face_tracking_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

FaceTrackingErrorForecast forecast(
  const Eigen::Vector3d & signed_error,
  const Eigen::Vector3d & feedforward,
  const Eigen::Vector3d & achieved,
  const double horizon_sec, const double guard)
{
  FaceTrackingErrorForecast result;
  result.signed_error_rad = signed_error;
  result.desired_feedforward_radps = feedforward;
  result.achieved_primary_radps = achieved;
  result.horizon_sec = horizon_sec;
  result.pointing_guard_rad = guard;
  result.roll_guard_rad = guard;
  return result;
}

BaseJointHandoffInput edgeInput(
  const double actual_position, const BaseJointDirection edge_direction,
  const BaseJointDirection tracking_direction)
{
  BaseJointHandoffInput result;
  result.working_edge_reached = true;
  result.actual_base_position_rad = actual_position;
  result.predicted_hold_position_rad = actual_position;
  result.edge_direction = edge_direction;
  result.tracking_yaw_direction = tracking_direction;
  return result;
}

PairYawResourceInput pairYawInput(
  double yaw_error, double target_bearing_rate,
  double j1_position, double j4_position);

TEST(FaceTrackingPolicy, StartsInExactTracking)
{
  FaceTrackingPolicy policy;
  EXPECT_EQ(policy.phase(), FaceTrackingPhase::kTracking);
}

TEST(FaceTrackingPolicy, EntersManeuverOnlyForQualifiedNeed)
{
  FaceTrackingPolicy policy;

  EXPECT_EQ(
    policy.update(FaceTrackingPolicyInput{false, false}), FaceTrackingPhase::kTracking);
  EXPECT_EQ(
    policy.update(FaceTrackingPolicyInput{true, false}), FaceTrackingPhase::kCompactManeuver);
}

TEST(FaceTrackingPolicy, CommittedManeuverSurvivesTargetChangesUntilComplete)
{
  FaceTrackingPolicy policy;
  ASSERT_EQ(
    policy.update(FaceTrackingPolicyInput{true, false}), FaceTrackingPhase::kCompactManeuver);

  // Face loss, a stopped/reversed target, or ordinary rearm are not completion.
  EXPECT_EQ(policy.update({}), FaceTrackingPhase::kCompactManeuver);
  EXPECT_EQ(
    policy.update(FaceTrackingPolicyInput{false, false}), FaceTrackingPhase::kCompactManeuver);

  EXPECT_EQ(policy.update(FaceTrackingPolicyInput{false, true}), FaceTrackingPhase::kTracking);
}

TEST(FaceTrackingPolicy, CompletionTakesPrecedenceOverAnotherNeedInSameUpdate)
{
  FaceTrackingPolicy policy;
  ASSERT_EQ(
    policy.update(FaceTrackingPolicyInput{true, false}), FaceTrackingPhase::kCompactManeuver);

  EXPECT_EQ(policy.update(FaceTrackingPolicyInput{true, true}),
    FaceTrackingPhase::kTracking);
}

TEST(FaceTrackingPolicy, ExplicitStopClearsCommittedPhase)
{
  FaceTrackingPolicy policy;
  ASSERT_EQ(
    policy.update(FaceTrackingPolicyInput{true, false}), FaceTrackingPhase::kCompactManeuver);

  policy.clearForStop();
  EXPECT_EQ(policy.phase(), FaceTrackingPhase::kTracking);
}

TEST(FaceTrackingPolicy, StationaryAlignedTargetDoesNotForecastGuardCrossing)
{
  const double guard = 5.0 * kPi / 180.0;

  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), 0.15, guard)));
}

TEST(FaceTrackingPolicy, PredictsMovingTargetShortfallBeforeCurrentErrorCrossesGuard)
{
  constexpr double horizon_sec = kFaceTrackingForecastHorizonSec;
  constexpr double target_rate_rad_s = 0.13;
  const double guard = kFaceTrackingErrorGuardRad;
  const double current_error = 4.0 * kPi / 180.0;
  const double achieved_rate_rad_s = 0.01;
  const double predicted_error = current_error +
    (target_rate_rad_s - achieved_rate_rad_s) * horizon_sec;
  ASSERT_LT(current_error, guard);
  ASSERT_GT(predicted_error, guard);

  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0),
      Eigen::Vector3d(target_rate_rad_s, 0.0, 0.0),
      Eigen::Vector3d(achieved_rate_rad_s, 0.0, 0.0), horizon_sec, guard)));
}

TEST(FaceTrackingPolicy, StationaryCorrectionDoesNotPredictGrowthFromGainResidual)
{
  const double guard = kFaceTrackingErrorGuardRad;
  const double current_error = 0.10;
  const double achieved_correction = 0.40;
  // A gain of 5 would request 0.50 rad/s, so naive requested-achieved residual
  // is +0.10 and wrongly predicts growth. The actual stationary error derivative
  // is 0 - achieved, which reduces 0.10 rad to 0.04 rad over this horizon.
  const double predicted_error = current_error - achieved_correction * 0.15;
  ASSERT_LT(std::abs(predicted_error), guard);

  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(achieved_correction, 0.0, 0.0), 0.15, guard)));
  // Even if a larger stationary error remains outside the guard at the forecast
  // horizon, a clearly shrinking signed error is not a worsening forecast.
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(0.12, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.10, 0.0, 0.0), 0.15, guard)));
}

TEST(FaceTrackingPolicy, ForecastShortfallAloneDoesNotQualifyCompactManeuver)
{
  const double guard = kFaceTrackingErrorGuardRad;
  const double current_error = 4.0 * kPi / 180.0;
  constexpr double target_rate_rad_s = 0.13;
  constexpr double achieved_rate_rad_s = 0.01;
  const bool shortfall_candidate = trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0),
      Eigen::Vector3d(target_rate_rad_s, 0.0, 0.0),
      Eigen::Vector3d(achieved_rate_rad_s, 0.0, 0.0),
      kFaceTrackingForecastHorizonSec, guard));
  ASSERT_TRUE(shortfall_candidate);

  const auto pair = assessPairYawResource(pairYawInput(0.0, 0.0, 0.0, 0.0));
  ASSERT_TRUE(pair.valid);
  EXPECT_FALSE(pair.demand_present);
  EXPECT_FALSE(pair.exhausted);
  FaceTrackingPolicy policy;
  EXPECT_EQ(policy.update(FaceTrackingPolicyInput{shortfall_candidate && pair.exhausted, false}),
    FaceTrackingPhase::kTracking);
}

TEST(FaceTrackingPolicy, PersistentUncorrectableStationaryErrorRemainsAQualifiedCandidate)
{
  const double guard = 5.0 * kPi / 180.0;
  const double stationary_error = 0.10;
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(stationary_error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), 0.15, guard)));
}

TEST(FaceTrackingPolicy, AdequateMovingTargetFeedforwardPreventsFalseTrigger)
{
  const double guard = 5.0 * kPi / 180.0;
  const double current_error = 4.0 * kPi / 180.0;
  constexpr double target_rate = 0.13;
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0),
      Eigen::Vector3d(target_rate, 0.0, 0.0),
      Eigen::Vector3d(target_rate, 0.0, 0.0), 0.15, guard)));
}

TEST(FaceTrackingPolicy, ClosingCorrectionDoesNotTriggerFromLinearOvershoot)
{
  const double guard = kFaceTrackingErrorGuardRad;
  const double current_error = 4.28 * kPi / 180.0;
  constexpr double achieved_rate = 42.6 * kPi / 180.0;
  const double horizon = kFaceTrackingForecastHorizonSec;
  const double predicted_error = current_error - achieved_rate * horizon;
  ASSERT_LT(current_error, guard);
  ASSERT_LT(predicted_error, -guard);

  // The extrapolation crosses zero and ends outside the opposite side of the
  // guard, but the current measured error is closing rapidly.
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(achieved_rate, 0.0, 0.0), horizon, guard)));
}

TEST(FaceTrackingPolicy, TangentialGrowthAndStalledOutsideErrorRemainCandidates)
{
  const double guard = kFaceTrackingErrorGuardRad;
  const double near_guard = 4.9 * kPi / 180.0;
  constexpr double tangential_rate = 20.0 * kPi / 180.0;

  // Zero radial velocity is not evidence that the error is closing; tangential
  // motion increases the pointing norm enough to cross the guard.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(near_guard, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, -tangential_rate, 0.0),
      kFaceTrackingForecastHorizonSec, guard)));

  // No achieved correction leaves an already out-of-guard error persistent.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(0.10, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), kFaceTrackingForecastHorizonSec, guard)));
}

TEST(FaceTrackingPolicy, TransverseGrowthCanOutweighOneClosingComponent)
{
  const double guard = kFaceTrackingErrorGuardRad;
  const double current_error = 4.9 * kPi / 180.0;
  const double closing_rate = 0.05;
  const double transverse_rate = 0.30;
  const double horizon = kFaceTrackingForecastHorizonSec;
  const Eigen::Vector2d error(current_error, 0.0);
  const Eigen::Vector2d rate(-closing_rate, transverse_rate);
  const Eigen::Vector2d predicted = error + horizon * rate;
  ASSERT_LT(error.dot(rate), 0.0);
  ASSERT_GT(predicted.norm(), error.norm());
  ASSERT_GT(predicted.norm(), guard);

  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(current_error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(closing_rate, -transverse_rate, 0.0), horizon, guard)));
}

TEST(FaceTrackingPolicy, ZeroErrorCanForecastUntrackedTargetMotion)
{
  const double guard = kFaceTrackingErrorGuardRad;
  constexpr double target_rate = 20.0 * kPi / 180.0;

  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(target_rate, 0.0, 0.0),
      Eigen::Vector3d::Zero(), kFaceTrackingForecastHorizonSec, guard)));

  // Small nonzero errors must still forecast a genuine outward target motion.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(1.0e-7, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(-target_rate, 0.0, 0.0),
      kFaceTrackingForecastHorizonSec, guard)));
}

TEST(FaceTrackingPolicy, ErrorAndAchievedVelocitySignsAreRespected)
{
  const double guard = 5.0 * kPi / 180.0;
  const double error = 4.0 * kPi / 180.0;
  // For positive error, negative achieved motion makes error grow.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(-0.15, 0.0, 0.0), 0.15, guard)));
  // Mirrored negative error plus mirrored correction is decreasing, not growth.
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(-error, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(-0.10, 0.0, 0.0), 0.15, guard)));
  // Roll uses signed error and then absolute predicted roll magnitude.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(0.0, 0.0, error), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, -0.15), 0.15, guard)));
  // Mirroring both roll error and outward motion preserves the decision.
  EXPECT_TRUE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(0.0, 0.0, -error), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, 0.15), 0.15, guard)));
  // A roll correction that crosses zero is clamped at alignment, like pointing.
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(0.0, 0.0, error), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.0, 0.0, 0.80), 0.15, guard)));
}

TEST(FaceTrackingPolicy, NearZeroNumericalNoiseDoesNotForecastGuardCrossing)
{
  const double guard = 5.0 * kPi / 180.0;
  EXPECT_FALSE(trackingErrorWillCrossGuard(forecast(
      Eigen::Vector3d(1.0e-7, -2.0e-7, 1.0e-7),
      Eigen::Vector3d(2.0e-6, 0.0, -1.0e-6),
      Eigen::Vector3d(-2.0e-6, 1.0e-6, 2.0e-6), 0.15, guard)));
}

TEST(FaceTrackingPolicy, RejectsInvalidForecastInputs)
{
  const double guard = 5.0 * kPi / 180.0;
  auto invalid_forecast = forecast(
    Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.10, 0.0, 0.0),
    Eigen::Vector3d::Zero(), 0.15, guard);
  invalid_forecast.horizon_sec = 0.0;

  EXPECT_FALSE(trackingErrorWillCrossGuard(invalid_forecast));
  invalid_forecast.horizon_sec = 0.15;
  invalid_forecast.signed_error_rad[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(trackingErrorWillCrossGuard(invalid_forecast));
}

PairYawResourceInput pairYawInput(
  const double yaw_error, const double target_bearing_rate,
  const double j1_position, const double j4_position)
{
  PairYawResourceInput input;
  input.signed_yaw_error_rad = yaw_error;
  input.target_bearing_rate_radps = target_bearing_rate;
  input.orientation_gain_per_sec = 2.0;
  input.pointing_deadband_rad = 0.01;
  input.target_rate_deadband_radps = 0.01;
  input.j1 = PairYawJointResourceInput{
    -1.0, j1_position, j1_position, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
  input.j4 = PairYawJointResourceInput{
    -1.0, j4_position, j4_position, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
  return input;
}

TEST(PairYawResource, NeedsYawDemandAndPhysicalDirectionalExhaustion)
{
  auto input = pairYawInput(0.0, 0.0, 1.9, 1.9);
  auto assessment = assessPairYawResource(input);
  ASSERT_TRUE(assessment.valid);
  EXPECT_FALSE(assessment.demand_present);
  EXPECT_FALSE(assessment.exhausted);

  // Near the yaw edge alone is not a reason to compact; an aligned target has
  // no yaw demand to exhaust the pair against.
  input.signed_yaw_error_rad = 0.2;
  input.j1.yaw_error_derivative_per_rad = 0.0;
  input.j4.yaw_error_derivative_per_rad = 0.0;
  input.j1.actual_position_rad = input.j1.predicted_hold_position_rad = 0.0;
  input.j4.actual_position_rad = input.j4.predicted_hold_position_rad = 0.0;
  assessment = assessPairYawResource(input);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_FALSE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, OneBoundDirectionIsOnlyACandidateWhenTheOtherSwivelIsFree)
{
  auto input = pairYawInput(0.0, 0.4, 1.9, 0.0);
  const auto assessment = assessPairYawResource(input);

  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, IdentifiesDirectionalPairCandidatesInEitherYawDirection)
{
  auto positive = pairYawInput(0.0, 0.4, 1.9, 1.9);
  auto assessment = assessPairYawResource(positive);
  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);

  auto negative = pairYawInput(0.0, -0.4, -1.9, -1.9);
  assessment = assessPairYawResource(negative);
  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, ReversingDemandAwayFromAnEdgeDoesNotTrigger)
{
  const auto assessment = assessPairYawResource(pairYawInput(-0.2, 0.0, 1.9, 1.9));

  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_FALSE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, ZeroJacobianWithoutARealJointBoundDoesNotLookExhausted)
{
  auto input = pairYawInput(0.0, 0.4, 0.0, 0.0);
  input.j1.yaw_error_derivative_per_rad = 0.0;
  input.j4.yaw_error_derivative_per_rad = 0.0;
  const auto assessment = assessPairYawResource(input);

  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_FALSE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, BrakingEndpointPastHardBoundConsumesDirectionalReserve)
{
  auto input = pairYawInput(0.0, 0.4, 1.8, 1.8);
  input.j1.predicted_hold_position_rad = 2.02;
  input.j4.predicted_hold_position_rad = 2.03;
  const auto assessment = assessPairYawResource(input);

  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
  EXPECT_DOUBLE_EQ(assessment.j1.remaining_joint_travel_rad, 0.0);
  EXPECT_DOUBLE_EQ(assessment.j4.remaining_joint_travel_rad, 0.0);
}

TEST(PairYawResource, HeldJ1DirectionConsumesItsLatchedYawResource)
{
  auto input = pairYawInput(0.0, 0.4, 0.0, 1.9);
  input.j1.held_direction = BaseJointDirection::kPositive;
  const auto assessment = assessPairYawResource(input);

  ASSERT_TRUE(assessment.valid);
  EXPECT_TRUE(assessment.j1.held_for_required_direction);
  ASSERT_TRUE(assessment.j1.directionally_bound);
  EXPECT_TRUE(assessment.pair_bound_candidate);
  EXPECT_FALSE(assessment.exhausted);
}

TEST(PairYawResource, RequiresSolvedCausalProbeAndUprightRelaxedPose)
{
  PairYawResourceAssessment assessment;
  assessment.valid = true;
  assessment.demand_present = true;
  assessment.pair_bound_candidate = true;
  assessment.probe_valid = true;
  assessment.actual_probe_solved = true;
  assessment.relaxed_probe_solved = true;
  assessment.actual_available_yaw_rad = 0.0;
  assessment.pair_relaxed_available_yaw_rad = kPairYawProbeDisplacementRad;
  assessment.actual_roll_residual_rad = 0.0;
  assessment.relaxed_roll_residual_rad = 0.0;
  assessment.actual_elevation_residual_rad = 0.0;
  assessment.relaxed_elevation_residual_rad = 0.0;
  assessment.probe_horizon_sec = kPairYawProbeHorizonSec;

  EXPECT_TRUE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.actual_probe_solved = false;
  assessment.actual_probe_infeasible = true;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.actual_probe_infeasible = false;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.actual_probe_solved = true;
  EXPECT_TRUE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.relaxed_probe_solved = false;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.relaxed_probe_solved = true;
  assessment.relaxed_roll_residual_rad = 2.0 * kPi / 180.0;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.relaxed_roll_residual_rad = 0.0;
  assessment.relaxed_elevation_residual_rad = 2.0 * kPi / 180.0;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
}

TEST(PairYawResource, RequiresAUsefulCausalImprovement)
{
  PairYawResourceAssessment assessment;
  assessment.valid = true;
  assessment.demand_present = true;
  assessment.pair_bound_candidate = true;
  assessment.probe_valid = true;
  assessment.actual_probe_solved = true;
  assessment.relaxed_probe_solved = true;
  assessment.actual_available_yaw_rad =
    0.79 * kPairYawProbeDisplacementRad;
  assessment.pair_relaxed_available_yaw_rad =
    0.80 * kPairYawProbeDisplacementRad;
  assessment.probe_horizon_sec = kPairYawProbeHorizonSec;

  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
  assessment.actual_available_yaw_rad = 0.0;
  assessment.pair_relaxed_available_yaw_rad =
    0.79 * kPairYawProbeDisplacementRad;
  EXPECT_FALSE(pairYawProbeShowsPairBoundCause(assessment));
}

TEST(BaseJointHandoffPolicy, LatchesTheJerkLimitedEndpointAtTheOutwardEdge)
{
  BaseJointHandoffPolicy policy;
  EXPECT_FALSE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kNone)).has_value());
  EXPECT_FALSE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kNegative)).has_value());

  auto approaching = edgeInput(
    6.05, BaseJointDirection::kPositive, BaseJointDirection::kPositive);
  approaching.predicted_hold_position_rad = 6.1;
  const auto hold = policy.update(approaching);
  ASSERT_TRUE(hold.has_value());
  EXPECT_DOUBLE_EQ(*hold, 6.1);
}

TEST(BaseJointHandoffPolicy, StaticTargetKeepsTheCapturedEdgePositionWithoutCentering)
{
  BaseJointHandoffPolicy policy;
  ASSERT_TRUE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kPositive)).has_value());

  auto static_face = edgeInput(
    5.95, BaseJointDirection::kPositive, BaseJointDirection::kPositive);
  static_face.predicted_hold_position_rad = 6.1;
  static_face.target_bearing_motion_direction = BaseJointDirection::kNone;
  const auto hold = policy.update(static_face);
  ASSERT_TRUE(hold.has_value());
  EXPECT_DOUBLE_EQ(*hold, 6.1);
  EXPECT_NE(*hold, static_face.actual_base_position_rad);
}

TEST(BaseJointHandoffPolicy, ReverseTargetBearingReleasesJ1ForOrdinaryTracking)
{
  BaseJointHandoffPolicy policy;
  ASSERT_TRUE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kPositive)).has_value());

  auto reverse_face = edgeInput(
    6.1, BaseJointDirection::kPositive, BaseJointDirection::kNegative);
  reverse_face.target_bearing_motion_direction = BaseJointDirection::kNegative;
  EXPECT_FALSE(policy.update(reverse_face).has_value());
  EXPECT_FALSE(policy.heldPositionRad().has_value());
}

TEST(BaseJointHandoffPolicy, UnqualifiedBearingNoiseDoesNotReleaseTheHold)
{
  BaseJointHandoffPolicy policy;
  ASSERT_TRUE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kPositive)).has_value());

  auto noisy_static_face = edgeInput(
    6.1, BaseJointDirection::kPositive, BaseJointDirection::kPositive);
  noisy_static_face.target_bearing_motion_direction = BaseJointDirection::kNone;
  const auto hold = policy.update(noisy_static_face);
  ASSERT_TRUE(hold.has_value());
  EXPECT_DOUBLE_EQ(*hold, 6.1);
}

TEST(BaseJointHandoffPolicy, ValidTargetChangeReleasesTheOldHold)
{
  BaseJointHandoffPolicy policy;
  ASSERT_TRUE(policy.update(edgeInput(
      6.1, BaseJointDirection::kPositive, BaseJointDirection::kPositive)).has_value());

  auto changed_target = edgeInput(
    6.1, BaseJointDirection::kPositive, BaseJointDirection::kNegative);
  changed_target.valid_target_changed = true;
  EXPECT_FALSE(policy.update(changed_target).has_value());
}

TEST(BaseJointHandoffPolicy, TargetBearingDeltaIsContinuousAcrossAngleWrap)
{
  constexpr double wrap_epsilon = 1.0e-3;
  const auto delta = targetBearingDeltaRad(kPi - wrap_epsilon, -kPi + wrap_epsilon);

  ASSERT_TRUE(delta.has_value());
  EXPECT_NEAR(*delta, 2.0 * wrap_epsilon, 1.0e-12);
}

}  // namespace
}  // namespace face_tracking_arm::control
