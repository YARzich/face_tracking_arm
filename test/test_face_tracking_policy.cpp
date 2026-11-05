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

J1BoundaryAssessment boundary(
  const double actual, const double brake_min, const double brake_max,
  const double lower = -2.0, const double upper = 2.0, const double margin = 0.1,
  const double position_error = 0.0)
{
  return assessJ1Boundary(J1BoundaryInput{
        actual, brake_min, brake_max, lower, upper, margin, position_error});
}

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

TEST(FaceTrackingPolicy, HorizontalEdgeForecastUsesDimensionlessHeightRate)
{
  auto input = forecast(
    Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d::Zero(),
    Eigen::Vector3d(0.0, 0.0, 100.0), 0.2, 0.08);
  input.horizontal_edge_axis = true;
  input.signed_edge_error = 0.03;
  input.desired_edge_rate = 0.0;
  input.achieved_edge_rate = 0.0;
  input.edge_guard = std::sin(5.1 * kPi / 180.0);
  EXPECT_FALSE(trackingErrorWillCrossGuard(input));

  input.signed_edge_error = 0.08;
  input.achieved_edge_rate = -0.05;
  EXPECT_TRUE(trackingErrorWillCrossGuard(input));
}

TEST(FaceTrackingPolicy, HorizontalEdgeClosingCorrectionDoesNotForecastPastZero)
{
  auto input = forecast(
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    0.30, 0.08);
  input.horizontal_edge_axis = true;
  input.signed_edge_error = -0.04;
  input.desired_edge_rate = 0.0;
  input.achieved_edge_rate = -0.50;
  input.edge_guard = std::sin(5.1 * kPi / 180.0);

  const double raw_prediction = input.signed_edge_error + input.horizon_sec *
    (input.desired_edge_rate - input.achieved_edge_rate);
  ASSERT_GT(raw_prediction, input.edge_guard);
  EXPECT_FALSE(trackingErrorWillCrossGuard(input));
}

TEST(FaceTrackingPolicy, HorizontalEdgeOutwardAndUnclosingForecastsKeepTheirMeaning)
{
  auto input = forecast(
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    0.30, 0.08);
  input.horizontal_edge_axis = true;
  input.edge_guard = std::sin(5.1 * kPi / 180.0);
  input.desired_edge_rate = 0.0;

  input.signed_edge_error = 0.08;
  input.achieved_edge_rate = -0.05;
  EXPECT_TRUE(trackingErrorWillCrossGuard(input));

  input.signed_edge_error = 0.08;
  input.achieved_edge_rate = 0.0;
  EXPECT_FALSE(trackingErrorWillCrossGuard(input));

  input.signed_edge_error = -0.04;
  input.achieved_edge_rate = -0.10;
  EXPECT_FALSE(trackingErrorWillCrossGuard(input));
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

  FaceTrackingPolicy policy;
  EXPECT_EQ(policy.update(FaceTrackingPolicyInput{shortfall_candidate, false}),
    FaceTrackingPhase::kCompactManeuver);
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

TEST(J1BoundaryAssessment, StaticActualAtEitherWorkingEdgeTriggers)
{
  const auto lower = boundary(-1.9, -1.9, -1.9);
  ASSERT_TRUE(lower.valid);
  EXPECT_EQ(lower.edge_direction, BaseJointDirection::kNegative);
  EXPECT_DOUBLE_EQ(lower.working_lower_rad, -1.9);
  EXPECT_DOUBLE_EQ(lower.working_upper_rad, 1.9);

  const auto upper = boundary(1.9, 1.9, 1.9);
  ASSERT_TRUE(upper.valid);
  EXPECT_EQ(upper.edge_direction, BaseJointDirection::kPositive);
}

TEST(J1BoundaryAssessment, BrakeExtremaTriggerEitherEdge)
{
  const auto lower = boundary(0.0, -1.9, 0.2);
  ASSERT_TRUE(lower.valid);
  EXPECT_EQ(lower.edge_direction, BaseJointDirection::kNegative);

  const auto upper = boundary(0.0, -0.2, 1.9);
  ASSERT_TRUE(upper.valid);
  EXPECT_EQ(upper.edge_direction, BaseJointDirection::kPositive);
}

TEST(J1BoundaryAssessment, PositionErrorCountsRoundoffAtEitherEdge)
{
  constexpr double position_error = 1.0e-12;
  const double lower_position = -1.9 + 0.5 * position_error;
  const auto lower = boundary(
    lower_position, lower_position, lower_position, -2.0, 2.0, 0.1, position_error);
  ASSERT_TRUE(lower.valid);
  EXPECT_EQ(lower.edge_direction, BaseJointDirection::kNegative);

  const double upper_position = 1.9 - 0.5 * position_error;
  const auto upper = boundary(
    upper_position, upper_position, upper_position, -2.0, 2.0, 0.1, position_error);
  ASSERT_TRUE(upper.valid);
  EXPECT_EQ(upper.edge_direction, BaseJointDirection::kPositive);
}

TEST(J1BoundaryAssessment, PositionErrorDoesNotTriggerFromInterior)
{
  constexpr double position_error = 1.0e-12;
  const double lower_position = -1.9 + 2.0 * position_error;
  const auto lower = boundary(
    lower_position, lower_position, lower_position, -2.0, 2.0, 0.1, position_error);
  ASSERT_TRUE(lower.valid);
  EXPECT_EQ(lower.edge_direction, BaseJointDirection::kNone);

  const double upper_position = 1.9 - 2.0 * position_error;
  const auto upper = boundary(
    upper_position, upper_position, upper_position, -2.0, 2.0, 0.1, position_error);
  ASSERT_TRUE(upper.valid);
  EXPECT_EQ(upper.edge_direction, BaseJointDirection::kNone);
}

TEST(J1BoundaryAssessment, InteriorDoesNotTriggerAndBothEdgesAreInvalid)
{
  const auto interior = boundary(0.0, -0.2, 0.2);
  ASSERT_TRUE(interior.valid);
  EXPECT_EQ(interior.edge_direction, BaseJointDirection::kNone);

  EXPECT_FALSE(boundary(0.0, -2.0, 2.0).valid);
}

TEST(J1BoundaryAssessment, RejectsInvalidLimitsAndNonfiniteExtrema)
{
  EXPECT_FALSE(boundary(0.0, 0.2, -0.2).valid);
  EXPECT_FALSE(boundary(0.0, -0.2, 0.2, -0.1, 0.1, 0.1).valid);
  EXPECT_FALSE(boundary(0.0, -0.2, 0.2, -2.0, 2.0, 0.1, -1.0e-12).valid);
  EXPECT_FALSE(boundary(
      std::numeric_limits<double>::quiet_NaN(), -0.2, 0.2).valid);
  EXPECT_FALSE(boundary(
      0.0, -0.2, 0.2, -2.0, 2.0, 0.1,
      std::numeric_limits<double>::infinity()).valid);
}

}  // namespace
}  // namespace face_tracking_arm::control
