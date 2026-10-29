// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>

#include "servo_telemetry.hpp"

namespace face_tracking_arm
{
namespace
{

TEST(ServoTelemetry, TickResetKeepsLifetimeEvidence)
{
  ServoTelemetry telemetry;
  telemetry.observeClearances(0.05, 0.08);
  telemetry.observeClearances(0.07, std::numeric_limits<double>::infinity());
  telemetry.last_minimum_self_distance_m = 0.07;
  telemetry.last_position_error_m = 0.1;
  telemetry.last_active_collision_rows = 5;
  telemetry.segment_rejection_count = 2;
  ASSERT_TRUE(telemetry.control_tick_statistics.add(4.0));

  telemetry.resetTick();

  EXPECT_FALSE(telemetry.last_minimum_self_distance_m.has_value());
  EXPECT_FALSE(telemetry.last_position_error_m.has_value());
  EXPECT_EQ(telemetry.last_active_collision_rows, 0U);
  EXPECT_EQ(telemetry.segment_rejection_count, 2U);
  EXPECT_DOUBLE_EQ(*telemetry.minimum_observed_self_distance_m, 0.05);
  EXPECT_DOUBLE_EQ(*telemetry.minimum_observed_world_distance_m, 0.08);
  EXPECT_DOUBLE_EQ(*telemetry.control_tick_statistics.mean(), 4.0);
}

TEST(ServoTelemetry, CountsModeTransitionsSeparatelyFromTicks)
{
  ServoTelemetry telemetry;
  telemetry.observeMode(ControllerMode::kBraking);
  telemetry.observeMode(ControllerMode::kBraking);
  telemetry.observeMode(ControllerMode::kTracking);
  telemetry.observeMode(ControllerMode::kWaitingForSafeState);
  telemetry.observeMode(ControllerMode::kWaitingForSafeState);
  telemetry.observeMode(ControllerMode::kBraking);

  EXPECT_EQ(telemetry.braking_tick_count, 3U);
  EXPECT_EQ(telemetry.braking_event_count, 2U);
  EXPECT_EQ(telemetry.emergency_tick_count, 2U);
  EXPECT_EQ(telemetry.emergency_event_count, 1U);
}

TEST(ServoTelemetry, CountsAvailablePartialRankCommandWithoutCallingItSolverFailure)
{
  ServoTelemetry telemetry;
  control::QpResult result;
  result.status = control::QpStatus::kPartial;
  result.joint_velocity = Eigen::VectorXd::Zero(1);

  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);

  EXPECT_TRUE(result.command_available());
  EXPECT_EQ(telemetry.solver_failure_count, 0U);
  EXPECT_EQ(telemetry.primary_only_count, 0U);
  EXPECT_EQ(telemetry.lower_rank_partial_count, 1U);
}

TEST(ServoTelemetry, CountsRecoveredQpRetriesWithoutCountingSolverFailure)
{
  ServoTelemetry telemetry;
  control::QpResult recovered;
  recovered.status = control::QpStatus::kSolved;
  recovered.primary_status = control::QpStageStatus::kSolved;
  recovered.primary_retry_count = 1;
  recovered.joint_velocity = Eigen::VectorXd::Zero(1);

  telemetry.recordQpResult(std::chrono::steady_clock::now(), recovered);
  EXPECT_EQ(telemetry.qp_retry_count, 1U);
  EXPECT_EQ(telemetry.solver_failure_count, 0U);

  control::QpResult next_default_result;
  next_default_result.status = control::QpStatus::kSolved;
  next_default_result.primary_status = control::QpStageStatus::kSolved;
  next_default_result.joint_velocity = Eigen::VectorXd::Zero(1);
  telemetry.recordQpResult(std::chrono::steady_clock::now(), next_default_result);

  EXPECT_EQ(telemetry.qp_retry_count, 1U);
  EXPECT_EQ(telemetry.solver_failure_count, 0U);
}

TEST(ServoTelemetry, RecordsLastRankOutcomesAndDistanceRateOnlyForScalarTask)
{
  ServoTelemetry telemetry;
  control::QpResult result;
  result.status = control::QpStatus::kSolved;
  result.primary_status = control::QpStageStatus::kSolved;
  result.secondary_status = control::QpStageStatus::kNotRun;
  result.tertiary_status = control::QpStageStatus::kSolvedInaccurate;
  result.distance_status = control::QpStageStatus::kSolved;
  result.primary_iterations = 12;
  result.tertiary_iterations = 7;
  result.distance_iterations = 4;
  result.primary_solve_time_sec = 0.0004;
  result.tertiary_solve_time_sec = 0.0003;
  result.distance_solve_time_sec = 0.0002;
  result.requested_distance_rate = -0.18;
  result.achieved_distance_rate = -0.06;

  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);

  EXPECT_EQ(telemetry.last_qp_rank_count, 3U);
  EXPECT_EQ(telemetry.last_qp_primary_status, control::QpStageStatus::kSolved);
  EXPECT_EQ(telemetry.last_qp_secondary_status, control::QpStageStatus::kNotRun);
  EXPECT_EQ(telemetry.last_qp_tertiary_status, control::QpStageStatus::kSolvedInaccurate);
  EXPECT_EQ(telemetry.last_qp_distance_status, control::QpStageStatus::kSolved);
  EXPECT_EQ(telemetry.last_qp_primary_iterations, 12);
  EXPECT_EQ(telemetry.last_qp_tertiary_iterations, 7);
  EXPECT_EQ(telemetry.last_qp_distance_iterations, 4);
  EXPECT_DOUBLE_EQ(telemetry.last_qp_primary_solve_time_sec, 0.0004);
  EXPECT_DOUBLE_EQ(telemetry.last_qp_tertiary_solve_time_sec, 0.0003);
  EXPECT_DOUBLE_EQ(telemetry.last_qp_distance_solve_time_sec, 0.0002);
  ASSERT_TRUE(telemetry.last_requested_distance_rate.has_value());
  ASSERT_TRUE(telemetry.last_achieved_distance_rate.has_value());
  EXPECT_DOUBLE_EQ(*telemetry.last_requested_distance_rate, -0.18);
  EXPECT_DOUBLE_EQ(*telemetry.last_achieved_distance_rate, -0.06);

  // A future distance objective may include tangential rows. Keep the scalar
  // rate diagnostic unavailable for that shape instead of reporting row 0 as
  // the achieved radial rate.
  result.requested_distance_rate.reset();
  result.achieved_distance_rate.reset();
  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);
  EXPECT_FALSE(telemetry.last_requested_distance_rate.has_value());
  EXPECT_FALSE(telemetry.last_achieved_distance_rate.has_value());
}

TEST(ServoTelemetry, CountsGuardedPrimaryFallbackAndBothPrimaryAttempts)
{
  ServoTelemetry telemetry;
  control::QpResult result;
  result.status = control::QpStatus::kPrimaryOnly;
  result.preferred_primary_status = control::QpStageStatus::kMaxIterations;
  result.preferred_primary_fallback_used = true;
  result.primary_status = control::QpStageStatus::kSolved;
  result.primary_iterations = 237;
  result.primary_solve_time_sec = 0.0008;

  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);
  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);

  EXPECT_EQ(telemetry.preferred_primary_fallback_count, 2U);
  EXPECT_TRUE(telemetry.last_qp_preferred_primary_fallback_used);
  EXPECT_EQ(telemetry.last_qp_preferred_primary_status, control::QpStageStatus::kMaxIterations);
  EXPECT_EQ(telemetry.last_qp_primary_iterations, 237);
  EXPECT_DOUBLE_EQ(telemetry.last_qp_primary_solve_time_sec, 0.0008);
  EXPECT_EQ(telemetry.last_qp_rank_count, 2U);
}

TEST(ServoTelemetry, CountsDistanceFeasibleApproximationWithoutHidingRawFailure)
{
  ServoTelemetry telemetry;
  control::QpResult result;
  result.status = control::QpStatus::kPartial;
  result.distance_status = control::QpStageStatus::kMaxIterations;
  result.distance_feasible_approximation_used = true;
  result.distance_feasible_approximation_alpha = 0.131740381606;
  result.joint_velocity = Eigen::VectorXd::Zero(1);

  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);
  telemetry.recordQpResult(std::chrono::steady_clock::now(), result);

  EXPECT_EQ(telemetry.last_qp_distance_status, control::QpStageStatus::kMaxIterations);
  EXPECT_TRUE(telemetry.last_qp_distance_feasible_approximation_used);
  EXPECT_DOUBLE_EQ(telemetry.last_qp_distance_feasible_approximation_alpha, 0.131740381606);
  EXPECT_EQ(telemetry.distance_feasible_approximation_count, 2U);
  EXPECT_EQ(telemetry.lower_rank_partial_count, 2U);
  EXPECT_EQ(telemetry.solver_failure_count, 0U);
}

TEST(ServoTelemetry, RejectionDoesNotKeepDistancesFromPreviousAttempt)
{
  ServoTelemetry telemetry;
  control::CollisionSegmentValidationResult result;
  result.input_valid = true;
  result.unsafe = true;
  result.minimum_self_distance_m = 0.01;
  result.minimum_world_distance_m = 0.02;
  result.minimum_distance_m = 0.01;
  result.first_unsafe_fraction = 0.5;
  result.closest_pair = "screen <-> arm";
  telemetry.recordSegmentRejection(CommandAttempt::kTracking, result);
  EXPECT_DOUBLE_EQ(*telemetry.last_segment_rejection_minimum_self_distance_m, 0.01);

  result = control::CollisionSegmentValidationResult{};
  result.failure_reason = "no collision environment";
  telemetry.recordSegmentRejection(CommandAttempt::kBraking, result);

  EXPECT_EQ(telemetry.segment_rejection_count, 2U);
  EXPECT_EQ(telemetry.tracking_segment_rejection_count, 1U);
  EXPECT_EQ(telemetry.braking_segment_rejection_count, 1U);
  EXPECT_EQ(telemetry.last_segment_rejection_reason, "no collision environment");
  EXPECT_EQ(telemetry.last_segment_rejection_closest_pair, "unavailable");
  EXPECT_FALSE(telemetry.last_segment_rejection_minimum_self_distance_m.has_value());
  EXPECT_FALSE(telemetry.last_segment_rejection_minimum_world_distance_m.has_value());
  EXPECT_FALSE(telemetry.last_segment_rejection_minimum_distance_m.has_value());
  EXPECT_FALSE(telemetry.last_segment_rejection_first_unsafe_fraction.has_value());
}

TEST(ServoTelemetry, CommandJerkUsesControlPeriodAndRejectsNonFiniteEvidence)
{
  ServoTelemetry telemetry;
  control::JointMotionState current;
  current.velocity = Eigen::Vector2d(0.1, -0.2);
  current.acceleration = Eigen::Vector2d(0.5, -0.3);
  telemetry.observeMotionState(current);
  moveit_servo::KinematicState next(2);
  next.velocities = Eigen::Vector2d(0.3, -0.4);
  next.accelerations = Eigen::Vector2d(0.6, -0.5);
  telemetry.observeCommandTransition(current, next, 0.1);

  EXPECT_DOUBLE_EQ(telemetry.max_observed_joint_velocity_rad_s, 0.4);
  EXPECT_DOUBLE_EQ(telemetry.max_observed_joint_acceleration_rad_s2, 0.6);
  EXPECT_DOUBLE_EQ(telemetry.max_observed_joint_jerk_rad_s3, 2.0);

  next.accelerations[0] = std::numeric_limits<double>::quiet_NaN();
  telemetry.observeCommandTransition(current, next, 0.1);
  EXPECT_DOUBLE_EQ(telemetry.max_observed_joint_acceleration_rad_s2, 0.6);
  EXPECT_DOUBLE_EQ(telemetry.max_observed_joint_jerk_rad_s3, 2.0);
}

}  // namespace
}  // namespace face_tracking_arm
