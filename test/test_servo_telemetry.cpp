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
