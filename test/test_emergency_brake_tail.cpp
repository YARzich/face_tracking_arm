// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "face_tracking_arm/emergency_brake_tail.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kTolerance = 1.0e-9;

JointMotionState makeState(const Eigen::Index joint_count)
{
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(joint_count);
  state.velocity = Eigen::VectorXd::Zero(joint_count);
  state.acceleration = Eigen::VectorXd::Zero(joint_count);
  return state;
}

JointMotionLimits makeLimits(const Eigen::Index joint_count)
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(joint_count, -2.0);
  limits.upper_position = Eigen::VectorXd::Constant(joint_count, 2.0);
  limits.position_margin = Eigen::VectorXd::Constant(joint_count, 0.1);
  limits.max_velocity = Eigen::VectorXd::Constant(joint_count, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(joint_count, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(joint_count, 20.0);
  return limits;
}

void expectValidTail(
  const EmergencyBrakeTail & tail, const JointMotionLimits & limits,
  const double period_sec)
{
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  ASSERT_GE(tail.points.size(), 3U);
  ASSERT_LT(tail.first_stationary_point, tail.points.size());
  const Eigen::Index joint_count = tail.points.front().position.size();
  for (std::size_t point = 1; point < tail.points.size(); ++point) {
    const auto & previous = tail.points[point - 1U];
    const auto & current = tail.points[point];
    ASSERT_EQ(current.position.size(), joint_count);
    EXPECT_TRUE(current.position.allFinite());
    EXPECT_TRUE(current.velocity.allFinite());
    EXPECT_TRUE(current.acceleration.allFinite());
    for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
      EXPECT_NEAR(
        current.position[joint],
        previous.position[joint] + current.velocity[joint] * period_sec,
        kTolerance) << "point " << point << ", joint " << joint;
      EXPECT_NEAR(
        current.acceleration[joint],
        (current.velocity[joint] - previous.velocity[joint]) / period_sec,
        kTolerance) << "point " << point << ", joint " << joint;
      const double jerk =
        (current.acceleration[joint] - previous.acceleration[joint]) / period_sec;
      EXPECT_LE(std::abs(current.velocity[joint]), limits.max_velocity[joint] + kTolerance);
      EXPECT_LE(
        std::abs(current.acceleration[joint]),
        limits.max_acceleration[joint] + kTolerance);
      EXPECT_LE(std::abs(jerk), limits.max_jerk[joint] + kTolerance);
      EXPECT_GE(
        current.position[joint],
        limits.lower_position[joint] + limits.position_margin[joint] - kTolerance);
      EXPECT_LE(
        current.position[joint],
        limits.upper_position[joint] - limits.position_margin[joint] + kTolerance);
    }
  }
  for (std::size_t point = tail.first_stationary_point;
    point < tail.points.size(); ++point)
  {
    EXPECT_TRUE(tail.points[point].velocity.isZero(0.0));
    EXPECT_TRUE(tail.points[point].acceleration.isZero(0.0));
    EXPECT_TRUE(tail.points[point].position.isApprox(
        tail.points[tail.first_stationary_point].position, 0.0));
  }
}

TEST(EmergencyBrakeTail, StationaryBranchPublishesAVisibleHoldSuffix)
{
  const EmergencyBrakeTailGenerator generator;
  const JointMotionState state = makeState(2);
  const EmergencyBrakeTail tail = generator.generate(state, makeLimits(2));

  expectValidTail(tail, makeLimits(2), generator.config().period_sec);
  EXPECT_EQ(tail.first_stationary_point, 0U);
  EXPECT_EQ(
    tail.points.size(), generator.config().terminal_hold_points + 1U);
}

TEST(EmergencyBrakeTail, UsesExactBackwardEulerJerkLimitedBraking)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState state = makeState(1);
  state.velocity[0] = 0.4;
  state.acceleration[0] = 0.1;
  const JointMotionLimits limits = makeLimits(1);

  const EmergencyBrakeTail tail = generator.generate(state, limits);

  expectValidTail(tail, limits, generator.config().period_sec);
  ASSERT_GT(tail.first_stationary_point, 2U);
  EXPECT_NEAR(tail.points[1].acceleration[0], -0.1, kTolerance);
  EXPECT_NEAR(tail.points[1].velocity[0], 0.399, kTolerance);
  for (std::size_t point = 1; point <= tail.first_stationary_point; ++point) {
    EXPECT_GE(tail.points[point].velocity[0], -kTolerance);
    EXPECT_GE(
      tail.points[point].position[0], tail.points[point - 1U].position[0] - kTolerance);
  }
}

TEST(EmergencyBrakeTail, MirrorsPositiveAndNegativeMotion)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState positive = makeState(1);
  positive.position[0] = 0.2;
  positive.velocity[0] = 0.4;
  positive.acceleration[0] = 0.1;
  JointMotionState negative = positive;
  negative.position = -positive.position;
  negative.velocity = -positive.velocity;
  negative.acceleration = -positive.acceleration;
  const JointMotionLimits limits = makeLimits(1);

  const EmergencyBrakeTail positive_tail = generator.generate(positive, limits);
  const EmergencyBrakeTail negative_tail = generator.generate(negative, limits);

  expectValidTail(positive_tail, limits, generator.config().period_sec);
  expectValidTail(negative_tail, limits, generator.config().period_sec);
  ASSERT_EQ(positive_tail.points.size(), negative_tail.points.size());
  for (std::size_t point = 0; point < positive_tail.points.size(); ++point) {
    EXPECT_TRUE(positive_tail.points[point].position.isApprox(
        -negative_tail.points[point].position, kTolerance));
    EXPECT_TRUE(positive_tail.points[point].velocity.isApprox(
        -negative_tail.points[point].velocity, kTolerance));
    EXPECT_TRUE(positive_tail.points[point].acceleration.isApprox(
        -negative_tail.points[point].acceleration, kTolerance));
  }
}

TEST(EmergencyBrakeTail, HoldsStoppedJointsWhileOtherJointsFinish)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState state = makeState(3);
  state.velocity << 0.0, 0.4, -0.2;
  state.acceleration << 0.0, 0.1, 0.0;
  const JointMotionLimits limits = makeLimits(3);

  const EmergencyBrakeTail tail = generator.generate(state, limits);

  expectValidTail(tail, limits, generator.config().period_sec);
  for (const auto & point : tail.points) {
    EXPECT_DOUBLE_EQ(point.position[0], 0.0);
    EXPECT_DOUBLE_EQ(point.velocity[0], 0.0);
    EXPECT_DOUBLE_EQ(point.acceleration[0], 0.0);
  }
}

TEST(EmergencyBrakeTail, ResolvesTheMinimumUnavoidableVelocityReversalSmoothly)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState state = makeState(1);
  state.velocity[0] = 0.001;
  state.acceleration[0] = -2.0;
  const JointMotionLimits limits = makeLimits(1);

  const EmergencyBrakeTail tail = generator.generate(state, limits);

  expectValidTail(tail, limits, generator.config().period_sec);
  double minimum_velocity = 0.0;
  for (const auto & point : tail.points) {
    minimum_velocity = std::min(minimum_velocity, point.velocity[0]);
  }
  EXPECT_LT(minimum_velocity, 0.0);
  EXPECT_GT(minimum_velocity, -0.11);
}

TEST(EmergencyBrakeTail, RejectsAStateOutsideTheVelocityViabilityEnvelope)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState state = makeState(1);
  state.velocity[0] = 1.0;
  state.acceleration[0] = 2.0;

  const EmergencyBrakeTail tail = generator.generate(state, makeLimits(1));

  EXPECT_EQ(tail.status, EmergencyBrakeTailStatus::kInfeasible);
  EXPECT_FALSE(tail.command_available());
  EXPECT_TRUE(tail.points.empty());
}

TEST(EmergencyBrakeTail, StopsARepresentativeGridOfLite6MotionStates)
{
  const EmergencyBrakeTailGenerator generator;
  const JointMotionLimits limits = makeLimits(1);
  for (int velocity_step = -8; velocity_step <= 8; ++velocity_step) {
    for (int acceleration_step = -7; acceleration_step <= 7; ++acceleration_step) {
      JointMotionState state = makeState(1);
      state.velocity[0] = 0.1 * static_cast<double>(velocity_step);
      state.acceleration[0] = 0.2 * static_cast<double>(acceleration_step);

      const EmergencyBrakeTail tail = generator.generate(state, limits);

      ASSERT_TRUE(tail.command_available()) <<
        "velocity=" << state.velocity[0] <<
        ", acceleration=" << state.acceleration[0] <<
        ", reason=" << tail.failure_reason;
      expectValidTail(tail, limits, generator.config().period_sec);
    }
  }
}

TEST(EmergencyBrakeTail, RejectsAStopThatWouldCrossTheSafePositionBound)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState state = makeState(1);
  state.position[0] = 1.89;
  state.velocity[0] = 0.4;
  state.acceleration[0] = 0.0;

  const EmergencyBrakeTail tail = generator.generate(state, makeLimits(1));

  EXPECT_EQ(tail.status, EmergencyBrakeTailStatus::kInfeasible);
  EXPECT_FALSE(tail.command_available());
  EXPECT_TRUE(tail.points.empty());
}

TEST(EmergencyBrakeTail, RejectsNonFiniteAndDimensionallyInvalidInput)
{
  const EmergencyBrakeTailGenerator generator;
  JointMotionState non_finite = makeState(1);
  non_finite.velocity[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    generator.generate(non_finite, makeLimits(1)).status,
    EmergencyBrakeTailStatus::kInvalidInput);

  JointMotionState wrong_size = makeState(2);
  wrong_size.acceleration.resize(1);
  EXPECT_EQ(
    generator.generate(wrong_size, makeLimits(2)).status,
    EmergencyBrakeTailStatus::kInvalidInput);
}

TEST(EmergencyBrakeTail, EnforcesABoundedGenerationHorizon)
{
  EmergencyBrakeTailConfig config;
  config.maximum_braking_steps = 1U;
  const EmergencyBrakeTailGenerator generator{config};
  JointMotionState state = makeState(1);
  state.velocity[0] = 0.4;

  const EmergencyBrakeTail tail = generator.generate(state, makeLimits(1));

  EXPECT_EQ(tail.status, EmergencyBrakeTailStatus::kMaximumStepsExceeded);
  EXPECT_FALSE(tail.command_available());
  EXPECT_TRUE(tail.points.empty());
}

TEST(EmergencyBrakeTail, RejectsConfigurationsThatCouldHideTheTerminalPoint)
{
  EmergencyBrakeTailConfig config;
  config.terminal_hold_points = 1U;
  EXPECT_THROW(EmergencyBrakeTailGenerator{config}, std::invalid_argument);

  config = EmergencyBrakeTailConfig{};
  config.period_sec = 0.0;
  EXPECT_THROW(EmergencyBrakeTailGenerator{config}, std::invalid_argument);
}

}  // namespace
}  // namespace face_tracking_arm::control
