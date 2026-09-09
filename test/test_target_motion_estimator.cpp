// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "face_tracking_arm/target_motion_estimator.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kTolerance = 1.0e-9;

void expectZero(const TargetMotionEstimate & estimate)
{
  EXPECT_TRUE(estimate.face_velocity.isZero(0.0));
  EXPECT_TRUE(estimate.goal_velocity.isZero(0.0));
}

TargetMotionEstimate movingSample(TargetMotionEstimator & estimator, const double stamp)
{
  return estimator.update(
    Eigen::Vector3d(0.3, -0.1, 0.2) * stamp,
    Eigen::Vector3d(0.06, 0.03, -0.02) * stamp, stamp, stamp);
}

TEST(TargetMotionEstimator, FirstAndStationaryObservationsAreExactlyZero)
{
  TargetMotionEstimator estimator;
  for (int sample = 0; sample < 100; ++sample) {
    const double stamp = sample * 0.01;
    expectZero(estimator.update(
        Eigen::Vector3d(0.85, 0.1, 1.68), Eigen::Vector3d(0.2, 0.1, 1.1), stamp, stamp));
  }
}

TEST(TargetMotionEstimator, ConstantVelocityConvergesUsingObservationTime)
{
  TargetMotionEstimator estimator;
  TargetMotionEstimate estimate;
  for (int sample = 0; sample <= 100; ++sample) {
    estimate = movingSample(estimator, sample * 0.02);
  }
  EXPECT_TRUE(estimate.face_velocity.isApprox(Eigen::Vector3d(0.3, -0.1, 0.2), kTolerance));
  EXPECT_TRUE(estimate.goal_velocity.isApprox(Eigen::Vector3d(0.06, 0.03, -0.02), kTolerance));
}

TEST(TargetMotionEstimator, DuplicateStampsDoNotDifferentiateOrDecayAgain)
{
  TargetMotionEstimator estimator;
  expectZero(movingSample(estimator, 0.0));
  const auto original = movingSample(estimator, 0.02);
  ASSERT_GT(original.face_velocity.norm(), 0.0);
  for (int tick = 1; tick <= 5; ++tick) {
    const auto repeated = estimator.update(
      Eigen::Vector3d::Ones(), -Eigen::Vector3d::Ones(), 0.02, 0.02 + tick * 0.01);
    EXPECT_TRUE(repeated.face_velocity.isApprox(original.face_velocity, 0.0));
    EXPECT_TRUE(repeated.goal_velocity.isApprox(original.goal_velocity, 0.0));
  }
}

TEST(TargetMotionEstimator, StaleInputReturnsZeroAndStartsAFreshEstimate)
{
  TargetMotionEstimator estimator;
  expectZero(movingSample(estimator, 0.0));
  ASSERT_GT(movingSample(estimator, 0.02).face_velocity.norm(), 0.0);
  expectZero(estimator.update(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.02, 0.121));
  expectZero(movingSample(estimator, 0.14));
  EXPECT_GT(movingSample(estimator, 0.16).face_velocity.norm(), 0.0);
}

TEST(TargetMotionEstimator, StationaryNewMeasurementsDecayToExactZero)
{
  TargetMotionEstimator estimator;
  for (int sample = 0; sample <= 50; ++sample) {
    (void)movingSample(estimator, sample * 0.02);
  }
  TargetMotionEstimate estimate;
  for (int sample = 1; sample <= 50; ++sample) {
    const double stamp = 1.0 + sample * 0.02;
    estimate = estimator.update(
      Eigen::Vector3d(0.3, -0.1, 0.2), Eigen::Vector3d(0.06, 0.03, -0.02), stamp, stamp);
  }
  expectZero(estimate);
}

TEST(TargetMotionEstimator, FaceJumpResetsBothVelocityEstimates)
{
  TargetMotionEstimator estimator;
  expectZero(movingSample(estimator, 0.0));
  ASSERT_GT(movingSample(estimator, 0.02).face_velocity.norm(), 0.0);
  const Eigen::Vector3d new_face(1.0, 0.0, 0.0);
  const Eigen::Vector3d new_goal(0.4, 0.0, 0.0);
  expectZero(estimator.update(new_face, new_goal, 0.04, 0.04));
  expectZero(estimator.update(new_face, new_goal, 0.06, 0.06));
}

TEST(TargetMotionEstimator, InvalidInputsAndRewindsResetTheFilter)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int invalid_case = 0; invalid_case < 7; ++invalid_case) {
    TargetMotionEstimator estimator;
    expectZero(movingSample(estimator, 1.0));
    ASSERT_GT(movingSample(estimator, 1.02).face_velocity.norm(), 0.0);
    Eigen::Vector3d face = Eigen::Vector3d::Zero();
    Eigen::Vector3d goal = Eigen::Vector3d::Zero();
    double measurement = 1.04;
    double control = 1.04;
    switch (invalid_case) {
      case 0: face.x() = nan; break;
      case 1: goal.z() = nan; break;
      case 2: measurement = nan; break;
      case 3: control = nan; break;
      case 4: measurement = 1.01; break;
      case 5: measurement = 1.01; control = 1.01; break;
      case 6: measurement = -1.0; break;
    }
    expectZero(estimator.update(face, goal, measurement, control));
    expectZero(estimator.update(face.setZero(), goal.setZero(), 1.06, 1.06));
  }
}

TEST(TargetMotionEstimator, InvalidSampleIntervalsStartANewAnchor)
{
  for (const double interval : {0.005, 0.21}) {
    TargetMotionEstimator estimator;
    expectZero(movingSample(estimator, 0.0));
    ASSERT_GT(movingSample(estimator, 0.02).face_velocity.norm(), 0.0);
    expectZero(movingSample(estimator, 0.02 + interval));
    EXPECT_GT(movingSample(estimator, 0.04 + interval).face_velocity.norm(), 0.0);
  }
}

TEST(TargetMotionEstimator, SmallFutureClockSkewRetainsTheFilter)
{
  TargetMotionEstimator estimator;
  TargetMotionEstimate estimate;
  for (int sample = 0; sample <= 100; ++sample) {
    const double control = sample * 0.02;
    const double measurement = control + 0.005;
    estimate = estimator.update(
      Eigen::Vector3d(0.3, 0.0, 0.0) * measurement,
      Eigen::Vector3d(0.05, 0.0, 0.0) * measurement, measurement, control);
  }
  EXPECT_NEAR(estimate.face_velocity.x(), 0.3, kTolerance);
  EXPECT_NEAR(estimate.goal_velocity.x(), 0.05, kTolerance);
  expectZero(estimator.update(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 2.041, 2.02));
}

TEST(TargetMotionEstimator, SpeedLimitsPreserveDirectionsAndBoundBothOutputs)
{
  TargetMotionEstimator estimator;
  const Eigen::Vector3d face_velocity(1.0, 2.0, 2.0);
  const Eigen::Vector3d goal_velocity(0.3, 0.4, 0.0);
  TargetMotionEstimate estimate;
  for (int sample = 0; sample <= 100; ++sample) {
    const double stamp = sample * 0.02;
    estimate = estimator.update(face_velocity * stamp, goal_velocity * stamp, stamp, stamp);
    EXPECT_LE(estimate.face_velocity.norm(), 1.5 + kTolerance);
    EXPECT_LE(estimate.goal_velocity.norm(), 0.18 + kTolerance);
  }
  EXPECT_TRUE(estimate.face_velocity.isApprox(face_velocity.normalized() * 1.5, kTolerance));
  EXPECT_TRUE(estimate.goal_velocity.isApprox(goal_velocity.normalized() * 0.18, kTolerance));
}

TEST(TargetMotionEstimator, ExplicitResetRequiresANewObservationPair)
{
  TargetMotionEstimator estimator;
  expectZero(movingSample(estimator, 0.0));
  ASSERT_GT(movingSample(estimator, 0.02).face_velocity.norm(), 0.0);
  estimator.reset();
  expectZero(movingSample(estimator, 0.04));
  EXPECT_GT(movingSample(estimator, 0.06).face_velocity.norm(), 0.0);
}

TEST(TargetMotionEstimator, InvalidConfigurationIsRejected)
{
  TargetMotionEstimatorConfig config;
  config.filter_time_constant_sec = 0.0;
  EXPECT_THROW(TargetMotionEstimator{config}, std::invalid_argument);
  config = TargetMotionEstimatorConfig{};
  config.minimum_sample_interval_sec = 0.3;
  EXPECT_THROW(TargetMotionEstimator{config}, std::invalid_argument);
  config = TargetMotionEstimatorConfig{};
  config.maximum_future_skew_sec = -0.01;
  EXPECT_THROW(TargetMotionEstimator{config}, std::invalid_argument);
}

}  // namespace
}  // namespace face_tracking_arm::control
