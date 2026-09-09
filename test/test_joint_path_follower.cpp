// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "face_tracking_arm/joint_path_follower.hpp"

namespace
{

using face_tracking_arm::control::JointPathFollower;
using face_tracking_arm::control::JointPathFollowerConfig;
using face_tracking_arm::control::JointPathStatus;

Eigen::VectorXd point(const double first, const double second = 0.0)
{
  Eigen::VectorXd result(2);
  result << first, second;
  return result;
}

TEST(JointPathFollower, AttachesLateWithoutReturningToTheOldStart)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.3), 1.0));
  const auto result = follower.update(point(0.3), 1.0);
  EXPECT_EQ(result.status, JointPathStatus::kTracking);
  EXPECT_TRUE(result.reference_position.isApprox(point(0.4), 1.0e-12));
  EXPECT_NEAR(result.total_length_rad, 0.7, 1.0e-12);
  EXPECT_DOUBLE_EQ(result.progress_rad, 0.0);
}

TEST(JointPathFollower, ConnectsAnAcceptedOffsetWithoutImmediateCorridorFailure)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.3, 0.15), 0.0));
  const auto result = follower.update(point(0.3, 0.15), 0.01);
  EXPECT_EQ(result.status, JointPathStatus::kTracking);
  EXPECT_TRUE(result.reference_position.isApprox(point(0.3, 0.05), 1.0e-12));
  EXPECT_DOUBLE_EQ(result.distance_from_path_rad, 0.0);
}

TEST(JointPathFollower, ProgressNeverMovesBackwards)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  const auto forward = follower.update(point(0.4), 0.3);
  const auto backward = follower.update(point(0.35), 0.4);
  EXPECT_EQ(backward.status, JointPathStatus::kTracking);
  EXPECT_DOUBLE_EQ(backward.progress_rad, forward.progress_rad);
  EXPECT_TRUE(backward.reference_position.isApprox(point(0.5), 1.0e-12));
}

TEST(JointPathFollower, LookaheadUsesPolylineArcAcrossACorner)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath(
      {point(0.0), point(0.2), point(0.2, 0.3)}, point(0.0), 0.0));
  const auto result = follower.update(point(0.15), 0.1);
  EXPECT_TRUE(result.reference_position.isApprox(point(0.2, 0.05), 1.0e-12));
  EXPECT_NEAR(result.progress_rad, 0.15, 1.0e-12);
  EXPECT_NEAR(result.total_length_rad, 0.5, 1.0e-12);
}

TEST(JointPathFollower, DoesNotJumpToADistantSelfIntersectionOrClosedPathEnd)
{
  JointPathFollowerConfig config;
  config.projection_search_segments = 2;
  JointPathFollower follower(config);
  const std::vector<Eigen::VectorXd> path{
    point(0.0), point(1.0), point(1.0, 1.0), point(-1.0, 1.0),
    point(-1.0), point(0.0)};
  ASSERT_TRUE(follower.setPath(path, point(0.0), 0.0));
  const auto first = follower.update(point(0.0), 0.01);
  EXPECT_EQ(first.status, JointPathStatus::kTracking);
  EXPECT_DOUBLE_EQ(first.progress_rad, 0.0);
  EXPECT_NEAR(first.total_length_rad, 6.0, 1.0e-12);
  const auto nearby = follower.update(point(0.2), 0.1);
  EXPECT_NEAR(nearby.progress_rad, 0.2, 1.0e-12);
}

TEST(JointPathFollower, RejectsAttachmentOnlyNearADistantPathSection)
{
  JointPathFollowerConfig config;
  config.projection_search_segments = 1;
  JointPathFollower follower(config);
  EXPECT_FALSE(follower.setPath(
      {point(0.0), point(1.0), point(1.0, 1.0), point(0.0, 1.0)},
      point(0.0, 1.0), 0.0));
}

TEST(JointPathFollower, DetectsStallWithoutBeingKeptAliveByTinyNoise)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  EXPECT_EQ(follower.update(point(0.001), 0.5).status, JointPathStatus::kTracking);
  EXPECT_EQ(follower.update(point(0.002), 1.0).status, JointPathStatus::kTracking);
  EXPECT_EQ(follower.update(point(0.003), 1.5).status, JointPathStatus::kStalled);
  EXPECT_FALSE(follower.active());
}

TEST(JointPathFollower, MeaningfulProgressRefreshesStallDeadline)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  EXPECT_EQ(follower.update(point(0.1), 1.0).status, JointPathStatus::kTracking);
  EXPECT_EQ(follower.update(point(0.1), 2.0).status, JointPathStatus::kTracking);
  EXPECT_EQ(follower.update(point(0.1), 2.5).status, JointPathStatus::kStalled);
}

TEST(JointPathFollower, LeavesCorridorAsTerminalInvalidState)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  const auto result = follower.update(point(0.2, 0.13), 0.1);
  EXPECT_EQ(result.status, JointPathStatus::kInvalid);
  EXPECT_NEAR(result.distance_from_path_rad, 0.13, 1.0e-12);
  EXPECT_EQ(follower.update(point(0.3), 0.2).status, JointPathStatus::kInvalid);
}

TEST(JointPathFollower, ArrivesAndCanBeResetOrAssignedAnotherPath)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  const auto result = follower.update(point(0.99), 0.5);
  EXPECT_EQ(result.status, JointPathStatus::kArrived);
  EXPECT_TRUE(result.reference_position.isApprox(point(1.0), 1.0e-12));
  EXPECT_FALSE(follower.active());
  follower.reset();
  EXPECT_EQ(follower.update(point(0.0), 0.0).status, JointPathStatus::kInvalid);
  ASSERT_TRUE(follower.setPath({point(0.0), point(0.0), point(0.0)}, point(0.0), 0.0));
  EXPECT_EQ(follower.update(point(0.0), 0.0).status, JointPathStatus::kArrived);
}

TEST(JointPathFollower, RejectsInvalidInputAndPreservesThePreviousAcceptedPath)
{
  JointPathFollower follower;
  ASSERT_TRUE(follower.setPath({point(0.0), point(1.0)}, point(0.0), 0.0));
  EXPECT_FALSE(follower.setPath({}, point(0.0), 0.0));
  EXPECT_FALSE(follower.setPath({Eigen::VectorXd::Zero(3)}, point(0.0), 0.0));
  EXPECT_FALSE(follower.setPath({point(std::numeric_limits<double>::infinity())}, point(0.0), 0.0));
  EXPECT_FALSE(follower.setPath({point(0.0), point(1.0)}, point(0.0), -1.0));
  EXPECT_FALSE(follower.setPath({point(0.0), point(1.0)}, point(0.0, 0.21), 0.0));
  EXPECT_TRUE(follower.active());
  EXPECT_EQ(follower.update(point(0.0), -0.1).status, JointPathStatus::kInvalid);
}

TEST(JointPathFollower, RejectsInvalidConfiguration)
{
  JointPathFollowerConfig config;
  config.lookahead_distance_rad = 0.0;
  EXPECT_THROW((JointPathFollower{config}), std::invalid_argument);
  config = JointPathFollowerConfig{};
  config.projection_search_segments = 0U;
  EXPECT_THROW((JointPathFollower{config}), std::invalid_argument);
}

}  // namespace
