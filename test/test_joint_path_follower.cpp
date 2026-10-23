// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "face_tracking_arm/joint_path_follower.hpp"

namespace
{

using face_tracking_arm::control::JointPathFollower;
using face_tracking_arm::control::JointPathFollowerConfig;
using face_tracking_arm::control::JointPathStatus;
using face_tracking_arm::control::JointPathUpdate;

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

TEST(JointPathFollower, ExposesProjectionRemainderAndIgnoresDenseCollinearVertices)
{
  JointPathFollower coarse_follower;
  JointPathFollower dense_follower;
  const std::vector<Eigen::VectorXd> coarse_path{point(0.0), point(0.2)};
  const std::vector<Eigen::VectorXd> dense_path{
    point(0.0), point(0.02), point(0.04), point(0.06), point(0.08),
    point(0.10), point(0.12), point(0.14), point(0.16), point(0.18), point(0.20)};
  ASSERT_TRUE(coarse_follower.setPath(coarse_path, point(0.0), 0.0));
  ASSERT_TRUE(dense_follower.setPath(dense_path, point(0.0), 0.0));

  const auto coarse = coarse_follower.update(point(0.0), 0.01);
  const auto dense = dense_follower.update(point(0.0), 0.01);
  ASSERT_EQ(coarse.status, JointPathStatus::kTracking);
  ASSERT_EQ(dense.status, JointPathStatus::kTracking);
  EXPECT_TRUE(coarse.projected_position.isApprox(point(0.0), 1.0e-12));
  EXPECT_TRUE(dense.projected_position.isApprox(point(0.0), 1.0e-12));
  EXPECT_NEAR(coarse.remaining_distance_rad, 0.2, 1.0e-12);
  EXPECT_NEAR(dense.remaining_distance_rad, 0.2, 1.0e-12);
  EXPECT_FALSE(coarse.lookahead_crosses_corner);
  EXPECT_FALSE(dense.lookahead_crosses_corner);
  EXPECT_TRUE(coarse.reference_position.isApprox(dense.reference_position, 1.0e-12));
}

TEST(JointPathFollower, DetectsDirectionChangeInsideLookaheadIncludingAttachmentBridge)
{
  JointPathFollower corner_follower;
  ASSERT_TRUE(corner_follower.setPath(
      {point(0.0), point(0.08), point(0.08, 0.2)}, point(0.0), 0.0));
  const auto corner = corner_follower.update(point(0.0), 0.01);
  ASSERT_EQ(corner.status, JointPathStatus::kTracking);
  EXPECT_TRUE(corner.lookahead_crosses_corner);

  JointPathFollower bridge_follower;
  ASSERT_TRUE(bridge_follower.setPath(
      {point(0.0), point(0.2)}, point(0.0, 0.02), 0.0));
  const auto bridge = bridge_follower.update(point(0.0, 0.02), 0.01);
  ASSERT_EQ(bridge.status, JointPathStatus::kTracking);
  EXPECT_TRUE(bridge.lookahead_crosses_corner);
}

TEST(JointPathFollower, ProjectionWindowCoversLookaheadArcIndependentOfWaypointDensity)
{
  std::vector<Eigen::VectorXd> coarse_path;
  std::vector<Eigen::VectorXd> dense_path;
  constexpr double kQuarterTurnRad{1.5707963267948966};
  constexpr double kRadius{0.10};
  constexpr int kCoarseSegments{4};
  for (int index = 0; index <= kCoarseSegments; ++index) {
    const double angle = kQuarterTurnRad * static_cast<double>(index) /
      static_cast<double>(kCoarseSegments);
    coarse_path.push_back(point(kRadius * std::sin(angle), kRadius * (1.0 - std::cos(angle))));
  }
  for (std::size_t segment = 0U; segment + 1U < coarse_path.size(); ++segment) {
    for (int subdivision = 0; subdivision < 10; ++subdivision) {
      const double fraction = static_cast<double>(subdivision) / 10.0;
      dense_path.push_back(
        coarse_path[segment] + fraction * (coarse_path[segment + 1U] - coarse_path[segment]));
    }
  }
  dense_path.push_back(coarse_path.back());
  const auto referenceAtArc = [](
    const std::vector<Eigen::VectorXd> & path, double arc) -> Eigen::VectorXd {
      for (std::size_t index = 0U; index + 1U < path.size(); ++index) {
        const Eigen::VectorXd delta = path[index + 1U] - path[index];
        const double length = delta.norm();
        if (arc <= length) {
          return path[index] + (arc / length) * delta;
        }
        arc -= length;
      }
      return path.back();
    };

  const auto run_to_end = [](const std::vector<Eigen::VectorXd> & path) {
      JointPathFollower follower;
      EXPECT_TRUE(follower.setPath(path, path.front(), 0.0));
      Eigen::VectorXd current = path.front();
      JointPathUpdate update;
      int updates = 0;
      for (; updates < 10; ++updates) {
        update = follower.update(current, 0.01 * static_cast<double>(updates + 1));
        EXPECT_NE(update.status, JointPathStatus::kInvalid);
        EXPECT_NE(update.status, JointPathStatus::kStalled);
        if (update.status == JointPathStatus::kArrived) {
          break;
        }
        current = update.reference_position;
      }
      return std::make_pair(update, updates + 1);
    };

  JointPathFollower dense_follower;
  JointPathFollower coarse_follower;
  ASSERT_TRUE(dense_follower.setPath(dense_path, dense_path.front(), 0.0));
  ASSERT_TRUE(coarse_follower.setPath(coarse_path, coarse_path.front(), 0.0));
  const auto dense_first = dense_follower.update(dense_path.front(), 0.01);
  const auto coarse_first = coarse_follower.update(coarse_path.front(), 0.01);
  ASSERT_EQ(dense_first.status, JointPathStatus::kTracking);
  ASSERT_EQ(coarse_first.status, JointPathStatus::kTracking);
  EXPECT_TRUE(dense_first.reference_position.isApprox(coarse_first.reference_position, 1.0e-12));
  EXPECT_TRUE(dense_first.reference_position.isApprox(referenceAtArc(coarse_path, 0.10), 1.0e-12));

  const auto dense_result = run_to_end(dense_path);
  const auto coarse_result = run_to_end(coarse_path);
  EXPECT_EQ(dense_result.first.status, JointPathStatus::kArrived);
  EXPECT_EQ(coarse_result.first.status, JointPathStatus::kArrived);
  EXPECT_EQ(dense_result.second, coarse_result.second);
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
