// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "face_tracking_arm/compact_path_execution.hpp"

namespace face_tracking_arm::control
{
namespace
{

JointMotionState sample(const double position, const double velocity)
{
  return JointMotionState{
    Eigen::VectorXd::Constant(1, position),
    Eigen::VectorXd::Constant(1, velocity),
    Eigen::VectorXd::Zero(1)};
}

TEST(CompactPathExecutionTest, FailedPublicationKeepsCursorAndProgress)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {sample(0.0, 0.0), sample(0.1, 0.2), sample(0.2, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 0.1, 0.2};
  JointMotionLimits prefix_limits;
  execution.install(std::move(profile), std::move(prefix_limits));

  execution.recordCommandPublished(false);
  EXPECT_EQ(execution.cursor(), 0U);
  EXPECT_DOUBLE_EQ(execution.progressRad(), 0.0);

  execution.recordCommandPublished(true);
  EXPECT_EQ(execution.cursor(), 1U);
  EXPECT_DOUBLE_EQ(execution.progressRad(), 0.1);
}

TEST(CompactPathExecutionTest, QueueMatchingIsForwardOnlyAndDoesNotAdvanceCursor)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {sample(0.0, 0.0), sample(0.1, 0.2), sample(0.2, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 0.1, 0.2};
  execution.install(std::move(profile), JointMotionLimits{});

  const auto future = execution.matchQueueState(sample(0.1, 0.2), 1.0e-8, 1.0e-8, 1.0e-8);
  EXPECT_EQ(future.status, CompactQueueMatchStatus::kMatchedFuture);
  EXPECT_EQ(future.sample_index, 1U);
  EXPECT_EQ(execution.cursor(), 0U);

  execution.recordCommandPublished(true);
  const auto old = execution.matchQueueState(sample(0.0, 0.0), 1.0e-8, 1.0e-8, 1.0e-8);
  EXPECT_EQ(old.status, CompactQueueMatchStatus::kNoMatch);
}

TEST(CompactPathExecutionTest, RemainingRoutePreservesIntervalMetadata)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {
    sample(0.0, 0.1), sample(1.0, 0.1), sample(3.0, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 1.0, 3.0};
  profile.speed_scale_per_interval = {0.8, 0.5};
  execution.install(std::move(profile), JointMotionLimits{});

  const auto suffix = execution.remainingRoute(0.5);
  ASSERT_TRUE(suffix);
  ASSERT_EQ(suffix->path.size(), 3U);
  ASSERT_EQ(suffix->speed_scales.size(), 2U);
  EXPECT_DOUBLE_EQ(suffix->path[0][0], 0.5);
  EXPECT_DOUBLE_EQ(suffix->path[1][0], 1.0);
  EXPECT_DOUBLE_EQ(suffix->path[2][0], 3.0);
  EXPECT_DOUBLE_EQ(suffix->speed_scales[0], 0.8);
  EXPECT_DOUBLE_EQ(suffix->speed_scales[1], 0.5);
}

TEST(CompactPathExecutionTest, InstallingEmptyProfileClearsActiveExecution)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {sample(0.0, 0.0), sample(0.1, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 0.1};
  execution.install(std::move(profile), JointMotionLimits{});
  ASSERT_TRUE(execution.active());

  execution.install(CompactJointProfile{}, JointMotionLimits{});

  EXPECT_FALSE(execution.active());
  EXPECT_EQ(execution.cursor(), 0U);
  EXPECT_DOUBLE_EQ(execution.progressRad(), 0.0);
  EXPECT_EQ(execution.totalPathLengthRad(), 0.0);
  EXPECT_EQ(execution.matchQueueState(sample(0.0, 0.0), 1.0e-8, 1.0e-8, 1.0e-8).status,
    CompactQueueMatchStatus::kNoMatch);
}

TEST(CompactPathExecutionTest, ProjectionRejectsDistinctForwardMatches)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {
    sample(0.0, 1.0), sample(1.0, 1.0), sample(0.0, 1.0), sample(1.0, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 1.0, 2.0, 3.0};
  profile.speed_scale_per_interval = {1.0, 1.0, 1.0};
  execution.install(std::move(profile), JointMotionLimits{});

  const auto projected = execution.projectQueueStateToProgress(sample(0.5, 1.0), 1.0e-9);

  EXPECT_FALSE(projected.has_value());
  EXPECT_EQ(execution.cursor(), 0U);
  EXPECT_DOUBLE_EQ(execution.progressRad(), 0.0);
}

TEST(CompactPathExecutionTest, ResyncStartsAtForwardPublishedBrakeEndpoint)
{
  CompactPathExecution execution;
  CompactJointProfile profile;
  profile.samples = {
    sample(0.0, 1.0), sample(1.0, 1.0), sample(2.0, 1.0), sample(3.0, 0.0)};
  profile.path_progress_rad_per_sample = {0.0, 1.0, 2.0, 3.0};
  profile.speed_scale_per_interval = {0.7, 0.7, 0.7};
  execution.install(std::move(profile), JointMotionLimits{});

  const auto queued_progress = execution.projectQueueStateToProgress(sample(0.5, 1.0), 1.0e-9);
  ASSERT_TRUE(queued_progress);
  const auto stopped_progress = execution.projectQueueStateToProgress(
    sample(1.5, 0.0), 1.0e-9, *queued_progress);
  ASSERT_TRUE(stopped_progress);
  EXPECT_GT(*stopped_progress, *queued_progress);

  const auto suffix = execution.remainingRoute(*stopped_progress);
  ASSERT_TRUE(suffix);
  ASSERT_FALSE(suffix->path.empty());
  EXPECT_DOUBLE_EQ(suffix->path.front()[0], 1.5);
  EXPECT_DOUBLE_EQ(suffix->start_progress_rad, 1.5);
}

}  // namespace
}  // namespace face_tracking_arm::control
