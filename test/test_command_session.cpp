// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "command_session.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;
constexpr std::int64_t kOriginNs = 1'000'000'000;
constexpr std::int64_t kPeriodNs = 10'000'000;

rclcpp::Time stamp(const std::int64_t offset_ns = 0)
{
  return rclcpp::Time(kOriginNs + offset_ns, RCL_ROS_TIME);
}

std::deque<moveit_servo::KinematicState> publishedQueue()
{
  const double positions[] = {0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3};
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0; index < std::size(positions); ++index) {
    moveit_servo::KinematicState point(1);
    point.joint_names = {"joint1"};
    point.positions[0] = positions[index];
    point.velocities[0] = index < 3U ? 10.0 : 0.0;
    point.accelerations[0] = 0.0;
    point.time_stamp = stamp(static_cast<std::int64_t>(index) * kPeriodNs);
    queue.push_back(std::move(point));
  }
  return queue;
}

class CommandSessionTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
    ASSERT_EQ(session_.time_grid->commandStamp(kOriginNs, kPeriodNs, 0),
      kOriginNs + kPeriodNs);
    installPublication(publishedQueue());
    session_.max_queue_points = 8;
    session_.epoch_active = true;
    session_.epoch_has_published = true;
  }

  void installPublication(std::deque<moveit_servo::KinematicState> queue)
  {
    session_.forgetPublished();
    session_.queue = {queue[0], queue[1]};
    const auto stationary_time = queue[3].time_stamp;
    session_.active_publication_id = session_.publications.append(
      std::move(queue), stationary_time);
    ASSERT_TRUE(session_.active_publication_id.has_value());
  }

  TimelineRecoveryStatus recover(const std::uint64_t skipped_periods)
  {
    return session_.recoverTimeline(skipped_periods, stamp(20'000'000), reason_, 0.2, 1);
  }

  CommandSession session_;
  std::string reason_;
};

TEST_F(CommandSessionTest, DiscardBranchKeepsImmutablePublicationUntilExplicitlyForgotten)
{
  const auto id = session_.active_publication_id;
  session_.bootstrap_ack_pending = true;
  session_.bootstrap_publication_id = id;
  session_.bootstrap_trajectory_origin = stamp();
  session_.bootstrap_seed_positions = Eigen::VectorXd::Constant(1, 0.0);
  session_.published_tail_rearm_pending = true;

  session_.discardBranch();

  EXPECT_TRUE(session_.queue.empty());
  EXPECT_FALSE(session_.epoch_active);
  EXPECT_FALSE(session_.epoch_has_published);
  EXPECT_FALSE(session_.bootstrap_ack_pending);
  EXPECT_FALSE(session_.bootstrap_publication_id);
  EXPECT_FALSE(session_.bootstrap_trajectory_origin);
  EXPECT_FALSE(session_.bootstrap_seed_positions);
  EXPECT_EQ(session_.active_publication_id, id);
  ASSERT_NE(session_.activePublication(), nullptr);
  EXPECT_EQ(session_.activePublication()->execution_queue.size(), 7U);
  // A fresh epoch may start at an unrelated time after the old branch is dropped.
  EXPECT_EQ(session_.time_grid->commandStamp(2 * kOriginNs, 0, 0), 2 * kOriginNs);

  session_.forgetPublished();
  EXPECT_TRUE(session_.publications.empty());
  EXPECT_FALSE(session_.active_publication_id);
  EXPECT_EQ(session_.activePublication(), nullptr);
}

TEST_F(CommandSessionTest, RetainedPublicationStillOwnsHoldAfterRearmDiscardsEpoch)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());
  ASSERT_TRUE(session_.hasRetainedPublication());

  session_.discardBranch();

  EXPECT_FALSE(session_.epoch_has_published);
  EXPECT_EQ(session_.active_publication_id, publication_id);
  EXPECT_TRUE(session_.hasRetainedPublication());
  EXPECT_EQ(session_.activePublication()->id, *publication_id);
}

TEST_F(CommandSessionTest, StartupRetryPreservesPendingStationaryResetOwnership)
{
  session_.bootstrap_reset_ack_pending = true;
  session_.bootstrap_reset_positions = Eigen::VectorXd::Constant(1, 0.118631);
  session_.bootstrap_reset_published_time = stamp();

  session_.discardBranch(true);

  EXPECT_TRUE(session_.bootstrap_reset_ack_pending);
  ASSERT_TRUE(session_.bootstrap_reset_positions.has_value());
  EXPECT_DOUBLE_EQ((*session_.bootstrap_reset_positions)[0], 0.118631);
  EXPECT_EQ(session_.bootstrap_reset_published_time, stamp());
  EXPECT_TRUE(session_.hasRetainedPublication());
  EXPECT_EQ(session_.bootstrap_reset_stable_samples, 0U);
  EXPECT_FALSE(session_.last_bootstrap_reset_sample_time.has_value());
}

TEST(ServoRearmStateTest, AllowsStationaryFeedbackOffsetOnlyForOwnedSafePhase)
{
  constexpr double kCommandTolerance = kControllerCommandAlignmentToleranceRad;
  constexpr double kFollowingBound = 0.03;
  constexpr double kObservedOffset = 0.118631;
  EXPECT_TRUE(settled_rearm_state(
      0.0, kObservedOffset, 0.0, 0.0, kCommandTolerance, kFollowingBound,
      true, true, true));
  EXPECT_FALSE(settled_rearm_state(
      0.0, kObservedOffset, 0.0, 0.0, kCommandTolerance, kFollowingBound,
      true, true, false));
  EXPECT_FALSE(settled_rearm_state(
      0.0, kObservedOffset, 0.0, 0.0, kCommandTolerance, kFollowingBound,
      false, true, true));
  EXPECT_FALSE(settled_rearm_state(
      0.0, kObservedOffset, 0.0, 0.0, kCommandTolerance, kFollowingBound,
      true, false, true));
  EXPECT_FALSE(settled_rearm_state(
      0.0, kObservedOffset, 0.02, 0.0, kCommandTolerance, kFollowingBound,
      true, true, true));
  EXPECT_FALSE(settled_rearm_state(
      0.0, kObservedOffset, 0.0, 0.02, kCommandTolerance, kFollowingBound,
      true, true, true));
  EXPECT_FALSE(settled_rearm_state(
      2.0 * kCommandTolerance, kObservedOffset, 0.0, 0.0,
      kCommandTolerance, kFollowingBound, true, true, true));
}

TEST_F(CommandSessionTest, BootstrapResetNeedsThreeFreshSettledOwnedSamples)
{
  session_.bootstrap_reset_ack_pending = true;
  session_.bootstrap_reset_published_time = stamp();
  const Eigen::VectorXd target = Eigen::VectorXd::Constant(1, 0.118631);
  const Eigen::VectorXd stopped = Eigen::VectorXd::Zero(1);
  const auto acknowledge = [this, &target, &stopped](
    const std::int64_t sample_offset, const Eigen::VectorXd & reference,
    const Eigen::VectorXd & output, const Eigen::VectorXd & feedback,
    const Eigen::VectorXd & reference_velocity, const Eigen::VectorXd & feedback_velocity,
    const bool safe) {
      return session_.observeBootstrapResetAcknowledgement(
        stamp(sample_offset), target, reference, output, feedback, reference_velocity,
        feedback_velocity, kControllerCommandAlignmentToleranceRad, 0.006,
        kStoppedVelocityRadps, safe);
    };

  EXPECT_FALSE(acknowledge(0, stopped, target, target, stopped, stopped, true));
  EXPECT_EQ(session_.bootstrap_reset_stable_samples, 0U);
  EXPECT_FALSE(acknowledge(kPeriodNs, target, target, target, stopped, stopped, true));
  EXPECT_FALSE(acknowledge(kPeriodNs, target, target, target, stopped, stopped, true));
  EXPECT_EQ(session_.bootstrap_reset_stable_samples, 1U);
  EXPECT_FALSE(acknowledge(2 * kPeriodNs, target, target, target, stopped, stopped, false));
  EXPECT_EQ(session_.bootstrap_reset_stable_samples, 0U);
  EXPECT_FALSE(acknowledge(3 * kPeriodNs, target, target, target, stopped, stopped, true));
  EXPECT_FALSE(acknowledge(4 * kPeriodNs, target, target, target,
      Eigen::VectorXd::Constant(1, 0.02), stopped, true));
  EXPECT_EQ(session_.bootstrap_reset_stable_samples, 0U);
  EXPECT_FALSE(acknowledge(5 * kPeriodNs, target, target, target, stopped, stopped, true));
  EXPECT_FALSE(acknowledge(6 * kPeriodNs, target, target, target, stopped, stopped, true));
  EXPECT_TRUE(acknowledge(7 * kPeriodNs, target, target, target, stopped, stopped, true));
}

TEST_F(CommandSessionTest, BootstrapSourceHoldAcceptsStableOffsetButRejectsMotionAndStaleness)
{
  const Eigen::VectorXd reference = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd output = reference;
  const Eigen::VectorXd actual = Eigen::VectorXd::Constant(1, 0.118631);
  const Eigen::VectorXd stopped = Eigen::VectorXd::Zero(1);
  const auto observe = [this, &reference, &output, &actual, &stopped](
    const std::int64_t sample_offset, const Eigen::VectorXd & reference_velocity,
    const Eigen::VectorXd & feedback_velocity, const bool safe) {
      return session_.observeBootstrapSourceHold(
        stamp(sample_offset), reference, output, actual, reference_velocity, feedback_velocity,
        kControllerCommandAlignmentToleranceRad, 0.006, kStoppedVelocityRadps, safe);
    };

  EXPECT_FALSE(observe(0, stopped, stopped, true));
  EXPECT_FALSE(observe(0, stopped, stopped, true));
  EXPECT_EQ(session_.bootstrap_source_stable_samples, 1U);
  EXPECT_FALSE(observe(kPeriodNs, stopped, stopped, false));
  EXPECT_EQ(session_.bootstrap_source_stable_samples, 0U);
  EXPECT_FALSE(observe(2 * kPeriodNs, stopped, stopped, true));
  EXPECT_FALSE(observe(3 * kPeriodNs, Eigen::VectorXd::Constant(1, 0.02), stopped, true));
  EXPECT_EQ(session_.bootstrap_source_stable_samples, 0U);
  EXPECT_FALSE(observe(4 * kPeriodNs, stopped, stopped, true));
  EXPECT_FALSE(observe(5 * kPeriodNs, stopped, stopped, true));
  EXPECT_TRUE(observe(6 * kPeriodNs, stopped, stopped, true));
}

TEST_F(CommandSessionTest, RestartSettlingPreservesPendingTailAndDeadline)
{
  session_.stable_rearm_samples = 5;
  session_.last_rearm_sample_time = stamp();
  session_.published_tail_rearm_pending = true;
  const auto wall_start = CommandSession::WallTime{} + 10s;
  session_.startRearmDeadline(stamp(), wall_start);
  const auto deadline = session_.rearm_deadline;
  session_.restartSettling();
  EXPECT_EQ(session_.stable_rearm_samples, 0U);
  EXPECT_FALSE(session_.last_rearm_sample_time);
  EXPECT_TRUE(session_.published_tail_rearm_pending);
  EXPECT_EQ(session_.rearm_deadline, deadline);
}

TEST_F(CommandSessionTest, DeadlinesIncludePublishedStopAndExpireOnEitherClock)
{
  const auto wall_start = CommandSession::WallTime{} + 10s;
  session_.published_tail_rearm_pending = true;
  session_.startRearmDeadline(stamp(), wall_start);
  ASSERT_TRUE(session_.rearm_deadline);
  ASSERT_TRUE(session_.rearm_wall_deadline);
  EXPECT_EQ(*session_.rearm_deadline, stamp(2'030'000'000));
  EXPECT_NEAR(std::chrono::duration<double>(
      *session_.rearm_wall_deadline - wall_start).count(), 2.03, 1.0e-8);
  EXPECT_FALSE(session_.rearmDeadlineExpired(stamp(2'000'000'000), wall_start + 2s));
  EXPECT_TRUE(session_.rearmDeadlineExpired(stamp(2'030'000'000), wall_start));
  EXPECT_TRUE(session_.rearmDeadlineExpired(stamp(), *session_.rearm_wall_deadline));

  const auto original_ros_deadline = session_.rearm_deadline;
  const auto original_wall_deadline = session_.rearm_wall_deadline;
  session_.ensureRearmDeadline(stamp(500'000'000), wall_start + 500ms);
  EXPECT_EQ(session_.rearm_deadline, original_ros_deadline);
  EXPECT_EQ(session_.rearm_wall_deadline, original_wall_deadline);
  session_.clearRearmDeadlines();
  EXPECT_FALSE(session_.rearmDeadlineExpired(stamp(10'000'000'000), wall_start + 10s));
}

TEST_F(CommandSessionTest, RearmTimeoutReportIsOneShotAndNewWindowCanRecover)
{
  const auto wall_start = CommandSession::WallTime{} + 10s;
  session_.startRearmDeadline(stamp(), wall_start);
  ASSERT_TRUE(session_.rearmDeadlineExpired(stamp(2'000'000'000), wall_start + 2s));
  EXPECT_TRUE(session_.claimRearmTimeoutReport());
  EXPECT_FALSE(session_.claimRearmTimeoutReport());

  // Reporting a timeout does not disable readiness checks or poison a later window.
  session_.clearRearmDeadlines();
  session_.startRearmDeadline(stamp(3'000'000'000), wall_start + 3s);
  EXPECT_FALSE(session_.rearmDeadlineExpired(stamp(3'100'000'000), wall_start + 3100ms));
  EXPECT_TRUE(session_.claimRearmTimeoutReport());
}

TEST_F(CommandSessionTest, NewSettlingBudgetStartsAtNowWhenPublishedStopIsInThePast)
{
  const auto wall_start = CommandSession::WallTime{} + 10s;
  session_.published_tail_rearm_pending = true;
  session_.ensureRearmDeadline(stamp(100'000'000), wall_start);
  EXPECT_EQ(session_.rearm_deadline, stamp(2'100'000'000));
  EXPECT_EQ(session_.rearm_wall_deadline, wall_start + 2s);
}

TEST_F(CommandSessionTest, HealthyPublicationBudgetIgnoresBootstrapAcknowledgement)
{
  session_.consecutive_automatic_rearms = 3;
  session_.bootstrap_ack_pending = true;
  for (int index = 0; index < 150; ++index) {
    session_.recordHealthyPublication();
  }
  EXPECT_EQ(session_.healthy_publications_since_rearm, 0U);
  EXPECT_EQ(session_.consecutive_automatic_rearms, 3U);
  session_.bootstrap_ack_pending = false;
  for (int index = 0; index < 99; ++index) {
    session_.recordHealthyPublication();
  }
  EXPECT_EQ(session_.consecutive_automatic_rearms, 3U);
  session_.recordHealthyPublication();
  EXPECT_EQ(session_.consecutive_automatic_rearms, 0U);
  EXPECT_EQ(session_.healthy_publications_since_rearm, 0U);
}

TEST_F(CommandSessionTest, RecoveryAdoptsOnlyTheExactAlreadyTransmittedPrefix)
{
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  ASSERT_EQ(recover(2), TimelineRecoveryStatus::kSuccess) << reason_;
  ASSERT_EQ(session_.queue.size(), 4U);
  for (std::size_t index = 0; index < session_.queue.size(); ++index) {
    const auto & actual = session_.queue[index];
    const auto & expected = published->execution_queue[index];
    EXPECT_EQ(actual.time_stamp, expected.time_stamp);
    EXPECT_TRUE(actual.positions.isApprox(expected.positions));
    EXPECT_TRUE(actual.velocities.isApprox(expected.velocities));
    EXPECT_TRUE(actual.accelerations.isApprox(expected.accelerations));
  }
  EXPECT_EQ(session_.timeline_recovery_events, 1U);
  EXPECT_EQ(session_.timeline_recovered_periods, 2U);
  EXPECT_EQ(session_.time_grid->commandStamp(kOriginNs, 0, 0),
    kOriginNs + 4 * kPeriodNs);
}

TEST_F(CommandSessionTest, DeferredDeadlineRetryUsesRequiredPublishedAnchorWithoutDoubleCounting)
{
  ASSERT_TRUE(session_.deferCommandDeadlineRetry());
  EXPECT_FALSE(session_.deferCommandDeadlineRetry());
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(kPeriodNs));

  // At 11 ms, preserve the 30 ms initial lead (20 ms physical minimum plus
  // one 10 ms compute period). Three published periods are needed; one
  // scheduler skip must not be added again to that same command-grid deficit.
  ASSERT_EQ(session_.timelineRecoveryPeriods(1U, stamp(11'000'000).nanoseconds(),
      30'000'000), 3U);
  control::ControlTimeGrid unadvanced_grid{kPeriodNs};
  ASSERT_EQ(unadvanced_grid.commandStamp(kOriginNs, kPeriodNs, 0),
    kOriginNs + kPeriodNs);
  EXPECT_FALSE(unadvanced_grid.commandStamp(
      stamp(16'000'000).nanoseconds(), 50'000'000, 20'000'000));
  ASSERT_EQ(session_.recoverTimeline(3U, stamp(11'000'000), reason_, 0.2, 1),
    TimelineRecoveryStatus::kSuccess) << reason_;
  session_.completeDeferredCommandDeadlineRetry();

  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  ASSERT_EQ(session_.queue.back().time_stamp, stamp(40'000'000));
  EXPECT_TRUE(session_.queue.back().positions.isApprox(published->execution_queue[4].positions));
  EXPECT_TRUE(session_.queue.back().velocities.isApprox(published->execution_queue[4].velocities));
  EXPECT_TRUE(session_.queue.back().accelerations.isApprox(
      published->execution_queue[4].accelerations));
  // A 5 ms QP/publication calculation still leaves more than the 20 ms
  // physical command lead after the recovered 30 ms epoch lead.
  EXPECT_EQ(session_.time_grid->commandStamp(
      stamp(16'000'000).nanoseconds(), 50'000'000, 20'000'000),
    stamp(50'000'000).nanoseconds());

  // A second miss before a healthy publication falls back to the caller's
  // normal re-arm path. A successful publication grants one new retry.
  EXPECT_FALSE(session_.deferCommandDeadlineRetry());
  session_.recordHealthyPublication();
  EXPECT_TRUE(session_.deferCommandDeadlineRetry());
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationAdoptsExactlyOneTransmittedPoint)
{
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  const auto status = session_.continuePublishedBrake(
    stamp(), 20'000'000, reason_, 0.2, 1);

  ASSERT_EQ(status, PublishedBrakeContinuationStatus::kContinued) << reason_;
  ASSERT_EQ(session_.queue.size(), 3U);
  const auto & next = published->execution_queue[2];
  EXPECT_EQ(session_.queue.back().time_stamp, next.time_stamp);
  EXPECT_TRUE(session_.queue.back().positions.isApprox(next.positions));
  EXPECT_TRUE(session_.queue.back().velocities.isApprox(next.velocities));
  EXPECT_TRUE(session_.queue.back().accelerations.isApprox(next.accelerations));
  EXPECT_EQ(session_.published_brake_continuation_count, 1U);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.timeline_recovered_periods, 0U);

  for (std::size_t expected_index = 3U; expected_index <= 5U; ++expected_index) {
    const auto repeated = session_.continuePublishedBrake(
      stamp(), 20'000'000, reason_, 0.2, 1);
    ASSERT_EQ(repeated, PublishedBrakeContinuationStatus::kContinued) << reason_;
    EXPECT_EQ(session_.queue.back().time_stamp,
      published->execution_queue[expected_index].time_stamp);
  }
  EXPECT_EQ(session_.published_brake_continuation_count, 4U);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.continuePublishedBrake(
      stamp(), 20'000'000, reason_, 0.2, 1),
    PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix);
  EXPECT_EQ(session_.queue.back().time_stamp, published->execution_queue[5].time_stamp);
  EXPECT_NE(session_.queue.back().time_stamp, published->execution_queue.back().time_stamp);
  EXPECT_EQ(session_.time_grid->commandStamp(kOriginNs, 0, 0),
    published->execution_queue[5].time_stamp.nanoseconds() + kPeriodNs);
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationRequiresNextTickCommandLead)
{
  const auto queue_before = session_.queue;
  const auto periods_before = session_.time_grid->periodsUntilCommandLead(
    stamp(5'000'000).nanoseconds(), 30'000'000);
  ASSERT_TRUE(periods_before.has_value());

  const auto status = session_.continuePublishedBrake(
    stamp(5'000'000), 30'000'000, reason_, 0.2, 1);

  EXPECT_EQ(status, PublishedBrakeContinuationStatus::kNotEligible);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  EXPECT_EQ(session_.queue.back().time_stamp, queue_before.back().time_stamp);
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(5'000'000).nanoseconds(), 30'000'000), periods_before);
  EXPECT_EQ(session_.published_brake_continuation_count, 0U);
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationRejectsInactiveEpochAndRearmStates)
{
  const auto queue_size_before = session_.queue.size();
  const auto queue_stamp_before = session_.queue.back().time_stamp;
  const auto periods_before = session_.time_grid->periodsUntilCommandLead(
    stamp().nanoseconds(), 20'000'000);
  ASSERT_TRUE(periods_before.has_value());
  const auto expect_not_eligible_without_mutation = [this, queue_size_before,
      queue_stamp_before, periods_before]() {
      EXPECT_EQ(session_.continuePublishedBrake(
          stamp(), 20'000'000, reason_, 0.2, 1),
        PublishedBrakeContinuationStatus::kNotEligible);
      EXPECT_EQ(session_.queue.size(), queue_size_before);
      EXPECT_EQ(session_.queue.back().time_stamp, queue_stamp_before);
      EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
          stamp().nanoseconds(), 20'000'000), periods_before);
      EXPECT_EQ(session_.published_brake_continuation_count, 0U);
    };

  session_.rearm_pending = true;
  expect_not_eligible_without_mutation();
  session_.rearm_pending = false;
  session_.published_tail_rearm_pending = true;
  expect_not_eligible_without_mutation();
  session_.published_tail_rearm_pending = false;
  session_.bootstrap_ack_pending = true;
  expect_not_eligible_without_mutation();
  session_.bootstrap_ack_pending = false;
  session_.epoch_active = false;
  expect_not_eligible_without_mutation();
  session_.epoch_active = true;
  session_.epoch_has_published = false;
  expect_not_eligible_without_mutation();
  session_.epoch_has_published = true;
  session_.forgetPublished();
  expect_not_eligible_without_mutation();
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationNeverUsesWithheldTerminalPoint)
{
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  session_.queue = {published->execution_queue[4], published->execution_queue[5]};
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue[4].time_stamp.nanoseconds()));

  const auto status = session_.continuePublishedBrake(
    stamp(20'000'000), 0, reason_, 0.2, 1);

  EXPECT_EQ(status, PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix);
  EXPECT_EQ(session_.queue.back().time_stamp, published->execution_queue[5].time_stamp);
  EXPECT_NE(session_.queue.back().time_stamp, published->execution_queue.back().time_stamp);
  EXPECT_EQ(session_.published_brake_continuation_count, 0U);
}

TEST_F(CommandSessionTest, ObservedSkipsCoverTheDeadlineDeficitWithoutBeingAddedTwice)
{
  ASSERT_TRUE(session_.deferCommandDeadlineRetry());
  // One additional period is needed for the minimum lead, but two ordinary
  // scheduler skips already move the exact published anchor far enough.
  EXPECT_EQ(session_.timelineRecoveryPeriods(2U, stamp(1'000'000).nanoseconds(),
      30'000'000), 2U);
}

TEST_F(CommandSessionTest, DiscardingTheBranchClearsDeferredDeadlineRetry)
{
  ASSERT_TRUE(session_.deferCommandDeadlineRetry());
  session_.discardBranch();
  EXPECT_FALSE(session_.command_deadline_retry_pending);
  EXPECT_FALSE(session_.command_deadline_retry_used);
  EXPECT_FALSE(session_.deferCommandDeadlineRetry());
}

TEST_F(CommandSessionTest, RecoveryDropsOldHistoryToRespectQueueCapacity)
{
  session_.max_queue_points = 3;
  ASSERT_EQ(recover(2), TimelineRecoveryStatus::kSuccess) << reason_;
  EXPECT_EQ(session_.queue.size(), 3U);
  EXPECT_EQ(session_.queue.front().time_stamp, stamp(kPeriodNs));
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(3 * kPeriodNs));
}

TEST_F(CommandSessionTest, WithheldSentinelCannotExtendTheRecoveryHorizon)
{
  EXPECT_EQ(recover(5), TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  EXPECT_NE(reason_.find("suffix is too short"), std::string::npos);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(kPeriodNs));
  EXPECT_EQ(recover(4), TimelineRecoveryStatus::kSuccess) << reason_;
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(5 * kPeriodNs));
}

TEST_F(CommandSessionTest, RecoveryCannotExceedHardQueueCapacity)
{
  session_.max_queue_points = 2;
  EXPECT_EQ(recover(2), TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  EXPECT_NE(reason_.find("hard bound"), std::string::npos);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(kPeriodNs));
}

TEST_F(CommandSessionTest, MismatchedBranchAndNonUniformPublishedGridAreRejected)
{
  session_.queue.back().positions[0] += 0.01;
  EXPECT_EQ(recover(1), TimelineRecoveryStatus::kInvariantFailure);
  EXPECT_NE(reason_.find("exact prefix"), std::string::npos);

  auto uneven_queue = publishedQueue();
  uneven_queue[2].time_stamp = stamp(21'000'000);
  installPublication(std::move(uneven_queue));
  EXPECT_EQ(recover(2), TimelineRecoveryStatus::kInvariantFailure);
  EXPECT_NE(reason_.find("fixed command grid"), std::string::npos);
  EXPECT_EQ(session_.timeline_recovered_periods, 0U);
}

TEST_F(CommandSessionTest, RecoveryCannotAdoptAnAnchorFromAnotherCommandEpoch)
{
  session_.time_grid->resetCommandEpoch();
  ASSERT_EQ(session_.time_grid->commandStamp(kOriginNs + 1, kPeriodNs, 0),
    kOriginNs + kPeriodNs + 1);
  EXPECT_EQ(recover(1), TimelineRecoveryStatus::kInvariantFailure);
  EXPECT_NE(reason_.find("active command grid"), std::string::npos);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(kPeriodNs));
}

TEST_F(CommandSessionTest, RecoveryRequiresAPublicationOnlyWhenAnActiveEpochSkippedTime)
{
  session_.forgetPublished();
  EXPECT_EQ(recover(0), TimelineRecoveryStatus::kSuccess);
  EXPECT_EQ(recover(1), TimelineRecoveryStatus::kInvariantFailure);
  session_.epoch_active = false;
  EXPECT_EQ(recover(10), TimelineRecoveryStatus::kSuccess);
}

}  // namespace
}  // namespace face_tracking_arm
