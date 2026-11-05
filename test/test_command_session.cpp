// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
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

std::deque<moveit_servo::KinematicState> longPublishedQueue(
  const std::size_t count = 8U)
{
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0; index < count; ++index) {
    moveit_servo::KinematicState point(1);
    point.joint_names = {"joint1"};
    point.positions[0] = 0.1 * static_cast<double>(std::min(index, std::size_t{3}));
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

TEST_F(CommandSessionTest, PublishedBrakeContinuationCatchesUpExactLateTransmittedSamples)
{
  for (const std::uint64_t periods : {2U, 3U, 5U}) {
    session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
    ASSERT_EQ(session_.time_grid->commandStamp(kOriginNs, kPeriodNs, 0),
      kOriginNs + kPeriodNs);
    installPublication(longPublishedQueue());
    const auto * published = session_.activePublication();
    ASSERT_NE(published, nullptr);
    const auto lead_ns = 15'000'000 +
      static_cast<std::int64_t>(periods) * kPeriodNs;
    const auto required_periods = session_.time_grid->periodsUntilCommandLead(
      stamp(5'000'000).nanoseconds(), lead_ns);
    ASSERT_TRUE(required_periods.has_value());
    ASSERT_EQ(*required_periods, periods);
    const auto continuation_count_before = session_.published_brake_continuation_count;

    const auto status = session_.continuePublishedBrake(
      stamp(5'000'000), lead_ns, reason_, 0.2, 1);

    ASSERT_EQ(status, PublishedBrakeContinuationStatus::kContinued) << reason_;
    ASSERT_EQ(session_.queue.size(), 2U + periods);
    const auto & expected = published->execution_queue[1U + periods];
    EXPECT_EQ(session_.queue.back().time_stamp, expected.time_stamp);
    EXPECT_TRUE(session_.queue.back().positions.isApprox(expected.positions));
    EXPECT_TRUE(session_.queue.back().velocities.isApprox(expected.velocities));
    EXPECT_TRUE(session_.queue.back().accelerations.isApprox(expected.accelerations));
    EXPECT_EQ(session_.time_grid->commandStamp(kOriginNs, 0, 0),
      expected.time_stamp.nanoseconds() + kPeriodNs);
    EXPECT_EQ(session_.published_brake_continuation_count,
      continuation_count_before + 1U);
    EXPECT_EQ(session_.timeline_recovery_events, 0U);
    EXPECT_EQ(session_.timeline_recovered_periods, 0U);
  }
}

TEST_F(CommandSessionTest, ContinuationLeadIsNotRecoveredAgainAsTimerSkips)
{
  session_.max_queue_points = 13U;
  auto transmitted = longPublishedQueue(32U);
  installPublication(transmitted);
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);

  // The online branch already contains exact command states from t=0 through
  // t=60 ms. A late validation at t=45 ms continues the transmitted suffix
  // until the next command has the requested 50 ms lead (anchor at 90 ms).
  session_.queue.clear();
  for (std::size_t index = 0; index <= 6U; ++index) {
    session_.queue.push_back(published->execution_queue[index]);
  }
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue[6U].time_stamp.nanoseconds()));
  ASSERT_EQ(session_.time_grid->observe(stamp().nanoseconds()).status,
    control::ControlTimeStatus::kDue);

  ASSERT_EQ(session_.continuePublishedBrake(
      stamp(45'000'000), 50'000'000, reason_, 0.05, 1),
    PublishedBrakeContinuationStatus::kContinued) << reason_;
  ASSERT_EQ(session_.queue.back().time_stamp, stamp(90'000'000));

  // The next timer observation reports four skips from its t=0 schedule. But
  // the continued command grid already reaches t=100 ms, exactly 50 ms ahead
  // of this t=50 ms callback. Replaying those same four periods again creates
  // a 90 ms lead and fills the 13-point queue before the ordinary append.
  const auto observation = session_.time_grid->observe(stamp(50'000'000).nanoseconds());
  ASSERT_EQ(observation.status, control::ControlTimeStatus::kDue);
  ASSERT_EQ(observation.skipped_periods, 4U);
  ASSERT_EQ(observation.scheduled_time_ns, stamp(50'000'000).nanoseconds());
  ASSERT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(50'000'000).nanoseconds(), 50'000'000), 0U);

  const auto recovery_periods = session_.timelineRecoveryPeriods(
    observation.skipped_periods, stamp(50'000'000).nanoseconds(), 50'000'000);
  ASSERT_TRUE(recovery_periods.has_value());
  // The command-grid deficit is authoritative: the next command stamp is
  // already 50 ms ahead after continuation, so no additional recovery point
  // is needed even though the callback timer reports four skipped periods.
  const auto required_periods = session_.time_grid->periodsUntilCommandLead(
    stamp(50'000'000).nanoseconds(), 50'000'000);
  ASSERT_TRUE(required_periods.has_value());
  EXPECT_EQ(*recovery_periods, *required_periods);
  ASSERT_EQ(*required_periods, 0U);
  if (*recovery_periods > 0U) {
    ASSERT_EQ(session_.recoverTimeline(
        *recovery_periods, stamp(50'000'000), reason_, 0.05, 1),
      TimelineRecoveryStatus::kSuccess) << reason_;
  }
  session_.pruneExpiredHistory(stamp(50'000'000), 0.05);

  // The next command must remain the exact transmitted t=100 ms point.
  const auto next_stamp = session_.time_grid->commandStamp(
    stamp(50'000'000).nanoseconds(), 50'000'000, 20'000'000);
  ASSERT_TRUE(next_stamp.has_value());
  EXPECT_EQ(*next_stamp, stamp(100'000'000).nanoseconds());
  const auto next = std::find_if(
    published->execution_queue.begin(), published->execution_queue.end(),
    [&next_stamp](const auto & point) {
      return point.time_stamp.nanoseconds() == *next_stamp;
    });
  ASSERT_NE(next, published->execution_queue.end());

  // This is the production append bound check; no future queue point is
  // discarded to make capacity.
  EXPECT_LE(session_.queue.size() + 1U, session_.max_queue_points);
  session_.queue.push_back(*next);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(100'000'000));
  EXPECT_LE(session_.queue.size(), session_.max_queue_points);
}

TEST_F(CommandSessionTest, StartupWithoutCommandGridAnchorUsesObservedSkips)
{
  session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
  session_.epoch_active = false;
  session_.epoch_has_published = false;

  EXPECT_EQ(session_.timelineRecoveryPeriods(0U, stamp().nanoseconds(), 50'000'000), 0U);
  EXPECT_EQ(session_.timelineRecoveryPeriods(3U, stamp().nanoseconds(), 50'000'000), 3U);

  session_.epoch_active = true;
  EXPECT_FALSE(session_.timelineRecoveryPeriods(
      3U, stamp().nanoseconds(), 50'000'000).has_value());
  session_.epoch_active = false;
  session_.command_deadline_retry_pending = true;
  EXPECT_FALSE(session_.timelineRecoveryPeriods(
      3U, stamp().nanoseconds(), 50'000'000).has_value());
}

TEST_F(CommandSessionTest, CompactTimelineRecoveryUsesFortyMillisecondLeadHysteresis)
{
  struct TimingCase
  {
    std::int64_t callback_time_ns;
    std::int64_t last_command_stamp_ns;
    std::int64_t expected_next_grid_lead_ns;
  };
  const TimingCase cases[] = {
    {437'127'000'000LL, 437'165'000'000LL, 48'000'000LL},
    {605'295'000'000LL, 605'325'000'000LL, 40'000'000LL},
  };
  for (const auto & test_case : cases) {
    session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
    const auto now_ns = test_case.callback_time_ns;
    ASSERT_EQ(session_.time_grid->commandStamp(test_case.last_command_stamp_ns, 0, 0),
      test_case.last_command_stamp_ns);
    EXPECT_EQ(test_case.last_command_stamp_ns + kPeriodNs - now_ns,
      test_case.expected_next_grid_lead_ns);

    const auto periods = session_.timelineRecoveryPeriods(
      1U, now_ns, 50'000'000, 40'000'000);
    ASSERT_TRUE(periods.has_value());
    EXPECT_EQ(*periods, 0U) << "callback time " << test_case.callback_time_ns;
    EXPECT_EQ(session_.timelineRecoveryPeriods(
        1U, now_ns, 50'000'000), 1U);
  }
}

TEST_F(CommandSessionTest, CompactTimelineRecoveryBelowHysteresisPreservesRetryToFifty)
{
  constexpr std::int64_t callback_time_ns = 605'325'000'000LL;
  constexpr std::int64_t now_ns = callback_time_ns;
  session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
  ASSERT_TRUE(session_.time_grid->commandStamp(now_ns + 39'000'000 - kPeriodNs, 0, 0));

  ASSERT_EQ(session_.timelineRecoveryPeriods(
      0U, now_ns, 50'000'000, 40'000'000), 2U);
  session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
  ASSERT_TRUE(session_.time_grid->commandStamp(now_ns + 30'000'000 - kPeriodNs, 0, 0));
  ASSERT_EQ(session_.timelineRecoveryPeriods(
      0U, now_ns, 50'000'000, 40'000'000), 2U);

  // A pending deadline retry always retains the existing 50 ms recovery.
  session_.time_grid = std::make_unique<control::ControlTimeGrid>(kPeriodNs);
  ASSERT_TRUE(session_.time_grid->commandStamp(now_ns + 48'000'000 - kPeriodNs, 0, 0));
  session_.command_deadline_retry_pending = true;
  EXPECT_EQ(session_.timelineRecoveryPeriods(
      0U, now_ns, 50'000'000, 40'000'000), 1U);
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

TEST_F(CommandSessionTest, VerifiedTerminalHoldResumeExtendsOnlyLocalCommandQueue)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());

  const auto status = session_.resumeVerifiedTerminalHold(
    *publication_id, stamp(100'000'000), 20'000'000, 1, true, reason_);

  EXPECT_EQ(status, TimelineRecoveryStatus::kSuccess);
  EXPECT_EQ(session_.active_publication_id, publication_id);
  ASSERT_NE(session_.activePublication(), nullptr);
  EXPECT_EQ(session_.activePublication()->execution_queue.size(), 7U);
  ASSERT_FALSE(session_.queue.empty());
  EXPECT_EQ(session_.queue.size(), 1U);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(110'000'000));
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(100'000'000).nanoseconds(), 20'000'000), 0U);
  EXPECT_DOUBLE_EQ(session_.queue.back().positions[0], 0.3);
  EXPECT_DOUBLE_EQ(session_.queue.back().velocities[0], 0.0);
  EXPECT_DOUBLE_EQ(session_.queue.back().accelerations[0], 0.0);
}

TEST_F(CommandSessionTest, VerifiedTerminalHoldPredicateAcceptsOnlyResumedStationaryCommand)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;

  EXPECT_TRUE(session_.hasVerifiedTerminalHold(1U));
}

TEST_F(CommandSessionTest, VerifiedTerminalHoldPredicateRequiresProofAtSentinelTimestamp)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  const auto transmitted_end = std::prev(published->execution_queue.end(), 2);
  auto unverified_anchor = *transmitted_end;
  unverified_anchor.time_stamp = published->execution_queue.back().time_stamp;
  session_.queue.clear();
  session_.queue.push_back(std::move(unverified_anchor));
  session_.verified_terminal_hold_publication_id.reset();
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue.back().time_stamp.nanoseconds()));

  // The withheld sentinel's timestamp can also be a valid future virtual-hold
  // stamp after proof; without the publication marker, it has no such proof.
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.verified_terminal_hold_publication_id = session_.active_publication_id;
  EXPECT_TRUE(session_.hasVerifiedTerminalHold(1U));
}

TEST_F(CommandSessionTest, VerifiedTerminalHoldPredicateRejectsChangedPositionOrMotion)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;
  session_.queue.front().positions[0] += 1.0e-6;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.queue.front().positions[0] = 0.3;

  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(200'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;
  session_.queue.front().velocities[0] = 1.0e-6;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.queue.front().velocities[0] = 0.0;

  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(300'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;
  session_.queue.front().accelerations[0] = 1.0e-6;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
}

TEST_F(CommandSessionTest,
      VerifiedTerminalHoldPredicateRejectsChangedPublicationEpochAndPendingReset)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;

  session_.verified_terminal_hold_publication_id = *session_.active_publication_id + 1U;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.verified_terminal_hold_publication_id = session_.active_publication_id;
  session_.epoch_has_published = false;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.epoch_has_published = true;
  session_.bootstrap_reset_ack_pending = true;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
  session_.bootstrap_reset_ack_pending = false;
  session_.rearm_pending = true;
  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
}

TEST_F(CommandSessionTest, VerifiedTerminalHoldPredicateRejectsCommandGridMismatch)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;
  session_.time_grid->resetCommandEpoch();
  ASSERT_TRUE(session_.time_grid->commandStamp(
      session_.queue.front().time_stamp.nanoseconds() + kPeriodNs, 0, 0).has_value());

  EXPECT_FALSE(session_.hasVerifiedTerminalHold(1U));
}

TEST_F(CommandSessionTest, TerminalHoldExpiryWithoutSettledProofCannotExtendQueue)
{
  const auto queue_before = session_.queue;
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(100'000'000), 20'000'000, 1, false, reason_),
    TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  for (std::size_t index = 0U; index < queue_before.size(); ++index) {
    EXPECT_EQ(session_.queue[index].time_stamp, queue_before[index].time_stamp);
    EXPECT_TRUE(session_.queue[index].positions.isApprox(queue_before[index].positions));
  }
}

TEST_F(CommandSessionTest, TerminalHoldResumeRejectsChangedPublicationIdentity)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  const auto queue_before = session_.queue;

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id + 1U, stamp(100'000'000), 20'000'000,
      1, true, reason_),
    TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(100'000'000).nanoseconds(), 20'000'000), 10U);
}

TEST_F(CommandSessionTest, TerminalHoldResumeRejectsWithheldSentinelAsStationaryAnchor)
{
  auto * published = const_cast<control::PublishedTrajectoryRecord *>(
    session_.activePublication());
  ASSERT_NE(published, nullptr);
  published->stationary_time = published->execution_queue.back().time_stamp;

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_),
    TimelineRecoveryStatus::kInvariantFailure);
}

TEST_F(CommandSessionTest, TerminalHoldResumeRejectsNonstationaryPublishedSuffix)
{
  auto * published = const_cast<control::PublishedTrajectoryRecord *>(
    session_.activePublication());
  ASSERT_NE(published, nullptr);
  published->execution_queue.back().velocities[0] = 0.01;

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_),
    TimelineRecoveryStatus::kInvariantFailure);
}

TEST_F(CommandSessionTest, TerminalHoldResumeRejectsZeroQueueCapacityWithoutMutation)
{
  session_.max_queue_points = 0U;
  const auto queue_before = session_.queue;
  const auto grid_before = session_.time_grid->periodsUntilCommandLead(
    stamp(100'000'000).nanoseconds(), 20'000'000);

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_),
    TimelineRecoveryStatus::kInvariantFailure);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(100'000'000).nanoseconds(), 20'000'000), grid_before);
}

TEST_F(CommandSessionTest, TerminalHoldResumeAcceptsGridAlreadyAtTransmittedEnd)
{
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  const auto transmitted_end = std::prev(published->execution_queue.end(), 2);
  session_.queue.clear();
  for (auto point = published->execution_queue.begin(); point != std::next(transmitted_end);
    ++point)
  {
    session_.queue.push_back(*point);
  }
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      transmitted_end->time_stamp.nanoseconds()));

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_),
    TimelineRecoveryStatus::kSuccess);
  EXPECT_LE(session_.queue.size(), session_.max_queue_points);
  EXPECT_DOUBLE_EQ(session_.queue.back().positions[0], 0.3);
  EXPECT_DOUBLE_EQ(session_.queue.back().velocities[0], 0.0);
}

TEST_F(CommandSessionTest, TerminalHoldResumeCanContinueTheSameVerifiedPublication)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(100'000'000), 20'000'000, 1, true, reason_),
    TimelineRecoveryStatus::kSuccess) << reason_;
  const auto first_anchor = session_.queue.back().time_stamp;
  EXPECT_EQ(session_.recoverTimeline(
      1U, stamp(200'000'000), reason_, 0.02, 1),
    TimelineRecoveryStatus::kExhaustedPublishedSuffix);

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(200'000'000), 20'000'000, 1, true, reason_),
    TimelineRecoveryStatus::kSuccess) << reason_;
  EXPECT_GT(session_.queue.back().time_stamp, first_anchor);
  EXPECT_LE(session_.queue.size(), session_.max_queue_points);
  EXPECT_EQ(session_.active_publication_id, publication_id);
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationKeepsVerifiedVirtualHoldUnchanged)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(100'000'000), 20'000'000, 1, true, reason_),
    TimelineRecoveryStatus::kSuccess) << reason_;
  const auto queue_before = session_.queue;
  const auto grid_before = session_.time_grid->periodsUntilCommandLead(
    stamp(200'000'000).nanoseconds(), 20'000'000);

  EXPECT_EQ(session_.continuePublishedBrake(
      stamp(200'000'000), 20'000'000, reason_, 0.2, 1),
    PublishedBrakeContinuationStatus::kContinued);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  EXPECT_EQ(session_.queue.back().time_stamp, queue_before.back().time_stamp);
  EXPECT_TRUE(session_.queue.back().positions.isApprox(queue_before.back().positions));
  EXPECT_TRUE(session_.queue.back().velocities.isApprox(queue_before.back().velocities));
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(200'000'000).nanoseconds(), 20'000'000), grid_before);
  EXPECT_EQ(session_.active_publication_id, publication_id);
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationRejectsCorruptVerifiedVirtualHold)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(100'000'000), 20'000'000, 1, true, reason_),
    TimelineRecoveryStatus::kSuccess) << reason_;
  session_.queue.back().velocities[0] = 0.01;

  EXPECT_NE(session_.continuePublishedBrake(
      stamp(200'000'000), 20'000'000, reason_, 0.2, 1),
    PublishedBrakeContinuationStatus::kContinued);
}

TEST_F(CommandSessionTest, PublishedBrakeContinuationRejectsVirtualHoldWithoutProof)
{
  ASSERT_TRUE(session_.active_publication_id.has_value());
  ASSERT_EQ(session_.resumeVerifiedTerminalHold(
      *session_.active_publication_id, stamp(100'000'000), 20'000'000,
      1, true, reason_), TimelineRecoveryStatus::kSuccess) << reason_;
  session_.verified_terminal_hold_publication_id.reset();

  EXPECT_NE(session_.continuePublishedBrake(
      stamp(200'000'000), 20'000'000, reason_, 0.2, 1),
    PublishedBrakeContinuationStatus::kContinued);
}

TEST_F(CommandSessionTest, LongVerifiedHoldReschedulesWithoutSynthesizingElapsedSamples)
{
  const auto publication_id = session_.active_publication_id;
  ASSERT_TRUE(publication_id.has_value());
  session_.max_queue_points = 8U;

  EXPECT_EQ(session_.resumeVerifiedTerminalHold(
      *publication_id, stamp(10'000'000'000), 20'000'000, 1, true, reason_),
    TimelineRecoveryStatus::kSuccess) << reason_;
  EXPECT_EQ(session_.queue.size(), 1U);
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(10'000'000'000).nanoseconds(), 20'000'000), 0U);
  EXPECT_DOUBLE_EQ(session_.queue.back().positions[0], 0.3);
  EXPECT_DOUBLE_EQ(session_.queue.back().velocities[0], 0.0);
}

TEST_F(CommandSessionTest, RecoveryCannotExceedHardQueueCapacity)
{
  session_.max_queue_points = 1;
  EXPECT_EQ(recover(2), TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  EXPECT_NE(reason_.find("hard bound"), std::string::npos);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(kPeriodNs));
}

TEST_F(CommandSessionTest, ExpiredHistoryPruningMakesRoomForTheNextAppend)
{
  session_.max_queue_points = 13U;
  installPublication(longPublishedQueue(16U));
  ASSERT_EQ(session_.recoverTimeline(
      12U, stamp(70'000'000), reason_, 0.05, 1), TimelineRecoveryStatus::kSuccess) << reason_;
  ASSERT_EQ(session_.queue.size(), 12U);

  const auto fresh_scheduling_time = stamp(105'000'000);
  session_.pruneExpiredHistory(fresh_scheduling_time, 0.05);

  ASSERT_EQ(session_.queue.size(), 8U);
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  for (std::size_t index = 6U; index <= 13U; ++index) {
    const auto & retained = session_.queue[index - 6U];
    const auto & expected = published->execution_queue[index];
    EXPECT_EQ(retained.time_stamp, expected.time_stamp);
    EXPECT_TRUE(retained.positions.isApprox(expected.positions));
    EXPECT_TRUE(retained.velocities.isApprox(expected.velocities));
    EXPECT_TRUE(retained.accelerations.isApprox(expected.accelerations));
  }

  auto next = published->execution_queue[14U];
  EXPECT_GT(next.time_stamp,
    fresh_scheduling_time + rclcpp::Duration::from_nanoseconds(30'000'000));
  session_.queue.push_back(std::move(next));
  EXPECT_EQ(session_.queue.size(), 9U);
  EXPECT_LE(session_.queue.size(), session_.max_queue_points);
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(14 * kPeriodNs));
}

TEST_F(CommandSessionTest, ExpiredHistoryPruningKeepsFuturePointsAndTrueOverflow)
{
  session_.max_queue_points = 13U;
  session_.queue.clear();
  for (std::size_t index = 0; index < session_.max_queue_points; ++index) {
    moveit_servo::KinematicState point(1);
    point.joint_names = {"joint1"};
    point.positions[0] = static_cast<double>(index);
    point.velocities[0] = 0.0;
    point.accelerations[0] = 0.0;
    point.time_stamp = stamp(200'000'000 + static_cast<std::int64_t>(index) * kPeriodNs);
    session_.queue.push_back(std::move(point));
  }
  const auto future_queue = session_.queue;

  session_.pruneExpiredHistory(stamp(100'000'000), 0.02);

  ASSERT_EQ(session_.queue.size(), session_.max_queue_points);
  EXPECT_GT(session_.queue.size() + 1U, session_.max_queue_points);
  for (std::size_t index = 0; index < future_queue.size(); ++index) {
    EXPECT_EQ(session_.queue[index].time_stamp, future_queue[index].time_stamp);
    EXPECT_TRUE(session_.queue[index].positions.isApprox(future_queue[index].positions));
  }
}

TEST_F(CommandSessionTest, DelayedBrakeContinuationRetainsOnlyUnexpiredValidatedPrefix)
{
  session_.max_queue_points = 13U;
  auto transmitted = longPublishedQueue(40U);
  installPublication(std::move(transmitted));
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);

  session_.queue.clear();
  for (std::size_t index = 0U; index <= 6U; ++index) {
    session_.queue.push_back(published->execution_queue[index]);
  }
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue[6U].time_stamp.nanoseconds()));
  const auto required_periods = session_.time_grid->periodsUntilCommandLead(
    stamp(165'000'000).nanoseconds(), 50'000'000);
  ASSERT_EQ(required_periods, 15U);

  const auto status = session_.recoverTimeline(
    *required_periods, stamp(165'000'000), reason_, 0.05, 1,
    TimelineRecoveryPurpose::kPublishedBrakeContinuation);

  ASSERT_EQ(status, TimelineRecoveryStatus::kSuccess) << reason_;
  // Keep only transmitted samples still inside the 50 ms retained window
  // through the new lead anchor. Expired crossed samples must still have been
  // validated before being omitted from the retained queue.
  ASSERT_EQ(session_.queue.size(), 10U);
  const auto expect_exact_point = [this, published](
    const std::size_t actual_index, const std::size_t published_index) {
      const auto & actual = session_.queue[actual_index];
      const auto & expected = published->execution_queue[published_index];
      EXPECT_EQ(actual.time_stamp, expected.time_stamp);
      EXPECT_TRUE(actual.positions.isApprox(expected.positions));
      EXPECT_TRUE(actual.velocities.isApprox(expected.velocities));
      EXPECT_TRUE(actual.accelerations.isApprox(expected.accelerations));
    };
  for (std::size_t actual_index = 0U; actual_index < session_.queue.size(); ++actual_index) {
    expect_exact_point(actual_index, actual_index + 12U);
  }
  EXPECT_EQ(session_.queue.back().time_stamp, stamp(210'000'000));
  EXPECT_EQ(session_.time_grid->commandStamp(
      stamp(165'000'000).nanoseconds(), 50'000'000, 50'000'000),
    stamp(220'000'000).nanoseconds());
  EXPECT_EQ(session_.published_brake_continuation_count, 1U);
  EXPECT_EQ(session_.timeline_recovery_events, 0U);
  EXPECT_EQ(session_.timeline_recovered_periods, 0U);
  EXPECT_LE(session_.queue.size(), session_.max_queue_points);
}

TEST_F(CommandSessionTest, CorruptExpiredCrossedPointRejectsContinuationWithoutMutation)
{
  session_.max_queue_points = 13U;
  auto transmitted = longPublishedQueue(40U);
  installPublication(std::move(transmitted));
  auto * published = const_cast<control::PublishedTrajectoryRecord *>(
    session_.activePublication());
  ASSERT_NE(published, nullptr);
  // Corrupt an already accepted record to exercise validation of an expired
  // crossed point; the publication history correctly rejects this at append.
  published->execution_queue[8U].positions[0] = std::numeric_limits<double>::quiet_NaN();

  session_.queue.clear();
  for (std::size_t index = 0U; index <= 6U; ++index) {
    session_.queue.push_back(published->execution_queue[index]);
  }
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue[6U].time_stamp.nanoseconds()));
  const auto queue_before = session_.queue;
  const auto periods_before = session_.time_grid->periodsUntilCommandLead(
    stamp(165'000'000).nanoseconds(), 50'000'000);
  ASSERT_EQ(periods_before, 15U);

  EXPECT_EQ(session_.recoverTimeline(
      *periods_before, stamp(165'000'000), reason_, 0.05, 1,
      TimelineRecoveryPurpose::kPublishedBrakeContinuation),
    TimelineRecoveryStatus::kInvariantFailure);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  for (std::size_t index = 0U; index < queue_before.size(); ++index) {
    EXPECT_EQ(session_.queue[index].time_stamp, queue_before[index].time_stamp);
    EXPECT_TRUE(session_.queue[index].positions.isApprox(queue_before[index].positions));
    EXPECT_TRUE(session_.queue[index].velocities.isApprox(queue_before[index].velocities));
    EXPECT_TRUE(session_.queue[index].accelerations.isApprox(queue_before[index].accelerations));
  }
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(165'000'000).nanoseconds(), 50'000'000), periods_before);
  EXPECT_EQ(session_.published_brake_continuation_count, 0U);
}

TEST_F(CommandSessionTest, DelayedRecoveryCannotDropFuturePointsToFitHardBound)
{
  session_.max_queue_points = 13U;
  installPublication(longPublishedQueue(40U));
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);

  session_.queue.clear();
  for (std::size_t index = 6U; index <= 18U; ++index) {
    session_.queue.push_back(published->execution_queue[index]);
  }
  ASSERT_TRUE(session_.time_grid->adoptPublishedStamp(
      published->execution_queue[18U].time_stamp.nanoseconds()));
  const auto queue_before = session_.queue;
  const auto periods_before = session_.time_grid->periodsUntilCommandLead(
    stamp(165'000'000).nanoseconds(), 50'000'000);

  EXPECT_EQ(session_.recoverTimeline(
      15U, stamp(165'000'000), reason_, 0.05, 1,
      TimelineRecoveryPurpose::kPublishedBrakeContinuation),
    TimelineRecoveryStatus::kExhaustedPublishedSuffix);
  EXPECT_NE(reason_.find("hard bound"), std::string::npos);
  ASSERT_EQ(session_.queue.size(), queue_before.size());
  for (std::size_t index = 0U; index < queue_before.size(); ++index) {
    EXPECT_EQ(session_.queue[index].time_stamp, queue_before[index].time_stamp);
    EXPECT_TRUE(session_.queue[index].positions.isApprox(queue_before[index].positions));
  }
  EXPECT_EQ(session_.time_grid->periodsUntilCommandLead(
      stamp(165'000'000).nanoseconds(), 50'000'000), periods_before);
  EXPECT_EQ(session_.published_brake_continuation_count, 0U);
}

TEST_F(CommandSessionTest, FullyExpiredCrossedPrefixStillRetainsAndAdoptsAnchor)
{
  installPublication(longPublishedQueue(20U));
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  session_.queue.clear();
  session_.queue.push_back(published->execution_queue[0U]);

  ASSERT_EQ(session_.recoverTimeline(
      4U, stamp(200'000'000), reason_, 0.05, 1,
      TimelineRecoveryPurpose::kPublishedBrakeContinuation),
    TimelineRecoveryStatus::kSuccess) << reason_;

  ASSERT_EQ(session_.queue.size(), 1U);
  EXPECT_EQ(session_.queue.back().time_stamp, published->execution_queue[4U].time_stamp);
  EXPECT_EQ(session_.time_grid->commandStamp(kOriginNs, 0, 0),
    published->execution_queue[4U].time_stamp.nanoseconds() + kPeriodNs);
}

TEST_F(CommandSessionTest, ExpiredOldAnchorDoesNotConsumeCapacityForRetainedRecovery)
{
  session_.max_queue_points = 13U;
  installPublication(longPublishedQueue(20U));
  const auto * published = session_.activePublication();
  ASSERT_NE(published, nullptr);
  session_.queue.clear();
  session_.queue.push_back(published->execution_queue[0U]);

  ASSERT_EQ(session_.recoverTimeline(
      13U, stamp(60'000'000), reason_, 0.05, 1,
      TimelineRecoveryPurpose::kPublishedBrakeContinuation),
    TimelineRecoveryStatus::kSuccess) << reason_;

  ASSERT_EQ(session_.queue.size(), session_.max_queue_points);
  for (std::size_t index = 0U; index < session_.queue.size(); ++index) {
    EXPECT_EQ(session_.queue[index].time_stamp, published->execution_queue[index + 1U].time_stamp);
  }
  EXPECT_EQ(session_.queue.back().time_stamp, published->execution_queue[13U].time_stamp);
  EXPECT_EQ(session_.time_grid->commandStamp(kOriginNs, 0, 0),
    published->execution_queue[13U].time_stamp.nanoseconds() + kPeriodNs);
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
