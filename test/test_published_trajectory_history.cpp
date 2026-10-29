// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <moveit_servo/utils/datatypes.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>

#include "face_tracking_arm/published_trajectory_history.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr std::int64_t kSecondNs = 1'000'000'000;
constexpr std::int64_t kPeriodNs = 10'000'000;
constexpr std::int64_t kPendingOldStartNs = 108'500'000'000;
constexpr std::int64_t kPendingReplacementStartNs = 108'750'000'000;
constexpr std::int64_t kPendingMeasurementNs = 108'708'000'000;
constexpr std::int64_t kPendingHybridHeaderLeadNs = 2'000'000;

void setStamp(builtin_interfaces::msg::Time & stamp, const std::int64_t nanoseconds)
{
  stamp.sec = static_cast<std::int32_t>(nanoseconds / kSecondNs);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % kSecondNs);
}

void setDuration(
  builtin_interfaces::msg::Duration & duration, const std::int64_t nanoseconds)
{
  std::int64_t seconds = nanoseconds / kSecondNs;
  std::int64_t subsecond = nanoseconds % kSecondNs;
  if (subsecond < 0) {
    --seconds;
    subsecond += kSecondNs;
  }
  duration.sec = static_cast<std::int32_t>(seconds);
  duration.nanosec = static_cast<std::uint32_t>(subsecond);
}

std::deque<moveit_servo::KinematicState> makeQueue(
  const std::int64_t start_ns, const double offset = 0.0)
{
  constexpr double positions[] = {0.0, 0.1, 0.2, 0.2, 0.2};
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0; index < std::size(positions); ++index) {
    moveit_servo::KinematicState state(1);
    state.joint_names = {"joint1"};
    state.positions[0] = offset + positions[index];
    state.velocities[0] = index < 2U ? 10.0 : 0.0;
    state.accelerations[0] = 0.0;
    state.time_stamp = rclcpp::Time(
      start_ns + static_cast<std::int64_t>(index) * kPeriodNs, RCL_ROS_TIME);
    queue.push_back(std::move(state));
  }
  return queue;
}

std::deque<moveit_servo::KinematicState> makeReplacementQueue(
  const std::int64_t start_ns, const double start_position, const double start_velocity = 0.0)
{
  constexpr double increments[] = {0.0, 0.1, 0.2, 0.2, 0.2};
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0; index < std::size(increments); ++index) {
    moveit_servo::KinematicState state(1);
    state.joint_names = {"joint1"};
    state.positions[0] = start_position + increments[index];
    state.velocities[0] = index == 0U ? start_velocity : (index < 2U ? 10.0 : 0.0);
    state.accelerations[0] = 0.0;
    state.time_stamp = rclcpp::Time(
      start_ns + static_cast<std::int64_t>(index) * kPeriodNs, RCL_ROS_TIME);
    queue.push_back(std::move(state));
  }
  return queue;
}

std::deque<moveit_servo::KinematicState> makeConstantQueue(
  const std::int64_t start_ns, const double position)
{
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0U; index < 5U; ++index) {
    moveit_servo::KinematicState state(1);
    state.joint_names = {"joint1"};
    state.positions[0] = position;
    state.velocities[0] = 0.0;
    state.accelerations[0] = 0.0;
    state.time_stamp = rclcpp::Time(
      start_ns + static_cast<std::int64_t>(index) * kPeriodNs, RCL_ROS_TIME);
    queue.push_back(std::move(state));
  }
  return queue;
}

std::deque<moveit_servo::KinematicState> makeFutureFiniteQueue(
  const std::int64_t start_ns, const double start_position, const double next_position,
  const double front_velocity = 0.0, const double front_acceleration = 0.0,
  const std::int64_t point_period_ns = kPeriodNs)
{
  std::deque<moveit_servo::KinematicState> queue;
  for (std::size_t index = 0U; index < 5U; ++index) {
    moveit_servo::KinematicState state(1);
    state.joint_names = {"joint1"};
    state.positions[0] = index == 0U ? start_position : next_position;
    state.velocities[0] = index == 0U ? front_velocity :
      (index == 1U ? (next_position - start_position) /
      (static_cast<double>(point_period_ns) * 1.0e-9) : 0.0);
    state.accelerations[0] = index == 0U ? front_acceleration : 0.0;
    state.time_stamp = rclcpp::Time(
      start_ns + static_cast<std::int64_t>(index) * point_period_ns, RCL_ROS_TIME);
    queue.push_back(std::move(state));
  }
  return queue;
}

control_msgs::msg::JointTrajectoryControllerState makeControllerState(
  std::int64_t header_ns, std::int64_t reference_offset_ns,
  std::int64_t feedback_offset_ns, double reference_position, double output_position);

PublicationMatch pendingTerminalMatch(
  PublishedTrajectoryHistory & history, const double replacement_start_position = 0.2,
  const double replacement_start_velocity = 0.0)
{
  PublicationMatch no_match;
  const auto old_id = history.append(
    makeQueue(kPendingOldStartNs),
    rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  if (!old_id.has_value()) {
    return no_match;
  }
  const auto replacement_id = history.append(
    makeReplacementQueue(
      kPendingReplacementStartNs, replacement_start_position, replacement_start_velocity),
    rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  if (!replacement_id.has_value()) {
    return no_match;
  }
  const auto controller_state = makeControllerState(
    kPendingOldStartNs, 60'000'000, 0, 0.2, 0.2);
  return history.match(
    controller_state, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);
}

control_msgs::msg::JointTrajectoryControllerState makeControllerState(
  const std::int64_t header_ns, const std::int64_t reference_offset_ns,
  const std::int64_t feedback_offset_ns, const double reference_position,
  const double output_position)
{
  control_msgs::msg::JointTrajectoryControllerState state;
  setStamp(state.header.stamp, header_ns);
  setDuration(state.reference.time_from_start, reference_offset_ns);
  setDuration(state.feedback.time_from_start, feedback_offset_ns);
  state.reference.positions = {reference_position};
  state.output.positions = {output_position};
  return state;
}

TEST(PublishedTrajectoryHistory, RejectsInvalidCapacityAndRecords)
{
  EXPECT_THROW(PublishedTrajectoryHistory{0}, std::invalid_argument);

  PublishedTrajectoryHistory history{2};
  auto too_short = makeQueue(kSecondNs);
  too_short.resize(2);
  EXPECT_FALSE(history.append(
      std::move(too_short), rclcpp::Time(kSecondNs + kPeriodNs, RCL_ROS_TIME)));

  auto moving_terminal = makeQueue(kSecondNs);
  moving_terminal.back().positions[0] = 0.3;
  EXPECT_FALSE(history.append(
      std::move(moving_terminal), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));

  auto nonfinite = makeQueue(kSecondNs);
  nonfinite.front().positions[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(history.append(
      std::move(nonfinite), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  EXPECT_TRUE(history.empty());
}

TEST(PublishedTrajectoryHistory, KeepsOnlyTheConfiguredNumberOfRecords)
{
  PublishedTrajectoryHistory history{2};
  const auto first = history.append(
    makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  const auto second = history.append(
    makeQueue(2 * kSecondNs),
    rclcpp::Time(2 * kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  const auto third = history.append(
    makeQueue(3 * kSecondNs),
    rclcpp::Time(3 * kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));

  ASSERT_EQ(first, 1U);
  ASSERT_EQ(second, 2U);
  ASSERT_EQ(third, 3U);
  EXPECT_EQ(history.size(), 2U);
  EXPECT_EQ(history.capacity(), 2U);
  EXPECT_EQ(history.find(*first), nullptr);
  ASSERT_NE(history.find(*second), nullptr);
  ASSERT_NE(history.latest(), nullptr);
  EXPECT_EQ(history.latest()->id, *third);

  history.clear();
  EXPECT_TRUE(history.empty());
  EXPECT_EQ(history.latest(), nullptr);
}

TEST(PublishedTrajectoryHistory, CorrectsJtcTrajectoryClockOffset)
{
  const auto state = makeControllerState(
    10 * kSecondNs, 2 * kSecondNs, 2 * kSecondNs + 30'000'000, 0.0, 0.0);

  const auto times = PublishedTrajectoryHistory::correctedSampleTimes(
    state, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs));

  ASSERT_TRUE(times.has_value());
  EXPECT_EQ(times->trajectory_origin.nanoseconds(), 8 * kSecondNs - 30'000'000);
  EXPECT_EQ(times->reference.nanoseconds(), 10 * kSecondNs - 30'000'000);
  EXPECT_EQ(times->output.nanoseconds(), 10 * kSecondNs - 20'000'000);
}

TEST(PublishedTrajectoryHistory, RejectsMalformedControllerTimes)
{
  auto state = makeControllerState(kSecondNs, 0, 0, 0.0, 0.0);
  EXPECT_FALSE(PublishedTrajectoryHistory::correctedSampleTimes(
      state, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(0)));

  state.reference.time_from_start.nanosec = 1'000'000'000;
  EXPECT_FALSE(PublishedTrajectoryHistory::correctedSampleTimes(
      state, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs)));
}

TEST(PublishedTrajectoryHistory, MatchesTheLatestPublication)
{
  PublishedTrajectoryHistory history{4};
  const auto id = history.append(
    makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(id.has_value());
  const auto state = makeControllerState(
    2 * kSecondNs, 15'000'000, kSecondNs, 0.15, 0.2);

  const PublicationMatch match = history.match(
    state, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kLatest);
  EXPECT_EQ(match.publication_id, *id);
  EXPECT_NEAR(match.expected_reference.positions[0], 0.15, 1.0e-12);
  EXPECT_NEAR(match.expected_reference.velocities[0], 10.0, 1.0e-12);
  EXPECT_TRUE(match.expected_output.terminal_hold);
}

TEST(PublishedTrajectoryHistory, HardwareReadUsesItsOwnInterpolatedTrajectoryTime)
{
  PublishedTrajectoryHistory history{4};
  const auto id = history.append(
    makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(id.has_value());
  // Ownership is checked at JTC's own corrected reference/output times.
  const auto jtc = makeControllerState(
    2 * kSecondNs, 15'000'000, kSecondNs, 0.15, 0.2);
  const auto match = history.match(
    jtc, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);
  ASSERT_TRUE(match.matched());

  // The coherent hardware sample was taken earlier. Its expected value is
  // interpolated at its read timestamp, not copied from the newer JTC sample.
  const auto actual_expected = history.expectedAt(
    *id, rclcpp::Time(kSecondNs + 5'000'000, RCL_ROS_TIME));
  ASSERT_TRUE(actual_expected.has_value());
  EXPECT_NEAR(actual_expected->positions[0], 0.05, 1.0e-12);
  EXPECT_NEAR(actual_expected->velocities[0], 10.0, 1.0e-12);
  EXPECT_NE(match.expected_reference.positions[0], actual_expected->positions[0]);
}

TEST(PublishedTrajectoryHistory, NewJtcStampDoesNotValidateAnOldCommandPhase)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  auto stale_phase = makeControllerState(
    3 * kSecondNs, 15'000'000, kSecondNs, 0.15, 0.2);
  const auto match = history.match(
    stale_phase, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);
  EXPECT_FALSE(match.matched());
}

TEST(PublishedTrajectoryHistory, DelayedStateCanMatchAnOlderPublication)
{
  PublishedTrajectoryHistory history{4};
  const auto old_id = history.append(
    makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(old_id.has_value());
  ASSERT_TRUE(history.append(
      makeQueue(2 * kSecondNs, 1.0),
      rclcpp::Time(2 * kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  const auto delayed_state = makeControllerState(
    2 * kSecondNs, 15'000'000, kSecondNs, 0.15, 0.2);

  const PublicationMatch match = history.match(
    delayed_state, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kOlder);
  EXPECT_EQ(match.publication_id, *old_id);
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      match, rclcpp::Time(2 * kSecondNs, RCL_ROS_TIME)));
}

TEST(PublishedTrajectoryHistory, RecognizesOlderHoldBeforeItsStationaryReplacementStarts)
{
  PublishedTrajectoryHistory history{4};
  const PublicationMatch match = pendingTerminalMatch(history);

  ASSERT_TRUE(match.matched());
  ASSERT_EQ(match.kind, PublicationMatchKind::kOlder);
  EXPECT_TRUE(history.isPendingTerminalReplacement(
      match, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));
}

TEST(PublishedTrajectoryHistory, PendingReplacementRejectsAnOlderMovingReference)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kPendingOldStartNs),
      rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  ASSERT_TRUE(history.append(
      makeReplacementQueue(kPendingReplacementStartNs, 0.2),
      rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  const auto moving_reference = makeControllerState(
    kPendingOldStartNs, kPeriodNs, 0, 0.1, 0.2);

  const PublicationMatch match = history.match(
    moving_reference, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  ASSERT_EQ(match.kind, PublicationMatchKind::kOlder);
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      match, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));
}

TEST(PublishedTrajectoryHistory, PendingReplacementRejectsMovingOrDifferentFirstCommand)
{
  PublishedTrajectoryHistory moving_history{4};
  const auto moving_match = pendingTerminalMatch(moving_history, 0.2, 1.0e-6);
  ASSERT_EQ(moving_match.kind, PublicationMatchKind::kOlder);
  EXPECT_FALSE(moving_history.isPendingTerminalReplacement(
      moving_match, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));

  PublishedTrajectoryHistory different_history{4};
  const auto different_match = pendingTerminalMatch(different_history, 0.21);
  ASSERT_EQ(different_match.kind, PublicationMatchKind::kOlder);
  EXPECT_FALSE(different_history.isPendingTerminalReplacement(
      different_match, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));
}

TEST(PublishedTrajectoryHistory, PendingReplacementRequiresMatchingOriginAndRetainedIds)
{
  PublishedTrajectoryHistory history{4};
  auto match = pendingTerminalMatch(history);
  ASSERT_EQ(match.kind, PublicationMatchKind::kOlder);

  auto wrong_origin = match;
  wrong_origin.sample_times.trajectory_origin = rclcpp::Time(
    kPendingOldStartNs + kPeriodNs, RCL_ROS_TIME);
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      wrong_origin, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));

  auto missing_id = match;
  ++missing_id.publication_id;
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      missing_id, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));

  auto latest_match = match;
  latest_match.kind = PublicationMatchKind::kLatest;
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      latest_match, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));
}

TEST(PublishedTrajectoryHistory, PendingReplacementRequiresReferenceOutputAndMeasurementBeforeStart)
{
  PublishedTrajectoryHistory history{4};
  const auto match = pendingTerminalMatch(history);
  ASSERT_EQ(match.kind, PublicationMatchKind::kOlder);
  const rclcpp::Time replacement_start(kPendingReplacementStartNs, RCL_ROS_TIME);

  auto expired_reference = match;
  expired_reference.sample_times.reference = replacement_start;
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      expired_reference, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));

  auto expired_output = match;
  expired_output.sample_times.output = replacement_start;
  EXPECT_FALSE(history.isPendingTerminalReplacement(
      expired_output, rclcpp::Time(kPendingMeasurementNs, RCL_ROS_TIME)));

  EXPECT_FALSE(history.isPendingTerminalReplacement(
      match,
      rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs - 1, RCL_ROS_TIME)));
  EXPECT_FALSE(history.isPendingTerminalReplacement(match, replacement_start));
}

TEST(PublishedTrajectoryHistory, BridgesFutureOriginHoldAndMixedFiniteOutput)
{
  PublishedTrajectoryHistory history{4};
  const auto old_id = history.append(
    makeQueue(kPendingOldStartNs),
    rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(old_id.has_value());
  ASSERT_TRUE(history.append(
      makeConstantQueue(kPendingReplacementStartNs, 0.2),
      rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  const auto latest_id = history.append(
    makeFutureFiniteQueue(kPendingReplacementStartNs, 0.2, 0.21),
    rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(latest_id.has_value());

  const auto before_start = rclcpp::Time(
    kPendingReplacementStartNs - 2 * kPeriodNs, RCL_ROS_TIME);
  const auto expected_measurement = history.expectedAt(
    *latest_id, before_start + rclcpp::Duration::from_nanoseconds(kPeriodNs / 2));
  ASSERT_TRUE(expected_measurement.has_value());
  EXPECT_TRUE(expected_measurement->terminal_hold);
  EXPECT_DOUBLE_EQ(expected_measurement->positions[0], 0.2);
  EXPECT_DOUBLE_EQ(expected_measurement->velocities[0], 0.0);

  auto all_prefront = makeControllerState(
    before_start.nanoseconds(), -2 * kPeriodNs, -2 * kPeriodNs, 0.2, 0.2);
  all_prefront.reference.velocities = {0.0};
  const auto prefront_match = history.match(
    all_prefront, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);
  ASSERT_TRUE(prefront_match.matched());
  EXPECT_EQ(prefront_match.kind, PublicationMatchKind::kLatest);
  EXPECT_EQ(prefront_match.publication_id, *latest_id);
  EXPECT_TRUE(prefront_match.expected_reference.terminal_hold);
  EXPECT_TRUE(prefront_match.expected_output.terminal_hold);

  const auto hybrid_header = rclcpp::Time(
    kPendingReplacementStartNs - kPendingHybridHeaderLeadNs, RCL_ROS_TIME);
  auto hybrid = makeControllerState(
    hybrid_header.nanoseconds(), -kPendingHybridHeaderLeadNs, -kPendingHybridHeaderLeadNs,
    0.2, 0.208);
  hybrid.reference.velocities = {0.0};
  hybrid.output.velocities = {1.0};
  const auto hybrid_match = history.match(
    hybrid, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);
  ASSERT_TRUE(hybrid_match.matched());
  EXPECT_EQ(hybrid_match.kind, PublicationMatchKind::kLatest);
  EXPECT_EQ(hybrid_match.publication_id, *latest_id);
  EXPECT_TRUE(hybrid_match.expected_reference.terminal_hold);
  EXPECT_DOUBLE_EQ(hybrid_match.expected_reference.positions[0], 0.2);
  EXPECT_FALSE(hybrid_match.expected_output.terminal_hold);
  EXPECT_NEAR(hybrid_match.expected_output.positions[0], 0.208, 1.0e-12);
  EXPECT_NEAR(hybrid_match.expected_output.velocities[0], 1.0, 1.0e-12);

  auto wrong_reference = hybrid;
  wrong_reference.reference.positions[0] += 0.01;
  EXPECT_FALSE(history.match(
      wrong_reference, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());
  auto wrong_output = hybrid;
  wrong_output.output.positions[0] += 0.01;
  EXPECT_FALSE(history.match(
      wrong_output, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());

  auto moving_prefront_reference = all_prefront;
  moving_prefront_reference.reference.velocities = {0.1};
  EXPECT_FALSE(history.match(
      moving_prefront_reference, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());
  auto moving_prefront_output = all_prefront;
  moving_prefront_output.output.velocities = {0.1};
  EXPECT_FALSE(history.match(
      moving_prefront_output, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());

  auto wrong_origin = all_prefront;
  setDuration(wrong_origin.reference.time_from_start, -3 * kPeriodNs / 2);
  setDuration(wrong_origin.feedback.time_from_start, -3 * kPeriodNs / 2);
  EXPECT_FALSE(history.match(
      wrong_origin, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());
}

TEST(PublishedTrajectoryHistory, DoesNotBridgeBrokenOrMovingPredecessorChains)
{
  const auto before_start = rclcpp::Time(
    kPendingReplacementStartNs - kPeriodNs, RCL_ROS_TIME);
  for (int broken_field = 0; broken_field < 4; ++broken_field) {
    PublishedTrajectoryHistory history{4};
    ASSERT_TRUE(history.append(
        makeQueue(kPendingOldStartNs),
        rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
    std::int64_t middle_start = kPendingReplacementStartNs;
    double front_position = 0.2;
    double front_velocity = 0.0;
    double front_acceleration = 0.0;
    if (broken_field == 0) {
      middle_start += kPeriodNs;
    } else if (broken_field == 1) {
      front_position += 0.01;
    } else if (broken_field == 2) {
      front_velocity = 0.1;
    } else {
      front_acceleration = 0.1;
    }
    ASSERT_TRUE(history.append(
        makeFutureFiniteQueue(
          middle_start, front_position, 0.22, front_velocity, front_acceleration),
        rclcpp::Time(middle_start + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
    const auto latest_id = history.append(
      makeFutureFiniteQueue(kPendingReplacementStartNs, 0.2, 0.21),
      rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
    ASSERT_TRUE(latest_id.has_value());
    EXPECT_FALSE(history.expectedAt(*latest_id, before_start).has_value())
      << "broken future link field " << broken_field;
  }

  PublishedTrajectoryHistory evicted_root{2};
  ASSERT_TRUE(evicted_root.append(
      makeQueue(kPendingOldStartNs),
      rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  ASSERT_TRUE(evicted_root.append(
      makeConstantQueue(kPendingReplacementStartNs, 0.2),
      rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  const auto latest_id = evicted_root.append(
    makeFutureFiniteQueue(kPendingReplacementStartNs, 0.2, 0.21),
    rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(latest_id.has_value());
  EXPECT_EQ(evicted_root.find(*latest_id - 2U), nullptr);
  EXPECT_FALSE(evicted_root.expectedAt(*latest_id, before_start).has_value());

  PublishedTrajectoryHistory moving_root{4};
  const auto moving_root_id = moving_root.append(
    makeQueue(kPendingOldStartNs),
    rclcpp::Time(kPendingOldStartNs + 3 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(moving_root_id.has_value());
  ASSERT_TRUE(moving_root.append(
      makeConstantQueue(kPendingReplacementStartNs, 0.2),
      rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  const auto moving_latest_id = moving_root.append(
    makeFutureFiniteQueue(kPendingReplacementStartNs, 0.2, 0.21),
    rclcpp::Time(kPendingReplacementStartNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(moving_latest_id.has_value());
  const rclcpp::Time before_stationary(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME);
  const auto still_moving = moving_root.expectedAt(*moving_root_id, before_stationary);
  ASSERT_TRUE(still_moving.has_value());
  EXPECT_FALSE(still_moving->terminal_hold);
  EXPECT_GT(std::abs(still_moving->velocities[0]), 0.0);
  EXPECT_FALSE(moving_root.expectedAt(*moving_latest_id, before_stationary).has_value());
}

TEST(PublishedTrajectoryHistory, HybridOutputCannotExtrapolateBeyondFiniteReplacement)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kPendingOldStartNs),
      rclcpp::Time(kPendingOldStartNs + 2 * kPeriodNs, RCL_ROS_TIME)).has_value());
  ASSERT_TRUE(history.append(
      makeConstantQueue(kPendingReplacementStartNs, 0.2),
      rclcpp::Time(kPendingReplacementStartNs + kPeriodNs, RCL_ROS_TIME)).has_value());
  ASSERT_TRUE(history.append(
      makeFutureFiniteQueue(kPendingReplacementStartNs, 0.2, 0.21, 0.0, 0.0, 1'000'000),
      rclcpp::Time(kPendingReplacementStartNs + 2'000'000, RCL_ROS_TIME)).has_value());

  const auto header = rclcpp::Time(
    kPendingReplacementStartNs - kPendingHybridHeaderLeadNs, RCL_ROS_TIME);
  auto hybrid = makeControllerState(
    header.nanoseconds(), -kPendingHybridHeaderLeadNs, -kPendingHybridHeaderLeadNs,
    0.2, 0.21);
  hybrid.reference.velocities = {0.0};
  hybrid.output.velocities = {0.0};
  EXPECT_FALSE(history.match(
      hybrid, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());
}

TEST(PublishedTrajectoryHistory, LatestTerminalHoldExtendsPastTransmittedEnd)
{
  PublishedTrajectoryHistory history{4};
  const auto publication_id = history.append(
    makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME));
  ASSERT_TRUE(publication_id.has_value());
  const auto held_state = makeControllerState(
    2 * kSecondNs, 100'000'000, kSecondNs, 0.2, 0.2);

  const PublicationMatch match = history.match(
    held_state, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kLatest);
  EXPECT_TRUE(match.expected_reference.terminal_hold);
  EXPECT_TRUE(match.expected_output.terminal_hold);
  EXPECT_DOUBLE_EQ(match.expected_reference.velocities[0], 0.0);

  const auto terminal_prefix = history.expectedAt(
    *publication_id, rclcpp::Time(2 * kSecondNs + 100'000'000, RCL_ROS_TIME));
  ASSERT_TRUE(terminal_prefix.has_value());
  EXPECT_TRUE(terminal_prefix->terminal_hold);
  EXPECT_DOUBLE_EQ(terminal_prefix->positions[0], 0.2);
  EXPECT_DOUBLE_EQ(terminal_prefix->velocities[0], 0.0);

  const auto mismatched_hold = makeControllerState(
    2 * kSecondNs, 100'000'000, kSecondNs, 0.21, 0.21);
  EXPECT_FALSE(history.match(
      mismatched_hold, RCL_ROS_TIME,
      rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9).matched());
}

TEST(PublishedTrajectoryHistory, ExactOriginAllowsSupersededTerminalHold)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  ASSERT_TRUE(history.append(
      makeQueue(2 * kSecondNs, 1.0),
      rclcpp::Time(2 * kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  const auto stale_hold = makeControllerState(
    2 * kSecondNs, 100'000'000, kSecondNs, 0.2, 0.2);

  const PublicationMatch match = history.match(
    stale_hold, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kOlder);
}

TEST(PublishedTrajectoryHistory, WrongOriginCannotImpersonateLatestTerminalHold)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  const auto stale_hold_with_wrong_origin = makeControllerState(
    3 * kSecondNs, 100'000'000, kSecondNs, 0.2, 0.2);

  const PublicationMatch match = history.match(
    stale_hold_with_wrong_origin, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  EXPECT_FALSE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kNone);
}

TEST(PublishedTrajectoryHistory, SharedExplicitPrefixCanMatchAcrossReplacementOrigins)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  const auto shared_prefix_state = makeControllerState(
    2 * kSecondNs, 15'000'000, kSecondNs + kPeriodNs, 0.05, 0.15);

  const PublicationMatch match = history.match(
    shared_prefix_state, RCL_ROS_TIME,
    rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-9);

  ASSERT_TRUE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kLatest);
  EXPECT_NE(match.sample_times.trajectory_origin.nanoseconds(), kSecondNs);
}

TEST(PublishedTrajectoryHistory, PositionMismatchReturnsNone)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
  const auto state = makeControllerState(
    2 * kSecondNs, 15'000'000, kSecondNs, 0.15, 0.25);

  const PublicationMatch match = history.match(
    state, RCL_ROS_TIME, rclcpp::Duration::from_nanoseconds(kPeriodNs), 1.0e-4);

  EXPECT_FALSE(match.matched());
  EXPECT_EQ(match.kind, PublicationMatchKind::kNone);
}

}  // namespace
}  // namespace face_tracking_arm::control
