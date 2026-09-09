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

void setStamp(builtin_interfaces::msg::Time & stamp, const std::int64_t nanoseconds)
{
  stamp.sec = static_cast<std::int32_t>(nanoseconds / kSecondNs);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % kSecondNs);
}

void setDuration(
  builtin_interfaces::msg::Duration & duration, const std::int64_t nanoseconds)
{
  duration.sec = static_cast<std::int32_t>(nanoseconds / kSecondNs);
  duration.nanosec = static_cast<std::uint32_t>(nanoseconds % kSecondNs);
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
}

TEST(PublishedTrajectoryHistory, LatestTerminalHoldExtendsPastTransmittedEnd)
{
  PublishedTrajectoryHistory history{4};
  ASSERT_TRUE(history.append(
      makeQueue(kSecondNs), rclcpp::Time(kSecondNs + 2 * kPeriodNs, RCL_ROS_TIME)));
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
