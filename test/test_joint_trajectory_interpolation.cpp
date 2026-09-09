// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

#include <joint_trajectory_controller/interpolation_methods.hpp>
#include <joint_trajectory_controller/trajectory.hpp>
#include <moveit_servo/moveit_servo_lib_parameters.hpp>
#include <moveit_servo/utils/common.hpp>
#include <rclcpp/time.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace face_tracking_arm
{
namespace
{

using joint_trajectory_controller::Trajectory;
using joint_trajectory_controller::TrajectoryPointConstIter;
using joint_trajectory_controller::interpolation_methods::InterpolationMethod;

constexpr std::int64_t kHeaderTimeNs = 1'000'000'000;
constexpr std::int64_t kPeriodNs = 10'000'000;

std::shared_ptr<trajectory_msgs::msg::JointTrajectory> make_position_only_trajectory()
{
  auto trajectory = std::make_shared<trajectory_msgs::msg::JointTrajectory>();
  trajectory->header.stamp.sec = 1;
  trajectory->joint_names = {"joint1"};
  trajectory->points.resize(2);
  trajectory->points[0].positions = {0.10};
  trajectory->points[0].time_from_start.nanosec = 0;
  trajectory->points[1].positions = {0.11};
  trajectory->points[1].time_from_start.nanosec =
    static_cast<std::uint32_t>(kPeriodNs);
  return trajectory;
}

Trajectory make_trajectory()
{
  Trajectory trajectory(make_position_only_trajectory());
  trajectory_msgs::msg::JointTrajectoryPoint preceding_point;
  preceding_point.positions = {0.09};
  trajectory.set_point_before_trajectory_msg(
    rclcpp::Time(kHeaderTimeNs - kPeriodNs, RCL_ROS_TIME), preceding_point);
  return trajectory;
}

trajectory_msgs::msg::JointTrajectoryPoint sample(
  Trajectory & trajectory, const std::int64_t time_ns,
  const InterpolationMethod interpolation_method)
{
  trajectory_msgs::msg::JointTrajectoryPoint output;
  TrajectoryPointConstIter start;
  TrajectoryPointConstIter end;
  EXPECT_TRUE(trajectory.sample(
      rclcpp::Time(time_ns, RCL_ROS_TIME), interpolation_method, output, start, end,
      false));
  return output;
}

TEST(JointTrajectoryInterpolation, PositionOnlySplinesDoNotSkipTheFirstPoint)
{
  Trajectory trajectory = make_trajectory();

  const auto at_start = sample(
    trajectory, kHeaderTimeNs, InterpolationMethod::VARIABLE_DEGREE_SPLINE);
  ASSERT_EQ(at_start.positions.size(), 1U);
  EXPECT_NEAR(at_start.positions[0], 0.10, 1.0e-12);

  const auto at_midpoint = sample(
    trajectory, kHeaderTimeNs + kPeriodNs / 2,
    InterpolationMethod::VARIABLE_DEGREE_SPLINE);
  ASSERT_EQ(at_midpoint.positions.size(), 1U);
  EXPECT_NEAR(at_midpoint.positions[0], 0.105, 1.0e-12);

  const auto at_end = sample(
    trajectory, kHeaderTimeNs + kPeriodNs,
    InterpolationMethod::VARIABLE_DEGREE_SPLINE);
  ASSERT_EQ(at_end.positions.size(), 1U);
  EXPECT_NEAR(at_end.positions[0], 0.11, 1.0e-12);
}

TEST(JointTrajectoryInterpolation, NoneModeWouldSkipToTheFollowingPoint)
{
  Trajectory trajectory = make_trajectory();

  const auto at_start = sample(
    trajectory, kHeaderTimeNs, InterpolationMethod::NONE);
  ASSERT_EQ(at_start.positions.size(), 1U);
  EXPECT_NEAR(at_start.positions[0], 0.11, 1.0e-12);
}

TEST(JointTrajectoryInterpolation, MatchingControllerGridPreservesDiscreteAcceleration)
{
  constexpr double acceleration_limit = 2.0;
  const auto maximum_sampled_acceleration = [](const std::int64_t command_period_ns) {
      auto message = std::make_shared<trajectory_msgs::msg::JointTrajectory>();
      message->header.stamp.sec = 1;
      message->joint_names = {"joint1"};
      double position = 0.0;
      const double command_period = static_cast<double>(command_period_ns) * 1.0e-9;
      for (int index = 0; index < 12; ++index) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {position};
        point.time_from_start.nanosec =
          static_cast<std::uint32_t>(index * command_period_ns);
        message->points.push_back(point);
        const double next_velocity =
          acceleration_limit * static_cast<double>(index + 1) * command_period;
        position += next_velocity * command_period;
      }
      Trajectory trajectory(message);
      trajectory.set_point_before_trajectory_msg(
        rclcpp::Time(kHeaderTimeNs - kPeriodNs, RCL_ROS_TIME), message->points.front());
      double maximum = 0.0;
      // Sample away from the knots to also cover an offset controller phase.
      auto previous = sample(
        trajectory, kHeaderTimeNs + command_period_ns + kPeriodNs / 4,
        InterpolationMethod::VARIABLE_DEGREE_SPLINE);
      for (std::int64_t offset = command_period_ns + kPeriodNs;
        offset < 10 * command_period_ns; offset += kPeriodNs)
      {
        const auto current = sample(
          trajectory, kHeaderTimeNs + offset + kPeriodNs / 4,
          InterpolationMethod::VARIABLE_DEGREE_SPLINE);
        maximum = std::max(
          maximum, std::abs(current.velocities[0] - previous.velocities[0]) /
          (static_cast<double>(kPeriodNs) * 1.0e-9));
        previous = current;
      }
      return maximum;
    };

  // A 50 Hz position-only stream produces alternating zero/double increments
  // when sampled by a 100 Hz JTC. The 100 Hz stream preserves the discrete limit.
  EXPECT_NEAR(maximum_sampled_acceleration(2 * kPeriodNs), 4.0, 1.0e-10);
  EXPECT_NEAR(maximum_sampled_acceleration(kPeriodNs), acceleration_limit, 1.0e-10);
}

TEST(JointTrajectoryInterpolation, ComposerWithholdsOnlyTheFinalBrakeSentinel)
{
  servo::Params parameters;
  parameters.publish_joint_positions = true;
  parameters.publish_joint_velocities = false;
  parameters.publish_joint_accelerations = false;

  std::deque<moveit_servo::KinematicState> execution_queue;
  for (int index = 0; index < 5; ++index) {
    moveit_servo::KinematicState state(1);
    state.joint_names = {"joint1"};
    state.positions[0] = index < 2 ? 0.01 * static_cast<double>(index) : 0.02;
    state.velocities[0] = index < 2 ? 1.0 : 0.0;
    state.accelerations[0] = 0.0;
    state.time_stamp = rclcpp::Time(
      kHeaderTimeNs + static_cast<std::int64_t>(index) * kPeriodNs,
      RCL_ROS_TIME);
    execution_queue.push_back(std::move(state));
  }

  const auto trajectory = moveit_servo::composeTrajectoryMessage(
    parameters, execution_queue);

  ASSERT_TRUE(trajectory.has_value());
  ASSERT_EQ(trajectory->points.size(), execution_queue.size() - 1U);
  ASSERT_EQ(trajectory->points[2].positions.size(), 1U);
  ASSERT_EQ(trajectory->points[3].positions.size(), 1U);
  EXPECT_DOUBLE_EQ(trajectory->points[2].positions[0], 0.02);
  EXPECT_DOUBLE_EQ(trajectory->points[3].positions[0], 0.02);
  EXPECT_TRUE(trajectory->points[2].velocities.empty());
  EXPECT_TRUE(trajectory->points[3].accelerations.empty());
}

}  // namespace
}  // namespace face_tracking_arm
