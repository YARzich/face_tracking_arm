// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <joint_trajectory_controller/interpolation_methods.hpp>
#include <joint_trajectory_controller/trajectory.hpp>
#include <moveit_servo/moveit_servo_lib_parameters.hpp>
#include <moveit_servo/utils/common.hpp>
#include <rclcpp/time.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "face_tracking_arm/emergency_brake_tail.hpp"

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

TEST(JointTrajectoryInterpolation, MissedReplacementKeepsCheckedPrefixAndTailInBounds)
{
  constexpr double lower_bound = 0.0;
  constexpr double upper_bound = 0.8;
  constexpr std::int64_t sample_step_ns = 1'000'000;
  constexpr double period_sec = 0.01;
  constexpr double source_position = 0.7991;
  constexpr double candidate_velocity = 0.002;

  control::JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(1, lower_bound);
  limits.upper_position = Eigen::VectorXd::Constant(1, upper_bound);
  limits.position_margin = Eigen::VectorXd::Zero(1);
  limits.max_velocity = Eigen::VectorXd::Constant(1, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(1, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(1, 20.0);
  limits.numerical_policy = control::NumericalPolicy{
    control::Radians{1.0e-12}, control::RadiansPerSecond{1.0e-12},
    control::RadiansPerSecondSquared{1.0e-12},
    control::RadiansPerSecondCubed{1.0e-12}, control::SolverResidual{1.0e-6}};

  control::JointMotionState candidate;
  candidate.position = Eigen::VectorXd::Constant(
    1, source_position + candidate_velocity * period_sec);
  candidate.velocity = Eigen::VectorXd::Constant(1, candidate_velocity);
  candidate.acceleration = Eigen::VectorXd::Constant(
    1, (candidate_velocity - 0.0) / period_sec);
  const control::EmergencyBrakeTail tail =
    control::EmergencyBrakeTailGenerator().generate(candidate, limits);
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  ASSERT_GE(tail.points.size(), 3U);

  const auto make_message = [](const std::vector<double> & positions) {
      auto message = std::make_shared<trajectory_msgs::msg::JointTrajectory>();
      message->header.stamp.sec = 1;
      message->joint_names = {"joint1"};
      for (std::size_t index = 0; index < positions.size(); ++index) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {positions[index]};
        point.time_from_start.nanosec = static_cast<std::uint32_t>(
          static_cast<std::int64_t>(index) * kPeriodNs);
        message->points.push_back(std::move(point));
      }
      return message;
    };

  // The old publication has a validated source prefix and its checked stationary suffix.
  std::vector<double> old_positions{source_position, source_position};
  std::vector<double> replacement_positions = old_positions;
  for (const auto & point : tail.points) {
    ASSERT_EQ(point.position.size(), 1);
    EXPECT_GE(point.position[0], lower_bound);
    EXPECT_LE(point.position[0], upper_bound);
    old_positions.push_back(source_position);
    replacement_positions.push_back(point.position[0]);
  }
  auto old_message = make_message(old_positions);
  auto replacement_message = make_message(replacement_positions);
  Trajectory old_publication(old_message);
  Trajectory replacement(replacement_message);
  trajectory_msgs::msg::JointTrajectoryPoint preceding_point;
  preceding_point.positions = {source_position};
  const rclcpp::Time preceding_time(kHeaderTimeNs - kPeriodNs, RCL_ROS_TIME);
  old_publication.set_point_before_trajectory_msg(preceding_time, preceding_point);
  replacement.set_point_before_trajectory_msg(preceding_time, preceding_point);

  // The replacement preserves the queued prefix exactly. If it is missed, the controller
  // keeps following the previously checked hold; if received, the actual generated braking
  // suffix remains in bounds. Position-only JTC interpolation is linear between these points.
  ASSERT_EQ(old_message->points.size(), replacement_message->points.size());
  for (std::size_t index = 0; index < 2U; ++index) {
    EXPECT_DOUBLE_EQ(
      old_message->points[index].positions[0], replacement_message->points[index].positions[0]);
  }
  const auto trajectory_end_ns = static_cast<std::int64_t>(
    old_message->points.size() - 1U) * kPeriodNs;
  for (std::int64_t offset_ns = 0; offset_ns <= trajectory_end_ns;
    offset_ns += sample_step_ns)
  {
    const auto old_sample = sample(
      old_publication, kHeaderTimeNs + offset_ns,
      InterpolationMethod::VARIABLE_DEGREE_SPLINE);
    const auto replacement_sample = sample(
      replacement, kHeaderTimeNs + offset_ns,
      InterpolationMethod::VARIABLE_DEGREE_SPLINE);
    ASSERT_EQ(old_sample.positions.size(), 1U);
    ASSERT_EQ(replacement_sample.positions.size(), 1U);
    EXPECT_GE(old_sample.positions[0], lower_bound);
    EXPECT_LE(old_sample.positions[0], upper_bound);
    EXPECT_GE(replacement_sample.positions[0], lower_bound);
    EXPECT_LE(replacement_sample.positions[0], upper_bound);
  }

  const auto missed_replacement_sample = sample(
    old_publication, kHeaderTimeNs + 15'000'000,
    InterpolationMethod::VARIABLE_DEGREE_SPLINE);
  EXPECT_DOUBLE_EQ(missed_replacement_sample.positions[0], source_position);
}

TEST(JointTrajectoryInterpolation, NoneModeWouldSkipToTheFollowingPoint)
{
  Trajectory trajectory = make_trajectory();

  const auto at_start = sample(
    trajectory, kHeaderTimeNs, InterpolationMethod::NONE);
  ASSERT_EQ(at_start.positions.size(), 1U);
  EXPECT_NEAR(at_start.positions[0], 0.11, 1.0e-12);
}

TEST(JointTrajectoryInterpolation, FloatSampleDifferencesStayInsideRoundoffEnvelope)
{
  Trajectory trajectory = make_trajectory();
  constexpr std::int64_t kControllerSampleNs = 1'000'000;
  constexpr double kControllerSampleSec =
    static_cast<double>(kControllerSampleNs) * 1.0e-9;
  std::vector<double> nominal;
  std::vector<double> wire;
  double maximum_half_ulp = 0.0;
  for (std::int64_t offset_ns = 0; offset_ns <= kPeriodNs;
    offset_ns += kControllerSampleNs)
  {
    const double value = sample(
      trajectory, kHeaderTimeNs + offset_ns,
      InterpolationMethod::VARIABLE_DEGREE_SPLINE).positions.at(0);
    const float float_value = static_cast<float>(value);
    const float lower = std::nextafter(float_value,
      -std::numeric_limits<float>::infinity());
    const float upper = std::nextafter(float_value,
      std::numeric_limits<float>::infinity());
    const double half_ulp = 0.5 * std::max(
      static_cast<double>(float_value) - static_cast<double>(lower),
      static_cast<double>(upper) - static_cast<double>(float_value));
    maximum_half_ulp = std::max(maximum_half_ulp, half_ulp);
    nominal.push_back(value);
    wire.push_back(static_cast<double>(float_value));
    EXPECT_LE(std::abs(wire.back() - nominal.back()), maximum_half_ulp);
  }

  std::vector<double> nominal_velocity, wire_velocity;
  for (std::size_t index = 1; index < nominal.size(); ++index) {
    nominal_velocity.push_back((nominal[index] - nominal[index - 1U]) /
      kControllerSampleSec);
    wire_velocity.push_back((wire[index] - wire[index - 1U]) /
      kControllerSampleSec);
    EXPECT_LE(std::abs(wire_velocity.back() - nominal_velocity.back()),
      2.0 * maximum_half_ulp / kControllerSampleSec + 1.0e-12);
  }

  std::vector<double> nominal_acceleration, wire_acceleration;
  for (std::size_t index = 1; index < nominal_velocity.size(); ++index) {
    nominal_acceleration.push_back((nominal_velocity[index] - nominal_velocity[index - 1U]) /
      kControllerSampleSec);
    wire_acceleration.push_back((wire_velocity[index] - wire_velocity[index - 1U]) /
      kControllerSampleSec);
    EXPECT_LE(std::abs(wire_acceleration.back() - nominal_acceleration.back()),
      4.0 * maximum_half_ulp / std::pow(kControllerSampleSec, 2) + 1.0e-9);
  }
  for (std::size_t index = 1; index < nominal_acceleration.size(); ++index) {
    const double nominal_jerk =
      (nominal_acceleration[index] - nominal_acceleration[index - 1U]) /
      kControllerSampleSec;
    const double wire_jerk =
      (wire_acceleration[index] - wire_acceleration[index - 1U]) /
      kControllerSampleSec;
    EXPECT_LE(std::abs(wire_jerk - nominal_jerk),
      8.0 * maximum_half_ulp / std::pow(kControllerSampleSec, 3) + 1.0e-6);
  }
}

TEST(JointTrajectoryInterpolation, ZeroStampedStationaryResetSamplesMeasuredHoldAtTimeZero)
{
  auto message = std::make_shared<trajectory_msgs::msg::JointTrajectory>();
  message->joint_names = {"joint1"};
  trajectory_msgs::msg::JointTrajectoryPoint measured_hold;
  measured_hold.positions = {0.118631};
  measured_hold.velocities = {0.0};
  measured_hold.accelerations = {0.0};
  message->points = {measured_hold};
  Trajectory trajectory(message);
  trajectory_msgs::msg::JointTrajectoryPoint old_desired_state;
  old_desired_state.positions = {0.0};
  old_desired_state.velocities = {0.0};
  old_desired_state.accelerations = {0.0};
  trajectory.set_point_before_trajectory_msg(
    rclcpp::Time(kHeaderTimeNs - kPeriodNs, RCL_ROS_TIME), old_desired_state);

  const auto at_zero = sample(
    trajectory, kHeaderTimeNs, InterpolationMethod::VARIABLE_DEGREE_SPLINE);
  const auto next_update = sample(
    trajectory, kHeaderTimeNs + kPeriodNs, InterpolationMethod::VARIABLE_DEGREE_SPLINE);

  ASSERT_EQ(at_zero.positions.size(), 1U);
  EXPECT_NEAR(at_zero.positions[0], 0.118631, 1.0e-12);
  EXPECT_NEAR(at_zero.velocities[0], 0.0, 1.0e-12);
  ASSERT_EQ(next_update.positions.size(), 1U);
  EXPECT_NEAR(next_update.positions[0], 0.118631, 1.0e-12);
  EXPECT_NEAR(next_update.velocities[0], 0.0, 1.0e-12);
}

TEST(JointTrajectoryInterpolation, FutureDatedStationaryResetInterpolatesFromOldDesiredState)
{
  auto message = std::make_shared<trajectory_msgs::msg::JointTrajectory>();
  message->header.stamp.sec = 1;
  message->header.stamp.nanosec = static_cast<std::uint32_t>(3 * kPeriodNs);
  message->joint_names = {"joint1"};
  trajectory_msgs::msg::JointTrajectoryPoint measured_hold;
  measured_hold.positions = {0.118631};
  measured_hold.velocities = {0.0};
  measured_hold.accelerations = {0.0};
  message->points = {measured_hold};
  Trajectory trajectory(message);
  trajectory_msgs::msg::JointTrajectoryPoint old_desired_state;
  old_desired_state.positions = {0.0};
  old_desired_state.velocities = {0.0};
  old_desired_state.accelerations = {0.0};
  trajectory.set_point_before_trajectory_msg(
    rclcpp::Time(kHeaderTimeNs - kPeriodNs, RCL_ROS_TIME), old_desired_state);

  const auto before_start = sample(
    trajectory, kHeaderTimeNs + kPeriodNs, InterpolationMethod::VARIABLE_DEGREE_SPLINE);

  ASSERT_EQ(before_start.positions.size(), 1U);
  EXPECT_GT(before_start.positions[0], 0.0);
  EXPECT_LT(before_start.positions[0], 0.118631);
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
