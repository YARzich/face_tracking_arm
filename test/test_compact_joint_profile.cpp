// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "face_tracking_arm/compact_joint_profile.hpp"
#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

JointMotionLimits make_limits(const Eigen::Index count)
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(count, -2.0);
  limits.upper_position = Eigen::VectorXd::Constant(count, 2.0);
  limits.position_margin = Eigen::VectorXd::Zero(count);
  limits.max_velocity = Eigen::VectorXd::Constant(count, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(count, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(count, 20.0);
  return limits;
}

JointMotionState make_start(const std::initializer_list<double> position)
{
  JointMotionState start;
  start.position = Eigen::Map<const Eigen::VectorXd>(position.begin(), position.size());
  start.velocity = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(position.size()));
  start.acceleration = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(position.size()));
  return start;
}

JointMotionLimits make_runtime_compact_limits(
  const double maximum_velocity = 3.0,
  const double maximum_acceleration = 8.0,
  const double maximum_jerk = 80.0)
{
  constexpr double kTwoPi = 6.28318530717958647692;
  constexpr double kPi = 3.14159265358979323846;
  JointMotionLimits limits;
  limits.lower_position = (Eigen::Matrix<double, 6, 1>() <<
    -kTwoPi, -2.059, -3.927, -kTwoPi, -1.69297, -kTwoPi).finished();
  limits.upper_position = (Eigen::Matrix<double, 6, 1>() <<
    kTwoPi, 2.0944, 0.19198, kTwoPi, kPi, kTwoPi).finished();
  limits.position_margin = Eigen::VectorXd::Zero(6);
  limits.max_velocity = Eigen::VectorXd::Constant(6, maximum_velocity);
  limits.max_acceleration = Eigen::VectorXd::Constant(6, maximum_acceleration);
  limits.max_jerk = Eigen::VectorXd::Constant(6, maximum_jerk);
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-kTwoPi, kTwoPi}, maximum_velocity,
    maximum_acceleration, maximum_jerk, 0.01,
    SolverResidual{1.0e-5});
  EXPECT_TRUE(policy.has_value());
  if (policy) {
    limits.numerical_policy = *policy;
  }
  return limits;
}

JointMotionState make_recorded_large_offset_start()
{
  JointMotionState start;
  start.position = (Eigen::Matrix<double, 6, 1>() <<
    6.242596651391844, 0.2532663046125725, -1.668817503160594,
    -0.01405052604479681, -0.7974477179101805, 6.255262952338858).finished();
  start.velocity = (Eigen::Matrix<double, 6, 1>() <<
    1.0654971746012344e-9, -0.01613889988632024, 0.010754807278748283,
    -0.40864940291490287, -0.0018078756099597529, 0.3392258070261897).finished();
  start.acceleration = (Eigen::Matrix<double, 6, 1>() <<
    1.0660273304713289e-7, -0.28626433472655566, 0.34114516316999305,
    0.33892018939357116, -0.18082025931285553, -1.9305812561488445).finished();
  return start;
}

Eigen::VectorXd recorded_first_attached_waypoint()
{
  return (Eigen::Matrix<double, 6, 1>() <<
         6.236689477739856, 0.24402370531497092, -1.688822621075801,
         0.047994572916991156, -0.7565968624191373, 6.197831295812177).finished();
}

void expect_discrete_contract(
  const CompactJointProfile & profile,
  const JointMotionLimits & limits,
  const double period_sec,
  const double limit_scale = 1.0)
{
  ASSERT_FALSE(profile.samples.empty());
  double observed_duration = 0.0;
  const Eigen::Index count = profile.samples.front().position.size();
  ASSERT_GT(count, 0);
  for (std::size_t sample = 1U; sample < profile.samples.size(); ++sample) {
    const auto & previous = profile.samples[sample - 1U];
    const auto & current = profile.samples[sample];
    EXPECT_EQ(current.position.size(), count);
    ASSERT_EQ(current.velocity.size(), count);
    ASSERT_EQ(current.acceleration.size(), count);
    const Eigen::VectorXd expected_acceleration =
      (current.velocity - previous.velocity) / period_sec;
    const Eigen::VectorXd jerk = (current.acceleration - previous.acceleration) / period_sec;
    EXPECT_NEAR((current.position - previous.position - current.velocity * period_sec).norm(),
      0.0, 1.0e-9);
    EXPECT_NEAR((current.acceleration - expected_acceleration).norm(), 0.0, 1.0e-9);
    EXPECT_LE((current.position - limits.upper_position).maxCoeff(), 1.0e-9);
    EXPECT_LE((limits.lower_position - current.position).maxCoeff(), 1.0e-9);
    EXPECT_LE((current.velocity.cwiseAbs().array() -
      (limits.max_velocity * limit_scale).array()).maxCoeff(), 1.0e-6);
    EXPECT_LE((current.acceleration.cwiseAbs().array() -
      (limits.max_acceleration * limit_scale * limit_scale).array()).maxCoeff(), 1.0e-6);
    EXPECT_LE((jerk.cwiseAbs().array() -
      (limits.max_jerk * limit_scale * limit_scale * limit_scale).array()).maxCoeff(), 1.0e-5);
    observed_duration += period_sec;
  }
  EXPECT_NEAR(profile.duration_sec, observed_duration, 1.0e-9);
}

std::size_t countInteriorNearStopEpisodes(const CompactJointProfile & profile);

void record_profile_metrics(
  const std::string & name, const CompactJointProfile & profile,
  const double wall_duration_ms, const double period_sec)
{
  double max_velocity = 0.0;
  double max_acceleration = 0.0;
  double max_jerk = 0.0;
  for (std::size_t index = 1U; index < profile.samples.size(); ++index) {
    const auto & previous = profile.samples[index - 1U];
    const auto & current = profile.samples[index];
    max_velocity = std::max(max_velocity, current.velocity.cwiseAbs().maxCoeff());
    max_acceleration = std::max(
      max_acceleration, current.acceleration.cwiseAbs().maxCoeff());
    max_jerk = std::max(max_jerk,
        ((current.acceleration - previous.acceleration) / period_sec).cwiseAbs().maxCoeff());
  }
  std::cout << "[PROFILE_METRIC] route=" << name
            << " builder=" << profile.builder_mode
            << " duration_s=" << profile.duration_sec
            << " wall_ms=" << wall_duration_ms
            << " geometry_prepare_ms=" << profile.geometry_prepare_duration_sec * 1000.0
            << " profile_generation_ms=" << profile.profile_generation_duration_sec * 1000.0
            << " path_length_rad=" << profile.path_length_rad
            << " interior_nearstop_episodes=" << countInteriorNearStopEpisodes(profile)
            << " max_v=" << max_velocity << " max_a=" << max_acceleration
            << " max_j=" << max_jerk;
  if (!profile.builder_fallback_reason.empty()) {
    std::cout << " fallback_reason=" << profile.builder_fallback_reason;
  }
  std::cout << '\n';
}

std::vector<Eigen::VectorXd> readPPathFixture()
{
  std::ifstream input(FACE_TRACKING_ARM_P_PATH_R1_FIXTURE);
  if (!input) {
    throw std::runtime_error("cannot open P-PATH R1 fixture");
  }
  std::string line;
  std::getline(input, line);
  std::vector<Eigen::VectorXd> path;
  while (std::getline(input, line)) {
    std::stringstream row(line);
    std::string value;
    std::getline(row, value, ',');  // fixture index
    Eigen::VectorXd point(6);
    for (Eigen::Index joint = 0; joint < point.size(); ++joint) {
      if (!std::getline(row, value, ',')) {
        throw std::runtime_error("malformed P-PATH R1 fixture");
      }
      point[joint] = std::stod(value);
    }
    path.push_back(std::move(point));
  }
  return path;
}

JointMotionLimits makePPathLimits()
{
  constexpr double kTwoPi = 6.28318530717958647692;
  constexpr double kPi = 3.14159265358979323846;
  JointMotionLimits limits;
  limits.lower_position = (Eigen::Matrix<double, 6, 1>() <<
    -kTwoPi, -2.059488, -3.92699, -kTwoPi, -1.692969, -kTwoPi).finished();
  limits.upper_position = (Eigen::Matrix<double, 6, 1>() <<
    kTwoPi, 2.094395, 0.191986, kTwoPi, kPi, kTwoPi).finished();
  limits.position_margin = Eigen::VectorXd::Zero(6);
  limits.max_velocity = Eigen::VectorXd::Constant(6, 3.1);
  limits.max_acceleration = Eigen::VectorXd::Constant(6, 19.0);
  limits.max_jerk = Eigen::VectorXd::Constant(6, 400.0);
  return limits;
}

std::size_t countInteriorNearStopEpisodes(const CompactJointProfile & profile)
{
  constexpr double kNearStopRadPerSec = 0.01;
  if (profile.samples.size() < 3U) {
    return 0U;
  }
  const auto near_stop = [](const JointMotionState & sample) {
      return sample.velocity.cwiseAbs().maxCoeff() < kNearStopRadPerSec;
    };
  std::size_t episodes = 0U;
  std::size_t index = 1U;
  while (index + 1U < profile.samples.size()) {
    if (!near_stop(profile.samples[index])) {
      ++index;
      continue;
    }
    const std::size_t run_start = index;
    while (index + 1U < profile.samples.size() && near_stop(profile.samples[index])) {
      ++index;
    }
    const bool bounded_by_motion = run_start > 1U &&
      index + 1U < profile.samples.size() &&
      !near_stop(profile.samples[run_start - 1U]) && !near_stop(profile.samples[index]);
    if (bounded_by_motion) {
      ++episodes;
    }
  }
  return episodes;
}

void expectBrakeTailAvailableFromEverySample(
  const std::string & route_name,
  const CompactJointProfile & profile,
  JointMotionLimits limits,
  const double period_sec)
{
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-6.28318530717958647692, 6.28318530717958647692},
    3.1, 19.0, 400.0, period_sec, SolverResidual{1.0e-5});
  ASSERT_TRUE(policy.has_value());
  limits.numerical_policy = *policy;

  const auto started = std::chrono::steady_clock::now();
  std::size_t failure_count = 0U;
  std::ostringstream first_failure;
  for (std::size_t index = 0U; index < profile.samples.size(); ++index) {
    const auto & sample = profile.samples[index];
    Eigen::VectorXd direction = sample.velocity;
    if (direction.isZero(0.0)) {
      direction = sample.acceleration;
    }
    const auto tail = generateCompactProfileBrakeTail(
      sample, direction, limits, EmergencyBrakeTailConfig{period_sec});
    if (tail.status != EmergencyBrakeTailStatus::kSuccess) {
      ++failure_count;
      if (failure_count == 1U) {
        first_failure << "index=" << index << " status=" << static_cast<int>(tail.status)
                      << " reason=" << tail.failure_reason << " q=" << sample.position.transpose()
                      << " v=" << sample.velocity.transpose()
                      << " a=" << sample.acceleration.transpose();
      }
    }
  }
  const double wall_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  std::cout << "[BRAKE_SCAN] route=" << route_name
            << " samples=" << profile.samples.size()
            << " failures=" << failure_count
            << " wall_ms=" << wall_ms;
  if (failure_count > 0U) {
    std::cout << " first_failure={" << first_failure.str() << "}";
  }
  std::cout << '\n';
  EXPECT_EQ(failure_count, 0U) << route_name << ": " << first_failure.str();
}

std::vector<Eigen::VectorXd> makePPathArc(const std::size_t segment_count)
{
  Eigen::VectorXd start(6), goal(6);
  start << 6.1883686799358966, 0.3239841022272904, -1.9537049418177228,
    2.1038388719789514, 1.2617313829869721, -1.6409001317067762;
  goal << 0.0, -0.2715944, -2.8699983, 0.0, -1.151386, -3.14159265358979323846;
  const double delta_a = goal[2] - start[2];
  const double delta_b = goal[3] - start[3];
  const double chord = std::hypot(delta_a, delta_b);
  const double radius = chord * 0.5;
  const double chord_a = delta_a / chord;
  const double chord_b = delta_b / chord;
  const double normal_a = -chord_b;
  const double normal_b = chord_a;
  const double center_a = 0.5 * (start[2] + goal[2]);
  const double center_b = 0.5 * (start[3] + goal[3]);
  std::vector<Eigen::VectorXd> path;
  path.reserve(segment_count + 1U);
  for (std::size_t index = 0U; index <= segment_count; ++index) {
    const double t = static_cast<double>(index) / static_cast<double>(segment_count);
    const double theta = std::acos(-1.0) * t;
    Eigen::VectorXd point = start + t * (goal - start);
    point[2] = center_a - radius * std::cos(theta) * chord_a +
      radius * std::sin(theta) * normal_a;
    point[3] = center_b - radius * std::cos(theta) * chord_b +
      radius * std::sin(theta) * normal_b;
    path.push_back(std::move(point));
  }
  path.front() = start;
  path.back() = goal;
  return path;
}

TEST(CompactJointProfile, PPathR1ProjectedCurveHasNoInteriorNearStops)
{
  constexpr double kPeriod = 0.01;
  const auto path = readPPathFixture();
  ASSERT_EQ(path.size(), 944U);
  const auto limits = makePPathLimits();
  const auto start = make_start({
        path.front()[0], path.front()[1], path.front()[2],
        path.front()[3], path.front()[4], path.front()[5]});

  std::string failure_reason;
  const auto build_started = std::chrono::steady_clock::now();
  const auto profile = buildCompactJointProfile(path, start, limits, kPeriod, 1.0, {},
    &failure_reason);
  const double wall_duration_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - build_started).count();

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  record_profile_metrics("R1", *profile, wall_duration_ms, kPeriod);
  expectBrakeTailAvailableFromEverySample("R1", *profile, limits, kPeriod);
  EXPECT_EQ(profile->checked_path_vertices.size() - 1U, 360U);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U)
    << "near-stops must be computed from the sampled joint velocities; "
       "whole-profile start/end stops are excluded";
  EXPECT_NEAR((profile->samples.front().position - path.front()).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR((profile->samples.back().position - path.back()).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.front().velocity.norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(profile->samples.front().acceleration.norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(profile->samples.back().velocity.norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(profile->samples.back().acceleration.norm(), 0.0, 1.0e-12);
  expect_discrete_contract(*profile, limits, kPeriod);
}

TEST(CompactJointProfile, PublishesProgressAndScaleMetadataForSampleGeometry)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0});
  const auto limits = make_limits(1);
  const std::vector<Eigen::VectorXd> path{
    start.position, Eigen::VectorXd::Constant(1, 0.05), Eigen::VectorXd::Constant(1, 0.1)};
  std::string failure_reason;
  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 0.5, {0.5, 0.8}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  ASSERT_EQ(profile->path_progress_rad_per_sample.size(), profile->samples.size());
  ASSERT_EQ(profile->speed_scale_per_interval.size() + 1U, profile->samples.size());
  EXPECT_DOUBLE_EQ(profile->path_progress_rad_per_sample.front(), 0.0);
  EXPECT_NEAR(profile->path_progress_rad_per_sample.back(), profile->path_length_rad, 1.0e-12);
  for (std::size_t index = 1U; index < profile->path_progress_rad_per_sample.size(); ++index) {
    EXPECT_GE(profile->path_progress_rad_per_sample[index],
      profile->path_progress_rad_per_sample[index - 1U]);
  }
  EXPECT_DOUBLE_EQ(profile->speed_scale_per_interval.front(), 0.5);
  EXPECT_DOUBLE_EQ(profile->speed_scale_per_interval.back(), 0.8);
  const std::size_t sample_index = static_cast<std::size_t>(std::distance(
    profile->samples.begin(), std::max_element(profile->samples.begin(), profile->samples.end(),
      [](const JointMotionState & left, const JointMotionState & right) {
        return left.velocity.norm() < right.velocity.norm();
      })));
  const auto projected = projectCompactProfileProgress(
    *profile, profile->samples[sample_index].position,
    profile->samples[sample_index].velocity,
    profile->path_progress_rad_per_sample[sample_index],
    profile->path_progress_rad_per_sample[sample_index], 1.0e-12);
  ASSERT_TRUE(projected.has_value());
  EXPECT_NEAR(*projected, profile->path_progress_rad_per_sample[sample_index], 1.0e-12);
}

TEST(CompactJointProfile, PPathR2ArcDensificationDoesNotCreateInteriorNearStops)
{
  constexpr double kPeriod = 0.01;
  const auto limits = makePPathLimits();
  for (const std::size_t segment_count : {10U, 50U, 100U, 200U}) {
    SCOPED_TRACE("arc segment count=" + std::to_string(segment_count));
    const auto path = makePPathArc(segment_count);
    const auto start = make_start({
          path.front()[0], path.front()[1], path.front()[2],
          path.front()[3], path.front()[4], path.front()[5]});
    const auto build_started = std::chrono::steady_clock::now();
    const auto profile = buildCompactJointProfile(path, start, limits, kPeriod);
    const double wall_duration_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - build_started).count();

    ASSERT_TRUE(profile.has_value());
    record_profile_metrics("R2_" + std::to_string(segment_count),
      *profile, wall_duration_ms, kPeriod);
    expectBrakeTailAvailableFromEverySample(
      "R2_" + std::to_string(segment_count), *profile, limits, kPeriod);
    EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
    EXPECT_NEAR(profile->samples.front().velocity.norm(), 0.0, 1.0e-12);
    EXPECT_NEAR(profile->samples.front().acceleration.norm(), 0.0, 1.0e-12);
    EXPECT_NEAR(profile->samples.back().velocity.norm(), 0.0, 1.0e-12);
    EXPECT_NEAR(profile->samples.back().acceleration.norm(), 0.0, 1.0e-12);
    expect_discrete_contract(*profile, limits, kPeriod);
  }
}

TEST(CompactJointProfile, StraightPathControlHasOnlyWholeProfileEndpointStops)
{
  constexpr double kPeriod = 0.01;
  const auto limits = makePPathLimits();
  Eigen::VectorXd start_position(6), goal(6);
  start_position << 6.1883686799358966, 0.3239841022272904, -1.9537049418177228,
    2.1038388719789514, 1.2617313829869721, -1.6409001317067762;
  goal << 0.0, -0.2715944, -2.8699983, 0.0, -1.151386, -3.14159265358979323846;
  std::vector<Eigen::VectorXd> path{start_position, goal};
  const auto start = make_start({
        start_position[0], start_position[1], start_position[2],
        start_position[3], start_position[4], start_position[5]});

  const auto build_started = std::chrono::steady_clock::now();
  const auto profile = buildCompactJointProfile(path, start, limits, kPeriod);
  const double wall_duration_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - build_started).count();

  ASSERT_TRUE(profile.has_value());
  record_profile_metrics("straight_control", *profile, wall_duration_ms, kPeriod);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
  expect_discrete_contract(*profile, limits, kPeriod);
}

TEST(CompactJointProfile, DirectStagedRoutePreservesPathAndBrakeability)
{
  constexpr double kPeriod = 0.01;
  constexpr double kPi = 3.14159265358979323846;
  const JointMotionState start = make_start({
        6.1275, -1.3259, -0.4535, -1.8243, -1.0063, 1.9440});
  Eigen::VectorXd goal(6);
  goal << 0.0, -0.2715944, -2.8699983, 0.0, -1.151386, kPi;
  Eigen::VectorXd staged_prefix = goal;
  staged_prefix[0] = start.position[0] + std::clamp(
    goal[0] - start.position[0], -0.35, 0.35);
  staged_prefix[3] = start.position[3];
  const std::vector<Eigen::VectorXd> path{start.position, staged_prefix, goal};
  const auto limits = make_runtime_compact_limits(3.1, 19.0, 400.0);
  ASSERT_TRUE(limits.numerical_policy.has_value());

  std::string failure_reason;
  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {1.0, 1.0}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  ASSERT_EQ(profile->builder_mode.rfind("totg_phase_local_clock", 0U), 0U)
    << profile->builder_mode;
  record_profile_metrics("direct_staged_route", *profile,
    (profile->geometry_prepare_duration_sec + profile->profile_generation_duration_sec) * 1000.0,
    kPeriod);
  ASSERT_EQ(profile->checked_path_vertices.size(), path.size());
  for (std::size_t index = 0U; index < path.size(); ++index) {
    EXPECT_TRUE(profile->checked_path_vertices[index].isApprox(path[index], 1.0e-12));
  }
  ASSERT_EQ(profile->checked_path_segment_scales.size(), 2U);
  EXPECT_DOUBLE_EQ(profile->checked_path_segment_scales[0], 1.0);
  EXPECT_DOUBLE_EQ(profile->checked_path_segment_scales[1], 1.0);
  EXPECT_LE(profile->duration_sec, 4.0);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
  EXPECT_EQ(profile->samples.back().position.size(), goal.size());
  EXPECT_TRUE((profile->samples.back().position.array() == goal.array()).all());
  EXPECT_TRUE(profile->samples.back().velocity.isZero(0.0));
  EXPECT_TRUE(profile->samples.back().acceleration.isZero(0.0));
  expect_discrete_contract(*profile, limits, kPeriod);
  expectBrakeTailAvailableFromEverySample(
    "direct_staged_route", *profile, limits, kPeriod);
}

TEST(CompactJointProfile, DirectStagedRouteUsesComputedJerkReserve)
{
  constexpr double kPeriod = 0.01;
  constexpr double kPi = 3.14159265358979323846;
  const JointMotionState start = make_start({
        6.1275, -1.3259, -0.4535, -1.8243, -1.0063, 1.9440});
  Eigen::VectorXd goal(6);
  goal << 0.0, -0.2715944, -2.8699983, 0.0, -1.151386, kPi;
  Eigen::VectorXd staged_prefix = goal;
  staged_prefix[0] = start.position[0] + std::clamp(
    goal[0] - start.position[0], -0.35, 0.35);
  staged_prefix[3] = start.position[3];
  const std::vector<Eigen::VectorXd> path{start.position, staged_prefix, goal};
  auto limits = make_runtime_compact_limits(3.1, 19.0, 420.0);
  ASSERT_TRUE(limits.numerical_policy.has_value());

  std::string failure_reason;
  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {1.0, 1.0}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  ASSERT_EQ(profile->builder_mode.rfind("totg_phase_local_clock", 0U), 0U)
    << profile->builder_mode;
  EXPECT_LE(profile->duration_sec, 4.0);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
  expect_discrete_contract(*profile, limits, kPeriod);
  expectBrakeTailAvailableFromEverySample(
    "direct_staged_route_j420", *profile, limits, kPeriod);
}

TEST(CompactJointProfile, ClosedRouteNeverUsesStraightScalarShortcut)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.0, 0.1).finished(),
    start.position};
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, make_limits(2), kPeriod, 1.0, {}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  EXPECT_EQ(profile->builder_mode.rfind("totg_time_warp", 0U), 0U)
    << profile->builder_mode;
  EXPECT_NE(profile->builder_fallback_reason.find("queue_limit:jerk:axis="),
    std::string::npos)
    << profile->builder_fallback_reason;
  EXPECT_NE(profile->builder_fallback_reason.find(":actual="), std::string::npos)
    << profile->builder_fallback_reason;
  EXPECT_NE(profile->builder_fallback_reason.find(":limit="), std::string::npos)
    << profile->builder_fallback_reason;
  expect_discrete_contract(*profile, make_limits(2), kPeriod);
}

TEST(CompactJointProfile, SmallErrorAtRestSettlesOnOneBoundedProfile)
{
  constexpr double kPeriod = 0.01;
  constexpr double kGoal = 0.0;
  const auto start = make_start({0.01});
  const std::vector<Eigen::VectorXd> path{
    start.position, Eigen::VectorXd::Constant(1, kGoal)};

  const auto profile = buildCompactJointProfile(
    path, start, make_limits(1), kPeriod);

  ASSERT_TRUE(profile.has_value());
  ASSERT_GE(profile->samples.size(), 2U);
  EXPECT_LE(profile->duration_sec, 5.0);
  EXPECT_NEAR(profile->duration_sec,
    kPeriod * static_cast<double>(profile->samples.size() - 1U), 1.0e-9);
  EXPECT_NEAR(profile->samples.front().position[0], start.position[0], 1.0e-12);
  EXPECT_NEAR(profile->samples.front().velocity[0], 0.0, 1.0e-12);
  EXPECT_NEAR(profile->samples.front().acceleration[0], 0.0, 1.0e-12);
  EXPECT_NEAR(profile->samples.back().position[0], kGoal, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().velocity[0], 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().acceleration[0], 0.0, 1.0e-6);

  const auto limits = make_limits(1);
  expect_discrete_contract(*profile, limits, kPeriod);
  double max_abs_velocity = 0.0;
  double max_abs_acceleration = 0.0;
  double max_abs_jerk = 0.0;
  for (std::size_t index = 1U; index < profile->samples.size(); ++index) {
    const auto & previous = profile->samples[index - 1U];
    const auto & current = profile->samples[index];
    EXPECT_NEAR(current.position[0],
      previous.position[0] + current.velocity[0] * kPeriod, 1.0e-9);
    EXPECT_LE(current.position[0], previous.position[0] + 1.0e-9);
    EXPECT_GE(current.position[0], kGoal - 1.0e-9);
    max_abs_velocity = std::max(max_abs_velocity, std::abs(current.velocity[0]));
    max_abs_acceleration = std::max(
      max_abs_acceleration, std::abs(current.acceleration[0]));
    max_abs_jerk = std::max(max_abs_jerk,
      std::abs(current.acceleration[0] - previous.acceleration[0]) / kPeriod);
  }
  EXPECT_LE(max_abs_velocity, limits.max_velocity[0] + 1.0e-6);
  EXPECT_LE(max_abs_acceleration, limits.max_acceleration[0] + 1.0e-6);
  EXPECT_LE(max_abs_jerk, limits.max_jerk[0] + 1.0e-5);
}

TEST(CompactJointProfile, AlreadyAtGoalWithZeroMotionProducesStationaryHold)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.4});
  const std::vector<Eigen::VectorXd> path{start.position, start.position};
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, make_limits(1), kPeriod, 1.0, {}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  ASSERT_GE(profile->samples.size(), 2U);
  EXPECT_NEAR(profile->duration_sec,
    kPeriod * static_cast<double>(profile->samples.size() - 1U), 1.0e-12);
  for (const auto & sample : profile->samples) {
    EXPECT_NEAR(sample.position[0], start.position[0], 1.0e-12);
    EXPECT_DOUBLE_EQ(sample.velocity[0], 0.0);
    EXPECT_DOUBLE_EQ(sample.acceleration[0], 0.0);
  }
}

TEST(CompactJointProfile, SplicesPublishedPrefixExactlyBeforeCompactContinuation)
{
  constexpr double kPeriod = 0.01;
  const auto compact_limits = make_limits(1);
  auto ordinary_prefix_limits = compact_limits;
  ordinary_prefix_limits.max_velocity.setConstant(0.5);
  ordinary_prefix_limits.max_acceleration.setConstant(1.0);
  ordinary_prefix_limits.max_jerk.setConstant(10.0);

  std::vector<JointMotionState> published_prefix;
  const auto add = [&published_prefix](
    const double q, const double v, const double a) {
      JointMotionState sample;
      sample.position = Eigen::VectorXd::Constant(1, q);
      sample.velocity = Eigen::VectorXd::Constant(1, v);
      sample.acceleration = Eigen::VectorXd::Constant(1, a);
      published_prefix.push_back(std::move(sample));
    };
  add(0.0, 0.002, 0.0);
  add(0.00001, 0.001, -0.1);
  add(0.00001, 0.0, -0.1);
  add(0.00001, 0.0, 0.0);

  auto continuation_start = published_prefix.back();
  const Eigen::VectorXd goal = Eigen::VectorXd::Constant(1, 0.8);
  const std::vector<Eigen::VectorXd> path{continuation_start.position, goal};
  auto continuation = buildCompactJointProfile(
    path, continuation_start, compact_limits, kPeriod);
  ASSERT_TRUE(continuation);
  const auto continuation_before_splice = *continuation;
  std::string failure_reason;

  ASSERT_TRUE(prependValidatedCompactPrefix(
      *continuation, published_prefix, kPeriod, &failure_reason)) << failure_reason;
  ASSERT_EQ(continuation->prefix_endpoint_sample_index, published_prefix.size() - 1U);
  ASSERT_EQ(continuation->samples.size(), published_prefix.size() +
    continuation_before_splice.samples.size() - 1U);
  EXPECT_NEAR(continuation->duration_sec,
    continuation_before_splice.duration_sec +
    kPeriod * static_cast<double>(published_prefix.size() - 1U), 1.0e-12);
  EXPECT_NEAR(continuation->path_length_rad,
    continuation_before_splice.path_length_rad + 0.00001, 1.0e-12);

  double prefix_max_velocity = 0.0;
  for (std::size_t index = 0U; index < published_prefix.size(); ++index) {
    const auto & actual = continuation->samples[index];
    const auto & expected = published_prefix[index];
    EXPECT_TRUE(actual.position.isApprox(expected.position, 0.0));
    EXPECT_TRUE(actual.velocity.isApprox(expected.velocity, 0.0));
    EXPECT_TRUE(actual.acceleration.isApprox(expected.acceleration, 0.0));
    prefix_max_velocity = std::max(prefix_max_velocity, actual.velocity.cwiseAbs().maxCoeff());
  }
  EXPECT_LE(prefix_max_velocity, ordinary_prefix_limits.max_velocity.maxCoeff());
  ASSERT_EQ(
    continuation->samples[published_prefix.size() - 1U].velocity.norm(), 0.0);
  ASSERT_EQ(
    continuation->samples[published_prefix.size() - 1U].acceleration.norm(), 0.0);

  double continuation_max_velocity = 0.0;
  for (std::size_t index = published_prefix.size(); index < continuation->samples.size(); ++index) {
    const auto & sample = continuation->samples[index];
    continuation_max_velocity = std::max(
      continuation_max_velocity, sample.velocity.cwiseAbs().maxCoeff());
    for (Eigen::Index joint = 0; joint < sample.velocity.size(); ++joint) {
      EXPECT_LE(std::abs(sample.velocity[joint]), compact_limits.max_velocity[joint] + 1.0e-6);
    }
  }
  EXPECT_GT(continuation_max_velocity, ordinary_prefix_limits.max_velocity.maxCoeff());
  EXPECT_LE(continuation_max_velocity, compact_limits.max_velocity.maxCoeff() + 1.0e-6);
  expect_discrete_contract(*continuation, compact_limits, kPeriod);

  auto invalid_prefix = published_prefix;
  invalid_prefix.back().position[0] += 1.0e-6;
  auto preserved = continuation_before_splice;
  EXPECT_FALSE(prependValidatedCompactPrefix(
      preserved, invalid_prefix, kPeriod, &failure_reason));
  ASSERT_EQ(preserved.samples.size(), continuation_before_splice.samples.size());
  for (std::size_t index = 0U; index < preserved.samples.size(); ++index) {
    EXPECT_TRUE(preserved.samples[index].position.isApprox(
        continuation_before_splice.samples[index].position, 0.0));
    EXPECT_TRUE(preserved.samples[index].velocity.isApprox(
        continuation_before_splice.samples[index].velocity, 0.0));
    EXPECT_TRUE(preserved.samples[index].acceleration.isApprox(
        continuation_before_splice.samples[index].acceleration, 0.0));
  }
}

TEST(CompactJointProfile, PreservesAlignedNonzeroStartVelocityAndAcceleration)
{
  constexpr double kPeriod = 0.01;
  auto start = make_start({0.01});
  start.velocity[0] = -0.04;
  start.acceleration[0] = -0.20;
  const std::vector<Eigen::VectorXd> path{
    start.position, Eigen::VectorXd::Constant(1, 0.0)};

  const auto profile = buildCompactJointProfile(path, start, make_limits(1), kPeriod);

  ASSERT_TRUE(profile.has_value());
  ASSERT_GE(profile->samples.size(), 2U);
  EXPECT_DOUBLE_EQ(profile->samples.front().position[0], start.position[0]);
  EXPECT_DOUBLE_EQ(profile->samples.front().velocity[0], start.velocity[0]);
  EXPECT_DOUBLE_EQ(profile->samples.front().acceleration[0], start.acceleration[0]);
  const double first_discrete_jerk =
    (profile->samples[1].acceleration[0] - start.acceleration[0]) / kPeriod;
  EXPECT_LE(std::abs(first_discrete_jerk), 20.0 + 1.0e-5);
  EXPECT_NEAR(profile->samples.back().position[0], 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().velocity[0], 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().acceleration[0], 0.0, 1.0e-6);
  expect_discrete_contract(*profile, make_limits(1), kPeriod);
}

TEST(CompactJointProfile, RecordedNonparallelStartRequiresValidatedPrefix)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_recorded_large_offset_start();
  const Eigen::VectorXd first_attached_waypoint = recorded_first_attached_waypoint();
  const std::vector<Eigen::VectorXd> path{start.position, first_attached_waypoint};
  auto limits = make_runtime_compact_limits();
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {1.0}, &failure_reason);

  EXPECT_FALSE(profile.has_value());
  EXPECT_FALSE(failure_reason.empty());
}

TEST(CompactJointProfile, RecordedLargeOffsetSimplifiedRouteStaysInsideLimits)
{
  constexpr double kPeriod = 0.01;
  const Eigen::VectorXd first_attached_waypoint = recorded_first_attached_waypoint();
  const Eigen::VectorXd long_turn_endpoint = (Eigen::Matrix<double, 6, 1>() <<
    5.892596650735637, -0.27159440000000007, -2.8699983,
    0.047994572916991156, 1.5707963267948966, 6.263185307179587).finished();
  const Eigen::VectorXd goal = (Eigen::Matrix<double, 6, 1>() <<
    0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966,
    6.263185307179587).finished();
  // The validated brake prefix ends at first_attached_waypoint. Continue the
  // captured route from that stationary state through the fold and long turn.
  const auto start = make_start({
        first_attached_waypoint[0], first_attached_waypoint[1],
        first_attached_waypoint[2], first_attached_waypoint[3],
        first_attached_waypoint[4], first_attached_waypoint[5]});
  // These are the exact remaining points retained by the production
  // collinearity simplifier from the captured route.
  const std::vector<Eigen::VectorXd> path{
    start.position, long_turn_endpoint, goal};
  struct ProfileCase
  {
    const char * name;
    double max_velocity;
    double max_acceleration;
    double max_jerk;
    double speed_scale;
    std::vector<double> phase_scales;
  };
  const std::vector<ProfileCase> cases{
    {"nominal", 3.0, 8.0, 80.0, 1.0, {1.0, 1.0}},
    {"baseline_caps", 1.0, 2.0, 20.0, 1.0, {1.0, 1.0}},
    {"common_half", 3.0, 8.0, 80.0, 0.5, {1.0, 1.0}},
    {"fold_half", 3.0, 8.0, 80.0, 1.0, {0.5, 1.0}},
    {"turn_half", 3.0, 8.0, 80.0, 1.0, {1.0, 0.5}},
    {"common_and_phase_half", 3.0, 8.0, 80.0, 0.5, {0.5, 0.5}},
  };

  for (const auto & profile_case : cases) {
    SCOPED_TRACE(profile_case.name);
    const auto limits = make_runtime_compact_limits(
      profile_case.max_velocity, profile_case.max_acceleration, profile_case.max_jerk);
    std::string failure_reason;
    const auto profile = buildCompactJointProfile(
      path, start, limits, kPeriod, profile_case.speed_scale,
      profile_case.phase_scales, &failure_reason);

    ASSERT_TRUE(profile.has_value()) << failure_reason;
    ASSERT_GE(profile->samples.size(), 2U);
    EXPECT_NEAR((profile->samples.front().position - start.position).norm(), 0.0, 1.0e-15);
    EXPECT_NEAR((profile->samples.front().velocity - start.velocity).norm(), 0.0, 1.0e-15);
    EXPECT_NEAR(
      (profile->samples.front().acceleration - start.acceleration).norm(), 0.0, 1.0e-15);
    EXPECT_NEAR((profile->samples.back().position - goal).norm(), 0.0, 1.0e-6);
    EXPECT_NEAR(profile->samples.back().velocity.norm(), 0.0, 1.0e-6);
    EXPECT_NEAR(profile->samples.back().acceleration.norm(), 0.0, 1.0e-6);
    expect_discrete_contract(*profile, limits, kPeriod);
  }
}

TEST(CompactJointProfile, AlignedVelocityAwayFromGoalTurnsAroundSmoothly)
{
  constexpr double kPeriod = 0.01;
  auto start = make_start({0.01});
  start.velocity[0] = 0.04;
  start.acceleration[0] = 0.20;
  const std::vector<Eigen::VectorXd> path{
    start.position, Eigen::VectorXd::Constant(1, 0.0)};

  const auto profile = buildCompactJointProfile(path, start, make_limits(1), kPeriod);

  ASSERT_TRUE(profile.has_value());
  ASSERT_GE(profile->samples.size(), 3U);
  EXPECT_DOUBLE_EQ(profile->samples.front().position[0], start.position[0]);
  EXPECT_DOUBLE_EQ(profile->samples.front().velocity[0], start.velocity[0]);
  EXPECT_DOUBLE_EQ(profile->samples.front().acceleration[0], start.acceleration[0]);
  const auto away_sample = std::find_if(profile->samples.begin() + 1,
      profile->samples.end(), [&](const JointMotionState & sample) {
        return sample.position[0] > start.position[0] + 1.0e-6;
      });
  ASSERT_NE(away_sample, profile->samples.end());
  const auto toward_sample = std::find_if(away_sample + 1,
      profile->samples.end(), [&](const JointMotionState & sample) {
        return sample.position[0] < away_sample->position[0] - 1.0e-6;
      });
  ASSERT_NE(toward_sample, profile->samples.end());
  EXPECT_NEAR(profile->samples.back().position[0], 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().velocity[0], 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().acceleration[0], 0.0, 1.0e-6);
  expect_discrete_contract(*profile, make_limits(1), kPeriod);
}

TEST(CompactJointProfile, NonparallelStartOnShortFirstLegRequiresValidatedPrefix)
{
  constexpr double kPeriod = 0.01;
  auto start = make_start({0.0, 0.0});
  start.velocity[1] = 0.05;
  start.acceleration[1] = 0.20;
  const Eigen::VectorXd short_leg_end = (Eigen::Vector2d() << 0.01, 0.0).finished();
  const Eigen::VectorXd goal = (Eigen::Vector2d() << 0.01, 0.05).finished();
  const std::vector<Eigen::VectorXd> path{start.position, short_leg_end, goal};
  const auto limits = make_limits(2);

  const auto profile = buildCompactJointProfile(path, start, limits, kPeriod);

  EXPECT_FALSE(profile.has_value());
}

TEST(CompactJointProfile, PassesCornerWithoutStoppingAndKeepsDerivativesBounded)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const Eigen::VectorXd corner = (Eigen::Vector2d() << 0.01, 0.0).finished();
  const Eigen::VectorXd goal = (Eigen::Vector2d() << 0.01, 0.01).finished();
  const std::vector<Eigen::VectorXd> path{start.position, corner, goal};
  const auto build_started = std::chrono::steady_clock::now();
  std::string failure_reason;
  const auto profile = buildCompactJointProfile(
    path, start, make_limits(2), kPeriod, 1.0, {}, &failure_reason);
  const double wall_duration_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - build_started).count();

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  const auto profile_without_diagnostics = buildCompactJointProfile(
    path, start, make_limits(2), kPeriod);
  ASSERT_TRUE(profile_without_diagnostics.has_value());
  EXPECT_DOUBLE_EQ(profile_without_diagnostics->duration_sec, profile->duration_sec);
  EXPECT_NE(profile->builder_mode.find("_uniform_dilated"), std::string::npos)
    << profile->builder_mode;
  record_profile_metrics("direct_corner", *profile, wall_duration_ms, kPeriod);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
  EXPECT_NEAR((profile->samples.back().position - goal).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().velocity.norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().acceleration.norm(), 0.0, 1.0e-6);
  expect_discrete_contract(*profile, make_limits(2), kPeriod);
}

TEST(CompactJointProfile, PhaseScaleChangesDurationWithoutExceedingPhysicalCaps)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  const auto limits = make_limits(2);

  std::string slow_failure;
  const auto slower = buildCompactJointProfile(
    path, start, limits, kPeriod, 0.5, {}, &slow_failure);
  const auto nominal = buildCompactJointProfile(path, start, limits, kPeriod, 1.0);
  const auto faster = buildCompactJointProfile(path, start, limits, kPeriod, 2.0);
  std::string slow_first_phase_failure;
  const auto slow_first_phase = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &slow_first_phase_failure);
  const auto fast_first_phase = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {2.0, 1.0});
  const auto mixed_scales = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.8, 0.6});

  ASSERT_TRUE(slower.has_value()) << slow_failure;
  ASSERT_TRUE(nominal.has_value());
  ASSERT_TRUE(faster.has_value());
  ASSERT_TRUE(slow_first_phase.has_value()) << slow_first_phase_failure;
  ASSERT_TRUE(fast_first_phase.has_value());
  ASSERT_TRUE(mixed_scales.has_value());
  record_profile_metrics("phase_scale_nominal", *nominal,
    (nominal->geometry_prepare_duration_sec + nominal->profile_generation_duration_sec) * 1000.0,
    kPeriod);
  record_profile_metrics("phase_scale_fast_first", *fast_first_phase,
    (fast_first_phase->geometry_prepare_duration_sec +
    fast_first_phase->profile_generation_duration_sec) * 1000.0, kPeriod);
  EXPECT_GT(slower->duration_sec, nominal->duration_sec);
  EXPECT_DOUBLE_EQ(faster->duration_sec, nominal->duration_sec);
  expect_discrete_contract(*slower, limits, kPeriod, 0.5);
  expect_discrete_contract(*nominal, limits, kPeriod);
  expect_discrete_contract(*faster, limits, kPeriod);
  expect_discrete_contract(*slow_first_phase, limits, kPeriod);
  expect_discrete_contract(*fast_first_phase, limits, kPeriod);
  expect_discrete_contract(*mixed_scales, limits, kPeriod);
  ASSERT_EQ(mixed_scales->speed_scale_per_interval.size() + 1U,
    mixed_scales->samples.size());
  for (std::size_t sample = 1U; sample < mixed_scales->samples.size(); ++sample) {
    const double phase_scale = std::min(
      mixed_scales->speed_scale_per_interval[sample - 1U], 1.0);
    const auto & previous = mixed_scales->samples[sample - 1U];
    const auto & current = mixed_scales->samples[sample];
    const Eigen::VectorXd jerk =
      (current.acceleration - previous.acceleration) / kPeriod;
    EXPECT_LE((current.velocity.cwiseAbs().array() -
      (limits.max_velocity * phase_scale).array()).maxCoeff(), 1.0e-6);
    EXPECT_LE((current.acceleration.cwiseAbs().array() -
      (limits.max_acceleration * phase_scale * phase_scale).array()).maxCoeff(), 1.0e-6);
    EXPECT_LE((jerk.cwiseAbs().array() -
      (limits.max_jerk * phase_scale * phase_scale * phase_scale).array()).maxCoeff(), 1.0e-5);
  }

  constexpr double kSourceBoundaryProgress = 0.1;
  const auto phase_boundary_time = [&](const CompactJointProfile & profile) {
      if (profile.path_progress_rad_per_sample.size() != profile.samples.size()) {
        return std::optional<double>{};
      }
      const auto boundary = std::lower_bound(profile.path_progress_rad_per_sample.begin(),
        profile.path_progress_rad_per_sample.end(), kSourceBoundaryProgress);
      if (boundary == profile.path_progress_rad_per_sample.end()) {
        return std::optional<double>{};
      }
      return std::optional<double>{static_cast<double>(std::distance(
        profile.path_progress_rad_per_sample.begin(), boundary)) * kPeriod};
    };
  const auto nominal_boundary_time = phase_boundary_time(*nominal);
  const auto slow_boundary_time = phase_boundary_time(*slow_first_phase);
  const auto fast_boundary_time = phase_boundary_time(*fast_first_phase);
  ASSERT_TRUE(nominal_boundary_time.has_value());
  ASSERT_TRUE(slow_boundary_time.has_value());
  ASSERT_TRUE(fast_boundary_time.has_value());
  const auto scale_transition = std::adjacent_find(
    slow_first_phase->speed_scale_per_interval.begin(),
    slow_first_phase->speed_scale_per_interval.end(),
    [](const double left, const double right) {return std::abs(left - right) > 1.0e-12;});
  ASSERT_NE(scale_transition, slow_first_phase->speed_scale_per_interval.end());
  EXPECT_NEAR(*scale_transition, 0.5, 1.0e-12);
  EXPECT_NEAR(*(scale_transition + 1), 1.0, 1.0e-12);
  EXPECT_EQ(countInteriorNearStopEpisodes(*slow_first_phase), 0U);
  const std::size_t transition_interval = static_cast<std::size_t>(std::distance(
    slow_first_phase->speed_scale_per_interval.begin(), scale_transition));
  ASSERT_LT(transition_interval + 1U, slow_first_phase->samples.size());
  const auto & transition_sample = slow_first_phase->samples[transition_interval + 1U];
  EXPECT_GT(transition_sample.velocity.norm(), 0.0);
  EXPECT_GT(*slow_boundary_time, *nominal_boundary_time);
  // Keep the requested source scale for later retiming even when the physical
  // cap saturates it; phase-local partitioning may shift the boundary tick.
  EXPECT_DOUBLE_EQ(fast_first_phase->speed_scale_per_interval.front(), 2.0);

  const auto mixed_boundary_time = phase_boundary_time(*mixed_scales);
  ASSERT_TRUE(mixed_boundary_time.has_value());
  EXPECT_GT(*mixed_boundary_time, *nominal_boundary_time);
  EXPECT_DOUBLE_EQ(mixed_scales->speed_scale_per_interval.front(), 0.8);
  const auto mixed_transition = std::adjacent_find(
    mixed_scales->speed_scale_per_interval.begin(),
    mixed_scales->speed_scale_per_interval.end(),
    [](const double left, const double right) {return std::abs(left - right) > 1.0e-12;});
  ASSERT_NE(mixed_transition, mixed_scales->speed_scale_per_interval.end());
  EXPECT_DOUBLE_EQ(*mixed_transition, 0.8);
  EXPECT_DOUBLE_EQ(*(mixed_transition + 1), 0.6);
  EXPECT_EQ(countInteriorNearStopEpisodes(*mixed_scales), 0U);
}

TEST(CompactJointProfile, TotgAtPhysicalAccelerationCapUsesBoundedClockFeasibility)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  auto limits = make_limits(2);
  limits.max_jerk.setConstant(400.0);
  std::string failure_reason;

  // min(A, J*dt/2) == A here; the first clock attempt has no acceleration
  // reserve at scale 1. The bounded clock-rate search must create real reserve
  // and still respect the unchanged discrete physical limits.
  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &failure_reason);

  ASSERT_TRUE(profile.has_value()) << failure_reason;
  EXPECT_EQ(profile->builder_mode, "totg_time_warp");
  EXPECT_LE(profile->samples.size(), 100000U);
  EXPECT_EQ(countInteriorNearStopEpisodes(*profile), 0U);
  EXPECT_NEAR((profile->samples.back().position - path.back()).norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().velocity.norm(), 0.0, 1.0e-6);
  EXPECT_NEAR(profile->samples.back().acceleration.norm(), 0.0, 1.0e-6);
  expect_discrete_contract(*profile, limits, kPeriod);
}

TEST(CompactJointProfile, TotgDurationBeyondExistingSampleBoundFailsBeforeExpansion)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  auto limits = make_limits(2);
  limits.max_velocity.setConstant(1.0e-12);
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &failure_reason);

  EXPECT_FALSE(profile.has_value());
  EXPECT_NE(failure_reason.find("existing profile sample bound"), std::string::npos)
    << failure_reason;
}

TEST(CompactJointProfile, TinyAccelerationFailsPreflightBeforeTotgIntegration)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  auto limits = make_limits(2);
  limits.max_acceleration.setConstant(1.0e-12);
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &failure_reason);

  EXPECT_FALSE(profile.has_value());
  EXPECT_NE(failure_reason.find("lower-bound duration"), std::string::npos)
    << failure_reason;
}

TEST(CompactJointProfile, TinySingleAxisAccelerationFailsPreflightBeforeTotgIntegration)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  auto limits = make_limits(2);
  limits.max_acceleration[0] = 1.0e-12;
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &failure_reason);

  EXPECT_FALSE(profile.has_value());
  EXPECT_NE(failure_reason.find("lower-bound duration"), std::string::npos)
    << failure_reason;
}

TEST(CompactJointProfile, TinyAxisVelocityFailsPreflightBeforeTotgIntegration)
{
  constexpr double kPeriod = 0.01;
  const auto start = make_start({0.0, 0.0});
  const std::vector<Eigen::VectorXd> path{
    start.position,
    (Eigen::Vector2d() << 0.1, 0.0).finished(),
    (Eigen::Vector2d() << 0.1, 0.1).finished()};
  auto limits = make_limits(2);
  limits.max_velocity[0] = 1.0e-12;
  std::string failure_reason;

  const auto profile = buildCompactJointProfile(
    path, start, limits, kPeriod, 1.0, {0.5, 1.0}, &failure_reason);

  EXPECT_FALSE(profile.has_value());
  EXPECT_NE(failure_reason.find("lower-bound duration"), std::string::npos)
    << failure_reason;
}

TEST(CompactJointProfile, FloorCrossingInsideStraightLegChangesScaleAtNextCorner)
{
  constexpr double kPeriod = 0.01;
  constexpr double kFoldScale = 0.5;
  constexpr double kTurnScale = 0.8;
  const auto start = make_start({0.0, 0.0});
  const Eigen::VectorXd floor_crossing = (Eigen::Vector2d() << 0.01, 0.0).finished();
  const Eigen::VectorXd straight_end = (Eigen::Vector2d() << 0.02, 0.0).finished();
  const Eigen::VectorXd corner_goal = (Eigen::Vector2d() << 0.02, 0.01).finished();
  const std::vector<Eigen::VectorXd> path{
    start.position, floor_crossing, floor_crossing, straight_end, corner_goal};
  const std::vector<double> radii{0.5, 0.39, 0.39, 0.35, 0.3};

  const auto scales = assignCompactPhaseScales(
    path, radii, 0.4, kFoldScale, kTurnScale);

  ASSERT_TRUE(scales.has_value());
  ASSERT_EQ(scales->size(), path.size() - 1U);
  EXPECT_EQ(*scales, (std::vector<double>{kFoldScale, kFoldScale, kFoldScale, kTurnScale}));
  std::string failure_reason;
  const auto profile = buildCompactJointProfile(
    path, start, make_limits(2), kPeriod, 1.0, *scales, &failure_reason);
  ASSERT_TRUE(profile.has_value()) << failure_reason;
  EXPECT_EQ(profile->checked_path_vertices.size(), 3U);
  const auto stopped_at_floor_crossing = std::find_if(
    profile->samples.begin() + 1, profile->samples.end(),
    [&](const JointMotionState & sample) {
      return (sample.position - floor_crossing).norm() <= 1.0e-12 &&
             sample.velocity.norm() <= 1.0e-9;
    });
  EXPECT_EQ(stopped_at_floor_crossing, profile->samples.end());
  const double route_length = profile->path_progress_rad_per_sample.back();
  double minimum_transit_speed = std::numeric_limits<double>::infinity();
  double previous_source_progress = 0.0;
  const Eigen::VectorXd first_leg = straight_end - start.position;
  const Eigen::VectorXd second_leg = corner_goal - straight_end;
  for (std::size_t index = 1U; index < profile->samples.size(); ++index) {
    const double progress = profile->path_progress_rad_per_sample[index];
    if (progress > 1.0e-6 && progress < route_length - 1.0e-6) {
      minimum_transit_speed = std::min(minimum_transit_speed,
        profile->samples[index].velocity.cwiseAbs().maxCoeff());
      EXPECT_GT((profile->samples[index].position -
        profile->samples[index - 1U].position).norm(), 0.0)
        << "the phase-transition slowdown must continue advancing in q";
    }
    const Eigen::VectorXd & q = profile->samples[index].position;
    const auto project = [&](const Eigen::VectorXd & from, const Eigen::VectorXd & delta,
      const double start_progress) {
        const double fraction = std::clamp((q - from).dot(delta) / delta.squaredNorm(), 0.0, 1.0);
        const double residual = (q - (from + fraction * delta)).norm();
        return std::pair<double, double>{residual, start_progress + fraction * delta.norm()};
      };
    const auto first_projection = project(start.position, first_leg, 0.0);
    const auto second_projection = project(straight_end, second_leg, first_leg.norm());
    const double source_progress = first_projection.first <= second_projection.first ?
      first_projection.second : second_projection.second;
    EXPECT_GE(source_progress + kCompactProfilePositionToleranceRad, previous_source_progress)
      << "sample q must not backtrack along the source route";
    previous_source_progress = std::max(previous_source_progress, source_progress);
  }
  EXPECT_TRUE(std::isfinite(minimum_transit_speed));
  EXPECT_GT(minimum_transit_speed, 0.0);
  ASSERT_EQ(profile->checked_path_vertices.back().size(), corner_goal.size());
  EXPECT_NEAR((profile->checked_path_vertices.back() - corner_goal).norm(), 0.0, 1.0e-12);
  expect_discrete_contract(*profile, make_limits(2), kPeriod);
}

TEST(CompactJointProfile, AlreadyFoldedStartUsesTurnScaleAcrossDuplicateVertices)
{
  const auto start = make_start({0.0, 0.0});
  const Eigen::VectorXd corner = (Eigen::Vector2d() << 0.01, 0.0).finished();
  const Eigen::VectorXd goal = (Eigen::Vector2d() << 0.01, 0.01).finished();
  const std::vector<Eigen::VectorXd> path{start.position, start.position, corner, goal};
  const std::vector<double> radii{0.3, 0.3, 0.3, 0.3};

  const auto scales = assignCompactPhaseScales(path, radii, 0.4, 0.5, 0.8);

  ASSERT_TRUE(scales.has_value());
  EXPECT_EQ(*scales, (std::vector<double>{0.8, 0.8, 0.8}));
}

TEST(CompactJointProfile, BrakeChecksNearParallelAccelerationResidualAgainstMappedJerk)
{
  constexpr double kPeriod = 0.01;
  constexpr double kResidual = 1.0e-5;
  auto limits = make_limits(2);
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-2.0, 2.0}, 1.0, 2.0, 20.0, kPeriod,
    SolverResidual{1.0e-5});
  ASSERT_TRUE(policy.has_value());
  limits.numerical_policy = *policy;

  const Eigen::Vector2d direction(0.6, 0.8);
  const Eigen::Vector2d orthogonal(-0.8, 0.6);
  ASSERT_GT(kResidual, policy->acceleration_error.value);
  ASSERT_LT(kResidual, limits.max_jerk.minCoeff() * kPeriod);
  const auto make_branch = [&](const double residual_sign) {
      JointMotionState branch;
      branch.position = (Eigen::Vector2d() << 0.2, -0.1).finished();
      branch.velocity = direction * 0.2;
      branch.acceleration = direction * 0.3 +
        residual_sign * orthogonal * kResidual;
      return branch;
    };

  const auto safe_branch = make_branch(-1.0);
  const auto safe_tail = generateCompactProfileBrakeTail(
    safe_branch, direction, limits, EmergencyBrakeTailConfig{kPeriod});
  ASSERT_EQ(safe_tail.status, EmergencyBrakeTailStatus::kSuccess)
    << safe_tail.failure_reason;
  ASSERT_GT(safe_tail.points.size(), 1U);
  EXPECT_TRUE(safe_tail.points.front().position.isApprox(safe_branch.position, 0.0));
  EXPECT_TRUE(safe_tail.points.front().velocity.isApprox(safe_branch.velocity, 0.0));
  EXPECT_TRUE(safe_tail.points.front().acceleration.isApprox(safe_branch.acceleration, 0.0));
  const auto & first_safe_step = safe_tail.points[1U];
  const Eigen::VectorXd finite_difference_acceleration =
    (first_safe_step.velocity - safe_branch.velocity) / kPeriod;
  EXPECT_LE((first_safe_step.acceleration - finite_difference_acceleration).norm(),
    policy->acceleration_error.value * std::sqrt(2.0) + 1.0e-15);
  EXPECT_LE(((first_safe_step.acceleration - safe_branch.acceleration) / kPeriod).
    cwiseAbs().maxCoeff(), limits.max_jerk.maxCoeff() + policy->jerk_error.value);
  EXPECT_NEAR((first_safe_step.position - safe_branch.position -
    first_safe_step.velocity * kPeriod).norm(), 0.0, 1.0e-12);

  const auto unsafe_branch = make_branch(1.0);
  const auto unsafe_tail = generateCompactProfileBrakeTail(
    unsafe_branch, direction, limits, EmergencyBrakeTailConfig{kPeriod});
  ASSERT_EQ(unsafe_tail.status, EmergencyBrakeTailStatus::kSuccess)
    << unsafe_tail.failure_reason;
  ASSERT_GT(unsafe_tail.points.size(), 1U);
  const Eigen::VectorXd unsafe_first_jerk =
    (unsafe_tail.points[1U].acceleration - unsafe_branch.acceleration) / kPeriod;
  EXPECT_LE(unsafe_first_jerk.cwiseAbs().maxCoeff(),
    limits.max_jerk.maxCoeff() + policy->jerk_error.value);

  auto impossible_branch = make_branch(1.0);
  impossible_branch.acceleration += orthogonal * 0.4;
  const auto impossible_tail = generateCompactProfileBrakeTail(
    impossible_branch, direction, limits, EmergencyBrakeTailConfig{kPeriod});
  EXPECT_EQ(impossible_tail.status, EmergencyBrakeTailStatus::kInfeasible);
  EXPECT_NE(impossible_tail.failure_reason.find("residual jerk has no reserve"),
    std::string::npos) << impossible_tail.failure_reason;
}

TEST(CompactJointProfile, ReplaysCapturedRuntimeBrakeBranchWithRoundedState)
{
  constexpr double kPeriod = 0.01;
  auto limits = make_runtime_compact_limits(3.1, 19.0, 400.0);
  ASSERT_TRUE(limits.numerical_policy.has_value());
  JointMotionState branch;
  branch.position = (Eigen::Matrix<double, 6, 1>() <<
    6.127503042, -1.325903415, -0.453511811,
    -1.824253999, -1.006257878, 1.944006417).finished();
  branch.velocity = (Eigen::Matrix<double, 6, 1>() <<
    0.109867425, 0.050431788, 0.011623737,
    -0.099107487, -0.076204979, -0.020422932).finished();
  branch.acceleration = (Eigen::Matrix<double, 6, 1>() <<
    1.997540752, 0.916919194, 0.211335507,
    -1.801910296, -1.385511235, -0.371316959).finished();

  const auto tail = generateCompactProfileBrakeTail(
    branch, branch.velocity, limits, EmergencyBrakeTailConfig{kPeriod});

  ASSERT_EQ(tail.status, EmergencyBrakeTailStatus::kSuccess) << tail.failure_reason;
  ASSERT_GT(tail.points.size(), 1U);
  const Eigen::VectorXd direction = branch.velocity.normalized();
  const Eigen::VectorXd velocity_residual =
    branch.velocity - direction * direction.dot(branch.velocity);
  const Eigen::VectorXd acceleration_residual =
    branch.acceleration - direction * direction.dot(branch.acceleration);
  const Eigen::VectorXd residual_jerk = acceleration_residual / kPeriod +
    velocity_residual / (kPeriod * kPeriod);
  const Eigen::VectorXd actual_first_jerk =
    (tail.points[1U].acceleration - branch.acceleration) / kPeriod;
  const double scalar_first_jerk = direction.dot(actual_first_jerk + residual_jerk);
  const double old_projected_jerk_cap = limits.max_jerk.maxCoeff() /
    direction.cwiseAbs().maxCoeff();
  const Eigen::VectorXd unreserved_first_jerk = direction *
    std::copysign(old_projected_jerk_cap, scalar_first_jerk) - residual_jerk;
  Eigen::Index limiting_axis = 0;
  const double unreserved_peak = unreserved_first_jerk.cwiseAbs().maxCoeff(&limiting_axis);
  EXPECT_GT(unreserved_peak - limits.max_jerk[limiting_axis], 0.0)
    << "the old projected cap must reproduce the captured first-sample violation";
  EXPECT_LE(actual_first_jerk.cwiseAbs().maxCoeff(),
    limits.max_jerk.maxCoeff() + limits.numerical_policy->jerk_error.value);
  for (std::size_t index = 1U; index < tail.points.size(); ++index) {
    const Eigen::VectorXd jerk =
      (tail.points[index].acceleration - tail.points[index - 1U].acceleration) / kPeriod;
    EXPECT_LE(jerk.cwiseAbs().maxCoeff(),
      limits.max_jerk.maxCoeff() + limits.numerical_policy->jerk_error.value);
  }
}

TEST(CompactJointProfile, RejectsNonFiniteSpeedScale)
{
  const auto start = make_start({0.01});
  const std::vector<Eigen::VectorXd> path{
    start.position, Eigen::VectorXd::Constant(1, 0.0)};
  EXPECT_FALSE(buildCompactJointProfile(
      path, start, make_limits(1), 0.01,
      std::numeric_limits<double>::infinity()).has_value());
}

}  // namespace
}  // namespace face_tracking_arm::control
