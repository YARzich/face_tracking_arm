// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_PATH_TIMING_HPP_
#define COMPACT_PATH_TIMING_HPP_

#include <Eigen/Core>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <ruckig/ruckig.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>

#include "face_tracking_arm/compact_joint_profile.hpp"

namespace face_tracking_arm::control::detail::path_timing
{

inline constexpr std::size_t kMaximumProfileSamples{100000U};
inline constexpr double kTotgIntegrationStepSec{0.001};

struct ClockSegment
{
  ruckig::Trajectory<1> trajectory;
  double duration{0.0};
  double tau_start{0.0};
  double relative_scale{1.0};
};

struct ClockSchedule
{
  std::vector<ClockSegment> clocks;
  std::vector<double> phase_tau;
  double duration{0.0};
};

/// Build the Ruckig clock schedule for precomputed TOTG phase boundaries.
[[nodiscard]] std::optional<ClockSchedule> buildClockSchedule(
  const trajectory_processing::Trajectory & totg,
  Eigen::Index joint_count,
  std::vector<double> phase_tau,
  std::vector<double> phase_scales,
  const JointMotionLimits & limits,
  double control_period_sec,
  double speed_scale,
  double requested_speed_scale,
  double velocity_reserve,
  double acceleration_reserve,
  double jerk_reserve,
  const Eigen::VectorXd & totg_acceleration,
  bool phase_local_clock,
  std::string * failure_reason);

}  // namespace face_tracking_arm::control::detail::path_timing

#endif  // COMPACT_PATH_TIMING_HPP_
