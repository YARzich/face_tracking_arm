// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_TOTG_TIME_WARP_HPP_
#define COMPACT_TOTG_TIME_WARP_HPP_

#include <Eigen/Core>

#include <string>
#include <vector>

#include "face_tracking_arm/compact_joint_profile.hpp"

namespace face_tracking_arm::control::detail
{

/// Build a sampled queue profile by time-warping MoveIt TOTG's smoothed path.
/// Validates fast, phase-local acceleration/coast, and bounded-acceleration
/// clocks on the shared fixed-grid sampler, including per-sample brake tails;
/// queue or brake refusals may use a bounded uniform polynomial time dilation.
/// Relative phase scales are source-route metadata; request-wide scale is
/// applied separately. Returns null on infeasible clock or discrete limits.
[[nodiscard]] std::optional<CompactJointProfile> buildTotgTimeWarpProfile(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & relative_phase_scales,
  const JointMotionState & stationary_start,
  const JointMotionLimits & limits,
  double control_period_sec,
  double speed_scale,
  double requested_speed_scale,
  std::string * failure_reason);

}  // namespace face_tracking_arm::control::detail

#endif  // COMPACT_TOTG_TIME_WARP_HPP_
