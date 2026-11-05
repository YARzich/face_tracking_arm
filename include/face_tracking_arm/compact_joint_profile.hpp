// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_JOINT_PROFILE_HPP_
#define FACE_TRACKING_ARM__COMPACT_JOINT_PROFILE_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{

inline constexpr double kCompactProfilePositionToleranceRad{1.0e-6};

/// Samples of a jerk-limited profile on the controller's fixed time grid.
/// Samples include the starting state and use the same position/velocity
/// integration convention as the published servo queue.
struct CompactJointProfile
{
  std::vector<JointMotionState> samples;
  /// Cumulative joint-space distance (rad) of each authoritative sample from
  /// samples.front(). The vector is parallel to samples and is nondecreasing.
  std::vector<double> path_progress_rad_per_sample;
  /// Relative source phase scale for each transition i -> i+1 in samples.
  /// The vector has samples.size()-1 entries; the request-wide compact speed
  /// scale is applied separately when rebuilding a profile.
  std::vector<double> speed_scale_per_interval;
  /// Checked straight-leg route used to generate these samples.
  /// Retained for legacy consumers; smooth-path geometry is represented by
  /// samples[*].position and its progress vector above.
  std::vector<Eigen::VectorXd> checked_path_vertices;
  std::vector<double> checked_path_segment_scales;
  /// Legacy scalar-leg terminal indices used to map relative scales in the
  /// straight scalar builder. Smooth TOTG profiles leave this vector empty.
  std::vector<std::size_t> leg_end_sample_indices;
  double duration_sec{0.0};
  double path_length_rad{0.0};
  /// Low-cardinality timing for profile preparation and generation.
  std::string builder_mode{"straight_scalar"};
  /// Exact reason the fast TOTG acceleration candidate was rejected, if a
  /// bounded fallback candidate was accepted.
  std::string builder_fallback_reason;
  double geometry_prepare_duration_sec{0.0};
  double profile_generation_duration_sec{0.0};
  /// Cursor index of the stationary endpoint of a prepended published
  /// braking prefix. Zero means compact limits apply from the first sample.
  std::size_t prefix_endpoint_sample_index{0U};
};

/// Project a queue state onto the authoritative sampled profile geometry in a
/// bounded forward progress window. Returns null for excessive residual,
/// opposing velocity, or ambiguous projections at distinct progress values.
[[nodiscard]] std::optional<double> projectCompactProfileProgress(
  const CompactJointProfile & profile,
  const Eigen::VectorXd & position,
  const Eigen::VectorXd & velocity,
  double minimum_progress_rad,
  double maximum_progress_rad,
  double residual_limit_rad);

/// Assign phase scales while keeping a radius-floor transition inside an
/// existing straight leg. A turn scale begins only after that leg's corner.
[[nodiscard]] std::optional<std::vector<double>> assignCompactPhaseScales(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & assembly_radii_m,
  double radius_floor_m,
  double fold_speed_scale,
  double turn_speed_scale);

/// Build a jerk-limited time law along a joint-space route.
/// Straight uniform-scale routes use the scalar profile; other routes use
/// MoveIt TOTG geometry with a Ruckig time warp to avoid waypoint stops.
/// `speed_scale` time-warps the profile: v limits scale by s, acceleration by
/// s^2, and jerk by s^3. Invalid limits, incompatible initial motion, or a
/// discretized sample outside the shared q/v/a/jerk limits fail closed.
[[nodiscard]] std::optional<CompactJointProfile> buildCompactJointProfile(
  const std::vector<Eigen::VectorXd> & path,
  const JointMotionState & start,
  const JointMotionLimits & limits,
  double control_period_sec,
  double speed_scale = 1.0,
  const std::vector<double> & segment_speed_scales = {},
  std::string * failure_reason = nullptr);

/// Prepend a previously validated, fixed-grid queue prefix ending at the
/// continuation's stationary initial state. The prefix start is retained and
/// the duplicate stationary endpoint is emitted only once.
[[nodiscard]] bool prependValidatedCompactPrefix(
  CompactJointProfile & continuation,
  const std::vector<JointMotionState> & prefix,
  double control_period_sec,
  std::string * failure_reason = nullptr);

/// Generate a joint-limit-safe braking suffix using one existing scalar
/// braking rollout projected along the checked PATH segment direction.
[[nodiscard]] EmergencyBrakeTail generateCompactProfileBrakeTail(
  const JointMotionState & branch_state,
  const Eigen::VectorXd & path_direction,
  const JointMotionLimits & limits,
  const EmergencyBrakeTailConfig & config);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_JOINT_PROFILE_HPP_
