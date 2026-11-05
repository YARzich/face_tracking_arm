// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_DYNAMICS_HPP_
#define FACE_TRACKING_ARM__COMPACT_DYNAMICS_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <moveit/robot_model/robot_model.hpp>

#include "face_tracking_arm/compact_joint_profile.hpp"

namespace face_tracking_arm::control
{

enum class CompactDynamicsStatus
{
  kPass,
  kRetimed,
  kPrefixViolation,
  kStaticInfeasible,
  kNoFeasibleScale,
  kInvalidInput,
  kSolverFailure
};

struct CompactDynamicsResult
{
  CompactDynamicsStatus status{CompactDynamicsStatus::kInvalidInput};
  std::optional<CompactJointProfile> profile;
  /// Requested speed scale multiplied by the selected common tempo correction.
  double applied_speed_scale{1.0};
  /// Peak absolute required effort before correction, in group variable order.
  std::vector<double> peak_required_effort_nm_before;
  /// Peak absolute required effort in the returned profile, in group variable order.
  std::vector<double> peak_required_effort_nm_after;
  /// Maximum required effort / URDF limit across all joints, before and after.
  double maximum_effort_ratio_before{0.0};
  double maximum_effort_ratio_after{0.0};
  std::size_t checked_profile_samples{0U};
  std::size_t checked_interpolation_samples{0U};
  /// A published prefix is checked but cannot be modified by this component.
  bool immutable_prefix_over_limit{false};
  std::string failure_reason;
  double elapsed_ms{0.0};
};

/// One-time inverse dynamics model for a compact profile.
///
/// Effort limits and URDF joint dynamics are read in group variable order from
/// the supplied RobotModel. The caller must pass the effective model used by
/// the controller, including any already-applied effort overrides.
class CompactDynamicsModel final
{
public:
  CompactDynamicsModel(
    moveit::core::RobotModelConstPtr robot_model,
    const std::string & group_name,
    const std::string & base_link_name,
    const Eigen::Vector3d & gravity_world = Eigen::Vector3d{0.0, 0.0, -9.81});
  ~CompactDynamicsModel();

  CompactDynamicsModel(const CompactDynamicsModel &) = delete;
  CompactDynamicsModel & operator=(const CompactDynamicsModel &) = delete;
  CompactDynamicsModel(CompactDynamicsModel &&) noexcept;
  CompactDynamicsModel & operator=(CompactDynamicsModel &&) noexcept;

  /// Verify a prepared profile and, if needed, rebuild only its mutable suffix
  /// with a bounded common time-scale correction. The fixed published prefix
  /// remains byte-for-byte unchanged and is reported separately if over limit.
  [[nodiscard]] CompactDynamicsResult validateAndRetime(
    const CompactJointProfile & input,
    const JointMotionLimits & motion_limits,
    double control_period_sec,
    double requested_speed_scale) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_DYNAMICS_HPP_
