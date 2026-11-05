// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POSTURE_TASK_BUILDER_HPP_
#define FACE_TRACKING_ARM__POSTURE_TASK_BUILDER_HPP_

#include <Eigen/Core>

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"
#include "face_tracking_arm/posture_preference.hpp"

namespace face_tracking_arm::control
{

struct PostureTaskBuilderParameters
{
  double position_gain{1.5};
  double joint_position_margin_rad{0.10};
  double maximum_linear_reference_mps{0.18};
  double maximum_angular_reference_radps{0.70};
  double control_period_sec{0.01};
  double residual_command_latency_sec{0.020};
};

enum class PostureTaskMode
{
  kPreserve,
  kRecover,
  kUnavailable
};

enum class PostureUnavailableReason
{
  kNone,
  kCalibrationUnavailable,
  kKinematicsUnavailable,
  kInvalidObjective,
  kComfortBoundsUnavailable
};

struct PostureTaskResult
{
  PostureTaskMode mode{PostureTaskMode::kUnavailable};
  PostureUnavailableReason unavailable_reason{PostureUnavailableReason::kNone};
};

[[nodiscard]] const char * postureTaskModeName(PostureTaskMode mode) noexcept;
[[nodiscard]] const char * postureUnavailableReasonName(
  PostureUnavailableReason reason) noexcept;

/// Builds the posture-related rows for ordinary FACE tracking.
class PostureTaskBuilder final
{
public:
  PostureTaskBuilder(
    moveit::core::RobotModelConstPtr robot_model,
    const moveit::core::JointModelGroup * group,
    const JointMotionLimits & limits,
    PostureTaskBuilderParameters parameters);

  /// Replaces secondary/tertiary objectives and may append a preferred-primary
  /// constraint. Reports the resulting posture mode and any unavailability reason.
  [[nodiscard]] PostureTaskResult apply(
    const moveit::core::RobotState & state, HierarchicalVelocityTask & task) const;

  [[nodiscard]] double comfortableJoint3Lower() const;
  [[nodiscard]] double comfortableJoint3Upper() const;

private:
  void calibrate();

  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  const moveit::core::LinkModel * root_link_{nullptr};
  const moveit::core::LinkModel * shoulder_link_{nullptr};
  const moveit::core::LinkModel * elbow_link_{nullptr};
  const moveit::core::LinkModel * wrist_link_{nullptr};
  JointMotionLimits limits_;
  PostureTaskBuilderParameters parameters_;
  PosturePreferenceConfig preference_config_;
  bool calibrated_{false};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__POSTURE_TASK_BUILDER_HPP_
