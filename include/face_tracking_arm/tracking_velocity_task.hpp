// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_
#define FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{

struct TrackingVelocityTaskConfig
{
  double position_gain{1.5};
  double orientation_gain{4.0};
  double position_deadband_m{0.03};
  double pointing_deadband_rad{0.005};
  double roll_deadband_rad{0.01};
  double maximum_linear_reference_mps{0.18};
  double maximum_angular_reference_radps{0.70};
  double roll_weight{0.5};
};

struct GazeKinematics
{
  /// World pose with +X forward, +Y left, +Z up (convert optical axes before use).
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
  /// World [linear; angular] Jacobian at the gaze origin, in the same joint order.
  Eigen::MatrixXd jacobian;
};

struct TrackingVelocityTaskResult
{
  HierarchicalVelocityTask task;
  double position_error_m{0.0};
  double pointing_error_rad{0.0};
  double roll_error_rad{0.0};
  /// All errors are within their deadbands and both combined references are zero.
  /// This is a geometric condition, not evidence that the robot has stopped.
  bool accepted{false};
};

/// Build a pure velocity task in a common world/planning frame.
///
/// The Jacobian maps qdot to [TCP linear velocity; world angular velocity].
/// With a face, both pointing and upright roll account for the changing line of
/// sight caused by TCP translation. Without a face, target_rotation is fixed.
/// Optional velocities are expressed in the same planning frame. Face velocity
/// adds the desired look-at frame's angular motion; target-position velocity adds
/// translational feed-forward. Speed caps apply after adding feedback and feed-forward.
/// An optional gaze frame replaces only the primary pointing task. Secondary
/// translation and roll always refer to the monitor, including a displaced camera.
/// The primary stage contains two pointing rows; translation and soft roll are secondary.
/// No joint posture or collision-separation preference is introduced here.
/// Vertical rays remain valid: upright roll fades out near the pole.
/// Returns nullopt for invalid input or coincident face and TCP.
[[nodiscard]] std::optional<TrackingVelocityTaskResult> makeTrackingVelocityTask(
  const Eigen::Isometry3d & current,
  const Eigen::MatrixXd & jacobian,
  const Eigen::Vector3d & target_position,
  const std::optional<Eigen::Vector3d> & face_position,
  const Eigen::Matrix3d & target_rotation,
  const TrackingVelocityTaskConfig & config = {},
  const Eigen::Vector3d & face_velocity = Eigen::Vector3d::Zero(),
  const Eigen::Vector3d & target_position_velocity = Eigen::Vector3d::Zero(),
  const std::optional<GazeKinematics> & gaze = std::nullopt);

/// Track a joint-path lookahead point without wrapping bounded revolute joints.
/// The reference is scaled uniformly to preserve its joint-space direction.
/// Leave deadband at zero for intermediate waypoints; the caller owns completion.
[[nodiscard]] std::optional<HierarchicalVelocityTask> makeJointPathVelocityTask(
  const Eigen::VectorXd & current_position,
  const Eigen::VectorXd & waypoint,
  double gain,
  double maximum_velocity_radps,
  double deadband_rad = 0.0);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_
