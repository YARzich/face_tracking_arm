// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_
#define FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"
#include "face_tracking_arm/horizontal_edge_upright.hpp"

namespace face_tracking_arm::control
{

struct TrackingVelocityTaskConfig
{
  bool horizontal_edge_primary{true};
  double position_gain{1.5};
  double orientation_gain{4.0};
  double position_deadband_m{0.03};
  double pointing_deadband_rad{0.005};
  /// Deadband for the horizontal screen-edge tilt objective.
  double roll_deadband_rad{0.01};
  double maximum_linear_reference_mps{0.18};
  double maximum_angular_reference_radps{0.70};
  /// Relative roll residual weight in the primary full-orientation FACE task.
  /// The field name is retained for parameter/API compatibility.
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
  /// Signed [screen-normal left, screen-normal up, upright roll] task error.
  /// This matches the three ordinary FACE primary rows.
  Eigen::Vector3d signed_face_primary_error_rad{Eigen::Vector3d::Zero()};
  /// Target-only angular feed-forward in those rows, excluding orientation_gain.
  Eigen::Vector3d face_primary_feedforward_radps{Eigen::Vector3d::Zero()};
  bool has_face_primary_roll{false};
  /// All errors are within their deadbands and both combined references are zero.
  /// This is a geometric condition, not evidence that the robot has stopped.
  bool accepted{false};
};

struct UprightRollPathTaskResult
{
  HierarchicalVelocityTask task;
  double roll_error_rad{0.0};
};

struct HorizontalEdgePathTaskResult
{
  HierarchicalVelocityTask task;
  double signed_edge_sine{0.0};
  double edge_tilt_error_rad{0.0};
};

struct FaceReturnPointingTask
{
  Eigen::MatrixXd matrix;
  Eigen::Vector2d reference{Eigen::Vector2d::Zero()};
  bool used_normal_fallback{false};
};

/// Build a pure velocity task in a common world/planning frame.
///
/// The Jacobian maps qdot to [TCP linear velocity; world angular velocity].
/// With a face, pointing and horizontal-edge height account for the changing line of
/// sight caused by TCP translation. Without a face, target_rotation is fixed.
/// Optional velocities are expressed in the same planning frame. Face velocity
/// adds the desired look-at frame's angular motion; target-position velocity adds
/// translational feed-forward. Speed caps apply after adding feedback and feed-forward.
/// Ordinary FACE tracking prioritizes both screen-normal axes and gravity-upright
/// screen roll, with translation secondary. An optional gaze frame replaces the
/// two-axis pointing task while physical screen roll remains primary.
/// Fixed-orientation tracking without a face keeps roll secondary. The
/// roll_weight name is retained for parameter/API compatibility even though it
/// weights a primary residual for ordinary FACE.
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

/// Preserve only the ordinary FACE screen-edge primary row for a guarded
/// inward correction. The caller must prove this is a fresh FACE task and that
/// row 2 is the upright screen-edge objective. Secondary identity/zero motion
/// keeps joint speed small while the hard constraints remain owned by the QP.
[[nodiscard]] std::optional<HierarchicalVelocityTask> makeFaceEdgeOnlyRecoveryTask(
  const HierarchicalVelocityTask & face_task, bool has_face_primary_roll);

/// Track face azimuth and elevation during compact RETURN_FACE reacquisition.
/// The returned rows use analytic current-normal and moving-ray Jacobians;
/// normal_matrix/reference provide the existing pointing rows as a finite
/// fallback when azimuth/elevation coordinates approach their vertical pole.
[[nodiscard]] std::optional<FaceReturnPointingTask> makeFaceReturnPointingTask(
  const Eigen::Isometry3d & current_screen_pose,
  const Eigen::MatrixXd & screen_jacobian,
  const Eigen::Vector3d & face_position,
  const Eigen::Vector3d & face_velocity,
  const TrackingVelocityTaskConfig & config,
  const Eigen::MatrixXd & normal_matrix,
  const Eigen::VectorXd & normal_reference,
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

/// Keep a moving compact path upright without retaining FACE pointing or yaw.
/// The primary row tracks screen roll relative to world +Z; a joint-path
/// velocity reference is secondary. The task rejects screen normals too close
/// to world vertical, where upright roll is undefined.
[[nodiscard]] std::optional<UprightRollPathTaskResult> makeUprightRollPathTask(
  const Eigen::Isometry3d & current_screen_pose,
  const Eigen::MatrixXd & screen_jacobian,
  const Eigen::VectorXd & path_velocity_reference,
  const TrackingVelocityTaskConfig & config = {});

/// Keep the monitor's horizontal screen edge level without a screen-normal pole.
[[nodiscard]] std::optional<HorizontalEdgePathTaskResult> makeHorizontalEdgePathTask(
  const Eigen::Isometry3d & current_screen_pose,
  const Eigen::MatrixXd & screen_jacobian,
  const Eigen::VectorXd & path_velocity_reference,
  const TrackingVelocityTaskConfig & config = {});

[[nodiscard]] std::optional<double> measureHorizontalEdgeTiltError(
  const Eigen::Isometry3d & current_screen_pose);

/// Measure screen roll against the gravity-upright frame; returns nullopt near
/// the vertical-normal singularity where upright roll is undefined.
[[nodiscard]] std::optional<double> measureUprightRollError(
  const Eigen::Isometry3d & current_screen_pose);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__TRACKING_VELOCITY_TASK_HPP_
