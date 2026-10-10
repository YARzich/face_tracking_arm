// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_UPRIGHT_HPP_
#define FACE_TRACKING_ARM__COMPACT_UPRIGHT_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

namespace face_tracking_arm::control
{

inline constexpr double kCompactMinimumHorizontalProjectionNorm{0.10};
inline constexpr double kCompactPitchValidationToleranceRad{1.0e-9};

struct CompactUprightPose
{
  Eigen::Vector3d screen_normal{Eigen::Vector3d::UnitX()};
  double signed_pitch_rad{0.0};
  double signed_roll_error_rad{0.0};
  double horizontal_projection_norm{1.0};
};

struct CompactOrientationGuideBounds
{
  double absolute_x_axis_tolerance_rad{0.0};
  double absolute_y_axis_tolerance_rad{0.0};
};

/// Measures signed screen-normal elevation and gravity-relative roll.
/// Returns nullopt when the screen normal is too close to vertical for roll to be defined.
[[nodiscard]] std::optional<CompactUprightPose> evaluateCompactUprightPose(
  const Eigen::Matrix3d & screen_rotation,
  double minimum_horizontal_projection_norm = kCompactMinimumHorizontalProjectionNorm);

/// Computes d(signed screen pitch)/dq from a world-frame angular Jacobian.
/// The Jacobian columns are angular velocities in the same frame as screen_normal.
[[nodiscard]] std::optional<Eigen::RowVectorXd> signedScreenPitchGradient(
  const Eigen::Vector3d & screen_normal,
  const Eigen::MatrixXd & screen_angular_jacobian,
  double minimum_horizontal_projection_norm = kCompactMinimumHorizontalProjectionNorm);

/// Computes d(gravity-relative screen roll error)/dq from a world-frame angular Jacobian.
/// The Jacobian columns are angular velocities in the same frame as screen_rotation.
[[nodiscard]] std::optional<Eigen::RowVectorXd> signedScreenRollErrorGradient(
  const Eigen::Matrix3d & screen_rotation,
  const Eigen::MatrixXd & screen_angular_jacobian,
  double minimum_horizontal_projection_norm = kCompactMinimumHorizontalProjectionNorm);

/// Returns a conservative MoveIt XYZ orientation-constraint box containing the
/// physical pitch/roll envelope, with yaw about world up left unconstrained.
[[nodiscard]] std::optional<CompactOrientationGuideBounds> compactOrientationGuideBounds(
  double maximum_abs_pitch_rad, double maximum_roll_error_rad,
  double minimum_horizontal_projection_norm = kCompactMinimumHorizontalProjectionNorm);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_UPRIGHT_HPP_
