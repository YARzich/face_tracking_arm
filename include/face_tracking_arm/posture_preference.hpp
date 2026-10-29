// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POSTURE_PREFERENCE_HPP_
#define FACE_TRACKING_ARM__POSTURE_PREFERENCE_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <limits>

namespace face_tracking_arm::control
{

struct PosturePreferenceConfig
{
  std::size_t j3_index{2U};
  std::size_t j5_index{4U};
  std::size_t j6_index{5U};
  double joint3_angle_offset_rad{0.0};
  double preferred_bend_sign{1.0};
  double comfortable_joint3_lower{std::numeric_limits<double>::quiet_NaN()};
  double comfortable_joint3_upper{std::numeric_limits<double>::quiet_NaN()};
  double lower_j5{-1.693};
  double upper_j5{3.14159265358979323846};
  double lower_j6{-6.28318530717958647692};
  double upper_j6{6.28318530717958647692};
  double position_margin_j5_rad{0.10};
  double position_margin_j6_rad{0.10};
  double position_gain{1.5};
  double maximum_linear_reference_mps{0.18};
  double maximum_angular_reference_radps{0.70};
  double control_period_sec{0.01};
};

struct PosturePreferenceObjective
{
  Eigen::MatrixXd matrix;
  Eigen::VectorXd reference;
  Eigen::VectorXd weights;
  bool valid{false};
  /// True when no soft posture correction row is active.
  bool comfortable{false};
  /// True when joint3 lies in the calibrated normal-bend interval.
  bool branch_comfortable{false};
};

struct PostureBendCalibration
{
  double angle_offset_rad{0.0};
  double preferred_sign{0.0};
  double minimum_comfortable_sin{0.0};
  double joint3_lower{0.0};
  double joint3_upper{0.0};
  bool valid{false};
};

/// Calibrate one continuous signed-bend lobe from the model's named rest and
/// search configurations. The supplied angles come from FK and the URDF joint3 axis.
[[nodiscard]] PostureBendCalibration calibratePostureBend(
  double rest_joint3, double rest_angle, double search_joint3, double search_angle,
  double physical_joint3_lower, double physical_joint3_upper);

/// Softly return joint3 to the model-calibrated normal-bend interval, then
/// prefer the face side and avoid physical J5/J6 limits. Positions and
/// Jacobians use one world frame. No joint-position centering is applied.
[[nodiscard]] PosturePreferenceObjective makePosturePreferenceObjective(
  const Eigen::Vector3d & base, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face,
  const Eigen::MatrixXd & base_jacobian,
  const Eigen::MatrixXd & elbow_jacobian, const Eigen::MatrixXd & wrist_jacobian,
  const Eigen::VectorXd & joint_position,
  const PosturePreferenceConfig & config = {});

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__POSTURE_PREFERENCE_HPP_
