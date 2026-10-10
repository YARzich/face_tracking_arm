// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_upright.hpp"

#include <algorithm>
#include <cmath>

namespace face_tracking_arm::control
{
namespace
{

bool validProjectionLimit(const double value)
{
  return std::isfinite(value) && value > 0.0 && value < 1.0;
}

}  // namespace

std::optional<CompactUprightPose> evaluateCompactUprightPose(
  const Eigen::Matrix3d & screen_rotation, const double minimum_horizontal_projection_norm)
{
  if (!screen_rotation.allFinite() || !validProjectionLimit(minimum_horizontal_projection_norm)) {
    return std::nullopt;
  }

  Eigen::Vector3d normal = screen_rotation.col(0);
  const double normal_norm = normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-12) {
    return std::nullopt;
  }
  normal /= normal_norm;

  const double horizontal_projection_norm = std::hypot(normal.x(), normal.y());
  if (!std::isfinite(horizontal_projection_norm) ||
    horizontal_projection_norm < minimum_horizontal_projection_norm)
  {
    return std::nullopt;
  }

  Eigen::Vector3d actual_up = screen_rotation.col(2);
  actual_up -= actual_up.dot(normal) * normal;
  const double actual_up_norm = actual_up.norm();
  if (!std::isfinite(actual_up_norm) || actual_up_norm <= 1.0e-12) {
    return std::nullopt;
  }
  actual_up /= actual_up_norm;

  Eigen::Vector3d desired_up = Eigen::Vector3d::UnitZ() - normal.z() * normal;
  const double desired_up_norm = desired_up.norm();
  if (!std::isfinite(desired_up_norm) || desired_up_norm < minimum_horizontal_projection_norm) {
    return std::nullopt;
  }
  desired_up /= desired_up_norm;

  const double pitch = std::atan2(normal.z(), horizontal_projection_norm);
  const double roll_error = std::atan2(
    normal.dot(actual_up.cross(desired_up)), actual_up.dot(desired_up));
  if (!std::isfinite(pitch) || !std::isfinite(roll_error)) {
    return std::nullopt;
  }

  return CompactUprightPose{normal, pitch, roll_error, horizontal_projection_norm};
}

std::optional<Eigen::RowVectorXd> signedScreenPitchGradient(
  const Eigen::Vector3d & screen_normal, const Eigen::MatrixXd & screen_angular_jacobian,
  const double minimum_horizontal_projection_norm)
{
  if (!screen_normal.allFinite() || !screen_angular_jacobian.allFinite() ||
    screen_angular_jacobian.rows() != 3 || screen_angular_jacobian.cols() == 0 ||
    !validProjectionLimit(minimum_horizontal_projection_norm))
  {
    return std::nullopt;
  }

  Eigen::Vector3d normal = screen_normal;
  const double normal_norm = normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-12) {
    return std::nullopt;
  }
  normal /= normal_norm;
  const double horizontal_projection_norm = std::hypot(normal.x(), normal.y());
  if (!std::isfinite(horizontal_projection_norm) ||
    horizontal_projection_norm < minimum_horizontal_projection_norm)
  {
    return std::nullopt;
  }

  // For a unit normal, d(asin(n.z))/dq = (n x world_up)^T J_omega / ||n_xy||.
  const Eigen::RowVectorXd gradient =
    (normal.cross(Eigen::Vector3d::UnitZ()).transpose() * screen_angular_jacobian) /
    horizontal_projection_norm;
  if (!gradient.allFinite()) {
    return std::nullopt;
  }
  return gradient;
}

std::optional<Eigen::RowVectorXd> signedScreenRollErrorGradient(
  const Eigen::Matrix3d & screen_rotation, const Eigen::MatrixXd & screen_angular_jacobian,
  const double minimum_horizontal_projection_norm)
{
  if (!screen_rotation.allFinite() || !screen_angular_jacobian.allFinite() ||
    screen_angular_jacobian.rows() != 3 || screen_angular_jacobian.cols() == 0 ||
    !validProjectionLimit(minimum_horizontal_projection_norm))
  {
    return std::nullopt;
  }
  const auto pose = evaluateCompactUprightPose(
    screen_rotation, minimum_horizontal_projection_norm);
  if (!pose) {
    return std::nullopt;
  }

  const Eigen::Vector3d normal = pose->screen_normal;
  Eigen::Vector3d actual_up = screen_rotation.col(2);
  actual_up -= actual_up.dot(normal) * normal;
  const double actual_up_norm = actual_up.norm();
  if (!std::isfinite(actual_up_norm) || actual_up_norm <= 1.0e-12) {
    return std::nullopt;
  }
  actual_up /= actual_up_norm;
  Eigen::Vector3d desired_up =
    Eigen::Vector3d::UnitZ() - normal.z() * normal;
  const double desired_up_norm = desired_up.norm();
  if (!std::isfinite(desired_up_norm) ||
    desired_up_norm < minimum_horizontal_projection_norm)
  {
    return std::nullopt;
  }
  desired_up /= desired_up_norm;

  const double sine = normal.dot(actual_up.cross(desired_up));
  const double cosine = actual_up.dot(desired_up);
  const double angle_norm_squared = sine * sine + cosine * cosine;
  if (!std::isfinite(angle_norm_squared) || angle_norm_squared <= 1.0e-12) {
    return std::nullopt;
  }

  Eigen::RowVectorXd gradient(screen_angular_jacobian.cols());
  const Eigen::Vector3d screen_up = screen_rotation.col(2);
  const double up_normal_dot = screen_up.dot(normal);
  for (Eigen::Index column = 0; column < screen_angular_jacobian.cols(); ++column) {
    const Eigen::Vector3d angular_velocity = screen_angular_jacobian.col(column);
    const Eigen::Vector3d normal_velocity = angular_velocity.cross(normal);
    const Eigen::Vector3d screen_up_velocity = angular_velocity.cross(screen_up);
    const double up_normal_dot_velocity =
      screen_up_velocity.dot(normal) + screen_up.dot(normal_velocity);
    const Eigen::Vector3d actual_up_projection_velocity =
      screen_up_velocity - up_normal_dot_velocity * normal -
      up_normal_dot * normal_velocity;
    const Eigen::Vector3d actual_up_velocity =
      (Eigen::Matrix3d::Identity() - actual_up * actual_up.transpose()) *
      actual_up_projection_velocity / actual_up_norm;

    const Eigen::Vector3d desired_up_projection_velocity =
      -normal_velocity.z() * normal - normal.z() * normal_velocity;
    const Eigen::Vector3d desired_up_velocity =
      (Eigen::Matrix3d::Identity() - desired_up * desired_up.transpose()) *
      desired_up_projection_velocity / desired_up_norm;

    const double sine_velocity =
      normal_velocity.dot(actual_up.cross(desired_up)) +
      normal.dot(actual_up_velocity.cross(desired_up) +
      actual_up.cross(desired_up_velocity));
    const double cosine_velocity =
      actual_up_velocity.dot(desired_up) + actual_up.dot(desired_up_velocity);
    gradient[column] =
      (cosine * sine_velocity - sine * cosine_velocity) / angle_norm_squared;
  }
  if (!gradient.allFinite()) {
    return std::nullopt;
  }
  return gradient;
}

std::optional<CompactOrientationGuideBounds> compactOrientationGuideBounds(
  const double maximum_abs_pitch_rad, const double maximum_roll_error_rad,
  const double minimum_horizontal_projection_norm)
{
  constexpr double kPi = 3.14159265358979323846;
  if (!std::isfinite(maximum_abs_pitch_rad) || maximum_abs_pitch_rad < 0.0 ||
    maximum_abs_pitch_rad > 0.5 * kPi || !std::isfinite(maximum_roll_error_rad) ||
    maximum_roll_error_rad < 0.0 || maximum_roll_error_rad > kPi ||
    !validProjectionLimit(minimum_horizontal_projection_norm))
  {
    return std::nullopt;
  }

  // At and above a half-turn of permitted roll, do not constrain sampling by
  // orientation; the exact compact predicate remains the acceptance test.
  if (maximum_roll_error_rad >= 0.5 * kPi) {
    return CompactOrientationGuideBounds{kPi, 0.5 * kPi};
  }

  const double pitch_limit = std::min(
    maximum_abs_pitch_rad, std::acos(minimum_horizontal_projection_norm));
  const double cosine_pitch = std::max(1.0e-8, std::cos(pitch_limit));
  const double tangent_pitch = std::tan(pitch_limit);
  const double tangent_roll = std::tan(maximum_roll_error_rad);
  const double x_tolerance = std::atan(std::sqrt(
      tangent_pitch * tangent_pitch +
        (tangent_roll / cosine_pitch) * (tangent_roll / cosine_pitch)));
  const double y_sine_bound = std::sqrt(
    std::pow(std::sin(pitch_limit) * std::cos(maximum_roll_error_rad), 2) +
    std::pow(std::sin(maximum_roll_error_rad), 2));
  if (!std::isfinite(x_tolerance) || !std::isfinite(y_sine_bound) || y_sine_bound >= 1.0) {
    return std::nullopt;
  }
  const double y_tolerance = std::asin(y_sine_bound);
  if (!std::isfinite(y_tolerance)) {
    return std::nullopt;
  }
  return CompactOrientationGuideBounds{x_tolerance, y_tolerance};
}

}  // namespace face_tracking_arm::control
