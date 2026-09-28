// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/tracking_velocity_task.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include "face_tracking_arm/pointing_geometry.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kGeometryEpsilon = 1.0e-6;
constexpr double kRotationTolerance = 1.0e-6;

bool positive_finite(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool nonnegative_finite(const double value)
{
  return std::isfinite(value) && value >= 0.0;
}

bool valid_rotation(const Eigen::Matrix3d & rotation)
{
  return rotation.allFinite() &&
         (rotation.transpose() * rotation).isApprox(
    Eigen::Matrix3d::Identity(), kRotationTolerance) &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

bool valid_config(const TrackingVelocityTaskConfig & config)
{
  return positive_finite(config.position_gain) &&
         positive_finite(config.orientation_gain) &&
         nonnegative_finite(config.position_deadband_m) &&
         nonnegative_finite(config.pointing_deadband_rad) &&
         nonnegative_finite(config.roll_deadband_rad) &&
         positive_finite(config.maximum_linear_reference_mps) &&
         positive_finite(config.maximum_angular_reference_radps) &&
         positive_finite(config.roll_weight);
}

Eigen::Matrix3d skew(const Eigen::Vector3d & vector)
{
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(),
    vector.z(), 0.0, -vector.x(),
    -vector.y(), vector.x(), 0.0;
  return result;
}

double outside_deadband(const double value, const double deadband)
{
  return std::copysign(std::max(0.0, std::abs(value) - deadband), value);
}

Eigen::Vector3d outside_deadband(const Eigen::Vector3d & value, const double deadband)
{
  const double norm = value.norm();
  return norm > deadband ? Eigen::Vector3d(value * ((norm - deadband) / norm)) :
         Eigen::Vector3d::Zero();
}

Eigen::Vector3d limit_norm(const Eigen::Vector3d & value, const double maximum)
{
  const double norm = value.norm();
  return norm > maximum ? Eigen::Vector3d(value * (maximum / norm)) : value;
}

}  // namespace

std::optional<TrackingVelocityTaskResult> makeTrackingVelocityTask(
  const Eigen::Isometry3d & current,
  const Eigen::MatrixXd & jacobian,
  const Eigen::Vector3d & target_position,
  const std::optional<Eigen::Vector3d> & face_position,
  const Eigen::Matrix3d & target_rotation,
  const TrackingVelocityTaskConfig & config,
  const Eigen::Vector3d & face_velocity,
  const Eigen::Vector3d & target_position_velocity,
  const std::optional<GazeKinematics> & gaze)
{
  if (!valid_config(config) || !current.matrix().allFinite() ||
    !valid_rotation(current.rotation()) || !target_position.allFinite() ||
    jacobian.rows() != 6 || jacobian.cols() == 0 || !jacobian.allFinite() ||
    !face_velocity.allFinite() || !target_position_velocity.allFinite())
  {
    return std::nullopt;
  }

  Eigen::Matrix3d desired_rotation = target_rotation;
  double face_distance = 0.0;
  double up_projection_norm = 1.0;
  double roll_authority = 1.0;
  if (face_position.has_value()) {
    if (!face_position->allFinite()) {
      return std::nullopt;
    }
    const Eigen::Vector3d ray = *face_position - current.translation();
    face_distance = ray.norm();
    if (!std::isfinite(face_distance) || face_distance <= kGeometryEpsilon) {
      return std::nullopt;
    }
    const Eigen::Vector3d direction = ray / face_distance;
    up_projection_norm = std::sqrt(std::max(0.0, 1.0 - direction.z() * direction.z()));
    // Upright roll becomes unobservable at a vertical ray. Fade that preference
    // out rather than dropping a perfectly valid pointing target or amplifying noise.
    roll_authority = std::min(1.0, up_projection_norm / 0.10);
    desired_rotation = *pointingRotation(ray);
  } else if (!valid_rotation(desired_rotation)) {
    return std::nullopt;
  }

  const Eigen::Vector3d normal = desired_rotation.col(0);
  const Eigen::Vector3d left = desired_rotation.col(1);
  const Eigen::Vector3d up = desired_rotation.col(2);
  const Eigen::Quaterniond pointing_rotation = Eigen::Quaterniond::FromTwoVectors(
    current.rotation().col(0), normal);
  const Eigen::AngleAxisd pointing_angle_axis(pointing_rotation);
  const Eigen::Vector3d pointing_error =
    pointing_angle_axis.axis() * pointing_angle_axis.angle();
  // Compare roll after the shortest pointing correction so pointing error is
  // not counted a second time as screen roll, including opposite screen normals.
  const Eigen::Vector3d transported_up = pointing_rotation * current.rotation().col(2);
  const double signed_roll_error = std::atan2(
    normal.dot(transported_up.cross(up)), transported_up.dot(up));
  const Eigen::Vector3d position_error = target_position - current.translation();

  TrackingVelocityTaskResult result;
  result.position_error_m = position_error.norm();
  result.pointing_error_rad = pointing_angle_axis.angle();
  result.roll_error_rad = std::abs(signed_roll_error);
  if (!std::isfinite(result.position_error_m) || !pointing_error.allFinite() ||
    !std::isfinite(signed_roll_error))
  {
    return std::nullopt;
  }
  const Eigen::Vector3d pointing_reference = config.orientation_gain *
    outside_deadband(pointing_error, config.pointing_deadband_rad);
  Eigen::Vector3d angular_reference;
  angular_reference << left.dot(pointing_reference), up.dot(pointing_reference),
    config.orientation_gain * outside_deadband(signed_roll_error, config.roll_deadband_rad);
  if (face_position.has_value()) {
    // Face motion rotates the desired frame in the opposite direction from
    // equal TCP translation. Its upright-frame spin also contributes to roll.
    const Eigen::Vector3d desired_angular_velocity =
      skew(normal) * face_velocity / face_distance + normal *
      (normal.z() / (std::max(0.10, up_projection_norm) * face_distance) *
      left.dot(face_velocity));
    angular_reference += Eigen::Vector3d(
      left.dot(desired_angular_velocity), up.dot(desired_angular_velocity),
      normal.dot(desired_angular_velocity));
  }
  const double pointing_speed = angular_reference.head<2>().norm();
  if (pointing_speed > config.maximum_angular_reference_radps) {
    angular_reference.head<2>() *= config.maximum_angular_reference_radps / pointing_speed;
  }
  angular_reference[2] = roll_authority * std::clamp(angular_reference[2],
    -config.maximum_angular_reference_radps, config.maximum_angular_reference_radps);
  const Eigen::Vector3d linear_reference = limit_norm(
    config.position_gain * outside_deadband(position_error, config.position_deadband_m) +
    target_position_velocity,
    config.maximum_linear_reference_mps);
  result.accepted = result.position_error_m <= config.position_deadband_m &&
    result.pointing_error_rad <= config.pointing_deadband_rad &&
    (roll_authority <= kGeometryEpsilon || result.roll_error_rad <= config.roll_deadband_rad) &&
    angular_reference.isZero(0.0) && linear_reference.isZero(0.0);

  const auto linear_jacobian = jacobian.topRows(3);
  const auto angular_jacobian = jacobian.bottomRows(3);
  Eigen::MatrixXd relative_angular_jacobian = angular_jacobian;
  if (face_position.has_value()) {
    // d_dot = -(I-dd') Jv qdot / distance. The desired frame rotates with
    // angular velocity d x d_dot plus its upright-frame spin about d.
    // Subtracting that motion from Jw gives the positive translation terms.
    relative_angular_jacobian += skew(normal) * linear_jacobian / face_distance;
    relative_angular_jacobian += normal *
      (normal.z() / (std::max(0.10, up_projection_norm) * face_distance) *
      left.transpose() * linear_jacobian);
  }

  result.task.primary_matrix.resize(2, jacobian.cols());
  result.task.primary_matrix.row(0) = left.transpose() * relative_angular_jacobian;
  result.task.primary_matrix.row(1) = up.transpose() * relative_angular_jacobian;
  result.task.primary_reference = angular_reference.head<2>();
  result.task.primary_weights = Eigen::Vector2d::Ones();
  result.task.secondary_matrix.resize(4, jacobian.cols());
  result.task.secondary_matrix.topRows(3) = linear_jacobian;
  result.task.secondary_matrix.row(3) =
    roll_authority * normal.transpose() * relative_angular_jacobian;
  result.task.secondary_reference.resize(4);
  result.task.secondary_reference.head<3>() = linear_reference;
  result.task.secondary_reference[3] = angular_reference[2];
  result.task.secondary_weights = Eigen::Vector4d(1.0, 1.0, 1.0, config.roll_weight);
  if (!result.task.primary_matrix.allFinite() || !result.task.secondary_matrix.allFinite() ||
    !angular_reference.allFinite() ||
    !linear_reference.allFinite())
  {
    return std::nullopt;
  }
  if (gaze && face_position) {
    if (gaze->jacobian.cols() != jacobian.cols()) {
      return std::nullopt;
    }
    const auto pointing = makeTrackingVelocityTask(
      gaze->pose, gaze->jacobian, gaze->pose.translation(), face_position,
      target_rotation, config, face_velocity);
    if (!pointing) {
      return std::nullopt;
    }
    result.task.primary_matrix = pointing->task.primary_matrix;
    result.task.primary_reference = pointing->task.primary_reference;
    result.task.primary_weights = pointing->task.primary_weights;
    result.pointing_error_rad = pointing->pointing_error_rad;
    result.accepted = result.position_error_m <= config.position_deadband_m &&
      result.pointing_error_rad <= config.pointing_deadband_rad &&
      (roll_authority <= kGeometryEpsilon || result.roll_error_rad <= config.roll_deadband_rad) &&
      result.task.primary_reference.isZero(0.0) && result.task.secondary_reference.isZero(0.0);
  }
  return result;
}

std::optional<HierarchicalVelocityTask> makeJointPathVelocityTask(
  const Eigen::VectorXd & current_position,
  const Eigen::VectorXd & waypoint,
  const double gain,
  const double maximum_velocity_radps,
  const double deadband_rad)
{
  if (current_position.size() == 0 || current_position.size() != waypoint.size() ||
    !current_position.allFinite() || !waypoint.allFinite() || !positive_finite(gain) ||
    !positive_finite(maximum_velocity_radps) || !nonnegative_finite(deadband_rad))
  {
    return std::nullopt;
  }
  const Eigen::VectorXd error = waypoint - current_position;
  const double maximum_error = error.cwiseAbs().maxCoeff();
  if (!std::isfinite(maximum_error)) {
    return std::nullopt;
  }
  Eigen::VectorXd reference = Eigen::VectorXd::Zero(error.size());
  if (maximum_error > deadband_rad) {
    const double maximum_speed = std::min(
      gain * (maximum_error - deadband_rad), maximum_velocity_radps);
    reference = error * (maximum_speed / maximum_error);
  }
  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(error.size(), error.size());
  task.primary_reference = reference;
  task.primary_weights = Eigen::VectorXd::Ones(error.size());
  task.secondary_matrix.resize(0, error.size());
  task.secondary_reference.resize(0);
  task.secondary_weights.resize(0);
  return task;
}

}  // namespace face_tracking_arm::control
