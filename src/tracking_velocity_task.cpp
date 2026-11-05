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
  const bool distance_disabled = std::isnan(config.face_distance_min_m) &&
    std::isnan(config.face_distance_max_m);
  const bool distance_enabled = std::isfinite(config.face_distance_min_m) &&
    std::isfinite(config.face_distance_max_m) && config.face_distance_min_m > 0.0 &&
    config.face_distance_max_m > config.face_distance_min_m;
  return (distance_disabled || distance_enabled) &&
         positive_finite(config.position_gain) &&
         positive_finite(config.orientation_gain) &&
         nonnegative_finite(config.position_deadband_m) &&
         nonnegative_finite(config.pointing_deadband_rad) &&
         nonnegative_finite(config.roll_deadband_rad) &&
         positive_finite(config.maximum_linear_reference_mps) &&
         positive_finite(config.control_period_sec) &&
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

double signed_roll_error(
  const Eigen::Vector3d & normal,
  const Eigen::Vector3d & actual_up,
  const Eigen::Vector3d & desired_up)
{
  return std::atan2(
    normal.dot(actual_up.cross(desired_up)), actual_up.dot(desired_up));
}

Eigen::Vector3d outside_deadband(const Eigen::Vector3d & value, const double deadband)
{
  const double norm = value.norm();
  return norm > deadband ? Eigen::Vector3d(value * ((norm - deadband) / norm)) :
         Eigen::Vector3d::Zero();
}

Eigen::Vector2d outside_deadband(const Eigen::Vector2d & value, const double deadband)
{
  const double norm = value.norm();
  return norm > deadband ? Eigen::Vector2d(value * ((norm - deadband) / norm)) :
         Eigen::Vector2d::Zero();
}

Eigen::Vector3d limit_norm(const Eigen::Vector3d & value, const double maximum)
{
  const double norm = value.norm();
  return norm > maximum ? Eigen::Vector3d(value * (maximum / norm)) : value;
}

struct TrackingGeometry
{
  Eigen::Vector3d normal{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d left{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d up{Eigen::Vector3d::UnitZ()};
  Eigen::MatrixXd relative_angular_jacobian;
  Eigen::MatrixXd primary_matrix;
  Eigen::VectorXd primary_reference = Eigen::Vector2d::Zero();
  Eigen::Vector3d signed_primary_error{Eigen::Vector3d::Zero()};
  Eigen::Vector3d primary_feedforward{Eigen::Vector3d::Zero()};
  double pointing_error_rad{0.0};
  double signed_roll_error{0.0};
  double roll_authority{1.0};
  double roll_reference{0.0};
};

std::optional<TrackingGeometry> computeTrackingGeometry(
  const Eigen::Isometry3d & current,
  const Eigen::MatrixXd & jacobian,
  const std::optional<Eigen::Vector3d> & face_position,
  const Eigen::Matrix3d & target_rotation,
  const TrackingVelocityTaskConfig & config,
  const Eigen::Vector3d & face_velocity,
  const bool compute_primary,
  const bool include_roll_in_primary)
{
  if (!current.matrix().allFinite() || !valid_rotation(current.rotation()) ||
    jacobian.rows() != 6 || jacobian.cols() == 0 || !jacobian.allFinite() ||
    !face_velocity.allFinite())
  {
    return std::nullopt;
  }

  Eigen::Matrix3d desired_rotation = target_rotation;
  double face_distance = 0.0;
  double up_projection_norm = 1.0;
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
    desired_rotation = *pointingRotation(ray);
  } else if (!valid_rotation(desired_rotation)) {
    return std::nullopt;
  }

  TrackingGeometry result;
  result.normal = desired_rotation.col(0);
  result.left = desired_rotation.col(1);
  result.up = desired_rotation.col(2);
  const Eigen::Quaterniond pointing_rotation = Eigen::Quaterniond::FromTwoVectors(
    current.rotation().col(0), result.normal);
  const Eigen::AngleAxisd pointing_angle_axis(pointing_rotation);
  const Eigen::Vector3d pointing_error =
    pointing_angle_axis.axis() * pointing_angle_axis.angle();
  const Eigen::Vector3d transported_up = pointing_rotation * current.rotation().col(2);
  result.signed_roll_error = signed_roll_error(
    result.normal, transported_up, result.up);
  result.pointing_error_rad = pointing_angle_axis.angle();
  result.signed_primary_error <<
    result.left.dot(pointing_error), result.up.dot(pointing_error), result.signed_roll_error;
  result.roll_authority = std::min(1.0, up_projection_norm / 0.10);

  Eigen::Vector3d desired_angular_velocity = Eigen::Vector3d::Zero();
  if (face_position.has_value()) {
    desired_angular_velocity =
      skew(result.normal) * face_velocity / face_distance + result.normal *
      (result.normal.z() / (std::max(0.10, up_projection_norm) * face_distance) *
      result.left.dot(face_velocity));
  }
  result.primary_feedforward << result.left.dot(desired_angular_velocity),
    result.up.dot(desired_angular_velocity),
    result.roll_authority * result.normal.dot(desired_angular_velocity);
  result.roll_reference = result.roll_authority * std::clamp(
    config.orientation_gain * outside_deadband(
      result.signed_roll_error, config.roll_deadband_rad) +
    result.normal.dot(desired_angular_velocity),
    -config.maximum_angular_reference_radps, config.maximum_angular_reference_radps);

  result.relative_angular_jacobian = jacobian.bottomRows(3);
  if (face_position.has_value()) {
    const auto linear_jacobian = jacobian.topRows(3);
    result.relative_angular_jacobian +=
      skew(result.normal) * linear_jacobian / face_distance;
    result.relative_angular_jacobian += result.normal *
      (result.normal.z() / (std::max(0.10, up_projection_norm) * face_distance) *
      result.left.transpose() * linear_jacobian);
  }

  if (compute_primary) {
    const Eigen::Vector2d pointing_error_2d(
      result.left.dot(pointing_error), result.up.dot(pointing_error));
    const Eigen::Vector2d pointing_feedforward_2d(
      result.left.dot(desired_angular_velocity), result.up.dot(desired_angular_velocity));
    Eigen::Vector2d reference = outside_deadband(Eigen::Vector2d(
        config.orientation_gain * pointing_error_2d + pointing_feedforward_2d),
      config.orientation_gain * config.pointing_deadband_rad);
    const double pointing_speed = reference.norm();
    if (pointing_speed > config.maximum_angular_reference_radps) {
      reference *= config.maximum_angular_reference_radps / pointing_speed;
    }
    const Eigen::Index primary_rows = include_roll_in_primary ? 3 : 2;
    result.primary_matrix.resize(primary_rows, jacobian.cols());
    result.primary_matrix.row(0) = result.left.transpose() * result.relative_angular_jacobian;
    result.primary_matrix.row(1) = result.up.transpose() * result.relative_angular_jacobian;
    result.primary_reference.resize(primary_rows);
    result.primary_reference.head<2>() = reference;
    if (include_roll_in_primary) {
      result.primary_matrix.row(2) = result.roll_authority * result.normal.transpose() *
        result.relative_angular_jacobian;
      result.primary_reference[2] = result.roll_reference;
    }
  }

  if (!result.relative_angular_jacobian.allFinite() ||
    !result.signed_primary_error.allFinite() || !result.primary_feedforward.allFinite() ||
    (compute_primary &&
    (!result.primary_matrix.allFinite() || !result.primary_reference.allFinite())) ||
    !std::isfinite(result.pointing_error_rad) || !std::isfinite(result.signed_roll_error) ||
    !std::isfinite(result.roll_reference))
  {
    return std::nullopt;
  }
  return result;
}

}  // namespace

std::optional<HierarchicalVelocityTask> makeFaceEdgeOnlyRecoveryTask(
  const HierarchicalVelocityTask & face_task, const bool has_face_primary_roll)
{
  const Eigen::Index joints = face_task.primary_matrix.cols();
  if (!has_face_primary_roll || joints <= 0 || face_task.primary_matrix.rows() != 3 ||
    face_task.primary_reference.size() != 3 || face_task.primary_weights.size() != 3 ||
    !face_task.primary_matrix.allFinite() || !face_task.primary_reference.allFinite() ||
    !face_task.primary_weights.allFinite() || (face_task.primary_weights.array() <= 0.0).any())
  {
    return std::nullopt;
  }

  HierarchicalVelocityTask recovery;
  recovery.primary_matrix = face_task.primary_matrix.row(2);
  recovery.primary_reference = Eigen::VectorXd::Constant(1, face_task.primary_reference[2]);
  recovery.primary_weights = Eigen::VectorXd::Constant(1, face_task.primary_weights[2]);
  recovery.secondary_matrix = Eigen::MatrixXd::Identity(joints, joints);
  recovery.secondary_reference = Eigen::VectorXd::Zero(joints);
  recovery.secondary_weights = Eigen::VectorXd::Ones(joints);
  return recovery;
}

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

  const bool use_gaze_primary = gaze.has_value() && face_position.has_value();
  const auto screen_geometry = computeTrackingGeometry(
    current, jacobian, face_position, target_rotation, config, face_velocity,
    !use_gaze_primary, !use_gaze_primary && face_position.has_value());
  if (!screen_geometry) {
    return std::nullopt;
  }

  const TrackingGeometry * primary_geometry = &*screen_geometry;
  std::optional<TrackingGeometry> gaze_geometry;
  std::optional<HorizontalEdgePathTaskResult> screen_horizontal_edge;
  if (use_gaze_primary) {
    if (gaze->jacobian.cols() != jacobian.cols()) {
      return std::nullopt;
    }
    gaze_geometry = computeTrackingGeometry(
      gaze->pose, gaze->jacobian, face_position, target_rotation, config, face_velocity,
      true, false);
    if (!gaze_geometry) {
      return std::nullopt;
    }
    primary_geometry = &*gaze_geometry;
  }
  if (face_position && config.horizontal_edge_primary) {
    screen_horizontal_edge = makeHorizontalEdgePathTask(
      current, jacobian, Eigen::VectorXd::Zero(jacobian.cols()), config);
    if (!screen_horizontal_edge) {
      return std::nullopt;
    }
  }

  const Eigen::Vector3d position_error = target_position - current.translation();

  TrackingVelocityTaskResult result;
  result.position_error_m = position_error.norm();
  result.pointing_error_rad = primary_geometry->pointing_error_rad;
  result.roll_error_rad = face_position && screen_horizontal_edge ?
    std::abs(screen_horizontal_edge->edge_tilt_error_rad) :
    std::abs(screen_geometry->signed_roll_error);
  result.signed_face_primary_error_rad = primary_geometry->signed_primary_error;
  result.face_primary_feedforward_radps = primary_geometry->primary_feedforward;
  result.has_face_primary_roll = face_position.has_value();
  if (face_position) {
    if (screen_horizontal_edge) {
      result.signed_face_primary_error_rad[2] = screen_horizontal_edge->edge_tilt_error_rad;
      // The edge-height task depends only on the current screen rotation.
      result.face_primary_feedforward_radps[2] = 0.0;
    } else {
      // Keep the existing finite face-ray fallback when the edge task is not
      // defined; the fallback is not treated as a valid edge-upright measure.
      result.signed_face_primary_error_rad[2] = screen_geometry->signed_roll_error;
      result.face_primary_feedforward_radps[2] =
        screen_geometry->primary_feedforward[2];
    }
  }
  if (!std::isfinite(result.position_error_m)) {
    return std::nullopt;
  }
  const Eigen::Vector3d linear_reference = limit_norm(
    config.position_gain * outside_deadband(position_error, config.position_deadband_m) +
    target_position_velocity,
    config.maximum_linear_reference_mps);

  const auto linear_jacobian = jacobian.topRows(3);
  const bool face_distance_enabled = face_position.has_value() &&
    std::isfinite(config.face_distance_min_m) && std::isfinite(config.face_distance_max_m);
  if (face_distance_enabled) {
    const Eigen::Vector3d monitor_from_face = current.translation() - *face_position;
    const double face_distance_m = monitor_from_face.norm();
    if (!std::isfinite(face_distance_m) || face_distance_m <= kGeometryEpsilon) {
      return std::nullopt;
    }
    const Eigen::Vector3d radial = monitor_from_face / face_distance_m;
    result.face_distance_matrix.resize(1, jacobian.cols());
    result.face_distance_matrix.row(0) = radial.transpose() * linear_jacobian;
    const double face_radial_velocity = radial.dot(face_velocity);
    const double lower_rate =
      (config.face_distance_min_m - face_distance_m) * config.position_gain +
      face_radial_velocity;
    const double upper_rate =
      (config.face_distance_max_m - face_distance_m) * config.position_gain +
      face_radial_velocity;
    LinearVelocityConstraint interval;
    interval.coefficients = result.face_distance_matrix.row(0).transpose();
    interval.lower_bound = lower_rate;
    interval.upper_bound = upper_rate;
    result.face_distance_outside_band =
      face_distance_m<config.face_distance_min_m ||
        face_distance_m> config.face_distance_max_m;
    result.face_distance_interval = std::move(interval);
    const double correction_error = face_distance_m < config.face_distance_min_m ?
      config.face_distance_min_m - face_distance_m :
      (face_distance_m > config.face_distance_max_m ?
      config.face_distance_max_m - face_distance_m : 0.0);
    const double distance_reference = result.face_distance_outside_band ?
      config.position_gain * correction_error + face_radial_velocity : 0.0;
    result.face_distance_reference = Eigen::VectorXd::Constant(1, std::clamp(
        distance_reference, -config.maximum_linear_reference_mps,
        config.maximum_linear_reference_mps));
    result.position_error_m = std::max(
      0.0, std::max(config.face_distance_min_m - face_distance_m,
      face_distance_m - config.face_distance_max_m));
  }
  result.task.primary_matrix = primary_geometry->primary_matrix;
  result.task.primary_reference = primary_geometry->primary_reference;
  if (face_position) {
    result.task.primary_matrix.resize(3, jacobian.cols());
    result.task.primary_matrix.topRows(2) = primary_geometry->primary_matrix.topRows(2);
    if (screen_horizontal_edge) {
      result.task.primary_matrix.row(2) =
        screen_horizontal_edge->task.primary_matrix.row(0);
    } else {
      result.task.primary_matrix.row(2) = screen_geometry->roll_authority *
        screen_geometry->normal.transpose() * screen_geometry->relative_angular_jacobian;
    }
    result.task.primary_reference.resize(3);
    result.task.primary_reference.head<2>() = primary_geometry->primary_reference.head<2>();
    result.task.primary_reference[2] = screen_horizontal_edge ?
      screen_horizontal_edge->task.primary_reference[0] : screen_geometry->roll_reference;
    result.task.primary_weights = Eigen::Vector3d(1.0, 1.0, config.roll_weight);
    if (face_distance_enabled) {
      result.task.secondary_matrix.resize(0, jacobian.cols());
      result.task.secondary_reference.resize(0);
      result.task.secondary_weights.resize(0);
      if (result.face_distance_outside_band) {
        const Eigen::Vector3d face_radial =
          (current.translation() - *face_position).normalized();
        Eigen::Vector3d follow_velocity = Eigen::Vector3d::Zero();
        const Eigen::Vector2d first_joint_pointing_jacobian =
          primary_geometry->primary_matrix.topRows(2).col(0);
        const Eigen::Vector2d pointing_reference =
          primary_geometry->primary_reference.head<2>();
        const Eigen::Vector3d first_joint_angular_jacobian =
          jacobian.bottomRows(3).col(0);
        const double first_joint_angular_speed = first_joint_angular_jacobian.norm();
        const double pointing_authority_squared =
          first_joint_pointing_jacobian.squaredNorm();
        if (!pointing_reference.isZero(0.0) &&
          std::isfinite(first_joint_angular_speed) &&
          first_joint_angular_speed > std::sqrt(std::numeric_limits<double>::epsilon()) &&
          std::isfinite(pointing_authority_squared) &&
          pointing_authority_squared > std::numeric_limits<double>::epsilon())
        {
          const double least_squares_rate =
            first_joint_pointing_jacobian.dot(pointing_reference) /
            pointing_authority_squared;
          const double maximum_joint_rate =
            config.maximum_angular_reference_radps / first_joint_angular_speed;
          if (std::isfinite(least_squares_rate) && std::isfinite(maximum_joint_rate)) {
            const double follow_joint_rate = std::clamp(
              least_squares_rate, -maximum_joint_rate, maximum_joint_rate);
            const Eigen::Vector3d candidate = linear_jacobian.col(0) * follow_joint_rate;
            if (candidate.allFinite()) {
              follow_velocity = candidate;
            }
          }
        }

        const Eigen::Matrix3d tangent_projection =
          Eigen::Matrix3d::Identity() - face_radial * face_radial.transpose();
        result.task.distance_matrix = linear_jacobian;
        result.task.distance_reference = face_radial * result.face_distance_reference[0] +
          tangent_projection * follow_velocity;
        result.task.distance_weights = Eigen::VectorXd::Ones(3);
      } else {
        result.task.distance_matrix.resize(0, jacobian.cols());
        result.task.distance_reference.resize(0);
        result.task.distance_weights.resize(0);
      }
    } else {
      result.task.secondary_matrix = linear_jacobian;
      result.task.secondary_reference = linear_reference;
      result.task.secondary_weights = Eigen::Vector3d::Ones();
    }
  } else {
    result.task.primary_weights = Eigen::Vector2d::Ones();
    result.task.secondary_matrix.resize(4, jacobian.cols());
    result.task.secondary_matrix.topRows(3) = linear_jacobian;
    result.task.secondary_matrix.row(3) = screen_geometry->roll_authority *
      screen_geometry->normal.transpose() * screen_geometry->relative_angular_jacobian;
    result.task.secondary_reference.resize(4);
    result.task.secondary_reference.head<3>() = linear_reference;
    result.task.secondary_reference[3] = screen_geometry->roll_reference;
    result.task.secondary_weights = Eigen::Vector4d(1.0, 1.0, 1.0, config.roll_weight);
  }
  if (!result.task.primary_matrix.allFinite() || !result.task.secondary_matrix.allFinite() ||
    !linear_reference.allFinite())
  {
    return std::nullopt;
  }
  const bool roll_is_observable = face_position ? screen_horizontal_edge.has_value() :
    screen_geometry->roll_authority > kGeometryEpsilon;
  result.accepted = (face_distance_enabled ? result.position_error_m == 0.0 :
    result.position_error_m <= config.position_deadband_m) &&
    result.pointing_error_rad <= config.pointing_deadband_rad &&
    (!roll_is_observable || result.roll_error_rad <= config.roll_deadband_rad) &&
    result.task.primary_reference.isZero(0.0) && result.task.secondary_reference.isZero(0.0) &&
    result.task.distance_reference.isZero(0.0);
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

std::optional<UprightRollPathTaskResult> makeUprightRollPathTask(
  const Eigen::Isometry3d & current_screen_pose,
  const Eigen::MatrixXd & screen_jacobian,
  const Eigen::VectorXd & path_velocity_reference,
  const TrackingVelocityTaskConfig & config)
{
  constexpr double kMinimumUprightProjection = 0.10;
  if (!valid_config(config) || !current_screen_pose.matrix().allFinite() ||
    !valid_rotation(current_screen_pose.rotation()) || screen_jacobian.rows() != 6 ||
    screen_jacobian.cols() == 0 || !screen_jacobian.allFinite() ||
    path_velocity_reference.size() != screen_jacobian.cols() ||
    !path_velocity_reference.allFinite())
  {
    return std::nullopt;
  }

  const Eigen::Vector3d normal = current_screen_pose.rotation().col(0);
  const double upright_projection_norm = std::sqrt(std::max(
      0.0, 1.0 - normal.z() * normal.z()));
  if (!std::isfinite(upright_projection_norm) ||
    upright_projection_norm < kMinimumUprightProjection)
  {
    return std::nullopt;
  }

  // Reuse the package's +X-forward, gravity-upright screen frame definition.
  const auto upright_frame = face_tracking_arm::pointingRotation(normal);
  if (!upright_frame) {
    return std::nullopt;
  }
  const Eigen::Vector3d desired_up = upright_frame->col(2);
  const auto measured_roll_error = measureUprightRollError(current_screen_pose);
  if (!measured_roll_error) {
    return std::nullopt;
  }
  const double roll_error = *measured_roll_error;

  // As the screen normal moves, the gravity-upright frame also spins about
  // that normal. Remove that induced spin so this row controls roll only.
  // This is the projected-up kinematics used by FACE tracking, without a face
  // ray or target-motion feed-forward.
  const Eigen::RowVector3d relative_roll_angular_row =
    normal.transpose() -
    (normal.z() / upright_projection_norm) * desired_up.transpose();

  UprightRollPathTaskResult result;
  result.roll_error_rad = roll_error;
  result.task.primary_matrix.resize(1, screen_jacobian.cols());
  result.task.primary_matrix.row(0) =
    relative_roll_angular_row * screen_jacobian.bottomRows(3);
  result.task.primary_reference.resize(1);
  result.task.primary_reference[0] = std::clamp(
    config.orientation_gain * outside_deadband(roll_error, config.roll_deadband_rad),
    -config.maximum_angular_reference_radps, config.maximum_angular_reference_radps);
  result.task.primary_weights = Eigen::VectorXd::Ones(1);

  // Keep XYZ and yaw out of the primary task. The supplied q-path command is
  // an ordinary secondary objective and cannot trade away roll priority.
  result.task.secondary_matrix = Eigen::MatrixXd::Identity(
    screen_jacobian.cols(), screen_jacobian.cols());
  result.task.secondary_reference = path_velocity_reference;
  result.task.secondary_weights = Eigen::VectorXd::Ones(screen_jacobian.cols());
  if (!result.task.primary_matrix.allFinite() ||
    !result.task.primary_reference.allFinite())
  {
    return std::nullopt;
  }
  return result;
}

std::optional<HorizontalEdgePathTaskResult> makeHorizontalEdgePathTask(
  const Eigen::Isometry3d & current_screen_pose,
  const Eigen::MatrixXd & screen_jacobian,
  const Eigen::VectorXd & path_velocity_reference,
  const TrackingVelocityTaskConfig & config)
{
  if (!valid_config(config) || !current_screen_pose.matrix().allFinite() ||
    screen_jacobian.rows() != 6 ||
    screen_jacobian.cols() == 0 || !screen_jacobian.allFinite() ||
    path_velocity_reference.size() != screen_jacobian.cols() ||
    !path_velocity_reference.allFinite())
  {
    return std::nullopt;
  }
  const Eigen::Matrix3d rotation = current_screen_pose.rotation();
  if (!validEdgeRotation(rotation) || rotation(2, 2) < 0.0) {
    return std::nullopt;
  }
  const double edge_tilt_rad = std::asin(std::clamp(rotation(2, 1), -1.0, 1.0));
  const Eigen::RowVectorXd sine_gradient =
    (rotation.col(1).cross(Eigen::Vector3d::UnitZ())).transpose() *
    screen_jacobian.bottomRows(3);
  const double edge_deadband_sine = std::sin(std::min(
    config.roll_deadband_rad, 0.5 * std::acos(-1.0)));

  HorizontalEdgePathTaskResult result;
  result.signed_edge_sine = rotation(2, 1);
  result.edge_tilt_error_rad = -edge_tilt_rad;
  result.task.primary_matrix.resize(1, screen_jacobian.cols());
  result.task.primary_matrix.row(0) = sine_gradient;
  result.task.primary_reference.resize(1);
  result.task.primary_reference[0] = std::clamp(
    config.orientation_gain * outside_deadband(-rotation(2, 1), edge_deadband_sine),
    -config.maximum_angular_reference_radps, config.maximum_angular_reference_radps);
  result.task.primary_weights = Eigen::VectorXd::Ones(1);
  result.task.secondary_matrix = Eigen::MatrixXd::Identity(
    screen_jacobian.cols(), screen_jacobian.cols());
  result.task.secondary_reference = path_velocity_reference;
  result.task.secondary_weights = Eigen::VectorXd::Ones(screen_jacobian.cols());
  if (!result.task.primary_matrix.allFinite() || !result.task.primary_reference.allFinite()) {
    return std::nullopt;
  }
  return result;
}

std::optional<double> measureHorizontalEdgeTiltError(
  const Eigen::Isometry3d & current_screen_pose)
{
  if (!current_screen_pose.matrix().allFinite() ||
    !validEdgeRotation(current_screen_pose.rotation()))
  {
    return std::nullopt;
  }
  return -std::asin(std::clamp(current_screen_pose.rotation()(2, 1), -1.0, 1.0));
}

std::optional<double> measureUprightRollError(
  const Eigen::Isometry3d & current_screen_pose)
{
  constexpr double kMinimumUprightProjection = 0.10;
  if (!current_screen_pose.matrix().allFinite() ||
    !valid_rotation(current_screen_pose.rotation()))
  {
    return std::nullopt;
  }
  const Eigen::Vector3d normal = current_screen_pose.rotation().col(0);
  const Eigen::Vector3d actual_up = current_screen_pose.rotation().col(2);
  const double projection = std::sqrt(std::max(0.0, 1.0 - normal.z() * normal.z()));
  if (!std::isfinite(projection) || projection < kMinimumUprightProjection) {
    return std::nullopt;
  }
  const auto upright_frame = face_tracking_arm::pointingRotation(normal);
  if (!upright_frame) {
    return std::nullopt;
  }
  const double error = signed_roll_error(normal, actual_up, upright_frame->col(2));
  return std::isfinite(error) ? std::optional<double>{error} : std::nullopt;
}

}  // namespace face_tracking_arm::control
