// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/posture_preference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "face_tracking_arm/collision_constraints.hpp"

namespace face_tracking_arm::control
{
PostureBendCalibration calibratePostureBend(
  const double rest_joint3, const double rest_angle, const double search_joint3,
  const double search_angle, const double physical_joint3_lower,
  const double physical_joint3_upper)
{
  PostureBendCalibration result;
  constexpr double kTwoPi = 6.28318530717958647692;
  if (!std::isfinite(rest_joint3) || !std::isfinite(rest_angle) ||
    !std::isfinite(search_joint3) || !std::isfinite(search_angle) ||
    !std::isfinite(physical_joint3_lower) || !std::isfinite(physical_joint3_upper) ||
    physical_joint3_upper <= physical_joint3_lower)
  {
    return result;
  }
  const double offset = std::remainder(rest_angle - rest_joint3, kTwoPi);
  const double rest_sin = std::sin(rest_joint3 + offset);
  const double search_offset_error = std::remainder(
    (search_angle - search_joint3) - offset, kTwoPi);
  const double search_sin = std::sin(search_joint3 + offset);
  if (!std::isfinite(offset) || !std::isfinite(rest_sin) ||
    !std::isfinite(search_offset_error) || std::abs(search_offset_error) > 1.0e-6 ||
    !std::isfinite(search_sin) || rest_sin == 0.0 || search_sin == 0.0 ||
    std::signbit(rest_sin) != std::signbit(search_sin))
  {
    return result;
  }

  const double preferred_sign = std::signbit(rest_sin) ? -1.0 : 1.0;
  const double minimum_sin = std::min(std::abs(rest_sin), std::abs(search_sin));
  const double phase_rest = preferred_sign * (rest_joint3 + offset);
  const double phase_search = preferred_sign * (search_joint3 + offset);
  const double edge = std::asin(minimum_sin);
  const double center = 0.5 * (phase_rest + phase_search);
  const double nearest_turn = std::round((center - 0.5 * 3.14159265358979323846) / kTwoPi);
  double phase_lower = std::numeric_limits<double>::quiet_NaN();
  double phase_upper = std::numeric_limits<double>::quiet_NaN();
  for (int delta = -2; delta <= 2; ++delta) {
    const double turn = nearest_turn + static_cast<double>(delta);
    const double candidate_lower = edge + turn * kTwoPi;
    const double candidate_upper = 3.14159265358979323846 - edge +
      turn * kTwoPi;
    constexpr double kAngleTolerance = 1.0e-9;
    if (phase_rest >= candidate_lower - kAngleTolerance &&
      phase_rest <= candidate_upper + kAngleTolerance &&
      phase_search >= candidate_lower - kAngleTolerance &&
      phase_search <= candidate_upper + kAngleTolerance)
    {
      phase_lower = candidate_lower;
      phase_upper = candidate_upper;
      break;
    }
  }
  if (!std::isfinite(phase_lower) || !std::isfinite(phase_upper)) {
    return result;
  }
  double q_lower = preferred_sign * phase_lower - offset;
  double q_upper = preferred_sign * phase_upper - offset;
  if (q_lower > q_upper) {
    std::swap(q_lower, q_upper);
  }
  q_lower = std::max(q_lower, physical_joint3_lower);
  q_upper = std::min(q_upper, physical_joint3_upper);
  const auto in_interval = [q_lower, q_upper](const double value) {
      return value >= q_lower - 1.0e-9 && value <= q_upper + 1.0e-9;
    };
  if (!std::isfinite(q_lower) || !std::isfinite(q_upper) || q_upper <= q_lower ||
    !in_interval(rest_joint3) || !in_interval(search_joint3))
  {
    return result;
  }
  result.angle_offset_rad = offset;
  result.preferred_sign = preferred_sign;
  result.minimum_comfortable_sin = minimum_sin;
  result.joint3_lower = q_lower;
  result.joint3_upper = q_upper;
  result.valid = true;
  return result;
}

std::optional<LinearVelocityConstraint> makeJoint3ComfortBounds(
  const JointMotionState & motion, const JointMotionLimits & limits,
  const Eigen::Index joint_index, const double lower_position,
  const double upper_position, const double period_sec,
  const double residual_latency_sec)
{
  const Eigen::Index count = motion.position.size();
  if (joint_index < 0 || joint_index >= count || motion.velocity.size() != count ||
    motion.acceleration.size() != count || limits.max_velocity.size() != count ||
    limits.max_acceleration.size() != count || limits.max_jerk.size() != count ||
    !motion.position.allFinite() || !motion.velocity.allFinite() ||
    !motion.acceleration.allFinite() || !std::isfinite(lower_position) ||
    !std::isfinite(upper_position) || upper_position <= lower_position ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0)
  {
    return std::nullopt;
  }

  const double position = motion.position[joint_index];
  const double velocity = motion.velocity[joint_index];
  const double acceleration = motion.acceleration[joint_index];
  const double max_velocity = limits.max_velocity[joint_index];
  const double max_acceleration = limits.max_acceleration[joint_index];
  const double max_jerk = limits.max_jerk[joint_index];
  if (!std::isfinite(max_velocity) || max_velocity <= 0.0 ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
    !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
    position < lower_position || position > upper_position ||
    !limits.numerical_policy || !limits.numerical_policy->valid() ||
    classifyVelocity(RadiansPerSecond{velocity}, RadiansPerSecond{max_velocity},
      limits.numerical_policy->velocity_error) == BoundaryClass::kOutside ||
    classifyAcceleration(RadiansPerSecondSquared{acceleration},
      RadiansPerSecondSquared{max_acceleration},
      limits.numerical_policy->acceleration_error) == BoundaryClass::kOutside)
  {
    return std::nullopt;
  }

  const double bounded_acceleration = std::clamp(
    acceleration, -max_acceleration, max_acceleration);
  const double jerk_step = max_jerk * period_sec;
  const double minimum_acceleration = std::max(
    -max_acceleration, bounded_acceleration - jerk_step);
  const double maximum_acceleration = std::min(
    max_acceleration, bounded_acceleration + jerk_step);
  const double minimum_reachable_velocity = std::max(
    -max_velocity, velocity + minimum_acceleration * period_sec);
  const double maximum_reachable_velocity = std::min(
    max_velocity, velocity + maximum_acceleration * period_sec);
  if (!std::isfinite(minimum_reachable_velocity) ||
    !std::isfinite(maximum_reachable_velocity) ||
    minimum_reachable_velocity > maximum_reachable_velocity)
  {
    return std::nullopt;
  }

  const double upper_velocity = maximumSafeVelocityTowardBoundary(
    upper_position - position, velocity, minimum_reachable_velocity,
    maximum_reachable_velocity, max_acceleration, max_jerk,
    period_sec, residual_latency_sec);
  const double lower_velocity = -maximumSafeVelocityTowardBoundary(
    position - lower_position, -velocity, -maximum_reachable_velocity,
    -minimum_reachable_velocity, max_acceleration, max_jerk,
    period_sec, residual_latency_sec);
  if (!std::isfinite(lower_velocity) || !std::isfinite(upper_velocity) ||
    lower_velocity > upper_velocity)
  {
    return std::nullopt;
  }

  LinearVelocityConstraint constraint;
  constraint.coefficients = Eigen::VectorXd::Zero(count);
  constraint.coefficients[joint_index] = 1.0;
  constraint.lower_bound = lower_velocity;
  constraint.upper_bound = upper_velocity;
  return constraint;
}

PosturePreferenceObjective makePosturePreferenceObjective(
  const Eigen::Vector3d & shoulder, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist,
  const Eigen::MatrixXd & shoulder_jacobian, const Eigen::MatrixXd & elbow_jacobian,
  const Eigen::MatrixXd & wrist_jacobian, const Eigen::VectorXd & joint_position,
  const PosturePreferenceConfig & config)
{
  PosturePreferenceObjective result;
  const Eigen::Index count = joint_position.size();
  if (count <= 0 || !joint_position.allFinite() ||
    config.j3_index >= static_cast<std::size_t>(count) ||
    config.j5_index >= static_cast<std::size_t>(count) ||
    config.j6_index >= static_cast<std::size_t>(count) ||
    !std::isfinite(config.joint3_angle_offset_rad) ||
    !std::isfinite(config.preferred_bend_sign) ||
    std::abs(std::abs(config.preferred_bend_sign) - 1.0) > 1.0e-9 ||
    !std::isfinite(config.comfortable_joint3_lower) ||
    !std::isfinite(config.comfortable_joint3_upper) ||
    config.comfortable_joint3_upper <= config.comfortable_joint3_lower ||
    !std::isfinite(config.position_margin_j5_rad) || config.position_margin_j5_rad < 0.0 ||
    !std::isfinite(config.position_margin_j6_rad) || config.position_margin_j6_rad < 0.0 ||
    !std::isfinite(config.position_gain) || config.position_gain <= 0.0 ||
    !std::isfinite(config.maximum_linear_reference_mps) ||
    config.maximum_linear_reference_mps <= 0.0 ||
    !std::isfinite(config.maximum_angular_reference_radps) ||
    config.maximum_angular_reference_radps <= 0.0 ||
    !std::isfinite(config.control_period_sec) || config.control_period_sec <= 0.0)
  {
    return result;
  }

  std::vector<Eigen::RowVectorXd> rows;
  std::vector<double> references;
  result.joint3_correction.matrix.resize(0, count);
  result.joint3_correction.reference.resize(0);
  result.joint3_correction.weights.resize(0);
  const Eigen::Index j3 = static_cast<Eigen::Index>(config.j3_index);
  const double q3 = joint_position[j3];
  const bool comfortable = q3 >= config.comfortable_joint3_lower &&
    q3 <= config.comfortable_joint3_upper;
  result.branch_comfortable = comfortable;
  const double one_step_reserve = std::min(
    config.maximum_angular_reference_radps * config.control_period_sec,
    0.5 * (config.comfortable_joint3_upper - config.comfortable_joint3_lower));
  if (!comfortable) {
    const bool below_interval = q3 < config.comfortable_joint3_lower;
    const double boundary = below_interval ?
      config.comfortable_joint3_lower : config.comfortable_joint3_upper;
    const double branch_target = boundary + (below_interval ? one_step_reserve : -one_step_reserve);
    Eigen::RowVectorXd branch_row = Eigen::RowVectorXd::Zero(count);
    branch_row[j3] = 1.0;
    const double q3_rate = std::clamp(
      config.position_gain * (branch_target - q3),
      -config.maximum_angular_reference_radps,
      config.maximum_angular_reference_radps);
    result.joint3_correction.matrix = branch_row;
    result.joint3_correction.reference = Eigen::VectorXd::Constant(1, q3_rate);
    result.joint3_correction.weights = Eigen::VectorXd::Ones(1);
  }

  constexpr double kIndeterminateActivation{std::numeric_limits<double>::epsilon()};
  const Eigen::Vector3d line = wrist - shoulder;
  const Eigen::Vector3d upper = Eigen::Vector3d::UnitZ();
  const double line_squared = line.squaredNorm();
  if (shoulder.allFinite() && elbow.allFinite() && wrist.allFinite() &&
    shoulder_jacobian.rows() == 3 && elbow_jacobian.rows() == 3 &&
    wrist_jacobian.rows() == 3 && shoulder_jacobian.cols() == count &&
    elbow_jacobian.cols() == count && wrist_jacobian.cols() == count &&
    shoulder_jacobian.allFinite() && elbow_jacobian.allFinite() &&
    wrist_jacobian.allFinite() && std::isfinite(line_squared))
  {
    if (line_squared <= std::numeric_limits<double>::epsilon()) {
      result.geometry_comfortable = true;
    } else {
      const Eigen::Vector3d offset = elbow - shoulder;
      const double along = offset.dot(line);
      const double up_along = upper.dot(line);
      const auto h = evaluateUpwardBend(shoulder, elbow, wrist);
      const double reserve = config.maximum_linear_reference_mps * config.control_period_sec;
      const double horizontal_squared = line.head<2>().squaredNorm();
      const double forearm_squared = (wrist - elbow).squaredNorm();
      // The horizontal lever arm naturally fades near a vertical or collapsed line.
      const double activation = horizontal_squared / (line_squared + forearm_squared);
      if (std::isfinite(along) && std::isfinite(up_along) && h &&
        std::isfinite(reserve) && std::isfinite(horizontal_squared) &&
        std::isfinite(forearm_squared) && std::isfinite(activation))
      {
        result.geometry_comfortable =
          *h >= reserve || activation <= kIndeterminateActivation;
        if (comfortable && !result.geometry_comfortable && activation > 0.0) {
          const double squared = line_squared;
          const double product = along * up_along;
          const Eigen::Vector3d gradient_offset = upper - line * (up_along / squared);
          const Eigen::Vector3d gradient_line =
            -(offset * up_along + upper * along) / squared +
            line * (2.0 * product / (squared * squared));
          const Eigen::Vector3d gradient_shoulder = -gradient_offset - gradient_line;
          Eigen::RowVectorXd clearance_row(count);
          for (Eigen::Index index = 0; index < count; ++index) {
            clearance_row[index] = activation * (
              gradient_shoulder.dot(shoulder_jacobian.col(index)) +
              gradient_offset.dot(elbow_jacobian.col(index)) +
              gradient_line.dot(wrist_jacobian.col(index)));
          }
          if (clearance_row.allFinite()) {
            const double desired_rate = std::clamp(
              config.position_gain * (reserve - *h),
              0.0, config.maximum_linear_reference_mps);
            rows.push_back(std::move(clearance_row));
            references.push_back(activation * desired_rate);
          }
        }
      }
    }
  }

  const auto add_limit_margin = [&](const std::size_t index, const double lower,
    const double upper, const double margin) {
      if (margin <= 0.0 || !std::isfinite(lower) || !std::isfinite(upper) || upper <= lower) {
        return;
      }
      const double q = joint_position[static_cast<Eigen::Index>(index)];
      double velocity = 0.0;
      if (q - lower < margin) {
        velocity = config.position_gain * (lower + margin - q);
      } else if (upper - q < margin) {
        velocity = config.position_gain * (upper - margin - q);
      } else {
        return;
      }
      Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(count);
      row[static_cast<Eigen::Index>(index)] = 1.0;
      rows.push_back(std::move(row));
      references.push_back(std::clamp(
        velocity, -config.maximum_angular_reference_radps,
        config.maximum_angular_reference_radps));
    };
  add_limit_margin(config.j5_index, config.lower_j5, config.upper_j5,
    config.position_margin_j5_rad);
  add_limit_margin(config.j6_index, config.lower_j6, config.upper_j6,
    config.position_margin_j6_rad);

  result.valid = true;
  result.other_corrections.matrix.resize(static_cast<Eigen::Index>(rows.size()), count);
  result.other_corrections.reference.resize(static_cast<Eigen::Index>(references.size()));
  result.other_corrections.weights = Eigen::VectorXd::Ones(
    static_cast<Eigen::Index>(rows.size()));
  for (std::size_t index = 0; index < rows.size(); ++index) {
    result.other_corrections.matrix.row(static_cast<Eigen::Index>(index)) = rows[index];
    result.other_corrections.reference[static_cast<Eigen::Index>(index)] = references[index];
  }
  result.comfortable = rows.empty() && result.joint3_correction.matrix.rows() == 0;
  return result;
}
}  // namespace face_tracking_arm::control
