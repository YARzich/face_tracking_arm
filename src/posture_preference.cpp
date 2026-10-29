// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/posture_preference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

PosturePreferenceObjective makePosturePreferenceObjective(
  const Eigen::Vector3d & base, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face,
  const Eigen::MatrixXd & base_jacobian, const Eigen::MatrixXd & elbow_jacobian,
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
  const Eigen::Index j3 = static_cast<Eigen::Index>(config.j3_index);
  const double q3 = joint_position[j3];
  const bool comfortable = q3 >= config.comfortable_joint3_lower &&
    q3 <= config.comfortable_joint3_upper;
  result.branch_comfortable = comfortable;
  const double one_step_reserve = std::min(
    config.maximum_angular_reference_radps * config.control_period_sec,
    0.5 * (config.comfortable_joint3_upper - config.comfortable_joint3_lower));
  double face_side_scale = 0.0;
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
    rows.push_back(std::move(branch_row));
    references.push_back(q3_rate);
  } else {
    const double distance_to_lower = q3 - config.comfortable_joint3_lower;
    const double distance_to_upper = config.comfortable_joint3_upper - q3;
    const bool near_lower = distance_to_lower <= distance_to_upper;
    const double distance_to_boundary = near_lower ? distance_to_lower : distance_to_upper;
    face_side_scale = one_step_reserve > 0.0 ?
      std::clamp(distance_to_boundary / one_step_reserve, 0.0, 1.0) : 1.0;
  }

  const auto add_elbow_geometry = [&]() {
      if (!base.allFinite() || !elbow.allFinite() || !wrist.allFinite() ||
        !face.allFinite() || base_jacobian.rows() != 3 || elbow_jacobian.rows() != 3 ||
        wrist_jacobian.rows() != 3 || base_jacobian.cols() != count ||
        elbow_jacobian.cols() != count || wrist_jacobian.cols() != count ||
        !base_jacobian.allFinite() || !elbow_jacobian.allFinite() ||
        !wrist_jacobian.allFinite())
      {
        return;
      }
      const Eigen::Vector3d e = elbow - base;
      const Eigen::Vector3d w = wrist - base;
      const Eigen::Vector3d f = face - base;
      const double d = w.squaredNorm();
      const double forearm_length = (wrist - elbow).stableNorm();
      if (!std::isfinite(d) || d <= 0.0 || !std::isfinite(forearm_length) ||
        forearm_length <= std::numeric_limits<double>::epsilon())
      {
        return;
      }

      const double a = e.dot(w);
      const double b = f.dot(w);
      if (!std::isfinite(a) || !std::isfinite(b)) {
        return;
      }
      const double c = e.dot(f) - (a * b) / d;
      const Eigen::Vector3d f_perp = f - w * (b / d);
      const double r_squared = f_perp.squaredNorm();
      if (!std::isfinite(c) || !f_perp.allFinite() || !std::isfinite(r_squared) ||
        r_squared <= 0.0)
      {
        return;
      }

      const double reserve = config.maximum_linear_reference_mps * config.control_period_sec;
      const double r = std::sqrt(r_squared);
      const double denominator = std::sqrt(r_squared + forearm_length * forearm_length);
      const double clearance_residual = (-c - reserve * r) / denominator;
      if (!std::isfinite(reserve) || !std::isfinite(denominator) || denominator <= 0.0 ||
        !std::isfinite(clearance_residual) || clearance_residual >= 0.0)
      {
        return;
      }

      const double line_activation = d / (d + forearm_length * forearm_length);
      const double face_activation = r_squared / (r_squared + forearm_length * forearm_length);
      const double activation = line_activation * face_activation;
      if (!std::isfinite(activation) || activation <= 0.0) {
        return;
      }

      Eigen::RowVectorXd clearance_row(count);
      const double squared_forearm_length = forearm_length * forearm_length;
      for (Eigen::Index index = 0; index < count; ++index) {
        const Eigen::Vector3d de = elbow_jacobian.col(index) - base_jacobian.col(index);
        const Eigen::Vector3d dw = wrist_jacobian.col(index) - base_jacobian.col(index);
        const Eigen::Vector3d forearm_derivative = dw - de;
        const Eigen::Vector3d df = -base_jacobian.col(index);
        const double da = de.dot(w) + e.dot(dw);
        const double db = df.dot(w) + f.dot(dw);
        const double dd = 2.0 * w.dot(dw);
        const double dc = de.dot(f) + e.dot(df) - (da * b + a * db) / d +
          (a * b * dd) / (d * d);
        const Eigen::Vector3d df_perp = df - dw * (b / d) - w * (db / d - b * dd / (d * d));
        const double dr = f_perp.dot(df_perp) / r;
        const double d_forearm_length =
          (wrist - elbow).dot(forearm_derivative) / forearm_length;
        const double d_residual = -dc / denominator +
          (c * r - reserve * squared_forearm_length) * dr /
          (denominator * denominator * denominator) +
          (c + reserve * r) * forearm_length * d_forearm_length /
          (denominator * denominator * denominator);
        clearance_row[index] = d_residual / forearm_length -
          clearance_residual * d_forearm_length / squared_forearm_length;
      }
      if (!clearance_row.allFinite()) {
        return;
      }

      const double desired_rate = std::clamp(
        -config.position_gain * clearance_residual,
        0.0, config.maximum_linear_reference_mps);
      rows.push_back(face_side_scale * activation * clearance_row);
      references.push_back(face_side_scale * activation * desired_rate / forearm_length);
    };
  if (comfortable) {
    add_elbow_geometry();
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
  result.matrix.resize(static_cast<Eigen::Index>(rows.size()), count);
  result.reference.resize(static_cast<Eigen::Index>(references.size()));
  result.weights = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(rows.size()));
  for (std::size_t index = 0; index < rows.size(); ++index) {
    result.matrix.row(static_cast<Eigen::Index>(index)) = rows[index];
    result.reference[static_cast<Eigen::Index>(index)] = references[index];
  }
  result.comfortable = rows.empty();
  return result;
}
}  // namespace face_tracking_arm::control
