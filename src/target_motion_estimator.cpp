// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/target_motion_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace face_tracking_arm::control
{
namespace
{

constexpr double kTimeToleranceSec = 1.0e-9;
constexpr double kStationarySpeedMps = 1.0e-4;

bool positiveFinite(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

Eigen::Vector3d limitedVelocity(const Eigen::Vector3d & velocity, const double maximum)
{
  const double norm = velocity.stableNorm();
  return norm > maximum ? Eigen::Vector3d(velocity * (maximum / norm)) : velocity;
}

TargetMotionEstimate withoutNumericalResidual(TargetMotionEstimate estimate)
{
  if (estimate.face_velocity.norm() < kStationarySpeedMps) {
    estimate.face_velocity.setZero();
  }
  if (estimate.goal_velocity.norm() < kStationarySpeedMps) {
    estimate.goal_velocity.setZero();
  }
  return estimate;
}

}  // namespace

TargetMotionEstimator::TargetMotionEstimator(TargetMotionEstimatorConfig config)
: config_(config)
{
  if (!positiveFinite(config_.filter_time_constant_sec) ||
    !positiveFinite(config_.minimum_sample_interval_sec) ||
    !positiveFinite(config_.maximum_sample_interval_sec) ||
    config_.minimum_sample_interval_sec > config_.maximum_sample_interval_sec ||
    !positiveFinite(config_.stale_timeout_sec) ||
    !std::isfinite(config_.maximum_future_skew_sec) || config_.maximum_future_skew_sec < 0.0 ||
    !positiveFinite(config_.maximum_face_jump_m) ||
    !positiveFinite(config_.maximum_face_speed_mps) ||
    !positiveFinite(config_.maximum_goal_speed_mps))
  {
    throw std::invalid_argument("Invalid target-motion estimator configuration");
  }
}

void TargetMotionEstimator::reset() noexcept
{
  observation_.reset();
  last_control_time_sec_.reset();
  filtered_ = TargetMotionEstimate{};
}

TargetMotionEstimate TargetMotionEstimator::update(
  const Eigen::Vector3d & face_position,
  const Eigen::Vector3d & goal_position,
  const double measurement_time_sec, const double control_time_sec)
{
  if (!face_position.allFinite() || !goal_position.allFinite() ||
    !std::isfinite(measurement_time_sec) || !std::isfinite(control_time_sec) ||
    measurement_time_sec < 0.0 || control_time_sec < 0.0 ||
    measurement_time_sec - control_time_sec >
    config_.maximum_future_skew_sec + kTimeToleranceSec)
  {
    reset();
    return {};
  }
  if ((last_control_time_sec_ && control_time_sec < *last_control_time_sec_) ||
    (observation_ && measurement_time_sec < observation_->stamp_sec))
  {
    reset();
  }
  const double age_sec = std::max(0.0, control_time_sec - measurement_time_sec);
  if (age_sec > config_.stale_timeout_sec + kTimeToleranceSec) {
    reset();
    return {};
  }
  last_control_time_sec_ = control_time_sec;
  if (!observation_) {
    observation_ = Observation{face_position, goal_position, measurement_time_sec};
    return {};
  }
  if (measurement_time_sec == observation_->stamp_sec) {
    return withoutNumericalResidual(filtered_);
  }

  const double interval = measurement_time_sec - observation_->stamp_sec;
  const Eigen::Vector3d face_delta = face_position - observation_->face;
  const Eigen::Vector3d goal_delta = goal_position - observation_->goal;
  observation_ = Observation{face_position, goal_position, measurement_time_sec};
  if (interval + kTimeToleranceSec < config_.minimum_sample_interval_sec ||
    interval > config_.maximum_sample_interval_sec + kTimeToleranceSec ||
    !face_delta.allFinite() || !goal_delta.allFinite() ||
    face_delta.stableNorm() > config_.maximum_face_jump_m)
  {
    filtered_ = TargetMotionEstimate{};
    return {};
  }
  const Eigen::Vector3d face_velocity = face_delta / interval;
  const Eigen::Vector3d goal_velocity = goal_delta / interval;
  if (!face_velocity.allFinite() || !goal_velocity.allFinite()) {
    reset();
    return {};
  }
  const double alpha = -std::expm1(-interval / config_.filter_time_constant_sec);
  filtered_.face_velocity = limitedVelocity(
    (1.0 - alpha) * filtered_.face_velocity +
    alpha * limitedVelocity(face_velocity, config_.maximum_face_speed_mps),
    config_.maximum_face_speed_mps);
  filtered_.goal_velocity = limitedVelocity(
    (1.0 - alpha) * filtered_.goal_velocity +
    alpha * limitedVelocity(goal_velocity, config_.maximum_goal_speed_mps),
    config_.maximum_goal_speed_mps);
  return withoutNumericalResidual(filtered_);
}

}  // namespace face_tracking_arm::control
