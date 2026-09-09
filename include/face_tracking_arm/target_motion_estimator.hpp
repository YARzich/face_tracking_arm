// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_
#define FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_

#include <Eigen/Core>

#include <optional>

namespace face_tracking_arm::control
{

struct TargetMotionEstimatorConfig
{
  double filter_time_constant_sec{0.08};
  double minimum_sample_interval_sec{0.01};
  double maximum_sample_interval_sec{0.20};
  double stale_timeout_sec{0.10};
  double maximum_future_skew_sec{0.02};
  double maximum_face_jump_m{0.15};
  double maximum_face_speed_mps{1.5};
  double maximum_goal_speed_mps{0.18};
};

struct TargetMotionEstimate
{
  Eigen::Vector3d face_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d goal_velocity{Eigen::Vector3d::Zero()};
};

/// Small observation-time velocity filter, without position extrapolation or ROS.
/// The owner calls reset() when changing target modes or coordinate frames.
class TargetMotionEstimator final
{
public:
  explicit TargetMotionEstimator(TargetMotionEstimatorConfig config = {});

  /// Differentiate new measurement stamps only; repeated stamps retain the filter
  /// until stale. Invalid/rewound/stale observations reset it; jumps and sample
  /// gaps start a new zero-velocity anchor. Bounded future clock skew has age zero.
  /// Speeds are norm-limited before filtering; output norms below 1e-4 m/s snap
  /// to exact zero so settled targets do not retain numerical feedforward.
  [[nodiscard]] TargetMotionEstimate update(
    const Eigen::Vector3d & face_position,
    const Eigen::Vector3d & goal_position,
    double measurement_time_sec, double control_time_sec);

  void reset() noexcept;

private:
  struct Observation
  {
    Eigen::Vector3d face;
    Eigen::Vector3d goal;
    double stamp_sec;
  };

  TargetMotionEstimatorConfig config_;
  std::optional<Observation> observation_;
  std::optional<double> last_control_time_sec_;
  TargetMotionEstimate filtered_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_
