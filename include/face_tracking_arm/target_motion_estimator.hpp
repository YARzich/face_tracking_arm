// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_
#define FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_

#include <Eigen/Core>

#include <optional>
#include <vector>

namespace face_tracking_arm::control
{

enum class MotionTimingMode
{
  kManual,
  kAuto,
};

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
  MotionTimingMode timing_mode{MotionTimingMode::kManual};
  /// Hard upper bound for source-sample gaps; AUTO only tightens this bound.
  double maximum_source_gap_sec{0.50};
  /// Maximum age at which the last real sample can still anchor new history.
  double history_timeout_sec{0.50};
  /// Bounded allowance for transport and scheduling delay after cadence-based expiry.
  double delivery_delay_reserve_sec{0.05};
};

struct TargetMotionEstimate
{
  Eigen::Vector3d face_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d goal_velocity{Eigen::Vector3d::Zero()};
};

/// Observation-time velocity filter and bounded source-time face extrapolator.
/// The owner calls reset() when changing target modes or coordinate frames.
class TargetMotionEstimator final
{
public:
  explicit TargetMotionEstimator(TargetMotionEstimatorConfig config = {});

  /// Differentiate new measurement stamps only; repeated stamps retain the live
  /// estimate until stale. A stale live estimate is zeroed while cadence history
  /// remains bounded independently. Invalid/rewound observations reset both;
  /// jumps and long sample gaps start a new zero-velocity anchor. Bounded future
  /// clock skew has age zero.
  /// Speeds are norm-limited before filtering; output norms below 1e-4 m/s snap
  /// to exact zero so settled targets do not retain numerical feedforward.
  [[nodiscard]] TargetMotionEstimate update(
    const Eigen::Vector3d & face_position,
    const Eigen::Vector3d & goal_position,
    double measurement_time_sec, double control_time_sec);

  /// Extrapolate the latest real face observation without changing its stamp.
  /// Returns empty once the configured stale timeout or time-order checks fail.
  [[nodiscard]] std::optional<Eigen::Vector3d> predictFacePosition(
    double control_time_sec);

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
  std::vector<double> valid_sample_intervals_sec_;

  [[nodiscard]] double effectiveStaleTimeoutSec() const;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__TARGET_MOTION_ESTIMATOR_HPP_
