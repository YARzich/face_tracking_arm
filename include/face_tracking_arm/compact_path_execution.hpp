// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_PATH_EXECUTION_HPP_
#define FACE_TRACKING_ARM__COMPACT_PATH_EXECUTION_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "face_tracking_arm/compact_joint_profile.hpp"

namespace face_tracking_arm::control
{

enum class CompactQueueMatchStatus
{
  kMatchedCurrent,
  kMatchedFuture,
  kNoMatch
};

struct CompactQueueMatch
{
  CompactQueueMatchStatus status{CompactQueueMatchStatus::kNoMatch};
  std::size_t sample_index{0U};
};

struct CompactSmoothPathSuffix
{
  std::vector<Eigen::VectorXd> path;
  std::vector<double> speed_scales;
  double start_progress_rad{0.0};
};

/// Owns the active sampled PATH and its publication cursor.
/// The controller thread owns this object; it does not read or mutate a queue.
class CompactPathExecution final
{
public:
  void install(
    CompactJointProfile profile, JointMotionLimits prefix_limits,
    double initial_progress_rad = 0.0, std::size_t initial_cursor = 0U);
  void clear() noexcept;

  [[nodiscard]] bool active() const noexcept;
  [[nodiscard]] const CompactJointProfile * profile() const noexcept;
  [[nodiscard]] std::size_t cursor() const noexcept;
  [[nodiscard]] std::size_t prefixEndpoint() const noexcept;
  [[nodiscard]] double progressRad() const noexcept;
  [[nodiscard]] double absoluteProgressRad(double profile_progress_rad) const noexcept;
  [[nodiscard]] double totalPathLengthRad() const noexcept;
  [[nodiscard]] const JointMotionLimits & prefixLimits() const noexcept;

  /// Find an exact current or future profile sample matching queue q/v/a.
  /// This is observational: it never advances the cursor.
  [[nodiscard]] CompactQueueMatch matchQueueState(
    const JointMotionState & queue_state, double position_tolerance_rad,
    double velocity_tolerance_rad_s, double acceleration_tolerance_rad_s2) const noexcept;
  /// Commit a queue match only after the caller establishes that the matching
  /// sample belongs to a successfully published command record.
  [[nodiscard]] bool commitPublishedQueueMatch(std::size_t sample_index) noexcept;

  /// Project a queue state only onto the forward, not-yet-executed profile arc.
  /// Returns no value when the smooth geometry cannot identify one safe progress.
  [[nodiscard]] std::optional<double> projectQueueStateToProgress(
    const JointMotionState & queue_state, double residual_limit_rad,
    std::optional<double> minimum_progress_rad = std::nullopt) const;

  /// Slice the authoritative sampled geometry and interval metadata at progress.
  [[nodiscard]] std::optional<CompactSmoothPathSuffix> remainingRoute(
    double progress_rad) const;

  /// Advance exactly one sample only after its command was published.
  void recordCommandPublished(bool published) noexcept;

  /// Construct the branch brake for the currently committed sample.
  [[nodiscard]] EmergencyBrakeTail brakingTail(
    const JointMotionState & candidate_state, const JointMotionLimits & limits,
    const EmergencyBrakeTailConfig & config) const;

private:
  std::optional<CompactJointProfile> profile_;
  std::size_t cursor_{0U};
  std::size_t prefix_endpoint_sample_index_{0U};
  double progress_rad_{0.0};
  double profile_progress_origin_rad_{0.0};
  JointMotionLimits prefix_limits_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_PATH_EXECUTION_HPP_
