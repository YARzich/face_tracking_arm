// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMMAND_SESSION_HPP_
#define COMMAND_SESSION_HPP_

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include "face_tracking_arm/control_time_grid.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"
#include "servo_types.hpp"

namespace face_tracking_arm
{
enum class TimelineRecoveryStatus
{
  kSuccess,
  kExhaustedPublishedSuffix,
  kInvariantFailure,
};

enum class TimelineRecoveryPurpose
{
  kDeadline,
  kPublishedBrakeContinuation,
};

enum class PublishedBrakeContinuationStatus
{
  kContinued,
  kNotEligible,
  kExhaustedPublishedSuffix,
  kInvariantFailure,
};

// Owned exclusively by the control thread. The online branch can be discarded
// while the controller continues executing the last immutable published stop.
struct CommandSession
{
  using WallTime = std::chrono::steady_clock::time_point;

  const control::PublishedTrajectoryRecord * activePublication() const noexcept;
  TimelineRecoveryStatus recoverTimeline(
    std::uint64_t skipped_periods, const rclcpp::Time & current_time,
    std::string & failure_reason, double retained_history_sec, std::size_t joint_count,
    TimelineRecoveryPurpose purpose = TimelineRecoveryPurpose::kDeadline);
  PublishedBrakeContinuationStatus continuePublishedBrake(
    const rclcpp::Time & decision_time, std::int64_t required_next_command_lead_ns,
    std::string & failure_reason, double retained_history_sec, std::size_t joint_count);
  [[nodiscard]] bool deferCommandDeadlineRetry() noexcept;
  [[nodiscard]] std::optional<std::uint64_t> timelineRecoveryPeriods(
    std::uint64_t observed_skipped_periods, std::int64_t current_time_ns,
    std::int64_t target_command_lead_ns) const;
  void completeDeferredCommandDeadlineRetry() noexcept;
  void discardBranch(bool preserveBootstrapReset = false);
  void forgetPublished();
  [[nodiscard]] bool hasRetainedPublication() const noexcept;
  void restartSettling() noexcept;
  void clearBootstrap() noexcept;
  bool observeBootstrapResetAcknowledgement(
    const rclcpp::Time & sample_time, const Eigen::VectorXd & target_positions,
    const Eigen::VectorXd & reference_positions, const Eigen::VectorXd & output_positions,
    const Eigen::VectorXd & feedback_positions, const Eigen::VectorXd & reference_velocities,
    const Eigen::VectorXd & feedback_velocities, double command_tolerance,
    double feedback_position_tolerance, double stopped_velocity, bool feedback_safe);
  bool observeBootstrapSourceHold(
    const rclcpp::Time & sample_time, const Eigen::VectorXd & reference_positions,
    const Eigen::VectorXd & output_positions, const Eigen::VectorXd & feedback_positions,
    const Eigen::VectorXd & reference_velocities, const Eigen::VectorXd & feedback_velocities,
    double command_tolerance, double feedback_position_tolerance,
    double stopped_velocity, bool feedback_safe);
  void startRearmDeadline(const rclcpp::Time & current_time, const WallTime & wall_time);
  void ensureRearmDeadline(const rclcpp::Time & current_time, const WallTime & wall_time);
  void clearRearmDeadlines() noexcept;
  bool rearmDeadlineExpired(
    const rclcpp::Time & current_time,
    const WallTime & wall_time) const noexcept;
  void recordHealthyPublication() noexcept;

  std::unique_ptr<control::ControlTimeGrid> time_grid;
  control::PublishedTrajectoryHistory publications{
    kPublishedTrajectoryHistoryCapacity};
  std::deque<moveit_servo::KinematicState> queue;
  std::optional<control::PublicationId> active_publication_id;
  std::size_t max_queue_points{0};
  std::size_t startup_warmup_cycles{0};
  bool startup_warmup_complete{false};
  bool epoch_active{false};
  bool epoch_has_published{false};
  bool bootstrap_ack_pending{false};
  std::optional<control::PublicationId> bootstrap_publication_id;
  std::optional<rclcpp::Time> bootstrap_trajectory_origin;
  std::optional<Eigen::VectorXd> bootstrap_seed_positions;
  bool bootstrap_reset_ack_pending{false};
  std::size_t bootstrap_reset_stable_samples{0};
  std::optional<rclcpp::Time> bootstrap_reset_published_time;
  std::optional<rclcpp::Time> last_bootstrap_reset_sample_time;
  std::optional<Eigen::VectorXd> bootstrap_reset_positions;
  std::size_t bootstrap_source_stable_samples{0};
  std::optional<rclcpp::Time> last_bootstrap_source_sample_time;
  std::optional<Eigen::VectorXd> bootstrap_source_reference_positions;
  std::optional<Eigen::VectorXd> bootstrap_source_feedback_positions;
  bool rearm_pending{false};
  bool published_tail_rearm_pending{false};
  std::size_t stable_rearm_samples{0};
  std::optional<rclcpp::Time> last_rearm_sample_time;
  std::optional<rclcpp::Time> rearm_deadline;
  std::optional<std::chrono::steady_clock::time_point> rearm_wall_deadline;
  std::size_t consecutive_automatic_rearms{0};
  std::size_t healthy_publications_since_rearm{0};
  std::uint64_t timeline_recovery_events{0};
  std::uint64_t timeline_recovered_periods{0};
  std::uint64_t published_brake_continuation_count{0};
  bool command_deadline_retry_pending{false};
  bool command_deadline_retry_used{false};
};

}  // namespace face_tracking_arm

#endif  // COMMAND_SESSION_HPP_
