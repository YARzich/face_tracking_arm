// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "command_session.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

namespace face_tracking_arm
{
namespace
{
constexpr std::size_t kHealthyPublicationsToResetRearmBudget = 100;
}  // namespace

const control::PublishedTrajectoryRecord * CommandSession::activePublication()
const noexcept
{
  return active_publication_id.has_value() ?
         publications.find(*active_publication_id) : nullptr;
}

bool CommandSession::deferCommandDeadlineRetry() noexcept
{
  if (command_deadline_retry_pending || command_deadline_retry_used ||
    !time_grid || !epoch_active || !epoch_has_published || queue.empty() ||
    activePublication() == nullptr)
  {
    return false;
  }
  command_deadline_retry_pending = true;
  command_deadline_retry_used = true;
  return true;
}

std::optional<std::uint64_t> CommandSession::timelineRecoveryPeriods(
  const std::uint64_t observed_skipped_periods, const std::int64_t current_time_ns,
  const std::int64_t target_command_lead_ns) const
{
  if (!command_deadline_retry_pending) {
    return observed_skipped_periods;
  }
  if (!time_grid) {
    return std::nullopt;
  }
  const auto required_periods = time_grid->periodsUntilCommandLead(
    current_time_ns, target_command_lead_ns);
  if (!required_periods.has_value()) {
    return std::nullopt;
  }
  // Both counts are relative to the current command-grid branch. Taking the
  // maximum avoids counting the same delayed periods twice.
  return std::max(observed_skipped_periods, *required_periods);
}

void CommandSession::completeDeferredCommandDeadlineRetry() noexcept
{
  command_deadline_retry_pending = false;
}

TimelineRecoveryStatus CommandSession::recoverTimeline(
  const std::uint64_t skipped_periods, const rclcpp::Time & current_time,
  std::string & failure_reason, const double retained_history_sec,
  const std::size_t joint_count, const TimelineRecoveryPurpose purpose)
{
  if (!epoch_active || skipped_periods == 0U) {
    return TimelineRecoveryStatus::kSuccess;
  }
  const auto * published_record = activePublication();
  if (!epoch_has_published || queue.empty() ||
    published_record == nullptr || published_record->execution_queue.size() < 3U)
  {
    failure_reason = "active epoch has no recoverable published suffix";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const rclcpp::Duration retained_history = rclcpp::Duration::from_seconds(
    retained_history_sec);
  while (queue.size() > 1U &&
    queue.front().time_stamp < current_time - retained_history)
  {
    queue.pop_front();
  }

  // The final execution-queue entry is intentionally withheld by MoveIt
  // Servo's trajectory composer, so it must never become a recovery anchor.
  const auto & published_queue = published_record->execution_queue;
  const auto transmitted_end = std::prev(published_queue.end());
  const auto current_branch = std::find_if(
    published_queue.begin(), transmitted_end,
    [this](const moveit_servo::KinematicState & state) {
      return state.time_stamp == queue.back().time_stamp;
    });
  if (current_branch == transmitted_end ||
    !current_branch->positions.isApprox(queue.back().positions, kFiniteEpsilon) ||
    !current_branch->velocities.isApprox(queue.back().velocities, kFiniteEpsilon) ||
    !current_branch->accelerations.isApprox(
      queue.back().accelerations, kFiniteEpsilon))
  {
    failure_reason = "current branch is not an exact prefix of the transmitted trajectory";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const auto available_periods = static_cast<std::uint64_t>(
    std::distance(std::next(current_branch), transmitted_end));
  if (skipped_periods > available_periods) {
    failure_reason = "transmitted braking suffix is too short for the delayed callback";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }
  const auto recovery_anchor = std::next(
    current_branch, static_cast<std::ptrdiff_t>(skipped_periods));

  const std::size_t recovered_periods = static_cast<std::size_t>(skipped_periods);
  // Retained past points are useful for matching a replacement, but they are
  // not part of the future safety horizon. Drop only the oldest history when
  // a delayed callback needs room for already-published future samples.
  while (queue.size() > 1U &&
    queue.size() + recovered_periods > max_queue_points)
  {
    queue.pop_front();
  }
  if (queue.size() + recovered_periods > max_queue_points) {
    failure_reason = "published recovery span exceeds the command queue hard bound";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }

  auto previous_stamp = current_branch->time_stamp;
  for (auto point = std::next(current_branch); point != std::next(recovery_anchor); ++point) {
    if (!finite_state(*point, joint_count) ||
      point->time_stamp - previous_stamp !=
      rclcpp::Duration::from_nanoseconds(time_grid->periodNs()))
    {
      failure_reason = "published recovery prefix left the fixed command grid";
      return TimelineRecoveryStatus::kInvariantFailure;
    }
    previous_stamp = point->time_stamp;
  }
  if (!time_grid->adoptPublishedStamp(recovery_anchor->time_stamp.nanoseconds())) {
    failure_reason = "recovery anchor was not on the active command grid";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  queue.insert(
    queue.end(), std::next(current_branch), std::next(recovery_anchor));
  if (purpose == TimelineRecoveryPurpose::kDeadline) {
    ++timeline_recovery_events;
    timeline_recovered_periods += recovered_periods;
  } else {
    ++published_brake_continuation_count;
  }
  return TimelineRecoveryStatus::kSuccess;
}

PublishedBrakeContinuationStatus CommandSession::continuePublishedBrake(
  const rclcpp::Time & decision_time, const std::int64_t required_next_command_lead_ns,
  std::string & failure_reason, const double retained_history_sec,
  const std::size_t joint_count)
{
  if (!time_grid || !epoch_active || !epoch_has_published || rearm_pending ||
    published_tail_rearm_pending ||
    bootstrap_ack_pending || activePublication() == nullptr || queue.empty())
  {
    return PublishedBrakeContinuationStatus::kNotEligible;
  }
  const auto required_periods = time_grid->periodsUntilCommandLead(
    decision_time.nanoseconds(), required_next_command_lead_ns);
  if (!required_periods.has_value() || *required_periods > 1U) {
    return PublishedBrakeContinuationStatus::kNotEligible;
  }

  const TimelineRecoveryStatus recovery_status = recoverTimeline(
    1U, decision_time, failure_reason, retained_history_sec, joint_count,
    TimelineRecoveryPurpose::kPublishedBrakeContinuation);
  switch (recovery_status) {
    case TimelineRecoveryStatus::kSuccess:
      return PublishedBrakeContinuationStatus::kContinued;
    case TimelineRecoveryStatus::kExhaustedPublishedSuffix:
      return PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix;
    case TimelineRecoveryStatus::kInvariantFailure:
      return PublishedBrakeContinuationStatus::kInvariantFailure;
  }
  return PublishedBrakeContinuationStatus::kInvariantFailure;
}

void CommandSession::clearBootstrap() noexcept
{
  bootstrap_ack_pending = false;
  bootstrap_publication_id.reset();
  bootstrap_trajectory_origin.reset();
  bootstrap_seed_positions.reset();
  bootstrap_reset_ack_pending = false;
  bootstrap_reset_stable_samples = 0U;
  bootstrap_reset_published_time.reset();
  last_bootstrap_reset_sample_time.reset();
  bootstrap_reset_positions.reset();
  bootstrap_source_stable_samples = 0U;
  last_bootstrap_source_sample_time.reset();
  bootstrap_source_reference_positions.reset();
  bootstrap_source_feedback_positions.reset();
}

bool CommandSession::observeBootstrapResetAcknowledgement(
  const rclcpp::Time & sample_time, const Eigen::VectorXd & target_positions,
  const Eigen::VectorXd & reference_positions, const Eigen::VectorXd & output_positions,
  const Eigen::VectorXd & feedback_positions, const Eigen::VectorXd & reference_velocities,
  const Eigen::VectorXd & feedback_velocities, const double command_tolerance,
  const double feedback_position_tolerance, const double stopped_velocity,
  const bool feedback_safe)
{
  if (!bootstrap_reset_ack_pending || !target_positions.allFinite() ||
    !reference_positions.allFinite() || !output_positions.allFinite() ||
    !feedback_positions.allFinite() || !reference_velocities.allFinite() ||
    !feedback_velocities.allFinite() || !std::isfinite(command_tolerance) ||
    !std::isfinite(feedback_position_tolerance) || !std::isfinite(stopped_velocity) ||
    command_tolerance < 0.0 || feedback_position_tolerance < 0.0 || stopped_velocity < 0.0 ||
    target_positions.size() == 0 || reference_positions.size() != target_positions.size() ||
    output_positions.size() != target_positions.size() ||
    feedback_positions.size() != target_positions.size() ||
    reference_velocities.size() != target_positions.size() ||
    feedback_velocities.size() != target_positions.size() || !feedback_safe)
  {
    bootstrap_reset_stable_samples = 0U;
    last_bootstrap_reset_sample_time.reset();
    return false;
  }
  if (!bootstrap_reset_published_time.has_value() ||
    sample_time.get_clock_type() != bootstrap_reset_published_time->get_clock_type() ||
    sample_time <= *bootstrap_reset_published_time)
  {
    bootstrap_reset_stable_samples = 0U;
    return false;
  }
  if (last_bootstrap_reset_sample_time.has_value() &&
    sample_time <= *last_bootstrap_reset_sample_time)
  {
    return false;
  }
  last_bootstrap_reset_sample_time = sample_time;
  if ((reference_positions - target_positions).cwiseAbs().maxCoeff() > command_tolerance ||
    (output_positions - target_positions).cwiseAbs().maxCoeff() > command_tolerance ||
    (feedback_positions - target_positions).cwiseAbs().maxCoeff() > feedback_position_tolerance ||
    reference_velocities.cwiseAbs().maxCoeff() > stopped_velocity ||
    feedback_velocities.cwiseAbs().maxCoeff() > stopped_velocity)
  {
    bootstrap_reset_stable_samples = 0U;
    return false;
  }
  ++bootstrap_reset_stable_samples;
  return bootstrap_reset_stable_samples >= 3U;
}

bool CommandSession::observeBootstrapSourceHold(
  const rclcpp::Time & sample_time, const Eigen::VectorXd & reference_positions,
  const Eigen::VectorXd & output_positions, const Eigen::VectorXd & feedback_positions,
  const Eigen::VectorXd & reference_velocities, const Eigen::VectorXd & feedback_velocities,
  const double command_tolerance, const double feedback_position_tolerance,
  const double stopped_velocity, const bool feedback_safe)
{
  if (!reference_positions.allFinite() || !output_positions.allFinite() ||
    !feedback_positions.allFinite() || !reference_velocities.allFinite() ||
    !feedback_velocities.allFinite() || !std::isfinite(command_tolerance) ||
    !std::isfinite(feedback_position_tolerance) || !std::isfinite(stopped_velocity) ||
    command_tolerance < 0.0 || feedback_position_tolerance < 0.0 || stopped_velocity < 0.0 ||
    reference_positions.size() == 0 || output_positions.size() != reference_positions.size() ||
    feedback_positions.size() != reference_positions.size() ||
    reference_velocities.size() != reference_positions.size() ||
    feedback_velocities.size() != reference_positions.size() || !feedback_safe ||
    (reference_positions - output_positions).cwiseAbs().maxCoeff() > command_tolerance ||
    reference_velocities.cwiseAbs().maxCoeff() > stopped_velocity ||
    feedback_velocities.cwiseAbs().maxCoeff() > stopped_velocity)
  {
    bootstrap_source_stable_samples = 0U;
    last_bootstrap_source_sample_time.reset();
    bootstrap_source_reference_positions.reset();
    bootstrap_source_feedback_positions.reset();
    return false;
  }
  if (last_bootstrap_source_sample_time.has_value() &&
    sample_time <= *last_bootstrap_source_sample_time)
  {
    return false;
  }
  last_bootstrap_source_sample_time = sample_time;
  const bool stable = bootstrap_source_reference_positions.has_value() &&
    bootstrap_source_feedback_positions.has_value() &&
    bootstrap_source_reference_positions->size() == reference_positions.size() &&
    bootstrap_source_feedback_positions->size() == feedback_positions.size() &&
    (*bootstrap_source_reference_positions - reference_positions).cwiseAbs().maxCoeff() <=
    command_tolerance &&
    (*bootstrap_source_feedback_positions - feedback_positions).cwiseAbs().maxCoeff() <=
    feedback_position_tolerance;
  bootstrap_source_reference_positions = reference_positions;
  bootstrap_source_feedback_positions = feedback_positions;
  bootstrap_source_stable_samples = stable ? bootstrap_source_stable_samples + 1U : 1U;
  return bootstrap_source_stable_samples >= 3U;
}

void CommandSession::startRearmDeadline(
  const rclcpp::Time & current_time, const WallTime & wall_time)
{
  rearm_timeout_reported = false;
  rclcpp::Time settle_start = current_time;
  if (published_tail_rearm_pending) {
    const auto * published_record = activePublication();
    if (published_record != nullptr && published_record->stationary_time > settle_start) {
      settle_start = published_record->stationary_time;
    }
  }
  rearm_deadline = settle_start +
    rclcpp::Duration::from_seconds(kSafeRearmSettleBudgetSec);
  const double remaining_tail_sec = std::max(
    0.0, (settle_start - current_time).seconds());
  rearm_wall_deadline = wall_time +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(remaining_tail_sec + kSafeRearmSettleBudgetSec));
}

void CommandSession::ensureRearmDeadline(
  const rclcpp::Time & current_time, const WallTime & wall_time)
{
  if (!rearm_deadline.has_value() || !rearm_wall_deadline.has_value()) {
    startRearmDeadline(current_time, wall_time);
  }
}

void CommandSession::clearRearmDeadlines() noexcept
{
  rearm_deadline.reset();
  rearm_wall_deadline.reset();
  rearm_timeout_reported = false;
}

bool CommandSession::claimRearmTimeoutReport() noexcept
{
  if (rearm_timeout_reported) {
    return false;
  }
  rearm_timeout_reported = true;
  return true;
}

bool CommandSession::rearmDeadlineExpired(
  const rclcpp::Time & current_time,
  const std::chrono::steady_clock::time_point & wall_time) const noexcept
{
  const bool ros_deadline_expired =
    rearm_deadline.has_value() && current_time >= *rearm_deadline;
  const bool wall_deadline_expired =
    rearm_wall_deadline.has_value() && wall_time >= *rearm_wall_deadline;
  return ros_deadline_expired || wall_deadline_expired;
}

void CommandSession::recordHealthyPublication() noexcept
{
  command_deadline_retry_pending = false;
  command_deadline_retry_used = false;
  if (consecutive_automatic_rearms == 0U || bootstrap_ack_pending) {
    return;
  }
  ++healthy_publications_since_rearm;
  if (healthy_publications_since_rearm >=
    kHealthyPublicationsToResetRearmBudget)
  {
    consecutive_automatic_rearms = 0U;
    healthy_publications_since_rearm = 0U;
  }
}

void CommandSession::discardBranch(const bool preserveBootstrapReset)
{
  const bool reset_ack_pending = preserveBootstrapReset && bootstrap_reset_ack_pending;
  std::optional<Eigen::VectorXd> reset_positions;
  std::optional<rclcpp::Time> reset_published_time;
  if (reset_ack_pending && bootstrap_reset_positions.has_value()) {
    reset_positions.emplace(*bootstrap_reset_positions);
    reset_published_time = bootstrap_reset_published_time;
  }
  queue.clear();
  time_grid->resetCommandEpoch();
  epoch_active = false;
  epoch_has_published = false;
  command_deadline_retry_pending = false;
  command_deadline_retry_used = false;
  clearBootstrap();
  if (reset_ack_pending && reset_positions.has_value()) {
    bootstrap_reset_ack_pending = true;
    bootstrap_reset_positions = reset_positions;
    bootstrap_reset_published_time = reset_published_time;
  }
}

void CommandSession::forgetPublished()
{
  publications.clear();
  active_publication_id.reset();
}

bool CommandSession::hasRetainedPublication() const noexcept
{
  return activePublication() != nullptr;
}

void CommandSession::restartSettling() noexcept
{
  stable_rearm_samples = 0;
  last_rearm_sample_time.reset();
}

}  // namespace face_tracking_arm
