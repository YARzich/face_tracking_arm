// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "command_session.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
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

bool CommandSession::hasVerifiedTerminalHold(const std::size_t joint_count) const noexcept
{
  if (joint_count == 0U || !epoch_active || !epoch_has_published || rearm_pending ||
    published_tail_rearm_pending || bootstrap_ack_pending || bootstrap_reset_ack_pending ||
    command_deadline_retry_pending || queue.size() != 1U || !time_grid ||
    !active_publication_id.has_value() || !verified_terminal_hold_publication_id.has_value() ||
    *verified_terminal_hold_publication_id != *active_publication_id ||
    max_queue_points < 2U)
  {
    return false;
  }

  const auto * record = activePublication();
  const std::int64_t period_ns = time_grid->periodNs();
  if (record == nullptr || record->id != *active_publication_id || period_ns <= 0 ||
    record->execution_queue.size() < 3U)
  {
    return false;
  }

  const auto transmitted_end = std::prev(record->execution_queue.end(), 2);
  const auto & anchor = queue.front();
  const std::int64_t anchor_ns = anchor.time_stamp.nanoseconds();
  const std::int64_t terminal_ns = transmitted_end->time_stamp.nanoseconds();
  if (transmitted_end->joint_names.size() != joint_count ||
    anchor.joint_names != transmitted_end->joint_names || !finite_state(anchor, joint_count) ||
    anchor.time_stamp.get_clock_type() != transmitted_end->time_stamp.get_clock_type() ||
    anchor_ns <= terminal_ns || anchor_ns % period_ns != terminal_ns % period_ns ||
    !(anchor.positions.array() == transmitted_end->positions.array()).all() ||
    !anchor.velocities.isZero(0.0) || !anchor.accelerations.isZero(0.0))
  {
    return false;
  }

  try {
    if (anchor_ns > std::numeric_limits<std::int64_t>::max() - period_ns) {
      return false;
    }
    auto checked_grid = *time_grid;
    const auto next_stamp = checked_grid.commandStamp(anchor_ns, 0, 0);
    return next_stamp.has_value() && *next_stamp == anchor_ns + period_ns;
  } catch (...) {
    return false;
  }
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
  if (!time_grid) {
    return command_deadline_retry_pending ?
           std::nullopt : std::optional<std::uint64_t>{observed_skipped_periods};
  }
  // Timer skips describe callback cadence, not the current command-grid lead.
  // A published-brake continuation may already have advanced that grid, so
  // using the observed count here would recover the same intervals twice.
  const auto required_periods = time_grid->periodsUntilCommandLead(
    current_time_ns, target_command_lead_ns);
  if (!required_periods.has_value()) {
    // An inactive startup/reset epoch has no command anchor yet; retain its
    // callback-based behavior. An active epoch must have a usable anchor.
    return command_deadline_retry_pending || epoch_active ?
           std::nullopt : std::optional<std::uint64_t>{observed_skipped_periods};
  }
  return required_periods;
}

void CommandSession::completeDeferredCommandDeadlineRetry() noexcept
{
  command_deadline_retry_pending = false;
}

void CommandSession::pruneExpiredHistory(
  const rclcpp::Time & current_time, const double retained_history_sec)
{
  const rclcpp::Duration retained_history = rclcpp::Duration::from_seconds(
    retained_history_sec);
  while (queue.size() > 1U &&
    queue.front().time_stamp < current_time - retained_history)
  {
    queue.pop_front();
  }
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
  const auto retention_cutoff = current_time - retained_history;

  // The final execution-queue entry is intentionally withheld by MoveIt
  // Servo's trajectory composer, so it must never become a recovery anchor.
  const auto & published_queue = published_record->execution_queue;
  const auto transmitted_end = std::prev(published_queue.end());
  const auto current_branch = std::find_if(
    published_queue.begin(), transmitted_end,
    [this](const moveit_servo::KinematicState & state) {
      return state.time_stamp == queue.back().time_stamp;
    });
  if (current_branch == transmitted_end &&
    verified_terminal_hold_publication_id == active_publication_id &&
    current_time >= published_record->stationary_time)
  {
    const auto terminal = std::prev(published_queue.end(), 2);
    const std::int64_t extension_ns =
      queue.back().time_stamp.nanoseconds() - terminal->time_stamp.nanoseconds();
    if (finite_state(queue.back(), joint_count) &&
      queue.back().time_stamp.get_clock_type() == current_time.get_clock_type() &&
      extension_ns > 0 && extension_ns % time_grid->periodNs() == 0 &&
      queue.back().positions.isApprox(terminal->positions, kFiniteEpsilon) &&
      queue.back().velocities.isZero(kFiniteEpsilon) &&
      queue.back().accelerations.isZero(kFiniteEpsilon))
    {
      failure_reason = "verified terminal hold needs a fresh settled check before extension";
      return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
    }
  }
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

  // Validate the full transmitted prefix above, including expired samples,
  // but retain only entries still inside the history window. Work on a bounded
  // candidate so a failed capacity check cannot mutate the live command queue.
  const auto recovery_end = std::next(recovery_anchor);
  auto first_retained = std::next(current_branch);
  while (first_retained != recovery_end && first_retained->time_stamp < retention_cutoff) {
    ++first_retained;
  }
  // The adopted command-grid stamp must always be represented by the queue,
  // even when the entire crossed prefix is older than the retention window.
  if (first_retained == recovery_end) {
    first_retained = recovery_anchor;
  }
  std::size_t retained_recovery_points = static_cast<std::size_t>(
    std::distance(first_retained, recovery_end));

  auto recovered_queue = queue;
  while (recovered_queue.size() > 1U &&
    recovered_queue.front().time_stamp < retention_cutoff)
  {
    recovered_queue.pop_front();
  }
  // If the only surviving old queue entry is expired, the retained transmitted
  // prefix (which always includes its anchor) supersedes it completely.
  if (recovered_queue.size() == 1U &&
    recovered_queue.front().time_stamp < retention_cutoff &&
    retained_recovery_points > 0U)
  {
    recovered_queue.clear();
  }
  while (!recovered_queue.empty() &&
    recovered_queue.size() + retained_recovery_points > max_queue_points &&
    recovered_queue.front().time_stamp < current_time)
  {
    recovered_queue.pop_front();
  }
  while (recovered_queue.size() + retained_recovery_points > max_queue_points &&
    first_retained != recovery_anchor && first_retained->time_stamp < current_time)
  {
    ++first_retained;
    --retained_recovery_points;
  }
  if (recovered_queue.size() + retained_recovery_points > max_queue_points) {
    failure_reason = "published recovery span exceeds the command queue hard bound";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }
  if (!time_grid->adoptPublishedStamp(recovery_anchor->time_stamp.nanoseconds())) {
    failure_reason = "recovery anchor was not on the active command grid";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  for (auto point = first_retained; point != recovery_end; ++point) {
    recovered_queue.push_back(*point);
  }
  queue.swap(recovered_queue);
  const std::size_t recovered_periods = static_cast<std::size_t>(skipped_periods);
  if (purpose == TimelineRecoveryPurpose::kDeadline) {
    ++timeline_recovery_events;
    timeline_recovered_periods += recovered_periods;
  } else {
    ++published_brake_continuation_count;
  }
  return TimelineRecoveryStatus::kSuccess;
}

TimelineRecoveryStatus CommandSession::resumeVerifiedTerminalHold(
  const control::PublicationId expected_publication_id,
  const rclcpp::Time & current_time, const std::int64_t target_command_lead_ns,
  const std::size_t joint_count, const bool settled_proof, std::string & failure_reason)
{
  if (!settled_proof || !epoch_active || !epoch_has_published || rearm_pending ||
    published_tail_rearm_pending || bootstrap_ack_pending || queue.empty() || !time_grid ||
    !active_publication_id.has_value() || *active_publication_id != expected_publication_id)
  {
    failure_reason = "verified terminal hold no longer owns the active command epoch";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }
  if (target_command_lead_ns < 0 || joint_count == 0U) {
    failure_reason = "terminal hold resume has invalid scheduling bounds";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const auto * record = publications.find(expected_publication_id);
  if (record == nullptr || record != activePublication() ||
    record->execution_queue.size() < 3U ||
    record->stationary_time.get_clock_type() != current_time.get_clock_type() ||
    current_time < record->stationary_time)
  {
    failure_reason = "verified terminal hold publication is missing or not yet stationary";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }

  const auto & published_queue = record->execution_queue;
  const auto transmitted_end = std::prev(published_queue.end(), 2);
  const auto stationary = std::find_if(
    published_queue.begin(), std::next(transmitted_end), [record](const auto & state) {
      return state.time_stamp == record->stationary_time;
    });
  if (stationary == std::next(transmitted_end) ||
    !finite_state(*transmitted_end, joint_count) ||
    transmitted_end->joint_names.size() != joint_count ||
    transmitted_end->time_stamp.get_clock_type() != current_time.get_clock_type())
  {
    failure_reason = "published terminal hold is not a transmitted, finite command";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const Eigen::VectorXd & terminal_position = transmitted_end->positions;
  for (auto point = stationary; point != published_queue.end(); ++point) {
    if (!finite_state(*point, joint_count) || point->joint_names != queue.front().joint_names ||
      point->time_stamp.get_clock_type() != current_time.get_clock_type() ||
      !point->positions.isApprox(terminal_position, kFiniteEpsilon) ||
      !point->velocities.isZero(kFiniteEpsilon) ||
      !point->accelerations.isZero(kFiniteEpsilon))
    {
      failure_reason = "published terminal suffix does not hold one exact stopped command";
      return TimelineRecoveryStatus::kInvariantFailure;
    }
    if (point != stationary && point->time_stamp - std::prev(point)->time_stamp !=
      rclcpp::Duration::from_nanoseconds(time_grid->periodNs()))
    {
      failure_reason = "published terminal hold left the active fixed command grid";
      return TimelineRecoveryStatus::kInvariantFailure;
    }
  }

  const auto current_branch = std::find_if(
    published_queue.begin(), std::next(transmitted_end), [this](const auto & state) {
      return state.time_stamp == queue.back().time_stamp;
    });
  const bool exact_published_branch = current_branch != std::next(transmitted_end) &&
    current_branch->positions.isApprox(queue.back().positions, kFiniteEpsilon) &&
    current_branch->velocities.isApprox(queue.back().velocities, kFiniteEpsilon) &&
    current_branch->accelerations.isApprox(queue.back().accelerations, kFiniteEpsilon);
  const bool verified_virtual_hold =
    verified_terminal_hold_publication_id == expected_publication_id &&
    finite_state(queue.back(), joint_count) &&
    queue.back().time_stamp.get_clock_type() == current_time.get_clock_type() &&
    queue.back().time_stamp > transmitted_end->time_stamp &&
    (queue.back().time_stamp.nanoseconds() -
    transmitted_end->time_stamp.nanoseconds()) % time_grid->periodNs() == 0 &&
    queue.back().positions.isApprox(terminal_position, kFiniteEpsilon) &&
    queue.back().velocities.isZero(kFiniteEpsilon) &&
    queue.back().accelerations.isZero(kFiniteEpsilon);
  if (!exact_published_branch && !verified_virtual_hold) {
    failure_reason = "current command branch is not an exact published or verified hold prefix";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const std::int64_t period_ns = time_grid->periodNs();
  if (period_ns <= 0 || max_queue_points < 2U) {
    failure_reason = "terminal hold resume has no usable command-grid capacity";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  auto resumed_grid = *time_grid;
  const rclcpp::Time base_anchor = verified_virtual_hold ?
    queue.back().time_stamp : transmitted_end->time_stamp;
  if (!verified_virtual_hold && queue.back().time_stamp < base_anchor &&
    !resumed_grid.adoptPublishedStamp(base_anchor.nanoseconds()))
  {
    failure_reason = "published terminal command is not later on the active command grid";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  if (!verified_virtual_hold && queue.back().time_stamp > base_anchor) {
    failure_reason = "current command branch advanced beyond the published terminal command";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  const auto required_periods = resumed_grid.periodsUntilCommandLead(
    current_time.nanoseconds(), target_command_lead_ns);
  if (!required_periods.has_value()) {
    failure_reason = "terminal hold resume lost its command-grid anchor";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  if (*required_periods > static_cast<std::uint64_t>(
      (std::numeric_limits<std::int64_t>::max() - base_anchor.nanoseconds()) / period_ns))
  {
    failure_reason = "verified terminal hold timestamp exceeds the representable command grid";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  std::uint64_t anchor_periods = *required_periods;
  std::int64_t final_anchor_ns = base_anchor.nanoseconds() +
    static_cast<std::int64_t>(anchor_periods) * period_ns;
  while (final_anchor_ns <= current_time.nanoseconds()) {
    if (anchor_periods == std::numeric_limits<std::uint64_t>::max() ||
      anchor_periods >= static_cast<std::uint64_t>(
        (std::numeric_limits<std::int64_t>::max() - base_anchor.nanoseconds()) / period_ns))
    {
      failure_reason = "verified terminal hold timestamp exceeds the representable command grid";
      return TimelineRecoveryStatus::kInvariantFailure;
    }
    ++anchor_periods;
    final_anchor_ns = base_anchor.nanoseconds() +
      static_cast<std::int64_t>(anchor_periods) * period_ns;
  }
  const rclcpp::Time final_anchor(final_anchor_ns, current_time.get_clock_type());
  if (anchor_periods > 0U &&
    !resumed_grid.adoptPublishedStamp(final_anchor.nanoseconds()))
  {
    failure_reason = "terminal hold resume could not adopt its phase-aligned command anchor";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  if (final_anchor < record->stationary_time) {
    failure_reason = "phase-aligned terminal hold anchor precedes the published stationary point";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  moveit_servo::KinematicState hold(static_cast<int>(joint_count));
  hold.joint_names = transmitted_end->joint_names;
  hold.positions = terminal_position;
  hold.velocities.setZero();
  hold.accelerations.setZero();
  hold.time_stamp = final_anchor;
  if (!finite_state(hold, joint_count)) {
    failure_reason = "terminal hold extension produced a non-finite command sample";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  queue.clear();
  queue.push_back(std::move(hold));
  *time_grid = std::move(resumed_grid);
  verified_terminal_hold_publication_id = expected_publication_id;
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
  if (!required_periods.has_value()) {
    return PublishedBrakeContinuationStatus::kNotEligible;
  }

  const auto * published_record = activePublication();
  if (published_record != nullptr && active_publication_id.has_value() &&
    verified_terminal_hold_publication_id == active_publication_id &&
    decision_time >= published_record->stationary_time &&
    published_record->execution_queue.size() >= 3U)
  {
    const auto transmitted_end = std::prev(published_record->execution_queue.end(), 2);
    const auto expected_hold = publications.expectedAt(
      published_record->id, queue.back().time_stamp);
    const std::int64_t extension_ns =
      queue.back().time_stamp.nanoseconds() - transmitted_end->time_stamp.nanoseconds();
    if (finite_state(queue.back(), joint_count) &&
      queue.back().time_stamp.get_clock_type() == decision_time.get_clock_type() &&
      time_grid->periodNs() > 0 && extension_ns > 0 &&
      extension_ns % time_grid->periodNs() == 0 &&
      expected_hold.has_value() && expected_hold->terminal_hold &&
      expected_hold->positions.isApprox(queue.back().positions, kFiniteEpsilon) &&
      expected_hold->velocities.isZero(kFiniteEpsilon) &&
      queue.back().velocities.isZero(kFiniteEpsilon) &&
      queue.back().accelerations.isZero(kFiniteEpsilon))
    {
      ++published_brake_continuation_count;
      return PublishedBrakeContinuationStatus::kContinued;
    }
  }

  const TimelineRecoveryStatus recovery_status = recoverTimeline(
    std::max<std::uint64_t>(1U, *required_periods), decision_time,
    failure_reason, retained_history_sec, joint_count,
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
  verified_terminal_hold_publication_id.reset();
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
  verified_terminal_hold_publication_id.reset();
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
