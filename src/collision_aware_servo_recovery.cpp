// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr std::size_t kMaximumConsecutiveAutomaticRearms = 3;

constexpr std::size_t kHealthyPublicationsToResetRearmBudget = 100;

}  // namespace

void CollisionAwareServoComponent::clear_bootstrap_ack_state() noexcept
{
  bootstrap_ack_pending_ = false;
  bootstrap_publication_id_.reset();
  bootstrap_trajectory_origin_.reset();
  bootstrap_seed_positions_.reset();
}

void CollisionAwareServoComponent::start_rearm_deadline(const rclcpp::Time & current_time)
{
  rclcpp::Time settle_start = current_time;
  if (published_tail_rearm_pending_) {
    const auto * published_record = active_published_record();
    if (published_record != nullptr && published_record->stationary_time > settle_start) {
      settle_start = published_record->stationary_time;
    }
  }
  rearm_deadline_ = settle_start +
    rclcpp::Duration::from_seconds(kSafeRearmSettleBudgetSec);
  const double remaining_tail_sec = std::max(
    0.0, (settle_start - current_time).seconds());
  rearm_wall_deadline_ = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(remaining_tail_sec + kSafeRearmSettleBudgetSec));
}

void CollisionAwareServoComponent::ensure_rearm_deadline(const rclcpp::Time & current_time)
{
  if (!rearm_deadline_.has_value() || !rearm_wall_deadline_.has_value()) {
    start_rearm_deadline(current_time);
  }
}

void CollisionAwareServoComponent::clear_rearm_deadlines() noexcept
{
  rearm_deadline_.reset();
  rearm_wall_deadline_.reset();
}

bool CollisionAwareServoComponent::rearm_deadline_expired(
  const rclcpp::Time & current_time,
  const std::chrono::steady_clock::time_point & wall_time) const noexcept
{
  const bool ros_deadline_expired =
    rearm_deadline_.has_value() && current_time >= *rearm_deadline_;
  const bool wall_deadline_expired =
    rearm_wall_deadline_.has_value() && wall_time >= *rearm_wall_deadline_;
  return ros_deadline_expired || wall_deadline_expired;
}

void CollisionAwareServoComponent::record_healthy_publication() noexcept
{
  if (consecutive_automatic_rearms_ == 0U || bootstrap_ack_pending_) {
    return;
  }
  ++healthy_publications_since_rearm_;
  if (healthy_publications_since_rearm_ >=
    kHealthyPublicationsToResetRearmBudget)
  {
    consecutive_automatic_rearms_ = 0U;
    healthy_publications_since_rearm_ = 0U;
  }
}

void CollisionAwareServoComponent::enter_latched_halt(std::string message)
{
  if (halt_latched_.exchange(true, std::memory_order_acq_rel)) {
    rearm_pending_ = false;
    published_tail_rearm_pending_ = false;
    stable_rearm_samples_ = 0;
    last_rearm_sample_time_.reset();
    clear_rearm_deadlines();
    clear_bootstrap_ack_state();
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION,
      latched_halt_reason_.empty() ? message : latched_halt_reason_);
    return;
  }
  // Do not replace the active JTC command here. Every successful publication
  // already contains a checked jerk-limited suffix to a stationary hold, so
  // silence is the only transport-independent way to preserve that stop.
  controller_mode_ = ControllerMode::kLatchedHalt;
  latched_halt_reason_ = message;
  command_queue_.clear();
  control_time_grid_->resetCommandEpoch();
  command_epoch_active_ = false;
  command_epoch_has_published_ = false;
  clear_bootstrap_ack_state();
  rearm_pending_ = false;
  published_tail_rearm_pending_ = false;
  stable_rearm_samples_ = 0;
  last_rearm_sample_time_.reset();
  clear_rearm_deadlines();
  publish_status(moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION, latched_halt_reason_);
}

void CollisionAwareServoComponent::reset_feedback_derivative_history()
{
  last_feedback_sample_time_.reset();
  last_feedback_velocity_.reset();
  last_feedback_acceleration_.reset();
  last_feedback_jerk_.reset();
}

void CollisionAwareServoComponent::enter_startup_retry(std::string message)
{
  // Before the first successful publication the trajectory controller is
  // still holding its own stationary reference. A transiently stale sample
  // must restart the settled-state gate, not create a permanent safety latch.
  command_queue_.clear();
  published_trajectory_history_.clear();
  active_publication_id_.reset();
  control_time_grid_->resetCommandEpoch();
  command_epoch_active_ = false;
  command_epoch_has_published_ = false;
  clear_bootstrap_ack_state();
  controller_mode_ = ControllerMode::kBraking;
  rearm_pending_ = true;
  published_tail_rearm_pending_ = false;
  stable_rearm_samples_ = 0;
  last_rearm_sample_time_.reset();
  ensure_rearm_deadline(now());
  reset_feedback_derivative_history();
  pause_terminal_pending_.store(false, std::memory_order_release);
  ++startup_retry_count_;
  last_startup_retry_reason_ = message;
  publish_status(
    moveit_msgs::msg::ServoStatus::NO_WARNING,
    std::move(message) + "; waiting for a fresh settled hold before retry");
}

void CollisionAwareServoComponent::handle_recoverable_control_gap(std::string message)
{
  if (command_epoch_has_published_) {
    enter_controlled_rearm(std::move(message));
  } else {
    enter_startup_retry(std::move(message));
  }
}

void CollisionAwareServoComponent::enter_controlled_rearm(std::string message)
{
  motion_reference_->reset();
  const auto * published_record = active_published_record();
  if (!command_epoch_has_published_ || published_record == nullptr ||
    published_record->execution_queue.empty())
  {
    enter_latched_halt(std::move(message));
    return;
  }
  if (consecutive_automatic_rearms_ >= kMaximumConsecutiveAutomaticRearms) {
    ++rearm_budget_exhaustion_count_;
    enter_latched_halt(
      "Automatic recovery budget exhausted after repeated control gaps: " + message);
    return;
  }
  ++consecutive_automatic_rearms_;
  healthy_publications_since_rearm_ = 0U;

  // Do not replace the controller command. Its immutable suffix already
  // brakes to an exact hold. Discard only the unpublished online branch,
  // wait for measured settling, then bootstrap a fresh fixed-grid epoch.
  command_queue_.clear();
  control_time_grid_->resetCommandEpoch();
  command_epoch_active_ = false;
  command_epoch_has_published_ = false;
  clear_bootstrap_ack_state();
  controller_mode_ = ControllerMode::kBraking;
  rearm_pending_ = true;
  const rclcpp::Time rearm_start = now();
  published_tail_rearm_pending_ = rearm_start < published_record->stationary_time;
  stable_rearm_samples_ = 0;
  last_rearm_sample_time_.reset();
  start_rearm_deadline(rearm_start);
  reset_feedback_derivative_history();
  pause_terminal_pending_.store(false, std::memory_order_release);
  ++controlled_rearm_count_;
  last_controlled_rearm_reason_ = message;
  publish_status(
    published_tail_rearm_pending_ ?
    moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION :
    moveit_msgs::msg::ServoStatus::NO_WARNING,
    std::move(message) +
    (published_tail_rearm_pending_ ?
    "; executing the pre-published smooth stop before automatic re-arm" :
    "; published stop is complete, confirming the settled state before automatic re-arm"));
}

}  // namespace face_tracking_arm
