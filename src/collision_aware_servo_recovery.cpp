// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

namespace face_tracking_arm
{
void CollisionAwareServoComponent::enter_safety_wait(std::string message)
{
  if (controller_mode_ == ControllerMode::kWaitingForSafeState &&
    latched_halt_reason_ == message)
  {
    return;
  }
  // Keep the last checked stop on the controller. Reconsider fresh feedback
  // continuously; this software wait never clears a drive fault or an E-stop.
  halt_latched_.store(true, std::memory_order_release);
  proven_q_hold_.reset();
  proven_q_hold_epoch_.reset();
  controller_mode_ = ControllerMode::kWaitingForSafeState;
  latched_halt_reason_ = std::move(message);
  const auto * record = commands_.activePublication();
  const auto current_time = now();
  commands_.published_tail_rearm_pending = record && current_time < record->stationary_time;
  commands_.discardBranch();
  commands_.rearm_pending = true;
  commands_.restartSettling();
  commands_.ensureRearmDeadline(current_time, std::chrono::steady_clock::now());
  reset_feedback_derivative_history();
  motion_reference_->reset(true);
  publish_status(moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION, latched_halt_reason_);
}

void CollisionAwareServoComponent::handle_rearm_settle_timeout(std::string message)
{
  if (require_hardware_io_status_) {
    request_hardware_inhibit();
  }
  if (!commands_.claimRearmTimeoutReport()) {
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION,
      latched_halt_reason_.empty() ? "Safe re-arm timed out after its bounded settle window" :
      latched_halt_reason_);
    return;
  }
  ++telemetry_.rearm_timeout_count;
  enter_safety_wait(
    "Safe re-arm timed out after its bounded settle window: " + std::move(message));
}

void CollisionAwareServoComponent::reset_feedback_derivative_history()
{
  last_feedback_sample_time_.reset();
  last_feedback_velocity_.reset();
  last_feedback_acceleration_.reset();
}

void CollisionAwareServoComponent::enter_startup_retry(std::string message)
{
  // Before the first successful publication the trajectory controller is
  // still holding its own stationary reference. A transiently stale sample
  // must restart the settled-state gate, not create a permanent safety latch.
  const bool reset_ack_pending = commands_.bootstrap_reset_ack_pending;
  proven_q_hold_.reset();
  proven_q_hold_epoch_.reset();
  commands_.discardBranch(reset_ack_pending);
  if (!reset_ack_pending) {
    commands_.forgetPublished();
  }
  controller_mode_ = ControllerMode::kBraking;
  commands_.rearm_pending = true;
  commands_.published_tail_rearm_pending = false;
  commands_.restartSettling();
  commands_.ensureRearmDeadline(now(), std::chrono::steady_clock::now());
  reset_feedback_derivative_history();
  pause_terminal_pending_.store(false, std::memory_order_release);
  ++telemetry_.startup_retry_count;
  telemetry_.last_startup_retry_reason = message;
  publish_status(
    moveit_msgs::msg::ServoStatus::NO_WARNING,
    std::move(message) + "; waiting for a fresh settled hold before retry");
}

void CollisionAwareServoComponent::handle_recoverable_control_gap(std::string message)
{
  if (commands_.epoch_has_published || commands_.hasRetainedPublication()) {
    enter_controlled_rearm(std::move(message));
  } else {
    enter_startup_retry(std::move(message));
  }
}

void CollisionAwareServoComponent::enter_controlled_rearm(std::string message)
{
  if (require_hardware_io_status_ &&
    (!cycle_hardware_io_status_.has_value() ||
    !cycle_hardware_io_status_->valid ||
    cycle_hardware_io_status_->sample.epoch != proven_q_hold_epoch_))
  {
    proven_q_hold_.reset();
    proven_q_hold_epoch_.reset();
  }
  motion_reference_->reset(true);
  const auto * published_record = commands_.activePublication();
  if (published_record == nullptr || published_record->execution_queue.empty()) {
    if (require_hardware_io_status_) {
      request_hardware_inhibit();
    }
    enter_safety_wait(std::move(message));
    return;
  }
  ++commands_.consecutive_automatic_rearms;
  commands_.healthy_publications_since_rearm = 0U;

  // Do not replace the controller command. Its immutable suffix already
  // brakes to an exact hold. Discard only the unpublished online branch,
  // wait for measured settling, then bootstrap a fresh fixed-grid epoch.
  commands_.discardBranch();
  controller_mode_ = ControllerMode::kBraking;
  commands_.rearm_pending = true;
  const rclcpp::Time rearm_start = now();
  commands_.published_tail_rearm_pending = rearm_start < published_record->stationary_time;
  commands_.restartSettling();
  commands_.startRearmDeadline(rearm_start, std::chrono::steady_clock::now());
  reset_feedback_derivative_history();
  pause_terminal_pending_.store(false, std::memory_order_release);
  ++telemetry_.controlled_rearm_count;
  telemetry_.last_controlled_rearm_reason = message;
  publish_status(
    commands_.published_tail_rearm_pending ?
    moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION :
    moveit_msgs::msg::ServoStatus::NO_WARNING,
    std::move(message) +
    (commands_.published_tail_rearm_pending ?
    "; executing the pre-published smooth stop before automatic re-arm" :
    "; published stop is complete, confirming the settled state before automatic re-arm"));
}

bool CollisionAwareServoComponent::prepare_control_cycle(
  const rclcpp::Time & current_time,
  const std::chrono::steady_clock::time_point & current_wall_time,
  const control::ControlTimeObservation & time_observation)
{
  if (time_observation.status == control::ControlTimeStatus::kRewind) {
    ++telemetry_.clock_rewind_count;
    if (require_hardware_io_status_ && (commands_.epoch_has_published ||
      commands_.hasRetainedPublication() || commands_.published_tail_rearm_pending ||
      commands_.bootstrap_ack_pending || commands_.bootstrap_reset_ack_pending))
    {
      request_hardware_inhibit();
    }
    if (motion_reference_) {
      motion_reference_->reset();
    }
    proven_q_hold_.reset();
    proven_q_hold_epoch_.reset();
    commands_.discardBranch();
    commands_.forgetPublished();
    commands_.rearm_pending = false;
    commands_.published_tail_rearm_pending = false;
    commands_.restartSettling();
    commands_.clearRearmDeadlines();
    reset_feedback_derivative_history();
    initial_state_received_ = false;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_pose_.reset();
    }
    {
      std::lock_guard<std::mutex> lock(controller_state_mutex_);
      latest_controller_state_.reset();
    }
    controller_mode_ = ControllerMode::kWaitingForSafeState;
    halt_latched_.store(true, std::memory_order_release);
    commands_.rearm_pending = true;
    latched_halt_reason_ =
      "ROS time rewound; waiting for fresh feedback and a settled controller";
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION, latched_halt_reason_);
    return false;
  }
  const bool rearm_requested = reset_queue_requested_.exchange(
    false, std::memory_order_acq_rel);
  if (rearm_requested) {
    motion_reference_->reset();
    const auto * published_record = commands_.activePublication();
    commands_.published_tail_rearm_pending = published_record != nullptr &&
      current_time < published_record->stationary_time;
    commands_.discardBranch();
    commands_.rearm_pending = true;
    commands_.restartSettling();
    commands_.clearRearmDeadlines();
    commands_.consecutive_automatic_rearms = 0U;
    commands_.healthy_publications_since_rearm = 0U;
    reset_feedback_derivative_history();
    if (controller_mode_ != ControllerMode::kWaitingForSafeState) {
      controller_mode_ = ControllerMode::kBraking;
    }
  }
  if (commands_.rearm_pending) {
    commands_.ensureRearmDeadline(current_time, std::chrono::steady_clock::now());
  }
  if (time_observation.status == control::ControlTimeStatus::kNotAdvancing) {
    ++telemetry_.clock_not_advancing_observations;
    if ((commands_.rearm_pending || commands_.bootstrap_ack_pending) &&
      commands_.rearmDeadlineExpired(current_time, current_wall_time))
    {
      if (commands_.claimRearmTimeoutReport()) {
        ++telemetry_.rearm_timeout_count;
      }
      if (require_hardware_io_status_ &&
        (commands_.bootstrap_ack_pending || commands_.published_tail_rearm_pending))
      {
        request_hardware_inhibit();
      }
      enter_safety_wait(
        commands_.bootstrap_ack_pending ?
        "Trajectory controller bootstrap acknowledgement timed out while ROS time was frozen" :
        "Safe re-arm timed out while ROS time was frozen");
      return false;
    }
    publish_status(
      controller_mode_ == ControllerMode::kWaitingForSafeState ?
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      controller_mode_ == ControllerMode::kWaitingForSafeState ?
      (latched_halt_reason_.empty() ? "Waiting for a safe controller state" :
      latched_halt_reason_) : "ROS time is not advancing");
    return false;
  }
  if (time_observation.status == control::ControlTimeStatus::kWait) {
    if ((commands_.rearm_pending || commands_.bootstrap_ack_pending) &&
      commands_.rearmDeadlineExpired(current_time, current_wall_time))
    {
      if (commands_.claimRearmTimeoutReport()) {
        ++telemetry_.rearm_timeout_count;
      }
      if (require_hardware_io_status_ &&
        (commands_.bootstrap_ack_pending || commands_.published_tail_rearm_pending))
      {
        request_hardware_inhibit();
      }
      enter_safety_wait(
        commands_.bootstrap_ack_pending ?
        "Trajectory controller bootstrap acknowledgement timed out in wall time" :
        "Safe re-arm timed out in wall time");
    }
    return false;
  }
  telemetry_.skipped_control_periods += time_observation.skipped_periods;

  telemetry_.resetTick();
  if (!commands_.rearm_pending && controller_mode_ == ControllerMode::kWaitingForSafeState) {
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION,
      latched_halt_reason_.empty() ? "Waiting for a safe controller state" :
      latched_halt_reason_);
    return false;
  }
  if (paused_.load(std::memory_order_acquire) &&
    pause_requested_.load(std::memory_order_acquire))
  {
    publish_status(moveit_msgs::msg::ServoStatus::NO_WARNING, "Servoing paused");
    return false;
  }

  if (trajectory_publisher_->get_subscription_count() == 0U ||
    !trajectory_controller_active_.load(std::memory_order_acquire))
  {
    if (commands_.rearm_pending) {
      if (commands_.published_tail_rearm_pending) {
        if (require_hardware_io_status_) {
          request_hardware_inhibit();
        }
        enter_safety_wait(
          "Trajectory controller was lost while a published braking tail was in flight");
      } else if (commands_.rearmDeadlineExpired(current_time, current_wall_time)) {
        if (commands_.claimRearmTimeoutReport()) {
          ++telemetry_.rearm_timeout_count;
        }
        if (require_hardware_io_status_) {
          request_hardware_inhibit();
        }
        enter_safety_wait(
          "Safe re-arm timed out waiting for an active subscribed trajectory controller");
      } else {
        publish_status(
          halt_latched_.load(std::memory_order_acquire) ?
          moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
          moveit_msgs::msg::ServoStatus::NO_WARNING,
          "Safe re-arm is waiting for an active subscribed trajectory controller");
      }
    } else {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait("Trajectory controller is not active and subscribed");
    }
    return false;
  }

  if (commands_.rearm_pending) {
    if (commands_.published_tail_rearm_pending &&
      planning_scene_collision_revision_.load(std::memory_order_acquire) !=
      last_validated_scene_revision_)
    {
      ++telemetry_.planning_scene_invalidation_count;
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(
        "PlanningScene collision geometry changed while a published braking tail was in flight");
      return false;
    }
    std::string rearm_reason;
    const RearmReadiness rearm_readiness = safe_rearm_ready(
      current_time, rearm_reason);
    if (rearm_readiness == RearmReadiness::kUnsafe) {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(std::move(rearm_reason));
      return false;
    }
    if (rearm_readiness == RearmReadiness::kWaiting) {
      if (commands_.rearmDeadlineExpired(current_time, current_wall_time)) {
        handle_rearm_settle_timeout(std::move(rearm_reason));
        return false;
      }
      publish_status(
        halt_latched_.load(std::memory_order_acquire) ?
        moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
        (commands_.published_tail_rearm_pending ?
        moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION :
        moveit_msgs::msg::ServoStatus::NO_WARNING),
        std::move(rearm_reason));
      return false;
    }
    commands_.rearm_pending = false;
    commands_.published_tail_rearm_pending = false;
    commands_.restartSettling();
    commands_.forgetPublished();
    commands_.clearBootstrap();
    halt_latched_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    controller_mode_ = ControllerMode::kBraking;
    latched_halt_reason_.clear();
  }

  return true;
}

}  // namespace face_tracking_arm
