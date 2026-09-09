// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

const char * controller_mode_name(const ControllerMode mode)
{
  switch (mode) {
    case ControllerMode::kTracking:
      return "TRACKING";
    case ControllerMode::kBraking:
      return "BRAKING";
    case ControllerMode::kLatchedHalt:
      return "LATCHED_HALT";
  }
  return "UNKNOWN";
}

diagnostic_msgs::msg::KeyValue key_value(std::string key, std::string value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = std::move(key);
  result.value = std::move(value);
  return result;
}

std::string optional_double_string(const std::optional<double> & value)
{
  return value.has_value() ? std::to_string(*value) : "unavailable";
}

}  // namespace

void CollisionAwareServoComponent::observe_controller_mode() noexcept
{
  if (controller_mode_ == ControllerMode::kBraking) {
    ++braking_tick_count_;
  } else if (controller_mode_ == ControllerMode::kLatchedHalt) {
    ++emergency_tick_count_;
  }

  if (!last_observed_controller_mode_.has_value() ||
    *last_observed_controller_mode_ != controller_mode_)
  {
    if (controller_mode_ == ControllerMode::kBraking) {
      ++braking_event_count_;
    } else if (controller_mode_ == ControllerMode::kLatchedHalt) {
      ++emergency_event_count_;
    }
    last_observed_controller_mode_ = controller_mode_;
  }
}

void CollisionAwareServoComponent::publish_runtime_diagnostics()
{
  const auto control_mean_ms = control_tick_statistics_.mean();
  const auto control_p99_ms = control_tick_statistics_.p99();
  const auto control_max_ms = control_tick_statistics_.maximum();
  const auto qp_mean_ms = qp_statistics_.mean();
  const auto qp_p99_ms = qp_statistics_.p99();
  const auto qp_max_ms = qp_statistics_.maximum();
  const auto collision_build_mean_ms = collision_build_statistics_.mean();
  const auto collision_build_p99_ms = collision_build_statistics_.p99();
  const auto collision_build_max_ms = collision_build_statistics_.maximum();
  const auto segment_validation_mean_ms = segment_validation_statistics_.mean();
  const auto segment_validation_p99_ms = segment_validation_statistics_.p99();
  const auto segment_validation_max_ms = segment_validation_statistics_.maximum();
  const auto path_validation_mean_ms = path_validation_statistics_.mean();
  const auto path_validation_p99_ms = path_validation_statistics_.p99();
  const auto path_validation_max_ms = path_validation_statistics_.maximum();
  const auto pointing_error_mean_rad = pointing_error_statistics_.mean();
  const auto pointing_error_p99_rad = pointing_error_statistics_.p99();
  const auto pointing_error_max_rad = pointing_error_statistics_.maximum();

  const rclcpp::Time current_time = now();
  const auto current_wall_time = std::chrono::steady_clock::now();
  std::optional<double> latest_pose_age_ms;
  std::optional<double> rearm_deadline_remaining_sec;
  std::optional<double> rearm_wall_deadline_remaining_sec;
  if (rearm_deadline_.has_value() && *rearm_deadline_ >= current_time) {
    rearm_deadline_remaining_sec = (*rearm_deadline_ - current_time).seconds();
  }
  if (rearm_wall_deadline_.has_value() &&
    *rearm_wall_deadline_ >= current_wall_time)
  {
    rearm_wall_deadline_remaining_sec = std::chrono::duration<double>(
      *rearm_wall_deadline_ - current_wall_time).count();
  }
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (latest_pose_.has_value()) {
      const rclcpp::Time pose_stamp(
        latest_pose_->header.stamp, current_time.get_clock_type());
      latest_pose_age_ms = 1000.0 * (current_time - pose_stamp).seconds();
    }
  }
  std::optional<double> queue_horizon_sec;
  std::optional<double> published_queue_horizon_sec;
  std::optional<double> braking_tail_stop_remaining_sec;
  if (!command_queue_.empty()) {
    queue_horizon_sec = std::max(
      0.0, (command_queue_.back().time_stamp - current_time).seconds());
  }
  const auto * published_record = active_published_record();
  if (published_record != nullptr && published_record->execution_queue.size() >= 2U) {
    published_queue_horizon_sec = std::max(
      0.0,
      (std::prev(published_record->execution_queue.end(), 2)->time_stamp -
      current_time).seconds());
    braking_tail_stop_remaining_sec = std::max(
      0.0, (published_record->stationary_time - current_time).seconds());
  }

  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = current_time;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "face_tracking_arm/collision_aware_servo";
  status.hardware_id = "lite6_gazebo";
  if (controller_mode_ == ControllerMode::kLatchedHalt) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  } else if (last_servo_status_code_ == moveit_msgs::msg::ServoStatus::NO_WARNING) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
  }
  status.message = controller_mode_name(controller_mode_);
  status.values.reserve(112);
  status.values.push_back(key_value("mode", controller_mode_name(controller_mode_)));
  status.values.push_back(key_value(
    "servo_status_code", std::to_string(last_servo_status_code_)));
  status.values.push_back(key_value("servo_status_message", last_servo_status_message_));
  const auto & reference = motion_reference_->diagnostics();
  status.values.push_back(key_value("motion_reference_state", reference.state));
  status.values.push_back(key_value("last_plan_message", reference.last_plan_message));
  status.values.push_back(key_value("plans_requested", std::to_string(reference.plans_requested)));
  status.values.push_back(key_value("plans_accepted", std::to_string(reference.plans_accepted)));
  status.values.push_back(key_value("plans_rejected", std::to_string(reference.plans_rejected)));
  status.values.push_back(key_value("paths_completed", std::to_string(reference.paths_completed)));
  status.values.push_back(key_value("paths_abandoned", std::to_string(reference.paths_abandoned)));
  status.values.push_back(key_value("last_plan_wall_ms",
      std::to_string(reference.last_plan_wall_ms)));
  status.values.push_back(key_value("path_progress_rad",
      std::to_string(reference.path_progress_rad)));
  status.values.push_back(key_value("path_distance_rad",
      std::to_string(reference.path_distance_rad)));
  status.values.push_back(key_value("roll_error_rad",
      optional_double_string(last_roll_error_rad_)));
  status.values.push_back(key_value(
    "pose_messages_received", std::to_string(
      pose_message_count_.load(std::memory_order_relaxed))));
  status.values.push_back(key_value(
    "pose_messages_accepted", std::to_string(
      pose_accepted_count_.load(std::memory_order_relaxed))));
  status.values.push_back(key_value(
    "pose_messages_invalid", std::to_string(
      pose_invalid_count_.load(std::memory_order_relaxed))));
  status.values.push_back(key_value(
    "pose_messages_future", std::to_string(
      pose_future_count_.load(std::memory_order_relaxed))));
  status.values.push_back(key_value(
    "pose_messages_not_newer", std::to_string(
      pose_not_newer_count_.load(std::memory_order_relaxed))));
  status.values.push_back(key_value(
    "latest_pose_age_ms", optional_double_string(latest_pose_age_ms)));
  status.values.push_back(key_value(
    "pose_commands_enabled",
    accept_pose_commands_.load(std::memory_order_acquire) ? "true" : "false"));
  status.values.push_back(key_value(
    "pause_requested",
    pause_requested_.load(std::memory_order_acquire) ? "true" : "false"));
  status.values.push_back(key_value(
    "control_tick_mean_ms", optional_double_string(control_mean_ms)));
  status.values.push_back(key_value(
    "control_tick_p99_ms", optional_double_string(control_p99_ms)));
  status.values.push_back(key_value(
    "control_tick_max_ms", optional_double_string(control_max_ms)));
  status.values.push_back(key_value(
    "control_tick_lifetime_max_ms", std::to_string(lifetime_maximum_control_tick_ms_)));
  status.values.push_back(key_value(
    "control_cycle_mean_ms", optional_double_string(control_cycle_statistics_.mean())));
  status.values.push_back(key_value(
    "control_cycle_p99_ms", optional_double_string(control_cycle_statistics_.p99())));
  status.values.push_back(key_value(
    "control_cycle_max_ms", optional_double_string(control_cycle_statistics_.maximum())));
  status.values.push_back(key_value("qp_mean_ms", optional_double_string(qp_mean_ms)));
  status.values.push_back(key_value("qp_p99_ms", optional_double_string(qp_p99_ms)));
  status.values.push_back(key_value("qp_max_ms", optional_double_string(qp_max_ms)));
  status.values.push_back(key_value(
    "collision_build_mean_ms", optional_double_string(collision_build_mean_ms)));
  status.values.push_back(key_value(
    "collision_self_query_mean_ms",
    optional_double_string(collision_self_query_statistics_.mean())));
  status.values.push_back(key_value(
    "collision_world_query_mean_ms",
    optional_double_string(collision_world_query_statistics_.mean())));
  status.values.push_back(key_value(
    "collision_build_p99_ms", optional_double_string(collision_build_p99_ms)));
  status.values.push_back(key_value(
    "collision_build_max_ms", optional_double_string(collision_build_max_ms)));
  status.values.push_back(key_value(
    "segment_validation_mean_ms", optional_double_string(segment_validation_mean_ms)));
  status.values.push_back(key_value(
    "segment_validation_p99_ms", optional_double_string(segment_validation_p99_ms)));
  status.values.push_back(key_value(
    "segment_validation_max_ms", optional_double_string(segment_validation_max_ms)));
  status.values.push_back(key_value(
    "path_validation_mean_ms", optional_double_string(path_validation_mean_ms)));
  status.values.push_back(key_value(
    "path_validation_p99_ms", optional_double_string(path_validation_p99_ms)));
  status.values.push_back(key_value(
    "path_validation_max_ms", optional_double_string(path_validation_max_ms)));
  status.values.push_back(key_value(
    "missed_control_deadlines", std::to_string(missed_control_deadlines_)));
  status.values.push_back(key_value(
    "skipped_control_periods", std::to_string(skipped_control_periods_)));
  status.values.push_back(key_value(
    "timeline_recovery_events", std::to_string(timeline_recovery_events_)));
  status.values.push_back(key_value(
    "timeline_recovered_periods", std::to_string(timeline_recovered_periods_)));
  status.values.push_back(key_value(
    "timeline_recovery_failures", std::to_string(timeline_recovery_failures_)));
  status.values.push_back(key_value(
    "controlled_rearm_count", std::to_string(controlled_rearm_count_)));
  status.values.push_back(key_value(
    "last_controlled_rearm_reason", last_controlled_rearm_reason_));
  status.values.push_back(key_value(
    "startup_retry_count", std::to_string(startup_retry_count_)));
  status.values.push_back(key_value(
    "last_startup_retry_reason", last_startup_retry_reason_));
  status.values.push_back(key_value(
    "clock_not_advancing_observations",
    std::to_string(clock_not_advancing_observations_)));
  status.values.push_back(key_value(
    "clock_rewind_count", std::to_string(clock_rewind_count_)));
  status.values.push_back(key_value(
    "queue_horizon_sec", optional_double_string(queue_horizon_sec)));
  status.values.push_back(key_value(
    "published_queue_horizon_sec",
    optional_double_string(published_queue_horizon_sec)));
  status.values.push_back(key_value("queue_size", std::to_string(command_queue_.size())));
  status.values.push_back(key_value(
    "queue_hard_limit", std::to_string(max_command_queue_points_)));
  status.values.push_back(key_value(
    "published_execution_queue_size",
    std::to_string(
      published_record == nullptr ? 0U : published_record->execution_queue.size())));
  status.values.push_back(key_value(
    "published_trajectory_history_size",
    std::to_string(published_trajectory_history_.size())));
  status.values.push_back(key_value(
    "braking_tail_stop_remaining_sec",
    optional_double_string(braking_tail_stop_remaining_sec)));
  status.values.push_back(key_value(
    "trajectory_controller_subscriptions",
    std::to_string(trajectory_publisher_->get_subscription_count())));
  status.values.push_back(key_value(
    "trajectory_controller_active",
    trajectory_controller_active_.load(std::memory_order_acquire) ? "true" : "false"));
  status.values.push_back(key_value(
    "published_command_count", std::to_string(published_command_count_)));
  status.values.push_back(key_value(
    "last_published_horizon_sec",
    optional_double_string(last_published_horizon_sec_)));
  status.values.push_back(key_value(
    "minimum_published_horizon_sec",
    optional_double_string(minimum_published_horizon_sec_)));
  status.values.push_back(key_value(
    "last_replacement_lead_sec",
    optional_double_string(last_replacement_lead_sec_)));
  status.values.push_back(key_value(
    "minimum_replacement_lead_sec",
    optional_double_string(minimum_replacement_lead_sec_)));
  status.values.push_back(key_value(
    "startup_warmup_cycles",
    std::to_string(startup_warmup_cycles_)));
  status.values.push_back(key_value(
    "startup_warmup_complete", startup_warmup_complete_ ? "true" : "false"));
  status.values.push_back(key_value(
    "bootstrap_publication_count", std::to_string(bootstrap_publication_count_)));
  status.values.push_back(key_value(
    "bootstrap_ack_pending", bootstrap_ack_pending_ ? "true" : "false"));
  status.values.push_back(key_value(
    "bootstrap_publication_id",
    bootstrap_publication_id_.has_value() ?
    std::to_string(*bootstrap_publication_id_) : "unavailable"));
  status.values.push_back(key_value(
    "bootstrap_acknowledgement_count",
    std::to_string(bootstrap_acknowledgement_count_)));
  status.values.push_back(key_value(
    "bootstrap_segment_validation_count",
    std::to_string(bootstrap_segment_validation_count_)));
  status.values.push_back(key_value("rearm_pending", rearm_pending_ ? "true" : "false"));
  status.values.push_back(key_value(
    "published_tail_rearm_pending",
    published_tail_rearm_pending_ ? "true" : "false"));
  status.values.push_back(key_value(
    "stable_rearm_samples", std::to_string(stable_rearm_samples_)));
  status.values.push_back(key_value(
    "rearm_deadline_remaining_sec",
    optional_double_string(rearm_deadline_remaining_sec)));
  status.values.push_back(key_value(
    "rearm_wall_deadline_remaining_sec",
    optional_double_string(rearm_wall_deadline_remaining_sec)));
  status.values.push_back(key_value(
    "rearm_timeout_count", std::to_string(rearm_timeout_count_)));
  status.values.push_back(key_value(
    "consecutive_automatic_rearms",
    std::to_string(consecutive_automatic_rearms_)));
  status.values.push_back(key_value(
    "healthy_publications_since_rearm",
    std::to_string(healthy_publications_since_rearm_)));
  status.values.push_back(key_value(
    "rearm_budget_exhaustion_count",
    std::to_string(rearm_budget_exhaustion_count_)));
  status.values.push_back(key_value(
    "controller_speed_scaling_factor",
    optional_double_string(last_controller_speed_scaling_factor_)));
  status.values.push_back(key_value(
    "controller_speed_scaling_failure_count",
    std::to_string(controller_speed_scaling_failure_count_)));
  status.values.push_back(key_value(
    "controller_reference_alignment_error_rad",
    optional_double_string(last_controller_reference_alignment_error_rad_)));
  status.values.push_back(key_value(
    "controller_output_alignment_error_rad",
    optional_double_string(last_controller_output_alignment_error_rad_)));
  status.values.push_back(key_value(
    "max_controller_reference_alignment_error_rad",
    std::to_string(max_controller_reference_alignment_error_rad_)));
  status.values.push_back(key_value(
    "max_controller_output_alignment_error_rad",
    std::to_string(max_controller_output_alignment_error_rad_)));
  status.values.push_back(key_value(
    "command_alignment_check_count", std::to_string(command_alignment_check_count_)));
  status.values.push_back(key_value(
    "command_alignment_failure_count", std::to_string(command_alignment_failure_count_)));
  status.values.push_back(key_value(
    "older_publication_recovery_count",
    std::to_string(older_publication_recovery_count_)));
  status.values.push_back(key_value(
    "following_position_error_rad",
    optional_double_string(last_following_position_error_rad_)));
  status.values.push_back(key_value(
    "following_position_error_joint", last_following_position_error_joint_));
  status.values.push_back(key_value(
    "following_velocity_error_rad_s",
    optional_double_string(last_following_velocity_error_rad_s_)));
  status.values.push_back(key_value(
    "following_velocity_error_joint", last_following_velocity_error_joint_));
  status.values.push_back(key_value(
    "max_following_position_error_rad",
    std::to_string(max_following_position_error_rad_)));
  status.values.push_back(key_value(
    "max_following_position_error_joint", max_following_position_error_joint_));
  status.values.push_back(key_value(
    "max_following_velocity_error_rad_s",
    std::to_string(max_following_velocity_error_rad_s_)));
  status.values.push_back(key_value(
    "max_following_velocity_error_joint", max_following_velocity_error_joint_));
  status.values.push_back(key_value(
    "following_check_count", std::to_string(following_check_count_)));
  status.values.push_back(key_value(
    "following_error_count", std::to_string(following_error_count_)));
  status.values.push_back(key_value(
    "condition_number", optional_double_string(last_condition_number_)));
  status.values.push_back(key_value(
    "maximum_condition_number", std::to_string(maximum_condition_number_)));
  status.values.push_back(key_value(
    "singularity_guidance_requested",
    last_singularity_guidance_requested_ ? "true" : "false"));
  status.values.push_back(key_value(
    "singularity_guidance_available",
    last_singularity_guidance_available_ ? "true" : "false"));
  status.values.push_back(key_value(
    "singularity_guidance_activation",
    std::to_string(last_singularity_guidance_activation_)));
  status.values.push_back(key_value(
    "singularity_gradient_norm",
    optional_double_string(last_singularity_gradient_norm_)));
  status.values.push_back(key_value(
    "singularity_guidance_reference_rad_s",
    optional_double_string(last_singularity_guidance_reference_rad_s_)));
  status.values.push_back(key_value(
    "singularity_guidance_achieved_rad_s",
    optional_double_string(last_singularity_guidance_achieved_rad_s_)));
  status.values.push_back(key_value(
    "minimum_self_distance_m", optional_double_string(last_minimum_self_distance_m_)));
  status.values.push_back(key_value(
    "minimum_world_distance_m", optional_double_string(last_minimum_world_distance_m_)));
  status.values.push_back(key_value(
    "minimum_robust_headroom_m",
    optional_double_string(last_minimum_robust_headroom_m_)));
  status.values.push_back(key_value(
    "minimum_robust_headroom_pair", last_minimum_robust_headroom_pair_));
  status.values.push_back(key_value(
    "minimum_observed_self_distance_m",
    optional_double_string(minimum_observed_self_distance_m_)));
  status.values.push_back(key_value(
    "minimum_observed_world_distance_m",
    optional_double_string(minimum_observed_world_distance_m_)));
  status.values.push_back(key_value(
    "minimum_observed_robust_headroom_m",
    optional_double_string(minimum_observed_robust_headroom_m_)));
  status.values.push_back(key_value(
    "minimum_observed_robust_headroom_pair", minimum_observed_robust_headroom_pair_));
  status.values.push_back(key_value(
    "actual_minimum_self_distance_m",
    optional_double_string(last_actual_minimum_self_distance_m_)));
  status.values.push_back(key_value(
    "actual_minimum_world_distance_m",
    optional_double_string(last_actual_minimum_world_distance_m_)));
  status.values.push_back(key_value("actual_closest_pair", last_actual_closest_pair_));
  status.values.push_back(key_value(
    "actual_state_validation_count", std::to_string(actual_state_validation_count_)));
  status.values.push_back(key_value(
    "actual_state_validation_failure_count",
    std::to_string(actual_state_validation_failure_count_)));
  status.values.push_back(key_value(
    "planning_scene_collision_revision",
    std::to_string(planning_scene_collision_revision_.load(std::memory_order_acquire))));
  status.values.push_back(key_value(
    "planning_scene_invalidation_count",
    std::to_string(planning_scene_invalidation_count_)));
  status.values.push_back(key_value("closest_pair", last_closest_pair_));
  status.values.push_back(key_value(
    "active_collision_rows", std::to_string(last_active_collision_rows_)));
  status.values.push_back(key_value(
    "outside_collision_viability_contacts",
    std::to_string(last_outside_collision_viability_contacts_)));
  status.values.push_back(key_value(
    "position_error_m", optional_double_string(last_position_error_m_)));
  status.values.push_back(key_value(
    "screen_normal_pointing_error_rad",
    optional_double_string(last_screen_normal_pointing_error_rad_)));
  status.values.push_back(key_value(
    "screen_normal_pointing_error_mean_rad",
    optional_double_string(pointing_error_mean_rad)));
  status.values.push_back(key_value(
    "screen_normal_pointing_error_p99_rad",
    optional_double_string(pointing_error_p99_rad)));
  status.values.push_back(key_value(
    "screen_normal_pointing_error_max_rad",
    optional_double_string(pointing_error_max_rad)));
  status.values.push_back(key_value(
    "max_observed_joint_velocity_rad_s",
    std::to_string(max_observed_joint_velocity_rad_s_)));
  status.values.push_back(key_value(
    "max_observed_joint_acceleration_rad_s2",
    std::to_string(max_observed_joint_acceleration_rad_s2_)));
  status.values.push_back(key_value(
    "max_observed_joint_jerk_rad_s3",
    std::to_string(max_observed_joint_jerk_rad_s3_)));
  status.values.push_back(key_value(
    "max_feedback_joint_velocity_rad_s",
    std::to_string(max_feedback_joint_velocity_rad_s_)));
  status.values.push_back(key_value(
    "max_feedback_joint_velocity_name", max_feedback_joint_velocity_name_));
  status.values.push_back(key_value(
    "max_feedback_joint_acceleration_rad_s2",
    std::to_string(max_feedback_joint_acceleration_rad_s2_)));
  status.values.push_back(key_value(
    "max_feedback_joint_acceleration_name", max_feedback_joint_acceleration_name_));
  status.values.push_back(key_value(
    "max_feedback_joint_jerk_rad_s3",
    std::to_string(max_feedback_joint_jerk_rad_s3_)));
  status.values.push_back(key_value(
    "max_feedback_joint_jerk_name", max_feedback_joint_jerk_name_));
  status.values.push_back(key_value(
    "feedback_motion_limit_failure_count",
    std::to_string(feedback_motion_limit_failure_count_)));
  status.values.push_back(key_value(
    "solver_failure_count", std::to_string(solver_failure_count_)));
  status.values.push_back(key_value(
    "primary_only_count", std::to_string(primary_only_count_)));
  status.values.push_back(key_value(
    "braking_event_count", std::to_string(braking_event_count_)));
  status.values.push_back(key_value(
    "braking_tick_count", std::to_string(braking_tick_count_)));
  status.values.push_back(key_value(
    "emergency_event_count", std::to_string(emergency_event_count_)));
  status.values.push_back(key_value(
    "emergency_tick_count", std::to_string(emergency_tick_count_)));
  status.values.push_back(key_value(
    "last_braking_tail_points", std::to_string(last_braking_tail_points_)));
  status.values.push_back(key_value(
    "maximum_braking_tail_points", std::to_string(maximum_braking_tail_points_)));
  status.values.push_back(key_value(
    "braking_tail_generation_failure_count",
    std::to_string(braking_tail_generation_failure_count_)));
  status.values.push_back(key_value(
    "braking_tail_segment_validation_count",
    std::to_string(braking_tail_segment_validation_count_)));
  status.values.push_back(key_value(
    "braking_tail_collision_rejection_count",
    std::to_string(braking_tail_collision_rejection_count_)));
  status.values.push_back(key_value(
    "task_failure_count", std::to_string(task_failure_count_)));
  status.values.push_back(key_value(
    "moveit_validation_failure_count",
    std::to_string(moveit_validation_failure_count_)));
  status.values.push_back(key_value(
    "segment_validation_count", std::to_string(segment_validation_count_)));
  status.values.push_back(key_value(
    "segment_rejection_count", std::to_string(segment_rejection_count_)));
  status.values.push_back(key_value(
    "tracking_segment_rejection_count",
    std::to_string(tracking_segment_rejection_count_)));
  status.values.push_back(key_value(
    "braking_segment_rejection_count",
    std::to_string(braking_segment_rejection_count_)));
  status.values.push_back(key_value(
    "last_segment_rejection_attempt", last_segment_rejection_attempt_));
  status.values.push_back(key_value(
    "last_segment_rejection_reason", last_segment_rejection_reason_));
  status.values.push_back(key_value(
    "last_segment_rejection_input_valid",
    last_segment_rejection_input_valid_ ? "true" : "false"));
  status.values.push_back(key_value(
    "last_segment_rejection_unsafe",
    last_segment_rejection_unsafe_ ? "true" : "false"));
  status.values.push_back(key_value(
    "last_segment_rejection_evaluated_samples",
    std::to_string(last_segment_rejection_evaluated_samples_)));
  status.values.push_back(key_value(
    "last_segment_rejection_first_unsafe_sample",
    last_segment_rejection_first_unsafe_sample_ ==
    std::numeric_limits<std::size_t>::max() ?
    "unavailable" : std::to_string(last_segment_rejection_first_unsafe_sample_)));
  status.values.push_back(key_value(
    "last_segment_rejection_closest_pair", last_segment_rejection_closest_pair_));
  status.values.push_back(key_value(
    "last_segment_rejection_first_unsafe_fraction",
    optional_double_string(last_segment_rejection_first_unsafe_fraction_)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_self_distance_m",
    optional_double_string(last_segment_rejection_minimum_self_distance_m_)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_world_distance_m",
    optional_double_string(last_segment_rejection_minimum_world_distance_m_)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_distance_m",
    optional_double_string(last_segment_rejection_minimum_distance_m_)));

  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void CollisionAwareServoComponent::reset_tick_telemetry()
{
  last_condition_number_.reset();
  last_minimum_self_distance_m_.reset();
  last_minimum_world_distance_m_.reset();
  last_minimum_robust_headroom_m_.reset();
  last_minimum_robust_headroom_pair_ = "unavailable";
  last_closest_pair_ = "unavailable";
  last_active_collision_rows_ = 0;
  last_outside_collision_viability_contacts_ = 0;
  last_position_error_m_.reset();
  last_roll_error_rad_.reset();
  last_screen_normal_pointing_error_rad_.reset();
}

void CollisionAwareServoComponent::observe_minimum(
  std::optional<double> & observed_minimum,
  const double value) noexcept
{
  if (!std::isfinite(value)) {
    return;
  }
  if (!observed_minimum.has_value() || value < *observed_minimum) {
    observed_minimum = value;
  }
}

void CollisionAwareServoComponent::observe_clearances(
  const double self_distance_m,
  const double world_distance_m) noexcept
{
  observe_minimum(minimum_observed_self_distance_m_, self_distance_m);
  observe_minimum(minimum_observed_world_distance_m_, world_distance_m);
}

void CollisionAwareServoComponent::observe_motion_state(
  const control::JointMotionState & state) noexcept
{
  if (state.velocity.size() > 0 && state.velocity.allFinite()) {
    max_observed_joint_velocity_rad_s_ = std::max(
      max_observed_joint_velocity_rad_s_, state.velocity.cwiseAbs().maxCoeff());
  }
  if (state.acceleration.size() > 0 && state.acceleration.allFinite()) {
    max_observed_joint_acceleration_rad_s2_ = std::max(
      max_observed_joint_acceleration_rad_s2_,
      state.acceleration.cwiseAbs().maxCoeff());
  }
}

void CollisionAwareServoComponent::observe_command_transition(
  const control::JointMotionState & current_state,
  const moveit_servo::KinematicState & next_state) noexcept
{
  if (next_state.velocities.size() > 0 && next_state.velocities.allFinite()) {
    max_observed_joint_velocity_rad_s_ = std::max(
      max_observed_joint_velocity_rad_s_, next_state.velocities.cwiseAbs().maxCoeff());
  }
  if (next_state.accelerations.size() > 0 && next_state.accelerations.allFinite()) {
    max_observed_joint_acceleration_rad_s2_ = std::max(
      max_observed_joint_acceleration_rad_s2_,
      next_state.accelerations.cwiseAbs().maxCoeff());
  }
  if (current_state.acceleration.size() == next_state.accelerations.size() &&
    next_state.accelerations.size() > 0)
  {
    double maximum_jerk = 0.0;
    for (Eigen::Index index = 0; index < next_state.accelerations.size(); ++index) {
      const double jerk = std::abs(
        (next_state.accelerations[index] - current_state.acceleration[index]) /
        parameters_.control_period_sec);
      if (!std::isfinite(jerk)) {
        return;
      }
      maximum_jerk = std::max(maximum_jerk, jerk);
    }
    max_observed_joint_jerk_rad_s3_ = std::max(
      max_observed_joint_jerk_rad_s3_, maximum_jerk);
  }
}

void CollisionAwareServoComponent::record_qp_result(
  const std::chrono::steady_clock::time_point & start,
  const control::QpResult & result) noexcept
{
  const double duration_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  (void)qp_statistics_.add(duration_ms);
  if (!result.command_available()) {
    ++solver_failure_count_;
  }
  if (result.status == control::QpStatus::kPrimaryOnly) {
    ++primary_only_count_;
  }
}

void CollisionAwareServoComponent::log_qp_failure(
  const CommandAttempt attempt,
  const control::QpResult & result,
  const control::JointMotionState & motion_state,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const control::CollisionConstraintDiagnostics & collision_diagnostics,
  const double condition_number) const
{
  RCLCPP_WARN(
    get_logger(),
    "%s QP command unavailable: status=%d, primary=%d, secondary=%d, "
    "primary_iterations=%d, secondary_iterations=%d, constraints=%zu, "
    "distance_rows=%zu, outside_viability=%zu, condition_number=%.3f, "
    "min_distance=%.6f m, pair='%s'",
    command_attempt_name(attempt), static_cast<int>(result.status),
    static_cast<int>(result.primary_status), static_cast<int>(result.secondary_status),
    result.primary_iterations, result.secondary_iterations, constraints.size(),
    collision_diagnostics.active_distance_constraints,
    collision_diagnostics.outside_viability_contacts, condition_number,
    collision_diagnostics.minimum_distance_m, collision_diagnostics.closest_pair.c_str());
  for (const auto & pair : collision_diagnostics.pairs) {
    if (!pair.constraint_added) {
      continue;
    }
    RCLCPP_WARN(
      get_logger(),
      "Active distance row: '%s <-> %s', distance=%.6f m, gradient_norm=%.6f, "
      "jerk_safe=%.6f m/s, reachable=[%.6f, %.6f] m/s, applied_cap=%.6f m/s, "
      "outside_viability=%s",
      pair.first_body.c_str(), pair.second_body.c_str(), pair.distance_m,
      pair.gradient_norm, pair.jerk_aware_safe_approach_speed_mps,
      pair.minimum_reachable_approach_speed_mps,
      pair.maximum_reachable_approach_speed_mps, pair.applied_approach_speed_cap_mps,
      pair.outside_viability ? "true" : "false");
  }
  for (Eigen::Index index = 0; index < motion_state.position.size(); ++index) {
    RCLCPP_WARN(
      get_logger(), "QP state %s: q=%.6f, qdot=%.6f, qddot=%.6f",
      parameters_.joint_names[static_cast<std::size_t>(index)].c_str(),
      motion_state.position[index], motion_state.velocity[index],
      motion_state.acceleration[index]);
  }
}

void CollisionAwareServoComponent::record_segment_rejection(
  const CommandAttempt attempt,
  const control::CollisionSegmentValidationResult & result)
{
  ++segment_rejection_count_;
  if (attempt == CommandAttempt::kTracking) {
    ++tracking_segment_rejection_count_;
  } else {
    ++braking_segment_rejection_count_;
  }
  last_segment_rejection_attempt_ = command_attempt_name(attempt);
  last_segment_rejection_reason_ = result.failure_reason.empty() ?
    "hard clearance or protected-joint corridor violated" : result.failure_reason;
  last_segment_rejection_input_valid_ = result.input_valid;
  last_segment_rejection_unsafe_ = result.unsafe;
  last_segment_rejection_evaluated_samples_ = result.evaluated_samples;
  last_segment_rejection_first_unsafe_sample_ = result.first_unsafe_sample;
  last_segment_rejection_closest_pair_ = result.closest_pair.empty() ?
    "unavailable" : result.closest_pair;
  if (std::isfinite(result.first_unsafe_fraction)) {
    last_segment_rejection_first_unsafe_fraction_ = result.first_unsafe_fraction;
  } else {
    last_segment_rejection_first_unsafe_fraction_.reset();
  }
  if (std::isfinite(result.minimum_self_distance_m)) {
    last_segment_rejection_minimum_self_distance_m_ = result.minimum_self_distance_m;
  } else {
    last_segment_rejection_minimum_self_distance_m_.reset();
  }
  if (std::isfinite(result.minimum_world_distance_m)) {
    last_segment_rejection_minimum_world_distance_m_ = result.minimum_world_distance_m;
  } else {
    last_segment_rejection_minimum_world_distance_m_.reset();
  }
  if (std::isfinite(result.minimum_distance_m)) {
    last_segment_rejection_minimum_distance_m_ = result.minimum_distance_m;
  } else {
    last_segment_rejection_minimum_distance_m_.reset();
  }

  RCLCPP_WARN(
    get_logger(),
    "%s segment rejected: input_valid=%s, unsafe=%s, reason='%s', "
    "evaluated=%zu/%zu, first_unsafe_sample=%zu, first_unsafe_fraction=%.6f, "
    "min_self=%.6f m, min_world=%.6f m, min_distance=%.6f m, pair='%s'",
    command_attempt_name(attempt), result.input_valid ? "true" : "false",
    result.unsafe ? "true" : "false", last_segment_rejection_reason_.c_str(),
    result.evaluated_samples, result.requested_substeps + 1U,
    result.first_unsafe_sample, result.first_unsafe_fraction,
    result.minimum_self_distance_m, result.minimum_world_distance_m,
    result.minimum_distance_m, result.closest_pair.c_str());
}

void CollisionAwareServoComponent::publish_status(const std::int8_t code, std::string message)
{
  last_servo_status_code_ = code;
  last_servo_status_message_ = message;
  moveit_msgs::msg::ServoStatus status;
  status.code = code;
  status.message = std::move(message);
  status_publisher_->publish(status);
}

}  // namespace face_tracking_arm
