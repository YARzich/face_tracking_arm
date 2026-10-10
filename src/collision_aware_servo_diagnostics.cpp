// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

#include <iomanip>
#include <sstream>
#include <vector>

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
    case ControllerMode::kWaitingForSafeState:
      return "WAITING_FOR_SAFE_STATE";
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

std::string json_string(const std::string & value)
{
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (const char character : value) {
    switch (character) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (static_cast<unsigned char>(character) >= 0x20U) {
          escaped.push_back(character);
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string compact_event_json(const control::CompactEvent & event)
{
  return "{\"event\":" + json_string(event.event) +
         ",\"stamp_ns\":" + std::to_string(event.stamp_ns) +
         ",\"generation_id\":" + std::to_string(event.generation_id) +
         (event.reason.empty() ? std::string{} : ",\"reason\":" + json_string(event.reason)) +
         ",\"details\":" + event.details_json + "}";
}

void append_compact_tracking_diagnostics(
  diagnostic_msgs::msg::DiagnosticStatus & status,
  const control::MotionReferenceDiagnostics & reference,
  const std::vector<control::CompactEvent> & compact_events)
{
  status.values.push_back(key_value("motion_reference_state", reference.state));
  status.values.push_back(key_value("tracking_phase", reference.tracking_phase));
  status.values.push_back(key_value("recovery_mode", reference.recovery_mode));
  status.values.push_back(key_value(
      "disabled_limit_hold_latched",
      reference.disabled_limit_hold_latched ? "true" : "false"));
  status.values.push_back(key_value("compact_phase", reference.compact_phase));
  status.values.push_back(key_value(
      "compact_generation", std::to_string(reference.compact_generation)));
  status.values.push_back(key_value("compact_reason", reference.compact_reason));
  status.values.push_back(key_value(
      "compact_goal_reached", std::to_string(reference.compact_goal_reached)));
  status.values.push_back(key_value(
      "j1_hold_direction", std::to_string(reference.j1_hold_direction)));
  status.values.push_back(key_value("j1_hold_q", std::to_string(reference.j1_hold_q)));
  status.values.push_back(key_value(
      "compact_entry_radius_cap_m", std::to_string(reference.compact_entry_radius_cap_m)));
  status.values.push_back(key_value(
      "compact_effective_radius_cap_m",
      std::to_string(reference.compact_effective_radius_cap_m)));
  status.values.push_back(key_value(
      "compact_pitch_cap_rad", std::to_string(reference.compact_pitch_cap_rad)));
  status.values.push_back(key_value(
      "compact_radius_floor_latched", reference.compact_radius_floor_latched ? "true" : "false"));
  status.values.push_back(key_value(
      "upright_roll_guard_armed", reference.upright_roll_guard_armed ? "true" : "false"));
  status.values.push_back(key_value(
      "compact_actual_settled", reference.compact_actual_settled ? "true" : "false"));
  status.values.push_back(key_value(
      "compact_return_full_face_active",
      reference.compact_return_full_face_active ? "true" : "false"));
  status.values.push_back(key_value(
      "compact_return_face_qualified",
      reference.compact_return_face_qualified ? "true" : "false"));
  status.values.push_back(key_value(
      "compact_return_face_dwell_sec", std::to_string(reference.compact_return_face_dwell_sec)));
  status.values.push_back(key_value(
      "planned_path_max_radius_m", std::to_string(reference.planned_path_max_radius_m)));
  status.values.push_back(key_value(
      "planned_path_max_roll_error_rad",
      std::to_string(reference.planned_path_max_roll_error_rad)));
  status.values.push_back(key_value(
      "planned_path_max_abs_pitch_rad",
      std::to_string(reference.planned_path_max_abs_pitch_rad)));
  status.values.push_back(key_value(
      "compact_actual_goal_error_rad",
      optional_double_string(reference.compact_actual_goal_error_rad)));
  status.values.push_back(key_value(
      "compact_actual_speed_max_rad_s",
      optional_double_string(reference.compact_actual_speed_max_rad_s)));
  status.values.push_back(key_value(
      "compact_queue_tail_goal_error_rad",
      optional_double_string(reference.compact_queue_tail_goal_error_rad)));
  status.values.push_back(key_value(
      "compact_queue_tail_speed_max_rad_s",
      optional_double_string(reference.compact_queue_tail_speed_max_rad_s)));
  status.values.push_back(key_value(
      "compact_queue_tail_acceleration_max_rad_s2",
      optional_double_string(reference.compact_queue_tail_acceleration_max_rad_s2)));
  status.values.push_back(key_value("compact_event_count",
      std::to_string(compact_events.size())));
  status.values.push_back(key_value("compact_event_overflow_count",
      std::to_string(reference.compact_event_overflow_count)));
  for (std::size_t index = 0; index < compact_events.size(); ++index) {
    std::string key{"compact_event_"};
    key.append(index < 10U ? "00" : (index < 100U ? "0" : ""));
    key += std::to_string(index);
    status.values.push_back(key_value(std::move(key), compact_event_json(compact_events[index])));
  }
}

}  // namespace

void CollisionAwareServoComponent::publish_runtime_diagnostics()
{
  const auto control_mean_ms = telemetry_.control_tick_statistics.mean();
  const auto control_p99_ms = telemetry_.control_tick_statistics.p99();
  const auto control_max_ms = telemetry_.control_tick_statistics.maximum();
  const auto qp_mean_ms = telemetry_.qp_statistics.mean();
  const auto qp_p99_ms = telemetry_.qp_statistics.p99();
  const auto qp_max_ms = telemetry_.qp_statistics.maximum();
  const auto collision_build_mean_ms = telemetry_.collision_build_statistics.mean();
  const auto collision_build_p99_ms = telemetry_.collision_build_statistics.p99();
  const auto collision_build_max_ms = telemetry_.collision_build_statistics.maximum();
  const auto segment_validation_mean_ms = telemetry_.segment_validation_statistics.mean();
  const auto segment_validation_p99_ms = telemetry_.segment_validation_statistics.p99();
  const auto segment_validation_max_ms = telemetry_.segment_validation_statistics.maximum();
  const auto path_validation_mean_ms = telemetry_.path_validation_statistics.mean();
  const auto path_validation_p99_ms = telemetry_.path_validation_statistics.p99();
  const auto path_validation_max_ms = telemetry_.path_validation_statistics.maximum();
  const auto pointing_error_mean_rad = telemetry_.pointing_error_statistics.mean();
  const auto pointing_error_p99_rad = telemetry_.pointing_error_statistics.p99();
  const auto pointing_error_max_rad = telemetry_.pointing_error_statistics.maximum();

  const rclcpp::Time current_time = now();
  const auto current_wall_time = std::chrono::steady_clock::now();
  std::optional<double> latest_pose_age_ms;
  std::optional<double> rearm_deadline_remaining_sec;
  std::optional<double> rearm_wall_deadline_remaining_sec;
  if (commands_.rearm_deadline.has_value() && *commands_.rearm_deadline >= current_time) {
    rearm_deadline_remaining_sec = (*commands_.rearm_deadline - current_time).seconds();
  }
  if (commands_.rearm_wall_deadline.has_value() &&
    *commands_.rearm_wall_deadline >= current_wall_time)
  {
    rearm_wall_deadline_remaining_sec = std::chrono::duration<double>(
      *commands_.rearm_wall_deadline - current_wall_time).count();
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
  if (!commands_.queue.empty()) {
    queue_horizon_sec = std::max(
      0.0, (commands_.queue.back().time_stamp - current_time).seconds());
  }
  const auto * published_record = commands_.activePublication();
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
  status.hardware_id = parameters_.planning_group_name;
  if (controller_mode_ == ControllerMode::kWaitingForSafeState) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  } else if (telemetry_.last_servo_status_code == moveit_msgs::msg::ServoStatus::NO_WARNING) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
  }
  status.message = controller_mode_name(controller_mode_);
  status.values.reserve(128);
  status.values.push_back(key_value("mode", controller_mode_name(controller_mode_)));
  status.values.push_back(key_value(
    "servo_status_code", std::to_string(telemetry_.last_servo_status_code)));
  status.values.push_back(key_value("servo_status_message", telemetry_.last_servo_status_message));
  const auto & reference = motion_reference_->diagnostics();
  const auto compact_events = motion_reference_->drainCompactEvents();
  append_compact_tracking_diagnostics(status, reference, compact_events);
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
      optional_double_string(telemetry_.last_roll_error_rad)));
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
    "control_tick_lifetime_max_ms", std::to_string(telemetry_.lifetime_maximum_control_tick_ms)));
  status.values.push_back(key_value(
    "control_cycle_mean_ms", optional_double_string(telemetry_.control_cycle_statistics.mean())));
  status.values.push_back(key_value(
    "control_cycle_p99_ms", optional_double_string(telemetry_.control_cycle_statistics.p99())));
  status.values.push_back(key_value(
    "control_cycle_max_ms", optional_double_string(telemetry_.control_cycle_statistics.maximum())));
  status.values.push_back(key_value("qp_mean_ms", optional_double_string(qp_mean_ms)));
  status.values.push_back(key_value("qp_p99_ms", optional_double_string(qp_p99_ms)));
  status.values.push_back(key_value("qp_max_ms", optional_double_string(qp_max_ms)));
  status.values.push_back(key_value(
    "collision_build_mean_ms", optional_double_string(collision_build_mean_ms)));
  status.values.push_back(key_value(
    "collision_self_query_mean_ms",
    optional_double_string(telemetry_.collision_self_query_statistics.mean())));
  status.values.push_back(key_value(
    "collision_world_query_mean_ms",
    optional_double_string(telemetry_.collision_world_query_statistics.mean())));
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
    "missed_control_deadlines", std::to_string(telemetry_.missed_control_deadlines)));
  status.values.push_back(key_value(
    "skipped_control_periods", std::to_string(telemetry_.skipped_control_periods)));
  status.values.push_back(key_value(
    "timeline_recovery_events", std::to_string(commands_.timeline_recovery_events)));
  status.values.push_back(key_value(
    "timeline_recovered_periods", std::to_string(commands_.timeline_recovered_periods)));
  status.values.push_back(key_value(
    "timeline_recovery_failures", std::to_string(telemetry_.timeline_recovery_failures)));
  status.values.push_back(key_value(
    "published_brake_continuation_count",
    std::to_string(commands_.published_brake_continuation_count)));
  status.values.push_back(key_value(
    "controlled_rearm_count", std::to_string(telemetry_.controlled_rearm_count)));
  status.values.push_back(key_value(
    "last_controlled_rearm_reason", telemetry_.last_controlled_rearm_reason));
  status.values.push_back(key_value(
    "startup_retry_count", std::to_string(telemetry_.startup_retry_count)));
  status.values.push_back(key_value(
    "last_startup_retry_reason", telemetry_.last_startup_retry_reason));
  status.values.push_back(key_value(
    "clock_not_advancing_observations",
    std::to_string(telemetry_.clock_not_advancing_observations)));
  status.values.push_back(key_value(
    "clock_rewind_count", std::to_string(telemetry_.clock_rewind_count)));
  status.values.push_back(key_value(
    "queue_horizon_sec", optional_double_string(queue_horizon_sec)));
  status.values.push_back(key_value(
    "published_queue_horizon_sec",
    optional_double_string(published_queue_horizon_sec)));
  status.values.push_back(key_value("queue_size", std::to_string(commands_.queue.size())));
  status.values.push_back(key_value(
    "queue_hard_limit", std::to_string(commands_.max_queue_points)));
  status.values.push_back(key_value(
    "published_execution_queue_size",
    std::to_string(
      published_record == nullptr ? 0U : published_record->execution_queue.size())));
  status.values.push_back(key_value(
    "published_trajectory_history_size",
    std::to_string(commands_.publications.size())));
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
    "published_command_count", std::to_string(telemetry_.published_command_count)));
  status.values.push_back(key_value(
    "last_published_horizon_sec",
    optional_double_string(telemetry_.last_published_horizon_sec)));
  status.values.push_back(key_value(
    "minimum_published_horizon_sec",
    optional_double_string(telemetry_.minimum_published_horizon_sec)));
  status.values.push_back(key_value(
    "last_replacement_lead_sec",
    optional_double_string(telemetry_.last_replacement_lead_sec)));
  status.values.push_back(key_value(
    "minimum_replacement_lead_sec",
    optional_double_string(telemetry_.minimum_replacement_lead_sec)));
  status.values.push_back(key_value(
    "startup_warmup_cycles",
    std::to_string(commands_.startup_warmup_cycles)));
  status.values.push_back(key_value(
    "startup_warmup_complete", commands_.startup_warmup_complete ? "true" : "false"));
  status.values.push_back(key_value(
    "bootstrap_publication_count", std::to_string(telemetry_.bootstrap_publication_count)));
  status.values.push_back(key_value(
    "bootstrap_ack_pending", commands_.bootstrap_ack_pending ? "true" : "false"));
  status.values.push_back(key_value(
    "bootstrap_publication_id",
    commands_.bootstrap_publication_id.has_value() ?
    std::to_string(*commands_.bootstrap_publication_id) : "unavailable"));
  status.values.push_back(key_value(
    "bootstrap_acknowledgement_count",
    std::to_string(telemetry_.bootstrap_acknowledgement_count)));
  status.values.push_back(key_value(
    "bootstrap_segment_validation_count",
    std::to_string(telemetry_.bootstrap_segment_validation_count)));
  status.values.push_back(key_value("rearm_pending", commands_.rearm_pending ? "true" : "false"));
  status.values.push_back(key_value(
    "published_tail_rearm_pending",
    commands_.published_tail_rearm_pending ? "true" : "false"));
  status.values.push_back(key_value(
    "stable_rearm_samples", std::to_string(commands_.stable_rearm_samples)));
  status.values.push_back(key_value(
    "rearm_deadline_remaining_sec",
    optional_double_string(rearm_deadline_remaining_sec)));
  status.values.push_back(key_value(
    "rearm_wall_deadline_remaining_sec",
    optional_double_string(rearm_wall_deadline_remaining_sec)));
  status.values.push_back(key_value(
    "rearm_timeout_count", std::to_string(telemetry_.rearm_timeout_count)));
  status.values.push_back(key_value(
    "consecutive_automatic_rearms",
    std::to_string(commands_.consecutive_automatic_rearms)));
  status.values.push_back(key_value(
    "healthy_publications_since_rearm",
    std::to_string(commands_.healthy_publications_since_rearm)));
  status.values.push_back(key_value(
    "controller_speed_scaling_factor",
    optional_double_string(telemetry_.last_controller_speed_scaling_factor)));
  status.values.push_back(key_value(
    "controller_speed_scaling_failure_count",
    std::to_string(telemetry_.controller_speed_scaling_failure_count)));
  status.values.push_back(key_value(
    "controller_reference_alignment_error_rad",
    optional_double_string(telemetry_.last_controller_reference_alignment_error_rad)));
  status.values.push_back(key_value(
    "controller_output_alignment_error_rad",
    optional_double_string(telemetry_.last_controller_output_alignment_error_rad)));
  status.values.push_back(key_value(
    "max_controller_reference_alignment_error_rad",
    std::to_string(telemetry_.max_controller_reference_alignment_error_rad)));
  status.values.push_back(key_value(
    "max_controller_output_alignment_error_rad",
    std::to_string(telemetry_.max_controller_output_alignment_error_rad)));
  status.values.push_back(key_value(
    "command_alignment_check_count", std::to_string(telemetry_.command_alignment_check_count)));
  status.values.push_back(key_value(
    "command_alignment_failure_count", std::to_string(telemetry_.command_alignment_failure_count)));
  status.values.push_back(key_value(
    "older_publication_recovery_count",
    std::to_string(telemetry_.older_publication_recovery_count)));
  status.values.push_back(key_value(
    "following_position_error_rad",
    optional_double_string(telemetry_.last_following_position_error_rad)));
  status.values.push_back(key_value(
    "following_position_error_joint", telemetry_.last_following_position_error_joint));
  status.values.push_back(key_value(
    "following_velocity_error_rad_s",
    optional_double_string(telemetry_.last_following_velocity_error_rad_s)));
  status.values.push_back(key_value(
    "following_velocity_error_joint", telemetry_.last_following_velocity_error_joint));
  status.values.push_back(key_value(
    "max_following_position_error_rad",
    std::to_string(telemetry_.max_following_position_error_rad)));
  status.values.push_back(key_value(
    "max_following_position_error_joint", telemetry_.max_following_position_error_joint));
  status.values.push_back(key_value(
    "max_following_velocity_error_rad_s",
    std::to_string(telemetry_.max_following_velocity_error_rad_s)));
  status.values.push_back(key_value(
    "max_following_velocity_error_joint", telemetry_.max_following_velocity_error_joint));
  status.values.push_back(key_value(
    "following_check_count", std::to_string(telemetry_.following_check_count)));
  status.values.push_back(key_value(
    "following_error_count", std::to_string(telemetry_.following_error_count)));
  // Compatibility keys for retired mechanisms; no measurement exists.
  for (const auto * key : {"condition_number", "maximum_condition_number",
      "singularity_guidance_requested", "singularity_guidance_available",
      "singularity_guidance_activation", "singularity_gradient_norm",
      "singularity_guidance_reference_rad_s", "singularity_guidance_achieved_rad_s",
      "rearm_budget_exhaustion_count"})
  {
    status.values.push_back(key_value(key, "unavailable"));
  }
  status.values.push_back(key_value(
    "minimum_self_distance_m", optional_double_string(telemetry_.last_minimum_self_distance_m)));
  status.values.push_back(key_value(
    "minimum_world_distance_m", optional_double_string(telemetry_.last_minimum_world_distance_m)));
  status.values.push_back(key_value(
    "minimum_robust_headroom_m",
    optional_double_string(telemetry_.last_minimum_robust_headroom_m)));
  status.values.push_back(key_value(
    "minimum_robust_headroom_pair", telemetry_.last_minimum_robust_headroom_pair));
  status.values.push_back(key_value(
    "minimum_observed_self_distance_m",
    optional_double_string(telemetry_.minimum_observed_self_distance_m)));
  status.values.push_back(key_value(
    "minimum_observed_world_distance_m",
    optional_double_string(telemetry_.minimum_observed_world_distance_m)));
  status.values.push_back(key_value(
    "minimum_observed_robust_headroom_m",
    optional_double_string(telemetry_.minimum_observed_robust_headroom_m)));
  status.values.push_back(key_value(
    "minimum_observed_robust_headroom_pair", telemetry_.minimum_observed_robust_headroom_pair));
  status.values.push_back(key_value(
    "actual_minimum_self_distance_m",
    optional_double_string(telemetry_.last_actual_minimum_self_distance_m)));
  status.values.push_back(key_value(
    "actual_minimum_world_distance_m",
    optional_double_string(telemetry_.last_actual_minimum_world_distance_m)));
  status.values.push_back(key_value("actual_closest_pair", telemetry_.last_actual_closest_pair));
  status.values.push_back(key_value(
    "actual_state_validation_count", std::to_string(telemetry_.actual_state_validation_count)));
  status.values.push_back(key_value(
    "actual_state_validation_failure_count",
    std::to_string(telemetry_.actual_state_validation_failure_count)));
  status.values.push_back(key_value(
    "planning_scene_collision_revision",
    std::to_string(planning_scene_collision_revision_.load(std::memory_order_acquire))));
  status.values.push_back(key_value(
    "planning_scene_invalidation_count",
    std::to_string(telemetry_.planning_scene_invalidation_count)));
  status.values.push_back(key_value("closest_pair", telemetry_.last_closest_pair));
  status.values.push_back(key_value(
    "active_collision_rows", std::to_string(telemetry_.last_active_collision_rows)));
  status.values.push_back(key_value(
    "outside_collision_viability_contacts",
    std::to_string(telemetry_.last_outside_collision_viability_contacts)));
  status.values.push_back(key_value(
    "position_error_m", optional_double_string(telemetry_.last_position_error_m)));
  status.values.push_back(key_value(
    "screen_normal_pointing_error_rad",
    optional_double_string(telemetry_.last_screen_normal_pointing_error_rad)));
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
    std::to_string(telemetry_.max_observed_joint_velocity_rad_s)));
  status.values.push_back(key_value(
    "max_observed_joint_acceleration_rad_s2",
    std::to_string(telemetry_.max_observed_joint_acceleration_rad_s2)));
  status.values.push_back(key_value(
    "max_observed_joint_jerk_rad_s3",
    std::to_string(telemetry_.max_observed_joint_jerk_rad_s3)));
  status.values.push_back(key_value(
    "max_feedback_joint_velocity_rad_s",
    std::to_string(telemetry_.max_feedback_joint_velocity_rad_s)));
  status.values.push_back(key_value(
    "max_feedback_joint_velocity_name", telemetry_.max_feedback_joint_velocity_name));
  status.values.push_back(key_value(
    "max_feedback_joint_acceleration_rad_s2",
    std::to_string(telemetry_.max_feedback_joint_acceleration_rad_s2)));
  status.values.push_back(key_value(
    "max_feedback_joint_acceleration_name", telemetry_.max_feedback_joint_acceleration_name));
  status.values.push_back(key_value(
    "max_feedback_joint_jerk_rad_s3",
    std::to_string(telemetry_.max_feedback_joint_jerk_rad_s3)));
  status.values.push_back(key_value(
    "max_feedback_joint_jerk_name", telemetry_.max_feedback_joint_jerk_name));
  status.values.push_back(key_value(
    "feedback_motion_limit_failure_count",
    std::to_string(telemetry_.feedback_motion_limit_failure_count)));
  status.values.push_back(key_value(
    "solver_failure_count", std::to_string(telemetry_.solver_failure_count)));
  status.values.push_back(key_value(
    "primary_only_count", std::to_string(telemetry_.primary_only_count)));
  status.values.push_back(key_value(
    "braking_event_count", std::to_string(telemetry_.braking_event_count)));
  status.values.push_back(key_value(
    "braking_tick_count", std::to_string(telemetry_.braking_tick_count)));
  status.values.push_back(key_value(
    "emergency_event_count", std::to_string(telemetry_.emergency_event_count)));
  status.values.push_back(key_value(
    "emergency_tick_count", std::to_string(telemetry_.emergency_tick_count)));
  status.values.push_back(key_value(
    "last_braking_tail_points", std::to_string(telemetry_.last_braking_tail_points)));
  status.values.push_back(key_value(
    "maximum_braking_tail_points", std::to_string(telemetry_.maximum_braking_tail_points)));
  status.values.push_back(key_value(
    "braking_tail_generation_failure_count",
    std::to_string(telemetry_.braking_tail_generation_failure_count)));
  status.values.push_back(key_value(
    "braking_tail_segment_validation_count",
    std::to_string(telemetry_.braking_tail_segment_validation_count)));
  status.values.push_back(key_value(
    "braking_tail_collision_rejection_count",
    std::to_string(telemetry_.braking_tail_collision_rejection_count)));
  status.values.push_back(key_value(
    "braking_tail_distance_query_count",
    std::to_string(telemetry_.braking_tail_distance_query_count)));
  status.values.push_back(key_value(
    "braking_tail_certified_path_block_count",
    std::to_string(telemetry_.braking_tail_certified_path_block_count)));
  status.values.push_back(key_value(
    "braking_tail_inconclusive_path_block_count",
    std::to_string(telemetry_.braking_tail_inconclusive_path_block_count)));
  status.values.push_back(key_value(
    "last_braking_tail_certified_segment_count",
    std::to_string(telemetry_.last_braking_tail_certified_segment_count)));
  status.values.push_back(key_value(
    "last_braking_tail_uncertified_segment_count",
    std::to_string(telemetry_.last_braking_tail_uncertified_segment_count)));
  status.values.push_back(key_value(
    "task_failure_count", "unavailable"));
  status.values.push_back(key_value(
    "moveit_validation_failure_count",
    std::to_string(telemetry_.moveit_validation_failure_count)));
  status.values.push_back(key_value(
    "segment_validation_count", std::to_string(telemetry_.segment_validation_count)));
  status.values.push_back(key_value(
    "segment_rejection_count", std::to_string(telemetry_.segment_rejection_count)));
  status.values.push_back(key_value(
    "tracking_segment_rejection_count",
    std::to_string(telemetry_.tracking_segment_rejection_count)));
  status.values.push_back(key_value(
    "braking_segment_rejection_count",
    std::to_string(telemetry_.braking_segment_rejection_count)));
  status.values.push_back(key_value(
    "last_segment_rejection_attempt", telemetry_.last_segment_rejection_attempt));
  status.values.push_back(key_value(
    "last_segment_rejection_reason", telemetry_.last_segment_rejection_reason));
  status.values.push_back(key_value(
    "last_segment_rejection_input_valid",
    telemetry_.last_segment_rejection_input_valid ? "true" : "false"));
  status.values.push_back(key_value(
    "last_segment_rejection_unsafe",
    telemetry_.last_segment_rejection_unsafe ? "true" : "false"));
  status.values.push_back(key_value(
    "last_segment_rejection_evaluated_samples",
    std::to_string(telemetry_.last_segment_rejection_evaluated_samples)));
  status.values.push_back(key_value(
    "last_segment_rejection_first_unsafe_sample",
    telemetry_.last_segment_rejection_first_unsafe_sample ==
    std::numeric_limits<std::size_t>::max() ?
    "unavailable" : std::to_string(telemetry_.last_segment_rejection_first_unsafe_sample)));
  status.values.push_back(key_value(
    "last_segment_rejection_closest_pair", telemetry_.last_segment_rejection_closest_pair));
  status.values.push_back(key_value(
    "last_segment_rejection_first_unsafe_fraction",
    optional_double_string(telemetry_.last_segment_rejection_first_unsafe_fraction)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_self_distance_m",
    optional_double_string(telemetry_.last_segment_rejection_minimum_self_distance_m)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_world_distance_m",
    optional_double_string(telemetry_.last_segment_rejection_minimum_world_distance_m)));
  status.values.push_back(key_value(
    "last_segment_rejection_minimum_distance_m",
    optional_double_string(telemetry_.last_segment_rejection_minimum_distance_m)));

  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void CollisionAwareServoComponent::log_qp_failure(
  const CommandAttempt attempt,
  const control::QpResult & result,
  const control::JointMotionState & motion_state,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const control::CollisionConstraintDiagnostics & collision_diagnostics) const
{
  RCLCPP_WARN(
    get_logger(),
    "%s QP command unavailable: status=%d, primary=%d, secondary=%d, "
    "primary_iterations=%d, secondary_iterations=%d, constraints=%zu, "
    "distance_rows=%zu, outside_viability=%zu, "
    "min_distance=%.6f m, pair='%s'",
    command_attempt_name(attempt), static_cast<int>(result.status),
    static_cast<int>(result.primary_status), static_cast<int>(result.secondary_status),
    result.primary_iterations, result.secondary_iterations, constraints.size(),
    collision_diagnostics.active_distance_constraints,
    collision_diagnostics.outside_viability_contacts,
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
  for (std::size_t row_index = 0; row_index < constraints.size(); ++row_index) {
    const auto & row = constraints[row_index];
    std::ostringstream coefficients;
    coefficients << std::setprecision(17) << '[';
    for (Eigen::Index index = 0; index < row.coefficients.size(); ++index) {
      if (index != 0) {
        coefficients << ", ";
      }
      coefficients << row.coefficients[index];
    }
    coefficients << ']';
    RCLCPP_WARN(
      get_logger(), "QP input row %zu: lower=%.17g, upper=%.17g, coefficients=%s",
      row_index, row.lower_bound, row.upper_bound, coefficients.str().c_str());
  }
  const auto log_limit_vector = [this](const char * name, const Eigen::VectorXd & values) {
      std::ostringstream formatted;
      formatted << std::setprecision(17) << '[';
      for (Eigen::Index index = 0; index < values.size(); ++index) {
        if (index != 0) {
          formatted << ", ";
        }
        formatted << values[index];
      }
      formatted << ']';
      RCLCPP_WARN(get_logger(), "QP motion limit %s=%s", name, formatted.str().c_str());
    };
  const auto & qp_config = velocity_qp_->config();
  RCLCPP_WARN(
    get_logger(), "QP config period_sec=%.17g, qp_residual_latency_sec=%.17g, "
    "collision_residual_latency_sec=%.17g, solver_abs_tol=%.17g, "
    "solver_rel_tol=%.17g, solver_time_limit_sec=%.17g, solver_max_iterations=%d, "
    "solution_feasibility_tolerance=%.17g",
    qp_config.period_sec, qp_config.residual_command_latency_sec,
    parameters_.residual_command_latency_sec, qp_config.solver_absolute_tolerance,
    qp_config.solver_relative_tolerance, qp_config.solver_time_limit_sec,
    qp_config.solver_max_iterations, qp_config.solution_feasibility_tolerance);
  log_limit_vector("lower_position", motion_limits_.lower_position);
  log_limit_vector("upper_position", motion_limits_.upper_position);
  log_limit_vector("max_velocity", motion_limits_.max_velocity);
  log_limit_vector("max_acceleration", motion_limits_.max_acceleration);
  log_limit_vector("max_jerk", motion_limits_.max_jerk);
  for (Eigen::Index index = 0; index < motion_state.position.size(); ++index) {
    RCLCPP_WARN(
      get_logger(), "QP state %s: q=%.17g, qdot=%.17g, qddot=%.17g",
      parameters_.joint_names[static_cast<std::size_t>(index)].c_str(),
      motion_state.position[index], motion_state.velocity[index],
      motion_state.acceleration[index]);
  }
}

void CollisionAwareServoComponent::record_segment_rejection(
  const CommandAttempt attempt,
  const control::CollisionSegmentValidationResult & result)
{
  telemetry_.recordSegmentRejection(attempt, result);

  RCLCPP_WARN(
    get_logger(),
    "%s segment rejected: input_valid=%s, unsafe=%s, reason='%s', "
    "query_samples=%zu, dense_grid_samples=%zu, distance_queries=%zu, "
    "certified_blocks=%zu, inconclusive_blocks=%zu, "
    "first_unsafe_sample=%zu, first_unsafe_fraction=%.6f, "
    "min_self=%.6f m, min_world=%.6f m, min_distance=%.6f m, pair='%s'",
    command_attempt_name(attempt), result.input_valid ? "true" : "false",
    result.unsafe ? "true" : "false", telemetry_.last_segment_rejection_reason.c_str(),
    result.evaluated_samples, result.requested_substeps + 1U,
    result.distance_query_count, result.certified_path_block_count,
    result.inconclusive_path_block_count,
    result.first_unsafe_sample, result.first_unsafe_fraction,
    result.minimum_self_distance_m, result.minimum_world_distance_m,
    result.minimum_distance_m, result.closest_pair.c_str());
}

void CollisionAwareServoComponent::publish_status(const std::int8_t code, std::string message)
{
  telemetry_.last_servo_status_code = code;
  telemetry_.last_servo_status_message = message;
  moveit_msgs::msg::ServoStatus status;
  status.code = code;
  status.message = std::move(message);
  status_publisher_->publish(status);
}

}  // namespace face_tracking_arm
