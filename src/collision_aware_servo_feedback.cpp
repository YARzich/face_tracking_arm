// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr std::size_t kRequiredStableRearmSamples = 3;

constexpr double kFeedbackLimitTolerance = 1.0e-6;

constexpr double kMaximumRecoverableFollowingVelocityErrorRadps = 0.300;

}  // namespace

std::optional<control_msgs::msg::JointTrajectoryControllerState> CollisionAwareServoComponent::
current_controller_state() const
{
  std::lock_guard<std::mutex> lock(controller_state_mutex_);
  if (!latest_controller_state_.has_value()) {
    return std::nullopt;
  }
  const auto & state = *latest_controller_state_;
  const rclcpp::Time current_time = now();
  const rclcpp::Time stamp(state.header.stamp, current_time.get_clock_type());
  const rclcpp::Time oldest_allowed = current_time -
    rclcpp::Duration::from_seconds(parameters_.state_feedback_timeout_sec);
  const std::size_t size = parameters_.joint_names.size();
  const auto finite_vector = [](const std::vector<double> & values) {
      return std::all_of(values.begin(), values.end(), [](const double value) {
                 return std::isfinite(value);
        });
    };
  if (stamp.nanoseconds() <= 0 || stamp > current_time || stamp < oldest_allowed ||
    state.joint_names != parameters_.joint_names ||
    state.reference.positions.size() != size || state.reference.velocities.size() != size ||
    state.feedback.positions.size() != size || state.feedback.velocities.size() != size ||
    state.output.positions.size() != size ||
    !finite_vector(state.reference.positions) || !finite_vector(state.reference.velocities) ||
    !finite_vector(state.feedback.positions) || !finite_vector(state.feedback.velocities) ||
    !finite_vector(state.output.positions) || !std::isfinite(state.speed_scaling_factor))
  {
    return std::nullopt;
  }
  return state;
}

ActualFeedbackSafety CollisionAwareServoComponent::validate_controller_feedback_collision_state(
  const control_msgs::msg::JointTrajectoryControllerState & controller_state,
  const std::string & context, std::string & reason)
{
  if ((command_epoch_active_ || published_tail_rearm_pending_) &&
    planning_scene_collision_revision_.load(std::memory_order_acquire) !=
    last_validated_scene_revision_)
  {
    ++planning_scene_invalidation_count_;
    reason = context + " detected changed collision geometry for a published trajectory";
    return ActualFeedbackSafety::kUnsafe;
  }
  auto measured_robot_state =
    planning_scene_monitor_->getStateMonitor()->getCurrentState();
  if (!measured_robot_state) {
    reason = context + " is waiting for the measured MoveIt robot state";
    return ActualFeedbackSafety::kWaiting;
  }
  measured_robot_state = std::make_shared<moveit::core::RobotState>(*measured_robot_state);
  measured_robot_state->setJointGroupPositions(
    joint_model_group_, controller_state.feedback.positions);
  measured_robot_state->update(true);

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
  if (!planning_scene) {
    reason = context + " lost the PlanningScene";
    return ActualFeedbackSafety::kUnsafe;
  }
  const control::CollisionActualStateValidationResult validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *measured_robot_state, *joint_model_group_);
  ++actual_state_validation_count_;
  last_actual_minimum_self_distance_m_ =
    std::isfinite(validation.minimum_self_distance_m) ?
    std::optional<double>{validation.minimum_self_distance_m} : std::nullopt;
  last_actual_minimum_world_distance_m_ =
    std::isfinite(validation.minimum_world_distance_m) ?
    std::optional<double>{validation.minimum_world_distance_m} : std::nullopt;
  last_actual_closest_pair_ = validation.closest_pair.empty() ?
    "outside hard-clearance query" : validation.closest_pair;
  observe_clearances(
    validation.minimum_self_distance_m, validation.minimum_world_distance_m);
  if (!validation.input_valid || validation.unsafe) {
    ++actual_state_validation_failure_count_;
    reason = !validation.input_valid ?
      context + " actual-state validation failed: " + validation.failure_reason :
      context + " measured a hard-clearance/corridor violation for " +
      last_actual_closest_pair_;
    return ActualFeedbackSafety::kUnsafe;
  }
  return ActualFeedbackSafety::kSafe;
}

RearmReadiness CollisionAwareServoComponent::safe_rearm_ready(
  const rclcpp::Time & current_time, std::string & reason)
{
  const auto state_monitor = planning_scene_monitor_->getStateMonitor();
  const rclcpp::Time oldest_allowed_state = current_time -
    rclcpp::Duration::from_seconds(parameters_.state_feedback_timeout_sec);
  const auto [measured_robot_state, state_time] = state_monitor->getCurrentStateAndTime();
  if (!measured_robot_state || state_time > current_time || state_time < oldest_allowed_state ||
    !state_monitor->haveCompleteState(oldest_allowed_state))
  {
    stable_rearm_samples_ = 0;
    reason = "Safe re-arm is waiting for fresh complete robot feedback";
    return RearmReadiness::kWaiting;
  }

  const auto state = current_controller_state();
  if (!state.has_value()) {
    stable_rearm_samples_ = 0;
    reason = "Safe re-arm is waiting for fresh finite controller feedback";
    return RearmReadiness::kWaiting;
  }
  const rclcpp::Time sample_time(state->header.stamp, current_time.get_clock_type());
  if (last_rearm_sample_time_.has_value() && sample_time <= *last_rearm_sample_time_) {
    reason = "Safe re-arm is waiting for the next controller feedback sample";
    return RearmReadiness::kWaiting;
  }
  last_rearm_sample_time_ = sample_time;
  observe_controller_feedback(*state, current_time.get_clock_type());

  const auto maximum_absolute_difference = [](
    const std::vector<double> & first, const std::vector<double> & second) {
      if (first.size() != second.size()) {
        return std::numeric_limits<double>::infinity();
      }
      double maximum = 0.0;
      for (std::size_t index = 0; index < first.size(); ++index) {
        maximum = std::max(maximum, std::abs(first[index] - second[index]));
      }
      return maximum;
    };
  const auto maximum_expected_difference = [](
    const std::vector<double> & actual, const Eigen::VectorXd & expected) {
      if (actual.size() != static_cast<std::size_t>(expected.size())) {
        return std::numeric_limits<double>::infinity();
      }
      double maximum = 0.0;
      for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(
          maximum,
          std::abs(actual[index] - expected[static_cast<Eigen::Index>(index)]));
      }
      return maximum;
    };
  const auto maximum_absolute = [](const std::vector<double> & values) {
      double maximum = 0.0;
      for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
      }
      return maximum;
    };
  const double reference_output_error = maximum_absolute_difference(
    state->reference.positions, state->output.positions);
  const double following_position_error = maximum_absolute_difference(
    state->reference.positions, state->feedback.positions);
  const double reference_velocity = maximum_absolute(state->reference.velocities);
  const double feedback_velocity = maximum_absolute(state->feedback.velocities);

  if (std::abs(state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    ++controller_speed_scaling_failure_count_;
    reason = "Safe re-arm detected trajectory-controller speed scaling";
    return RearmReadiness::kUnsafe;
  }

  std::string feedback_failure;
  if (!feedback_motion_within_limits(*state, feedback_failure)) {
    ++feedback_motion_limit_failure_count_;
    reason = "Safe re-arm measured unsafe motion: " + feedback_failure;
    return RearmReadiness::kUnsafe;
  }
  const ActualFeedbackSafety feedback_safety =
    validate_controller_feedback_collision_state(*state, "Safe re-arm", reason);
  if (feedback_safety == ActualFeedbackSafety::kWaiting) {
    stable_rearm_samples_ = 0;
    return RearmReadiness::kWaiting;
  }
  if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
    return RearmReadiness::kUnsafe;
  }

  if (published_tail_rearm_pending_) {
    const rclcpp::Duration controller_period = rclcpp::Duration::from_nanoseconds(
      seconds_to_nanoseconds(parameters_.trajectory_controller_period_sec));
    const control::PublicationMatch publication_match =
      published_trajectory_history_.match(
      *state, current_time.get_clock_type(), controller_period,
      kControllerCommandAlignmentToleranceRad);
    if (!publication_match.matched()) {
      ++command_alignment_failure_count_;
      const bool unmatched_state_is_settled =
        reference_output_error <= kControllerCommandAlignmentToleranceRad &&
        following_position_error <= parameters_.collision_tracking_error_bound_rad &&
        reference_velocity <= kStoppedVelocityRadps &&
        feedback_velocity <= kStoppedVelocityRadps;
      if (!unmatched_state_is_settled) {
        stable_rearm_samples_ = 0;
        reason =
          "Safe re-arm is waiting for an unmatched controller phase to settle";
        return RearmReadiness::kWaiting;
      }
    } else {
      const bool publication_changed = !active_publication_id_.has_value() ||
        *active_publication_id_ != publication_match.publication_id;
      if (publication_changed) {
        stable_rearm_samples_ = 0;
        if (publication_match.kind == control::PublicationMatchKind::kOlder) {
          ++older_publication_recovery_count_;
        }
        active_publication_id_ = publication_match.publication_id;
      }
      const auto * published_record =
        published_trajectory_history_.find(publication_match.publication_id);
      if (published_record == nullptr || published_record->execution_queue.size() < 3U) {
        reason = "Safe re-arm lost the published braking-tail record";
        return RearmReadiness::kUnsafe;
      }
      if (publication_changed) {
        const rclcpp::Time required_ros_deadline = published_record->stationary_time +
          rclcpp::Duration::from_seconds(kSafeRearmSettleBudgetSec);
        if (!rearm_deadline_.has_value() || required_ros_deadline > *rearm_deadline_) {
          rearm_deadline_ = required_ros_deadline;
        }
        const double remaining_tail_sec = std::max(
          0.0, (published_record->stationary_time - current_time).seconds());
        const auto required_wall_deadline = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(
            remaining_tail_sec + kSafeRearmSettleBudgetSec));
        if (!rearm_wall_deadline_.has_value() ||
          required_wall_deadline > *rearm_wall_deadline_)
        {
          rearm_wall_deadline_ = required_wall_deadline;
        }
      }
      const auto & published_queue = published_record->execution_queue;
      const control::ControllerSampleTimes & corrected_times =
        publication_match.sample_times;
      last_controller_reference_alignment_error_rad_ =
        publication_match.reference_error_rad;
      last_controller_output_alignment_error_rad_ =
        publication_match.output_error_rad;
      max_controller_reference_alignment_error_rad_ = std::max(
        max_controller_reference_alignment_error_rad_,
        publication_match.reference_error_rad);
      max_controller_output_alignment_error_rad_ = std::max(
        max_controller_output_alignment_error_rad_,
        publication_match.output_error_rad);
      command_alignment_check_count_ += 2U;
      const rclcpp::Time transmitted_end =
        std::prev(published_queue.end(), 2)->time_stamp;
      if (corrected_times.reference < published_queue.front().time_stamp ||
        (corrected_times.output > transmitted_end &&
        corrected_times.reference < published_record->stationary_time))
      {
        ++command_alignment_failure_count_;
        reason = "Safe re-arm controller sample left the published braking-tail horizon";
        stable_rearm_samples_ = 0;
        return RearmReadiness::kWaiting;
      }

      if (corrected_times.reference < published_record->stationary_time) {
        stable_rearm_samples_ = 0;
        reason = "Safe re-arm is monitoring the pre-published braking tail";
        return RearmReadiness::kWaiting;
      }

      const double terminal_reference_error = maximum_expected_difference(
        state->reference.positions, published_queue.back().positions);
      const double terminal_output_error = maximum_expected_difference(
        state->output.positions, published_queue.back().positions);
      if (terminal_reference_error > kControllerCommandAlignmentToleranceRad ||
        terminal_output_error > kControllerCommandAlignmentToleranceRad ||
        reference_output_error > kControllerCommandAlignmentToleranceRad ||
        reference_velocity > kStoppedVelocityRadps)
      {
        ++command_alignment_failure_count_;
        const bool controller_is_settled_at_a_safe_state =
          reference_output_error <= kControllerCommandAlignmentToleranceRad &&
          following_position_error <= parameters_.collision_tracking_error_bound_rad &&
          reference_velocity <= kStoppedVelocityRadps &&
          feedback_velocity <= kStoppedVelocityRadps;
        if (!controller_is_settled_at_a_safe_state) {
          stable_rearm_samples_ = 0;
          reason =
            "Safe re-arm is waiting for the trajectory controller to settle";
          return RearmReadiness::kWaiting;
        }
      }
    }
  }

  const bool stopped_and_settled =
    reference_output_error <= kControllerCommandAlignmentToleranceRad &&
    following_position_error <= parameters_.collision_tracking_error_bound_rad &&
    reference_velocity <= kStoppedVelocityRadps &&
    feedback_velocity <= kStoppedVelocityRadps;
  if (!stopped_and_settled) {
    stable_rearm_samples_ = 0;
    reason =
      "Safe re-arm requires a settled hold: command=" +
      std::to_string(reference_output_error) + " rad, following=" +
      std::to_string(following_position_error) + " rad, reference_velocity=" +
      std::to_string(reference_velocity) + " rad/s, feedback_velocity=" +
      std::to_string(feedback_velocity) + " rad/s";
    return RearmReadiness::kWaiting;
  }

  ++stable_rearm_samples_;
  if (stable_rearm_samples_ < kRequiredStableRearmSamples) {
    reason = "Safe re-arm is confirming a settled hold (" +
      std::to_string(stable_rearm_samples_) + "/" +
      std::to_string(kRequiredStableRearmSamples) + ")";
    return RearmReadiness::kWaiting;
  }
  return RearmReadiness::kReady;
}

std::optional<moveit_servo::KinematicState> CollisionAwareServoComponent::
predicted_or_measured_state(
  const rclcpp::Time & current_time, std::string & failure_reason,
  bool & requires_latched_halt)
{
  requires_latched_halt = false;
  const auto state_monitor = planning_scene_monitor_->getStateMonitor();
  const rclcpp::Time oldest_allowed_state = current_time -
    rclcpp::Duration::from_seconds(parameters_.state_feedback_timeout_sec);
  const auto [measured_robot_state, state_time] = state_monitor->getCurrentStateAndTime();
  if (!measured_robot_state || state_time > current_time || state_time < oldest_allowed_state ||
    !state_monitor->haveCompleteState(oldest_allowed_state))
  {
    return std::nullopt;
  }
  moveit_servo::KinematicState measured(parameters_.joint_names.size());
  measured.joint_names = parameters_.joint_names;
  measured_robot_state->copyJointGroupPositions(joint_model_group_, measured.positions);
  measured_robot_state->copyJointGroupVelocities(joint_model_group_, measured.velocities);
  measured.accelerations.setZero();
  if (!finite_state(measured, parameters_.joint_names.size())) {
    return std::nullopt;
  }
  measured.time_stamp = state_time;

  const auto controller_state = current_controller_state();
  if (!controller_state.has_value()) {
    failure_reason = "Fresh finite trajectory controller state is unavailable";
    return std::nullopt;
  }
  observe_controller_feedback(*controller_state, current_time.get_clock_type());
  if (!feedback_motion_within_limits(*controller_state, failure_reason)) {
    ++feedback_motion_limit_failure_count_;
    requires_latched_halt = true;
    return std::nullopt;
  }
  last_controller_speed_scaling_factor_ = controller_state->speed_scaling_factor;
  if (std::abs(controller_state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    ++controller_speed_scaling_failure_count_;
    failure_reason =
      "Trajectory controller speed scaling changed the checked command timeline: factor=" +
      std::to_string(controller_state->speed_scaling_factor);
    requires_latched_halt = true;
    return std::nullopt;
  }

  bool following_window_active = false;
  if (command_epoch_has_published_) {
    const rclcpp::Duration controller_period = rclcpp::Duration::from_nanoseconds(
      seconds_to_nanoseconds(parameters_.trajectory_controller_period_sec));
    const auto corrected_times =
      control::PublishedTrajectoryHistory::correctedSampleTimes(
      *controller_state, current_time.get_clock_type(), controller_period);
    const auto * latest_publication = published_trajectory_history_.latest();
    if (latest_publication == nullptr) {
      failure_reason =
        "Published command epoch has no retained checked trajectory";
      ++command_alignment_failure_count_;
      requires_latched_halt = true;
      return std::nullopt;
    }
    if (!corrected_times.has_value()) {
      failure_reason =
        "Trajectory controller state has no valid checked publication phase";
      ++command_alignment_failure_count_;
      std::string collision_failure;
      const ActualFeedbackSafety feedback_safety =
        validate_controller_feedback_collision_state(
        *controller_state, "Unmatched controller phase", collision_failure);
      if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
        failure_reason += "; " + collision_failure;
        requires_latched_halt = true;
      }
      return std::nullopt;
    }

    const control::PublicationMatch publication_match =
      published_trajectory_history_.match(
      *controller_state, current_time.get_clock_type(), controller_period,
      kControllerCommandAlignmentToleranceRad);
    if (!publication_match.matched()) {
      const auto maximum_expected_difference = [](
        const std::vector<double> & actual, const Eigen::VectorXd & expected) {
          if (actual.size() != static_cast<std::size_t>(expected.size())) {
            return std::numeric_limits<double>::infinity();
          }
          double maximum = 0.0;
          for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum = std::max(
              maximum,
              std::abs(actual[index] - expected[static_cast<Eigen::Index>(index)]));
          }
          return maximum;
        };
      const auto maximum_absolute = [](const std::vector<double> & values) {
          double maximum = 0.0;
          for (const double value : values) {
            maximum = std::max(maximum, std::abs(value));
          }
          return maximum;
        };
      // Several safe future-dated replacements can be published before JTC
      // reaches the first command timestamp. During that bounded phase its
      // old hold is valid only while it still equals the immutable bootstrap
      // seed, not whichever replacement happens to be latest in history.
      const bool preserving_validated_bootstrap_hold =
        bootstrap_ack_pending_ &&
        bootstrap_trajectory_origin_.has_value() &&
        bootstrap_seed_positions_.has_value() &&
        corrected_times->reference < *bootstrap_trajectory_origin_ &&
        maximum_expected_difference(
          controller_state->reference.positions, *bootstrap_seed_positions_) <=
        kControllerCommandAlignmentToleranceRad &&
        maximum_expected_difference(
          controller_state->output.positions, *bootstrap_seed_positions_) <=
        kControllerCommandAlignmentToleranceRad &&
        maximum_expected_difference(
          controller_state->feedback.positions, *bootstrap_seed_positions_) <=
        parameters_.collision_tracking_error_bound_rad &&
        maximum_absolute(controller_state->reference.velocities) <=
        kStoppedVelocityRadps &&
        maximum_absolute(controller_state->feedback.velocities) <=
        kStoppedVelocityRadps;
      if (preserving_validated_bootstrap_hold) {
        following_window_active = false;
      } else {
        failure_reason =
          "Trajectory controller reference/output do not match any retained checked "
          "publication";
        ++command_alignment_failure_count_;
        std::string collision_failure;
        const ActualFeedbackSafety feedback_safety =
          validate_controller_feedback_collision_state(
          *controller_state, "Unmatched controller phase", collision_failure);
        if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
          failure_reason += "; " + collision_failure;
          requires_latched_halt = true;
        }
        return std::nullopt;
      }
    } else {
      last_controller_reference_alignment_error_rad_ =
        publication_match.reference_error_rad;
      last_controller_output_alignment_error_rad_ = publication_match.output_error_rad;
      max_controller_reference_alignment_error_rad_ = std::max(
        max_controller_reference_alignment_error_rad_,
        publication_match.reference_error_rad);
      max_controller_output_alignment_error_rad_ = std::max(
        max_controller_output_alignment_error_rad_, publication_match.output_error_rad);
      command_alignment_check_count_ += 2U;
      following_window_active = true;

      if (publication_match.kind == control::PublicationMatchKind::kOlder) {
        active_publication_id_ = publication_match.publication_id;
        ++older_publication_recovery_count_;
        failure_reason =
          "Trajectory controller is executing an older checked publication";
        return std::nullopt;
      }
      active_publication_id_ = publication_match.publication_id;
      if (bootstrap_ack_pending_) {
        if (bootstrap_trajectory_origin_.has_value() &&
          *bootstrap_trajectory_origin_ == publication_match.sample_times.trajectory_origin)
        {
          ++bootstrap_acknowledgement_count_;
          clear_bootstrap_ack_state();
          clear_rearm_deadlines();
        }
      }
    }
  }

  if (following_window_active) {
    ++following_check_count_;
    const auto maximum_absolute_difference = [](
      const std::vector<double> & reference, const std::vector<double> & feedback) {
        double maximum = 0.0;
        std::size_t maximum_index = 0U;
        for (std::size_t index = 0; index < reference.size(); ++index) {
          const double error = std::abs(reference[index] - feedback[index]);
          if (error > maximum) {
            maximum = error;
            maximum_index = index;
          }
        }
        return std::pair<double, std::size_t>{maximum, maximum_index};
      };
    const auto [position_error, position_error_index] = maximum_absolute_difference(
      controller_state->reference.positions, controller_state->feedback.positions);
    const auto [velocity_error, velocity_error_index] = maximum_absolute_difference(
      controller_state->reference.velocities, controller_state->feedback.velocities);
    last_following_position_error_rad_ = position_error;
    last_following_velocity_error_rad_s_ = velocity_error;
    last_following_position_error_joint_ = parameters_.joint_names[position_error_index];
    last_following_velocity_error_joint_ = parameters_.joint_names[velocity_error_index];
    if (position_error > max_following_position_error_rad_) {
      max_following_position_error_rad_ = position_error;
      max_following_position_error_joint_ = last_following_position_error_joint_;
    }
    if (velocity_error > max_following_velocity_error_rad_s_) {
      max_following_velocity_error_rad_s_ = velocity_error;
      max_following_velocity_error_joint_ = last_following_velocity_error_joint_;
    }
    const control::FollowingErrorPolicy following_policy{
      parameters_.following_position_tolerance_rad,
      parameters_.following_velocity_tolerance_rad_s,
      parameters_.collision_tracking_error_bound_rad,
      kMaximumRecoverableFollowingVelocityErrorRadps,
    };
    const control::FollowingErrorAction following_action =
      control::classifyFollowingError(position_error, velocity_error, following_policy);
    if (following_action != control::FollowingErrorAction::kContinue) {
      ++following_error_count_;
      failure_reason =
        "Trajectory following error: position=" + std::to_string(position_error) +
        " rad (" + last_following_position_error_joint_ + "), velocity=" +
        std::to_string(velocity_error) + " rad/s (" +
        last_following_velocity_error_joint_ + ")";
      // Validate the exact JTC feedback sample that triggered recovery. The
      // independently updated PlanningSceneMonitor state can be one callback
      // behind and is not sufficient for this transition.
      std::string collision_failure;
      const ActualFeedbackSafety feedback_safety =
        validate_controller_feedback_collision_state(
        *controller_state, "Following-error recovery", collision_failure);
      if (feedback_safety != ActualFeedbackSafety::kSafe) {
        failure_reason += "; " + collision_failure;
        requires_latched_halt = true;
        return std::nullopt;
      }

      // Stop extending the command and let its checked braking suffix run.
      // Following accuracy alone never creates a permanent halt: the
      // controlled re-arm path checks physical bounds and collision state on
      // every sample, then waits for a settled hold.
      requires_latched_halt = false;
      return std::nullopt;
    }
  }

  while (!command_queue_.empty() &&
    command_queue_.front().time_stamp <
    current_time - rclcpp::Duration::from_seconds(servo_parameters_.max_expected_latency))
  {
    command_queue_.pop_front();
  }
  if (!command_queue_.empty() && command_queue_.back().time_stamp > current_time) {
    initial_state_received_ = true;
    return command_queue_.back();
  }
  if (command_epoch_active_) {
    failure_reason = "Buffered command timeline was exhausted";
    return std::nullopt;
  }
  command_queue_.clear();
  measured.time_stamp = current_time;
  initial_state_received_ = true;
  return measured;
}

void CollisionAwareServoComponent::observe_controller_feedback(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  const rcl_clock_type_t clock_type)
{
  const rclcpp::Time stamp(state.header.stamp, clock_type);
  if (last_feedback_sample_time_.has_value() && stamp <= *last_feedback_sample_time_) {
    return;
  }
  const Eigen::Map<const Eigen::VectorXd> velocity(
    state.feedback.velocities.data(),
    static_cast<Eigen::Index>(state.feedback.velocities.size()));
  Eigen::Index velocity_index = 0;
  const double maximum_velocity = velocity.cwiseAbs().maxCoeff(&velocity_index);
  if (maximum_velocity > max_feedback_joint_velocity_rad_s_) {
    max_feedback_joint_velocity_rad_s_ = maximum_velocity;
    max_feedback_joint_velocity_name_ =
      parameters_.joint_names[static_cast<std::size_t>(velocity_index)];
  }

  if (last_feedback_sample_time_.has_value() && last_feedback_velocity_.has_value()) {
    const double elapsed_sec = (stamp - *last_feedback_sample_time_).seconds();
    const Eigen::VectorXd acceleration =
      (velocity - *last_feedback_velocity_) / elapsed_sec;
    if (acceleration.allFinite()) {
      Eigen::Index acceleration_index = 0;
      const double maximum_acceleration =
        acceleration.cwiseAbs().maxCoeff(&acceleration_index);
      if (maximum_acceleration > max_feedback_joint_acceleration_rad_s2_) {
        max_feedback_joint_acceleration_rad_s2_ = maximum_acceleration;
        max_feedback_joint_acceleration_name_ =
          parameters_.joint_names[static_cast<std::size_t>(acceleration_index)];
      }
      if (last_feedback_acceleration_.has_value()) {
        const Eigen::VectorXd jerk =
          (acceleration - *last_feedback_acceleration_) / elapsed_sec;
        if (jerk.allFinite()) {
          Eigen::Index jerk_index = 0;
          const double maximum_jerk = jerk.cwiseAbs().maxCoeff(&jerk_index);
          if (maximum_jerk > max_feedback_joint_jerk_rad_s3_) {
            max_feedback_joint_jerk_rad_s3_ = maximum_jerk;
            max_feedback_joint_jerk_name_ =
              parameters_.joint_names[static_cast<std::size_t>(jerk_index)];
          }
          last_feedback_jerk_ = jerk;
        }
      }
      last_feedback_acceleration_ = acceleration;
    }
  }
  last_feedback_sample_time_ = stamp;
  last_feedback_velocity_ = velocity;
}

bool CollisionAwareServoComponent::feedback_motion_within_limits(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  std::string & failure_reason) const
{
  const Eigen::Map<const Eigen::VectorXd> position(
    state.feedback.positions.data(),
    static_cast<Eigen::Index>(state.feedback.positions.size()));
  const Eigen::Map<const Eigen::VectorXd> velocity(
    state.feedback.velocities.data(),
    static_cast<Eigen::Index>(state.feedback.velocities.size()));
  const Eigen::VectorXd physical_position_margin =
    (motion_limits_.position_margin.array() -
    parameters_.collision_tracking_error_bound_rad).max(0.0);
  const Eigen::VectorXd safe_lower =
    motion_limits_.lower_position + physical_position_margin;
  const Eigen::VectorXd safe_upper =
    motion_limits_.upper_position - physical_position_margin;

  if ((position.array() < safe_lower.array() - kFeedbackLimitTolerance).any() ||
    (position.array() > safe_upper.array() + kFeedbackLimitTolerance).any())
  {
    failure_reason = "Measured joint position left the feedback-safe limit corridor";
    return false;
  }
  if (physical_joint_velocity_limits_.size() != velocity.size() ||
    (velocity.cwiseAbs().array() >
    physical_joint_velocity_limits_.array() + kFeedbackLimitTolerance).any())
  {
    failure_reason = "Measured joint velocity exceeds the robot's physical URDF limit";
    return false;
  }
  // The controller reports position and velocity, but not acceleration or
  // jerk. Their finite-difference estimates remain useful telemetry, yet are
  // too sensitive to sample jitter to serve as a hard stop. The published
  // command itself is still constrained by the exact acceleration/jerk
  // limits in every QP step and in the braking-tail generator.
  return true;
}

}  // namespace face_tracking_arm
