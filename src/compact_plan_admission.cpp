// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_plan_admission.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "face_tracking_arm/compact_joint_profile.hpp"
namespace face_tracking_arm::control::detail
{
namespace
{
constexpr double kCompactMetricTolerance{1.0e-6};
constexpr double kMaximumAttachDistanceRad{0.20};
constexpr double kGoalMatchTolerance{1.0e-6};

std::string jsonNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string jsonFiniteNumber(const double value)
{
  return std::isfinite(value) ? jsonNumber(value) : std::string{"null"};
}

bool compactMotionLimitsValid(const JointMotionLimits & limits, const Eigen::Index count)
{
  return count > 0 && limits.lower_position.size() == count &&
         limits.upper_position.size() == count && limits.max_velocity.size() == count &&
         limits.max_acceleration.size() == count && limits.max_jerk.size() == count &&
         limits.lower_position.allFinite() && limits.upper_position.allFinite() &&
         limits.max_velocity.allFinite() && limits.max_acceleration.allFinite() &&
         limits.max_jerk.allFinite() &&
         (limits.lower_position.array() <= limits.upper_position.array()).all() &&
         (limits.max_velocity.array() > 0.0).all() &&
         (limits.max_acceleration.array() > 0.0).all() &&
         (limits.max_jerk.array() > 0.0).all() && limits.numerical_policy &&
         limits.numerical_policy->valid();
}

std::optional<CompactJointProfile> makeCompactReturnProfile(
  const BackgroundPathResult & plan, const std::uint64_t scene_revision,
  const Eigen::VectorXd & folded_goal, const double radius_cap_m,
  const JointMotionLimits & limits, const double control_period_sec,
  const double speed_scale)
{
  if (!plan.success || plan.scene_revision != scene_revision ||
    !plan.compact_return_path || folded_goal.size() == 0 || !folded_goal.allFinite())
  {
    return std::nullopt;
  }
  const auto & return_path = *plan.compact_return_path;
  const auto & metrics = return_path.validation_metrics;
  if (return_path.path.size() < 2U ||
    return_path.path.front().size() != folded_goal.size() ||
    (return_path.path.front() - folded_goal).lpNorm<Eigen::Infinity>() > 1.0e-12 ||
    return_path.goal_positions.size() != folded_goal.size() ||
    !return_path.goal_positions.allFinite() ||
    return_path.path.back().size() != return_path.goal_positions.size() ||
    (return_path.path.back() - return_path.goal_positions).lpNorm<Eigen::Infinity>() > 1.0e-12 ||
    metrics.checked_samples < 2U ||
    !std::isfinite(metrics.maximum_assembly_radius_m) ||
    metrics.maximum_assembly_radius_m < 0.0 ||
    metrics.maximum_assembly_radius_m > radius_cap_m + kCompactMetricTolerance ||
    !std::isfinite(metrics.effective_radius_cap_m) || metrics.effective_radius_cap_m <= 0.0 ||
    metrics.maximum_assembly_radius_m >
    metrics.effective_radius_cap_m + kCompactMetricTolerance)
  {
    return std::nullopt;
  }
  const JointMotionState stationary_start{
    folded_goal, Eigen::VectorXd::Zero(folded_goal.size()),
    Eigen::VectorXd::Zero(folded_goal.size())};
  return buildCompactJointProfile(
    return_path.path, stationary_start, limits, control_period_sec, speed_scale);
}

std::string metricsAndCapsJson(
  const BackgroundPathResult & plan, const double submitted_cap, const double current_cap)
{
  const auto & metrics = *plan.validation_metrics;
  return std::string{"\"actual_metrics\":{"} +
         "\"maximum_assembly_radius_m\":" + jsonFiniteNumber(metrics.maximum_assembly_radius_m) +
         ",\"maximum_abs_screen_pitch_rad\":" +
         jsonFiniteNumber(metrics.maximum_abs_screen_pitch_rad) +
         ",\"maximum_roll_error_rad\":" + jsonFiniteNumber(metrics.maximum_roll_error_rad) + "}," +
         "\"submitted_caps\":{\"radius_m\":" + jsonFiniteNumber(submitted_cap) + "}," +
         "\"current_caps\":{\"radius_m\":" + jsonFiniteNumber(current_cap) + "}," +
         "\"compact_path_sample_count\":" + std::to_string(plan.compact_path_samples.size()) + "," +
         (plan.compact_goal_metrics ?
         std::string{"\"exact_goal_metrics\":{"} +
         "\"maximum_assembly_radius_m\":" +
         jsonFiniteNumber(plan.compact_goal_metrics->maximum_assembly_radius_m) +
         ",\"maximum_abs_screen_pitch_rad\":" +
         jsonFiniteNumber(plan.compact_goal_metrics->maximum_abs_screen_pitch_rad) +
         ",\"maximum_roll_error_rad\":" +
         jsonFiniteNumber(plan.compact_goal_metrics->maximum_roll_error_rad) + "}," :
         "\"exact_goal_metrics\":null,") +
         (plan.compact_path_samples.empty() ? "\"first_sample_suffix_metrics\":null" :
         std::string{"\"first_sample_suffix_metrics\":{"} +
         "\"maximum_assembly_radius_m\":" + jsonFiniteNumber(
      plan.compact_path_samples.front().suffix_maximum_assembly_radius_m) +
         ",\"maximum_abs_screen_pitch_rad\":" + jsonFiniteNumber(
      plan.compact_path_samples.front().suffix_maximum_abs_screen_pitch_rad) +
         ",\"maximum_roll_error_rad\":" + jsonFiniteNumber(
      plan.compact_path_samples.front().suffix_maximum_roll_error_rad) + "}");
}

CompactPlanAdmissionResult reject(std::string reason, std::string details = {})
{
  CompactPlanAdmissionResult result;
  result.disposition = CompactAdmissionDisposition::kReject;
  result.reason = std::move(reason);
  result.additional_details = std::move(details);
  return result;
}

}  // namespace

CompactPlanAdmissionResult admitCompactPreparationPlan(
  const CompactPreparationAdmissionInput & input,
  const CompactPlanAdmissionEnvironment & environment)
{
  const auto & plan = input.plan;
  const auto & candidate = input.candidate;
  const auto * actual = environment.actual_feedback;
  const auto * queue = environment.command_queue_state;
  const auto & state = environment.state;
  const auto & group = environment.group;

  const bool matches_request = plan.profile_id == candidate.profile_id &&
    plan.goal_positions.size() == candidate.bounded_positions.size() &&
    plan.goal_positions.allFinite() &&
    (plan.goal_positions - candidate.bounded_positions).lpNorm<Eigen::Infinity>() <=
    kGoalMatchTolerance &&
    plan.start_positions.size() == input.submitted_start_positions.size() &&
    plan.start_positions.allFinite() &&
    (plan.start_positions - input.submitted_start_positions).lpNorm<Eigen::Infinity>() <=
    kGoalMatchTolerance;
  if (!plan.success || !matches_request || !plan.validation_metrics) {
    return reject("compact candidate plan failed or did not match its request: " + plan.message);
  }
  if (!actual || !queue || !input.fresh_actual_feedback ||
    actual->position.size() != plan.start_positions.size() ||
    actual->velocity.size() != plan.start_positions.size() ||
    actual->acceleration.size() != plan.start_positions.size() ||
    queue->position.size() != plan.start_positions.size() ||
    !actual->position.allFinite() || !actual->velocity.allFinite() ||
    !actual->acceleration.allFinite() || !queue->position.allFinite() ||
    queue->velocity.size() != plan.start_positions.size() ||
    queue->acceleration.size() != plan.start_positions.size() ||
    !queue->velocity.allFinite() || !queue->acceleration.allFinite())
  {
    CompactPlanAdmissionResult result;
    result.disposition = CompactAdmissionDisposition::kWait;
    result.reason = "compact candidate awaits fresh actual and queue state";
    return result;
  }
  if (plan.path.size() < 2U ||
    std::any_of(plan.path.begin(), plan.path.end(), [&plan](const Eigen::VectorXd & point) {
      return point.size() != plan.start_positions.size() || !point.allFinite();
    }) ||
    (plan.path.front() - plan.start_positions).lpNorm<Eigen::Infinity>() >
    kGoalMatchTolerance ||
    (plan.path.back() - candidate.bounded_positions).lpNorm<Eigen::Infinity>() >
    input.planner_goal_tolerance_rad + 1.0e-6)
  {
    return reject("compact candidate returned an incomplete path");
  }
  const auto & return_path = plan.compact_return_path;
  const bool return_matches = input.return_region && return_path &&
    return_path->path.size() >= 2U && return_path->goal_positions.size() ==
    candidate.bounded_positions.size() && return_path->goal_positions.allFinite() &&
    (return_path->path.front() - candidate.bounded_positions).
    lpNorm<Eigen::Infinity>() <= kGoalMatchTolerance &&
    return_path->path.back().size() == return_path->goal_positions.size() &&
    (return_path->path.back() - return_path->goal_positions).
    lpNorm<Eigen::Infinity>() <= kGoalMatchTolerance &&
    std::all_of(return_path->path.begin(), return_path->path.end(), [&plan](
        const Eigen::VectorXd & point) {
        return point.size() == plan.start_positions.size() && point.allFinite();
        }) &&
    environment.validation.validateCompactReturnRegion(
      state, return_path->goal_positions, *input.return_region);
  if (!return_matches) {
    return reject("compact candidate lacks a checked endpoint in its frozen return region");
  }

  double entry_radius_cap = input.minimum_entry_radius_cap_m;
  double entry_max_pitch = 0.0;
  double entry_max_roll = 0.0;
  Eigen::VectorXd state_position;
  state.copyJointGroupPositions(&group, state_position);
  bool entry_valid = environment.validation.includeCompactEntryEnvelopePosition(
    state, state_position, &entry_radius_cap,
    &entry_max_pitch, &entry_max_roll);
  entry_valid = environment.validation.includeCompactEntryEnvelopePosition(
    state, actual->position, &entry_radius_cap, &entry_max_pitch, &entry_max_roll) && entry_valid;
  entry_valid = environment.validation.includeCompactEntryEnvelopePosition(
    state, queue->position, &entry_radius_cap, &entry_max_pitch, &entry_max_roll) && entry_valid;
  entry_valid = environment.validation.includeCompactEntryEnvelope(
    state, environment.time_sec, environment.active_publication,
    environment.latest_publication, &entry_radius_cap,
    &entry_max_pitch, &entry_max_roll) && entry_valid;
  if (!entry_valid) {
    return reject("compact candidate entry envelope is invalid");
  }
  const auto joint_count = static_cast<Eigen::Index>(group.getVariableCount());
  if (!compactMotionLimitsValid(environment.compact_limits, joint_count) ||
    !compactMotionLimitsValid(environment.prefix_limits, joint_count))
  {
    return reject("compact candidate could not be timed: compact motion limits are invalid");
  }
  const std::string metrics_details = metricsAndCapsJson(
    plan, input.submitted_entry_radius_cap_m, entry_radius_cap);

  std::string prefix_failure;
  const auto prefix = findCompactPublishedPrefix(
    *queue, environment.time_sec, environment.latest_publication,
    environment.active_publication, environment.control_period_sec,
    environment.prefix_limits, &prefix_failure);
  if (!prefix) {
    auto result = reject("compact candidate has no validated stationary prefix: " +
      prefix_failure);
    result.prefix_failure = prefix_failure;
    return result;
  }
  const JointMotionState compact_start = prefix->samples.back();
  const auto attachment = selectCompactAttachment(
    plan, actual->position, compact_start.position, entry_radius_cap);
  if (!attachment.valid) {
    const std::string reason = plan.compact_path_samples.empty() ?
      "compact candidate path metrics exceed current entry caps" :
      "compact candidate has no valid certified suffix attachment";
    return reject(reason + "; " + attachment.failure_reason + "; " + metrics_details,
      metrics_details);
  }
  const double actual_to_attachment = (actual->position - attachment.position).norm();
  const double queue_to_attachment = (compact_start.position - attachment.position).norm();
  if (!std::isfinite(actual_to_attachment) || !std::isfinite(queue_to_attachment) ||
    queue_to_attachment > kMaximumAttachDistanceRad + 1.0e-9)
  {
    return reject("compact candidate attachment moved outside the checked window");
  }

  CompactBridgeValidation bridge;
  bridge.radius_cap = entry_radius_cap;
  const auto validate_bridge = [&](
    const Eigen::VectorXd & from, const Eigen::VectorXd & to, std::size_t sample_index) {
      return environment.validation.validateCompactBridgeSegment(
        state, from, to, sample_index, bridge);
    };
  CompactAdmittedPlan admitted;
  admitted.entry_radius_cap_m = entry_radius_cap;
  admitted.entry_max_pitch_rad = entry_max_pitch;
  admitted.entry_max_roll_rad = entry_max_roll;
  admitted.actual_to_attachment_rad = actual_to_attachment;
  admitted.queue_to_attachment_rad = queue_to_attachment;
  admitted.published_prefix = *prefix;
  admitted.compact_start = compact_start;
  admitted.attachment = attachment;
  admitted.path.reserve(plan.compact_path_samples.size() + 2U);
  std::vector<double> radii;
  radii.reserve(plan.compact_path_samples.size() + 2U);
  admitted.path.push_back(compact_start.position);
  moveit::core::RobotState queue_geometry_state(state);
  queue_geometry_state.setJointGroupPositions(&group, compact_start.position);
  queue_geometry_state.update(true);
  radii.push_back(environment.geometry ?
    environment.geometry->maximumAssemblyRadius(queue_geometry_state) :
    std::numeric_limits<double>::quiet_NaN());
  if ((attachment.position - admitted.path.back()).norm() > 1.0e-12) {
    admitted.path.push_back(attachment.position);
    radii.push_back(attachment.maximum_radius_m);
  }
  if (attachment.used_certificate) {
    for (std::size_t index = attachment.sample_index + 1U;
      index < plan.compact_path_samples.size(); ++index)
    {
      const auto & sample = plan.compact_path_samples[index];
      if ((sample.positions - admitted.path.back()).norm() > 1.0e-12) {
        admitted.path.push_back(sample.positions);
        radii.push_back(sample.maximum_assembly_radius_m);
      }
    }
  } else {
    for (std::size_t index = 1U; index < plan.path.size(); ++index) {
      if ((plan.path[index] - admitted.path.back()).norm() > 1.0e-12) {
        admitted.path.push_back(plan.path[index]);
        moveit::core::RobotState path_state(state);
        path_state.setJointGroupPositions(&group, plan.path[index]);
        path_state.update(true);
        radii.push_back(environment.geometry ?
          environment.geometry->maximumAssemblyRadius(path_state) :
          std::numeric_limits<double>::quiet_NaN());
      }
    }
  }
  if (radii.size() != admitted.path.size()) {
    return reject("compact candidate phase boundary has incomplete geometry metrics");
  }
  const auto scales = assignCompactPhaseScales(
    admitted.path, radii, kCompactRadiusFloorM + 1.0e-6,
    environment.compact_fold_speed_scale, environment.compact_turn_speed_scale);
  if (!scales) {
    return reject("compact candidate phase scale could not be assigned");
  }
  admitted.phase_scales = *scales;
  CompactPlanAdmissionResult result;
  result.metadata.phase_scales_computed = true;
  result.metadata.profile_tempo_limited = std::any_of(
    scales->begin(), scales->end(), [&environment](const double scale) {
      return environment.compact_speed_scale * scale > 1.0;
    });

  if (prefix->samples.size() > 1U) {
    bool bridge_valid = true;
    for (std::size_t index = 1U; bridge_valid && index < prefix->samples.size(); ++index) {
      bridge_valid = validate_bridge(
        prefix->samples[index - 1U].position, prefix->samples[index].position, index);
    }
    moveit::core::RobotState bridge_start(state);
    bridge_start.setJointGroupPositions(&group, prefix->samples.front().position);
    bridge_start.update(true);
    const auto collision = environment.collision_validator.validatePath(
      environment.scene, bridge_start, group, prefix->samples,
      environment.collision_validator.config().segment_validation_substeps);
    bridge_valid = bridge_valid && collision.input_valid && !collision.unsafe;
    if (!bridge_valid) {
      result.disposition = CompactAdmissionDisposition::kReject;
      if (!bridge.failure_reason.empty()) {
        result.reason = "compact candidate published prefix geometry failed: " +
          bridge.failure_reason;
        result.additional_details = bridge.failure_details;
      } else {
        result.reason = "compact candidate published prefix collision validation failed: " +
          collision.failure_reason + "; sample=" +
          std::to_string(collision.first_unsafe_sample) + ", fraction=" +
          jsonFiniteNumber(collision.first_unsafe_fraction) + ", closest_pair=" +
          collision.closest_pair + ", minimum_distance_m=" +
          jsonFiniteNumber(collision.minimum_distance_m) + ", budget_exhausted=" +
          (collision.budget_exhausted ? "true" : "false");
        result.additional_details = "\"bridge_collision_input_valid\":" +
          std::string(collision.input_valid ? "true" : "false") +
          ",\"bridge_collision_unsafe\":" + (collision.unsafe ? "true" : "false") +
          ",\"bridge_collision_robust_clearance_violated\":" +
          (collision.robust_clearance_violated ? "true" : "false") +
          ",\"bridge_collision_physical_clearance_violated\":" +
          (collision.physical_clearance_violated ? "true" : "false") +
          ",\"bridge_collision_first_unsafe_sample\":" +
          std::to_string(collision.first_unsafe_sample) +
          ",\"bridge_collision_first_unsafe_fraction\":" +
          jsonFiniteNumber(collision.first_unsafe_fraction) +
          ",\"bridge_collision_minimum_distance_m\":" +
          jsonFiniteNumber(collision.minimum_distance_m);
      }
      return result;
    }
  }
  const JointMotionState stationary_attachment{
    attachment.position, Eigen::VectorXd::Zero(attachment.position.size()),
    Eigen::VectorXd::Zero(attachment.position.size())};
  moveit::core::RobotState connector_state(state);
  connector_state.setJointGroupPositions(&group, compact_start.position);
  connector_state.update(true);
  const auto connector = environment.collision_validator.validateSegment(
    environment.scene, connector_state, group, compact_start, stationary_attachment,
    environment.collision_validator.config().segment_validation_substeps);
  const bool attachment_bridge_valid = validate_bridge(
    compact_start.position, attachment.position, 0U);
  if (!connector.input_valid || connector.unsafe || !attachment_bridge_valid) {
    result.disposition = CompactAdmissionDisposition::kReject;
    if (!bridge.failure_reason.empty()) {
      result.reason = "compact candidate attachment bridge failed: " + bridge.failure_reason;
      result.additional_details = bridge.failure_details;
    } else {
      result.reason = "compact candidate attachment collision validation failed: " +
        connector.failure_reason + "; first_sample=" +
        std::to_string(connector.first_unsafe_sample) + ", fraction=" +
        jsonFiniteNumber(connector.first_unsafe_fraction) + ", closest_pair=" +
        connector.closest_pair + ", minimum_distance_m=" +
        jsonFiniteNumber(connector.minimum_distance_m);
    }
    return result;
  }
  auto return_profile = makeCompactReturnProfile(
    plan, input.scene_revision, candidate.bounded_positions, entry_radius_cap,
    environment.compact_limits, environment.control_period_sec,
    environment.compact_speed_scale * environment.compact_return_speed_scale);
  if (!return_profile) {
    result.disposition = CompactAdmissionDisposition::kReject;
    result.reason = "compact candidate return could not be timed and certified";
    return result;
  }
  admitted.bridge_validation = std::move(bridge);
  admitted.return_profile = std::move(*return_profile);
  result.disposition = CompactAdmissionDisposition::kReady;
  result.admitted = std::move(admitted);
  return result;
}

CompactPlanAdmissionResult admitCommittedCompactPlan(
  const CompactCommittedAdmissionInput & input,
  const CompactPlanAdmissionEnvironment & environment)
{
  const auto & plan = input.plan;
  const auto & goal = input.goal;
  const auto & state = environment.state;
  const auto * queue = environment.command_queue_state;
  const bool radius_valid = plan.validation_metrics &&
    std::isfinite(plan.validation_metrics->maximum_assembly_radius_m) &&
    plan.validation_metrics->maximum_assembly_radius_m <= input.effective_radius_cap_m + 1.0e-6;
  const auto & return_path = plan.compact_return_path;
  const bool return_valid = goal.return_region && return_path &&
    return_path->path.size() >= 2U && return_path->goal_positions.size() ==
    goal.bounded_positions.size() && return_path->goal_positions.allFinite() &&
    (return_path->path.front() - goal.bounded_positions).
    lpNorm<Eigen::Infinity>() <= 1.0e-6 &&
    return_path->path.back().size() == return_path->goal_positions.size() &&
    (return_path->path.back() - return_path->goal_positions).
    lpNorm<Eigen::Infinity>() <= 1.0e-6 &&
    environment.validation.validateCompactReturnRegion(
      state, return_path->goal_positions, *goal.return_region);
  bool accepted = plan.success && plan.scene_revision == input.scene_revision &&
    plan.profile_id == goal.profile_id && radius_valid && return_valid && queue &&
    queue->position.size() == plan.start_positions.size() &&
    queue->velocity.size() == plan.start_positions.size() &&
    queue->acceleration.size() == plan.start_positions.size() && plan.path.size() >= 2U &&
    (plan.path.front() - queue->position).lpNorm<Eigen::Infinity>() <= 1.0e-6;
  CompactPlanAdmissionResult result;
  if (!accepted) {
    return result;
  }
  auto prefix = findCompactPublishedPrefix(
    *queue, environment.time_sec, environment.latest_publication,
    environment.active_publication, environment.control_period_sec,
    environment.prefix_limits, &result.prefix_failure);
  if (!prefix) {
    return result;
  }
  CompactAdmittedPlan admitted;
  admitted.entry_radius_cap_m = input.effective_radius_cap_m;
  admitted.published_prefix = *prefix;
  admitted.compact_start = prefix->samples.back();
  Eigen::VectorXd actual_position;
  state.copyJointGroupPositions(&environment.group, actual_position);
  const auto attachment = selectCompactAttachment(
    plan, actual_position, admitted.compact_start.position, input.effective_radius_cap_m);
  accepted = attachment.valid;
  if (accepted) {
    const double actual_distance = (actual_position - attachment.position).norm();
    accepted = std::isfinite(actual_distance) &&
      (admitted.compact_start.position - attachment.position).norm() <=
      kMaximumAttachDistanceRad + 1.0e-9;
  }
  std::vector<double> radii;
  if (accepted) {
    admitted.path.push_back(admitted.compact_start.position);
    moveit::core::RobotState start_state(state);
    start_state.setJointGroupPositions(&environment.group, admitted.compact_start.position);
    start_state.update(true);
    radii.push_back(environment.geometry ?
      environment.geometry->maximumAssemblyRadius(start_state) :
      std::numeric_limits<double>::quiet_NaN());
    if ((attachment.position - admitted.path.back()).norm() > 1.0e-12) {
      admitted.path.push_back(attachment.position);
      radii.push_back(attachment.maximum_radius_m);
    }
    if (attachment.used_certificate) {
      for (std::size_t index = attachment.sample_index + 1U;
        index < plan.compact_path_samples.size(); ++index)
      {
        const auto & sample = plan.compact_path_samples[index];
        if ((sample.positions - admitted.path.back()).norm() > 1.0e-12) {
          admitted.path.push_back(sample.positions);
          radii.push_back(sample.maximum_assembly_radius_m);
        }
      }
    } else {
      for (std::size_t index = 1U; index < plan.path.size(); ++index) {
        if ((plan.path[index] - admitted.path.back()).norm() > 1.0e-12) {
          admitted.path.push_back(plan.path[index]);
          moveit::core::RobotState point_state(state);
          point_state.setJointGroupPositions(&environment.group, plan.path[index]);
          point_state.update(true);
          radii.push_back(environment.geometry ?
            environment.geometry->maximumAssemblyRadius(point_state) :
            std::numeric_limits<double>::quiet_NaN());
        }
      }
    }
    accepted = !admitted.path.empty() && admitted.path.size() == radii.size();
    if (accepted && (attachment.position - admitted.compact_start.position).norm() > 1.0e-12) {
      const std::size_t steps = static_cast<std::size_t>(std::max(1.0, std::ceil(
          (attachment.position - admitted.compact_start.position).
          lpNorm<Eigen::Infinity>() / 0.02)));
      for (std::size_t step = 0U; accepted && step <= steps; ++step) {
        const double fraction = static_cast<double>(step) / static_cast<double>(steps);
        std::string violation;
        accepted = !environment.compact_active ||
          environment.validation.validateCompactPathPosition(
          state, admitted.compact_start.position + fraction *
          (attachment.position - admitted.compact_start.position),
          input.effective_radius_cap_m, &violation);
      }
      moveit::core::RobotState connector_state(state);
      connector_state.setJointGroupPositions(&environment.group, admitted.compact_start.position);
      connector_state.update(true);
      const JointMotionState stopped_start{
        admitted.compact_start.position,
        Eigen::VectorXd::Zero(admitted.compact_start.position.size()),
        Eigen::VectorXd::Zero(admitted.compact_start.position.size())};
      const JointMotionState stopped_attachment{
        attachment.position, Eigen::VectorXd::Zero(attachment.position.size()),
        Eigen::VectorXd::Zero(attachment.position.size())};
      const auto collision = environment.collision_validator.validateSegment(
        environment.scene, connector_state, environment.group, stopped_start,
        stopped_attachment, environment.collision_validator.config().segment_validation_substeps);
      accepted = accepted && collision.input_valid && !collision.unsafe;
    }
  }
  if (!accepted) {
    return result;
  }

  const auto scales = assignCompactPhaseScales(
    admitted.path, radii, kCompactRadiusFloorM + 1.0e-6,
    environment.compact_fold_speed_scale, environment.compact_turn_speed_scale);
  if (!scales) {
    return result;
  }
  admitted.phase_scales = *scales;
  if (prefix->samples.size() > 1U) {
    const auto & bridge_samples = prefix->samples;
    for (const auto & sample : bridge_samples) {
      std::string violation;
      accepted = !environment.compact_active ||
        environment.validation.validateCompactPathPosition(
        state, sample.position, input.effective_radius_cap_m, &violation);
      if (!accepted) {
        break;
      }
    }
    moveit::core::RobotState bridge_start_state(state);
    bridge_start_state.setJointGroupPositions(&environment.group, bridge_samples.front().position);
    bridge_start_state.update(true);
    const auto collision = environment.collision_validator.validatePath(
      environment.scene, bridge_start_state, environment.group, bridge_samples,
      environment.collision_validator.config().segment_validation_substeps);
    accepted = accepted && collision.input_valid && !collision.unsafe;
  }
  if (!accepted) {
    return result;
  }
  auto return_profile = makeCompactReturnProfile(
    plan, input.scene_revision, goal.bounded_positions, input.return_radius_cap_m,
    environment.compact_limits, environment.control_period_sec,
    environment.compact_speed_scale * environment.compact_return_speed_scale);
  if (!return_profile) {
    return result;
  }
  admitted.return_profile = std::move(*return_profile);
  result.disposition = CompactAdmissionDisposition::kReady;
  result.metadata.phase_scales_computed = true;
  result.metadata.profile_tempo_limited = std::any_of(
    scales->begin(), scales->end(), [&environment](const double scale) {
      return environment.compact_speed_scale * scale > 1.0;
    });
  result.admitted = std::move(admitted);
  return result;
}

}  // namespace face_tracking_arm::control::detail
