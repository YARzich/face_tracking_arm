// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/motion_reference.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kRestArrivalRad{0.025};
constexpr double kRestReleaseRad{0.060};
constexpr double kSettledVelocityRadps{0.01};
constexpr double kSettledAccelerationRadps2{0.03};
constexpr double kCompactGoalToleranceRad{0.025};
constexpr double kCompactReserveRad{0.35};
constexpr double kCompactRadiusFloorM{0.40};
constexpr double kCompactMinimumPitchCapRad{0.5235987755982988};
constexpr double kCompactRollLimitRad{0.0890117918517108};
constexpr double kCompactMetricTolerance{1.0e-6};
constexpr double kFaceReturnDwellSec{0.30};
constexpr double kPairYawProbePositionRelaxationRad{0.17453292519943295};  // 10 deg
constexpr double kPairYawProbePoseHoldToleranceRad{0.017453292519943295};  // 1 deg
// Compact constraints act at the predicted queue tail. Its published remainder
// through stop is included in the immutable entry envelope, so it must not be
// charged a second command-latency interval. Collision constraints still use
// the configured residual latency.
constexpr double kCompactQueueTailResidualLatencySec{0.0};

std::string jsonNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string jsonVector(const Eigen::VectorXd & value)
{
  std::string result{"["};
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (index > 0) {
      result += ',';
    }
    result += jsonNumber(value[index]);
  }
  result += ']';
  return result;
}

std::string pairYawAssessmentJson(const PairYawResourceAssessment & assessment)
{
  const auto joint_json = [](const PairYawJointResourceAssessment & joint) {
      return std::string{"{\"yaw_error_derivative_per_rad\":"} +
             jsonNumber(joint.yaw_error_derivative_per_rad) +
             ",\"required_direction\":" +
             std::to_string(static_cast<int>(joint.required_direction)) +
             ",\"q_actual_rad\":" + jsonNumber(joint.actual_position_rad) +
             ",\"q_hold_rad\":" + jsonNumber(joint.predicted_hold_position_rad) +
             ",\"lower_rad\":" + jsonNumber(joint.lower_position_rad) +
             ",\"upper_rad\":" + jsonNumber(joint.upper_position_rad) +
             ",\"position_margin_rad\":" + jsonNumber(joint.position_margin_rad) +
             ",\"remaining_joint_travel_rad\":" +
             jsonNumber(joint.remaining_joint_travel_rad) +
             ",\"remaining_yaw_rad\":" + jsonNumber(joint.remaining_yaw_rad) +
             ",\"held_for_required_direction\":" +
             (joint.held_for_required_direction ? "true" : "false") +
             ",\"directionally_bound\":" +
             (joint.directionally_bound ? "true}" : "false}");
    };
  return std::string{"{\"valid\":"} + (assessment.valid ? "true" : "false") +
         ",\"demand_present\":" + (assessment.demand_present ? "true" : "false") +
         ",\"pair_bound_candidate\":" +
         (assessment.pair_bound_candidate ? "true" : "false") +
         ",\"probe_valid\":" + (assessment.probe_valid ? "true" : "false") +
         ",\"actual_probe_solved\":" +
         (assessment.actual_probe_solved ? "true" : "false") +
         ",\"actual_probe_infeasible\":" +
         (assessment.actual_probe_infeasible ? "true" : "false") +
         ",\"relaxed_probe_solved\":" +
         (assessment.relaxed_probe_solved ? "true" : "false") +
         ",\"pair_bound_cause\":" +
         (assessment.pair_bound_cause ? "true" : "false") +
         ",\"exhausted\":" + (assessment.exhausted ? "true" : "false") +
         ",\"signed_yaw_error_rad\":" + jsonNumber(assessment.signed_yaw_error_rad) +
         ",\"target_bearing_rate_radps\":" +
         jsonNumber(assessment.target_bearing_rate_radps) +
         ",\"yaw_demand_radps\":" + jsonNumber(assessment.yaw_demand_radps) +
         ",\"probe_horizon_sec\":" + jsonNumber(assessment.probe_horizon_sec) +
         ",\"actual_available_yaw_rad\":" +
         jsonNumber(assessment.actual_available_yaw_rad) +
         ",\"pair_relaxed_available_yaw_rad\":" +
         jsonNumber(assessment.pair_relaxed_available_yaw_rad) +
         ",\"actual_roll_residual_rad\":" +
         jsonNumber(assessment.actual_roll_residual_rad) +
         ",\"relaxed_roll_residual_rad\":" +
         jsonNumber(assessment.relaxed_roll_residual_rad) +
         ",\"actual_elevation_residual_rad\":" +
         jsonNumber(assessment.actual_elevation_residual_rad) +
         ",\"relaxed_elevation_residual_rad\":" +
         jsonNumber(assessment.relaxed_elevation_residual_rad) +
         ",\"j1\":" + joint_json(assessment.j1) +
         ",\"j4\":" + joint_json(assessment.j4) + "}";
}

struct CompactAttachmentSelection
{
  bool valid{false};
  bool used_certificate{false};
  std::size_t sample_index{0U};
  std::size_t segment_index{0U};
  double segment_fraction{std::numeric_limits<double>::quiet_NaN()};
  Eigen::VectorXd position;
  double maximum_radius_m{std::numeric_limits<double>::quiet_NaN()};
  double maximum_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_abs_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
};

/// Select an entry using only worker samples and suffix maxima from the same
/// immutable coarse path. Certificate gaps fail closed before suffix maxima are trusted.
CompactAttachmentSelection selectCompactAttachment(
  const BackgroundPathResult & plan, const Eigen::VectorXd & actual,
  const Eigen::VectorXd & queue, const double radius_cap_m,
  const double pitch_cap_rad, const double roll_cap_rad)
{
  constexpr double kMaximumAttachDistanceRad{0.20};
  constexpr std::size_t kMaximumCompactCertificateSamples{8192U};
  CompactAttachmentSelection result;
  if (plan.path.size() < 2U) {
    return result;
  }
  const auto within_caps = [radius_cap_m, pitch_cap_rad, roll_cap_rad](
    const double radius, const double roll, const double pitch) {
      return std::isfinite(radius) && radius >= 0.0 &&
             radius <= radius_cap_m + kCompactMetricTolerance &&
             std::isfinite(roll) && roll >= 0.0 &&
             roll <= roll_cap_rad + kCompactPitchValidationToleranceRad &&
             std::isfinite(pitch) && pitch >= 0.0 &&
             pitch <= pitch_cap_rad + kCompactPitchValidationToleranceRad;
    };
  if (plan.compact_path_samples.empty()) {
    if (!plan.validation_metrics || !within_caps(
        plan.validation_metrics->maximum_assembly_radius_m,
        plan.validation_metrics->maximum_roll_error_rad,
        plan.validation_metrics->maximum_abs_screen_pitch_rad))
    {
      return result;
    }
    result.valid = true;
    result.position = plan.start_positions;
    result.maximum_radius_m = plan.validation_metrics->maximum_assembly_radius_m;
    result.maximum_roll_error_rad = plan.validation_metrics->maximum_roll_error_rad;
    result.maximum_abs_pitch_rad = plan.validation_metrics->maximum_abs_screen_pitch_rad;
    return result;
  }

  result.used_certificate = true;
  if (plan.compact_path_samples.size() > kMaximumCompactCertificateSamples ||
    !plan.compact_goal_metrics)
  {
    return result;
  }
  const auto finite_metrics = [](const double radius, const double roll, const double pitch) {
      return std::isfinite(radius) && std::isfinite(roll) && std::isfinite(pitch) &&
             radius >= 0.0 && roll >= 0.0 && pitch >= 0.0;
    };
  const auto & goal_metrics = *plan.compact_goal_metrics;
  if (!finite_metrics(goal_metrics.maximum_assembly_radius_m,
      goal_metrics.maximum_roll_error_rad, goal_metrics.maximum_abs_screen_pitch_rad) ||
    !within_caps(goal_metrics.maximum_assembly_radius_m, goal_metrics.maximum_roll_error_rad,
    goal_metrics.maximum_abs_screen_pitch_rad))
  {
    return result;
  }

  bool certificate_valid = true;
  double suffix_radius = goal_metrics.maximum_assembly_radius_m;
  double suffix_roll = goal_metrics.maximum_roll_error_rad;
  double suffix_pitch = goal_metrics.maximum_abs_screen_pitch_rad;
  for (std::size_t reverse_index = plan.compact_path_samples.size();
    certificate_valid && reverse_index > 0U; --reverse_index)
  {
    const auto & sample = plan.compact_path_samples[reverse_index - 1U];
    certificate_valid = sample.original_segment_index < plan.path.size() - 1U &&
      sample.positions.size() == plan.start_positions.size() && sample.positions.allFinite() &&
      std::isfinite(sample.fraction) && sample.fraction >= 0.0 && sample.fraction <= 1.0 &&
      finite_metrics(sample.maximum_assembly_radius_m, sample.maximum_roll_error_rad,
      sample.maximum_abs_screen_pitch_rad) &&
      finite_metrics(sample.suffix_maximum_assembly_radius_m,
      sample.suffix_maximum_roll_error_rad, sample.suffix_maximum_abs_screen_pitch_rad);
    if (!certificate_valid) {
      break;
    }
    const Eigen::VectorXd expected_position =
      plan.path[sample.original_segment_index] + sample.fraction *
      (plan.path[sample.original_segment_index + 1U] -
      plan.path[sample.original_segment_index]);
    certificate_valid = sample.positions.isApprox(expected_position, 1.0e-9);
    if (reverse_index < plan.compact_path_samples.size()) {
      const auto & following = plan.compact_path_samples[reverse_index];
      const bool same_segment = following.original_segment_index ==
        sample.original_segment_index;
      const bool next_segment = following.original_segment_index ==
        sample.original_segment_index + 1U;
      const double sample_gap =
        (following.positions - sample.positions).lpNorm<Eigen::Infinity>();
      certificate_valid = certificate_valid &&
        (same_segment || (next_segment && std::abs(sample.fraction - 1.0) <= 1.0e-9)) &&
        (!same_segment || following.fraction >= sample.fraction) &&
        std::isfinite(sample_gap) && sample_gap <= 0.0100001;
    }
    suffix_radius = std::max(suffix_radius, sample.maximum_assembly_radius_m);
    suffix_roll = std::max(suffix_roll, sample.maximum_roll_error_rad);
    suffix_pitch = std::max(suffix_pitch, sample.maximum_abs_screen_pitch_rad);
    certificate_valid = certificate_valid &&
      std::abs(sample.suffix_maximum_assembly_radius_m - suffix_radius) <= 1.0e-9 &&
      std::abs(sample.suffix_maximum_roll_error_rad - suffix_roll) <= 1.0e-9 &&
      std::abs(sample.suffix_maximum_abs_screen_pitch_rad - suffix_pitch) <= 1.0e-9;
  }
  if (!certificate_valid) {
    return result;
  }
  const auto & first = plan.compact_path_samples.front();
  const auto & last = plan.compact_path_samples.back();
  if (first.original_segment_index != 0U || std::abs(first.fraction) > 1.0e-12 ||
    !first.positions.isApprox(plan.path.front(), 1.0e-12) ||
    last.original_segment_index != plan.path.size() - 2U ||
    std::abs(last.fraction - 1.0) > 1.0e-12 ||
    !last.positions.isApprox(plan.path.back(), 1.0e-12))
  {
    return result;
  }

  const std::size_t first_segment_limit = std::min<std::size_t>(4U, plan.path.size() - 1U);
  double selected_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0U; index < plan.compact_path_samples.size(); ++index) {
    const auto & sample = plan.compact_path_samples[index];
    const double actual_distance = (actual - sample.positions).norm();
    const double queue_distance = (queue - sample.positions).norm();
    if (sample.original_segment_index >= first_segment_limit ||
      !std::isfinite(actual_distance) || !std::isfinite(queue_distance) ||
      actual_distance > kMaximumAttachDistanceRad + 1.0e-9 ||
      queue_distance > kMaximumAttachDistanceRad + 1.0e-9 ||
      !within_caps(sample.suffix_maximum_assembly_radius_m,
      sample.suffix_maximum_roll_error_rad,
      sample.suffix_maximum_abs_screen_pitch_rad))
    {
      continue;
    }
    if (!result.valid || actual_distance < selected_distance - 1.0e-12) {
      result.valid = true;
      result.sample_index = index;
      result.segment_index = sample.original_segment_index;
      result.segment_fraction = sample.fraction;
      result.position = sample.positions;
      result.maximum_radius_m = sample.suffix_maximum_assembly_radius_m;
      result.maximum_roll_error_rad = sample.suffix_maximum_roll_error_rad;
      result.maximum_abs_pitch_rad = sample.suffix_maximum_abs_screen_pitch_rad;
      selected_distance = actual_distance;
    }
  }
  return result;
}

template<typename T>
T readOrDeclareParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name, const T & fallback,
  const rcl_interfaces::msg::ParameterDescriptor & descriptor)
{
  if (node->has_parameter(name)) {
    T value{};
    node->get_parameter(name, value);
    return value;
  }
  return node->declare_parameter<T>(name, fallback, descriptor);
}

const moveit::core::LinkModel * rigidRoot(const moveit::core::LinkModel * link)
{
  while (link->getParentJointModel() &&
    link->getParentJointModel()->getType() == moveit::core::JointModel::FIXED &&
    link->getParentLinkModel())
  {
    link = link->getParentLinkModel();
  }
  return link;
}

std::int64_t toNanoseconds(const double time_sec)
{
  if (!std::isfinite(time_sec) || time_sec < 0.0 ||
    time_sec > static_cast<double>(std::numeric_limits<std::int64_t>::max()) * 1.0e-9)
  {
    return 0;
  }
  return static_cast<std::int64_t>(std::llround(time_sec * 1.0e9));
}

std::optional<LinearVelocityConstraint> makeOutwardStoppingConstraint(
  const Eigen::VectorXd & outward_gradient, const double remaining_distance,
  const JointMotionState & motion_state, const JointMotionLimits & limits,
  const double period_sec, const double residual_latency_sec)
{
  if (outward_gradient.size() != motion_state.position.size() ||
    !outward_gradient.allFinite() || !std::isfinite(remaining_distance))
  {
    return std::nullopt;
  }
  double maximum_outward_speed = 0.0;
  try {
    maximum_outward_speed = maximumSafeCollisionApproachSpeed(
      std::max(0.0, remaining_distance), -outward_gradient, motion_state, limits,
      period_sec, residual_latency_sec);
    if (motion_state.velocity.lpNorm<Eigen::Infinity>() <= 1.0e-12 &&
      motion_state.acceleration.lpNorm<Eigen::Infinity>() <= 1.0e-12)
    {
      // A stationary command can hold exactly on a virtual envelope boundary.
      maximum_outward_speed = std::max(0.0, maximum_outward_speed);
    }
  } catch (const std::exception &) {
    return std::nullopt;
  }
  if (!std::isfinite(maximum_outward_speed)) {
    return std::nullopt;
  }
  LinearVelocityConstraint row;
  row.coefficients = outward_gradient;
  row.upper_bound = maximum_outward_speed;
  return row;
}

}  // namespace

CompactPathReferenceSettings scaleCompactPathReference(
  const double gain, const double maximum_velocity_radps, const double speed_scale)
{
  if (!std::isfinite(gain) || gain <= 0.0 ||
    !std::isfinite(maximum_velocity_radps) || maximum_velocity_radps <= 0.0 ||
    !std::isfinite(speed_scale) || speed_scale <= 0.0)
  {
    throw std::invalid_argument(
            "compact path gain, velocity and speed scale must be finite and positive");
  }
  const double scaled_gain = gain * speed_scale;
  const double scaled_velocity = maximum_velocity_radps * speed_scale;
  if (!std::isfinite(scaled_gain) || scaled_gain <= 0.0 ||
    !std::isfinite(scaled_velocity) || scaled_velocity <= 0.0)
  {
    throw std::invalid_argument("compact_maneuver.speed_scale is outside usable range");
  }
  return {scaled_gain, scaled_velocity};
}

double preferredBaseJointAngle(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Isometry3d & base_pose_in_planning_frame)
{
  const Eigen::Vector3d ray_in_base = base_pose_in_planning_frame.linear().transpose() *
    (face_in_planning_frame - base_pose_in_planning_frame.translation());
  return std::atan2(ray_in_base.y(), ray_in_base.x());
}

MotionReference::MotionReference(
  const rclcpp::Node::SharedPtr & node,
  const planning_scene_monitor::PlanningSceneMonitorPtr & scene_monitor,
  ControllerParameters parameters, JointMotionLimits limits,
  CollisionConstraintConfig collision_config)
: parameters_(std::move(parameters)), limits_(std::move(limits)),
  collision_validator_(collision_config), robot_model_(scene_monitor->getRobotModel()),
  group_(robot_model_->getJointModelGroup(parameters_.planning_group_name)),
  command_link_(robot_model_->getLinkModel(parameters_.command_frame)),
  base_link_(robot_model_->getLinkModel("link_base")),
  compact_geometry_validator_(nullptr)
{
  if (!base_link_) {
    base_link_ = robot_model_->getRootLink();
  }
  if (!group_ || !command_link_ || !base_link_) {
    throw std::invalid_argument("Motion reference requires the planning group and screen link");
  }
  if (!parameters_.gaze_frame.empty()) {
    const auto * gaze_link = robot_model_->getLinkModel(parameters_.gaze_frame);
    if (!gaze_link || rigidRoot(gaze_link) != rigidRoot(command_link_)) {
      throw std::invalid_argument(
              "gaze_frame must be a camera optical link rigidly fixed to the screen");
    }
  }

  TargetMotionEstimatorConfig target_motion_config;
  target_motion_config.stale_timeout_sec = parameters_.source_timeout_sec;
  target_motion_config.maximum_sample_interval_sec = parameters_.source_timeout_sec;
  target_motion_config.maximum_source_gap_sec = parameters_.source_timeout_sec;
  target_motion_config.timing_mode = parameters_.timing_mode == "auto" ?
    MotionTimingMode::kAuto : MotionTimingMode::kManual;
  target_motion_estimator_ = TargetMotionEstimator(target_motion_config);

  moveit::core::RobotState named_state(robot_model_);
  named_state.setToDefaultValues();
  if (!named_state.setToDefaultValues(group_, "rest")) {
    throw std::invalid_argument("Motion reference requires a named rest posture");
  }
  named_state.copyJointGroupPositions(group_, rest_positions_);

  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  search_enabled_ = readOrDeclareParameter(node, "search_enabled", true, descriptor);
  compact_speed_scale_ = readOrDeclareParameter(
    node, "compact_maneuver.speed_scale", 1.0, descriptor);
  if (!std::isfinite(compact_speed_scale_) || compact_speed_scale_ <= 0.0) {
    throw std::invalid_argument("compact_maneuver.speed_scale must be finite and positive");
  }
  compact_path_settings_ = scaleCompactPathReference(6.0, 0.70, compact_speed_scale_);
  compact_goal_settings_ = scaleCompactPathReference(3.0, 0.10, compact_speed_scale_);
  if (search_enabled_) {
    if (!named_state.setToDefaultValues(group_, "search")) {
      throw std::invalid_argument("Idle search requires a named search posture");
    }
    named_state.copyJointGroupPositions(group_, search_positions_);
    const auto & names = group_->getVariableNames();
    const auto base = std::find(names.begin(), names.end(), "joint1");
    if (base == names.end()) {
      throw std::invalid_argument("Idle search requires joint1");
    }
    search_base_index_ = std::distance(names.begin(), base);
    search_config_.speed_rad_s = readOrDeclareParameter(
      node, "search_speed_rad_s", 0.30, descriptor);
    search_config_.sweep_half_range_rad = readOrDeclareParameter(
      node, "search_sweep_half_range_rad", 2.80, descriptor);
    search_config_.local_half_range_rad = readOrDeclareParameter(
      node, "search_local_half_range_rad", 0.35, descriptor);
    search_config_.local_duration_sec = readOrDeclareParameter(
      node, "search_local_duration_sec", 6.0, descriptor);
    search_.emplace();
    search_->start(rest_positions_, search_positions_, limits_.lower_position +
      limits_.position_margin, limits_.upper_position - limits_.position_margin,
      search_base_index_, search_config_, false, 0.0);
  }

  named_state.setToDefaultValues();
  std::vector<double> configured_goal = readOrDeclareParameter(
    node, "compact_maneuver.goal_joints_rad", std::vector<double>{}, descriptor);
  if (configured_goal.empty()) {
    select_default_compact_goal_ = true;
    for (const auto * profile : {
        "compact_turn_negative_wrist", "compact_turn_positive_wrist"})
    {
      if (!named_state.setToDefaultValues(group_, profile)) {
        continue;
      }
      CompactGoal candidate;
      candidate.profile_id = profile;
      named_state.copyJointGroupPositions(group_, candidate.bounded_positions);
      if (compact_goal_.profile_id.empty() ||
        candidate.profile_id == "compact_turn_negative_wrist")
      {
        compact_goal_ = candidate;
      }
      default_compact_goals_.push_back(std::move(candidate));
    }
  } else {
    if (configured_goal.size() != static_cast<std::size_t>(group_->getVariableCount())) {
      throw std::invalid_argument(
              "compact_maneuver.goal_joints_rad must match the planning group variable count");
    }
    compact_goal_.profile_id = "configured_goal_override";
    compact_goal_.bounded_positions = Eigen::Map<const Eigen::VectorXd>(
      configured_goal.data(), static_cast<Eigen::Index>(configured_goal.size()));
    if (!compact_goal_.bounded_positions.allFinite() ||
      (compact_goal_.bounded_positions.array() < limits_.lower_position.array()).any() ||
      (compact_goal_.bounded_positions.array() > limits_.upper_position.array()).any())
    {
      throw std::invalid_argument(
              "compact_maneuver.goal_joints_rad must be finite and inside joint bounds");
    }
  }
  compact_goal_.path_requirements.base_link_name = "link_base";
  compact_goal_.path_requirements.screen_link_name = parameters_.command_frame;
  compact_goal_.path_requirements.maximum_roll_error_rad = kCompactRollLimitRad;
  compact_goal_.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
  for (auto & candidate : default_compact_goals_) {
    candidate.path_requirements.base_link_name = "link_base";
    candidate.path_requirements.screen_link_name = parameters_.command_frame;
    candidate.path_requirements.maximum_roll_error_rad = kCompactRollLimitRad;
    candidate.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
  }
  if (!compact_goal_.profile_id.empty()) {
    compact_goal_.path_requirements.base_link_name = base_link_->getName();
    compact_geometry_validator_ = std::make_unique<CompactGeometryValidator>(
      robot_model_, base_link_->getName());
  }

  BackgroundPathPlannerConfig planner_config;
  planner_config.move_group_name = parameters_.planning_group_name;
  planner_goal_tolerance_rad_ = planner_config.joint_tolerance_rad;
  planner_ = std::make_unique<BackgroundPathPlanner>(
    node, robot_model_, scene_monitor, planner_config,
    [validator = collision_validator_, limits = limits_, group = group_] (
      const planning_scene::PlanningScene & scene, const moveit::core::RobotState & state)
    {
      Eigen::VectorXd position;
      state.copyJointGroupPositions(group, position);
      if ((position.array() < limits.lower_position.array()).any() ||
      (position.array() > limits.upper_position.array()).any())
      {
        return false;
      }
      JointMotionState stationary{position, Eigen::VectorXd::Zero(position.size()),
        Eigen::VectorXd::Zero(position.size())};
      const auto validation = validator.validatePose(scene, state, *group, stationary);
      return validation.input_valid && !validation.unsafe;
    });
}

void MotionReference::appendCompactEvent(
  std::string event, const double time_sec, std::string reason, std::string details_json)
{
  if (compact_events_.size() == kCompactEventCapacity) {
    ++compact_event_overflow_count_;
    compact_events_.pop_front();
  }
  compact_events_.push_back(CompactEvent{
      std::move(event), toNanoseconds(time_sec), diagnostics_.compact_generation,
      std::move(reason), std::move(details_json)});
  diagnostics_.compact_event_overflow_count = compact_event_overflow_count_;
}

std::vector<CompactEvent> MotionReference::drainCompactEvents()
{
  std::vector<CompactEvent> drained;
  drained.reserve(compact_events_.size());
  while (!compact_events_.empty()) {
    drained.push_back(std::move(compact_events_.front()));
    compact_events_.pop_front();
  }
  return drained;
}

bool MotionReference::compactManeuverActive() const noexcept
{
  return face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
         compact_commit_initialized_;
}

double MotionReference::effectiveCompactRadiusCap() const noexcept
{
  return compact_radius_floor_latched_ ? kCompactRadiusFloorM :
         compact_goal_.path_requirements.maximum_assembly_radius_m;
}

std::vector<CompactGoal> MotionReference::selectDefaultCompactGoals(
  const moveit::core::RobotState & state,
  const planning_scene::PlanningScene & scene,
  const double measured_joint6) const
{
  const auto & names = group_->getVariableNames();
  const auto joint6 = std::find(names.begin(), names.end(), "joint6");
  if (joint6 == names.end() || !std::isfinite(measured_joint6) ||
    limits_.lower_position.size() != static_cast<Eigen::Index>(names.size()) ||
    limits_.upper_position.size() != static_cast<Eigen::Index>(names.size()))
  {
    return {};
  }
  const auto joint6_index = static_cast<Eigen::Index>(std::distance(names.begin(), joint6));
  constexpr double kTwoPi{6.28318530717958647692};
  constexpr double kBoundTolerance{1.0e-10};
  constexpr double kBoundInsetRad{0.02};

  struct RankedCandidate
  {
    CompactGoal goal;
    double joint6_distance{0.0};
    std::size_t winding_rank{0U};
  };
  std::vector<std::vector<RankedCandidate>> candidates_by_profile;
  const auto valid_candidate = [this, &state, &scene](const CompactGoal & candidate) {
      if (candidate.bounded_positions.size() !=
        static_cast<Eigen::Index>(group_->getVariableCount()) ||
        !candidate.bounded_positions.allFinite() ||
        (candidate.bounded_positions.array() < limits_.lower_position.array()).any() ||
        (candidate.bounded_positions.array() > limits_.upper_position.array()).any())
      {
        return false;
      }
      moveit::core::RobotState goal_state(state);
      goal_state.setJointGroupPositions(group_, candidate.bounded_positions);
      goal_state.update(true);
      const auto pose = evaluateCompactUprightPose(
        goal_state.getGlobalLinkTransform(command_link_).linear());
      const double radius = compact_geometry_validator_->maximumAssemblyRadius(goal_state);
      if (!pose || !std::isfinite(radius) || radius > kCompactRadiusFloorM + 1.0e-6 ||
        !std::isfinite(pose->signed_roll_error_rad) ||
        std::abs(pose->signed_roll_error_rad) >
        kCompactRollLimitRad + kCompactPitchValidationToleranceRad ||
        !std::isfinite(pose->signed_pitch_rad))
      {
        return false;
      }
      Eigen::VectorXd position;
      goal_state.copyJointGroupPositions(group_, position);
      const JointMotionState stationary{
        position, Eigen::VectorXd::Zero(position.size()),
        Eigen::VectorXd::Zero(position.size())};
      const auto clearance = collision_validator_.validatePose(
        scene, goal_state, *group_, stationary);
      return clearance.input_valid && !clearance.unsafe;
    };

  for (const auto & profile : default_compact_goals_) {
    if (profile.bounded_positions.size() != static_cast<Eigen::Index>(names.size())) {
      continue;
    }
    const double nominal_joint6 = profile.bounded_positions[joint6_index];
    std::vector<RankedCandidate> profile_candidates;
    for (int winding = -2; winding <= 2; ++winding) {
      CompactGoal candidate = profile;
      double equivalent_joint6 = nominal_joint6 + winding * kTwoPi;
      if (equivalent_joint6 < limits_.lower_position[joint6_index] - kBoundTolerance ||
        equivalent_joint6 > limits_.upper_position[joint6_index] + kBoundTolerance)
      {
        continue;
      }
      const bool at_joint6_lower_bound = std::abs(
        equivalent_joint6 - limits_.lower_position[joint6_index]) <= kBoundTolerance;
      const bool at_joint6_upper_bound = std::abs(
        equivalent_joint6 - limits_.upper_position[joint6_index]) <= kBoundTolerance;
      if (at_joint6_lower_bound) {
        equivalent_joint6 = limits_.lower_position[joint6_index] + kBoundInsetRad;
      }
      if (!at_joint6_lower_bound && at_joint6_upper_bound) {
        equivalent_joint6 = limits_.upper_position[joint6_index] - kBoundInsetRad;
      }
      candidate.bounded_positions[joint6_index] = equivalent_joint6;
      if (!valid_candidate(candidate)) {
        continue;
      }

      profile_candidates.push_back(RankedCandidate{
          std::move(candidate), std::abs(equivalent_joint6 - measured_joint6), 0U});
    }
    std::stable_sort(profile_candidates.begin(), profile_candidates.end(),
      [](const RankedCandidate & lhs, const RankedCandidate & rhs) {
        return lhs.joint6_distance < rhs.joint6_distance;
      });
    for (std::size_t index = 0; index < profile_candidates.size(); ++index) {
      profile_candidates[index].winding_rank = index;
    }
    candidates_by_profile.push_back(std::move(profile_candidates));
  }

  std::vector<CompactGoal> result;
  std::size_t maximum_winding_rank = 0U;
  for (const auto & profile_candidates : candidates_by_profile) {
    maximum_winding_rank = std::max(maximum_winding_rank, profile_candidates.size());
  }
  for (std::size_t rank = 0U; rank < maximum_winding_rank; ++rank) {
    std::vector<RankedCandidate> tier;
    for (const auto & profile_candidates : candidates_by_profile) {
      const auto candidate = std::find_if(
        profile_candidates.begin(), profile_candidates.end(),
        [rank](const RankedCandidate & value) {return value.winding_rank == rank;});
      if (candidate != profile_candidates.end()) {
        tier.push_back(*candidate);
      }
    }
    std::sort(tier.begin(), tier.end(),
      [](const RankedCandidate & lhs, const RankedCandidate & rhs) {
        if (lhs.joint6_distance != rhs.joint6_distance) {
          return lhs.joint6_distance < rhs.joint6_distance;
        }
        return lhs.goal.profile_id == "compact_turn_positive_wrist" &&
               rhs.goal.profile_id != "compact_turn_positive_wrist";
      });
    for (auto & candidate : tier) {
      result.push_back(std::move(candidate.goal));
    }
  }
  return result;
}

bool MotionReference::includeCompactEntryEnvelopePosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  if ((maximum_radius != nullptr && !compact_geometry_validator_) ||
    position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    !position.allFinite())
  {
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  if (maximum_radius != nullptr) {
    const double radius = compact_geometry_validator_->maximumAssemblyRadius(candidate);
    if (!std::isfinite(radius)) {
      return false;
    }
    *maximum_radius = std::max(*maximum_radius, radius);
  }
  if (maximum_abs_pitch_rad != nullptr || maximum_abs_roll_rad != nullptr) {
    const auto pose = evaluateCompactUprightPose(
      candidate.getGlobalLinkTransform(command_link_).linear());
    if (!pose ||
      (maximum_abs_pitch_rad != nullptr && !std::isfinite(pose->signed_pitch_rad)) ||
      (maximum_abs_roll_rad != nullptr && !std::isfinite(pose->signed_roll_error_rad)))
    {
      return false;
    }
    if (maximum_abs_pitch_rad != nullptr) {
      *maximum_abs_pitch_rad = std::max(
        *maximum_abs_pitch_rad, std::abs(pose->signed_pitch_rad));
    }
    if (maximum_abs_roll_rad != nullptr) {
      *maximum_abs_roll_rad = std::max(
        *maximum_abs_roll_rad, std::abs(pose->signed_roll_error_rad));
    }
  }
  return true;
}

bool MotionReference::includeCompactEntryEnvelope(
  const moveit::core::RobotState & state, const double time_sec,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  if (!includeCompactEntryEnvelopePosition(
      state, groupPosition(state), maximum_radius, maximum_abs_pitch_rad,
      maximum_abs_roll_rad))
  {
    return false;
  }
  const auto include_publication = [this, &state, time_sec, &maximum_radius,
      maximum_abs_pitch_rad, maximum_abs_roll_rad](
    const PublishedTrajectoryRecord * publication) {
      if (publication == nullptr || publication->execution_queue.empty()) {
        return true;
      }
      const double stop_time_sec = publication->stationary_time.seconds();
      if (!std::isfinite(stop_time_sec)) {
        return false;
      }
      if (time_sec >= stop_time_sec) {
        return true;
      }
      const moveit_servo::KinematicState * previous = nullptr;
      std::size_t next_index = publication->execution_queue.size();
      for (std::size_t index = 0; index < publication->execution_queue.size(); ++index) {
        const auto & sample = publication->execution_queue[index];
        const double sample_time_sec = sample.time_stamp.seconds();
        if (!std::isfinite(sample_time_sec)) {
          return false;
        }
        if (sample_time_sec < time_sec) {
          previous = &sample;
        } else {
          next_index = index;
          break;
        }
      }
      const auto include_interpolated = [this, &state, &maximum_radius,
          maximum_abs_pitch_rad, maximum_abs_roll_rad](
        const moveit_servo::KinematicState & before,
        const moveit_servo::KinematicState & after, const double at_time_sec) {
          const double before_time_sec = before.time_stamp.seconds();
          const double after_time_sec = after.time_stamp.seconds();
          if (after_time_sec <= before_time_sec ||
            before.positions.size() != after.positions.size() ||
            !before.positions.allFinite() || !after.positions.allFinite())
          {
            return false;
          }
          const double fraction = std::clamp(
            (at_time_sec - before_time_sec) / (after_time_sec - before_time_sec), 0.0, 1.0);
          const Eigen::VectorXd position =
            before.positions + fraction * (after.positions - before.positions);
          return includeCompactEntryEnvelopePosition(state, position, maximum_radius,
            maximum_abs_pitch_rad, maximum_abs_roll_rad);
        };
      if (previous != nullptr && next_index < publication->execution_queue.size()) {
        const auto & next = publication->execution_queue[next_index];
        const double next_time_sec = next.time_stamp.seconds();
        if (!includeCompactEntryEnvelopePosition(
            state, previous->positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
        if (!include_interpolated(*previous, next, time_sec)) {
          return false;
        }
        if (next_time_sec > stop_time_sec && previous->time_stamp.seconds() < stop_time_sec &&
          !include_interpolated(*previous, next, stop_time_sec))
        {
          return false;
        }
      } else if (previous != nullptr && time_sec < stop_time_sec) {
        // A publication with no future sample holds its final point until stop.
        if (!includeCompactEntryEnvelopePosition(
            state, previous->positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
      }
      for (std::size_t index = next_index;
        index < publication->execution_queue.size(); ++index)
      {
        const auto & sample = publication->execution_queue[index];
        if (sample.time_stamp.seconds() > stop_time_sec) {
          break;
        }
        if (sample.time_stamp.seconds() >= time_sec &&
          !includeCompactEntryEnvelopePosition(
            state, sample.positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
      }
      return true;
    };
  if (!include_publication(active_publication)) {
    return false;
  }
  return latest_publication == active_publication || include_publication(latest_publication);
}

bool MotionReference::maybeArmFaceUprightRollGuard(
  const moveit::core::RobotState & state, const double time_sec,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (upright_roll_guard_armed_) {
    return true;
  }
  if (!actual_feedback || !actual_feedback_time_sec || !command_queue_state ||
    !std::isfinite(time_sec) || !std::isfinite(*actual_feedback_time_sec) ||
    time_sec < *actual_feedback_time_sec ||
    time_sec - *actual_feedback_time_sec > parameters_.state_feedback_timeout_sec)
  {
    return false;
  }
  const auto joint_count = static_cast<Eigen::Index>(group_->getVariableCount());
  if (actual_feedback->position.size() != joint_count ||
    actual_feedback->velocity.size() != joint_count ||
    command_queue_state->position.size() != joint_count ||
    command_queue_state->velocity.size() != joint_count ||
    command_queue_state->acceleration.size() != joint_count ||
    !actual_feedback->position.allFinite() || !actual_feedback->velocity.allFinite() ||
    !command_queue_state->position.allFinite() || !command_queue_state->velocity.allFinite() ||
    !command_queue_state->acceleration.allFinite())
  {
    return false;
  }

  const auto roll_at = [this, &state](const Eigen::VectorXd & position)
    -> std::optional<double> {
      moveit::core::RobotState candidate(state);
      candidate.setJointGroupPositions(group_, position);
      candidate.update(true);
      const auto pose = evaluateCompactUprightPose(
        candidate.getGlobalLinkTransform(command_link_).linear());
      if (!pose || !std::isfinite(pose->signed_roll_error_rad)) {
        return std::nullopt;
      }
      return std::abs(pose->signed_roll_error_rad);
    };
  const auto actual_roll = roll_at(actual_feedback->position);
  const auto command_roll = roll_at(command_queue_state->position);
  if (!actual_roll || !command_roll ||
    *actual_roll > kCompactRollLimitRad + kCompactPitchValidationToleranceRad ||
    *command_roll > kCompactRollLimitRad + kCompactPitchValidationToleranceRad)
  {
    return false;
  }

  const bool active_available = active_publication != nullptr;
  const bool latest_available = latest_publication != nullptr;
  const bool no_publication_history = !active_available && !latest_available;
  if (active_available != latest_available) {
    return false;
  }
  const auto has_stopped_record = [joint_count](
    const PublishedTrajectoryRecord * publication) {
      if (publication == nullptr || publication->execution_queue.empty() ||
        !std::isfinite(publication->stationary_time.seconds()))
      {
        return false;
      }
      const auto stationary = std::find_if(
        publication->execution_queue.begin(), publication->execution_queue.end(),
        [publication](const auto & sample) {
          return sample.time_stamp == publication->stationary_time;
        });
      if (stationary == publication->execution_queue.end() ||
        stationary->positions.size() != joint_count ||
        stationary->velocities.size() != joint_count ||
        stationary->accelerations.size() != joint_count ||
        !stationary->positions.allFinite() || !stationary->velocities.allFinite() ||
        !stationary->accelerations.allFinite())
      {
        return false;
      }
      return stationary->velocities.lpNorm<Eigen::Infinity>() <= 1.0e-6 &&
             stationary->accelerations.lpNorm<Eigen::Infinity>() <= 1.0e-6;
    };
  if (!no_publication_history &&
    (!has_stopped_record(active_publication) ||
    !has_stopped_record(latest_publication)))
  {
    return false;
  }
  if (no_publication_history) {
    const bool positions_match =
      (actual_feedback->position - command_queue_state->position).
      lpNorm<Eigen::Infinity>() <= kRestArrivalRad;
    if (!positions_match || !motionSettled(*actual_feedback, command_queue_state)) {
      return false;
    }
  }

  double maximum_abs_roll_rad = std::max(*actual_roll, *command_roll);
  if (!includeCompactEntryEnvelope(
      state, time_sec, active_publication, latest_publication,
      nullptr, nullptr, &maximum_abs_roll_rad) ||
    maximum_abs_roll_rad > kCompactRollLimitRad + kCompactPitchValidationToleranceRad)
  {
    return false;
  }

  upright_roll_guard_armed_ = true;
  diagnostics_.upright_roll_guard_armed = true;
  const std::string details =
    "{\"actual_abs_roll_error_rad\":" + jsonNumber(*actual_roll) +
    ",\"command_abs_roll_error_rad\":" + jsonNumber(*command_roll) +
    ",\"fullstop_max_abs_roll_error_rad\":" + jsonNumber(maximum_abs_roll_rad) +
    ",\"maximum_abs_roll_cap_rad\":" + jsonNumber(kCompactRollLimitRad) + "}";
  appendCompactEvent("upright_roll_guard_armed", time_sec,
    "actual_command_and_published_stop_envelope_within_cap", details);
  return true;
}

void MotionReference::maybeLatchCompactRadiusCap(
  const moveit::core::RobotState & state, const double time_sec,
  const double publication_lookback_sec,
  const JointMotionState & actual_feedback, const JointMotionState & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (!compactManeuverActive() || compact_radius_floor_latched_ ||
    actual_feedback.position.size() != group_->getVariableCount() ||
    command_queue_state.position.size() != group_->getVariableCount() ||
    !actual_feedback.position.allFinite() || !command_queue_state.position.allFinite())
  {
    return;
  }
  double actual_radius = 0.0;
  double command_radius = 0.0;
  if (!includeCompactEntryEnvelopePosition(
      state, actual_feedback.position, &actual_radius) ||
    !includeCompactEntryEnvelopePosition(
      state, command_queue_state.position, &command_radius) ||
    std::max(actual_radius, command_radius) > kCompactRadiusFloorM + 1.0e-6)
  {
    return;
  }

  // Queue geometry is expensive; inspect it only after measured and queued q
  // have both entered the floor, then latch once for this commitment.
  double envelope_radius = 0.0;
  if (!includeCompactEntryEnvelope(
      state, publication_lookback_sec, active_publication, latest_publication, &envelope_radius) ||
    envelope_radius > kCompactRadiusFloorM + 1.0e-6)
  {
    return;
  }

  compact_radius_floor_latched_ = true;
  const double total_envelope_radius = std::max(actual_radius, command_radius);
  const std::string details =
    "{\"entry_radius_cap_m\":" + jsonNumber(diagnostics_.compact_entry_radius_cap_m) +
    ",\"effective_radius_cap_m\":" + jsonNumber(kCompactRadiusFloorM) +
    ",\"actual_feedback_stamp_ns\":" +
    std::to_string(toNanoseconds(publication_lookback_sec)) +
    ",\"actual_position_rad\":" + jsonVector(actual_feedback.position) +
    ",\"command_position_rad\":" + jsonVector(command_queue_state.position) +
    ",\"actual_radius_m\":" + jsonNumber(actual_radius) +
    ",\"command_radius_m\":" + jsonNumber(command_radius) +
    ",\"published_remainder_radius_m\":" + jsonNumber(envelope_radius) +
    ",\"radius_envelope_max_m\":" +
    jsonNumber(std::max(total_envelope_radius, envelope_radius)) + "}";
  appendCompactEvent(
    "compact_radius_floor_latched", time_sec, "full_envelope_inside_floor", details);
}

bool MotionReference::validateCompactPathPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (!compactManeuverActive()) {
    return true;
  }
  if (!compact_geometry_validator_ || position.size() != group_->getVariableCount() ||
    !position.allFinite())
  {
    if (violation) {
      *violation = "state is unavailable";
    }
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  const double radius = compact_geometry_validator_->maximumAssemblyRadius(candidate);
  if (!std::isfinite(radius) || radius > effectiveCompactRadiusCap() + 1.0e-6) {
    if (violation) {
      *violation = "assembly radius";
    }
    return false;
  }
  if (!validateUprightRollPose(candidate, violation)) {
    return false;
  }
  if (compact_face_return_pending_) {
    return true;
  }
  const auto pose = evaluateCompactUprightPose(
    candidate.getGlobalLinkTransform(command_link_).linear());
  const auto pitch_cap = compact_goal_.path_requirements.maximum_abs_screen_pitch_rad;
  if (!pitch_cap || !std::isfinite(*pitch_cap) || !pose ||
    !std::isfinite(pose->signed_pitch_rad))
  {
    if (violation) {
      *violation = "screen pitch state is unavailable";
    }
    return false;
  }
  if (std::abs(pose->signed_pitch_rad) > *pitch_cap) {
    if (std::abs(pose->signed_pitch_rad) <=
      *pitch_cap + kCompactPitchValidationToleranceRad)
    {
      return true;
    }
    if (violation) {
      *violation = "screen pitch";
    }
    return false;
  }
  return true;
}

bool MotionReference::validateUprightRollPose(
  const moveit::core::RobotState & state, std::string * violation) const
{
  const auto pose = evaluateCompactUprightPose(
    state.getGlobalLinkTransform(command_link_).linear());
  if (!pose || !std::isfinite(pose->signed_roll_error_rad)) {
    if (violation) {
      *violation = "screen roll state is unavailable";
    }
    return false;
  }
  if (std::abs(pose->signed_roll_error_rad) >
    kCompactRollLimitRad + kCompactPitchValidationToleranceRad)
  {
    if (violation) {
      *violation = "screen roll";
    }
    return false;
  }
  return true;
}

bool MotionReference::validateUprightRollPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (!group_ || !command_link_ ||
    position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    !position.allFinite())
  {
    if (violation) {
      *violation = "screen roll state is unavailable";
    }
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  return validateUprightRollPose(candidate, violation);
}

bool MotionReference::validateCommandPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  if (compactManeuverActive()) {
    return validateCompactPathPosition(state, position, violation);
  }
  if (upright_roll_guard_armed_) {
    return validateUprightRollPosition(state, position, violation);
  }
  if (violation) {
    violation->clear();
  }
  return true;
}

void MotionReference::reset(const bool preserve_compact_maneuver)
{
  const bool preserve_committed_maneuver = preserve_compact_maneuver &&
    compact_commit_initialized_;
  const bool preserve_return_stage = preserve_committed_maneuver &&
    compact_face_return_pending_ && compact_goal_reached_;
  planner_->cancel();
  follower_.reset();
  target_motion_estimator_.reset();
  base_handoff_policy_.clear();
  pending_generation_.reset();
  compact_preparation_.reset();
  active_scene_revision_.reset();
  last_face_return_evidence_time_sec_.reset();
  face_return_dwell_sec_ = 0.0;
  compact_return_full_face_active_ = false;
  rest_mode_ = RestMode::kNone;
  search_pattern_.clear();
  rest_completed_ = false;
  rest_settling_ = false;
  target_id_.reset();
  handoff_target_id_.reset();
  target_bearing_motion_sign_ = 0;
  target_bearing_moving_ = false;
  compact_path_adopted_ = false;
  compact_goal_reached_ = preserve_return_stage;
  compact_path_endpoint_reached_ = preserve_return_stage;
  compact_goal_event_emitted_ = preserve_return_stage && compact_goal_event_emitted_;
  compact_face_return_pending_ = preserve_return_stage;
  compact_plan_pending_ = false;
  compact_return_full_face_active_ = false;
  diagnostics_.compact_return_full_face_active = false;
  diagnostics_.compact_return_face_dwell_sec = 0.0;
  diagnostics_.compact_return_face_qualified = false;
  diagnostics_.compact_actual_settled = false;
  diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_roll_error_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
  if (!preserve_committed_maneuver) {
    face_policy_.clearForStop();
    compact_commit_initialized_ = false;
    compact_radius_floor_latched_ = false;
    diagnostics_.compact_radius_floor_latched = false;
    diagnostics_.compact_effective_radius_cap_m =
      std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_actual_settled = false;
    // A re-arm may adopt another path, but path_started is emitted once per
    // compact generation and records its first actual path execution.
    compact_path_started_event_emitted_ = false;
    if (!preserve_compact_maneuver) {
      upright_roll_guard_armed_ = false;
      diagnostics_.upright_roll_guard_armed = false;
      compact_goal_.path_requirements.maximum_abs_screen_pitch_rad.reset();
      compact_goal_.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
      diagnostics_.compact_entry_radius_cap_m =
        std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_radius_m =
        std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_roll_error_rad =
        std::numeric_limits<double>::quiet_NaN();
      diagnostics_.compact_generation = 0U;
      diagnostics_.compact_reason.clear();
    }
  }
  diagnostics_.compact_goal_reached = preserve_return_stage ? 1.0 : 0.0;
  diagnostics_.compact_phase = preserve_return_stage ?
    "RETURN_FACE" : face_policy_.phase() == FaceTrackingPhase::kCompactManeuver ?
    "PLANNING" : "INACTIVE";
  diagnostics_.tracking_phase = preserve_return_stage ? "COMPACT_RETURN_FACE" : "FACE";
  diagnostics_.state = preserve_return_stage ? "COMPACT_RETURN_FACE" : "HOLD";
}

void MotionReference::rejectPath()
{
  if (compact_preparation_ && !compact_commit_initialized_) {
    return;
  }
  if (follower_.active()) {
    ++diagnostics_.paths_abandoned;
    follower_.reset();
    if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver) {
      compact_path_adopted_ = false;
    }
  }
}

const MotionReferenceDiagnostics & MotionReference::diagnostics() const noexcept
{
  return diagnostics_;
}

Eigen::VectorXd MotionReference::groupPosition(const moveit::core::RobotState & state) const
{
  Eigen::VectorXd position;
  state.copyJointGroupPositions(group_, position);
  return position;
}

void MotionReference::updateBaseHandoff(
  const moveit::core::RobotState & state,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & feedback_time_sec,
  const msg::TrackingTarget & target, const TargetMotionEstimate & target_motion,
  const std::optional<JointMotionState> & command_queue_state,
  MotionReferenceResult & result)
{
  const auto & names = group_->getVariableNames();
  const auto base = std::find(names.begin(), names.end(), "joint1");
  if (base == names.end()) {
    return;
  }
  const auto index = static_cast<Eigen::Index>(std::distance(names.begin(), base));
  const bool target_changed = target.target_id >= 0 && handoff_target_id_ &&
    *handoff_target_id_ != target.target_id;
  const bool release_for_pair_yaw = base_handoff_policy_.heldPositionRad() &&
    pair_yaw_assessment_.valid && pair_yaw_assessment_.demand_present &&
    pair_yaw_assessment_.j1.required_direction != BaseJointDirection::kNone &&
    pair_yaw_assessment_.j1.required_direction != base_handoff_policy_.heldDirection();
  if (release_for_pair_yaw) {
    base_handoff_policy_.clear();
  }
  if (target.target_id >= 0) {
    handoff_target_id_ = target.target_id;
  }

  const Eigen::Isometry3d base_pose = state.getGlobalLinkTransform(base_link_);
  const Eigen::Vector3d ray = base_pose.linear().transpose() *
    (Eigen::Vector3d(target.face.x, target.face.y, target.face.z) - base_pose.translation());
  const Eigen::Vector3d velocity = base_pose.linear().transpose() * target_motion.face_velocity;
  const double radial_sq = ray.head<2>().squaredNorm();
  BaseJointDirection target_motion_direction = BaseJointDirection::kNone;
  if (radial_sq > 1.0e-9 && ray.allFinite() && velocity.allFinite()) {
    const double rate = (ray.x() * velocity.y() - ray.y() * velocity.x()) / radial_sq;
    if (std::isfinite(rate) && std::abs(rate) >= 0.02) {
      target_motion_direction = rate > 0.0 ? BaseJointDirection::kPositive :
        BaseJointDirection::kNegative;
    }
  }

  BaseJointDirection edge_direction = BaseJointDirection::kNone;
  if (actual_feedback && feedback_time_sec && std::isfinite(*feedback_time_sec) &&
    actual_feedback->position.size() == limits_.lower_position.size() &&
    actual_feedback->velocity.size() == limits_.lower_position.size() &&
    actual_feedback->acceleration.size() == limits_.lower_position.size() &&
    actual_feedback->position.allFinite() && actual_feedback->velocity.allFinite() &&
    actual_feedback->acceleration.allFinite())
  {
    const double q = actual_feedback->position[index];
    const auto reserve = measureBoundedJointReserve(
      q, limits_.lower_position[index], limits_.upper_position[index]);
    if (reserve.inside_bounds && reserve.lower_margin_rad <=
      limits_.position_margin[index] + 0.10)
    {
      edge_direction = BaseJointDirection::kNegative;
    }
    if (edge_direction == BaseJointDirection::kNone && reserve.inside_bounds &&
      reserve.upper_margin_rad <= limits_.position_margin[index] + 0.10)
    {
      edge_direction = BaseJointDirection::kPositive;
    }
  }

  BaseJointDirection yaw_direction = BaseJointDirection::kNone;
  if (result.tracking && result.tracking->task.primary_matrix.cols() > index) {
    const auto & task = result.tracking->task;
    const double requested = task.primary_matrix.col(index).dot(task.primary_reference);
    if (std::isfinite(requested) && std::abs(requested) >= 1.0e-6) {
      yaw_direction = requested > 0.0 ? BaseJointDirection::kPositive :
        BaseJointDirection::kNegative;
    }
  }

  double predicted_hold_position = 0.0;
  bool have_brake_endpoint = false;
  const auto & brake_state = command_queue_state ? command_queue_state : actual_feedback;
  if (brake_state && brake_state->position.size() == limits_.lower_position.size() &&
    brake_state->velocity.size() == limits_.lower_position.size() &&
    brake_state->acceleration.size() == limits_.lower_position.size())
  {
    detail::ScalarBrakingOptions options;
    options.period_sec = parameters_.control_period_sec;
    options.maximum_braking_steps = 1000U;
    const auto rollout = detail::rolloutScalarBraking(
      detail::ScalarBrakingState{brake_state->position[index],
        brake_state->velocity[index], brake_state->acceleration[index]},
      detail::ScalarBrakingLimits{limits_.max_velocity[index],
        limits_.max_acceleration[index], limits_.max_jerk[index]}, options, true);
    if (rollout.commandAvailable() && !rollout.points.empty()) {
      predicted_hold_position = rollout.points.back().position;
      have_brake_endpoint = predicted_hold_position >= limits_.lower_position[index] &&
        predicted_hold_position <= limits_.upper_position[index];
    }
  }

  BaseJointHandoffInput input;
  input.working_edge_reached = !release_for_pair_yaw &&
    edge_direction != BaseJointDirection::kNone &&
    have_brake_endpoint;
  const bool valid_actual_base = actual_feedback &&
    actual_feedback->position.size() == limits_.lower_position.size() &&
    index >= 0 && index < actual_feedback->position.size() &&
    actual_feedback->position.allFinite();
  input.actual_base_position_rad = valid_actual_base ?
    actual_feedback->position[index] : std::numeric_limits<double>::quiet_NaN();
  input.predicted_hold_position_rad = predicted_hold_position;
  input.edge_direction = edge_direction;
  input.tracking_yaw_direction = yaw_direction;
  input.target_bearing_motion_direction = target_motion_direction;
  input.valid_target_changed = target_changed;
  const auto held_position = base_handoff_policy_.update(input);
  if (held_position) {
    result.j1_hold_position_rad = held_position;
    result.j1_joint_index = static_cast<std::size_t>(index);
    result.j1_hold_direction = static_cast<int>(base_handoff_policy_.heldDirection());
    diagnostics_.j1_hold_direction = result.j1_hold_direction;
    diagnostics_.j1_hold_q = *held_position;
  } else {
    diagnostics_.j1_hold_direction = 0;
    diagnostics_.j1_hold_q = 0.0;
  }
}

std::optional<HierarchicalVelocityTask> MotionReference::faceTask(
  const moveit::core::RobotState & state, const msg::TrackingTarget & target,
  const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
  const bool face_tracking, FaceReturnPointingTask * return_pointing)
{
  Eigen::MatrixXd jacobian;
  if (!state.getJacobian(group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false)) {
    return std::nullopt;
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  std::optional<GazeKinematics> gaze;
  if (!parameters_.gaze_frame.empty()) {
    const auto * gaze_link = robot_model_->getLinkModel(parameters_.gaze_frame);
    GazeKinematics kinematics;
    kinematics.pose = state.getGlobalLinkTransform(gaze_link);
    kinematics.pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    if (state.getJacobian(group_, gaze_link, Eigen::Vector3d::Zero(),
      kinematics.jacobian, false))
    {
      if (root) {
        const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
        kinematics.jacobian.topRows(3) = (rotation * kinematics.jacobian.topRows(3)).eval();
        kinematics.jacobian.bottomRows(3) =
          (rotation * kinematics.jacobian.bottomRows(3)).eval();
      }
      gaze = std::move(kinematics);
    }
  }
  TrackingVelocityTaskConfig config;
  config.position_gain = parameters_.position_gain;
  config.orientation_gain = parameters_.orientation_gain;
  config.position_deadband_m = parameters_.position_deadband_m;
  config.pointing_deadband_rad = parameters_.pointing_deadband_rad;
  config.maximum_linear_reference_mps = parameters_.maximum_linear_reference_mps;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  config.roll_weight = parameters_.secondary_roll_weight;
  const Eigen::Quaterniond orientation(
    target.pose.orientation.w, target.pose.orientation.x,
    target.pose.orientation.y, target.pose.orientation.z);
  if (!orientation.coeffs().allFinite() || orientation.norm() < 1.0e-9) {
    return std::nullopt;
  }
  const auto task = makeTrackingVelocityTask(
    state.getGlobalLinkTransform(command_link_), jacobian,
    Eigen::Vector3d(target.pose.position.x, target.pose.position.y, target.pose.position.z),
    face_tracking ? std::optional<Eigen::Vector3d>{Eigen::Vector3d(
      target.face.x, target.face.y, target.face.z)} : std::nullopt,
    orientation.normalized().toRotationMatrix(), config, target_motion.face_velocity,
    target_motion.goal_velocity, gaze);
  if (!task) {
    return std::nullopt;
  }
  if (return_pointing) {
    if (!face_tracking) {
      return std::nullopt;
    }
    const auto pointing = makeFaceReturnPointingTask(
      state.getGlobalLinkTransform(command_link_), jacobian,
      Eigen::Vector3d(target.face.x, target.face.y, target.face.z),
      target_motion.face_velocity, config, task->task.primary_matrix.topRows(2),
      task->task.primary_reference.head(2), gaze);
    if (!pointing) {
      return std::nullopt;
    }
    *return_pointing = *pointing;
  }
  result.tracking = *task;
  return task->task;
}

std::optional<MotionReferenceResult::FaceAlignmentMeasurement>
MotionReference::measureActualFaceAlignment(
  const moveit::core::RobotState & state, const msg::TrackingTarget & target,
  const double time_sec, const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec) const
{
  if (!actual_feedback || actual_feedback->position.size() != group_->getVariableCount() ||
    !actual_feedback->position.allFinite())
  {
    return std::nullopt;
  }
  const auto * measurement_link = parameters_.gaze_frame.empty() ? command_link_ :
    robot_model_->getLinkModel(parameters_.gaze_frame);
  if (!measurement_link) {
    return std::nullopt;
  }
  moveit::core::RobotState actual_state(state);
  actual_state.setJointGroupPositions(group_, actual_feedback->position);
  actual_state.update(true);
  Eigen::Isometry3d measurement_pose = actual_state.getGlobalLinkTransform(measurement_link);
  if (!parameters_.gaze_frame.empty()) {
    measurement_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
  }
  const Eigen::Vector3d face_position(target.face.x, target.face.y, target.face.z);
  const Eigen::Vector3d face_ray = face_position - measurement_pose.translation();
  const double ray_norm = face_ray.norm();
  const Eigen::Isometry3d actual_screen_pose =
    actual_state.getGlobalLinkTransform(command_link_);
  const auto actual_roll_error = measureUprightRollError(actual_screen_pose);
  if (!face_ray.allFinite() || !std::isfinite(ray_norm) || ray_norm <= 1.0e-9 ||
    !actual_roll_error)
  {
    return std::nullopt;
  }
  const Eigen::Vector3d current_normal = measurement_pose.linear().col(0);
  const Eigen::Vector3d target_normal = face_ray / ray_norm;
  const double pointing_error = std::acos(std::clamp(
      current_normal.dot(target_normal), -1.0, 1.0));
  if (!std::isfinite(pointing_error)) {
    return std::nullopt;
  }
  const double feedback_time = actual_feedback_time_sec.value_or(time_sec);
  return MotionReferenceResult::FaceAlignmentMeasurement{
    pointing_error, *actual_roll_error,
    static_cast<std::int64_t>(std::llround(feedback_time * 1.0e9))};
}

PairYawResourceAssessment MotionReference::evaluatePairYawResource(
  const moveit::core::RobotState & state, const msg::TrackingTarget & target,
  const TargetMotionEstimate & target_motion,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state) const
{
  PairYawResourceAssessment invalid;
  const auto & names = group_->getVariableNames();
  const auto joint1 = std::find(names.begin(), names.end(), "joint1");
  const auto joint4 = std::find(names.begin(), names.end(), "joint4");
  if (joint1 == names.end() || joint4 == names.end() || !actual_feedback ||
    actual_feedback->position.size() != static_cast<Eigen::Index>(names.size()) ||
    actual_feedback->velocity.size() != actual_feedback->position.size() ||
    actual_feedback->acceleration.size() != actual_feedback->position.size() ||
    !actual_feedback->position.allFinite() || !actual_feedback->velocity.allFinite() ||
    !actual_feedback->acceleration.allFinite())
  {
    return invalid;
  }
  const auto joint1_index = static_cast<Eigen::Index>(std::distance(names.begin(), joint1));
  const auto joint4_index = static_cast<Eigen::Index>(std::distance(names.begin(), joint4));
  const auto * tracking_link = parameters_.gaze_frame.empty() ? command_link_ :
    robot_model_->getLinkModel(parameters_.gaze_frame);
  if (!tracking_link) {
    return invalid;
  }

  moveit::core::RobotState actual_state(state);
  actual_state.setJointGroupPositions(group_, actual_feedback->position);
  actual_state.update(true);
  const Eigen::Isometry3d base_pose = actual_state.getGlobalLinkTransform(base_link_);
  Eigen::Isometry3d tracking_pose = actual_state.getGlobalLinkTransform(tracking_link);
  if (!parameters_.gaze_frame.empty()) {
    tracking_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
  }
  const Eigen::Vector3d face_position(target.face.x, target.face.y, target.face.z);
  const Eigen::Vector3d ray_world = face_position - tracking_pose.translation();
  const Eigen::Vector3d ray = base_pose.linear().transpose() * ray_world;
  const Eigen::Vector3d normal = base_pose.linear().transpose() * tracking_pose.linear().col(0);
  const double ray_squared = ray.head<2>().squaredNorm();
  const double normal_squared = normal.head<2>().squaredNorm();
  if (!ray.allFinite() || !normal.allFinite() || !std::isfinite(ray_squared) ||
    !std::isfinite(normal_squared) || ray_squared <= 1.0e-9 || normal_squared <= 1.0e-9)
  {
    return invalid;
  }

  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  const auto world_jacobian = [&](
    const moveit::core::LinkModel * link, Eigen::MatrixXd & jacobian) {
      if (!actual_state.getJacobian(group_, link, Eigen::Vector3d::Zero(), jacobian, false) ||
        jacobian.rows() != 6 || jacobian.cols() != static_cast<Eigen::Index>(names.size()))
      {
        return false;
      }
      if (root) {
        const Eigen::Matrix3d rotation = actual_state.getGlobalLinkTransform(root).linear();
        jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
        jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
      }
      return jacobian.allFinite();
    };
  Eigen::MatrixXd tracking_jacobian;
  if (!world_jacobian(tracking_link, tracking_jacobian)) {
    return invalid;
  }
  // The supported xArm6 planning group is rooted at fixed link_base. Its
  // transform is constant with respect to the six arm joints.
  Eigen::MatrixXd base_jacobian = Eigen::MatrixXd::Zero(6, names.size());
  if (base_link_ != root && !world_jacobian(base_link_, base_jacobian)) {
    return invalid;
  }

  PairYawResourceInput input;
  const double bearing = std::atan2(ray.y(), ray.x());
  const double normal_heading = std::atan2(normal.y(), normal.x());
  input.signed_yaw_error_rad = std::atan2(
    std::sin(bearing - normal_heading), std::cos(bearing - normal_heading));
  const Eigen::Vector3d target_velocity_base =
    base_pose.linear().transpose() * target_motion.face_velocity;
  input.target_bearing_rate_radps =
    (ray.x() * target_velocity_base.y() - ray.y() * target_velocity_base.x()) /
    ray_squared;
  input.orientation_gain_per_sec = parameters_.orientation_gain;
  input.pointing_deadband_rad = parameters_.pointing_deadband_rad;
  input.target_rate_deadband_radps = 0.02;

  const auto make_joint_input = [&](const Eigen::Index index,
    const BaseJointDirection held_direction) -> std::optional<PairYawJointResourceInput> {
      if (index < 0 || index >= limits_.lower_position.size() ||
        index >= limits_.upper_position.size() || index >= limits_.position_margin.size() ||
        index >= limits_.max_velocity.size() || index >= limits_.max_acceleration.size() ||
        index >= limits_.max_jerk.size())
      {
        return std::nullopt;
      }
      const auto & brake_state = command_queue_state ? command_queue_state : actual_feedback;
      if (!brake_state || brake_state->position.size() != actual_feedback->position.size() ||
        brake_state->velocity.size() != actual_feedback->position.size() ||
        brake_state->acceleration.size() != actual_feedback->position.size() ||
        !brake_state->position.allFinite() || !brake_state->velocity.allFinite() ||
        !brake_state->acceleration.allFinite())
      {
        return std::nullopt;
      }
      detail::ScalarBrakingOptions options;
      options.period_sec = parameters_.control_period_sec;
      options.maximum_braking_steps = 1000U;
      const auto rollout = detail::rolloutScalarBraking(
        detail::ScalarBrakingState{brake_state->position[index],
          brake_state->velocity[index], brake_state->acceleration[index]},
        detail::ScalarBrakingLimits{limits_.max_velocity[index],
          limits_.max_acceleration[index], limits_.max_jerk[index]}, options, true);
      if (!rollout.commandAvailable() || rollout.points.empty()) {
        return std::nullopt;
      }
      PairYawJointResourceInput joint;
      joint.actual_position_rad = actual_feedback->position[index];
      joint.predicted_hold_position_rad = rollout.points.back().position;
      joint.lower_position_rad = limits_.lower_position[index];
      joint.upper_position_rad = limits_.upper_position[index];
      joint.position_margin_rad = limits_.position_margin[index];
      joint.held_direction = held_direction;
      const Eigen::Vector3d tracking_origin = tracking_pose.translation();
      const Eigen::Vector3d world_ray = face_position - tracking_origin;
      const Eigen::Vector3d normal_world = tracking_pose.linear().col(0);
      const Eigen::Vector3d base_angular = base_jacobian.bottomRows(3).col(index);
      const Eigen::Vector3d tracking_angular = tracking_jacobian.bottomRows(3).col(index);
      const Eigen::Vector3d tracking_linear = tracking_jacobian.topRows(3).col(index);
      const Eigen::Vector3d ray_rate_base = base_pose.linear().transpose() *
        (-tracking_linear - base_angular.cross(world_ray));
      const Eigen::Vector3d normal_rate_base = base_pose.linear().transpose() *
        ((tracking_angular - base_angular).cross(normal_world));
      const double bearing_rate =
        (ray.x() * ray_rate_base.y() - ray.y() * ray_rate_base.x()) / ray_squared;
      const double normal_rate =
        (normal.x() * normal_rate_base.y() - normal.y() * normal_rate_base.x()) /
        normal_squared;
      joint.yaw_error_derivative_per_rad = bearing_rate - normal_rate;
      if (!std::isfinite(joint.yaw_error_derivative_per_rad)) {
        return std::nullopt;
      }
      return joint;
    };

  const BaseJointDirection j1_held = base_handoff_policy_.heldPositionRad() ?
    base_handoff_policy_.heldDirection() : BaseJointDirection::kNone;
  const auto j1_input = make_joint_input(joint1_index, j1_held);
  const auto j4_input = make_joint_input(joint4_index, BaseJointDirection::kNone);
  if (!j1_input || !j4_input) {
    return invalid;
  }
  input.j1 = *j1_input;
  input.j4 = *j4_input;
  PairYawResourceAssessment assessment = assessPairYawResource(input);
  if (!assessment.valid || !assessment.demand_present ||
    !assessment.pair_bound_candidate)
  {
    return assessment;
  }

  // This is an admission-only one-second reserve probe, not a controller command.
  // Zero velocity/acceleration deliberately measures available position/orientation
  // reserve without treating a transient queue acceleration as pair exhaustion.
  assessment.probe_horizon_sec = kPairYawProbeHorizonSec;
  JointMotionState probe_state;
  probe_state.position = actual_feedback->position;
  probe_state.velocity = Eigen::VectorXd::Zero(probe_state.position.size());
  probe_state.acceleration = Eigen::VectorXd::Zero(probe_state.position.size());
  JointMotionLimits probe_limits = limits_;
  Eigen::VectorXd working_lower = limits_.lower_position + limits_.position_margin;
  Eigen::VectorXd working_upper = limits_.upper_position - limits_.position_margin;
  const double horizon = assessment.probe_horizon_sec;
  for (Eigen::Index index = 0; index < probe_state.position.size(); ++index) {
    // Probe the preferred working interval. The actual hardware limits remain
    // untouched and are still enforced by the production command QP.
    if (working_lower[index] >= working_upper[index]) {
      return assessment;
    }
  }

  Eigen::MatrixXd command_jacobian;
  if (!world_jacobian(command_link_, command_jacobian)) {
    return assessment;
  }
  const Eigen::Isometry3d command_pose = actual_state.getGlobalLinkTransform(command_link_);
  TrackingVelocityTaskConfig upright_config;
  upright_config.orientation_gain = parameters_.orientation_gain;
  upright_config.maximum_angular_reference_radps =
    parameters_.maximum_angular_reference_radps;
  const auto roll_task = makeUprightRollPathTask(
    command_pose, command_jacobian,
    Eigen::VectorXd::Zero(probe_state.position.size()), upright_config);
  if (!roll_task || roll_task->task.primary_matrix.rows() != 1 ||
    roll_task->task.primary_matrix.cols() != probe_state.position.size() ||
    !roll_task->task.primary_reference.allFinite())
  {
    return assessment;
  }

  Eigen::VectorXd yaw_gradient(probe_state.position.size());
  Eigen::VectorXd elevation_gradient(probe_state.position.size());
  const double ray_range = std::sqrt(ray_squared);
  const double ray_norm_squared = ray.squaredNorm();
  const double normal_range = std::sqrt(normal_squared);
  if (!std::isfinite(ray_range) || !std::isfinite(normal_range) ||
    ray_range <= 1.0e-6 || normal_range <= 1.0e-6 || ray_norm_squared <= 1.0e-9)
  {
    return assessment;
  }
  const Eigen::Vector3d tracking_origin = tracking_pose.translation();
  const Eigen::Vector3d world_ray = face_position - tracking_origin;
  const Eigen::Vector3d normal_world = tracking_pose.linear().col(0);
  for (Eigen::Index index = 0; index < probe_state.position.size(); ++index) {
    const Eigen::Vector3d base_angular = base_jacobian.bottomRows(3).col(index);
    const Eigen::Vector3d tracking_angular = tracking_jacobian.bottomRows(3).col(index);
    const Eigen::Vector3d tracking_linear = tracking_jacobian.topRows(3).col(index);
    const Eigen::Vector3d ray_rate_base = base_pose.linear().transpose() *
      (-tracking_linear - base_angular.cross(world_ray));
    const Eigen::Vector3d normal_rate_base = base_pose.linear().transpose() *
      ((tracking_angular - base_angular).cross(normal_world));
    const double ray_bearing_rate =
      (ray.x() * ray_rate_base.y() - ray.y() * ray_rate_base.x()) / ray_squared;
    const double normal_bearing_rate =
      (normal.x() * normal_rate_base.y() - normal.y() * normal_rate_base.x()) /
      normal_squared;
    const double ray_elevation_rate =
      (ray_range * ray_rate_base.z() - ray.z() *
      (ray.x() * ray_rate_base.x() + ray.y() * ray_rate_base.y()) / ray_range) /
      ray_norm_squared;
    const double normal_elevation_rate =
      (normal_range * normal_rate_base.z() - normal.z() *
      (normal.x() * normal_rate_base.x() + normal.y() * normal_rate_base.y()) /
      normal_range) / normal.squaredNorm();
    yaw_gradient[index] = ray_bearing_rate - normal_bearing_rate;
    elevation_gradient[index] = ray_elevation_rate - normal_elevation_rate;
  }
  if (!yaw_gradient.allFinite() || !elevation_gradient.allFinite() ||
    yaw_gradient.norm() <= 1.0e-9)
  {
    return assessment;
  }

  HierarchicalVelocityTask probe_task;
  probe_task.primary_matrix = yaw_gradient.transpose();
  probe_task.primary_reference = Eigen::VectorXd::Constant(
    1, -std::copysign(kPairYawProbeDisplacementRad / horizon, assessment.yaw_demand_radps));
  probe_task.primary_weights = Eigen::VectorXd::Ones(1);
  probe_task.secondary_matrix = Eigen::MatrixXd(0, probe_state.position.size());
  probe_task.secondary_reference = Eigen::VectorXd(0);
  probe_task.secondary_weights = Eigen::VectorXd(0);

  std::vector<LinearVelocityConstraint> upright_constraints;
  const auto add_hold_constraint = [&upright_constraints, horizon](
    const Eigen::VectorXd & gradient) {
      if (gradient.size() == 0 || !gradient.allFinite() || gradient.norm() <= 1.0e-9) {
        return false;
      }
      LinearVelocityConstraint row;
      row.coefficients = gradient;
      row.lower_bound = -kPairYawProbePoseHoldToleranceRad / horizon;
      row.upper_bound = kPairYawProbePoseHoldToleranceRad / horizon;
      upright_constraints.push_back(std::move(row));
      return true;
    };
  const Eigen::VectorXd roll_gradient =
    roll_task->task.primary_matrix.row(0).transpose();
  if (!add_hold_constraint(elevation_gradient) || !add_hold_constraint(roll_gradient)) {
    return assessment;
  }

  // Bound the one-second local linearization to 10 degrees per joint. This
  // trust region prevents large joint deltas from overstating useful yaw reserve.
  const auto travel_constraints = [&probe_state, horizon](
    const Eigen::VectorXd & lower, const Eigen::VectorXd & upper) {
      std::vector<LinearVelocityConstraint> constraints;
      constraints.reserve(static_cast<std::size_t>(2 * probe_state.position.size()));
      for (Eigen::Index index = 0; index < probe_state.position.size(); ++index) {
        const double q = probe_state.position[index];
        const double positive_travel = std::min(
          kPairYawProbePositionRelaxationRad, std::max(0.0, upper[index] - q));
        const double negative_travel = std::min(
          kPairYawProbePositionRelaxationRad, std::max(0.0, q - lower[index]));
        LinearVelocityConstraint positive;
        positive.coefficients = Eigen::VectorXd::Zero(probe_state.position.size());
        positive.coefficients[index] = 1.0;
        positive.upper_bound = positive_travel / horizon;
        constraints.push_back(std::move(positive));
        LinearVelocityConstraint negative;
        negative.coefficients = Eigen::VectorXd::Zero(probe_state.position.size());
        negative.coefficients[index] = -1.0;
        negative.upper_bound = negative_travel / horizon;
        constraints.push_back(std::move(negative));
      }
      return constraints;
    };
  const std::vector<LinearVelocityConstraint> pair_hold = [&]() {
      std::vector<LinearVelocityConstraint> rows;
      if (assessment.j1.held_for_required_direction) {
        const Eigen::Index index = joint1_index;
        LinearVelocityConstraint hold;
        hold.coefficients = Eigen::VectorXd::Zero(probe_state.position.size());
        hold.coefficients[index] = 1.0;
        hold.lower_bound = 0.0;
        hold.upper_bound = 0.0;
        rows.push_back(std::move(hold));
      }
      return rows;
    }();

  const auto merge_constraints = [](
    std::vector<LinearVelocityConstraint> first,
    const std::vector<LinearVelocityConstraint> & second) {
      first.insert(first.end(), second.begin(), second.end());
      return first;
    };
  const auto solve_probe = [&](const JointMotionLimits & local_limits,
    const std::vector<LinearVelocityConstraint> & constraints) {
      HierarchicalVelocityQpConfig config;
      config.period_sec = horizon;
      config.residual_command_latency_sec = 0.0;
      return HierarchicalVelocityQp(config).solve(
        probe_task, probe_state, local_limits, constraints);
    };
  const auto current_constraints = merge_constraints(
    merge_constraints(travel_constraints(working_lower, working_upper), upright_constraints),
    pair_hold);
  const QpResult actual_probe = solve_probe(probe_limits, current_constraints);
  assessment.actual_probe_solved = actual_probe.command_available();
  assessment.actual_probe_infeasible = actual_probe.status == QpStatus::kInfeasible;
  if (assessment.actual_probe_solved && actual_probe.achieved_primary_velocity.size() == 1) {
    assessment.actual_available_yaw_rad = std::max(0.0,
      -std::copysign(1.0, assessment.yaw_demand_radps) *
      actual_probe.achieved_primary_velocity[0] * horizon);
  }
  const auto pose_residual = [&](const QpResult & probe, const Eigen::VectorXd & row) {
      return probe.command_available() ?
             std::abs(row.dot(probe.joint_velocity)) * horizon :
             kPairYawProbePoseHoldToleranceRad * 2.0;
    };
  assessment.actual_roll_residual_rad = pose_residual(actual_probe, roll_gradient);
  assessment.actual_elevation_residual_rad = pose_residual(actual_probe, elevation_gradient);

  JointMotionLimits relaxed_limits = probe_limits;
  Eigen::VectorXd relaxed_lower = working_lower;
  Eigen::VectorXd relaxed_upper = working_upper;
  const auto relax_direction = [&](const PairYawJointResourceAssessment & joint,
    const Eigen::Index index) {
      if (!joint.directionally_bound) {
        return;
      }
      if (joint.required_direction == BaseJointDirection::kPositive) {
        relaxed_limits.upper_position[index] += kPairYawProbePositionRelaxationRad;
        relaxed_upper[index] += kPairYawProbePositionRelaxationRad;
      } else if (joint.required_direction == BaseJointDirection::kNegative) {
        relaxed_limits.lower_position[index] -= kPairYawProbePositionRelaxationRad;
        relaxed_lower[index] -= kPairYawProbePositionRelaxationRad;
      }
    };
  relax_direction(assessment.j1, joint1_index);
  relax_direction(assessment.j4, joint4_index);
  const auto relaxed_constraints = merge_constraints(
    merge_constraints(travel_constraints(relaxed_lower, relaxed_upper), upright_constraints),
    std::vector<LinearVelocityConstraint>{});
  const QpResult relaxed_probe = solve_probe(relaxed_limits, relaxed_constraints);
  assessment.relaxed_probe_solved = relaxed_probe.command_available();
  if (assessment.relaxed_probe_solved && relaxed_probe.achieved_primary_velocity.size() == 1) {
    assessment.pair_relaxed_available_yaw_rad = std::max(0.0,
      -std::copysign(1.0, assessment.yaw_demand_radps) *
      relaxed_probe.achieved_primary_velocity[0] * horizon);
  }
  assessment.relaxed_roll_residual_rad = pose_residual(relaxed_probe, roll_gradient);
  assessment.relaxed_elevation_residual_rad =
    pose_residual(relaxed_probe, elevation_gradient);
  assessment.probe_valid = assessment.actual_probe_solved;
  assessment.pair_bound_cause = pairYawProbeShowsPairBoundCause(assessment);
  assessment.exhausted = assessment.pair_bound_cause;
  return assessment;
}

std::optional<HierarchicalVelocityTask> MotionReference::faceReturnTask(
  const moveit::core::RobotState & state, const msg::TrackingTarget & target,
  const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
  const double time_sec, const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec, const bool fresh_actual_state)
{
  const bool use_fresh_actual = fresh_actual_state && actual_feedback &&
    actual_feedback->position.size() == group_->getVariableCount() &&
    actual_feedback->position.allFinite();
  FaceReturnPointingTask return_pointing;
  auto task = faceTask(state, target, target_motion, result, true,
    compact_return_full_face_active_ ? nullptr : &return_pointing);
  if (!task || !result.tracking) {
    return task;
  }

  if (!result.tracking->has_face_primary_roll || task->primary_matrix.rows() != 3 ||
    task->primary_reference.size() != 3 || task->primary_weights.size() != 3 ||
    task->secondary_matrix.rows() != 3 || task->secondary_reference.size() != 3 ||
    task->secondary_weights.size() != 3)
  {
    return std::nullopt;
  }

  result.actual_face_alignment = use_fresh_actual ?
    measureActualFaceAlignment(
    state, target, time_sec, actual_feedback, actual_feedback_time_sec) : std::nullopt;
  if (!compact_return_full_face_active_ && use_fresh_actual &&
    result.actual_face_alignment &&
    result.actual_face_alignment->pointing_error_rad <= kFaceTrackingErrorGuardRad &&
    std::abs(result.actual_face_alignment->upright_roll_error_rad) <=
    kFaceTrackingErrorGuardRad)
  {
    compact_return_full_face_active_ = true;
    diagnostics_.compact_return_full_face_active = true;
    const std::string details =
      "{\"actual_feedback_stamp_ns\":" +
      std::to_string(result.actual_face_alignment->stamp_ns) +
      ",\"actual_pointing_error_rad\":" +
      jsonNumber(result.actual_face_alignment->pointing_error_rad) +
      ",\"actual_upright_roll_error_rad\":" +
      jsonNumber(result.actual_face_alignment->upright_roll_error_rad) + "}";
    appendCompactEvent(
      "face_return_full_face_started", time_sec, "actual_face_aligned", details);
  }
  if (compact_return_full_face_active_) {
    return task;
  }

  const Eigen::RowVectorXd roll_row = task->primary_matrix.row(2);
  const double roll_reference = task->primary_reference[2];
  const double roll_weight = task->primary_weights[2];
  const Eigen::MatrixXd normal_matrix = return_pointing.matrix;
  const Eigen::VectorXd normal_reference = return_pointing.reference;
  const Eigen::VectorXd normal_weights = task->primary_weights.head(2);
  task->primary_matrix.resize(1, roll_row.size());
  task->primary_matrix.row(0) = roll_row;
  task->primary_reference = Eigen::VectorXd::Constant(1, roll_reference);
  task->primary_weights = Eigen::VectorXd::Constant(1, roll_weight);
  // During explicit reacquisition, do not spend the radius budget on TCP
  // translation. Preserve azimuth/elevation pointing as a lower-priority task.
  task->secondary_matrix = normal_matrix;
  task->secondary_reference = normal_reference;
  task->secondary_weights = normal_weights;
  return task;
}

void MotionReference::recordFaceCommand(
  const TrackingVelocityTaskResult & tracking, const QpResult & qp_result,
  const double sim_time_sec, const bool published,
  const std::optional<MotionReferenceResult::FaceAlignmentMeasurement> & actual_face_alignment)
{
  if (!published || !qp_result.command_available() ||
    qp_result.achieved_primary_velocity.size() == 0 ||
    !qp_result.achieved_primary_velocity.allFinite() || !std::isfinite(sim_time_sec))
  {
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    diagnostics_.compact_return_face_qualified = false;
    return;
  }
  if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
    compact_face_return_pending_ && compact_goal_reached_)
  {
    const bool full_face_command = compact_return_full_face_active_ &&
      qp_result.achieved_primary_velocity.size() == 3 && tracking.has_face_primary_roll;
    const bool face_is_aligned = actual_face_alignment &&
      actual_face_alignment->pointing_error_rad <= kFaceTrackingErrorGuardRad &&
      std::abs(actual_face_alignment->upright_roll_error_rad) <=
      kFaceTrackingErrorGuardRad;
    bool return_face_qualified = false;
    if (full_face_command && face_is_aligned) {
      FaceTrackingErrorForecast forecast;
      forecast.signed_error_rad = tracking.signed_face_primary_error_rad;
      forecast.desired_feedforward_radps = tracking.face_primary_feedforward_radps;
      forecast.achieved_primary_radps = qp_result.achieved_primary_velocity;
      forecast.horizon_sec = kFaceTrackingForecastHorizonSec;
      forecast.pointing_guard_rad = kFaceTrackingErrorGuardRad;
      forecast.roll_guard_rad = kFaceTrackingErrorGuardRad;
      const bool forecast_valid = forecast.signed_error_rad.allFinite() &&
        forecast.desired_feedforward_radps.allFinite() &&
        forecast.achieved_primary_radps.allFinite();
      return_face_qualified = forecast_valid && !trackingErrorWillCrossGuard(forecast);
    }
    diagnostics_.compact_return_face_qualified = return_face_qualified;
    if (!return_face_qualified) {
      face_return_dwell_sec_ = 0.0;
      last_face_return_evidence_time_sec_.reset();
      diagnostics_.compact_return_face_dwell_sec = 0.0;
      return;
    }
    if (last_face_return_evidence_time_sec_ &&
      sim_time_sec > *last_face_return_evidence_time_sec_)
    {
      const double dt = sim_time_sec - *last_face_return_evidence_time_sec_;
      face_return_dwell_sec_ = dt <= 0.05 ? face_return_dwell_sec_ + dt : 0.0;
    } else {
      face_return_dwell_sec_ = 0.0;
    }
    last_face_return_evidence_time_sec_ = sim_time_sec;
    diagnostics_.compact_return_face_dwell_sec = face_return_dwell_sec_;
    if (face_return_dwell_sec_ + 1.0e-9 < kFaceReturnDwellSec) {
      return;
    }
    std::string details{"{}"};
    if (actual_face_alignment) {
      details = "{\"actual_feedback_stamp_ns\":" +
        std::to_string(actual_face_alignment->stamp_ns) +
        ",\"actual_pointing_error_rad\":" +
        jsonNumber(actual_face_alignment->pointing_error_rad) +
        ",\"actual_upright_roll_error_rad\":" +
        jsonNumber(actual_face_alignment->upright_roll_error_rad) +
        ",\"command_signed_primary_error_rad\":" +
        jsonVector(tracking.signed_face_primary_error_rad) +
        ",\"command_feedforward_rad_s\":" +
        jsonVector(tracking.face_primary_feedforward_radps) +
        ",\"achieved_full_face_velocity_rad_s\":" +
        jsonVector(qp_result.achieved_primary_velocity) + "}";
    }
    appendCompactEvent("face_return_qualified", sim_time_sec, "full_face_stable", details);
    appendCompactEvent(
      "face_returned", sim_time_sec, "full_face_task_reacquired", details);
    appendCompactEvent("completed", sim_time_sec, diagnostics_.compact_reason, details);
    static_cast<void>(face_policy_.update(FaceTrackingPolicyInput{false, true}));
    compact_face_return_pending_ = false;
    compact_return_full_face_active_ = false;
    diagnostics_.compact_return_full_face_active = false;
    diagnostics_.compact_return_face_qualified = false;
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    compact_commit_initialized_ = false;
    compact_radius_floor_latched_ = false;
    compact_goal_.path_requirements.maximum_abs_screen_pitch_rad.reset();
    diagnostics_.compact_radius_floor_latched = false;
    diagnostics_.compact_effective_radius_cap_m =
      std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_actual_settled = false;
    compact_goal_reached_ = false;
    compact_goal_event_emitted_ = false;
    diagnostics_.compact_phase = "COMPLETE";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE";
    diagnostics_.compact_goal_reached = 0.0;
    active_scene_revision_.reset();
    return;
  }
  if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
    !compact_commit_initialized_)
  {
    // Preparation admission and cancellation use the fresh pair-yaw gate in update().
    return;
  }
}

void MotionReference::requestCompactManeuver(
  const moveit::core::RobotState & state, const double time_sec,
  const planning_scene::PlanningScene & scene,
  const std::uint64_t scene_revision, std::string reason,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (face_policy_.phase() != FaceTrackingPhase::kCompactManeuver ||
    compact_commit_initialized_ || pending_generation_ ||
    (compact_preparation_ && !compact_preparation_->exhausted))
  {
    return;
  }
  if (!pair_yaw_assessment_.valid || !pair_yaw_assessment_.exhausted) {
    planner_->cancel();
    pending_generation_.reset();
    compact_preparation_.reset();
    compact_plan_pending_ = false;
    face_policy_.clearForStop();
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    diagnostics_.compact_return_face_qualified = false;
    diagnostics_.compact_phase = "PREPARATION_CANCELLED";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_PAIR_RESOURCE_RESTORED";
    appendCompactEvent("preparation_cancelled", time_sec,
      "pair_yaw_resource_available", pairYawAssessmentJson(pair_yaw_assessment_));
    active_scene_revision_.reset();
    return;
  }
  if (!upright_roll_guard_armed_) {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_COMMIT_WAITING_FOR_ROLL_GUARD";
    return;
  }
  if (compact_goal_.profile_id.empty() || !compact_geometry_validator_) {
    ++diagnostics_.compact_generation;
    diagnostics_.compact_reason = "no_verified_compact_profile_for_robot_model";
    appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason);
    face_policy_.clearForStop();
    diagnostics_.compact_phase = "UNAVAILABLE";
    diagnostics_.state = "COMPACT_UNAVAILABLE";
    return;
  }
  if (!actual_feedback || !command_queue_state ||
    actual_feedback->position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    command_queue_state->position.size() != actual_feedback->position.size() ||
    !actual_feedback->position.allFinite() || !command_queue_state->position.allFinite())
  {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_PREPARING_WAITING_FOR_STATE";
    return;
  }

  Eigen::VectorXd exhausted_key = actual_feedback->position;
  if (compact_preparation_ && compact_preparation_->exhausted) {
    constexpr double kRetryJointChangeRad{0.05};
    const bool meaningful_state_change =
      compact_preparation_->exhausted_start_positions.size() != exhausted_key.size() ||
      (compact_preparation_->exhausted_start_positions - exhausted_key).
      lpNorm<Eigen::Infinity>() >= kRetryJointChangeRad;
    if (!meaningful_state_change &&
      compact_preparation_->exhausted_scene_revision == scene_revision)
    {
      face_policy_.clearForStop();
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
      return;
    }
    compact_preparation_.reset();
  }

  double entry_radius_cap = kCompactRadiusFloorM;
  double entry_envelope_max_abs_pitch_rad = 0.0;
  double entry_envelope_max_abs_roll_rad = 0.0;
  bool entry_envelope_valid = includeCompactEntryEnvelopePosition(
    state, groupPosition(state), &entry_radius_cap, &entry_envelope_max_abs_pitch_rad,
    &entry_envelope_max_abs_roll_rad);
  if (actual_feedback) {
    entry_envelope_valid = includeCompactEntryEnvelopePosition(
      state, actual_feedback->position, &entry_radius_cap,
      &entry_envelope_max_abs_pitch_rad, &entry_envelope_max_abs_roll_rad) &&
      entry_envelope_valid;
  }
  if (command_queue_state) {
    entry_envelope_valid = includeCompactEntryEnvelopePosition(
      state, command_queue_state->position, &entry_radius_cap,
      &entry_envelope_max_abs_pitch_rad, &entry_envelope_max_abs_roll_rad) &&
      entry_envelope_valid;
  }
  entry_envelope_valid = includeCompactEntryEnvelope(
    state, time_sec, active_publication, latest_publication, &entry_radius_cap,
    &entry_envelope_max_abs_pitch_rad, &entry_envelope_max_abs_roll_rad) &&
    entry_envelope_valid;

  if (!entry_envelope_valid ||
    entry_envelope_max_abs_roll_rad > kCompactRollLimitRad +
    kCompactPitchValidationToleranceRad)
  {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_PREPARING_WAITING_FOR_ROLL_SAFE_ENVELOPE";
    return;
  }

  std::vector<CompactGoal> candidates;
  if (select_default_compact_goal_) {
    const auto & names = group_->getVariableNames();
    const auto joint6 = std::find(names.begin(), names.end(), "joint6");
    if (joint6 == names.end() ||
      actual_feedback->position.size() != static_cast<Eigen::Index>(names.size()))
    {
      ++diagnostics_.compact_generation;
      diagnostics_.compact_reason = "compact_default_goal_unavailable";
      appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason);
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "UNAVAILABLE";
      diagnostics_.state = "COMPACT_UNAVAILABLE";
      return;
    }
    const auto joint6_index = static_cast<Eigen::Index>(
      std::distance(names.begin(), joint6));
    candidates = selectDefaultCompactGoals(
      state, scene, actual_feedback->position[joint6_index]);
    if (candidates.empty()) {
      ++diagnostics_.compact_generation;
      diagnostics_.compact_reason = "no_default_compact_goal_passed_pose_validation";
      CompactPreparation exhausted;
      exhausted.scene_revision = scene_revision;
      exhausted.exhausted_start_positions = actual_feedback->position;
      exhausted.exhausted_scene_revision = scene_revision;
      exhausted.exhausted = true;
      compact_preparation_ = std::move(exhausted);
      appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason,
        "{\"candidate_count\":0}");
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
      return;
    }
  } else {
    candidates.push_back(compact_goal_);
  }

  for (auto & candidate : candidates) {
    if (candidate.profile_id == "compact_turn_negative_wrist" ||
      candidate.profile_id == "compact_turn_positive_wrist")
    {
      candidate.path_requirements.staged_route = CompactStagedRouteRequirements{};
    }
  }

  CompactPreparation preparation;
  preparation.candidates = std::move(candidates);
  preparation.submitted_start_positions = actual_feedback->position;
  preparation.scene_revision = scene_revision;
  preparation.entry_radius_cap_m = entry_radius_cap;
  preparation.maximum_abs_screen_pitch_rad = std::max(
    kCompactMinimumPitchCapRad, entry_envelope_max_abs_pitch_rad);
  compact_preparation_ = std::move(preparation);
  face_return_dwell_sec_ = 0.0;
  last_face_return_evidence_time_sec_.reset();
  diagnostics_.compact_return_face_dwell_sec = 0.0;
  diagnostics_.compact_return_face_qualified = false;
  ++diagnostics_.compact_generation;
  diagnostics_.compact_reason = std::move(reason);
  diagnostics_.compact_phase = "PREPARING";
  diagnostics_.tracking_phase = "FACE";
  diagnostics_.state = "FACE_COMPACT_PREPARING";
  diagnostics_.last_plan_message = "compact path preparation started";
  appendCompactEvent("preparing", time_sec, diagnostics_.compact_reason,
    "{\"candidate_count\":" +
    std::to_string(compact_preparation_->candidates.size()) + "}");
}

void MotionReference::submitCompactPlan(
  const moveit::core::RobotState & state, const std::uint64_t scene_revision,
  const double time_sec)
{
  if (pending_generation_ || planner_->isBusy()) {
    return;
  }
  ++diagnostics_.plans_requested;
  CompactGoal request = compact_goal_;
  request.path_requirements.maximum_assembly_radius_m = effectiveCompactRadiusCap();
  pending_generation_ = planner_->submit(state, request, scene_revision);
  active_scene_revision_ = scene_revision;
  compact_plan_pending_ = true;
  diagnostics_.compact_phase = "PLANNING";
  if (diagnostics_.last_plan_message.empty() ||
    diagnostics_.last_plan_message == "not requested")
  {
    diagnostics_.last_plan_message = "compact path submitted";
  }
  (void)time_sec;
}

void MotionReference::submitCompactPreparationPlan(
  const moveit::core::RobotState & state, const double time_sec,
  const std::uint64_t scene_revision,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (!compact_preparation_ || compact_preparation_->exhausted || pending_generation_ ||
    planner_->isBusy() || !actual_feedback || !command_queue_state)
  {
    return;
  }
  const auto valid_motion_state = [this](const JointMotionState & motion) {
      const auto size = static_cast<Eigen::Index>(group_->getVariableCount());
      return motion.position.size() == size && motion.velocity.size() == size &&
             motion.acceleration.size() == size && motion.position.allFinite() &&
             motion.velocity.allFinite() && motion.acceleration.allFinite();
    };
  if (!valid_motion_state(*actual_feedback) || !valid_motion_state(*command_queue_state)) {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_WAITING_FOR_VALID_MOTION_STATE";
    return;
  }
  auto & preparation = *compact_preparation_;
  if (scene_revision != preparation.scene_revision ||
    preparation.candidate_index >= preparation.candidates.size())
  {
    return;
  }

  double entry_radius_cap = kCompactRadiusFloorM;
  double maximum_abs_pitch = 0.0;
  double maximum_abs_roll = 0.0;
  bool envelope_valid = includeCompactEntryEnvelopePosition(
    state, groupPosition(state), &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll);
  envelope_valid = includeCompactEntryEnvelopePosition(
    state, actual_feedback->position, &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll) && envelope_valid;
  envelope_valid = includeCompactEntryEnvelopePosition(
    state, command_queue_state->position, &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll) && envelope_valid;
  envelope_valid = includeCompactEntryEnvelope(
    state, time_sec, active_publication, latest_publication, &entry_radius_cap,
    &maximum_abs_pitch, &maximum_abs_roll) && envelope_valid;
  if (!envelope_valid || maximum_abs_roll > kCompactRollLimitRad +
    kCompactPitchValidationToleranceRad)
  {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_ENTRY_UNSAFE";
    return;
  }

  CompactGoal request = preparation.candidates[preparation.candidate_index];
  preparation.submitted_start_positions = actual_feedback->position;
  moveit::core::RobotState goal_state(state);
  goal_state.setJointGroupPositions(group_, request.bounded_positions);
  goal_state.update(true);
  const auto goal_upright = evaluateCompactUprightPose(
    goal_state.getGlobalLinkTransform(command_link_).linear());
  if (!goal_upright || !std::isfinite(goal_upright->signed_pitch_rad)) {
    ++diagnostics_.plans_rejected;
    const auto & rejected_candidate = preparation.candidates[preparation.candidate_index];
    appendCompactEvent("candidate_rejected", time_sec,
      "compact candidate goal pitch is unavailable",
      "{\"candidate_index\":" + std::to_string(preparation.candidate_index) +
      ",\"profile_id\":\"" + rejected_candidate.profile_id +
      "\",\"goal_joints_rad\":" + jsonVector(rejected_candidate.bounded_positions) +
      "}");
    ++preparation.candidate_index;
    if (preparation.candidate_index >= preparation.candidates.size()) {
      preparation.exhausted = true;
      preparation.exhausted_start_positions = preparation.submitted_start_positions;
      preparation.exhausted_scene_revision = scene_revision;
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
    }
    return;
  }
  const double pitch_cap = std::max({
      kCompactMinimumPitchCapRad, maximum_abs_pitch,
      std::abs(goal_upright->signed_pitch_rad)});
  request.path_requirements.maximum_assembly_radius_m = entry_radius_cap;
  request.path_requirements.maximum_abs_screen_pitch_rad = pitch_cap;
  request.path_requirements.base_link_name = base_link_->getName();
  request.path_requirements.screen_link_name = command_link_->getName();

  moveit::core::RobotState start(state);
  start.setJointGroupPositions(group_, actual_feedback->position);
  start.update(true);
  preparation.entry_radius_cap_m = entry_radius_cap;
  preparation.maximum_abs_screen_pitch_rad = pitch_cap;
  preparation.scene_revision = scene_revision;
  ++diagnostics_.plans_requested;
  pending_generation_ = planner_->submit(start, request, scene_revision);
  preparation.generation = *pending_generation_;
  compact_plan_pending_ = true;
  diagnostics_.compact_phase = "PREPARING";
  diagnostics_.tracking_phase = "FACE";
  diagnostics_.state = "FACE_COMPACT_PREPARING";
  diagnostics_.last_plan_message = "compact candidate path submitted: " + request.profile_id;
}

bool MotionReference::compactSettled(
  const JointMotionState & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state,
  const Eigen::VectorXd & goal) const
{
  // Actual feedback provides measured q/v only; its acceleration is zero-filled.
  // The command tail is the controller's source for the jerk-limited v/a state.
  return compactGoalPositionReached(actual_feedback, goal) &&
         motionSettled(actual_feedback, command_queue_state);
}

void MotionReference::recordCompactSettledEvent(
  const double time_sec, const bool settled_now,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state)
{
  if (settled_now && !diagnostics_.compact_actual_settled && actual_feedback &&
    actual_feedback_time_sec && command_queue_state)
  {
    const double actual_goal_error =
      (actual_feedback->position - compact_goal_.bounded_positions).
      lpNorm<Eigen::Infinity>();
    const double actual_speed = actual_feedback->velocity.lpNorm<Eigen::Infinity>();
    const double queue_goal_error =
      (command_queue_state->position - compact_goal_.bounded_positions).
      lpNorm<Eigen::Infinity>();
    const double queue_speed = command_queue_state->velocity.lpNorm<Eigen::Infinity>();
    const double queue_acceleration =
      command_queue_state->acceleration.lpNorm<Eigen::Infinity>();
    const std::string details =
      "{\"actual_feedback_stamp_ns\":" +
      std::to_string(toNanoseconds(*actual_feedback_time_sec)) +
      ",\"actual_goal_error_max_rad\":" + jsonNumber(actual_goal_error) +
      ",\"actual_speed_max_rad_s\":" + jsonNumber(actual_speed) +
      ",\"actual_position_rad\":" + jsonVector(actual_feedback->position) +
      ",\"actual_velocity_rad_s\":" + jsonVector(actual_feedback->velocity) +
      ",\"command_queue_goal_error_max_rad\":" + jsonNumber(queue_goal_error) +
      ",\"command_queue_speed_max_rad_s\":" + jsonNumber(queue_speed) +
      ",\"command_queue_acceleration_max_rad_s2\":" +
      jsonNumber(queue_acceleration) +
      ",\"command_queue_position_rad\":" + jsonVector(command_queue_state->position) +
      ",\"command_queue_velocity_rad_s\":" + jsonVector(command_queue_state->velocity) +
      ",\"command_queue_acceleration_rad_s2\":" +
      jsonVector(command_queue_state->acceleration) + "}";
    appendCompactEvent("compact_goal_settled", time_sec, "actual_and_queue_stopped", details);
  }
  diagnostics_.compact_actual_settled = settled_now;
}

bool MotionReference::appendCompactPathConstraints(
  const moveit::core::RobotState & state, const JointMotionState & motion_state,
  MotionReferenceResult & result) const
{
  if (!compact_geometry_validator_ || !group_ || !motion_state.position.allFinite() ||
    motion_state.position.size() != static_cast<Eigen::Index>(group_->getVariableCount()))
  {
    return false;
  }
  moveit::core::RobotState task_state(state);
  task_state.setJointGroupPositions(group_, motion_state.position);
  task_state.update(true);
  const auto support_set = compact_geometry_validator_->perLinkRadiusSupports(task_state, *group_);
  const double radius_cap = effectiveCompactRadiusCap();
  if (!support_set || !std::isfinite(support_set->maximum_radius_m) ||
    !std::isfinite(radius_cap) ||
    support_set->maximum_radius_m > radius_cap + 1.0e-6)
  {
    return false;
  }

  std::vector<LinearVelocityConstraint> constraints;
  constraints.reserve(support_set->supports.size());
  for (const auto & support : support_set->supports) {
    if (!std::isfinite(support.radius_m) ||
      support.outward_gradient.size() != motion_state.position.size() ||
      !support.outward_gradient.allFinite() || support.radius_m > radius_cap + 1.0e-6)
    {
      return false;
    }
    const double gradient_norm = support.outward_gradient.norm();
    if (!std::isfinite(gradient_norm)) {
      return false;
    }
    if (gradient_norm <= 1.0e-10) {
      if (support.radius_m > radius_cap) {
        return false;
      }
      continue;
    }

    // The entry cap can be initialized from this exact pose. Reserve only
    // consumes positive slack; a virtual negative gap must not force motion
    // inward when the measured assembly is already within its committed cap.
    const double remaining_radius_m = std::max(
      0.0, radius_cap - support.radius_m -
      collision_validator_.config().numerical_distance_reserve_m);
    const auto row = makeOutwardStoppingConstraint(
      support.outward_gradient, remaining_radius_m, motion_state, limits_,
      parameters_.control_period_sec, kCompactQueueTailResidualLatencySec);
    if (!row) {
      return false;
    }
    constraints.push_back(*row);
  }
  result.compact_radius_constraints = std::move(constraints);
  return appendUprightConstraints(
    state, motion_state, result, !compact_face_return_pending_);
}

bool MotionReference::appendUprightConstraints(
  const moveit::core::RobotState & state, const JointMotionState & motion_state,
  MotionReferenceResult & result, const bool constrain_pitch) const
{
  if (!group_ || !command_link_ || !motion_state.position.allFinite() ||
    motion_state.position.size() != static_cast<Eigen::Index>(group_->getVariableCount()))
  {
    return false;
  }
  moveit::core::RobotState task_state(state);
  task_state.setJointGroupPositions(group_, motion_state.position);
  task_state.update(true);

  const auto pitch_cap = compact_goal_.path_requirements.maximum_abs_screen_pitch_rad;
  if (constrain_pitch && (!pitch_cap || !std::isfinite(*pitch_cap))) {
    return false;
  }
  const auto upright_pose = evaluateCompactUprightPose(
    task_state.getGlobalLinkTransform(command_link_).linear());
  Eigen::MatrixXd jacobian;
  if (!upright_pose ||
    (constrain_pitch && !std::isfinite(upright_pose->signed_pitch_rad)) ||
    !task_state.getJacobian(
      group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false))
  {
    return false;
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = task_state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig roll_config;
  roll_config.orientation_gain = parameters_.orientation_gain;
  roll_config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const auto roll_task = makeUprightRollPathTask(
    task_state.getGlobalLinkTransform(command_link_), jacobian,
    Eigen::VectorXd::Zero(motion_state.position.size()), roll_config);
  if (!roll_task || roll_task->task.primary_matrix.rows() != 1 ||
    roll_task->task.primary_matrix.cols() != motion_state.position.size() ||
    !roll_task->task.primary_matrix.allFinite() ||
    !std::isfinite(roll_task->roll_error_rad))
  {
    return false;
  }
  const double signed_roll_error = roll_task->roll_error_rad;
  if (std::abs(signed_roll_error) >
    kCompactRollLimitRad + kCompactPitchValidationToleranceRad)
  {
    return false;
  }
  const Eigen::VectorXd signed_roll_gradient =
    -roll_task->task.primary_matrix.row(0).transpose();
  if (!signed_roll_gradient.allFinite()) {
    return false;
  }
  std::vector<LinearVelocityConstraint> roll_constraints;
  roll_constraints.reserve(2U);
  for (const double direction : {1.0, -1.0}) {
    const double signed_roll = direction * signed_roll_error;
    const Eigen::VectorXd outward_gradient = direction * signed_roll_gradient;
    if (signed_roll > kCompactRollLimitRad + kCompactPitchValidationToleranceRad) {
      return false;
    }
    if (outward_gradient.norm() <= 1.0e-10) {
      if (signed_roll > kCompactRollLimitRad) {
        return false;
      }
      continue;
    }
    const double remaining_roll_rad = std::max(0.0, kCompactRollLimitRad - signed_roll);
    const auto row = makeOutwardStoppingConstraint(
      outward_gradient, remaining_roll_rad, motion_state, limits_,
      parameters_.control_period_sec, kCompactQueueTailResidualLatencySec);
    if (!row) {
      return false;
    }
    roll_constraints.push_back(*row);
  }
  result.compact_roll_constraints = std::move(roll_constraints);

  if (!constrain_pitch) {
    return true;
  }

  const auto pitch_gradient = signedScreenPitchGradient(
    upright_pose->screen_normal, jacobian.bottomRows(3));
  if (!pitch_gradient || pitch_gradient->size() != motion_state.position.size() ||
    !pitch_gradient->allFinite())
  {
    return false;
  }

  const Eigen::VectorXd signed_gradient = pitch_gradient->transpose();
  const double gradient_norm = signed_gradient.norm();
  if (!std::isfinite(gradient_norm)) {
    return false;
  }
  if (gradient_norm <= 1.0e-10) {
    return std::abs(upright_pose->signed_pitch_rad) <=
           *pitch_cap + kCompactPitchValidationToleranceRad;
  }

  std::vector<LinearVelocityConstraint> pitch_constraints;
  pitch_constraints.reserve(2U);
  for (const double direction : {1.0, -1.0}) {
    const double signed_pitch = direction * upright_pose->signed_pitch_rad;
    if (signed_pitch > *pitch_cap + kCompactPitchValidationToleranceRad) {
      return false;
    }
    const Eigen::VectorXd outward_gradient = direction * signed_gradient;
    const double remaining_pitch_rad = std::max(0.0, *pitch_cap - signed_pitch);
    const auto row = makeOutwardStoppingConstraint(
      outward_gradient, remaining_pitch_rad, motion_state, limits_,
      parameters_.control_period_sec, kCompactQueueTailResidualLatencySec);
    if (!row) {
      return false;
    }
    pitch_constraints.push_back(*row);
  }
  result.compact_pitch_constraints = std::move(pitch_constraints);
  return true;
}

bool MotionReference::appendFaceRollGuardConstraints(
  const moveit::core::RobotState & state,
  const std::optional<JointMotionState> & command_queue_state,
  MotionReferenceResult & result) const
{
  if (!upright_roll_guard_armed_) {
    return true;
  }
  if (!command_queue_state || !appendUprightConstraints(
      state, *command_queue_state, result, false))
  {
    result.compact_roll_constraints.clear();
    result.motion_reference_support_unavailable = true;
    return false;
  }
  return true;
}

bool MotionReference::motionSettled(
  const JointMotionState & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state) const
{
  if (!command_queue_state || actual_feedback.position.size() == 0 ||
    actual_feedback.velocity.size() != actual_feedback.position.size() ||
    command_queue_state->position.size() != actual_feedback.position.size() ||
    command_queue_state->velocity.size() != actual_feedback.position.size() ||
    command_queue_state->acceleration.size() != actual_feedback.position.size() ||
    !actual_feedback.position.allFinite() || !actual_feedback.velocity.allFinite() ||
    !command_queue_state->position.allFinite() || !command_queue_state->velocity.allFinite() ||
    !command_queue_state->acceleration.allFinite())
  {
    return false;
  }
  return actual_feedback.velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
         command_queue_state->velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
         command_queue_state->acceleration.lpNorm<Eigen::Infinity>() <
         kSettledAccelerationRadps2;
}

bool MotionReference::compactGoalPositionReached(
  const JointMotionState & feedback, const Eigen::VectorXd & goal) const
{
  if (feedback.position.size() != goal.size() || !feedback.position.allFinite() ||
    (feedback.position - goal).lpNorm<Eigen::Infinity>() > kCompactGoalToleranceRad ||
    feedback.position.size() != limits_.lower_position.size())
  {
    return false;
  }
  // The committed profile is selected to restore both bounded winding axes.
  // Never wrap their actual coordinates to manufacture reserve.
  for (const char * name : {"joint1", "joint4"}) {
    const auto found = std::find(
      group_->getVariableNames().begin(), group_->getVariableNames().end(), name);
    if (found == group_->getVariableNames().end()) {
      return false;
    }
    const auto index = std::distance(group_->getVariableNames().begin(), found);
    const auto reserve = measureBoundedJointReserve(
      feedback.position[index], limits_.lower_position[index], limits_.upper_position[index]);
    if (!reserve.inside_bounds || reserve.nearest_bound_margin_rad < kCompactReserveRad) {
      return false;
    }
  }
  return true;
}

std::optional<HierarchicalVelocityTask> MotionReference::pathTask(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  const Eigen::VectorXd & waypoint, const double gain, const double speed,
  const bool upright_roll_primary, double * measured_roll_error)
{
  const auto joint_task = makeJointPathVelocityTask(position, waypoint, gain, speed);
  if (!joint_task) {
    return std::nullopt;
  }
  if (!upright_roll_primary) {
    return joint_task;
  }
  Eigen::MatrixXd jacobian;
  if (!state.getJacobian(group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false)) {
    return std::nullopt;
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig config;
  config.orientation_gain = parameters_.orientation_gain;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const auto roll_task = makeUprightRollPathTask(
    state.getGlobalLinkTransform(command_link_), jacobian,
    joint_task->primary_reference, config);
  if (!roll_task) {
    return std::nullopt;
  }
  if (measured_roll_error) {
    *measured_roll_error = roll_task->roll_error_rad;
  }
  return roll_task->task;
}

std::optional<HierarchicalVelocityTask> MotionReference::compactTask(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  const double time_sec)
{
  if (follower_.active()) {
    const auto path = follower_.update(position, time_sec);
    diagnostics_.path_progress_rad = path.progress_rad;
    diagnostics_.path_distance_rad = path.distance_from_path_rad;
    if (path.status == JointPathStatus::kTracking) {
      diagnostics_.compact_phase = "PATH";
      if (!compact_path_started_event_emitted_) {
        appendCompactEvent("path_started", time_sec, diagnostics_.compact_reason);
        compact_path_started_event_emitted_ = true;
      }
      return pathTask(
        state, position, path.reference_position,
        compact_path_settings_.gain,
        compact_path_settings_.maximum_velocity_radps, false);
    }
    if (path.status == JointPathStatus::kArrived) {
      ++diagnostics_.paths_completed;
      compact_path_endpoint_reached_ = true;
      diagnostics_.compact_phase = "SETTLING";
      follower_.reset();
    } else {
      ++diagnostics_.paths_abandoned;
      follower_.reset();
      compact_path_adopted_ = false;
      compact_plan_pending_ = false;
      diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_roll_error_rad =
        std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.compact_phase = "REPLANNING";
      diagnostics_.last_plan_message = "compact follower lost its checked path";
      return std::nullopt;
    }
  }
  if (!compact_path_endpoint_reached_) {
    if (compact_plan_pending_) {
      return std::nullopt;
    }
    return std::nullopt;
  }
  if ((position - compact_goal_.bounded_positions).lpNorm<Eigen::Infinity>() <=
    kCompactGoalToleranceRad)
  {
    return std::nullopt;
  }
  return pathTask(
    state, position, compact_goal_.bounded_positions,
    compact_goal_settings_.gain,
    compact_goal_settings_.maximum_velocity_radps, false);
}

void MotionReference::consumeCompactPreparationResult(
  const BackgroundPathResult & plan, const moveit::core::RobotState & state,
  const double time_sec, const planning_scene::PlanningScene & scene,
  const std::uint64_t scene_revision,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (!compact_preparation_ || !pending_generation_ ||
    plan.generation != *pending_generation_ ||
    plan.generation != compact_preparation_->generation)
  {
    return;
  }
  auto & preparation = *compact_preparation_;
  pending_generation_.reset();
  compact_plan_pending_ = false;
  diagnostics_.last_plan_wall_ms = 1000.0 * plan.elapsed_time_s;
  diagnostics_.last_plan_message = plan.message;

  if (plan.scene_revision != scene_revision ||
    preparation.scene_revision != scene_revision)
  {
    // Scene tokens are opaque revisions. A result validated against an older
    // scene cannot be advanced or adopted in this one.
    planner_->cancel();
    compact_preparation_.reset();
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_SCENE_CHANGED";
    diagnostics_.last_plan_message = "discarded compact result from stale scene";
    return;
  }
  if (preparation.candidate_index >= preparation.candidates.size()) {
    return;
  }

  const auto reject_candidate = [this, &preparation,
      scene_revision, time_sec, plan_generation = plan.generation](
    const std::string & reason, const std::string & additional_details = {}) {
      ++diagnostics_.plans_rejected;
      diagnostics_.last_plan_message = reason;
      const auto & rejected_candidate = preparation.candidates[preparation.candidate_index];
      appendCompactEvent("candidate_rejected", time_sec, reason,
        "{\"candidate_index\":" + std::to_string(preparation.candidate_index) +
        ",\"profile_id\":\"" + rejected_candidate.profile_id +
        "\",\"goal_joints_rad\":" + jsonVector(rejected_candidate.bounded_positions) +
        ",\"plan_generation\":" + std::to_string(plan_generation) +
        (additional_details.empty() ? std::string{} : "," + additional_details) + "}");
      ++preparation.candidate_index;
      if (preparation.candidate_index < preparation.candidates.size()) {
        diagnostics_.compact_phase = "PREPARING";
        diagnostics_.tracking_phase = "FACE";
        diagnostics_.state = "FACE_COMPACT_PREPARING";
        return;
      }
      preparation.exhausted = true;
      preparation.exhausted_start_positions = preparation.submitted_start_positions;
      preparation.exhausted_scene_revision = scene_revision;
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
    };

  const CompactGoal candidate = preparation.candidates[preparation.candidate_index];
  constexpr double kGoalMatchTolerance{1.0e-6};
  const bool result_matches_request =
    plan.profile_id == candidate.profile_id &&
    plan.goal_positions.size() == candidate.bounded_positions.size() &&
    plan.goal_positions.allFinite() &&
    (plan.goal_positions - candidate.bounded_positions).
    lpNorm<Eigen::Infinity>() <= kGoalMatchTolerance &&
    plan.start_positions.size() == preparation.submitted_start_positions.size() &&
    plan.start_positions.allFinite() &&
    (plan.start_positions - preparation.submitted_start_positions).
    lpNorm<Eigen::Infinity>() <= kGoalMatchTolerance;
  if (!plan.success || !result_matches_request || !plan.validation_metrics) {
    reject_candidate("compact candidate plan failed or did not match its request: " +
      plan.message);
    return;
  }
  if (!actual_feedback || !actual_feedback_time_sec || !command_queue_state ||
    !std::isfinite(*actual_feedback_time_sec) || time_sec < *actual_feedback_time_sec ||
    time_sec - *actual_feedback_time_sec > parameters_.state_feedback_timeout_sec ||
    actual_feedback->position.size() != plan.start_positions.size() ||
    actual_feedback->velocity.size() != plan.start_positions.size() ||
    actual_feedback->acceleration.size() != plan.start_positions.size() ||
    command_queue_state->position.size() != plan.start_positions.size() ||
    !actual_feedback->position.allFinite() || !actual_feedback->velocity.allFinite() ||
    !actual_feedback->acceleration.allFinite() ||
    !command_queue_state->position.allFinite() ||
    command_queue_state->velocity.size() != plan.start_positions.size() ||
    command_queue_state->acceleration.size() != plan.start_positions.size() ||
    !command_queue_state->velocity.allFinite() || !command_queue_state->acceleration.allFinite())
  {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_WAITING_FOR_FRESH_STATE";
    diagnostics_.last_plan_message =
      "compact candidate will be resubmitted after fresh actual and queue state";
    return;
  }
  if (plan.path.size() < 2U ||
    std::any_of(plan.path.begin(), plan.path.end(), [&plan](const Eigen::VectorXd & point) {
      return point.size() != plan.start_positions.size() || !point.allFinite();
    }) ||
    (plan.path.front() - plan.start_positions).lpNorm<Eigen::Infinity>() >
    kGoalMatchTolerance ||
    (plan.path.back() - candidate.bounded_positions).lpNorm<Eigen::Infinity>() >
    planner_goal_tolerance_rad_ + 1.0e-6)
  {
    reject_candidate("compact candidate returned an incomplete path");
    return;
  }

  double entry_radius_cap = kCompactRadiusFloorM;
  double entry_max_pitch = 0.0;
  double entry_max_roll = 0.0;
  bool entry_valid = includeCompactEntryEnvelopePosition(
    state, groupPosition(state), &entry_radius_cap, &entry_max_pitch, &entry_max_roll);
  entry_valid = includeCompactEntryEnvelopePosition(
    state, actual_feedback->position, &entry_radius_cap, &entry_max_pitch,
    &entry_max_roll) && entry_valid;
  entry_valid = includeCompactEntryEnvelopePosition(
    state, command_queue_state->position, &entry_radius_cap, &entry_max_pitch,
    &entry_max_roll) && entry_valid;
  entry_valid = includeCompactEntryEnvelope(
    state, time_sec, active_publication, latest_publication, &entry_radius_cap,
    &entry_max_pitch, &entry_max_roll) && entry_valid;
  if (!entry_valid || entry_max_roll > kCompactRollLimitRad +
    kCompactPitchValidationToleranceRad)
  {
    reject_candidate("compact candidate entry envelope is invalid or outside roll limit");
    return;
  }

  moveit::core::RobotState goal_state(state);
  goal_state.setJointGroupPositions(group_, candidate.bounded_positions);
  goal_state.update(true);
  const auto goal_upright = evaluateCompactUprightPose(
    goal_state.getGlobalLinkTransform(command_link_).linear());
  if (!goal_upright || !std::isfinite(goal_upright->signed_pitch_rad)) {
    reject_candidate("compact candidate goal pitch is unavailable");
    return;
  }
  const double pitch_cap = std::max({
      kCompactMinimumPitchCapRad, entry_max_pitch,
      std::abs(goal_upright->signed_pitch_rad)});
  const auto & metrics = *plan.validation_metrics;
  const std::size_t accepted_candidate_index = preparation.candidate_index;
  const auto metrics_and_caps_json = [&]() {
      const auto json_finite_number = [](const double value) {
          return std::isfinite(value) ? jsonNumber(value) : std::string{"null"};
        };
      return std::string{"\"actual_metrics\":{"} +
             "\"maximum_assembly_radius_m\":" +
             json_finite_number(metrics.maximum_assembly_radius_m) +
             ",\"maximum_abs_screen_pitch_rad\":" +
             json_finite_number(metrics.maximum_abs_screen_pitch_rad) +
             ",\"maximum_roll_error_rad\":" +
             json_finite_number(metrics.maximum_roll_error_rad) + "}," +
             "\"submitted_caps\":{" +
             "\"radius_m\":" + json_finite_number(preparation.entry_radius_cap_m) +
             ",\"pitch_rad\":" +
             json_finite_number(preparation.maximum_abs_screen_pitch_rad) +
             ",\"roll_rad\":" + json_finite_number(kCompactRollLimitRad) + "}," +
             "\"current_caps\":{" +
             "\"radius_m\":" + json_finite_number(entry_radius_cap) +
             ",\"pitch_rad\":" + json_finite_number(pitch_cap) +
             ",\"roll_rad\":" + json_finite_number(kCompactRollLimitRad) + "}," +
             "\"compact_path_sample_count\":" +
             std::to_string(plan.compact_path_samples.size()) + "," +
             (plan.compact_goal_metrics ?
             std::string{"\"exact_goal_metrics\":{"} +
             "\"maximum_assembly_radius_m\":" + json_finite_number(
               plan.compact_goal_metrics->maximum_assembly_radius_m) +
             ",\"maximum_abs_screen_pitch_rad\":" + json_finite_number(
               plan.compact_goal_metrics->maximum_abs_screen_pitch_rad) +
             ",\"maximum_roll_error_rad\":" + json_finite_number(
               plan.compact_goal_metrics->maximum_roll_error_rad) + "}," :
             "\"exact_goal_metrics\":null,") +
             (plan.compact_path_samples.empty() ?
             "\"first_sample_suffix_metrics\":null" :
             std::string{"\"first_sample_suffix_metrics\":{"} +
             "\"maximum_assembly_radius_m\":" + json_finite_number(
               plan.compact_path_samples.front().suffix_maximum_assembly_radius_m) +
             ",\"maximum_abs_screen_pitch_rad\":" + json_finite_number(
               plan.compact_path_samples.front().suffix_maximum_abs_screen_pitch_rad) +
             ",\"maximum_roll_error_rad\":" + json_finite_number(
               plan.compact_path_samples.front().suffix_maximum_roll_error_rad) + "}");
    };
  constexpr double kMaximumAttachDistanceRad{0.20};
  const auto attachment = selectCompactAttachment(
    plan, actual_feedback->position, command_queue_state->position, entry_radius_cap,
    pitch_cap, kCompactRollLimitRad);
  if (!attachment.valid) {
    const std::string reason = plan.compact_path_samples.empty() ?
      "compact candidate path metrics exceed current entry caps" :
      "compact candidate has no valid certified suffix attachment";
    reject_candidate(reason + "; " + metrics_and_caps_json(), metrics_and_caps_json());
    return;
  }
  const Eigen::VectorXd & attachment_position = attachment.position;
  const double actual_to_attachment =
    (actual_feedback->position - attachment_position).norm();
  const double queue_to_attachment =
    (command_queue_state->position - attachment_position).norm();
  if (!std::isfinite(actual_to_attachment) || !std::isfinite(queue_to_attachment) ||
    actual_to_attachment > kMaximumAttachDistanceRad + 1.0e-9 ||
    queue_to_attachment > kMaximumAttachDistanceRad + 1.0e-9)
  {
    reject_candidate("compact candidate attachment moved outside the checked window");
    return;
  }

  double bridge_radius = 0.0;
  double bridge_max_pitch = 0.0;
  double bridge_max_roll = 0.0;
  const auto validate_bridge = [this, &state, &bridge_radius,
      &bridge_max_pitch, &bridge_max_roll, entry_radius_cap, pitch_cap](
    const Eigen::VectorXd & from, const Eigen::VectorXd & to) {
      if (from.size() != to.size() || from.size() != limits_.lower_position.size() ||
        !from.allFinite() || !to.allFinite())
      {
        return false;
      }
      constexpr double kMaximumJointStepRad{0.02};
      const double span = (to - from).lpNorm<Eigen::Infinity>();
      const auto steps = static_cast<std::size_t>(std::max(
        1.0, std::ceil(span / kMaximumJointStepRad)));
      if (steps > 4096U) {
        return false;
      }
      for (std::size_t step = 0U; step <= steps; ++step) {
        const double fraction = static_cast<double>(step) / static_cast<double>(steps);
        const Eigen::VectorXd position = from + fraction * (to - from);
        if ((position.array() < limits_.lower_position.array()).any() ||
          (position.array() > limits_.upper_position.array()).any())
        {
          return false;
        }
        moveit::core::RobotState bridge_state(state);
        bridge_state.setJointGroupPositions(group_, position);
        bridge_state.update(true);
        const double radius = compact_geometry_validator_->maximumAssemblyRadius(bridge_state);
        const auto pose = evaluateCompactUprightPose(
          bridge_state.getGlobalLinkTransform(command_link_).linear());
        if (!std::isfinite(radius) || radius > entry_radius_cap + 1.0e-6 || !pose ||
          !std::isfinite(pose->signed_pitch_rad) ||
          std::abs(pose->signed_pitch_rad) > pitch_cap +
          kCompactPitchValidationToleranceRad ||
          !std::isfinite(pose->signed_roll_error_rad) ||
          std::abs(pose->signed_roll_error_rad) > kCompactRollLimitRad +
          kCompactPitchValidationToleranceRad)
        {
          return false;
        }
        bridge_radius = std::max(bridge_radius, radius);
        bridge_max_pitch = std::max(bridge_max_pitch, std::abs(pose->signed_pitch_rad));
        bridge_max_roll = std::max(bridge_max_roll, std::abs(pose->signed_roll_error_rad));
      }
      return true;
    };
  const JointMotionState stationary_start{
    attachment_position, Eigen::VectorXd::Zero(attachment_position.size()),
    Eigen::VectorXd::Zero(attachment_position.size())};
  moveit::core::RobotState actual_connector_state(state);
  actual_connector_state.setJointGroupPositions(group_, actual_feedback->position);
  actual_connector_state.update(true);
  moveit::core::RobotState queue_connector_state(state);
  queue_connector_state.setJointGroupPositions(group_, command_queue_state->position);
  queue_connector_state.update(true);
  const auto actual_connector = collision_validator_.validateSegment(
    scene, actual_connector_state, *group_, *actual_feedback, stationary_start,
    collision_validator_.config().segment_validation_substeps);
  const auto queue_connector = collision_validator_.validateSegment(
    scene, queue_connector_state, *group_, *command_queue_state, stationary_start,
    collision_validator_.config().segment_validation_substeps);
  if (!actual_connector.input_valid || actual_connector.unsafe ||
    !queue_connector.input_valid || queue_connector.unsafe ||
    !validate_bridge(actual_feedback->position, attachment_position) ||
    !validate_bridge(command_queue_state->position, attachment_position))
  {
    reject_candidate(
        "compact candidate attachment bridge failed bounds, geometry, or collision validation");
    return;
  }

  std::vector<Eigen::VectorXd> attached_path;
  attached_path.reserve(plan.path.size() + 2U);
  // compactTask() follows groupPosition(state), which is the predicted command
  // state passed by the servo. Seed the follower from that same queue state;
  // measured feedback remains independently connected and validated above.
  attached_path.push_back(command_queue_state->position);
  if ((attachment_position - attached_path.back()).norm() > 1.0e-12) {
    attached_path.push_back(attachment_position);
  }
  const std::size_t first_path_index = attachment.used_certificate ?
    attachment.segment_index + 1U : 1U;
  for (std::size_t path_index = first_path_index; path_index < plan.path.size(); ++path_index) {
    if ((plan.path[path_index] - attached_path.back()).norm() > 1.0e-12) {
      attached_path.push_back(plan.path[path_index]);
    }
  }
  if (attached_path.empty()) {
    reject_candidate("compact candidate attachment leaves no path to its goal");
    return;
  }
  if (!pair_yaw_assessment_.valid || !pair_yaw_assessment_.exhausted) {
    planner_->cancel();
    pending_generation_.reset();
    compact_preparation_.reset();
    compact_plan_pending_ = false;
    face_policy_.clearForStop();
    diagnostics_.compact_phase = "PREPARATION_CANCELLED";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_PAIR_RESOURCE_RESTORED";
    appendCompactEvent("preparation_cancelled", time_sec,
      "pair_yaw_resource_available_before_commit",
      pairYawAssessmentJson(pair_yaw_assessment_));
    return;
  }
  if (!follower_.setPath(attached_path, command_queue_state->position, time_sec)) {
    reject_candidate("compact candidate follower rejected its checked attachment");
    return;
  }

  CompactGoal committed_goal = candidate;
  committed_goal.path_requirements.maximum_assembly_radius_m = entry_radius_cap;
  committed_goal.path_requirements.maximum_abs_screen_pitch_rad = pitch_cap;
  committed_goal.path_requirements.base_link_name = base_link_->getName();
  committed_goal.path_requirements.screen_link_name = command_link_->getName();
  compact_goal_ = std::move(committed_goal);
  compact_preparation_.reset();
  compact_commit_initialized_ = true;
  base_handoff_policy_.clear();
  face_return_dwell_sec_ = 0.0;
  last_face_return_evidence_time_sec_.reset();
  diagnostics_.j1_hold_direction = 0;
  diagnostics_.j1_hold_q = 0.0;
  compact_radius_floor_latched_ = false;
  compact_return_full_face_active_ = false;
  compact_plan_pending_ = false;
  compact_path_adopted_ = true;
  compact_goal_reached_ = false;
  compact_path_endpoint_reached_ = false;
  compact_face_return_pending_ = false;
  compact_goal_event_emitted_ = false;
  compact_path_started_event_emitted_ = false;
  diagnostics_.compact_goal_reached = 0.0;
  diagnostics_.compact_entry_radius_cap_m = entry_radius_cap;
  diagnostics_.compact_effective_radius_cap_m = entry_radius_cap;
  diagnostics_.compact_pitch_cap_rad = pitch_cap;
  diagnostics_.compact_radius_floor_latched = false;
  diagnostics_.compact_return_full_face_active = false;
  diagnostics_.compact_return_face_dwell_sec = 0.0;
  diagnostics_.compact_return_face_qualified = false;
  diagnostics_.compact_actual_settled = false;
  diagnostics_.planned_path_max_radius_m = std::max(
    attachment.maximum_radius_m, bridge_radius);
  diagnostics_.planned_path_max_roll_error_rad = std::max(
    attachment.maximum_roll_error_rad, bridge_max_roll);
  diagnostics_.planned_path_max_abs_pitch_rad = std::max(
    attachment.maximum_abs_pitch_rad, bridge_max_pitch);
  diagnostics_.compact_reason = diagnostics_.compact_reason.empty() ?
    "pair_yaw_resource_exhausted" : diagnostics_.compact_reason;
  diagnostics_.compact_phase = "PATH";
  diagnostics_.tracking_phase = "COMPACT";
  diagnostics_.state = "COMPACT_PATH";
  diagnostics_.last_plan_message = "compact path committed: " + plan.message;
  active_scene_revision_ = scene_revision;
  ++diagnostics_.plans_accepted;

  const auto committed_goal_pose = evaluateCompactUprightPose(
    goal_state.getGlobalLinkTransform(command_link_).linear());
  const double goal_abs_pitch = committed_goal_pose ?
    std::abs(committed_goal_pose->signed_pitch_rad) :
    std::numeric_limits<double>::quiet_NaN();
  const std::string details =
    "{\"goal_profile\":\"" + compact_goal_.profile_id +
    "\",\"goal_joints_rad\":" + jsonVector(compact_goal_.bounded_positions) +
    ",\"staged_route_enabled\":" +
    (compact_goal_.path_requirements.staged_route ? "true" : "false") +
    ",\"entry_radius_cap_m\":" + jsonNumber(entry_radius_cap) +
    ",\"entry_envelope_max_abs_pitch_rad\":" + jsonNumber(entry_max_pitch) +
    ",\"entry_envelope_max_abs_roll_rad\":" + jsonNumber(entry_max_roll) +
    ",\"goal_abs_pitch_rad\":" + jsonNumber(goal_abs_pitch) +
    ",\"maximum_abs_screen_pitch_cap_rad\":" + jsonNumber(pitch_cap) +
    ",\"candidate_index\":" + std::to_string(accepted_candidate_index) +
    ",\"attachment\":{" +
    "\"used_certificate\":" + (attachment.used_certificate ? "true" : "false") +
    ",\"sample_index\":" + (attachment.used_certificate ?
    std::to_string(attachment.sample_index) : "null") +
    ",\"segment_index\":" + (attachment.used_certificate ?
    std::to_string(attachment.segment_index) : "null") +
    ",\"segment_fraction\":" + (attachment.used_certificate ?
    jsonNumber(attachment.segment_fraction) : "null") +
    ",\"positions_rad\":" + jsonVector(attachment_position) +
    ",\"actual_bridge_distance_rad\":" + jsonNumber(actual_to_attachment) +
    ",\"queue_bridge_distance_rad\":" + jsonNumber(queue_to_attachment) +
    ",\"suffix_maximum_radius_m\":" + jsonNumber(attachment.maximum_radius_m) +
    ",\"suffix_maximum_abs_pitch_rad\":" +
    jsonNumber(attachment.maximum_abs_pitch_rad) +
    ",\"suffix_maximum_roll_error_rad\":" +
    jsonNumber(attachment.maximum_roll_error_rad) +
    "},\"worker_full_path_metrics\":{" +
    "\"maximum_assembly_radius_m\":" + jsonNumber(metrics.maximum_assembly_radius_m) +
    ",\"maximum_abs_screen_pitch_rad\":" +
    jsonNumber(metrics.maximum_abs_screen_pitch_rad) +
    ",\"maximum_roll_error_rad\":" + jsonNumber(metrics.maximum_roll_error_rad) +
    "},\"follower_seed\":\"command_queue_state\",\"follower_seed_positions_rad\":" +
    jsonVector(command_queue_state->position) + "}";
  std::string trigger_details = details;
  trigger_details.pop_back();
  trigger_details += ",\"pair_yaw_gate\":" +
    pairYawAssessmentJson(pair_yaw_assessment_) + "}";
  appendCompactEvent("triggered", time_sec, diagnostics_.compact_reason, trigger_details);
}

void MotionReference::adoptCommittedCompactPlan(
  const BackgroundPathResult & plan, const Eigen::VectorXd & current_position,
  const double time_sec, const std::uint64_t scene_revision)
{
  if (!pending_generation_ || plan.generation != *pending_generation_) {
    return;
  }
  pending_generation_.reset();
  diagnostics_.last_plan_wall_ms = 1000.0 * plan.elapsed_time_s;
  diagnostics_.last_plan_message = plan.message;
  const bool radius_within_effective_cap = plan.validation_metrics &&
    std::isfinite(plan.validation_metrics->maximum_assembly_radius_m) &&
    plan.validation_metrics->maximum_assembly_radius_m <=
    effectiveCompactRadiusCap() + 1.0e-6;
  const double roll_cap = compact_goal_.path_requirements.maximum_roll_error_rad;
  const bool roll_within_committed_cap = plan.validation_metrics &&
    std::isfinite(plan.validation_metrics->maximum_roll_error_rad) &&
    plan.validation_metrics->maximum_roll_error_rad <=
    roll_cap + kCompactPitchValidationToleranceRad;
  const auto pitch_cap = compact_goal_.path_requirements.maximum_abs_screen_pitch_rad;
  const bool pitch_within_committed_cap = plan.validation_metrics && pitch_cap &&
    std::isfinite(plan.validation_metrics->maximum_abs_screen_pitch_rad) &&
    plan.validation_metrics->maximum_abs_screen_pitch_rad <=
    *pitch_cap + kCompactPitchValidationToleranceRad;
  const bool accepted = plan.success && plan.scene_revision == scene_revision &&
    plan.profile_id == compact_goal_.profile_id && radius_within_effective_cap &&
    roll_within_committed_cap && pitch_within_committed_cap &&
    follower_.setPath(plan.path, current_position, time_sec);
  if (accepted) {
    ++diagnostics_.plans_accepted;
    compact_plan_pending_ = false;
    compact_path_adopted_ = true;
    active_scene_revision_ = scene_revision;
    diagnostics_.compact_phase = "PATH";
    diagnostics_.planned_path_max_radius_m =
      plan.validation_metrics->maximum_assembly_radius_m;
    diagnostics_.planned_path_max_roll_error_rad =
      plan.validation_metrics->maximum_roll_error_rad;
    diagnostics_.planned_path_max_abs_pitch_rad =
      plan.validation_metrics->maximum_abs_screen_pitch_rad;
    diagnostics_.last_plan_message = "compact path accepted: " + plan.message;
    return;
  }
  ++diagnostics_.plans_rejected;
  compact_plan_pending_ = false;
  compact_path_adopted_ = false;
  diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_roll_error_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_phase = "REPLANNING";
  diagnostics_.last_plan_message = "compact path rejected: " + plan.message;
}

void MotionReference::updateFaceTargetIdentity(
  const std::optional<msg::TrackingTarget> & target, const double time_sec)
{
  if (!target || target->mode != msg::TrackingTarget::FACE || target->target_id < 0) {
    return;
  }
  if (target_id_ && *target_id_ != target->target_id) {
    if (!compact_commit_initialized_ &&
      face_policy_.phase() == FaceTrackingPhase::kCompactManeuver)
    {
      planner_->cancel();
      pending_generation_.reset();
      compact_preparation_.reset();
      compact_plan_pending_ = false;
      face_policy_.clearForStop();
      face_return_dwell_sec_ = 0.0;
      last_face_return_evidence_time_sec_.reset();
      diagnostics_.compact_return_face_dwell_sec = 0.0;
      diagnostics_.compact_return_face_qualified = false;
      diagnostics_.compact_phase = "PREPARATION_CANCELLED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_TARGET_CHANGED";
    }
    target_motion_estimator_.reset();
    target_bearing_motion_sign_ = 0;
    target_bearing_moving_ = false;
    appendCompactEvent("target_id_changed", time_sec, "selector_track_generation_changed");
  }
  target_id_ = target->target_id;
}

bool MotionReference::handleCompactPreparation(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const TargetMotionEstimate & target_motion,
  const planning_scene::PlanningScene & scene, const double time_sec,
  const std::uint64_t scene_revision, const Eigen::VectorXd & position,
  const bool fresh_actual_state,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication,
  MotionReferenceResult & result)
{
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const bool explicit_hold = target && target->mode == msg::TrackingTarget::HOLD &&
    !target->allow_recovery;
  if (!has_face || explicit_hold) {
    planner_->cancel();
    pending_generation_.reset();
    compact_preparation_.reset();
    compact_plan_pending_ = false;
    face_policy_.clearForStop();
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    diagnostics_.compact_return_face_qualified = false;
    diagnostics_.compact_phase = "PREPARATION_CANCELLED";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_TARGET_LOST";
    return true;
  }

  if (compact_preparation_ && compact_preparation_->scene_revision != scene_revision) {
    planner_->cancel();
    pending_generation_.reset();
    compact_plan_pending_ = false;
    compact_preparation_.reset();
    diagnostics_.last_plan_message =
      "restarting compact preparation for changed scene revision";
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    diagnostics_.compact_return_face_qualified = false;
  }
  bool preparation_cancelled = false;
  if (!pair_yaw_assessment_.valid || !pair_yaw_assessment_.exhausted) {
    planner_->cancel();
    pending_generation_.reset();
    compact_preparation_.reset();
    compact_plan_pending_ = false;
    face_policy_.clearForStop();
    face_return_dwell_sec_ = 0.0;
    last_face_return_evidence_time_sec_.reset();
    diagnostics_.compact_return_face_dwell_sec = 0.0;
    diagnostics_.compact_return_face_qualified = false;
    diagnostics_.compact_phase = "PREPARATION_CANCELLED";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_PAIR_RESOURCE_RESTORED";
    appendCompactEvent("preparation_cancelled", time_sec,
      "pair_yaw_resource_available", pairYawAssessmentJson(pair_yaw_assessment_));
    active_scene_revision_.reset();
    preparation_cancelled = true;
  }
  if (!preparation_cancelled && !upright_roll_guard_armed_) {
    double current_roll_error = std::numeric_limits<double>::infinity();
    auto upright_recovery = pathTask(
      state, position, position, 1.0, 0.01, true, &current_roll_error);
    const bool outside_roll_admission = !std::isfinite(current_roll_error) ||
      std::abs(current_roll_error) >
      compact_goal_.path_requirements.maximum_roll_error_rad;
    if (outside_roll_admission && upright_recovery) {
      diagnostics_.compact_phase = "UPRIGHT_START";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "COMPACT_UPRIGHT_START";
      result.task = std::move(upright_recovery);
      return true;
    }
  }

  if (!preparation_cancelled) {
    requestCompactManeuver(state, time_sec, scene, scene_revision,
      diagnostics_.compact_reason.empty() ? "pair_yaw_resource_exhausted" :
      diagnostics_.compact_reason,
      fresh_actual_state ? actual_feedback : std::nullopt,
      command_queue_state, active_publication, latest_publication);
  }
  if (!preparation_cancelled && compact_preparation_ && pending_generation_) {
    if (auto plan = planner_->takeResult()) {
      consumeCompactPreparationResult(
        *plan, state, time_sec, scene, scene_revision, actual_feedback,
        actual_feedback_time_sec, command_queue_state,
        active_publication, latest_publication);
    }
  }
  if (!compact_commit_initialized_) {
    if (compact_preparation_ && !compact_preparation_->exhausted &&
      !pending_generation_ && !planner_->isBusy() && fresh_actual_state &&
      command_queue_state)
    {
      moveit::core::RobotState start(state);
      start.setJointGroupPositions(group_, actual_feedback->position);
      start.update(true);
      submitCompactPreparationPlan(
        start, time_sec, scene_revision, actual_feedback, command_queue_state,
        active_publication, latest_publication);
    }
    result.task = faceTask(state, *target, target_motion, result);
    result.actual_face_alignment = fresh_actual_state ?
      measureActualFaceAlignment(
      state, *target, time_sec, actual_feedback, actual_feedback_time_sec) : std::nullopt;
    static_cast<void>(appendFaceRollGuardConstraints(
        state, command_queue_state, result));
    updateBaseHandoff(state, actual_feedback, actual_feedback_time_sec, *target,
      target_motion, command_queue_state, result);
    diagnostics_.tracking_phase = "FACE";
    if (!compact_preparation_ || compact_preparation_->exhausted) {
      diagnostics_.compact_phase = compact_preparation_ ?
        "PREPARATION_EXHAUSTED" : "COMMIT_WAIT";
    } else {
      diagnostics_.compact_phase = "PREPARING";
    }
    if (preparation_cancelled) {
      diagnostics_.compact_phase = "PREPARATION_CANCELLED";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_PAIR_RESOURCE_RESTORED";
    } else if (!compact_preparation_) {
      diagnostics_.state = face_policy_.phase() == FaceTrackingPhase::kCompactManeuver ?
        "FACE_PAIR_YAW_RESOURCE_EXHAUSTED" : "FACE_COMPACT_PREPARING_WAIT";
    } else if (!compact_preparation_->exhausted) {
      diagnostics_.state = pending_generation_ || planner_->isBusy() ?
        "FACE_COMPACT_PREPARING" : "FACE_COMPACT_PREPARING_WAIT";
    }
    return true;
  }
  return false;
}

MotionReferenceResult MotionReference::update(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const planning_scene::PlanningScene & scene,
  const double time_sec, const std::uint64_t scene_revision,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  MotionReferenceResult result;
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const bool healthy_face_loss = target && target->allow_recovery && !has_face;
  const auto mode = target ? target->mode : msg::TrackingTarget::HOLD;
  const bool idle = mode == msg::TrackingTarget::REST || mode == msg::TrackingTarget::SEARCH;
  if (idle || ((!has_face && !healthy_face_loss) &&
    face_policy_.phase() != FaceTrackingPhase::kCompactManeuver))
  {
    upright_roll_guard_armed_ = false;
    diagnostics_.upright_roll_guard_armed = false;
  }
  if (rest_mode_ != RestMode::kNone && !idle) {
    planner_->cancel();
    pending_generation_.reset();
    follower_.reset();
    rest_mode_ = RestMode::kNone;
    rest_settling_ = false;
  }
  updateFaceTargetIdentity(target, time_sec);
  TargetMotionEstimate target_motion;
  if (has_face) {
    const double measurement_time = static_cast<double>(target->face_stamp.sec) +
      1.0e-9 * static_cast<double>(target->face_stamp.nanosec);
    target_motion = target_motion_estimator_.update(
      Eigen::Vector3d(target->face.x, target->face.y, target->face.z),
      Eigen::Vector3d(target->pose.position.x, target->pose.position.y,
      target->pose.position.z), measurement_time, time_sec);
    const Eigen::Isometry3d base_pose = state.getGlobalLinkTransform(base_link_);
    const Eigen::Vector3d ray = base_pose.linear().transpose() *
      (Eigen::Vector3d(target->face.x, target->face.y, target->face.z) -
      base_pose.translation());
    const Eigen::Vector3d face_velocity_base =
      base_pose.linear().transpose() * target_motion.face_velocity;
    const double ray_squared = ray.head<2>().squaredNorm();
    int bearing_sign = 0;
    if (ray_squared > 1.0e-9 && ray.allFinite() && face_velocity_base.allFinite()) {
      const double rate = (ray.x() * face_velocity_base.y() -
        ray.y() * face_velocity_base.x()) / ray_squared;
      if (std::isfinite(rate) && std::abs(rate) >= 0.02) {
        bearing_sign = rate > 0.0 ? 1 : -1;
      }
    }
    if (bearing_sign != 0 && target_bearing_motion_sign_ != 0 &&
      bearing_sign != target_bearing_motion_sign_)
    {
      appendCompactEvent("target_reverse_intent", time_sec,
        "filtered_target_bearing_reversed", "{\"from_sign\":" +
        std::to_string(target_bearing_motion_sign_) + ",\"to_sign\":" +
        std::to_string(bearing_sign) + "}");
    } else if (bearing_sign == 0 && target_bearing_moving_) {
      appendCompactEvent("target_stopped", time_sec, "filtered_target_bearing_stationary");
    }
    if (bearing_sign != 0) {
      target_bearing_motion_sign_ = bearing_sign;
    }
    target_bearing_moving_ = bearing_sign != 0;
  }

  Eigen::VectorXd position = groupPosition(state);
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
  state.copyJointGroupVelocities(group_, velocity);
  state.copyJointGroupAccelerations(group_, acceleration);
  const bool settled = velocity.size() == position.size() &&
    acceleration.size() == position.size() &&
    velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
    acceleration.lpNorm<Eigen::Infinity>() < kSettledAccelerationRadps2;
  const bool fresh_actual_state = actual_feedback && actual_feedback_time_sec &&
    std::isfinite(*actual_feedback_time_sec) && time_sec >= *actual_feedback_time_sec &&
    time_sec - *actual_feedback_time_sec <= parameters_.state_feedback_timeout_sec &&
    actual_feedback->position.size() == position.size() &&
    actual_feedback->velocity.size() == position.size() &&
    actual_feedback->position.allFinite() && actual_feedback->velocity.allFinite();
  pair_yaw_assessment_ = has_face && fresh_actual_state && !compact_commit_initialized_ ?
    evaluatePairYawResource(state, *target, target_motion, actual_feedback,
      command_queue_state) : PairYawResourceAssessment{};
  if (!upright_roll_guard_armed_ && (has_face || (healthy_face_loss &&
    !idle && face_policy_.phase() == FaceTrackingPhase::kCompactManeuver)))
  {
    static_cast<void>(maybeArmFaceUprightRollGuard(
        state, time_sec, actual_feedback, actual_feedback_time_sec, command_queue_state,
        active_publication, latest_publication));
  }
  diagnostics_.upright_roll_guard_armed = upright_roll_guard_armed_;
  recordCompactSettledEvent(time_sec, compact_goal_reached_ && fresh_actual_state &&
    compactSettled(*actual_feedback, command_queue_state, compact_goal_.bounded_positions),
    actual_feedback, actual_feedback_time_sec, command_queue_state);
  if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
    compact_commit_initialized_ && compact_goal_.bounded_positions.size() == position.size())
  {
    if (fresh_actual_state) {
      diagnostics_.compact_actual_goal_error_rad =
        (actual_feedback->position - compact_goal_.bounded_positions).lpNorm<Eigen::Infinity>();
      diagnostics_.compact_actual_speed_max_rad_s =
        actual_feedback->velocity.lpNorm<Eigen::Infinity>();
    } else {
      diagnostics_.compact_actual_goal_error_rad.reset();
      diagnostics_.compact_actual_speed_max_rad_s.reset();
    }
    if (command_queue_state && command_queue_state->position.size() == position.size() &&
      command_queue_state->velocity.size() == position.size() &&
      command_queue_state->acceleration.size() == position.size() &&
      command_queue_state->position.allFinite() && command_queue_state->velocity.allFinite() &&
      command_queue_state->acceleration.allFinite())
    {
      diagnostics_.compact_queue_tail_goal_error_rad =
        (command_queue_state->position - compact_goal_.bounded_positions).
        lpNorm<Eigen::Infinity>();
      diagnostics_.compact_queue_tail_speed_max_rad_s =
        command_queue_state->velocity.lpNorm<Eigen::Infinity>();
      diagnostics_.compact_queue_tail_acceleration_max_rad_s2 =
        command_queue_state->acceleration.lpNorm<Eigen::Infinity>();
    } else {
      diagnostics_.compact_queue_tail_goal_error_rad.reset();
      diagnostics_.compact_queue_tail_speed_max_rad_s.reset();
      diagnostics_.compact_queue_tail_acceleration_max_rad_s2.reset();
    }
  }

  if (fresh_actual_state && command_queue_state) {
    maybeLatchCompactRadiusCap(
      state, time_sec, std::min(time_sec, *actual_feedback_time_sec),
      *actual_feedback, *command_queue_state,
      active_publication, latest_publication);
  }
  diagnostics_.compact_radius_floor_latched = compact_radius_floor_latched_;
  diagnostics_.compact_effective_radius_cap_m = compact_commit_initialized_ ?
    effectiveCompactRadiusCap() : std::numeric_limits<double>::quiet_NaN();

  if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver) {
    const bool explicit_hold = mode == msg::TrackingTarget::HOLD &&
      (!target || !target->allow_recovery);
    if (!compact_commit_initialized_ && handleCompactPreparation(
        state, target, target_motion, scene, time_sec, scene_revision, position,
        fresh_actual_state, actual_feedback, actual_feedback_time_sec,
        command_queue_state, active_publication, latest_publication, result))
    {
      return result;
    }
    diagnostics_.tracking_phase = "COMPACT";
    bool compact_radius_constraints_computed = false;
    const auto compute_compact_radius_constraints = [&]() {
        if (compact_radius_constraints_computed || !compact_commit_initialized_) {
          return;
        }
        compact_radius_constraints_computed = true;
        if (!command_queue_state || !appendCompactPathConstraints(
            state, *command_queue_state, result))
        {
          result.motion_reference_support_unavailable = true;
        }
      };
    if (explicit_hold || !target ||
      (!has_face && !healthy_face_loss))
    {
      diagnostics_.state = "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND";
      return result;
    }
    compute_compact_radius_constraints();
    if (compact_face_return_pending_ && has_face) {
      result.task = faceReturnTask(
        state, *target, target_motion, result, time_sec, actual_feedback,
        actual_feedback_time_sec, fresh_actual_state);
      diagnostics_.compact_phase = "RETURN_FACE";
      diagnostics_.tracking_phase = "COMPACT_RETURN_FACE";
      diagnostics_.state = "COMPACT_RETURN_FACE";
      return result;
    }
    if (active_scene_revision_ && *active_scene_revision_ != scene_revision &&
      !pending_generation_)
    {
      follower_.reset();
      compact_path_adopted_ = false;
      compact_plan_pending_ = false;
    }
    if (auto plan = planner_->takeResult()) {
      adoptCommittedCompactPlan(*plan, position, time_sec, scene_revision);
    }
    double current_roll_error = std::numeric_limits<double>::infinity();
    auto upright_hold = pathTask(state, position, position, 1.0, 0.01, true,
      &current_roll_error);
    if (!upright_hold) {
      diagnostics_.state = "COMPACT_UPRIGHT_UNAVAILABLE";
      return result;
    }
    if (std::abs(current_roll_error) > compact_goal_.path_requirements.maximum_roll_error_rad) {
      diagnostics_.compact_phase = "UPRIGHT_START";
      diagnostics_.state = "COMPACT_UPRIGHT_START";
      result.task = std::move(upright_hold);
      return result;
    }
    if (!compact_path_adopted_ && !pending_generation_ && !planner_->isBusy() &&
      fresh_actual_state && motionSettled(*actual_feedback, command_queue_state))
    {
      moveit::core::RobotState start(state);
      start.setJointGroupPositions(group_, actual_feedback->position);
      start.update(true);
      position = actual_feedback->position;
      submitCompactPlan(start, scene_revision, time_sec);
    }
    result.task = compactTask(state, position, time_sec);
    if (compact_path_endpoint_reached_ && fresh_actual_state &&
      compactGoalPositionReached(*actual_feedback, compact_goal_.bounded_positions))
    {
      compact_goal_reached_ = true;
      diagnostics_.compact_goal_reached = 1.0;
      if (!compact_goal_event_emitted_) {
        appendCompactEvent("goal_reached", time_sec, diagnostics_.compact_reason);
        compact_goal_event_emitted_ = true;
      }
      const bool compact_settled = compactSettled(
        *actual_feedback, command_queue_state, compact_goal_.bounded_positions);
      recordCompactSettledEvent(
        time_sec, compact_settled, actual_feedback, actual_feedback_time_sec,
        command_queue_state);
      if (compact_settled) {
        if (has_face) {
          compact_face_return_pending_ = true;
          result.compact_pitch_constraints.clear();
          result.task = faceReturnTask(
            state, *target, target_motion, result, time_sec, actual_feedback,
            actual_feedback_time_sec, fresh_actual_state);
          diagnostics_.compact_phase = "RETURN_FACE";
          diagnostics_.tracking_phase = "COMPACT_RETURN_FACE";
          diagnostics_.state = "COMPACT_RETURN_FACE";
          result.follows_path = false;
        } else if (healthy_face_loss) {
          appendCompactEvent("completed", time_sec, diagnostics_.compact_reason);
          static_cast<void>(face_policy_.update(FaceTrackingPolicyInput{false, true}));
          compact_commit_initialized_ = false;
          compact_radius_floor_latched_ = false;
          compact_goal_.path_requirements.maximum_abs_screen_pitch_rad.reset();
          result.compact_radius_constraints.clear();
          result.compact_roll_constraints.clear();
          result.compact_pitch_constraints.clear();
          diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
          diagnostics_.planned_path_max_roll_error_rad =
            std::numeric_limits<double>::quiet_NaN();
          diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
          compact_return_full_face_active_ = false;
          diagnostics_.compact_radius_floor_latched = false;
          diagnostics_.compact_effective_radius_cap_m =
            std::numeric_limits<double>::quiet_NaN();
          diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
          diagnostics_.compact_return_full_face_active = false;
          diagnostics_.compact_return_face_qualified = false;
          diagnostics_.compact_return_face_dwell_sec = 0.0;
          diagnostics_.compact_actual_settled = false;
          compact_goal_reached_ = false;
          compact_goal_event_emitted_ = false;
          compact_path_endpoint_reached_ = false;
          compact_path_adopted_ = false;
          compact_plan_pending_ = false;
          follower_.reset();
          diagnostics_.compact_phase = "COMPLETE";
          diagnostics_.tracking_phase = "FACE";
          diagnostics_.state = "NO_FACE";
          diagnostics_.compact_goal_reached = 0.0;
          active_scene_revision_.reset();
        }
      }
    }
    result.follows_path = compact_path_adopted_ && !compact_face_return_pending_ &&
      result.task.has_value();
    if (result.task) {
      if (compact_face_return_pending_) {
        return result;
      }
      diagnostics_.state = !compact_path_adopted_ ? "COMPACT_PLANNING" :
        (compact_path_endpoint_reached_ ? "COMPACT_SETTLING" : "COMPACT_PATH");
      return result;
    }
    if (compact_path_endpoint_reached_) {
      diagnostics_.state = "COMPACT_SETTLING";
      return result;
    }
    diagnostics_.state = "COMPACT_PLANNING";
    return result;
  }

  if (mode == msg::TrackingTarget::HOLD) {
    diagnostics_.state = "HOLD";
    diagnostics_.tracking_phase = "FACE";
    return result;
  }

  if (mode == msg::TrackingTarget::SEARCH && !search_enabled_) {
    diagnostics_.state = "SEARCH_UNAVAILABLE";
    diagnostics_.last_plan_message = "SEARCH rejected: search is disabled";
    return result;
  }
  if (mode == msg::TrackingTarget::REST || mode == msg::TrackingTarget::SEARCH) {
    const RestMode requested = mode == msg::TrackingTarget::REST ? RestMode::kRest :
      RestMode::kSearch;
    if (rest_mode_ != requested ||
      (requested == RestMode::kSearch && search_pattern_ != target->search_pattern))
    {
      planner_->cancel();
      pending_generation_.reset();
      follower_.reset();
      rest_mode_ = requested;
      rest_completed_ = false;
      rest_settling_ = false;
      if (requested == RestMode::kSearch) {
        search_pattern_ = target->search_pattern;
        search_->start(position, search_positions_, limits_.lower_position +
          limits_.position_margin, limits_.upper_position - limits_.position_margin,
          search_base_index_, search_config_, search_pattern_ == "local_then_sweep", time_sec);
      }
    }
    if (search_ && requested == RestMode::kSearch) {
      search_->update(position, time_sec);
      if (!search_->preparing()) {
        result.task = pathTask(
          state, position, search_->goal(), 2.0, search_config_.speed_rad_s, true);
        result.follows_path = result.task.has_value();
        diagnostics_.state = search_->phase();
        return result;
      }
    }
    const Eigen::VectorXd & goal = requested == RestMode::kSearch ?
      search_->goal() : rest_positions_;
    if (follower_.active()) {
      const auto path = follower_.update(position, time_sec);
      diagnostics_.path_progress_rad = path.progress_rad;
      diagnostics_.path_distance_rad = path.distance_from_path_rad;
      if (path.status == JointPathStatus::kTracking) {
        result.task = pathTask(state, position, path.reference_position, 3.0, 0.35, true);
        result.follows_path = true;
        diagnostics_.state = requested == RestMode::kSearch ? "SEARCH_PATH" : "REST_PATH";
        return result;
      }
      follower_.reset();
      if (path.status == JointPathStatus::kArrived) {
        ++diagnostics_.paths_completed;
        rest_settling_ = true;
      } else if (requested == RestMode::kSearch) {
        ++diagnostics_.paths_abandoned;
        search_->retryFrom(position, time_sec);
      }
    }
    const double error = (position - goal).lpNorm<Eigen::Infinity>();
    const bool actual_settled = actual_feedback ?
      (actual_feedback->velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
      actual_feedback->acceleration.lpNorm<Eigen::Infinity>() < kSettledAccelerationRadps2) :
      settled;
    if (error < kRestArrivalRad && actual_settled) {
      if (requested == RestMode::kRest) {
        rest_completed_ = true;
        rest_settling_ = false;
      } else {
        search_->prepared(position, time_sec);
        rest_settling_ = false;
        result.task = pathTask(state, position, search_->goal(), 2.0,
          search_config_.speed_rad_s, true);
        result.follows_path = result.task.has_value();
      }
      diagnostics_.state = requested == RestMode::kSearch ? search_->phase() : "REST_HOLD";
      return result;
    }
    if (requested == RestMode::kRest && rest_completed_ && error > kRestReleaseRad) {
      rest_completed_ = false;
    }
    if (requested == RestMode::kRest && rest_completed_ && !actual_settled) {
      rest_settling_ = true;
    }
    if (requested == RestMode::kRest && rest_settling_) {
      diagnostics_.state = "REST_SETTLING";
      if (error >= kRestArrivalRad) {
        result.task = pathTask(state, position, goal, 3.0, 0.10, true);
        result.follows_path = result.task.has_value();
      }
      return result;
    }
    if (requested == RestMode::kSearch && search_->preparing()) {
      if (!pending_generation_ && !planner_->isBusy()) {
        pending_generation_ = planner_->submit(state, goal, scene_revision);
        ++diagnostics_.plans_requested;
      }
      if (auto plan = planner_->takeResult()) {
        if (pending_generation_ && plan->generation == *pending_generation_) {
          pending_generation_.reset();
          if (plan->success && plan->scene_revision == scene_revision &&
            follower_.setPath(plan->path, position, time_sec))
          {
            ++diagnostics_.plans_accepted;
          } else {
            ++diagnostics_.plans_rejected;
            search_->retryFrom(position, time_sec);
          }
        }
      }
      diagnostics_.state = "SEARCH_PLANNING";
      return result;
    }
    if (requested == RestMode::kRest && pending_generation_) {
      if (auto plan = planner_->takeResult()) {
        if (plan->generation == *pending_generation_) {
          pending_generation_.reset();
          diagnostics_.last_plan_message = plan->message;
          if (plan->success && plan->scene_revision == scene_revision &&
            follower_.setPath(plan->path, position, time_sec))
          {
            ++diagnostics_.plans_accepted;
          } else {
            ++diagnostics_.plans_rejected;
            diagnostics_.last_plan_message = "REST path rejected: " + plan->message;
          }
        }
      }
    }
    if (requested == RestMode::kRest && !rest_completed_ && !follower_.active() &&
      !pending_generation_ && !planner_->isBusy())
    {
      pending_generation_ = planner_->submit(state, goal, scene_revision);
      ++diagnostics_.plans_requested;
      diagnostics_.state = "REST_PLANNING";
      return result;
    }
    if (requested == RestMode::kRest && !rest_completed_ && !follower_.active()) {
      diagnostics_.state = "REST_PLANNING";
      return result;
    }
    diagnostics_.state = "SEARCH_SETTLING";
    return result;
  }

  rest_mode_ = RestMode::kNone;
  rest_completed_ = false;
  rest_settling_ = false;
  if (has_face || mode == msg::TrackingTarget::POSE) {
    result.task = faceTask(state, *target, target_motion, result, has_face);
    if (!has_face) {
      diagnostics_.state = result.task ? "POSE" : "INVALID_TASK";
      diagnostics_.tracking_phase = "POSE";
      return result;
    }
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = result.task ? "FACE" : "INVALID_TASK";
    if (has_face) {
      static_cast<void>(appendFaceRollGuardConstraints(
          state, command_queue_state, result));
    }
    updateBaseHandoff(state, actual_feedback, actual_feedback_time_sec, *target,
      target_motion, command_queue_state, result);
    if (face_policy_.phase() == FaceTrackingPhase::kTracking &&
      pair_yaw_assessment_.valid && pair_yaw_assessment_.exhausted)
    {
      const std::string pair_details = pairYawAssessmentJson(pair_yaw_assessment_);
      if (compact_goal_.profile_id.empty() || !compact_geometry_validator_) {
        if (!compact_profile_unavailable_reported_) {
          diagnostics_.compact_reason = "no_verified_compact_profile_for_robot_model";
          diagnostics_.compact_phase = "UNAVAILABLE";
          appendCompactEvent("compact_unavailable", time_sec, diagnostics_.compact_reason,
            pair_details);
          compact_profile_unavailable_reported_ = true;
        }
      } else {
        diagnostics_.compact_reason = "pair_yaw_resource_exhausted";
        diagnostics_.compact_phase = "COMMIT_WAIT";
        diagnostics_.state = "FACE_PAIR_YAW_RESOURCE_EXHAUSTED";
        static_cast<void>(face_policy_.update(FaceTrackingPolicyInput{true, false}));
        appendCompactEvent("pair_resource_exhausted", time_sec,
          diagnostics_.compact_reason, pair_details);
      }
    }
    return result;
  }

  diagnostics_.state = "NO_FACE";
  diagnostics_.tracking_phase = "FACE";
  return result;
}

}  // namespace face_tracking_arm::control
