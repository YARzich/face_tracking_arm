// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/motion_reference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "face_tracking_arm/posture_recovery_policy.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"

namespace face_tracking_arm::control
{
namespace
{

JointMotionState stationaryState(const Eigen::VectorXd & position)
{
  return {position, Eigen::VectorXd::Zero(position.size()),
    Eigen::VectorXd::Zero(position.size())};
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

}  // namespace

MotionReference::MotionReference(
  const rclcpp::Node::SharedPtr & node,
  const planning_scene_monitor::PlanningSceneMonitorPtr & scene_monitor,
  ControllerParameters parameters, JointMotionLimits limits,
  CollisionConstraintConfig collision_config)
: parameters_(std::move(parameters)), limits_(std::move(limits)),
  collision_validator_(collision_config),
  group_(scene_monitor->getRobotModel()->getJointModelGroup(parameters_.planning_group_name)),
  command_link_(scene_monitor->getRobotModel()->getLinkModel(parameters_.command_frame))
{
  moveit::core::RobotState rest(scene_monitor->getRobotModel());
  rest.setToDefaultValues();
  if (!group_ || !command_link_ || !rest.setToDefaultValues(group_, "rest")) {
    throw std::invalid_argument(
            "Motion reference requires the group, control frame and named rest");
  }
  if (!parameters_.gaze_frame.empty()) {
    gaze_link_ = scene_monitor->getRobotModel()->getLinkModel(parameters_.gaze_frame);
    if (!gaze_link_ || rigidRoot(gaze_link_) != rigidRoot(command_link_)) {
      throw std::invalid_argument(
          "gaze_frame must be a camera optical link rigidly fixed to the screen");
    }
  }
  rest.copyJointGroupPositions(group_, rest_positions_);
  if (!rest.setToDefaultValues(group_, "search")) {
    throw std::invalid_argument("Motion reference requires a named search posture");
  }
  rest.copyJointGroupPositions(group_, search_positions_);
  const auto & names = group_->getVariableNames();
  const auto base = std::find(names.begin(), names.end(), "joint1");
  if (base == names.end()) {
    throw std::invalid_argument("Idle search requires joint1");
  }
  search_base_index_ = std::distance(names.begin(), base);
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  search_config_.speed_rad_s = node->declare_parameter<double>(
    "search_speed_rad_s", 0.30, descriptor);
  search_config_.sweep_half_range_rad = node->declare_parameter<double>(
    "search_sweep_half_range_rad", 2.80, descriptor);
  search_config_.local_half_range_rad = node->declare_parameter<double>(
    "search_local_half_range_rad", 0.35, descriptor);
  search_config_.local_duration_sec = node->declare_parameter<double>(
    "search_local_duration_sec", 6.0, descriptor);
  search_.start(rest_positions_, search_positions_, limits_.lower_position +
    limits_.position_margin, limits_.upper_position - limits_.position_margin,
    search_base_index_, search_config_, false, 0.0);
  BackgroundPathPlannerConfig config;
  config.move_group_name = parameters_.planning_group_name;
  // Each capture owns its immutable data; the worker never calls back into the
  // control component or touches its live queue or mutable RobotState.
  planner_ = std::make_unique<BackgroundPathPlanner>(
    node, scene_monitor->getRobotModel(), scene_monitor, config,
    [validator = collision_validator_, limits = limits_, group = group_](
      const planning_scene::PlanningScene & scene, const moveit::core::RobotState & state)
    {
      Eigen::VectorXd position;
      state.copyJointGroupPositions(group, position);
      if ((position.array() < limits.lower_position.array()).any() ||
      (position.array() > limits.upper_position.array()).any())
      {
        return false;
      }
      const auto motion = stationaryState(position);
      const auto checked = validator.validateSegment(scene, state, *group, motion, motion, 2);
      return checked.input_valid && !checked.unsafe;
    });
}

void MotionReference::reset()
{
  if (mode_ == msg::TrackingTarget::FACE && (pending_generation_ || follower_.active())) {
    recovery_failures_ = std::min(1000u, recovery_failures_ + 1);
  }
  planner_->cancel();
  follower_.reset();
  target_motion_estimator_.reset();
  mode_.reset();
  search_pattern_.clear();
  pending_generation_.reset();
  next_plan_time_sec_ = 0.0;
  rest_completed_ = false;
  rest_settling_ = false;
  acquisition_planning_ = false;
  recovery_requested_ = false;
  pointing_progress_.reset();
  planned_face_.reset();
  diagnostics_.state = "HOLD";
}

void MotionReference::rejectPath()
{
  if (follower_.active() || rest_settling_) {
    ++diagnostics_.paths_abandoned;
    recovery_failures_ = std::min(1000u, recovery_failures_ + 1);
    follower_.reset();
    rest_settling_ = false;
  }
}

const MotionReferenceDiagnostics & MotionReference::diagnostics() const noexcept
{
  return diagnostics_;
}

MotionReferenceResult MotionReference::update(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const planning_scene::PlanningScene & scene,
  const double time_sec, const std::uint64_t scene_revision)
{
  MotionReferenceResult result;
  const bool live_face = target && target->mode == msg::TrackingTarget::FACE;
  // Only a positively identified perception loss may finish an adopted detour.
  // Missing commands, explicit HOLD and safety/pause commands always cancel it.
  const bool finishing_detour = target && target->allow_recovery && !live_face &&
    mode_ == msg::TrackingTarget::FACE && follower_.active() && path_relaxation_ >= 2;
  const std::uint8_t mode = finishing_detour ? msg::TrackingTarget::FACE :
    (target ? target->mode : msg::TrackingTarget::HOLD);
  if (!mode_ || mode != *mode_ ||
    (mode == msg::TrackingTarget::SEARCH && search_pattern_ != target->search_pattern))
  {
    reset();
    mode_ = mode;
    if (mode == msg::TrackingTarget::SEARCH) {
      Eigen::VectorXd current;
      state.copyJointGroupPositions(group_, current);
      search_pattern_ = target->search_pattern;
      search_.start(current, search_positions_, limits_.lower_position + limits_.position_margin,
        limits_.upper_position - limits_.position_margin, search_base_index_, search_config_,
        search_pattern_ == "local_then_sweep", time_sec);
    }
  }
  if (mode == msg::TrackingTarget::HOLD) {
    diagnostics_.state = "HOLD";
    return result;
  }
  Eigen::VectorXd position;
  state.copyJointGroupPositions(group_, position);
  const bool searching = mode == msg::TrackingTarget::SEARCH;
  if (searching) {
    search_.update(position, time_sec);
    if (!search_.preparing()) {
      diagnostics_.state = search_.phase();
      result.task = makeJointPathVelocityTask(position, search_.goal(), 2.0,
          search_config_.speed_rad_s);
      result.follows_path = true;
      return result;
    }
  }
  TargetMotionEstimate target_motion;
  if (live_face) {
    const double measurement_time_sec = static_cast<double>(target->face_stamp.sec) +
      1.0e-9 * static_cast<double>(target->face_stamp.nanosec);
    target_motion = target_motion_estimator_.update(
      Eigen::Vector3d(target->face.x, target->face.y, target->face.z),
      Eigen::Vector3d(target->pose.position.x, target->pose.position.y, target->pose.position.z),
      measurement_time_sec, time_sec);
  }
  if (live_face || mode == msg::TrackingTarget::POSE) {
    Eigen::MatrixXd jacobian;
    if (state.getJacobian(group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false)) {
      // MoveIt returns the Jacobian in the group's root-link frame. References
      // and optical extrinsics use the planning/world frame, including tilted bases.
      const auto * root = group_->getJointModels().front()->getParentLinkModel();
      if (root) {
        const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
        jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
        jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
      }
      TrackingVelocityTaskConfig config;
      config.position_gain = parameters_.position_gain;
      config.orientation_gain = parameters_.orientation_gain;
      config.position_deadband_m = parameters_.position_deadband_m;
      config.pointing_deadband_rad = follower_.active() && path_relaxation_ == 1 ?
        0.35 : parameters_.pointing_deadband_rad;
      config.maximum_linear_reference_mps = parameters_.maximum_linear_reference_mps;
      config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
      config.roll_weight = parameters_.secondary_roll_weight;
      const Eigen::Quaterniond orientation(target->pose.orientation.w,
        target->pose.orientation.x, target->pose.orientation.y, target->pose.orientation.z);
      std::optional<Eigen::Vector3d> face;
      if (live_face) {
        face = Eigen::Vector3d(target->face.x, target->face.y, target->face.z);
      }
      std::optional<GazeKinematics> gaze;
      if (live_face && gaze_link_) {
        gaze.emplace();
        gaze->pose = state.getGlobalLinkTransform(gaze_link_);
        gaze->pose.linear() *= opticalToPointingRotation();
        gaze->jacobian = jacobian;
        const Eigen::Vector3d offset = gaze->pose.translation() -
          state.getGlobalLinkTransform(command_link_).translation();
        // The rigid mount changes only Jv: v_camera = v_screen + omega x offset.
        for (Eigen::Index joint = 0; joint < jacobian.cols(); ++joint) {
          gaze->jacobian.col(joint).head<3>() +=
            Eigen::Vector3d(jacobian.col(joint).tail<3>()).cross(offset);
        }
      }
      result.tracking = makeTrackingVelocityTask(
        state.getGlobalLinkTransform(command_link_), jacobian,
        Eigen::Vector3d(target->pose.position.x, target->pose.position.y, target->pose.position.z),
        face, orientation.normalized().toRotationMatrix(), config,
        target_motion.face_velocity, target_motion.goal_velocity, gaze);
    }
  }

  if (auto plan = planner_->takeResult()) {
    diagnostics_.last_plan_message = plan->message;
    diagnostics_.last_plan_wall_ms = 1000.0 * plan->elapsed_time_s;
    const bool current = pending_generation_ && plan->generation == *pending_generation_ &&
      plan->scene_revision == scene_revision;
    pending_generation_.reset();
    bool accepted = current && plan->success && follower_.setPath(plan->path, position, time_sec);
    if (accepted) {
      acquisition_planning_ = false;
      path_relaxation_ = pending_relaxation_;
      diagnostics_.recovery_relaxation = path_relaxation_;
      ++diagnostics_.plans_accepted;
    } else {
      follower_.reset();
      ++diagnostics_.plans_rejected;
      recovery_failures_ = std::min(1000u, recovery_failures_ + 1);
    }
    next_plan_time_sec_ = time_sec + (accepted ? 0.75 :
      std::min(3.0, 0.75 + 0.25 * recovery_failures_));
  }

  if (follower_.active()) {
    const auto path = follower_.update(position, time_sec);
    diagnostics_.path_progress_rad = path.progress_rad;
    diagnostics_.path_distance_rad = path.distance_from_path_rad;
    if (path.status == JointPathStatus::kTracking) {
      // This waypoint is a velocity objective, not an executed straight chord.
      // The executor validates the actual constrained next step; checking the
      // whole lookahead here duplicates mesh work and starves the command queue.
      diagnostics_.state = searching ? "SEARCH_PATH" :
        (mode == msg::TrackingTarget::REST ? "REST_PATH" : "POSTURE_PATH");
      const bool returning_to_rest = mode == msg::TrackingTarget::REST;
      result.task = makeJointPathVelocityTask(
        position, path.reference_position, returning_to_rest ? 3.0 : 6.0,
        searching ? search_config_.speed_rad_s : (returning_to_rest ? 0.35 : 0.70));
      if (mode == msg::TrackingTarget::FACE && path_relaxation_ < 2 && !result.tracking) {
        result.task.reset();
        diagnostics_.state = "INVALID_TASK";
        return result;
      }
      const bool retain_pointing = path_relaxation_ == 0 ||
        (path_relaxation_ == 1 && result.tracking && result.tracking->pointing_error_rad >= 0.35);
      if (mode == msg::TrackingTarget::FACE && retain_pointing && result.task) {
        // Stage 1 is a soft cone, not a hard angular constraint: inside it the
        // joint path has priority, outside it pointing correction returns.
        // Unlike a zero pointing reference, omitting those rows actually frees
        // the two directions. The executor still bounds q/v/a and collisions.
        result.task->secondary_matrix = result.task->primary_matrix;
        result.task->secondary_reference = result.task->primary_reference;
        result.task->secondary_weights = result.task->primary_weights;
        result.task->primary_matrix = result.tracking->task.primary_matrix;
        result.task->primary_reference = result.tracking->task.primary_reference;
        result.task->primary_weights = result.tracking->task.primary_weights;
      }
      result.follows_path = true;
      return result;
    }
    if (path.status == JointPathStatus::kArrived) {
      ++diagnostics_.paths_completed;
      rest_settling_ = mode == msg::TrackingTarget::REST || searching;
      next_plan_time_sec_ = time_sec + 1.0;
      recovery_requested_ = false;
      pointing_progress_.reset();
    } else {
      ++diagnostics_.paths_abandoned;
      recovery_failures_ = std::min(1000u, recovery_failures_ + 1);
      next_plan_time_sec_ = time_sec + 0.75;
    }
    follower_.reset();
  }
  if (finishing_detour) {
    reset();
    return update(state, target, scene, time_sec, scene_revision);
  }

  const Eigen::VectorXd & idle_goal = searching ? search_.goal() : rest_positions_;
  const double rest_error = (position - idle_goal).lpNorm<Eigen::Infinity>();
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
  state.copyJointGroupVelocities(group_, velocity);
  state.copyJointGroupAccelerations(group_, acceleration);
  const bool settled = velocity.lpNorm<Eigen::Infinity>() < 0.01 &&
    acceleration.lpNorm<Eigen::Infinity>() < 0.03;
  if (searching && rest_error < 0.025 && settled) {
    search_.prepared();
    rest_settling_ = false;
    diagnostics_.state = search_.phase();
    result.task = makeJointPathVelocityTask(position, search_.goal(), 2.0,
        search_config_.speed_rad_s);
    result.follows_path = true;
    return result;
  }
  if (mode == msg::TrackingTarget::REST && rest_error < 0.025 && settled) {
    rest_completed_ = true;
    rest_settling_ = false;
  }
  if (rest_completed_ && rest_error > 0.06) {
    rest_completed_ = false;
  }
  if (rest_settling_) {
    diagnostics_.state = searching ? "SEARCH_SETTLING" : "REST_SETTLING";
    // Inside the arrival tolerance, brake with the existing q/v/a limits.
    // Chasing tiny residuals with a low jerk budget can sustain a limit cycle.
    if (rest_error < 0.025) {
      return result;
    }
    result.task = makeJointPathVelocityTask(position, idle_goal, 3.0, 0.10);
    result.follows_path = true;
    return result;
  }
  bool posture_needed = false;
  if (live_face) {
    if (result.tracking && pointing_progress_.update(
        result.tracking->pointing_error_rad, time_sec))
    {
      recovery_requested_ = true;
    }
    if (result.tracking && result.tracking->pointing_error_rad <= 0.03 &&
      !pending_generation_ && !recovery_requested_)
    {
      recovery_requested_ = false;
      recovery_failures_ = 0;
    }
    // A large initial pointing error needs a global posture before local motion.
    if (result.tracking && result.tracking->pointing_error_rad > 2.0) {
      // Keep waiting through small changes near the entry threshold. A new
      // face, mode change or accepted plan ends this acquisition request.
      acquisition_planning_ = true;
      recovery_requested_ = true;
    }
    for (Eigen::Index joint = 0; joint < position.size(); ++joint) {
      if (needsPostureRecovery(
          position[joint], velocity[joint], limits_.lower_position[joint],
          limits_.upper_position[joint], limits_.position_margin[joint],
          target_motion.face_velocity.norm()))
      {
        posture_needed = true;
      }
    }
  }
  const bool requires_settled_start = mode == msg::TrackingTarget::FACE &&
    (acquisition_planning_ || (recovery_requested_ && recoveryRelaxation(recovery_failures_) >= 2));
  if (((mode == msg::TrackingTarget::REST && !rest_completed_) || searching ||
    posture_needed || acquisition_planning_ || recovery_requested_) &&
    (!requires_settled_start || settled) &&
    !pending_generation_ && !planner_->isBusy() && time_sec >= next_plan_time_sec_)
  {
    if (mode == msg::TrackingTarget::FACE) {
      const auto & base = state.getGlobalLinkTransform("link_base").translation();
      planned_face_ = Eigen::Vector3d(target->face.x, target->face.y, target->face.z);
      BackgroundPathPlanner::PoseGoal goal;
      goal.link_name = parameters_.command_frame;
      goal.gaze_link_name = parameters_.gaze_frame;
      goal.preferred_base_angle_rad = std::atan2(
        target->face.y - base.y(), target->face.x - base.x());
      goal.pose.translation() = Eigen::Vector3d(
        target->pose.position.x, target->pose.position.y, target->pose.position.z);
      goal.face_position = planned_face_;
      goal.relaxation = recoveryRelaxation(recovery_failures_);
      goal.attempt = recovery_attempt_++;
      pending_relaxation_ = goal.relaxation;
      diagnostics_.recovery_relaxation = goal.relaxation;
      recovery_requested_ = true;
      if (const auto orientation = pointingRotation(*planned_face_ - goal.pose.translation())) {
        goal.pose.linear() = *orientation;
      }
      pending_generation_ = planner_->submit(state, goal, scene_revision);
    } else {
      pending_relaxation_ = 2;
      pending_generation_ = planner_->submit(state, idle_goal, scene_revision);
    }
    ++diagnostics_.plans_requested;
    next_plan_time_sec_ = time_sec + 0.75;
  }
  if (mode == msg::TrackingTarget::REST || searching) {
    diagnostics_.state = searching ? "SEARCH_PLANNING" :
      (rest_completed_ ? "REST_HOLD" : "REST_PLANNING");
    return result;
  }
  if (requires_settled_start) {
    // After less disruptive attempts failed, settle the planning start instead
    // of repeatedly outrunning path attachment. The same executor brakes its
    // existing q/v/a timeline; diagnostics retain the live pointing error.
    diagnostics_.state = acquisition_planning_ ? "ACQUIRE_PLANNING" : "RECOVERY_PLANNING";
    return result;
  }

  diagnostics_.state = "FACE";
  if (result.tracking && !result.tracking->accepted) {
    result.task = result.tracking->task;
  }
  return result;
}

}  // namespace face_tracking_arm::control
