// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/background_path_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/logging.hpp>

namespace face_tracking_arm::control
{
namespace
{
using ErrorCode = moveit_msgs::msg::MoveItErrorCodes;
using SteadyClock = std::chrono::steady_clock;
constexpr double kPi = 3.14159265358979323846;
constexpr double kRecoveryEndpointMaximumOffsetRad = 3.5;

bool positiveFinite(double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool unwoundRecoveryEndpoint(
  const moveit::core::RobotModel & model, const moveit::core::JointModelGroup & group,
  const moveit::core::RobotState & state)
{
  for (const auto & name : group.getVariableNames()) {
    const auto & bounds = model.getVariableBounds(name);
    if (model.getJointOfVariable(name)->getType() == moveit::core::JointModel::REVOLUTE &&
      bounds.position_bounded_ && bounds.max_position_ - bounds.min_position_ > 2.0 * kPi)
    {
      const double center = 0.5 * (bounds.min_position_ + bounds.max_position_);
      if (std::abs(state.getVariablePosition(name) - center) > kRecoveryEndpointMaximumOffsetRad) {
        return false;
      }
    }
  }
  return true;
}

// This changes only an IK initial guess. The measured start and returned IK
// solutions retain their bounded joint coordinates throughout the entire path.
Eigen::VectorXd centeredSeed(
  const moveit::core::RobotModel & model, const moveit::core::JointModelGroup & group,
  Eigen::VectorXd positions)
{
  const auto & names = group.getVariableNames();
  for (Eigen::Index index = 0; index < positions.size(); ++index) {
    const auto * joint = model.getJointOfVariable(names[index]);
    const auto & bounds = model.getVariableBounds(names[index]);
    if (!bounds.position_bounded_) {
      continue;
    }
    if (joint->getType() == moveit::core::JointModel::REVOLUTE &&
      bounds.max_position_ - bounds.min_position_ >= 2.0 * kPi)
    {
      const double center = 0.5 * (bounds.min_position_ + bounds.max_position_);
      positions[index] = center + std::remainder(positions[index] - center, 2.0 * kPi);
    }
    positions[index] = std::clamp(positions[index], bounds.min_position_, bounds.max_position_);
  }
  return positions;
}

double goalScore(
  const moveit::core::RobotModel & model, const moveit::core::JointModelGroup & group,
  const Eigen::VectorXd & positions, const Eigen::VectorXd & start)
{
  double score = 0.1 * (positions - start).squaredNorm();
  const auto & names = group.getVariableNames();
  for (Eigen::Index index = 0; index < positions.size(); ++index) {
    const auto & bounds = model.getVariableBounds(names[index]);
    const double half_range = 0.5 * (bounds.max_position_ - bounds.min_position_);
    if (!bounds.position_bounded_ || half_range <= 0.0) {
      continue;
    }
    const double offset = std::abs(
      positions[index] - 0.5 * (bounds.min_position_ + bounds.max_position_));
    const double near_limit = std::max(0.0, (offset / half_range - 0.75) / 0.25);
    score += 2.0 * near_limit * near_limit;
    if (model.getJointOfVariable(names[index])->getType() == moveit::core::JointModel::REVOLUTE &&
      half_range >= kPi)
    {
      const double winding = std::max(0.0, offset - kPi);
      score += 2.0 * winding * winding;
    }
  }
  return score;
}
}  // namespace

struct BackgroundPathPlanner::Impl
{
  struct Request
  {
    moveit::core::RobotState start;
    Eigen::VectorXd goal;
    std::optional<PoseGoal> pose_goal;
    std::uint64_t generation;
    std::uint64_t scene_revision;
  };

  Impl(
    const rclcpp::Node::SharedPtr & node,
    moveit::core::RobotModelConstPtr model,
    planning_scene_monitor::PlanningSceneMonitorPtr monitor,
    BackgroundPathPlannerConfig planner_config, StateValidity validity)
  : robot_model(std::move(model)), scene_monitor(std::move(monitor)),
    config(std::move(planner_config)), additional_validity(std::move(validity))
  {
    if (!node || !robot_model || !scene_monitor || !scene_monitor->getPlanningScene()) {
      throw std::invalid_argument("Background planner requires a node, robot and planning scene");
    }
    group = robot_model->getJointModelGroup(config.move_group_name);
    if (!group || group->getVariableCount() == 0 ||
      scene_monitor->getRobotModel().get() != robot_model.get() ||
      !positiveFinite(config.planning_time_s) ||
      !positiveFinite(config.joint_tolerance_rad) ||
      !positiveFinite(config.validation_joint_step_rad) ||
      !positiveFinite(config.ik_time_budget_s) || config.ik_time_budget_s > 0.1 ||
      config.maximum_validation_samples < 2)
    {
      throw std::invalid_argument("Invalid background planner configuration or robot model");
    }
    // OMPL 2.12 reads declared parameters, not undeclared NodeOptions overrides.
    // Composable controllers deliberately declare their own parameters, so make
    // only this pipeline's supplied configuration visible before plugin loading.
    const std::string prefix = config.pipeline_namespace + ".";
    for (const auto & [name, value] :
      node->get_node_parameters_interface()->get_parameter_overrides())
    {
      if (name.compare(0, prefix.size(), prefix) == 0 && !node->has_parameter(name)) {
        node->declare_parameter(name, value);
      }
    }
    // No request adapters that alter the explicit start; no time parameterization
    // or display/execution adapters. This worker returns geometry only.
    pipeline = std::make_unique<planning_pipeline::PlanningPipeline>(
      robot_model, node, config.pipeline_namespace,
      std::vector<std::string>{"ompl_interface/OMPLPlanner"},
      std::vector<std::string>{}, std::vector<std::string>{});
    const auto manager = pipeline->getPlannerManager("ompl_interface/OMPLPlanner");
    const std::string planner_key = config.move_group_name + "[" + config.planner_id + "]";
    if (!manager || manager->getPlannerConfigurations().count(planner_key) == 0) {
      throw std::invalid_argument(
              "Missing background OMPL configuration '" + planner_key +
              "' under ROS parameter namespace '" + config.pipeline_namespace + "'");
    }
    const auto & settings = manager->getPlannerConfigurations().at(planner_key).config;
    const auto type = settings.find("type");
    if (type == settings.end() || type->second.empty()) {
      throw std::invalid_argument("Background OMPL configuration has no planner type");
    }
    const auto resolution = settings.find("longest_valid_segment_fraction");
    RCLCPP_INFO(
      node->get_logger(), "Background planner configured: %s, type=%s, segment_fraction=%s",
      planner_key.c_str(), type->second.c_str(),
      resolution == settings.end() ? "OMPL default" : resolution->second.c_str());
    worker = std::thread([this]() {run();});
  }

  ~Impl()
  {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      ++latest_generation;
      pending.reset();
    }
    condition.notify_one();
    pipeline->terminate();
    worker.join();
  }

  bool obsolete(std::uint64_t generation) const
  {
    const std::lock_guard<std::mutex> lock(mutex);
    return stopping || generation != latest_generation;
  }

  std::optional<Eigen::VectorXd> selectPoseGoal(
    const Request & request, const planning_scene::PlanningScene & scene)
  {
    const auto & target = *request.pose_goal;
    if (!group->getSolverInstance() || !group->canSetStateFromIK(target.link_name)) {
      return {};
    }
    const auto deadline = SteadyClock::now() +
      std::chrono::duration<double>(config.ik_time_budget_s);
    Eigen::VectorXd start_positions;
    request.start.copyJointGroupPositions(group, start_positions);
    moveit::core::RobotState rest(request.start);
    const auto & state_names = group->getDefaultStateNames();
    if (std::find(state_names.begin(), state_names.end(), config.ik_rest_state_name) !=
      state_names.end())
    {
      rest.setToDefaultValues(group, config.ik_rest_state_name);
    } else {
      std::vector<double> defaults;
      group->getVariableDefaultPositions(defaults);
      rest.setJointGroupPositions(group, defaults);
    }
    Eigen::VectorXd rest_positions;
    rest.copyJointGroupPositions(group, rest_positions);
    const auto & names = group->getVariableNames();
    const auto base = std::find(names.begin(), names.end(), config.ik_base_joint_name);
    const Eigen::Index base_index = base == names.end() ? -1 : std::distance(names.begin(), base);
    if (base_index >= 0) {
      rest_positions[base_index] = target.preferred_base_angle_rad;
    }
    rest_positions = centeredSeed(*robot_model, *group, rest_positions);
    std::vector<Eigen::VectorXd> seeds{rest_positions};
    if (base_index >= 0) {
      for (const double offset : {kPi, -kPi}) {
        Eigen::VectorXd seed = rest_positions;
        seed[base_index] += offset;
        seeds.push_back(centeredSeed(*robot_model, *group, std::move(seed)));
      }
    }
    seeds.push_back(centeredSeed(*robot_model, *group, start_positions));
    // Two additional elbow seeds for the Lite6 chain; they are guesses, never
    // imposed goals. Different groups still have the default/current seeds.
    const auto shoulder = std::find(names.begin(), names.end(), "joint2");
    const auto elbow = std::find(names.begin(), names.end(), "joint3");
    if (shoulder != names.end() && elbow != names.end()) {
      for (const double sign : {-1.0, 1.0}) {
        Eigen::VectorXd seed = rest_positions;
        seed[std::distance(names.begin(), shoulder)] += sign * 0.8;
        seed[std::distance(names.begin(), elbow)] -= sign * 1.2;
        seeds.push_back(centeredSeed(*robot_model, *group, std::move(seed)));
      }
    }
    if (base_index >= 0) {
      Eigen::VectorXd seed = centeredSeed(*robot_model, *group, start_positions);
      seed[base_index] = rest_positions[base_index];
      seeds.push_back(std::move(seed));
    }

    const auto valid_ik = [this, &scene, &request, deadline](
      moveit::core::RobotState * candidate, const moveit::core::JointModelGroup * candidate_group,
      const double * values)
      {
        if (obsolete(request.generation) || SteadyClock::now() >= deadline) {
          return false;
        }
        candidate->setJointGroupPositions(candidate_group, values);
        candidate->update();
        return candidate->satisfiesBounds(candidate_group) &&
               unwoundRecoveryEndpoint(*robot_model, *candidate_group, *candidate) &&
               scene.isStateValid(*candidate, config.move_group_name);
      };
    double best_score = std::numeric_limits<double>::infinity();
    std::optional<Eigen::VectorXd> best;
    for (const auto & seed : seeds) {
      const double remaining = std::chrono::duration<double>(deadline - SteadyClock::now()).count();
      if (remaining <= 0.0 || obsolete(request.generation)) {
        break;
      }
      moveit::core::RobotState candidate(request.start);
      candidate.setJointGroupPositions(group, seed);
      if (!candidate.setFromIK(
          group, target.pose, target.link_name, std::min(0.01, remaining), valid_ik))
      {
        continue;
      }
      candidate.update();
      const auto & achieved = candidate.getGlobalLinkTransform(target.link_name);
      if ((achieved.translation() - target.pose.translation()).norm() > 0.001 ||
        Eigen::AngleAxisd(achieved.linear().transpose() * target.pose.linear()).angle() > 0.001)
      {
        continue;
      }
      Eigen::VectorXd positions;
      candidate.copyJointGroupPositions(group, positions);
      if (!positions.allFinite() || !unwoundRecoveryEndpoint(*robot_model, *group, candidate)) {
        continue;
      }
      double score = goalScore(*robot_model, *group, positions, start_positions);
      if (base_index >= 0) {
        const double base_deviation = positions[base_index] - rest_positions[base_index];
        score += 0.25 * base_deviation * base_deviation;
      }
      if (score < best_score) {
        best_score = score;
        best = std::move(positions);
      }
    }
    return best;
  }

  BackgroundPathResult plan(const Request & request)
  {
    BackgroundPathResult result;
    result.generation = request.generation;
    result.scene_revision = request.scene_revision;
    request.start.copyJointGroupPositions(group, result.start_positions);
    result.goal_positions = request.goal;
    result.error_code = ErrorCode::PLANNING_FAILED;
    if (obsolete(request.generation)) {
      return result;
    }

    planning_scene::PlanningScenePtr scene;
    planning_scene::StateFeasibilityFn prior;
    {
      const planning_scene_monitor::LockedPlanningSceneRO locked_scene(scene_monitor);
      prior = locked_scene->getStateFeasibilityPredicate();
      scene = planning_scene::PlanningScene::clone(locked_scene);
    }
    scene->decoupleParent();
    if (prior || additional_validity) {
      // A raw scene pointer avoids a cycle: the private scene owns its predicate
      // and outlives every use during this synchronous planning call.
      scene->setStateFeasibilityPredicate(
        [prior, extra = additional_validity, snapshot = scene.get()](
          const moveit::core::RobotState & state, bool verbose)
        {
          return (!prior || prior(state, verbose)) && (!extra || extra(*snapshot, state));
        });
    }

    moveit::core::RobotState start(request.start);
    start.update();
    scene->setCurrentState(start);
    if (!start.satisfiesBounds(group) || !scene->isStateValid(start, config.move_group_name)) {
      result.error_code = ErrorCode::START_STATE_IN_COLLISION;
      result.message = "Start violates joint bounds, collision or additional clearance constraints";
      return result;
    }
    moveit::core::RobotState goal(start);
    if (request.pose_goal) {
      const auto positions = selectPoseGoal(request, *scene);
      if (!positions) {
        result.error_code = ErrorCode::NO_IK_SOLUTION;
        result.message = "No valid endpoint IK solution within the bounded candidate search";
        return result;
      }
      result.goal_positions = *positions;
    }
    goal.setJointGroupPositions(group, result.goal_positions);
    goal.update();
    if (!goal.satisfiesBounds(group) || !scene->isStateValid(goal, config.move_group_name)) {
      result.error_code = ErrorCode::GOAL_IN_COLLISION;
      result.message = "Goal violates joint bounds, collision or additional clearance constraints";
      return result;
    }

    planning_interface::MotionPlanRequest motion_request;
    motion_request.group_name = config.move_group_name;
    motion_request.planner_id = config.planner_id;
    motion_request.allowed_planning_time = config.planning_time_s;
    motion_request.num_planning_attempts = 1;
    // MoveIt applies this sampling volume to planar/floating (SE2/SE3) joints
    // only. The fixed-base Lite6 still uses precisely its original joint bounds.
    motion_request.workspace_parameters.header.frame_id = robot_model->getModelFrame();
    motion_request.workspace_parameters.min_corner.x = -10.0;
    motion_request.workspace_parameters.min_corner.y = -10.0;
    motion_request.workspace_parameters.min_corner.z = -10.0;
    motion_request.workspace_parameters.max_corner.x = 10.0;
    motion_request.workspace_parameters.max_corner.y = 10.0;
    motion_request.workspace_parameters.max_corner.z = 10.0;
    moveit::core::robotStateToRobotStateMsg(start, motion_request.start_state);
    motion_request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
        goal, group, config.joint_tolerance_rad, config.joint_tolerance_rad));
    if (obsolete(request.generation)) {
      return result;
    }
    planning_interface::MotionPlanResponse response;
    const bool planned = pipeline->generatePlan(scene, motion_request, response, false);
    result.error_code = response.error_code.val;
    if (!planned || !response || !response.trajectory ||
      response.trajectory->getWayPointCount() == 0)
    {
      result.message = "OMPL did not return a solution";
      return result;
    }
    if (obsolete(request.generation)) {
      return result;
    }

    result.path.reserve(response.trajectory->getWayPointCount() + 1);
    result.path.push_back(result.start_positions);
    moveit::core::RobotState sample(start);
    std::size_t checked_samples = 1;
    for (std::size_t index = 0; index < response.trajectory->getWayPointCount(); ++index) {
      Eigen::VectorXd positions;
      response.trajectory->getWayPoint(index).copyJointGroupPositions(group, positions);
      if (!positions.allFinite()) {
        result.message = "Non-finite position in OMPL path";
        result.path.clear();
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        return result;
      }
      const Eigen::VectorXd previous = result.path.back();
      const double distance = (positions - previous).lpNorm<Eigen::Infinity>();
      if (distance < 1.0e-12) {
        continue;
      }
      const double required_samples = std::ceil(distance / config.validation_joint_step_rad);
      if (required_samples > static_cast<double>(
          config.maximum_validation_samples - checked_samples))
      {
        result.message = "Path exceeds the bounded validation sample budget";
        result.path.clear();
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        return result;
      }
      const std::size_t subdivisions = static_cast<std::size_t>(required_samples);
      for (std::size_t step = 1; step <= subdivisions; ++step) {
        if (obsolete(request.generation)) {
          return result;
        }
        const double fraction = static_cast<double>(step) / static_cast<double>(subdivisions);
        // Deliberately linear in the bounded joint coordinates used by the
        // executor; a shortest-angle interpolation could hide an invalid wrap.
        sample.setJointGroupPositions(group, previous + fraction * (positions - previous));
        sample.update();
        if (!sample.satisfiesBounds(group) ||
          !scene->isStateValid(sample, config.move_group_name))
        {
          result.message = "An interpolated path sample violates safety constraints";
          result.path.clear();
          result.error_code = ErrorCode::INVALID_MOTION_PLAN;
          return result;
        }
        ++checked_samples;
      }
      result.path.push_back(std::move(positions));
    }
    if ((result.path.back() - result.goal_positions).lpNorm<Eigen::Infinity>() >
      config.joint_tolerance_rad + 1.0e-6)
    {
      result.message = "OMPL path does not reach the requested joint goal tolerance";
      result.path.clear();
      result.error_code = ErrorCode::INVALID_MOTION_PLAN;
      return result;
    }
    result.success = true;
    result.error_code = ErrorCode::SUCCESS;
    result.message = "Geometric path validated on the worker scene snapshot";
    return result;
  }

  void run()
  {
    while (true) {
      std::optional<Request> request;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]() {return stopping || pending.has_value();});
        if (stopping) {
          return;
        }
        request = std::move(pending);
        pending.reset();
        active = true;
      }
      const auto begin = SteadyClock::now();
      BackgroundPathResult result;
      try {
        result = plan(*request);
      } catch (const std::exception & exception) {
        result.generation = request->generation;
        result.scene_revision = request->scene_revision;
        request->start.copyJointGroupPositions(group, result.start_positions);
        result.goal_positions = request->goal;
        result.error_code = ErrorCode::FAILURE;
        result.message = exception.what();
      }
      result.elapsed_time_s = std::chrono::duration<double>(SteadyClock::now() - begin).count();
      {
        const std::lock_guard<std::mutex> lock(mutex);
        active = false;
        if (!stopping && request->generation == latest_generation) {
          completed = std::move(result);
        }
      }
    }
  }

  moveit::core::RobotModelConstPtr robot_model;
  planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor;
  BackgroundPathPlannerConfig config;
  StateValidity additional_validity;
  const moveit::core::JointModelGroup * group{nullptr};
  std::unique_ptr<planning_pipeline::PlanningPipeline> pipeline;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::optional<Request> pending;
  std::optional<BackgroundPathResult> completed;
  std::uint64_t latest_generation{0};
  bool active{false};
  bool stopping{false};
  std::thread worker;
};

BackgroundPathPlanner::BackgroundPathPlanner(
  const rclcpp::Node::SharedPtr & node, moveit::core::RobotModelConstPtr robot_model,
  planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor,
  BackgroundPathPlannerConfig config, StateValidity additional_validity)
: impl_(std::make_unique<Impl>(
      node, std::move(robot_model), std::move(scene_monitor), std::move(config),
      std::move(additional_validity)))
{
}

BackgroundPathPlanner::~BackgroundPathPlanner() = default;

std::uint64_t BackgroundPathPlanner::submit(
  const moveit::core::RobotState & start, const Eigen::VectorXd & goal_positions,
  std::uint64_t scene_revision)
{
  if (start.getRobotModel().get() != impl_->robot_model.get() ||
    goal_positions.size() != static_cast<Eigen::Index>(impl_->group->getVariableCount()) ||
    !goal_positions.allFinite())
  {
    throw std::invalid_argument("Background planner request has a different model or invalid goal");
  }
  Eigen::VectorXd start_positions;
  start.copyJointGroupPositions(impl_->group, start_positions);
  if (!start_positions.allFinite()) {
    throw std::invalid_argument("Background planner start contains non-finite joint positions");
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::uint64_t generation = ++impl_->latest_generation;
  impl_->pending.emplace(Impl::Request{start, goal_positions, {}, generation, scene_revision});
  impl_->completed.reset();
  impl_->condition.notify_one();
  return generation;
}

std::uint64_t BackgroundPathPlanner::submit(
  const moveit::core::RobotState & start, const PoseGoal & goal, std::uint64_t scene_revision)
{
  if (start.getRobotModel().get() != impl_->robot_model.get() ||
    !impl_->robot_model->hasLinkModel(goal.link_name) || !goal.pose.matrix().allFinite() ||
    !goal.pose.linear().isUnitary(1.0e-6) || goal.pose.linear().determinant() < 0.0 ||
    !goal.pose.matrix().row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)) ||
    !std::isfinite(goal.preferred_base_angle_rad))
  {
    throw std::invalid_argument("Background planner pose goal has an invalid model, link or pose");
  }
  Eigen::VectorXd start_positions;
  start.copyJointGroupPositions(impl_->group, start_positions);
  if (!start_positions.allFinite()) {
    throw std::invalid_argument("Background planner start contains non-finite joint positions");
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::uint64_t generation = ++impl_->latest_generation;
  impl_->pending.emplace(Impl::Request{start, {}, goal, generation, scene_revision});
  impl_->completed.reset();
  impl_->condition.notify_one();
  return generation;
}

void BackgroundPathPlanner::cancel()
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  ++impl_->latest_generation;
  impl_->pending.reset();
  impl_->completed.reset();
}

bool BackgroundPathPlanner::isBusy() const
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->active || impl_->pending.has_value();
}

std::optional<BackgroundPathResult> BackgroundPathPlanner::takeResult()
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  auto result = std::move(impl_->completed);
  impl_->completed.reset();
  return result;
}

}  // namespace face_tracking_arm::control
