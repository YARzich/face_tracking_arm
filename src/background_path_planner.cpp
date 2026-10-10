// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/background_path_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/compact_upright_path.hpp"

#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace face_tracking_arm::control
{
namespace
{
using ErrorCode = moveit_msgs::msg::MoveItErrorCodes;
using SteadyClock = std::chrono::steady_clock;
constexpr double kPi = 3.14159265358979323846;
constexpr double kCompactValidationJointStepRad = 0.01;
constexpr std::size_t kMaximumCompactValidationSamples = 8192;
constexpr double kStartBoundNormalizationTolerance = 1.0e-9;
constexpr double kMaximumStagedEarlyYawShiftRad = 0.35;
constexpr double kMaximumStagedCompactRadiusM = 0.40;
constexpr double kStagedProfileJointToleranceRad = 1.0e-6;
constexpr double kStagedProfileQ6ToleranceRad = 0.0201;

bool positiveFinite(double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool normalizeStartWithinBounds(
  moveit::core::RobotState & start, const moveit::core::JointModelGroup * group)
{
  Eigen::VectorXd positions;
  start.copyJointGroupPositions(group, positions);
  if (!positions.allFinite()) {
    return false;
  }
  const auto & names = group->getVariableNames();
  if (positions.size() != static_cast<Eigen::Index>(names.size())) {
    return false;
  }
  for (Eigen::Index index = 0; index < positions.size(); ++index) {
    const auto & bounds = start.getRobotModel()->getVariableBounds(
      names[static_cast<std::size_t>(index)]);
    if (!bounds.position_bounded_) {
      continue;
    }
    if (positions[index] < bounds.min_position_) {
      if (bounds.min_position_ - positions[index] > kStartBoundNormalizationTolerance) {
        return false;
      }
      positions[index] = bounds.min_position_;
    } else if (positions[index] > bounds.max_position_) {
      if (positions[index] - bounds.max_position_ > kStartBoundNormalizationTolerance) {
        return false;
      }
      positions[index] = bounds.max_position_;
    }
  }
  start.setJointGroupPositions(group, positions);
  start.update();
  return start.satisfiesBounds(group);
}

std::optional<double> uprightRollError(
  const moveit::core::RobotState & state, const std::string & screen_link_name)
{
  const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(screen_link_name).linear();
  const auto pose = evaluateCompactUprightPose(rotation);
  return pose ? std::optional<double>(std::abs(pose->signed_roll_error_rad)) : std::nullopt;
}

std::optional<double> screenPitch(
  const moveit::core::RobotState & state, const std::string & screen_link_name)
{
  const auto pose = evaluateCompactUprightPose(
    state.getGlobalLinkTransform(screen_link_name).linear());
  return pose ? std::optional<double>(pose->signed_pitch_rad) : std::nullopt;
}

bool addCompactOrientationGuide(
  moveit::core::RobotState const & start, const PathRequirements & requirements,
  planning_interface::MotionPlanRequest & motion_request)
{
  if (!requirements.maximum_abs_screen_pitch_rad) {
    return true;
  }
  const auto guide_bounds = compactOrientationGuideBounds(
    *requirements.maximum_abs_screen_pitch_rad, requirements.maximum_roll_error_rad);
  if (!guide_bounds) {
    return false;
  }

  const auto pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform(requirements.screen_link_name).linear());
  if (!pose) {
    return false;
  }
  const double yaw = std::atan2(pose->screen_normal.y(), pose->screen_normal.x());
  const Eigen::Matrix3d target_rotation =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  moveit_msgs::msg::OrientationConstraint constraint;
  constraint.header.frame_id = start.getRobotModel()->getModelFrame();
  constraint.link_name = requirements.screen_link_name;
  constraint.orientation = tf2::toMsg(Eigen::Quaterniond(target_rotation));
  constexpr double kGuideNumericalMargin = 1.0e-4;
  constraint.absolute_x_axis_tolerance =
    guide_bounds->absolute_x_axis_tolerance_rad + kGuideNumericalMargin;
  constraint.absolute_y_axis_tolerance =
    guide_bounds->absolute_y_axis_tolerance_rad + kGuideNumericalMargin;
  // Rotation about world up is free. Exact screen-normal-axis roll is still checked
  // by evaluateCompactUprightPose on every planner and execution-path sample.
  constraint.absolute_z_axis_tolerance = kPi;
  constraint.weight = 1.0;
  motion_request.path_constraints.orientation_constraints.push_back(constraint);
  return true;
}


}  // namespace

struct BackgroundPathPlanner::Impl
{
  struct Request
  {
    moveit::core::RobotState start;
    Eigen::VectorXd goal;
    std::optional<CompactGoal> compact_goal;
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
    if (!group || group->getJointModels().empty()) {
      throw std::invalid_argument("Background planner joint group is empty or unavailable");
    }
    const auto & variable_names = group->getVariableNames();
    const auto yaw_joint = std::find(variable_names.begin(), variable_names.end(), "joint1");
    const auto reset_joint = std::find(variable_names.begin(), variable_names.end(), "joint4");
    const auto wrist_pitch_joint = std::find(variable_names.begin(), variable_names.end(),
        "joint5");
    const auto joint6 = std::find(variable_names.begin(), variable_names.end(), "joint6");
    const auto index_of = [&variable_names](const auto iterator) -> std::optional<std::size_t> {
        if (iterator == variable_names.end()) {
          return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(variable_names.begin(), iterator));
      };
    staged_yaw_index = index_of(yaw_joint);
    staged_reset_index = index_of(reset_joint);
    staged_wrist_pitch_index = index_of(wrist_pitch_joint);
    staged_joint6_index = index_of(joint6);
    const auto * first_joint_parent = group->getJointModels().front()->getParentLinkModel();
    if (!first_joint_parent) {
      throw std::invalid_argument("Background planner group has no fixed-base parent link");
    }
    base_link_name = first_joint_parent->getName();
    try {
      compact_geometry = std::make_unique<CompactGeometryValidator>(robot_model, base_link_name);
    } catch (const std::exception & exception) {
      compact_geometry_error = exception.what();
      RCLCPP_WARN(
        node->get_logger(), "Compact path geometry is unavailable for model '%s': %s",
        robot_model->getName().c_str(), compact_geometry_error.c_str());
    }
    if (group->getVariableCount() == 0 ||
      scene_monitor->getRobotModel().get() != robot_model.get() ||
      !positiveFinite(config.planning_time_s) ||
      !positiveFinite(config.compact_planning_time_s) ||
      !positiveFinite(config.joint_tolerance_rad) ||
      !positiveFinite(config.validation_joint_step_rad) ||
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
    // Compact requests add a native orientation guide but keep bounded joint
    // interpolation. Requests without path constraints remain ordinary joint-space plans.
    const std::string joint_space_parameter = prefix + config.move_group_name +
      ".enforce_joint_model_state_space";
    if (!node->has_parameter(joint_space_parameter)) {
      node->declare_parameter(joint_space_parameter, true);
    } else if (!node->get_parameter(joint_space_parameter).as_bool()) {
      node->set_parameter(rclcpp::Parameter(joint_space_parameter, true));
    }
    // No request adapters that alter the explicit start; no time parameterization
    // or display/execution adapters. This worker returns geometry only.
    pipeline = std::make_unique<planning_pipeline::PlanningPipeline>(
      robot_model, node, config.pipeline_namespace,
      std::vector<std::string>{"ompl_interface/OMPLPlanner"},
      std::vector<std::string>{}, std::vector<std::string>{});
    planner_manager = pipeline->getPlannerManager("ompl_interface/OMPLPlanner");
    planner_key = config.move_group_name + "[" + config.planner_id + "]";
    if (!planner_manager || planner_manager->getPlannerConfigurations().count(planner_key) == 0) {
      throw std::invalid_argument(
              "Missing background OMPL configuration '" + planner_key +
              "' under ROS parameter namespace '" + config.pipeline_namespace + "'");
    }
    default_planner_configurations = planner_manager->getPlannerConfigurations();
    const auto & settings = default_planner_configurations.at(planner_key).config;
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

  BackgroundPathResult plan(const Request & request)
  {
    const auto plan_start = SteadyClock::now();
    BackgroundPathResult result;
    result.generation = request.generation;
    result.scene_revision = request.scene_revision;
    result.goal_positions = request.goal;
    if (request.compact_goal) {
      result.profile_id = request.compact_goal->profile_id;
      PathValidationMetrics metrics;
      metrics.effective_radius_cap_m =
        request.compact_goal->path_requirements.maximum_assembly_radius_m;
      metrics.minimum_robust_clearance_m = std::numeric_limits<double>::quiet_NaN();
      result.validation_metrics = metrics;
      if (!compact_geometry) {
        result.error_code = ErrorCode::FAILURE;
        result.message = "Compact path geometry is unavailable: " + compact_geometry_error;
        return result;
      }
    }
    std::vector<CompactPathSampleCertificate> compact_certificate_samples;
    std::optional<PathValidationMetrics> exact_compact_goal_metrics;
    std::optional<double> compact_start_roll_error;
    std::optional<double> compact_start_pitch;
    double compact_start_radius = std::numeric_limits<double>::quiet_NaN();
    const auto append_compact_certificate_sample =
      [this, &request, &compact_certificate_samples](
      const Eigen::VectorXd & positions, const std::size_t segment_index,
      const double fraction, const double radius, const double roll,
      const double pitch) -> bool
      {
        if (!request.compact_goal) {
          return true;
        }
        if (compact_certificate_samples.size() >= kMaximumCompactValidationSamples ||
          positions.size() != static_cast<Eigen::Index>(group->getVariableCount()) ||
          !positions.allFinite() || !std::isfinite(fraction) || fraction < 0.0 ||
          fraction > 1.0 || !std::isfinite(radius) || !std::isfinite(roll) ||
          !std::isfinite(pitch))
        {
          return false;
        }
        CompactPathSampleCertificate sample;
        sample.positions = positions;
        sample.original_segment_index = segment_index;
        sample.fraction = fraction;
        sample.maximum_assembly_radius_m = radius;
        sample.maximum_roll_error_rad = roll;
        sample.maximum_abs_screen_pitch_rad = pitch;
        compact_certificate_samples.push_back(std::move(sample));
        return true;
      };
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
    const auto compact_valid = [this, &request](
      const moveit::core::RobotState & state) -> bool
      {
        if (!request.compact_goal) {
          return true;
        }
        const auto & requirements = request.compact_goal->path_requirements;
        const auto roll_error = uprightRollError(state, requirements.screen_link_name);
        const auto pitch = screenPitch(state, requirements.screen_link_name);
        const double radius = compact_geometry->maximumAssemblyRadius(state);
        const bool pitch_valid = !requirements.maximum_abs_screen_pitch_rad ||
          (pitch && std::abs(*pitch) <= *requirements.maximum_abs_screen_pitch_rad +
          kCompactPitchValidationToleranceRad);
        return roll_error && pitch && std::isfinite(radius) && pitch_valid &&
               *roll_error <= requirements.maximum_roll_error_rad &&
               radius <= requirements.maximum_assembly_radius_m;
      };
    if (prior || additional_validity) {
      // A raw scene pointer avoids a cycle: the private scene owns its predicate
      // and outlives every use during this synchronous planning call.
      scene->setStateFeasibilityPredicate(
        [prior, extra = additional_validity, compact_valid, snapshot = scene.get()](
          const moveit::core::RobotState & state, bool verbose)
        {
          return (!prior || prior(state, verbose)) && (!extra || extra(*snapshot, state)) &&
                 compact_valid(state);
        });
    } else if (request.compact_goal) {
      scene->setStateFeasibilityPredicate(
        [compact_valid](const moveit::core::RobotState & state, bool) {
          return compact_valid(state);
        });
    }

    moveit::core::RobotState start(request.start);
    start.update();
    if (!normalizeStartWithinBounds(start, group)) {
      result.error_code = ErrorCode::START_STATE_INVALID;
      result.message = "Start violates bounded joint limits";
      return result;
    }
    start.copyJointGroupPositions(group, result.start_positions);
    scene->setCurrentState(start);
    if (request.compact_goal) {
      const auto & requirements = request.compact_goal->path_requirements;
      const auto roll = uprightRollError(start, requirements.screen_link_name);
      const auto pitch = screenPitch(start, requirements.screen_link_name);
      const double radius = compact_geometry->maximumAssemblyRadius(start);
      compact_start_roll_error = roll;
      compact_start_pitch = pitch;
      compact_start_radius = radius;
      if (roll) {
        result.validation_metrics->maximum_roll_error_rad = std::abs(*roll);
      }
      if (pitch) {
        result.validation_metrics->maximum_abs_screen_pitch_rad = std::abs(*pitch);
      }
      result.validation_metrics->maximum_assembly_radius_m = radius;
      if (!roll || !pitch || !std::isfinite(radius) ||
        std::abs(*roll) > requirements.maximum_roll_error_rad ||
        (requirements.maximum_abs_screen_pitch_rad &&
        std::abs(*pitch) > *requirements.maximum_abs_screen_pitch_rad +
        kCompactPitchValidationToleranceRad) ||
        radius > requirements.maximum_assembly_radius_m)
      {
        result.error_code = ErrorCode::START_STATE_INVALID;
        result.message = "Compact path start violates its pitch, roll or radius requirement";
        return result;
      }
      result.validation_metrics->checked_samples = 1;
    }
    if (!scene->isStateValid(start, config.move_group_name)) {
      result.error_code = ErrorCode::START_STATE_IN_COLLISION;
      result.message = "Start violates collision or additional clearance constraints";
      return result;
    }
    if (request.compact_goal && (!compact_start_roll_error || !compact_start_pitch ||
      !append_compact_certificate_sample(
        result.start_positions, 0U, 0.0, compact_start_radius,
        *compact_start_roll_error, *compact_start_pitch)))
    {
      result.error_code = ErrorCode::INVALID_MOTION_PLAN;
      result.message = "Compact start certificate is malformed or exceeds its sample budget";
      return result;
    }
    moveit::core::RobotState goal(start);
    goal.setJointGroupPositions(group, result.goal_positions);
    goal.update();
    if (!goal.satisfiesBounds(group)) {
      result.error_code = ErrorCode::GOAL_IN_COLLISION;
      result.message = "Goal violates bounded joint limits";
      return result;
    }
    if (request.compact_goal) {
      const auto & requirements = request.compact_goal->path_requirements;
      const auto roll = uprightRollError(goal, requirements.screen_link_name);
      const auto pitch = screenPitch(goal, requirements.screen_link_name);
      const double radius = compact_geometry->maximumAssemblyRadius(goal);
      if (roll) {
        result.validation_metrics->maximum_roll_error_rad = std::max(
          result.validation_metrics->maximum_roll_error_rad, std::abs(*roll));
      }
      if (pitch) {
        result.validation_metrics->maximum_abs_screen_pitch_rad = std::max(
          result.validation_metrics->maximum_abs_screen_pitch_rad, std::abs(*pitch));
      }
      if (std::isfinite(radius)) {
        result.validation_metrics->maximum_assembly_radius_m = std::max(
          result.validation_metrics->maximum_assembly_radius_m, radius);
      }
      if (!roll || !pitch || !std::isfinite(radius) ||
        std::abs(*roll) > requirements.maximum_roll_error_rad ||
        (requirements.maximum_abs_screen_pitch_rad &&
        std::abs(*pitch) > *requirements.maximum_abs_screen_pitch_rad +
        kCompactPitchValidationToleranceRad) ||
        radius > requirements.maximum_assembly_radius_m)
      {
        result.error_code = ErrorCode::GOAL_IN_COLLISION;
        result.message = "Compact goal violates its pitch, roll or radius requirement";
        return result;
      }
      PathValidationMetrics goal_metrics;
      goal_metrics.maximum_roll_error_rad = std::abs(*roll);
      goal_metrics.maximum_abs_screen_pitch_rad = std::abs(*pitch);
      goal_metrics.maximum_assembly_radius_m = radius;
      goal_metrics.effective_radius_cap_m = requirements.maximum_assembly_radius_m;
      goal_metrics.checked_samples = 1U;
      exact_compact_goal_metrics = goal_metrics;
    }
    if (!scene->isStateValid(goal, config.move_group_name)) {
      result.error_code = ErrorCode::GOAL_IN_COLLISION;
      result.message = "Goal violates collision or additional clearance constraints";
      return result;
    }

    Eigen::VectorXd planning_goal_positions = result.goal_positions;
    bool staged_route_active = false;
    bool staged_route_direct_complete = false;
    bool staged_prefix_available = false;
    bool compact_radius_entered = false;
    auto append_direct_segment = [this, &request, &scene, &result,
        &compact_certificate_samples,
        &append_compact_certificate_sample](
      const Eigen::VectorXd & from, const Eigen::VectorXd & to,
      const double segment_radius_cap, const double compact_floor,
      bool & compact_entered, std::vector<Eigen::VectorXd> & path) -> bool
      {
        const double distance = (to - from).lpNorm<Eigen::Infinity>();
        const std::size_t subdivisions = std::max<std::size_t>(
          1U, static_cast<std::size_t>(std::ceil(
            distance / kCompactValidationJointStepRad)));
        if (subdivisions > kMaximumCompactValidationSamples ||
          result.validation_metrics->checked_samples >
          kMaximumCompactValidationSamples - subdivisions ||
          compact_certificate_samples.size() >
          kMaximumCompactValidationSamples - subdivisions)
        {
          return false;
        }
        moveit::core::RobotState sample(request.start);
        const std::size_t segment_index = path.empty() ? 0U : path.size() - 1U;
        for (std::size_t index = 1; index <= subdivisions; ++index) {
          if (obsolete(request.generation)) {
            return false;
          }
          const double fraction = static_cast<double>(index) /
            static_cast<double>(subdivisions);
          const Eigen::VectorXd positions = from + fraction * (to - from);
          sample.setJointGroupPositions(group, positions);
          sample.update();
          const double radius = compact_geometry->maximumAssemblyRadius(sample);
          if (std::isfinite(radius) && radius <= compact_floor + 1.0e-9) {
            compact_entered = true;
          } else if (compact_entered) {
            return false;
          }
          if (!sample.satisfiesBounds(group) || !scene->isStateValid(
              sample, config.move_group_name) || !std::isfinite(radius) ||
            radius > segment_radius_cap + 1.0e-9)
          {
            return false;
          }
          const auto roll = uprightRollError(
            sample, request.compact_goal->path_requirements.screen_link_name);
          const auto pitch = screenPitch(
            sample, request.compact_goal->path_requirements.screen_link_name);
          if (!roll || !pitch ||
            std::abs(*roll) > request.compact_goal->path_requirements.maximum_roll_error_rad ||
            (request.compact_goal->path_requirements.maximum_abs_screen_pitch_rad &&
            std::abs(*pitch) >
            *request.compact_goal->path_requirements.maximum_abs_screen_pitch_rad +
            kCompactPitchValidationToleranceRad))
          {
            return false;
          }
          result.validation_metrics->maximum_roll_error_rad = std::max(
            result.validation_metrics->maximum_roll_error_rad, std::abs(*roll));
          result.validation_metrics->maximum_abs_screen_pitch_rad = std::max(
            result.validation_metrics->maximum_abs_screen_pitch_rad, std::abs(*pitch));
          result.validation_metrics->maximum_assembly_radius_m = std::max(
            result.validation_metrics->maximum_assembly_radius_m, radius);
          ++result.validation_metrics->checked_samples;
          if (!append_compact_certificate_sample(
              positions, segment_index, fraction, radius, std::abs(*roll), std::abs(*pitch)))
          {
            return false;
          }
        }
        path.push_back(to);
        return true;
      };
    const bool staged_profile_requested = request.compact_goal &&
      (request.compact_goal->profile_id == "compact_turn_negative_wrist" ||
      request.compact_goal->profile_id == "compact_turn_positive_wrist") &&
      request.compact_goal->path_requirements.staged_route.has_value();
    const bool supported_staged_model = staged_profile_requested &&
      robot_model->getName() == "xarm6" && group->getVariableCount() == 6 &&
      staged_yaw_index && staged_reset_index && staged_wrist_pitch_index && staged_joint6_index;
    bool known_staged_profile = false;
    if (supported_staged_model) {
      const Eigen::VectorXd & committed_goal = result.goal_positions;
      moveit::core::RobotState named_profile(robot_model);
      if (!named_profile.setToDefaultValues(group, request.compact_goal->profile_id)) {
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        result.message = "Staged compact profile is missing its named SRDF posture";
        return result;
      }
      Eigen::VectorXd expected_positions;
      named_profile.copyJointGroupPositions(group, expected_positions);
      known_staged_profile = expected_positions.size() == committed_goal.size();
      for (Eigen::Index index = 0; known_staged_profile && index < committed_goal.size(); ++index) {
        double difference = committed_goal[index] - expected_positions[index];
        double tolerance = kStagedProfileJointToleranceRad;
        if (index == static_cast<Eigen::Index>(*staged_joint6_index)) {
          difference = std::remainder(difference, 2.0 * kPi);
          tolerance = kStagedProfileQ6ToleranceRad;
        }
        known_staged_profile = std::isfinite(difference) &&
          std::abs(difference) <= tolerance;
      }
      if (!known_staged_profile) {
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        result.message = "Staged compact goal does not match its named SRDF posture";
        return result;
      }
    }
    if (known_staged_profile) {
      const auto & staged = *request.compact_goal->path_requirements.staged_route;
      Eigen::VectorXd prefix = result.goal_positions;
      const Eigen::Index yaw_index = static_cast<Eigen::Index>(*staged_yaw_index);
      const Eigen::Index reset_index = static_cast<Eigen::Index>(*staged_reset_index);
      const double yaw_delta = prefix[yaw_index] - result.start_positions[yaw_index];
      prefix[yaw_index] = result.start_positions[yaw_index] + std::clamp(
        yaw_delta, -staged.maximum_early_yaw_shift_rad,
        staged.maximum_early_yaw_shift_rad);
      prefix[reset_index] = result.start_positions[reset_index];

      moveit::core::RobotState prefix_state(start);
      prefix_state.setJointGroupPositions(group, prefix);
      prefix_state.update();
      const double prefix_radius = compact_geometry->maximumAssemblyRadius(prefix_state);
      staged_prefix_available = prefix_state.satisfiesBounds(group) &&
        std::isfinite(prefix_radius) && prefix_radius <= staged.maximum_compact_radius_m &&
        scene->isStateValid(prefix_state, config.move_group_name);
      staged_route_active = true;
      planning_goal_positions = staged_prefix_available ? prefix : result.goal_positions;
      const double start_radius = compact_geometry->maximumAssemblyRadius(start);
      compact_radius_entered = std::isfinite(start_radius) &&
        start_radius <= staged.maximum_compact_radius_m;
      const auto metrics_before_direct = result.validation_metrics;
      const bool compact_entered_before_direct = compact_radius_entered;
      const std::size_t certificate_samples_before_direct =
        compact_certificate_samples.size();
      if (staged_prefix_available) {
        std::vector<Eigen::VectorXd> direct_path{result.start_positions};
        bool direct_valid = append_direct_segment(
          result.start_positions, prefix,
          request.compact_goal->path_requirements.maximum_assembly_radius_m,
          staged.maximum_compact_radius_m, compact_radius_entered, direct_path);
        if (direct_valid) {
          direct_valid = append_direct_segment(
            direct_path.back(), result.goal_positions,
            staged.maximum_compact_radius_m, staged.maximum_compact_radius_m,
            compact_radius_entered, direct_path);
        }
        if (direct_valid) {
          result.path = std::move(direct_path);
          staged_route_direct_complete = true;
          result.message = "Direct staged compact route validated";
        }
      }
      if (!staged_route_direct_complete) {
        result.validation_metrics = metrics_before_direct;
        compact_radius_entered = compact_entered_before_direct;
        compact_certificate_samples.resize(certificate_samples_before_direct);
        const auto projection_cancelled = [&]() {
            return obsolete(request.generation) ||
                   std::chrono::duration<double>(SteadyClock::now() - plan_start).count() >=
                   config.compact_planning_time_s;
          };
        CompactStagedProjectionConfig projection_config;
        projection_config.yaw_joint_index = *staged_yaw_index;
        projection_config.reset_joint_index = *staged_reset_index;
        projection_config.wrist_pitch_joint_index = *staged_wrist_pitch_index;
        projection_config.wrist_roll_joint_index = *staged_joint6_index;
        projection_config.maximum_early_yaw_shift_rad = staged.maximum_early_yaw_shift_rad;
        projection_config.maximum_abs_pitch_rad =
          request.compact_goal->path_requirements.maximum_abs_screen_pitch_rad.value_or(0.5 * kPi);
        projection_config.maximum_roll_error_rad =
          request.compact_goal->path_requirements.maximum_roll_error_rad;
        projection_config.maximum_joint_step_rad = kCompactValidationJointStepRad;
        projection_config.maximum_samples = kMaximumCompactValidationSamples;
        const auto projected_path = projectCompactStagedPath(
          start, result.goal_positions, group,
          request.compact_goal->path_requirements.screen_link_name,
          projection_config, projection_cancelled);
        if (projected_path && !obsolete(request.generation)) {
          std::vector<Eigen::VectorXd> validated_path{result.start_positions};
          bool valid = projected_path->fold_end_index > 0U &&
            projected_path->fold_end_index < projected_path->waypoints.size();
          if (valid) {
            moveit::core::RobotState fold_end_state(start);
            fold_end_state.setJointGroupPositions(
              group, projected_path->waypoints[projected_path->fold_end_index]);
            fold_end_state.update();
            const double fold_end_radius =
              compact_geometry->maximumAssemblyRadius(fold_end_state);
            valid = std::isfinite(fold_end_radius) &&
              fold_end_radius <= staged.maximum_compact_radius_m + 1.0e-9;
          }
          for (std::size_t index = 1; valid && index < projected_path->waypoints.size(); ++index) {
            if (obsolete(request.generation)) {
              return result;
            }
            const double segment_cap = compact_radius_entered ?
              staged.maximum_compact_radius_m :
              request.compact_goal->path_requirements.maximum_assembly_radius_m;
            if (!append_direct_segment(
                projected_path->waypoints[index - 1U], projected_path->waypoints[index],
                segment_cap, staged.maximum_compact_radius_m,
                compact_radius_entered, validated_path))
            {
              valid = false;
              break;
            }
          }
          if (valid && validated_path.back().isApprox(result.goal_positions, 1.0e-12)) {
            staged_route_direct_complete = true;
            result.path = std::move(validated_path);
            result.message = "Projected upright staged compact route validated";
          }
        }
        if (!staged_route_direct_complete) {
          result.validation_metrics = metrics_before_direct;
          compact_radius_entered = compact_entered_before_direct;
          compact_certificate_samples.resize(certificate_samples_before_direct);
          result.path.clear();
        }
      }
      if (!staged_route_direct_complete && !staged_prefix_available) {
        const double elapsed_planning_sec =
          std::chrono::duration<double>(SteadyClock::now() - plan_start).count();
        if (elapsed_planning_sec >= config.compact_planning_time_s) {
          result.error_code = ErrorCode::TIMED_OUT;
          result.message = "Compact planning time budget exhausted";
        } else {
          result.error_code = ErrorCode::INVALID_MOTION_PLAN;
          result.message = "Validated staged compact prefix is unavailable";
        }
        return result;
      }
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
    moveit::core::RobotState planning_goal(start);
    planning_goal.setJointGroupPositions(group, planning_goal_positions);
    planning_goal.update();
    motion_request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
        planning_goal, group, config.joint_tolerance_rad, config.joint_tolerance_rad));
    // This conservative MoveIt orientation box guides constrained sampling.
    // The worker feasibility predicate and explicit path validator remain the
    // authority for physical pitch, gravity-upright roll and assembly radius.
    if (request.compact_goal && !addCompactOrientationGuide(
        start, request.compact_goal->path_requirements, motion_request))
    {
      result.error_code = ErrorCode::FAILURE;
      result.message = "Compact orientation guide could not be constructed";
      return result;
    }
    if (obsolete(request.generation)) {
      return result;
    }
    planning_interface::MotionPlanResponse response;
    if (!staged_route_direct_complete) {
      bool planned = false;
      if (request.compact_goal) {
        const double elapsed_planning_sec =
          std::chrono::duration<double>(SteadyClock::now() - plan_start).count();
        const double remaining_planning_sec =
          config.compact_planning_time_s - elapsed_planning_sec;
        if (!std::isfinite(remaining_planning_sec) || remaining_planning_sec <= 0.0) {
          result.error_code = ErrorCode::TIMED_OUT;
          result.message = "Compact planning time budget exhausted";
          return result;
        }
        motion_request.allowed_planning_time = remaining_planning_sec;
        auto compact_configurations = default_planner_configurations;
        auto compact_config = compact_configurations.find(planner_key);
        if (compact_config == compact_configurations.end()) {
          result.error_code = ErrorCode::FAILURE;
          result.message = "Compact planner configuration is unavailable";
          return result;
        }
        // OMPL reads this through boost::lexical_cast<bool>, which accepts the
        // numeric representation used by MoveIt planner configs (0/1).
        compact_config->second.config["simplify_solutions"] = "0";
        planner_manager->setPlannerConfigurations(compact_configurations);
        try {
          planned = pipeline->generatePlan(scene, motion_request, response, false);
        } catch (...) {
          planner_manager->setPlannerConfigurations(default_planner_configurations);
          throw;
        }
        planner_manager->setPlannerConfigurations(default_planner_configurations);
      } else {
        planned = pipeline->generatePlan(scene, motion_request, response, false);
      }
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
        const std::size_t original_segment_index = result.path.size() - 1U;
        const double distance = (positions - previous).lpNorm<Eigen::Infinity>();
        if (distance < 1.0e-12) {
          continue;
        }
        const double validation_step_rad = request.compact_goal ?
          kCompactValidationJointStepRad : config.validation_joint_step_rad;
        const std::size_t maximum_samples = request.compact_goal ?
          kMaximumCompactValidationSamples : config.maximum_validation_samples;
        const double required_samples = std::ceil(distance / validation_step_rad);
        if (required_samples > static_cast<double>(maximum_samples - checked_samples)) {
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
          const double fraction = static_cast<double>(step) /
            static_cast<double>(subdivisions);
          // Deliberately linear in the bounded joint coordinates used by the
          // executor; a shortest-angle interpolation could hide an invalid wrap.
          const Eigen::VectorXd sampled_positions = previous +
            fraction * (positions - previous);
          sample.setJointGroupPositions(group, sampled_positions);
          sample.update();
          if (!sample.satisfiesBounds(group) ||
            !scene->isStateValid(sample, config.move_group_name))
          {
            result.message = "An interpolated path sample violates safety constraints";
            result.path.clear();
            result.error_code = ErrorCode::INVALID_MOTION_PLAN;
            return result;
          }
          if (request.compact_goal) {
            const auto & requirements = request.compact_goal->path_requirements;
            const auto roll = uprightRollError(sample, requirements.screen_link_name);
            const auto pitch = screenPitch(sample, requirements.screen_link_name);
            const double radius = compact_geometry->maximumAssemblyRadius(sample);
            if (roll) {
              result.validation_metrics->maximum_roll_error_rad = std::max(
                result.validation_metrics->maximum_roll_error_rad, std::abs(*roll));
            }
            if (pitch) {
              result.validation_metrics->maximum_abs_screen_pitch_rad = std::max(
                result.validation_metrics->maximum_abs_screen_pitch_rad, std::abs(*pitch));
            }
            if (std::isfinite(radius)) {
              result.validation_metrics->maximum_assembly_radius_m = std::max(
                result.validation_metrics->maximum_assembly_radius_m, radius);
            }
            const auto staged = requirements.staged_route;
            if (staged_route_active && std::isfinite(radius) &&
              radius <= staged->maximum_compact_radius_m + 1.0e-9)
            {
              compact_radius_entered = true;
            }
            const bool compact_radius_exit = staged_route_active && compact_radius_entered &&
              (!std::isfinite(radius) || radius > staged->maximum_compact_radius_m + 1.0e-9);
            if (!roll || !pitch || !std::isfinite(radius) || compact_radius_exit ||
              std::abs(*roll) > requirements.maximum_roll_error_rad ||
              (requirements.maximum_abs_screen_pitch_rad &&
              std::abs(*pitch) > *requirements.maximum_abs_screen_pitch_rad +
              kCompactPitchValidationToleranceRad) ||
              radius > requirements.maximum_assembly_radius_m)
            {
              result.message = "An interpolated compact-path sample violates pitch/roll/radius";
              result.path.clear();
              result.error_code = ErrorCode::INVALID_MOTION_PLAN;
              return result;
            }
            ++result.validation_metrics->checked_samples;
            if (!append_compact_certificate_sample(
                sampled_positions, original_segment_index, fraction, radius,
                std::abs(*roll), std::abs(*pitch)))
            {
              result.message = "Compact path certificate exceeds its sample budget";
              result.path.clear();
              result.error_code = ErrorCode::INVALID_MOTION_PLAN;
              return result;
            }
          }
          ++checked_samples;
        }
        result.path.push_back(std::move(positions));
      }
      if ((result.path.back() - planning_goal_positions).lpNorm<Eigen::Infinity>() >
        config.joint_tolerance_rad + 1.0e-6)
      {
        result.message = staged_route_active ?
          "OMPL fold path does not reach its validated prefix" :
          "OMPL path does not reach the requested joint goal tolerance";
        result.path.clear();
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        return result;
      }
    }

    if (staged_route_active && !staged_route_direct_complete && staged_prefix_available) {
      std::vector<Eigen::VectorXd> staged_path = result.path;
      if (!append_direct_segment(
          staged_path.back(), result.goal_positions,
          request.compact_goal->path_requirements.staged_route->maximum_compact_radius_m,
          request.compact_goal->path_requirements.staged_route->maximum_compact_radius_m,
          compact_radius_entered, staged_path))
      {
        result.message = "Compact suffix violates bounds, uprightness, collision or radius floor";
        result.path.clear();
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        return result;
      }
      result.path = std::move(staged_path);
      result.message = "Staged compact fold and direct reset legs validated";
    }

    if ((result.path.back() - result.goal_positions).lpNorm<Eigen::Infinity>() >
      config.joint_tolerance_rad + 1.0e-6)
    {
      result.message = "OMPL path does not reach the requested joint goal tolerance";
      result.path.clear();
      result.error_code = ErrorCode::INVALID_MOTION_PLAN;
      return result;
    }
    if (request.compact_goal && result.path.size() == 1U) {
      // Keep the planner's established zero-motion success case and represent
      // its single checked state as both endpoints of a degenerate segment.
      if (!compact_start_roll_error || !compact_start_pitch ||
        !append_compact_certificate_sample(
          result.path.front(), 0U, 1.0, compact_start_radius,
          *compact_start_roll_error, *compact_start_pitch))
      {
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        result.message = "Compact zero-motion certificate is malformed";
        return result;
      }
      const Eigen::VectorXd duplicate_endpoint = result.path.front();
      result.path.push_back(duplicate_endpoint);
    }
    if (request.compact_goal) {
      const auto & goal_metrics = exact_compact_goal_metrics;
      bool certificate_valid = goal_metrics && result.path.size() >= 2U &&
        compact_certificate_samples.size() >= 2U &&
        compact_certificate_samples.size() <= kMaximumCompactValidationSamples;
      if (certificate_valid) {
        const auto & first = compact_certificate_samples.front();
        const auto & last = compact_certificate_samples.back();
        certificate_valid = first.original_segment_index == 0U &&
          std::abs(first.fraction) <= 1.0e-12 &&
          first.positions.isApprox(result.path.front(), 1.0e-12) &&
          last.original_segment_index == result.path.size() - 2U &&
          std::abs(last.fraction - 1.0) <= 1.0e-12 &&
          last.positions.isApprox(result.path.back(), 1.0e-12);
      }
      for (std::size_t index = 0U;
        certificate_valid && index < compact_certificate_samples.size(); ++index)
      {
        const auto & sample = compact_certificate_samples[index];
        certificate_valid = sample.original_segment_index < result.path.size() - 1U &&
          std::isfinite(sample.fraction) && sample.fraction >= 0.0 && sample.fraction <= 1.0 &&
          sample.positions.size() == result.start_positions.size() &&
          sample.positions.allFinite() && std::isfinite(sample.maximum_assembly_radius_m) &&
          std::isfinite(sample.maximum_roll_error_rad) &&
          std::isfinite(sample.maximum_abs_screen_pitch_rad);
        if (!certificate_valid) {
          break;
        }
        const Eigen::VectorXd expected_position =
          result.path[sample.original_segment_index] + sample.fraction *
          (result.path[sample.original_segment_index + 1U] -
          result.path[sample.original_segment_index]);
        certificate_valid = sample.positions.isApprox(expected_position, 1.0e-9);
        if (index > 0U && certificate_valid) {
          const auto & previous = compact_certificate_samples[index - 1U];
          const bool same_segment =
            previous.original_segment_index == sample.original_segment_index;
          const bool next_segment = sample.original_segment_index ==
            previous.original_segment_index + 1U &&
            std::abs(previous.fraction - 1.0) <= 1.0e-12;
          const double sample_step =
            (sample.positions - previous.positions).lpNorm<Eigen::Infinity>();
          certificate_valid = (same_segment && sample.fraction > previous.fraction) ||
            (next_segment && sample.fraction > 0.0);
          certificate_valid = certificate_valid && std::isfinite(sample_step) &&
            sample_step <= kCompactValidationJointStepRad + 1.0e-9;
        }
      }
      if (!certificate_valid || !goal_metrics ||
        !std::isfinite(goal_metrics->maximum_assembly_radius_m) ||
        !std::isfinite(goal_metrics->maximum_roll_error_rad) ||
        !std::isfinite(goal_metrics->maximum_abs_screen_pitch_rad))
      {
        result.error_code = ErrorCode::INVALID_MOTION_PLAN;
        result.message = "Compact path certificate is incomplete or malformed";
        result.path.clear();
        return result;
      }

      double suffix_maximum_radius = goal_metrics->maximum_assembly_radius_m;
      double suffix_maximum_roll = goal_metrics->maximum_roll_error_rad;
      double suffix_maximum_pitch = goal_metrics->maximum_abs_screen_pitch_rad;
      for (auto sample = compact_certificate_samples.rbegin();
        sample != compact_certificate_samples.rend(); ++sample)
      {
        suffix_maximum_radius = std::max(
          suffix_maximum_radius, sample->maximum_assembly_radius_m);
        suffix_maximum_roll = std::max(suffix_maximum_roll, sample->maximum_roll_error_rad);
        suffix_maximum_pitch = std::max(
          suffix_maximum_pitch, sample->maximum_abs_screen_pitch_rad);
        sample->suffix_maximum_assembly_radius_m = suffix_maximum_radius;
        sample->suffix_maximum_roll_error_rad = suffix_maximum_roll;
        sample->suffix_maximum_abs_screen_pitch_rad = suffix_maximum_pitch;
      }
      result.compact_path_samples = std::move(compact_certificate_samples);
      result.compact_goal_metrics = std::move(exact_compact_goal_metrics);
    }
    result.success = true;
    result.error_code = ErrorCode::SUCCESS;
    if (!staged_route_active) {
      result.message = "Geometric path validated on the worker scene snapshot";
    }
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
  planning_interface::PlannerManagerPtr planner_manager;
  planning_interface::PlannerConfigurationMap default_planner_configurations;
  std::string planner_key;
  std::unique_ptr<CompactGeometryValidator> compact_geometry;
  std::string compact_geometry_error;
  std::string base_link_name;
  std::optional<std::size_t> staged_yaw_index;
  std::optional<std::size_t> staged_reset_index;
  std::optional<std::size_t> staged_wrist_pitch_index;
  std::optional<std::size_t> staged_joint6_index;
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
  const moveit::core::RobotState & start, const CompactGoal & goal,
  const std::uint64_t scene_revision)
{
  const auto & requirements = goal.path_requirements;
  if (start.getRobotModel().get() != impl_->robot_model.get() || goal.profile_id.empty() ||
    goal.bounded_positions.size() !=
    static_cast<Eigen::Index>(impl_->group->getVariableCount()) ||
    !goal.bounded_positions.allFinite() || requirements.base_link_name.empty() ||
    requirements.screen_link_name.empty() ||
    requirements.base_link_name != impl_->base_link_name ||
    !impl_->robot_model->hasLinkModel(requirements.screen_link_name) ||
    !positiveFinite(requirements.maximum_roll_error_rad) ||
    requirements.maximum_roll_error_rad > kPi ||
    (requirements.maximum_abs_screen_pitch_rad &&
    (!std::isfinite(*requirements.maximum_abs_screen_pitch_rad) ||
    *requirements.maximum_abs_screen_pitch_rad < 0.0 ||
    *requirements.maximum_abs_screen_pitch_rad > 0.5 * kPi)) ||
    !positiveFinite(requirements.maximum_assembly_radius_m) ||
    (requirements.staged_route &&
    (!std::isfinite(requirements.staged_route->maximum_early_yaw_shift_rad) ||
    requirements.staged_route->maximum_early_yaw_shift_rad < 0.0 ||
    requirements.staged_route->maximum_early_yaw_shift_rad >
    kMaximumStagedEarlyYawShiftRad + 1.0e-12 ||
    !positiveFinite(requirements.staged_route->maximum_compact_radius_m) ||
    requirements.staged_route->maximum_compact_radius_m >
    kMaximumStagedCompactRadiusM + 1.0e-12)))
  {
    throw std::invalid_argument("Compact goal has an invalid profile, model or path requirement");
  }
  Eigen::VectorXd start_positions;
  start.copyJointGroupPositions(impl_->group, start_positions);
  if (!start_positions.allFinite()) {
    throw std::invalid_argument("Compact path start contains non-finite joint positions");
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const std::uint64_t generation = ++impl_->latest_generation;
  impl_->pending.emplace(Impl::Request{
      start, goal.bounded_positions, goal, generation, scene_revision});
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
