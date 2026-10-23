// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/compact_upright_path.hpp"

namespace face_tracking_arm::control
{
namespace
{

using namespace std::chrono_literals;

constexpr double kPi = 3.14159265358979323846;
constexpr double kPitchLimitRad = 30.0 * kPi / 180.0;
constexpr double kV7PositivePitchLimitRad = 37.7124 * kPi / 180.0;
constexpr double kRollLimitRad = 5.1 * kPi / 180.0;
constexpr double kCompactRadiusLimitM = 0.40;
constexpr double kPathTolerance = 1.0e-9;

std::string readModelFile(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot read xArm6 test model: " + path);
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class CompactUprightPathTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    options.parameter_overrides({
          rclcpp::Parameter("robot_description", readModelFile(TEST_ROBOT_URDF_FILE)),
          rclcpp::Parameter("robot_description_semantic", readModelFile(TEST_ROBOT_SRDF_FILE)),
          rclcpp::Parameter(
          "robot_description_kinematics.xarm6.kinematics_solver",
          "kdl_kinematics_plugin/KDLKinematicsPlugin"),
          rclcpp::Parameter(
          "robot_description_kinematics.xarm6.kinematics_solver_timeout", 0.01),
          rclcpp::Parameter(
          "recovery_planner.planner_configs.RRTConnectkConfigDefault.type",
          "geometric::RRTConnect"),
          rclcpp::Parameter(
          "recovery_planner.planner_configs.RRTConnectkConfigDefault.range", 0.10),
          rclcpp::Parameter(
          "recovery_planner.xarm6.planner_configs",
          std::vector<std::string>{"RRTConnectkConfigDefault"}),
          rclcpp::Parameter("recovery_planner.xarm6.longest_valid_segment_fraction", 0.00019),
      });
    node_ = std::make_shared<rclcpp::Node>("compact_upright_path_test", options);
    monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node_, "robot_description");
    model_ = monitor_->getRobotModel();
    ASSERT_TRUE(model_);
    group_ = model_->getJointModelGroup("xarm6");
    ASSERT_NE(group_, nullptr);
    ASSERT_EQ(group_->getVariableCount(), 6U);
    ASSERT_EQ(group_->getVariableNames(), (std::vector<std::string>{
          "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}));
    planner_config_.move_group_name = "xarm6";
    planner_config_.compact_planning_time_s = 0.5;
    planner_config_.validation_joint_step_rad = 0.01;

    {
      planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
      ASSERT_TRUE(scene->getCurrentStateNonConst().setToDefaultValues(group_, "rest"));
      scene->getCurrentStateNonConst().update();
    }
  }

  void TearDown() override
  {
    monitor_.reset();
    model_.reset();
    group_ = nullptr;
    node_.reset();
  }

  moveit::core::RobotState v9BrakedStart() const
  {
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    Eigen::VectorXd positions(6);
    // Captured immediately after braking, at the v9 request to plan the
    // negative-wrist compact goal (production_pitch_probe/v9_settled_request_7130.csv).
    positions << 5.8437294395305805, -1.2879770008425657, -0.6126221869634798,
      -1.5484269828015693, -0.5952125584770421, 1.7526456557977512;
    state.setJointGroupPositions(group_, positions);
    state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    state.update();
    return state;
  }

  moveit::core::RobotState v7PositiveBrakedStart() const
  {
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    Eigen::VectorXd positions(6);
    // Captured positive-branch request from production_pitch_probe/gen1_v7_request_actual.csv.
    positions << 6.231937380637336, 0.25929263003009057, -1.671189107531851,
      -0.00236942962278436, -0.8073188450598627, 6.275581068044038;
    state.setJointGroupPositions(group_, positions);
    state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    state.update();
    return state;
  }

  Eigen::VectorXd namedGoal(const std::string & profile) const
  {
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    if (!state.setToDefaultValues(group_, profile)) {
      throw std::runtime_error("xArm6 SRDF is missing profile: " + profile);
    }
    Eigen::VectorXd result;
    state.copyJointGroupPositions(group_, result);
    return result;
  }

  Eigen::VectorXd v7PositiveUpperGoal() const
  {
    Eigen::VectorXd result(6);
    // Exact bounded endpoint from the successful v7 staged positive-upper probe.
    result << 0.0, -0.2715944, -2.8699983, kPi, 0.5 * kPi, 2.0 * kPi - 0.02;
    return result;
  }

  CompactStagedProjectionConfig projectionConfig() const
  {
    CompactStagedProjectionConfig config;
    config.maximum_abs_pitch_rad = kPitchLimitRad;
    config.maximum_roll_error_rad = kRollLimitRad;
    config.maximum_joint_step_rad = 0.01;
    config.maximum_samples = 8192U;
    return config;
  }

  void expectPathContract(
    const moveit::core::RobotState & start, const Eigen::VectorXd & goal,
    const std::vector<Eigen::VectorXd> & path,
    const CompactStagedProjectionConfig & config) const
  {
    ASSERT_GE(path.size(), 2U);
    Eigen::VectorXd start_positions;
    start.copyJointGroupPositions(group_, start_positions);
    ASSERT_TRUE(path.front().isApprox(start_positions, 0.0));
    ASSERT_TRUE(path.back().isApprox(goal, 0.0));

    moveit::core::RobotState sample(model_);
    for (std::size_t index = 0; index < path.size(); ++index) {
      ASSERT_EQ(path[index].size(), 6);
      ASSERT_TRUE(path[index].allFinite());
      sample.setJointGroupPositions(group_, path[index]);
      sample.update();
      ASSERT_TRUE(sample.satisfiesBounds(group_)) << "sample=" << index;
      const auto pose = evaluateCompactUprightPose(
        sample.getGlobalLinkTransform("monitor_control_frame").linear());
      ASSERT_TRUE(pose) << "screen pole at sample=" << index;
      EXPECT_LE(std::abs(pose->signed_pitch_rad),
        config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad)
        << "sample=" << index;
      EXPECT_LE(std::abs(pose->signed_roll_error_rad),
        config.maximum_roll_error_rad + kPathTolerance) << "sample=" << index;
      if (index > 0U) {
        const double max_step = (path[index] - path[index - 1U]).lpNorm<Eigen::Infinity>();
        EXPECT_LE(max_step, config.maximum_joint_step_rad + kPathTolerance)
          << "segment=" << (index - 1U);
      }
    }
  }

  std::optional<BackgroundPathResult> waitForResult(BackgroundPathPlanner & planner) const
  {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (auto result = planner.takeResult()) {
        return result;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene_monitor::PlanningSceneMonitorPtr monitor_;
  moveit::core::RobotModelConstPtr model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  BackgroundPathPlannerConfig planner_config_;
};

TEST_F(CompactUprightPathTest, ProjectsExactV9NegativeGoalAndMirroredYawWithinDenseLimits)
{
  auto start = v9BrakedStart();
  const auto goal = namedGoal("compact_turn_negative_wrist");
  const auto config = projectionConfig();
  ASSERT_TRUE(start.satisfiesBounds(group_));

  const auto projected = projectCompactStagedPath(
    start, goal, group_, "monitor_control_frame", config);
  ASSERT_TRUE(projected);
  expectPathContract(start, goal, projected->waypoints, config);
  ASSERT_LT(projected->fold_end_index + 1U, projected->waypoints.size());
  EXPECT_DOUBLE_EQ(
    projected->waypoints[projected->fold_end_index][3],
    start.getVariablePosition("joint4"));
  EXPECT_NE(
    projected->waypoints[projected->fold_end_index + 1U][3],
    start.getVariablePosition("joint4"));

  auto mirrored = v9BrakedStart();
  mirrored.setVariablePosition("joint1", -mirrored.getVariablePosition("joint1"));
  mirrored.update();
  ASSERT_TRUE(mirrored.satisfiesBounds(group_));
  const auto mirrored_path = projectCompactStagedPath(
    mirrored, goal, group_, "monitor_control_frame", config);
  ASSERT_TRUE(mirrored_path);
  expectPathContract(mirrored, goal, mirrored_path->waypoints, config);
}

TEST_F(CompactUprightPathTest, ProjectsKnownV7PositiveUpperRouteAtItsRecordedPitchCap)
{
  const auto start = v7PositiveBrakedStart();
  const Eigen::VectorXd goal = v7PositiveUpperGoal();
  const auto config = projectionConfig();
  auto positive_config = config;
  positive_config.maximum_abs_pitch_rad = kV7PositivePitchLimitRad;

  const auto projected = projectCompactStagedPath(
    start, goal, group_, "monitor_control_frame", positive_config);
  ASSERT_TRUE(projected);
  expectPathContract(start, goal, projected->waypoints, positive_config);
}

TEST_F(CompactUprightPathTest, RejectsCancelledBudgetAndOutOfBoundsRequests)
{
  const auto start = v9BrakedStart();
  const auto goal = namedGoal("compact_turn_negative_wrist");
  auto config = projectionConfig();

  EXPECT_FALSE(projectCompactStagedPath(
      start, goal, group_, "monitor_control_frame", config, []() {return true;}));

  config.maximum_samples = 2U;
  EXPECT_FALSE(projectCompactStagedPath(
      start, goal, group_, "monitor_control_frame", config));

  config = projectionConfig();
  Eigen::VectorXd outside_goal = goal;
  const auto & bounds = model_->getVariableBounds("joint2");
  outside_goal[1] = bounds.max_position_ + 0.1;
  EXPECT_FALSE(projectCompactStagedPath(
      start, outside_goal, group_, "monitor_control_frame", config));
}

TEST_F(CompactUprightPathTest, BackgroundPlannerReturnsCollisionAndRadiusValidatedRoute)
{
  const auto start = v9BrakedStart();
  const auto goal_positions = namedGoal("compact_turn_negative_wrist");
  CompactGoal goal;
  goal.profile_id = "compact_turn_negative_wrist";
  goal.bounded_positions = goal_positions;
  goal.path_requirements.base_link_name = "link_base";
  goal.path_requirements.screen_link_name = "monitor_control_frame";
  goal.path_requirements.maximum_roll_error_rad = kRollLimitRad;
  goal.path_requirements.maximum_abs_screen_pitch_rad = kPitchLimitRad;
  goal.path_requirements.maximum_assembly_radius_m = kCompactRadiusLimitM;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{
    0.35, kCompactRadiusLimitM};

  // Match the production geometric validator: the robot collision model is the
  // CMake-expanded xarm6_table xacro used by runtime, with the runtime-only
  // ros2_control/Gazebo plugin elements omitted from this planning fixture.
  moveit_msgs::msg::CollisionObject table;
  table.header.frame_id = "world";
  table.id = "round_table";
  table.operation = table.ADD;
  shape_msgs::msg::SolidPrimitive cylinder;
  cylinder.type = cylinder.CYLINDER;
  cylinder.dimensions = {0.05, 0.50};
  table.primitives.push_back(cylinder);
  geometry_msgs::msg::Pose table_pose;
  table_pose.position.z = 0.725;
  table_pose.orientation.w = 1.0;
  table.primitive_poses.push_back(table_pose);
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
    ASSERT_TRUE(scene->processCollisionObjectMsg(table));
  }

  CollisionConstraintConfig collision_config;
  const auto numerical_policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-3.14, 3.14}, 1.0, 2.0, 20.0, 0.01, SolverResidual{1.0e-6});
  ASSERT_TRUE(numerical_policy.has_value());
  collision_config.numerical_policy = *numerical_policy;
  const auto & joint_names = group_->getVariableNames();
  const auto joint5 = std::find(joint_names.begin(), joint_names.end(), "joint5");
  ASSERT_NE(joint5, joint_names.end());
  const auto & joint5_bounds = model_->getVariableBounds(*joint5);
  collision_config.protected_joint_name = "joint5";
  collision_config.protected_joint_lower_rad = joint5_bounds.min_position_;
  collision_config.protected_joint_upper_rad = joint5_bounds.max_position_;
  collision_config.invariant_mount_neighbor_name = "link6";
  collision_config.residual_latency_sec = 0.02;
  configureGeometryBounds(collision_config, *model_, *group_);
  const CollisionConstraintBuilder collision_validator(collision_config);
  auto validated_states = std::make_shared<std::atomic_size_t>(0U);
  const auto robust_clearance = [model = model_, group = group_, collision_validator,
      validated_states](const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & candidate) {
      ++(*validated_states);
      Eigen::VectorXd positions;
      candidate.copyJointGroupPositions(group, positions);
      for (Eigen::Index index = 0; index < positions.size(); ++index) {
        const auto & bounds = model->getVariableBounds(
          group->getVariableNames()[static_cast<std::size_t>(index)]);
        if (positions[index] < bounds.min_position_ || positions[index] > bounds.max_position_) {
          return false;
        }
      }
      const JointMotionState stationary{
        positions, Eigen::VectorXd::Zero(positions.size()),
        Eigen::VectorXd::Zero(positions.size())};
      const auto checked = collision_validator.validatePose(scene, candidate, *group, stationary);
      return checked.input_valid && !checked.unsafe;
    };

  BackgroundPathPlanner planner(
    node_, model_, monitor_, planner_config_, robust_clearance);
  const auto generation = planner.submit(start, goal, 7130U);
  const auto result = waitForResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->message, "Projected upright staged compact route validated");
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->scene_revision, 7130U);
  EXPECT_EQ(result->profile_id, goal.profile_id);
  ASSERT_GE(result->path.size(), 2U);
  EXPECT_TRUE(result->path.front().isApprox(result->start_positions, 1.0e-12));
  EXPECT_TRUE(result->path.back().isApprox(goal_positions, 0.0));
  EXPECT_TRUE(result->goal_positions.isApprox(goal_positions, 0.0));
  ASSERT_TRUE(result->validation_metrics);
  EXPECT_LE(result->validation_metrics->maximum_roll_error_rad, kRollLimitRad + kPathTolerance);
  EXPECT_LE(result->validation_metrics->maximum_abs_screen_pitch_rad,
    kPitchLimitRad + kCompactPitchValidationToleranceRad);
  EXPECT_LE(result->validation_metrics->maximum_assembly_radius_m,
    kCompactRadiusLimitM + kPathTolerance);
  EXPECT_DOUBLE_EQ(
    result->validation_metrics->effective_radius_cap_m, kCompactRadiusLimitM);
  EXPECT_GE(result->validation_metrics->checked_samples, result->path.size());
  EXPECT_GT(*validated_states, 0U);

  moveit::core::RobotState endpoint(model_);
  endpoint.setJointGroupPositions(group_, result->path.back());
  endpoint.update();
  const CompactGeometryValidator geometry(model_, "link_base");
  const double start_q4 = start.getVariablePosition("joint4");
  bool reset_started = false;
  for (std::size_t index = 1; index < result->path.size(); ++index) {
    if (std::abs(result->path[index][3] - start_q4) > 1.0e-9) {
      moveit::core::RobotState before_reset(model_);
      before_reset.setJointGroupPositions(group_, result->path[index - 1U]);
      before_reset.update();
      EXPECT_LE(
        geometry.maximumAssemblyRadius(before_reset),
        kCompactRadiusLimitM + kPathTolerance);
      reset_started = true;
      break;
    }
  }
  EXPECT_TRUE(reset_started);
  EXPECT_LE(geometry.maximumAssemblyRadius(endpoint), kCompactRadiusLimitM + kPathTolerance);
}

TEST_F(CompactUprightPathTest, ProjectionAndOmplShareOneCompactPlanningBudget)
{
  const auto start = v9BrakedStart();
  CompactGoal goal;
  goal.profile_id = "compact_turn_negative_wrist";
  goal.bounded_positions = namedGoal(goal.profile_id);
  goal.path_requirements.base_link_name = "link_base";
  goal.path_requirements.screen_link_name = "monitor_control_frame";
  goal.path_requirements.maximum_roll_error_rad = kRollLimitRad;
  goal.path_requirements.maximum_abs_screen_pitch_rad = kPitchLimitRad;
  goal.path_requirements.maximum_assembly_radius_m = kCompactRadiusLimitM;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{
    0.35, kCompactRadiusLimitM};

  auto short_budget = planner_config_;
  short_budget.compact_planning_time_s = 1.0e-9;
  BackgroundPathPlanner planner(node_, model_, monitor_, short_budget);
  static_cast<void>(planner.submit(start, goal, 7131U));
  const auto result = waitForResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT);
  EXPECT_EQ(result->message, "Compact planning time budget exhausted");
  EXPECT_TRUE(result->path.empty());
}

}  // namespace
}  // namespace face_tracking_arm::control
