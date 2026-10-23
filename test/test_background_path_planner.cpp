// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <srdfdom/srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/horizontal_edge_upright.hpp"

namespace face_tracking_arm::control
{
namespace
{
using namespace std::chrono_literals;

moveit::core::RobotModelConstPtr makeGeometryTestModel(
  const std::string & urdf_xml, const std::string & srdf_xml = {})
{
  const auto urdf_model = urdf::parseURDF(urdf_xml);
  if (!urdf_model) {
    return {};
  }
  auto semantic_model = std::make_shared<srdf::Model>();
  const std::string semantic_description = srdf_xml.empty() ?
    "<robot name=\"" + urdf_model->getName() + "\"/>" : srdf_xml;
  if (!semantic_model->initString(
      *urdf_model, semantic_description))
  {
    return {};
  }
  return std::make_shared<moveit::core::RobotModel>(urdf_model, semantic_model);
}

constexpr char kRobotDescription[] =
  R"(
<robot name="planner_slider">
  <link name="world"/>
  <link name="base"/>
  <link name="carriage"/>
  <link name="tip">
    <collision><geometry><sphere radius="0.04"/></geometry></collision>
  </link>
  <joint name="x" type="prismatic">
    <parent link="base"/><child link="carriage"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="y" type="prismatic">
    <parent link="carriage"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="10 20 0.75"/>
  </joint>
</robot>)";

constexpr char kRobotSemantic[] =
  R"(
<robot name="planner_slider">
  <group name="slider"><chain base_link="base" tip_link="tip"/></group>
</robot>)";

constexpr char kSixAxisRobotDescription[] =
  R"(
<robot name="xarm6">
  <link name="base"/>
  <link name="link1"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="link2"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="link3"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="link4"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="link5"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <link name="tip"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
  <joint name="joint5" type="revolute">
    <parent link="link4"/><child link="link5"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
  <joint name="joint6" type="revolute">
    <parent link="link5"/><child link="tip"/><origin xyz="0 0 0.10"/>
    <axis xyz="0 0 1"/><limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
</robot>)";

constexpr char kSixAxisRobotSemantic[] =
  R"(
<robot name="xarm6">
  <group name="slider"><chain base_link="base" tip_link="tip"/></group>
  <group_state name="compact_turn_positive_wrist" group="slider">
    <joint name="joint1" value="0.0"/>
    <joint name="joint2" value="-0.2715944"/>
    <joint name="joint3" value="-2.8699983"/>
    <joint name="joint4" value="0.0"/>
    <joint name="joint5" value="1.5707963267948966"/>
    <joint name="joint6" value="0.0"/>
  </group_state>
  <group_state name="compact_turn_negative_wrist" group="slider">
    <joint name="joint1" value="0.0"/>
    <joint name="joint2" value="-0.2715944"/>
    <joint name="joint3" value="-2.8699983"/>
    <joint name="joint4" value="0.0"/>
    <joint name="joint5" value="-1.151386"/>
    <joint name="joint6" value="3.141592653589793"/>
  </group_state>
</robot>)";

class RosEnvironment : public ::testing::Environment
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

[[maybe_unused]] const auto * ros_environment =
  ::testing::AddGlobalTestEnvironment(new RosEnvironment());

class BackgroundPathPlannerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    options.parameter_overrides({
          rclcpp::Parameter("robot_description", robotDescription()),
          rclcpp::Parameter("robot_description_semantic", robotSemantic()),
          rclcpp::Parameter(
        "robot_description_kinematics.slider.kinematics_solver",
        "kdl_kinematics_plugin/KDLKinematicsPlugin"),
          rclcpp::Parameter("robot_description_kinematics.slider.kinematics_solver_timeout", 0.01),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.type", "geometric::RRTConnect"),
          rclcpp::Parameter(
        "recovery_planner.slider.planner_configs",
        std::vector<std::string>{"RRTConnectkConfigDefault"}),
          rclcpp::Parameter("recovery_planner.slider.longest_valid_segment_fraction", 0.005),
    });
    node_ = std::make_shared<rclcpp::Node>("background_planner_test", options);
    monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node_, "robot_description");
    model_ = monitor_->getRobotModel();
    ASSERT_TRUE(model_);
    config_.move_group_name = "slider";
    config_.planning_time_s = 0.3;
    config_.joint_tolerance_rad = 0.001;
    config_.validation_joint_step_rad = 0.02;
  }

  virtual const char * robotDescription() const
  {
    return kRobotDescription;
  }

  virtual const char * robotSemantic() const
  {
    return kRobotSemantic;
  }

  moveit::core::RobotState state(double x, double y) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    result.setVariablePosition("x", x);
    result.setVariablePosition("y", y);
    result.update();
    return result;
  }

  std::optional<BackgroundPathResult> waitResult(BackgroundPathPlanner & planner) const
  {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (auto result = planner.takeResult()) {
        return result;
      }
      std::this_thread::sleep_for(1ms);
    }
    return {};
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene_monitor::PlanningSceneMonitorPtr monitor_;
  moveit::core::RobotModelConstPtr model_;
  BackgroundPathPlannerConfig config_;
};

class BackgroundStagedPathPlannerTest : public BackgroundPathPlannerTest
{
protected:
  const char * robotDescription() const override
  {
    return kSixAxisRobotDescription;
  }

  const char * robotSemantic() const override
  {
    return kSixAxisRobotSemantic;
  }
};

void expectCompactCertificateMatchesPath(
  const BackgroundPathResult & result, moveit::core::RobotModelConstPtr model,
  const std::string & base_link_name, const std::string & screen_link_name)
{
  ASSERT_TRUE(result.success);
  ASSERT_TRUE(result.validation_metrics);
  ASSERT_TRUE(result.compact_goal_metrics);
  ASSERT_GE(result.path.size(), 2U);
  ASSERT_TRUE(model);
  const auto * group = model->getJointModelGroup("slider");
  ASSERT_TRUE(group);
  const CompactGeometryValidator geometry(model, base_link_name);
  moveit::core::RobotState measured_state(model);
  measured_state.setToDefaultValues();
  const auto & samples = result.compact_path_samples;
  ASSERT_GE(samples.size(), 2U);
  ASSERT_LE(samples.size(), 8192U);
  EXPECT_EQ(samples.size(), result.validation_metrics->checked_samples);
  EXPECT_TRUE(samples.front().positions.isApprox(result.path.front(), 1.0e-12));
  EXPECT_EQ(samples.front().original_segment_index, 0U);
  EXPECT_DOUBLE_EQ(samples.front().fraction, 0.0);
  EXPECT_TRUE(samples.back().positions.isApprox(result.path.back(), 1.0e-12));
  EXPECT_EQ(samples.back().original_segment_index, result.path.size() - 2U);
  EXPECT_DOUBLE_EQ(samples.back().fraction, 1.0);

  double suffix_radius = result.compact_goal_metrics->maximum_assembly_radius_m;
  double suffix_roll = result.compact_goal_metrics->maximum_roll_error_rad;
  double suffix_pitch = result.compact_goal_metrics->maximum_abs_screen_pitch_rad;
  double whole_path_radius = 0.0;
  double whole_path_roll = 0.0;
  double whole_path_pitch = 0.0;
  for (std::size_t reverse_index = samples.size(); reverse_index > 0U; --reverse_index) {
    const auto index = reverse_index - 1U;
    const auto & sample = samples[index];
    ASSERT_LT(sample.original_segment_index, result.path.size() - 1U);
    ASSERT_TRUE(std::isfinite(sample.fraction));
    ASSERT_GE(sample.fraction, 0.0);
    ASSERT_LE(sample.fraction, 1.0);
    const Eigen::VectorXd expected_position =
      result.path[sample.original_segment_index] + sample.fraction *
      (result.path[sample.original_segment_index + 1U] -
      result.path[sample.original_segment_index]);
    EXPECT_TRUE(sample.positions.isApprox(expected_position, 1.0e-9));
    if (index > 0U) {
      const auto & previous = samples[index - 1U];
      const bool same_segment =
        previous.original_segment_index == sample.original_segment_index;
      const bool next_segment = sample.original_segment_index ==
        previous.original_segment_index + 1U && std::abs(previous.fraction - 1.0) <= 1.0e-12;
      EXPECT_TRUE((same_segment && sample.fraction > previous.fraction) ||
        (next_segment && sample.fraction > 0.0));
      EXPECT_LE(
        (sample.positions - previous.positions).lpNorm<Eigen::Infinity>(), 0.010000001);
    }
    EXPECT_TRUE(sample.positions.allFinite());
    EXPECT_TRUE(std::isfinite(sample.maximum_assembly_radius_m));
    EXPECT_TRUE(std::isfinite(sample.maximum_roll_error_rad));
    EXPECT_TRUE(std::isfinite(sample.maximum_abs_screen_pitch_rad));
    measured_state.setJointGroupPositions(group, sample.positions);
    measured_state.update(true);
    const auto measured_pose = evaluateCompactUprightPose(
      measured_state.getGlobalLinkTransform(screen_link_name).linear());
    ASSERT_TRUE(measured_pose);
    EXPECT_NEAR(
      sample.maximum_assembly_radius_m, geometry.maximumAssemblyRadius(measured_state), 1.0e-12);
    EXPECT_NEAR(
      sample.maximum_roll_error_rad, std::abs(measured_pose->signed_roll_error_rad), 1.0e-12);
    EXPECT_NEAR(
      sample.maximum_abs_screen_pitch_rad, std::abs(measured_pose->signed_pitch_rad), 1.0e-12);
    suffix_radius = std::max(suffix_radius, sample.maximum_assembly_radius_m);
    suffix_roll = std::max(suffix_roll, sample.maximum_roll_error_rad);
    suffix_pitch = std::max(suffix_pitch, sample.maximum_abs_screen_pitch_rad);
    EXPECT_DOUBLE_EQ(sample.suffix_maximum_assembly_radius_m, suffix_radius);
    EXPECT_DOUBLE_EQ(sample.suffix_maximum_roll_error_rad, suffix_roll);
    EXPECT_DOUBLE_EQ(sample.suffix_maximum_abs_screen_pitch_rad, suffix_pitch);
    whole_path_radius = std::max(whole_path_radius, sample.maximum_assembly_radius_m);
    whole_path_roll = std::max(whole_path_roll, sample.maximum_roll_error_rad);
    whole_path_pitch = std::max(whole_path_pitch, sample.maximum_abs_screen_pitch_rad);
  }
  whole_path_radius = std::max(
    whole_path_radius, result.compact_goal_metrics->maximum_assembly_radius_m);
  whole_path_roll = std::max(
    whole_path_roll, result.compact_goal_metrics->maximum_roll_error_rad);
  whole_path_pitch = std::max(
    whole_path_pitch, result.compact_goal_metrics->maximum_abs_screen_pitch_rad);
  EXPECT_DOUBLE_EQ(result.validation_metrics->maximum_assembly_radius_m, whole_path_radius);
  EXPECT_DOUBLE_EQ(result.validation_metrics->maximum_roll_error_rad, whole_path_roll);
  EXPECT_DOUBLE_EQ(result.validation_metrics->maximum_abs_screen_pitch_rad, whole_path_pitch);

  measured_state.setJointGroupPositions(group, result.goal_positions);
  measured_state.update(true);
  const auto goal_pose = evaluateCompactUprightPose(
    measured_state.getGlobalLinkTransform(screen_link_name).linear());
  ASSERT_TRUE(goal_pose);
  EXPECT_NEAR(result.compact_goal_metrics->maximum_assembly_radius_m,
    geometry.maximumAssemblyRadius(measured_state), 1.0e-12);
  EXPECT_NEAR(result.compact_goal_metrics->maximum_roll_error_rad,
    std::abs(goal_pose->signed_roll_error_rad), 1.0e-12);
  EXPECT_NEAR(result.compact_goal_metrics->maximum_abs_screen_pitch_rad,
    std::abs(goal_pose->signed_pitch_rad), 1.0e-12);
}

TEST_F(BackgroundPathPlannerTest, PlansAroundWorldObstacleAndAppliesExtraFeasibility)
{
  moveit_msgs::msg::CollisionObject obstacle;
  obstacle.header.frame_id = "base";
  obstacle.id = "box";
  obstacle.operation = obstacle.ADD;
  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = {0.3, 0.3, 0.3};
  obstacle.primitives.push_back(box);
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  obstacle.primitive_poses.push_back(pose);
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
    ASSERT_TRUE(scene->processCollisionObjectMsg(obstacle));
  }
  const auto upper_half_plane = [](
    const planning_scene::PlanningScene &, const moveit::core::RobotState & candidate)
    {
      return candidate.getVariablePosition("y") >= -0.02;
    };
  BackgroundPathPlanner planner(node_, model_, monitor_, config_, upper_half_plane);
  const Eigen::Vector2d goal(0.6, 0.0);
  const auto generation = planner.submit(state(-0.6, 0.0), goal, 37);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->scene_revision, 37U);
  ASSERT_GE(result->path.size(), 3U);
  EXPECT_TRUE(result->path.front().isApprox(Eigen::Vector2d(-0.6, 0.0)));
  EXPECT_LT((result->path.back() - goal).lpNorm<Eigen::Infinity>(), 0.002);
  double maximum_y = 0.0;
  for (const auto & waypoint : result->path) {
    EXPECT_GE(waypoint[1], -0.02);
    maximum_y = std::max(maximum_y, waypoint[1]);
  }
  EXPECT_GT(maximum_y, 0.19);
  EXPECT_FALSE(planner.takeResult());
}

TEST_F(BackgroundPathPlannerTest, DeclaresPipelineOverridesBeforeOmplLoadsItsConfiguration)
{
  // Reproduce a composable node: the override values exist, but the node has
  // declared its other parameters without declaring the OMPL namespace.
  const auto names = node_->list_parameters({"recovery_planner"}, 10).names;
  ASSERT_FALSE(names.empty());
  for (const auto & name : names) {
    node_->undeclare_parameter(name);
  }
  EXPECT_FALSE(node_->has_parameter("recovery_planner.slider.planner_configs"));
  // Construction checks OMPL's loaded map for slider[RRTConnectkConfigDefault],
  // so silently falling back to its unnamed defaults also fails this regression.
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  EXPECT_TRUE(node_->has_parameter("recovery_planner.slider.planner_configs"));
  static_cast<void>(planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.3, 0.2), 54));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->success) << result->message;
}

TEST_F(BackgroundPathPlannerTest, RejectsNonPositiveCompactPlanningBudget)
{
  auto invalid_config = config_;
  invalid_config.compact_planning_time_s = 0.0;
  EXPECT_THROW(
    BackgroundPathPlanner(node_, model_, monitor_, invalid_config), std::invalid_argument);
}

TEST_F(BackgroundPathPlannerTest, RejectsInvalidRestEdgePlannerSettings)
{
  auto invalid_config = config_;
  invalid_config.rest_edge_planning_time_s = 0.0;
  EXPECT_THROW(
    BackgroundPathPlanner(node_, model_, monitor_, invalid_config), std::invalid_argument);

  invalid_config = config_;
  invalid_config.rest_edge_planner_range = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    BackgroundPathPlanner(node_, model_, monitor_, invalid_config), std::invalid_argument);
}

TEST_F(BackgroundPathPlannerTest, RejectsGoalOutsideAdditionalSafetyCorridor)
{
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_,
    [](const planning_scene::PlanningScene &, const moveit::core::RobotState & candidate) {
      return candidate.getVariablePosition("y") >= 0.0;
    });
  static_cast<void>(planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.6, -0.2), 3));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::GOAL_IN_COLLISION);
  EXPECT_TRUE(result->path.empty());
}

TEST_F(BackgroundPathPlannerTest, ReplacesPendingRequestsAndCancelDropsActiveResult)
{
  struct Gate
  {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered{false};
    bool released{false};
  };
  const auto gate = std::make_shared<Gate>();
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_,
    [gate](const planning_scene::PlanningScene &, const moveit::core::RobotState &) {
      std::unique_lock<std::mutex> lock(gate->mutex);
      gate->entered = true;
      gate->condition.notify_one();
      return gate->condition.wait_for(lock, 2s, [gate]() {return gate->released;});
    });
  const auto first = planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.6, 0.2), 1);
  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    ASSERT_TRUE(gate->condition.wait_for(lock, 2s, [gate]() {return gate->entered;}));
  }
  const auto second = planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.2, 0.2), 2);
  const auto third = planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.4, 0.2), 3);
  EXPECT_GT(second, first);
  EXPECT_GT(third, second);
  planner.cancel();
  {
    const std::lock_guard<std::mutex> lock(gate->mutex);
    gate->released = true;
  }
  gate->condition.notify_one();
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (planner.isBusy() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_FALSE(planner.isBusy());
  EXPECT_FALSE(planner.takeResult());
  const auto latest = planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.3, 0.2), 42);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_GT(latest, third);
  EXPECT_EQ(result->generation, latest);
  EXPECT_EQ(result->scene_revision, 42U);
  EXPECT_NEAR(result->path.back()[0], 0.3, 0.002);
}

TEST_F(BackgroundPathPlannerTest, PreservesTheMonitoredScenesExistingFeasibilityPredicate)
{
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
    scene->setStateFeasibilityPredicate(
      [](const moveit::core::RobotState & candidate, bool) {
        return candidate.getVariablePosition("y") >= 0.0;
      });
  }
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_,
    [](const planning_scene::PlanningScene &, const moveit::core::RobotState &) {
      return true;
    });
  static_cast<void>(planner.submit(state(-0.6, 0.2), Eigen::Vector2d(0.6, -0.2), 9));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::GOAL_IN_COLLISION);
}

class BackgroundRevolutePlannerTest : public BackgroundPathPlannerTest
{
protected:
  const char * robotDescription() const override
  {
    return
      R"(
<robot name="planner_slider">
  <link name="base"/>
  <link name="tip"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
    <limit lower="-6.28318530718" upper="6.28318530718" effort="10" velocity="1"/>
  </joint>
</robot>)";
  }
};

class BackgroundHorizontalEdgePlannerTest : public BackgroundPathPlannerTest
{
protected:
  const char * robotDescription() const override
  {
    return
      R"(
<robot name="edge_planner">
  <link name="base"/><link name="middle"/><link name="tip"/>
  <joint name="edge_joint1" type="revolute">
    <parent link="base"/><child link="middle"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="3.14159265359" effort="10" velocity="1"/>
  </joint>
  <joint name="edge_joint2" type="revolute">
    <parent link="middle"/><child link="tip"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="3.14159265359" effort="10" velocity="1"/>
  </joint>
</robot>)";
  }

  const char * robotSemantic() const override
  {
    return
      R"(<robot name="edge_planner">
  <group name="slider"><chain base_link="base" tip_link="tip"/></group>
</robot>)";
  }
};

class BackgroundHighPitchEdgePlannerTest : public BackgroundPathPlannerTest
{
protected:
  const char * robotDescription() const override
  {
    return
      R"(
<robot name="high_pitch_edge_planner">
  <link name="base"/><link name="tip"/>
  <joint name="pitch_joint" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-3.14159265359" upper="3.14159265359" effort="10" velocity="1"/>
  </joint>
</robot>)";
  }

  const char * robotSemantic() const override
  {
    return
      R"(<robot name="high_pitch_edge_planner">
  <group name="slider"><chain base_link="base" tip_link="tip"/></group>
</robot>)";
  }
};

TEST_F(BackgroundPathPlannerTest, CompactGoalValidatesUprightRadiusAndReturnsFacts)
{
  EXPECT_EQ(model_->getRootLinkName(), "world");
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  CompactGoal goal;
  goal.profile_id = "slider_compact";
  goal.bounded_positions = Eigen::Vector2d(0.4, 0.2);
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.70;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.25;
  // This generic two-axis model has no joint1/joint4 pair, so the internal
  // staged route request must leave its established planning behavior intact.
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  const auto generation = planner.submit(state(-0.6, 0.2), goal, 61);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->scene_revision, 61U);
  EXPECT_EQ(result->profile_id, "slider_compact");
  ASSERT_TRUE(result->validation_metrics);
  EXPECT_LE(result->validation_metrics->maximum_roll_error_rad, 0.05);
  EXPECT_LE(result->validation_metrics->maximum_abs_screen_pitch_rad, 0.25);
  EXPECT_LE(result->validation_metrics->maximum_assembly_radius_m, 0.70);
  EXPECT_DOUBLE_EQ(result->validation_metrics->effective_radius_cap_m, 0.70);
  EXPECT_GE(result->validation_metrics->checked_samples, 90U);
  EXPECT_TRUE(std::isnan(result->validation_metrics->minimum_robust_clearance_m));
  EXPECT_TRUE(result->path.front().isApprox(Eigen::Vector2d(-0.6, 0.2)));
  EXPECT_LT((result->path.back() - goal.bounded_positions).lpNorm<Eigen::Infinity>(), 0.002);
  expectCompactCertificateMatchesPath(*result, model_, "base", "tip");
}

TEST_F(BackgroundStagedPathPlannerTest, DirectStagedPathPreservesTheCommittedGoalAndJointBounds)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  CompactGoal goal;
  goal.profile_id = "compact_turn_positive_wrist";
  goal.bounded_positions.resize(6);
  goal.bounded_positions << 0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966,
    6.28318530718 - 0.02;
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.5;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  Eigen::VectorXd start_positions(6);
  start_positions << 6.28318530718, 0.0, 0.0, 1.0, 0.0, 0.0;
  start.setJointGroupPositions(model_->getJointModelGroup("slider"), start_positions);
  start.update();
  ASSERT_TRUE(start.satisfiesBounds(model_->getJointModelGroup("slider")));

  const auto generation = planner.submit(start, goal, 71);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->scene_revision, 71U);
  EXPECT_EQ(result->profile_id, goal.profile_id);
  ASSERT_EQ(result->path.size(), 3U);
  EXPECT_TRUE(result->path.front().isApprox(start_positions, 1.0e-12));
  EXPECT_NEAR(result->path[1][0], start_positions[0] - 0.35, 1.0e-12);
  EXPECT_DOUBLE_EQ(result->path[1][3], start_positions[3]);
  EXPECT_TRUE(result->path.back().isApprox(goal.bounded_positions, 1.0e-12));
  EXPECT_TRUE(result->goal_positions.isApprox(goal.bounded_positions, 1.0e-12));
  ASSERT_TRUE(result->validation_metrics);
  EXPECT_LE(result->validation_metrics->maximum_assembly_radius_m, 0.40);
  expectCompactCertificateMatchesPath(*result, model_, "base", "tip");
}

TEST_F(BackgroundStagedPathPlannerTest, NegativeWristNamedPresetUsesTheStagedRoute)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  CompactGoal goal;
  goal.profile_id = "compact_turn_negative_wrist";
  goal.bounded_positions.resize(6);
  goal.bounded_positions << 0.0, -0.2715944, -2.8699983, 0.0, -1.151386,
    3.141592653589793;
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.5;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  Eigen::VectorXd start_positions(6);
  start_positions << -6.28318530718, 0.0, 0.0, -1.0, 0.0, 0.0;
  start.setJointGroupPositions(model_->getJointModelGroup("slider"), start_positions);
  start.update();
  ASSERT_TRUE(start.satisfiesBounds(model_->getJointModelGroup("slider")));

  const auto generation = planner.submit(start, goal, 75);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->profile_id, goal.profile_id);
  ASSERT_EQ(result->path.size(), 3U);
  EXPECT_TRUE(result->path.front().isApprox(start_positions, 1.0e-12));
  EXPECT_NEAR(result->path[1][0], start_positions[0] + 0.35, 1.0e-12);
  EXPECT_DOUBLE_EQ(result->path[1][3], start_positions[3]);
  EXPECT_TRUE(result->path.back().isApprox(goal.bounded_positions, 1.0e-12));
  EXPECT_TRUE(result->goal_positions.isApprox(goal.bounded_positions, 1.0e-12));
  ASSERT_TRUE(result->validation_metrics);
  EXPECT_LE(result->validation_metrics->maximum_abs_screen_pitch_rad, 0.05);
  EXPECT_LE(result->validation_metrics->maximum_roll_error_rad, 0.05);
  expectCompactCertificateMatchesPath(*result, model_, "base", "tip");
}

TEST_F(
  BackgroundStagedPathPlannerTest,
  InvalidStagedCompactRouteRejectsWithoutPlanningToAnotherGoal)
{
  const auto reject_suffix_corridor = [](
    const planning_scene::PlanningScene &, const moveit::core::RobotState & state) {
      const double yaw = state.getVariablePosition("joint1");
      const double reset = state.getVariablePosition("joint4");
      return !(yaw > 0.2 && yaw < 0.4 && reset > 0.4 && reset < 0.6);
    };
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_, reject_suffix_corridor);
  CompactGoal goal;
  goal.profile_id = "compact_turn_positive_wrist";
  goal.bounded_positions.resize(6);
  goal.bounded_positions << 0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966, 0.0;
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.5;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  Eigen::VectorXd start_positions(6);
  start_positions << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  start.setJointGroupPositions(model_->getJointModelGroup("slider"), start_positions);
  start.update();

  static_cast<void>(planner.submit(start, goal, 72));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN);
  EXPECT_FALSE(result->message.empty());
  EXPECT_TRUE(result->path.empty());
  EXPECT_TRUE(result->compact_path_samples.empty());
  EXPECT_FALSE(result->compact_goal_metrics);
  EXPECT_TRUE(result->goal_positions.isApprox(goal.bounded_positions, 1.0e-12));
}

TEST_F(BackgroundStagedPathPlannerTest, InvalidKnownPrefixDoesNotFallBackToTheFullGoal)
{
  const auto reject_prefix = [](
    const planning_scene::PlanningScene &, const moveit::core::RobotState & state) {
      const double yaw = state.getVariablePosition("joint1");
      const double reset = state.getVariablePosition("joint4");
      return !(yaw > 0.5 && yaw < 0.8 && reset > 0.9);
    };
  BackgroundPathPlanner planner(node_, model_, monitor_, config_, reject_prefix);
  CompactGoal goal;
  goal.profile_id = "compact_turn_positive_wrist";
  goal.bounded_positions.resize(6);
  goal.bounded_positions << 0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966, 0.0;
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.5;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  Eigen::VectorXd start_positions(6);
  start_positions << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  start.setJointGroupPositions(model_->getJointModelGroup("slider"), start_positions);
  start.update();

  static_cast<void>(planner.submit(start, goal, 73));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN);
  EXPECT_EQ(result->message, "Validated staged compact prefix is unavailable");
  EXPECT_TRUE(result->path.empty());
  EXPECT_TRUE(result->goal_positions.isApprox(goal.bounded_positions, 1.0e-12));
}

TEST_F(BackgroundStagedPathPlannerTest, StagedGoalMustMatchNamedSrdfPreset)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  CompactGoal goal;
  goal.profile_id = "compact_turn_positive_wrist";
  goal.bounded_positions.resize(6);
  goal.bounded_positions << 0.0, -0.2615944, -2.8699983, 0.0, 1.5707963267948966, 0.0;
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_roll_error_rad = 0.05;
  goal.path_requirements.maximum_abs_screen_pitch_rad = 0.05;
  goal.path_requirements.maximum_assembly_radius_m = 0.5;
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{};

  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  Eigen::VectorXd start_positions(6);
  start_positions << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  start.setJointGroupPositions(model_->getJointModelGroup("slider"), start_positions);
  start.update();

  static_cast<void>(planner.submit(start, goal, 74));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN);
  EXPECT_EQ(result->message, "Staged compact goal does not match its named SRDF posture");
  EXPECT_TRUE(result->path.empty());
  EXPECT_TRUE(result->goal_positions.isApprox(goal.bounded_positions, 1.0e-12));
}

TEST_F(BackgroundPathPlannerTest, CompactGoalRejectsInvalidProfileAndNarrowStartEnvelope)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  CompactGoal goal;
  goal.profile_id = "slider_compact";
  goal.bounded_positions = Eigen::Vector2d(0.4, 0.2);
  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.screen_link_name = "tip";
  goal.path_requirements.maximum_assembly_radius_m = 0.65;

  static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 62));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID);
  EXPECT_TRUE(result->path.empty());
  EXPECT_EQ(result->profile_id, "slider_compact");
  ASSERT_TRUE(result->validation_metrics);
  EXPECT_DOUBLE_EQ(result->validation_metrics->effective_radius_cap_m, 0.65);

  goal.path_requirements.base_link_name = "not_the_root";
  EXPECT_THROW(static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 63)),
    std::invalid_argument);

  goal.path_requirements.base_link_name = "base";
  goal.path_requirements.maximum_abs_screen_pitch_rad = -0.01;
  EXPECT_THROW(static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 64)),
    std::invalid_argument);

  goal.path_requirements.maximum_abs_screen_pitch_rad.reset();
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{0.36, 0.40};
  EXPECT_THROW(static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 65)),
    std::invalid_argument);
  goal.path_requirements.staged_route = CompactStagedRouteRequirements{0.35, 0.401};
  EXPECT_THROW(static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 66)),
    std::invalid_argument);
}

TEST(CompactGeometryValidatorTest, IncludesVisualPrimitiveOutsideCollisionAtItsOrigin)
{
  const std::string description =
    R"(
<robot name="visual_box">
  <link name="base"/>
  <link name="tip">
    <collision><geometry><sphere radius="0.04"/></geometry></collision>
    <visual>
      <origin xyz="0.5 0 0"/>
      <geometry><box size="0.2 0.1 0.1"/></geometry>
    </visual>
  </link>
  <joint name="tip_fixed" type="fixed">
    <parent link="base"/><child link="tip"/>
  </joint>
</robot>)";
  const auto model = makeGeometryTestModel(description);
  ASSERT_TRUE(model);
  moveit::core::RobotState state(model);
  state.update();
  const CompactGeometryValidator validator(model, "base");

  EXPECT_NEAR(validator.maximumAssemblyRadius(state), std::hypot(0.6, 0.05), 1.0e-12);
}

TEST(CompactGeometryValidatorTest, AppliesMeshScaleAndOriginAndIgnoresInteriorPoints)
{
  const auto mesh_path = std::filesystem::temp_directory_path() /
    ("compact_geometry_visual_" + std::to_string(::getpid()) + ".stl");
  {
    std::ofstream mesh(mesh_path);
    ASSERT_TRUE(mesh.good());
    mesh <<
      R"(solid tetra
facet normal 0 0 -1
outer loop
vertex 0 0 0
vertex 0 1 0
vertex 0.3333333333333333 0.3333333333333333 0
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 0 1 0
vertex 1 0 0
vertex 0.3333333333333333 0.3333333333333333 0
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 1 0 0
vertex 0 0 0
vertex 0.3333333333333333 0.3333333333333333 0
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 0 0 0
vertex 0 1 0
vertex 1 0 0
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 0 1
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 0 0 0
vertex 0 0 1
vertex 0 1 0
endloop
endfacet
facet normal 1 1 1
outer loop
vertex 1 0 0
vertex 0 1 0
vertex 0 0 1
endloop
endfacet
endsolid tetra
)";
  }
  const std::string mesh_uri = "file://" + mesh_path.string();
  const std::string description =
    "<robot name=\"scaled_visual_mesh\">"
    "<link name=\"base\"/>"
    "<link name=\"tip\">"
    "<collision><geometry><sphere radius=\"0.04\"/></geometry></collision>"
    "<visual><origin xyz=\"0.25 0.5 0\"/><geometry>"
    "<mesh filename=\"" + mesh_uri + "\" scale=\"2 1 1\"/>"
    "</geometry></visual></link>"
    "<joint name=\"tip_fixed\" type=\"fixed\"><parent link=\"base\"/>"
    "<child link=\"tip\"/></joint></robot>";
  const auto model = makeGeometryTestModel(description);
  ASSERT_TRUE(model);
  moveit::core::RobotState state(model);
  state.update();
  const CompactGeometryValidator validator(model, "base");
  std::filesystem::remove(mesh_path);

  EXPECT_NEAR(validator.maximumAssemblyRadius(state), std::hypot(2.25, 0.5), 1.0e-9);
}

TEST(CompactGeometryValidatorTest, CylinderEnvelopeNeverLosesVerticalOrTiltedRadius)
{
  const auto vertical_model =
    makeGeometryTestModel(
        R"(
<robot name="vertical_cylinder">
  <link name="base">
    <collision><geometry><cylinder radius="0.2" length="1.0"/></geometry></collision>
  </link>
</robot>)");
  ASSERT_TRUE(vertical_model);
  moveit::core::RobotState vertical_state(vertical_model);
  vertical_state.update();
  const CompactGeometryValidator vertical_validator(vertical_model, "base");
  EXPECT_NEAR(vertical_validator.maximumAssemblyRadius(vertical_state), 0.2, 1.0e-12);

  const auto tilted_model =
    makeGeometryTestModel(
        R"(
<robot name="tilted_cylinder">
  <link name="base"/>
  <link name="cylinder">
    <collision><geometry><cylinder radius="0.2" length="1.0"/></geometry></collision>
  </link>
  <joint name="base_to_cylinder" type="fixed">
    <parent link="base"/><child link="cylinder"/>
    <origin xyz="0.3 0.4 0" rpy="0 1.5707963267948966 0"/>
  </joint>
</robot>)");
  ASSERT_TRUE(tilted_model);
  moveit::core::RobotState tilted_state(tilted_model);
  tilted_state.update();
  const CompactGeometryValidator tilted_validator(tilted_model, "base");
  EXPECT_NEAR(
    tilted_validator.maximumAssemblyRadius(tilted_state), std::hypot(0.8, 0.4) + 0.2,
    1.0e-12);
}

TEST(CompactGeometryValidatorTest, CoactiveSupportsRetainBothRadialDerivatives)
{
  const auto model = makeGeometryTestModel(
    R"(
<robot name="coactive_supports">
  <link name="world"/>
  <link name="base"><collision><geometry><sphere radius="0.2"/></geometry></collision></link>
  <link name="tip">
    <collision><origin xyz="1 0 1"/><geometry><sphere radius="0.01"/></geometry></collision>
    <collision><origin xyz="1 0 -1"/><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="2 -3 0.7" rpy="0.2 -0.3 0.5"/>
  </joint>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>)",
    R"(<robot name="coactive_supports">
  <group name="arm"><chain base_link="base" tip_link="tip"/></group>
</robot>)");
  ASSERT_TRUE(model);
  const auto * group = model->getJointModelGroup("arm");
  ASSERT_NE(group, nullptr);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.update();
  const CompactGeometryValidator validator(model, "base");

  const auto supports = validator.perLinkRadiusSupports(state, *group);
  ASSERT_TRUE(supports);
  EXPECT_NEAR(supports->maximum_radius_m, 1.01, 1.0e-12);
  ASSERT_EQ(supports->supports.size(), 2U);
  EXPECT_NEAR(supports->supports[0].radius_m, supports->supports[1].radius_m, 1.0e-12);
  ASSERT_EQ(supports->supports[0].outward_gradient.size(), 1);
  ASSERT_EQ(supports->supports[1].outward_gradient.size(), 1);
  EXPECT_NEAR(std::abs(supports->supports[0].outward_gradient[0]), 1.0, 1.0e-12);
  EXPECT_NEAR(std::abs(supports->supports[1].outward_gradient[0]), 1.0, 1.0e-12);
  EXPECT_NEAR(
    supports->supports[0].outward_gradient[0] +
    supports->supports[1].outward_gradient[0], 0.0, 1.0e-12);
  EXPECT_FALSE(validator.perLinkRadiusSupports(state, *group, 1.0e-6, 1));
}

TEST(CompactGeometryValidatorTest, SupportGradientMatchesRadiusFiniteDifferenceInShiftedBase)
{
  const auto model = makeGeometryTestModel(
    R"(
<robot name="support_gradient">
  <link name="world"/><link name="base"/>
  <link name="tip">
    <collision><origin xyz="0.7 0.2 0.5"/><geometry><sphere radius="0.03"/></geometry></collision>
  </link>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="10 -4 1.2" rpy="0.4 -0.2 0.7"/>
  </joint>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>)",
    R"(<robot name="support_gradient">
  <group name="arm"><chain base_link="base" tip_link="tip"/></group>
</robot>)");
  ASSERT_TRUE(model);
  const auto * group = model->getJointModelGroup("arm");
  ASSERT_NE(group, nullptr);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  constexpr double position = 0.31;
  constexpr double epsilon = 1.0e-6;
  state.setVariablePosition("hinge", position);
  state.update();
  const CompactGeometryValidator validator(model, "base");

  const auto supports = validator.perLinkRadiusSupports(state, *group);
  ASSERT_TRUE(supports);
  ASSERT_EQ(supports->supports.size(), 1U);
  const double analytic_gradient = supports->supports.front().outward_gradient[0];

  auto plus = state;
  plus.setVariablePosition("hinge", position + epsilon);
  plus.update();
  auto minus = state;
  minus.setVariablePosition("hinge", position - epsilon);
  minus.update();
  const double numeric_gradient =
    (validator.maximumAssemblyRadius(plus) - validator.maximumAssemblyRadius(minus)) /
    (2.0 * epsilon);
  EXPECT_NEAR(analytic_gradient, numeric_gradient, 1.0e-8);
}

TEST(CompactGeometryValidatorTest, StaticDenseBaseMeshDoesNotConsumeSupportRows)
{
  constexpr std::size_t kRingVertices = 80;
  constexpr double kRadius = 0.5;
  constexpr double kHalfHeight = 0.05;
  constexpr double kTwoPi = 6.2831853071795864769;
  const auto mesh_path = std::filesystem::temp_directory_path() /
    ("compact_geometry_static_ring_" + std::to_string(::getpid()) + ".stl");
  {
    std::ofstream mesh(mesh_path);
    ASSERT_TRUE(mesh.good());
    mesh << std::setprecision(17) << "solid static_ring\n";
    const auto vertex = [&mesh, kTwoPi, kRingVertices, kRadius](
      const std::size_t index, const double z) {
        const double angle = kTwoPi * static_cast<double>(index) /
          static_cast<double>(kRingVertices);
        mesh << "vertex " << kRadius * std::cos(angle) << ' '
             << kRadius * std::sin(angle) << ' ' << z << '\n';
      };
    for (std::size_t index = 0; index < kRingVertices; ++index) {
      const std::size_t next = (index + 1U) % kRingVertices;
      for (const double z : {-kHalfHeight, kHalfHeight}) {
        mesh << "facet normal 0 0 0\nouter loop\n";
        vertex(index, z);
        vertex(next, z);
        mesh << "vertex 0 0 " << z << "\nendloop\nendfacet\n";
      }
      mesh << "facet normal 0 0 0\nouter loop\n";
      vertex(index, -kHalfHeight);
      vertex(next, -kHalfHeight);
      vertex(next, kHalfHeight);
      mesh << "endloop\nendfacet\n";
      mesh << "facet normal 0 0 0\nouter loop\n";
      vertex(index, -kHalfHeight);
      vertex(next, kHalfHeight);
      vertex(index, kHalfHeight);
      mesh << "endloop\nendfacet\n";
    }
    mesh << "endsolid static_ring\n";
  }

  const std::string description =
    "<robot name=\"static_dense_base\"><link name=\"world\"/><link name=\"base\">"
    "<collision><geometry><mesh filename=\"file://" + mesh_path.string() +
    "\"/></geometry></collision></link><link name=\"tip\"/>"
    "<joint name=\"world_to_base\" type=\"fixed\"><parent link=\"world\"/>"
    "<child link=\"base\"/></joint><joint name=\"slide\" type=\"prismatic\">"
    "<parent link=\"base\"/><child link=\"tip\"/><axis xyz=\"1 0 0\"/>"
    "<limit lower=\"-1\" upper=\"1\" effort=\"10\" velocity=\"1\"/>"
    "</joint></robot>";
  const auto model = makeGeometryTestModel(
    description,
    R"(<robot name="static_dense_base">
  <group name="arm"><chain base_link="base" tip_link="tip"/></group>
</robot>)");
  ASSERT_TRUE(model);
  const auto * group = model->getJointModelGroup("arm");
  ASSERT_NE(group, nullptr);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.update();
  const CompactGeometryValidator validator(model, "base");
  std::filesystem::remove(mesh_path);

  const auto supports = validator.perLinkRadiusSupports(state, *group);
  ASSERT_TRUE(supports);
  EXPECT_NEAR(supports->maximum_radius_m, kRadius, 1.0e-6);
  EXPECT_TRUE(supports->supports.empty());
}

TEST(CompactGeometryValidatorTest, MovingBaseRetainsGradientForWorldFixedGeometry)
{
  const auto model = makeGeometryTestModel(
    R"(
<robot name="moving_base_fixed_geometry">
  <link name="world"/><link name="base"/><link name="marker">
    <collision><origin xyz="1 0 0"/><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="world"/><child link="base"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="world_to_marker" type="fixed">
    <parent link="world"/><child link="marker"/>
  </joint>
</robot>)",
    R"(<robot name="moving_base_fixed_geometry">
  <group name="arm"><chain base_link="world" tip_link="base"/></group>
</robot>)");
  ASSERT_TRUE(model);
  const auto * group = model->getJointModelGroup("arm");
  ASSERT_NE(group, nullptr);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.setVariablePosition("slide", 0.2);
  state.update();
  const CompactGeometryValidator validator(model, "base");

  const auto supports = validator.perLinkRadiusSupports(state, *group);
  ASSERT_TRUE(supports);
  ASSERT_EQ(supports->supports.size(), 1U);
  EXPECT_NEAR(supports->supports.front().radius_m, 0.81, 1.0e-12);
  ASSERT_EQ(supports->supports.front().outward_gradient.size(), 1);
  EXPECT_NEAR(supports->supports.front().outward_gradient[0], -1.0, 1.0e-12);
}

TEST(BoundedJointReserveTest, ReportsBothPhysicalMarginsWithoutAngleWrapping)
{
  const auto reserve = measureBoundedJointReserve(5.9, -2.0 * M_PI, 2.0 * M_PI);
  ASSERT_TRUE(reserve.inside_bounds);
  EXPECT_NEAR(reserve.lower_margin_rad, 5.9 + 2.0 * M_PI, 1.0e-12);
  EXPECT_NEAR(reserve.upper_margin_rad, 2.0 * M_PI - 5.9, 1.0e-12);
  EXPECT_DOUBLE_EQ(reserve.nearest_bound_margin_rad, reserve.upper_margin_rad);
  EXPECT_FALSE(measureBoundedJointReserve(7.0, -2.0 * M_PI, 2.0 * M_PI).inside_bounds);
}

TEST_F(BackgroundRevolutePlannerTest, ExplicitJointGoalKeepsTheOriginalWideBounds)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("joint1", 5.9);
  start.update();
  static_cast<void>(planner.submit(start, Eigen::VectorXd::Constant(1, 6.0), 56));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_DOUBLE_EQ(result->path.front()[0], 5.9);
  EXPECT_NEAR(result->path.back()[0], 6.0, 0.002);
}

TEST_F(BackgroundRevolutePlannerTest, NormalizesOnlyNumericalStartBoundaryError)
{
  const auto & bounds = model_->getVariableBounds("joint1");
  const auto startAt = [this](const double position) {
      moveit::core::RobotState start(model_);
      start.setToDefaultValues();
      start.setVariablePosition("joint1", position);
      start.update();
      return start;
    };
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);

  const auto goal = Eigen::VectorXd::Constant(1, bounds.min_position_ + 0.1);
  static_cast<void>(planner.submit(
    startAt(bounds.min_position_ - 5.0e-10), goal, 57));
  const auto normalized_result = waitResult(planner);
  ASSERT_TRUE(normalized_result);
  ASSERT_TRUE(normalized_result->success) << normalized_result->message;
  EXPECT_DOUBLE_EQ(normalized_result->path.front()[0], bounds.min_position_);

  static_cast<void>(planner.submit(
    startAt(bounds.min_position_ - 2.0e-9), goal, 58));
  const auto outside_result = waitResult(planner);
  ASSERT_TRUE(outside_result);
  EXPECT_FALSE(outside_result->success);
  EXPECT_EQ(outside_result->error_code, moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID);
  EXPECT_TRUE(outside_result->path.empty());
}

TEST_F(BackgroundHorizontalEdgePlannerTest, RejectsInteriorEdgeViolationWithValidEndpoints)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  constexpr double kPi = 3.14159265358979323846;
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("edge_joint1", 0.0);
  start.setVariablePosition("edge_joint2", 0.0);
  start.update();
  moveit::core::RobotState goal_state = start;
  goal_state.setVariablePosition("edge_joint1", kPi);
  goal_state.setVariablePosition("edge_joint2", kPi);
  goal_state.update();
  const double sine_cap = std::sin(5.1 * kPi / 180.0);
  const auto edge_valid = [sine_cap](const moveit::core::RobotState & state) {
      const Eigen::Matrix3d rotation = state.getGlobalLinkTransform("tip").linear();
      return std::abs(rotation(2, 1)) <= sine_cap && rotation(2, 2) >= 0.0;
    };
  ASSERT_TRUE(edge_valid(start));
  ASSERT_TRUE(edge_valid(goal_state));
  moveit::core::RobotState violating_midpoint = start;
  violating_midpoint.setVariablePosition("edge_joint1", 0.5 * kPi);
  violating_midpoint.setVariablePosition("edge_joint2", 0.5 * kPi);
  violating_midpoint.update();
  ASSERT_FALSE(edge_valid(violating_midpoint));

  const auto generation = planner.submit(
    start, Eigen::Vector2d::Constant(kPi),
    HorizontalEdgePathRequirement{"tip", sine_cap}, 72);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->generation, generation);
  EXPECT_FALSE(result->success);
  EXPECT_NE(result->error_code, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  EXPECT_TRUE(result->path.empty());
}

TEST_F(BackgroundHighPitchEdgePlannerTest, AcceptsAValidHighPitchRestPath)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  constexpr double kPi = 3.14159265358979323846;
  const double start_angle = 85.0 * kPi / 180.0;
  const double goal_angle = 85.2 * kPi / 180.0;
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("pitch_joint", start_angle);
  start.update();
  const double sine_cap = std::sin(5.1 * kPi / 180.0);

  static_cast<void>(planner.submit(
    start, Eigen::VectorXd::Constant(1, goal_angle),
    HorizontalEdgePathRequirement{"tip", sine_cap}, 73));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_NEAR(result->path.front()[0], start_angle, 1.0e-12);
  EXPECT_NEAR(result->path.back()[0], goal_angle, config_.joint_tolerance_rad + 1.0e-12);
}

TEST_F(BackgroundHorizontalEdgePlannerTest, RestEdgeRequestDoesNotConstrainLaterOrdinaryPlan)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  constexpr double kPi = 3.14159265358979323846;
  const double sine_cap = std::sin(5.1 * kPi / 180.0);
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("edge_joint1", 0.0);
  start.update();

  const auto rest_generation = planner.submit(
    start, Eigen::Vector2d(2.0 * kPi / 180.0, 0.0),
    HorizontalEdgePathRequirement{"tip", sine_cap}, 74);
  const auto rest_result = waitResult(planner);
  ASSERT_TRUE(rest_result);
  ASSERT_TRUE(rest_result->success) << rest_result->message;
  EXPECT_EQ(rest_result->generation, rest_generation);

  moveit::core::RobotState ordinary_goal(start);
  ordinary_goal.setVariablePosition("edge_joint1", 0.5 * kPi);
  ordinary_goal.update();
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(
      ordinary_goal.getGlobalLinkTransform("tip").linear(), sine_cap));
  const auto ordinary_generation = planner.submit(
    start, Eigen::Vector2d(0.5 * kPi, 0.0), 75);
  const auto ordinary_result = waitResult(planner);
  ASSERT_TRUE(ordinary_result);
  EXPECT_EQ(ordinary_result->generation, ordinary_generation);
  EXPECT_TRUE(ordinary_result->success) << ordinary_result->message;
}

}  // namespace
}  // namespace face_tracking_arm::control
