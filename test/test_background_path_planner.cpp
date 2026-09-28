// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "face_tracking_arm/background_path_planner.hpp"

namespace face_tracking_arm::control
{
namespace
{
using namespace std::chrono_literals;

constexpr char kRobotDescription[] =
  R"(
<robot name="planner_slider">
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
</robot>)";

constexpr char kRobotSemantic[] =
  R"(
<robot name="planner_slider">
  <group name="slider"><chain base_link="base" tip_link="tip"/></group>
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
          rclcpp::Parameter("robot_description_semantic", kRobotSemantic),
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

TEST_F(BackgroundPathPlannerTest, ResolvesPoseGoalUsingTheConfiguredKdlPlugin)
{
  ASSERT_TRUE(model_->getJointModelGroup("slider")->getSolverInstance());
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.pose = state(0.45, 0.2).getGlobalLinkTransform("tip");
  const auto start = state(-0.6, 0.2);
  const auto generation = planner.submit(start, goal, 51);
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_EQ(result->generation, generation);
  EXPECT_EQ(result->scene_revision, 51U);
  EXPECT_NEAR(result->goal_positions[0], 0.45, 0.001);
  EXPECT_NEAR(result->goal_positions[1], 0.2, 0.001);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("x"), -0.6);
  EXPECT_TRUE(result->path.front().isApprox(Eigen::Vector2d(-0.6, 0.2)));
}

TEST_F(BackgroundPathPlannerTest, RejectsPoseWhenIkCannotSatisfyAdditionalClearance)
{
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_,
    [](const planning_scene::PlanningScene &, const moveit::core::RobotState & candidate) {
      return candidate.getVariablePosition("y") >= 0.0;
    });
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.pose = state(0.45, -0.2).getGlobalLinkTransform("tip");
  static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 52));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(result->error_code, moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION);
  EXPECT_TRUE(result->path.empty());
}

TEST_F(BackgroundPathPlannerTest, PointingGoalDoesNotRequireThePreferredCartesianPosition)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.pose.translation() = Eigen::Vector3d(10.0, 10.0, 10.0);
  goal.face_position = Eigen::Vector3d(2.0, 0.0, 0.0);
  static_cast<void>(planner.submit(state(-0.6, 0.2), goal, 70));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  auto achieved = state(result->goal_positions[0], result->goal_positions[1]);
  const auto & pose = achieved.getGlobalLinkTransform("tip");
  const Eigen::Vector3d direction = (*goal.face_position - pose.translation()).normalized();
  EXPECT_GT(pose.linear().col(0).dot(direction), std::cos(0.025));
  EXPECT_GT((pose.translation() - goal.pose.translation()).norm(), 10.0);
}

TEST_F(BackgroundPathPlannerTest, FailedPointingMayRelaxToDiverseCheckedJointDetours)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.face_position = Eigen::Vector3d(0.0, 0.0, 2.0);
  const auto start = state(-0.6, 0.2);
  static_cast<void>(planner.submit(start, goal, 71));
  auto result = waitResult(planner);
  ASSERT_TRUE(result);
  EXPECT_FALSE(result->success);  // The slider cannot point upward.
  goal.relaxation = 2;
  Eigen::VectorXd first_goal;
  for (const std::uint64_t attempt : {2U, 5U}) {
    goal.attempt = attempt;
    static_cast<void>(planner.submit(start, goal, 72));
    result = waitResult(planner);
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->success) << result->message;
    EXPECT_GT((result->goal_positions - result->start_positions).norm(), 0.08);
    if (first_goal.size() != 0) {
      EXPECT_GT((first_goal - result->goal_positions).norm(), 0.01);
    }
    first_goal = result->goal_positions;
    for (const auto & waypoint : result->path) {
      EXPECT_LE(waypoint.cwiseAbs().maxCoeff(), 1.0);
    }
  }
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

TEST_F(BackgroundRevolutePlannerTest, CentersOnlyIkSeedsAndPlansFromTheActualWoundStart)
{
  BackgroundPathPlanner planner(node_, model_, monitor_, config_);
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("joint1", 5.9);
  start.update();
  moveit::core::RobotState desired(start);
  desired.setVariablePosition("joint1", 0.2);
  desired.update();
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.pose = desired.getGlobalLinkTransform("tip");
  goal.preferred_base_angle_rad = 0.2;
  static_cast<void>(planner.submit(start, goal, 53));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_NEAR(result->goal_positions[0], 0.2, 0.001);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 5.9);
  EXPECT_DOUBLE_EQ(result->path.front()[0], 5.9);
  EXPECT_NEAR(result->path.back()[0], 0.2, 0.002);
}

TEST_F(BackgroundRevolutePlannerTest, AllowsSafeEndpointWithoutMandatoryCompleteUnwinding)
{
  BackgroundPathPlanner planner(
    node_, model_, monitor_, config_,
    [](const planning_scene::PlanningScene &, const moveit::core::RobotState & candidate) {
      return candidate.getVariablePosition("joint1") > 4.0;
    });
  moveit::core::RobotState start(model_);
  start.setToDefaultValues();
  start.setVariablePosition("joint1", 5.9);
  start.update();
  moveit::core::RobotState desired(start);
  desired.setVariablePosition("joint1", -0.2);
  desired.update();
  BackgroundPathPlanner::PoseGoal goal;
  goal.link_name = "tip";
  goal.pose = desired.getGlobalLinkTransform("tip");
  goal.preferred_base_angle_rad = -0.2;
  static_cast<void>(planner.submit(start, goal, 55));
  const auto result = waitResult(planner);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->success) << result->message;
  EXPECT_NEAR(result->goal_positions[0], 2.0 * M_PI - 0.2, 0.001);
  for (const auto & waypoint : result->path) {
    EXPECT_GT(waypoint[0], 4.0);
  }
  EXPECT_DOUBLE_EQ(result->start_positions[0], 5.9);
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

}  // namespace
}  // namespace face_tracking_arm::control
