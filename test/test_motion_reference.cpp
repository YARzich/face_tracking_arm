// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Geometry>
#include <geometric_shapes/shapes.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "face_tracking_arm/motion_reference.hpp"

namespace face_tracking_arm::control
{
namespace
{

using namespace std::chrono_literals;

constexpr std::uint64_t kSceneRevision = 11;

std::string readModelFile(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot read test robot model: " + path);
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/// The real xArm6 description is expanded by CMake, never by a running test.
/// No controller, sensor, simulator or robot command publisher is started here.
class MotionReferenceTest : public ::testing::Test
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
    std::string urdf = readModelFile(TEST_ROBOT_URDF_FILE);
    const auto closing_robot = urdf.rfind("</robot>");
    ASSERT_NE(closing_robot, std::string::npos);
    urdf.insert(closing_robot,
          R"(
      <link name="test_camera_optical"/>
      <joint name="test_camera_mount" type="fixed">
        <parent link="monitor_control_frame"/><child link="test_camera_optical"/>
        <origin xyz="0 0 0.125" rpy="-1.57079632679 0 -1.57079632679"/>
      </joint>)");
    options.parameter_overrides({
          rclcpp::Parameter("robot_description", urdf),
          rclcpp::Parameter("robot_description_semantic", readModelFile(TEST_ROBOT_SRDF_FILE)),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver",
        "kdl_kinematics_plugin/KDLKinematicsPlugin"),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver_timeout", 0.01),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.type", "geometric::RRTConnect"),
          rclcpp::Parameter(
        "recovery_planner.xarm6.planner_configs",
        std::vector<std::string>{"RRTConnectkConfigDefault"}),
          rclcpp::Parameter("recovery_planner.xarm6.longest_valid_segment_fraction", 0.005),
    });
    node_ = std::make_shared<rclcpp::Node>("motion_reference_test", options);
    monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node_, "robot_description");
    model_ = monitor_->getRobotModel();
    ASSERT_TRUE(model_);
    group_ = model_->getJointModelGroup("xarm6");
    ASSERT_NE(group_, nullptr);
    ASSERT_EQ(group_->getVariableCount(), 6U);
    parameters_.joint_names = group_->getVariableNames();
    parameters_.gaze_frame = gazeFrame();
    limits_.lower_position.resize(6);
    limits_.upper_position.resize(6);
    limits_.position_margin = Eigen::VectorXd::Constant(6, 0.10);
    limits_.max_velocity = Eigen::VectorXd::Constant(6, 1.0);
    limits_.max_acceleration = Eigen::VectorXd::Constant(6, 2.0);
    limits_.max_jerk = Eigen::VectorXd::Constant(6, 20.0);
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      const auto & bounds = model_->getVariableBounds(parameters_.joint_names[joint]);
      limits_.lower_position[joint] = bounds.min_position_;
      limits_.upper_position[joint] = bounds.max_position_;
    }
    {
      planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
      Eigen::Isometry3d tabletop = Eigen::Isometry3d::Identity();
      tabletop.translation().z() = 0.725;
      scene->getWorldNonConst()->addToObject(
        "round_table", std::make_shared<shapes::Cylinder>(0.5, 0.05), tabletop);
      scene->getAllowedCollisionMatrixNonConst().setEntry("link_base", "round_table", true);
      ASSERT_TRUE(scene->getCurrentStateNonConst().setToDefaultValues(group_, "rest"));
      scene->getCurrentStateNonConst().update();
    }
    collision_config_.protected_joint_lower_rad = limits_.lower_position[4];
    collision_config_.protected_joint_upper_rad = limits_.upper_position[4];
    collision_config_.invariant_mount_neighbor_name = "link6";
    configureGeometryBounds(collision_config_, *model_, *group_);
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    clock_origin_ = std::chrono::steady_clock::now();
  }

  virtual std::string gazeFrame() const {return "";}

  moveit::core::RobotState state(const double base_angle = 0.0) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    if (!result.setToDefaultValues(group_, "rest")) {
      throw std::runtime_error("The real robot model must contain a named rest state");
    }
    result.setVariablePosition("joint1", base_angle);
    result.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    result.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    result.update();
    return result;
  }

  msg::TrackingTarget target(const std::uint8_t mode) const
  {
    msg::TrackingTarget result;
    result.header.frame_id = "world";
    result.mode = mode;
    result.pose.orientation.w = 1.0;
    result.pose.position.x = 0.2;
    result.pose.position.z = 1.05;
    result.face.x = 0.85;
    result.face.z = 1.68;
    return result;
  }

  MotionReferenceResult update(
    const moveit::core::RobotState & robot_state,
    const std::optional<msg::TrackingTarget> & command)
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    return reference_->update(robot_state, command, *planning_scene, now, kSceneRevision);
  }

  std::optional<MotionReferenceResult> waitForRestPath(
    const moveit::core::RobotState & robot_state)
  {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto result = update(robot_state, target(msg::TrackingTarget::REST));
      if (result.follows_path) {
        return result;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  }

  void expectPendingRestPreempted(const std::uint8_t new_mode)
  {
    const auto start = state(0.4);
    const auto waiting = update(start, target(msg::TrackingTarget::REST));
    EXPECT_FALSE(waiting.task);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
    // The first update submits but cannot adopt its newly submitted result.
    // Change mode immediately, then keep polling past the planner's 250 ms budget
    // so a late completion from REST has an opportunity to reach the selector.
    const auto deadline = std::chrono::steady_clock::now() + 350ms;
    do {
      const auto selected = update(start, target(new_mode));
      EXPECT_FALSE(selected.follows_path);
      EXPECT_EQ(reference_->diagnostics().plans_accepted, 0U);
      if (new_mode == msg::TrackingTarget::HOLD) {
        EXPECT_FALSE(selected.task);
        EXPECT_EQ(reference_->diagnostics().state, "HOLD");
      } else {
        EXPECT_TRUE(selected.tracking);
        EXPECT_EQ(reference_->diagnostics().state, "FACE");
      }
      std::this_thread::sleep_for(2ms);
    } while (std::chrono::steady_clock::now() < deadline);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  }

  bool adoptRelaxedDetour(
    const moveit::core::RobotState & start, double & now, unsigned int desired_level = 2)
  {
    auto command = target(msg::TrackingTarget::FACE);
    command.face.x = -0.85;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      now += 0.02;
      command.face_stamp = rclcpp::Time(static_cast<int64_t>(now * 1.0e9));
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      const auto result = reference_->update(start, command, *planning_scene, now, kSceneRevision);
      if (result.follows_path) {
        if (reference_->diagnostics().recovery_relaxation >= desired_level) {
          return reference_->diagnostics().recovery_relaxation == desired_level;
        }
        // Exercise the same escalation used when the executor rejects a path.
        reference_->rejectPath();
      }
      std::this_thread::sleep_for(2ms);
    }
    return false;
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene_monitor::PlanningSceneMonitorPtr monitor_;
  moveit::core::RobotModelConstPtr model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  ControllerParameters parameters_;
  JointMotionLimits limits_;
  CollisionConstraintConfig collision_config_;
  std::unique_ptr<MotionReference> reference_;
  std::chrono::steady_clock::time_point clock_origin_;
};

class MotionReferenceCameraTest : public MotionReferenceTest
{
protected:
  std::string gazeFrame() const override {return "test_camera_optical";}
};

TEST_F(MotionReferenceCameraTest, OpticalFaceCenterUsesTheMountedCameraOrigin)
{
  const auto start = state();
  const auto & camera = start.getGlobalLinkTransform("test_camera_optical");
  const auto & monitor = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d face = camera.translation() + 0.8 * camera.linear().col(2);
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();
  command.pose.position.x = monitor.translation().x();
  command.pose.position.y = monitor.translation().y();
  command.pose.position.z = monitor.translation().z();
  const auto result = update(start, command);
  ASSERT_TRUE(result.tracking);
  EXPECT_NEAR(result.tracking->pointing_error_rad, 0.0, 1.0e-6);
  EXPECT_TRUE(result.tracking->task.primary_reference.isZero(1.0e-6));
  EXPECT_GT(std::acos(std::clamp(monitor.linear().col(0).dot(
      (face - monitor.translation()).normalized()), -1.0, 1.0)), 0.10);
}

TEST_F(MotionReferenceTest, HoldAndMissingTargetProduceNoTaskOrPlanning)
{
  const auto start = state(0.4);
  for (const auto & command : {std::optional<msg::TrackingTarget>{},
      std::optional<msg::TrackingTarget>{target(msg::TrackingTarget::HOLD)}})
  {
    const auto result = update(start, command);
    EXPECT_FALSE(result.task);
    EXPECT_FALSE(result.tracking);
    EXPECT_FALSE(result.follows_path);
    EXPECT_EQ(reference_->diagnostics().state, "HOLD");
    EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  }
}

TEST_F(MotionReferenceTest, PerceptionLossMayFinishDetourButExplicitHoldAndPauseCancel)
{
  const auto start = state();
  double now = 0.0;
  ASSERT_TRUE(adoptRelaxedDetour(start, now)) << reference_->diagnostics().last_plan_message;
  auto lost = target(msg::TrackingTarget::HOLD);
  lost.allow_recovery = true;
  const auto selected = [&](const std::optional<msg::TrackingTarget> & command) {
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(start, command, *planning_scene, now += 0.01, kSceneRevision);
    };
  const auto continued = selected(lost);
  ASSERT_TRUE(continued.follows_path);
  ASSERT_TRUE(continued.task);
  EXPECT_TRUE(continued.task->primary_matrix.isIdentity());
  EXPECT_FALSE(continued.tracking);  // Stale face is never a new observation.
  EXPECT_FALSE(selected(std::nullopt).follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");

  ASSERT_TRUE(adoptRelaxedDetour(start, now)) << reference_->diagnostics().last_plan_message;
  lost.allow_recovery = false;
  EXPECT_FALSE(selected(lost).follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  auto reacquired = target(msg::TrackingTarget::FACE);
  reacquired.face.x = -0.85;
  (void)selected(reacquired);
  // Pause stops motion, but does not repeat already unsuccessful restrictive
  // recovery levels after every intermittent observation or feedback rearm.
  EXPECT_EQ(reference_->diagnostics().recovery_relaxation, 2U);
}

TEST_F(MotionReferenceTest, IntermediateRecoveryFreesMotionInsideItsSoftPointingCone)
{
  const auto start = state();
  double now = 0.0;
  ASSERT_TRUE(adoptRelaxedDetour(start, now, 1)) << reference_->diagnostics().last_plan_message;
  const auto plans_accepted = reference_->diagnostics().plans_accepted;
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  for (const double angle : {0.20, 0.50, 0.20}) {
    const Eigen::Vector3d face = tcp.translation() + 0.8 *
      (Eigen::AngleAxisd(angle, tcp.linear().col(2)) * tcp.linear().col(0));
    auto command = target(msg::TrackingTarget::FACE);
    command.face.x = face.x();
    command.face.y = face.y();
    command.face.z = face.z();
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto result = reference_->update(start, command, *planning_scene, now += 0.01,
          kSceneRevision);
    ASSERT_TRUE(result.task);
    ASSERT_TRUE(result.follows_path);
    EXPECT_EQ(reference_->diagnostics().plans_accepted, plans_accepted);
    if (angle < 0.35) {
      EXPECT_TRUE(result.task->primary_matrix.isIdentity());
      EXPECT_GT(result.task->primary_reference.norm(), 0.0);
    } else {
      EXPECT_EQ(result.task->primary_matrix.rows(), 2);
      EXPECT_TRUE(result.task->secondary_matrix.isIdentity());
      EXPECT_GT(result.task->primary_reference.norm(), 0.0);
    }
  }
}

TEST_F(MotionReferenceTest, LastRecoveryLevelPlansOnlyAfterVelocityAndAccelerationSettle)
{
  auto start = state();
  double now = 0.0;
  ASSERT_TRUE(adoptRelaxedDetour(start, now)) << reference_->diagnostics().last_plan_message;
  reference_->rejectPath();
  const auto requests_before = reference_->diagnostics().plans_requested;
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d face = tcp.translation() + 0.8 *
    (Eigen::AngleAxisd(0.3, tcp.linear().col(2)) * tcp.linear().col(0));
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();
  // Move past the retry cooldown. Neither ongoing motion nor zero velocity
  // with remaining acceleration is an appropriate start for a geometric path.
  for (const Eigen::Vector2d derivatives : {Eigen::Vector2d(0.12, 0.0),
      Eigen::Vector2d(0.0, 0.04), Eigen::Vector2d::Zero().eval()})
  {
    start.setJointGroupVelocities(group_, Eigen::VectorXd::Constant(6, derivatives[0]));
    start.setJointGroupAccelerations(group_, Eigen::VectorXd::Constant(6, derivatives[1]));
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto result = reference_->update(start, command, *planning_scene, now += 2.0,
          kSceneRevision);
    ASSERT_TRUE(result.tracking);
    EXPECT_NEAR(result.tracking->pointing_error_rad, 0.3, 1.0e-6);
    EXPECT_FALSE(result.task);
    EXPECT_FALSE(result.follows_path);
    EXPECT_EQ(reference_->diagnostics().state, "RECOVERY_PLANNING");
    EXPECT_EQ(reference_->diagnostics().plans_requested,
      requests_before + (derivatives.isZero() ? 1U : 0U));
  }
}

TEST_F(MotionReferenceTest, StaticMisalignmentEventuallyRequestsRecovery)
{
  const auto start = state();
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d face = tcp.translation() + 0.8 *
    (Eigen::AngleAxisd(0.3, tcp.linear().col(2)) * tcp.linear().col(0));
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();
  for (const double now : {0.0, 1.0, 1.6}) {
    command.face_stamp = rclcpp::Time(static_cast<int64_t>(now * 1.0e9));
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto result = reference_->update(start, command, *planning_scene, now, kSceneRevision);
    EXPECT_TRUE(result.tracking);
    EXPECT_TRUE(result.task);  // The initial recovery level keeps live tracking.
  }
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}

TEST_F(MotionReferenceTest, NamedRestPoseHoldsWithoutRequestingAPlan)
{
  const auto result = update(state(), target(msg::TrackingTarget::REST));
  EXPECT_FALSE(result.task);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "REST_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, FaceBehindRestRequiresAPathBeforeAnyLocalMotion)
{
  auto start = state();
  const Eigen::VectorXd velocity =
    (Eigen::VectorXd(6) << 0.005, -0.002, 0.0, 0.0, 0.0, 0.0).finished();
  const Eigen::VectorXd acceleration =
    (Eigen::VectorXd(6) << 0.015, 0.0, 0.0, 0.0, 0.0, 0.0).finished();
  start.setJointGroupVelocities(group_, velocity);
  start.setJointGroupAccelerations(group_, acceleration);
  Eigen::VectorXd position_before;
  start.copyJointGroupPositions(group_, position_before);

  const Eigen::Vector3d face(-0.85, 0.0, 1.68);
  const Eigen::Vector3d center = start.getGlobalLinkTransform("link1").translation();
  // The same 0.42 m reach sphere as the human-height simulation target.
  const Eigen::Vector3d goal_position = center + 0.42 * (face - center).normalized();
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();
  command.pose.position.x = goal_position.x();
  command.pose.position.y = goal_position.y();
  command.pose.position.z = goal_position.z();
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Quaterniond pointing_error = Eigen::Quaterniond::FromTwoVectors(
    tcp.linear().col(0), (face - tcp.translation()).normalized());
  ASSERT_GT(Eigen::AngleAxisd(pointing_error).angle(), 2.0);

  const auto first = update(start, command);
  EXPECT_FALSE(first.task);
  EXPECT_FALSE(first.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "ACQUIRE_PLANNING");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 0U);

  std::optional<MotionReferenceResult> adopted;
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto selected = update(start, command);
    if (selected.follows_path) {
      adopted = std::move(selected);
      break;
    }
    EXPECT_FALSE(selected.task);
    EXPECT_EQ(reference_->diagnostics().state, "ACQUIRE_PLANNING");
    std::this_thread::sleep_for(2ms);
  }
  ASSERT_TRUE(adopted) << reference_->diagnostics().state << ": " <<
    reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(adopted->task);
  EXPECT_EQ(reference_->diagnostics().state, "POSTURE_PATH");
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  if (reference_->diagnostics().recovery_relaxation < 2) {
    EXPECT_EQ(adopted->task->primary_matrix.rows(), 2);
    EXPECT_TRUE(adopted->task->secondary_matrix.isIdentity());
  } else {
    EXPECT_TRUE(adopted->task->primary_matrix.isIdentity());
  }
  EXPECT_GT(adopted->task->primary_reference.norm(), 0.0);

  Eigen::VectorXd position_after;
  Eigen::VectorXd velocity_after;
  Eigen::VectorXd acceleration_after;
  start.copyJointGroupPositions(group_, position_after);
  start.copyJointGroupVelocities(group_, velocity_after);
  start.copyJointGroupAccelerations(group_, acceleration_after);
  EXPECT_TRUE(position_after.isApprox(position_before, 0.0));
  EXPECT_TRUE(velocity_after.isApprox(velocity, 0.0));
  EXPECT_TRUE(acceleration_after.isApprox(acceleration, 0.0));
}

TEST_F(MotionReferenceTest, AcquisitionRemainsLatchedAcrossThePointingThreshold)
{
  const auto start = state();
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d first_ray =
    Eigen::AngleAxisd(2.01, tcp.linear().col(2)) * tcp.linear().col(0);
  const Eigen::Vector3d second_ray =
    Eigen::AngleAxisd(1.99, tcp.linear().col(2)) * tcp.linear().col(0);
  const Eigen::Vector3d first_face = tcp.translation() + 0.8 * first_ray;
  const Eigen::Vector3d second_face = tcp.translation() + 0.8 * second_ray;
  ASSERT_LT((second_face - first_face).norm(), 0.5);
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = first_face.x();
  command.face.y = first_face.y();
  command.face.z = first_face.z();
  const auto first = update(start, command);
  ASSERT_FALSE(first.task);
  ASSERT_EQ(reference_->diagnostics().state, "ACQUIRE_PLANNING");
  ASSERT_EQ(reference_->diagnostics().plans_requested, 1U);

  command.face.x = second_face.x();
  command.face.y = second_face.y();
  command.face.z = second_face.z();
  const auto second = update(start, command);
  EXPECT_TRUE(second.tracking);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  if (second.follows_path) {
    // A fast worker may already have completed the original request. Only its
    // adopted joint path may replace the waiting state, never local tracking.
    EXPECT_TRUE(second.task);
    EXPECT_EQ(reference_->diagnostics().state, "POSTURE_PATH");
    EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  } else {
    EXPECT_FALSE(second.task);
    EXPECT_EQ(reference_->diagnostics().state, "ACQUIRE_PLANNING");
  }
}

TEST_F(MotionReferenceTest, RealClearancePredicateAllowsRestPlanningAndPathAdoption)
{
  const auto start = state(0.4);
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState motion{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto validity = CollisionConstraintBuilder(collision_config_).validateSegment(
      *planning_scene, start, *group_, motion, motion, 2);
    ASSERT_TRUE(validity.input_valid) << validity.failure_reason;
    ASSERT_FALSE(validity.unsafe) << validity.failure_reason;
  }
  const auto result = waitForRestPath(start);
  ASSERT_TRUE(result) << reference_->diagnostics().state << ": " <<
    reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(result->task);
  EXPECT_TRUE(result->follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "REST_PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_TRUE(result->task->primary_matrix.isIdentity());
  EXPECT_GT(result->task->primary_reference.norm(), 0.0);
  EXPECT_LE(result->task->primary_reference.cwiseAbs().maxCoeff(), 0.70 + 1.0e-9);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 0.4);
}

TEST_F(MotionReferenceTest, HoldCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::HOLD);
}

TEST_F(MotionReferenceTest, FaceCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::FACE);
}

TEST_F(MotionReferenceTest, HoldImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(result.task);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}

TEST_F(MotionReferenceTest, FaceImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::FACE));
  EXPECT_TRUE(result.tracking);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}

TEST_F(MotionReferenceTest, RejectedRestSettlingReturnsToPlanningWithoutResettingMotion)
{
  auto moving = state(0.4);
  auto selected = waitForRestPath(moving);
  ASSERT_TRUE(selected) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().state, "REST_PATH");

  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd acceleration = Eigen::VectorXd::Zero(6);
  velocity[0] = 0.04;
  acceleration[0] = 0.05;
  moving.setJointGroupVelocities(group_, velocity);
  moving.setJointGroupAccelerations(group_, acceleration);
  // Advance along the returned references rather than jumping past the follower's
  // local segment window. Nonzero derivatives prevent automatic REST_HOLD.
  for (int step = 0; step < 500 && reference_->diagnostics().state == "REST_PATH"; ++step) {
    ASSERT_TRUE(selected->task);
    ASSERT_TRUE(selected->task->primary_matrix.isIdentity());
    Eigen::VectorXd position;
    moving.copyJointGroupPositions(group_, position);
    position += 0.02 * selected->task->primary_reference;
    moving.setJointGroupPositions(group_, position);
    moving.update();
    selected = update(moving, target(msg::TrackingTarget::REST));
  }
  ASSERT_EQ(reference_->diagnostics().state, "REST_SETTLING");
  ASSERT_EQ(reference_->diagnostics().paths_completed, 1U);

  moving = state(0.01);
  moving.setJointGroupVelocities(group_, velocity);
  moving.setJointGroupAccelerations(group_, acceleration);
  const auto braking = update(moving, target(msg::TrackingTarget::REST));
  EXPECT_FALSE(braking.task);
  EXPECT_EQ(reference_->diagnostics().state, "REST_SETTLING");
  moving = state(0.04);
  moving.setJointGroupVelocities(group_, velocity);
  moving.setJointGroupAccelerations(group_, acceleration);
  const auto settling = update(moving, target(msg::TrackingTarget::REST));
  ASSERT_TRUE(settling.follows_path);
  ASSERT_TRUE(settling.task);
  ASSERT_EQ(reference_->diagnostics().state, "REST_SETTLING");
  Eigen::VectorXd position_before;
  moving.copyJointGroupPositions(group_, position_before);
  const auto abandoned_before = reference_->diagnostics().paths_abandoned;

  reference_->rejectPath();
  const auto replanning = update(moving, target(msg::TrackingTarget::REST));
  EXPECT_EQ(reference_->diagnostics().state, "REST_PLANNING");
  EXPECT_FALSE(replanning.follows_path);
  EXPECT_FALSE(replanning.task);
  EXPECT_EQ(reference_->diagnostics().paths_abandoned, abandoned_before + 1U);

  Eigen::VectorXd position_after;
  Eigen::VectorXd velocity_after;
  Eigen::VectorXd acceleration_after;
  moving.copyJointGroupPositions(group_, position_after);
  moving.copyJointGroupVelocities(group_, velocity_after);
  moving.copyJointGroupAccelerations(group_, acceleration_after);
  EXPECT_TRUE(position_after.isApprox(position_before, 0.0));
  EXPECT_TRUE(velocity_after.isApprox(velocity, 0.0));
  EXPECT_TRUE(acceleration_after.isApprox(acceleration, 0.0));
}

TEST_F(MotionReferenceTest, StaticFaceDoesNotUnwindAWideJoint)
{
  const auto start = state(4.0);
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Quaterniond orientation(tcp.linear());
  const Eigen::Vector3d face = tcp.translation() + 0.8 * tcp.linear().col(0);
  auto command = target(msg::TrackingTarget::FACE);
  command.pose.position.x = tcp.translation().x();
  command.pose.position.y = tcp.translation().y();
  command.pose.position.z = tcp.translation().z();
  command.pose.orientation.x = orientation.x();
  command.pose.orientation.y = orientation.y();
  command.pose.orientation.z = orientation.z();
  command.pose.orientation.w = orientation.w();
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();

  // Fresh observations of a stationary face must preserve tracking without
  // requesting a posture change solely because of the current q1 angle.
  for (int step = 1; step <= 100; ++step) {
    const double now = step * 0.02;
    command.face_stamp = rclcpp::Time(static_cast<int64_t>(now * 1.0e9));
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto selected = reference_->update(start, command, *planning_scene, now, kSceneRevision);
    EXPECT_TRUE(selected.tracking);
    EXPECT_FALSE(selected.follows_path);
    EXPECT_EQ(reference_->diagnostics().state, "FACE");
    EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  }
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 4.0);
}

TEST_F(MotionReferenceTest, AdoptedPosturePathSurvivesFaceMotionButYieldsToHold)
{
  double simulated_time = 1.0;
  const auto update = [this, &simulated_time](
    const moveit::core::RobotState & state, msg::TrackingTarget command)
    {
      simulated_time += 0.02;
      command.face_stamp = rclcpp::Time(static_cast<int64_t>(simulated_time * 1.0e9));
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(state, command, *planning_scene, simulated_time, kSceneRevision);
    };
  auto start = state(5.0);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[0] = 0.13;
  start.setJointGroupVelocities(group_, velocity);
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Quaterniond orientation(tcp.linear());
  const Eigen::Vector3d face = tcp.translation() + 0.8 * tcp.linear().col(0);
  auto command = target(msg::TrackingTarget::FACE);
  command.pose.position.x = tcp.translation().x();
  command.pose.position.y = tcp.translation().y();
  command.pose.position.z = tcp.translation().z();
  command.pose.orientation.x = orientation.x();
  command.pose.orientation.y = orientation.y();
  command.pose.orientation.z = orientation.z();
  command.pose.orientation.w = orientation.w();
  command.face.x = face.x();
  command.face.y = face.y();
  command.face.z = face.z();

  // Seed a stationary sample, then observe motion while q1 approaches its end.
  (void)update(start, command);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  command.face.x += 0.004;
  const auto first = update(start, command);
  EXPECT_TRUE(first.tracking);
  EXPECT_FALSE(first.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);

  std::optional<MotionReferenceResult> adopted;
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto result = update(start, command);
    if (result.follows_path) {
      adopted = std::move(result);
      break;
    }
    std::this_thread::sleep_for(2ms);
  }
  ASSERT_TRUE(adopted) << reference_->diagnostics().state << ": " <<
    reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(adopted->task);
  ASSERT_EQ(reference_->diagnostics().state, "POSTURE_PATH");
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  ASSERT_GT(adopted->task->primary_reference.norm() +
    adopted->task->secondary_reference.norm(), 0.0);
  const auto requests_before = reference_->diagnostics().plans_requested;
  const auto abandoned_before = reference_->diagnostics().paths_abandoned;

  // The same selected face has moved farther than the pending-plan retarget
  // threshold. An adopted maneuver must retain its path instead of restarting
  // on every short arc of a person walking around the robot.
  command.face.x += 0.6;
  const auto continued = update(start, command);
  ASSERT_TRUE(continued.follows_path);
  ASSERT_TRUE(continued.task);
  EXPECT_EQ(reference_->diagnostics().state, "POSTURE_PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_before);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().paths_abandoned, abandoned_before);
  // The path remains adopted while its pointing rows follow the fresh face.
  if (reference_->diagnostics().recovery_relaxation < 2) {
    EXPECT_TRUE(continued.task->secondary_reference.isApprox(
        adopted->task->secondary_reference, 1.0e-12));
  }

  const auto held = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_FALSE(held.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_before);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 5.0);
}

TEST_F(MotionReferenceTest, SearchUsesTheSameExecutorAndIsPreemptedByHoldAndFace)
{
  auto start = state();
  ASSERT_TRUE(start.setToDefaultValues(group_, "search"));
  start.update();
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_TRUE(scanned.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_SWEEP");
  EXPECT_GT(scanned.task->primary_reference[0], 0.0);
  EXPECT_LE(scanned.task->primary_reference.norm(), .30001);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[0] = .2;
  start.setJointGroupVelocities(group_, velocity);
  const auto held = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
  auto face = target(msg::TrackingTarget::FACE);
  face.face.x = 2.0;
  const auto tracked = update(start, face);
  EXPECT_FALSE(tracked.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
}

TEST_F(MotionReferenceTest, LocalSearchKeepsTheLastPostureAndStartsWithoutPlanning)
{
  auto start = state(.7);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_GT(scanned.task->primary_reference[0], 0.0);
  EXPECT_LT(scanned.task->primary_reference.tail(5).norm(), 1e-9);
}

}  // namespace
}  // namespace face_tracking_arm::control
