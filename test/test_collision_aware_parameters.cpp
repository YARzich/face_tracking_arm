// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit_servo/moveit_servo_lib_parameters.hpp>

#include "face_tracking_arm/collision_aware_parameters.hpp"

namespace face_tracking_arm
{
namespace
{

constexpr std::array<const char *, 12> kDeprecatedNames{{
  "joint_centering_activation_fraction",
  "joint_centering_max_velocity_rad_s",
  "joint_centering_weight",
  "collision_avoidance_buffer_m",
  "collision_avoidance_max_velocity_mps",
  "collision_avoidance_lookahead_sec",
  "collision_avoidance_weight",
  "lower_singularity_threshold",
  "hard_stop_singularity_threshold",
  "singularity_recovery_gain",
  "singularity_gradient_step_rad",
  "monitor_near_distance_lipschitz_m_per_rad",
}};

class ControllerParameterTest : public ::testing::Test
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

  static rclcpp::Node::SharedPtr makeNode(std::vector<rclcpp::Parameter> overrides = {})
  {
    rclcpp::NodeOptions options;
    options.use_global_arguments(false);
    options.enable_rosout(false);
    options.start_parameter_services(false);
    options.start_parameter_event_publisher(false);
    options.parameter_overrides(std::move(overrides));
    auto node = std::make_shared<rclcpp::Node>("controller_parameter_test", options);
    node->get_logger().set_level(rclcpp::Logger::Level::Warn);
    return node;
  }
};

TEST_F(ControllerParameterTest, DefaultsRemainValidWithoutDeprecationWarnings)
{
  auto node = makeNode();
  testing::internal::CaptureStderr();
  const auto parameters = declareControllerParameters(*node);
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(output.find("deprecated"), std::string::npos);
  EXPECT_NO_THROW(validateControllerParameters(parameters));
  for (const auto * name : kDeprecatedNames) {
    EXPECT_EQ(node->get_parameter(name).get_type(), rclcpp::ParameterType::PARAMETER_DOUBLE);
    EXPECT_NE(node->describe_parameter(name).description.find("ignored"), std::string::npos);
  }
}

TEST_F(ControllerParameterTest, DeprecatedOverridesWarnButDoNotRejectWorkingConfiguration)
{
  std::vector<rclcpp::Parameter> overrides;
  // These values were rejected despite having no influence on motion.
  for (const auto * name : kDeprecatedNames) {
    overrides.emplace_back(name, -1.0);
  }
  overrides.back() = rclcpp::Parameter(
    "monitor_near_distance_lipschitz_m_per_rad",
    std::numeric_limits<double>::quiet_NaN());
  overrides.emplace_back("orientation_gain", 3.0);
  auto node = makeNode(std::move(overrides));
  testing::internal::CaptureStderr();
  const auto parameters = declareControllerParameters(*node);
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_NO_THROW(validateControllerParameters(parameters));
  EXPECT_DOUBLE_EQ(parameters.orientation_gain, 3.0);
  for (const auto * name : kDeprecatedNames) {
    if (std::string(name) == "monitor_near_distance_lipschitz_m_per_rad") {
      EXPECT_TRUE(std::isnan(node->get_parameter(name).as_double()));
    } else {
      EXPECT_DOUBLE_EQ(node->get_parameter(name).as_double(), -1.0);
    }
    EXPECT_NE(output.find(std::string("Parameter '") + name + "' is deprecated"),
      std::string::npos);
  }
}

TEST_F(ControllerParameterTest, ExplicitDeprecatedDefaultAlsoWarns)
{
  auto node = makeNode({rclcpp::Parameter("lower_singularity_threshold", 40.0)});
  testing::internal::CaptureStderr();
  const auto parameters = declareControllerParameters(*node);
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_NO_THROW(validateControllerParameters(parameters));
  EXPECT_NE(output.find("Parameter 'lower_singularity_threshold' is deprecated"),
    std::string::npos);
}

TEST_F(ControllerParameterTest, ActiveOverridesStillUseSafetyValidation)
{
  auto node = makeNode({rclcpp::Parameter("hard_clearance_m", 0.0)});
  const auto parameters = declareControllerParameters(*node);
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
}

TEST(ControllerParameterValidation, RejectsUnsafeMotionFeedbackAndCollisionSettings)
{
  const std::vector<std::pair<double ControllerParameters::*, double>> invalid_values{
    {&ControllerParameters::hard_clearance_m, 0.014},
    {&ControllerParameters::collision_query_distance_m, 0.10},
    {&ControllerParameters::residual_command_latency_sec, 0.0},
    {&ControllerParameters::numerical_distance_reserve_m, 0.0},
    {&ControllerParameters::state_feedback_timeout_sec, 0.20},
    {&ControllerParameters::following_position_tolerance_rad, 0.001},
    {&ControllerParameters::control_period_sec, 0.030},
    {&ControllerParameters::solver_time_limit_sec, 0.020},
    {&ControllerParameters::maximum_linear_reference_mps, 0.0},
    {&ControllerParameters::maximum_angular_reference_radps,
      std::numeric_limits<double>::quiet_NaN()},
  };
  for (std::size_t index = 0; index < invalid_values.size(); ++index) {
    SCOPED_TRACE(index);
    ControllerParameters parameters;
    const auto & [member, value] = invalid_values[index];
    parameters.*member = value;
    EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
  }
}

TEST(ControllerParameterValidation, HardwareConfigurationLimitsMatchCppBoundaries)
{
  ControllerParameters parameters;
  parameters.position_gain = 0.0001;
  parameters.solver_time_limit_sec = 0.003;
  parameters.maximum_collision_constraints = 24;
  EXPECT_NO_THROW(validateControllerParameters(parameters));

  parameters.maximum_collision_constraints = 64;
  EXPECT_NO_THROW(validateControllerParameters(parameters));

  parameters.maximum_collision_constraints = 23;
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
  parameters.maximum_collision_constraints = 65;
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);

  parameters.maximum_collision_constraints = 24;
  parameters.solver_time_limit_sec = 0.003001;
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
  parameters.solver_time_limit_sec = 0.003;
  parameters.position_gain = 0.0;
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
}

TEST(ControllerParameterValidation, EmptyMonitorGuardIgnoresItsUnusedLimits)
{
  ControllerParameters parameters;
  parameters.monitor_guard_min_position_rad =
    std::numeric_limits<double>::quiet_NaN();
  parameters.monitor_guard_max_position_rad =
    -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(validateControllerParameters(parameters));

  parameters.monitor_guard_joint_name = "joint_5";
  EXPECT_THROW(validateControllerParameters(parameters), std::invalid_argument);
}

TEST_F(ControllerParameterTest, CustomServoHelperUsesSharedSettingsAndFixedOutputContract)
{
  auto node = makeNode({
        rclcpp::Parameter("moveit_servo.thread_priority", std::int64_t{17}),
        rclcpp::Parameter("moveit_servo.max_expected_latency", 0.06),
        rclcpp::Parameter("moveit_servo.command_out_topic",
          std::string("/custom/joint_trajectory")),
        rclcpp::Parameter("moveit_servo.joint_topic", std::string("/custom/joint_states")),
        rclcpp::Parameter(
        "moveit_servo.monitored_planning_scene_topic", std::string("/custom/planning_scene")),
        rclcpp::Parameter("moveit_servo.is_primary_planning_scene_monitor", false),
      // Full legacy Servo YAML settings remain accepted as overrides, but the
      // custom backend deliberately doesn't consume or constrain these values.
        rclcpp::Parameter("moveit_servo.move_group_name", std::string("different_group")),
        rclcpp::Parameter("moveit_servo.pose_command_in_topic", std::string("/unused/pose")),
        rclcpp::Parameter("moveit_servo.cartesian_command_in_topic", std::string("/unused/twist")),
        rclcpp::Parameter("moveit_servo.joint_command_in_topic", std::string("/unused/joints")),
        rclcpp::Parameter("moveit_servo.status_topic", std::string("/unused/status")),
        rclcpp::Parameter("moveit_servo.command_in_type", std::string("unitless")),
        rclcpp::Parameter("moveit_servo.incoming_command_timeout", -1.0),
        rclcpp::Parameter("moveit_servo.apply_twist_commands_about_ee_frame", false),
        rclcpp::Parameter("moveit_servo.scale.linear", -1.0),
        rclcpp::Parameter("moveit_servo.scale.rotational", -1.0),
        rclcpp::Parameter("moveit_servo.scale.joint", -1.0),
        rclcpp::Parameter("moveit_servo.pose_tracking.linear_tolerance", -1.0),
        rclcpp::Parameter("moveit_servo.pose_tracking.angular_tolerance", -1.0),
        rclcpp::Parameter(
        "moveit_servo.command_out_type", std::string("invalid/legacy_override")),
        rclcpp::Parameter("moveit_servo.publish_period", 0.5),
        rclcpp::Parameter("moveit_servo.publish_joint_positions", false),
        rclcpp::Parameter("moveit_servo.publish_joint_velocities", true),
        rclcpp::Parameter("moveit_servo.publish_joint_accelerations", true),
        rclcpp::Parameter("moveit_servo.check_octomap_collisions", true),
        rclcpp::Parameter("moveit_servo.check_collisions", true),
        rclcpp::Parameter("moveit_servo.use_smoothing", true),
        rclcpp::Parameter(
        "moveit_servo.smoothing_filter_plugin_name", std::string("invalid/legacy_plugin")),
        rclcpp::Parameter("moveit_servo.collision_check_rate", -1.0),
        rclcpp::Parameter("moveit_servo.self_collision_proximity_threshold", -1.0),
        rclcpp::Parameter("moveit_servo.scene_collision_proximity_threshold", -1.0),
        rclcpp::Parameter("moveit_servo.lower_singularity_threshold", -1.0),
        rclcpp::Parameter("moveit_servo.hard_stop_singularity_threshold", -1.0),
        rclcpp::Parameter("moveit_servo.leaving_singularity_threshold_multiplier", -1.0),
        rclcpp::Parameter("moveit_servo.singularity_step_scale", -1.0),
        rclcpp::Parameter("moveit_servo.halt_all_joints_in_joint_mode", false),
        rclcpp::Parameter("moveit_servo.halt_all_joints_in_cartesian_mode", false),
        rclcpp::Parameter("moveit_servo.joint_limit_margins", std::vector<double>{-1.0}),
        rclcpp::Parameter("moveit_servo.override_velocity_scaling_factor", -1.0)});
  ControllerParameters controller;
  controller.control_period_sec = 0.02;
  controller.planning_group_name = "lite6";

  const auto parameters = declareCollisionAwareServoParameters(*node, controller);
  EXPECT_FALSE(node->has_parameter("moveit_servo.publish_period"));
  EXPECT_FALSE(node->has_parameter("moveit_servo.publish_joint_positions"));
  EXPECT_FALSE(node->has_parameter("moveit_servo.check_collisions"));
  EXPECT_FALSE(node->has_parameter("moveit_servo.move_group_name"));
  EXPECT_EQ(parameters.thread_priority, 17);
  EXPECT_DOUBLE_EQ(parameters.max_expected_latency, 0.06);
  EXPECT_EQ(parameters.command_out_topic, "/custom/joint_trajectory");
  EXPECT_EQ(parameters.joint_topic, "/custom/joint_states");
  EXPECT_EQ(parameters.monitored_planning_scene_topic, "/custom/planning_scene");
  EXPECT_FALSE(parameters.is_primary_planning_scene_monitor);
  EXPECT_DOUBLE_EQ(parameters.publish_period, 0.02);
  EXPECT_EQ(parameters.move_group_name, "lite6");
  EXPECT_EQ(parameters.command_out_type, "trajectory_msgs/JointTrajectory");
  EXPECT_EQ(parameters.command_in_type, "speed_units");
  EXPECT_TRUE(parameters.publish_joint_positions);
  EXPECT_FALSE(parameters.publish_joint_velocities);
  EXPECT_FALSE(parameters.publish_joint_accelerations);
  EXPECT_FALSE(parameters.check_collisions);
  EXPECT_FALSE(parameters.use_smoothing);
}

}  // namespace
}  // namespace face_tracking_arm
