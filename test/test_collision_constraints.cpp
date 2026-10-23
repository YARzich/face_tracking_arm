// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Geometry>

#include <geometric_shapes/shapes.h>
#include <gtest/gtest.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr char kSliderUrdf[] =
  R"(
<robot name="slider">
  <link name="base">
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <link name="moving">
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/>
    <child link="moving"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="100" velocity="1"/>
  </joint>
  <link name="near">
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <joint name="near_joint" type="fixed">
    <parent link="moving"/>
    <child link="near"/>
    <origin xyz="0 10 0"/>
  </joint>
  <link name="mount_neighbor">
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <joint name="mount_neighbor_joint" type="fixed">
    <parent link="near"/>
    <child link="mount_neighbor"/>
    <origin xyz="0 -9.88 0"/>
  </joint>
</robot>)";

constexpr char kSliderSrdf[] =
  R"(
<robot name="slider">
  <group name="arm">
    <chain base_link="base" tip_link="mount_neighbor"/>
  </group>
</robot>)";

struct SliderFixture
{
  SliderFixture(
    const std::string & robot_description = kSliderUrdf,
    const std::string & semantic_description = kSliderSrdf)
  {
    const auto urdf_model = urdf::parseURDF(robot_description);
    if (!urdf_model) {
      return;
    }
    auto mutable_srdf = std::make_shared<srdf::Model>();
    if (!mutable_srdf->initString(*urdf_model, semantic_description)) {
      return;
    }
    robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, mutable_srdf);
    scene = std::make_shared<planning_scene::PlanningScene>(robot_model);
    group = robot_model->getJointModelGroup("arm");
  }

  moveit::core::RobotModelPtr robot_model;
  planning_scene::PlanningScenePtr scene;
  const moveit::core::JointModelGroup * group{nullptr};
};

std::string commonAncestorUrdf()
{
  std::string description = kSliderUrdf;
  description.insert(description.find("<link name=\"base\">"),
        R"(
  <link name="world"/>
  <joint name="common" type="revolute">
    <parent link="world"/><child link="base"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="100" velocity="1"/>
  </joint>
)");
  return description;
}

constexpr char kCommonAncestorSrdf[] =
  R"(
<robot name="slider">
  <group name="arm"><chain base_link="world" tip_link="mount_neighbor"/></group>
</robot>)";

JointMotionState twoDofMotion(const double common, const double relative)
{
  return {Eigen::Vector2d(common, relative), Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()};
}

[[nodiscard]] JointMotionLimits oneDofLimits()
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(1, -1.0);
  limits.upper_position = Eigen::VectorXd::Constant(1, 1.0);
  limits.position_margin = Eigen::VectorXd::Constant(1, 0.01);
  limits.max_velocity = Eigen::VectorXd::Constant(1, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(1, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(1, 20.0);
  limits.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-6}};
  return limits;
}

[[nodiscard]] JointMotionLimits twoDofLimits()
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(2, -10.0);
  limits.upper_position = Eigen::VectorXd::Constant(2, 10.0);
  limits.position_margin = Eigen::VectorXd::Zero(2);
  limits.max_velocity = Eigen::VectorXd::Constant(2, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(2, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(2, 20.0);
  limits.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-6}};
  return limits;
}

[[nodiscard]] JointMotionState motionAt(const double position)
{
  JointMotionState state;
  state.position = Eigen::VectorXd::Constant(1, position);
  state.velocity = Eigen::VectorXd::Zero(1);
  state.acceleration = Eigen::VectorXd::Zero(1);
  return state;
}

[[nodiscard]] CollisionConstraintConfig sliderConfig()
{
  CollisionConstraintConfig config;
  config.hard_clearance_m = 0.015;
  config.query_distance_m = 0.15;
  config.maximum_constraint_rows = 8;
  config.protected_joint_name = "slide";
  config.protected_joint_lower_rad = -0.8;
  config.protected_joint_upper_rad = 0.8;
  config.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-6}};
  config.monitor_link_name = "moving";
  config.monitor_near_link_name = "near";
  config.invariant_mount_neighbor_name = "mount_neighbor";
  return config;
}

[[nodiscard]] const CollisionPairDiagnostic * findWorldPair(
  const CollisionConstraintResult & result, const std::string & object_name)
{
  const auto diagnostic = std::find_if(
    result.diagnostics.pairs.begin(), result.diagnostics.pairs.end(),
    [&object_name](const CollisionPairDiagnostic & pair) {
      return pair.kind == CollisionPairKind::kWorld &&
             (pair.first_body == object_name || pair.second_body == object_name);
    });
  return diagnostic == result.diagnostics.pairs.end() ? nullptr : &*diagnostic;
}

[[nodiscard]] const CollisionSegmentSampleDiagnostic * findDensePathSample(
  const CollisionSegmentValidationResult & result, const std::size_t sample_index)
{
  const auto sample = std::find_if(
    result.samples.begin(), result.samples.end(),
    [sample_index](const CollisionSegmentSampleDiagnostic & candidate) {
      return candidate.sample_index == sample_index &&
             !candidate.path_block_certificate_attempt;
    });
  return sample == result.samples.end() ? nullptr : &*sample;
}

[[nodiscard]] std::pair<double, double> oneDofReachableVelocityInterval(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const double period_sec)
{
  const double minimum_acceleration = std::max(
    -limits.max_acceleration[0],
    state.acceleration[0] - limits.max_jerk[0] * period_sec);
  const double maximum_acceleration = std::min(
    limits.max_acceleration[0],
    state.acceleration[0] + limits.max_jerk[0] * period_sec);
  const double lower_position = limits.lower_position[0] + limits.position_margin[0];
  const double upper_position = limits.upper_position[0] - limits.position_margin[0];
  const double lower_velocity = std::max({
        -limits.max_velocity[0],
        state.velocity[0] + minimum_acceleration * period_sec,
        (lower_position - state.position[0]) / period_sec});
  const double upper_velocity = std::min({
        limits.max_velocity[0],
        state.velocity[0] + maximum_acceleration * period_sec,
        (upper_position - state.position[0]) / period_sec});
  return {lower_velocity, upper_velocity};
}

[[nodiscard]] double simulateDiscreteStoppingDistance(
  const double toward_velocity,
  const double toward_acceleration,
  const double max_acceleration,
  const double max_jerk,
  const double period_sec,
  const double residual_latency_sec)
{
  double position = 0.0;
  double velocity = toward_velocity;
  double acceleration = toward_acceleration;
  double maximum_position = 0.0;
  const double acceleration_step = max_jerk * period_sec;
  const auto integrate_tick = [&]() {
      velocity += acceleration * period_sec;
      position += velocity * period_sec;
      maximum_position = std::max(maximum_position, position);
    };

  const auto residual_ticks = static_cast<std::size_t>(std::ceil(
      residual_latency_sec / period_sec));
  for (std::size_t tick = 0; tick < residual_ticks; ++tick) {
    acceleration = std::min(max_acceleration, acceleration + acceleration_step);
    integrate_tick();
  }
  for (std::size_t tick = 0;
    tick < 1000U && (velocity > 0.0 || acceleration > 0.0); ++tick)
  {
    acceleration = std::max(-max_acceleration, acceleration - acceleration_step);
    integrate_tick();
  }
  return maximum_position;
}

TEST(CollisionConstraints, StoppingDistanceGrowsWithVelocityAndLatency)
{
  const double slow = jerkLimitedStoppingDistance(0.2, 0.0, 2.0, 20.0, 0.01, 0.0);
  const double fast = jerkLimitedStoppingDistance(0.8, 0.0, 2.0, 20.0, 0.01, 0.0);
  const double delayed = jerkLimitedStoppingDistance(
    0.8, 0.0, 2.0, 20.0, 0.01, 0.02);

  EXPECT_GT(slow, 0.0);
  EXPECT_GT(fast, slow);
  EXPECT_GT(delayed, fast);
}

TEST(CollisionConstraints, BoundaryVelocityRespectsTheStoppingEnvelope)
{
  constexpr double remaining = 0.20;
  constexpr double current_velocity = 0.35;
  constexpr double current_acceleration = 0.4;
  constexpr double max_acceleration = 2.0;
  constexpr double max_jerk = 20.0;
  constexpr double period = 0.01;
  constexpr double latency = 0.02;
  const double minimum_reachable_acceleration = std::max(
    -max_acceleration, current_acceleration - max_jerk * period);
  const double maximum_reachable_acceleration = std::min(
    max_acceleration, current_acceleration + max_jerk * period);
  const double minimum_reachable_velocity =
    current_velocity + minimum_reachable_acceleration * period;
  const double maximum_reachable_velocity =
    current_velocity + maximum_reachable_acceleration * period;

  const double safe = maximumSafeVelocityTowardBoundary(
    remaining, current_velocity,
    minimum_reachable_velocity, maximum_reachable_velocity,
    max_acceleration, max_jerk, period, latency);
  const double next_acceleration = (safe - current_velocity) / period;
  const double required = safe * period + jerkLimitedStoppingDistance(
    safe, next_acceleration, max_acceleration, max_jerk, period, latency);
  const double independently_simulated = safe * period + simulateDiscreteStoppingDistance(
    safe, next_acceleration, max_acceleration, max_jerk, period, latency);

  EXPECT_GE(safe, minimum_reachable_velocity);
  EXPECT_LE(safe, maximum_reachable_velocity);
  EXPECT_LE(std::abs(next_acceleration), max_acceleration + 1.0e-12);
  EXPECT_LE(required, remaining + 1.0e-10);
  EXPECT_NEAR(required, independently_simulated, 1.0e-12);
}

TEST(CollisionConstraints, StoppingDistanceRejectsAccelerationOutsidePhysicalBounds)
{
  EXPECT_THROW(
    static_cast<void>(jerkLimitedStoppingDistance(
      0.2, 2.01, 2.0, 20.0, 0.01, 0.02)),
    std::invalid_argument);
  EXPECT_THROW(
    static_cast<void>(jerkLimitedStoppingDistance(
      0.2, -2.01, 2.0, 20.0, 0.01, 0.02)),
    std::invalid_argument);
}

TEST(CollisionConstraints, DiscreteResidualCatchesContinuousCounterexampleAndMirror)
{
  constexpr double period = 0.01;
  constexpr double latency = 0.02;
  constexpr double max_acceleration = 2.0;
  constexpr double max_jerk = 20.0;
  constexpr double remaining_distance = 0.000020;
  constexpr double current_approach_speed = 0.002;
  constexpr double minimum_reachable_approach_speed = 0.0;
  constexpr double maximum_reachable_approach_speed = 0.004;
  constexpr double minimum_reachable_acceleration = -0.2;

  const double stopping_distance = jerkLimitedStoppingDistance(
    minimum_reachable_approach_speed, minimum_reachable_acceleration,
    max_acceleration, max_jerk, period, latency);
  EXPECT_NEAR(stopping_distance, 0.000040, 1.0e-15);
  EXPECT_NEAR(
    stopping_distance,
    simulateDiscreteStoppingDistance(
      minimum_reachable_approach_speed, minimum_reachable_acceleration,
      max_acceleration, max_jerk, period, latency),
    1.0e-15);
  EXPECT_GT(stopping_distance, remaining_distance);

  const double safe_speed = maximumSafeVelocityTowardBoundary(
    remaining_distance, current_approach_speed,
    minimum_reachable_approach_speed, maximum_reachable_approach_speed,
    max_acceleration, max_jerk, period, latency);
  EXPECT_DOUBLE_EQ(safe_speed, minimum_reachable_approach_speed);

  for (const double direction : {1.0, -1.0}) {
    const Eigen::VectorXd gradient = Eigen::VectorXd::Constant(1, direction);
    JointMotionState state = motionAt(0.0);
    state.velocity[0] = -direction * current_approach_speed;
    EXPECT_DOUBLE_EQ(
      maximumSafeCollisionApproachSpeed(
        remaining_distance, gradient, state, oneDofLimits(), period, latency),
      minimum_reachable_approach_speed);
  }
}

TEST(CollisionConstraints, VectorStoppingPreservesIndividualAccelerationSaturationAndMirror)
{
  constexpr double period = 0.01;
  constexpr double latency = 0.02;
  Eigen::VectorXd gradient(2);
  gradient << 0.5, 0.5;
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(2);
  state.velocity.resize(2);
  state.velocity << -0.5, -0.5;
  state.acceleration.resize(2);
  state.acceleration << 2.0, -2.0;
  Eigen::VectorXd candidate_velocity(2);
  candidate_velocity << -0.48, -0.52;

  const double stopping_distance = jerkLimitedCollisionStoppingDistance(
    gradient, state, candidate_velocity, twoDofLimits(), period, latency);
  EXPECT_NEAR(stopping_distance, 0.1156, 1.0e-12);

  // Mirroring the gradient and every joint derivative must produce the same
  // approach-coordinate trajectory and stopping excursion.
  JointMotionState mirrored_state = state;
  mirrored_state.velocity = -state.velocity;
  mirrored_state.acceleration = -state.acceleration;
  EXPECT_NEAR(
    jerkLimitedCollisionStoppingDistance(
      -gradient, mirrored_state, -candidate_velocity,
      twoDofLimits(), period, latency),
    stopping_distance, 1.0e-12);
}

TEST(CollisionConstraints, VectorApproachEnvelopeRejectsScalarSaturationCounterexample)
{
  constexpr double period = 0.01;
  constexpr double latency = 0.02;
  constexpr double remaining_distance = 0.1204;
  Eigen::VectorXd gradient(2);
  gradient << 0.5, 0.5;
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(2);
  state.velocity.resize(2);
  state.velocity << -0.5, -0.5;
  state.acceleration.resize(2);
  state.acceleration << 2.0, -2.0;
  Eigen::VectorXd unsafe_candidate(2);
  unsafe_candidate << -0.48, -0.52;

  const double unsafe_approach_speed = -gradient.dot(unsafe_candidate);
  const double unsafe_required_distance = unsafe_approach_speed * period +
    jerkLimitedCollisionStoppingDistance(
    gradient, state, unsafe_candidate, twoDofLimits(), period, latency);
  ASSERT_NEAR(unsafe_approach_speed, 0.5, 1.0e-15);
  ASSERT_NEAR(unsafe_required_distance, 0.1206, 1.0e-12);
  ASSERT_GT(unsafe_required_distance, remaining_distance);

  const double safe_cap = maximumSafeCollisionApproachSpeed(
    remaining_distance, gradient, state, twoDofLimits(), period, latency);
  EXPECT_GE(safe_cap, 0.499);
  EXPECT_LT(safe_cap, unsafe_approach_speed);

  JointMotionState mirrored_state = state;
  mirrored_state.velocity = -state.velocity;
  mirrored_state.acceleration = -state.acceleration;
  EXPECT_NEAR(
    maximumSafeCollisionApproachSpeed(
      remaining_distance, -gradient, mirrored_state,
      twoDofLimits(), period, latency),
    safe_cap, 1.0e-12);
}

TEST(CollisionConstraints, ResidualLatencyRoundsUpToCompleteDiscreteTicks)
{
  constexpr double period = 0.01;
  constexpr double max_acceleration = 2.0;
  constexpr double max_jerk = 20.0;
  constexpr double velocity = 0.0;
  constexpr double acceleration = -0.2;

  EXPECT_DOUBLE_EQ(
    jerkLimitedStoppingDistance(
      velocity, acceleration, max_acceleration, max_jerk, period, period),
    0.0);
  EXPECT_NEAR(
    jerkLimitedStoppingDistance(
      velocity, acceleration, max_acceleration, max_jerk,
      period, period + 1.0e-6),
    0.000040, 1.0e-15);
}

TEST(CollisionConstraints, RequiredStoppingDistanceIsMonotonicOnReachableIntervals)
{
  constexpr double max_acceleration = 2.0;
  constexpr double max_jerk = 20.0;
  constexpr double period = 0.01;
  constexpr double latency = 0.02;
  const std::vector<std::pair<double, double>> states{
    {0.18, 0.0},
    {0.40, 1.0},
    {-0.05, -1.2},
  };

  for (const auto & [current_velocity, current_acceleration] : states) {
    const double minimum_acceleration = std::max(
      -max_acceleration, current_acceleration - max_jerk * period);
    const double maximum_acceleration = std::min(
      max_acceleration, current_acceleration + max_jerk * period);
    const double minimum_velocity = current_velocity + minimum_acceleration * period;
    const double maximum_velocity = current_velocity + maximum_acceleration * period;
    double previous_required_distance = -std::numeric_limits<double>::infinity();
    for (int sample = 0; sample <= 100; ++sample) {
      const double fraction = static_cast<double>(sample) / 100.0;
      const double candidate_velocity =
        (1.0 - fraction) * minimum_velocity + fraction * maximum_velocity;
      const double candidate_acceleration =
        (candidate_velocity - current_velocity) / period;
      const double required_distance = candidate_velocity * period +
        jerkLimitedStoppingDistance(
        candidate_velocity, candidate_acceleration,
        max_acceleration, max_jerk, period, latency);
      EXPECT_GE(required_distance + 1.0e-12, previous_required_distance);
      previous_required_distance = required_distance;
    }
  }
}

TEST(CollisionConstraints, CollisionApproachEnvelopeBrakesBeforeTheHardBoundary)
{
  const Eigen::VectorXd gradient = Eigen::VectorXd::Constant(1, 1.0);
  JointMotionState state = motionAt(0.0);
  state.velocity[0] = -0.020;
  state.acceleration[0] = -0.10;
  const JointMotionLimits limits = oneDofLimits();

  const double far_safe_speed = maximumSafeCollisionApproachSpeed(
    0.060, gradient, state, limits, 0.01, 0.02);
  const double near_safe_speed = maximumSafeCollisionApproachSpeed(
    0.0001, gradient, state, limits, 0.01, 0.02);

  EXPECT_GT(far_safe_speed, near_safe_speed);
  EXPECT_GT(near_safe_speed, 0.0);

  // Once outside viability, the envelope returns the least approach that is physically
  // reachable on the next tick instead of inventing an infeasible instantaneous slowdown.
  const double next_acceleration = std::min(
    limits.max_acceleration[0],
    state.acceleration[0] + limits.max_jerk[0] * 0.01);
  const double most_separating_next_velocity =
    state.velocity[0] + next_acceleration * 0.01;
  const double minimum_reachable_approach_speed =
    -gradient[0] * most_separating_next_velocity;
  EXPECT_NEAR(near_safe_speed, minimum_reachable_approach_speed, 1.0e-12);
}

TEST(CollisionConstraints, RejectsContradictorySafetyProfile)
{
  EXPECT_DOUBLE_EQ(CollisionConstraintConfig{}.query_distance_m, 0.20);
  EXPECT_DOUBLE_EQ(CollisionConstraintConfig{}.tracking_position_error_bound_rad, 0.006);
  EXPECT_DOUBLE_EQ(CollisionConstraintConfig{}.numerical_distance_reserve_m, 0.0001);
  EXPECT_DOUBLE_EQ(CollisionConstraintConfig{}.default_distance_lipschitz_m_per_rad, 2.6193);
  EXPECT_DOUBLE_EQ(
    CollisionConstraintConfig{}.monitor_near_distance_lipschitz_m_per_rad, 0.4543);

  CollisionConstraintConfig config;
  config.query_distance_m = config.hard_clearance_m;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.segment_validation_substeps = 1U;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.protected_joint_lower_rad = config.protected_joint_upper_rad;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.tracking_position_error_bound_rad = -0.001;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.numerical_distance_reserve_m = -0.001;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.default_distance_lipschitz_m_per_rad =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.query_distance_m = 0.030;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.separation_buffer_m = -0.01;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.separation_buffer_m = config.query_distance_m;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.separation_max_velocity_mps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);

  config = CollisionConstraintConfig{};
  config.separation_lookahead_sec = -0.1;
  EXPECT_THROW(static_cast<void>(CollisionConstraintBuilder{config}), std::invalid_argument);
}

TEST(CollisionConstraints, SeparationGuidanceActsBeforeHardBrakingAndKeepsHardRows)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.50;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.01, 0.1), obstacle_pose);
  auto config = sliderConfig();
  CollisionConstraintBuilder original(config);
  config.separation_buffer_m = 0.06;
  CollisionConstraintBuilder guided(config);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  auto motion = motionAt(0.35);
  state.setJointGroupPositions(fixture.group, motion.position);
  state.update();
  const auto result = guided.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  const auto baseline = original.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid);
  // Both moving bodies receive an outward preference from the world obstacle;
  // the fixed-distance mount-to-monitor pair must not add a third row.
  ASSERT_EQ(result.separation_guidance.size(), 2U);
  for (const auto & guidance : result.separation_guidance) {
    EXPECT_LT(guidance.distance_gradient[0], 0.0);
    EXPECT_GT(guidance.reference_velocity_mps, 0.0);
    EXPECT_GT(guidance.activation, 0.0);
    EXPECT_LT(guidance.activation, 1.0);
  }
  ASSERT_EQ(result.constraints.size(), baseline.constraints.size());
  for (std::size_t index = 0; index < baseline.constraints.size(); ++index) {
    EXPECT_TRUE(result.constraints[index].coefficients.isApprox(
        baseline.constraints[index].coefficients));
    EXPECT_DOUBLE_EQ(
      result.constraints[index].lower_bound, baseline.constraints[index].lower_bound);
    EXPECT_DOUBLE_EQ(
      result.constraints[index].upper_bound, baseline.constraints[index].upper_bound);
  }
  motion = motionAt(0.20);
  state.setJointGroupPositions(fixture.group, motion.position);
  state.update();
  const auto far = guided.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(far.diagnostics.valid);
  EXPECT_TRUE(far.separation_guidance.empty());

  config.separation_lookahead_sec = 0.5;
  CollisionConstraintBuilder predictive(config);
  motion = motionAt(0.30);
  state.setJointGroupPositions(fixture.group, motion.position);
  state.update();
  const auto still = predictive.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  EXPECT_TRUE(still.separation_guidance.empty());
  // The same distance needs earlier guidance when the bodies are closing fast.
  motion.velocity[0] = 0.20;
  const auto approaching = predictive.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(approaching.diagnostics.valid);
  EXPECT_FALSE(approaching.separation_guidance.empty());
  motion.velocity[0] = -0.20;
  const auto separating = predictive.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(separating.diagnostics.valid);
  EXPECT_TRUE(separating.separation_guidance.empty());
}

TEST(CollisionConstraints, QpEnvelopeUsesPairSpecificRobustRemainingDistance)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.184;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);
  const JointMotionState stationary = motionAt(-0.30);
  const CollisionConstraintConfig config = sliderConfig();
  const auto result = CollisionConstraintBuilder(config).build(
    *fixture.scene, robot_state, *fixture.group, stationary, oneDofLimits(), 0.01);

  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  const auto * wall_pair = findWorldPair(result, "wall");
  ASSERT_NE(wall_pair, nullptr);
  constexpr double expected_required_clearance =
    0.015 + 0.0001 + 2.6193 * (0.006 + 0.002 * 0.01 / 6.0);
  EXPECT_NEAR(wall_pair->distance_m, 0.016, 1.0e-9);
  EXPECT_NEAR(wall_pair->distance_lipschitz_m_per_rad, 2.6193, 1.0e-12);
  EXPECT_NEAR(wall_pair->required_clearance_m, expected_required_clearance, 1.0e-12);
  EXPECT_TRUE(wall_pair->robust_clearance_violated);
  EXPECT_TRUE(result.diagnostics.robust_clearance_violated);
  EXPECT_FALSE(result.diagnostics.hard_clearance_violated);
  EXPECT_LT(wall_pair->applied_approach_speed_cap_mps, 0.0);
}

TEST(CollisionConstraints, QpNextStepReserveMatchesSegmentValidator)
{
  constexpr double period = 0.01;
  constexpr double legacy_candidate_velocity = 0.002;
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);

  CollisionConstraintConfig config = sliderConfig();
  config.residual_latency_sec = 0.0;
  config.segment_validation_substeps = 3U;
  const JointMotionLimits limits = oneDofLimits();
  JointMotionState motion = motionAt(-0.30);
  motion.velocity[0] = legacy_candidate_velocity;

  Eigen::VectorXd legacy_velocity(1);
  legacy_velocity[0] = legacy_candidate_velocity;
  const Eigen::VectorXd distance_gradient = Eigen::VectorXd::Constant(1, -1.0);
  const double pair_lipschitz = config.default_distance_lipschitz_m_per_rad;
  const double base_required_clearance = config.hard_clearance_m +
    config.numerical_distance_reserve_m +
    pair_lipschitz * config.tracking_position_error_bound_rad;
  const double candidate_step_toward_contact =
    -distance_gradient.dot(legacy_velocity) * period;
  const double candidate_stop_tail = jerkLimitedCollisionStoppingDistance(
    distance_gradient, motion, legacy_velocity, limits, period, config.residual_latency_sec);
  const double validator_inter_sample_reserve = pair_lipschitz * legacy_candidate_velocity *
    period / (2.0 * static_cast<double>(config.segment_validation_substeps));
  ASSERT_GT(validator_inter_sample_reserve, 0.0);

  // The old QP envelope leaves only a fraction of the segment check's extra reserve after
  // this reachable command's first integration step.
  const double remaining_margin = 0.25 * validator_inter_sample_reserve;
  const double initial_distance = base_required_clearance + candidate_step_toward_contact +
    candidate_stop_tail + remaining_margin;
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = motion.position[0] + 0.1 + initial_distance;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(fixture.group, motion.position);
  robot_state.update(true);
  const CollisionConstraintResult collision = CollisionConstraintBuilder(config).build(
    *fixture.scene, robot_state, *fixture.group, motion, limits, period);
  ASSERT_TRUE(collision.diagnostics.valid) << collision.diagnostics.failure_reason;
  const CollisionPairDiagnostic * wall_pair = findWorldPair(collision, "wall");
  ASSERT_NE(wall_pair, nullptr);
  ASSERT_EQ(collision.diagnostics.active_distance_constraints, 1U);
  ASSERT_EQ(collision.constraints.size(), 1U);
  const Eigen::VectorXd actual_distance_gradient = collision.constraints.back().coefficients;
  ASSERT_EQ(actual_distance_gradient.size(), 1);
  EXPECT_NEAR(actual_distance_gradient[0], distance_gradient[0], 1.0e-12);
  const auto [reachable_minimum, reachable_maximum] =
    oneDofReachableVelocityInterval(motion, limits, period);
  const double qp_inter_sample_reserve = pair_lipschitz *
    std::max(std::abs(reachable_minimum), std::abs(reachable_maximum)) * period /
    (2.0 * static_cast<double>(config.segment_validation_substeps));
  EXPECT_NEAR(
    wall_pair->required_clearance_m,
    base_required_clearance + qp_inter_sample_reserve, 1.0e-12);

  const double old_qp_safe_approach_speed = maximumSafeCollisionApproachSpeed(
    wall_pair->distance_m - base_required_clearance,
    actual_distance_gradient, motion, limits, period, config.residual_latency_sec);
  ASSERT_GE(old_qp_safe_approach_speed, legacy_candidate_velocity);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = period;
  qp_config.residual_command_latency_sec = config.residual_latency_sec;
  qp_config.solver_time_limit_sec = 0.1;
  qp_config.solver_max_iterations = 2000;
  const HierarchicalVelocityQp solver(qp_config);
  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(1, 1);
  task.primary_reference = Eigen::VectorXd::Constant(1, legacy_candidate_velocity);
  task.primary_weights = Eigen::VectorXd::Ones(1);
  task.secondary_matrix = Eigen::MatrixXd(0, 1);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);

  std::vector<LinearVelocityConstraint> legacy_constraints = collision.constraints;
  legacy_constraints.back().lower_bound = -old_qp_safe_approach_speed;
  const QpResult legacy_qp = solver.solve(task, motion, limits, legacy_constraints);
  ASSERT_TRUE(legacy_qp.command_available());
  EXPECT_NEAR(legacy_qp.joint_velocity[0], legacy_candidate_velocity, 1.0e-6);

  const auto next_motion = [&motion, period](const double velocity) {
      JointMotionState next;
      next.position = motion.position + Eigen::VectorXd::Constant(1, velocity * period);
      next.velocity = Eigen::VectorXd::Constant(1, velocity);
      next.acceleration = Eigen::VectorXd::Constant(
        1, (velocity - motion.velocity[0]) / period);
      return next;
    };
  const CollisionSegmentValidationResult legacy_segment = CollisionConstraintBuilder(
    config).validateSegment(
    *fixture.scene, robot_state, *fixture.group, motion,
    next_motion(legacy_qp.joint_velocity[0]), config.segment_validation_substeps);
  ASSERT_TRUE(legacy_segment.input_valid) << legacy_segment.failure_reason;
  EXPECT_TRUE(legacy_segment.unsafe);
  EXPECT_TRUE(legacy_segment.robust_clearance_violated);

  const QpResult corrected_qp = solver.solve(task, motion, limits, collision.constraints);
  ASSERT_TRUE(corrected_qp.command_available());
  EXPECT_LE(corrected_qp.joint_velocity[0], legacy_qp.joint_velocity[0] + 1.0e-8);
  const CollisionSegmentValidationResult corrected_segment = CollisionConstraintBuilder(
    config).validateSegment(
    *fixture.scene, robot_state, *fixture.group, motion,
    next_motion(corrected_qp.joint_velocity[0]), config.segment_validation_substeps);
  ASSERT_TRUE(corrected_segment.input_valid) << corrected_segment.failure_reason;
  EXPECT_FALSE(corrected_segment.unsafe) << corrected_segment.failure_reason;
}

TEST(CollisionConstraints, BuildSingleDoesNotLetInvariantHideControllablePair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);

  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.125));
  robot_state.update(true);

  const CollisionConstraintResult result = CollisionConstraintBuilder(sliderConfig()).build(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.125), oneDofLimits(), 0.01);

  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  const auto find_self_pair = [&result](const std::string & first, const std::string & second) {
      const auto pair = std::find_if(
        result.diagnostics.pairs.begin(), result.diagnostics.pairs.end(),
        [&first, &second](const CollisionPairDiagnostic & diagnostic) {
          return diagnostic.kind == CollisionPairKind::kSelf &&
                 diagnostic.first_body == first && diagnostic.second_body == second;
        });
      return pair == result.diagnostics.pairs.end() ? nullptr : &*pair;
    };
  const CollisionPairDiagnostic * invariant = find_self_pair("mount_neighbor", "moving");
  const CollisionPairDiagnostic * controllable = find_self_pair("base", "moving");

  ASSERT_NE(invariant, nullptr);
  ASSERT_NE(controllable, nullptr);
  EXPECT_TRUE(invariant->invariant_pair);
  EXPECT_FALSE(invariant->constraint_added);
  EXPECT_NEAR(invariant->distance_m, 0.020, 1.0e-9);
  EXPECT_FALSE(controllable->invariant_pair);
  EXPECT_TRUE(controllable->constraint_added);
  EXPECT_TRUE(controllable->robust_clearance_violated);
  EXPECT_NEAR(controllable->distance_m, 0.025, 1.0e-9);
  EXPECT_EQ(result.diagnostics.closest_pair, "mount_neighbor <-> moving");
  EXPECT_EQ(result.diagnostics.closest_controllable_pair, "base <-> moving");
  EXPECT_EQ(result.diagnostics.skipped_invariant_contacts, 1U);
  EXPECT_GE(result.diagnostics.active_distance_constraints, 1U);

  std::set<std::pair<std::string, std::string>> unique_body_pairs;
  for (const CollisionPairDiagnostic & pair : result.diagnostics.pairs) {
    EXPECT_TRUE(unique_body_pairs.emplace(pair.first_body, pair.second_body).second);
  }

  const CollisionConstraintResult repeated = CollisionConstraintBuilder(sliderConfig()).build(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.125), oneDofLimits(), 0.01);
  ASSERT_TRUE(repeated.diagnostics.valid) << repeated.diagnostics.failure_reason;
  ASSERT_EQ(repeated.diagnostics.pairs.size(), result.diagnostics.pairs.size());
  for (std::size_t index = 0; index < result.diagnostics.pairs.size(); ++index) {
    EXPECT_EQ(
      repeated.diagnostics.pairs[index].first_body,
      result.diagnostics.pairs[index].first_body);
    EXPECT_EQ(
      repeated.diagnostics.pairs[index].second_body,
      result.diagnostics.pairs[index].second_body);
    EXPECT_DOUBLE_EQ(
      repeated.diagnostics.pairs[index].distance_m,
      result.diagnostics.pairs[index].distance_m);
  }
}

TEST(CollisionConstraints, SegmentBatchKeepsNonNearestRobustlyUnsafePair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const JointMotionState pose = motionAt(0.125);
  state.setJointGroupPositions(fixture.group, pose.position);
  state.update(true);

  const auto result = builder.validateSegment(
    *fixture.scene, state, *fixture.group, pose, pose, 2U);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  ASSERT_TRUE(result.unsafe);
  ASSERT_TRUE(result.robust_clearance_violated);
  ASSERT_EQ(result.first_unsafe_sample, 0U);
  ASSERT_EQ(result.samples.size(), 3U);
  // The nearer mount-neighbor pair is safe under its tighter profile. The farther
  // base-to-moving pair violates the default reserve and must not be hidden by the
  // global nearest result returned by MoveIt.
  const auto * sample = findDensePathSample(result, 0U);
  ASSERT_NE(sample, nullptr);
  EXPECT_EQ(sample->closest_pair, "mount_neighbor <-> moving");
  EXPECT_TRUE(sample->robust_clearance_violated);
  EXPECT_NE(result.failure_reason.find("base <-> moving"), std::string::npos);
  // One batch for self distances and one for world distances per sampled pose.
  EXPECT_EQ(result.distance_query_count, 6U);
}

TEST(CollisionConstraints, WorldDistanceGradientMatchesFiniteDifference)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.robot_model, nullptr);
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);

  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.184;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);

  JointMotionState nominal_motion = motionAt(-0.30);
  nominal_motion.velocity[0] = 0.18;
  const auto nominal = builder.build(
    *fixture.scene, robot_state, *fixture.group,
    nominal_motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(nominal.diagnostics.valid) << nominal.diagnostics.failure_reason;
  ASSERT_GE(nominal.constraints.size(), 1U);
  ASSERT_TRUE(std::isfinite(nominal.diagnostics.minimum_world_distance_m));
  const auto * wall_pair = findWorldPair(nominal, "wall");
  ASSERT_NE(wall_pair, nullptr);
  EXPECT_TRUE(wall_pair->outside_viability);
  EXPECT_GE(nominal.diagnostics.outside_viability_contacts, 1U);
  EXPECT_NEAR(
    wall_pair->applied_approach_speed_cap_mps,
    wall_pair->minimum_reachable_approach_speed_mps, 1.0e-12);
  const auto [lower_velocity, upper_velocity] =
    oneDofReachableVelocityInterval(nominal_motion, oneDofLimits(), 0.01);
  EXPECT_LE(lower_velocity, upper_velocity);
  EXPECT_GE(
    nominal.constraints[0].coefficients[0] * lower_velocity + 1.0e-12,
    nominal.constraints[0].lower_bound);

  constexpr double epsilon = 1.0e-5;
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30 + epsilon));
  robot_state.update(true);
  JointMotionState perturbed_motion = motionAt(-0.30 + epsilon);
  perturbed_motion.velocity[0] = 0.18;
  const auto perturbed = builder.build(
    *fixture.scene, robot_state, *fixture.group,
    perturbed_motion, oneDofLimits(), 0.01);
  ASSERT_TRUE(perturbed.diagnostics.valid) << perturbed.diagnostics.failure_reason;

  const double finite_difference =
    (perturbed.diagnostics.minimum_world_distance_m -
    nominal.diagnostics.minimum_world_distance_m) / epsilon;
  EXPECT_NEAR(nominal.constraints[0].coefficients[0], finite_difference, 2.0e-3);
  EXPECT_NEAR(finite_difference, -1.0, 2.0e-3);
}

TEST(CollisionConstraints, CollisionRowHasExactSignMirrorAtTheLowerSide)
{
  const auto build_case = [](
    const double position, const double velocity, const double wall_position)
    {
      SliderFixture fixture;
      if (fixture.scene == nullptr || fixture.group == nullptr) {
        return CollisionConstraintResult{};
      }
      Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
      obstacle_pose.translation().x() = wall_position;
      fixture.scene->getWorldNonConst()->addToObject(
        "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

      moveit::core::RobotState robot_state(fixture.robot_model);
      robot_state.setToDefaultValues();
      robot_state.setJointGroupPositions(
        fixture.group, Eigen::VectorXd::Constant(1, position));
      robot_state.update(true);
      JointMotionState motion = motionAt(position);
      motion.velocity[0] = velocity;
      return CollisionConstraintBuilder(sliderConfig()).build(
        *fixture.scene, robot_state, *fixture.group, motion, oneDofLimits(), 0.01);
    };

  const CollisionConstraintResult upper_side = build_case(-0.30, 0.18, -0.184);
  const CollisionConstraintResult lower_side = build_case(0.30, -0.18, 0.184);
  ASSERT_TRUE(upper_side.diagnostics.valid) << upper_side.diagnostics.failure_reason;
  ASSERT_TRUE(lower_side.diagnostics.valid) << lower_side.diagnostics.failure_reason;
  ASSERT_GE(upper_side.constraints.size(), 1U);
  ASSERT_GE(lower_side.constraints.size(), 1U);
  const auto * upper_pair = findWorldPair(upper_side, "wall");
  const auto * lower_pair = findWorldPair(lower_side, "wall");
  ASSERT_NE(upper_pair, nullptr);
  ASSERT_NE(lower_pair, nullptr);

  EXPECT_NEAR(upper_pair->distance_m, lower_pair->distance_m, 1.0e-12);
  EXPECT_NEAR(
    upper_pair->minimum_reachable_approach_speed_mps,
    lower_pair->minimum_reachable_approach_speed_mps, 1.0e-12);
  EXPECT_NEAR(
    upper_pair->maximum_reachable_approach_speed_mps,
    lower_pair->maximum_reachable_approach_speed_mps, 1.0e-12);
  EXPECT_NEAR(
    upper_pair->applied_approach_speed_cap_mps,
    lower_pair->applied_approach_speed_cap_mps, 1.0e-12);
  EXPECT_EQ(upper_pair->outside_viability, lower_pair->outside_viability);
  EXPECT_NEAR(
    upper_side.constraints[0].coefficients[0],
    -lower_side.constraints[0].coefficients[0], 1.0e-12);
  EXPECT_NEAR(
    upper_side.constraints[0].lower_bound,
    lower_side.constraints[0].lower_bound, 1.0e-12);
}

TEST(CollisionConstraints, QueryBoundaryApproachBrakesWithoutCrossingHardClearance)
{
  constexpr double period = 0.01;
  constexpr double desired_approach_speed = 0.18;
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.12;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  const CollisionConstraintConfig config = sliderConfig();
  const JointMotionLimits limits = oneDofLimits();
  CollisionConstraintBuilder builder(config);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  JointMotionState motion = motionAt(-0.30);
  motion.velocity[0] = desired_approach_speed;
  bool braking_started = false;
  bool first_stop_reached = false;
  bool settled = false;
  double minimum_clearance = std::numeric_limits<double>::infinity();

  for (int tick = 0; tick < 500; ++tick) {
    robot_state.setJointGroupPositions(fixture.group, motion.position);
    robot_state.update(true);
    const CollisionConstraintResult result = builder.build(
      *fixture.scene, robot_state, *fixture.group, motion, limits, period);
    ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
    const auto * wall_pair = findWorldPair(result, "wall");
    ASSERT_NE(wall_pair, nullptr);
    minimum_clearance = std::min(minimum_clearance, wall_pair->distance_m);
    EXPECT_GE(wall_pair->distance_m + 1.0e-10, config.hard_clearance_m);

    const auto [lower_velocity, upper_velocity] =
      oneDofReachableVelocityInterval(motion, limits, period);
    EXPECT_NEAR(
      wall_pair->minimum_reachable_approach_speed_mps, lower_velocity, 1.0e-12);
    EXPECT_NEAR(
      wall_pair->maximum_reachable_approach_speed_mps, upper_velocity, 1.0e-12);
    if (tick == 0) {
      EXPECT_NEAR(wall_pair->distance_m, 0.080, 1.0e-12);
      EXPECT_FALSE(wall_pair->constraint_added);
      EXPECT_EQ(result.diagnostics.active_distance_constraints, 0U);
      EXPECT_TRUE(result.constraints.empty());
      EXPECT_GE(
        wall_pair->jerk_aware_safe_approach_speed_mps + 1.0e-12,
        upper_velocity);
    }
    braking_started = braking_started || wall_pair->constraint_added;
    const double desired_velocity = braking_started ? 0.0 : desired_approach_speed;
    const double capped_velocity = wall_pair->constraint_added ?
      std::min(desired_velocity, wall_pair->applied_approach_speed_cap_mps) :
      desired_velocity;
    const double command_velocity = std::clamp(
      capped_velocity, lower_velocity, upper_velocity);

    for (const auto & constraint : result.constraints) {
      const double value = constraint.coefficients[0] * command_velocity;
      EXPECT_GE(value + 1.0e-10, constraint.lower_bound);
      EXPECT_LE(value - 1.0e-10, constraint.upper_bound);
    }
    if (braking_started && !first_stop_reached) {
      EXPECT_LE(
        std::max(0.0, command_velocity),
        std::max(0.0, motion.velocity[0]) + 1.0e-12);
      first_stop_reached = command_velocity <= 0.0;
    }

    const double next_acceleration =
      (command_velocity - motion.velocity[0]) / period;
    EXPECT_LE(std::abs(next_acceleration), limits.max_acceleration[0] + 1.0e-10);
    EXPECT_LE(
      std::abs(next_acceleration - motion.acceleration[0]),
      limits.max_jerk[0] * period + 1.0e-10);
    motion.position[0] += command_velocity * period;
    motion.velocity[0] = command_velocity;
    motion.acceleration[0] = next_acceleration;
    if (braking_started && std::abs(motion.velocity[0]) < 1.0e-10 &&
      std::abs(motion.acceleration[0]) < 1.0e-10)
    {
      settled = true;
      break;
    }
  }

  EXPECT_TRUE(braking_started);
  EXPECT_TRUE(first_stop_reached);
  EXPECT_TRUE(settled);
  EXPECT_GE(minimum_clearance + 1.0e-10, config.hard_clearance_m);
}

TEST(CollisionConstraints, SafeSegmentAndActivationSelection)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.10;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);
  const auto constraints = builder.build(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), oneDofLimits(), 0.01);
  ASSERT_TRUE(constraints.diagnostics.valid) << constraints.diagnostics.failure_reason;
  EXPECT_TRUE(builder.shouldValidateSegment(constraints.diagnostics, 0.12));
  EXPECT_FALSE(builder.shouldValidateSegment(constraints.diagnostics, 0.08));

  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), motionAt(-0.25), 10);
  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  EXPECT_FALSE(segment.unsafe);
  EXPECT_EQ(segment.evaluated_samples, 11U);
  EXPECT_GT(segment.minimum_distance_m, sliderConfig().hard_clearance_m);
}

TEST(CollisionConstraints, CertificateUsesRelativeMotionAndPreservesInvariantClearance)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.133);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, start, twoDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  EXPECT_GT(result.diagnostics.skipped_invariant_contacts, 0U);

  const auto safe_end = twoDofMotion(0.02, 0.1328);
  EXPECT_TRUE(builder.canCertifySegment(result.diagnostics, start.position, safe_end.position));
  const auto sampled = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, safe_end, 2);
  ASSERT_TRUE(sampled.input_valid) << sampled.failure_reason;
  EXPECT_FALSE(sampled.unsafe) << sampled.failure_reason;
  // Common rotation does not change self geometry, but relative translation still
  // consumes the full configured Lipschitz reserve in either direction.
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position, twoDofMotion(0.02, 0.131).position));
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, twoDofMotion(0.02, 0.131).position, start.position));
}

TEST(CollisionConstraints, CertificateChargesCommonAncestorForWorldPairs)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.435;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.3);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, start, twoDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  ASSERT_NE(findWorldPair(result, "wall"), nullptr);
  EXPECT_TRUE(builder.canCertifySegment(
      result.diagnostics, start.position, twoDofMotion(0.001, 0.3).position));
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position, twoDofMotion(0.002, 0.3).position));
}

TEST(CollisionConstraints, CertificateBoundsPairsOmittedBeyondTheQueryDistance)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.6;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = motionAt(0.3);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, start, oneDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  ASSERT_EQ(findWorldPair(result, "wall"), nullptr);
  EXPECT_TRUE(builder.canCertifySegment(
      result.diagnostics, start.position, motionAt(0.31).position));

  // No listed controllable pair can veto this motion. The global query-radius
  // bound must reject it before the initially omitted wall becomes dangerous.
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position, motionAt(0.35).position));
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position, motionAt(0.51).position));
  const auto unsafe = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, motionAt(0.51), 10);
  ASSERT_TRUE(unsafe.input_valid) << unsafe.failure_reason;
  EXPECT_TRUE(unsafe.unsafe);
}

TEST(CollisionConstraints, PathCertificateIncludesIntermediateSuffixExtrema)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.6;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = motionAt(0.3);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, start, oneDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;

  const std::vector<JointMotionState> safe_path{
    motionAt(0.3), motionAt(0.305), motionAt(0.31)};
  EXPECT_TRUE(builder.canCertifyPath(result.diagnostics, safe_path));

  // The path returns to its safe start after an unsafe middle waypoint. A check
  // that considered only the first and last point would incorrectly certify it.
  const std::vector<JointMotionState> unsafe_suffix{
    motionAt(0.3), motionAt(0.51), motionAt(0.3)};
  EXPECT_FALSE(builder.canCertifyPath(result.diagnostics, unsafe_suffix));
}

TEST(CollisionConstraints, UnsafePathReportsItsFirstSampleWithoutEvaluatingTheRemainder)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.6;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  const auto first = motionAt(0.51);
  const auto second = motionAt(0.3);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, first.position);
  state.update(true);

  const auto result = builder.validatePath(
    *fixture.scene, state, *fixture.group, {first, second}, 10U);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  ASSERT_TRUE(result.unsafe);
  EXPECT_EQ(result.first_unsafe_sample, 0U);
  EXPECT_GE(result.evaluated_samples, 1U);
  EXPECT_GT(result.distance_query_count, 0U);
  const auto * first_unsafe = findDensePathSample(result, 0U);
  ASSERT_NE(first_unsafe, nullptr);
  EXPECT_TRUE(first_unsafe->collision_detected || first_unsafe->robust_clearance_violated);
  EXPECT_LT(result.evaluated_samples, result.requested_substeps + 1U);
}

TEST(CollisionConstraints, CertificateRejectsIncompleteDataAndProtectedCorridorExit)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = motionAt(0.799);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, start, oneDofLimits(), 0.01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  ASSERT_TRUE(builder.canCertifySegment(result.diagnostics, start.position, start.position));
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position, motionAt(0.801).position));
  EXPECT_FALSE(builder.canCertifySegment({}, start.position, start.position));
  EXPECT_FALSE(builder.canCertifySegment(result.diagnostics, Eigen::VectorXd{}, start.position));
  EXPECT_FALSE(builder.canCertifySegment(
      result.diagnostics, start.position,
      motionAt(std::numeric_limits<double>::quiet_NaN()).position));

  auto incomplete = result.diagnostics;
  incomplete.distance_query_complete = false;
  EXPECT_FALSE(builder.canCertifySegment(incomplete, start.position, start.position));
  incomplete = result.diagnostics;
  ASSERT_FALSE(incomplete.pairs.empty());
  incomplete.pairs.pop_back();
  EXPECT_FALSE(builder.canCertifySegment(incomplete, start.position, start.position));
  incomplete = result.diagnostics;
  incomplete.pairs.front().relative_motion_variables.clear();
  EXPECT_FALSE(builder.canCertifySegment(incomplete, start.position, start.position));
  incomplete = result.diagnostics;
  incomplete.pairs.front().distance_m = 0.01;
  EXPECT_FALSE(builder.canCertifySegment(incomplete, start.position, start.position));
}

TEST(CollisionConstraints, SegmentDoesNotLetSafeInvariantHideUnsafeGenericPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.125));
  robot_state.update(true);

  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.125), motionAt(0.125), 2);

  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  ASSERT_TRUE(segment.unsafe);
  EXPECT_TRUE(segment.robust_clearance_violated);
  EXPECT_FALSE(segment.samples.front().hard_clearance_violated);
  EXPECT_TRUE(segment.samples.front().robust_clearance_violated);
  // The 20 mm fixed mount pair is physically closest and robustly safe with kappa=0.
  // The iterative SINGLE query must allow it temporarily, then find the 25 mm generic pair
  // whose required robust clearance is 30.8158 mm.
  EXPECT_NEAR(segment.minimum_self_distance_m, 0.020, 1.0e-9);
  EXPECT_EQ(segment.closest_pair, "mount_neighbor <-> moving");
  EXPECT_NE(segment.failure_reason.find("base <-> moving"), std::string::npos);
  EXPECT_NE(segment.failure_reason.find("distance="), std::string::npos);
  EXPECT_NE(segment.failure_reason.find("required="), std::string::npos);
}

TEST(CollisionConstraints, SampledWorldDistancesPreservePairSpecificReservesAndClosestPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("moving", "mount_neighbor", true);
  CollisionConstraintConfig config = sliderConfig();
  config.distance_bounds[{"near", ""}] = 0.0;
  config.distance_bounds[{"moving", ""}] = config.default_distance_lipschitz_m_per_rad;
  CollisionConstraintBuilder builder(config);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation() = Eigen::Vector3d(0.30, 10.12, 0.0);
  fixture.scene->getWorldNonConst()->addToObject(
    "near_wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  obstacle.translation() = Eigen::Vector3d(0.30, -0.125, 0.0);
  fixture.scene->getWorldNonConst()->addToObject(
    "moving_wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, motionAt(0.30).position);
  state.update(true);

  // The nearest world pair is safe; the farther pair violates its larger reserve.
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto result = builder.validatePath(
      *fixture.scene, state, *fixture.group, {motionAt(0.30), motionAt(0.30)}, 2);
    ASSERT_TRUE(result.input_valid) << result.failure_reason;
    ASSERT_TRUE(result.unsafe);
    EXPECT_EQ(result.first_unsafe_sample, 0U);
    EXPECT_NEAR(result.minimum_world_distance_m, 0.020, 1.0e-9);
    EXPECT_EQ(result.closest_pair, "near <-> near_wall");
    EXPECT_NE(result.failure_reason.find("moving <-> moving_wall"), std::string::npos);
    const auto * dense_failure = findDensePathSample(result, 0U);
    ASSERT_NE(dense_failure, nullptr);
    EXPECT_FALSE(dense_failure->collision_detected);
    EXPECT_FALSE(dense_failure->hard_clearance_violated);
    EXPECT_TRUE(dense_failure->robust_clearance_violated);
  }
  // Querying a sample must not change the scene's collision exclusions.
  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("moving", "moving_wall", true);
  EXPECT_FALSE(builder.validateSegment(
      *fixture.scene, state, *fixture.group, motionAt(0.30), motionAt(0.30), 2).unsafe);
}

TEST(CollisionConstraints, EmptySampledDistancesKeepInfiniteClearanceAndNoClosestPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  auto & acm = fixture.scene->getAllowedCollisionMatrixNonConst();
  const auto & links = fixture.robot_model->getLinkModelNames();
  acm.setEntry(links, links, true);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, motionAt(0.30).position);
  state.update(true);
  CollisionConstraintBuilder builder(sliderConfig());
  const auto result = builder.validatePath(
    *fixture.scene, state, *fixture.group, {motionAt(0.30), motionAt(0.32)}, 2);
  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_FALSE(result.unsafe);
  EXPECT_TRUE(result.closest_pair.empty());
  EXPECT_TRUE(result.failure_reason.empty());
  EXPECT_EQ(result.minimum_distance_m, std::numeric_limits<double>::infinity());
  EXPECT_EQ(result.certified_path_block_count, 1U);
  ASSERT_EQ(result.samples.size(), 1U);
  EXPECT_TRUE(result.samples.front().path_block_certificate_attempt);
  EXPECT_TRUE(result.samples.front().path_block_certified);
  for (const auto & sample : result.samples) {
    EXPECT_EQ(sample.minimum_self_distance_m, std::numeric_limits<double>::infinity());
    EXPECT_EQ(sample.minimum_world_distance_m, std::numeric_limits<double>::infinity());
    EXPECT_FALSE(sample.collision_detected);
  }
}

TEST(CollisionConstraints, SampledSignedPenetrationRemainsUnsafeWithOtherClearPairs)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, motionAt(0.30).position);
  state.update(true);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation() = Eigen::Vector3d(0.39, 0.0, 0.0);
  fixture.scene->getWorldNonConst()->addToObject(
    "penetrating_wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  const auto result = builder.validateSegment(
    *fixture.scene, state, *fixture.group, motionAt(0.30), motionAt(0.30), 2);
  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_TRUE(result.unsafe);
  EXPECT_EQ(result.first_unsafe_sample, 0U);
  EXPECT_NEAR(result.minimum_world_distance_m, -0.01, 1.0e-9);
  EXPECT_EQ(result.closest_pair, "moving <-> penetrating_wall");
  for (const auto & sample : result.samples) {
    EXPECT_TRUE(sample.collision_detected);
    EXPECT_TRUE(sample.hard_clearance_violated);
    EXPECT_TRUE(sample.robust_clearance_violated);
  }
}

TEST(CollisionConstraints, SampledQueryLeavesPairsBeyondItsThresholdSafe)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintConfig config = sliderConfig();
  config.tracking_position_error_bound_rad = 0.0;
  CollisionConstraintBuilder builder(config);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, motionAt(0.30).position);
  state.update(true);
  // Every pair is beyond the 15.1 mm query threshold; the closest is at 20 mm.
  const auto result = builder.validateSegment(
    *fixture.scene, state, *fixture.group, motionAt(0.30), motionAt(0.30), 2);
  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_FALSE(result.unsafe);
  EXPECT_GE(result.minimum_self_distance_m, 0.019999);
  EXPECT_TRUE(result.failure_reason.empty());
}

TEST(CollisionConstraints, SampledEqualDistancesKeepClosestPairAndFailureDeterministic)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.41;
  for (const std::string name : {"wall_a", "wall_b"}) {
    fixture.scene->getWorldNonConst()->addToObject(
      name, std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  }
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(fixture.group, motionAt(0.30).position);
  state.update(true);
  CollisionConstraintBuilder builder(sliderConfig());
  const auto first = builder.validateSegment(
    *fixture.scene, state, *fixture.group, motionAt(0.30), motionAt(0.30), 2);
  ASSERT_TRUE(first.input_valid) << first.failure_reason;
  ASSERT_TRUE(first.unsafe);
  EXPECT_NEAR(first.minimum_world_distance_m, 0.01, 1.0e-9);
  EXPECT_NE(first.failure_reason.find(first.closest_pair), std::string::npos);
  for (int repeat = 0; repeat < 5; ++repeat) {
    const auto result = builder.validateSegment(
      *fixture.scene, state, *fixture.group, motionAt(0.30), motionAt(0.30), 2);
    EXPECT_EQ(result.closest_pair, first.closest_pair);
    EXPECT_EQ(result.failure_reason, first.failure_reason);
    EXPECT_EQ(result.first_unsafe_sample, first.first_unsafe_sample);
    EXPECT_DOUBLE_EQ(result.minimum_world_distance_m, first.minimum_world_distance_m);
  }
}

TEST(CollisionConstraints, SegmentUsesTighterMonitorNearLipschitzProfile)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintConfig config = sliderConfig();
  config.monitor_near_link_name = "mount_neighbor";
  config.invariant_mount_neighbor_name = "near";
  CollisionConstraintBuilder builder(config);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);

  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), motionAt(-0.30), 2);

  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  EXPECT_FALSE(segment.unsafe);
  EXPECT_FALSE(segment.robust_clearance_violated);
  EXPECT_NEAR(segment.minimum_self_distance_m, 0.020, 1.0e-9);
  // 15 mm + 0.1 mm + 0.4543 m/rad * 6 mrad = 17.8258 mm, below this 20 mm pair.
}

TEST(CollisionConstraints, MoreSubstepsReduceCertifiedInterSampleReserve)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.133));
  robot_state.update(true);

  const auto coarse = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.133), motionAt(0.139), 2);
  const auto fine = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.133), motionAt(0.139), 10);

  ASSERT_TRUE(coarse.input_valid) << coarse.failure_reason;
  ASSERT_TRUE(fine.input_valid) << fine.failure_reason;
  EXPECT_TRUE(coarse.unsafe);
  EXPECT_TRUE(coarse.robust_clearance_violated);
  EXPECT_FALSE(fine.unsafe);
  EXPECT_FALSE(fine.robust_clearance_violated);
}

TEST(CollisionConstraints, CommonAncestorMotionDoesNotInflateSelfPairSamplingReserve)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  ASSERT_EQ(fixture.group->getVariableCount(), 2U);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.133);
  const auto middle = twoDofMotion(0.1, 0.133);
  const auto end = twoDofMotion(0.2, 0.133);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);

  const auto segment = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, middle, 2);
  const auto path = builder.validatePath(
    *fixture.scene, state, *fixture.group, {start, middle, end}, 2);
  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  EXPECT_FALSE(segment.unsafe) << segment.failure_reason;
  EXPECT_FALSE(path.unsafe) << path.failure_reason;
  // The generic base/moving clearance is 33 mm: above the unchanged 30.8158 mm
  // tracking reserve, but below the former reserve charged for common rotation.
  EXPECT_FALSE(segment.robust_clearance_violated);
  EXPECT_FALSE(path.robust_clearance_violated);
}

TEST(CollisionConstraints, RelativeMotionKeepsItsReserveIncludingAtSharedPathWaypoints)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.133);
  const auto end = twoDofMotion(0.1, 0.139);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto coarse = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 2);
  const auto fine = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 10);
  ASSERT_TRUE(coarse.input_valid) << coarse.failure_reason;
  ASSERT_TRUE(fine.input_valid) << fine.failure_reason;
  EXPECT_TRUE(coarse.unsafe);
  EXPECT_TRUE(coarse.robust_clearance_violated);
  EXPECT_FALSE(fine.unsafe) << fine.failure_reason;

  const auto path = builder.validatePath(
    *fixture.scene, state, *fixture.group,
    {start, twoDofMotion(0.1, 0.133), twoDofMotion(0.2, 0.139)}, 2);
  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  ASSERT_TRUE(path.unsafe);
  EXPECT_EQ(path.first_unsafe_sample, 2U);
  const auto * first_sample = findDensePathSample(path, 0U);
  const auto * unsafe_sample = findDensePathSample(path, 2U);
  ASSERT_NE(first_sample, nullptr);
  ASSERT_NE(unsafe_sample, nullptr);
  EXPECT_FALSE(first_sample->robust_clearance_violated);
  EXPECT_TRUE(unsafe_sample->robust_clearance_violated);
}

TEST(CollisionConstraints, CommonAncestorMotionStillChargesRobotWorldSamplingReserve)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.435;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.3);
  const auto end = twoDofMotion(0.02, 0.3);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto coarse = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 2);
  const auto fine = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 20);
  ASSERT_TRUE(coarse.input_valid) << coarse.failure_reason;
  ASSERT_TRUE(fine.input_valid) << fine.failure_reason;
  EXPECT_TRUE(coarse.unsafe);
  EXPECT_TRUE(coarse.robust_clearance_violated);
  EXPECT_FALSE(fine.unsafe) << fine.failure_reason;
  EXPECT_NE(coarse.failure_reason.find("world"), std::string::npos);
  EXPECT_NE(coarse.failure_reason.find("wall"), std::string::npos);
}

TEST(CollisionConstraints, WorldPairIgnoresMotionOutsideItsRobotBodyAncestry)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle = Eigen::Isometry3d::Identity();
  obstacle.translation().x() = 0.135;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.5);
  const auto end = twoDofMotion(0.0, 0.6);
  state.setJointGroupPositions(fixture.group, start.position);
  state.update(true);
  const auto segment = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 2);
  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  EXPECT_FALSE(segment.unsafe) << segment.failure_reason;
  EXPECT_NEAR(segment.minimum_world_distance_m, 0.035, 1.0e-9);
}

TEST(CollisionConstraints, AttachedBodyUsesItsParentLinkForCommonAncestorMotion)
{
  SliderFixture fixture(commonAncestorUrdf(), kCommonAncestorSrdf);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto start = twoDofMotion(0.0, 0.3);
  const auto end = twoDofMotion(0.1, 0.3);
  state.setJointGroupPositions(fixture.group, start.position);
  Eigen::Isometry3d attached_pose = Eigen::Isometry3d::Identity();
  attached_pose.translation().x() = -0.19;
  state.attachBody(
    "payload", attached_pose, {std::make_shared<shapes::Box>(0.05, 0.05, 0.05)},
    {Eigen::Isometry3d::Identity()}, std::set<std::string>{"moving"}, "moving");
  state.update(true);
  const auto segment = builder.validateSegment(
    *fixture.scene, state, *fixture.group, start, end, 2);
  const auto path = builder.validatePath(
    *fixture.scene, state, *fixture.group, {start, end}, 2);
  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  EXPECT_FALSE(segment.unsafe) << segment.failure_reason;
  EXPECT_FALSE(path.unsafe) << path.failure_reason;
  // The payload/base pair has 35 mm clearance and moves rigidly together.
  // Its clearance reserve must exclude joint motion shared by both bodies.
}

TEST(CollisionConstraints, SegmentStillChecksConfiguredInvariantPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintConfig config = sliderConfig();
  config.hard_clearance_m = 0.021;
  CollisionConstraintBuilder builder(config);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);

  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), motionAt(-0.25), 2);

  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  ASSERT_TRUE(segment.unsafe);
  ASSERT_EQ(segment.samples.size(), 3U);
  EXPECT_NEAR(segment.minimum_self_distance_m, 0.020, 1.0e-9);
  EXPECT_EQ(segment.closest_pair, "mount_neighbor <-> moving");
  for (const auto & sample : segment.samples) {
    EXPECT_EQ(sample.skipped_invariant_contacts, 0U);
  }
}

TEST(CollisionConstraints, SegmentFindsWorldCollisionBetweenSafeEndpoints)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.50;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.30));
  robot_state.update(true);
  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.30), motionAt(0.70), 10);

  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  EXPECT_TRUE(segment.unsafe);
  EXPECT_GT(segment.first_unsafe_sample, 0U);
  EXPECT_LT(segment.first_unsafe_fraction, 0.5);
  EXPECT_LE(segment.minimum_world_distance_m, 0.0);
  EXPECT_NE(segment.closest_pair.find("wall"), std::string::npos);
}

TEST(CollisionConstraints, SegmentFindsSelfCollisionBetweenSafeEndpoints)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.20));
  robot_state.update(true);
  const auto segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.20), motionAt(0.20), 10);

  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  EXPECT_TRUE(segment.unsafe);
  EXPECT_GT(segment.first_unsafe_sample, 0U);
  EXPECT_LT(segment.first_unsafe_fraction, 0.5);
  EXPECT_LE(segment.minimum_self_distance_m, 0.0);
  EXPECT_NE(segment.closest_pair.find("base"), std::string::npos);
}

TEST(CollisionConstraints, SegmentUsesRobustBoundaryAbovePhysicalHardClearance)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.131));
  robot_state.update(true);

  const auto above_robust_boundary = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.131), motionAt(0.131), 2);
  ASSERT_TRUE(above_robust_boundary.input_valid) << above_robust_boundary.failure_reason;
  EXPECT_FALSE(above_robust_boundary.unsafe);

  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.130));
  robot_state.update(true);
  const auto below_robust_boundary = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(0.130), motionAt(0.130), 2);
  ASSERT_TRUE(below_robust_boundary.input_valid) << below_robust_boundary.failure_reason;
  EXPECT_TRUE(below_robust_boundary.unsafe);
  EXPECT_TRUE(below_robust_boundary.robust_clearance_violated);
  EXPECT_FALSE(below_robust_boundary.samples.front().hard_clearance_violated);
}

TEST(CollisionConstraints, StationaryPoseMatchesSegmentDecisionWithOneSample)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  const auto pose = motionAt(0.130);
  robot_state.setJointGroupPositions(fixture.group, pose.position);
  robot_state.update(true);

  const auto stationary_segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group, pose, pose, 2);
  const auto single_pose = builder.validatePose(
    *fixture.scene, robot_state, *fixture.group, pose);

  ASSERT_TRUE(stationary_segment.input_valid) << stationary_segment.failure_reason;
  ASSERT_TRUE(single_pose.input_valid) << single_pose.failure_reason;
  EXPECT_EQ(stationary_segment.unsafe, single_pose.unsafe);
  EXPECT_EQ(stationary_segment.robust_clearance_violated, single_pose.robust_clearance_violated);
  EXPECT_EQ(
    stationary_segment.samples.front().collision_detected,
    single_pose.samples.front().collision_detected);
  EXPECT_DOUBLE_EQ(stationary_segment.minimum_self_distance_m, single_pose.minimum_self_distance_m);
  EXPECT_DOUBLE_EQ(
    stationary_segment.minimum_world_distance_m, single_pose.minimum_world_distance_m);
  EXPECT_EQ(single_pose.evaluated_samples, 1U);
  EXPECT_EQ(single_pose.samples.size(), 1U);
  EXPECT_EQ(stationary_segment.evaluated_samples, 3U);
  EXPECT_EQ(stationary_segment.distance_query_count, single_pose.distance_query_count * 3U);
  EXPECT_TRUE(single_pose.unsafe);
  EXPECT_TRUE(single_pose.robust_clearance_violated);
  EXPECT_FALSE(single_pose.samples.front().hard_clearance_violated);
}

TEST(CollisionConstraints, StationaryPoseHonorsContactsAndRejectsInvalidGeometryOrState)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.19;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);
  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("near", "mount_neighbor", true);
  const auto pose = motionAt(-0.30);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(fixture.group, pose.position);
  robot_state.update(true);
  CollisionConstraintBuilder builder(sliderConfig());

  const auto collision = builder.validatePose(*fixture.scene, robot_state, *fixture.group, pose);
  ASSERT_TRUE(collision.input_valid) << collision.failure_reason;
  EXPECT_TRUE(collision.unsafe);
  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("moving", "wall", true);
  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("mount_neighbor", "wall", true);
  const auto allowed = builder.validatePose(*fixture.scene, robot_state, *fixture.group, pose);
  ASSERT_TRUE(allowed.input_valid) << allowed.failure_reason;
  EXPECT_FALSE(allowed.unsafe) << allowed.failure_reason << "; closest=" << allowed.closest_pair;

  auto invalid = pose;
  invalid.position[0] = std::numeric_limits<double>::quiet_NaN();
  const auto invalid_state = builder.validatePose(
    *fixture.scene, robot_state, *fixture.group, invalid);
  EXPECT_FALSE(invalid_state.input_valid);
  EXPECT_TRUE(invalid_state.unsafe);

  std::string description = kSliderUrdf;
  const auto start = description.find("<link name=\"near\">");
  description.replace(start, description.find("</link>", start) + 7U - start,
    "<link name=\"near\"/>");
  SliderFixture geometry_off(description);
  ASSERT_NE(geometry_off.group, nullptr);
  moveit::core::RobotState geometry_state(geometry_off.robot_model);
  geometry_state.setToDefaultValues();
  geometry_state.update(true);
  const auto no_geometry = CollisionConstraintBuilder(sliderConfig()).validatePose(
    *geometry_off.scene, geometry_state, *geometry_off.group, motionAt(0.0));
  EXPECT_FALSE(no_geometry.input_valid);
  EXPECT_TRUE(no_geometry.unsafe);
  EXPECT_NE(no_geometry.failure_reason.find("'near' has no collision geometry"), std::string::npos);
}

TEST(CollisionConstraints, InvalidSegmentInputFailsClosed)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);

  const auto too_few_substeps = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), motionAt(-0.25), 1);
  EXPECT_FALSE(too_few_substeps.input_valid);
  EXPECT_TRUE(too_few_substeps.unsafe);

  JointMotionState invalid_candidate = motionAt(-0.25);
  invalid_candidate.position[0] = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group,
    motionAt(-0.30), invalid_candidate, 2);
  EXPECT_FALSE(non_finite.input_valid);
  EXPECT_TRUE(non_finite.unsafe);
}

TEST(CollisionConstraints, SegmentRejectsOverflowAndCapacityBeforeAllocatingSamples)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  const CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(fixture.group, motionAt(-0.30).position);
  robot_state.update(true);
  const std::vector<CollisionSegmentSampleDiagnostic> samples;
  for (const std::size_t substeps : {
        std::numeric_limits<std::size_t>::max(), samples.max_size()})
  {
    SCOPED_TRACE(substeps);
    CollisionSegmentValidationResult result;
    ASSERT_NO_THROW(result = builder.validateSegment(
        *fixture.scene, robot_state, *fixture.group, motionAt(-0.30), motionAt(-0.28), substeps));
    EXPECT_FALSE(result.input_valid);
    EXPECT_TRUE(result.unsafe);
    EXPECT_EQ(result.requested_substeps, substeps);
    EXPECT_EQ(result.evaluated_samples, 0U);
    EXPECT_TRUE(result.samples.empty());
    EXPECT_NE(result.failure_reason.find("sample count"), std::string::npos);
  }
}

void expectEquivalentSamples(
  const CollisionSegmentValidationResult & segment,
  const CollisionSegmentValidationResult & path)
{
  ASSERT_TRUE(segment.input_valid) << segment.failure_reason;
  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  EXPECT_EQ(path.unsafe, segment.unsafe);
  EXPECT_EQ(path.requested_substeps, segment.requested_substeps);
  EXPECT_EQ(path.first_unsafe_sample, segment.first_unsafe_sample);
  EXPECT_EQ(path.samples.size(), path.evaluated_samples);
  if (segment.unsafe) {
    EXPECT_DOUBLE_EQ(path.first_unsafe_fraction, segment.first_unsafe_fraction);
  } else {
    EXPECT_EQ(path.robust_clearance_violated, segment.robust_clearance_violated);
    EXPECT_TRUE(std::isnan(segment.first_unsafe_fraction));
    EXPECT_TRUE(std::isnan(path.first_unsafe_fraction));
  }
  for (const auto & right : path.samples) {
    if (right.path_block_certificate_attempt) {
      continue;
    }
    ASSERT_LT(right.sample_index, segment.samples.size());
    const auto & left = segment.samples[right.sample_index];
    EXPECT_DOUBLE_EQ(right.interpolation_fraction, left.interpolation_fraction);
    EXPECT_DOUBLE_EQ(right.protected_joint_position_rad, left.protected_joint_position_rad);
    EXPECT_DOUBLE_EQ(right.minimum_self_distance_m, left.minimum_self_distance_m);
    EXPECT_DOUBLE_EQ(right.minimum_world_distance_m, left.minimum_world_distance_m);
    EXPECT_DOUBLE_EQ(right.minimum_distance_m, left.minimum_distance_m);
    EXPECT_EQ(right.closest_pair, left.closest_pair);
    EXPECT_EQ(right.collision_detected, left.collision_detected);
    EXPECT_EQ(right.hard_clearance_violated, left.hard_clearance_violated);
    EXPECT_EQ(right.robust_clearance_violated, left.robust_clearance_violated);
    EXPECT_EQ(right.protected_joint_corridor_violated, left.protected_joint_corridor_violated);
    EXPECT_EQ(right.skipped_invariant_contacts, left.skipped_invariant_contacts);
  }
}

TEST(CollisionConstraints, SingleSegmentAndPathHaveEquivalentClearanceDiagnostics)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.50;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);
  const CollisionConstraintBuilder builder(sliderConfig());
  const std::vector<std::pair<double, double>> endpoints{
    {-0.30, -0.26}, {-0.13, -0.135}, {-0.30, 0.30}, {0.20, 0.70}, {0.79, 0.82}};
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();

  for (const bool allow_world_contact : {false, true}) {
    fixture.scene->getAllowedCollisionMatrixNonConst().setEntry(
      "moving", "wall", allow_world_contact);
    for (const auto & [start, finish] : endpoints) {
      SCOPED_TRACE(start);
      SCOPED_TRACE(finish);
      std::vector<JointMotionState> waypoints{motionAt(start), motionAt(finish)};
      // Distance queries depend on position; finite derivatives must not change the verdict.
      waypoints[0].velocity[0] = 0.30;
      waypoints[0].acceleration[0] = -0.20;
      waypoints[1].velocity[0] = -0.10;
      waypoints[1].acceleration[0] = 0.50;
      robot_state.setJointGroupPositions(fixture.group, waypoints.front().position);
      robot_state.update(true);
      for (const std::size_t substeps : {2U, 7U, 20U}) {
        SCOPED_TRACE(substeps);
        expectEquivalentSamples(
          builder.validateSegment(
            *fixture.scene, robot_state, *fixture.group, waypoints[0], waypoints[1], substeps),
          builder.validatePath(*fixture.scene, robot_state, *fixture.group, waypoints, substeps));
      }
    }
  }
}

TEST(CollisionConstraints, SingleSegmentAndPathRejectTheSameInvalidMotionStates)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  const CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(fixture.group, motionAt(-0.30).position);
  robot_state.update(true);
  for (const bool wrong_dimension : {false, true}) {
    std::vector<JointMotionState> waypoints{motionAt(-0.30), motionAt(-0.28)};
    if (wrong_dimension) {
      waypoints[1].acceleration = Eigen::VectorXd::Zero(2);
    } else {
      waypoints[1].velocity[0] = std::numeric_limits<double>::quiet_NaN();
    }
    const auto segment = builder.validateSegment(
      *fixture.scene, robot_state, *fixture.group, waypoints[0], waypoints[1], 2U);
    const auto path = builder.validatePath(
      *fixture.scene, robot_state, *fixture.group, waypoints, 2U);
    EXPECT_FALSE(segment.input_valid);
    EXPECT_FALSE(path.input_valid);
    EXPECT_TRUE(segment.unsafe);
    EXPECT_TRUE(path.unsafe);
    EXPECT_EQ(segment.evaluated_samples, 0U);
    EXPECT_EQ(path.evaluated_samples, 0U);
    EXPECT_EQ(segment.failure_reason, path.failure_reason);
  }
}

TEST(CollisionConstraints, SafePathBlockUsesGlobalDenseGridIndices)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);
  const std::vector<JointMotionState> waypoints{
    motionAt(-0.30), motionAt(-0.28), motionAt(-0.26)};

  const auto path = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, waypoints, 3);
  const auto first_segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group, waypoints[0], waypoints[1], 3);
  moveit::core::RobotState middle_robot_state(robot_state);
  middle_robot_state.setJointGroupPositions(fixture.group, waypoints[1].position);
  middle_robot_state.update(true);
  const auto second_segment = builder.validateSegment(
    *fixture.scene, middle_robot_state, *fixture.group, waypoints[1], waypoints[2], 3);

  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  EXPECT_FALSE(path.unsafe);
  EXPECT_EQ(path.requested_substeps, 6U);
  EXPECT_EQ(path.evaluated_samples, path.samples.size());
  EXPECT_LT(path.evaluated_samples, path.requested_substeps + 1U);
  EXPECT_GT(path.distance_query_count, 0U);
  EXPECT_EQ(path.certified_path_block_count, 1U);
  ASSERT_EQ(path.samples.size(), 1U);
  EXPECT_EQ(path.samples.front().sample_index, 3U);
  EXPECT_DOUBLE_EQ(path.samples.front().interpolation_fraction, 0.5);
  EXPECT_DOUBLE_EQ(path.samples.front().protected_joint_position_rad, -0.28);
  EXPECT_TRUE(path.samples.front().path_block_certified);
  EXPECT_EQ(path.samples.front().path_block_end_sample_index, 6U);
  EXPECT_EQ(path.unsafe, first_segment.unsafe || second_segment.unsafe);
  EXPECT_NEAR(
    path.minimum_self_distance_m,
    std::min(
      first_segment.minimum_self_distance_m,
      second_segment.minimum_self_distance_m),
    1.0e-12);
  EXPECT_DOUBLE_EQ(
    path.minimum_world_distance_m,
    std::min(
      first_segment.minimum_world_distance_m,
      second_segment.minimum_world_distance_m));
}

TEST(CollisionConstraints, UnsafePathStopsAtFirstViolationInsideSecondSegment)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.50;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.20));
  robot_state.update(true);
  const std::vector<JointMotionState> waypoints{
    motionAt(0.20), motionAt(0.30), motionAt(0.70)};

  const auto path = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, waypoints, 10);

  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  EXPECT_TRUE(path.unsafe);
  EXPECT_EQ(path.requested_substeps, 20U);
  EXPECT_EQ(path.evaluated_samples, path.samples.size());
  EXPECT_EQ(path.first_unsafe_sample, 11U);
  EXPECT_DOUBLE_EQ(path.first_unsafe_fraction, 11.0 / 20.0);
  EXPECT_TRUE(path.robust_clearance_violated);
  EXPECT_LT(path.evaluated_samples, path.requested_substeps + 1U);
  const auto * first_unsafe = findDensePathSample(path, 11U);
  ASSERT_NE(first_unsafe, nullptr);
  EXPECT_TRUE(first_unsafe->robust_clearance_violated);
}

TEST(CollisionConstraints, AdaptivePathCertificateMatchesDenseOracleForInteriorObstacle)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.50;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState start_state(fixture.robot_model);
  start_state.setToDefaultValues();
  start_state.setJointGroupPositions(fixture.group, motionAt(0.20).position);
  start_state.update(true);
  const std::vector<JointMotionState> waypoints{
    motionAt(0.20), motionAt(0.70), motionAt(0.20)};

  // Independent dense oracle: the first segment crosses the wall before reaching
  // the intermediate waypoint. Its shared-endpoint reserve matches the reverse
  // segment's reserve componentwise.
  const auto dense_first_segment = builder.validateSegment(
    *fixture.scene, start_state, *fixture.group, waypoints[0], waypoints[1], 10U);
  const auto adaptive_path = builder.validatePath(
    *fixture.scene, start_state, *fixture.group, waypoints, 10U);

  ASSERT_TRUE(dense_first_segment.input_valid) << dense_first_segment.failure_reason;
  ASSERT_TRUE(adaptive_path.input_valid) << adaptive_path.failure_reason;
  ASSERT_TRUE(dense_first_segment.unsafe);
  ASSERT_TRUE(adaptive_path.unsafe);
  ASSERT_LT(dense_first_segment.first_unsafe_sample, 10U);
  EXPECT_EQ(adaptive_path.first_unsafe_sample, dense_first_segment.first_unsafe_sample);
  const auto * dense_first_unsafe = findDensePathSample(
    dense_first_segment, dense_first_segment.first_unsafe_sample);
  const auto * path_first_unsafe = findDensePathSample(
    adaptive_path, adaptive_path.first_unsafe_sample);
  ASSERT_NE(dense_first_unsafe, nullptr);
  ASSERT_NE(path_first_unsafe, nullptr);
  EXPECT_EQ(path_first_unsafe->collision_detected, dense_first_unsafe->collision_detected);
  EXPECT_EQ(
    path_first_unsafe->robust_clearance_violated,
    dense_first_unsafe->robust_clearance_violated);
  EXPECT_EQ(path_first_unsafe->closest_pair, dense_first_unsafe->closest_pair);
}

TEST(CollisionConstraints, SharedWaypointMarginViolationPrecedesLaterCollision)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = 0.72;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.20));
  robot_state.update(true);
  const std::vector<JointMotionState> waypoints{
    motionAt(0.20), motionAt(0.21), motionAt(0.70)};

  const auto first_segment = builder.validateSegment(
    *fixture.scene, robot_state, *fixture.group, waypoints[0], waypoints[1], 2U);
  const auto path = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, waypoints, 2U);

  ASSERT_TRUE(first_segment.input_valid) << first_segment.failure_reason;
  EXPECT_FALSE(first_segment.unsafe) << first_segment.failure_reason;
  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  ASSERT_TRUE(path.unsafe);
  EXPECT_EQ(path.requested_substeps, 4U);
  // Global sample 2 is the shared waypoint. Its outgoing segment has much larger
  // inter-sample uncertainty (0.49 / (2 * 2) = 0.1225 rad), so the path rejects
  // that otherwise-clear pose before reaching the later collision at sample 4.
  EXPECT_EQ(path.first_unsafe_sample, 2U);
  EXPECT_EQ(path.evaluated_samples, path.samples.size());
  const auto * safe_before = findDensePathSample(path, 1U);
  const auto * first_unsafe = findDensePathSample(path, 2U);
  ASSERT_NE(safe_before, nullptr);
  EXPECT_FALSE(safe_before->robust_clearance_violated);
  ASSERT_NE(first_unsafe, nullptr);
  EXPECT_TRUE(first_unsafe->robust_clearance_violated);
  EXPECT_FALSE(first_unsafe->collision_detected);
  EXPECT_GT(path.minimum_world_distance_m, 0.0);

  moveit::core::RobotState endpoint_state(robot_state);
  endpoint_state.setJointGroupPositions(fixture.group, waypoints.back().position);
  endpoint_state.update(true);
  const auto later_collision = builder.validatePose(
    *fixture.scene, endpoint_state, *fixture.group, waypoints.back());
  ASSERT_TRUE(later_collision.input_valid) << later_collision.failure_reason;
  EXPECT_TRUE(later_collision.unsafe);
  ASSERT_EQ(later_collision.samples.size(), 1U);
  EXPECT_TRUE(later_collision.samples.front().collision_detected);
  EXPECT_NE(later_collision.closest_pair.find("wall"), std::string::npos);
}

TEST(CollisionConstraints, PathSharedWaypointUsesBothAdjacentSegmentReserves)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.131));
  robot_state.update(true);
  const std::vector<JointMotionState> waypoints{
    motionAt(0.131), motionAt(0.131), motionAt(0.139)};

  const auto path = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, waypoints, 2);

  ASSERT_TRUE(path.input_valid) << path.failure_reason;
  ASSERT_TRUE(path.unsafe);
  EXPECT_EQ(path.first_unsafe_sample, 2U);
  EXPECT_DOUBLE_EQ(path.first_unsafe_fraction, 0.5);
  const auto * safe_start = findDensePathSample(path, 0U);
  const auto * safe_adjacent = findDensePathSample(path, 1U);
  const auto * unsafe_shared = findDensePathSample(path, 2U);
  ASSERT_NE(safe_start, nullptr);
  ASSERT_NE(safe_adjacent, nullptr);
  ASSERT_NE(unsafe_shared, nullptr);
  EXPECT_FALSE(safe_start->robust_clearance_violated);
  EXPECT_FALSE(safe_adjacent->robust_clearance_violated);
  EXPECT_TRUE(unsafe_shared->robust_clearance_violated);
}

TEST(CollisionConstraints, InvalidPathInputFailsClosed)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);
  const std::vector<JointMotionState> valid_waypoints{
    motionAt(-0.30), motionAt(-0.28)};

  const auto too_few_waypoints = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, {motionAt(-0.30)}, 2);
  EXPECT_FALSE(too_few_waypoints.input_valid);
  EXPECT_TRUE(too_few_waypoints.unsafe);

  const auto too_few_substeps = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, valid_waypoints, 1);
  EXPECT_FALSE(too_few_substeps.input_valid);
  EXPECT_TRUE(too_few_substeps.unsafe);

  const std::vector<JointMotionState> three_waypoints{
    motionAt(-0.30), motionAt(-0.29), motionAt(-0.28)};
  const auto overflowing_sample_count = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, three_waypoints,
    std::numeric_limits<std::size_t>::max());
  EXPECT_FALSE(overflowing_sample_count.input_valid);
  EXPECT_TRUE(overflowing_sample_count.unsafe);
  EXPECT_NE(overflowing_sample_count.failure_reason.find("overflows"), std::string::npos);

  auto non_finite_waypoints = three_waypoints;
  non_finite_waypoints[1].velocity[0] = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, non_finite_waypoints, 2);
  EXPECT_FALSE(non_finite.input_valid);
  EXPECT_TRUE(non_finite.unsafe);

  auto wrong_dimension_waypoints = valid_waypoints;
  wrong_dimension_waypoints[1].acceleration = Eigen::VectorXd::Zero(2);
  const auto wrong_dimension = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, wrong_dimension_waypoints, 2);
  EXPECT_FALSE(wrong_dimension.input_valid);
  EXPECT_TRUE(wrong_dimension.unsafe);

  auto mismatched_start = valid_waypoints;
  mismatched_start[0] = motionAt(-0.29);
  const auto mismatch = builder.validatePath(
    *fixture.scene, robot_state, *fixture.group, mismatched_start, 2);
  EXPECT_FALSE(mismatch.input_valid);
  EXPECT_TRUE(mismatch.unsafe);
}

TEST(CollisionConstraints, ActualStateCheckReportsWorldClearanceAndClosestPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.19;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);

  const auto result = CollisionConstraintBuilder(sliderConfig()).validateActualState(
    *fixture.scene, robot_state, *fixture.group);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_TRUE(result.unsafe);
  EXPECT_FALSE(result.collision_detected);
  EXPECT_TRUE(result.hard_clearance_violated);
  EXPECT_FALSE(result.protected_joint_corridor_violated);
  EXPECT_NEAR(result.minimum_world_distance_m, 0.010, 1.0e-9);
  EXPECT_NEAR(result.minimum_distance_m, result.minimum_world_distance_m, 1.0e-12);
  EXPECT_EQ(result.closest_pair, "moving <-> wall");
}

TEST(CollisionConstraints, ActualStateCheckReportsSelfClearanceAndClosestPair)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.114));
  robot_state.update(true);

  const auto result = CollisionConstraintBuilder(sliderConfig()).validateActualState(
    *fixture.scene, robot_state, *fixture.group);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_TRUE(result.unsafe);
  EXPECT_FALSE(result.collision_detected);
  EXPECT_TRUE(result.hard_clearance_violated);
  EXPECT_NEAR(result.minimum_self_distance_m, 0.014, 1.0e-9);
  EXPECT_NEAR(result.minimum_distance_m, result.minimum_self_distance_m, 1.0e-12);
  EXPECT_EQ(result.closest_pair, "base <-> moving");
}

TEST(CollisionConstraints, ActualStateCheckHonorsAllowedCollisionMatrix)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation().x() = -0.19;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), obstacle_pose);

  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, -0.30));
  robot_state.update(true);
  CollisionConstraintBuilder builder(sliderConfig());
  ASSERT_TRUE(builder.validateActualState(
      *fixture.scene, robot_state, *fixture.group).unsafe);

  fixture.scene->getAllowedCollisionMatrixNonConst().setEntry("moving", "wall", true);
  const auto allowed = builder.validateActualState(
    *fixture.scene, robot_state, *fixture.group);

  ASSERT_TRUE(allowed.input_valid) << allowed.failure_reason;
  EXPECT_FALSE(allowed.unsafe);
  EXPECT_FALSE(allowed.collision_detected);
  EXPECT_FALSE(allowed.hard_clearance_violated);
}

TEST(CollisionConstraints, ActualStateCheckRejectsProtectedJointOutsideCorridor)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group, Eigen::VectorXd::Constant(1, 0.81));
  robot_state.update(true);

  const auto result = CollisionConstraintBuilder(sliderConfig()).validateActualState(
    *fixture.scene, robot_state, *fixture.group);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_TRUE(result.unsafe);
  EXPECT_TRUE(result.protected_joint_corridor_violated);
  EXPECT_DOUBLE_EQ(result.protected_joint_position_rad, 0.81);
  EXPECT_FALSE(result.hard_clearance_violated);
}

TEST(CollisionConstraints, DoubleFeedbackBoundaryToleranceDoesNotExpandNominalBounds)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  const auto config = sliderConfig();
  const CollisionConstraintBuilder builder(config);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  const auto validate_at = [&](const double position) {
      robot_state.setJointGroupPositions(
        fixture.group, Eigen::VectorXd::Constant(1, position));
      robot_state.update(true);
      return builder.validateActualState(*fixture.scene, robot_state, *fixture.group);
    };
  const double feedback_tolerance = kControllerPositionFeedbackTolerance.value;

  for (const double boundary : {config.protected_joint_lower_rad,
      config.protected_joint_upper_rad})
  {
    for (const double offset : {0.0, -1.0e-12, 1.0e-12}) {
      const auto result = validate_at(boundary + offset);
      ASSERT_TRUE(result.input_valid) << result.failure_reason;
      EXPECT_EQ(result.protected_joint_boundary_class, BoundaryClass::kBoundary);
      EXPECT_FALSE(result.protected_joint_corridor_violated);
      EXPECT_FALSE(result.unsafe);
      EXPECT_FALSE(result.collision_detected);
      EXPECT_FALSE(result.hard_clearance_violated);
    }

    const double outside = boundary < 0.0 ?
      boundary - 1.1 * feedback_tolerance : boundary + 1.1 * feedback_tolerance;
    const auto outside_result = validate_at(outside);
    ASSERT_TRUE(outside_result.input_valid) << outside_result.failure_reason;
    EXPECT_EQ(outside_result.protected_joint_boundary_class, BoundaryClass::kOutside);
    EXPECT_TRUE(outside_result.protected_joint_corridor_violated);
    EXPECT_TRUE(outside_result.unsafe);
    EXPECT_FALSE(outside_result.collision_detected);
    EXPECT_FALSE(outside_result.hard_clearance_violated);
  }
}

TEST(CollisionConstraints, BuilderKeepsCorridorDiagnosticsWithoutJ5LatencyRow)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  CollisionConstraintBuilder builder(sliderConfig());
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();

  auto inside = motionAt(0.7991);
  robot_state.setJointGroupPositions(fixture.group, inside.position);
  robot_state.update(true);
  const auto inside_result = builder.build(
    *fixture.scene, robot_state, *fixture.group, inside, oneDofLimits(), 0.01);
  ASSERT_TRUE(inside_result.diagnostics.valid) << inside_result.diagnostics.failure_reason;
  EXPECT_FALSE(inside_result.diagnostics.protected_joint_corridor_violated);
  EXPECT_EQ(inside_result.diagnostics.protected_joint_index, 0U);
  EXPECT_EQ(
    inside_result.constraints.size(), inside_result.diagnostics.active_distance_constraints);

  const auto [minimum_velocity, maximum_velocity] =
    oneDofReachableVelocityInterval(inside, oneDofLimits(), 0.01);
  const double old_j5_cap = maximumSafeVelocityTowardBoundary(
    sliderConfig().protected_joint_upper_rad - inside.position[0], inside.velocity[0],
    minimum_velocity, maximum_velocity, oneDofLimits().max_acceleration[0],
    oneDofLimits().max_jerk[0], 0.01, 0.02);
  EXPECT_LT(old_j5_cap, maximum_velocity);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = 0.01;
  qp_config.solver_time_limit_sec = 0.1;
  qp_config.solver_max_iterations = 2000;
  const HierarchicalVelocityQp solver(qp_config);
  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(1, 1);
  task.primary_reference = Eigen::VectorXd::Constant(1, 1.0);
  task.primary_weights = Eigen::VectorXd::Ones(1);
  task.secondary_matrix = Eigen::MatrixXd(0, 1);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);
  const QpResult qp_result = solver.solve(
    task, inside, oneDofLimits(), inside_result.constraints);
  ASSERT_TRUE(qp_result.command_available());
  EXPECT_NEAR(qp_result.joint_velocity[0], maximum_velocity, 1.0e-6);
  EXPECT_GT(qp_result.joint_velocity[0], old_j5_cap);

  JointMotionState candidate;
  candidate.position = inside.position + qp_result.joint_velocity * 0.01;
  candidate.velocity = qp_result.joint_velocity;
  candidate.acceleration = (candidate.velocity - inside.velocity) / 0.01;
  const EmergencyBrakeTail tail = EmergencyBrakeTailGenerator().generate(
    candidate, oneDofLimits());
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  for (const auto & point : tail.points) {
    EXPECT_GE(point.position[0], oneDofLimits().lower_position[0]);
    EXPECT_LE(point.position[0], sliderConfig().protected_joint_upper_rad);
  }

  auto outside = motionAt(0.81);
  robot_state.setJointGroupPositions(fixture.group, outside.position);
  robot_state.update(true);
  const auto outside_result = builder.build(
    *fixture.scene, robot_state, *fixture.group, outside, oneDofLimits(), 0.01);
  ASSERT_TRUE(outside_result.diagnostics.valid) << outside_result.diagnostics.failure_reason;
  EXPECT_TRUE(outside_result.diagnostics.protected_joint_corridor_violated);
}

TEST(CollisionConstraints, FloatEncodedProtectedBoundaryIsReportedUnresolved)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  const auto config = sliderConfig();
  const double measured = static_cast<double>(static_cast<float>(config.protected_joint_upper_rad));
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(fixture.group, Eigen::VectorXd::Constant(1, measured));
  robot_state.update(true);

  const auto result = CollisionConstraintBuilder(config).validateActualState(
    *fixture.scene, robot_state, *fixture.group,
    ActualPositionEncoding::kSdkFloatPromotedToDouble);

  ASSERT_TRUE(result.input_valid) << result.failure_reason;
  EXPECT_EQ(result.protected_joint_boundary_class, BoundaryClass::kBoundary);
  EXPECT_TRUE(result.protected_joint_corridor_violated);
  EXPECT_TRUE(result.unsafe);
}

TEST(CollisionConstraints, InvalidActualStateCheckFailsClosed)
{
  SliderFixture fixture;
  ASSERT_NE(fixture.scene, nullptr);
  ASSERT_NE(fixture.group, nullptr);
  moveit::core::RobotState robot_state(fixture.robot_model);
  robot_state.setToDefaultValues();
  robot_state.setJointGroupPositions(
    fixture.group,
    Eigen::VectorXd::Constant(1, std::numeric_limits<double>::quiet_NaN()));

  const auto result = CollisionConstraintBuilder(sliderConfig()).validateActualState(
    *fixture.scene, robot_state, *fixture.group);

  EXPECT_FALSE(result.input_valid);
  EXPECT_TRUE(result.unsafe);
  EXPECT_NE(result.failure_reason.find("non-finite"), std::string::npos);
}


TEST(CollisionGeometryBounds, DerivesBoundsAndCancelsCommonAncestorMotion)
{
  std::string description = kSliderUrdf;
  description.replace(description.find("prismatic"), 9U, "revolute");
  SliderFixture fixture(description);
  ASSERT_NE(fixture.group, nullptr);
  auto config = sliderConfig();
  configureGeometryBounds(config, *fixture.robot_model, *fixture.group);
  const auto world = std::make_pair(std::string("moving"), std::string());
  const auto fixed_pair = std::make_pair(std::string("mount_neighbor"), std::string("moving"));
  const auto relative_pair = std::make_pair(std::string("base"), std::string("moving"));
  EXPECT_NEAR(config.distance_bounds.at(world), std::sqrt(3.0) * 0.05, 1.0e-9);
  EXPECT_DOUBLE_EQ(config.distance_bounds.at(fixed_pair), 0.0);
  EXPECT_NEAR(config.distance_bounds.at(relative_pair), config.distance_bounds.at(world), 1.0e-9);
}

TEST(CollisionConstraints, MonitorFrameWithoutGeometryKeepsCameraCollisionChecks)
{
  std::string description = kSliderUrdf;
  const auto start = description.find("<link name=\"moving\">");
  description.replace(start, description.find("</link>", start) + 7U - start,
    "<link name=\"moving\"/>");
  description.insert(description.find("</robot>"),
        R"(
  <link name="camera">
    <collision><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <joint name="camera_mount" type="fixed">
    <parent link="moving"/><child link="camera"/><origin xyz="0 0.5 0"/>
  </joint>
)");
  SliderFixture fixture(description);
  ASSERT_NE(fixture.group, nullptr);
  Eigen::Isometry3d obstacle_pose = Eigen::Isometry3d::Identity();
  obstacle_pose.translation() = Eigen::Vector3d(.40, .5, 0);
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(.08, .08, .08), obstacle_pose);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  const auto motion = motionAt(.30);
  state.setJointGroupPositions(fixture.group, motion.position);
  state.update(true);
  const CollisionConstraintBuilder builder(sliderConfig());
  const auto result = builder.build(
    *fixture.scene, state, *fixture.group, motion, oneDofLimits(), .01);
  ASSERT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
  const auto * pair = findWorldPair(result, "wall");
  ASSERT_NE(pair, nullptr);
  EXPECT_TRUE(pair->first_body == "camera" || pair->second_body == "camera");
  EXPECT_NEAR(pair->distance_m, .02, 1.0e-9);
  EXPECT_TRUE(std::isfinite(pair->gradient_norm));
  for (const auto & constraint : result.constraints) {
    EXPECT_TRUE(constraint.coefficients.allFinite());
    EXPECT_FALSE(std::isnan(constraint.lower_bound));
    EXPECT_FALSE(std::isnan(constraint.upper_bound));
  }
  const auto actual = builder.validateActualState(*fixture.scene, state, *fixture.group);
  EXPECT_TRUE(actual.input_valid) << actual.failure_reason;
  EXPECT_FALSE(actual.unsafe);
  // The actual-state check queries only inside hard_clearance_m (15 mm).
  // Move the obstacle closer to verify that the remaining camera still stops it.
  fixture.scene->getWorldNonConst()->removeObject("wall");
  obstacle_pose.translation().x() = .39;
  fixture.scene->getWorldNonConst()->addToObject(
    "wall", std::make_shared<shapes::Box>(.08, .08, .08), obstacle_pose);
  const auto close = builder.validateActualState(*fixture.scene, state, *fixture.group);
  EXPECT_TRUE(close.input_valid) << close.failure_reason;
  EXPECT_NEAR(close.minimum_world_distance_m, .01, 1.0e-9);
  EXPECT_EQ(close.closest_pair, "camera <-> wall");
  EXPECT_TRUE(close.hard_clearance_violated);
  EXPECT_TRUE(close.unsafe);

  description.replace(description.find("prismatic"), 9U, "revolute");
  SliderFixture revolute(description);
  auto config = sliderConfig();
  configureGeometryBounds(config, *revolute.robot_model, *revolute.group);
  EXPECT_EQ(config.distance_bounds.count({"moving", ""}), 0U);
  EXPECT_GT(config.distance_bounds.at({"camera", ""}), 0.0);
  for (const auto & entry : config.distance_bounds) {
    EXPECT_TRUE(std::isfinite(entry.second));
    EXPECT_GE(entry.second, 0.0);
  }
}

TEST(CollisionConstraints, MissingGeometryOnArmLinksIsStillRejected)
{
  std::string description = kSliderUrdf;
  const auto start = description.find("<link name=\"near\">");
  description.replace(start, description.find("</link>", start) + 7U - start,
    "<link name=\"near\"/>");
  SliderFixture fixture(description);
  ASSERT_NE(fixture.group, nullptr);
  moveit::core::RobotState state(fixture.robot_model);
  state.setToDefaultValues();
  state.update(true);
  const auto actual = CollisionConstraintBuilder(sliderConfig()).validateActualState(
    *fixture.scene, state, *fixture.group);
  EXPECT_FALSE(actual.input_valid);
  EXPECT_NE(actual.failure_reason.find("'near' has no collision geometry"), std::string::npos);
}


TEST(CollisionConstraints, PhysicalBoundaryStartDoesNotRequireOneTickMarginRecovery)
{
  SliderFixture fixture;
  auto config = sliderConfig();
  config.protected_joint_lower_rad = -1.0;
  config.protected_joint_upper_rad = 1.0;
  auto limits = oneDofLimits();
  limits.position_margin[0] = 0.1;
  for (const double position : {-0.999, 0.999}) {
    auto motion = motionAt(position);
    moveit::core::RobotState state(fixture.robot_model);
    state.setToDefaultValues();
    state.setJointGroupPositions(fixture.group, motion.position);
    state.update();
    const auto result = CollisionConstraintBuilder(config).build(
      *fixture.scene, state, *fixture.group, motion, limits, 0.01);
    EXPECT_TRUE(result.diagnostics.valid) << result.diagnostics.failure_reason;
    EXPECT_FALSE(result.diagnostics.protected_joint_corridor_violated);
  }
}

}  // namespace
}  // namespace face_tracking_arm::control
