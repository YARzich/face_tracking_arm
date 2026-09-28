// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kTolerance = 5.0e-5;

HierarchicalVelocityQpConfig make_config()
{
  HierarchicalVelocityQpConfig config;
  config.residual_command_latency_sec = 0.0;
  config.velocity_regularization = 1.0e-8;
  config.continuity_weight = 0.0;
  config.primary_preservation_tolerance = 1.0e-4;
  config.solution_feasibility_tolerance = 1.0e-6;
  config.solver_absolute_tolerance = 1.0e-8;
  config.solver_relative_tolerance = 1.0e-8;
  config.solver_time_limit_sec = 0.1;
  config.solver_max_iterations = 500;
  return config;
}

JointMotionState make_state(const Eigen::Index joint_count)
{
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(joint_count);
  state.velocity = Eigen::VectorXd::Zero(joint_count);
  state.acceleration = Eigen::VectorXd::Zero(joint_count);
  return state;
}

JointMotionLimits make_limits(const Eigen::Index joint_count)
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(joint_count, -10.0);
  limits.upper_position = Eigen::VectorXd::Constant(joint_count, 10.0);
  limits.position_margin = Eigen::VectorXd::Zero(joint_count);
  limits.max_velocity = Eigen::VectorXd::Constant(joint_count, 10.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(joint_count, 1000.0);
  limits.max_jerk = Eigen::VectorXd::Constant(joint_count, 1.0e7);
  return limits;
}

HierarchicalVelocityTask make_task(
  const Eigen::MatrixXd & primary_matrix,
  const Eigen::VectorXd & primary_reference,
  const Eigen::MatrixXd & secondary_matrix,
  const Eigen::VectorXd & secondary_reference)
{
  HierarchicalVelocityTask task;
  task.primary_matrix = primary_matrix;
  task.primary_reference = primary_reference;
  task.primary_weights = Eigen::VectorXd::Ones(primary_matrix.rows());
  task.secondary_matrix = secondary_matrix;
  task.secondary_reference = secondary_reference;
  task.secondary_weights = Eigen::VectorXd::Ones(secondary_matrix.rows());
  return task;
}

struct DiscreteEnvelopeSample
{
  double maximum_position{0.0};
  double maximum_velocity{0.0};
};

DiscreteEnvelopeSample simulate_discrete_adverse_then_brake(
  const double current_position,
  const double current_velocity,
  const double candidate_velocity,
  const double signed_direction,
  const double max_acceleration,
  const double max_jerk,
  const HierarchicalVelocityQpConfig & config)
{
  double position = signed_direction *
    (current_position + candidate_velocity * config.period_sec);
  double velocity = signed_direction * candidate_velocity;
  double acceleration = signed_direction *
    (candidate_velocity - current_velocity) / config.period_sec;
  const double acceleration_step = max_jerk * config.period_sec;
  DiscreteEnvelopeSample sample{position, velocity};
  const auto integrate_tick = [&]() {
      velocity += acceleration * config.period_sec;
      position += velocity * config.period_sec;
      sample.maximum_position = std::max(sample.maximum_position, position);
      sample.maximum_velocity = std::max(sample.maximum_velocity, velocity);
    };

  const auto residual_ticks = static_cast<std::size_t>(std::ceil(
      config.residual_command_latency_sec / config.period_sec));
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
  return sample;
}

TEST(JointCenteringGuidance, LeavesMiddleFreeAndPushesInwardAtBothEnds)
{
  EXPECT_DOUBLE_EQ(jointCenteringGuidance(0.0, -6.0, 6.0, 0.35, 0.4).activation, 0.0);
  EXPECT_DOUBLE_EQ(jointCenteringGuidance(2.0, -6.0, 6.0, 0.35, 0.4).velocity, 0.0);
  const auto upper = jointCenteringGuidance(5.0, -6.0, 6.0, 0.35, 0.4);
  const auto lower = jointCenteringGuidance(-5.0, -6.0, 6.0, 0.35, 0.4);
  EXPECT_GT(upper.activation, 0.0);
  EXPECT_LT(upper.activation, 1.0);
  EXPECT_LT(upper.velocity, 0.0);
  EXPECT_NEAR(upper.velocity, -lower.velocity, 1.0e-12);
  EXPECT_DOUBLE_EQ(jointCenteringGuidance(6.0, -6.0, 6.0, 0.35, 0.4).velocity, -0.4);
  EXPECT_DOUBLE_EQ(jointCenteringGuidance(-7.0, -6.0, 6.0, 0.35, 0.4).velocity, 0.4);
  EXPECT_DOUBLE_EQ(jointCenteringGuidance(10.0, 4.0, 16.0, 0.35, 0.4).velocity, 0.0);
}

TEST(JointCenteringGuidance, EntersSmoothlyAndRejectsInvalidParameters)
{
  const auto edge = jointCenteringGuidance(2.1, -6.0, 6.0, 0.35, 0.4);
  const auto nearby = jointCenteringGuidance(2.10001, -6.0, 6.0, 0.35, 0.4);
  EXPECT_NEAR(edge.activation, 0.0, 1.0e-12);
  EXPECT_NEAR(nearby.velocity, 0.0, 1.0e-9);
  EXPECT_THROW(jointCenteringGuidance(0.0, 1.0, -1.0, 0.35, 0.4), std::invalid_argument);
  EXPECT_THROW(jointCenteringGuidance(0.0, -1.0, 1.0, 1.0, 0.4), std::invalid_argument);
  EXPECT_THROW(jointCenteringGuidance(0.0, -1.0, 1.0, 0.35, 0.0), std::invalid_argument);
  EXPECT_THROW(
    jointCenteringGuidance(std::numeric_limits<double>::quiet_NaN(), -1.0, 1.0, 0.35, 0.4),
    std::invalid_argument);
}

TEST(HierarchicalVelocityQp, CentersBoundedJointWithoutLosingPrimaryOrHardLimits)
{
  const auto config = make_config();
  HierarchicalVelocityQp solver(config);
  auto state = make_state(2);
  auto limits = make_limits(2);
  state.position[0] = 5.0;
  limits.lower_position[0] = -6.0;
  limits.upper_position[0] = 6.0;
  limits.max_velocity[0] = 0.05;
  const auto guidance = jointCenteringGuidance(5.0, -6.0, 6.0, 0.35, 0.4);
  Eigen::Matrix<double, 1, 2> primary;
  primary << 1.0, 1.0;
  Eigen::Matrix<double, 1, 2> secondary;
  secondary << 1.0, 0.0;
  const auto task = make_task(
    primary, Eigen::VectorXd::Constant(1, 0.1),
    secondary, Eigen::VectorXd::Constant(1, guidance.velocity));
  const auto result = solver.solve(task, state, limits);
  ASSERT_TRUE(result.command_available());
  EXPECT_LT(result.joint_velocity[0], -0.01);
  EXPECT_GE(result.joint_velocity[0], -0.05 - config.solution_feasibility_tolerance);
  EXPECT_NEAR(result.joint_velocity.sum(), 0.1, config.primary_preservation_tolerance * 1.1);
}

TEST(HierarchicalVelocityQp, PreservesPrimaryTaskWhileUsingItsNullSpace)
{
  const HierarchicalVelocityQpConfig config = make_config();
  const HierarchicalVelocityQp solver{config};
  Eigen::Matrix<double, 1, 2> primary;
  primary << 1.0, 1.0;
  Eigen::Matrix<double, 1, 2> secondary;
  secondary << 1.0, -1.0;
  const HierarchicalVelocityTask task = make_task(
    primary, Eigen::VectorXd::Constant(1, 1.0),
    secondary, Eigen::VectorXd::Constant(1, 2.0));

  const QpResult result = solver.solve(task, make_state(2), make_limits(2));

  ASSERT_TRUE(result.command_available());
  EXPECT_TRUE(result.fully_solved());
  ASSERT_EQ(result.joint_velocity.size(), 2);
  EXPECT_NEAR(
    (primary * result.joint_velocity)[0], 1.0,
    config.primary_preservation_tolerance + kTolerance);
  EXPECT_NEAR((secondary * result.joint_velocity)[0], 2.0, kTolerance);
  EXPECT_NEAR(result.achieved_primary_velocity[0], 1.0,
    config.primary_preservation_tolerance + kTolerance);
}

TEST(HierarchicalVelocityQp, PrimaryOrientationWinsAConflictingSecondaryRequest)
{
  const HierarchicalVelocityQpConfig config = make_config();
  const HierarchicalVelocityQp solver{config};
  Eigen::Matrix<double, 1, 1> same_joint;
  same_joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    same_joint, Eigen::VectorXd::Constant(1, 0.8),
    same_joint, Eigen::VectorXd::Constant(1, -0.8));

  const QpResult result = solver.solve(task, make_state(1), make_limits(1));

  ASSERT_TRUE(result.fully_solved());
  EXPECT_GE(
    result.joint_velocity[0],
    0.8 - config.primary_preservation_tolerance - kTolerance);
  EXPECT_LT(result.joint_velocity[0], 0.81);
}

TEST(HierarchicalVelocityQp, AppliesGenericHardVelocityConstraintsInBothStages)
{
  const HierarchicalVelocityQp solver{make_config()};
  Eigen::Matrix<double, 1, 2> primary;
  primary << 1.0, 0.0;
  Eigen::Matrix<double, 1, 2> secondary;
  secondary << 0.0, 1.0;
  const HierarchicalVelocityTask task = make_task(
    primary, Eigen::VectorXd::Constant(1, -2.0),
    secondary, Eigen::VectorXd::Constant(1, 3.0));
  LinearVelocityConstraint safety_row;
  safety_row.coefficients = Eigen::Vector2d{1.0, 0.0};
  safety_row.lower_bound = 0.3;
  safety_row.upper_bound = 0.4;

  const QpResult result = solver.solve(
    task, make_state(2), make_limits(2), {safety_row});

  ASSERT_TRUE(result.command_available());
  EXPECT_GE(result.joint_velocity[0], 0.3 - kTolerance);
  EXPECT_LE(result.joint_velocity[0], 0.4 + kTolerance);
}

TEST(HierarchicalVelocityQp, BrakesAtTheJerkLimitedBoundaryInsteadOfJumpingToZero)
{
  const HierarchicalVelocityQpConfig config = make_config();
  const HierarchicalVelocityQp solver{config};
  JointMotionState state = make_state(1);
  state.velocity[0] = 0.4;
  state.acceleration[0] = 0.1;
  JointMotionLimits limits = make_limits(1);
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;

  const QpResult first = solver.brake(state, limits);

  ASSERT_TRUE(first.command_available());
  ASSERT_EQ(first.joint_velocity.size(), 1);
  const double first_acceleration =
    (first.joint_velocity[0] - state.velocity[0]) / config.period_sec;
  const double first_jerk =
    (first_acceleration - state.acceleration[0]) / config.period_sec;
  EXPECT_NEAR(first.joint_velocity[0], 0.399, kTolerance);
  EXPECT_LE(std::abs(first_acceleration), limits.max_acceleration[0] + kTolerance);
  EXPECT_LE(std::abs(first_jerk), limits.max_jerk[0] + kTolerance);

  state.position[0] += first.joint_velocity[0] * config.period_sec;
  state.velocity[0] = first.joint_velocity[0];
  state.acceleration[0] = first_acceleration;
  const QpResult second = solver.brake(state, limits);

  ASSERT_TRUE(second.command_available());
  const double second_acceleration =
    (second.joint_velocity[0] - state.velocity[0]) / config.period_sec;
  const double second_jerk =
    (second_acceleration - state.acceleration[0]) / config.period_sec;
  EXPECT_LT(second.joint_velocity[0], first.joint_velocity[0]);
  EXPECT_NEAR(second.joint_velocity[0], 0.396, kTolerance);
  EXPECT_LE(std::abs(second_jerk), limits.max_jerk[0] + kTolerance);
}

TEST(HierarchicalVelocityQp, TwentyMillisecondLowPowerPeriodPreservesMotionLimits)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.period_sec = 0.02;
  const HierarchicalVelocityQp solver{config};
  JointMotionState state = make_state(1);
  state.velocity[0] = 0.4;
  state.acceleration[0] = 0.5;
  JointMotionLimits limits = make_limits(1);
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 0.8), Eigen::MatrixXd{0, 1},
    Eigen::VectorXd{0});

  const QpResult result = solver.solve(task, state, limits);
  ASSERT_TRUE(result.command_available());
  const double acceleration =
    (result.joint_velocity[0] - state.velocity[0]) / config.period_sec;
  const double jerk = (acceleration - state.acceleration[0]) / config.period_sec;
  EXPECT_LE(std::abs(result.joint_velocity[0]), limits.max_velocity[0] + kTolerance);
  EXPECT_LE(std::abs(acceleration), limits.max_acceleration[0] + kTolerance);
  EXPECT_LE(std::abs(jerk), limits.max_jerk[0] + kTolerance);

  JointMotionState next = state;
  next.position[0] += result.joint_velocity[0] * config.period_sec;
  next.velocity = result.joint_velocity;
  next.acceleration[0] = acceleration;
  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = config.period_sec;
  const EmergencyBrakeTail tail =
    EmergencyBrakeTailGenerator{tail_config}.generate(next, limits);
  EXPECT_TRUE(tail.command_available()) << tail.failure_reason;
}

TEST(HierarchicalVelocityQp, BrakingRetainsExternalSafetyRows)
{
  const HierarchicalVelocityQp solver{make_config()};
  JointMotionState state = make_state(1);
  LinearVelocityConstraint constraint;
  constraint.coefficients = Eigen::VectorXd::Ones(1);
  constraint.lower_bound = 0.25;
  constraint.upper_bound = std::numeric_limits<double>::infinity();

  const QpResult result = solver.brake(state, make_limits(1), {constraint});

  ASSERT_TRUE(result.command_available());
  EXPECT_NEAR(result.joint_velocity[0], 0.25, kTolerance);
}

TEST(HierarchicalVelocityQp, PhysicalPositionBoundPreventsACommandFurtherIntoTheLimit)
{
  const HierarchicalVelocityQp solver{make_config()};
  JointMotionState state = make_state(1);
  state.position[0] = 1.0;
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -1.0;
  limits.upper_position[0] = 1.0;
  limits.position_margin[0] = 0.1;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 1.0),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});

  const QpResult result = solver.solve(task, state, limits);

  ASSERT_TRUE(result.command_available());
  EXPECT_LE(result.joint_velocity[0], kTolerance);
}

TEST(HierarchicalVelocityQp, BrakingEnvelopeActsBeforeTheOneStepPositionBound)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.02;
  const HierarchicalVelocityQp solver{config};
  JointMotionState state = make_state(1);
  state.position[0] = 0.885;
  state.velocity[0] = 0.5;
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -1.0;
  limits.upper_position[0] = 1.0;
  limits.position_margin[0] = 0.1;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 1.0),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});

  const QpResult result = solver.solve(task, state, limits);

  ASSERT_TRUE(result.command_available());
  // Jerk alone would permit 0.502 rad/s and the one-step position bound is
  // still 11.5 rad/s. The lower value therefore comes from the stopping envelope.
  EXPECT_LT(result.joint_velocity[0], 0.502 - 1.0e-5);
  EXPECT_GE(result.joint_velocity[0], 0.498 - kTolerance);
  EXPECT_LT(
    state.position[0] + result.joint_velocity[0] * config.period_sec,
    limits.upper_position[0]);
}

TEST(HierarchicalVelocityQp, VelocityEnvelopePreventsTheObservedRecursiveFailure)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.02;
  const HierarchicalVelocityQp solver{config};
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -100.0;
  limits.upper_position[0] = 100.0;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 2.0),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});

  JointMotionState observed_failure = make_state(1);
  observed_failure.velocity[0] = 0.98745;
  observed_failure.acceleration[0] = 1.800;
  EXPECT_EQ(
    solver.solve(task, observed_failure, limits).status,
    QpStatus::kInfeasible);

  JointMotionState state = make_state(1);
  state.velocity[0] = 0.856;
  state.acceleration[0] = 1.4;
  const double old_one_step_upper = state.velocity[0] +
    std::min(
    limits.max_acceleration[0],
    state.acceleration[0] + limits.max_jerk[0] * config.period_sec) *
    config.period_sec;

  QpResult result = solver.solve(task, state, limits);
  ASSERT_TRUE(result.command_available());
  EXPECT_LE(result.joint_velocity[0], old_one_step_upper + kTolerance);
  EXPECT_LE(
    simulate_discrete_adverse_then_brake(
      state.position[0], state.velocity[0], result.joint_velocity[0], 1.0,
      limits.max_acceleration[0], limits.max_jerk[0], config).maximum_velocity,
    limits.max_velocity[0] + config.solution_feasibility_tolerance);

  for (int tick = 0; tick < 120; ++tick) {
    const double next_acceleration =
      (result.joint_velocity[0] - state.velocity[0]) / config.period_sec;
    state.position[0] += result.joint_velocity[0] * config.period_sec;
    state.velocity = result.joint_velocity;
    state.acceleration[0] = next_acceleration;

    const double minimum_following_acceleration = std::max(
      -limits.max_acceleration[0],
      state.acceleration[0] - limits.max_jerk[0] * config.period_sec);
    EXPECT_LE(
      state.velocity[0] + minimum_following_acceleration * config.period_sec,
      limits.max_velocity[0] + 1.0e-9) << "tick " << tick;
    EXPECT_LE(state.velocity[0], limits.max_velocity[0] + 1.0e-10);

    result = solver.solve(task, state, limits);
    ASSERT_TRUE(result.command_available()) << "tick " << tick;
  }

  JointMotionState mirrored_state = make_state(1);
  mirrored_state.velocity[0] = -0.856;
  mirrored_state.acceleration[0] = -1.4;
  HierarchicalVelocityTask mirrored_task = task;
  mirrored_task.primary_reference[0] = -2.0;
  const QpResult mirrored_result = solver.solve(mirrored_task, mirrored_state, limits);
  ASSERT_TRUE(mirrored_result.command_available());
  EXPECT_GE(mirrored_result.joint_velocity[0], -old_one_step_upper - kTolerance);
  EXPECT_LE(
    simulate_discrete_adverse_then_brake(
      mirrored_state.position[0], mirrored_state.velocity[0],
      mirrored_result.joint_velocity[0], -1.0,
      limits.max_acceleration[0], limits.max_jerk[0], config).maximum_velocity,
    limits.max_velocity[0] + config.solution_feasibility_tolerance);
}

TEST(HierarchicalVelocityQp, ImmediatePublishedBrakeTailKeepsViableHighAccelerationRecoverable)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.0;
  const HierarchicalVelocityQp solver{config};
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -100.0;
  limits.upper_position[0] = 100.0;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;

  // The formerly accepted captured state exceeds the strict discrete velocity
  // envelope by roughly 0.15 microrad/s. Numerical solver tolerance must not
  // turn that excess into extra physical range.
  JointMotionState state = make_state(1);
  state.velocity[0] = 0.977005;
  state.acceleration[0] = 1.059903;
  EXPECT_EQ(solver.brake(state, limits).status, QpStatus::kInfeasible);

  // A neighboring state with real headroom remains recursively feasible.
  state.velocity[0] = 0.9768;
  QpResult result = solver.brake(state, limits);

  ASSERT_TRUE(result.command_available());
  const double next_acceleration =
    (result.joint_velocity[0] - state.velocity[0]) / config.period_sec;
  EXPECT_LE(next_acceleration, state.acceleration[0] + kTolerance);
  EXPECT_GE(
    next_acceleration,
    state.acceleration[0] - limits.max_jerk[0] * config.period_sec - kTolerance);
  EXPECT_LE(result.joint_velocity[0], limits.max_velocity[0] + kTolerance);

  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 2.0),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
  for (int tick = 0; tick < 200; ++tick) {
    state.position[0] += result.joint_velocity[0] * config.period_sec;
    state.acceleration[0] =
      (result.joint_velocity[0] - state.velocity[0]) / config.period_sec;
    state.velocity = result.joint_velocity;
    result = solver.solve(task, state, limits);
    ASSERT_TRUE(result.command_available()) <<
      "tick " << tick << ", velocity=" << state.velocity[0] <<
      ", acceleration=" << state.acceleration[0];
    EXPECT_LE(
      result.joint_velocity[0],
      limits.max_velocity[0] + config.solution_feasibility_tolerance);
  }
}

TEST(HierarchicalVelocityQp, ProductionToleranceNeverExpandsPhysicalJointLimits)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.solution_feasibility_tolerance = 2.0e-5;
  config.solver_absolute_tolerance = 1.0e-5;
  config.solver_relative_tolerance = 1.0e-5;
  const HierarchicalVelocityQp solver{config};

  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -1.0;
  limits.upper_position[0] = 1.0;
  limits.position_margin[0] = 0.1;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;

  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 2.0),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});

  JointMotionState state = make_state(1);
  state.position[0] = 0.55;
  state.velocity[0] = 0.80;
  state.acceleration[0] = 1.0;
  for (int tick = 0; tick < 100; ++tick) {
    const QpResult result = solver.solve(task, state, limits);
    ASSERT_TRUE(result.command_available()) << "tick " << tick;

    const double next_velocity = result.joint_velocity[0];
    const double next_acceleration =
      (next_velocity - state.velocity[0]) / config.period_sec;
    const double next_position = state.position[0] + next_velocity * config.period_sec;
    EXPECT_LE(std::abs(next_velocity), limits.max_velocity[0] + 1.0e-12);
    EXPECT_LE(std::abs(next_acceleration), limits.max_acceleration[0] + 1.0e-10);
    EXPECT_LE(
      std::abs(next_acceleration - state.acceleration[0]),
      limits.max_jerk[0] * config.period_sec + 1.0e-10);
    EXPECT_GE(
      next_position, limits.lower_position[0] - 1.0e-12);
    EXPECT_LE(
      next_position, limits.upper_position[0] + 1.0e-12);

    state.position[0] = next_position;
    state.velocity[0] = next_velocity;
    state.acceleration[0] = next_acceleration;
  }
}

TEST(HierarchicalVelocityQp, DiscreteVelocityEnvelopeRejectsContinuousCounterexampleAndMirror)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.02;
  const HierarchicalVelocityQp solver{config};
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -100.0;
  limits.upper_position[0] = 100.0;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;

  for (const double direction : {1.0, -1.0}) {
    JointMotionState state = make_state(1);
    state.velocity[0] = direction * 0.99995;
    state.acceleration[0] = direction * -0.175;
    const double candidate_velocity = direction * 0.9982;
    const auto unsafe_envelope = simulate_discrete_adverse_then_brake(
      state.position[0], state.velocity[0], candidate_velocity, direction,
      limits.max_acceleration[0], limits.max_jerk[0], config);
    EXPECT_NEAR(unsafe_envelope.maximum_velocity, 1.00095, 1.0e-12);
    EXPECT_GT(unsafe_envelope.maximum_velocity, limits.max_velocity[0]);

    LinearVelocityConstraint fixed_candidate;
    fixed_candidate.coefficients = Eigen::VectorXd::Ones(1);
    fixed_candidate.lower_bound = candidate_velocity;
    fixed_candidate.upper_bound = candidate_velocity;
    const HierarchicalVelocityTask task = make_task(
      joint, Eigen::VectorXd::Constant(1, direction),
      Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
    EXPECT_EQ(
      solver.solve(task, state, limits, {fixed_candidate}).status,
      QpStatus::kInfeasible);

    const QpResult result = solver.solve(task, state, limits);
    ASSERT_TRUE(result.command_available());
    const auto safe_envelope = simulate_discrete_adverse_then_brake(
      state.position[0], state.velocity[0], result.joint_velocity[0], direction,
      limits.max_acceleration[0], limits.max_jerk[0], config);
    EXPECT_LE(
      safe_envelope.maximum_velocity,
      limits.max_velocity[0] + config.solution_feasibility_tolerance);
  }
}

TEST(HierarchicalVelocityQp, ResidualLatencyRoundsUpToCompleteDiscreteTicks)
{
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -100.0;
  limits.upper_position[0] = 100.0;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  JointMotionState state = make_state(1);
  state.velocity[0] = 0.99995;
  state.acceleration[0] = -0.175;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Constant(1, 0.9982),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
  LinearVelocityConstraint fixed_candidate;
  fixed_candidate.coefficients = Eigen::VectorXd::Ones(1);
  fixed_candidate.lower_bound = 0.9982;
  fixed_candidate.upper_bound = 0.9982;

  HierarchicalVelocityQpConfig one_tick_config = make_config();
  one_tick_config.residual_command_latency_sec = one_tick_config.period_sec;
  EXPECT_TRUE(
    HierarchicalVelocityQp{one_tick_config}.solve(
      task, state, limits, {fixed_candidate}).command_available());

  HierarchicalVelocityQpConfig rounded_config = make_config();
  rounded_config.residual_command_latency_sec = rounded_config.period_sec + 1.0e-6;
  EXPECT_EQ(
    HierarchicalVelocityQp{rounded_config}.solve(
      task, state, limits, {fixed_candidate}).status,
    QpStatus::kInfeasible);
}

TEST(HierarchicalVelocityQp, DiscretePositionEnvelopeProtectsBothJointLimits)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.02;
  const HierarchicalVelocityQp solver{config};
  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -1.0;
  limits.upper_position[0] = 1.0;
  limits.position_margin[0] = 0.1;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;

  for (const double direction : {1.0, -1.0}) {
    JointMotionState state = make_state(1);
    state.position[0] = direction * 0.785;
    state.velocity[0] = direction * 0.5;
    const HierarchicalVelocityTask task = make_task(
      joint, Eigen::VectorXd::Constant(1, direction),
      Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});

    const QpResult result = solver.solve(task, state, limits);

    ASSERT_TRUE(result.command_available());
    const auto envelope = simulate_discrete_adverse_then_brake(
      state.position[0], state.velocity[0], result.joint_velocity[0], direction,
      limits.max_acceleration[0], limits.max_jerk[0], config);
    const double signed_safe_limit = direction > 0.0 ?
      limits.upper_position[0] :
      -(limits.lower_position[0]);
    EXPECT_LE(
      envelope.maximum_position,
      signed_safe_limit + config.solution_feasibility_tolerance);
    EXPECT_LE(
      envelope.maximum_velocity,
      limits.max_velocity[0] + config.solution_feasibility_tolerance);
  }
}

TEST(HierarchicalVelocityQp, AcceptedJoint5BoundaryCommandAlwaysHasAnExactBrakeTail)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.solution_feasibility_tolerance = 2.0e-5;
  const HierarchicalVelocityQp solver{config};
  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = config.period_sec;
  tail_config.comparison_tolerance = 2.0 * config.solution_feasibility_tolerance;
  const EmergencyBrakeTailGenerator tail_generator{tail_config};

  JointMotionLimits limits = make_limits(1);
  limits.lower_position[0] = -1.6;
  limits.upper_position[0] = 1.6;
  limits.position_margin[0] = 0.0;
  limits.max_velocity[0] = 1.0;
  limits.max_acceleration[0] = 2.0;
  limits.max_jerk[0] = 20.0;
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;

  for (const double direction : {-1.0, 1.0}) {
    // Captured from the rejected joint5 branch in the people scenario. The
    // positive form is its exact mirror at the opposite corridor boundary.
    JointMotionState captured_branch = make_state(1);
    captured_branch.position[0] = direction * 1.598814320;
    captured_branch.velocity[0] = direction * 0.060322130;
    captured_branch.acceleration[0] = direction * -1.467020631;
    ASSERT_FALSE(tail_generator.generate(captured_branch, limits).command_available());

    JointMotionState previous = make_state(1);
    previous.position[0] = captured_branch.position[0] -
      captured_branch.velocity[0] * config.period_sec;
    previous.velocity[0] = captured_branch.velocity[0] -
      captured_branch.acceleration[0] * config.period_sec;
    previous.acceleration[0] = captured_branch.acceleration[0];
    const HierarchicalVelocityTask task = make_task(
      joint, Eigen::VectorXd::Constant(1, captured_branch.velocity[0]),
      Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
    LinearVelocityConstraint captured_candidate;
    captured_candidate.coefficients = Eigen::VectorXd::Ones(1);
    captured_candidate.lower_bound = captured_branch.velocity[0];
    captured_candidate.upper_bound = captured_branch.velocity[0];

    const QpResult rejected = solver.solve(task, previous, limits, {captured_candidate});
    EXPECT_FALSE(rejected.command_available());

    const QpResult accepted = solver.brake(previous, limits);
    ASSERT_TRUE(accepted.command_available());
    JointMotionState accepted_branch = make_state(1);
    accepted_branch.position[0] =
      previous.position[0] + accepted.joint_velocity[0] * config.period_sec;
    accepted_branch.velocity = accepted.joint_velocity;
    accepted_branch.acceleration[0] =
      (accepted.joint_velocity[0] - previous.velocity[0]) / config.period_sec;
    const EmergencyBrakeTail exact_tail = tail_generator.generate(accepted_branch, limits);
    EXPECT_TRUE(exact_tail.command_available()) << exact_tail.failure_reason;
  }
}

TEST(HierarchicalVelocityQp, SingularityPreferenceCanBeUnreachableWhilePhysicalBrakeIsSafe)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.residual_command_latency_sec = 0.02;
  config.solution_feasibility_tolerance = 2.0e-5;
  config.solver_absolute_tolerance = 1.0e-5;
  config.solver_relative_tolerance = 1.0e-5;
  const HierarchicalVelocityQp solver{config};

  JointMotionState state;
  state.position.resize(6);
  state.position << 0.306490, -0.263427, 0.726567, -0.405537, -0.151429, 0.505958;
  state.velocity.resize(6);
  state.velocity << -0.149844, -0.317944, -0.298388, 0.014177, 0.344805, 0.115399;
  state.acceleration.resize(6);
  state.acceleration << -0.343747, 2.0, 1.485221, 1.200399, -0.784452, -1.723018;

  JointMotionLimits limits = make_limits(6);
  limits.lower_position <<
    -2.0 * M_PI, -2.61799, -0.061087, -2.0 * M_PI, -1.60, -2.0 * M_PI;
  limits.upper_position <<
    2.0 * M_PI, 2.61799, 5.235988, 2.0 * M_PI, 1.60, 2.0 * M_PI;
  limits.position_margin << 0.10, 0.10, 0.10, 0.10, 0.0, 0.10;
  limits.max_velocity.setConstant(1.0);
  limits.max_acceleration.setConstant(2.0);
  limits.max_jerk.setConstant(20.0);

  Eigen::MatrixXd primary = Eigen::MatrixXd::Zero(1, 6);
  primary(0, 0) = 1.0;
  const HierarchicalVelocityTask task = make_task(
    primary, Eigen::VectorXd::Zero(1),
    Eigen::MatrixXd{0, 6}, Eigen::VectorXd{0});

  LinearVelocityConstraint monitor_guard;
  monitor_guard.coefficients = Eigen::VectorXd::Zero(6);
  monitor_guard.coefficients[4] = 1.0;
  monitor_guard.lower_bound = 0.33496048;
  monitor_guard.upper_bound = 0.33896048;

  LinearVelocityConstraint singularity_preference;
  singularity_preference.coefficients.resize(6);
  singularity_preference.coefficients <<
    0.0, 0.017869062418, -0.005622217758, -0.002187156721,
    -0.155465928406, 0.0;
  singularity_preference.lower_bound = -0.024100216762;

  // Captured from the people scenario at condition number 40.733. Even the
  // optimistic jerk-only box can achieve at most this derivative on the next
  // tick, so the relative-degree-one barrier asks for an instantaneous change
  // that the real acceleration/jerk limits forbid.
  Eigen::VectorXd optimistic_velocity = state.velocity;
  for (Eigen::Index index = 0; index < optimistic_velocity.size(); ++index) {
    const double minimum_acceleration = std::max(
      -limits.max_acceleration[index],
      state.acceleration[index] - limits.max_jerk[index] * config.period_sec);
    const double maximum_acceleration = std::min(
      limits.max_acceleration[index],
      state.acceleration[index] + limits.max_jerk[index] * config.period_sec);
    const double selected_acceleration =
      singularity_preference.coefficients[index] >= 0.0 ?
      maximum_acceleration : minimum_acceleration;
    optimistic_velocity[index] += selected_acceleration * config.period_sec;
  }
  const double optimistic_derivative =
    singularity_preference.coefficients.dot(optimistic_velocity);
  EXPECT_NEAR(optimistic_derivative, -0.055846465, 1.0e-6);
  EXPECT_LT(optimistic_derivative, singularity_preference.lower_bound);

  const QpResult blocked = solver.solve(
    task, state, limits, {monitor_guard, singularity_preference});
  EXPECT_EQ(blocked.status, QpStatus::kInfeasible);
  EXPECT_EQ(blocked.primary_status, QpStageStatus::kInfeasible);

  // The same escape direction is always feasible as a secondary objective:
  // it can improve the posture as far as this tick's physical limits allow,
  // but it can never stop screen pointing or safe braking.
  const QpResult primary_only = solver.solve(task, state, limits, {monitor_guard});
  ASSERT_TRUE(primary_only.command_available());
  HierarchicalVelocityTask soft_task = task;
  soft_task.secondary_matrix.resize(1, 6);
  soft_task.secondary_matrix.row(0) =
    singularity_preference.coefficients.normalized().transpose();
  soft_task.secondary_reference = Eigen::VectorXd::Constant(1, 0.20);
  soft_task.secondary_weights = Eigen::VectorXd::Constant(1, 2.0);
  const QpResult guided = solver.solve(soft_task, state, limits, {monitor_guard});
  ASSERT_TRUE(guided.command_available());
  ASSERT_EQ(guided.joint_velocity.size(), 6);
  EXPECT_NEAR(
    guided.achieved_primary_velocity[0], primary_only.achieved_primary_velocity[0],
    config.primary_preservation_tolerance + config.solution_feasibility_tolerance);

  const QpResult physical_brake = solver.brake(state, limits, {monitor_guard});
  ASSERT_TRUE(physical_brake.command_available());
  JointMotionState next = state;
  next.position += physical_brake.joint_velocity * config.period_sec;
  next.velocity = physical_brake.joint_velocity;
  next.acceleration =
    (physical_brake.joint_velocity - state.velocity) / config.period_sec;
  EXPECT_TRUE((next.acceleration.cwiseAbs().array() <=
    limits.max_acceleration.array() + 1.0e-8).all());
  EXPECT_TRUE((((next.acceleration - state.acceleration) / config.period_sec)
    .cwiseAbs().array() <= limits.max_jerk.array() + 1.0e-6).all());

  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = config.period_sec;
  tail_config.comparison_tolerance = 2.0 * config.solution_feasibility_tolerance;
  const EmergencyBrakeTail tail =
    EmergencyBrakeTailGenerator{tail_config}.generate(next, limits);
  EXPECT_TRUE(tail.command_available()) << tail.failure_reason;
}

TEST(HierarchicalVelocityQp, ReportsContradictorySafetyRowsAsInfeasible)
{
  const HierarchicalVelocityQp solver{make_config()};
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  const HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Zero(1),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
  LinearVelocityConstraint positive;
  positive.coefficients = Eigen::VectorXd::Ones(1);
  positive.lower_bound = 0.5;
  LinearVelocityConstraint negative;
  negative.coefficients = Eigen::VectorXd::Ones(1);
  negative.upper_bound = -0.5;

  const QpResult result = solver.solve(
    task, make_state(1), make_limits(1), {positive, negative});

  EXPECT_EQ(result.status, QpStatus::kInfeasible);
  EXPECT_FALSE(result.command_available());
  EXPECT_EQ(result.joint_velocity.size(), 0);
}

TEST(HierarchicalVelocityQp, RejectsNonFiniteAndDimensionallyInvalidInputs)
{
  const HierarchicalVelocityQp solver{make_config()};
  Eigen::Matrix<double, 1, 1> joint;
  joint << 1.0;
  HierarchicalVelocityTask task = make_task(
    joint, Eigen::VectorXd::Zero(1),
    Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
  JointMotionState state = make_state(1);

  state.velocity[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    solver.solve(task, state, make_limits(1)).status,
    QpStatus::kInvalidInput);

  state = make_state(1);
  task.primary_reference = Eigen::VectorXd::Zero(2);
  EXPECT_EQ(
    solver.solve(task, state, make_limits(1)).status,
    QpStatus::kInvalidInput);
}

TEST(HierarchicalVelocityQp, RejectsInvalidConfigurationAtConstruction)
{
  HierarchicalVelocityQpConfig config = make_config();
  config.period_sec = 0.0;
  EXPECT_THROW(HierarchicalVelocityQp{config}, std::invalid_argument);

  config = make_config();
  config.primary_preservation_tolerance =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(HierarchicalVelocityQp{config}, std::invalid_argument);

  config = make_config();
  config.solver_max_iterations = 0;
  EXPECT_THROW(HierarchicalVelocityQp{config}, std::invalid_argument);

  config = make_config();
  config.residual_command_latency_sec = -0.01;
  EXPECT_THROW(HierarchicalVelocityQp{config}, std::invalid_argument);
}

TEST(HierarchicalVelocityQp, RandomCommandsRespectAllDiscreteJointLimits)
{
  const HierarchicalVelocityQpConfig config = make_config();
  const HierarchicalVelocityQp solver{config};
  constexpr Eigen::Index kJointCount = 6;
  std::mt19937 generator{42U};
  std::uniform_real_distribution<double> position_distribution{-0.5, 0.5};
  std::uniform_real_distribution<double> velocity_distribution{-0.8, 0.8};
  std::uniform_real_distribution<double> acceleration_distribution{-1.5, 1.5};
  std::uniform_real_distribution<double> task_distribution{-1.0, 1.0};

  for (int sample = 0; sample < 50; ++sample) {
    JointMotionState state = make_state(kJointCount);
    JointMotionLimits limits = make_limits(kJointCount);
    limits.lower_position.setConstant(-2.0);
    limits.upper_position.setConstant(2.0);
    limits.position_margin.setConstant(0.1);
    limits.max_velocity.setConstant(1.0);
    limits.max_acceleration.setConstant(2.0);
    limits.max_jerk.setConstant(20.0);
    for (Eigen::Index joint = 0; joint < kJointCount; ++joint) {
      state.position[joint] = position_distribution(generator);
      state.velocity[joint] = velocity_distribution(generator);
      state.acceleration[joint] = acceleration_distribution(generator);
    }

    Eigen::MatrixXd primary(2, kJointCount);
    Eigen::MatrixXd secondary(4, kJointCount);
    Eigen::Vector2d primary_reference;
    Eigen::Vector4d secondary_reference;
    for (Eigen::Index row = 0; row < primary.rows(); ++row) {
      primary_reference[row] = task_distribution(generator);
      for (Eigen::Index column = 0; column < primary.cols(); ++column) {
        primary(row, column) = task_distribution(generator);
      }
    }
    for (Eigen::Index row = 0; row < secondary.rows(); ++row) {
      secondary_reference[row] = task_distribution(generator);
      for (Eigen::Index column = 0; column < secondary.cols(); ++column) {
        secondary(row, column) = task_distribution(generator);
      }
    }
    const HierarchicalVelocityTask task = make_task(
      primary, primary_reference, secondary, secondary_reference);

    const QpResult result = solver.solve(task, state, limits);

    ASSERT_TRUE(result.command_available()) << "random sample " << sample;
    for (Eigen::Index joint = 0; joint < kJointCount; ++joint) {
      const double acceleration =
        (result.joint_velocity[joint] - state.velocity[joint]) / config.period_sec;
      const double jerk =
        (acceleration - state.acceleration[joint]) / config.period_sec;
      const double next_position =
        state.position[joint] + result.joint_velocity[joint] * config.period_sec;
      EXPECT_LE(std::abs(result.joint_velocity[joint]),
        limits.max_velocity[joint] + 1.0e-10);
      EXPECT_LE(std::abs(acceleration), limits.max_acceleration[joint] + 1.0e-8);
      EXPECT_LE(std::abs(jerk), limits.max_jerk[joint] + 1.0e-6);
      EXPECT_GE(next_position,
        limits.lower_position[joint] - 1.0e-10);
      EXPECT_LE(next_position,
        limits.upper_position[joint] + 1.0e-10);
    }
  }
}


TEST(HierarchicalVelocityQp, EscapesPreferredMarginGraduallyWithAValidStopAtEveryStep)
{
  const auto config = make_config();
  const HierarchicalVelocityQp solver{config};
  const EmergencyBrakeTailGenerator brake;
  auto limits = make_limits(1);
  limits.lower_position[0] = -1.0;
  limits.upper_position[0] = 1.0;
  limits.position_margin[0] = 0.1;
  limits.max_velocity[0] = 0.25;
  limits.max_acceleration[0] = 0.5;
  limits.max_jerk[0] = 5.0;
  for (const double direction : {-1.0, 1.0}) {
    auto state = make_state(1);
    state.position[0] = direction * 0.999;
    const auto task = make_task(
      Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Constant(1, -direction * 0.2),
      Eigen::MatrixXd{0, 1}, Eigen::VectorXd{0});
    for (int step = 0; step < 250; ++step) {
      const auto command = solver.solve(task, state, limits);
      ASSERT_TRUE(command.command_available()) << "step " << step;
      const auto acceleration = ((command.joint_velocity - state.velocity) /
        config.period_sec).eval();
      EXPECT_LE(std::abs(acceleration[0] - state.acceleration[0]),
        limits.max_jerk[0] * config.period_sec + kTolerance);
      state.acceleration = acceleration;
      state.velocity = command.joint_velocity;
      state.position += state.velocity * config.period_sec;
      EXPECT_LE(std::abs(state.position[0]), 1.0 + kTolerance);
      ASSERT_TRUE(brake.generate(state, limits).command_available());
    }
    EXPECT_LT(std::abs(state.position[0]), 0.9);
  }
}

}  // namespace
}  // namespace face_tracking_arm::control
