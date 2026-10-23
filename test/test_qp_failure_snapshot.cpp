// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <osqp.h>

#include <limits>
#include <sstream>
#include <vector>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"
#include "face_tracking_arm/qp_failure_snapshot_io.hpp"
#include "face_tracking_arm/qp_failure_snapshot_mailbox.hpp"

namespace face_tracking_arm::control
{
namespace
{

JointMotionState state()
{
  JointMotionState value;
  value.position = Eigen::VectorXd::Zero(1);
  value.velocity = Eigen::VectorXd::Zero(1);
  value.acceleration = Eigen::VectorXd::Zero(1);
  return value;
}

JointMotionLimits limits()
{
  JointMotionLimits value;
  value.lower_position = Eigen::VectorXd::Constant(1, -2.0);
  value.upper_position = Eigen::VectorXd::Constant(1, 2.0);
  value.position_margin = Eigen::VectorXd::Zero(1);
  value.max_velocity = Eigen::VectorXd::Constant(1, 1.0);
  value.max_acceleration = Eigen::VectorXd::Constant(1, 2.0);
  value.max_jerk = Eigen::VectorXd::Constant(1, 10.0);
  value.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-8}};
  return value;
}

HierarchicalVelocityTask task(const double target)
{
  HierarchicalVelocityTask value;
  value.primary_matrix = Eigen::MatrixXd::Ones(1, 1);
  value.primary_reference = Eigen::VectorXd::Constant(1, target);
  value.primary_weights = Eigen::VectorXd::Ones(1);
  value.secondary_matrix.resize(0, 1);
  value.secondary_reference.resize(0);
  value.secondary_weights.resize(0);
  return value;
}

TEST(QpFailureSnapshot, CapturesAndReplaysInfeasibleMatrixAndCenteredCoordinates)
{
  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  std::vector<LinearVelocityConstraint> contradictory(2);
  contradictory[0].coefficients = Eigen::VectorXd::Ones(1);
  contradictory[0].lower_bound = 0.5;
  contradictory[1].coefficients = Eigen::VectorXd::Ones(1);
  contradictory[1].upper_bound = -0.5;

  const auto result = solver.solve(task(0.2), state(), limits(), contradictory);
  EXPECT_EQ(result.primary_status, QpStageStatus::kInfeasible);
  ASSERT_TRUE(result.primary_failure_snapshot.has_value());
  const auto & snapshot = *result.primary_failure_snapshot;
  EXPECT_EQ(snapshot.stage, QpSolveStage::kPrimary);
  EXPECT_EQ(result.primary_snapshot_disposition, QpFailureSnapshotDisposition::kCaptured);
  EXPECT_EQ(snapshot.osqp_version, OSQP_VERSION);
  EXPECT_EQ(snapshot.c_int_size, sizeof(c_int));
  EXPECT_EQ(snapshot.c_float_size, sizeof(c_float));
  EXPECT_EQ(snapshot.variable_count, 1);
  EXPECT_GT(snapshot.constraint_count, 0);
  EXPECT_EQ(snapshot.p_column_pointers.size(), 2U);
  EXPECT_EQ(snapshot.a_column_pointers.size(), 2U);
  ASSERT_EQ(snapshot.centered_warm_start.size(), 1U);
  ASSERT_EQ(snapshot.variable_shift.size(), 1U);
  EXPECT_EQ(snapshot.max_iter, config.solver_max_iterations);
  EXPECT_DOUBLE_EQ(snapshot.eps_abs, config.solver_absolute_tolerance);
  EXPECT_DOUBLE_EQ(snapshot.solver_budget_sec, config.solver_time_limit_sec);

  const auto replay = replayQpFailureSnapshot(snapshot);
  ASSERT_EQ(replay.setup_return_code, 0);
  EXPECT_EQ(replay.status, QpStageStatus::kInfeasible);
  EXPECT_EQ(replay.raw_status_value, snapshot.raw_status_value);
  std::stringstream encoded;
  ASSERT_TRUE(writeQpFailureSnapshot(encoded, snapshot));
  QpFailureSnapshot decoded;
  ASSERT_TRUE(readQpFailureSnapshot(encoded, decoded));
  EXPECT_EQ(decoded.p_column_pointers, snapshot.p_column_pointers);
  EXPECT_EQ(decoded.a_row_indices, snapshot.a_row_indices);
  EXPECT_EQ(decoded.q, snapshot.q);
  EXPECT_EQ(decoded.lower, snapshot.lower);
  EXPECT_EQ(decoded.centered_warm_start, snapshot.centered_warm_start);
  EXPECT_EQ(decoded.osqp_version, snapshot.osqp_version);
  EXPECT_EQ(decoded.c_int_size, snapshot.c_int_size);
  EXPECT_EQ(decoded.c_float_size, snapshot.c_float_size);
  EXPECT_DOUBLE_EQ(decoded.adaptive_rho_tolerance, snapshot.adaptive_rho_tolerance);
  EXPECT_EQ(replayQpFailureSnapshot(decoded).status, QpStageStatus::kInfeasible);

  auto mismatched_build = decoded;
  mismatched_build.osqp_version += "-other";
  const auto incompatible = replayQpFailureSnapshot(mismatched_build);
  EXPECT_FALSE(incompatible.compatible_solver_build);
  EXPECT_EQ(incompatible.setup_return_code, -2);
}

TEST(QpFailureSnapshot, CapturesMaximumIterationFailure)
{
  HierarchicalVelocityQpConfig config;
  config.solver_max_iterations = 1;
  config.solver_absolute_tolerance = 1.0e-12;
  config.solver_relative_tolerance = 1.0e-12;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  const auto result = solver.solve(task(0.731), state(), limits());
  EXPECT_EQ(result.primary_status, QpStageStatus::kMaxIterations);
  EXPECT_EQ(result.primary_snapshot_disposition, QpFailureSnapshotDisposition::kCaptured);
  ASSERT_TRUE(result.primary_failure_snapshot.has_value());
  EXPECT_EQ(result.primary_failure_snapshot->max_iter, 1);
  const auto replay = replayQpFailureSnapshot(*result.primary_failure_snapshot);
  ASSERT_EQ(replay.setup_return_code, 0);
  EXPECT_EQ(replay.status, QpStageStatus::kMaxIterations);
}

TEST(QpFailureSnapshot, MapsSolverStatusesIncludingOptionalTimeLimit)
{
  EXPECT_EQ(classifyQpSolverStatus(OSQP_SOLVED), QpStageStatus::kSolved);
  EXPECT_EQ(classifyQpSolverStatus(OSQP_MAX_ITER_REACHED), QpStageStatus::kMaxIterations);
  EXPECT_EQ(classifyQpSolverStatus(OSQP_PRIMAL_INFEASIBLE), QpStageStatus::kInfeasible);
#ifdef OSQP_TIME_LIMIT_REACHED
  EXPECT_EQ(classifyQpSolverStatus(OSQP_TIME_LIMIT_REACHED), QpStageStatus::kTimeLimit);
#endif
}

TEST(QpFailureSnapshot, SuccessfulCommandDoesNotAllocateFailureSnapshot)
{
  HierarchicalVelocityQpConfig config;
  HierarchicalVelocityQp solver(config);
  const auto result = solver.solve(task(0.1), state(), limits());
  ASSERT_TRUE(result.command_available());
  EXPECT_FALSE(result.primary_failure_snapshot.has_value());
  EXPECT_FALSE(result.secondary_failure_snapshot.has_value());
  EXPECT_EQ(result.primary_snapshot_disposition, QpFailureSnapshotDisposition::kNotNeeded);
}

TEST(QpFailureSnapshot, ReportsShapeLimitInsteadOfSilentlyDroppingFailure)
{
  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  std::vector<LinearVelocityConstraint> contradictory(
    kQpFailureSnapshotMaximumRows);
  for (auto & constraint : contradictory) {
    constraint.coefficients = Eigen::VectorXd::Ones(1);
  }
  contradictory[0].lower_bound = 0.5;
  contradictory[1].upper_bound = -0.5;

  const auto result = solver.solve(task(0.2), state(), limits(), contradictory);
  EXPECT_EQ(result.primary_status, QpStageStatus::kInfeasible);
  EXPECT_FALSE(result.primary_failure_snapshot.has_value());
  EXPECT_EQ(result.primary_snapshot_disposition,
    QpFailureSnapshotDisposition::kShapeLimitExceeded);
}

TEST(QpFailureSnapshot, CapturesAtMaximumBoundedAssembledShape)
{
  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  // One joint row is always assembled, leaving maxRows-1 external rows.
  std::vector<LinearVelocityConstraint> contradictory(
    kQpFailureSnapshotMaximumRows - 1U);
  for (auto & constraint : contradictory) {
    constraint.coefficients = Eigen::VectorXd::Ones(1);
  }
  contradictory[0].lower_bound = 0.5;
  contradictory[1].upper_bound = -0.5;

  const auto result = solver.solve(task(0.2), state(), limits(), contradictory);
  ASSERT_EQ(result.primary_status, QpStageStatus::kInfeasible);
  ASSERT_EQ(result.primary_snapshot_disposition, QpFailureSnapshotDisposition::kCaptured);
  ASSERT_TRUE(result.primary_failure_snapshot.has_value());
  EXPECT_EQ(result.primary_failure_snapshot->constraint_count,
    static_cast<int>(kQpFailureSnapshotMaximumRows));
  std::stringstream encoded;
  ASSERT_TRUE(writeQpFailureSnapshot(encoded, *result.primary_failure_snapshot));
  QpFailureSnapshot decoded;
  ASSERT_TRUE(readQpFailureSnapshot(encoded, decoded));
  EXPECT_EQ(decoded.constraint_count, static_cast<int>(kQpFailureSnapshotMaximumRows));
}

TEST(QpFailureSnapshot, RejectsTruncatedAndOversizedFiles)
{
  QpFailureSnapshot snapshot;
  std::stringstream truncated("face_tracking_arm_qp_snapshot_v2\nstage 0\n");
  EXPECT_FALSE(readQpFailureSnapshot(truncated, snapshot));

  snapshot.variable_count = 7;
  snapshot.constraint_count = 1;
  std::stringstream encoded;
  EXPECT_FALSE(writeQpFailureSnapshot(encoded, snapshot));
}

TEST(QpFailureSnapshotMailbox, HandsOffOneBoundedFailureAndCountsDrops)
{
  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  std::vector<LinearVelocityConstraint> contradictory(2);
  contradictory[0].coefficients = Eigen::VectorXd::Ones(1);
  contradictory[0].lower_bound = 0.5;
  contradictory[1].coefficients = Eigen::VectorXd::Ones(1);
  contradictory[1].upper_bound = -0.5;
  QpResult result = solver.solve(task(0.2), state(), limits(), contradictory);

  QpFailureSnapshotMailbox mailbox;
  ASSERT_TRUE(mailbox.trySubmit(std::move(result)));
  QpResult second = solver.solve(task(0.2), state(), limits(), contradictory);
  EXPECT_FALSE(mailbox.trySubmit(std::move(second)));
  EXPECT_EQ(mailbox.droppedCount(), 1U);
  const auto event = mailbox.tryTake();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->event_id, 1U);
  EXPECT_EQ(event->primary_disposition, QpFailureSnapshotDisposition::kCaptured);
  ASSERT_TRUE(event->primary.has_value());
  EXPECT_EQ(event->primary->variable_count, 1);
  EXPECT_FALSE(mailbox.tryTake().has_value());
}

TEST(QpFailureSnapshotMailbox, RecordsShapeLimitWithoutMatrix)
{
  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  std::vector<LinearVelocityConstraint> contradictory(
    kQpFailureSnapshotMaximumRows);
  for (auto & constraint : contradictory) {
    constraint.coefficients = Eigen::VectorXd::Ones(1);
  }
  contradictory[0].lower_bound = 0.5;
  contradictory[1].upper_bound = -0.5;
  QpResult result = solver.solve(task(0.2), state(), limits(), contradictory);

  QpFailureSnapshotMailbox mailbox;
  ASSERT_TRUE(mailbox.trySubmit(std::move(result)));
  const auto event = mailbox.tryTake();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->primary_disposition,
    QpFailureSnapshotDisposition::kShapeLimitExceeded);
  EXPECT_FALSE(event->primary.has_value());
}

}  // namespace
}  // namespace face_tracking_arm::control
