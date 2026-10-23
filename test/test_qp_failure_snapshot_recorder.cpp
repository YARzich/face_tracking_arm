// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <Eigen/Core>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "face_tracking_arm/qp_failure_snapshot_recorder.hpp"

using face_tracking_arm::control::HierarchicalVelocityQp;
using face_tracking_arm::control::HierarchicalVelocityQpConfig;
using face_tracking_arm::control::HierarchicalVelocityTask;
using face_tracking_arm::control::JointMotionLimits;
using face_tracking_arm::control::JointMotionState;
using face_tracking_arm::control::LinearVelocityConstraint;
using face_tracking_arm::control::QpFailureSnapshotEvent;
using face_tracking_arm::control::QpFailureSnapshotDisposition;
using face_tracking_arm::control::NumericalPolicy;
using face_tracking_arm::control::Radians;
using face_tracking_arm::control::RadiansPerSecond;
using face_tracking_arm::control::RadiansPerSecondSquared;
using face_tracking_arm::control::RadiansPerSecondCubed;
using face_tracking_arm::control::SolverResidual;
using face_tracking_arm::control::writeQpFailureSnapshotEvent;

namespace
{

int fail(const std::filesystem::path & directory, const int code)
{
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  return code;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    return 2;
  }
  const auto directory = std::filesystem::temp_directory_path() /
    ("face_tracking_arm_qp_recorder_" + std::to_string(getpid()));
  std::filesystem::create_directories(directory);

  HierarchicalVelocityQpConfig config;
  config.solver_time_limit_sec = 1.0;
  HierarchicalVelocityQp solver(config);
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(1);
  state.velocity = Eigen::VectorXd::Zero(1);
  state.acceleration = Eigen::VectorXd::Zero(1);
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(1, -2.0);
  limits.upper_position = Eigen::VectorXd::Constant(1, 2.0);
  limits.position_margin = Eigen::VectorXd::Zero(1);
  limits.max_velocity = Eigen::VectorXd::Constant(1, 1.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(1, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(1, 10.0);
  limits.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-8}};
  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Ones(1, 1);
  task.primary_reference = Eigen::VectorXd::Constant(1, 0.2);
  task.primary_weights = Eigen::VectorXd::Ones(1);
  task.secondary_matrix.resize(0, 1);
  task.secondary_reference.resize(0);
  task.secondary_weights.resize(0);
  std::vector<LinearVelocityConstraint> constraints(2);
  constraints[0].coefficients = Eigen::VectorXd::Ones(1);
  constraints[0].lower_bound = 0.5;
  constraints[1].coefficients = Eigen::VectorXd::Ones(1);
  constraints[1].upper_bound = -0.5;
  const auto result = solver.solve(task, state, limits, constraints);
  if (result.primary_snapshot_disposition != QpFailureSnapshotDisposition::kCaptured ||
    !result.primary_failure_snapshot.has_value())
  {
    return fail(directory, 3);
  }

  QpFailureSnapshotEvent event;
  event.event_id = 1U;
  event.ros_stamp_ns = 101U;
  event.steady_stamp_ns = 201U;
  event.hardware_epoch = 301U;
  event.hardware_read_sequence = 401U;
  event.active_publication_id = 501U;
  event.latest_publication_id = 601U;
  event.primary_disposition = result.primary_snapshot_disposition;
  event.primary = result.primary_failure_snapshot;
  if (!writeQpFailureSnapshotEvent(directory, event).index_ok) {
    return fail(directory, 4);
  }
  event.event_id = 2U;
  event.ros_stamp_ns = 102U;
  event.hardware_epoch = 302U;
  event.hardware_read_sequence = 402U;
  event.active_publication_id = 502U;
  event.latest_publication_id = 602U;
  const auto write_result = writeQpFailureSnapshotEvent(directory, event);
  if (!write_result.index_ok || !write_result.snapshots_ok ||
    !std::filesystem::exists(write_result.primary_path))
  {
    return fail(directory, 5);
  }

  std::ifstream index(directory / "events.tsv");
  std::string contents((std::istreambuf_iterator<char>(index)),
    std::istreambuf_iterator<char>());
  if (contents.find("2\t102\t201\t302\t402\t502\t602\t") == std::string::npos) {
    return fail(directory, 6);
  }

  const pid_t child = fork();
  if (child == 0) {
    execl(argv[1], argv[1], write_result.primary_path.c_str(), nullptr);
    _exit(127);
  }
  if (child < 0) {
    return fail(directory, 7);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    return fail(directory, 8);
  }
  return fail(directory, 0);
}
