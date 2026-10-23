// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <face_tracking_arm/qp_failure_snapshot_io.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::cerr << "usage: qp_snapshot_replay SNAPSHOT_FILE\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  face_tracking_arm::control::QpFailureSnapshot snapshot;
  if (!input || !face_tracking_arm::control::readQpFailureSnapshot(input, snapshot)) {
    std::cerr << "invalid or unsupported QP snapshot\n";
    return 2;
  }
  const auto replay = face_tracking_arm::control::replayQpFailureSnapshot(snapshot);
  if (!replay.compatible_solver_build) {
    std::cerr << "snapshot OSQP build mismatch: recorded version=" << snapshot.osqp_version
              << " c_int=" << snapshot.c_int_size << " c_float=" << snapshot.c_float_size
              << "; current build is incompatible\n";
    return 3;
  }
  std::cout << "stage=" << static_cast<int>(snapshot.stage)
            << " original_status=" << static_cast<int>(snapshot.classified_status)
            << " replay_status=" << static_cast<int>(replay.status)
            << " raw_status=" << replay.raw_status_value
            << " iterations=" << replay.iterations
            << " solve_time_sec=" << replay.solve_time_sec
            << " primal_residual=" << replay.primal_residual
            << " dual_residual=" << replay.dual_residual
            << " original_solve_time_sec=" << snapshot.observed_solve_time_sec
            << " recorded_budget_sec=" << snapshot.solver_budget_sec << '\n';
  return replay.setup_return_code == 0 && replay.solve_return_code == 0 ? 0 : 1;
}
