// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/qp_failure_snapshot_io.hpp"

#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{

constexpr char kMagic[] = "face_tracking_arm_qp_snapshot_v2";
constexpr std::size_t kMaxJoints = kQpFailureSnapshotMaximumJoints;
constexpr std::size_t kMaxRows = kQpFailureSnapshotMaximumRows;
constexpr std::size_t kMaxPNonzeros = 21U;
constexpr std::size_t kMaxANonzeros = kQpFailureSnapshotMaximumANonzeros;

template<typename Vector>
void writeVector(std::ostream & out, const char * label, const Vector & values)
{
  out << label << ' ' << values.size();
  for (const auto value : values) {
    out << ' ' << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  }
  out << '\n';
}

template<typename T>
bool readValue(std::istream & in, T & value)
{
  std::string token;
  if (!(in >> token)) {
    return false;
  }
  try {
    if constexpr (std::is_integral_v<T>) {
      std::size_t consumed = 0U;
      const auto parsed = std::stoll(token, &consumed);
      if (consumed != token.size() || parsed < std::numeric_limits<T>::lowest() ||
        parsed > std::numeric_limits<T>::max())
      {
        return false;
      }
      value = static_cast<T>(parsed);
    } else {
      std::size_t consumed = 0U;
      const double parsed = std::stod(token, &consumed);
      if (consumed != token.size()) {
        return false;
      }
      value = static_cast<T>(parsed);
    }
  } catch (...) {
    return false;
  }
  return true;
}

template<typename Vector>
bool readVector(
  std::istream & in, const char * expected_label, Vector & values,
  const std::size_t maximum_size)
{
  std::string label;
  std::size_t size = 0U;
  if (!(in >> label >> size) || label != expected_label || size > maximum_size) {
    return false;
  }
  values.resize(size);
  for (auto & value : values) {
    if (!readValue(in, value)) {
      return false;
    }
  }
  return true;
}

bool readLabel(std::istream & in, const char * expected)
{
  std::string label;
  return static_cast<bool>(in >> label) && label == expected;
}

}  // namespace

bool writeQpFailureSnapshot(std::ostream & output, const QpFailureSnapshot & snapshot)
{
  if (snapshot.variable_count <= 0 || snapshot.variable_count > static_cast<int>(kMaxJoints) ||
    snapshot.constraint_count <= 0 || snapshot.constraint_count > static_cast<int>(kMaxRows) ||
    snapshot.p_values.size() > kMaxPNonzeros || snapshot.a_values.size() > kMaxANonzeros)
  {
    return false;
  }
  output << kMagic << '\n';
  output << "stage " << static_cast<int>(snapshot.stage) << '\n';
  output << "classified " << static_cast<int>(snapshot.classified_status) << '\n';
  output << "codes " << snapshot.setup_return_code << ' ' << snapshot.solve_return_code << ' '
         << snapshot.raw_status_value << '\n';
  output << "dimensions " << snapshot.variable_count << ' ' << snapshot.constraint_count << '\n';
  if (snapshot.osqp_version.empty() || snapshot.osqp_version.find_first_of(" \t\r\n") !=
    std::string::npos || snapshot.c_int_size == 0U || snapshot.c_float_size == 0U)
  {
    return false;
  }
  output << "build " << snapshot.osqp_version << ' ' << snapshot.c_int_size << ' '
         << snapshot.c_float_size << '\n';
  output << "settings " << std::setprecision(std::numeric_limits<double>::max_digits10)
         << snapshot.rho << ' ' << snapshot.sigma << ' ' << snapshot.scaling << ' '
         << snapshot.adaptive_rho << ' ' << snapshot.adaptive_rho_interval << ' '
         << snapshot.adaptive_rho_tolerance << ' ' << snapshot.adaptive_rho_fraction << ' '
         << snapshot.max_iter << ' ' << snapshot.eps_abs << ' ' << snapshot.eps_rel << ' '
         << snapshot.eps_prim_inf << ' ' << snapshot.eps_dual_inf << ' ' << snapshot.alpha << ' '
         << snapshot.linsys_solver << ' ' << snapshot.delta << ' ' << snapshot.polish << ' '
         << snapshot.polish_refine_iter << ' ' << snapshot.verbose << ' '
         << snapshot.scaled_termination << ' ' << snapshot.check_termination << ' '
         << snapshot.warm_start << ' ' << snapshot.time_limit_sec << ' '
         << snapshot.observed_solve_time_sec << ' ' << snapshot.solver_budget_sec << '\n';
  writeVector(output, "Pp", snapshot.p_column_pointers);
  writeVector(output, "Pi", snapshot.p_row_indices);
  writeVector(output, "Px", snapshot.p_values);
  writeVector(output, "Ap", snapshot.a_column_pointers);
  writeVector(output, "Ai", snapshot.a_row_indices);
  writeVector(output, "Ax", snapshot.a_values);
  writeVector(output, "q", snapshot.q);
  writeVector(output, "l", snapshot.lower);
  writeVector(output, "u", snapshot.upper);
  writeVector(output, "warm", snapshot.centered_warm_start);
  writeVector(output, "shift", snapshot.variable_shift);
  return static_cast<bool>(output);
}

bool readQpFailureSnapshot(std::istream & input, QpFailureSnapshot & snapshot)
{
  std::string magic;
  int stage = 0;
  int classified = 0;
  int variable_count = 0;
  int constraint_count = 0;
  std::string osqp_version;
  if (!(input >> magic) || magic != kMagic || !readLabel(input, "stage") ||
    !readValue(input, stage) || !readLabel(input, "classified") ||
    !readValue(input, classified) || !readLabel(input, "codes") ||
    !readValue(input, snapshot.setup_return_code) ||
    !readValue(input, snapshot.solve_return_code) || !readValue(input, snapshot.raw_status_value) ||
    !readLabel(input, "dimensions") || !readValue(input, variable_count) ||
    !readValue(input, constraint_count) || variable_count <= 0 ||
    variable_count > static_cast<int>(kMaxJoints) || constraint_count <= 0 ||
    constraint_count > static_cast<int>(kMaxRows) || stage < 0 || stage > 2 ||
    classified < 0 || classified > static_cast<int>(QpStageStatus::kSolverError) ||
    !readLabel(input, "build") || !(input >> osqp_version) ||
    !readValue(input, snapshot.c_int_size) || !readValue(input, snapshot.c_float_size) ||
    !readLabel(input, "settings"))
  {
    return false;
  }
  snapshot.stage = static_cast<QpSolveStage>(stage);
  snapshot.classified_status = static_cast<QpStageStatus>(classified);
  snapshot.variable_count = variable_count;
  snapshot.constraint_count = constraint_count;
  snapshot.osqp_version = std::move(osqp_version);
  if (!readValue(input, snapshot.rho) || !readValue(input, snapshot.sigma) ||
    !readValue(input, snapshot.scaling) || !readValue(input, snapshot.adaptive_rho) ||
    !readValue(input, snapshot.adaptive_rho_interval) ||
    !readValue(input, snapshot.adaptive_rho_tolerance) ||
    !readValue(input, snapshot.adaptive_rho_fraction) || !readValue(input, snapshot.max_iter) ||
    !readValue(input, snapshot.eps_abs) || !readValue(input, snapshot.eps_rel) ||
    !readValue(input, snapshot.eps_prim_inf) || !readValue(input, snapshot.eps_dual_inf) ||
    !readValue(input, snapshot.alpha) || !readValue(input, snapshot.linsys_solver) ||
    !readValue(input, snapshot.delta) || !readValue(input, snapshot.polish) ||
    !readValue(input, snapshot.polish_refine_iter) || !readValue(input, snapshot.verbose) ||
    !readValue(input, snapshot.scaled_termination) ||
    !readValue(input, snapshot.check_termination) || !readValue(input, snapshot.warm_start) ||
    !readValue(input, snapshot.time_limit_sec) ||
    !readValue(input, snapshot.observed_solve_time_sec) ||
    !readValue(input, snapshot.solver_budget_sec))
  {
    return false;
  }
  if (!readVector(input, "Pp", snapshot.p_column_pointers, kMaxJoints + 1U) ||
    !readVector(input, "Pi", snapshot.p_row_indices, kMaxPNonzeros) ||
    !readVector(input, "Px", snapshot.p_values, kMaxPNonzeros) ||
    !readVector(input, "Ap", snapshot.a_column_pointers, kMaxJoints + 1U) ||
    !readVector(input, "Ai", snapshot.a_row_indices, kMaxANonzeros) ||
    !readVector(input, "Ax", snapshot.a_values, kMaxANonzeros) ||
    !readVector(input, "q", snapshot.q, kMaxJoints) ||
    !readVector(input, "l", snapshot.lower, kMaxRows) ||
    !readVector(input, "u", snapshot.upper, kMaxRows) ||
    !readVector(input, "warm", snapshot.centered_warm_start, kMaxJoints) ||
    !readVector(input, "shift", snapshot.variable_shift, kMaxJoints))
  {
    return false;
  }
  return snapshot.p_column_pointers.size() == static_cast<std::size_t>(variable_count + 1) &&
         snapshot.a_column_pointers.size() == static_cast<std::size_t>(variable_count + 1) &&
         snapshot.q.size() == static_cast<std::size_t>(variable_count) &&
         snapshot.lower.size() == static_cast<std::size_t>(constraint_count) &&
         snapshot.upper.size() == static_cast<std::size_t>(constraint_count) &&
         snapshot.centered_warm_start.size() == static_cast<std::size_t>(variable_count) &&
         snapshot.variable_shift.size() == static_cast<std::size_t>(variable_count) &&
         snapshot.p_row_indices.size() == snapshot.p_values.size() &&
         snapshot.a_row_indices.size() == snapshot.a_values.size();
}

}  // namespace face_tracking_arm::control
