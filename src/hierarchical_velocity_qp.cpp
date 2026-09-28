// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

#include <osqp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

enum class InputCheck
{
  kValid,
  kInvalid,
  kInfeasible,
};

struct ConstraintMatrix
{
  Eigen::MatrixXd matrix;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

struct SparseArrays
{
  std::vector<c_int> column_pointers;
  std::vector<c_int> row_indices;
  std::vector<c_float> values;
};

struct StageResult
{
  QpStageStatus status{QpStageStatus::kSolverError};
  Eigen::VectorXd velocity;
  int iterations{0};
};

struct WorkspaceDeleter
{
  void operator()(OSQPWorkspace * workspace) const noexcept
  {
    if (workspace != nullptr) {
      (void)osqp_cleanup(workspace);
    }
  }
};

using WorkspacePtr = std::unique_ptr<OSQPWorkspace, WorkspaceDeleter>;

bool is_finite(const Eigen::VectorXd & vector)
{
  return vector.array().isFinite().all();
}

bool is_finite(const Eigen::MatrixXd & matrix)
{
  return matrix.array().isFinite().all();
}

bool valid_config(const HierarchicalVelocityQpConfig & config)
{
  return std::isfinite(config.period_sec) && config.period_sec > 0.0 &&
         std::isfinite(config.residual_command_latency_sec) &&
         config.residual_command_latency_sec >= 0.0 &&
         std::isfinite(config.velocity_regularization) &&
         config.velocity_regularization > 0.0 &&
         std::isfinite(config.continuity_weight) && config.continuity_weight >= 0.0 &&
         std::isfinite(config.primary_preservation_tolerance) &&
         config.primary_preservation_tolerance > 0.0 &&
         std::isfinite(config.solution_feasibility_tolerance) &&
         config.solution_feasibility_tolerance > 0.0 &&
         std::isfinite(config.solver_absolute_tolerance) &&
         config.solver_absolute_tolerance > 0.0 &&
         std::isfinite(config.solver_relative_tolerance) &&
         config.solver_relative_tolerance > 0.0 &&
         std::isfinite(config.solver_time_limit_sec) &&
         config.solver_time_limit_sec > 0.0 &&
         config.solver_max_iterations > 0;
}

struct DiscreteStoppingEnvelope
{
  double maximum_position_excursion{0.0};
  double maximum_velocity{-std::numeric_limits<double>::infinity()};
};

/// Simulates the exact backward-Euler braking policy used by the published tail.
///
/// The candidate velocity is already the next queued state. Residual latency is rounded
/// up to complete controller ticks and applies adverse jerk before the shared braking
/// policy takes over.
bool discrete_stopping_envelope(
  const double signed_velocity,
  const double signed_acceleration,
  const double max_velocity,
  const double max_acceleration,
  const double max_jerk,
  const HierarchicalVelocityQpConfig & config,
  DiscreteStoppingEnvelope & envelope)
{
  constexpr std::size_t kMaximumSimulationTicks = 10000;
  const double residual_ticks_value = std::ceil(
    config.residual_command_latency_sec / config.period_sec);
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks))
  {
    return false;
  }
  const auto residual_ticks = static_cast<std::size_t>(residual_ticks_value);
  detail::ScalarBrakingOptions options;
  options.period_sec = config.period_sec;
  // The tail validator uses this same margin. Keeping the policy decision on
  // one tolerance prevents the QP and its reconstructed suffix from selecting
  // different terminal acceleration samples at a numerical boundary.
  options.comparison_tolerance = 2.0 * config.solution_feasibility_tolerance;
  options.maximum_braking_steps = kMaximumSimulationTicks;
  options.adverse_ticks = residual_ticks;
  const detail::ScalarBrakingRollout rollout = detail::rolloutScalarBraking(
    detail::ScalarBrakingState{0.0, signed_velocity, signed_acceleration},
    detail::ScalarBrakingLimits{max_velocity, max_acceleration, max_jerk},
    options, false);
  if (!rollout.commandAvailable()) {
    return false;
  }
  envelope.maximum_position_excursion = rollout.maximum_position;
  envelope.maximum_velocity = rollout.maximum_velocity;
  return true;
}

bool safe_before_velocity_limit(
  const double candidate_velocity,
  const double current_velocity,
  const double signed_direction,
  const double max_velocity,
  const double max_acceleration,
  const double max_jerk,
  const HierarchicalVelocityQpConfig & config)
{
  const double candidate_acceleration =
    (candidate_velocity - current_velocity) / config.period_sec;
  DiscreteStoppingEnvelope envelope;
  if (!discrete_stopping_envelope(
    signed_direction * candidate_velocity,
    signed_direction * candidate_acceleration,
    max_velocity,
    max_acceleration,
    max_jerk,
    config, envelope))
  {
    return false;
  }
  // Solver feasibility tolerance is not extra physical velocity.  Keep the
  // recursively feasible envelope inside the real joint limit so the next
  // control tick cannot start outside the strict collision-builder box.
  return envelope.maximum_velocity <= max_velocity;
}

InputCheck enforce_velocity_braking_invariant(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index index,
  const HierarchicalVelocityQpConfig & config,
  double & lower_velocity,
  double & upper_velocity)
{
  // 32 halvings are already substantially finer than the configured QP
  // feasibility tolerance, while these checks run in the 100 Hz hot path.
  constexpr int kBisectionIterations = 32;
  const double interior_margin = 2.0 * config.solution_feasibility_tolerance;
  const auto safe_upper = [&](const double candidate_velocity) {
      return safe_before_velocity_limit(
        candidate_velocity, state.velocity[index], 1.0,
        limits.max_velocity[index], limits.max_acceleration[index],
        limits.max_jerk[index], config);
    };
  if (!safe_upper(upper_velocity)) {
    if (!safe_upper(lower_velocity)) {
      return InputCheck::kInfeasible;
    }
    double safe = lower_velocity;
    double unsafe = upper_velocity;
    for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
      const double candidate = 0.5 * (safe + unsafe);
      if (safe_upper(candidate)) {
        safe = candidate;
      } else {
        unsafe = candidate;
      }
    }
    // Keep the selected command just inside the bisection boundary. Otherwise
    // a sub-microradian conversion difference in MoveIt can place the next
    // recursive solve infinitesimally outside the same viability set.
    upper_velocity = std::max(lower_velocity, safe - interior_margin);
  }

  const auto safe_lower = [&](const double candidate_velocity) {
      return safe_before_velocity_limit(
        candidate_velocity, state.velocity[index], -1.0,
        limits.max_velocity[index], limits.max_acceleration[index],
        limits.max_jerk[index], config);
    };
  if (!safe_lower(lower_velocity)) {
    if (!safe_lower(upper_velocity)) {
      return InputCheck::kInfeasible;
    }
    double unsafe = lower_velocity;
    double safe = upper_velocity;
    for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
      const double candidate = 0.5 * (unsafe + safe);
      if (safe_lower(candidate)) {
        safe = candidate;
      } else {
        unsafe = candidate;
      }
    }
    lower_velocity = std::min(upper_velocity, safe + interior_margin);
  }

  return lower_velocity <= upper_velocity ? InputCheck::kValid : InputCheck::kInfeasible;
}

bool safe_before_position_limit(
  const double candidate_velocity,
  const double current_position,
  const double current_velocity,
  const double signed_direction,
  const double safe_position_limit,
  const double max_velocity,
  const double max_acceleration,
  const double max_jerk,
  const HierarchicalVelocityQpConfig & config)
{
  const double candidate_acceleration =
    (candidate_velocity - current_velocity) / config.period_sec;
  const double next_signed_position = signed_direction *
    (current_position + candidate_velocity * config.period_sec);
  const double signed_limit = signed_direction * safe_position_limit;
  DiscreteStoppingEnvelope envelope;
  if (!discrete_stopping_envelope(
    signed_direction * candidate_velocity,
    signed_direction * candidate_acceleration,
    max_velocity,
    max_acceleration,
    max_jerk,
    config, envelope))
  {
    return false;
  }
  // As above, the numerical solver tolerance must not consume joint travel.
  // The bisection caller moves the selected bound into the safe interior.
  return next_signed_position + envelope.maximum_position_excursion <= signed_limit;
}

InputCheck enforce_braking_invariant(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index index,
  const HierarchicalVelocityQpConfig & config,
  double & lower_velocity,
  double & upper_velocity)
{
  // Keep the hot-path search bounded without chasing irrelevant
  // machine-precision digits.
  constexpr int kBisectionIterations = 32;
  const double interior_margin = 2.0 * config.solution_feasibility_tolerance;
  const double safe_lower_position =
    limits.lower_position[index];
  const double safe_upper_position =
    limits.upper_position[index];

  const auto safe_upper = [&](const double candidate_velocity) {
      return safe_before_position_limit(
        candidate_velocity, state.position[index], state.velocity[index],
        1.0, safe_upper_position,
        limits.max_velocity[index],
        limits.max_acceleration[index], limits.max_jerk[index], config);
    };
  if (!safe_upper(upper_velocity)) {
    if (!safe_upper(lower_velocity)) {
      return InputCheck::kInfeasible;
    }
    double safe = lower_velocity;
    double unsafe = upper_velocity;
    for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
      const double candidate = 0.5 * (safe + unsafe);
      if (safe_upper(candidate)) {
        safe = candidate;
      } else {
        unsafe = candidate;
      }
    }
    upper_velocity = std::max(lower_velocity, safe - interior_margin);
  }

  const auto safe_lower = [&](const double candidate_velocity) {
      return safe_before_position_limit(
        candidate_velocity, state.position[index], state.velocity[index],
        -1.0, safe_lower_position,
        limits.max_velocity[index],
        limits.max_acceleration[index], limits.max_jerk[index], config);
    };
  if (!safe_lower(lower_velocity)) {
    if (!safe_lower(upper_velocity)) {
      return InputCheck::kInfeasible;
    }
    double unsafe = lower_velocity;
    double safe = upper_velocity;
    for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
      const double candidate = 0.5 * (unsafe + safe);
      if (safe_lower(candidate)) {
        safe = candidate;
      } else {
        unsafe = candidate;
      }
    }
    lower_velocity = std::min(upper_velocity, safe + interior_margin);
  }

  return lower_velocity <= upper_velocity ? InputCheck::kValid : InputCheck::kInfeasible;
}

InputCheck validate_common_inputs(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const std::vector<LinearVelocityConstraint> & constraints)
{
  const Eigen::Index joint_count = state.position.size();
  if (joint_count <= 0 || state.velocity.size() != joint_count ||
    state.acceleration.size() != joint_count ||
    limits.lower_position.size() != joint_count ||
    limits.upper_position.size() != joint_count ||
    limits.position_margin.size() != joint_count ||
    limits.max_velocity.size() != joint_count ||
    limits.max_acceleration.size() != joint_count ||
    limits.max_jerk.size() != joint_count)
  {
    return InputCheck::kInvalid;
  }

  if (!is_finite(state.position) || !is_finite(state.velocity) ||
    !is_finite(state.acceleration) || !is_finite(limits.lower_position) ||
    !is_finite(limits.upper_position) || !is_finite(limits.position_margin) ||
    !is_finite(limits.max_velocity) || !is_finite(limits.max_acceleration) ||
    !is_finite(limits.max_jerk))
  {
    return InputCheck::kInvalid;
  }

  for (Eigen::Index index = 0; index < joint_count; ++index) {
    if (limits.lower_position[index] >= limits.upper_position[index] ||
      limits.position_margin[index] < 0.0 ||
      limits.max_velocity[index] <= 0.0 ||
      limits.max_acceleration[index] <= 0.0 ||
      limits.max_jerk[index] <= 0.0)
    {
      return InputCheck::kInvalid;
    }
    if (limits.lower_position[index] + limits.position_margin[index] >
      limits.upper_position[index] - limits.position_margin[index])
    {
      return InputCheck::kInvalid;
    }
  }

  for (const LinearVelocityConstraint & constraint : constraints) {
    if (constraint.coefficients.size() != joint_count ||
      !is_finite(constraint.coefficients) ||
      std::isnan(constraint.lower_bound) || std::isnan(constraint.upper_bound) ||
      constraint.lower_bound > constraint.upper_bound)
    {
      return InputCheck::kInvalid;
    }
    if (constraint.coefficients.isZero() &&
      (constraint.lower_bound > 0.0 || constraint.upper_bound < 0.0))
    {
      return InputCheck::kInfeasible;
    }
  }
  return InputCheck::kValid;
}

bool valid_objective(
  const Eigen::MatrixXd & matrix,
  const Eigen::VectorXd & reference,
  const Eigen::VectorXd & weights,
  const Eigen::Index joint_count,
  const bool allow_empty)
{
  if (matrix.cols() != joint_count || reference.size() != matrix.rows() ||
    weights.size() != matrix.rows() || (!allow_empty && matrix.rows() == 0) ||
    !is_finite(matrix) || !is_finite(reference) || !is_finite(weights))
  {
    return false;
  }
  return matrix.rows() == 0 || (weights.array() > 0.0).all();
}

InputCheck build_constraint_matrix(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const std::vector<LinearVelocityConstraint> & constraints,
  const HierarchicalVelocityQpConfig & config,
  ConstraintMatrix & output)
{
  const Eigen::Index joint_count = state.position.size();
  const Eigen::Index row_count =
    joint_count + static_cast<Eigen::Index>(constraints.size());
  output.matrix = Eigen::MatrixXd::Zero(row_count, joint_count);
  output.lower = Eigen::VectorXd::Constant(
    row_count, -std::numeric_limits<double>::infinity());
  output.upper = Eigen::VectorXd::Constant(
    row_count, std::numeric_limits<double>::infinity());

  for (Eigen::Index index = 0; index < joint_count; ++index) {
    const double minimum_acceleration = std::max(
      -limits.max_acceleration[index],
      state.acceleration[index] - limits.max_jerk[index] * config.period_sec);
    const double maximum_acceleration = std::min(
      limits.max_acceleration[index],
      state.acceleration[index] + limits.max_jerk[index] * config.period_sec);

    double lower_velocity = std::max(
      -limits.max_velocity[index],
      state.velocity[index] + minimum_acceleration * config.period_sec);
    double upper_velocity = std::min(
      limits.max_velocity[index],
      state.velocity[index] + maximum_acceleration * config.period_sec);

    const double safe_lower_position =
      limits.lower_position[index];
    const double safe_upper_position =
      limits.upper_position[index];
    lower_velocity = std::max(
      lower_velocity,
      (safe_lower_position - state.position[index]) / config.period_sec);
    upper_velocity = std::min(
      upper_velocity,
      (safe_upper_position - state.position[index]) / config.period_sec);

    if (!std::isfinite(lower_velocity) || !std::isfinite(upper_velocity)) {
      return InputCheck::kInvalid;
    }
    if (lower_velocity > upper_velocity) {
      return InputCheck::kInfeasible;
    }
    const InputCheck velocity_braking_check = enforce_velocity_braking_invariant(
      state, limits, index, config, lower_velocity, upper_velocity);
    if (velocity_braking_check != InputCheck::kValid) {
      return velocity_braking_check;
    }
    const InputCheck braking_check = enforce_braking_invariant(
      state, limits, index, config, lower_velocity, upper_velocity);
    if (braking_check != InputCheck::kValid) {
      return braking_check;
    }

    output.matrix(index, index) = 1.0;
    output.lower[index] = lower_velocity;
    output.upper[index] = upper_velocity;
  }

  for (std::size_t index = 0; index < constraints.size(); ++index) {
    const Eigen::Index row = joint_count + static_cast<Eigen::Index>(index);
    output.matrix.row(row) = constraints[index].coefficients.transpose();
    output.lower[row] = constraints[index].lower_bound;
    output.upper[row] = constraints[index].upper_bound;
  }
  return InputCheck::kValid;
}

SparseArrays make_upper_triangular_csc(const Eigen::MatrixXd & matrix)
{
  SparseArrays sparse;
  sparse.column_pointers.reserve(static_cast<std::size_t>(matrix.cols() + 1));
  sparse.row_indices.reserve(
    static_cast<std::size_t>(matrix.rows() * (matrix.rows() + 1) / 2));
  sparse.values.reserve(sparse.row_indices.capacity());
  sparse.column_pointers.push_back(0);

  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row <= column; ++row) {
      const double value = matrix(row, column);
      if (value != 0.0 || row == column) {
        sparse.row_indices.push_back(static_cast<c_int>(row));
        sparse.values.push_back(static_cast<c_float>(value));
      }
    }
    sparse.column_pointers.push_back(static_cast<c_int>(sparse.values.size()));
  }
  return sparse;
}

SparseArrays make_csc(const Eigen::MatrixXd & matrix)
{
  SparseArrays sparse;
  sparse.column_pointers.reserve(static_cast<std::size_t>(matrix.cols() + 1));
  sparse.row_indices.reserve(static_cast<std::size_t>(matrix.size()));
  sparse.values.reserve(static_cast<std::size_t>(matrix.size()));
  sparse.column_pointers.push_back(0);

  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double value = matrix(row, column);
      if (value != 0.0) {
        sparse.row_indices.push_back(static_cast<c_int>(row));
        sparse.values.push_back(static_cast<c_float>(value));
      }
    }
    sparse.column_pointers.push_back(static_cast<c_int>(sparse.values.size()));
  }
  return sparse;
}

c_float osqp_bound(const double bound)
{
  if (std::isinf(bound)) {
    return bound < 0.0 ? -OSQP_INFTY : OSQP_INFTY;
  }
  return static_cast<c_float>(bound);
}

QpStageStatus classify_stage_status(const c_int status)
{
  switch (status) {
    case OSQP_SOLVED:
      return QpStageStatus::kSolved;
    case OSQP_SOLVED_INACCURATE:
      return QpStageStatus::kSolvedInaccurate;
    case OSQP_PRIMAL_INFEASIBLE:
    case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
    case OSQP_DUAL_INFEASIBLE:
    case OSQP_DUAL_INFEASIBLE_INACCURATE:
      return QpStageStatus::kInfeasible;
    case OSQP_MAX_ITER_REACHED:
#ifdef OSQP_TIME_LIMIT_REACHED
    case OSQP_TIME_LIMIT_REACHED:
#endif
      return QpStageStatus::kMaxIterations;
    default:
      return QpStageStatus::kSolverError;
  }
}

bool satisfies_constraints(
  const Eigen::VectorXd & velocity,
  const ConstraintMatrix & constraints,
  const double tolerance)
{
  if (!is_finite(velocity)) {
    return false;
  }
  const Eigen::VectorXd value = constraints.matrix * velocity;
  if (!is_finite(value)) {
    return false;
  }
  for (Eigen::Index row = 0; row < value.size(); ++row) {
    if (std::isfinite(constraints.lower[row]) &&
      value[row] < constraints.lower[row] - tolerance)
    {
      return false;
    }
    if (std::isfinite(constraints.upper[row]) &&
      value[row] > constraints.upper[row] + tolerance)
    {
      return false;
    }
  }
  return true;
}

StageResult solve_stage(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const ConstraintMatrix & constraints,
  const Eigen::VectorXd & warm_start,
  const HierarchicalVelocityQpConfig & config)
{
  StageResult result;
  const Eigen::Index variable_count = gradient.size();
  if (hessian.rows() != variable_count || hessian.cols() != variable_count ||
    constraints.matrix.cols() != variable_count ||
    constraints.lower.size() != constraints.matrix.rows() ||
    constraints.upper.size() != constraints.matrix.rows() ||
    warm_start.size() != variable_count || !is_finite(hessian) ||
    !is_finite(gradient) || !is_finite(constraints.matrix) ||
    !is_finite(warm_start))
  {
    return result;
  }

  const Eigen::MatrixXd symmetric_hessian = 0.5 * (hessian + hessian.transpose());
  SparseArrays p_sparse = make_upper_triangular_csc(symmetric_hessian);
  SparseArrays a_sparse = make_csc(constraints.matrix);

  std::vector<c_float> linear_cost(static_cast<std::size_t>(variable_count));
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    linear_cost[static_cast<std::size_t>(index)] =
      static_cast<c_float>(gradient[index]);
  }
  std::vector<c_float> lower(
    static_cast<std::size_t>(constraints.lower.size()));
  std::vector<c_float> upper(
    static_cast<std::size_t>(constraints.upper.size()));
  for (Eigen::Index row = 0; row < constraints.lower.size(); ++row) {
    lower[static_cast<std::size_t>(row)] = osqp_bound(constraints.lower[row]);
    upper[static_cast<std::size_t>(row)] = osqp_bound(constraints.upper[row]);
  }

  csc p_matrix{
    static_cast<c_int>(p_sparse.values.size()),
    static_cast<c_int>(variable_count),
    static_cast<c_int>(variable_count),
    p_sparse.column_pointers.data(),
    p_sparse.row_indices.data(),
    p_sparse.values.data(),
    -1};
  csc a_matrix{
    static_cast<c_int>(a_sparse.values.size()),
    static_cast<c_int>(constraints.matrix.rows()),
    static_cast<c_int>(variable_count),
    a_sparse.column_pointers.data(),
    a_sparse.row_indices.data(),
    a_sparse.values.data(),
    -1};

  OSQPData data{};
  data.n = static_cast<c_int>(variable_count);
  data.m = static_cast<c_int>(constraints.matrix.rows());
  data.P = &p_matrix;
  data.A = &a_matrix;
  data.q = linear_cost.data();
  data.l = lower.data();
  data.u = upper.data();

  OSQPSettings settings{};
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.polish = 0;
  settings.warm_start = 1;
  settings.adaptive_rho_interval = 10;
  settings.scaled_termination = 1;
  settings.check_termination = 1;
  settings.max_iter = static_cast<c_int>(config.solver_max_iterations);
  settings.eps_abs = static_cast<c_float>(config.solver_absolute_tolerance);
  settings.eps_rel = static_cast<c_float>(config.solver_relative_tolerance);
  settings.time_limit = static_cast<c_float>(config.solver_time_limit_sec);

  OSQPWorkspace * raw_workspace = nullptr;
  const c_int setup_result = osqp_setup(&raw_workspace, &data, &settings);
  WorkspacePtr workspace{raw_workspace};
  if (setup_result != 0 || workspace == nullptr) {
    return result;
  }

  std::vector<c_float> initial_velocity(static_cast<std::size_t>(variable_count));
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    initial_velocity[static_cast<std::size_t>(index)] =
      static_cast<c_float>(warm_start[index]);
  }
  if (osqp_warm_start_x(workspace.get(), initial_velocity.data()) != 0) {
    return result;
  }

  const c_int solve_result = osqp_solve(workspace.get());
  if (solve_result != 0 || workspace->info == nullptr) {
    return result;
  }
  result.status = classify_stage_status(workspace->info->status_val);
  result.iterations = static_cast<int>(workspace->info->iter);
  if (result.status != QpStageStatus::kSolved &&
    result.status != QpStageStatus::kSolvedInaccurate)
  {
    return result;
  }
  if (workspace->solution == nullptr || workspace->solution->x == nullptr) {
    result.status = QpStageStatus::kSolverError;
    return result;
  }

  result.velocity.resize(variable_count);
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    result.velocity[index] = workspace->solution->x[index];
  }
  // The first N rows are the per-joint interval assembled by
  // build_constraint_matrix(). Clamp round-off at those boundaries so a small
  // accepted OSQP residual cannot become a velocity/acceleration/jerk violation.
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    result.velocity[index] = std::clamp(
      result.velocity[index], constraints.lower[index], constraints.upper[index]);
  }
  if (!satisfies_constraints(
      result.velocity, constraints, config.solution_feasibility_tolerance))
  {
    result.velocity.resize(0);
    result.status = QpStageStatus::kSolverError;
  }
  return result;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> make_tracking_objective(
  const Eigen::MatrixXd & matrix,
  const Eigen::VectorXd & reference,
  const Eigen::VectorXd & weights,
  const Eigen::VectorXd & previous_velocity,
  const HierarchicalVelocityQpConfig & config)
{
  const Eigen::Index joint_count = previous_velocity.size();
  const Eigen::MatrixXd weighted_matrix = weights.asDiagonal() * matrix;
  const Eigen::VectorXd weighted_reference = weights.array() * reference.array();

  Eigen::MatrixXd hessian = weighted_matrix.transpose() * weighted_matrix;
  hessian.diagonal().array() +=
    config.velocity_regularization + config.continuity_weight;
  Eigen::VectorXd gradient =
    -2.0 * weighted_matrix.transpose() * weighted_reference;
  gradient.noalias() -= 2.0 * config.continuity_weight * previous_velocity;
  hessian *= 2.0;

  if (hessian.rows() != joint_count) {
    return {Eigen::MatrixXd{}, Eigen::VectorXd{}};
  }
  return {std::move(hessian), std::move(gradient)};
}

QpStatus top_level_status(const QpStageStatus status)
{
  switch (status) {
    case QpStageStatus::kSolved:
      return QpStatus::kSolved;
    case QpStageStatus::kSolvedInaccurate:
      return QpStatus::kSolvedInaccurate;
    case QpStageStatus::kInfeasible:
      return QpStatus::kInfeasible;
    case QpStageStatus::kMaxIterations:
      return QpStatus::kMaxIterations;
    case QpStageStatus::kNotRun:
    case QpStageStatus::kSolverError:
      return QpStatus::kSolverError;
  }
  return QpStatus::kSolverError;
}

}  // namespace

bool QpResult::command_available() const noexcept
{
  return status == QpStatus::kSolved || status == QpStatus::kSolvedInaccurate ||
         status == QpStatus::kPrimaryOnly;
}

JointCenteringGuidance jointCenteringGuidance(
  const double position, const double lower, const double upper,
  const double activation_fraction, const double maximum_speed)
{
  if (!std::isfinite(position) || !std::isfinite(lower) || !std::isfinite(upper) ||
    lower >= upper || !std::isfinite(activation_fraction) ||
    activation_fraction <= 0.0 || activation_fraction >= 1.0 ||
    !std::isfinite(maximum_speed) || maximum_speed <= 0.0)
  {
    throw std::invalid_argument("invalid joint centering guidance input");
  }
  const double center = 0.5 * (lower + upper);
  const double half_range = 0.5 * (upper - lower);
  const double progress = std::clamp(
    (std::abs(position - center) / half_range - activation_fraction) /
    (1.0 - activation_fraction), 0.0, 1.0);
  const double activation = progress * progress * (3.0 - 2.0 * progress);
  return {activation, -std::copysign(maximum_speed * activation, position - center)};
}

bool QpResult::fully_solved() const noexcept
{
  return status == QpStatus::kSolved || status == QpStatus::kSolvedInaccurate;
}

HierarchicalVelocityQp::HierarchicalVelocityQp(HierarchicalVelocityQpConfig config)
: config_(std::move(config))
{
  if (!valid_config(config_)) {
    throw std::invalid_argument("hierarchical velocity QP configuration is invalid");
  }
}

QpResult HierarchicalVelocityQp::solve(
  const HierarchicalVelocityTask & task,
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const std::vector<LinearVelocityConstraint> & constraints) const
{
  QpResult result;
  const InputCheck common_check = validate_common_inputs(state, limits, constraints);
  if (common_check == InputCheck::kInvalid) {
    return result;
  }
  if (common_check == InputCheck::kInfeasible) {
    result.status = QpStatus::kInfeasible;
    return result;
  }

  const Eigen::Index joint_count = state.position.size();
  if (!valid_objective(
      task.primary_matrix, task.primary_reference, task.primary_weights,
      joint_count, false) ||
    !valid_objective(
      task.secondary_matrix, task.secondary_reference, task.secondary_weights,
      joint_count, true))
  {
    return result;
  }

  ConstraintMatrix base_constraints;
  const InputCheck bounds_check = build_constraint_matrix(
    state, limits, constraints, config_, base_constraints);
  if (bounds_check == InputCheck::kInvalid) {
    return result;
  }
  if (bounds_check == InputCheck::kInfeasible) {
    result.status = QpStatus::kInfeasible;
    return result;
  }

  const auto primary_objective = make_tracking_objective(
    task.primary_matrix, task.primary_reference, task.primary_weights,
    state.velocity, config_);
  const StageResult primary = solve_stage(
    primary_objective.first, primary_objective.second, base_constraints,
    state.velocity, config_);
  result.primary_status = primary.status;
  result.primary_iterations = primary.iterations;
  if (primary.status != QpStageStatus::kSolved &&
    primary.status != QpStageStatus::kSolvedInaccurate)
  {
    result.status = top_level_status(primary.status);
    return result;
  }

  result.joint_velocity = primary.velocity;
  result.achieved_primary_velocity = task.primary_matrix * primary.velocity;

  if (task.secondary_matrix.rows() == 0) {
    result.status = top_level_status(primary.status);
    return result;
  }

  ConstraintMatrix secondary_constraints;
  const Eigen::Index base_rows = base_constraints.matrix.rows();
  const Eigen::Index primary_rows = task.primary_matrix.rows();
  secondary_constraints.matrix.resize(base_rows + primary_rows, joint_count);
  secondary_constraints.matrix.topRows(base_rows) = base_constraints.matrix;
  secondary_constraints.matrix.bottomRows(primary_rows) = task.primary_matrix;
  secondary_constraints.lower.resize(base_rows + primary_rows);
  secondary_constraints.upper.resize(base_rows + primary_rows);
  secondary_constraints.lower.head(base_rows) = base_constraints.lower;
  secondary_constraints.upper.head(base_rows) = base_constraints.upper;
  secondary_constraints.lower.tail(primary_rows) =
    result.achieved_primary_velocity.array() - config_.primary_preservation_tolerance;
  secondary_constraints.upper.tail(primary_rows) =
    result.achieved_primary_velocity.array() + config_.primary_preservation_tolerance;

  const auto secondary_objective = make_tracking_objective(
    task.secondary_matrix, task.secondary_reference, task.secondary_weights,
    state.velocity, config_);
  const StageResult secondary = solve_stage(
    secondary_objective.first, secondary_objective.second,
    secondary_constraints, primary.velocity, config_);
  result.secondary_status = secondary.status;
  result.secondary_iterations = secondary.iterations;
  if (secondary.status != QpStageStatus::kSolved &&
    secondary.status != QpStageStatus::kSolvedInaccurate)
  {
    result.status = QpStatus::kPrimaryOnly;
    return result;
  }

  result.joint_velocity = secondary.velocity;
  result.achieved_primary_velocity = task.primary_matrix * secondary.velocity;
  result.status =
    primary.status == QpStageStatus::kSolvedInaccurate ||
    secondary.status == QpStageStatus::kSolvedInaccurate ?
    QpStatus::kSolvedInaccurate : QpStatus::kSolved;
  return result;
}

QpResult HierarchicalVelocityQp::brake(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const std::vector<LinearVelocityConstraint> & constraints) const
{
  QpResult result;
  const InputCheck common_check = validate_common_inputs(state, limits, constraints);
  if (common_check == InputCheck::kInvalid) {
    return result;
  }
  if (common_check == InputCheck::kInfeasible) {
    result.status = QpStatus::kInfeasible;
    return result;
  }

  ConstraintMatrix hard_constraints;
  const InputCheck bounds_check = build_constraint_matrix(
    state, limits, constraints, config_, hard_constraints);
  if (bounds_check == InputCheck::kInvalid) {
    return result;
  }
  if (bounds_check == InputCheck::kInfeasible) {
    result.status = QpStatus::kInfeasible;
    return result;
  }

  const Eigen::Index joint_count = state.position.size();
  const Eigen::MatrixXd hessian =
    2.0 * (1.0 + config_.velocity_regularization) *
    Eigen::MatrixXd::Identity(joint_count, joint_count);
  const Eigen::VectorXd gradient = Eigen::VectorXd::Zero(joint_count);
  const StageResult braking = solve_stage(
    hessian, gradient, hard_constraints, state.velocity, config_);
  result.primary_status = braking.status;
  result.primary_iterations = braking.iterations;
  result.status = top_level_status(braking.status);
  if (braking.status == QpStageStatus::kSolved ||
    braking.status == QpStageStatus::kSolvedInaccurate)
  {
    result.joint_velocity = braking.velocity;
  }
  return result;
}

const HierarchicalVelocityQpConfig & HierarchicalVelocityQp::config() const noexcept
{
  return config_;
}

}  // namespace face_tracking_arm::control
