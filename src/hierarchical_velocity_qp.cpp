// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

#include <osqp.h>

#include <algorithm>
#include <array>
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
  std::optional<QpFailureSnapshot> failure_snapshot;
  QpFailureSnapshotDisposition snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};
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

// OSQP's default primal-infeasibility certificate tolerance can classify a
// nearly tangent but feasible collision row as infeasible. Tighten only the
// certificate test; this does not alter primal constraints or acceptance.
constexpr double kSolverPrimalInfeasibilityTolerance = 1.0e-10;

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
  const NumericalPolicy & numerical_policy,
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
  // QP residual is a solver-domain tolerance, not a q/v/a/j allowance.
  options.numerical_policy = numerical_policy;
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

// Scoped to one joint while building its bounds: limits, period, latency, and
// numerical policy are immutable for that scope. Cache exact signed inputs only;
// this reuses a previously computed rollout without approximating its key.
class DiscreteStoppingEnvelopeCache
{
public:
  DiscreteStoppingEnvelopeCache(
    const double max_velocity, const double max_acceleration, const double max_jerk,
    const NumericalPolicy & numerical_policy,
    const HierarchicalVelocityQpConfig & config)
  : max_velocity_(max_velocity), max_acceleration_(max_acceleration),
    max_jerk_(max_jerk), numerical_policy_(numerical_policy), config_(config) {}

  bool get(
    const double signed_velocity, const double signed_acceleration,
    DiscreteStoppingEnvelope & envelope)
  {
    for (const Entry & entry : entries_) {
      if (entry.occupied && sameBits(entry.velocity, signed_velocity) &&
        sameBits(entry.acceleration, signed_acceleration))
      {
        envelope = entry.envelope;
        return entry.valid;
      }
    }
    Entry & entry = entries_[next_entry_];
    next_entry_ = (next_entry_ + 1U) % entries_.size();
    entry.velocity = signed_velocity;
    entry.acceleration = signed_acceleration;
    entry.envelope = {};
    entry.valid = discrete_stopping_envelope(
      signed_velocity, signed_acceleration, max_velocity_, max_acceleration_, max_jerk_,
      numerical_policy_, config_, entry.envelope);
    entry.occupied = true;
    envelope = entry.envelope;
    return entry.valid;
  }

private:
  static bool sameBits(const double lhs, const double rhs)
  {
    return lhs == rhs && std::signbit(lhs) == std::signbit(rhs);
  }

  struct Entry
  {
    double velocity{0.0};
    double acceleration{0.0};
    DiscreteStoppingEnvelope envelope{};
    bool occupied{false};
    bool valid{false};
  };

  double max_velocity_;
  double max_acceleration_;
  double max_jerk_;
  const NumericalPolicy & numerical_policy_;
  const HierarchicalVelocityQpConfig & config_;
  std::array<Entry, 2> entries_{};
  std::size_t next_entry_{0U};
};

bool safe_before_velocity_limit(
  const double candidate_velocity,
  const double current_velocity,
  const double signed_direction,
  const double max_velocity,
  const NumericalPolicy & numerical_policy,
  const HierarchicalVelocityQpConfig & config,
  DiscreteStoppingEnvelopeCache & profile_cache)
{
  const double candidate_acceleration =
    (candidate_velocity - current_velocity) / config.period_sec;
  DiscreteStoppingEnvelope envelope;
  if (!profile_cache.get(
    signed_direction * candidate_velocity,
    signed_direction * candidate_acceleration, envelope))
  {
    return false;
  }
  // Solver feasibility tolerance is not extra physical velocity.  Keep the
  // recursively feasible envelope inside the real joint limit so the next
  // control tick cannot start outside the strict collision-builder box.
  return envelope.maximum_velocity <=
         max_velocity - numerical_policy.velocity_error.value;
}

InputCheck enforce_velocity_braking_invariant(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index index,
  const HierarchicalVelocityQpConfig & config,
  DiscreteStoppingEnvelopeCache & profile_cache,
  double & lower_velocity,
  double & upper_velocity)
{
  // Keep the existing bounded 32-step search in the 100 Hz hot path.
  constexpr int kBisectionIterations = 32;
  const auto safe_upper = [&](const double candidate_velocity) {
      return safe_before_velocity_limit(
        candidate_velocity, state.velocity[index], 1.0,
        limits.max_velocity[index], *limits.numerical_policy, config, profile_cache);
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
    upper_velocity = safe;
  }

  const auto safe_lower = [&](const double candidate_velocity) {
      return safe_before_velocity_limit(
        candidate_velocity, state.velocity[index], -1.0,
        limits.max_velocity[index], *limits.numerical_policy, config, profile_cache);
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
    lower_velocity = safe;
  }

  return lower_velocity <= upper_velocity ? InputCheck::kValid : InputCheck::kInfeasible;
}

bool safe_before_position_limit(
  const double candidate_velocity,
  const double current_position,
  const double current_velocity,
  const double signed_direction,
  const double safe_position_limit,
  const NumericalPolicy & numerical_policy,
  const HierarchicalVelocityQpConfig & config,
  DiscreteStoppingEnvelopeCache & profile_cache)
{
  const double candidate_acceleration =
    (candidate_velocity - current_velocity) / config.period_sec;
  const double next_signed_position = signed_direction *
    (current_position + candidate_velocity * config.period_sec);
  const double signed_limit = signed_direction * safe_position_limit;
  DiscreteStoppingEnvelope envelope;
  if (!profile_cache.get(
    signed_direction * candidate_velocity,
    signed_direction * candidate_acceleration, envelope))
  {
    return false;
  }
  // For a true stationary hold, q + 0 * dt is exact and has no arithmetic
  // position uncertainty. Keep inclusive limit endpoints holdable; moving
  // candidates still reserve the typed q-operation budget. Solver residual
  // remains in QP units and is not part of this position check.
  const double position_operation_error =
    candidate_velocity == 0.0 && envelope.maximum_position_excursion == 0.0 ?
    0.0 : numerical_policy.position_error.value;
  return next_signed_position + envelope.maximum_position_excursion <=
         signed_limit - position_operation_error;
}

InputCheck enforce_braking_invariant(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index index,
  const HierarchicalVelocityQpConfig & config,
  DiscreteStoppingEnvelopeCache & profile_cache,
  double & lower_velocity,
  double & upper_velocity)
{
  // Keep the hot-path search bounded without chasing irrelevant
  // machine-precision digits.
  constexpr int kBisectionIterations = 32;
  const double safe_lower_position =
    limits.lower_position[index];
  const double safe_upper_position =
    limits.upper_position[index];

  const auto safe_upper = [&](const double candidate_velocity) {
      return safe_before_position_limit(
        candidate_velocity, state.position[index], state.velocity[index],
        1.0, safe_upper_position,
        *limits.numerical_policy, config, profile_cache);
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
    upper_velocity = safe;
  }

  const auto safe_lower = [&](const double candidate_velocity) {
      return safe_before_position_limit(
        candidate_velocity, state.position[index], state.velocity[index],
        -1.0, safe_lower_position,
        *limits.numerical_policy, config, profile_cache);
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
    lower_velocity = safe;
  }

  return lower_velocity <= upper_velocity ? InputCheck::kValid : InputCheck::kInfeasible;
}

InputCheck validate_common_inputs(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const std::vector<LinearVelocityConstraint> & constraints)
{
  const Eigen::Index joint_count = state.position.size();
  if (!limits.numerical_policy.has_value() || !limits.numerical_policy->valid() ||
    joint_count <= 0 || state.velocity.size() != joint_count ||
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
    if (classifyPosition(state.position[index],
        PositionInterval{limits.lower_position[index], limits.upper_position[index]},
        limits.numerical_policy->position_error) == BoundaryClass::kOutside ||
      classifyVelocity(RadiansPerSecond{state.velocity[index]},
        RadiansPerSecond{limits.max_velocity[index]},
        limits.numerical_policy->velocity_error) == BoundaryClass::kOutside ||
      classifyAcceleration(RadiansPerSecondSquared{state.acceleration[index]},
        RadiansPerSecondSquared{limits.max_acceleration[index]},
        limits.numerical_policy->acceleration_error) == BoundaryClass::kOutside)
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

    const double safe_lower_position = limits.lower_position[index];
    const double safe_upper_position = limits.upper_position[index];
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
    DiscreteStoppingEnvelopeCache profile_cache(
      limits.max_velocity[index], limits.max_acceleration[index], limits.max_jerk[index],
      *limits.numerical_policy, config);
    const InputCheck velocity_braking_check = enforce_velocity_braking_invariant(
      state, limits, index, config, profile_cache, lower_velocity, upper_velocity);
    if (velocity_braking_check != InputCheck::kValid) {
      return velocity_braking_check;
    }
    const InputCheck braking_check = enforce_braking_invariant(
      state, limits, index, config, profile_cache, lower_velocity, upper_velocity);
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
      return QpStageStatus::kMaxIterations;
#ifdef OSQP_TIME_LIMIT_REACHED
    case OSQP_TIME_LIMIT_REACHED:
      return QpStageStatus::kTimeLimit;
#endif
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
  const HierarchicalVelocityQpConfig & config,
  const QpSolveStage solve_stage_kind)
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
  Eigen::VectorXd variable_shift(variable_count);
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    const double lower = constraints.lower[index];
    const double upper = constraints.upper[index];
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
      return result;
    }
    variable_shift[index] = 0.5 * lower + 0.5 * upper;
  }

  // Center the bounded velocity variables before solving. The affine change
  // preserves the QP exactly while keeping the near-zero feasible corner away
  // from cancellation between absolute joint velocities and active safety rows.
  Eigen::VectorXd solver_gradient = gradient + symmetric_hessian * variable_shift;
  Eigen::VectorXd solver_warm_start = warm_start - variable_shift;
  Eigen::VectorXd solver_lower = constraints.lower;
  Eigen::VectorXd solver_upper = constraints.upper;
  const Eigen::VectorXd row_shift = constraints.matrix * variable_shift;
  if (!is_finite(variable_shift) || !is_finite(solver_gradient) ||
    !is_finite(solver_warm_start) || !is_finite(row_shift))
  {
    return result;
  }
  solver_lower -= row_shift;
  solver_upper -= row_shift;
  for (Eigen::Index row = 0; row < solver_lower.size(); ++row) {
    if (std::isnan(solver_lower[row]) || std::isnan(solver_upper[row]) ||
      (std::isfinite(constraints.lower[row]) && !std::isfinite(solver_lower[row])) ||
      (std::isfinite(constraints.upper[row]) && !std::isfinite(solver_upper[row])))
    {
      return result;
    }
  }

  SparseArrays p_sparse = make_upper_triangular_csc(symmetric_hessian);
  SparseArrays a_sparse = make_csc(constraints.matrix);

  std::vector<c_float> linear_cost(static_cast<std::size_t>(variable_count));
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    linear_cost[static_cast<std::size_t>(index)] =
      static_cast<c_float>(solver_gradient[index]);
  }
  std::vector<c_float> lower(
    static_cast<std::size_t>(solver_lower.size()));
  std::vector<c_float> upper(
    static_cast<std::size_t>(solver_upper.size()));
  for (Eigen::Index row = 0; row < solver_lower.size(); ++row) {
    lower[static_cast<std::size_t>(row)] = osqp_bound(solver_lower[row]);
    upper[static_cast<std::size_t>(row)] = osqp_bound(solver_upper[row]);
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
  settings.eps_prim_inf = static_cast<c_float>(kSolverPrimalInfeasibilityTolerance);
  settings.eps_abs = static_cast<c_float>(config.solver_absolute_tolerance);
  settings.eps_rel = static_cast<c_float>(config.solver_relative_tolerance);
  settings.time_limit = static_cast<c_float>(config.solver_time_limit_sec);

  std::vector<c_float> initial_velocity(static_cast<std::size_t>(variable_count));
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    initial_velocity[static_cast<std::size_t>(index)] =
      static_cast<c_float>(solver_warm_start[index]);
  }

  const auto capture_failure = [&](const int setup_code, const int solve_code,
    const int raw_status, const double observed_solve_time) {
      constexpr std::size_t kMaximumControllerJoints =
        kQpFailureSnapshotMaximumJoints;
      constexpr std::size_t kMaximumControllerRows =
        kQpFailureSnapshotMaximumRows;
      constexpr std::size_t kMaximumControllerPNonzeros = 21U;
      constexpr std::size_t kMaximumControllerANonzeros =
        kQpFailureSnapshotMaximumANonzeros;
      if (variable_count > static_cast<Eigen::Index>(kMaximumControllerJoints) ||
        constraints.matrix.rows() > static_cast<Eigen::Index>(kMaximumControllerRows) ||
        p_sparse.values.size() > kMaximumControllerPNonzeros ||
        a_sparse.values.size() > kMaximumControllerANonzeros)
      {
        result.snapshot_disposition = QpFailureSnapshotDisposition::kShapeLimitExceeded;
        return;
      }
      QpFailureSnapshot snapshot;
      snapshot.stage = solve_stage_kind;
      snapshot.classified_status = result.status;
      snapshot.setup_return_code = setup_code;
      snapshot.solve_return_code = solve_code;
      snapshot.raw_status_value = raw_status;
      snapshot.osqp_version = OSQP_VERSION;
      snapshot.c_int_size = sizeof(c_int);
      snapshot.c_float_size = sizeof(c_float);
      snapshot.observed_solve_time_sec = observed_solve_time;
      snapshot.solver_budget_sec = config.solver_time_limit_sec;
      snapshot.variable_count = data.n;
      snapshot.constraint_count = data.m;
      snapshot.p_column_pointers = std::move(p_sparse.column_pointers);
      snapshot.p_row_indices = std::move(p_sparse.row_indices);
      snapshot.p_values = std::move(p_sparse.values);
      snapshot.a_column_pointers = std::move(a_sparse.column_pointers);
      snapshot.a_row_indices = std::move(a_sparse.row_indices);
      snapshot.a_values = std::move(a_sparse.values);
      snapshot.q = std::move(linear_cost);
      snapshot.lower = std::move(lower);
      snapshot.upper = std::move(upper);
      snapshot.centered_warm_start = std::move(initial_velocity);
      snapshot.variable_shift.resize(static_cast<std::size_t>(variable_count));
      for (Eigen::Index i = 0; i < variable_count; ++i) {
        snapshot.variable_shift[static_cast<std::size_t>(i)] = variable_shift[i];
      }
      snapshot.rho = settings.rho;
      snapshot.sigma = settings.sigma;
      snapshot.scaling = settings.scaling;
#if EMBEDDED != 1
      snapshot.adaptive_rho = settings.adaptive_rho;
      snapshot.adaptive_rho_interval = settings.adaptive_rho_interval;
      snapshot.adaptive_rho_tolerance = settings.adaptive_rho_tolerance;
#ifdef PROFILING
      snapshot.adaptive_rho_fraction = settings.adaptive_rho_fraction;
#endif
#endif
      snapshot.max_iter = settings.max_iter;
      snapshot.eps_abs = settings.eps_abs;
      snapshot.eps_rel = settings.eps_rel;
      snapshot.eps_prim_inf = settings.eps_prim_inf;
      snapshot.eps_dual_inf = settings.eps_dual_inf;
      snapshot.alpha = settings.alpha;
      snapshot.linsys_solver = static_cast<int>(settings.linsys_solver);
#ifndef EMBEDDED
      snapshot.delta = settings.delta;
      snapshot.polish = settings.polish;
      snapshot.polish_refine_iter = settings.polish_refine_iter;
      snapshot.verbose = settings.verbose;
#endif
      snapshot.scaled_termination = settings.scaled_termination;
      snapshot.check_termination = settings.check_termination;
      snapshot.warm_start = settings.warm_start;
#ifdef PROFILING
      snapshot.time_limit_sec = settings.time_limit;
#endif
      result.failure_snapshot = std::move(snapshot);
      result.snapshot_disposition = QpFailureSnapshotDisposition::kCaptured;
    };

  OSQPWorkspace * raw_workspace = nullptr;
  const c_int setup_result = osqp_setup(&raw_workspace, &data, &settings);
  WorkspacePtr workspace{raw_workspace};
  if (setup_result != 0 || workspace == nullptr) {
    capture_failure(static_cast<int>(setup_result), 0, OSQP_UNSOLVED, 0.0);
    return result;
  }
  if (workspace->settings != nullptr) {
    settings = *workspace->settings;
  }

  if (osqp_warm_start_x(workspace.get(), initial_velocity.data()) != 0) {
    capture_failure(0, -1, OSQP_UNSOLVED, 0.0);
    return result;
  }

  const c_int solve_result = osqp_solve(workspace.get());
  if (solve_result != 0 || workspace->info == nullptr) {
    capture_failure(0, static_cast<int>(solve_result), OSQP_UNSOLVED, 0.0);
    return result;
  }
  result.status = classify_stage_status(workspace->info->status_val);
  result.iterations = static_cast<int>(workspace->info->iter);
  if (result.status != QpStageStatus::kSolved &&
    result.status != QpStageStatus::kSolvedInaccurate)
  {
    capture_failure(0, static_cast<int>(solve_result),
      static_cast<int>(workspace->info->status_val),
      static_cast<double>(workspace->info->solve_time));
    return result;
  }
  if (workspace->solution == nullptr || workspace->solution->x == nullptr) {
    result.status = QpStageStatus::kSolverError;
    capture_failure(0, static_cast<int>(solve_result),
      static_cast<int>(workspace->info->status_val),
      static_cast<double>(workspace->info->solve_time));
    return result;
  }

  result.velocity.resize(variable_count);
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    result.velocity[index] = workspace->solution->x[index] + variable_shift[index];
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
    capture_failure(0, static_cast<int>(solve_result),
      static_cast<int>(workspace->info->status_val),
      static_cast<double>(workspace->info->solve_time));
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
    case QpStageStatus::kTimeLimit:
      return QpStatus::kTimeLimit;
    case QpStageStatus::kNotRun:
    case QpStageStatus::kSolverError:
      return QpStatus::kSolverError;
  }
  return QpStatus::kSolverError;
}

}  // namespace

QpReplayResult replayQpFailureSnapshot(const QpFailureSnapshot & snapshot)
{
  QpReplayResult result;
  result.compatible_solver_build = snapshot.osqp_version == OSQP_VERSION &&
    snapshot.c_int_size == sizeof(c_int) && snapshot.c_float_size == sizeof(c_float);
  if (!result.compatible_solver_build) {
    result.setup_return_code = -2;
    return result;
  }
  const auto n = snapshot.variable_count;
  const auto m = snapshot.constraint_count;
  if (n <= 0 || m <= 0 || snapshot.p_column_pointers.size() !=
    static_cast<std::size_t>(n + 1) || snapshot.a_column_pointers.size() !=
    static_cast<std::size_t>(n + 1) || snapshot.q.size() != static_cast<std::size_t>(n) ||
    snapshot.lower.size() != static_cast<std::size_t>(m) ||
    snapshot.upper.size() != static_cast<std::size_t>(m) ||
    snapshot.centered_warm_start.size() != static_cast<std::size_t>(n) ||
    snapshot.variable_shift.size() != static_cast<std::size_t>(n) ||
    snapshot.p_row_indices.size() != snapshot.p_values.size() ||
    snapshot.a_row_indices.size() != snapshot.a_values.size())
  {
    result.setup_return_code = -1;
    return result;
  }
  const auto valid_csc = [](const std::vector<c_int> & indptr,
    const std::vector<c_int> & indices, const std::size_t nonzeros,
    const int rows, const bool upper_triangular) {
      if (indptr.empty() || indptr.front() != 0 ||
        indptr.back() != static_cast<c_int>(nonzeros))
      {
        return false;
      }
      for (std::size_t column = 0; column + 1U < indptr.size(); ++column) {
        if (indptr[column] < 0 || indptr[column] > indptr[column + 1U] ||
          indptr[column + 1U] > static_cast<c_int>(nonzeros))
        {
          return false;
        }
        for (c_int offset = indptr[column]; offset < indptr[column + 1U]; ++offset) {
          const c_int row = indices[static_cast<std::size_t>(offset)];
          if (row < 0 || row >= rows ||
            (upper_triangular && static_cast<std::size_t>(row) > column))
          {
            return false;
          }
        }
      }
      return true;
    };
  if (!valid_csc(snapshot.p_column_pointers, snapshot.p_row_indices,
      snapshot.p_values.size(), n, true) ||
    !valid_csc(snapshot.a_column_pointers, snapshot.a_row_indices,
      snapshot.a_values.size(), m, false) ||
    !std::all_of(snapshot.p_values.begin(), snapshot.p_values.end(),
    [](const c_float value) {return std::isfinite(static_cast<double>(value));}) ||
    !std::all_of(snapshot.a_values.begin(), snapshot.a_values.end(),
    [](const c_float value) {return std::isfinite(static_cast<double>(value));}) ||
    !std::all_of(snapshot.q.begin(), snapshot.q.end(),
    [](const c_float value) {return std::isfinite(static_cast<double>(value));}) ||
    !std::all_of(snapshot.lower.begin(), snapshot.lower.end(),
    [](const c_float value) {return !std::isnan(static_cast<double>(value));}) ||
    !std::all_of(snapshot.upper.begin(), snapshot.upper.end(),
    [](const c_float value) {return !std::isnan(static_cast<double>(value));}) ||
    !std::all_of(snapshot.centered_warm_start.begin(), snapshot.centered_warm_start.end(),
    [](const c_float value) {return std::isfinite(static_cast<double>(value));}) ||
    !std::all_of(snapshot.variable_shift.begin(), snapshot.variable_shift.end(),
    [](const double value) {return std::isfinite(value);}))
  {
    result.setup_return_code = -1;
    return result;
  }
  for (int row = 0; row < m; ++row) {
    if (snapshot.lower[static_cast<std::size_t>(row)] >
      snapshot.upper[static_cast<std::size_t>(row)])
    {
      result.setup_return_code = -1;
      return result;
    }
  }

  std::vector<c_int> p_indptr(snapshot.p_column_pointers.begin(), snapshot.p_column_pointers.end());
  std::vector<c_int> p_indices(snapshot.p_row_indices.begin(), snapshot.p_row_indices.end());
  std::vector<c_float> p_values(snapshot.p_values.begin(), snapshot.p_values.end());
  std::vector<c_int> a_indptr(snapshot.a_column_pointers.begin(), snapshot.a_column_pointers.end());
  std::vector<c_int> a_indices(snapshot.a_row_indices.begin(), snapshot.a_row_indices.end());
  std::vector<c_float> a_values(snapshot.a_values.begin(), snapshot.a_values.end());
  std::vector<c_float> q(snapshot.q.begin(), snapshot.q.end());
  std::vector<c_float> lower(snapshot.lower.begin(), snapshot.lower.end());
  std::vector<c_float> upper(snapshot.upper.begin(), snapshot.upper.end());
  std::vector<c_float> warm_start(snapshot.centered_warm_start.begin(),
    snapshot.centered_warm_start.end());
  csc p_matrix{static_cast<c_int>(p_values.size()), static_cast<c_int>(n), static_cast<c_int>(n),
    p_indptr.data(), p_indices.data(), p_values.data(), -1};
  csc a_matrix{static_cast<c_int>(a_values.size()), static_cast<c_int>(m), static_cast<c_int>(n),
    a_indptr.data(), a_indices.data(), a_values.data(), -1};
  OSQPData data{};
  data.n = static_cast<c_int>(n);
  data.m = static_cast<c_int>(m);
  data.P = &p_matrix;
  data.A = &a_matrix;
  data.q = q.data();
  data.l = lower.data();
  data.u = upper.data();

  OSQPSettings settings{};
  osqp_set_default_settings(&settings);
  settings.rho = static_cast<c_float>(snapshot.rho);
  settings.sigma = static_cast<c_float>(snapshot.sigma);
  settings.scaling = static_cast<c_int>(snapshot.scaling);
#if EMBEDDED != 1
  settings.adaptive_rho = static_cast<c_int>(snapshot.adaptive_rho);
  settings.adaptive_rho_interval = static_cast<c_int>(snapshot.adaptive_rho_interval);
  settings.adaptive_rho_tolerance = static_cast<c_float>(snapshot.adaptive_rho_tolerance);
#ifdef PROFILING
  settings.adaptive_rho_fraction = static_cast<c_float>(snapshot.adaptive_rho_fraction);
#endif
#endif
  settings.max_iter = static_cast<c_int>(snapshot.max_iter);
  settings.eps_abs = static_cast<c_float>(snapshot.eps_abs);
  settings.eps_rel = static_cast<c_float>(snapshot.eps_rel);
  settings.eps_prim_inf = static_cast<c_float>(snapshot.eps_prim_inf);
  settings.eps_dual_inf = static_cast<c_float>(snapshot.eps_dual_inf);
  settings.alpha = static_cast<c_float>(snapshot.alpha);
  settings.linsys_solver = static_cast<enum linsys_solver_type>(snapshot.linsys_solver);
#ifndef EMBEDDED
  settings.delta = static_cast<c_float>(snapshot.delta);
  settings.polish = static_cast<c_int>(snapshot.polish);
  settings.polish_refine_iter = static_cast<c_int>(snapshot.polish_refine_iter);
  settings.verbose = static_cast<c_int>(snapshot.verbose);
#endif
  settings.scaled_termination = static_cast<c_int>(snapshot.scaled_termination);
  settings.check_termination = static_cast<c_int>(snapshot.check_termination);
  settings.warm_start = static_cast<c_int>(snapshot.warm_start);
#ifdef PROFILING
  settings.time_limit = static_cast<c_float>(snapshot.time_limit_sec);
#endif

  OSQPWorkspace * raw_workspace = nullptr;
  result.setup_return_code = static_cast<int>(osqp_setup(&raw_workspace, &data, &settings));
  WorkspacePtr workspace{raw_workspace};
  if (result.setup_return_code != 0 || workspace == nullptr) {
    return result;
  }
  result.setup_return_code = static_cast<int>(osqp_warm_start_x(workspace.get(),
      warm_start.data()));
  if (result.setup_return_code != 0) {
    return result;
  }
  result.solve_return_code = static_cast<int>(osqp_solve(workspace.get()));
  if (workspace->info == nullptr) {
    return result;
  }
  result.raw_status_value = static_cast<int>(workspace->info->status_val);
  result.status = classify_stage_status(workspace->info->status_val);
  result.iterations = static_cast<int>(workspace->info->iter);
  result.solve_time_sec = static_cast<double>(workspace->info->solve_time);
  result.primal_residual = static_cast<double>(workspace->info->pri_res);
  result.dual_residual = static_cast<double>(workspace->info->dua_res);
  if (workspace->solution != nullptr && workspace->solution->x != nullptr) {
    result.physical_solution.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      result.physical_solution[static_cast<std::size_t>(i)] =
        static_cast<double>(workspace->solution->x[i]) +
        snapshot.variable_shift[static_cast<std::size_t>(i)];
    }
  }
  return result;
}

QpStageStatus classifyQpSolverStatus(const int raw_status_value) noexcept
{
  return classify_stage_status(static_cast<c_int>(raw_status_value));
}

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
  StageResult primary = solve_stage(
    primary_objective.first, primary_objective.second, base_constraints,
    state.velocity, config_, QpSolveStage::kPrimary);
  result.primary_status = primary.status;
  result.primary_iterations = primary.iterations;
  result.primary_failure_snapshot = std::move(primary.failure_snapshot);
  result.primary_snapshot_disposition = primary.snapshot_disposition;
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
  StageResult secondary = solve_stage(
    secondary_objective.first, secondary_objective.second,
    secondary_constraints, primary.velocity, config_, QpSolveStage::kSecondary);
  result.secondary_status = secondary.status;
  result.secondary_iterations = secondary.iterations;
  result.secondary_failure_snapshot = std::move(secondary.failure_snapshot);
  result.secondary_snapshot_disposition = secondary.snapshot_disposition;
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
  StageResult braking = solve_stage(
    hessian, gradient, hard_constraints, state.velocity, config_, QpSolveStage::kBrake);
  result.primary_status = braking.status;
  result.primary_iterations = braking.iterations;
  result.primary_failure_snapshot = std::move(braking.failure_snapshot);
  result.primary_snapshot_disposition = braking.snapshot_disposition;
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
