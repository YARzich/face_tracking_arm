// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

#include <osqp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "face_tracking_arm/detail/scalar_braking_policy.hpp"
#include "face_tracking_arm/detail/feasible_distance_approximation.hpp"

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
  QpStageStatus raw_solver_status{QpStageStatus::kSolverError};
  Eigen::VectorXd velocity;
  Eigen::VectorXd raw_velocity;
  int iterations{0};
  double solve_time_sec{0.0};
  int retry_attempts{0};
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

bool exact_box_single_row_minimizer(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const ConstraintMatrix & constraints,
  const double feasibility_tolerance,
  const std::chrono::steady_clock::time_point deadline,
  Eigen::VectorXd & solution)
{
  const Eigen::Index count = gradient.size();
  if (count <= 0 || hessian.rows() != count || hessian.cols() != count ||
    constraints.matrix.rows() != count + 1 || constraints.matrix.cols() != count ||
    constraints.lower.size() != count + 1 || constraints.upper.size() != count + 1 ||
    !is_finite(hessian) || !is_finite(gradient) || !is_finite(constraints.matrix))
  {
    return false;
  }

  Eigen::VectorXd lower(count);
  Eigen::VectorXd upper(count);
  Eigen::VectorXd diagonal(count);
  Eigen::VectorXd unconstrained(count);
  const double machine_epsilon = std::numeric_limits<double>::epsilon();
  for (Eigen::Index i = 0; i < count; ++i) {
    if (!std::isfinite(constraints.lower[i]) || !std::isfinite(constraints.upper[i]) ||
      constraints.lower[i] > constraints.upper[i] ||
      !std::isfinite(hessian(i, i)) || hessian(i, i) <= 0.0)
    {
      return false;
    }
    for (Eigen::Index j = 0; j < count; ++j) {
      if (i != j && (hessian(i, j) != 0.0 || hessian(j, i) != 0.0)) {
        return false;
      }
      if (constraints.matrix(i, j) != (i == j ? 1.0 : 0.0)) {
        return false;
      }
    }
    diagonal[i] = hessian(i, i);
    lower[i] = constraints.lower[i];
    upper[i] = constraints.upper[i];
    unconstrained[i] = -gradient[i] / diagonal[i];
    if (!std::isfinite(unconstrained[i])) {
      return false;
    }
  }

  const Eigen::VectorXd row = constraints.matrix.row(count).transpose();
  const double row_lower = constraints.lower[count];
  const double row_upper = constraints.upper[count];
  if (!is_finite(row) || std::isnan(row_lower) || std::isnan(row_upper) ||
    row_lower > row_upper)
  {
    return false;
  }
  Eigen::VectorXd candidate = unconstrained.cwiseMax(lower).cwiseMin(upper);
  double lambda = 0.0;
  double direction = 0.0;
  double target = 0.0;
  double row_at_candidate = row.dot(candidate);
  if (!std::isfinite(row_at_candidate)) {
    return false;
  }
  if (row_at_candidate < row_lower) {
    direction = 1.0;
    target = row_lower;
  } else if (row_at_candidate > row_upper) {
    direction = -1.0;
    target = -row_upper;
  } else {
    solution = std::move(candidate);
    return satisfies_constraints(solution, constraints, feasibility_tolerance);
  }
  if (!std::isfinite(target)) {
    return false;
  }

  Eigen::VectorXd signed_row = direction * row;
  const long double n_eps = static_cast<long double>(2 * count + 1) *
    static_cast<long double>(machine_epsilon);
  const long double gamma = n_eps / (1.0L - n_eps);
  long double max_row_value = 0.0L;
  long double row_abs_sum = 0.0L;
  for (Eigen::Index i = 0; i < count; ++i) {
    const double extremal = signed_row[i] >= 0.0 ? upper[i] : lower[i];
    const long double term = static_cast<long double>(signed_row[i]) *
      static_cast<long double>(extremal);
    max_row_value += term;
    row_abs_sum += std::abs(term);
  }
  const long double arithmetic_roundoff = gamma * row_abs_sum;
  if (static_cast<long double>(target) - max_row_value > arithmetic_roundoff) {
    return false;
  }

  std::vector<double> saturation_breakpoints;
  saturation_breakpoints.reserve(static_cast<std::size_t>(count));
  for (Eigen::Index i = 0; i < count; ++i) {
    const double c = signed_row[i];
    if (c > 0.0 && unconstrained[i] < upper[i]) {
      const double breakpoint = (upper[i] - unconstrained[i]) * diagonal[i] / c;
      if (std::isfinite(breakpoint)) {
        saturation_breakpoints.push_back(breakpoint);
      }
    } else if (c < 0.0 && unconstrained[i] > lower[i]) {
      const double breakpoint = (lower[i] - unconstrained[i]) * diagonal[i] / c;
      if (std::isfinite(breakpoint)) {
        saturation_breakpoints.push_back(breakpoint);
      }
    }
  }
  if (saturation_breakpoints.empty()) {
    return false;
  }
  std::sort(saturation_breakpoints.begin(), saturation_breakpoints.end());

  const auto at_multiplier = [&](const double multiplier) {
      return (unconstrained + multiplier * signed_row.cwiseQuotient(diagonal))
             .cwiseMax(lower).cwiseMin(upper);
    };
  double low = 0.0;
  double high = 0.0;
  bool bracket_found = false;
  for (const double breakpoint : saturation_breakpoints) {
    const Eigen::VectorXd at_breakpoint = at_multiplier(breakpoint);
    const double value = signed_row.dot(at_breakpoint);
    if (!std::isfinite(value)) {
      return false;
    }
    if (value >= target) {
      high = breakpoint;
      bracket_found = true;
      break;
    }
    low = breakpoint;
  }
  if (!bracket_found) {
    // The row bound can exceed the exact box extremum only by the bounded
    // floating-point dot-product error checked above. Use that exact box corner.
    lambda = saturation_breakpoints.back();
    candidate.resize(count);
    for (Eigen::Index i = 0; i < count; ++i) {
      candidate[i] = signed_row[i] >= 0.0 ? upper[i] : lower[i];
    }
    const double achieved_at_corner = signed_row.dot(candidate);
    if (!std::isfinite(achieved_at_corner) ||
      static_cast<long double>(target) - static_cast<long double>(achieved_at_corner) >
      arithmetic_roundoff)
    {
      return false;
    }
  } else {
    for (int iteration = 0; iteration < 64; ++iteration) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      const double middle = low + 0.5 * (high - low);
      if (middle == low || middle == high) {
        break;
      }
      const Eigen::VectorXd at_middle = at_multiplier(middle);
      const double value = signed_row.dot(at_middle);
      if (!std::isfinite(value)) {
        return false;
      }
      if (value >= target) {
        high = middle;
      } else {
        low = middle;
      }
    }
    lambda = high;
    candidate = at_multiplier(lambda);
  }
  const double achieved = signed_row.dot(candidate);
  if (!is_finite(candidate) || !std::isfinite(achieved) ||
    static_cast<long double>(target) - static_cast<long double>(achieved) >
    arithmetic_roundoff)
  {
    return false;
  }

  // Verify the diagonal-QP KKT conditions, including the box normal cone.
  const Eigen::VectorXd stationarity = diagonal.cwiseProduct(candidate) +
    gradient - lambda * signed_row;
  for (Eigen::Index i = 0; i < count; ++i) {
    if (lower[i] == upper[i]) {
      continue;
    }
    const double scale = std::max({1.0, std::abs(gradient[i]),
          std::abs(diagonal[i] * candidate[i]), std::abs(lambda * signed_row[i])});
    const double kkt_tolerance = 64.0 * machine_epsilon * scale;
    if (candidate[i] == lower[i]) {
      if (stationarity[i] < -kkt_tolerance) {
        return false;
      }
    } else if (candidate[i] == upper[i]) {
      if (stationarity[i] > kkt_tolerance) {
        return false;
      }
    } else if (std::abs(stationarity[i]) > kkt_tolerance) {
      return false;
    }
  }
  const long double objective = [&]() {
      long double value = 0.0L;
      for (Eigen::Index i = 0; i < count; ++i) {
        const long double x = static_cast<long double>(candidate[i]);
        value += 0.5L * static_cast<long double>(diagonal[i]) * x * x +
          static_cast<long double>(gradient[i]) * x;
      }
      return value;
    }();
  long double box_objective = 0.0L;
  for (Eigen::Index i = 0; i < count; ++i) {
    const long double x = static_cast<long double>(std::clamp(
        unconstrained[i], lower[i], upper[i]));
    box_objective += 0.5L * static_cast<long double>(diagonal[i]) * x * x +
      static_cast<long double>(gradient[i]) * x;
  }
  const long double objective_roundoff = 64.0L * machine_epsilon *
    std::max(1.0L, std::abs(objective));
  if (!std::isfinite(objective) || objective + objective_roundoff < box_objective ||
    !satisfies_constraints(candidate, constraints, feasibility_tolerance))
  {
    return false;
  }
  solution = std::move(candidate);
  return true;
}

StageResult solve_stage_once(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const ConstraintMatrix & constraints,
  const Eigen::VectorXd & warm_start,
  const HierarchicalVelocityQpConfig & config,
  const QpSolveStage solve_stage_kind,
  const std::chrono::steady_clock::time_point deadline,
  const bool primary_rho_retry = false)
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
  if (primary_rho_retry && solve_stage_kind == QpSolveStage::kPrimary) {
    // This bounded primary retry resolves the captured collision-bound corner
    // without changing tolerances, iterations, or the shared solver deadline.
    settings.rho = 100.0;
    settings.adaptive_rho = 0;
  }
  if (solve_stage_kind == QpSolveStage::kDistance) {
    // Captured production distance QPs converge more reliably with a less
    // aggressive penalty update interval. The iteration limit remains the
    // configured limit and the wall-clock budget stays unchanged.
    settings.rho = 2.0;
    settings.adaptive_rho_interval = 50;
  }
  settings.eps_prim_inf = static_cast<c_float>(kSolverPrimalInfeasibilityTolerance);
  settings.eps_abs = static_cast<c_float>(config.solver_absolute_tolerance);
  settings.eps_rel = static_cast<c_float>(config.solver_relative_tolerance);
  settings.time_limit = static_cast<c_float>(std::max(0.0,
      std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count()));

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
    const double remaining_time_sec = std::max(0.0,
      std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
    workspace->settings->time_limit = static_cast<c_float>(remaining_time_sec);
    settings = *workspace->settings;
  }

  if (osqp_warm_start_x(workspace.get(), initial_velocity.data()) != 0) {
    capture_failure(0, -1, OSQP_UNSOLVED, 0.0);
    return result;
  }

  const double remaining_time_sec = std::chrono::duration<double>(
    deadline - std::chrono::steady_clock::now()).count();
  if (remaining_time_sec <= 0.0) {
    result.status = QpStageStatus::kTimeLimit;
    result.raw_solver_status = QpStageStatus::kTimeLimit;
    capture_failure(0, 0, OSQP_UNSOLVED, 0.0);
    return result;
  }
  if (workspace->settings != nullptr) {
    workspace->settings->time_limit = static_cast<c_float>(remaining_time_sec);
    settings = *workspace->settings;
  }

  const c_int solve_result = osqp_solve(workspace.get());
  if (solve_result != 0 || workspace->info == nullptr) {
    capture_failure(0, static_cast<int>(solve_result), OSQP_UNSOLVED, 0.0);
    return result;
  }
  result.status = classify_stage_status(workspace->info->status_val);
  result.raw_solver_status = result.status;
  result.iterations = static_cast<int>(workspace->info->iter);
  result.solve_time_sec = static_cast<double>(workspace->info->solve_time);
  const bool has_raw_solution = workspace->solution != nullptr &&
    workspace->solution->x != nullptr;
  if (result.status != QpStageStatus::kSolved &&
    result.status != QpStageStatus::kSolvedInaccurate)
  {
    if (solve_stage_kind == QpSolveStage::kDistance && has_raw_solution) {
      result.raw_velocity.resize(variable_count);
      for (Eigen::Index index = 0; index < variable_count; ++index) {
        result.raw_velocity[index] = workspace->solution->x[index] + variable_shift[index];
      }
    }
    capture_failure(0, static_cast<int>(solve_result),
      static_cast<int>(workspace->info->status_val),
      static_cast<double>(workspace->info->solve_time));
    return result;
  }
  if (!has_raw_solution) {
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
    if (solve_stage_kind == QpSolveStage::kDistance) {
      result.raw_velocity.resize(variable_count);
      for (Eigen::Index index = 0; index < variable_count; ++index) {
        result.raw_velocity[index] = workspace->solution->x[index] + variable_shift[index];
      }
    }
    result.velocity.resize(0);
    result.status = QpStageStatus::kSolverError;
    capture_failure(0, static_cast<int>(solve_result),
      static_cast<int>(workspace->info->status_val),
      static_cast<double>(workspace->info->solve_time));
  }
  return result;
}

StageResult solve_stage_until(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const ConstraintMatrix & constraints,
  const Eigen::VectorXd & warm_start,
  const HierarchicalVelocityQpConfig & config,
  const QpSolveStage solve_stage_kind,
  const std::chrono::steady_clock::time_point deadline)
{
  StageResult first = solve_stage_once(
    hessian, gradient, constraints, warm_start, config, solve_stage_kind, deadline);

  const bool postcheck_failure =
    first.status == QpStageStatus::kSolverError &&
    (first.raw_solver_status == QpStageStatus::kSolved ||
    first.raw_solver_status == QpStageStatus::kSolvedInaccurate);
  const bool eligible_failure = first.status == QpStageStatus::kInfeasible ||
    first.status == QpStageStatus::kMaxIterations ||
    first.status == QpStageStatus::kTimeLimit || postcheck_failure;
  if (!eligible_failure || std::chrono::steady_clock::now() >= deadline) {
    return first;
  }

  Eigen::VectorXd exact_solution;
  const bool exact_solution_available = exact_box_single_row_minimizer(
    hessian, gradient, constraints, config.solution_feasibility_tolerance,
    deadline, exact_solution);
  if (std::chrono::steady_clock::now() >= deadline) {
    return first;
  }
  if (exact_solution_available && exact_solution.size() == warm_start.size()) {
    for (Eigen::Index index = 0; index < exact_solution.size(); ++index) {
      exact_solution[index] = std::clamp(
        exact_solution[index], constraints.lower[index], constraints.upper[index]);
    }
    if (satisfies_constraints(
        exact_solution, constraints, config.solution_feasibility_tolerance))
    {
      first.status = QpStageStatus::kSolved;
      // Keep raw_solver_status and the first-attempt snapshot as diagnostics. The
      // accepted command came from the exact diagonal box+row minimizer above.
      first.velocity = std::move(exact_solution);
      first.retry_attempts = 1;
      return first;
    }
  }

  if (solve_stage_kind != QpSolveStage::kPrimary ||
    first.status != QpStageStatus::kMaxIterations ||
    std::chrono::steady_clock::now() >= deadline)
  {
    return first;
  }

  StageResult retry = solve_stage_once(
    hessian, gradient, constraints, warm_start, config, solve_stage_kind,
    deadline, true);
  retry.iterations += first.iterations;
  retry.solve_time_sec += first.solve_time_sec;
  retry.retry_attempts += first.retry_attempts + 1;
  // Report the original failure even when the alternate settings recover.
  retry.failure_snapshot = std::move(first.failure_snapshot);
  retry.snapshot_disposition = first.snapshot_disposition;
  return retry;
}

StageResult solve_stage(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const ConstraintMatrix & constraints,
  const Eigen::VectorXd & warm_start,
  const HierarchicalVelocityQpConfig & config,
  const QpSolveStage solve_stage_kind)
{
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(config.solver_time_limit_sec));
  return solve_stage_until(
    hessian, gradient, constraints, warm_start, config, solve_stage_kind, deadline);
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

std::optional<detail::FeasibleDistanceApproximation>
detail::minimizeFeasibleDistanceSegment(
  const QpStageStatus stage_status, const QpStageStatus raw_solver_status,
  const Eigen::MatrixXd & hessian, const Eigen::VectorXd & gradient,
  const Eigen::MatrixXd & constraints, const Eigen::VectorXd & lower,
  const Eigen::VectorXd & upper, const Eigen::VectorXd & warm,
  const Eigen::VectorXd & raw, const double feasibility_tolerance)
{
  const bool retryable_failure = stage_status == QpStageStatus::kMaxIterations ||
    stage_status == QpStageStatus::kTimeLimit;
  const bool postcheck_rejected = stage_status == QpStageStatus::kSolverError &&
    (raw_solver_status == QpStageStatus::kSolved ||
    raw_solver_status == QpStageStatus::kSolvedInaccurate);
  const Eigen::Index joint_count = warm.size();
  if ((!retryable_failure && !postcheck_rejected) || joint_count <= 0 ||
    raw.size() != joint_count || hessian.rows() != joint_count ||
    hessian.cols() != joint_count || gradient.size() != joint_count ||
    constraints.cols() != joint_count || constraints.rows() < joint_count ||
    lower.size() != constraints.rows() || upper.size() != constraints.rows() ||
    !std::isfinite(feasibility_tolerance) || feasibility_tolerance < 0.0 ||
    !is_finite(hessian) || !is_finite(gradient) || !is_finite(constraints) ||
    !is_finite(warm) || !is_finite(raw))
  {
    return std::nullopt;
  }
  for (Eigen::Index row = 0; row < lower.size(); ++row) {
    if (std::isnan(lower[row]) || std::isnan(upper[row]) || lower[row] > upper[row]) {
      return std::nullopt;
    }
  }

  const Eigen::VectorXd warm_rows = constraints * warm;
  if (!is_finite(warm_rows)) {
    return std::nullopt;
  }
  double alpha_upper = 1.0;
  const Eigen::VectorXd direction = raw - warm;
  const Eigen::VectorXd row_direction = constraints * direction;
  for (Eigen::Index row = 0; row < constraints.rows(); ++row) {
    const double tolerance = row < joint_count ? 0.0 : feasibility_tolerance;
    const double row_lower = lower[row] - tolerance;
    const double row_upper = upper[row] + tolerance;
    const double start = warm_rows[row];
    const double delta = row_direction[row];
    if ((std::isfinite(row_lower) && start < row_lower) ||
      (std::isfinite(row_upper) && start > row_upper))
    {
      return std::nullopt;
    }
    if (delta > 0.0 && std::isfinite(row_upper)) {
      alpha_upper = std::min(alpha_upper, (row_upper - start) / delta);
    } else if (delta < 0.0 && std::isfinite(row_lower)) {
      alpha_upper = std::min(alpha_upper, (row_lower - start) / delta);
    }
  }
  alpha_upper = std::clamp(alpha_upper, 0.0, 1.0);
  if (alpha_upper > 0.0 && alpha_upper < 1.0) {
    alpha_upper = std::nextafter(alpha_upper, 0.0);
  }

  const Eigen::MatrixXd symmetric_hessian = 0.5 * (hessian + hessian.transpose());
  const double quadratic = direction.dot(symmetric_hessian * direction);
  const double linear = direction.dot(symmetric_hessian * warm + gradient);
  const double warm_cost = 0.5 * warm.dot(symmetric_hessian * warm) + gradient.dot(warm);
  if (!std::isfinite(quadratic) || !std::isfinite(linear) || !std::isfinite(warm_cost)) {
    return std::nullopt;
  }
  double alpha = 0.0;
  if (quadratic > 0.0) {
    alpha = std::clamp(-linear / quadratic, 0.0, alpha_upper);
  } else if (linear < 0.0) {
    alpha = alpha_upper;
  }

  detail::FeasibleDistanceApproximation approximation;
  approximation.alpha = alpha;
  approximation.velocity = warm + alpha * direction;
  for (Eigen::Index joint = 0; joint < joint_count; ++joint) {
    approximation.velocity[joint] = std::clamp(
      approximation.velocity[joint], lower[joint], upper[joint]);
  }
  const Eigen::VectorXd candidate_rows = constraints * approximation.velocity;
  if (!is_finite(candidate_rows)) {
    return std::nullopt;
  }
  for (Eigen::Index row = 0; row < constraints.rows(); ++row) {
    if ((std::isfinite(lower[row]) &&
      candidate_rows[row] < lower[row] - feasibility_tolerance) ||
      (std::isfinite(upper[row]) &&
      candidate_rows[row] > upper[row] + feasibility_tolerance))
    {
      return std::nullopt;
    }
  }
  const double candidate_cost =
    0.5 * approximation.velocity.dot(symmetric_hessian * approximation.velocity) +
    gradient.dot(approximation.velocity);
  if (!std::isfinite(candidate_cost) || candidate_cost > warm_cost) {
    approximation.velocity = warm;
    approximation.alpha = 0.0;
  }
  return approximation;
}

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
         status == QpStatus::kPrimaryOnly || status == QpStatus::kPartial;
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
  const auto optional_objective_valid = [&](
    const Eigen::MatrixXd & matrix, const Eigen::VectorXd & reference,
    const Eigen::VectorXd & weights) {
      return matrix.rows() == 0 ?
             (matrix.cols() == 0 || matrix.cols() == joint_count) &&
             reference.size() == 0 && weights.size() == 0 :
             valid_objective(matrix, reference, weights, joint_count, true);
    };
  if (!valid_objective(
      task.primary_matrix, task.primary_reference, task.primary_weights,
      joint_count, false) ||
    !valid_objective(
      task.secondary_matrix, task.secondary_reference, task.secondary_weights,
      joint_count, true) ||
    !optional_objective_valid(
      task.tertiary_matrix, task.tertiary_reference, task.tertiary_weights) ||
    !optional_objective_valid(
      task.distance_matrix, task.distance_reference, task.distance_weights))
  {
    return result;
  }

  const auto valid_constraint = [joint_count](const LinearVelocityConstraint & constraint) {
      return constraint.coefficients.size() == joint_count &&
             is_finite(constraint.coefficients) && !std::isnan(constraint.lower_bound) &&
             !std::isnan(constraint.upper_bound) &&
             constraint.lower_bound <= constraint.upper_bound;
    };
  for (const LinearVelocityConstraint & constraint : task.preferred_primary_constraints) {
    if (!valid_constraint(constraint)) {
      return result;
    }
  }
  for (const LinearVelocityConstraint & constraint : task.secondary_constraints) {
    if (!valid_constraint(constraint)) {
      return result;
    }
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
  if (task.distance_matrix.rows() == 1 && task.distance_reference.size() == 1 &&
    std::isfinite(task.distance_reference[0]))
  {
    result.requested_distance_rate = task.distance_reference[0];
  }
  const auto record_achieved_distance_rate = [&]() {
      if (result.command_available() && task.distance_matrix.rows() == 1 &&
        result.joint_velocity.size() == joint_count)
      {
        const double rate = task.distance_matrix.row(0).dot(result.joint_velocity);
        if (std::isfinite(rate)) {
          result.achieved_distance_rate = rate;
        }
      }
    };

  // Carry preferred GOOD rows and lower-rank constraints through every lower solve.
  // Empty objectives are skipped, so an empty objective never preserves damping.
  ConstraintMatrix lower_constraints = base_constraints;
  const auto append_task_constraints = [&](
    ConstraintMatrix & output, const auto & task_constraints) {
      if (task_constraints.empty()) {
        return;
      }
      const Eigen::Index old_rows = output.matrix.rows();
      const Eigen::Index extra_rows =
        static_cast<Eigen::Index>(task_constraints.size());
      output.matrix.conservativeResize(old_rows + extra_rows, joint_count);
      output.lower.conservativeResize(old_rows + extra_rows);
      output.upper.conservativeResize(old_rows + extra_rows);
      for (Eigen::Index row = 0; row < extra_rows; ++row) {
        const auto & constraint = task_constraints[static_cast<std::size_t>(row)];
        output.matrix.row(old_rows + row) = constraint.coefficients.transpose();
        output.lower[old_rows + row] = constraint.lower_bound;
        output.upper[old_rows + row] = constraint.upper_bound;
      }
    };
  append_task_constraints(lower_constraints, task.preferred_primary_constraints);
  append_task_constraints(lower_constraints, task.secondary_constraints);

  const auto with_preservation = [&](
    const std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> & preserved) {
      ConstraintMatrix output;
      Eigen::Index preserved_rows = 0;
      for (const auto & entry : preserved) {
        preserved_rows += entry.first.rows();
      }
      const Eigen::Index base_rows = lower_constraints.matrix.rows();
      output.matrix.resize(base_rows + preserved_rows, joint_count);
      output.lower.resize(base_rows + preserved_rows);
      output.upper.resize(base_rows + preserved_rows);
      output.matrix.topRows(base_rows) = lower_constraints.matrix;
      output.lower.head(base_rows) = lower_constraints.lower;
      output.upper.head(base_rows) = lower_constraints.upper;
      Eigen::Index row = base_rows;
      for (const auto & entry : preserved) {
        const Eigen::Index count = entry.first.rows();
        output.matrix.middleRows(row, count) = entry.first;
        output.lower.segment(row, count) = entry.second.array() -
          config_.primary_preservation_tolerance;
        output.upper.segment(row, count) = entry.second.array() +
          config_.primary_preservation_tolerance;
        row += count;
      }
      return output;
    };

  const auto succeeded = [](const QpStageStatus status) {
      return status == QpStageStatus::kSolved ||
             status == QpStageStatus::kSolvedInaccurate;
    };
  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> preserved;
  const auto run_rank = [&](
    const Eigen::MatrixXd & matrix, const Eigen::VectorXd & reference,
    const Eigen::VectorXd & weights, const QpSolveStage stage,
    const Eigen::VectorXd & warm_start) {
      const auto objective = make_tracking_objective(
        matrix, reference, weights, state.velocity, config_);
      return solve_stage(
        objective.first, objective.second, with_preservation(preserved),
        warm_start, config_, stage);
    };

  StageResult primary;
  const bool has_preferred_primary_constraints =
    !task.preferred_primary_constraints.empty();
  {
    const auto objective = make_tracking_objective(
      task.primary_matrix, task.primary_reference, task.primary_weights,
      state.velocity, config_);
    ConstraintMatrix guarded_constraints = base_constraints;
    if (has_preferred_primary_constraints) {
      append_task_constraints(guarded_constraints, task.preferred_primary_constraints);
    }
    primary = solve_stage(
      objective.first, objective.second, guarded_constraints, state.velocity,
      config_, QpSolveStage::kPrimary);
    if (has_preferred_primary_constraints) {
      result.preferred_primary_status = primary.status;
      result.primary_iterations = primary.iterations;
      result.primary_retry_count = primary.retry_attempts;
      result.primary_solve_time_sec = primary.solve_time_sec;
      result.primary_failure_snapshot = std::move(primary.failure_snapshot);
      result.primary_snapshot_disposition = primary.snapshot_disposition;
      if (!succeeded(primary.status)) {
        result.preferred_primary_fallback_used = true;
        StageResult unguarded = solve_stage(
          objective.first, objective.second, base_constraints, state.velocity,
          config_, QpSolveStage::kPrimary);
        result.primary_iterations += unguarded.iterations;
        result.primary_retry_count += unguarded.retry_attempts;
        result.primary_solve_time_sec += unguarded.solve_time_sec;
        if (!result.primary_failure_snapshot.has_value()) {
          result.primary_failure_snapshot = std::move(unguarded.failure_snapshot);
          result.primary_snapshot_disposition = unguarded.snapshot_disposition;
        }
        primary = std::move(unguarded);
      }
    }
  }
  result.primary_status = primary.status;
  if (!has_preferred_primary_constraints) {
    result.primary_iterations = primary.iterations;
    result.primary_retry_count = primary.retry_attempts;
    result.primary_solve_time_sec = primary.solve_time_sec;
    result.primary_failure_snapshot = std::move(primary.failure_snapshot);
    result.primary_snapshot_disposition = primary.snapshot_disposition;
  }
  if (!succeeded(primary.status)) {
    result.status = top_level_status(primary.status);
    return result;
  }
  result.joint_velocity = primary.velocity;
  result.achieved_primary_velocity = task.primary_matrix * primary.velocity;
  bool any_rank_inaccurate = primary.status == QpStageStatus::kSolvedInaccurate;
  preserved.emplace_back(task.primary_matrix, result.achieved_primary_velocity);
  if (result.preferred_primary_fallback_used) {
    result.status = QpStatus::kPrimaryOnly;
    record_achieved_distance_rate();
    return result;
  }

  const bool has_later_objective = task.secondary_matrix.rows() > 0 ||
    task.tertiary_matrix.rows() > 0 || task.distance_matrix.rows() > 0;
  const bool needs_constraints_only_secondary =
    task.secondary_matrix.rows() == 0 &&
    (!task.preferred_primary_constraints.empty() || !task.secondary_constraints.empty()) &&
    !has_later_objective;
  Eigen::VectorXd last_velocity = primary.velocity;

  const auto run_lower_rank = [&](
    const Eigen::MatrixXd & matrix, const Eigen::VectorXd & reference,
    const Eigen::VectorXd & weights, const QpSolveStage stage,
    QpStageStatus & stage_status, int & iterations, double & solve_time_sec,
    int & retry_count,
    std::optional<QpFailureSnapshot> & failure_snapshot,
    QpFailureSnapshotDisposition & snapshot_disposition,
    const bool constraints_only = false) {
      if (matrix.rows() == 0 && !constraints_only) {
        return true;
      }
      const bool idle_constraints_only_rank = constraints_only &&
        task.primary_reference.size() > 0 && task.primary_reference.isZero(0.0);
      StageResult stage_result;
      if (idle_constraints_only_rank) {
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
          joint_count, joint_count);
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(joint_count);
        const Eigen::VectorXd unit_weights = Eigen::VectorXd::Ones(joint_count);
        stage_result = run_rank(identity, zero, unit_weights, stage, last_velocity);
      } else {
        stage_result = run_rank(matrix, reference, weights, stage, last_velocity);
      }
      stage_status = stage_result.status;
      iterations = stage_result.iterations;
      solve_time_sec = stage_result.solve_time_sec;
      retry_count = stage_result.retry_attempts;
      failure_snapshot = std::move(stage_result.failure_snapshot);
      snapshot_disposition = stage_result.snapshot_disposition;
      if (!succeeded(stage_status)) {
        return false;
      }
      last_velocity = std::move(stage_result.velocity);
      result.joint_velocity = last_velocity;
      result.achieved_primary_velocity = task.primary_matrix * last_velocity;
      any_rank_inaccurate = any_rank_inaccurate ||
        stage_status == QpStageStatus::kSolvedInaccurate;
      if (matrix.rows() > 0) {
        preserved.emplace_back(matrix, matrix * last_velocity);
      }
      return true;
    };

  if (!run_lower_rank(
      task.secondary_matrix, task.secondary_reference, task.secondary_weights,
      QpSolveStage::kSecondary, result.secondary_status, result.secondary_iterations,
      result.secondary_solve_time_sec,
      result.secondary_retry_count,
      result.secondary_failure_snapshot, result.secondary_snapshot_disposition,
      needs_constraints_only_secondary))
  {
    result.status = QpStatus::kPrimaryOnly;
    record_achieved_distance_rate();
    return result;
  }
  if (!run_lower_rank(
      task.tertiary_matrix, task.tertiary_reference, task.tertiary_weights,
      QpSolveStage::kTertiary, result.tertiary_status, result.tertiary_iterations,
      result.tertiary_solve_time_sec,
      result.tertiary_retry_count,
      result.tertiary_failure_snapshot, result.tertiary_snapshot_disposition))
  {
    result.status = QpStatus::kPartial;
    record_achieved_distance_rate();
    return result;
  }
  if (task.distance_matrix.rows() > 0) {
    const auto distance_objective = make_tracking_objective(
      task.distance_matrix, task.distance_reference, task.distance_weights,
      state.velocity, config_);
    const ConstraintMatrix distance_constraints = with_preservation(preserved);
    StageResult distance = solve_stage(
      distance_objective.first, distance_objective.second, distance_constraints,
      last_velocity, config_, QpSolveStage::kDistance);
    result.distance_status = distance.status;
    result.distance_iterations += distance.iterations;
    result.distance_retry_count += distance.retry_attempts;
    result.distance_solve_time_sec += distance.solve_time_sec;
    result.distance_failure_snapshot = std::move(distance.failure_snapshot);
    result.distance_snapshot_disposition = distance.snapshot_disposition;
    if (succeeded(distance.status)) {
      last_velocity = std::move(distance.velocity);
      result.joint_velocity = last_velocity;
      result.achieved_primary_velocity = task.primary_matrix * last_velocity;
      any_rank_inaccurate = any_rank_inaccurate ||
        distance.status == QpStageStatus::kSolvedInaccurate;
    } else {
      const auto approximation = detail::minimizeFeasibleDistanceSegment(
        distance.status, distance.raw_solver_status,
        distance_objective.first, distance_objective.second,
        distance_constraints.matrix, distance_constraints.lower,
        distance_constraints.upper, last_velocity, distance.raw_velocity,
        config_.solution_feasibility_tolerance);
      if (approximation.has_value() && satisfies_constraints(
          approximation->velocity, distance_constraints,
          config_.solution_feasibility_tolerance))
      {
        result.joint_velocity = approximation->velocity;
        result.achieved_primary_velocity = task.primary_matrix * result.joint_velocity;
        result.distance_feasible_approximation_used = true;
        result.distance_feasible_approximation_alpha = approximation->alpha;
      }
      result.status = QpStatus::kPartial;
      record_achieved_distance_rate();
      return result;
    }
  }

  result.achieved_primary_velocity = task.primary_matrix * result.joint_velocity;
  result.status = any_rank_inaccurate ? QpStatus::kSolvedInaccurate : QpStatus::kSolved;
  record_achieved_distance_rate();
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
  result.primary_retry_count = braking.retry_attempts;
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
