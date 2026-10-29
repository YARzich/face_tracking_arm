// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__HIERARCHICAL_VELOCITY_QP_HPP_
#define FACE_TRACKING_ARM__HIERARCHICAL_VELOCITY_QP_HPP_

#include <Eigen/Core>
#include <osqp.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control
{

/// One hard inequality on the commanded joint velocity: lower <= a * qdot <= upper.
struct LinearVelocityConstraint
{
  Eigen::VectorXd coefficients;
  double lower_bound{-std::numeric_limits<double>::infinity()};
  double upper_bound{std::numeric_limits<double>::infinity()};
};

/// State at the end of the controller's command queue.
struct JointMotionState
{
  Eigen::VectorXd position;
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
};

/// Symmetric magnitude limits and hard position bounds for every controlled joint.
struct JointMotionLimits
{
  Eigen::VectorXd lower_position;
  Eigen::VectorXd upper_position;
  /// Preferred interior margin for posture selection; never a physical startup limit.
  Eigen::VectorXd position_margin;
  Eigen::VectorXd max_velocity;
  Eigen::VectorXd max_acceleration;
  Eigen::VectorXd max_jerk;
  /// Shared, dimensionally typed arithmetic policy for position/velocity/acceleration/jerk.
  /// Required by command generation; solver residual remains in QP configuration.
  std::optional<NumericalPolicy> numerical_policy;
};

struct JointCenteringGuidance
{
  double activation{0.0};
  double velocity{0.0};
};

/// Smooth inward preference near a bounded joint's ends, without wrapping its position.
[[nodiscard]] JointCenteringGuidance jointCenteringGuidance(
  double position, double lower, double upper, double activation_fraction,
  double maximum_speed);

/// Task-space velocity objectives, already expressed in a common command frame.
///
/// The primary rows normally represent task-critical motion. Ordinary FACE tracking
/// uses both screen-normal axes and upright roll as primary. Face-distance and
/// posture preferences occupy explicit lower ranks; GOOD comfort rows are preferred
/// hard rows for primary and lower ranks, with an unguarded primary fallback only when
/// the preferred rows conflict with physical constraints or the guarded solve fails.
/// compact PATH tracks the checked joint-space path as primary and enforces upright roll,
/// pitch, and radius through hard velocity constraints. RETURN_FACE retains the roll-primary
/// reacquisition task. Optional gaze and fixed-orientation tasks retain their own two-axis
/// pointing primary and may place translation and screen roll in secondary.
/// Each weight is a positive residual multiplier (the solver minimizes ||W(Ax-b)||^2).
struct HierarchicalVelocityTask
{
  Eigen::MatrixXd primary_matrix;
  Eigen::VectorXd primary_reference;
  Eigen::VectorXd primary_weights;
  Eigen::MatrixXd secondary_matrix;
  Eigen::VectorXd secondary_reference;
  Eigen::VectorXd secondary_weights;
  /// Optional GOOD-mode rows preferred by primary and retained by lower ranks on
  /// success. If this constrained primary fails, the solver retries against physical
  /// constraints alone and returns primary-only; these rows never relax physical bounds.
  std::vector<LinearVelocityConstraint> preferred_primary_constraints;
  /// Hard interval constraints owned by the secondary level. They do not constrain
  /// the primary solve; if incompatible with its achieved command, the solver keeps
  /// the primary-only command instead of letting a secondary objective violate them.
  std::vector<LinearVelocityConstraint> secondary_constraints;
  /// Optional lower-priority preference, normally elbow-side and joint-margin
  /// objectives. It is solved after secondary and preserved by distance.
  Eigen::MatrixXd tertiary_matrix;
  Eigen::VectorXd tertiary_reference;
  Eigen::VectorXd tertiary_weights;
  /// Optional final best-effort objective. Unreachable references remain a valid
  /// least-squares request and must not become hard constraints.
  Eigen::MatrixXd distance_matrix;
  Eigen::VectorXd distance_reference;
  Eigen::VectorXd distance_weights;
};

struct HierarchicalVelocityQpConfig
{
  double period_sec{0.01};
  /// Time during which adverse queued motion may still execute. The safety envelope
  /// conservatively rounds it up to complete controller periods.
  double residual_command_latency_sec{0.02};
  double velocity_regularization{1.0e-6};
  double continuity_weight{1.0e-3};
  double primary_preservation_tolerance{2.0e-4};
  double solution_feasibility_tolerance{2.0e-5};
  double solver_absolute_tolerance{1.0e-6};
  double solver_relative_tolerance{1.0e-6};
  double solver_time_limit_sec{0.003};
  int solver_max_iterations{500};
};

enum class QpStageStatus
{
  kNotRun,
  kSolved,
  kSolvedInaccurate,
  kInfeasible,
  kMaxIterations,
  kTimeLimit,
  kSolverError,
};

enum class QpSolveStage
{
  kPrimary,
  kSecondary,
  kBrake,
  kTertiary,
  kDistance,
};

enum class QpFailureSnapshotDisposition
{
  kNotNeeded,
  kCaptured,
  kShapeLimitExceeded,
};

inline constexpr std::size_t kQpFailureSnapshotMaximumJoints = 6U;
// The validated compact-path worst case is bounded by 64 radius supports plus
// its upright/physical rows; FACE has at most 64 collision rows plus its
// protected-joint row, tracking guards, and preservation rows for each rank.
// Keep the established 160-row cap; it affects diagnostics only, never admission.
inline constexpr std::size_t kQpFailureSnapshotMaximumRows = 160U;
inline constexpr std::size_t kQpFailureSnapshotMaximumANonzeros =
  kQpFailureSnapshotMaximumRows * kQpFailureSnapshotMaximumJoints;

/// Exact, in-memory inputs to one failed OSQP stage. Arrays use OSQP's CSC order;
/// `variable_shift` maps the centered solver variable back to physical joint speed.
/// Snapshots are created only on failure by moving buffers already owned by solve().
/// Storage is bounded to 6 joints, 160 assembled rows and dense A (960 nnz),
/// plus dense upper P (21 nnz). This covers the validated controller maximum
/// of 64 collision rows and 64 compact-radius support rows plus other
/// motion-reference constraints. Oversized
/// generic calls report a shape-limit disposition instead of losing the reason.
struct QpFailureSnapshot
{
  QpSolveStage stage{QpSolveStage::kPrimary};
  QpStageStatus classified_status{QpStageStatus::kSolverError};
  int setup_return_code{0};
  int solve_return_code{0};
  int raw_status_value{0};
  std::string osqp_version;
  std::size_t c_int_size{0U};
  std::size_t c_float_size{0U};
  int variable_count{0};
  int constraint_count{0};
  std::vector<c_int> p_column_pointers;
  std::vector<c_int> p_row_indices;
  std::vector<c_float> p_values;
  std::vector<c_int> a_column_pointers;
  std::vector<c_int> a_row_indices;
  std::vector<c_float> a_values;
  std::vector<c_float> q;
  std::vector<c_float> lower;
  std::vector<c_float> upper;
  std::vector<c_float> centered_warm_start;
  std::vector<double> variable_shift;
  // Exact effective settings, including values supplied by OSQP defaults.
  double rho{0.0};
  double sigma{0.0};
  int scaling{0};
  int adaptive_rho{0};
  int adaptive_rho_interval{0};
  double adaptive_rho_tolerance{0.0};
  double adaptive_rho_fraction{0.0};
  int max_iter{0};
  double eps_abs{0.0};
  double eps_rel{0.0};
  double eps_prim_inf{0.0};
  double eps_dual_inf{0.0};
  double alpha{0.0};
  int linsys_solver{0};
  double delta{0.0};
  int polish{0};
  int polish_refine_iter{0};
  int verbose{0};
  int scaled_termination{0};
  int check_termination{0};
  int warm_start{0};
  double time_limit_sec{0.0};
  double observed_solve_time_sec{0.0};
  double solver_budget_sec{0.0};
};

struct QpReplayResult
{
  QpStageStatus status{QpStageStatus::kSolverError};
  int setup_return_code{0};
  int solve_return_code{0};
  int raw_status_value{0};
  int iterations{0};
  double solve_time_sec{0.0};
  double primal_residual{0.0};
  double dual_residual{0.0};
  bool compatible_solver_build{false};
  std::vector<double> physical_solution;
};

/// Replays the captured matrix problem with its recorded effective settings.
/// Time-limit outcomes are inherently machine/scheduling dependent.
[[nodiscard]] QpReplayResult replayQpFailureSnapshot(const QpFailureSnapshot & snapshot);
[[nodiscard]] QpStageStatus classifyQpSolverStatus(int raw_status_value) noexcept;

enum class QpStatus
{
  kSolved,
  kSolvedInaccurate,
  /// The primary command is valid and safe, but the secondary stage failed.
  kPrimaryOnly,
  kInvalidInput,
  kInfeasible,
  kMaxIterations,
  kTimeLimit,
  kSolverError,
  /// A valid command is available, but a later optional objective stage failed.
  kPartial,
};

struct QpResult
{
  QpStatus status{QpStatus::kInvalidInput};
  QpStageStatus primary_status{QpStageStatus::kNotRun};
  /// Status of the guarded primary attempt; kNotRun when no preferred rows exist.
  QpStageStatus preferred_primary_status{QpStageStatus::kNotRun};
  bool preferred_primary_fallback_used{false};
  QpStageStatus secondary_status{QpStageStatus::kNotRun};
  QpStageStatus tertiary_status{QpStageStatus::kNotRun};
  QpStageStatus distance_status{QpStageStatus::kNotRun};
  /// True when the final distance command used a checked feasible segment approximation.
  bool distance_feasible_approximation_used{false};
  /// Selected segment fraction; meaningful only when the approximation flag is true.
  double distance_feasible_approximation_alpha{0.0};
  Eigen::VectorXd joint_velocity;
  Eigen::VectorXd achieved_primary_velocity;
  int primary_iterations{0};
  int secondary_iterations{0};
  int tertiary_iterations{0};
  int distance_iterations{0};
  /// Number of bounded post-solver recovery methods used by each stage.
  int primary_retry_count{0};
  int secondary_retry_count{0};
  int tertiary_retry_count{0};
  int distance_retry_count{0};
  double primary_solve_time_sec{0.0};
  double secondary_solve_time_sec{0.0};
  double tertiary_solve_time_sec{0.0};
  double distance_solve_time_sec{0.0};
  std::optional<double> requested_distance_rate;
  std::optional<double> achieved_distance_rate;
  std::optional<QpFailureSnapshot> primary_failure_snapshot;
  std::optional<QpFailureSnapshot> secondary_failure_snapshot;
  std::optional<QpFailureSnapshot> tertiary_failure_snapshot;
  std::optional<QpFailureSnapshot> distance_failure_snapshot;
  QpFailureSnapshotDisposition primary_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};
  QpFailureSnapshotDisposition secondary_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};
  QpFailureSnapshotDisposition tertiary_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};
  QpFailureSnapshotDisposition distance_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};

  /// True when joint_velocity is a checked command satisfying every hard constraint.
  [[nodiscard]] bool command_available() const noexcept;

  /// True only when every requested lexicographic stage completed.
  [[nodiscard]] bool fully_solved() const noexcept;
};

/// Lexicographic optimizer with optional secondary, tertiary and final distance ranks.
///
/// Each solved objective rank is preserved by lower ranks within
/// primary_preservation_tolerance. Secondary-level interval constraints apply after
/// the primary rank. Failed later stages retain the last valid command. Distance is a
/// soft final objective. OSQP workspaces are local to each call, so this object
/// contains no hidden state and can be tested deterministically.
/// Joint velocity and position bounds include jerk-limited braking invariants using
/// the queue's discrete backward-Euler update. Residual latency is conservatively
/// rounded up to whole controller ticks rather than integrated continuously.
class HierarchicalVelocityQp final
{
public:
  explicit HierarchicalVelocityQp(HierarchicalVelocityQpConfig config = {});

  [[nodiscard]] QpResult solve(
    const HierarchicalVelocityTask & task,
    const JointMotionState & state,
    const JointMotionLimits & limits,
    const std::vector<LinearVelocityConstraint> & constraints = {}) const;

  /// Minimize commanded speed while retaining the same hard motion and safety limits.
  /// This is a jerk-limited braking step, not an instantaneous zero command.
  [[nodiscard]] QpResult brake(
    const JointMotionState & state,
    const JointMotionLimits & limits,
    const std::vector<LinearVelocityConstraint> & constraints = {}) const;

  [[nodiscard]] const HierarchicalVelocityQpConfig & config() const noexcept;

private:
  HierarchicalVelocityQpConfig config_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__HIERARCHICAL_VELOCITY_QP_HPP_
