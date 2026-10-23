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
/// uses both screen-normal axes and upright roll as primary, with translation secondary;
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
};

enum class QpFailureSnapshotDisposition
{
  kNotNeeded,
  kCaptured,
  kShapeLimitExceeded,
};

inline constexpr std::size_t kQpFailureSnapshotMaximumJoints = 6U;
// Worst controller assembly is bounded by 64 collision rows plus their one
// protected-joint row, 64 compact-radius supports, at most 6 upright/pitch
// rows, 2 face rows, 1 J1 handoff row, 6 joint rows and at most 6 primary
// preservation rows (150 total). Round the snapshot cap to 160; this does not
// change solver admission. Generic oversized tasks retain an explicit
// shape-limit disposition.
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
};

struct QpResult
{
  QpStatus status{QpStatus::kInvalidInput};
  QpStageStatus primary_status{QpStageStatus::kNotRun};
  QpStageStatus secondary_status{QpStageStatus::kNotRun};
  Eigen::VectorXd joint_velocity;
  Eigen::VectorXd achieved_primary_velocity;
  int primary_iterations{0};
  int secondary_iterations{0};
  std::optional<QpFailureSnapshot> primary_failure_snapshot;
  std::optional<QpFailureSnapshot> secondary_failure_snapshot;
  QpFailureSnapshotDisposition primary_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};
  QpFailureSnapshotDisposition secondary_snapshot_disposition{
    QpFailureSnapshotDisposition::kNotNeeded};

  /// True when joint_velocity is a checked command satisfying every hard constraint.
  [[nodiscard]] bool command_available() const noexcept;

  /// True only when both lexicographic stages completed.
  [[nodiscard]] bool fully_solved() const noexcept;
};

/// Two-stage lexicographic joint-velocity optimizer.
///
/// Stage one tracks the primary task under all safety limits. Stage two tracks the
/// secondary task while constraining the primary task-space velocity to remain within
/// primary_preservation_tolerance of stage one's result. OSQP workspaces are local to
/// each call, so this object contains no hidden state and can be tested deterministically.
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
