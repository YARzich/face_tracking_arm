// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__HIERARCHICAL_VELOCITY_QP_HPP_
#define FACE_TRACKING_ARM__HIERARCHICAL_VELOCITY_QP_HPP_

#include <Eigen/Core>

#include <limits>
#include <vector>

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
/// The primary rows normally represent the two observable components of screen-normal
/// angular velocity. The secondary rows normally contain translation and screen roll.
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
  double primary_preservation_tolerance{2.0e-3};
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
  kSolverError,
};

enum class QpStatus
{
  kSolved,
  kSolvedInaccurate,
  /// The primary command is valid and safe, but the secondary stage failed.
  kPrimaryOnly,
  kInvalidInput,
  kInfeasible,
  kMaxIterations,
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
