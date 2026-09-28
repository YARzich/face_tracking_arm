// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COLLISION_CONSTRAINTS_HPP_
#define FACE_TRACKING_ARM__COLLISION_CONSTRAINTS_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace moveit::core
{
class JointModelGroup;
class RobotState;
class RobotModel;
}  // namespace moveit::core

namespace planning_scene
{
class PlanningScene;
}  // namespace planning_scene

namespace face_tracking_arm::control
{

/// Immutable safety profile used to turn PlanningScene distances into QP rows.
struct CollisionConstraintConfig
{
  double hard_clearance_m{0.015};
  double query_distance_m{0.200};
  /// Bound on the infinity norm of command-to-measured joint-position error.
  double tracking_position_error_bound_rad{0.006};
  /// Distance reserve for collision-backend and floating-point uncertainty.
  double numerical_distance_reserve_m{0.0001};
  /// Conservative distance Lipschitz bound for every pair without a tighter profile.
  double default_distance_lipschitz_m_per_rad{2.6193};
  /// Calibrated bound for the monitor-to-near-link pair.
  double monitor_near_distance_lipschitz_m_per_rad{0.4543};
  std::size_t maximum_constraint_rows{24};
  std::size_t maximum_contacts_per_pair{4};
  double minimum_gradient_norm{1.0e-8};
  /// Optional soft preference to preserve room before hard braking becomes necessary.
  double separation_buffer_m{0.0};
  double separation_max_velocity_mps{0.06};
  double separation_lookahead_sec{0.0};

  std::string monitor_link_name{"monitor_link"};
  std::string monitor_near_link_name{"link4"};
  std::string invariant_mount_neighbor_name{"link5"};

  std::string protected_joint_name{"joint5"};
  double protected_joint_lower_rad{-1.60};
  double protected_joint_upper_rad{1.60};
  double residual_latency_sec{0.020};

  /// Geometry-derived pair bounds; an empty second name denotes a world pair.
  std::map<std::pair<std::string, std::string>, double> distance_bounds;
};

/// Derive conservative distance sensitivity from the loaded collision geometry.
/// Called once at startup, outside the control loop; supports fixed/revolute chains.
void configureGeometryBounds(
  CollisionConstraintConfig & config, const moveit::core::RobotModel & model,
  const moveit::core::JointModelGroup & group);

enum class CollisionPairKind
{
  kSelf,
  kWorld,
};

/// A deterministic, canonical representation of one nearest-distance result.
struct CollisionPairDiagnostic
{
  CollisionPairKind kind{CollisionPairKind::kSelf};
  std::string first_body;
  std::string second_body;
  double distance_m{std::numeric_limits<double>::infinity()};
  double required_clearance_m{std::numeric_limits<double>::infinity()};
  double distance_lipschitz_m_per_rad{std::numeric_limits<double>::infinity()};
  double gradient_norm{0.0};
  double jerk_aware_safe_approach_speed_mps{0.0};
  double minimum_reachable_approach_speed_mps{0.0};
  double maximum_reachable_approach_speed_mps{0.0};
  double applied_approach_speed_cap_mps{0.0};
  bool constraint_added{false};
  bool invariant_pair{false};
  bool robust_clearance_violated{false};
  bool outside_viability{false};
  /// Variables that can change this pair's relative transform, in query group order.
  std::vector<bool> relative_motion_variables;
};

struct CollisionConstraintDiagnostics
{
  double self_query_ms{0.0};
  double world_query_ms{0.0};
  bool valid{false};
  bool collision_detected{false};
  bool hard_clearance_violated{false};
  bool robust_clearance_violated{false};
  bool protected_joint_corridor_violated{false};
  double minimum_self_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_world_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_controllable_distance_m{std::numeric_limits<double>::infinity()};
  std::string closest_pair;
  std::string closest_controllable_pair;
  std::size_t candidate_contacts{0};
  std::size_t active_distance_constraints{0};
  std::size_t skipped_invariant_contacts{0};
  std::size_t uncontrollable_contacts{0};
  std::size_t outside_viability_contacts{0};
  std::size_t truncated_constraints{0};
  std::string failure_reason;
  std::vector<CollisionPairDiagnostic> pairs;
  /// Complete SINGLE distance-query metadata for a same-scene segment certificate.
  bool distance_query_complete{false};
  double query_distance_m{0.0};
  Eigen::VectorXd query_joint_position;
  std::size_t protected_joint_index{std::numeric_limits<std::size_t>::max()};
};

struct CollisionSeparationGuidance
{
  Eigen::VectorXd distance_gradient;
  double activation{0.0};
  double reference_velocity_mps{0.0};
};

struct CollisionConstraintResult
{
  std::vector<LinearVelocityConstraint> constraints;
  std::vector<CollisionSeparationGuidance> separation_guidance;
  CollisionConstraintDiagnostics diagnostics;
};

/// Fail-closed result of a one-shot check of the measured robot state.
struct CollisionActualStateValidationResult
{
  bool input_valid{false};
  bool unsafe{true};
  bool collision_detected{false};
  bool hard_clearance_violated{false};
  bool protected_joint_corridor_violated{false};
  double protected_joint_position_rad{std::numeric_limits<double>::quiet_NaN()};
  double minimum_self_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_world_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_distance_m{std::numeric_limits<double>::infinity()};
  std::string closest_pair;
  std::string failure_reason;
};

struct CollisionSegmentSampleDiagnostic
{
  std::size_t sample_index{0};
  double interpolation_fraction{0.0};
  double protected_joint_position_rad{0.0};
  double minimum_self_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_world_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_distance_m{std::numeric_limits<double>::infinity()};
  std::string closest_pair;
  bool collision_detected{false};
  bool hard_clearance_violated{false};
  bool robust_clearance_violated{false};
  bool protected_joint_corridor_violated{false};
  std::size_t skipped_invariant_contacts{0};
};

struct CollisionSegmentValidationResult
{
  bool input_valid{false};
  bool unsafe{true};
  bool robust_clearance_violated{false};
  std::size_t requested_substeps{0};
  std::size_t evaluated_samples{0};
  std::size_t first_unsafe_sample{std::numeric_limits<std::size_t>::max()};
  double first_unsafe_fraction{std::numeric_limits<double>::quiet_NaN()};
  double minimum_self_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_world_distance_m{std::numeric_limits<double>::infinity()};
  double minimum_distance_m{std::numeric_limits<double>::infinity()};
  std::string closest_pair;
  /// Invalid input/query reason, or the first actual clearance violation with its pair.
  std::string failure_reason;
  std::vector<CollisionSegmentSampleDiagnostic> samples;
};

/// Discrete distance needed to stop motion toward a boundary with the fastest admissible jerk.
///
/// The input velocity/acceleration describe an already selected next command state. Residual
/// latency is conservatively rounded up to whole controller periods. Every later state uses
/// a_next = clamp(a + jerk * period), v_next = v + a_next * period and
/// q_next = q + v_next * period. Negative velocity means motion away from the boundary.
[[nodiscard]] double jerkLimitedStoppingDistance(
  double toward_velocity,
  double toward_acceleration,
  double max_acceleration,
  double max_jerk,
  double period_sec,
  double residual_latency_sec);

/// Largest reachable approach velocity that can be issued for one period and still stop before
/// a boundary. Returns minimum_reachable_velocity when the complete interval is outside the
/// viability envelope; callers can compare its required stopping distance with the remaining
/// distance to diagnose that state.
[[nodiscard]] double maximumSafeVelocityTowardBoundary(
  double remaining_distance,
  double current_velocity_toward_boundary,
  double minimum_reachable_velocity,
  double maximum_reachable_velocity,
  double max_acceleration,
  double max_jerk,
  double period_sec,
  double residual_latency_sec);

/// Exact discrete stopping excursion for one concrete candidate joint-velocity vector.
///
/// The candidate acceleration is reconstructed with backward Euler from motion_state.velocity.
/// Residual ticks apply the worst admissible per-joint jerk toward the collision, then every
/// joint applies its strongest admissible braking jerk. Acceleration saturation is kept per
/// joint; projected sums of acceleration and jerk are deliberately not used. The returned
/// distance starts at the candidate state and therefore excludes the candidate's first
/// q_next = q + qdot_next * period displacement.
[[nodiscard]] double jerkLimitedCollisionStoppingDistance(
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const Eigen::VectorXd & candidate_joint_velocity,
  const JointMotionLimits & motion_limits,
  double period_sec,
  double residual_latency_sec);

/// Projects per-joint limits onto a distance gradient and returns the largest approach speed
/// whose jerk-limited stopping excursion remains inside remaining_distance. The scalar cap is
/// evaluated with a component-wise candidate envelope, so every concrete reachable joint
/// velocity satisfying the returned row is covered by the per-joint stopping simulation.
[[nodiscard]] double maximumSafeCollisionApproachSpeed(
  double remaining_distance,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  double period_sec,
  double residual_latency_sec);

/// Builds hard linear constraints of the form lower <= coefficients * qdot_next <= upper.
///
/// The builder queries all FCL self/world distance pairs inside the query distance. The
/// returned rows are ordered deterministically by distance and body names, with the most
/// dangerous rows retained when maximum_constraint_rows is reached.
class CollisionConstraintBuilder final
{
public:
  explicit CollisionConstraintBuilder(CollisionConstraintConfig config = {});

  [[nodiscard]] CollisionConstraintResult build(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & robot_state,
    const moveit::core::JointModelGroup & joint_model_group,
    const JointMotionState & motion_state,
    const JointMotionLimits & motion_limits,
    double period_sec) const;

  /// Checks the measured RobotState against hard clearance and the protected-joint corridor.
  /// The two distance queries use SINGLE results limited to hard_clearance_m, without nearest
  /// points or gradients. Invalid input and collision-query failures are reported as unsafe.
  [[nodiscard]] CollisionActualStateValidationResult validateActualState(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & robot_state,
    const moveit::core::JointModelGroup & joint_model_group) const;

  /// Returns true only when a valid build result contains a controllable pair inside the
  /// requested activation distance. The activation distance must lie inside the configured
  /// hard-clearance/query interval.
  [[nodiscard]] bool shouldValidateSegment(
    const CollisionConstraintDiagnostics & diagnostics,
    double activation_distance_m) const;

  /// Conservatively certifies a straight joint-space segment without new collision queries.
  /// Distances and relative-motion masks come from build() on this builder. The scene, ACM,
  /// attached bodies and all variables outside the queried group must remain unchanged.
  /// Both endpoints are bounded relative to the stored query position; omitted pairs use
  /// query_distance_m as their initial lower bound and the maximum configured Lipschitz bound.
  /// True covers the unchanged robust clearances and protected-joint corridor along the segment.
  /// False is inconclusive: callers must use validateSegment(), not assume a collision.
  [[nodiscard]] bool canCertifySegment(
    const CollisionConstraintDiagnostics & diagnostics,
    const Eigen::VectorXd & start_position,
    const Eigen::VectorXd & end_position) const;

  /// Checks the complete straight joint-space segment using substeps equal intervals. Both
  /// endpoints and every internal boundary are sampled, for a total of substeps + 1 samples.
  /// Additional inter-sample uncertainty uses only ancestors affecting relative body motion;
  /// the configured tracking-error reserve and pair Lipschitz bounds remain unchanged.
  /// Invalid input is reported as unsafe and never throws.
  [[nodiscard]] CollisionSegmentValidationResult validateSegment(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & current_robot_state,
    const moveit::core::JointModelGroup & joint_model_group,
    const JointMotionState & current_motion_state,
    const JointMotionState & candidate_motion_state,
    std::size_t substeps) const;

  /// Checks a piecewise-linear joint-space path while sharing one collision-query setup.
  /// Waypoints include the path start and every later state. Shared endpoints are sampled
  /// once, so N segments produce N * substeps_per_segment + 1 globally indexed samples.
  /// Shared endpoints use the componentwise maximum uncertainty of the adjacent segments,
  /// then the same relative-motion ancestor mask as validateSegment.
  /// Invalid input is reported as unsafe and never throws.
  [[nodiscard]] CollisionSegmentValidationResult validatePath(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & start_robot_state,
    const moveit::core::JointModelGroup & joint_model_group,
    const std::vector<JointMotionState> & waypoints,
    std::size_t substeps_per_segment) const;

  [[nodiscard]] const CollisionConstraintConfig & config() const noexcept;

private:
  CollisionConstraintConfig config_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COLLISION_CONSTRAINTS_HPP_
