// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_
#define FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/node.hpp>

namespace face_tracking_arm::control
{

struct BackgroundPathPlannerConfig
{
  std::string move_group_name{"lite6_arm"};
  std::string planner_id{"RRTConnectkConfigDefault"};
  std::string pipeline_namespace{"recovery_planner"};
  double planning_time_s{0.25};
  /// Narrow compact-path feasibility regions need a longer OMPL sampling window.
  double compact_planning_time_s{3.0};
  /// REST edge-constrained requests use their own worker budget and RRT range.
  double rest_edge_planning_time_s{2.0};
  double rest_edge_planner_range{0.30};
  double joint_tolerance_rad{0.01};
  double validation_joint_step_rad{0.03};
  std::size_t maximum_validation_samples{4096};
};

struct PathValidationMetrics
{
  double maximum_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_abs_screen_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_assembly_radius_m{std::numeric_limits<double>::quiet_NaN()};
  double effective_radius_cap_m{std::numeric_limits<double>::quiet_NaN()};
  /// NaN when the configured validity callback does not expose clearance.
  double minimum_robust_clearance_m{std::numeric_limits<double>::quiet_NaN()};
  std::size_t checked_samples{0};
};

/// Worker-measured compact-path sample, mapped onto the immutable coarse path.
/// The suffix maxima include this sample and every later certificate sample.
struct CompactPathSampleCertificate
{
  Eigen::VectorXd positions;
  std::size_t original_segment_index{0};
  double fraction{std::numeric_limits<double>::quiet_NaN()};
  double maximum_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_abs_screen_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_assembly_radius_m{std::numeric_limits<double>::quiet_NaN()};
  double suffix_maximum_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double suffix_maximum_abs_screen_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
  double suffix_maximum_assembly_radius_m{std::numeric_limits<double>::quiet_NaN()};
};

struct BackgroundPathResult
{
  std::uint64_t generation{0};
  std::uint64_t scene_revision{0};
  Eigen::VectorXd start_positions;
  Eigen::VectorXd goal_positions;
  std::vector<Eigen::VectorXd> path;
  bool success{false};
  int error_code{0};
  /// Full worker duration, including snapshot and explicit path validation.
  double elapsed_time_s{0.0};
  std::string message;
  std::string profile_id;
  std::optional<PathValidationMetrics> validation_metrics;
  /// Present only for successful compact plans; samples follow the worker's
  /// checked interpolation and retain their mapping into `path`.
  std::vector<CompactPathSampleCertificate> compact_path_samples;
  /// Exact requested compact goal metrics, distinct from a tolerant path endpoint.
  std::optional<PathValidationMetrics> compact_goal_metrics;
};

/// Opt-in staged route for the known folded xArm6 compact preset. The planner
/// shifts joint1 toward the committed goal during folding while holding joint4,
/// then resets both joints on a directly validated compact suffix.
struct CompactStagedRouteRequirements
{
  double maximum_early_yaw_shift_rad{0.35};
  double maximum_compact_radius_m{0.40};
};

/// Request-specific constraints for a committed compact maneuver. The effective
/// radius and optional pitch caps are fixed by the caller at commitment.
struct PathRequirements
{
  std::string base_link_name;
  std::string screen_link_name;
  double maximum_roll_error_rad{5.1 * 3.14159265358979323846 / 180.0};
  double maximum_assembly_radius_m{0.40};
  /// Optional committed compact-path pitch envelope; FACE remains pitch-unbounded.
  std::optional<double> maximum_abs_screen_pitch_rad;
  /// Internal opt-in, set only for the prevalidated positive-wrist compact goal.
  /// An absent value preserves ordinary planning and explicit-goal behavior.
  std::optional<CompactStagedRouteRequirements> staged_route;
};

struct CompactGoal
{
  std::string profile_id;
  Eigen::VectorXd bounded_positions;
  PathRequirements path_requirements;
};

/// Request-local geometric guard for REST paths. The sine cap is
/// `sin(maximum_tilt_rad)` and is checked at every bounded worker interpolation
/// sample; it carries no compact radius or pitch requirement.
struct HorizontalEdgePathRequirement
{
  std::string screen_link_name;
  double maximum_abs_edge_sine{0.0};
};

/// A geometric reference generator. It never publishes or executes motion.
///
/// A single worker processes at most one active and one pending request. A newer
/// submit/cancel immediately invalidates older results; an active OMPL call may
/// finish its requested planning budget before the newest request starts.
/// The planning budget is not a hard wall-time limit for cloning or validation.
/// All path positions use the JointModelGroup variable order, without wrapping
/// bounded revolute joints. The caller owns adoption, live scene revalidation,
/// collision checks of executed segments, and continuity of its q/v/a state.
class BackgroundPathPlanner
{
public:
  /// Additional geometric safety predicate, on the worker's private scene.
  /// It must be thread-safe, own everything it captures, and must not recursively
  /// call PlanningScene::isStateValid/isStateFeasible. MoveIt still checks ordinary
  /// collisions and joint bounds separately. Use the same clearance profile as
  /// the executor; collision-free alone does not imply an executable QP reference.
  using StateValidity = std::function<bool(
        const planning_scene::PlanningScene &, const moveit::core::RobotState &)>;

  BackgroundPathPlanner(
    const rclcpp::Node::SharedPtr & node,
    moveit::core::RobotModelConstPtr robot_model,
    planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor,
    BackgroundPathPlannerConfig config,
    StateValidity additional_validity = {});
  ~BackgroundPathPlanner();

  BackgroundPathPlanner(const BackgroundPathPlanner &) = delete;
  BackgroundPathPlanner & operator=(const BackgroundPathPlanner &) = delete;

  /// Copies only the request. Scene locking/cloning occurs in the worker.
  /// scene_revision is an opaque caller token, echoed without claiming that the
  /// later snapshot still has that revision. The caller must reject stale tokens.
  [[nodiscard]] std::uint64_t submit(
    const moveit::core::RobotState & start,
    const Eigen::VectorXd & goal_positions, std::uint64_t scene_revision);

  /// Plans ordinary joint-space geometry with an additional sampled REST edge
  /// guard. The same bounded worker validation budget applies as to other paths.
  [[nodiscard]] std::uint64_t submit(
    const moveit::core::RobotState & start,
    const Eigen::VectorXd & goal_positions,
    const HorizontalEdgePathRequirement & edge_requirement,
    std::uint64_t scene_revision);

  /// Plans to an immutable bounded goal and validates every sampled state
  /// against the committed compact pitch, upright-roll and radius requirements.
  [[nodiscard]] std::uint64_t submit(
    const moveit::core::RobotState & start,
    const CompactGoal & goal, std::uint64_t scene_revision);

  void cancel();
  [[nodiscard]] bool isBusy() const;
  [[nodiscard]] std::optional<BackgroundPathResult> takeResult();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_
