// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_
#define FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <functional>
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
  double joint_tolerance_rad{0.01};
  double validation_joint_step_rad{0.03};
  std::size_t maximum_validation_samples{4096};
  double ik_time_budget_s{0.08};
  std::string ik_rest_state_name{"rest"};
  std::string ik_base_joint_name{"joint1"};
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
  struct PoseGoal
  {
    /// Desired link pose in RobotModel::getModelFrame(), not an unstamped TF frame.
    Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
    std::string link_name;
    double preferred_base_angle_rad{0.0};
    /// When set, pose.translation() is only a preference: sample reachable
    /// positions and free roll, keeping the link's +X directed toward this point.
    std::optional<Eigen::Vector3d> face_position;
    /// Optional optical camera link; look-at includes its fixed mounting transform.
    std::string gaze_link_name;
    /// 0: pointing, 1: relaxed pointing cone, 2: safe joint detour.
    unsigned int relaxation{0};
    /// Advances across retries to avoid repeatedly selecting the same endpoint.
    std::uint64_t attempt{0};
  };

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

  /// Selects a collision/clearance-valid joint goal using the configured MoveIt
  /// IK plugin and bounded candidate sampling. With face_position, translation
  /// and roll are free; relaxation can eventually admit a checked joint detour.
  /// The caller determines whether pointing is retained along the executed path.
  [[nodiscard]] std::uint64_t submit(
    const moveit::core::RobotState & start,
    const PoseGoal & goal, std::uint64_t scene_revision);

  void cancel();
  [[nodiscard]] bool isBusy() const;
  [[nodiscard]] std::optional<BackgroundPathResult> takeResult();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__BACKGROUND_PATH_PLANNER_HPP_
