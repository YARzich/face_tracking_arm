// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/posture_task_builder.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include <moveit/robot_model/revolute_joint_model.hpp>

namespace face_tracking_arm::control
{

const char * postureTaskModeName(const PostureTaskMode mode) noexcept
{
  switch (mode) {
    case PostureTaskMode::kPreserve:
      return "preserve";
    case PostureTaskMode::kRecover:
      return "recover";
    case PostureTaskMode::kUnavailable:
      return "unavailable";
  }
  return "unavailable";
}

const char * postureUnavailableReasonName(const PostureUnavailableReason reason) noexcept
{
  switch (reason) {
    case PostureUnavailableReason::kNone:
      return "none";
    case PostureUnavailableReason::kCalibrationUnavailable:
      return "calibration_unavailable";
    case PostureUnavailableReason::kKinematicsUnavailable:
      return "kinematics_unavailable";
    case PostureUnavailableReason::kInvalidObjective:
      return "invalid_objective";
    case PostureUnavailableReason::kComfortBoundsUnavailable:
      return "comfort_bounds_unavailable";
  }
  return "none";
}

PostureTaskBuilder::PostureTaskBuilder(
  moveit::core::RobotModelConstPtr robot_model,
  const moveit::core::JointModelGroup * group,
  const JointMotionLimits & limits,
  PostureTaskBuilderParameters parameters)
: robot_model_(std::move(robot_model)), group_(group), limits_(limits),
  parameters_(parameters)
{
  if (!robot_model_ || !group_) {
    throw std::invalid_argument("Posture task builder requires a robot model and planning group");
  }
  if (!group_->getJointModels().empty()) {
    root_link_ = group_->getJointModels().front()->getParentLinkModel();
  }
  shoulder_link_ = robot_model_->getLinkModel("link2");
  elbow_link_ = robot_model_->getLinkModel("link3");
  wrist_link_ = robot_model_->getLinkModel("link4");
  calibrate();
}

double PostureTaskBuilder::comfortableJoint3Lower() const
{
  return preference_config_.comfortable_joint3_lower;
}

double PostureTaskBuilder::comfortableJoint3Upper() const
{
  return preference_config_.comfortable_joint3_upper;
}

void PostureTaskBuilder::calibrate()
{
  const auto & joint_names = group_->getVariableNames();
  const auto joint3_position = std::find(joint_names.begin(), joint_names.end(), "joint3");
  const auto * joint3_model = robot_model_->getJointModel("joint3");
  const auto * joint3_revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(
    joint3_model);
  const auto named_bend_angle = [&](const std::string & pose_name, double & q3,
    double & bend_angle) {
      moveit::core::RobotState posture(robot_model_);
      posture.setToDefaultValues();
      if (!posture.setToDefaultValues(group_, pose_name)) {
        return false;
      }
      posture.update();
      Eigen::VectorXd positions;
      posture.copyJointGroupPositions(group_, positions);
      if (joint3_position == joint_names.end() || !joint3_model || !joint3_revolute ||
        !shoulder_link_ || !elbow_link_ || !wrist_link_ ||
        positions.size() != static_cast<Eigen::Index>(joint_names.size()))
      {
        return false;
      }
      q3 = positions[std::distance(joint_names.begin(), joint3_position)];
      const Eigen::Vector3d shoulder = posture.getGlobalLinkTransform(
        shoulder_link_).translation();
      const Eigen::Vector3d elbow = posture.getGlobalLinkTransform(
        elbow_link_).translation();
      const Eigen::Vector3d wrist = posture.getGlobalLinkTransform(
        wrist_link_).translation();
      const auto * parent = joint3_model->getParentLinkModel();
      const auto * child = joint3_model->getChildLinkModel();
      if (!parent || !child) {
        return false;
      }
      const Eigen::Vector3d axis = posture.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * joint3_revolute->getAxis();
      const Eigen::Vector3d upper_arm = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      if (!axis.allFinite() || !upper_arm.allFinite() || !forearm.allFinite() ||
        axis.squaredNorm() <= 0.0 || upper_arm.norm() <= 0.0 || forearm.norm() <= 0.0)
      {
        return false;
      }
      bend_angle = std::atan2(
        axis.normalized().dot(upper_arm.cross(forearm)), upper_arm.dot(forearm));
      return std::isfinite(q3) && std::isfinite(bend_angle);
    };

  double rest_joint3 = 0.0;
  double rest_bend_angle = 0.0;
  double search_joint3 = 0.0;
  double search_bend_angle = 0.0;
  const auto joint3_index = std::distance(joint_names.begin(), joint3_position);
  const bool joint3_limits_available = joint3_position != joint_names.end() &&
    limits_.lower_position.size() > joint3_index &&
    limits_.upper_position.size() > joint3_index;
  const bool rest_bend_available = joint3_limits_available &&
    named_bend_angle("rest", rest_joint3, rest_bend_angle);
  if (rest_bend_available) {
    search_joint3 = rest_joint3;
    search_bend_angle = rest_bend_angle;
    // A disabled optional search mode must not make posture calibration fail.
    // If no named search pose is present, the normal rest posture is the source.
    (void)named_bend_angle("search", search_joint3, search_bend_angle);
    const auto bend_calibration = calibratePostureBend(
      rest_joint3, rest_bend_angle, search_joint3, search_bend_angle,
      limits_.lower_position[joint3_index], limits_.upper_position[joint3_index]);
    if (bend_calibration.valid) {
      preference_config_.j3_index = joint3_index;
      preference_config_.joint3_angle_offset_rad = bend_calibration.angle_offset_rad;
      preference_config_.preferred_bend_sign = bend_calibration.preferred_sign;
      preference_config_.comfortable_joint3_lower = bend_calibration.joint3_lower;
      preference_config_.comfortable_joint3_upper = bend_calibration.joint3_upper;
      calibrated_ = true;
    }
  }
}

PostureTaskResult PostureTaskBuilder::apply(
  const moveit::core::RobotState & state, HierarchicalVelocityTask & task) const
{
  JointMotionState state_motion;
  state.copyJointGroupPositions(group_, state_motion.position);
  const bool have_brake_motion = state.hasVelocities() && state.hasAccelerations();
  if (have_brake_motion) {
    state.copyJointGroupVelocities(group_, state_motion.velocity);
    state.copyJointGroupAccelerations(group_, state_motion.acceleration);
  }
  const Eigen::Index joints = state_motion.position.size();
  const Eigen::VectorXd & q = state_motion.position;
  task.secondary_matrix.resize(0, joints);
  task.secondary_reference.resize(0);
  task.secondary_weights.resize(0);
  task.tertiary_matrix.resize(0, joints);
  task.tertiary_reference.resize(0);
  task.tertiary_weights.resize(0);
  const auto world_linear_jacobian = [&](const moveit::core::LinkModel * link,
    Eigen::MatrixXd & output) {
      Eigen::MatrixXd full;
      if (!link || !state.getJacobian(
          group_, link, Eigen::Vector3d::Zero(), full, false) ||
        full.rows() != 6 || full.cols() != q.size())
      {
        return false;
      }
      output = full.topRows(3);
      if (root_link_) {
        output = (state.getGlobalLinkTransform(root_link_).linear() * output).eval();
      }
      return output.allFinite();
    };
  Eigen::MatrixXd shoulder_jacobian, elbow_jacobian, wrist_jacobian;
  const bool posture_kinematics_available = shoulder_link_ && elbow_link_ && wrist_link_ &&
    q.size() == 6 && limits_.lower_position.size() > 5 &&
    limits_.upper_position.size() > 5 &&
    world_linear_jacobian(shoulder_link_, shoulder_jacobian) &&
    world_linear_jacobian(elbow_link_, elbow_jacobian) &&
    world_linear_jacobian(wrist_link_, wrist_jacobian);
  PosturePreferenceConfig posture_config = preference_config_;
  posture_config.position_margin_j5_rad = limits_.position_margin.size() > 4 ?
    limits_.position_margin[4] : parameters_.joint_position_margin_rad;
  posture_config.position_margin_j6_rad = limits_.position_margin.size() > 5 ?
    limits_.position_margin[5] : parameters_.joint_position_margin_rad;
  posture_config.position_gain = parameters_.position_gain;
  posture_config.maximum_linear_reference_mps = parameters_.maximum_linear_reference_mps;
  posture_config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  posture_config.control_period_sec = parameters_.control_period_sec;
  posture_config.lower_j5 = posture_kinematics_available ? limits_.lower_position[4] : 0.0;
  posture_config.upper_j5 = posture_kinematics_available ? limits_.upper_position[4] : 0.0;
  posture_config.lower_j6 = posture_kinematics_available ? limits_.lower_position[5] : 0.0;
  posture_config.upper_j6 = posture_kinematics_available ? limits_.upper_position[5] : 0.0;
  if (!calibrated_) {
    return {PostureTaskMode::kUnavailable,
      PostureUnavailableReason::kCalibrationUnavailable};
  }
  if (!posture_kinematics_available) {
    return {PostureTaskMode::kUnavailable,
      PostureUnavailableReason::kKinematicsUnavailable};
  }
  const auto comfort = makePosturePreferenceObjective(
    state.getGlobalLinkTransform(shoulder_link_).translation(),
    state.getGlobalLinkTransform(elbow_link_).translation(),
    state.getGlobalLinkTransform(wrist_link_).translation(), shoulder_jacobian,
    elbow_jacobian, wrist_jacobian, q, posture_config);
  if (!comfort.valid) {
    return {PostureTaskMode::kUnavailable,
      PostureUnavailableReason::kInvalidObjective};
  }
  if (comfort.joint3_correction.matrix.rows() > 0) {
    task.secondary_matrix = comfort.joint3_correction.matrix;
    task.secondary_reference = comfort.joint3_correction.reference;
    task.secondary_weights = comfort.joint3_correction.weights;
  }
  if (comfort.other_corrections.matrix.rows() > 0) {
    task.tertiary_matrix = comfort.other_corrections.matrix;
    task.tertiary_reference = comfort.other_corrections.reference;
    task.tertiary_weights = comfort.other_corrections.weights;
  }
  PostureTaskResult result{
    comfort.branch_comfortable && comfort.geometry_comfortable ?
    PostureTaskMode::kPreserve : PostureTaskMode::kRecover,
    PostureUnavailableReason::kNone};
  if (comfort.branch_comfortable && have_brake_motion) {
    const Eigen::Index joint3 = static_cast<Eigen::Index>(posture_config.j3_index);
    const auto preservation = makeJoint3ComfortBounds(
      state_motion, limits_, joint3, posture_config.comfortable_joint3_lower,
      posture_config.comfortable_joint3_upper, parameters_.control_period_sec,
      parameters_.residual_command_latency_sec);
    if (preservation) {
      task.preferred_primary_constraints.push_back(*preservation);
    } else {
      result.mode = PostureTaskMode::kUnavailable;
      result.unavailable_reason = PostureUnavailableReason::kComfortBoundsUnavailable;
    }
  }
  return result;
}

}  // namespace face_tracking_arm::control
