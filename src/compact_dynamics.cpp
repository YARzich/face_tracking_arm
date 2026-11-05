// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_dynamics.hpp"

#include <tinyxml2.h>
#include <urdf/model.h>
#include <urdf_parser/urdf_parser.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <moveit/dynamics_solver/dynamics_solver.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control
{
namespace
{

constexpr std::array<double, 4U> kCoarseTempoCorrections{0.8, 0.6, 0.4, 0.2};
constexpr std::size_t kTempoRefinementPasses{2U};
constexpr double kEffortComparisonTolerance{1.0e-9};
constexpr double kStaticVelocityTolerance{1.0e-12};

Eigen::Isometry3d toEigen(const urdf::Pose & pose)
{
  Eigen::Quaterniond rotation(
    pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z);
  if (!rotation.coeffs().allFinite() || rotation.norm() <= 0.0) {
    throw std::invalid_argument("URDF contains an invalid fixed-link rotation");
  }
  rotation.normalize();
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = rotation.toRotationMatrix();
  transform.translation() = Eigen::Vector3d(
    pose.position.x, pose.position.y, pose.position.z);
  if (!transform.matrix().allFinite()) {
    throw std::invalid_argument("URDF contains a nonfinite fixed-link transform");
  }
  return transform;
}

struct RigidBodyInertia
{
  double mass{0.0};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d inertia_at_center{Eigen::Matrix3d::Zero()};
};

RigidBodyInertia readBodyInertia(
  const urdf::Inertial & inertial, const Eigen::Isometry3d & body_in_link)
{
  if (!std::isfinite(inertial.mass) || inertial.mass < 0.0) {
    throw std::invalid_argument("URDF contains an invalid link mass");
  }
  const Eigen::Isometry3d inertial_in_link = body_in_link * toEigen(inertial.origin);
  Eigen::Matrix3d tensor;
  tensor << inertial.ixx, inertial.ixy, inertial.ixz,
    inertial.ixy, inertial.iyy, inertial.iyz,
    inertial.ixz, inertial.iyz, inertial.izz;
  if (!tensor.allFinite()) {
    throw std::invalid_argument("URDF contains a nonfinite inertia tensor");
  }
  return RigidBodyInertia{
    inertial.mass, inertial_in_link.translation(),
    inertial_in_link.linear() * tensor * inertial_in_link.linear().transpose()};
}

void collectFixedBranchBodies(
  const urdf::ModelInterface & source, const urdf::LinkSharedPtr & parent,
  const Eigen::Isometry3d & parent_from_link,
  const std::set<std::string> & group_links,
  std::vector<RigidBodyInertia> & bodies)
{
  for (const auto & joint : parent->child_joints) {
    if (!joint || joint->type != urdf::Joint::FIXED ||
      group_links.count(joint->child_link_name) != 0U)
    {
      continue;
    }
    urdf::LinkSharedPtr child;
    source.getLink(joint->child_link_name, child);
    if (!child) {
      throw std::invalid_argument("URDF fixed joint refers to a missing child link");
    }
    const Eigen::Isometry3d parent_from_child =
      parent_from_link * toEigen(joint->parent_to_joint_origin_transform);
    if (child->inertial) {
      if (child->inertial->mass > 0.0) {
        bodies.push_back(readBodyInertia(*child->inertial, parent_from_child));
      }
    }
    collectFixedBranchBodies(source, child, parent_from_child, group_links, bodies);
  }
}

urdf::InertialSharedPtr combineInertias(const std::vector<RigidBodyInertia> & bodies)
{
  double total_mass = 0.0;
  Eigen::Vector3d weighted_center = Eigen::Vector3d::Zero();
  for (const auto & body : bodies) {
    total_mass += body.mass;
    weighted_center += body.mass * body.center;
  }
  if (!std::isfinite(total_mass) || total_mass <= 0.0 || !weighted_center.allFinite()) {
    throw std::invalid_argument("Fixed payload has invalid total mass or center of mass");
  }
  const Eigen::Vector3d center = weighted_center / total_mass;
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
  for (const auto & body : bodies) {
    const Eigen::Vector3d offset = body.center - center;
    inertia += body.inertia_at_center + body.mass *
      (offset.squaredNorm() * Eigen::Matrix3d::Identity() - offset * offset.transpose());
  }
  if (!center.allFinite() || !inertia.allFinite()) {
    throw std::invalid_argument("Combined fixed-payload inertia is nonfinite");
  }
  auto result = std::make_shared<urdf::Inertial>();
  result->mass = total_mass;
  result->origin.position = urdf::Vector3(center.x(), center.y(), center.z());
  result->origin.rotation = urdf::Rotation(0.0, 0.0, 0.0, 1.0);
  result->ixx = inertia(0, 0);
  result->ixy = inertia(0, 1);
  result->ixz = inertia(0, 2);
  result->iyy = inertia(1, 1);
  result->iyz = inertia(1, 2);
  result->izz = inertia(2, 2);
  return result;
}

moveit::core::RobotModelConstPtr makeDynamicsRobotModel(
  const moveit::core::RobotModelConstPtr & source_model,
  const moveit::core::JointModelGroup & source_group)
{
  std::unique_ptr<tinyxml2::XMLDocument> exported_urdf(
    urdf::exportURDF(*source_model->getURDF()));
  if (!exported_urdf) {
    throw std::invalid_argument("Unable to serialize RobotModel URDF for payload aggregation");
  }
  tinyxml2::XMLPrinter printer;
  exported_urdf->Print(&printer);
  auto dynamics_urdf = std::make_shared<urdf::Model>();
  if (!dynamics_urdf->initString(printer.CStr())) {
    throw std::invalid_argument("Unable to copy RobotModel URDF for payload aggregation");
  }
  const auto & source_urdf = *source_model->getURDF();
  const auto & group_link_names = source_group.getLinkModelNames();
  const std::set<std::string> group_links(group_link_names.begin(), group_link_names.end());
  for (const auto & link_name : group_link_names) {
    urdf::LinkSharedPtr source_link;
    source_urdf.getLink(link_name, source_link);
    if (!source_link) {
      throw std::invalid_argument("RobotModel group refers to a missing URDF link");
    }
    std::vector<RigidBodyInertia> bodies;
    if (source_link->inertial && source_link->inertial->mass > 0.0) {
      bodies.push_back(readBodyInertia(*source_link->inertial, Eigen::Isometry3d::Identity()));
    }
    collectFixedBranchBodies(
      source_urdf, source_link, Eigen::Isometry3d::Identity(), group_links, bodies);
    const std::size_t original_body_count =
      source_link->inertial && source_link->inertial->mass > 0.0 ? 1U : 0U;
    if (bodies.size() <= original_body_count) {
      continue;
    }
    urdf::LinkSharedPtr dynamics_link;
    dynamics_urdf->getLink(link_name, dynamics_link);
    if (!dynamics_link) {
      throw std::invalid_argument("Copied URDF is missing a planning-group link");
    }
    dynamics_link->inertial = combineInertias(bodies);
  }
  auto mutable_model = std::make_shared<moveit::core::RobotModel>(
    dynamics_urdf, source_model->getSRDF());
  return mutable_model;
}

bool fixedRelative(
  const moveit::core::LinkModel * first,
  const moveit::core::LinkModel * second)
{
  if (!first || !second) {
    return false;
  }
  std::unordered_map<const moveit::core::LinkModel *, bool> first_ancestors;
  bool first_path_fixed = true;
  const moveit::core::LinkModel * link = first;
  first_ancestors.emplace(link, true);
  while (link->getParentJointModel() && link->getParentLinkModel()) {
    first_path_fixed = first_path_fixed &&
      link->getParentJointModel()->getType() == moveit::core::JointModel::FIXED;
    link = link->getParentLinkModel();
    first_ancestors.emplace(link, first_path_fixed);
  }
  bool second_path_fixed = true;
  link = second;
  while (link) {
    const auto common = first_ancestors.find(link);
    if (common != first_ancestors.end() && common->second && second_path_fixed) {
      return true;
    }
    if (!link->getParentJointModel() || !link->getParentLinkModel()) {
      break;
    }
    second_path_fixed = second_path_fixed &&
      link->getParentJointModel()->getType() == moveit::core::JointModel::FIXED;
    link = link->getParentLinkModel();
  }
  return false;
}

}  // namespace

struct CompactDynamicsModel::Impl
{
  Impl(
    moveit::core::RobotModelConstPtr input_model,
    const std::string & group_name,
    const std::string & base_link_name,
    const Eigen::Vector3d & gravity_world)
  : original_model(std::move(input_model))
  {
    if (!original_model || group_name.empty() || base_link_name.empty() ||
      !gravity_world.allFinite() || gravity_world.norm() <= 0.0)
    {
      throw std::invalid_argument("Compact dynamics requires a model, group, base and gravity");
    }
    const auto * original_group = original_model->getJointModelGroup(group_name);
    if (!original_group || original_group->getVariableCount() == 0U) {
      throw std::invalid_argument("Compact dynamics joint group is empty or unavailable");
    }
    const auto * solver_base = original_group->getJointModels().front()->getParentLinkModel();
    const auto * requested_base = original_model->getLinkModel(base_link_name);
    if (!solver_base || !requested_base || !fixedRelative(solver_base, requested_base)) {
      throw std::invalid_argument(
              "Requested dynamics base is not fixed-relative to the solver group base");
    }
    group_model = makeDynamicsRobotModel(original_model, *original_group);
    group = group_model->getJointModelGroup(group_name);
    const auto * base_link = group_model->getLinkModel(solver_base->getName());
    if (!group || !base_link || group->getVariableCount() != original_group->getVariableCount()) {
      throw std::invalid_argument("Dynamics RobotModel does not preserve the requested group");
    }
    moveit::core::RobotState default_state(group_model);
    default_state.setToDefaultValues();
    default_state.update(true);
    const Eigen::Vector3d gravity_base =
      default_state.getGlobalLinkTransform(base_link).linear().transpose() * gravity_world;
    if (!gravity_base.allFinite()) {
      throw std::invalid_argument("Gravity transformed into the group base frame is invalid");
    }
    geometry_msgs::msg::Vector3 gravity_msg;
    gravity_msg.x = gravity_base.x();
    gravity_msg.y = gravity_base.y();
    gravity_msg.z = gravity_base.z();
    solver = std::make_unique<dynamics_solver::DynamicsSolver>(
      group_model, group_name, gravity_msg);

    const auto & variable_names = group->getVariableNames();
    effort_limits.reserve(variable_names.size());
    damping.reserve(variable_names.size());
    friction.reserve(variable_names.size());
    for (const auto & variable_name : variable_names) {
      const auto joint = group_model->getURDF()->getJoint(variable_name);
      if (!joint || !joint->limits || !std::isfinite(joint->limits->effort) ||
        joint->limits->effort <= 0.0)
      {
        throw std::invalid_argument(
                "Compact dynamics requires a positive URDF effort limit for " + variable_name);
      }
      const double joint_damping = joint->dynamics ? joint->dynamics->damping : 0.0;
      const double joint_friction = joint->dynamics ? joint->dynamics->friction : 0.0;
      if (!std::isfinite(joint_damping) || joint_damping < 0.0 ||
        !std::isfinite(joint_friction) || joint_friction < 0.0)
      {
        throw std::invalid_argument(
                "URDF contains invalid damping or friction for " + variable_name);
      }
      effort_limits.push_back(joint->limits->effort);
      damping.push_back(joint_damping);
      friction.push_back(joint_friction);
    }
  }

  moveit::core::RobotModelConstPtr original_model;
  moveit::core::RobotModelConstPtr group_model;
  const moveit::core::JointModelGroup * group{nullptr};
  std::unique_ptr<dynamics_solver::DynamicsSolver> solver;
  std::vector<double> effort_limits;
  std::vector<double> damping;
  std::vector<double> friction;
};

CompactDynamicsModel::CompactDynamicsModel(
  moveit::core::RobotModelConstPtr robot_model,
  const std::string & group_name,
  const std::string & base_link_name,
  const Eigen::Vector3d & gravity_world)
: impl_(std::make_unique<Impl>(
      std::move(robot_model), group_name, base_link_name, gravity_world))
{
}

CompactDynamicsModel::~CompactDynamicsModel() = default;
CompactDynamicsModel::CompactDynamicsModel(CompactDynamicsModel &&) noexcept = default;
CompactDynamicsModel & CompactDynamicsModel::operator=(CompactDynamicsModel &&) noexcept = default;

namespace
{

struct EffortCheck
{
  bool valid{false};
  bool within_limits{false};
  std::vector<double> peaks;
  double maximum_ratio{0.0};
  std::string failure;
  std::size_t profile_samples{0U};
  std::size_t interpolation_samples{0U};
  bool prefix_over_limit{false};
  bool suffix_over_limit{false};
};

template<typename ModelImpl>
bool calculateRequiredEffort(
  const ModelImpl & model,
  const Eigen::VectorXd & position,
  const Eigen::VectorXd & velocity,
  const Eigen::VectorXd & acceleration,
  std::vector<double> & required,
  std::string & failure)
{
  const std::size_t count = model.effort_limits.size();
  if (position.size() != static_cast<Eigen::Index>(count) ||
    velocity.size() != static_cast<Eigen::Index>(count) ||
    acceleration.size() != static_cast<Eigen::Index>(count) ||
    !position.allFinite() || !velocity.allFinite() || !acceleration.allFinite())
  {
    failure = "profile state has invalid dimensions or nonfinite q/v/a";
    return false;
  }
  const std::vector<double> q(position.data(), position.data() + count);
  const std::vector<double> qd(velocity.data(), velocity.data() + count);
  const std::vector<double> qdd(acceleration.data(), acceleration.data() + count);
  const std::vector<geometry_msgs::msg::Wrench> wrenches(
    model.group->getLinkModelNames().size());
  std::vector<double> inverse_dynamics(count, 0.0);
  if (!model.solver->getTorques(q, qd, qdd, wrenches, inverse_dynamics)) {
    failure = "DynamicsSolver::getTorques rejected a profile state";
    return false;
  }
  required.resize(count);
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(inverse_dynamics[index])) {
      failure = "DynamicsSolver returned a nonfinite torque";
      return false;
    }
    const double viscous = model.damping[index] * qd[index];
    const double base_effort = inverse_dynamics[index] + viscous;
    const double friction = std::abs(qd[index]) > kStaticVelocityTolerance ?
      std::copysign(model.friction[index], qd[index]) : 0.0;
    required[index] = std::abs(base_effort + friction);
    if (!std::isfinite(required[index])) {
      failure = "URDF damping/friction produced a nonfinite torque";
      return false;
    }
  }
  return true;
}

template<typename ModelImpl>
EffortCheck checkProfileEffort(
  const ModelImpl & model,
  const CompactJointProfile & profile,
  const double control_period_sec)
{
  EffortCheck result;
  result.peaks.assign(model.effort_limits.size(), 0.0);
  if (profile.samples.empty() || !std::isfinite(control_period_sec) ||
    control_period_sec <= 0.0 ||
    profile.prefix_endpoint_sample_index >= profile.samples.size())
  {
    result.failure = "profile has no samples or an invalid prefix boundary";
    return result;
  }
  const std::size_t prefix_end = profile.prefix_endpoint_sample_index;
  const auto check_state = [&](
    const Eigen::VectorXd & q, const Eigen::VectorXd & qd, const Eigen::VectorXd & qdd,
    const bool is_prefix, const bool is_interpolation) {
      std::vector<double> required;
      if (!calculateRequiredEffort(model, q, qd, qdd, required, result.failure)) {
        return false;
      }
      if (is_interpolation) {
        ++result.interpolation_samples;
      } else {
        ++result.profile_samples;
      }
      bool over_limit = false;
      for (std::size_t joint = 0U; joint < required.size(); ++joint) {
        result.peaks[joint] = std::max(result.peaks[joint], std::abs(required[joint]));
        result.maximum_ratio = std::max(
          result.maximum_ratio, std::abs(required[joint]) / model.effort_limits[joint]);
        const double tolerance = kEffortComparisonTolerance *
          std::max(1.0, model.effort_limits[joint]);
        if (std::abs(required[joint]) > model.effort_limits[joint] + tolerance) {
          over_limit = true;
        }
      }
      if (over_limit && is_prefix) {
        result.prefix_over_limit = true;
      } else if (over_limit) {
        result.suffix_over_limit = true;
      }
      return true;
    };

  for (std::size_t sample = 0U; sample < profile.samples.size(); ++sample) {
    const auto & state = profile.samples[sample];
    if (!check_state(
        state.position, state.velocity, state.acceleration,
        prefix_end > 0U && sample <= prefix_end, false))
    {
      return result;
    }
    if (sample + 1U >= profile.samples.size()) {
      continue;
    }
    const auto & next = profile.samples[sample + 1U];
    if (next.position.size() != state.position.size()) {
      result.failure = "adjacent profile samples have incompatible joint counts";
      return result;
    }
    // The controller publishes positions only, so JTC linearly interpolates
    // between positions. Check the midpoint with that segment's constant slope.
    const Eigen::VectorXd midpoint_position = 0.5 * (state.position + next.position);
    const Eigen::VectorXd midpoint_velocity =
      (next.position - state.position) / control_period_sec;
    const Eigen::VectorXd midpoint_acceleration =
      Eigen::VectorXd::Zero(midpoint_position.size());
    if (!check_state(
        midpoint_position, midpoint_velocity, midpoint_acceleration,
        prefix_end > 0U && sample < prefix_end, true))
    {
      return result;
    }
  }
  result.valid = true;
  result.within_limits = !result.prefix_over_limit && !result.suffix_over_limit;
  return result;
}

template<typename ModelImpl>
bool staticPathFeasible(
  const ModelImpl & model,
  const CompactJointProfile & profile,
  const std::size_t prefix_endpoint,
  bool & static_infeasible,
  std::string & failure)
{
  static_infeasible = false;
  if (profile.checked_path_vertices.empty()) {
    failure = "profile has no checked path vertices";
    return false;
  }
  const Eigen::Index count = static_cast<Eigen::Index>(model.effort_limits.size());
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(count);
  for (std::size_t vertex = 0U; vertex < profile.checked_path_vertices.size(); ++vertex) {
    if (vertex == 0U && prefix_endpoint == 0U &&
      (profile.samples.front().velocity.lpNorm<Eigen::Infinity>() > kStaticVelocityTolerance ||
      profile.samples.front().acceleration.lpNorm<Eigen::Infinity>() > kStaticVelocityTolerance))
    {
      continue;
    }
    const auto & position = profile.checked_path_vertices[vertex];
    std::vector<double> required;
    if (!calculateRequiredEffort(model, position, zero, zero, required, failure)) {
      return false;
    }
    for (std::size_t joint = 0U; joint < required.size(); ++joint) {
      const double tolerance = kEffortComparisonTolerance *
        std::max(1.0, model.effort_limits[joint]);
      if (std::abs(required[joint]) > model.effort_limits[joint] + tolerance) {
        failure = "stationary waypoint " + std::to_string(vertex) +
          " exceeds effort limit at joint " +
          model.group->getVariableNames()[joint] +
          "; lowering tempo cannot remove gravity/load torque";
        static_infeasible = true;
        return false;
      }
    }
  }
  return true;
}

}  // namespace

CompactDynamicsResult CompactDynamicsModel::validateAndRetime(
  const CompactJointProfile & input,
  const JointMotionLimits & motion_limits,
  const double control_period_sec,
  const double requested_speed_scale) const
{
  const auto begin = std::chrono::steady_clock::now();
  CompactDynamicsResult output;
  const auto finish = [&output, &begin]() {
      output.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    };
  const std::size_t joint_count = impl_->effort_limits.size();
  if (input.samples.empty() || input.checked_path_vertices.empty() ||
    input.checked_path_segment_scales.size() + 1U != input.checked_path_vertices.size() ||
    input.prefix_endpoint_sample_index >= input.samples.size() ||
    !std::isfinite(control_period_sec) || control_period_sec <= 0.0 ||
    !std::isfinite(requested_speed_scale) || requested_speed_scale <= 0.0 ||
    motion_limits.lower_position.size() != static_cast<Eigen::Index>(joint_count) ||
    motion_limits.upper_position.size() != static_cast<Eigen::Index>(joint_count) ||
    motion_limits.max_velocity.size() != static_cast<Eigen::Index>(joint_count) ||
    motion_limits.max_acceleration.size() != static_cast<Eigen::Index>(joint_count) ||
    motion_limits.max_jerk.size() != static_cast<Eigen::Index>(joint_count) ||
    !motion_limits.lower_position.allFinite() || !motion_limits.upper_position.allFinite() ||
    !motion_limits.max_velocity.allFinite() || !motion_limits.max_acceleration.allFinite() ||
    !motion_limits.max_jerk.allFinite() ||
    !(motion_limits.upper_position.array() > motion_limits.lower_position.array()).all() ||
    !(motion_limits.max_velocity.array() > 0.0).all() ||
    !(motion_limits.max_acceleration.array() > 0.0).all() ||
    !(motion_limits.max_jerk.array() > 0.0).all())
  {
    output.status = CompactDynamicsStatus::kInvalidInput;
    output.failure_reason = "profile, motion limits, period or speed scale are invalid";
    finish();
    return output;
  }
  for (const auto & sample : input.samples) {
    if (sample.position.size() != static_cast<Eigen::Index>(joint_count) ||
      sample.velocity.size() != static_cast<Eigen::Index>(joint_count) ||
      sample.acceleration.size() != static_cast<Eigen::Index>(joint_count) ||
      !sample.position.allFinite() || !sample.velocity.allFinite() ||
      !sample.acceleration.allFinite())
    {
      output.status = CompactDynamicsStatus::kInvalidInput;
      output.failure_reason = "profile contains an invalid q/v/a sample";
      finish();
      return output;
    }
  }
  for (const auto & vertex : input.checked_path_vertices) {
    if (vertex.size() != static_cast<Eigen::Index>(joint_count) || !vertex.allFinite()) {
      output.status = CompactDynamicsStatus::kInvalidInput;
      output.failure_reason = "checked path contains an invalid vertex";
      finish();
      return output;
    }
  }
  if (std::any_of(input.checked_path_segment_scales.begin(),
    input.checked_path_segment_scales.end(), [](const double scale) {
      return !std::isfinite(scale) || scale <= 0.0;
    }) || input.checked_path_vertices.empty() ||
    input.checked_path_segment_scales.size() + 1U != input.checked_path_vertices.size() ||
    input.prefix_endpoint_sample_index >= input.samples.size() ||
    input.path_progress_rad_per_sample.size() != input.samples.size() ||
    input.speed_scale_per_interval.size() + 1U != input.samples.size() ||
    std::any_of(input.speed_scale_per_interval.begin(), input.speed_scale_per_interval.end(),
    [](const double scale) {return !std::isfinite(scale) || scale <= 0.0;}) ||
    !std::isfinite(input.duration_sec) || input.duration_sec < 0.0 ||
    !std::isfinite(input.path_length_rad) || input.path_length_rad<0.0 ||
    (input.checked_path_vertices.front() -
    input.samples[input.prefix_endpoint_sample_index].position).
    lpNorm<Eigen::Infinity>()> kCompactProfilePositionToleranceRad)
  {
    output.status = CompactDynamicsStatus::kInvalidInput;
    output.failure_reason = "profile metadata does not match its checked path or timing";
    finish();
    return output;
  }
  double expected_progress = 0.0;
  for (std::size_t index = 0; index < input.samples.size(); ++index) {
    const double reported_progress = input.path_progress_rad_per_sample[index];
    if (!std::isfinite(reported_progress) ||
      std::abs(reported_progress - expected_progress) > kCompactProfilePositionToleranceRad ||
      (index > 0U && reported_progress + kCompactProfilePositionToleranceRad <
      input.path_progress_rad_per_sample[index - 1U]))
    {
      output.status = CompactDynamicsStatus::kInvalidInput;
      output.failure_reason = "profile sample progress is invalid or does not match q geometry";
      finish();
      return output;
    }
    if (index + 1U < input.samples.size()) {
      expected_progress += (input.samples[index + 1U].position -
        input.samples[index].position).norm();
    }
  }
  if (std::abs(input.path_length_rad - expected_progress) >
    kCompactProfilePositionToleranceRad)
  {
    output.status = CompactDynamicsStatus::kInvalidInput;
    output.failure_reason = "profile path length does not match its sample geometry";
    finish();
    return output;
  }

  EffortCheck original_check;
  try {
    original_check = checkProfileEffort(*impl_, input, control_period_sec);
  } catch (const std::exception & exception) {
    output.status = CompactDynamicsStatus::kSolverFailure;
    output.failure_reason = std::string{"DynamicsSolver threw: "} + exception.what();
    finish();
    return output;
  }
  output.checked_profile_samples = original_check.profile_samples;
  output.checked_interpolation_samples = original_check.interpolation_samples;
  output.peak_required_effort_nm_before = original_check.peaks;
  output.peak_required_effort_nm_after = original_check.peaks;
  output.maximum_effort_ratio_before = original_check.maximum_ratio;
  output.maximum_effort_ratio_after = original_check.maximum_ratio;
  output.immutable_prefix_over_limit = original_check.prefix_over_limit;
  if (!original_check.valid) {
    output.status = CompactDynamicsStatus::kSolverFailure;
    output.failure_reason = original_check.failure;
    finish();
    return output;
  }
  if (!original_check.suffix_over_limit) {
    output.profile = input;
    output.applied_speed_scale = requested_speed_scale;
    output.status = original_check.prefix_over_limit ?
      CompactDynamicsStatus::kPrefixViolation : CompactDynamicsStatus::kPass;
    if (original_check.prefix_over_limit) {
      output.failure_reason =
        "immutable published prefix exceeds effort limits; suffix is feasible";
    }
    finish();
    return output;
  }

  std::string static_failure;
  bool static_infeasible = false;
  if (!staticPathFeasible(
      *impl_, input, input.prefix_endpoint_sample_index,
      static_infeasible, static_failure))
  {
    output.status = static_infeasible ?
      CompactDynamicsStatus::kStaticInfeasible : CompactDynamicsStatus::kSolverFailure;
    output.failure_reason = std::move(static_failure);
    finish();
    return output;
  }

  const std::size_t prefix_endpoint = input.prefix_endpoint_sample_index;
  std::vector<JointMotionState> immutable_prefix;
  if (prefix_endpoint > 0U) {
    immutable_prefix.assign(
      input.samples.begin(),
      input.samples.begin() + static_cast<std::ptrdiff_t>(prefix_endpoint + 1U));
  }
  struct CandidateAttempt
  {
    std::optional<CompactJointProfile> profile;
    EffortCheck effort;
    std::string failure;
    bool invalid_prefix{false};
    bool solver_failure{false};
  };
  const auto evaluate_candidate = [&](const double common_correction) {
      CandidateAttempt attempt;
      const double candidate_scale = requested_speed_scale * common_correction;
      std::string profile_failure;
      auto candidate = buildCompactJointProfile(
        input.checked_path_vertices, input.samples[input.prefix_endpoint_sample_index],
        motion_limits, control_period_sec, candidate_scale,
        input.checked_path_segment_scales, &profile_failure);
      if (!candidate) {
        attempt.failure = "Ruckig rejected common tempo scale " +
          std::to_string(common_correction) + ": " + profile_failure;
        return attempt;
      }
      if (!immutable_prefix.empty() && !prependValidatedCompactPrefix(
          *candidate, immutable_prefix, control_period_sec, &profile_failure))
      {
        attempt.invalid_prefix = true;
        attempt.failure = "published prefix cannot be preserved: " + profile_failure;
        return attempt;
      }
      try {
        attempt.effort = checkProfileEffort(*impl_, *candidate, control_period_sec);
      } catch (const std::exception & exception) {
        attempt.solver_failure = true;
        attempt.failure = std::string{"DynamicsSolver threw: "} + exception.what();
        return attempt;
      }
      if (!attempt.effort.valid) {
        attempt.solver_failure = true;
        attempt.failure = attempt.effort.failure;
        return attempt;
      }
      attempt.profile = std::move(candidate);
      return attempt;
    };

  double failed_correction = 1.0;
  double passing_correction = 0.0;
  std::optional<CompactJointProfile> best_profile;
  EffortCheck best_effort;
  std::string last_candidate_failure;
  const auto record_attempt = [&](const CandidateAttempt & attempt) {
      if (attempt.profile) {
        output.checked_profile_samples += attempt.effort.profile_samples;
        output.checked_interpolation_samples += attempt.effort.interpolation_samples;
        output.immutable_prefix_over_limit = output.immutable_prefix_over_limit ||
          attempt.effort.prefix_over_limit;
      }
    };
  for (const double correction : kCoarseTempoCorrections) {
    CandidateAttempt attempt = evaluate_candidate(correction);
    if (attempt.invalid_prefix || attempt.solver_failure) {
      output.status = attempt.invalid_prefix ? CompactDynamicsStatus::kInvalidInput :
        CompactDynamicsStatus::kSolverFailure;
      output.failure_reason = std::move(attempt.failure);
      finish();
      return output;
    }
    record_attempt(attempt);
    if (!attempt.profile || attempt.effort.suffix_over_limit) {
      failed_correction = correction;
      last_candidate_failure = attempt.failure.empty() ?
        "common tempo scale " + std::to_string(correction) +
        " still exceeds an effort limit" : std::move(attempt.failure);
      continue;
    }
    passing_correction = correction;
    best_profile = std::move(attempt.profile);
    best_effort = std::move(attempt.effort);
    break;
  }

  if (!best_profile) {
    output.status = CompactDynamicsStatus::kNoFeasibleScale;
    output.failure_reason = last_candidate_failure.empty() ?
      "bounded common tempo candidates did not satisfy all effort limits" :
      last_candidate_failure;
    finish();
    return output;
  }

  for (std::size_t refinement = 0U; refinement < kTempoRefinementPasses; ++refinement) {
    const double correction = 0.5 * (passing_correction + failed_correction);
    CandidateAttempt attempt = evaluate_candidate(correction);
    if (attempt.invalid_prefix || attempt.solver_failure) {
      output.status = attempt.invalid_prefix ? CompactDynamicsStatus::kInvalidInput :
        CompactDynamicsStatus::kSolverFailure;
      output.failure_reason = std::move(attempt.failure);
      finish();
      return output;
    }
    record_attempt(attempt);
    if (!attempt.profile || attempt.effort.suffix_over_limit) {
      failed_correction = correction;
      last_candidate_failure = attempt.failure.empty() ?
        "refined common tempo scale " + std::to_string(correction) +
        " still exceeds an effort limit" : std::move(attempt.failure);
      continue;
    }
    passing_correction = correction;
    best_profile = std::move(attempt.profile);
    best_effort = std::move(attempt.effort);
  }
  output.profile = std::move(best_profile);
  output.applied_speed_scale = requested_speed_scale * passing_correction;
  output.peak_required_effort_nm_after = best_effort.peaks;
  output.maximum_effort_ratio_after = best_effort.maximum_ratio;
  output.status = output.immutable_prefix_over_limit ?
    CompactDynamicsStatus::kPrefixViolation : CompactDynamicsStatus::kRetimed;
  if (output.immutable_prefix_over_limit) {
    output.failure_reason = "immutable published prefix exceeds effort limits; suffix was retimed";
  }
  finish();
  return output;
}

}  // namespace face_tracking_arm::control
