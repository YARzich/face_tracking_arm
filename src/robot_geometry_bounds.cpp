// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <moveit/robot_model/robot_model.hpp>

#include "face_tracking_arm/collision_constraints.hpp"

namespace face_tracking_arm::control
{
namespace
{

using JointRadii = std::map<std::string, double>;

double shapeRadius(const shapes::Shape & shape)
{
  if (shape.type == shapes::MESH) {
    const auto & mesh = static_cast<const shapes::Mesh &>(shape);
    double radius = 0.0;
    for (unsigned int index = 0; index < mesh.vertex_count; ++index) {
      radius = std::max(radius, Eigen::Map<const Eigen::Vector3d>(
          mesh.vertices + 3U * index).norm());
    }
    return radius;
  }
  const double radius = 0.5 * shapes::computeShapeExtents(&shape).norm();
  if (!std::isfinite(radius)) {
    throw std::invalid_argument("robot collision shapes must have finite extent");
  }
  return radius;
}

JointRadii jointRadii(
  const moveit::core::LinkModel & body, const moveit::core::JointModelGroup & group)
{
  double radius = 0.0;
  const auto & shapes = body.getShapes();
  const auto & origins = body.getCollisionOriginTransforms();
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    radius = std::max(radius, origins[index].translation().norm() + shapeRadius(*shapes[index]));
  }
  JointRadii result;
  for (const auto * link = &body; link != nullptr; link = link->getParentLinkModel()) {
    const auto * joint = link->getParentJointModel();
    if (joint && group.hasJointModel(joint->getName())) {
      if (joint->getType() == moveit::core::JointModel::REVOLUTE) {
        result.emplace(joint->getName(), radius);
      } else if (joint->getType() != moveit::core::JointModel::FIXED) {
        throw std::invalid_argument("geometry bounds require fixed/revolute arm joints");
      }
    }
    radius += link->getJointOriginTransform().translation().norm();
  }
  return result;
}

double relativeBound(const JointRadii & first, const JointRadii & second)
{
  double bound = 0.0;
  // Common ancestors rotate both bodies rigidly and cannot change their distance.
  for (const auto & [joint, radius] : first) {
    if (second.count(joint) == 0U) {
      bound += radius;
    }
  }
  for (const auto & [joint, radius] : second) {
    if (first.count(joint) == 0U) {
      bound += radius;
    }
  }
  return bound;
}

}  // namespace

void configureGeometryBounds(
  CollisionConstraintConfig & config, const moveit::core::RobotModel & model,
  const moveit::core::JointModelGroup & group)
{
  config.distance_bounds.clear();
  std::map<std::string, JointRadii> radii;
  double maximum_world_bound = 0.0;
  for (const auto * link : model.getLinkModelsWithCollisionGeometry()) {
    auto joint_radii = jointRadii(*link, group);
    const double world_bound = relativeBound(joint_radii, {});
    maximum_world_bound = std::max(maximum_world_bound, world_bound);
    config.distance_bounds[{link->getName(), ""}] = world_bound;
    radii.emplace(link->getName(), std::move(joint_radii));
  }
  for (auto first = radii.begin(); first != radii.end(); ++first) {
    for (auto second = std::next(first); second != radii.end(); ++second) {
      config.distance_bounds[{first->first, second->first}] =
        relativeBound(first->second, second->second);
    }
  }
  config.default_distance_lipschitz_m_per_rad = std::max(
    config.default_distance_lipschitz_m_per_rad, 2.0 * maximum_world_bound);
  config.monitor_near_distance_lipschitz_m_per_rad =
    config.default_distance_lipschitz_m_per_rad;
}

}  // namespace face_tracking_arm::control
