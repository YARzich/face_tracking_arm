// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TABLE_GEOMETRY_HPP_
#define FACE_TRACKING_ARM__TABLE_GEOMETRY_HPP_

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace face_tracking_arm::tracking
{

inline moveit_msgs::msg::CollisionObject make_table(
  const std::string & shape, const std::vector<double> & dimensions,
  const std::vector<double> & center, const double yaw)
{
  const auto positive = [](double x) {return std::isfinite(x) && x > 0.0;};
  const auto finite = [](double x) {return std::isfinite(x);};
  if ((shape != "box" && shape != "cylinder") ||
    dimensions.size() != (shape == "box" ? 3U : 2U) || center.size() != 3 ||
    !std::all_of(dimensions.begin(), dimensions.end(), positive) ||
    !std::all_of(center.begin(), center.end(), finite) || !std::isfinite(yaw))
  {
    throw std::invalid_argument("Invalid table shape, dimensions or pose");
  }
  moveit_msgs::msg::CollisionObject table;
  table.header.frame_id = "world";
  table.id = "round_table";  // Preserve the public collision-object identifier.
  table.operation = moveit_msgs::msg::CollisionObject::ADD;
  table.pose.orientation.w = 1.0;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape == "box" ? primitive.BOX : primitive.CYLINDER;
  primitive.dimensions.assign(dimensions.begin(), dimensions.end());
  geometry_msgs::msg::Pose pose;
  pose.position.x = center[0];
  pose.position.y = center[1];
  pose.position.z = center[2];
  pose.orientation.z = std::sin(yaw / 2.0);
  pose.orientation.w = std::cos(yaw / 2.0);
  table.primitives.push_back(primitive);
  table.primitive_poses.push_back(pose);
  return table;
}

inline Eigen::Isometry3d table_pose(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
    pose.orientation.y, pose.orientation.z);
  if (q.norm() > 1e-12) {
    result.linear() = q.normalized().toRotationMatrix();
  }
  return result;
}

inline bool has_table(
  const moveit_msgs::msg::PlanningScene & scene,
  const moveit_msgs::msg::CollisionObject & expected)
{
  for (const auto & object : scene.world.collision_objects) {
    if (object.id != expected.id || object.header.frame_id != expected.header.frame_id ||
      object.primitives.size() != 1 || object.primitive_poses.size() != 1 ||
      object.primitives[0] != expected.primitives[0])
    {
      continue;
    }
    const auto actual_pose = table_pose(object.pose) * table_pose(object.primitive_poses[0]);
    const auto expected_pose = table_pose(expected.pose) * table_pose(expected.primitive_poses[0]);
    if (actual_pose.matrix().isApprox(expected_pose.matrix(), 1e-8)) {
      return true;
    }
  }
  return false;
}

}  // namespace face_tracking_arm::tracking
#endif  // FACE_TRACKING_ARM__TABLE_GEOMETRY_HPP_
