// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_geometry_validator.hpp"

#include <geometric_shapes/bodies.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shapes.h>
#include <urdf/urdf/model.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{
enum class CachedPrimitive
{
  kVertices,
  kSphere,
  kCylinder,
  kCone
};
}  // namespace

struct CompactGeometryValidator::CachedShape
{
  const moveit::core::LinkModel * link{nullptr};
  std::size_t link_index{0};
  Eigen::Isometry3d origin{Eigen::Isometry3d::Identity()};
  CachedPrimitive primitive{CachedPrimitive::kVertices};
  std::vector<Eigen::Vector3d> vertices;
  double radius_m{0.0};
  double length_m{0.0};
};

namespace
{
template<typename Cached, typename Callback>
void forEachRadialSupport(const Cached & shape, Callback callback)
{
  if (shape.primitive == CachedPrimitive::kVertices) {
    for (const auto & vertex : shape.vertices) {
      callback(vertex, 0.0);
    }
    return;
  }
  switch (shape.primitive) {
    case CachedPrimitive::kSphere:
      callback(Eigen::Vector3d::Zero(), shape.radius_m);
      break;
    case CachedPrimitive::kCylinder:
    case CachedPrimitive::kCone:
      callback(Eigen::Vector3d(0.0, 0.0, -0.5 * shape.length_m), shape.radius_m);
      callback(Eigen::Vector3d(0.0, 0.0, 0.5 * shape.length_m), shape.radius_m);
      break;
    case CachedPrimitive::kVertices:
      break;
  }
}

struct RadiusCandidate
{
  std::size_t link_index{0};
  Eigen::Vector3d point_in_link{Eigen::Vector3d::Zero()};
  double radius_m{0.0};
};

bool sameSupportIdentity(const RadiusCandidate & lhs, const RadiusCandidate & rhs)
{
  return lhs.link_index == rhs.link_index &&
         lhs.point_in_link.isApprox(rhs.point_in_link, 1.0e-12) &&
         std::abs(lhs.radius_m - rhs.radius_m) <= 1.0e-12;
}

Eigen::Isometry3d transformFromUrdfPose(const urdf::Pose & pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  transform.linear() = Eigen::Quaterniond(
    pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z).toRotationMatrix();
  return transform;
}

template<typename Cached>
void cacheBoxVertices(Cached & cached, const Eigen::Vector3d & size)
{
  for (const double x : {-0.5, 0.5}) {
    for (const double y : {-0.5, 0.5}) {
      for (const double z : {-0.5, 0.5}) {
        cached.vertices.emplace_back(x * size.x(), y * size.y(), z * size.z());
      }
    }
  }
}

template<typename Cached>
void cacheMeshVertices(Cached & cached, const shapes::Mesh & mesh)
{
  try {
    const bodies::ConvexMesh convex_hull(&mesh);
    const auto & hull_vertices = convex_hull.getVertices();
    const bool usable_hull = hull_vertices.size() >= 4 && std::all_of(
      hull_vertices.begin(), hull_vertices.end(), [](const Eigen::Vector3d & vertex) {
        return vertex.allFinite();
      });
    if (usable_hull) {
      cached.vertices.assign(hull_vertices.begin(), hull_vertices.end());
      return;
    }
  } catch (const std::exception &) {
    // Preserve the exact original-vertex path for invalid or degenerate hulls.
  }

  cached.vertices.reserve(mesh.vertex_count);
  for (unsigned int vertex = 0; vertex < mesh.vertex_count; ++vertex) {
    cached.vertices.emplace_back(
      mesh.vertices[3U * vertex], mesh.vertices[3U * vertex + 1U],
      mesh.vertices[3U * vertex + 2U]);
  }
}

template<typename Cached>
double maximumPrimitiveRadius(const Cached & shape, const Eigen::Isometry3d & transform)
{
  const Eigen::Vector3d center = transform.translation();
  switch (shape.primitive) {
    case CachedPrimitive::kSphere:
      return center.head<2>().norm() + shape.radius_m;
    case CachedPrimitive::kCylinder:
      {
        const Eigen::Vector3d axis = transform.linear().col(2);
        const double half_length = 0.5 * shape.length_m;
        return std::max(
          (center - half_length * axis).head<2>().norm() + shape.radius_m,
          (center + half_length * axis).head<2>().norm() + shape.radius_m);
      }
    case CachedPrimitive::kCone:
      {
        const Eigen::Vector3d axis = transform.linear().col(2);
        const Eigen::Vector3d base = center - 0.5 * shape.length_m * axis;
        const Eigen::Vector3d tip = center + 0.5 * shape.length_m * axis;
        return std::max(
        base.head<2>().norm() + shape.radius_m,
        tip.head<2>().norm() + shape.radius_m);
      }
    case CachedPrimitive::kVertices:
      break;
  }
  return 0.0;
}
}  // namespace

BoundedJointReserve measureBoundedJointReserve(
  const double position_rad, const double lower_bound_rad,
  const double upper_bound_rad) noexcept
{
  BoundedJointReserve result;
  if (!std::isfinite(position_rad) || !std::isfinite(lower_bound_rad) ||
    !std::isfinite(upper_bound_rad) || upper_bound_rad <= lower_bound_rad)
  {
    return result;
  }
  result.lower_margin_rad = position_rad - lower_bound_rad;
  result.upper_margin_rad = upper_bound_rad - position_rad;
  result.nearest_bound_margin_rad =
    std::min(result.lower_margin_rad, result.upper_margin_rad);
  result.inside_bounds = result.lower_margin_rad >= 0.0 && result.upper_margin_rad >= 0.0;
  return result;
}

CompactGeometryValidator::CompactGeometryValidator(
  moveit::core::RobotModelConstPtr robot_model, std::string base_link_name)
: robot_model_(std::move(robot_model))
{
  if (!robot_model_) {
    throw std::invalid_argument("Compact geometry validator requires a robot model");
  }
  base_link_ = robot_model_->getLinkModel(base_link_name);
  if (!base_link_) {
    throw std::invalid_argument("Compact geometry validator base link is not in the model");
  }

  for (const auto * link : robot_model_->getLinkModels()) {
    const std::size_t link_index = links_.size();
    links_.push_back(link);
    const auto & shapes = link->getShapes();
    const auto & origins = link->getCollisionOriginTransforms();
    if (shapes.size() != origins.size()) {
      throw std::invalid_argument("Collision shapes and origin transforms do not match");
    }
    for (std::size_t index = 0; index < shapes.size(); ++index) {
      if (!shapes[index]) {
        throw std::invalid_argument("Robot model contains a null collision shape");
      }
      CachedShape cached;
      cached.link = link;
      cached.link_index = link_index;
      cached.origin = origins[index];
      const auto & shape = *shapes[index];
      switch (shape.type) {
        case shapes::MESH:
          {
            const auto & mesh = static_cast<const shapes::Mesh &>(shape);
            cacheMeshVertices(cached, mesh);
            break;
          }
        case shapes::BOX:
          {
            const auto & box = static_cast<const shapes::Box &>(shape);
            cacheBoxVertices(cached, Eigen::Vector3d(box.size[0], box.size[1], box.size[2]));
            break;
          }
        case shapes::SPHERE:
          cached.primitive = CachedPrimitive::kSphere;
          cached.radius_m = static_cast<const shapes::Sphere &>(shape).radius;
          break;
        case shapes::CYLINDER:
          {
            const auto & cylinder = static_cast<const shapes::Cylinder &>(shape);
            cached.primitive = CachedPrimitive::kCylinder;
            cached.radius_m = cylinder.radius;
            cached.length_m = cylinder.length;
            break;
          }
        case shapes::CONE:
          {
            const auto & cone = static_cast<const shapes::Cone &>(shape);
            cached.primitive = CachedPrimitive::kCone;
            cached.radius_m = cone.radius;
            cached.length_m = cone.length;
            break;
          }
        default:
          throw std::invalid_argument(
                  "Unsupported collision shape in compact radius validator");
      }
      if (cached.vertices.empty() && cached.radius_m <= 0.0) {
        throw std::invalid_argument("Collision shape has no measurable extent");
      }
      shapes_.push_back(std::move(cached));
    }

    const auto urdf_link = robot_model_->getURDF()->getLink(link->getName());
    if (!urdf_link) {
      throw std::invalid_argument("Robot model link is absent from its parsed URDF");
    }
    for (const auto & visual : urdf_link->visual_array) {
      if (!visual || !visual->geometry) {
        throw std::invalid_argument("Robot model contains a visual without geometry");
      }
      CachedShape cached;
      cached.link = link;
      cached.link_index = link_index;
      cached.origin = transformFromUrdfPose(visual->origin);
      switch (visual->geometry->type) {
        case urdf::Geometry::BOX:
          {
            const auto & box = static_cast<const urdf::Box &>(*visual->geometry);
            cacheBoxVertices(cached, Eigen::Vector3d(box.dim.x, box.dim.y, box.dim.z));
            break;
          }
        case urdf::Geometry::SPHERE:
          cached.primitive = CachedPrimitive::kSphere;
          cached.radius_m = static_cast<const urdf::Sphere &>(*visual->geometry).radius;
          break;
        case urdf::Geometry::CYLINDER:
          {
            const auto & cylinder = static_cast<const urdf::Cylinder &>(*visual->geometry);
            cached.primitive = CachedPrimitive::kCylinder;
            cached.radius_m = cylinder.radius;
            cached.length_m = cylinder.length;
            break;
          }
        case urdf::Geometry::MESH:
          {
            const auto & mesh = static_cast<const urdf::Mesh &>(*visual->geometry);
            const Eigen::Vector3d scale(mesh.scale.x, mesh.scale.y, mesh.scale.z);
            std::unique_ptr<shapes::Mesh> loaded_mesh(
              shapes::createMeshFromResource(mesh.filename, scale));
            if (!loaded_mesh) {
              throw std::invalid_argument("Unable to load a visual mesh for compact radius checks");
            }
            cacheMeshVertices(cached, *loaded_mesh);
            break;
          }
        default:
          throw std::invalid_argument("Unsupported visual geometry in compact radius validator");
      }
      if (cached.vertices.empty() && cached.radius_m <= 0.0) {
        throw std::invalid_argument("Visual shape has no measurable extent");
      }
      shapes_.push_back(std::move(cached));
    }
  }
  if (shapes_.empty()) {
    throw std::invalid_argument("Robot model has no visual or collision geometry to measure");
  }
}

CompactGeometryValidator::~CompactGeometryValidator() = default;

double CompactGeometryValidator::maximumAssemblyRadius(
  const moveit::core::RobotState & state) const
{
  if (state.getRobotModel().get() != robot_model_.get()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const Eigen::Isometry3d base_inverse =
    state.getGlobalLinkTransform(base_link_).inverse();
  double maximum_radius = 0.0;
  for (const auto & shape : shapes_) {
    const Eigen::Isometry3d transform = base_inverse *
      state.getGlobalLinkTransform(shape.link) * shape.origin;
    if (!transform.matrix().allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    forEachRadialSupport(shape, [&maximum_radius, &transform](
        const Eigen::Vector3d & local_point, const double radial_offset_m) {
        const Eigen::Vector3d point = transform * local_point;
        maximum_radius = std::max(
          maximum_radius, point.head<2>().norm() + radial_offset_m);
      });
  }
  return maximum_radius;
}

std::optional<CompactRadiusSupportSet> CompactGeometryValidator::perLinkRadiusSupports(
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group,
  const double coactive_tolerance_m,
  const std::size_t maximum_supports) const
{
  const Eigen::Index variable_count = static_cast<Eigen::Index>(group.getVariableCount());
  if (state.getRobotModel().get() != robot_model_.get() ||
    &group.getParentModel() != robot_model_.get() || variable_count <= 0 ||
    !std::isfinite(coactive_tolerance_m) || coactive_tolerance_m < 0.0 ||
    maximum_supports == 0 || links_.empty())
  {
    return std::nullopt;
  }

  const Eigen::Isometry3d base_transform = state.getGlobalLinkTransform(base_link_);
  if (!base_transform.matrix().allFinite()) {
    return std::nullopt;
  }
  const Eigen::Isometry3d base_inverse = base_transform.inverse();
  std::vector<Eigen::Isometry3d> base_to_link;
  base_to_link.reserve(links_.size());
  for (const auto * link : links_) {
    const Eigen::Isometry3d transform = base_inverse * state.getGlobalLinkTransform(link);
    if (!transform.matrix().allFinite()) {
      return std::nullopt;
    }
    base_to_link.push_back(transform);
  }

  std::vector<double> link_maximum(links_.size(), 0.0);
  std::vector<bool> link_has_geometry(links_.size(), false);
  for (const auto & shape : shapes_) {
    link_has_geometry[shape.link_index] = true;
    const Eigen::Isometry3d transform = base_to_link[shape.link_index] * shape.origin;
    if (!transform.matrix().allFinite()) {
      return std::nullopt;
    }
    forEachRadialSupport(shape, [&link_maximum, &transform, &shape](
        const Eigen::Vector3d & local_point, const double radial_offset_m) {
        const Eigen::Vector3d point = transform * local_point;
        link_maximum[shape.link_index] = std::max(
          link_maximum[shape.link_index], point.head<2>().norm() + radial_offset_m);
      });
  }

  CompactRadiusSupportSet result;
  result.maximum_radius_m = 0.0;
  for (std::size_t index = 0; index < link_maximum.size(); ++index) {
    if (link_has_geometry[index]) {
      result.maximum_radius_m = std::max(result.maximum_radius_m, link_maximum[index]);
    }
  }

  const auto & group_joints = group.getJointModels();
  if (group_joints.empty() || group_joints.front()->getParentLinkModel() == nullptr) {
    return std::nullopt;
  }
  const auto & updated_links = group.getUpdatedLinkModelsSet();

  std::vector<RadiusCandidate> candidates;
  candidates.reserve(std::min(maximum_supports, shapes_.size()));
  bool support_overflow = false;
  for (const auto & shape : shapes_) {
    // If neither the base nor this geometry link moves with the group, its
    // base-relative radius is constant and is already covered by the global
    // envelope. Do not spend bounded support rows on a fixed world envelope.
    if (updated_links.count(base_link_) == 0U &&
      updated_links.count(shape.link) == 0U)
    {
      continue;
    }
    const Eigen::Isometry3d transform = base_to_link[shape.link_index] * shape.origin;
    forEachRadialSupport(shape, [&candidates, &shape, &transform, &link_maximum,
      coactive_tolerance_m, maximum_supports, &support_overflow](
        const Eigen::Vector3d & local_point, const double radial_offset_m) {
        const Eigen::Vector3d point_in_link = shape.origin * local_point;
        const double radius_m = (transform * local_point).head<2>().norm() + radial_offset_m;
        if (link_maximum[shape.link_index] - radius_m > coactive_tolerance_m) {
          return;
        }
        const RadiusCandidate candidate{shape.link_index, point_in_link, radius_m};
        if (std::any_of(candidates.begin(), candidates.end(), [&candidate](
          const RadiusCandidate & retained) {
          return sameSupportIdentity(retained, candidate);
          }))
        {
          return;
        }
        if (candidates.size() >= maximum_supports) {
          support_overflow = true;
          return;
        }
        candidates.push_back(candidate);
      });
    if (support_overflow) {
      return std::nullopt;
    }
  }
  if (candidates.empty()) {
    return result;
  }
  const Eigen::Isometry3d reference_transform = state.getGlobalLinkTransform(
    group_joints.front()->getParentLinkModel());
  if (!reference_transform.matrix().allFinite()) {
    return std::nullopt;
  }
  const Eigen::Matrix3d reference_to_model = reference_transform.linear();
  Eigen::MatrixXd base_jacobian = Eigen::MatrixXd::Zero(6, variable_count);
  if (updated_links.count(base_link_) != 0U &&
    (!state.getJacobian(&group, base_link_, Eigen::Vector3d::Zero(), base_jacobian, false) ||
    base_jacobian.rows() != 6 || base_jacobian.cols() != variable_count ||
    !base_jacobian.array().isFinite().all()))
  {
    return std::nullopt;
  }
  Eigen::MatrixXd base_jacobian_model = base_jacobian;
  base_jacobian_model.topRows(3) = reference_to_model * base_jacobian.topRows(3);
  base_jacobian_model.bottomRows(3) = reference_to_model * base_jacobian.bottomRows(3);
  const Eigen::Vector3d base_origin_model = base_transform.translation();
  result.supports.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    const auto * link = links_[candidate.link_index];
    Eigen::MatrixXd point_jacobian = Eigen::MatrixXd::Zero(6, variable_count);
    if (updated_links.count(link) != 0U &&
      (!state.getJacobian(&group, link, candidate.point_in_link, point_jacobian, false) ||
      point_jacobian.rows() != 6 || point_jacobian.cols() != variable_count ||
      !point_jacobian.array().isFinite().all()))
    {
      return std::nullopt;
    }
    Eigen::MatrixXd point_jacobian_model = point_jacobian;
    point_jacobian_model.topRows(3) = reference_to_model * point_jacobian.topRows(3);
    point_jacobian_model.bottomRows(3) = reference_to_model * point_jacobian.bottomRows(3);
    const Eigen::Vector3d point_base = base_inverse *
      (state.getGlobalLinkTransform(link) * candidate.point_in_link);
    const Eigen::Vector3d point_model = state.getGlobalLinkTransform(link) *
      candidate.point_in_link;
    Eigen::MatrixXd point_velocity_model =
      point_jacobian_model.topRows(3) - base_jacobian_model.topRows(3);
    const Eigen::Vector3d point_offset_model = point_model - base_origin_model;
    for (Eigen::Index column = 0; column < variable_count; ++column) {
      // Remove the base-frame rotation before expressing the relative point
      // velocity in the base link's rotating coordinate frame.
      const Eigen::Vector3d base_angular_velocity =
        base_jacobian_model.block<3, 1>(3, column);
      point_velocity_model.col(column) += point_offset_model.cross(
        base_angular_velocity);
    }
    const Eigen::MatrixXd point_velocity_in_base =
      base_transform.linear().transpose() * point_velocity_model;

    CompactRadiusSupport support;
    support.radius_m = candidate.radius_m;
    support.outward_gradient = Eigen::VectorXd::Zero(variable_count);
    const double radial_norm = point_base.head<2>().norm();
    if (radial_norm > 1.0e-12) {
      Eigen::RowVector3d radial_direction = Eigen::RowVector3d::Zero();
      radial_direction.head<2>() = (point_base.head<2>() / radial_norm).transpose();
      support.outward_gradient = (
        radial_direction.head<2>() * point_velocity_in_base.topRows(2)).transpose();
    } else if (point_velocity_in_base.topRows(2).norm() > 1.0e-10) {
      return std::nullopt;
    }
    if (!support.outward_gradient.array().isFinite().all()) {
      return std::nullopt;
    }
    result.supports.push_back(std::move(support));
  }
  return result;
}

}  // namespace face_tracking_arm::control
