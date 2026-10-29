// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__DETAIL__FEASIBLE_DISTANCE_APPROXIMATION_HPP_
#define FACE_TRACKING_ARM__DETAIL__FEASIBLE_DISTANCE_APPROXIMATION_HPP_

#include <Eigen/Core>

#include <optional>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control::detail
{

struct FeasibleDistanceApproximation
{
  Eigen::VectorXd velocity;
  double alpha{0.0};
};

/// Selects a cost-nonincreasing point on the checked warm-to-raw distance segment.
/// The first N rows are exact joint intervals; other rows use the normal checker tolerance.
[[nodiscard]] std::optional<FeasibleDistanceApproximation>
minimizeFeasibleDistanceSegment(
  QpStageStatus stage_status, QpStageStatus raw_solver_status,
  const Eigen::MatrixXd & hessian, const Eigen::VectorXd & gradient,
  const Eigen::MatrixXd & constraints, const Eigen::VectorXd & lower,
  const Eigen::VectorXd & upper, const Eigen::VectorXd & warm,
  const Eigen::VectorXd & raw, double feasibility_tolerance);

}  // namespace face_tracking_arm::control::detail

#endif  // FACE_TRACKING_ARM__DETAIL__FEASIBLE_DISTANCE_APPROXIMATION_HPP_
