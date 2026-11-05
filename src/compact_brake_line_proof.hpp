// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_BRAKE_LINE_PROOF_HPP_
#define COMPACT_BRAKE_LINE_PROOF_HPP_

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <vector>

#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control
{

struct CompactBrakeLineProofContext
{
  std::uint64_t scene_revision{0U};
  std::uint64_t config_revision{0U};
  const void * model_identity{nullptr};
  const void * group_identity{nullptr};
  Eigen::VectorXd fixed_positions;
};

struct CompactBrakeLinePlan
{
  enum class Result
  {
    kMiss,
    kCovered,
    kValidate,
  };

  Result result{Result::kMiss};
  Eigen::VectorXd start_position;
  Eigen::VectorXd end_position;
  Eigen::VectorXd origin;
  Eigen::VectorXd direction;
  double tube_error_rad{0.0};
  double interval_end{0.0};
  CompactBrakeLineProofContext context;
};

/// One scene/config-bound straight q interval proven for compact brake reuse.
class CompactBrakeLineProof
{
public:
  [[nodiscard]] CompactBrakeLinePlan plan(
    const std::vector<JointMotionState> & tail,
    const CompactBrakeLineProofContext & context, double tube_error_rad,
    std::size_t protected_joint_index, const PositionInterval & protected_interval,
    Radians protected_position_error);

  void commit(const CompactBrakeLinePlan & plan);
  void clear() noexcept;

private:
  bool valid_{false};
  CompactBrakeLineProofContext context_;
  Eigen::VectorXd origin_;
  Eigen::VectorXd direction_;
  double interval_end_{0.0};
  double tube_error_rad_{0.0};
};

}  // namespace face_tracking_arm::control

#endif  // COMPACT_BRAKE_LINE_PROOF_HPP_
