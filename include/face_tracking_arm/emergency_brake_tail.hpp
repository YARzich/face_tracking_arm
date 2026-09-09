// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__EMERGENCY_BRAKE_TAIL_HPP_
#define FACE_TRACKING_ARM__EMERGENCY_BRAKE_TAIL_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{

/// Configuration of the pre-published, fixed-grid emergency braking suffix.
struct EmergencyBrakeTailConfig
{
  double period_sec{0.01};
  std::size_t maximum_braking_steps{1000};

  /// Extra stationary points after the first exact v=a=0 state. Keep at least
  /// two because MoveIt Servo's trajectory composer withholds its final state.
  std::size_t terminal_hold_points{2};
  /// Numerical tolerance for integrated position/velocity/acceleration. Jerk
  /// is a finite difference, so its matching tolerance is divided by period.
  double comparison_tolerance{1.0e-9};
};

enum class EmergencyBrakeTailStatus
{
  kSuccess,
  kInvalidInput,
  kInfeasible,
  kMaximumStepsExceeded,
};

/// A braking suffix whose first point is the caller-provided branch state.
///
/// Every later point lies exactly one configured period after the previous
/// point. Time stamps deliberately remain the ROS adapter's responsibility.
/// The final points have identical positions and exactly zero velocity and
/// acceleration, making the result suitable for a position-only JTC command.
struct EmergencyBrakeTail
{
  EmergencyBrakeTailStatus status{EmergencyBrakeTailStatus::kInvalidInput};
  std::vector<JointMotionState> points;
  std::size_t first_stationary_point{0};
  std::string failure_reason;

  [[nodiscard]] bool command_available() const noexcept;
};

/// Generates the braking suffix that must accompany every normal trajectory.
///
/// The generator is deterministic and has no ROS, controller or solver state.
/// It applies the same backward-Euler convention as the Servo command queue:
///
///   a_next = a + jerk * period
///   v_next = v + a_next * period
///   q_next = q + v_next * period
///
/// The result proves only joint position/velocity/acceleration/jerk limits.
/// The complete suffix still has to pass PlanningScene segment validation
/// before it is published.
class EmergencyBrakeTailGenerator final
{
public:
  explicit EmergencyBrakeTailGenerator(EmergencyBrakeTailConfig config = {});

  [[nodiscard]] EmergencyBrakeTail generate(
    const JointMotionState & branch_state,
    const JointMotionLimits & limits) const;

  [[nodiscard]] const EmergencyBrakeTailConfig & config() const noexcept;

private:
  EmergencyBrakeTailConfig config_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__EMERGENCY_BRAKE_TAIL_HPP_
