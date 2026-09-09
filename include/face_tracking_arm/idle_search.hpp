// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__IDLE_SEARCH_HPP_
#define FACE_TRACKING_ARM__IDLE_SEARCH_HPP_

#include <Eigen/Core>
#include <string>

namespace face_tracking_arm::control
{

struct IdleSearchConfig
{
  double speed_rad_s{0.30};
  double sweep_half_range_rad{2.80};
  double local_half_range_rad{0.35};
  double local_duration_sec{6.0};
};

/// Pure waypoint policy. The existing executor owns all velocity/acceleration/jerk limits.
class IdleSearch
{
public:
  void start(
    const Eigen::VectorXd & current, const Eigen::VectorXd & posture,
    const Eigen::VectorXd & lower, const Eigen::VectorXd & upper,
    Eigen::Index base_index, const IdleSearchConfig & config, bool local, double now);
  void update(const Eigen::VectorXd & current, double now);
  void prepared();
  [[nodiscard]] bool preparing() const;
  [[nodiscard]] const Eigen::VectorXd & goal() const;
  [[nodiscard]] std::string phase() const;

private:
  void prepare(const Eigen::VectorXd & current);
  IdleSearchConfig config_;
  Eigen::VectorXd goal_;
  Eigen::VectorXd posture_;
  Eigen::Index base_index_{0};
  double lower_{0.0};
  double upper_{0.0};
  double local_lower_{0.0};
  double local_upper_{0.0};
  double started_at_{0.0};
  int direction_{1};
  enum class Phase {kLocal, kPrepare, kSweep};
  Phase phase_{Phase::kPrepare};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__IDLE_SEARCH_HPP_
