// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/idle_search.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace face_tracking_arm::control
{

void IdleSearch::start(
  const Eigen::VectorXd & current, const Eigen::VectorXd & posture,
  const Eigen::VectorXd & lower, const Eigen::VectorXd & upper,
  const Eigen::Index base_index, const IdleSearchConfig & config, const bool local,
  const double now)
{
  if (current.size() != posture.size() || current.size() != lower.size() ||
    current.size() != upper.size() || base_index < 0 || base_index >= current.size() ||
    !current.allFinite() || !posture.allFinite() || !lower.allFinite() || !upper.allFinite() ||
    (lower.array() >= upper.array()).any() || !std::isfinite(now) ||
    !std::isfinite(config.speed_rad_s) || config.speed_rad_s <= 0.0 ||
    config.speed_rad_s > 0.50 || !std::isfinite(config.sweep_half_range_rad) ||
    config.sweep_half_range_rad <= 0.0 || !std::isfinite(config.local_half_range_rad) ||
    config.local_half_range_rad <= 0.0 || !std::isfinite(config.local_duration_sec) ||
    config.local_duration_sec < 0.0)
  {
    throw std::invalid_argument("Invalid idle search configuration");
  }
  config_ = config;
  base_index_ = base_index;
  posture_ = posture;
  // Choose actual bounded equivalents nearest the current branch, never wrap feedback.
  constexpr double turn = 6.283185307179586;
  for (Eigen::Index i = 0; i < posture.size(); ++i) {
    double nearest = posture[i];
    for (int k = -2; k <= 2; ++k) {
      const double candidate = posture[i] + static_cast<double>(k) * turn;
      if (candidate >= lower[i] && candidate <= upper[i] &&
        std::abs(candidate - current[i]) < std::abs(nearest - current[i]))
      {
        nearest = candidate;
      }
    }
    posture_[i] = nearest;
  }
  const double half = std::min(config.sweep_half_range_rad,
      0.5 * (upper[base_index] - lower[base_index]));
  const double center = std::clamp(current[base_index],
      lower[base_index] + half, upper[base_index] - half);
  lower_ = center - half;
  upper_ = center + half;
  local_lower_ = std::max(lower[base_index], current[base_index] - config.local_half_range_rad);
  local_upper_ = std::min(upper[base_index], current[base_index] + config.local_half_range_rad);
  started_at_ = now;
  direction_ = current[base_index] >= upper_ - 0.10 ? -1 : 1;
  phase_ = local && config.local_duration_sec > 0.0 ? Phase::kLocal : Phase::kPrepare;
  goal_ = current;
  if (preparing()) {
    prepare(current);
  } else {
    goal_[base_index_] = direction_ > 0 ? local_upper_ : local_lower_;
  }
}

void IdleSearch::prepare(const Eigen::VectorXd & current)
{
  phase_ = Phase::kPrepare;
  goal_ = posture_;
  goal_[base_index_] = std::clamp(current[base_index_], lower_, upper_);
}

void IdleSearch::update(const Eigen::VectorXd & current, const double now)
{
  if (phase_ == Phase::kLocal && now - started_at_ >= config_.local_duration_sec) {
    prepare(current);
  }
  if (preparing()) {
    return;
  }
  if (std::abs(current[base_index_] - goal_[base_index_]) < 0.03) {
    direction_ = -direction_;
  }
  const double low = phase_ == Phase::kLocal ? local_lower_ : lower_;
  const double high = phase_ == Phase::kLocal ? local_upper_ : upper_;
  goal_[base_index_] = direction_ > 0 ? high : low;
}

void IdleSearch::prepared()
{
  phase_ = Phase::kSweep;
  goal_[base_index_] = direction_ > 0 ? upper_ : lower_;
}

bool IdleSearch::preparing() const {return phase_ == Phase::kPrepare;}
const Eigen::VectorXd & IdleSearch::goal() const {return goal_;}
std::string IdleSearch::phase() const
{
  return phase_ == Phase::kLocal ? "SEARCH_LOCAL" :
         (preparing() ? "SEARCH_PLANNING" : "SEARCH_SWEEP");
}

}  // namespace face_tracking_arm::control
