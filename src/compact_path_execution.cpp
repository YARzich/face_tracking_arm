// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_path_execution.hpp"

#include <cmath>
#include <algorithm>
#include <string>
#include <utility>

namespace face_tracking_arm::control
{

void CompactPathExecution::install(
  CompactJointProfile profile, JointMotionLimits prefix_limits,
  const double initial_progress_rad, const std::size_t initial_cursor)
{
  if (profile.samples.empty()) {
    clear();
    return;
  }
  profile_ = std::move(profile);
  cursor_ = std::min(initial_cursor, profile_->samples.size() - 1U);
  prefix_endpoint_sample_index_ = profile_->prefix_endpoint_sample_index;
  progress_rad_ = std::isfinite(initial_progress_rad) ? initial_progress_rad : 0.0;
  double local_progress = 0.0;
  if (profile_->path_progress_rad_per_sample.size() == profile_->samples.size()) {
    local_progress = profile_->path_progress_rad_per_sample[cursor_];
  } else {
    for (std::size_t index = 1U; index <= cursor_; ++index) {
      local_progress += (profile_->samples[index].position -
        profile_->samples[index - 1U].position).norm();
    }
  }
  profile_progress_origin_rad_ = progress_rad_ - local_progress;
  prefix_limits_ = std::move(prefix_limits);
}

void CompactPathExecution::clear() noexcept
{
  profile_.reset();
  cursor_ = 0U;
  prefix_endpoint_sample_index_ = 0U;
  progress_rad_ = 0.0;
  profile_progress_origin_rad_ = 0.0;
}

bool CompactPathExecution::active() const noexcept
{
  return profile_.has_value();
}

const CompactJointProfile * CompactPathExecution::profile() const noexcept
{
  return profile_ ? &*profile_ : nullptr;
}

std::size_t CompactPathExecution::cursor() const noexcept
{
  return cursor_;
}

std::size_t CompactPathExecution::prefixEndpoint() const noexcept
{
  return prefix_endpoint_sample_index_;
}

double CompactPathExecution::progressRad() const noexcept
{
  return progress_rad_;
}

double CompactPathExecution::absoluteProgressRad(const double profile_progress_rad) const noexcept
{
  return profile_progress_origin_rad_ + profile_progress_rad;
}

double CompactPathExecution::totalPathLengthRad() const noexcept
{
  return profile_ ? profile_progress_origin_rad_ + profile_->path_length_rad :
         profile_progress_origin_rad_;
}

const JointMotionLimits & CompactPathExecution::prefixLimits() const noexcept
{
  return prefix_limits_;
}

CompactQueueMatch CompactPathExecution::matchQueueState(
  const JointMotionState & queue_state, const double position_tolerance_rad,
  const double velocity_tolerance_rad_s,
  const double acceleration_tolerance_rad_s2) const noexcept
{
  CompactQueueMatch result;
  if (!profile_ || cursor_ >= profile_->samples.size() ||
    !std::isfinite(position_tolerance_rad) || position_tolerance_rad < 0.0 ||
    !std::isfinite(velocity_tolerance_rad_s) || velocity_tolerance_rad_s < 0.0 ||
    !std::isfinite(acceleration_tolerance_rad_s2) || acceleration_tolerance_rad_s2 < 0.0)
  {
    return result;
  }
  const auto matches = [&](const JointMotionState & expected) {
      return queue_state.position.size() == expected.position.size() &&
             queue_state.velocity.size() == expected.velocity.size() &&
             queue_state.acceleration.size() == expected.acceleration.size() &&
             queue_state.position.allFinite() && queue_state.velocity.allFinite() &&
             queue_state.acceleration.allFinite() &&
             (queue_state.position - expected.position).lpNorm<Eigen::Infinity>() <=
             position_tolerance_rad &&
             (queue_state.velocity - expected.velocity).lpNorm<Eigen::Infinity>() <=
             velocity_tolerance_rad_s &&
             (queue_state.acceleration - expected.acceleration).lpNorm<Eigen::Infinity>() <=
             acceleration_tolerance_rad_s2;
    };
  if (matches(profile_->samples[cursor_])) {
    result.status = CompactQueueMatchStatus::kMatchedCurrent;
    result.sample_index = cursor_;
    return result;
  }
  for (std::size_t index = cursor_ + 1U; index < profile_->samples.size(); ++index) {
    if (matches(profile_->samples[index])) {
      result.status = CompactQueueMatchStatus::kMatchedFuture;
      result.sample_index = index;
      return result;
    }
  }
  return result;
}

bool CompactPathExecution::commitPublishedQueueMatch(const std::size_t sample_index) noexcept
{
  if (!profile_ || sample_index < cursor_ || sample_index >= profile_->samples.size()) {
    return false;
  }
  const auto & progress = profile_->path_progress_rad_per_sample;
  if (progress.size() == profile_->samples.size()) {
    progress_rad_ = profile_progress_origin_rad_ + progress[sample_index];
  } else {
    for (std::size_t index = cursor_ + 1U; index <= sample_index; ++index) {
      progress_rad_ += (profile_->samples[index].position -
        profile_->samples[index - 1U].position).norm();
    }
  }
  cursor_ = sample_index;
  return true;
}

std::optional<double> CompactPathExecution::projectQueueStateToProgress(
  const JointMotionState & queue_state, const double residual_limit_rad,
  const std::optional<double> minimum_progress_rad) const
{
  if (!profile_ || profile_->path_progress_rad_per_sample.size() !=
    profile_->samples.size() || profile_->samples.size() < 2U ||
    !std::isfinite(residual_limit_rad) || residual_limit_rad < 0.0 ||
    queue_state.position.size() == 0 || queue_state.velocity.size() !=
    queue_state.position.size() || !queue_state.position.allFinite() ||
    !queue_state.velocity.allFinite())
  {
    return std::nullopt;
  }
  const auto & progress = profile_->path_progress_rad_per_sample;
  if (cursor_ >= progress.size()) {
    return std::nullopt;
  }
  const double minimum_progress = minimum_progress_rad.value_or(progress[cursor_]);
  const double maximum_progress = progress.back();
  if (!std::isfinite(minimum_progress) || !std::isfinite(maximum_progress) ||
    maximum_progress < minimum_progress)
  {
    return std::nullopt;
  }
  return projectCompactProfileProgress(
    *profile_, queue_state.position, queue_state.velocity,
    minimum_progress, maximum_progress, residual_limit_rad);
}

std::optional<CompactSmoothPathSuffix> CompactPathExecution::remainingRoute(
  const double progress_rad) const
{
  if (!profile_ || !std::isfinite(progress_rad) || profile_->samples.empty() ||
    profile_->path_progress_rad_per_sample.size() != profile_->samples.size() ||
    profile_->speed_scale_per_interval.size() + 1U != profile_->samples.size())
  {
    return std::nullopt;
  }
  const auto & progress = profile_->path_progress_rad_per_sample;
  if (!std::isfinite(progress.front()) || !std::isfinite(progress.back()) ||
    progress_rad < progress.front() - 1.0e-9 || progress_rad > progress.back() + 1.0e-9)
  {
    return std::nullopt;
  }

  CompactSmoothPathSuffix suffix;
  suffix.start_progress_rad = std::clamp(progress_rad, progress.front(), progress.back());
  const auto & samples = profile_->samples;
  if (samples.size() == 1U || suffix.start_progress_rad >= progress.back() - 1.0e-12) {
    suffix.path.push_back(samples.back().position);
    return suffix;
  }
  std::size_t interval = 0U;
  while (interval + 1U < progress.size() &&
    progress[interval + 1U] <= suffix.start_progress_rad + 1.0e-12)
  {
    ++interval;
  }
  if (interval + 1U >= samples.size()) {
    return std::nullopt;
  }
  for (std::size_t index = 0U; index < progress.size(); ++index) {
    if (!std::isfinite(progress[index]) ||
      (index > 0U && progress[index] + 1.0e-12 < progress[index - 1U]))
    {
      return std::nullopt;
    }
  }
  const double span = progress[interval + 1U] - progress[interval];
  if (span <= 0.0) {
    return std::nullopt;
  }
  const double fraction = std::clamp(
    (suffix.start_progress_rad - progress[interval]) / span, 0.0, 1.0);
  suffix.path.push_back(
    samples[interval].position + fraction *
    (samples[interval + 1U].position - samples[interval].position));
  for (std::size_t index = interval; index + 1U < samples.size(); ++index) {
    suffix.speed_scales.push_back(profile_->speed_scale_per_interval[index]);
    suffix.path.push_back(samples[index + 1U].position);
  }
  return suffix;
}

void CompactPathExecution::recordCommandPublished(const bool published) noexcept
{
  if (!published || !profile_ || cursor_ + 1U >= profile_->samples.size()) {
    return;
  }
  const auto & samples = profile_->samples;
  if (profile_->path_progress_rad_per_sample.size() == samples.size()) {
    progress_rad_ = profile_progress_origin_rad_ +
      profile_->path_progress_rad_per_sample[cursor_ + 1U];
  } else {
    progress_rad_ += (samples[cursor_ + 1U].position - samples[cursor_].position).norm();
  }
  ++cursor_;
}

EmergencyBrakeTail CompactPathExecution::brakingTail(
  const JointMotionState & candidate_state, const JointMotionLimits & limits,
  const EmergencyBrakeTailConfig & config) const
{
  EmergencyBrakeTail failure;
  const auto fail = [&failure](const std::string & reason) {
      failure.status = EmergencyBrakeTailStatus::kInfeasible;
      failure.failure_reason = reason;
      return failure;
    };
  if (!profile_ || cursor_ + 1U >= profile_->samples.size() ||
    !std::isfinite(config.period_sec) || config.period_sec <= 0.0 ||
    config.maximum_braking_steps == 0U || config.terminal_hold_points < 2U)
  {
    failure.failure_reason = "no active direct compact PATH sample is available";
    return failure;
  }
  const auto & expected = profile_->samples[cursor_ + 1U];
  if (!limits.numerical_policy || !limits.numerical_policy->valid() ||
    candidate_state.position.size() != expected.position.size() ||
    candidate_state.velocity.size() != expected.velocity.size() ||
    candidate_state.acceleration.size() != expected.acceleration.size() ||
    !candidate_state.position.allFinite() || !candidate_state.velocity.allFinite() ||
    !candidate_state.acceleration.allFinite())
  {
    return fail("candidate state or selected compact command limits are invalid");
  }
  if ((candidate_state.velocity - expected.velocity).lpNorm<Eigen::Infinity>() >
    limits.numerical_policy->velocity_error.value)
  {
    return fail("candidate velocity does not match the next committed profile sample");
  }
  if (cursor_ < prefix_endpoint_sample_index_ &&
    ((candidate_state.position - expected.position).lpNorm<Eigen::Infinity>() >
    limits.numerical_policy->position_error.value ||
    (candidate_state.acceleration - expected.acceleration).lpNorm<Eigen::Infinity>() >
    limits.numerical_policy->acceleration_error.value))
  {
    return fail("candidate state does not match the next committed profile sample");
  }

  const auto sample_within_limits = [&limits, &config](
    const JointMotionState & sample, const JointMotionState * previous) {
      const Eigen::Index count = sample.position.size();
      if (count <= 0 || sample.velocity.size() != count ||
        sample.acceleration.size() != count || limits.lower_position.size() != count ||
        limits.upper_position.size() != count || limits.max_velocity.size() != count ||
        limits.max_acceleration.size() != count || limits.max_jerk.size() != count ||
        !sample.position.allFinite() || !sample.velocity.allFinite() ||
        !sample.acceleration.allFinite() ||
        (sample.position.array() < limits.lower_position.array()).any() ||
        (sample.position.array() > limits.upper_position.array()).any() ||
        (sample.velocity.cwiseAbs().array() > limits.max_velocity.array() +
        limits.numerical_policy->velocity_error.value).any() ||
        (sample.acceleration.cwiseAbs().array() > limits.max_acceleration.array() +
        limits.numerical_policy->acceleration_error.value).any())
      {
        return false;
      }
      if (previous != nullptr) {
        const Eigen::VectorXd jerk = (sample.acceleration - previous->acceleration) /
          config.period_sec;
        if (!jerk.allFinite() || (jerk.cwiseAbs().array() > limits.max_jerk.array() +
          limits.numerical_policy->jerk_error.value).any())
        {
          return false;
        }
      }
      return true;
    };
  if (!sample_within_limits(candidate_state, nullptr)) {
    return fail("candidate state exceeds selected compact command limits");
  }

  EmergencyBrakeTail tail;
  tail.points.push_back(candidate_state);
  const std::size_t candidate_index = cursor_ + 1U;
  if (cursor_ < prefix_endpoint_sample_index_) {
    if (candidate_index > prefix_endpoint_sample_index_) {
      return fail("candidate cursor is outside the copied published prefix");
    }
    for (std::size_t index = candidate_index + 1U;
      index <= prefix_endpoint_sample_index_; ++index)
    {
      const auto & sample = profile_->samples[index];
      if (!sample_within_limits(sample, &tail.points.back())) {
        return fail("copied published braking suffix exceeds selected limits at sample " +
          std::to_string(index - candidate_index));
      }
      tail.points.push_back(sample);
    }
    if (!tail.points.back().velocity.isZero(0.0) ||
      !tail.points.back().acceleration.isZero(0.0))
    {
      return fail("copied published prefix has no exact stationary endpoint");
    }
    tail.first_stationary_point = tail.points.size() - 1U;
    for (std::size_t hold = 0U; hold < config.terminal_hold_points; ++hold) {
      tail.points.push_back(tail.points[tail.first_stationary_point]);
    }
    tail.status = EmergencyBrakeTailStatus::kSuccess;
    return tail;
  }

  Eigen::VectorXd path_direction = Eigen::VectorXd::Zero(candidate_state.position.size());
  if (!candidate_state.velocity.isZero(0.0)) {
    path_direction = candidate_state.velocity;
  } else if (!candidate_state.acceleration.isZero(0.0)) {
    path_direction = candidate_state.acceleration;
  }
  return generateCompactProfileBrakeTail(candidate_state, path_direction, limits, config);
}

}  // namespace face_tracking_arm::control
