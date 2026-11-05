// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_path_attachment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "servo_types.hpp"

namespace face_tracking_arm::control::detail
{
namespace
{
constexpr double kCompactMetricTolerance{1.0e-6};
}  // namespace

CompactJointProfile compactProfileForPublishedPrefix(
  const std::vector<JointMotionState> & prefix, const double period_sec)
{
  CompactJointProfile profile;
  profile.samples = prefix;
  if (prefix.empty()) {
    return profile;
  }
  profile.prefix_endpoint_sample_index = prefix.size() - 1U;
  profile.duration_sec = static_cast<double>(profile.prefix_endpoint_sample_index) * period_sec;
  for (std::size_t index = 1U; index < prefix.size(); ++index) {
    profile.path_length_rad += (prefix[index].position - prefix[index - 1U].position).norm();
  }
  return profile;
}

/// Select an entry using only worker samples and suffix maxima from the same
/// immutable coarse path. Certificate gaps fail closed before suffix maxima are trusted.
CompactAttachmentSelection selectCompactAttachment(
  const BackgroundPathResult & plan, const Eigen::VectorXd & actual,
  const Eigen::VectorXd & queue, const double radius_cap_m)
{
  constexpr double kMaximumAttachDistanceRad{0.20};
  constexpr std::size_t kMaximumCompactCertificateSamples{8192U};
  CompactAttachmentSelection result;
  if (plan.path.size() < 2U) {
    result.failure_reason = "path_too_short";
    return result;
  }
  const auto within_caps = [radius_cap_m](const double radius) {
      return std::isfinite(radius) && radius >= 0.0 &&
             radius <= radius_cap_m + kCompactMetricTolerance;
    };
  if (plan.compact_path_samples.empty()) {
    if (!plan.validation_metrics || !within_caps(
        plan.validation_metrics->maximum_assembly_radius_m))
    {
      result.failure_reason = plan.validation_metrics ?
        "uncertified_path_metrics_outside_caps" : "missing_path_metrics";
      return result;
    }
    result.valid = true;
    result.position = plan.start_positions;
    result.maximum_radius_m = plan.validation_metrics->maximum_assembly_radius_m;
    result.maximum_roll_error_rad = plan.validation_metrics->maximum_roll_error_rad;
    result.maximum_abs_pitch_rad = plan.validation_metrics->maximum_abs_screen_pitch_rad;
    return result;
  }

  result.used_certificate = true;
  if (plan.compact_path_samples.size() > kMaximumCompactCertificateSamples ||
    !plan.compact_goal_metrics)
  {
    result.failure_reason = plan.compact_goal_metrics ?
      "certificate_sample_limit_exceeded" : "missing_exact_goal_metrics";
    return result;
  }
  const auto finite_radius = [](const double radius) {
      return std::isfinite(radius) && radius >= 0.0;
    };
  const auto & goal_metrics = *plan.compact_goal_metrics;
  if (!finite_radius(goal_metrics.maximum_assembly_radius_m) ||
    !within_caps(goal_metrics.maximum_assembly_radius_m))
  {
    result.failure_reason = "exact_goal_metrics_invalid_or_outside_caps";
    return result;
  }

  bool certificate_valid = true;
  std::string certificate_failure = "certificate_sample_invalid";
  double suffix_radius = goal_metrics.maximum_assembly_radius_m;
  for (std::size_t reverse_index = plan.compact_path_samples.size();
    certificate_valid && reverse_index > 0U; --reverse_index)
  {
    const auto & sample = plan.compact_path_samples[reverse_index - 1U];
    certificate_valid = sample.original_segment_index < plan.path.size() - 1U &&
      sample.positions.size() == plan.start_positions.size() && sample.positions.allFinite() &&
      std::isfinite(sample.fraction) && sample.fraction >= 0.0 && sample.fraction <= 1.0 &&
      finite_radius(sample.maximum_assembly_radius_m) &&
      finite_radius(sample.suffix_maximum_assembly_radius_m);
    if (!certificate_valid) {
      break;
    }
    const Eigen::VectorXd expected_position =
      plan.path[sample.original_segment_index] + sample.fraction *
      (plan.path[sample.original_segment_index + 1U] -
      plan.path[sample.original_segment_index]);
    certificate_valid = sample.positions.isApprox(expected_position, 1.0e-9);
    if (!certificate_valid) {
      certificate_failure = "certificate_mapping_mismatch";
      break;
    }
    if (reverse_index < plan.compact_path_samples.size()) {
      const auto & following = plan.compact_path_samples[reverse_index];
      const bool same_segment = following.original_segment_index ==
        sample.original_segment_index;
      const bool next_segment = following.original_segment_index ==
        sample.original_segment_index + 1U;
      const double sample_gap =
        (following.positions - sample.positions).lpNorm<Eigen::Infinity>();
      certificate_valid = certificate_valid &&
        (same_segment || (next_segment && std::abs(sample.fraction - 1.0) <= 1.0e-9)) &&
        (!same_segment || following.fraction >= sample.fraction) &&
        std::isfinite(sample_gap) && sample_gap <= 0.0100001;
      if (!certificate_valid) {
        certificate_failure = "certificate_gap_or_order_mismatch";
        break;
      }
    }
    suffix_radius = std::max(suffix_radius, sample.maximum_assembly_radius_m);
    certificate_valid = certificate_valid &&
      std::abs(sample.suffix_maximum_assembly_radius_m - suffix_radius) <= 1.0e-9;
    if (!certificate_valid) {
      certificate_failure = "certificate_suffix_metrics_mismatch";
    }
  }
  if (!certificate_valid) {
    result.failure_reason = certificate_failure;
    return result;
  }
  const auto & first = plan.compact_path_samples.front();
  const auto & last = plan.compact_path_samples.back();
  if (first.original_segment_index != 0U || std::abs(first.fraction) > 1.0e-12 ||
    !first.positions.isApprox(plan.path.front(), 1.0e-12) ||
    last.original_segment_index != plan.path.size() - 2U ||
    std::abs(last.fraction - 1.0) > 1.0e-12 ||
    !last.positions.isApprox(plan.path.back(), 1.0e-12))
  {
    result.failure_reason = "certificate_endpoint_mismatch";
    return result;
  }

  const std::size_t first_segment_limit = std::min<std::size_t>(4U, plan.path.size() - 1U);
  double selected_queue_distance = std::numeric_limits<double>::infinity();
  double selected_actual_distance = std::numeric_limits<double>::infinity();
  std::size_t queue_filtered = 0U;
  std::size_t cap_filtered = 0U;
  for (std::size_t index = 0U; index < plan.compact_path_samples.size(); ++index) {
    const auto & sample = plan.compact_path_samples[index];
    const double actual_distance = (actual - sample.positions).norm();
    const double queue_distance = (queue - sample.positions).norm();
    if (sample.original_segment_index >= first_segment_limit) {
      continue;
    }
    const bool queue_ok = std::isfinite(queue_distance) &&
      queue_distance <= kMaximumAttachDistanceRad + 1.0e-9;
    const bool actual_finite = std::isfinite(actual_distance);
    const bool caps_ok = within_caps(sample.suffix_maximum_assembly_radius_m);
    queue_filtered += queue_ok ? 0U : 1U;
    if (!caps_ok) {
      ++cap_filtered;
    }
    if (!actual_finite || !queue_ok || !caps_ok) {
      continue;
    }
    const bool queue_is_better = queue_distance < selected_queue_distance - 1.0e-12;
    const bool queue_is_tied = std::abs(queue_distance - selected_queue_distance) <= 1.0e-12;
    const bool actual_is_better = actual_distance < selected_actual_distance - 1.0e-12;
    if (!result.valid || queue_is_better || (queue_is_tied && actual_is_better)) {
      result.valid = true;
      result.sample_index = index;
      result.segment_index = sample.original_segment_index;
      result.segment_fraction = sample.fraction;
      result.position = sample.positions;
      result.maximum_radius_m = sample.suffix_maximum_assembly_radius_m;
      result.maximum_roll_error_rad = sample.suffix_maximum_roll_error_rad;
      result.maximum_abs_pitch_rad = sample.suffix_maximum_abs_screen_pitch_rad;
      selected_queue_distance = queue_distance;
      selected_actual_distance = actual_distance;
    }
  }
  if (!result.valid) {
    result.failure_reason = "no_sample:queue_filtered=" +
      std::to_string(queue_filtered) + ",cap_filtered=" + std::to_string(cap_filtered);
  }
  return result;
}

std::optional<CompactPublishedPrefix> findCompactPublishedPrefix(
  const JointMotionState & expected_start, const double now_sec,
  const PublishedTrajectoryRecord * latest,
  const PublishedTrajectoryRecord * active,
  const double period_sec, const JointMotionLimits & limits,
  std::string * failure_reason)
{
  const auto fail = [failure_reason](const std::string & reason) {
      if (failure_reason) {
        *failure_reason = reason;
      }
      return std::optional<CompactPublishedPrefix>{};
    };
  if (failure_reason) {
    failure_reason->clear();
  }
  const Eigen::Index count = expected_start.position.size();
  if (count <= 0 || expected_start.velocity.size() != count ||
    expected_start.acceleration.size() != count || !expected_start.position.allFinite() ||
    !expected_start.velocity.allFinite() || !expected_start.acceleration.allFinite() ||
    !std::isfinite(now_sec) || !std::isfinite(period_sec) || period_sec <= 0.0)
  {
    return fail("current queue state or controller period is invalid");
  }
  if (expected_start.velocity.isZero(kFiniteEpsilon) &&
    expected_start.acceleration.isZero(kFiniteEpsilon))
  {
    CompactPublishedPrefix result;
    result.samples.push_back(expected_start);
    return result;
  }
  if (limits.lower_position.size() != count || limits.upper_position.size() != count ||
    limits.max_velocity.size() != count || limits.max_acceleration.size() != count ||
    limits.max_jerk.size() != count || !limits.numerical_policy ||
    !limits.numerical_policy->valid())
  {
    return fail("current command limits are invalid for a moving prefix");
  }
  const PublishedTrajectoryRecord * records[2]{latest, active};
  for (std::size_t record_index = 0U; record_index < 2U; ++record_index) {
    const auto * record = records[record_index];
    if (record == nullptr || record->execution_queue.empty() ||
      (record_index == 1U && latest != nullptr && active != nullptr && latest->id == active->id))
    {
      continue;
    }
    std::vector<std::size_t> matches;
    for (std::size_t index = 0U; index < record->execution_queue.size(); ++index) {
      const auto & sample = record->execution_queue[index];
      if (sample.positions.size() != count || sample.velocities.size() != count ||
        sample.accelerations.size() != count || !sample.positions.allFinite() ||
        !sample.velocities.allFinite() || !sample.accelerations.allFinite())
      {
        matches.clear();
        break;
      }
      if ((sample.positions - expected_start.position).lpNorm<Eigen::Infinity>() <=
        kFiniteEpsilon &&
        (sample.velocities - expected_start.velocity).lpNorm<Eigen::Infinity>() <=
        kFiniteEpsilon &&
        (sample.accelerations - expected_start.acceleration).lpNorm<Eigen::Infinity>() <=
        kFiniteEpsilon)
      {
        matches.push_back(index);
      }
    }
    if (matches.empty()) {
      continue;
    }
    if (matches.size() != 1U) {
      return fail("moving queue state matches multiple samples in a published trajectory");
    }
    const std::size_t start_index = matches.front();
    const auto & start_sample = record->execution_queue[start_index];
    if (!std::isfinite(start_sample.time_stamp.seconds()) ||
      start_sample.time_stamp.seconds() + 1.0e-9 < now_sec)
    {
      continue;
    }
    const auto stationary = std::find_if(
      record->execution_queue.begin() + static_cast<std::ptrdiff_t>(start_index),
      record->execution_queue.end(), [&record](const auto & sample) {
        return sample.time_stamp == record->stationary_time;
      });
    if (stationary == record->execution_queue.end()) {
      return fail("published moving queue suffix has no stationary endpoint");
    }
    const std::size_t endpoint_index = static_cast<std::size_t>(
      std::distance(record->execution_queue.begin(), stationary));
    CompactPublishedPrefix result;
    result.samples.reserve(endpoint_index - start_index + 1U);
    for (std::size_t index = start_index; index <= endpoint_index; ++index) {
      const auto & sample = record->execution_queue[index];
      if (index > start_index) {
        const double interval = (sample.time_stamp -
          record->execution_queue[index - 1U].time_stamp).seconds();
        if (!std::isfinite(interval) || std::abs(interval - period_sec) > 1.0e-9) {
          return fail("published moving queue suffix is not on the controller time grid");
        }
      }
      if ((sample.positions.array() < limits.lower_position.array()).any() ||
        (sample.positions.array() > limits.upper_position.array()).any() ||
        (sample.velocities.cwiseAbs().array() > limits.max_velocity.array() +
        limits.numerical_policy->velocity_error.value).any() ||
        (sample.accelerations.cwiseAbs().array() > limits.max_acceleration.array() +
        limits.numerical_policy->acceleration_error.value).any())
      {
        return fail("published moving queue suffix exceeds its captured command limits");
      }
      if (index > start_index) {
        const auto & previous = record->execution_queue[index - 1U];
        const Eigen::VectorXd jerk = (sample.accelerations - previous.accelerations) /
          period_sec;
        if (!jerk.allFinite() || (jerk.cwiseAbs().array() > limits.max_jerk.array() +
          limits.numerical_policy->jerk_error.value).any())
        {
          return fail("published moving queue suffix exceeds its captured jerk limits");
        }
      }
      result.samples.push_back(JointMotionState{
          sample.positions, sample.velocities, sample.accelerations});
    }
    const auto & endpoint = result.samples.back();
    if (!endpoint.velocity.isZero(1.0e-12) || !endpoint.acceleration.isZero(1.0e-12)) {
      return fail("published moving queue suffix does not end at rest");
    }
    return result;
  }
  return fail("moving queue state has no exact future match in latest or active publication");
}

}  // namespace face_tracking_arm::control::detail
