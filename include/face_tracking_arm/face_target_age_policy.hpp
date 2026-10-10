// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__FACE_TARGET_AGE_POLICY_HPP_
#define FACE_TRACKING_ARM__FACE_TARGET_AGE_POLICY_HPP_

#include <cstdint>

namespace face_tracking_arm::control
{

inline constexpr std::int64_t kFaceSourceFutureSkewNs = 20'000'000;

/// Validate measured-face age separately from the bounded forecast interval.
/// All inputs are in one ROS clock domain. A forecast expiry cannot extend the
/// horizon configured by the receiving executor.
inline bool validFaceTargetAge(
  const std::int64_t source_stamp_ns, const std::int64_t command_stamp_ns,
  const std::int64_t now_ns, const std::int64_t live_age_limit_ns,
  const bool is_prediction, const std::int64_t prediction_expiry_ns,
  const std::int64_t prediction_horizon_ns)
{
  if (source_stamp_ns <= 0 || command_stamp_ns < source_stamp_ns || now_ns <= 0 ||
    live_age_limit_ns <= 0)
  {
    return false;
  }
  if (!is_prediction) {
    return prediction_expiry_ns == 0 &&
           command_stamp_ns - source_stamp_ns <= live_age_limit_ns &&
           ((now_ns >= source_stamp_ns && now_ns - source_stamp_ns <= live_age_limit_ns) ||
           (source_stamp_ns > now_ns && source_stamp_ns - now_ns <= kFaceSourceFutureSkewNs));
  }
  return prediction_horizon_ns > 0 && prediction_expiry_ns > source_stamp_ns &&
         prediction_expiry_ns - source_stamp_ns <= prediction_horizon_ns &&
         command_stamp_ns < prediction_expiry_ns && now_ns < prediction_expiry_ns;
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__FACE_TARGET_AGE_POLICY_HPP_
