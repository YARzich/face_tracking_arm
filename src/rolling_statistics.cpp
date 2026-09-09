// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/rolling_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace face_tracking_arm::telemetry
{

RollingStatistics::RollingStatistics(const std::size_t capacity)
: samples_(capacity, 0.0)
{
  if (capacity == 0) {
    throw std::invalid_argument("rolling statistics capacity must be positive");
  }
}

bool RollingStatistics::add(const double sample) noexcept
{
  if (!std::isfinite(sample)) {
    return false;
  }

  if (count_ < samples_.size()) {
    samples_[next_index_] = sample;
    sum_ += sample;
    if (count_ == 0 || sample > maximum_) {
      maximum_ = sample;
    }
    ++count_;
  } else {
    const double replaced_sample = samples_[next_index_];
    samples_[next_index_] = sample;
    sum_ += sample - replaced_sample;
    if (sample >= maximum_) {
      maximum_ = sample;
    } else if (replaced_sample == maximum_) {
      recompute_maximum();
    }
  }

  ++next_index_;
  if (next_index_ == samples_.size()) {
    next_index_ = 0;
  }
  return true;
}

void RollingStatistics::clear() noexcept
{
  next_index_ = 0;
  count_ = 0;
  sum_ = 0.0;
  maximum_ = 0.0;
}

std::size_t RollingStatistics::capacity() const noexcept
{
  return samples_.size();
}

std::size_t RollingStatistics::count() const noexcept
{
  return count_;
}

bool RollingStatistics::full() const noexcept
{
  return count_ == samples_.size();
}

std::optional<double> RollingStatistics::mean() const noexcept
{
  if (count_ == 0) {
    return std::nullopt;
  }
  return sum_ / static_cast<double>(count_);
}

std::optional<double> RollingStatistics::maximum() const noexcept
{
  if (count_ == 0) {
    return std::nullopt;
  }
  return maximum_;
}

std::optional<double> RollingStatistics::percentile(const double probability) const
{
  if (count_ == 0 || !std::isfinite(probability) ||
    probability < 0.0 || probability > 1.0)
  {
    return std::nullopt;
  }

  std::vector<double> ordered_samples;
  ordered_samples.reserve(count_);
  ordered_samples.insert(
    ordered_samples.end(), samples_.begin(), samples_.begin() + count_);

  const double nearest_rank = std::ceil(probability * static_cast<double>(count_));
  const std::size_t index = nearest_rank <= 1.0 ? 0 :
    static_cast<std::size_t>(nearest_rank) - 1;
  auto selected = ordered_samples.begin() + static_cast<std::ptrdiff_t>(index);
  std::nth_element(ordered_samples.begin(), selected, ordered_samples.end());
  return *selected;
}

std::optional<double> RollingStatistics::p99() const
{
  return percentile(0.99);
}

void RollingStatistics::recompute_maximum() noexcept
{
  maximum_ = samples_[0];
  for (std::size_t index = 1; index < count_; ++index) {
    maximum_ = std::max(maximum_, samples_[index]);
  }
}

}  // namespace face_tracking_arm::telemetry
