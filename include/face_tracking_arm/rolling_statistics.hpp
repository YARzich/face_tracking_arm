// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__ROLLING_STATISTICS_HPP_
#define FACE_TRACKING_ARM__ROLLING_STATISTICS_HPP_

#include <cstddef>
#include <optional>
#include <vector>

namespace face_tracking_arm::telemetry
{

/// Fixed-capacity rolling statistics for a high-frequency, thread-confined loop.
///
/// Storage is allocated in the constructor. add(), clear(), count(), mean(), and
/// maximum() never allocate. percentile() copies the active window and may allocate;
/// it is intended for low-frequency telemetry publication, not the control loop.
class RollingStatistics final
{
public:
  explicit RollingStatistics(std::size_t capacity);

  /// Add one finite sample, overwriting the oldest sample when the window is full.
  /// Returns false without changing the window when sample is NaN or infinite.
  [[nodiscard]] bool add(double sample) noexcept;

  void clear() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t count() const noexcept;
  [[nodiscard]] bool full() const noexcept;
  [[nodiscard]] std::optional<double> mean() const noexcept;
  [[nodiscard]] std::optional<double> maximum() const noexcept;

  /// Nearest-rank percentile for probability in [0, 1].
  /// Returns nullopt for an empty window or invalid probability.
  [[nodiscard]] std::optional<double> percentile(double probability) const;
  [[nodiscard]] std::optional<double> p99() const;

private:
  void recompute_maximum() noexcept;

  std::vector<double> samples_;
  std::size_t next_index_{0};
  std::size_t count_{0};
  double sum_{0.0};
  double maximum_{0.0};
};

}  // namespace face_tracking_arm::telemetry

#endif  // FACE_TRACKING_ARM__ROLLING_STATISTICS_HPP_
