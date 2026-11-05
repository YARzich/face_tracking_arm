// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_PROFILE_WORKER_HPP_
#define FACE_TRACKING_ARM__COMPACT_PROFILE_WORKER_HPP_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "face_tracking_arm/compact_dynamics.hpp"

namespace face_tracking_arm::control
{

enum class CompactProfileGeometryStatus
{
  kNotChecked,
  kPass,
  kInvalidInput,
  kUnsafe,
  kStaleScene,
  kValidatorFailure
};

/// Evidence that the final retimed profile was checked against one immutable
/// geometry/collision snapshot. A certificate is valid only for its exact
/// scene revision and profile sample/interval counts.
struct CompactProfileGeometryCertificate
{
  CompactProfileGeometryStatus status{CompactProfileGeometryStatus::kNotChecked};
  std::uint64_t scene_revision{0U};
  std::size_t checked_samples{0U};
  std::size_t checked_intervals{0U};
  std::string failure_reason;

  [[nodiscard]] bool passedFor(
    std::uint64_t expected_scene_revision,
    std::size_t expected_samples,
    std::size_t expected_intervals) const noexcept
  {
    return status == CompactProfileGeometryStatus::kPass &&
           scene_revision == expected_scene_revision &&
           checked_samples == expected_samples &&
           checked_intervals == expected_intervals;
  }
};

/// The callback runs on the worker thread and must own only immutable scene,
/// model, and geometry settings; it must not capture MotionReference, a live
/// PlanningSceneMonitor, or other mutable controller state. A pass certificate
/// covers every profile sample and each adjacent interval including the copied
/// prefix-to-suffix join.
using CompactProfileFinalGeometryValidator = std::function<
  CompactProfileGeometryCertificate(const CompactJointProfile &)>;

struct CompactProfileWorkRequest
{
  std::uint64_t generation{0U};
  std::uint64_t scene_revision{0U};
  /// Owned source geometry for the worker-side profile build. If empty, the
  /// prebuilt `profile` below is used by legacy Retimer tests/callers.
  std::vector<Eigen::VectorXd> source_route;
  std::vector<double> segment_speed_scales;
  JointMotionState start_state;
  /// Exact immutable controller queue prefix from current sample through its
  /// stationary endpoint. The builder starts at `published_prefix.back()`.
  std::vector<JointMotionState> published_prefix;
  JointMotionLimits prefix_limits;
  CompactProfileFinalGeometryValidator final_geometry_validator;
  CompactProfileFinalGeometryValidator return_final_geometry_validator;
  CompactJointProfile profile;
  std::optional<CompactJointProfile> return_profile;
  /// False only when a previously certified return remains stored by the caller.
  bool return_profile_required{true};
  JointMotionLimits limits;
  double control_period_sec{0.0};
  double requested_speed_scale{1.0};
  double return_speed_scale{1.0};
};

struct CompactProfileWorkResult
{
  std::uint64_t request_id{0U};
  std::uint64_t generation{0U};
  std::uint64_t scene_revision{0U};
  double build_elapsed_ms{0.0};
  double geometry_elapsed_ms{0.0};
  double total_elapsed_ms{0.0};
  CompactDynamicsResult profile;
  CompactProfileGeometryCertificate geometry_certificate;
  std::optional<CompactDynamicsResult> return_profile;
  std::optional<CompactProfileGeometryCertificate> return_geometry_certificate;
};

/// One latest-only worker for compact-profile building, effort checks,
/// retiming, and final geometry/collision certification.
class CompactProfileWorker final
{
public:
  using Retimer = std::function<CompactDynamicsResult(
        const CompactJointProfile &, const JointMotionLimits &, double, double)>;
  using ProfileBuilderForTest = std::function<std::optional<CompactJointProfile>(
        const std::vector<Eigen::VectorXd> &, const JointMotionState &,
        const JointMotionLimits &, double, double, const std::vector<double> &,
        std::string *)>;

  explicit CompactProfileWorker(std::shared_ptr<CompactDynamicsModel> model);
  /// Test seam for deterministic ordering, cancellation, and failure checks.
  explicit CompactProfileWorker(Retimer retimer);
  /// Narrow exception-path seam; production always uses buildCompactJointProfile.
  CompactProfileWorker(Retimer retimer, ProfileBuilderForTest builder_for_test);
  ~CompactProfileWorker();

  CompactProfileWorker(const CompactProfileWorker &) = delete;
  CompactProfileWorker & operator=(const CompactProfileWorker &) = delete;

  /// Replaces any pending request and invalidates the result of active work.
  [[nodiscard]] std::uint64_t submit(CompactProfileWorkRequest request);
  void cancel();
  [[nodiscard]] std::optional<CompactProfileWorkResult> takeResult();

private:
  struct WorkItem
  {
    std::uint64_t request_id{0U};
    CompactProfileWorkRequest request;
  };

  void run();

  Retimer retimer_;
  ProfileBuilderForTest builder_for_test_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<WorkItem> pending_;
  std::optional<CompactProfileWorkResult> result_;
  std::uint64_t next_request_id_{0U};
  std::uint64_t latest_request_id_{0U};
  bool stopping_{false};
  std::thread worker_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_PROFILE_WORKER_HPP_
