// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_RECORDER_HPP_
#define FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_RECORDER_HPP_

#include <filesystem>
#include <string>

#include "face_tracking_arm/qp_failure_snapshot_mailbox.hpp"

namespace face_tracking_arm::control
{

struct QpFailureSnapshotEventWriteResult
{
  bool snapshots_ok{true};
  bool index_ok{false};
  std::string primary_path;
  std::string secondary_path;
};

/// Persist one bounded failure event off the real-time path: one snapshot per
/// stage file and one index row that retains its control/hardware provenance.
[[nodiscard]] QpFailureSnapshotEventWriteResult writeQpFailureSnapshotEvent(
  const std::filesystem::path & directory,
  const QpFailureSnapshotEvent & event);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_RECORDER_HPP_
