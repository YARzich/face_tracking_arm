// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/qp_failure_snapshot_recorder.hpp"

#include <fstream>

#include "face_tracking_arm/qp_failure_snapshot_io.hpp"

namespace face_tracking_arm::control
{

QpFailureSnapshotEventWriteResult writeQpFailureSnapshotEvent(
  const std::filesystem::path & directory,
  const QpFailureSnapshotEvent & event)
{
  QpFailureSnapshotEventWriteResult result;
  const auto write_stage = [&directory, &event](
    const std::optional<QpFailureSnapshot> & snapshot,
    const char * stage,
    std::string & path) {
      if (!snapshot.has_value()) {
        return true;
      }
      path = (directory /
        ("event_" + std::to_string(event.event_id) + "_" + stage + ".qpsnap")).string();
      std::ofstream output(path, std::ios::out | std::ios::trunc);
      return output.good() && writeQpFailureSnapshot(output, *snapshot) && output.good();
    };
  result.snapshots_ok = write_stage(event.primary, "primary", result.primary_path) &&
    write_stage(event.secondary, "secondary", result.secondary_path);

  std::ofstream index(directory / "events.tsv", std::ios::out | std::ios::app);
  if (!index.good()) {
    return result;
  }
  index << event.event_id << '\t' << event.ros_stamp_ns << '\t'
        << event.steady_stamp_ns << '\t' << event.hardware_epoch << '\t'
        << event.hardware_read_sequence << '\t' << event.active_publication_id << '\t'
        << event.latest_publication_id << '\t'
        << static_cast<int>(event.primary_disposition) << '\t'
        << static_cast<int>(event.secondary_disposition) << '\t'
        << (result.snapshots_ok ? 1 : 0) << '\t' << result.primary_path << '\t'
        << result.secondary_path << '\n';
  result.index_ok = index.good();
  return result;
}

}  // namespace face_tracking_arm::control
