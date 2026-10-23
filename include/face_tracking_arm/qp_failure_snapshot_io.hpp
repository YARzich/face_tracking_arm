// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_IO_HPP_
#define FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_IO_HPP_

#include <iosfwd>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{

/// Writes/reads the versioned plain-text offline snapshot format. These functions
/// are intended for non-real-time recorder/replay tools, never the solver path.
[[nodiscard]] bool writeQpFailureSnapshot(
  std::ostream & output, const QpFailureSnapshot & snapshot);
[[nodiscard]] bool readQpFailureSnapshot(
  std::istream & input, QpFailureSnapshot & snapshot);

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_IO_HPP_
