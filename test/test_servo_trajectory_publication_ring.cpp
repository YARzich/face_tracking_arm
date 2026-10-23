// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <string>
#include <vector>

#include "face_tracking_arm/servo_trajectory_publication_ring.hpp"

struct FakeDuration
{
  std::int32_t sec{0};
  std::uint32_t nanosec{0U};
};

struct FakePoint
{
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> accelerations;
  std::vector<double> effort;
  FakeDuration time_from_start;
};

struct FakeStamp
{
  std::int32_t sec{0};
  std::uint32_t nanosec{0U};
};

struct FakeHeader
{
  FakeStamp stamp;
  std::string frame_id;
};

struct FakeTrajectory
{
  FakeHeader header;
  std::vector<std::string> joint_names;
  std::vector<FakePoint> points;
};

struct FakePublication
{
  FakeHeader header;
  std::uint64_t publication_id{1U};
  std::uint8_t publication_kind{1U};
  FakeStamp stationary_time;
  std::uint32_t tail_start_index{0U};
  std::uint64_t hardware_epoch{3U};
  std::uint64_t hardware_read_sequence{5U};
  std::uint64_t control_tick_sequence{7U};
  FakeTrajectory trajectory;
};

FakePublication make_publication(std::size_t point_count)
{
  FakePublication publication;
  publication.header.stamp = {10, 20U};
  publication.header.frame_id = "event_frame";
  publication.stationary_time = {11, 21U};
  publication.trajectory.header = publication.header;
  publication.trajectory.header.stamp = {12, 22U};
  publication.trajectory.header.frame_id = "trajectory_frame";
  publication.trajectory.joint_names = {"j1", "j2", "j3", "j4", "j5", "j6"};
  for (std::size_t index = 0; index < point_count; ++index) {
    FakePoint point;
    point.positions.assign(6U, static_cast<double>(index));
    point.time_from_start = {static_cast<std::int32_t>(index), 100U};
    publication.trajectory.points.push_back(point);
  }
  return publication;
}

int main()
{
  using face_tracking_arm::diagnostics::ServoTrajectoryPublicationRecord;
  using face_tracking_arm::diagnostics::copy_servo_trajectory_publication;
  using Ring = face_tracking_arm::diagnostics::ServoTrajectoryPublicationRing<2U>;

  auto publication = make_publication(
    face_tracking_arm::diagnostics::kTrajectoryPointCapacity);
  ServoTrajectoryPublicationRecord record;
  if (!copy_servo_trajectory_publication(publication, record)) {return 1;}
  if (record.publication_id != 1U || record.publication_kind != 1U ||
    record.point_count != 2048U ||
    record.points[2047].positions[4] != 2047.0 ||
    record.points[2047].time_sec != 2047 ||
    record.trajectory_stamp_sec != 12 || record.stamp_sec != 10 ||
    std::string(record.event_frame_id.data(), record.event_frame_id_size) != "event_frame" ||
    std::string(record.trajectory_frame_id.data(), record.trajectory_frame_id_size) !=
    "trajectory_frame")
  {
    return 2;
  }
  if (record.tail_start_index != 0U || record.hardware_epoch != 3U ||
    record.hardware_read_sequence != 5U || record.control_tick_sequence != 7U)
  {
    return 3;
  }

  Ring ring;
  if (!ring.tryEncodeAndPush(publication) || !ring.tryEncodeAndPush(publication)) {
    return 4;
  }
  if (ring.tryPush(record) || ring.droppedCount() != 1U) {return 5;}
  ServoTrajectoryPublicationRecord popped;
  if (!ring.tryPop(popped) || popped.point_count != record.point_count ||
    popped.points[2047].positions[4] != 2047.0)
  {
    return 6;
  }

  auto oversized = make_publication(2049U);
  if (copy_servo_trajectory_publication(oversized, record)) {return 7;}
  auto wrong_dof = make_publication(1U);
  wrong_dof.trajectory.joint_names.pop_back();
  if (copy_servo_trajectory_publication(wrong_dof, record)) {return 8;}
  auto wrong_point = make_publication(1U);
  wrong_point.trajectory.points[0].velocities.assign(6U, 1.0);
  if (copy_servo_trajectory_publication(wrong_point, record)) {return 9;}
  auto reset = make_publication(1U);
  reset.trajectory.points[0].velocities.assign(6U, 0.0);
  reset.trajectory.points[0].accelerations.assign(6U, 0.0);
  if (!copy_servo_trajectory_publication(reset, record) ||
    !record.points[0].has_velocities || !record.points[0].has_accelerations)
  {
    return 10;
  }
  auto long_frame = make_publication(1U);
  long_frame.trajectory.header.frame_id.assign(
    face_tracking_arm::diagnostics::kTrajectoryFrameCapacity, 'x');
  if (copy_servo_trajectory_publication(long_frame, record)) {return 11;}
  return EXIT_SUCCESS;
}
