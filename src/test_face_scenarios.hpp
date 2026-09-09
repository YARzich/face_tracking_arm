// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef TEST_FACE_SCENARIOS_HPP_
#define TEST_FACE_SCENARIOS_HPP_

#include <optional>
#include <string_view>

namespace face_tracking_arm::test_targets
{

enum class Scenario {kDisabled, kStationary, kCircle, kPeople, kWalkAround};

struct Point
{
  double x;
  double y;
  double z;
};

struct Sample
{
  std::optional<Point> target;
  std::string_view phase;
};

// Coordinates are in world, with z measured from the floor, not the tabletop.
// People: 24 s front sequence + eight sides for 6 s each + 48 s walk + 12 s loss/return.
inline constexpr double kPeopleCycleDurationSec = 132.0;
inline constexpr double kWalkDurationSec = 48.0;

Scenario parse_scenario(std::string_view name);
Sample sample(Scenario scenario, double elapsed_sec);

}  // namespace face_tracking_arm::test_targets

#endif  // TEST_FACE_SCENARIOS_HPP_
