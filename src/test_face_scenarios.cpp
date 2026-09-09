// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "test_face_scenarios.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace face_tracking_arm::test_targets
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kFaceBelowTopM = 0.12;
constexpr double kDefaultPersonHeightM = 1.80;
constexpr double kAroundRadiusM = 0.85;
constexpr double kFrontSequenceEndSec = 24.0;
constexpr double kSideDurationSec = 6.0;
constexpr double kWalkStartSec = 72.0;
constexpr double kWalkEndSec = kWalkStartSec + kWalkDurationSec;

struct SidePerson
{
  double height_m;
  std::string_view description;
};

// Counterclockwise in world XY, starting at +X. The table radius is 0.50 m.
constexpr std::array<SidePerson, 8> kSidePeople{{
  {1.60, "People: front (+X), height 1.60 m"},
  {1.75, "People: front-left (+X,+Y), height 1.75 m"},
  {1.90, "People: left (+Y), height 1.90 m"},
  {1.65, "People: rear-left (-X,+Y), height 1.65 m"},
  {1.80, "People: rear (-X), height 1.80 m"},
  {2.00, "People: rear-right (-X,-Y), height 2.00 m"},
  {1.70, "People: right (-Y), height 1.70 m"},
  {1.85, "People: front-right (+X,-Y), height 1.85 m"},
}};

double smooth_step(double value)
{
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

Point stationary_target()
{
  return {0.70, 0.0, kDefaultPersonHeightM - kFaceBelowTopM};
}

Point circle_target(double elapsed)
{
  const double phase = 2.0 * kPi * elapsed / 8.0;
  Point point = stationary_target();
  point.x += 0.03 * std::cos(phase);
  point.y = 0.08 * std::sin(phase);
  point.z += 0.03 * std::sin(0.5 * phase);
  return point;
}

Point person_a_target(double local_time)
{
  const double approach = smooth_step(local_time / 1.6);
  Point point;
  point.x = 0.76 - 0.27 * approach + 0.012 * std::sin(2.0 * kPi * local_time / 3.2);
  point.y = 0.41 - 0.10 * approach + 0.018 * std::sin(2.0 * kPi * local_time / 2.7);
  point.z = 1.60 - kFaceBelowTopM + 0.012 * std::sin(2.0 * kPi * local_time / 2.1);
  return point;
}

Point person_b_target(double local_time)
{
  const double approach = smooth_step(local_time / 1.5);
  Point point;
  point.x = 0.78 - 0.20 * approach + 0.010 * std::sin(2.0 * kPi * local_time / 3.0);
  point.y = -0.43 + 0.016 * std::sin(2.0 * kPi * local_time / 2.5);
  point.z = 2.00 - kFaceBelowTopM + 0.010 * std::sin(2.0 * kPi * local_time / 2.0);
  return point;
}

Point person_c_target(double local_time)
{
  const double approach = smooth_step(local_time / 1.4);
  const double fast_head_shift = smooth_step((local_time - 3.0) / 0.4);
  Point point;
  point.x = 0.73 - 0.16 * approach + 0.014 * std::sin(2.0 * kPi * local_time / 2.9);
  point.y = 0.31 + 0.018 * std::sin(2.0 * kPi * local_time / 2.4) - 0.11 * fast_head_shift;
  point.z = 1.80 - kFaceBelowTopM +
    0.012 * std::sin(2.0 * kPi * local_time / 2.2) + 0.025 * fast_head_shift;
  return point;
}

Point walk_around_target(double elapsed)
{
  // A periodic path also has continuous velocity across the loop boundary.
  const double phase = 2.0 * kPi * std::fmod(elapsed, kWalkDurationSec) / kWalkDurationSec;
  return {
    kAroundRadiusM * std::cos(phase),
    kAroundRadiusM * std::sin(phase),
    kDefaultPersonHeightM - kFaceBelowTopM + 0.01 * std::sin(24.0 * phase),
  };
}

Sample people_sample(double elapsed)
{
  const double t = std::fmod(elapsed, kPeopleCycleDurationSec);
  if (t < 2.4) {
    return {std::nullopt, "People: initial REST; no face target"};
  }
  if (t < 7.4) {
    return {person_a_target(t - 2.4), "People: person A, height 1.60 m, from the left"};
  }
  if (t < 10.0) {
    return {std::nullopt, "People: target lost; observing FACE -> HOLD -> REST"};
  }
  if (t < 13.8) {
    return {person_b_target(t - 10.0), "People: person B, height 2.00 m, from the right"};
  }
  if (t < 20.2) {
    return {person_c_target(t - 13.8), "People: direct switch to person C, height 1.80 m"};
  }
  if (t < kFrontSequenceEndSec) {
    return {std::nullopt, "People: front sequence target loss; observing HOLD -> REST"};
  }
  if (t < kWalkStartSec) {
    const auto index = static_cast<std::size_t>(
      (t - kFrontSequenceEndSec) / kSideDurationSec);
    const double angle = 2.0 * kPi * static_cast<double>(index) / kSidePeople.size();
    const auto & person = kSidePeople[index];
    return {
      Point{
        kAroundRadiusM * std::cos(angle), kAroundRadiusM * std::sin(angle),
        person.height_m - kFaceBelowTopM},
      person.description,
    };
  }
  if (t < kWalkEndSec) {
    return {walk_around_target(t - kWalkStartSec), "People: full walk around, height 1.80 m"};
  }
  return {std::nullopt, "People: final target loss; observing HOLD -> REST"};
}
}  // namespace

Scenario parse_scenario(std::string_view name)
{
  if (name == "disabled") {
    return Scenario::kDisabled;
  }
  if (name == "stationary") {
    return Scenario::kStationary;
  }
  if (name == "circle") {
    return Scenario::kCircle;
  }
  if (name == "people") {
    return Scenario::kPeople;
  }
  if (name == "walk_around") {
    return Scenario::kWalkAround;
  }
  throw std::invalid_argument(
      "scenario must be disabled, stationary, circle, people, or walk_around");
}

Sample sample(Scenario scenario, double elapsed_sec)
{
  if (!std::isfinite(elapsed_sec) || elapsed_sec < 0.0) {
    throw std::invalid_argument("scenario elapsed time must be finite and nonnegative");
  }
  switch (scenario) {
    case Scenario::kDisabled:
      return {std::nullopt, "Disabled"};
    case Scenario::kStationary:
      return {stationary_target(), "Stationary person, height 1.80 m"};
    case Scenario::kCircle:
      return {circle_target(elapsed_sec), "Small head motion, height 1.80 m"};
    case Scenario::kPeople:
      return people_sample(elapsed_sec);
    case Scenario::kWalkAround:
      return {walk_around_target(elapsed_sec), "Continuous walk around, height 1.80 m"};
  }
  throw std::invalid_argument("unknown scenario");
}

}  // namespace face_tracking_arm::test_targets
