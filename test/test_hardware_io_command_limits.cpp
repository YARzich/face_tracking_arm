// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "hardware_io_command_limits.hpp"

int main()
{
  std::array<double, 6> low{{-1, -1, -1, -1, -1, -1}};
  std::array<double, 6> high{{1, 1, 1, 1, 1, 1}};
  std::array<bool, 6> valid{{true, true, true, true, true, true}};
  std::array<double, 6> requested{{0, 0, 0, 0, 0, 0}};
  std::array<float, 6> wire{{0, 0, 0, 0, 0, 0}};
  const auto check = [&](bool expected) {
      const auto result = face_tracking_arm::hardware::validate_position_command(
            requested.data(), wire.data(), low.data(), high.data(), valid.data(), 6);
      return result.valid == expected;
    };
  const auto finite_check = [&](bool expected) {
      const auto result = face_tracking_arm::hardware::validate_finite_position_command(
            requested.data(), wire.data(), 6);
      return result.valid == expected;
    };
  if (!check(true)) {return 1;}
  requested[0] = low[0];
  wire[0] = static_cast<float>(requested[0]);
  if (!check(true)) {return 2;}
  requested[0] = high[0];
  wire[0] = static_cast<float>(requested[0]);
  if (!check(true)) {return 3;}
  requested[0] = std::nextafter(
        high[0], std::numeric_limits<double>::infinity());
  wire[0] = static_cast<float>(requested[0]);
  if (!check(false)) {return 4;}
  if (!finite_check(true)) {return 11;}
  if (!face_tracking_arm::hardware::position_command_allowed(false, true, true, false)) {
    return 12;
  }
  if (face_tracking_arm::hardware::position_command_allowed(true, true, true, false)) {
    return 13;
  }
  requested.fill(0); wire.fill(0);
  requested[3] = std::numeric_limits<double>::quiet_NaN();
  if (!check(false)) {return 5;}
  if (!finite_check(false)) {return 14;}
  if (face_tracking_arm::hardware::position_command_allowed(false, false, false, false)) {
    return 15;
  }
  const auto invalid_position = face_tracking_arm::hardware::validate_position_command(
        requested.data(), wire.data(), low.data(), high.data(), valid.data(), 6);
  if (invalid_position.joint_index != 3) {return 10;}
  requested[3] = std::numeric_limits<double>::infinity();
  if (!check(false)) {return 6;}
  requested.fill(0); wire.fill(0);
  wire[2] = std::numeric_limits<float>::infinity();
  if (!check(false)) {return 7;}
  if (!finite_check(false)) {return 16;}
  if (face_tracking_arm::hardware::position_command_allowed(false, false, false, false)) {
    return 17;
  }
  wire.fill(0); valid[3] = false;
  if (!check(false)) {return 8;}
  const auto finite_candidate =
    face_tracking_arm::hardware::validate_finite_position_command(
            requested.data(), wire.data(), 6);
  if (!finite_candidate.valid) {return 18;}
  if (!face_tracking_arm::hardware::position_command_allowed(
          false, false, finite_candidate.valid, false))
  {
    return 19;
  }
  if (face_tracking_arm::hardware::position_command_allowed(
          true, false, finite_candidate.valid, false))
  {
    return 20;
  }
  if (face_tracking_arm::hardware::validate_position_command(
            requested.data(), wire.data(), low.data(), high.data(), valid.data(), 5).valid)
  {
    return 9;
  }
  return EXIT_SUCCESS;
}
