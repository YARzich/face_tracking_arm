// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef DOCKER__PATCHES__HARDWARE_ACTIVATION_GUARD_HPP_
#define DOCKER__PATCHES__HARDWARE_ACTIVATION_GUARD_HPP_

namespace face_tracking_arm
{
namespace hardware
{

// Startup is explicit: only a ready arm may be enabled. State 5 is accepted
// solely after our own mode change. Paused, stopped, moving and unknown states
// require the operator. Getter failures never count as readiness.
template<typename Arm>
bool readyForActivation(Arm & arm, bool after_mode_change = false)
{
  int state = -1;
  int codes[2] = {-1, -1};
  return arm.get_state(&state) == 0 && arm.get_err_warn_code(codes) == 0 &&
         codes[0] == 0 && (state == 0 || state == 2 || (after_mode_change && state == 5));
}

template<typename Arm>
bool activateReadyArm(Arm & arm, int mode)
{
  if (!readyForActivation(arm) || arm.motion_enable(true) != 0 ||
    !readyForActivation(arm) || arm.set_mode(mode) != 0 ||
    !readyForActivation(arm, true))
  {
    return false;
  }
  // UFACTORY requires START after a mode change. This belongs only to explicit
  // activation, never runtime recovery. The firmware exposes no atomic
  // check-and-start operation: these checks are not a replacement for its E-stop.
  return arm.set_state(0) == 0 && readyForActivation(arm);
}

}  // namespace hardware
}  // namespace face_tracking_arm

#endif  // DOCKER__PATCHES__HARDWARE_ACTIVATION_GUARD_HPP_
