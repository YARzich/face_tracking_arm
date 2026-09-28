# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Check the actual driver activation policy using a fake SDK, without ROS or hardware."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_startup_checks_every_transition_and_never_resumes_an_observed_stop(tmp_path):
    source = tmp_path / 'activation.cpp'
    source.write_text(r"""
#include <array>
#include <cassert>
#include <string>
#include "hardware_activation_guard.hpp"

struct Arm
{
  std::array<int, 4> states{2, 2, 5, 2};
  std::array<int, 4> errors{};
  std::array<int, 4> warnings{};
  int sample = -1;
  int failed_state_read = -1;
  int failed_error_read = -1;
  char failed_command = '\0';
  std::string commands;

  int get_state(int *state)
  {
    ++sample;
    assert(sample < 4);
    *state = states[sample];
    return sample == failed_state_read ? -1 : 0;
  }

  int get_err_warn_code(int *codes)
  {
    codes[0] = errors[sample];
    codes[1] = warnings[sample];
    return sample == failed_error_read ? -1 : 0;
  }

  int motion_enable(bool enable)
  {
    assert(enable);
    commands += 'E';
    return failed_command == 'E' ? -1 : 0;
  }

  int set_mode(int mode)
  {
    assert(mode == 1 || mode == 4);
    commands += 'M';
    return failed_command == 'M' ? -1 : 0;
  }

  int set_state(int state)
  {
    assert(state == 0);
    commands += 'S';
    return failed_command == 'S' ? -1 : 0;
  }
};

int main()
{
  using face_tracking_arm::hardware::activateReadyArm;
  for (int mode : {1, 4}) {
    Arm ready;
    ready.warnings.fill(11);
    assert(activateReadyArm(ready, mode));
    assert(ready.commands == "EMS");
    assert(ready.warnings[0] == 11);
  }
  for (int state : {1, 3, 4, 5, 6, 99}) {
    Arm stopped;
    stopped.states[0] = state;
    assert(!activateReadyArm(stopped, 1));
    assert(stopped.commands.empty());
  }
  for (int stage : {1, 2}) {
    for (int state : {1, 3, 4, 6, 99}) {
      Arm stopped;
      stopped.states[stage] = state;
      assert(!activateReadyArm(stopped, 1));
      assert(stopped.commands.find('S') == std::string::npos);
    }
  }
  Arm externally_changed;
  externally_changed.states[1] = 5;
  assert(!activateReadyArm(externally_changed, 1));
  assert(externally_changed.commands == "E");
  for (int stage = 0; stage < 4; ++stage) {
    Arm fault;
    fault.errors[stage] = 23;
    assert(!activateReadyArm(fault, 1));
    if (stage < 3) assert(fault.commands.find('S') == std::string::npos);
    Arm stale;
    stale.failed_state_read = stage;
    assert(!activateReadyArm(stale, 1));
    Arm missing_error;
    missing_error.failed_error_read = stage;
    assert(!activateReadyArm(missing_error, 1));
  }
  for (char command : {'E', 'M', 'S'}) {
    Arm failing;
    failing.failed_command = command;
    assert(!activateReadyArm(failing, 1));
    assert(failing.commands.back() == command);
  }
}
""")
    executable = tmp_path / 'activation'
    subprocess.run(['c++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'docker/patches'), str(source), '-o', str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_pinned_patch_removes_explicit_fault_clears_and_requests_stop_on_shutdown():
    patch = (ROOT / 'docker/patches/xarm-readiness.patch').read_text()
    removed = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('-'))
    added = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+'))
    assert removed.count('->clean_error();') == 3
    assert removed.count('->clean_warn();') == 1
    assert '->clean_error(' not in added and '->clean_warn(' not in added
    assert added.count('->set_state(XARM_STATE::STOP);') == 2
    assert 'activateReadyArm(' in added
