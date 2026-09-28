# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""
Read-only preflight, separate from the motion driver's activation lifecycle.

The pinned SDK core is used deliberately: XArmAPI.connect() can clear warnings.
This adapter opens only a control socket and sends getter/limit-check requests.
No report thread, enable, mode/state change, warning reset or motion is requested.
"""

import argparse
import json
import math
import re
import socket

from .robot_profile import identify_robot, intersect_joint_limits, validate_joint_positions


def _result(response, operation):
    # SDK _check_code(non-move) accepts payloads with error/warning/not-ready
    # status bits. Controller faults are inspected separately, never cleared.
    if not response or response[0] not in (0, 1, 2, 9):
        code = response[0] if response else 'empty response'
        raise ValueError(f'Robot preflight {operation} failed ({code}); '
                         'inspect the controller status in UFACTORY Studio')
    return response[1:]


def _text(response, operation):
    return bytes(_result(response, operation)).split(b'\0', 1)[0].decode('ascii').strip()


def _identity(version):
    identity = re.search(r'(?:^|[^0-9])([567]),(\d+),', version)
    firmware = re.search(r'[vV]?(\d+)\.(\d+)\.(\d+)', version)
    if identity is None or firmware is None:
        raise ValueError('Controller version response does not identify the robot; '
                         'cannot safely select its profile')
    if tuple(map(int, firmware.groups())) < (1, 9, 0):
        raise ValueError('Robot preflight requires firmware 1.9.0 or newer '
                         'to read measured joint positions')
    return tuple(map(int, identity.groups()))


class _SdkReadOnlyClient:
    """Narrow adapter to the verified xArm SDK core at the pinned revision."""

    def __init__(self, ip, timeout_sec):
        try:
            from xarm.core.comm.socket_port import SocketPort
            from xarm.core.wrapper.uxbus_cmd_tcp import UxbusCmdTcp
        except ImportError as error:
            raise ValueError('xarm-python-sdk is required for the hardware preflight') from error
        # SocketPort changes the process default; retain the caller's setting.
        previous_timeout = socket.getdefaulttimeout()
        try:
            self._port = SocketPort(ip, 502, heartbeat=False, forbid_uds=True)
        finally:
            socket.setdefaulttimeout(previous_timeout)
        if not self._port.connected:
            self._port.close()
            raise ValueError('Cannot connect to robot.ip on TCP port 502')
        self._core = UxbusCmdTcp(self._port)
        self._core.set_timeout(timeout_sec)  # Local SDK request timeout, not a robot command.

    def get_version(self):
        return _text(self._core.get_version(), 'get_version')

    def get_robot_sn(self):
        return _text(self._core.get_robot_sn(), 'get_robot_sn')

    def get_joint_positions(self):
        result = _result(self._core.get_joint_states(num=1), 'get_joint_states')
        if len(result) != 7:
            raise ValueError('Incomplete measured joint-state response')
        return result[:6]

    def get_controller_status(self):
        state = _result(self._core.get_state(), 'get_state')
        errors = _result(self._core.get_err_code(), 'get_err_code')
        if len(state) != 1 or len(errors) != 2:
            raise ValueError('Incomplete controller-status response')
        return state[0], errors[0], errors[1]

    def get_reduced_states(self):
        result = _result(self._core.get_reduced_states(length=79), 'get_reduced_states')
        if len(result) != 7:
            raise ValueError('Incomplete reduced-mode response')
        return result

    def is_joint_limit(self, joints):
        result = _result(self._core.is_joint_limit(list(joints) + [0.0]), 'is_joint_limit')
        if len(result) != 1 or result[0] not in (0, 1):
            raise ValueError('Invalid joint-limit check response')
        return bool(result[0])

    def close(self):
        self._port.close()
        self._port.join(timeout=1.0)


def probe_robot(ip, expected_model='xarm6', joint_targets=None, timeout_sec=5.0,
                client_factory=None, require_ready=True):
    """
    Identify and cross-check a known profile without changing robot state.

    joint_targets maps names to six angles in radians. Effective bounds are the
    intersection with active reduced mode; its stored inactive envelope is only
    reported. Firmware queries validate sample poses, not all possible poses or
    collision-free paths. No limits are inferred or expanded from these queries.
    client_factory is injectable so the complete decision path is tested offline.
    """
    if not isinstance(ip, str) or not ip.strip():
        raise ValueError('robot.ip is required for hardware preflight')
    if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 30:
        raise ValueError('Robot preflight timeout must be in (0, 30] seconds')
    factory = client_factory or _SdkReadOnlyClient
    client = factory(ip, timeout_sec)
    try:
        version = client.get_version()
        axis, device_type = _identity(version)
        serial = client.get_robot_sn()
        profile = identify_robot(serial, axis, device_type, expected_model)
        state, error, warning = client.get_controller_status()
        if require_ready and (error or state not in (0, 2)):
            raise ValueError(f'Robot is not ready (state={state}, error={error}, '
                             f'warning={warning}); '
                             'resolve its status and make it ready in UFACTORY Studio first. '
                             'Preflight does not clear errors or release stops')
        current = list(client.get_joint_positions())
        states = client.get_reduced_states()
        if len(states) != 7 or states[0] not in (0, 1):
            raise ValueError('Invalid reduced-mode state')
        mode, _, _, max_speed, ranges, fence, rebound = states
        if len(ranges) != 14:
            raise ValueError('Expected seven reduced-mode joint-limit pairs')
        reduced = [list(ranges[i:i + 2]) for i in range(0, 12, 2)]
        bounds = profile['position_limits_rad']
        effective = intersect_joint_limits(bounds, reduced) if mode else bounds
        if mode and (not math.isfinite(max_speed) or max_speed <= 0):
            raise ValueError('Invalid reduced-mode maximum joint speed')
        targets = {'current measured position': current, **(joint_targets or {})}
        for name, position in targets.items():
            validate_joint_positions(position, effective, name)
            if client.is_joint_limit(position):
                raise ValueError(f'Controller rejects {name} under its joint limits')

        # Check each profile boundary from a neutral test vector. A rejection
        # invalidates the profile; it never justifies expanding or shifting it.
        midpoint = [(low + high) / 2 for low, high in effective]
        for i, (low, high) in enumerate(effective):
            inset = min(math.radians(.5), (high - low) / 4)
            for edge in (low + inset, high - inset):
                candidate = midpoint.copy()
                candidate[i] = edge
                if client.is_joint_limit(candidate):
                    raise ValueError(f'Controller rejects the selected profile near joint{i + 1}; '
                                     'limits cannot be identified safely from this profile')
        return {
            'profile': profile, 'serial_number': serial, 'firmware_version': version,
            'current_joints_rad': current, 'reduced_mode': bool(mode),
            'reduced_joint_limits_rad': reduced,
            'effective_position_limits_rad': effective,
            'reduced_max_joint_speed_rad_s': max_speed if mode else None,
            'safety_boundary_enabled': bool(fence), 'collision_rebound_enabled': bool(rebound),
            'controller_state': state, 'error_code': error, 'warning_code': warning,
        }
    finally:
        client.close()


def main():
    """Print a diagnostic snapshot; never activate the robot controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ip')
    parser.add_argument('--model', choices=('xarm6', 'lite6'), default='xarm6')
    args = parser.parse_args()
    try:
        result = probe_robot(args.ip, expected_model=args.model, require_ready=False)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
