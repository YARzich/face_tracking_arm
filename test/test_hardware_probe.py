# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Exercise the preflight decision path without a socket, SDK installation or arm."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup import hardware_probe as probe  # noqa: E402, I100


class FakeClient:
    def __init__(self):
        self.closed = False
        self.version = '6,6,XI1305,CB1300,v2.6.107'
        self.serial = 'XI13051234B123'
        self.angles = [5.8, 0, 0, 0, 0, 0]
        self.states = [0, [0] * 6, 100, .3, [-1, 1] * 7, 0, 0]
        self.checked = []
        self.reject = False
        self.status = (0, 0, 0)

    def get_version(self):
        return self.version

    def get_robot_sn(self):
        return self.serial

    def get_joint_positions(self):
        return self.angles

    def get_controller_status(self):
        return self.status

    def get_reduced_states(self):
        return self.states

    def is_joint_limit(self, positions):
        self.checked.append(list(positions))
        return self.reject

    def close(self):
        self.closed = True


def run(fake, **kwargs):
    return probe.probe_robot('192.0.2.1', client_factory=lambda ip, timeout: fake, **kwargs)


def test_measured_coordinates_are_preserved_and_inactive_reduced_limits_are_not_applied():
    fake = FakeClient()
    result = run(fake, joint_targets={'rest': [0] * 6})
    assert fake.closed
    assert result['current_joints_rad'] == [5.8, 0, 0, 0, 0, 0]
    assert result['profile']['model_num'] == 1305
    assert result['effective_position_limits_rad'][0] == [-2 * math.pi, 2 * math.pi]
    assert result['reduced_joint_limits_rad'][0] == [-1, 1]
    assert result['reduced_max_joint_speed_rad_s'] is None
    assert len(fake.checked) == 14
    assert fake.checked[0] == fake.angles


def test_active_reduced_envelope_is_applied_separately_from_factory_profile():
    fake = FakeClient()
    fake.angles = [0] * 6
    fake.states[0] = 1
    result = run(fake)
    assert result['effective_position_limits_rad'][0] == [-1, 1]
    assert result['profile']['position_limits_rad'][0] == [-2 * math.pi, 2 * math.pi]
    assert result['reduced_max_joint_speed_rad_s'] == .3
    assert fake.closed


@pytest.mark.parametrize('field,value,match', [
    ('serial', 'XIunknown', 'revision'),
    ('version', '6,9,LX1300,CB1300,v2.6.107', 'does not match'),
    ('version', 'v2.6.107', 'identify'),
    ('version', '6,6,XI1300,CB1300,v1.8.0', 'firmware'),
    ('angles', [math.nan] * 6, 'outside'),
    ('angles', [0] * 5, 'six'),
    ('reject', True, 'rejects'),
])
def test_preflight_failure_closes_connection_and_never_requests_motion(field, value, match):
    fake = FakeClient()
    setattr(fake, field, value)
    with pytest.raises(ValueError, match=match):
        run(fake)
    assert fake.closed


def test_rejected_boundary_does_not_expand_or_shift_profile():
    fake = FakeClient()

    def reject_boundary(position):
        return position[0] < -6
    fake.is_joint_limit = reject_boundary
    with pytest.raises(ValueError, match='selected profile near joint1'):
        run(fake)
    assert fake.closed


def test_bad_target_stops_before_any_profile_sampling():
    fake = FakeClient()
    with pytest.raises(ValueError, match='rest: joint3'):
        run(fake, joint_targets={'rest': [0, 0, 1, 0, 0, 0]})
    assert len(fake.checked) == 1
    assert fake.closed


def test_active_reduced_limits_cannot_accept_current_position_outside_envelope():
    fake = FakeClient()
    fake.states[0] = 1
    with pytest.raises(ValueError, match='current measured position: joint1'):
        run(fake)
    assert fake.closed


def test_sdk_transport_uses_only_getters_and_local_timeout(monkeypatch):
    calls = []

    class FakePort:
        connected = True

        def __init__(self, ip, port, **kwargs):
            calls.append(('connect', ip, port, kwargs))

        def close(self):
            calls.append(('close',))

        def join(self, **kwargs):
            calls.append(('join', kwargs))

    class FakeCore:
        def __init__(self, port):
            pass

        def set_timeout(self, seconds):
            calls.append(('timeout', seconds))

        def get_version(self):
            return [0, *b'6,6,XI1305,CB1300,v2.6.107\0']

        def get_robot_sn(self):
            return [0, *b'XI13051234B123\0']

        def get_joint_states(self, num):
            assert num == 1
            return [0, 0, 0, 0, 0, 0, 0, 99]

        def get_state(self):
            return [2, 0]

        def get_err_code(self):
            return [2, 0, 11]

        def get_reduced_states(self, length):
            assert length == 79
            return [0, 0, [0] * 6, 100, .3, [-1, 1] * 7, 0, 0]

        def is_joint_limit(self, angles):
            assert len(angles) == 7 and angles[-1] == 0
            return [0, 0]

    monkeypatch.setitem(sys.modules, 'xarm.core.comm.socket_port',
                        SimpleNamespace(SocketPort=FakePort))
    monkeypatch.setitem(sys.modules, 'xarm.core.wrapper.uxbus_cmd_tcp',
                        SimpleNamespace(UxbusCmdTcp=FakeCore))
    result = probe.probe_robot('192.0.2.1')
    assert result['current_joints_rad'] == [0] * 6
    assert result['warning_code'] == 11
    assert calls == [('connect', '192.0.2.1', 502, {'heartbeat': False, 'forbid_uds': True}),
                     ('timeout', 5.0), ('close',), ('join', {'timeout': 1.0})]


def test_transport_error_is_not_treated_as_a_controller_warning():
    with pytest.raises(ValueError, match='get_version failed'):
        probe._text([3, 0], 'get_version')


@pytest.mark.parametrize('status', [
    (1, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (99, 0, 0), (0, 23, 0),
])
def test_faults_and_stops_block_driver_activation_but_allow_diagnostic_read(status):
    fake = FakeClient()
    fake.status = status
    with pytest.raises(ValueError, match='make it ready in UFACTORY Studio'):
        run(fake)
    assert fake.closed
    result = run(fake, require_ready=False)
    assert (result['controller_state'], result['error_code'], result['warning_code']) == status


def test_warning_does_not_block_startup_or_get_cleared():
    fake = FakeClient()
    fake.status = (0, 0, 11)
    assert run(fake)['warning_code'] == 11


def test_pinned_sdk_decodes_measured_and_reduced_state_wire_payloads(monkeypatch):
    """Use the actual SDK parser when installed, with an entirely fake transport."""
    sdk = pytest.importorskip('xarm.core.wrapper.uxbus_cmd_tcp')
    import struct

    core = sdk.UxbusCmdTcp(None)
    sent = []

    def send(function, data, length):
        sent.append((function, data, length))
        return 1

    monkeypatch.setattr(core, 'send_modbus_request', send)
    angles = [5.8, -.3, -.5, 0, -.9, 0, 0]
    data = struct.pack('<7f', *angles)
    monkeypatch.setattr(core, 'recv_modbus_response', lambda *args: [2, *data])
    client = object.__new__(probe._SdkReadOnlyClient)
    client._core = core
    assert client.get_joint_positions() == pytest.approx(angles[:6])
    assert sent[-1][1:] == ([1], 1)

    data = (bytes([1]) + struct.pack('>6h', 100, -100, 100, -100, 100, -100)
            + struct.pack('<2f', 100.0, .3) + struct.pack('<14f', *([-1, 1] * 7))
            + bytes([0, 1]))
    monkeypatch.setattr(core, 'recv_modbus_response', lambda *args: [9, *data])
    result = client.get_reduced_states()
    assert result[0] == 1
    assert result[3] == pytest.approx(.3)
    assert result[4] == [-1, 1] * 7
    assert result[5:] == [0, 1]
