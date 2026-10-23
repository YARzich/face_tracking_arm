# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Bounded, nonblocking in-memory record of a hardware fault and its STOP response."""

from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time


@dataclass(frozen=True)
class FaultRecord:
    """One already-decoded event; payloads are expected to be small immutable values."""

    monotonic_time: float
    kind: str
    payload: tuple
    estimated_bytes: int = 1024


@dataclass(frozen=True)
class TimedContext:
    """Opaque companion snapshot with independent source and receipt times."""

    receipt_monotonic: float
    source_stamp_ns: int
    payload: tuple


class BoundedContextHistory:
    """Keep recent immutable companion snapshots for receipt-time joining."""

    def __init__(self, capacity=2048):
        if capacity < 1:
            raise ValueError('context capacity must be positive')
        self._items = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.dropped_events = 0
        self.overwritten_events = 0

    @property
    def capacity(self):
        return self._items.maxlen

    def record(self, source_stamp_ns, payload, receipt_monotonic=None):
        """Append a snapshot without waiting; return false if contended or invalid."""
        stamp = time.monotonic() if receipt_monotonic is None else receipt_monotonic
        try:
            source_stamp_ns = int(source_stamp_ns)
        except (TypeError, ValueError, OverflowError):
            self.dropped_events += 1
            return False
        if not math.isfinite(stamp) or source_stamp_ns < 0:
            self.dropped_events += 1
            return False
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            return False
        try:
            if len(self._items) == self._items.maxlen:
                self.overwritten_events += 1
            self._items.append(TimedContext(stamp, source_stamp_ns, tuple(payload)))
            return True
        finally:
            self._lock.release()

    def match(self, receipt_monotonic, maximum_age_sec, key=None):
        """Return a fresh prior snapshot, optionally matching named ID fields."""
        if (not math.isfinite(receipt_monotonic) or not math.isfinite(maximum_age_sec)
                or maximum_age_sec < 0):
            return None
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            return None
        try:
            for item in reversed(self._items):
                age = receipt_monotonic - item.receipt_monotonic
                if not 0.0 <= age <= maximum_age_sec:
                    continue
                if key is not None:
                    values = dict(item.payload)
                    if (values.get('hardware_epoch'), values.get('hardware_read_sequence')) != key:
                        continue
                return item, age
            return None
        finally:
            self._lock.release()

    def match_payload_stamp(self, field, source_stamp_ns, receipt_monotonic,
                            maximum_age_sec):
        """Return a fresh snapshot whose named two-part stamp exactly matches."""
        if (not math.isfinite(receipt_monotonic) or not math.isfinite(maximum_age_sec)
                or maximum_age_sec < 0):
            return None
        expected = (int(source_stamp_ns) // 1_000_000_000,
                    int(source_stamp_ns) % 1_000_000_000)
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            return None
        try:
            for item in reversed(self._items):
                age = receipt_monotonic - item.receipt_monotonic
                if not 0.0 <= age <= maximum_age_sec:
                    continue
                values = dict(item.payload)
                if values.get(field) == expected:
                    return item, age
            return None
        except (TypeError, ValueError, OverflowError):
            return None
        finally:
            self._lock.release()

    def match_payload_id(self, fields, expected_id, receipt_monotonic,
                         maximum_age_sec, hardware_epoch=None):
        """Find a fresh context that names an exact ID in one of its ID fields."""
        if (not math.isfinite(receipt_monotonic) or not math.isfinite(maximum_age_sec)
                or maximum_age_sec < 0):
            return None
        try:
            expected_id = int(expected_id)
            hardware_epoch = (None if hardware_epoch is None else int(hardware_epoch))
        except (TypeError, ValueError, OverflowError):
            return None
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            return None
        try:
            for item in reversed(self._items):
                age = receipt_monotonic - item.receipt_monotonic
                if not 0.0 <= age <= maximum_age_sec:
                    continue
                values = dict(item.payload)
                if (any(int(values.get(field, 0)) == expected_id for field in fields)
                        and (hardware_epoch is None
                             or int(values.get('hardware_epoch', -1)) == hardware_epoch)):
                    return item, age
            return None
        except (TypeError, ValueError, OverflowError):
            return None
        finally:
            self._lock.release()


class BoundedPublicationPayloadCache:
    """Keep recent full publications and pin IDs still named by execution context."""

    def __init__(self, capacity=256, max_bytes=16 * 1024 * 1024):
        if capacity < 1 or max_bytes < 1:
            raise ValueError('publication capacity must be positive')
        self._capacity = capacity
        self._items = OrderedDict()
        self._active_keys = set()
        self.dropped_events = 0
        self.overwritten_events = 0
        self._max_bytes = max_bytes
        self._stored_bytes = 0

    @property
    def capacity(self):
        return self._capacity

    @property
    def stored_bytes(self):
        return self._stored_bytes

    @property
    def byte_capacity(self):
        return self._max_bytes

    def record(self, epoch, publication_id, receipt_monotonic, payload):
        """Retain one immutable publication; reject malformed or duplicate IDs."""
        try:
            key = (int(epoch), int(publication_id))
            receipt = float(receipt_monotonic)
        except (TypeError, ValueError, OverflowError):
            self.dropped_events += 1
            return False
        if key[0] < 0 or key[1] <= 0 or not math.isfinite(receipt):
            self.dropped_events += 1
            return False
        if key in self._items:
            return False
        weight = self.estimate_payload_bytes(payload)
        if weight > self._max_bytes:
            self.dropped_events += 1
            return False
        self._items[key] = (receipt, payload, weight)
        self._stored_bytes += weight
        self._trim()
        return key in self._items

    def set_active(self, epoch, publication_ids):
        """Pin only IDs currently referenced by the executor's active/tail context."""
        try:
            epoch = int(epoch)
            self._active_keys = {
                (epoch, int(publication_id)) for publication_id in publication_ids
                if int(publication_id) > 0}
        except (TypeError, ValueError, OverflowError):
            self.dropped_events += 1
            return False
        for key in self._active_keys:
            if key in self._items:
                self._items.move_to_end(key)
        self._trim()
        return True

    def active_payloads(self):
        """Return the small active set with receipt times for dump-time pruning."""
        return tuple((key, *self._items[key][:2]) for key in sorted(self._active_keys)
                     if key in self._items)

    def missing_active_keys(self):
        """Return active IDs whose full publication event was never received."""
        return tuple(sorted(self._active_keys.difference(self._items)))

    def get_payload(self, epoch, publication_id):
        """Return the retained immutable payload for one driver-independent ID."""
        try:
            return self._items.get((int(epoch), int(publication_id)),
                                   (None, None, 0))[1]
        except (TypeError, ValueError, OverflowError):
            return None

    def _trim(self):
        while (len(self._items) > self._capacity + len(self._active_keys)
               or self._stored_bytes > self._max_bytes):
            removable = next((key for key in self._items if key not in self._active_keys), None)
            if removable is None:
                break
            removed = self._items.pop(removable)
            self._stored_bytes -= removed[2]
            self.overwritten_events += 1

    @staticmethod
    def estimate_payload_bytes(payload):
        """Conservatively bound decoded trajectory memory by its accepted point count."""
        details = dict(payload)
        points = details.get('points', ())
        if not isinstance(points, tuple):
            return 1024
        return 4096 + len(points) * 512


def missing_publication_payload_companions(pre_events, pinned_publications):
    """Return pinned payloads absent from retained full-publication events."""
    recorded_keys = set()
    for event in pre_events:
        if event.kind != 'servo_trajectory_publication' or not event.payload:
            continue
        try:
            details = dict(event.payload[-1])
            recorded_keys.add((int(details['hardware_epoch']),
                               int(details['publication_id'])))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return tuple(
        (key, receipt, payload) for key, receipt, payload in pinned_publications
        if key not in recorded_keys)


class HardwareFaultRecorder:
    """Retain bounded pre-fault history and a bounded post-fault response window."""

    def __init__(self, pre_capacity=16384, post_capacity=8192,
                 pre_window_sec=10.5, clock=time.monotonic,
                 pre_byte_capacity=64 * 1024 * 1024,
                 post_byte_capacity=64 * 1024 * 1024):
        if (pre_capacity < 1 or post_capacity < 1
                or not math.isfinite(pre_window_sec) or pre_window_sec < 10.0
                or pre_byte_capacity < 1 or post_byte_capacity < 1):
            raise ValueError('recorder capacities must be positive')
        self._pre = deque()
        self._pre_capacity = pre_capacity
        self._pre_byte_capacity = pre_byte_capacity
        self._pre_bytes = 0
        self._pre_window_sec = pre_window_sec
        self._pre_capacity_drops = 0
        self._pre_byte_drops = 0
        self._pre_byte_drop_times = deque(maxlen=pre_capacity)
        self._pre_capacity_drop_times = deque(maxlen=pre_capacity)
        self._untracked_pre_drop_until = None
        self._post = deque(maxlen=post_capacity)
        self._post_capacity = post_capacity
        self._post_byte_capacity = post_byte_capacity
        self._post_bytes = 0
        self._post_capacity_drops = 0
        self._post_byte_drops = 0
        self._post_contention_drops = 0
        self._transitions = deque(maxlen=64)
        self._transition_keys = set()
        self._transition_overflow = 0
        self._clock = clock
        self._lock = threading.Lock()
        self._fault_latched = False
        self._primary_fault = None
        self._dropped_events_at_latch = 0
        self._frozen_pre_fault = ()
        self._window_closed = False
        self.dropped_events = 0

    @property
    def post_capacity_drops(self):
        return self._post_capacity_drops

    @property
    def pre_stored_bytes(self):
        return self._pre_bytes

    @property
    def post_stored_bytes(self):
        return self._post_bytes

    @property
    def pre_window_sec(self):
        """Return the configured source-event retention horizon in seconds."""
        return self._pre_window_sec

    @property
    def primary_fault(self):
        return self._primary_fault

    @property
    def fault_latched(self):
        return self._fault_latched

    def record(self, kind, *payload, monotonic_time=None):
        """Append without waiting for another producer; return False when contended."""
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            if self._fault_latched:
                self._post_contention_drops += 1
            return False
        try:
            event = FaultRecord(self._clock() if monotonic_time is None else monotonic_time,
                                str(kind), tuple(payload),
                                self._estimate_event_bytes(kind, payload))
            if self._window_closed:
                return False
            if self._fault_latched:
                self._append_post(event)
                if kind in (
                        'inhibit_latched', 'inhibit_request', 'inhibit_service_ack',
                        'inhibit_status_confirmed', 'stop_request',
                        'stop_request_unconfirmed_timeout', 'stop_ack', 'stop_ack_timeout',
                        'stop_report_transition', 'stop_measured',
                        'stop_measurement_lost', 'stop_window_end'):
                    self._append_transition(event)
            else:
                self._prune_prehistory(event.monotonic_time)
                self._append_pre(event)
            return True
        finally:
            self._lock.release()

    def latch_fault(self, reason, *context, monotonic_time=None):
        """Freeze prehistory once; keep the first reason and append later context boundedly."""
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            if self._fault_latched:
                self._post_contention_drops += 1
            return False
        try:
            if not self._fault_latched:
                event = FaultRecord(self._clock() if monotonic_time is None else monotonic_time,
                                    'fault', (str(reason), *context), 1024)
                self._prune_prehistory(event.monotonic_time)
                self._frozen_pre_fault = tuple(self._pre)
                self._primary_fault = event
                self._dropped_events_at_latch = self.dropped_events
                self._fault_latched = True
                self._append_post(event)
                self._append_transition(event)
            else:
                event = FaultRecord(self._clock() if monotonic_time is None else monotonic_time,
                                    'fault_repeat', (str(reason), *context), 1024)
                self._append_post(event)
            return True
        finally:
            self._lock.release()

    def close_window(self, outcome, monotonic_time=None):
        """Stop post-fault recording after standstill evidence or its bounded timeout."""
        if not self._lock.acquire(blocking=False):
            self.dropped_events += 1
            if self._fault_latched:
                self._post_contention_drops += 1
            return False
        try:
            if not self._fault_latched or self._window_closed:
                return self._window_closed
            event = FaultRecord(self._clock() if monotonic_time is None else monotonic_time,
                                'stop_window_end', (str(outcome),), 1024)
            self._append_post(event)
            self._append_transition(event)
            self._window_closed = True
            return True
        finally:
            self._lock.release()

    def snapshot(self):
        """Return a consistent bounded copy for diagnostics outside the control path."""
        with self._lock:
            return (self._frozen_pre_fault, tuple(self._post), self._primary_fault,
                    tuple(self._transitions), self._window_closed)

    def write_dump(self, directory, metadata):
        """Persist one closed fault window without replacing an existing diagnostic file."""
        pre, post, primary, transitions, closed = self.snapshot()
        if not closed or primary is None:
            raise RuntimeError('fault record can be written only after the stop window closes')
        payload = {
            'metadata': metadata,
            'pre_fault_window': self._prehistory_metadata(pre, primary),
            'primary_fault': self._event_payload(primary),
            'pre_fault': [self._event_payload(event) for event in pre],
            'post_fault': [self._event_payload(event) for event in post],
            'transitions': [self._event_payload(event) for event in transitions],
            'dropped_events': self.dropped_events,
            'post_fault_window_capacity_drops': self._post_capacity_drops,
            'post_fault_window_byte_drops': self._post_byte_drops,
            'post_fault_window_contention_drops': self._post_contention_drops,
            'transition_overflow_count': self._transition_overflow,
            'post_fault_window_complete': all((
                self._post_capacity_drops == 0,
                self._post_contention_drops == 0,
                self._transition_overflow == 0)),
            'post_fault_byte_capacity': self._post_byte_capacity,
            'post_fault_estimated_bytes_retained': self._post_bytes,
        }
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        base = output_dir / f'hardware_fault_{stamp}_{os.getpid()}'
        for suffix in range(100):
            path = base.with_name(base.name + (f'_{suffix}' if suffix else '') + '.json')
            try:
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(self._json_safe(payload), stream,
                              separators=(',', ':'), allow_nan=False)
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f'could not allocate a unique fault record in {output_dir}')

    def _prune_prehistory(self, now):
        cutoff = now - self._pre_window_sec
        while self._pre and self._pre[0].monotonic_time <= cutoff:
            self._pre_bytes -= self._pre.popleft().estimated_bytes
        while self._pre_capacity_drop_times and self._pre_capacity_drop_times[0] <= cutoff:
            self._pre_capacity_drop_times.popleft()
        while self._pre_byte_drop_times and self._pre_byte_drop_times[0] <= cutoff:
            self._pre_byte_drop_times.popleft()
        if (self._untracked_pre_drop_until is not None
                and self._untracked_pre_drop_until <= now):
            self._untracked_pre_drop_until = None

    @staticmethod
    def _estimate_event_bytes(kind, payload):
        if kind == 'servo_trajectory_publication' and payload:
            try:
                details = dict(payload[-1])
                if isinstance(details.get('points'), tuple):
                    return BoundedPublicationPayloadCache.estimate_payload_bytes(
                        details.items())
            except (TypeError, ValueError):
                pass
        return 1024

    def _record_pre_drop(self, event, *, byte_drop=False):
        self._pre_capacity_drops += 1
        if byte_drop:
            self._pre_byte_drops += 1
        if len(self._pre_capacity_drop_times) == self._pre_capacity_drop_times.maxlen:
            evicted_drop = self._pre_capacity_drop_times[0]
            if evicted_drop > event.monotonic_time - self._pre_window_sec:
                expiry = evicted_drop + self._pre_window_sec
                self._untracked_pre_drop_until = max(
                    self._untracked_pre_drop_until or expiry, expiry)
        self._pre_capacity_drop_times.append(event.monotonic_time)
        if byte_drop:
            self._pre_byte_drop_times.append(event.monotonic_time)

    def _append_pre(self, event):
        if event.estimated_bytes > self._pre_byte_capacity:
            self._record_pre_drop(event, byte_drop=True)
            return
        byte_pressure = self._pre_bytes + event.estimated_bytes > self._pre_byte_capacity
        while (len(self._pre) >= self._pre_capacity
               or self._pre_bytes + event.estimated_bytes > self._pre_byte_capacity):
            removed = self._pre.popleft()
            self._pre_bytes -= removed.estimated_bytes
            self._record_pre_drop(event, byte_drop=byte_pressure)
        self._pre.append(event)
        self._pre_bytes += event.estimated_bytes

    def _append_post(self, event):
        if event.estimated_bytes > self._post_byte_capacity:
            self._post_capacity_drops += 1
            self._post_byte_drops += 1
            return
        while (len(self._post) >= self._post_capacity
               or self._post_bytes + event.estimated_bytes > self._post_byte_capacity):
            removed = self._post.popleft()
            self._post_bytes -= removed.estimated_bytes
            self._post_capacity_drops += 1
            if self._post_bytes + event.estimated_bytes > self._post_byte_capacity:
                self._post_byte_drops += 1
        self._post.append(event)
        self._post_bytes += event.estimated_bytes

    def _append_transition(self, event):
        key = (event.kind, repr(event.payload))
        if key in self._transition_keys:
            return
        if len(self._transitions) == self._transitions.maxlen:
            removed = self._transitions[0]
            self._transition_keys.discard((removed.kind, repr(removed.payload)))
            self._transition_overflow += 1
        self._transitions.append(event)
        self._transition_keys.add(key)

    def _prehistory_metadata(self, pre, primary):
        coverage = (0.0 if not pre or primary is None else max(
            0.0, primary.monotonic_time - pre[0].monotonic_time))
        cutoff = (0.0 if primary is None else
                  primary.monotonic_time - self._pre_window_sec)
        window_drops = sum(stamp >= cutoff for stamp in self._pre_capacity_drop_times)
        window_byte_drops = sum(stamp >= cutoff for stamp in self._pre_byte_drop_times)
        drop_tracking_overflow = (self._untracked_pre_drop_until is not None
                                  and self._untracked_pre_drop_until > primary.monotonic_time)
        return {
            'requested_pre_fault_window_sec': self._pre_window_sec,
            'pre_fault_coverage_sec': coverage,
            'pre_fault_window_capacity_drops': window_drops,
            'pre_fault_window_byte_drops': window_byte_drops,
            'pre_fault_byte_capacity_drops_total': self._pre_byte_drops,
            'pre_fault_byte_capacity': self._pre_byte_capacity,
            'pre_fault_estimated_bytes_retained': sum(
                event.estimated_bytes for event in pre),
            'pre_fault_drop_tracking_overflow': drop_tracking_overflow,
            'pre_fault_window_contention_drops': self._dropped_events_at_latch,
            'pre_fault_window_complete': (coverage >= 10.0 and window_drops == 0
                                          and not drop_tracking_overflow
                                          and self._dropped_events_at_latch == 0),
            'pre_fault_capacity_drops': self._pre_capacity_drops,
        }

    @staticmethod
    def _event_payload(event):
        return {
            'monotonic_time': event.monotonic_time,
            'kind': event.kind,
            'payload': event.payload,
            'estimated_bytes': event.estimated_bytes,
        }

    @staticmethod
    def _json_safe(value):
        """Convert non-finite telemetry to explicit strings during offline dump only."""
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return 'NaN'
            return 'Infinity' if value > 0 else '-Infinity'
        if isinstance(value, dict):
            return {str(key): HardwareFaultRecorder._json_safe(item)
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [HardwareFaultRecorder._json_safe(item) for item in value]
        return value
