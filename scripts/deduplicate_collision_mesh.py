# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Remove exactly repeated triangles from binary STL collision meshes."""

import argparse
import math
from pathlib import Path
import struct


def triangle_records(data):
    """Read validated binary STL records without rounding or moving vertices."""
    if len(data) < 84:
        raise ValueError('Binary STL header is incomplete')
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError('Binary STL size does not match its triangle count')
    for index in range(count):
        record = data[84 + index * 50:134 + index * 50]
        values = struct.unpack('<12fH', record)
        if not all(math.isfinite(value) for value in values[:12]):
            raise ValueError('Binary STL contains a non-finite coordinate or normal')
        yield record, tuple(
            sorted(tuple(values[offset:offset + 3]) for offset in (3, 6, 9))
        )


def deduplicate(data):
    """Keep the first original record for each exact unoriented triangle."""
    unique = set()
    records = []
    for record, vertices in triangle_records(data):
        if vertices not in unique:
            unique.add(vertices)
            records.append(record)
    return data[:80] + struct.pack('<I', len(records)) + b''.join(records)


def main():
    """Write a build-only mesh; preserve all unique faces and their coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    if arguments.source.resolve() == arguments.destination.resolve():
        parser.error('Source and destination must differ')
    result = deduplicate(arguments.source.read_bytes())
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    arguments.destination.write_bytes(result)


if __name__ == '__main__':
    main()
